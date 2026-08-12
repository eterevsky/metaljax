"""Bench texmo training chunks THROUGH THE PJRT PLUGIN (the jax route).

`scripts/texmo_topconfs.py` measures metal by driving `metaljax.engine`
directly, which is a Stage-1-only route: the Stage 2 plugin holds no Python
engine, so a run under `METALJAX_PLUGIN_PATH=.../libmetal_pjrt_native.dylib`
would silently keep measuring Stage 1.  This runner goes through jax itself
(`jax_platforms=metal`), so whichever plugin the environment selects is the
one measured, and the two stacks are compared on exactly the same route.

    .venv/bin/python scripts/bench_texmo_pjrt.py CONFIGS --out OUT.jsonl
        [--platform metal|cpu] [--steps 256] [--budget-s 3.0]
        [--only SUBSTR] [--limit N] [--warm-only]

CONFIGS is either the suite CSV (`benchmarks/texmo-suite.csv`: name, spec,
precision, batch, length, mode) or a top-configurations JSONL
(`spec, precision, batch, length, weights, ...`).

Metric definitions match `texmo_topconfs.py`: one jitted chunk of `--steps`
training steps (forward + backward + optimizer, scanned), `warmup` is the
first call including trace+compile, `ms_step` is the best chunk over an
auto-calibrated number of repetitions divided by the step count.  Results are
forced with `np.asarray` on every output leaf -- `jax.block_until_ready` is a
no-op on this backend (CLAUDE.md item 9).

Sequential by construction; hold the machine lock around it.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

TEXMO = Path(os.environ.get("TEXMO_DIR", str(Path.home() / "texmo")))
sys.path.insert(0, str(TEXMO))

_PLATFORM = os.environ.get("BENCH_PLATFORM", "metal")
for i, a in enumerate(sys.argv):
    if a == "--platform" and i + 1 < len(sys.argv):
        _PLATFORM = sys.argv[i + 1]

import jax

jax.config.update("jax_enable_x64", False)   # matches texmo.py
jax.config.update("jax_platforms", _PLATFORM)

import jax.numpy as jnp
import numpy as np

import logging

logging.disable(logging.INFO)

from texmo.tokens import set_tokens_dir
from texmo.dataset import DataSet
from texmo.configuration import Configuration
from texmo.spec_parser import parse_model2
from texmo.precision import Precision
from texmo.manager_jax import ManagerJax

set_tokens_dir(str(TEXMO / "tokens"))
train_set = DataSet(path=str(TEXMO / "data" / "pride.txt"))


def read_configs(path):
    """(name, spec, precision, batch, length, weights) per configuration."""
    p = Path(path)
    rows = []
    if p.suffix == ".csv":
        seen = set()
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("mode") != "scan" or row.get("error"):
                    continue
                # texmo renamed raw_fold -> fold (2026-08); old CSVs carry
                # the old spelling (texmo_gate.py does the same).
                spec = row["spec"].replace(".raw_fold", ".fold")
                key = (spec, int(row["batch"]), int(row["length"]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"name": row["name"], "spec": spec,
                             "precision": row.get("precision", "fp32"),
                             "batch": int(row["batch"]),
                             "length": int(row["length"])})
    else:
        for i, line in enumerate(open(p)):
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"name": f"tc{i:03d}-w{r.get('weights', 0)}",
                         "spec": r["spec"], "precision": r["precision"],
                         "batch": r["batch"], "length": r["length"],
                         "weights": r.get("weights"),
                         "lr": r.get("lr", 0.01),
                         "decay": r.get("decay", 1.0),
                         "cosine": r.get("cosine", False)})
    return rows


def build(row, nsteps):
    length = row["length"]
    conf = Configuration(
        parse_model2(row["spec"], precision=Precision(row["precision"])),
        lr=row.get("lr", 0.01), length=length, batch=row["batch"],
        steps=nsteps, decay=row.get("decay", 1.0),
        cosine=row.get("cosine", False),
    )
    manager = ManagerJax(conf=conf, system="metaljax-pjrt-bench",
                         dataset=train_set, test_sample_len=64, test_batch=8)
    tokens_name = manager.model_def.input.tokens_name
    data = manager.dataset.sample_tokens(
        ntokens=length, batch=nsteps * row["batch"], tokenset_name=tokens_name)
    batches = jnp.asarray(data).reshape(nsteps, row["batch"], length)
    return manager, (manager.weights, manager._opt_state, batches)


def force(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        np.asarray(leaf)


def bench(row, nsteps, budget_s, warm_only=False):
    manager, args = build(row, nsteps)
    t0 = time.perf_counter()
    out = manager._train_chunk(*args)
    force(out)
    warmup = time.perf_counter() - t0
    if warm_only:
        return manager, {"warmup_s": round(warmup, 4)}
    t0 = time.perf_counter()
    out = manager._train_chunk(*args)
    force(out)
    chunk_t = time.perf_counter() - t0
    reps = max(1, min(64, int(budget_s / max(chunk_t, 1e-4))))
    best = chunk_t
    for _ in range(reps - 1):
        t0 = time.perf_counter()
        out = manager._train_chunk(*args)
        force(out)
        best = min(best, time.perf_counter() - t0)
    return manager, {"warmup_s": round(warmup, 4),
                     "ms_step": round(1000 * best / nsteps, 4),
                     "chunk_ms": round(1000 * best, 3),
                     "reps": reps, "bench_chunk_steps": nsteps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform", default=_PLATFORM)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--budget-s", type=float, default=3.0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warm-only", action="store_true")
    ns = ap.parse_args()

    rows = read_configs(ns.configs)
    if ns.only:
        rows = [r for r in rows if ns.only in r["name"] or ns.only in r["spec"]]
    if ns.limit:
        rows = rows[:ns.limit]

    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    print(f"# platform={jax.default_backend()} plugin="
          f"{os.environ.get('METALJAX_PLUGIN_PATH', '(default)')} "
          f"configs={len(rows)} steps={ns.steps}", flush=True)
    with open(ns.out, "w") as f:
        for row in rows:
            rec = dict(row)
            rec["platform"] = jax.default_backend()
            t0 = time.perf_counter()
            manager = None
            try:
                manager, res = bench(row, ns.steps, ns.budget_s, ns.warm_only)
                rec.update(res)
                n_ok += 1
            except BaseException as e:            # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                n_err += 1
            rec["wall_s"] = round(time.perf_counter() - t0, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{rec['name']:<14} {row['spec'][:46]:<48} "
                  f"{row['precision']:<4} b{row['batch']:<5} l{row['length']:<5} "
                  f"{rec.get('ms_step', rec.get('error', '-'))} "
                  f"warm={rec.get('warmup_s', '-')}s [{rec['wall_s']}s]",
                  flush=True)
            del manager
            gc.collect()
            jax.clear_caches()
            gc.collect()
    print(f"\nSUMMARY: {n_ok} ok, {n_err} error -> {ns.out}", flush=True)


if __name__ == "__main__":
    main()
