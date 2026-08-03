"""Whole-model correctness sweep: metaljax vs jax-CPU on real texmo training.

For every benchmark-suite configuration, builds the actual ManagerJax model,
lowers ONE jitted training chunk (fwd + bwd + optimizer, scanned over a few
steps) to StableHLO, then executes the SAME module with the SAME inputs on
the jax CPU backend and through the metaljax interpreter, comparing every
output leaf (updated weights, optimizer state, per-step losses).

    .venv/bin/python scripts/texmo_check.py <suite.csv> [steps_per_chunk]

Prints one line per config: worst scale-normalized error across outputs.
Exercises every optimization path (mx.compile, msl scalar/vector/coop
kernels, accumulator fission, chunked loops) because it runs the real IR.
"""

import csv
import os
import sys
import time
from pathlib import Path

TEXMO = Path(os.environ.get("TEXMO_DIR", str(Path.home() / "texmo")))
sys.path.insert(0, str(TEXMO))

import jax

jax.config.update("jax_enable_x64", False)  # matches texmo.py
jax.config.update("jax_platforms", "cpu")

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

import mlx.core as mx
from metaljax import dtypes as mdt
from metaljax.interpreter import Interpreter

set_tokens_dir(str(TEXMO / "tokens"))
train_set = DataSet(path=str(TEXMO / "data" / "pride.txt"))


def configs_from_csv(path):
    seen = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["mode"] != "scan" or row.get("error"):
                continue
            # texmo renamed the raw_fold tokenset to fold (2026-08); older
            # suite CSVs still carry the old spec strings.
            spec = row["spec"].replace(".raw_fold", ".fold")
            key = (spec, int(row["batch"]), int(row["length"]))
            seen.setdefault(key, row["name"])
    return [(name, *key) for key, name in seen.items()]


def check_one(name, spec, batch, length, nsteps, rng):
    conf = Configuration(
        parse_model2(spec, precision=Precision("fp32")),
        lr=0.01, length=length, batch=batch, steps=nsteps,
        decay=1.0, cosine=False,
    )
    manager = ManagerJax(conf=conf, system="metaljax-check",
                         dataset=train_set, test_sample_len=64, test_batch=8)
    tokens_name = manager.model_def.input.tokens_name
    data = manager.dataset.sample_tokens(
        ntokens=length, batch=nsteps * batch, tokenset_name=tokens_name)
    batches = jnp.asarray(data).reshape(nsteps, batch, length)
    args = (manager.weights, manager._opt_state, batches)

    lowered = manager._train_chunk.lower(*args)
    text = lowered.as_text()

    flat_in = jax.tree_util.tree_leaves(args)
    ref = manager._train_chunk(*args)

    # Model sensitivity (CPU only): +1 ULP on every weight element. Configs
    # whose training step amplifies that to >>f32 noise are ill-conditioned;
    # cross-backend rounding differences will blow up identically, so the
    # pass tolerance must scale with this measured amplification.
    leaves, treedef = jax.tree_util.tree_flatten(args[0])
    pert = []
    for x in leaves:
        a = np.asarray(x).copy()
        if a.dtype.kind == "f" and a.size:
            f = a.reshape(-1)
            f[:] = np.nextafter(f, np.float32(1e9))
        pert.append(jnp.asarray(a))
    ref_p = manager._train_chunk(
        jax.tree_util.tree_unflatten(treedef, pert), *args[1:])
    sens = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(ref),
                    jax.tree_util.tree_leaves(ref_p)):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.dtype.kind != "f" or not a.size:
            continue
        d = float(np.abs(a - b).max())
        sc = max(float(np.abs(a).max()), 1e-3)
        sens = max(sens, d / sc)
    flat_ref = [np.asarray(x, dtype=np.float32)
                if np.asarray(x).dtype.kind == "f" else np.asarray(x)
                for x in jax.tree_util.tree_leaves(ref)]

    interp = Interpreter(text)
    outs = interp(*[mdt.to_mx(np.asarray(x)) for x in flat_in])
    mx.eval(*outs)

    worst = 0.0
    worst_i = -1
    n_bad = 0
    for i, (o, r) in enumerate(zip(outs, flat_ref)):
        h = np.array(o, copy=False)
        if h.dtype != r.dtype:
            h = h.astype(r.dtype)
        if h.shape != r.shape:
            return None, f"shape mismatch out[{i}]: {h.shape} vs {r.shape}"
        if r.dtype.kind == "f":
            if (np.isfinite(r) != np.isfinite(h)).any():
                n_bad += 1
                worst = float("inf")
                worst_i = i
                continue
            fin = np.isfinite(r)
            d = np.abs(h - r)[fin]
            if not d.size:
                continue
            scale = max(float(np.abs(r[fin]).max()), 1e-3)
            norm = float(d.max()) / scale
            if norm > worst:
                worst, worst_i = norm, i
        else:
            if not np.array_equal(h, r):
                n_bad += 1
                worst = float("inf")
                worst_i = i
    return (worst, worst_i, len(flat_ref), n_bad, sens), None


def main():
    suite_csv = sys.argv[1]
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tol = float(os.environ.get("CHECK_TOL", "2e-3"))
    rng = np.random.default_rng(0)
    configs = sorted(configs_from_csv(suite_csv))
    n_pass = n_fail = n_err = 0
    for spec, batch, length, name in [(s, b, l, n)
                                      for (n, s, b, l) in configs]:
        t0 = time.perf_counter()
        try:
            res, err = check_one(name, spec, batch, length, nsteps, rng)
        except Exception as e:
            n_err += 1
            print(f"ERROR  {name:<18} {spec[:44]:<46} {str(e)[:80]}",
                  flush=True)
            continue
        dt = time.perf_counter() - t0
        if err:
            n_err += 1
            print(f"ERROR  {name:<18} {spec[:44]:<46} {err}", flush=True)
            continue
        worst, wi, nout, nbad, sens = res
        # tolerance scales with the model's own 1-ULP amplification:
        # cross-backend rounding is worth a few hundred ULPs of seed.
        eff_tol = max(tol, sens * 500)
        ok = worst <= eff_tol and nbad == 0
        n_pass += ok
        n_fail += not ok
        tag = "ok" if ok else "FAIL"
        if ok and worst > tol:
            tag = "ok~"  # passed only via sensitivity scaling
        print(f"{tag:<6} {name:<18} {spec[:44]:<46} "
              f"worst={worst:.2e} sens={sens:.1e} tol={eff_tol:.1e} "
              f"out[{wi}]/{nout} bad={nbad} {dt:.0f}s", flush=True)
    print(f"\nSUMMARY: {n_pass} ok, {n_fail} FAIL, {n_err} error "
          f"(tol {tol})", flush=True)


if __name__ == "__main__":
    main()
