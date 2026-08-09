"""Tracked texmo micro-benchmark suite: top search configurations.

Reads an exported top-configurations JSONL (spec, precision, lr, length,
batch, steps, decay, cosine, weights, ...) and for each config:

  check  — correctness vs jax-CPU on the REAL training chunk (2 steps,
           sample length capped: scan noise diverges backends on long
           rollouts), sensitivity-scaled tolerance as in texmo_check.
  bench  — warmup (first chunk call, incl. compile) and steady ms/step
           on metaljax and on jax-CPU, reps auto-calibrated so each
           config costs a few seconds.

    .venv/bin/python scripts/texmo_topconfs.py top_confs.jsonl \
        [--out notes/data/texmo-topconfs-DATE.jsonl] [--check-only|--bench-only]
        [--filter SUBSTR]

Each config runs at its OWN precision (bf16 configs train in bf16).
Writes one JSON line per config with every metric; prints a summary.
Sequential; do not run alongside other GPU work.
"""

import argparse
import json
import os
import sys
import time
from datetime import date
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

import ml_dtypes

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

CHECK_STEPS = 2
CHECK_MAX_LEN = 128
BENCH_TARGET_S = 3.0    # steady-state measurement budget per config/platform
BENCH_CHUNK = 256       # production texmo chunking: one jitted scan of
                        # 256 training steps — per-step numbers must
                        # amortize dispatch over the SAME trip count


def build(row, nsteps, length):
    conf = Configuration(
        parse_model2(row["spec"], precision=Precision(row["precision"])),
        lr=row["lr"], length=length, batch=row["batch"], steps=nsteps,
        decay=row["decay"], cosine=row["cosine"],
    )
    manager = ManagerJax(conf=conf, system="metaljax-topconfs",
                         dataset=train_set, test_sample_len=64, test_batch=8)
    tokens_name = manager.model_def.input.tokens_name
    data = manager.dataset.sample_tokens(
        ntokens=length, batch=nsteps * row["batch"], tokenset_name=tokens_name)
    batches = jnp.asarray(data).reshape(nsteps, row["batch"], length)
    return manager, (manager.weights, manager._opt_state, batches)


def one_ulp_up(a):
    """+1 ULP on every positive-direction element, in the array's OWN
    dtype (np.nextafter has no bf16 support: bump the bit pattern)."""
    if a.dtype == ml_dtypes.bfloat16:
        bits = a.view(np.uint16).copy()
        # nextafter towards +inf: +1 on non-negative patterns, -1 on
        # negative ones (sign-magnitude), leave non-finite alone
        fin = np.isfinite(a.astype(np.float32))
        neg = (bits & 0x8000) != 0
        bits[fin & ~neg] += 1
        bits[fin & neg] -= 1
        return bits.view(ml_dtypes.bfloat16)
    out = a.copy().reshape(-1)
    out[:] = np.nextafter(out, out.dtype.type(np.inf))
    return out.reshape(a.shape)


def check_config(row):
    length = min(row["length"], CHECK_MAX_LEN)
    manager, args = build(row, CHECK_STEPS, length)
    lowered = manager._train_chunk.lower(*args)
    text = lowered.as_text()
    flat_in = jax.tree_util.tree_leaves(args)
    ref = manager._train_chunk(*args)

    # sensitivity: +1 ULP on every weight, in the weights' own dtype
    leaves, treedef = jax.tree_util.tree_flatten(args[0])
    pert = [jnp.asarray(one_ulp_up(np.asarray(x)))
            if np.asarray(x).dtype.kind in "fV" and np.asarray(x).size
            else x for x in leaves]
    ref_p = manager._train_chunk(
        jax.tree_util.tree_unflatten(treedef, pert), *args[1:])
    sens = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(ref),
                    jax.tree_util.tree_leaves(ref_p)):
        a = np.asarray(a).astype(np.float64)
        b = np.asarray(b).astype(np.float64)
        if not a.size:
            continue
        d = float(np.abs(a - b).max())
        sc = max(float(np.abs(a).max()), 1e-3)
        sens = max(sens, d / sc)

    interp = Interpreter(text)
    outs = interp(*[mdt.to_mx(np.asarray(x)) for x in flat_in])
    mx.eval(*outs)

    worst, worst_i, n_bad = 0.0, -1, 0
    flat_ref = [np.asarray(x) for x in jax.tree_util.tree_leaves(ref)]
    for i, (o, r) in enumerate(zip(outs, flat_ref)):
        h = np.array(mdt.to_np(o), copy=False)
        if h.dtype != r.dtype:
            h = h.astype(r.dtype)
        if h.shape != r.shape:
            return {"check": "shape-mismatch", "check_out": i}
        rf = r.astype(np.float64) if r.dtype.kind in "fV" else r
        hf = h.astype(np.float64) if h.dtype.kind in "fV" else h
        if r.dtype.kind in "fV":
            if (np.isfinite(rf) != np.isfinite(hf)).any():
                n_bad += 1
                worst, worst_i = float("inf"), i
                continue
            fin = np.isfinite(rf)
            d = np.abs(hf - rf)[fin]
            if not d.size:
                continue
            scale = max(float(np.abs(rf[fin]).max()), 1e-3)
            norm = float(d.max()) / scale
            if norm > worst:
                worst, worst_i = norm, i
        elif not np.array_equal(hf, rf):
            n_bad += 1
            worst, worst_i = float("inf"), i
    tol = max(2e-3 if row["precision"] == "fp32" else 2e-2, sens * 500)
    return {
        "check": "ok" if worst <= tol and n_bad == 0 else "FAIL",
        "check_worst": worst, "check_sens": sens, "check_tol": tol,
        "check_out": worst_i, "check_bad": n_bad,
    }


def bench_platform(row, run_chunk, args, nsteps):
    """Warmup = first call (trace+compile+first run); steady = best
    chunk over an auto-calibrated number of repetitions."""
    t0 = time.perf_counter()
    out = run_chunk(*args)
    jax.tree_util.tree_map(np.asarray, out)   # force (block_until_ready
    warmup = time.perf_counter() - t0          # is a no-op on metaljax)
    t0 = time.perf_counter()
    out = run_chunk(*args)
    jax.tree_util.tree_map(np.asarray, out)
    chunk_t = time.perf_counter() - t0
    reps = max(1, min(64, int(BENCH_TARGET_S / max(chunk_t, 1e-4))))
    best = chunk_t
    for _ in range(reps - 1):
        t0 = time.perf_counter()
        out = run_chunk(*args)
        jax.tree_util.tree_map(np.asarray, out)
        best = min(best, time.perf_counter() - t0)
    return warmup, best / nsteps


def bench_config(row):
    nsteps = BENCH_CHUNK
    manager, args = build(row, nsteps, row["length"])
    text = manager._train_chunk.lower(*args).as_text()
    flat_in = [np.asarray(x) for x in jax.tree_util.tree_leaves(args)]

    # --- metaljax (through engine.execute when the native engine is on:
    # the Stage-2 tape hooks there, and a bench that calls Interpreter
    # directly measures the Python path whatever METALJAX_ENGINE says)
    from metaljax import engine as mengine
    if mengine.NATIVE is None:
        interp = Interpreter(text)
        mx_args = [mdt.to_mx(x) for x in flat_in]

        def run_metal(*_):
            outs = interp(*mx_args)
            mx.eval(*outs)
            return ()
    else:
        ex = mengine.compile_program(text.encode(), "mlir")
        bufs = []
        for x in flat_in:
            a = np.ascontiguousarray(x)
            bufs.append(mengine.buffer_from_host(
                a.tobytes(), mengine._NP_TO_ENUM[a.dtype], list(a.shape),
                None, 0))

        def run_metal(*_):
            outs = mengine.execute(ex, bufs)
            mx.eval(*[o.data for o in outs])
            return ()

    m_warm, m_step = bench_platform(row, run_metal, (), nsteps)

    # --- jax CPU (same lowered chunk, jitted natively)
    cpu_args = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, jax.devices("cpu")[0]), args)
    c_warm, c_step = bench_platform(
        row, manager._train_chunk, cpu_args, nsteps)

    return {
        "metal_warmup_s": round(m_warm, 4),
        "metal_ms_step": round(1000 * m_step, 4),
        "cpu_warmup_s": round(c_warm, 4),
        "cpu_ms_step": round(1000 * c_step, 4),
        "ratio_cpu_over_metal": round(c_step / m_step, 3) if m_step else None,
        "bench_chunk_steps": nsteps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--filter", default=None)
    ns = ap.parse_args()

    rows = [json.loads(l) for l in open(ns.jsonl) if l.strip()]
    out_path = ns.out or f"notes/data/texmo-topconfs-{date.today()}.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_err = 0
    with open(out_path, "w") as out_f:
        for i, row in enumerate(rows):
            name = f"tc{i:03d}-w{row['weights']}"
            if ns.filter and ns.filter not in row["spec"]:
                continue
            rec = {"name": name, **{k: row[k] for k in
                   ("spec", "precision", "batch", "length", "weights")}}
            t0 = time.perf_counter()
            try:
                if not ns.bench_only:
                    rec.update(check_config(row))
                if not ns.check_only:
                    rec.update(bench_config(row))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:200]
                n_err += 1
                print(f"ERROR  {name:<12} {row['spec'][:52]:<54} "
                      f"{rec['error'][:60]}", flush=True)
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                continue
            rec["wall_s"] = round(time.perf_counter() - t0, 1)
            ck = rec.get("check", "-")
            if ck == "ok":
                n_ok += 1
            elif ck != "-":
                n_fail += 1
            print(f"{ck:<6} {name:<12} {row['spec'][:52]:<54} "
                  f"{row['precision']:<4} "
                  f"metal={rec.get('metal_ms_step', '-'):>9} "
                  f"cpu={rec.get('cpu_ms_step', '-'):>9} ms/step "
                  f"x{rec.get('ratio_cpu_over_metal', '-')} "
                  f"warm={rec.get('metal_warmup_s', '-')}s "
                  f"[{rec['wall_s']}s]", flush=True)
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
    print(f"\nSUMMARY: {n_ok} ok, {n_fail} FAIL, {n_err} error "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
