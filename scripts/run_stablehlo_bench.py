"""Run a StableHLO module as a benchmark on a chosen JAX platform.

Made for the openxla/xla benchmark suite (xla/tools/benchmarks): convert the
HLO-text benchmarks to StableHLO with `xla-translate --hlo-to-stablehlo`,
then run them here on cpu / metal / cuda with identical seeded inputs.

  JAX_PLATFORMS=metal,cpu python run_stablehlo_bench.py m.mlir --platform metal \
      --save-out ref.npz          # save outputs (as f32) for cross-checking
  python run_stablehlo_bench.py m.mlir --platform cpu --check ref.npz

Emits one JSON line on stdout (prefix RESULT:) with timing and status; all
diagnostics go to stderr. Timing is wall time per execute with device sync —
set METALJAX_SYNC=1 for the metal platform (jax.block_until_ready is a no-op
there; this script refuses to time metal without it).
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def parse_arg_types(text):
    """(shapes, np_dtypes) of the entry function's arguments."""
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo
    from jaxlib.mlir._mlir_libs import _jax_mlir_ext

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _jax_mlir_ext.register_dialects(reg)
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()
    stablehlo.register_dialect(ctx)
    with ctx:
        module = ir.Module.parse(text)
        entry = None
        for op in module.body.operations:
            if op.operation.name == "func.func":
                name = ir.StringAttr(op.attributes["sym_name"]).value
                if name == "main" or entry is None:
                    entry = op
        if entry is None:
            raise ValueError("no func.func in module")
        specs = []
        for a in entry.regions[0].blocks[0].arguments:
            t = ir.RankedTensorType(a.type)
            specs.append((tuple(t.shape), str(t.element_type)))
    return specs


_DT = {
    "f64": np.float64, "f32": np.float32, "f16": np.float16,
    "i64": np.int64, "i32": np.int32, "i16": np.int16, "i8": np.int8,
    "ui64": np.uint64, "ui32": np.uint32, "ui16": np.uint16, "ui8": np.uint8,
    "i1": np.bool_,
}


def np_dtype(elem):
    if elem == "bf16":
        import ml_dtypes
        return np.dtype(ml_dtypes.bfloat16)
    if elem in _DT:
        return np.dtype(_DT[elem])
    raise ValueError(f"unhandled element type {elem}")


def gen_inputs(specs, seed):
    rng = np.random.default_rng(seed)
    out = []
    for shape, elem in specs:
        dt = np_dtype(elem)
        if dt == np.bool_:
            a = np.zeros(shape, np.bool_)
        elif np.issubdtype(dt, np.integer):
            # Small non-negative values: safe as token ids / gather indices.
            a = rng.integers(0, 8, size=shape).astype(dt)
        else:
            a = (rng.standard_normal(shape) * 0.02).astype(dt)
        out.append(a)
    return out


def nbytes(specs):
    tot = 0
    for shape, elem in specs:
        n = 1
        for d in shape:
            n *= d
        tot += n * np_dtype(elem).itemsize
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--name", default=None)
    ap.add_argument("--platform", default="cpu")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mem-limit-gb", type=float, default=None,
                    help="skip if inputs+outputs exceed this")
    ap.add_argument("--save-out", default=None)
    ap.add_argument("--check", default=None,
                    help="npz of reference outputs to compare against")
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--atol", type=float, default=2e-2)
    args = ap.parse_args()
    name = args.name or os.path.basename(args.module)

    res = {"name": name, "platform": args.platform, "status": "ok"}

    def emit():
        print("RESULT:" + json.dumps(res), flush=True)

    text = open(args.module).read()
    in_specs = parse_arg_types(text)
    res["param_gb"] = round(nbytes(in_specs) / 1e9, 3)
    log(f"{name}: {len(in_specs)} args, {res['param_gb']} GB inputs")
    if args.mem_limit_gb and nbytes(in_specs) / 1e9 > args.mem_limit_gb:
        res["status"] = "skipped_mem"
        emit()
        return

    import jax
    from jax._src.lib import xla_client as xc

    if args.platform == "metal" and os.environ.get("METALJAX_SYNC") != "1":
        raise SystemExit("set METALJAX_SYNC=1: block_until_ready is a no-op "
                         "on the metal plugin, timings would be meaningless")

    dev = jax.devices(args.platform)[0]
    inputs = gen_inputs(in_specs, args.seed)

    t0 = time.perf_counter()
    try:
        exe = dev.client.compile_and_load(text, [dev], xc.CompileOptions())
    except Exception as e:
        res["status"] = "compile_error"
        res["error"] = str(e)[:400]
        emit()
        return
    res["compile_s"] = round(time.perf_counter() - t0, 2)
    log(f"{name}: compiled in {res['compile_s']}s on {dev}")

    dargs = [jax.device_put(a, dev) for a in inputs]
    del inputs

    try:
        for _ in range(max(1, args.warmup)):
            outs = exe.execute(dargs)
            for o in outs:
                o.block_until_ready()
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            outs = exe.execute(dargs)
            for o in outs:
                o.block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)
    except Exception as e:
        res["status"] = "run_error"
        res["error"] = str(e)[:400]
        emit()
        return

    times.sort()
    res["ms_min"] = round(times[0], 2)
    res["ms_median"] = round(times[len(times) // 2], 2)
    res["n_outputs"] = len(outs)
    log(f"{name} [{args.platform}]: min {res['ms_min']} ms, "
        f"median {res['ms_median']} ms over {args.reps} reps")

    if args.save_out or args.check:
        host = [np.asarray(o).astype(np.float32) for o in outs]
    if args.save_out:
        np.savez_compressed(args.save_out,
                            **{f"o{i}": h for i, h in enumerate(host)})
        log(f"{name}: saved {len(host)} outputs to {args.save_out}")
    if args.check:
        ref = np.load(args.check)
        worst_abs = worst_rel = 0.0
        n_mismatch = 0
        for i, h in enumerate(host):
            r = ref[f"o{i}"]
            if r.shape != h.shape:
                res["status"] = "check_shape_mismatch"
                break
            finite = np.isfinite(r) & np.isfinite(h)
            if not finite.all() and not (np.isfinite(r) == np.isfinite(h)).all():
                n_mismatch += 1
            d = np.abs(h - r)[finite]
            if d.size:
                worst_abs = max(worst_abs, float(d.max()))
                denom = np.abs(r[finite]) + 1e-6
                worst_rel = max(worst_rel, float((d / denom).max()))
            if not np.allclose(h[finite], r[finite],
                               rtol=args.rtol, atol=args.atol):
                n_mismatch += 1
        res["max_abs_err"] = f"{worst_abs:.3e}"
        res["max_rel_err"] = f"{worst_rel:.3e}"
        res["check"] = "PASS" if n_mismatch == 0 else f"FAIL({n_mismatch})"
        log(f"{name}: check {res['check']} abs {res['max_abs_err']} "
            f"rel {res['max_rel_err']}")
    emit()


if __name__ == "__main__":
    main()
