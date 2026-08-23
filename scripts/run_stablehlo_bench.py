"""Run a StableHLO module as a benchmark on a chosen JAX platform.

Made for the openxla/xla benchmark suite (xla/tools/benchmarks): convert the
HLO-text benchmarks to StableHLO with `xla-translate --hlo-to-stablehlo`,
then run them here on cpu / metal / cuda with identical seeded inputs.

  JAX_PLATFORMS=metal,cpu python run_stablehlo_bench.py m.mlir --platform metal \
      --save-out ref.npz          # save outputs (as f32) for cross-checking
  python run_stablehlo_bench.py m.mlir --platform cpu --check ref.npz

Emits one JSON line on stdout (prefix RESULT:) with timing and status; all
diagnostics go to stderr. Timing is wall time per execute with device sync:
every output is forced through numpy, because `block_until_ready` is a no-op
on the metal plugin (its PJRT events are born ready) and a loop that only
blocks would measure dispatch. This is the `force()` discipline of
scripts/bench_texmo_pjrt.py; the Stage-1-only METALJAX_SYNC knob it used to
demand died with that engine.
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
    """((shapes, np_dtypes) of entry args, sanitized module text).

    Strips mhlo.input_output_alias (buffer donation): a benchmark loop
    re-executes with the same argument buffers, which donation forbids.
    """
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
    # jax emits sdy (and sometimes mpmd) sharding ops even on one device, so a
    # captured maxtext module carries `sdy.mesh` at the top of the file.
    # `allow_unregistered_dialects` covers unknown ops in GENERIC form only; a
    # custom assembly like `sdy.mesh @mesh = <[...]>` still needs the dialect,
    # and without it this parse dies before any benchmark runs — which is what
    # kept notes/data/qwen3_8b_prefill_36layer.mlir, the 8B command-buffer
    # canary, from being re-run since 2026-08-03.  Same registration the
    # plugin does natively (plugin-native ingest registers the same dialects).
    # chlo for the same reason: `chlo.square` and friends survive into captured
    # jax modules, and they parse in custom assembly too.
    for _name in ("chlo", "sdy", "mpmd"):
        try:
            __import__(f"jaxlib.mlir.dialects.{_name}",
                       fromlist=[_name]).register_dialect(ctx)
        except Exception:
            pass
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
        if "mhlo.input_output_alias" in module.operation.attributes:
            del module.operation.attributes["mhlo.input_output_alias"]
            text = module.operation.get_asm(large_elements_limit=None)
    return specs, text


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


def gen_inputs(specs, seed, device=None):
    """Seeded inputs; with `device`, each array is moved to the device as it
    is generated so peak host memory stays one array, not the whole set."""
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
            a = (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(dt)
        if device is not None:
            import jax
            a = jax.device_put(a, device)
            # Sync periodically: async transfers pin their host arrays, and
            # thousands of queued puts can exhaust host RAM.
            if len(out) % 16 == 0:
                jax.block_until_ready(a)
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
    in_specs, text = parse_arg_types(text)
    res["param_gb"] = round(nbytes(in_specs) / 1e9, 3)
    log(f"{name}: {len(in_specs)} args, {res['param_gb']} GB inputs")
    if args.mem_limit_gb and nbytes(in_specs) / 1e9 > args.mem_limit_gb:
        res["status"] = "skipped_mem"
        emit()
        return

    import jax
    from jax._src.lib import xla_client as xc

    dev = jax.devices(args.platform)[0]

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

    dargs = gen_inputs(in_specs, args.seed, device=dev)

    def force(outs):
        """Settle every output, and MEAN it.

        `block_until_ready` is a no-op on the metal plugin -- its PJRT
        events are born ready and execute is async -- so a loop that only
        blocks measures dispatch, not work. Reading one element through
        numpy is what actually waits (the same `force()` discipline
        scripts/bench_texmo_pjrt.py uses). Cheap: one element per output,
        not the whole buffer.
        """
        for o in outs:
            o.block_until_ready()          # honest on cpu/cuda, free here
            np.asarray(o.reshape(-1)[:1] if o.ndim else o)

    try:
        for _ in range(max(1, args.warmup)):
            outs = exe.execute(dargs)
            force(outs)
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            outs = exe.execute(dargs)
            force(outs)
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
        worst_abs = worst_norm = 0.0
        n_mismatch = 0
        for i, h in enumerate(host):
            r = ref[f"o{i}"]
            if r.shape != h.shape:
                res["status"] = "check_shape_mismatch"
                break
            if (np.isfinite(r) != np.isfinite(h)).any():
                n_mismatch += 1
                continue
            finite = np.isfinite(r)
            d = np.abs(h - r)[finite]
            if d.size:
                worst_abs = max(worst_abs, float(d.max()))
                # Scale-normalized: bf16 outputs land on a coarse grid, so
                # judge error against the output's own magnitude.
                scale = max(float(np.abs(r[finite]).max()), 1e-6)
                norm = float(d.max()) / scale
                worst_norm = max(worst_norm, norm)
                if norm > args.rtol:
                    n_mismatch += 1
        res["max_abs_err"] = f"{worst_abs:.3e}"
        res["max_norm_err"] = f"{worst_norm:.3e}"
        res["check"] = "PASS" if n_mismatch == 0 else f"FAIL({n_mismatch})"
        log(f"{name}: check {res['check']} abs {res['max_abs_err']} "
            f"norm {res['max_norm_err']}")
    emit()


if __name__ == "__main__":
    main()
