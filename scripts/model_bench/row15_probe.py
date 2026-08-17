#!/usr/bin/env python
"""Row-15 mechanism probes — the cheap rungs of notes/row15-wrong-output-2026-08-17.md.

Row 15 (`qwix-int8-qwen3-8b`) completes and emits collapsed logits (every
decoded token is id 0, `!`), where row 14 — the SAME adapter, the same qwix
int8 overrides, a 0.6B config instead of an 8B one — emits coherent text.
This script is rung A of the ladder: it asks whether the int8 arithmetic
itself is wrong at 8B widths, using no checkpoint, no maxtext and ~2 GB, so
that the expensive model rungs are only spent on hypotheses this cannot kill.

    JAX_PLATFORMS=metal,cpu python scripts/model_bench/row15_probe.py dots
    JAX_PLATFORMS=metal,cpu python scripts/model_bench/row15_probe.py qwix
    JAX_PLATFORMS=metal,cpu python scripts/model_bench/row15_probe.py qwix --layers 36

Stack under test: whatever `METALJAX_PLUGIN_PATH` selects (unset = Stage 1's
`plugin/build/libmetal_pjrt.dylib`; a frozen native dylib = the native stack),
exactly as `scripts/release/g5_model.sh` does it.

`dots` is EXACT: an s8xs8->s32 contraction has one right answer, and both
engines take `ops/linalg.py::_int_dot_via_f32` (chunk 1024, f32 GEMM per
K-slice, i32 accumulation) whose whole correctness rests on MLX's f32 matmul
being exact on integers below 2**24. That assumption has never been tested at
row-15 widths; the numpy reference here is computed in int64 and is not an
approximation. `qwix` reproduces the row's real pattern — per-tensor dynamic
absmax quantization around that dot — and reports the collapse statistics
(fraction of zero codes, NaNs, flat-logit detection) rather than a tolerance,
because a collapse is what row 15 shows.

Every result line is JSON on stdout (prefix `RESULT:`); diagnostics on stderr.
"""

import argparse
import json
import os
import sys

import numpy as np


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def emit(rec):
    print("RESULT:" + json.dumps(rec), flush=True)


# The contractions each row actually runs, from the maxtext model configs
# (emb / heads / mlp / layers) — see notes/row15-wrong-output-2026-08-17.md §2.
# (label, M, K, N)
SHAPES_8B = [
    ("8b-qkv-proj", 8, 4096, 4096),      # q: 32 heads x 128
    ("8b-kv-proj", 8, 4096, 1024),       # kv: 8 heads x 128
    ("8b-attn-out", 8, 4096, 4096),
    ("8b-mlp-wi", 8, 4096, 12288),
    ("8b-mlp-wo", 8, 12288, 4096),       # 12 K-chunks
    ("8b-logits", 8, 4096, 151936),      # untied; 622 M elements of weight
]
SHAPES_06B = [
    ("06b-qkv-proj", 8, 1024, 2048),     # single K-chunk
    ("06b-kv-proj", 8, 1024, 1024),
    ("06b-attn-out", 8, 2048, 1024),
    ("06b-mlp-wi", 8, 1024, 3072),
    ("06b-mlp-wo", 8, 3072, 1024),
    ("06b-logits", 8, 1024, 151936),     # tied embedding
]


def _backends():
    import jax
    metal = [d for d in jax.devices() if d.platform == "metal"]
    cpu = [d for d in jax.devices("cpu")]
    if not metal:
        raise SystemExit("no metal device — run with JAX_PLATFORMS=metal,cpu")
    return metal[0], (cpu[0] if cpu else None)


def probe_dots(args):
    """s8 x s8 -> s32 against an exact int64 numpy reference."""
    import jax
    import jax.numpy as jnp

    metal, _ = _backends()
    rng = np.random.default_rng(args.seed)
    shapes = (SHAPES_8B if args.widths in ("8b", "both") else []) + \
             (SHAPES_06B if args.widths in ("06b", "both") else [])

    def f(a, b):
        return jnp.dot(a, b, preferred_element_type=jnp.int32)

    fj = jax.jit(f)
    bad = 0
    for label, m, k, n in shapes:
        # Full-range codes: |code| <= 127 is what absmax quantization emits,
        # and -128 is reachable, so the partial-sum bound is exercised.
        a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
        ref = np.matmul(a.astype(np.int64), b.astype(np.int64))
        assert np.abs(ref).max() < 2 ** 31, "reference itself overflows i32"
        with jax.default_device(metal):
            got = np.asarray(fj(jnp.asarray(a), jnp.asarray(b)))
        diff = got.astype(np.int64) - ref
        nbad = int(np.count_nonzero(diff))
        bad += nbad > 0
        emit({"probe": "dots", "shape": label, "m": m, "k": k, "n": n,
              "k_chunks": -(-k // 1024), "exact": nbad == 0,
              "mismatched": nbad, "of": int(ref.size),
              "max_abs_err": int(np.abs(diff).max()) if nbad else 0,
              "ref_max_abs": int(np.abs(ref).max())})
        log(f"  {label:16s} K={k:6d} N={n:6d}  "
            f"{'EXACT' if nbad == 0 else f'WRONG ({nbad} elements)'}")
    return bad


def _quantize(x, axis=None):
    """qwix's per-tensor dynamic int8: scale = absmax/127, round-half-even."""
    import jax.numpy as jnp
    absmax = jnp.max(jnp.abs(x), axis=axis, keepdims=axis is not None)
    scale = absmax / 127.0
    code = jnp.round(x / scale).astype(jnp.int8)
    return code, scale


def probe_qwix(args):
    """The real pattern: dynamic-quantized activations x dynamic-quantized weights.

    Reports metal-vs-CPU per layer on identical inputs, plus the collapse
    statistics row 15 shows (non-finite values, all-zero output, flat logits).
    The criterion is deliberately NOT a tolerance: what row 15 does is
    collapse, and a collapse is visible in `std` and `zero_frac` long before
    any rtol argument matters.

    `--structure scan` puts the layer stack in a `lax.scan` (what maxtext
    lowers, and what the engine replays in compiled chunks of `kmax=16`);
    `--structure unroll` runs the layers as straight-line code so each
    layer's output can be compared on its own and the ENTRY POINT of a
    divergence is located rather than inferred.
    """
    import jax
    import jax.numpy as jnp

    metal, cpu = _backends()
    if cpu is None:
        raise SystemExit("need a cpu device for the reference — "
                         "JAX_PLATFORMS=metal,cpu")
    rng = np.random.default_rng(args.seed)
    e, m_dim = (4096, 12288) if args.widths != "06b" else (1024, 3072)
    t, layers = args.tokens, args.layers

    x0 = (rng.standard_normal((t, e)) * 0.02).astype(np.float32)
    w1 = (rng.standard_normal((layers, e, m_dim)) / np.sqrt(e)).astype(np.float32)
    w2 = (rng.standard_normal((layers, m_dim, e)) / np.sqrt(m_dim)).astype(np.float32)

    def layer(x, w_i, w_o):
        xc, xs = _quantize(x)
        wc, ws = _quantize(w_i)
        h = jnp.dot(xc, wc, preferred_element_type=jnp.int32).astype(jnp.float32)
        h = (h * xs * ws).astype(jnp.bfloat16)
        h = jax.nn.silu(h)
        hc, hs = _quantize(h.astype(jnp.float32))
        wc2, ws2 = _quantize(w_o)
        y = jnp.dot(hc, wc2, preferred_element_type=jnp.int32).astype(jnp.float32)
        return x + (y * hs * ws2).astype(jnp.bfloat16).astype(jnp.float32)

    def scanned(x, wi, wo):
        def body(carry, ws):
            return layer(carry, ws[0], ws[1]), None
        out, _ = jax.lax.scan(body, x, (wi, wo))
        return out

    def unrolled(x, wi, wo):
        outs = []
        for i in range(layers):
            x = layer(x, wi[i], wo[i])
            outs.append(x)
        return outs

    fj = jax.jit(scanned if args.structure == "scan" else unrolled)
    per = {}
    for name, dev in (("cpu", cpu), ("metal", metal)):
        with jax.default_device(dev):
            r = fj(jnp.asarray(x0), jnp.asarray(w1), jnp.asarray(w2))
        per[name] = ([np.asarray(v, dtype=np.float32) for v in r]
                     if isinstance(r, (list, tuple))
                     else [np.asarray(r, dtype=np.float32)])
        log(f"  {name}: done ({len(per[name])} checkpoint(s))")

    first_bad = None
    for i, (g, c) in enumerate(zip(per["metal"], per["cpu"])):
        denom = float(np.sqrt((c ** 2).mean())) or 1.0
        rec = {"probe": "qwix", "structure": args.structure,
               "widths": args.widths, "layer": i, "layers": layers,
               "tokens": t, "emb": e, "mlp": m_dim,
               "rel_rms": float(np.sqrt(((g - c) ** 2).mean()) / denom),
               "metal_nonfinite": int(np.count_nonzero(~np.isfinite(g))),
               "cpu_nonfinite": int(np.count_nonzero(~np.isfinite(c))),
               "metal_zero_frac": float(np.count_nonzero(g == 0) / g.size),
               "cpu_zero_frac": float(np.count_nonzero(c == 0) / c.size),
               "metal_std": float(np.nan_to_num(g).std()),
               "cpu_std": float(c.std())}
        # A COLLAPSE, not a tolerance: metal lost the signal the CPU kept, or
        # produced values that are not numbers at all.
        rec["collapsed"] = bool(rec["metal_nonfinite"] or
                                (rec["cpu_std"] > 0 and
                                 rec["metal_std"] < 0.01 * rec["cpu_std"]))
        emit(rec)
        if (rec["collapsed"] or rec["rel_rms"] > 0.5) and first_bad is None:
            first_bad = i
            log(f"  FIRST DIVERGENCE at layer {i}: {rec}")
    return 1 if first_bad is not None else 0


def probe_big(args):
    """One dot whose WEIGHT OPERAND alone exceeds a Metal command buffer.

    `MLX_MAX_MB_PER_BUFFER` counts ELEMENTS, deduped per distinct buffer, so
    the shipped 512 is ~537 M elements per command buffer.  Row 15's untied
    `logits_dense` is 4096 x 151936 = **622 M elements** — bigger than the
    whole budget on its own, so MLX must cut a command buffer inside that one
    matmul, in the same place, on every call.  Row 14 ties its embeddings
    (155 M elements) and never crosses the line.

    That makes this the one candidate that is simultaneously (a) an MLX
    command-buffer split, (b) DETERMINISTIC rather than a lottery draw, and
    (c) present in row 15 and absent in row 14 — which is the exact shape of
    the evidence.  This probe asks it directly: the over-budget dot and an
    under-budget control, in bf16 and in the int8 path, against jax-CPU.

    Sweep `MLX_MAX_MB_PER_BUFFER` across the crossing (512 -> 2048) in
    separate processes: MLX latches the budget at load.
    """
    import jax
    import jax.numpy as jnp

    metal, cpu = _backends()
    if cpu is None:
        raise SystemExit("need a cpu device — JAX_PLATFORMS=metal,cpu")
    rng = np.random.default_rng(args.seed)
    budget_elems = int(os.environ.get("MLX_MAX_MB_PER_BUFFER", "512")) << 20
    cases = [("over-budget-logits", 8, 4096, 151936),   # 622 M elements
             ("under-budget-logits", 8, 4096, 65536),   # 268 M, control
             ("tied-embed-06b", 8, 1024, 151936)]       # 155 M, row 14's shape

    for label, m, k, n in cases:
        elems = k * n
        x = (rng.standard_normal((m, k)) * 0.05).astype(np.float32)
        w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
        for mode in ("bf16", "int8"):
            if mode == "bf16":
                def f(a, b):
                    return jnp.dot(a.astype(jnp.bfloat16),
                                   b.astype(jnp.bfloat16),
                                   preferred_element_type=jnp.float32)
            else:
                def f(a, b):
                    ac, asc = _quantize(a)
                    bc, bsc = _quantize(b)
                    return (jnp.dot(ac, bc, preferred_element_type=jnp.int32)
                            .astype(jnp.float32) * asc * bsc)
            fj = jax.jit(f)
            got = {}
            for name, dev in (("cpu", cpu), ("metal", metal)):
                with jax.default_device(dev):
                    got[name] = np.asarray(fj(jnp.asarray(x), jnp.asarray(w)),
                                           dtype=np.float32)
            g, c = got["metal"], got["cpu"]
            denom = float(np.sqrt((c ** 2).mean())) or 1.0
            emit({"probe": "big", "case": label, "mode": mode,
                  "weight_elems": elems,
                  "budget_elems": budget_elems,
                  "over_budget": elems > budget_elems,
                  "mlx_max_mb_per_buffer":
                      os.environ.get("MLX_MAX_MB_PER_BUFFER", "(default 512)"),
                  "rel_rms": float(np.sqrt(((g - c) ** 2).mean()) / denom),
                  "max_abs_err": float(np.abs(g - c).max()),
                  "metal_nonfinite": int(np.count_nonzero(~np.isfinite(g))),
                  "metal_std": float(np.nan_to_num(g).std()),
                  "cpu_std": float(c.std())})
            log(f"  {label:22s} {mode:5s} elems={elems/1e6:.0f}M "
                f"{'OVER' if elems > budget_elems else 'under'} budget")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["dots", "qwix", "big"])
    ap.add_argument("--widths", default="both", choices=["8b", "06b", "both"])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--structure", default="unroll", choices=["scan", "unroll"])
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    log(f"row15_probe {args.probe} widths={args.widths} "
        f"plugin={os.environ.get('METALJAX_PLUGIN_PATH', '(stage 1 default)')}")
    rc = {"dots": probe_dots, "qwix": probe_qwix, "big": probe_big}[args.probe](args)
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
