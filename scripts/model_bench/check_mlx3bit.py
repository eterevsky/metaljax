"""Differential check for the packed sub-byte (3-bit) quantized path.

The reference is MLX's own quantizer: `mx.quantize(w, group_size=64, bits=3)`
produces exactly the layout an mlx-community checkpoint ships -- uint32 words
holding a contiguous LSB-first bit stream whose elements STRADDLE word
boundaries, plus per-group `scales` and `biases` in the native affine form
`w = scales * q + biases` (an arbitrary float bias, NOT an integer zero point).

This script builds the in-graph unpack that a model must emit, runs it through
the metal backend (where the qmm recognizer is expected to verify it, repack it
at 3 bits and call `mx::quantized_matmul`) and through the CPU backend (where
it runs literally), and compares.  Both sides are also compared against MLX's
own `mx.dequantize`, which is what says the unpack itself is right rather than
merely self-consistent.

Run it under the machine lock; it is a GPU test.
"""

import argparse
import os
import sys

import numpy as np

GROUP = 64
BITS = 3


def unpack_codes(jnp, words, k):
    """uint32 words `[..., k*3/32]` -> unsigned codes `[..., k]`.

    Element i lives at bits [3i, 3i+3) of the row's flat little-endian stream,
    so one in three straddles a word boundary and has to be stitched from two.
    """
    lead = tuple(words.shape[:-1])
    per = 32 // np.gcd(32, BITS)          # codes per whole word run == 32
    nw = BITS // np.gcd(32, BITS)         # words those codes fill      == 3
    w = jnp.reshape(words, lead + (k // per, nw))
    outs = []
    for i in range(per):
        bit = i * BITS
        w0, off = bit // 32, bit % 32
        v = jnp.right_shift(w[..., w0], jnp.uint32(off))
        if off + BITS > 32:
            v = jnp.bitwise_or(
                v, jnp.left_shift(w[..., w0 + 1], jnp.uint32(32 - off)))
        outs.append(jnp.bitwise_and(v, jnp.uint32((1 << BITS) - 1)))
    return jnp.reshape(jnp.stack(outs, axis=-1), lead + (k,))


def dequant(jnp, words, scales, biases, k, dtype):
    """The affine reconstruction, at FULL contraction width.

    The scale/bias maps are broadcast to `[..., k]` rather than left as a
    grouped `[..., k/64, 64]` axis on purpose: the recognizer's affine path
    reads its maps along the contraction axis, and handing it the weight's real
    shape is what lets it measure the group size instead of being told.
    """
    q = unpack_codes(jnp, words, k).astype(dtype)

    def full(m):
        m = m.astype(dtype)
        return jnp.reshape(jnp.broadcast_to(m[..., None], m.shape + (GROUP,)),
                           tuple(m.shape[:-1]) + (k,))

    return q * full(scales) + full(biases)


def make_case(shape, seed):
    """Quantize a random weight with MLX itself; return the packed triple."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    rng = np.random.RandomState(seed)
    w = rng.randn(*shape).astype(np.float32)
    wq, scales, biases = mx.quantize(mx.array(w), group_size=GROUP, bits=BITS)
    ref = mx.dequantize(wq, scales, biases, group_size=GROUP, bits=BITS)
    mx.eval(wq, scales, biases, ref)
    return (np.array(wq), np.array(scales), np.array(biases), np.array(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "metal,cpu")
    import jax
    import jax.numpy as jnp

    dt = args.dtype
    metal = jax.devices("metal")[0]
    cpu = jax.devices("cpu")[0]
    print(f"[check] metal={metal} cpu={cpu} dtype={dt}", flush=True)

    fails = []

    def run(name, fn, arrays, tol):
        outs = {}
        for label, dev in (("cpu", cpu), ("metal", metal)):
            with jax.default_device(dev):
                xs = [jax.device_put(a, dev) for a in arrays]
                # np.array() forces the transfer: block_until_ready is a no-op
                # on this backend (events are born ready).
                outs[label] = np.array(jax.jit(fn)(*xs), dtype=np.float64)
        d = np.abs(outs["metal"] - outs["cpu"])
        scale = np.maximum(np.abs(outs["cpu"]), 1.0)
        rel = float(np.max(d / scale))
        ok = rel <= tol
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: max rel metal-vs-cpu "
              f"{rel:.3e} (tol {tol:g})", flush=True)
        if not ok:
            fails.append(name)
        return outs["metal"]

    # --- 1. the unpack itself, against mx.dequantize -----------------------
    n, k = 64, 256
    words, scales, biases, ref = make_case((n, k), 3)
    print(f"[check] packed {words.shape} {words.dtype}, scales {scales.shape}, "
          f"expected words {k * BITS // 32}", flush=True)
    for label, dev in (("cpu", cpu), ("metal", metal)):
        with jax.default_device(dev):
            got = np.array(jax.jit(
                lambda w, s, b: dequant(jnp, w, s, b, k, jnp.float32))(
                    jax.device_put(words, dev), jax.device_put(scales, dev),
                    jax.device_put(biases, dev)), dtype=np.float64)
        # 1 ULP of f32, not zero: both sides compute `q*s + b` but MLX's
        # dequantize kernel and XLA's literal chain are free to contract the
        # multiply-add differently.  What this pins down is the UNPACK -- a
        # wrong bit layout is off by a whole quantization step, not an ulp.
        err = float(np.max(np.abs(got - ref.astype(np.float64))))
        ok = err <= 1e-6
        print(f"[{'PASS' if ok else 'FAIL'}] unpack vs mx.dequantize ({label}): "
              f"max abs {err:.3e}", flush=True)
        if not ok:
            fails.append(f"unpack-{label}")

    # --- 2. a projection: the dense quantized_matmul path ------------------
    # The weight is [out, in] with its groups along `in`, so the contraction
    # MUST run along that last axis ("tk,nk->tn").  Contracting the other way
    # leaves the groups lying along the output axis and the recognizer
    # correctly declines -- which is what the first draft of this test did.
    x = np.random.RandomState(7).randn(6, k).astype(dt) * 0.5
    run("qmm 3-bit projection",
        lambda w, s, b, a: jnp.einsum(
            "tk,nk->tn", a, dequant(jnp, w, s, b, k, a.dtype)),
        [words, scales, biases, x], 2e-5 if dt == "float32" else 6e-3)

    # --- 3. batched experts: what AnalyzeMoe absorbs into gather_qmm -------
    ew, es, eb, _ = make_case((4, 32, 128), 11)
    xe = np.random.RandomState(9).randn(4, 3, 128).astype(dt) * 0.4
    run("qmm 3-bit batched experts",
        lambda w, s, b, a: jnp.einsum(
            "etm,ehm->eth", a, dequant(jnp, w, s, b, 128, a.dtype)),
        [ew, es, eb, xe], 2e-5 if dt == "float32" else 6e-3)

    # --- 4. a non-multiple-of-128 contraction, to exercise gs choice -------
    w2, s2, b2, _ = make_case((48, 192), 13)
    x2 = np.random.RandomState(17).randn(5, 192).astype(dt) * 0.5
    run("qmm 3-bit K=192 (gs must be 64, not 128)",
        lambda w, s, b, a: jnp.einsum(
            "tk,nk->tn", a, dequant(jnp, w, s, b, 192, a.dtype)),
        [w2, s2, b2, x2], 2e-5 if dt == "float32" else 6e-3)

    # --- 5. the row-20 expert pattern -------------------------------------
    # `th,emh->etm`: a 2-D activation against a 3-D weight whose expert axis is
    # a FREE OUTPUT dim, not a batching dim shared with the activation.  That
    # makes the pack's row count larger than the axis the blocking counts in,
    # which is the exact case the first alias implementation got wrong -- and
    # case 3 above does NOT cover it, because there the expert axis IS a
    # batching dim and the two counts coincide.
    ew2, es2, eb2, _ = make_case((4, 32, 128), 21)
    xt = np.random.RandomState(23).randn(5, 128).astype(dt) * 0.4
    run("qmm 3-bit experts, free expert dim (th,emh->etm)",
        lambda w, s, b, a: jnp.einsum(
            "th,emh->etm", a, dequant(jnp, w, s, b, 128, a.dtype)),
        [ew2, es2, eb2, xt], 2e-5 if dt == "float32" else 6e-3)

    print()
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("all 3-bit differentials PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
