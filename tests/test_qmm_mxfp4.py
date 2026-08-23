"""MXFP4 (OCP micro-scaling) weights through the quantized-matmul path.

MXFP4 stores a 4-bit E2M1 element -- a NON-UNIFORM 16-value grid, `0, +-0.5,
+-1, +-1.5, +-2, +-3, +-4, +-6` -- times one E8M0 (power-of-two) scale per 32
elements of a row. No affine `scale * (q - zp)` can represent that grid, so
the recognizer verifies and packs it separately.

The path under test is the native PJRT plugin, driven through plain JAX: it
implements the MXFP4 recognizer in C++ (controlled by the METALJAX_QMM*
environment variables), so everything here is end to end -- build the graph a
packed loader emits (nibble unpack, a 16-entry E2M1 table gather, a per-group
power-of-two scale, then the dot), jit it onto the metal device, and compare
against the CPU backend and against exact numpy arithmetic. Fallbacks
(off-grid values, non-power-of-two scales, a group size that is not 32) must
stay correct through the literal chain.

The batched form (`etm,ehm->eth`, one weight per expert) is covered too: a
mixture-of-experts checkpoint is the reason MXFP4 shows up at all, and its
second projection is a batched dot.

The Stage-1 Python engine's pack-primitive and introspection tests (pack
layout, code/scale extraction, pack-sharing counters, blocked packing, the
build-cache fingerprint) were removed at the Stage-1 retirement -- those
internals live in the native plugin now and are exercised only through the
numbers they produce.
"""

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

needs_qmm = pytest.mark.skipif(
    os.environ.get("METALJAX_QMM") == "0", reason="METALJAX_QMM=0")
needs_batch = pytest.mark.skipif(
    os.environ.get("METALJAX_QMM_BATCH") == "0", reason="METALJAX_QMM_BATCH=0")

# The E2M1 grid, indexed by the 4-bit code (bit 3 is the sign).
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def _metal():
    return jax.devices("metal")[0]


def _cpu():
    return jax.devices("cpu")[0]


# --------------------------------------------------------------------------
# numpy reference (the same arithmetic keras-hub's gpt-oss loader does)
# --------------------------------------------------------------------------


def np_dequant_mxfp4(blocks, scale_bytes):
    """`blocks` [..., K//2] uint8, `scale_bytes` [..., K//32] uint8 -> f32.

    The low nibble of byte j is element 2j and the high nibble is element
    2j+1 -- HF's `_blocks` convention, and MLX's packing order.
    """
    lo = np.bitwise_and(blocks, 0x0F)
    hi = np.right_shift(blocks, 4)
    nib = np.stack([lo, hi], axis=-1).reshape(blocks.shape[:-1] + (-1,))
    vals = E2M1[nib]
    scale = np.exp2(scale_bytes.astype(np.float32) - 127.0)
    k = vals.shape[-1]
    grouped = vals.reshape(vals.shape[:-1] + (k // 32, 32))
    return (grouped * scale[..., None]).reshape(vals.shape)


def np_pack_blocks(codes):
    """Codes [..., K] uint8 -> HF-style `blocks` bytes [..., K//2]."""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def _random_mxfp4(shape_nk, rng, exp_lo=118, exp_hi=132):
    """Codes + E8M0 bytes + the exact f32 weight they denote."""
    *lead, k = shape_nk
    codes = rng.integers(0, 16, size=tuple(lead) + (k,)).astype(np.uint8)
    codes.reshape(-1, k)[0, :16] = np.arange(16)      # force every code in
    sb = rng.integers(exp_lo, exp_hi,
                      size=tuple(lead) + (k // 32,)).astype(np.uint8)
    blocks = np_pack_blocks(codes)
    return codes, blocks, sb, np_dequant_mxfp4(blocks, sb)


# --------------------------------------------------------------------------
# the in-graph dequant chain (what a packed loader emits)
# --------------------------------------------------------------------------


def _tables(dtype):
    """The two constants the chain gathers through."""
    values = jnp.asarray(E2M1, dtype=dtype)
    # Kept in f32: byte 0 is 2**-127, a subnormal that Metal may flush. Real
    # MXFP4 checkpoints live far from it (gpt-oss uses 117..126), and a
    # flushed scale would fail the power-of-two check and fall back rather
    # than go wrong. Byte 255 is NaN by the OCP spec.
    table = np.ldexp(np.ones(256), np.arange(256) - 127)   # f64: no overflow
    table[255] = np.nan
    return values, jnp.asarray(table.astype(np.float32))


def mxfp4_weight(blocks, scale_bytes, k, dtype):
    """blocks [..., K//2] uint8 + E8M0 bytes [..., K//32] -> weight [..., K].

    Structurally identical to the keras version the bench harness installs:
    nibble unpack, a table gather for the E2M1 value, and a broadcast
    multiply by the per-group scale.
    """
    vtable, stable = _tables(dtype)
    lead = tuple(blocks.shape[:-1])
    lo = jnp.bitwise_and(blocks, jnp.uint8(0x0F))
    hi = jnp.right_shift(blocks, jnp.uint8(4))
    nib = jnp.reshape(jnp.stack([lo, hi], axis=-1), lead + (k,))
    vals = jnp.take(vtable, nib.astype(jnp.int32), axis=0)
    scale = jnp.take(stable, scale_bytes.astype(jnp.int32), axis=0)
    w = (jnp.reshape(vals, lead + (k // 32, 32))
         * scale[..., None].astype(dtype))
    return jnp.reshape(w, lead + (k,))


def dense_mxfp4(blocks, scale_bytes, x, k):
    """`th,nh->tn`: the plain (unbatched) projection."""
    w = mxfp4_weight(blocks, scale_bytes, k, x.dtype)
    return jnp.einsum("th,nh->tn", x, w)


def experts_mxfp4(blocks, scale_bytes, x, k):
    """`etm,ehm->eth`: one weight per expert, i.e. a BATCHED dot."""
    w = mxfp4_weight(blocks, scale_bytes, k, x.dtype)
    return jnp.einsum("etm,ehm->eth", x, w)


def _run(f, args, device):
    with jax.default_device(device):
        moved = [jax.device_put(a, device) for a in args]
        return np.asarray(jax.jit(f)(*moved))


def _no_worse_than(got, ref, exact, dtype):
    """The rewrite must land at least as close to the exact answer as the
    float matmul it replaced.

    It cannot be asserted bit-equal: a quantized matmul and a float matmul
    are different kernels and accumulate in a different order. The
    reconstructed weight itself is exact -- MXFP4 dequant is exact in f32
    and in bf16, since an E2M1 value carries one mantissa bit -- so any
    residual difference here is the dot's own summation, and it is bounded
    by the dtype.
    """
    scale = np.abs(exact).max()
    err_fused = np.abs(got.astype(np.float32) - exact).max() / scale
    err_float = np.abs(ref.astype(np.float32) - exact).max() / scale
    assert err_fused <= err_float * 1.5 + 1e-7, (err_fused, err_float)
    tol = 2e-2 if dtype == "bfloat16" else 1e-5
    np.testing.assert_allclose(got.astype(np.float32),
                               ref.astype(np.float32), rtol=tol,
                               atol=tol * scale)


@needs_qmm
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_dense_mxfp4_matches_a_float_matmul(dtype):
    n, k, t = 64, 128, 3
    rng = np.random.default_rng(1)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    x = (rng.standard_normal((t, k)) * 0.4).astype(dtype)
    args = (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x))

    got = _run(lambda *a: dense_mxfp4(*a, k=k), args, _metal())

    # Same backend, same shapes, weight handed over as a plain float array.
    ref = _run(lambda x_: jnp.einsum("th,nh->tn", x_, jnp.asarray(w, dtype)),
               (jnp.asarray(x),), _metal())
    _no_worse_than(got, ref, x.astype(np.float32) @ w.T, dtype)


@needs_qmm
@needs_batch
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_batched_experts_mxfp4(dtype):
    """The MoE second projection: a batching dimension over experts."""
    e, h, m, t = 4, 32, 64, 2
    rng = np.random.default_rng(2)
    _codes, blocks, sb, w = _random_mxfp4((e, h, m), rng)
    x = (rng.standard_normal((e, t, m)) * 0.4).astype(dtype)
    args = (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x))

    got = _run(lambda *a: experts_mxfp4(*a, k=m), args, _metal())

    ref = _run(
        lambda x_: jnp.einsum("etm,ehm->eth", x_, jnp.asarray(w, dtype)),
        (jnp.asarray(x),), _metal())
    exact = np.einsum("etm,ehm->eth", x.astype(np.float32), w)
    _no_worse_than(got, ref, exact, dtype)


@needs_qmm
def test_identical_weights_pack_once():
    """Two dots over the SAME weight inside one executable.

    jax lowers a decode loop's prefill and its while body as separate dots
    over one set of weights, so the plugin sees the same chain twice over
    the same buffers (and may build one shared pack for both); both dots
    must come out right.
    """
    n, k, t = 64, 128, 2
    rng = np.random.default_rng(9)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    x = (rng.standard_normal((t, k)) * 0.4).astype("bfloat16")

    def f(blocks_, sb_, x_):
        a = dense_mxfp4(blocks_, sb_, x_, k)
        b = dense_mxfp4(blocks_, sb_, jnp.tanh(x_), k)
        return a + b

    got = _run(f, (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x)),
               _metal())

    wf = jnp.asarray(w, "bfloat16")
    ref = _run(lambda x_: (jnp.einsum("th,nh->tn", x_, wf)
                           + jnp.einsum("th,nh->tn", jnp.tanh(x_), wf)),
               (jnp.asarray(x),), _metal())
    exact = (x.astype(np.float32) @ w.T
             + np.tanh(x.astype(np.float32)) @ w.T)
    _no_worse_than(got, ref, exact, "bfloat16")


@needs_qmm
def test_identical_weights_from_two_buffers_share_a_pack():
    """Same bytes, two placements: two device buffers holding one weight.

    Any pack dedupe the plugin does across genuinely distinct buffers must
    be content-addressed; both dots must come out right.
    """
    n, k, t = 64, 128, 2
    rng = np.random.default_rng(91)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    x = (rng.standard_normal((t, k)) * 0.4).astype("float32")

    def f(b1, s1, b2, s2, x_):
        return dense_mxfp4(b1, s1, x_, k) + dense_mxfp4(b2, s2, x_, k)

    dev = _metal()
    with jax.default_device(dev):
        # Two separate placements of the SAME bytes.
        args = [jax.device_put(v, dev) for v in
                (blocks, sb, blocks.copy(), sb.copy(), x)]
        got = np.asarray(jax.jit(f)(*args))
    exact = 2.0 * (x.astype(np.float32) @ w.T)
    np.testing.assert_allclose(got, exact, rtol=1e-5,
                               atol=1e-6 * np.abs(exact).max())


@needs_qmm
def test_different_weights_do_not_share_a_pack():
    """Two structurally identical chains whose weights differ: each dot must
    use its own weight (any pack sharing must be content-addressed)."""
    n, k, t = 64, 128, 2
    rng = np.random.default_rng(10)
    _c1, blocks1, sb1, w1 = _random_mxfp4((n, k), rng)
    _c2, blocks2, sb2, w2 = _random_mxfp4((n, k), rng)
    assert not np.array_equal(w1, w2)
    x = (rng.standard_normal((t, k)) * 0.4).astype("float32")

    def f(b1, s1, b2, s2, x_):
        return (dense_mxfp4(b1, s1, x_, k) + dense_mxfp4(b2, s2, x_, k))

    args = (jnp.asarray(blocks1), jnp.asarray(sb1), jnp.asarray(blocks2),
            jnp.asarray(sb2), jnp.asarray(x))
    got = _run(f, args, _metal())
    exact = x.astype(np.float32) @ w1.T + x.astype(np.float32) @ w2.T
    np.testing.assert_allclose(got, exact, rtol=1e-5,
                               atol=1e-6 * np.abs(exact).max())


@needs_qmm
def test_mxfp4_matches_the_cpu_backend():
    """The literal chain on jax-CPU is the independent check that the whole
    pipeline -- unpack, decode, scale, dot -- means what it should."""
    n, k, t = 96, 64, 2
    rng = np.random.default_rng(3)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    x = (rng.standard_normal((t, k)) * 0.5).astype("float32")
    args = (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x))
    got = _run(lambda *a: dense_mxfp4(*a, k=k), args, _metal())
    want = _run(lambda *a: dense_mxfp4(*a, k=k), args, _cpu())
    exact = x.astype(np.float32) @ w.T
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(got, exact, rtol=1e-6, atol=1e-5)


@needs_qmm
def test_mxfp4_in_a_decode_loop():
    """A while_loop carries the packed weights: the pack has to be followed
    out through the loop-invariant carry, and travel into the compiled body
    as an explicit input."""
    n = k = 64
    rng = np.random.default_rng(4)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng, exp_lo=123, exp_hi=128)
    x = (rng.standard_normal((2, k)) * 0.3).astype("bfloat16")

    def f(blocks_, sb_, x_, steps):
        def body(state):
            i, hs = state
            y = dense_mxfp4(blocks_, sb_, hs, k)
            return i + 1, jnp.tanh(y * jnp.bfloat16(0.5))
        return jax.lax.while_loop(lambda s: s[0] < steps, body, (0, x_))[1]

    args = (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x),
            jnp.int32(6))
    dev = _metal()
    with jax.default_device(dev):
        moved = [jax.device_put(a, dev) for a in args]
        jf = jax.jit(f)
        got = np.asarray(jf(*moved)).astype(np.float32)
        got2 = np.asarray(jf(*moved)).astype(np.float32)
    np.testing.assert_array_equal(got, got2)
    want = _run(f, args, _cpu()).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)


# --------------------------------------------------------------------------
# fallbacks: still correct, just not rewritten
# --------------------------------------------------------------------------


def _fallback(f, args):
    """The chain runs literally on the metal backend, and is still right."""
    got = _run(f, args, _metal())
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(np.asarray(got, np.float32),
                               np.asarray(want, np.float32),
                               rtol=1e-5, atol=1e-5)


@needs_qmm
def test_ordinary_float_graphs_recognize_nothing():
    """The guard that keeps every unquantized model on its old path.

    An RMS norm or a scaled residual is exactly `something *
    broadcast(something smaller)` -- the shape the recognizer probes on
    either dot operand. None of these may be adopted as a quantized weight;
    a wrong adoption would show up here as wrong numbers.
    """
    rng = np.random.default_rng(11)
    d, t = 128, 4
    w1 = jnp.asarray(rng.standard_normal((d, d)) * 0.1, "float32")
    w2 = jnp.asarray(rng.standard_normal((d, d)) * 0.1, "float32")
    g = jnp.asarray(rng.standard_normal(d) * 0.1, "float32")
    x = jnp.asarray(rng.standard_normal((t, d)) * 0.5, "float32")

    def f(x_, w1_, w2_, g_):
        # RMS norm (scale broadcast one-per-row), then two projections,
        # then a residual scaled by a learned per-channel gain.
        n = x_ * jax.lax.rsqrt(jnp.mean(x_ ** 2, -1, keepdims=True) + 1e-6)
        h = jnp.einsum("td,de->te", n, w1_ * g_[:, None])
        h = jnp.tanh(h)
        return jnp.einsum("td,de->te", h, w2_) + x_ * g_

    _fallback(f, (x, w1, w2, g))


@needs_qmm
def test_fallback_values_off_the_grid():
    """A 17th "code" -- a table whose entries are not the E2M1 grid."""
    n, k, t = 64, 64, 2
    rng = np.random.default_rng(5)
    _codes, blocks, sb, _w = _random_mxfp4((n, k), rng)
    x = (rng.standard_normal((t, k)) * 0.4).astype("float32")
    bad = E2M1.copy()
    bad[5] = 3.25                                    # off-grid magnitude

    def f(blocks_, sb_, x_):
        lead = tuple(blocks_.shape[:-1])
        lo = jnp.bitwise_and(blocks_, jnp.uint8(0x0F))
        hi = jnp.right_shift(blocks_, jnp.uint8(4))
        nib = jnp.reshape(jnp.stack([lo, hi], axis=-1), lead + (k,))
        vals = jnp.take(jnp.asarray(bad, x_.dtype), nib.astype(jnp.int32),
                        axis=0)
        _v, stable = _tables(x_.dtype)
        scale = jnp.take(stable, sb_.astype(jnp.int32), axis=0)
        w = (jnp.reshape(vals, lead + (k // 32, 32))
             * scale[..., None].astype(x_.dtype))
        return jnp.einsum("th,nh->tn", x_, jnp.reshape(w, lead + (k,)))

    _fallback(f, (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x)))


@needs_qmm
def test_fallback_scales_that_are_not_powers_of_two():
    """A per-group float scale (an affine-style quantization of an E2M1
    grid) is not MXFP4 and cannot be packed as one."""
    n, k, t = 64, 64, 2
    rng = np.random.default_rng(6)
    codes, blocks, _sb, _w = _random_mxfp4((n, k), rng)
    scale = ((rng.random((n, k // 32)).astype(np.float32) + 0.5) * 0.03)
    x = (rng.standard_normal((t, k)) * 0.4).astype("float32")

    def f(blocks_, scale_, x_):
        lead = tuple(blocks_.shape[:-1])
        lo = jnp.bitwise_and(blocks_, jnp.uint8(0x0F))
        hi = jnp.right_shift(blocks_, jnp.uint8(4))
        nib = jnp.reshape(jnp.stack([lo, hi], axis=-1), lead + (k,))
        vals = jnp.take(jnp.asarray(E2M1, x_.dtype), nib.astype(jnp.int32),
                        axis=0)
        w = jnp.reshape(vals, lead + (k // 32, 32)) * scale_[..., None]
        return jnp.einsum("th,nh->tn", x_, jnp.reshape(w, lead + (k,)))

    _fallback(f, (jnp.asarray(blocks), jnp.asarray(scale), jnp.asarray(x)))


@needs_qmm
def test_fallback_group_size_is_not_32():
    """MLX's MXFP4 kernel has exactly one legal group size."""
    n, k, t, gs = 64, 128, 2, 16
    rng = np.random.default_rng(8)
    codes = rng.integers(0, 16, size=(n, k)).astype(np.uint8)
    blocks = np_pack_blocks(codes)
    sb = rng.integers(118, 132, size=(n, k // gs)).astype(np.uint8)
    x = (rng.standard_normal((t, k)) * 0.4).astype("float32")

    def f(blocks_, sb_, x_):
        lead = tuple(blocks_.shape[:-1])
        lo = jnp.bitwise_and(blocks_, jnp.uint8(0x0F))
        hi = jnp.right_shift(blocks_, jnp.uint8(4))
        nib = jnp.reshape(jnp.stack([lo, hi], axis=-1), lead + (k,))
        vals = jnp.take(jnp.asarray(E2M1, x_.dtype), nib.astype(jnp.int32),
                        axis=0)
        _v, stable = _tables(x_.dtype)
        scale = jnp.take(stable, sb_.astype(jnp.int32), axis=0)
        w = (jnp.reshape(vals, lead + (k // gs, gs))
             * scale[..., None].astype(x_.dtype))
        return jnp.einsum("th,nh->tn", x_, jnp.reshape(w, lead + (k,)))

    # The scale broadcasts one value per 16, and MXFP4 has exactly one
    # group size: the chain must run literally.
    _fallback(f, (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x)))


# --------------------------------------------------------------------------
# one weight across several executables
# --------------------------------------------------------------------------


def _through_executables(shapes, blocks, sb, k, dtype="bfloat16"):
    """One MXFP4 weight through several executables; return [(x, out), ...].

    jax numbers a program's private helpers per module (`@_take`,
    `@_take_0`, ...), so several executables over one weight is the shape
    that punishes any name-keyed reuse in the plugin's pack cache. The
    executables are kept alive together and their numbers checked.
    """
    dev = _metal()
    fns, outs = [], []
    with jax.default_device(dev):
        moved = [jax.device_put(v, dev) for v in (blocks, sb)]
        for t in shapes:
            x = (np.random.default_rng(200 + t).standard_normal(
                (t, k)) * 0.4).astype(dtype)
            f = jax.jit(lambda b, s, xx: dense_mxfp4(b, s, xx, k=k))
            fns.append(f)        # keep every executable (and its pack) alive
            outs.append((x, np.asarray(f(*moved, jax.device_put(x, dev)))))
    return outs


@needs_qmm
def test_second_executable_reuses_the_mxfp4_pack():
    """Several executables over one placed weight: whatever pack reuse the
    plugin does across them, every executable must compute the same MXFP4
    matmul."""
    n, k = 64, 128
    rng = np.random.default_rng(77)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    outs = _through_executables((2, 5, 3), blocks, sb, k)
    for x, got in outs:
        exact = np.asarray(x, np.float32) @ w.T
        np.testing.assert_allclose(got.astype(np.float32), exact,
                                   rtol=2e-2, atol=2e-2 * np.abs(exact).max())
