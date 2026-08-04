"""MXFP4 (OCP micro-scaling) weights through the quantized-matmul recognizer.

MXFP4 stores a 4-bit E2M1 element -- a NON-UNIFORM 16-value grid, `0, +-0.5,
+-1, +-1.5, +-2, +-3, +-4, +-6` -- times one E8M0 (power-of-two) scale per 32
elements of a row. No affine `scale * (q - zp)` can represent that grid, so
the recognizer verifies and packs it separately and emits MLX's
`mode="mxfp4"` kernel.

Three layers of testing, mirroring test_qmm.py:

* the pack primitives (`mxfp4_codes`, `mxfp4_scale_bytes`, `pack_codes`)
  against a hand-computed layout and against `mx.dequantize(mode="mxfp4")`,
  which is what pins the layout MLX actually reads:

    - the packed codes are a little-endian uint32 bit stream, element i at
      bits [4i, 4i+4), which is the same layout the affine 4-bit packer
      emits;
    - the scale table is the RAW E8M0 byte (uint8), i.e. the f32 exponent
      field of the power-of-two scale;
    - there is no bias table.

  Both are also exactly what an HF MXFP4 checkpoint already stores, which is
  why the pack is a reinterpretation rather than a requantization.

  NOT against `mx.quantize(mode="mxfp4")`: that function is itself lossy on
  on-grid inputs (see `test_mlx_quantize_saturates_group_maxima`), so it is
  documented here as a bug rather than used as an oracle.

* the recognizer end to end THROUGH THE REAL BACKEND on the graph a packed
  loader emits: nibble unpack, a 16-entry E2M1 table gather, a per-group
  power-of-two scale, then the dot. Results must be BIT-EQUAL to a float
  matmul against the numpy-dequantized weight run on the same backend --
  MXFP4 dequant is exact in f32 and in bf16 (an E2M1 value carries one
  mantissa bit), so "close enough" is not the bar here.

* fallbacks: off-grid values, non-power-of-two scales and a group size that
  is not 32 must all fall back to the literal chain and stay correct.

The batched form (`etm,ehm->eth`, one weight per expert) is covered too: a
mixture-of-experts checkpoint is the reason MXFP4 shows up at all, and its
second projection is a batched dot.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from metaljax import qmm

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

needs_qmm = pytest.mark.skipif(not qmm.QMM_ENABLED, reason="METALJAX_QMM=0")
needs_batch = pytest.mark.skipif(not qmm._BATCH, reason="METALJAX_QMM_BATCH=0")
needs_build_cache = pytest.mark.skipif(
    not qmm._MAX_BUILT, reason="METALJAX_QMM_BUILD_CACHE=0")

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
# pack primitives
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_mxfp4_codes_recovers_every_code(dtype):
    """All 16 codes, in every float dtype an MXFP4 chain can decode into."""
    import mlx.core as mx
    mxdt = {"float32": mx.float32, "bfloat16": mx.bfloat16,
            "float16": mx.float16}[dtype]
    codes = np.tile(np.arange(16, dtype=np.uint8), 8).reshape(4, 32)
    vals = mx.array(E2M1[codes]).astype(mxdt)
    np.testing.assert_array_equal(np.array(qmm.mxfp4_codes(vals)), codes)


def test_mxfp4_codes_keeps_the_sign_of_zero():
    """Code 8 is -0.0 and code 0 is +0.0: the split is a bit test, not a
    comparison (`-0.0 == 0.0` is true)."""
    import mlx.core as mx
    vals = mx.array(np.array([[0.0, -0.0] * 16], np.float32))
    got = np.array(qmm.mxfp4_codes(vals))
    np.testing.assert_array_equal(got, np.array([[0, 8] * 16], np.uint8))


@pytest.mark.parametrize("byte", [1, 100, 127, 200, 254])
def test_mxfp4_scale_bytes_round_trip(byte):
    """The E8M0 byte is the f32 exponent field of `2**(byte-127)`."""
    import mlx.core as mx
    scale = np.full((2, 4), np.exp2(float(byte) - 127.0), np.float32)
    got = np.array(qmm.mxfp4_scale_bytes(mx.array(scale)))
    np.testing.assert_array_equal(got, np.full((2, 4), byte, np.uint8))


@pytest.mark.parametrize("bad", [3.0, 0.0, -2.0, np.inf, np.nan, 1.5e-38])
def test_mxfp4_scale_bytes_rejects(bad):
    """Anything that is not an exact positive normal power of two."""
    import mlx.core as mx
    scale = np.array([[1.0, float(bad)]], np.float32)
    with pytest.raises(qmm._Reject):
        qmm.mxfp4_scale_bytes(mx.array(scale))


@pytest.mark.parametrize("bad", [0.7, 5.0, 8.0, 0.25, np.nan])
def test_mxfp4_codes_rejects_off_grid(bad):
    import mlx.core as mx
    vals = np.concatenate([E2M1[:8], np.array([bad] * 8, np.float32)])
    with pytest.raises(qmm._Reject):
        qmm.mxfp4_codes(mx.array(vals.reshape(1, 16)))


def test_pack_layout_is_a_little_endian_nibble_stream():
    """Element i occupies bits [4i, 4i+4) of the uint32 stream.

    Hand-computed rather than taken from MLX, so that the two sides of the
    contract are checked independently: this pins what we WRITE, and
    `test_pack_layout_is_what_mlx_reads` pins that MLX reads the same thing.
    """
    import mlx.core as mx
    codes = np.arange(32, dtype=np.uint8) % 16
    packed = np.array(qmm.pack_codes(mx.array(codes.reshape(1, 32)), 4))
    want = [sum(int(codes[8 * word + i]) << (4 * i) for i in range(8))
            for word in range(4)]
    np.testing.assert_array_equal(packed[0], np.array(want, np.uint32))


@pytest.mark.parametrize("shape", [(4, 64), (3, 2, 128)])
def test_pack_layout_is_what_mlx_reads(shape):
    """The contract the whole branch rests on: MLX's MXFP4 dequant of our
    packed codes and E8M0 bytes reproduces the source weight EXACTLY.

    Any disagreement about nibble order, word order, or the scale encoding
    would show up here as garbage rather than as a rounding difference.
    """
    import mlx.core as mx
    rng = np.random.default_rng(sum(shape))
    codes, _blocks, sb, w = _random_mxfp4(shape, rng)
    packed = qmm.pack_codes(mx.array(codes), 4)
    scale = np.exp2(sb.astype(np.float32) - 127)
    ours = qmm.mxfp4_scale_bytes(mx.array(scale))
    np.testing.assert_array_equal(np.array(ours), sb)
    got = mx.dequantize(packed, mx.array(ours), mode="mxfp4")
    np.testing.assert_array_equal(np.array(got.astype(mx.float32)), w)


def _mlx_quantize_group(mag, kexp):
    """One group of 32 holding a single value `mag * 2**kexp`, round-tripped
    through MLX's own MXFP4 quantizer."""
    import mlx.core as mx
    w = np.zeros((1, 32), np.float32)
    w[0, 0] = np.float32(mag * 2.0 ** kexp)
    q, s = mx.quantize(mx.array(w), mode="mxfp4")
    got = np.array(mx.dequantize(q, s, mode="mxfp4").astype(mx.float32))
    return w, got


@pytest.mark.parametrize("mag", [0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("kexp", [0, -3, 5])
def test_mlx_quantize_saturates_group_maxima(mag, kexp):
    """MLX BUG (0.32): `mx.quantize(mode="mxfp4")` is LOSSY on inputs that
    already lie exactly on the MXFP4 grid.

    It picks `scale = 2**floor(log2(amax / 6))` and always encodes the group
    maximum as code 7 (the largest E2M1 value, 6.0). The floor makes
    `amax / scale >= 6`, with equality only when `amax / 6` is an exact
    power of two -- i.e. only when amax has mantissa 1.5 (0.75, 1.5, 3, 6,
    12, ... times a power of two). For every other magnitude the group
    maximum comes back multiplied by `6 / (amax/scale)`, which for a
    mantissa-1.0 maximum (0.5, 1, 2, 4 * 2**k) is exactly 0.75 -- a 25%
    error on the largest weight in the group.

    Hence this module never round-trips a checkpoint through `mx.quantize`:
    it derives the codes and the E8M0 byte from the factors the graph
    already carries, which is exact. The bug does not touch `mx.dequantize`
    or `mx.quantized_matmul` -- those read the stored layout faithfully,
    which is what `test_pack_layout_is_what_mlx_reads` pins.
    """
    import mlx.core as mx
    w, got = _mlx_quantize_group(mag, kexp)
    assert got[0, 0] == pytest.approx(0.75 * w[0, 0]), (w[0, 0], got[0, 0])

    # ... and the same value packed from its own factors (the grid code and
    # the power-of-two scale, which is what a checkpoint stores) is exact.
    codes = qmm.mxfp4_codes(mx.array(w / np.float32(2.0 ** kexp)))
    sb = qmm.mxfp4_scale_bytes(mx.array(np.full((1, 1), 2.0 ** kexp,
                                                np.float32)))
    ours = mx.dequantize(qmm.pack_codes(codes, 4), sb, mode="mxfp4")
    np.testing.assert_array_equal(np.array(ours.astype(mx.float32)), w)


@pytest.mark.parametrize("mag", [1.5, 3.0, 6.0])
@pytest.mark.parametrize("kexp", [0, -3, 5])
def test_mlx_quantize_is_exact_only_for_mantissa_1_5_maxima(mag, kexp):
    """The other side of the same rule: a group whose maximum is 1.5 * 2**k
    survives `mx.quantize` untouched."""
    w, got = _mlx_quantize_group(mag, kexp)
    np.testing.assert_array_equal(got, w)


@pytest.mark.parametrize("shape", [(4, 64), (3, 2, 128)])
def test_mxfp4_dequant_is_bit_exact(shape):
    """`dequantize(pack(...))` reproduces the numpy reference exactly, in
    f32 and in bf16 -- an E2M1 value has one mantissa bit, so scaling it by
    a power of two never needs more."""
    import mlx.core as mx
    rng = np.random.default_rng(7 + sum(shape))
    codes, _blocks, sb, w = _random_mxfp4(shape, rng)
    packed = qmm.pack_codes(mx.array(codes), 4)
    got = mx.dequantize(packed, mx.array(sb), mode="mxfp4")
    np.testing.assert_array_equal(np.array(got.astype(mx.float32)), w)
    np.testing.assert_array_equal(
        np.array(got.astype(mx.bfloat16).astype(mx.float32)), w)


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

    It cannot be asserted bit-equal: `quantized_matmul` and `matmul` are
    different kernels and accumulate in a different order. What IS bit-equal
    is the reconstructed weight -- pinned by
    `test_pack_layout_is_what_mlx_reads` -- so any residual difference here
    is the dot's own summation, and it is bounded by the dtype.
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

    qmm.reset_stats()
    got = _run(lambda *a: dense_mxfp4(*a, k=k), args, _metal())
    st = qmm.stats()
    assert st["packs"] == 1 and st["mxfp4"] == 1, st
    assert st["fallbacks"] == 0, st

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

    qmm.reset_stats()
    got = _run(lambda *a: experts_mxfp4(*a, k=m), args, _metal())
    st = qmm.stats()
    assert st["packs"] == 1 and st["mxfp4"] == 1 and st["batched"] == 1, st

    ref = _run(
        lambda x_: jnp.einsum("etm,ehm->eth", x_, jnp.asarray(w, dtype)),
        (jnp.asarray(x),), _metal())
    exact = np.einsum("etm,ehm->eth", x.astype(np.float32), w)
    _no_worse_than(got, ref, exact, dtype)


@needs_qmm
@needs_build_cache
def test_identical_weights_pack_once():
    """Two dots over the SAME weight share one packed copy.

    jax lowers a decode loop's prefill and its while body as separate dots
    over one set of weights, so without this a model's whole packed weight
    set is built and held twice (measured on gpt-oss-20b: 2 x 10.2 GB).
    Here the two chains are the same chain over the same buffers, so the
    BUILD cache answers the second one and the second build never runs;
    `test_identical_weights_from_two_buffers_share_a_pack` covers the case
    it cannot prove, where content-addressed sharing is still the answer.
    """
    n, k, t = 64, 128, 2
    rng = np.random.default_rng(9)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    x = (rng.standard_normal((t, k)) * 0.4).astype("bfloat16")

    def f(blocks_, sb_, x_):
        a = dense_mxfp4(blocks_, sb_, x_, k)
        b = dense_mxfp4(blocks_, sb_, jnp.tanh(x_), k)
        return a + b

    qmm.reset_stats()
    qmm._BUILT.clear()
    got = _run(f, (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x)),
               _metal())
    st = qmm.stats()
    assert st["packs"] == 1 and st["build_hits"] == 1, st

    wf = jnp.asarray(w, "bfloat16")
    ref = _run(lambda x_: (jnp.einsum("th,nh->tn", x_, wf)
                           + jnp.einsum("th,nh->tn", jnp.tanh(x_), wf)),
               (jnp.asarray(x),), _metal())
    exact = (x.astype(np.float32) @ w.T
             + np.tanh(x.astype(np.float32)) @ w.T)
    _no_worse_than(got, ref, exact, "bfloat16")


@needs_qmm
def test_identical_weights_from_two_buffers_share_a_pack():
    """Same bytes, two placements: the build cache cannot prove it, `_share`
    can.

    Buffer identity is what the build key names, and these really are two
    buffers -- so both weights get built and the content-addressed dedupe is
    what stops the model holding two copies. That is the case `_share`
    exists for once the build cache has taken the provable one.
    """
    n, k, t = 64, 128, 2
    rng = np.random.default_rng(91)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    x = (rng.standard_normal((t, k)) * 0.4).astype("float32")

    def f(b1, s1, b2, s2, x_):
        return dense_mxfp4(b1, s1, x_, k) + dense_mxfp4(b2, s2, x_, k)

    qmm.reset_stats()
    qmm._SHARED.clear()
    qmm._BUILT.clear()
    dev = _metal()
    with jax.default_device(dev):
        # Two separate placements of the SAME bytes.
        args = [jax.device_put(v, dev) for v in
                (blocks, sb, blocks.copy(), sb.copy(), x)]
        got = np.asarray(jax.jit(f)(*args))
    st = qmm.stats()
    assert st["packs"] == 2 and st["shared"] == 1, st
    assert st["build_hits"] == 0, st
    exact = 2.0 * (x.astype(np.float32) @ w.T)
    np.testing.assert_allclose(got, exact, rtol=1e-5,
                               atol=1e-6 * np.abs(exact).max())


@needs_qmm
def test_different_weights_do_not_share_a_pack():
    """The dedupe is content-addressed: two dots whose weights differ keep
    their own packs even though the chains are structurally identical."""
    n, k, t = 64, 128, 2
    rng = np.random.default_rng(10)
    _c1, blocks1, sb1, w1 = _random_mxfp4((n, k), rng)
    _c2, blocks2, sb2, w2 = _random_mxfp4((n, k), rng)
    assert not np.array_equal(w1, w2)
    x = (rng.standard_normal((t, k)) * 0.4).astype("float32")

    def f(b1, s1, b2, s2, x_):
        return (dense_mxfp4(b1, s1, x_, k) + dense_mxfp4(b2, s2, x_, k))

    qmm.reset_stats()
    args = (jnp.asarray(blocks1), jnp.asarray(sb1), jnp.asarray(blocks2),
            jnp.asarray(sb2), jnp.asarray(x))
    got = _run(f, args, _metal())
    st = qmm.stats()
    assert st["packs"] == 2 and st["shared"] == 0, st
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
    qmm.reset_stats()
    dev = _metal()
    with jax.default_device(dev):
        moved = [jax.device_put(a, dev) for a in args]
        jf = jax.jit(f)
        got = np.asarray(jf(*moved)).astype(np.float32)
        st = qmm.stats()
        assert st["packs"] == 1 and st["mxfp4"] == 1, st
        got2 = np.asarray(jf(*moved)).astype(np.float32)
    np.testing.assert_array_equal(got, got2)
    want = _run(f, args, _cpu()).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)


# --------------------------------------------------------------------------
# fallbacks: still correct, just not rewritten
# --------------------------------------------------------------------------


def _fallback(f, args, route):
    """The chain runs literally, and is still right.

    `route` says WHERE it was turned down, which is worth pinning: a
    "structural" rejection costs nothing, while a "verify" one is only
    discovered after the weight has been evaluated once at full size.
    """
    qmm.reset_stats()
    got = _run(f, args, _metal())
    st = qmm.stats()
    assert st["packs"] == 0, st
    if qmm.QMM_ENABLED:
        if route == "structural":
            assert st["recognized"] == 0, st
        else:
            assert st["recognized"] >= 1 and st["fallbacks"] >= 1, st
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(np.asarray(got, np.float32),
                               np.asarray(want, np.float32),
                               rtol=1e-5, atol=1e-5)


@needs_qmm
def test_ordinary_float_graphs_recognize_nothing():
    """The guard that keeps every unquantized model on its old path.

    Trying BOTH dot operands (needed for `th,emh->etm`) means the
    recognizer now looks at multiplies feeding either side, and an RMS norm
    or a scaled residual is exactly `something * broadcast(something
    smaller)`. None of these may be adopted: a match costs a full-weight
    evaluation before it can be turned down, and this is the shape of every
    training step in the acceptance workload.
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

    qmm.reset_stats()
    got = _run(f, (x, w1, w2, g), _metal())
    assert qmm.stats()["recognized"] == 0, qmm.stats()
    want = _run(f, (x, w1, w2, g), _cpu())
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


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

    _fallback(f, (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x)),
              route="verify")


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

    _fallback(f, (jnp.asarray(blocks), jnp.asarray(scale), jnp.asarray(x)),
              route="verify")


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

    # Turned down at ANALYSIS time: the scale broadcasts one value per 16,
    # and MXFP4 has exactly one group size. No weight is ever evaluated.
    _fallback(f, (jnp.asarray(blocks), jnp.asarray(sb), jnp.asarray(x)),
              route="structural")


# --------------------------------------------------------------------------
# row-blocked packing
# --------------------------------------------------------------------------


def _packs_of(monkeypatch, f, args, block_elems):
    """Run `f` on metal with a given block size; return the packs it built.

    The content-addressed pack table is cleared first: two runs of the same
    weight produce identical packs, which `_share` would otherwise ALIAS --
    and comparing an array against itself proves nothing.
    """
    import mlx.core as mx
    from metaljax import qmm as q

    built = []
    orig = q._build_pack

    def spy(interp, m, argv, leaves):
        pk = orig(interp, m, argv, leaves)
        built.append([np.array(a) for a in pk.arrays()])
        return pk

    q._SHARED.clear()
    q._BUILT.clear()
    monkeypatch.setattr(q, "_BLOCK_ELEMS", block_elems)
    monkeypatch.setattr(q, "_build_pack", spy)
    q.reset_stats()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    out = _run(f, args, _metal())
    peak = mx.get_peak_memory() - base
    monkeypatch.undo()
    return built, out, q.stats(), peak


@needs_qmm
@pytest.mark.parametrize("form", ["dense", "experts"])
def test_blocked_pack_is_bit_identical(monkeypatch, form):
    """Packing block by block is packing, exactly.

    The reconstruction is evaluated one slice of the weight's leading axis
    at a time (`qmm._Source`) so that a multi-GB chain never materializes
    whole; the packed codes, the E8M0 scale bytes and the dot's result must
    all be the SAME BITS as a whole evaluation produces.
    """
    e, n, k, t = 4, 128, 256, 3
    rng = np.random.default_rng(11)
    shape = (n, k) if form == "dense" else (e, n, k)
    _codes, blocks, sb, _w = _random_mxfp4(shape, rng)
    xshape = (t, k) if form == "dense" else (e, t, k)
    x = (rng.standard_normal(xshape) * 0.4).astype("bfloat16")
    fn = dense_mxfp4 if form == "dense" else experts_mxfp4
    # Host arrays, so that each run's `device_put` lands on FRESH buffers
    # and the second one really repacks (packs are cached on their source
    # buffers' identity).
    args = (blocks, sb, x)

    def f(*a):
        return fn(*a, k=k)

    whole, out_w, st_w, _ = _packs_of(monkeypatch, f, args, 1 << 40)
    small, out_b, st_b, _ = _packs_of(monkeypatch, f, args, 1 << 10)
    assert st_w["packs"] == st_b["packs"] == 1, (st_w, st_b)
    assert st_w["blocked"] == 0 and st_b["blocked"] == 1, (st_w, st_b)
    assert len(whole) == len(small) == 1
    for a, b in zip(whole[0], small[0]):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(out_w, out_b)


@needs_qmm
def test_blocked_pack_bounds_the_transient(monkeypatch):
    """The point of blocking: the pack costs a block, not a weight.

    An MXFP4 chain is several times wider than the weight it reconstructs
    (jax's `take` wrapper alone carries three int32 copies of the index
    tensor), so a whole evaluation of gpt-oss-20b's gate_up projection
    peaked at 7.7 GB of live buffers and spiked claimed memory by 15.9 GB --
    which is what kept killing the benchmark row. The bound asserted here is
    deliberately loose (half of the whole-evaluation peak); the real ratio
    at this size is ~8x, and on the real checkpoint it was 5x on live
    buffers and 10x on claimed.
    """
    e, n, k, t = 8, 2048, 1024, 2
    rng = np.random.default_rng(12)
    _codes, blocks, sb, _w = _random_mxfp4((e, n, k), rng)
    x = (rng.standard_normal((e, t, k)) * 0.4).astype("bfloat16")
    args = (blocks, sb, x)

    def f(*a):
        return experts_mxfp4(*a, k=k)

    _w_packs, out_w, st_w, peak_whole = _packs_of(monkeypatch, f, args,
                                                  1 << 40)
    _b_packs, out_b, st_b, peak_blocked = _packs_of(monkeypatch, f, args,
                                                    1 << 18)
    assert st_w["blocked"] == 0 and st_b["blocked"] == 1, (st_w, st_b)
    np.testing.assert_array_equal(out_w, out_b)
    assert peak_blocked < peak_whole / 2, (peak_blocked, peak_whole)
    # ... and in absolute terms it tracks the BLOCK (one expert here), not
    # the weight: ~15x a block's bf16 bytes at this shape, which is the
    # width of the chain plus the packed result accumulating.
    block_bytes = n * k * 2
    assert peak_blocked < 24 * block_bytes, (peak_blocked, block_bytes)


@needs_qmm
@needs_batch
def test_blocked_pack_reads_a_decode_table_whole(monkeypatch):
    """The blocked axis and the E2M1 table are both 16 long here.

    Whether a value is blocked is decided by the DEMAND coming down from
    the weight, not by its shape: the gather's indices are blocked and the
    table it gathers through is read whole, however its first dimension
    happens to compare to the expert count. A bottom-up rule would slice
    the table and quietly pack the wrong values.
    """
    e, n, k, t = 16, 64, 128, 2
    rng = np.random.default_rng(13)
    _codes, blocks, sb, w = _random_mxfp4((e, n, k), rng)
    x = (rng.standard_normal((e, t, k)) * 0.4).astype("bfloat16")
    args = (blocks, sb, x)

    def f(*a):
        return experts_mxfp4(*a, k=k)

    whole, out_w, st_w, _ = _packs_of(monkeypatch, f, args, 1 << 40)
    small, out_b, st_b, _ = _packs_of(monkeypatch, f, args, 1 << 10)
    assert st_b["blocked"] == 1 and st_b["fallbacks"] == 0, st_b
    for a, b in zip(whole[0], small[0]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(out_w, out_b)
    exact = np.einsum("etm,ehm->eth", np.asarray(x, np.float32), w)
    _no_worse_than(out_b, out_w, exact, "bfloat16")


# --------------------------------------------------------------------------
# the cross-executable build cache
# --------------------------------------------------------------------------


def _through_executables(shapes, blocks, sb, k, dtype="bfloat16"):
    """One MXFP4 weight through several executables; return (outs, stats).

    The MXFP4 chain is the one that makes the cache worth having: its E2M1
    lookup lowers to a private `@_take` function, so the fingerprint has to
    walk a CALLEE body -- and jax renumbers those per program, which is
    exactly what a symbol-name key would get wrong.
    """
    qmm._SHARED.clear()
    qmm._BUILT.clear()
    qmm.reset_stats()
    dev = _metal()
    fns, outs = [], []
    with jax.default_device(dev):
        moved = [jax.device_put(v, dev) for v in (blocks, sb)]
        for t in shapes:
            x = (np.random.default_rng(200 + t).standard_normal(
                (t, k)) * 0.4).astype(dtype)
            f = jax.jit(lambda b, s, xx: dense_mxfp4(b, s, xx, k=k))
            fns.append(f)                    # a dropped executable drops its
            outs.append((x, np.asarray(     # pack, and the cache is weak
                f(*moved, jax.device_put(x, dev)))))
    return outs, qmm.stats(), fns


@needs_qmm
@needs_build_cache
def test_second_executable_reuses_the_mxfp4_pack():
    n, k = 64, 128
    rng = np.random.default_rng(77)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    outs, st, _fns = _through_executables((2, 5, 3), blocks, sb, k)
    assert st["packs"] == 1, st
    assert st["build_hits"] == 2, st
    assert st["build_declines"] == 0, st
    for x, got in outs:
        exact = np.asarray(x, np.float32) @ w.T
        np.testing.assert_allclose(got.astype(np.float32), exact,
                                   rtol=2e-2, atol=2e-2 * np.abs(exact).max())


@needs_qmm
@needs_build_cache
def test_a_big_constant_declines_the_build_cache(monkeypatch):
    """A decode table too big to serialize: no cache, and no wrong reuse.

    The declining is the safety valve -- everything the fingerprint cannot
    cover exactly falls back to building, which is what happened before this
    cache existed. Here the E2M1 table itself is put over the limit.
    """
    monkeypatch.setattr(qmm, "_FP_DENSE_ELEMS", 8)
    n, k = 64, 128
    rng = np.random.default_rng(78)
    _codes, blocks, sb, w = _random_mxfp4((n, k), rng)
    outs, st, _fns = _through_executables((2, 5), blocks, sb, k)
    assert st["build_declines"] == 2, st     # one per match, once each
    assert st["build_hits"] == 0, st
    assert st["packs"] == 2, st
    for x, got in outs:
        exact = np.asarray(x, np.float32) @ w.T
        np.testing.assert_allclose(got.astype(np.float32), exact,
                                   rtol=2e-2, atol=2e-2 * np.abs(exact).max())


def _fingerprint_of(text):
    """The build-cache fingerprint of the one match in a module's text."""
    from jaxlib.mlir import ir
    from metaljax import _ir
    from metaljax.interpreter import Interpreter

    ctx = _ir.make_context()
    with ctx:
        module = ir.Module.parse(text)
    interp = Interpreter(module, context=ctx)
    st = qmm.analyze(interp)
    assert len(st.matches) == 1
    return qmm._fingerprint(interp, st.matches[0])[0]


@needs_qmm
def test_fingerprint_reads_callee_bodies_not_their_names():
    """A call is what its callee DOES, not what it is called.

    jax emits `jnp.take` as a private helper and numbers those per program
    (`@_take`, `@_take_0`), so two programs holding the same weight will
    disagree about the name; the same argument says two modules that bind
    one name to different bodies must never share a key.
    """
    import re

    n, k = 64, 128
    _codes, blocks, sb, _w = _random_mxfp4((n, k), np.random.default_rng(79))
    x = np.zeros((2, k), np.float32)
    text = jax.jit(lambda b, s, xx: dense_mxfp4(b, s, xx, k=k)).lower(
        blocks, sb, x).as_text()
    assert "func.func private @_take(" in text

    base = _fingerprint_of(text)
    # The gather, the reduce and the select live ONLY in the helpers, so
    # their presence is the proof that the bodies were walked at all.
    for name in ("stablehlo.gather", "stablehlo.reduce", "stablehlo.select"):
        assert name in base

    renamed = _fingerprint_of(text.replace("_take", "_pick_9")
                              .replace("_where", "_pick_8"))
    assert renamed == base

    start = text.index("func.func private @_take(")
    end = text.index("\n  }", start)
    body = text[start:end]
    mo = re.search(r"stablehlo\.constant dense<(-?\d+)>", body)
    edited = (text[:start] + body[:mo.start(1)] + "7" + body[mo.end(1):]
              + text[end:])
    assert _fingerprint_of(edited) != base
