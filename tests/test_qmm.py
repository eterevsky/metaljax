"""Quantized-matmul recognizer: exact repacking, rewriting, and fallbacks.

Two layers of testing:

* `pack_exact` against a numpy reference, including the bit-exactness matrix
  measured on MLX 0.32 (zero point 0 is exact for any f32 scale; a general
  zero point is exact only when the scale has few enough mantissa bits that
  `scale * (2**(b-1) + zp)` is representable -- always true for the bf16/f16
  scales real quantizers emit).

* The recognizer end to end THROUGH THE REAL BACKEND (the packing prologue
  lives in engine.execute, so the bare Interpreter used by `helpers.check`
  never exercises it). Every case is checked against jax-CPU *and* against an
  exactly-dequantized float32 reference: the rewrite accumulates in f32 over
  an exactly reconstructed weight, so it lands closer to the exact answer
  than the literal bf16 chain does.

The layer graphs are hand-built copies of what keras 3.15 emits for
`Dense.quantize("int4")` (sub-channel, block_size=128, asymmetric with a
zero point and a `g_idx` group ramp), its per-channel variant
(block_size=-1, symmetric, scale divides the OUTPUT) and `EinsumDense`
(`btd,dnh->btnh`, kernel reshaped right before the dot) -- keras is not
installed in this venv.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from metaljax import qmm

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# METALJAX_QMM=0 turns the whole rewrite off: the numeric comparisons below
# still run (the literal chain must be right too), the ones that assert the
# rewrite happened do not.
needs_qmm = pytest.mark.skipif(not qmm.ENABLED, reason="METALJAX_QMM=0")


def _metal():
    return jax.devices("metal")[0]


def _cpu():
    return jax.devices("cpu")[0]


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


def np_pack_codes(codes, bits):
    """packlib.py's generic bit-stream reference (any `bits`)."""
    c = np.ascontiguousarray(codes.astype(np.uint64))
    k = c.shape[-1]
    words = k * bits // 32
    flat = c.reshape(-1, k)
    out = np.zeros((flat.shape[0], words), dtype=np.uint64)
    for i in range(k):
        w0, off = divmod(i * bits, 32)
        v = flat[:, i]
        out[:, w0] |= (v << np.uint64(off)) & np.uint64(0xFFFFFFFF)
        if off + bits > 32:
            out[:, w0 + 1] |= v >> np.uint64(32 - off)
    return out.astype(np.uint32).reshape(*c.shape[:-1], words)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(1, 32), (5, 128), (3, 4, 64)])
def test_pack_codes_matches_reference(bits, shape):
    import mlx.core as mx
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1 << bits, size=shape).astype(np.uint32)
    got = np.array(qmm.pack_codes(mx.array(codes), bits))
    np.testing.assert_array_equal(got, np_pack_codes(codes, bits))


def _scale_kind(kind, shape, rng):
    s = (rng.random(shape).astype(np.float32) + 0.5) * 0.03
    if kind == "pow2":
        return np.exp2(rng.integers(-8, 2, size=shape)).astype(np.float32)
    if kind == "bf16":
        import ml_dtypes
        return s.astype(ml_dtypes.bfloat16).astype(np.float32)
    return s  # arbitrary f32


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("gs", [32, 64, 128])
@pytest.mark.parametrize("zp_kind", ["zero", "const", "random"])
@pytest.mark.parametrize("scale_kind", ["pow2", "bf16", "f32"])
def test_pack_exact(bits, gs, zp_kind, scale_kind):
    """`dequantize(pack_exact(...))` vs float32 `scale * (code - zp)`."""
    import mlx.core as mx
    rng = np.random.default_rng(bits * 100 + gs)
    n, k = 16, 256
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    codes = rng.integers(lo, hi + 1, size=(n, k)).astype(np.int32)
    codes[0, :4] = [lo, hi, 0, -1]  # force the extremes in
    ng = k // gs
    scale = _scale_kind(scale_kind, (n, ng), rng)
    if zp_kind == "zero":
        zp = np.zeros((n, ng), np.int32)
    elif zp_kind == "const":
        zp = np.full((n, ng), 3, np.int32)
    else:
        zp = rng.integers(lo, hi + 1, size=(n, ng)).astype(np.int32)

    want = scale.repeat(gs, axis=1) * (codes.astype(np.float32)
                                       - zp.repeat(gs, axis=1).astype(np.float32))
    w, s, b = qmm.pack_exact(mx.array(codes), mx.array(scale), mx.array(zp),
                             bits)
    got = np.array(mx.dequantize(w, s, b, group_size=gs, bits=bits))
    assert got.shape == want.shape
    exact = bool(np.all(got == want))
    # Measured on MLX 0.32 (report-mlxq section 2): the dequant is a single
    # FMA, so it is bit-exact whenever the folded bias is representable --
    # always for zp == 0 (a power-of-two multiple of the scale) and for any
    # zp when the scale has few mantissa bits.
    if zp_kind == "zero" or scale_kind in ("pow2", "bf16"):
        assert exact, np.abs(got - want).max()
    else:
        rel = np.abs(got - want).max() / np.abs(want).max()
        assert rel < 1e-5, rel


@pytest.mark.parametrize("shift", [0, 8, -20])
def test_pack_exact_arbitrary_code_offset(shift):
    """MLX's q_hat is unsigned, so any integer shift works as long as the
    bias undoes it -- that is what lets a [0, 15] code range pack at 4 bits."""
    import mlx.core as mx
    import ml_dtypes
    rng = np.random.default_rng(2)
    codes = rng.integers(-8, 8, size=(4, 128)).astype(np.int32) + shift
    # bf16-precision scales, as every real quantizer emits: the bias
    # `scale * offset` is then representable for ANY integer offset, which
    # is what makes the reconstruction exact off the power-of-two grid.
    scale = ((rng.random((4, 1)).astype(np.float32) + 0.5) * 0.1).astype(
        ml_dtypes.bfloat16).astype(np.float32)
    want = scale * codes.astype(np.float32)
    w, s, b = qmm.pack_exact(mx.array(codes), mx.array(scale), None, 4,
                             offset=-int(codes.min()))
    got = np.array(mx.dequantize(w, s, b, group_size=128, bits=4))
    np.testing.assert_array_equal(got, want)


def test_pack_exact_keeps_source_dtype_when_lossless():
    import mlx.core as mx
    if qmm._SCALE_WIDTH != "auto":
        pytest.skip("METALJAX_QMM_SCALES overrides the width policy")
    rng = np.random.default_rng(1)
    codes = rng.integers(-8, 8, size=(8, 128)).astype(np.int32)
    scale = np.exp2(rng.integers(-6, 0, size=(8, 1))).astype(np.float32)
    # zp == 0: bias = -8*scale is exact in bf16, so the narrow form is kept.
    w, s, b = qmm.pack_exact(mx.array(codes), mx.array(scale), None, 4,
                             scale_dtype=mx.bfloat16)
    assert s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16
    # a zero point that makes the bias unrepresentable in bf16 widens to f32
    scale = (rng.random((8, 1)).astype(np.float32) + 1.0)
    zp = np.full((8, 1), 3, np.int32)
    w, s, b = qmm.pack_exact(mx.array(codes), mx.array(scale), mx.array(zp), 4,
                             scale_dtype=mx.bfloat16)
    assert s.dtype == mx.float32 and b.dtype == mx.float32


# --------------------------------------------------------------------------
# keras-shaped graphs
# --------------------------------------------------------------------------


def _unpack_nibbles(packed, columns):
    """keras.quantizers.unpack_int4 for the packed (axis=-1) layout."""
    t = packed.T                                    # [cols/2, rows]
    low = jnp.bitwise_and(t, 15)
    high = jnp.bitwise_and(jax.lax.shift_right_arithmetic(t, jnp.int8(4)), 15)

    def to_signed(v):
        return jnp.bitwise_xor(v, jnp.int8(8)) - jnp.int8(8)

    stacked = jnp.stack([to_signed(low), to_signed(high)], axis=1)
    up = jnp.reshape(stacked, (-1,) + tuple(t.shape[1:]))
    return up[:columns, ...].T                      # [rows, cols]


def dense_sub(packed, scale, zero, g_idx, x, columns):
    """keras Dense._int4_call, sub-channel branch."""
    w = _unpack_nibbles(packed, columns)
    g = g_idx.astype(jnp.int32)
    s = jnp.take(scale, g, axis=0)
    z = jnp.take(zero, g, axis=0)
    return x @ ((w.astype(x.dtype) - z.astype(x.dtype)) * s)


def dense_perchannel(packed, scale, x, columns):
    """keras Dense._int4_call, per-channel branch: the scale divides out."""
    w = _unpack_nibbles(packed, columns)
    return (x @ w.astype(x.dtype)) / scale


def einsum_sub(packed, scale, zero, g_idx, x, out_shape):
    """keras EinsumDense._int4_call: dequantize, reshape, then einsum."""
    columns = int(np.prod(out_shape))
    w = _unpack_nibbles(packed, columns)
    g = g_idx.astype(jnp.int32)
    s = jnp.take(scale, g, axis=0)
    z = jnp.take(zero, g, axis=0)
    wf = (w.astype(x.dtype) - z.astype(x.dtype)) * s
    return jnp.einsum("btd,dnh->btnh", x,
                      jnp.reshape(wf, (w.shape[0],) + tuple(out_shape)))


def _quantize(rows, cols, block, dtype, seed=0, bits=4, g_idx=None):
    """Codes + scale/zero maps in keras' storage layout."""
    rng = np.random.default_rng(seed)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = rng.integers(lo, hi + 1, size=(rows, cols)).astype(np.int8)
    bsz = rows if block < 0 else block
    ng = rows // bsz
    scale = ((rng.random((ng, cols)).astype(np.float32) + 0.5) * 0.05)
    zero = (np.zeros((ng, cols), np.int8) if block < 0 else
            rng.integers(-3, 4, size=(ng, cols)).astype(np.int8))
    if g_idx is None:
        g_idx = (np.arange(rows) // bsz).astype(np.float32)
    packed = ((q[:, 0::2] & 0x0F) | (q[:, 1::2] << 4)).astype(np.int8)
    ref = (scale[g_idx.astype(np.int32)]
           * (q.astype(np.float32) - zero[g_idx.astype(np.int32)].astype(np.float32)))
    return q, packed, scale.astype(dtype), zero, g_idx, ref


def _run(f, args, device):
    with jax.default_device(device):
        moved = [jax.device_put(a, device) for a in args]
        return np.asarray(jax.jit(f)(*moved)).astype(np.float32)


def _compare(f, args, exact, packs=1):
    """Run on metal + CPU; both must match the exact f32 dequant reference,
    and metal must be no worse than CPU."""
    qmm.reset_stats()
    got = _run(f, args, _metal())
    want = _run(f, args, _cpu())
    if qmm.ENABLED:
        assert qmm.stats()["packs"] == packs, qmm.stats()
    scale = np.abs(exact).max()
    err_metal = np.abs(got - exact).max() / scale
    err_cpu = np.abs(want - exact).max() / scale
    assert got.shape == want.shape
    # bf16 outputs round at ~4e-3; the comparison that matters is that the
    # rewrite is not WORSE than the literal chain the CPU backend runs.
    # (Only under the default scale policy: METALJAX_QMM_SCALES=source
    # deliberately trades up to ~2x of this error for scale traffic.)
    if qmm._SCALE_WIDTH == "auto":
        assert err_metal <= err_cpu * 1.5 + 1e-6, (err_metal, err_cpu)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=1e-2 * scale)
    return err_metal, err_cpu


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_dense_subchannel(dtype):
    rows, cols = 256, 128
    q, packed, scale, zero, g_idx, wref = _quantize(rows, cols, 128, dtype)
    x = (np.random.default_rng(3).standard_normal((4, rows)) * 0.5).astype(dtype)
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    exact = x.astype(np.float32) @ wref
    _compare(lambda *a: dense_sub(*a, columns=cols), args, exact)


def test_dense_subchannel_int8_codes():
    """The same graph with codes that need all 8 bits."""
    rows, cols = 128, 64
    q, packed, scale, zero, g_idx, wref = _quantize(
        rows, cols, 64, "bfloat16", seed=5, bits=8)
    # keras only packs nibbles for int4; feed the int8 codes straight in.
    x = (np.random.default_rng(4).standard_normal((2, rows)) * 0.5).astype("bfloat16")

    def f(codes, scale_, zero_, g_idx_, x_):
        g = g_idx_.astype(jnp.int32)
        s = jnp.take(scale_, g, axis=0)
        z = jnp.take(zero_, g, axis=0)
        return x_ @ ((codes.astype(x_.dtype) - z.astype(x_.dtype)) * s)

    args = (jnp.asarray(q), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    exact = x.astype(np.float32) @ wref
    _compare(f, args, exact)


def test_dense_per_channel():
    rows, cols = 128, 64
    rng = np.random.default_rng(11)
    q = rng.integers(-8, 8, size=(rows, cols)).astype(np.int8)
    packed = ((q[:, 0::2] & 0x0F) | (q[:, 1::2] << 4)).astype(np.int8)
    # keras' per-channel scale is 7/absmax and DIVIDES the output.
    scale = (7.0 / np.maximum(np.abs(q).max(axis=0), 1)).astype("bfloat16")
    x = (rng.standard_normal((3, rows)) * 0.4).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(x))
    exact = (x.astype(np.float32) @ q.astype(np.float32)) / scale.astype(np.float32)
    _compare(lambda *a: dense_perchannel(*a, columns=cols), args, exact)


def test_einsum_dense_gemma_shape():
    d, heads, head_dim = 256, 4, 32
    cols = heads * head_dim
    q, packed, scale, zero, g_idx, wref = _quantize(d, cols, 128, "bfloat16",
                                                    seed=7)
    x = (np.random.default_rng(8).standard_normal((1, 2, d)) * 0.3).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    exact = np.einsum("btd,dnh->btnh", x.astype(np.float32),
                      wref.reshape(d, heads, head_dim))
    _compare(lambda *a: einsum_sub(*a, out_shape=(heads, head_dim)), args, exact)


def test_jnp_int4_weights():
    """The other real producer of this pattern: weights stored as jnp.int4
    with a per-output-channel scale (metaljax emulates i4 in int8 storage,
    so the codes come out exact)."""
    rows, cols = 128, 64
    rng = np.random.default_rng(50)
    codes = rng.integers(-8, 8, size=(rows, cols)).astype(np.int8)
    scale = ((rng.random(cols).astype(np.float32) + 0.5) * 0.02).astype("bfloat16")
    x = (rng.standard_normal((2, rows)) * 0.4).astype("bfloat16")

    def f(w, s, x_):
        return x_ @ (w.astype(x_.dtype) * s)

    args = (jnp.asarray(codes, dtype=jnp.int4), jnp.asarray(scale),
            jnp.asarray(x))
    exact = x.astype(np.float32) @ (codes.astype(np.float32)
                                    * scale.astype(np.float32))
    _compare(f, args, exact)


def test_group_permuting_g_idx_is_still_exact():
    """A g_idx that permutes whole groups keeps the map group-constant: the
    group scales are read off the MATERIALIZED map, so it fuses correctly."""
    rows, cols, block = 256, 64, 128
    g_idx = np.where(np.arange(rows) // block == 0, 1.0, 0.0).astype(np.float32)
    q, packed, scale, zero, g_idx, wref = _quantize(
        rows, cols, block, "bfloat16", seed=9, g_idx=g_idx)
    x = (np.random.default_rng(10).standard_normal((2, rows)) * 0.4).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    exact = x.astype(np.float32) @ wref
    _compare(lambda *a: dense_sub(*a, columns=cols), args, exact)


# --------------------------------------------------------------------------
# fallbacks: still correct, just not rewritten
# --------------------------------------------------------------------------


def _fallback_case(f, args, exact):
    """Recognized structurally, then rejected by a first-execute check."""
    qmm.reset_stats()
    got = _run(f, args, _metal())
    st = qmm.stats()
    assert st["packs"] == 0, st
    if qmm.ENABLED:
        assert st["fallbacks"] >= 1, st
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-2,
                               atol=1e-2 * np.abs(exact).max())


def test_fallback_shuffled_g_idx():
    """A g_idx that mixes groups breaks group-constancy -> literal chain."""
    rows, cols, block = 256, 64, 128
    rng = np.random.default_rng(12)
    g_idx = rng.permutation(np.arange(rows) // block).astype(np.float32)
    q, packed, scale, zero, g_idx, wref = _quantize(
        rows, cols, block, "bfloat16", seed=13, g_idx=g_idx)
    x = (rng.standard_normal((2, rows)) * 0.4).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    _fallback_case(lambda *a: dense_sub(*a, columns=cols), args,
                   x.astype(np.float32) @ wref)


def test_fallback_block_size_below_32():
    """No legal MLX group size divides a 16-element block."""
    rows, cols = 128, 64
    q, packed, scale, zero, g_idx, wref = _quantize(rows, cols, 16, "bfloat16",
                                                    seed=14)
    x = (np.random.default_rng(15).standard_normal((2, rows)) * 0.4).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    _fallback_case(lambda *a: dense_sub(*a, columns=cols), args,
                   x.astype(np.float32) @ wref)


def test_fallback_integer_zero_point_subtraction_that_wraps():
    """When the graph subtracts the zero point in int8 it WRAPS; the rewrite
    would compute it exactly, so such a dot must not be rewritten."""
    rows, cols = 128, 64
    rng = np.random.default_rng(60)
    codes = rng.integers(100, 128, size=(rows, cols)).astype(np.int8)
    zero = np.full((rows, cols), -120, np.int8)   # codes - zero overflows i8
    scale = ((rng.random((1, cols)).astype(np.float32) + 0.5)
             * 0.01).astype("bfloat16")
    x = (rng.standard_normal((2, rows)) * 0.4).astype("bfloat16")

    def f(c, z, s, x_):
        return x_ @ ((c - z).astype(x_.dtype) * s)

    args = (jnp.asarray(codes), jnp.asarray(zero), jnp.asarray(scale),
            jnp.asarray(x))
    _fallback_case(f, args, np.ones((2, cols), np.float32))


def test_integer_zero_point_subtraction_without_wrap_is_rewritten():
    """The same shape, in range: fused, and equal to the exact dequant."""
    rows, cols = 128, 64
    rng = np.random.default_rng(61)
    codes = rng.integers(-8, 8, size=(rows, cols)).astype(np.int8)
    zero = np.full((rows, cols), 3, np.int8)
    scale = ((rng.random((1, cols)).astype(np.float32) + 0.5)
             * 0.01).astype("bfloat16")
    x = (rng.standard_normal((2, rows)) * 0.4).astype("bfloat16")

    def f(c, z, s, x_):
        return x_ @ ((c - z).astype(x_.dtype) * s)

    args = (jnp.asarray(codes), jnp.asarray(zero), jnp.asarray(scale),
            jnp.asarray(x))
    exact = x.astype(np.float32) @ ((codes - zero).astype(np.float32)
                                    * scale.astype(np.float32))
    _compare(f, args, exact)


@needs_qmm
def test_fallback_dequantized_weight_used_twice():
    """A second consumer of the reconstructed weight means it is materialized
    anyway: the candidate is dropped at analysis time (never even packed)."""
    rows, cols = 128, 64
    q, packed, scale, zero, g_idx, wref = _quantize(rows, cols, 64, "bfloat16",
                                                    seed=16)
    x = (np.random.default_rng(17).standard_normal((2, rows)) * 0.4).astype("bfloat16")

    def f(packed_, scale_, zero_, g_idx_, x_):
        w = _unpack_nibbles(packed_, cols)
        g = g_idx_.astype(jnp.int32)
        wf = ((w.astype(x_.dtype) - jnp.take(zero_, g, axis=0).astype(x_.dtype))
              * jnp.take(scale_, g, axis=0))
        return x_ @ wf + jnp.sum(wf).astype(x_.dtype)

    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    qmm.reset_stats()
    got = _run(f, args, _metal())
    st = qmm.stats()
    assert st["packs"] == 0 and st["recognized"] == 0, st
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=1e-1)


def test_disabled_by_env(monkeypatch):
    """METALJAX_QMM=0 leaves the literal chain in place."""
    rows, cols = 128, 64
    q, packed, scale, zero, g_idx, wref = _quantize(rows, cols, 64, "bfloat16",
                                                    seed=18)
    x = (np.random.default_rng(19).standard_normal((2, rows)) * 0.4).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    monkeypatch.setattr(qmm, "ENABLED", False)
    qmm.reset_stats()
    got = _run(lambda *a: dense_sub(*a, columns=cols), args, _metal())
    assert qmm.stats() == {"recognized": 0, "packs": 0, "fallbacks": 0}
    want = _run(lambda *a: dense_sub(*a, columns=cols), args, _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=1e-1)


@needs_qmm
def test_new_weight_buffers_repack_without_stale_results():
    """A second set of weights through the SAME executable must not reuse
    the first pack (the packed arrays travel as compiled-graph inputs)."""
    rows, cols = 128, 64
    a = _quantize(rows, cols, 64, "bfloat16", seed=22)
    b = _quantize(rows, cols, 64, "bfloat16", seed=23)
    x = (np.random.default_rng(24).standard_normal((2, rows)) * 0.4).astype("bfloat16")
    dev = _metal()
    f = jax.jit(lambda *a_: dense_sub(*a_, columns=cols))
    qmm.reset_stats()
    outs = []
    with jax.default_device(dev):
        for part in (a, b):
            args = [jax.device_put(v, dev) for v in
                    (part[1], part[2], part[3], part[4], x)]
            outs.append(np.asarray(f(*args)).astype(np.float32))
    assert qmm.stats()["packs"] == 2, qmm.stats()
    for part, got in zip((a, b), outs):
        exact = x.astype(np.float32) @ part[5]
        np.testing.assert_allclose(got, exact, rtol=2e-2,
                                   atol=2e-2 * np.abs(exact).max())
    assert not np.allclose(outs[0], outs[1])


# --------------------------------------------------------------------------
# the decode loop (the reason this module exists)
# --------------------------------------------------------------------------


@needs_qmm
def test_decode_loop_packs_once_and_matches_cpu():
    """The keras sampler shape: a data-dependent while whose body holds the
    quantized layer. The weights are loop-invariant free captures, so the
    pack must happen once for the whole loop, not once per step."""
    rows = cols = 128
    q, packed, scale, zero, g_idx, wref = _quantize(rows, cols, 128, "bfloat16",
                                                    seed=20)
    x = (np.random.default_rng(21).standard_normal((1, rows)) * 0.3).astype("bfloat16")
    steps = 24

    def f(packed_, scale_, zero_, g_idx_, x_, n):
        def body(state):
            i, h = state
            y = dense_sub(packed_, scale_, zero_, g_idx_, h, cols)
            return i + 1, jnp.tanh(y * jnp.bfloat16(0.3))
        return jax.lax.while_loop(lambda s: s[0] < n, body, (0, x_))[1]

    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x), jnp.int32(steps))
    qmm.reset_stats()
    dev = _metal()
    with jax.default_device(dev):
        # Place the weights once, as a real decode loop does, then step.
        moved = [jax.device_put(a, dev) for a in args]
        jf = jax.jit(f)
        got = np.asarray(jf(*moved)).astype(np.float32)
        # `steps` iterations of a chain that would otherwise be repacked per
        # step, plus a second whole call: still exactly one pack.
        assert qmm.stats()["packs"] == 1, qmm.stats()
        got2 = np.asarray(jf(*moved)).astype(np.float32)
    assert qmm.stats()["packs"] == 1, qmm.stats()
    np.testing.assert_array_equal(got, got2)
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)


def _run_program(f, args):
    """Compile and execute through the engine directly, so the test can see
    the executable's own compile decision (what METALJAX_DEBUG prints as
    `exec ...: compile=True`)."""
    from helpers import lower_bytes
    from metaljax import dtypes as mdt
    from metaljax import engine

    ex = engine.compile_program(lower_bytes(f, *args), "mlir")
    bufs = []
    for a in jax.tree.leaves(args):
        arr = np.asarray(a)
        bufs.append(engine.MetalBuffer(mdt.to_mx(arr),
                                       engine._NP_TO_ENUM[arr.dtype],
                                       list(arr.shape)))
    outs = engine.execute(ex, bufs)
    return ex, [mdt.to_np(o.data) for o in outs]


@needs_qmm
def test_trace_budget_restored_by_the_rewrite():
    """Stacked quantized layers whose literal cost blows the trace budget
    still compile once their reconstruction chains are absorbed."""
    from metaljax.interpreter import COMPILE_ENABLED
    from metaljax.ops import control

    if not COMPILE_ENABLED:
        pytest.skip("METALJAX_COMPILE=0: nothing is traced")

    rows = cols = 64
    layers = 12
    parts = [_quantize(rows, cols, 64, "bfloat16", seed=30 + i)
             for i in range(layers)]
    x = (np.random.default_rng(31).standard_normal((1, rows)) * 0.3).astype("bfloat16")

    def f(ws, x_):
        h = x_
        for packed_, scale_, zero_, g_ in ws:
            h = jnp.tanh(dense_sub(packed_, scale_, zero_, g_, h, cols)
                         * jnp.bfloat16(0.2))
        return h

    ws = [(jnp.asarray(p[1]), jnp.asarray(p[2]), jnp.asarray(p[3]),
           jnp.asarray(p[4])) for p in parts]
    args = (ws, jnp.asarray(x))

    # The literal chain costs ~85 units per layer, the rewritten one ~5.
    budget = control._TRACE_BUDGET
    try:
        control._TRACE_BUDGET = layers * 20
        qmm.reset_stats()
        ex, outs = _run_program(f, args)
        assert qmm.stats()["packs"] == layers, qmm.stats()
        assert ex._can_compile is True   # the whole main still fits
        qmm.reset_stats()
        try:
            qmm.ENABLED = False
            plain, outs_plain = _run_program(f, args)
        finally:
            qmm.ENABLED = True
        assert plain._can_compile is False  # ... and would not without this
    finally:
        control._TRACE_BUDGET = budget
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(outs[0].astype(np.float32), want,
                               rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(outs_plain[0].astype(np.float32), want,
                               rtol=5e-2, atol=5e-2)


@needs_qmm
def test_cost_model_charges_the_fused_dot():
    """The recognized chain must not count toward the trace budget."""
    import io
    from metaljax.interpreter import Interpreter
    from metaljax.ops import control

    rows = cols = 128
    q, packed, scale, zero, g_idx, wref = _quantize(rows, cols, 128, "bfloat16",
                                                    seed=40)
    x = (np.random.default_rng(41).standard_normal((1, rows)) * 0.3).astype("bfloat16")
    lowered = jax.jit(lambda *a: dense_sub(*a, columns=cols)).lower(
        jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
        jnp.asarray(g_idx), jnp.asarray(x))
    buf = io.BytesIO()
    lowered.compiler_ir().operation.write_bytecode(buf)

    interp = Interpreter(buf.getvalue())
    plain = Interpreter(buf.getvalue())
    plain._qmm = qmm.State()          # analysis disabled: literal cost
    with interp.context:
        st = qmm.analyze(interp)
        fused = control._block_cost(interp, interp._main_block())
    with plain.context:
        literal = control._block_cost(plain, plain._main_block())
    assert len(st.matches) == 1
    assert fused <= 8, fused
    assert literal > 60, literal
