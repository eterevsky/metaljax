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
needs_qmm = pytest.mark.skipif(not qmm.QMM_ENABLED, reason="METALJAX_QMM=0")


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


def einsum_out_proj(packed, scale, zero, g_idx, x, n, h, d):
    """keras EinsumDense._int4_call for an attention output projection.

    `btnh,nhd->btd`: the kernel is stored (and grouped) as [n*h, d] in
    n-major order, then reshaped to [n, h, d]. jax lowers this einsum with
    `contracting_dims = [3, 2] x [1, 0]` -- the REVERSED pairing -- so the
    recognizer's canonical K axis runs h-major and the quantization groups
    arrive interleaved.
    """
    w = _unpack_nibbles(packed, d)
    g = g_idx.astype(jnp.int32)
    s = jnp.take(scale, g, axis=0)
    z = jnp.take(zero, g, axis=0)
    wf = (w.astype(x.dtype) - z.astype(x.dtype)) * s
    return jnp.einsum("btnh,nhd->btd", x, jnp.reshape(wf, (n, h, d)))


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


def _compare(f, args, exact, packs=1, perms=None):
    """Run on metal + CPU; both must match the exact f32 dequant reference,
    and metal must be no worse than CPU."""
    qmm.reset_stats()
    got = _run(f, args, _metal())
    want = _run(f, args, _cpu())
    if qmm.QMM_ENABLED:
        assert qmm.stats()["packs"] == packs, qmm.stats()
        if perms is not None:
            assert qmm.stats()["perms"] == perms, qmm.stats()
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
# interleaved groups: recovering the permutation
# --------------------------------------------------------------------------


def _canonical_scale_map(group_of_row, n, h, d=4):
    """The [N, K] scale map the recognizer materializes for `btnh,nhd->btd`.

    Canonical column j = hi * n + ni (h-major, the reversed contracting-dim
    pairing) reads storage row r = ni * h + hi (n-major, keras' layout).
    """
    import mlx.core as mx
    j = np.arange(n * h)
    r = (j % n) * h + (j // n)
    scale = np.arange(1, group_of_row.max() + 2, dtype=np.float32)[:, None]
    scale = scale * np.arange(1, d + 1, dtype=np.float32)[None, :]   # [ng, d]
    return mx.array(scale[group_of_row][r].T)                    # [d, K]


def _out_proj_case(n, h, d, block=128, seed=70, g_idx=None):
    rows = n * h
    q, packed, scale, zero, g_idx, wref = _quantize(
        rows, d, block, "bfloat16", seed=seed, g_idx=g_idx)
    x = (np.random.default_rng(seed + 1).standard_normal((1, 2, n, h))
         * 0.3).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    exact = np.einsum("btnh,nhd->btd", x.astype(np.float32),
                      wref.reshape(n, h, d))
    return args, exact


@needs_qmm
def test_einsum_out_projection_interleaved_groups():
    """The gemma/keras attention output projection: groups interleaved by the
    reversed contracting-dim pairing. Used to fall back; now regroups."""
    n, h, d = 8, 32, 96
    args, exact = _out_proj_case(n, h, d)
    _compare(lambda *a: einsum_out_proj(*a, n=n, h=h, d=d), args, exact,
             perms=1)


@needs_qmm
@pytest.mark.parametrize("n,h,block", [(4, 64, 128), (16, 32, 64),
                                       (2, 128, 128)])
def test_einsum_out_projection_shapes(n, h, block):
    d = 64
    args, exact = _out_proj_case(n, h, d, block=block, seed=80 + n)
    _compare(lambda *a: einsum_out_proj(*a, n=n, h=h, d=d), args, exact,
             perms=1)


@needs_qmm
def test_interleaved_groups_in_a_decode_loop():
    """The same layer inside a data-dependent while: the permutation has to
    travel into the compiled body with the packed weights."""
    n, h = 8, 32                    # n*h = 256: two groups of 128, interleaved
    d = n * h                       # square, so the loop can carry the state
    rows = n * h
    q, packed, scale, zero, g_idx, wref = _quantize(rows, d, 128, "bfloat16",
                                                    seed=90)
    x = (np.random.default_rng(91).standard_normal((1, 2, n, h))
         * 0.3).astype("bfloat16")
    steps = 8

    def f(packed_, scale_, zero_, g_idx_, x_, k):
        def body(state):
            i, hs = state
            y = einsum_out_proj(packed_, scale_, zero_, g_idx_, hs, n, h, d)
            y = jnp.tanh(y * jnp.bfloat16(0.3))
            return i + 1, jnp.reshape(y, (1, 2, n, h))
        return jax.lax.while_loop(lambda s: s[0] < k, body, (0, x_))[1]

    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x), jnp.int32(steps))
    qmm.reset_stats()
    dev = _metal()
    with jax.default_device(dev):
        moved = [jax.device_put(a, dev) for a in args]
        jf = jax.jit(f)
        got = np.asarray(jf(*moved)).astype(np.float32)
        assert qmm.stats() == {"recognized": 1, "packs": 1, "fallbacks": 0,
                               "perms": 1, "mxfp4": 0, "batched": 0,
                               "shared": 0}, \
            qmm.stats()
        got2 = np.asarray(jf(*moved)).astype(np.float32)
    np.testing.assert_array_equal(got, got2)
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)


@needs_qmm
def test_second_weight_set_with_a_different_permutation():
    """Two weight sets whose groups interleave DIFFERENTLY through one
    executable. The permutation travels as a graph input, so the second
    call must not reuse the first one's."""
    n, h, d = 8, 32, 64
    rows = n * h
    # (a) groups along the stored (n-major) axis; (b) groups by the parity of
    # the h index -- a different interleaving, hence a different permutation.
    alt = (np.arange(rows) % 2).astype(np.float32)
    parts = [_out_proj_case(n, h, d, seed=100),
             _out_proj_case(n, h, d, seed=110, g_idx=alt)]
    # The premise: the recognizer really does derive two different
    # permutations for these (checked on the maps it would build).
    perms = [qmm._regroup(rows, (_canonical_scale_map(g, n, h), None))
             for g in (np.arange(rows) // 128, alt.astype(np.int64))]
    assert all(p is not None for p in perms)
    assert not np.array_equal(*perms)
    qmm.reset_stats()
    dev = _metal()
    f = jax.jit(lambda *a: einsum_out_proj(*a, n=n, h=h, d=d))
    outs = []
    with jax.default_device(dev):
        for args, _exact in parts:
            moved = [jax.device_put(a, dev) for a in args]
            outs.append(np.asarray(f(*moved)).astype(np.float32))
    assert qmm.stats()["packs"] == 2 and qmm.stats()["perms"] == 2, qmm.stats()
    for (args, exact), got in zip(parts, outs):
        np.testing.assert_allclose(got, exact, rtol=3e-2,
                                   atol=3e-2 * np.abs(exact).max())
    assert not np.allclose(outs[0], outs[1])


def _maps(scale, zero=None):
    import mlx.core as mx
    return (mx.array(scale),
            None if zero is None else mx.array(zero))


def test_regroup_clusters_interleaved_columns():
    """The clustering itself: stride-interleaved groups become runs."""
    import mlx.core as mx
    rng = np.random.default_rng(120)
    k, ngroups, rows = 256, 2, 8
    order = np.arange(k) % ngroups          # stride-2 interleave
    gvals = rng.standard_normal((rows, ngroups)).astype(np.float32)
    scale = gvals[:, order]
    zero = np.arange(ngroups, dtype=np.float32)[order] * np.ones((rows, 1),
                                                                 np.float32)
    smap, zmap = _maps(scale, zero)
    assert qmm._pick_group(k, smap, zmap) is None
    perm = qmm._regroup(k, (smap, zmap))
    assert perm is not None
    assert sorted(perm.tolist()) == list(range(k))
    pi = mx.array(perm)
    assert qmm._pick_group(k, qmm._take_k(smap, pi),
                           qmm._take_k(zmap, pi)) == 128


def test_regroup_merges_duplicate_group_columns():
    """Two groups whose scale AND zero columns are identical cluster into one
    double-length run. Harmless: every value in the run is equal, so any
    group size dividing it reconstructs exactly."""
    import mlx.core as mx
    rng = np.random.default_rng(121)
    k, rows = 384, 8
    gvals = rng.standard_normal((rows, 3)).astype(np.float32)
    gvals[:, 2] = gvals[:, 0]               # group 2 duplicates group 0
    order = np.arange(k) % 3
    smap, zmap = _maps(gvals[:, order])
    perm = qmm._regroup(k, (smap, zmap))
    assert perm is not None
    pi = mx.array(perm)
    permuted = qmm._take_k(smap, pi)
    assert qmm._pick_group(k, permuted, None) == 128
    # the merged run is 256 long and constant, so the pack is still exact
    want = np.array(smap)[:, perm]
    np.testing.assert_array_equal(np.array(permuted), want)


def test_regroup_rejects_uneven_runs():
    """Cluster lengths with no common legal group size -> no permutation."""
    rng = np.random.default_rng(122)
    k, rows = 256, 4
    ids = np.concatenate([np.zeros(129), np.ones(127)]).astype(np.int64)
    gvals = rng.standard_normal((rows, 2)).astype(np.float32)
    smap, _ = _maps(gvals[:, ids])
    assert qmm._regroup(k, (smap, None)) is None
    # ... and a map whose every column differs
    smap, _ = _maps(rng.standard_normal((rows, k)).astype(np.float32))
    assert qmm._regroup(k, (smap, None)) is None


def test_regroup_short_circuits_on_already_grouped_maps():
    """An already-contiguous map must keep the identity (no-take) path."""
    rng = np.random.default_rng(123)
    k, rows = 256, 4
    gvals = rng.standard_normal((rows, 2)).astype(np.float32)
    smap, _ = _maps(np.repeat(gvals, 128, axis=1))
    assert qmm._pick_group(k, smap, None) == 128
    assert qmm._regroup(k, (smap, None)) is None


@needs_qmm
def test_identity_permutation_is_the_zero_overhead_path():
    """A dot whose groups are already contiguous carries no permutation."""
    rows, cols = 256, 128
    q, packed, scale, zero, g_idx, wref = _quantize(rows, cols, 128,
                                                    "bfloat16", seed=124)
    x = (np.random.default_rng(125).standard_normal((4, rows))
         * 0.5).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    exact = x.astype(np.float32) @ wref
    _compare(lambda *a: dense_sub(*a, columns=cols), args, exact, perms=0)


# --------------------------------------------------------------------------
# fallbacks: still correct, just not rewritten
# --------------------------------------------------------------------------


def _fallback_case(f, args, exact):
    """Recognized structurally, then rejected by a first-execute check."""
    qmm.reset_stats()
    got = _run(f, args, _metal())
    st = qmm.stats()
    assert st["packs"] == 0, st
    if qmm.QMM_ENABLED:
        assert st["fallbacks"] >= 1, st
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-2,
                               atol=1e-2 * np.abs(exact).max())


def test_shuffled_g_idx_is_regrouped():
    """A g_idx that scatters each group's members across the axis still has
    equal-sized groups of identical (scale, zero) columns: the permutation
    recovers them (it used to fall back to the literal chain)."""
    rows, cols, block = 256, 64, 128
    rng = np.random.default_rng(12)
    g_idx = rng.permutation(np.arange(rows) // block).astype(np.float32)
    q, packed, scale, zero, g_idx, wref = _quantize(
        rows, cols, block, "bfloat16", seed=13, g_idx=g_idx)
    x = (rng.standard_normal((2, rows)) * 0.4).astype("bfloat16")
    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    _compare(lambda *a: dense_sub(*a, columns=cols), args,
             x.astype(np.float32) @ wref, perms=1)


def test_fallback_uneven_group_sizes():
    """Groups of different sizes cannot be made to line up on any legal group
    boundary -> literal chain."""
    rows, cols, block = 256, 64, 128
    rng = np.random.default_rng(12)
    g_idx = (np.arange(rows) >= 129).astype(np.float32)   # 129 / 127
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
    # METALJAX_QMM=0 maps to QMM_ENABLED (the quantized-recognizer gate);
    # qmm.ENABLED is the shared prologue gate that MoE also rides on. On the
    # merged tree sdpa's overlap check reaches qmm.analyze even when the
    # prologue is off, so the recognizer gate is the one that must hold.
    monkeypatch.setattr(qmm, "QMM_ENABLED", False)
    monkeypatch.setattr(qmm, "ENABLED", False)
    qmm.reset_stats()
    got = _run(lambda *a: dense_sub(*a, columns=cols), args, _metal())
    assert qmm.stats() == {"recognized": 0, "packs": 0, "fallbacks": 0,
                           "perms": 0, "mxfp4": 0, "batched": 0,
                           "shared": 0}
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
def test_trace_budget_restored_for_interleaved_groups():
    """The real regression this fixes: a decode body of attention-projection
    layers whose groups are interleaved stayed EAGER because every one of
    them fell back to the literal chain."""
    from metaljax.interpreter import COMPILE_ENABLED
    from metaljax.ops import control

    if not COMPILE_ENABLED:
        pytest.skip("METALJAX_COMPILE=0: nothing is traced")

    n, h = 8, 32                     # two interleaved groups per layer
    d = n * h                        # square: the layers stack
    layers = 10
    parts = [_quantize(n * h, d, 128, "bfloat16", seed=140 + i)
             for i in range(layers)]
    x = (np.random.default_rng(141).standard_normal((1, 2, n, h))
         * 0.3).astype("bfloat16")

    def f(ws, x_):
        hs = x_
        for packed_, scale_, zero_, g_ in ws:
            y = einsum_out_proj(packed_, scale_, zero_, g_, hs, n, h, d)
            hs = jnp.reshape(jnp.tanh(y * jnp.bfloat16(0.2)), (1, 2, n, h))
        return hs

    ws = [(jnp.asarray(p[1]), jnp.asarray(p[2]), jnp.asarray(p[3]),
           jnp.asarray(p[4])) for p in parts]
    args = (ws, jnp.asarray(x))

    budget = control._TRACE_BUDGET
    try:
        control._TRACE_BUDGET = layers * 20
        qmm.reset_stats()
        ex, outs = _run_program(f, args)
        assert qmm.stats()["packs"] == layers, qmm.stats()
        assert qmm.stats()["perms"] == layers, qmm.stats()
        assert ex._can_compile is True
        qmm.reset_stats()
        try:
            qmm.ENABLED = False
            plain, outs_plain = _run_program(f, args)
        finally:
            qmm.ENABLED = True
        assert plain._can_compile is False
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
