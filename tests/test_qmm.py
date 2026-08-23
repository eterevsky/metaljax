"""Quantized-matmul recognizer, tested through the native PJRT plugin.

Every case drives plain JAX (`jax.jit` on the metal device, `device_put`)
and is checked against jax-CPU *and* against an exactly-dequantized float32
reference: the rewrite accumulates in f32 over an exactly reconstructed
weight, so it lands closer to the exact answer than the literal bf16 chain
does. The native recognizer reads METALJAX_QMM / METALJAX_QMM_SCALES /
METALJAX_QMM_BLOCK / METALJAX_QMM_BUILD_CACHE / METALJAX_QMM_BATCH; with
the rewrite off (METALJAX_QMM=0) the numeric comparisons here still hold
-- the literal chain must be right too.

The layer graphs are hand-built copies of what keras 3.15 emits for
`Dense.quantize("int4")` (sub-channel, block_size=128, asymmetric with a
zero point and a `g_idx` group ramp), its per-channel variant
(block_size=-1, symmetric, scale divides the OUTPUT) and `EinsumDense`
(`btd,dnh->btnh`, kernel reshaped right before the dot) -- keras is not
installed in this venv.

The Python-introspection tests (pack_exact/pack_codes bit-exactness,
_regroup clustering, stats counters, trace-budget and build-cache
internals) were removed at the Stage-1 retirement: the Python engine is
gone, the recognizer is re-implemented in C++ inside the plugin, and only
its numeric behavior is observable from here.
"""

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _metal():
    return jax.devices("metal")[0]


def _cpu():
    return jax.devices("cpu")[0]


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


def _compare(f, args, exact):
    """Run on metal + CPU; both must match the exact f32 dequant reference,
    and metal must be no worse than CPU."""
    got = _run(f, args, _metal())
    want = _run(f, args, _cpu())
    scale = np.abs(exact).max()
    err_metal = np.abs(got - exact).max() / scale
    err_cpu = np.abs(want - exact).max() / scale
    assert got.shape == want.shape
    # bf16 outputs round at ~4e-3; the comparison that matters is that the
    # rewrite is not WORSE than the literal chain the CPU backend runs.
    # (Only under the default scale policy: METALJAX_QMM_SCALES=source
    # deliberately trades up to ~2x of this error for scale traffic.)
    if os.environ.get("METALJAX_QMM_SCALES", "auto") == "auto":
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


def test_einsum_out_projection_interleaved_groups():
    """The gemma/keras attention output projection: groups interleaved by the
    reversed contracting-dim pairing. Used to fall back; now regroups."""
    n, h, d = 8, 32, 96
    args, exact = _out_proj_case(n, h, d)
    _compare(lambda *a: einsum_out_proj(*a, n=n, h=h, d=d), args, exact)


@pytest.mark.parametrize("n,h,block", [(4, 64, 128), (16, 32, 64),
                                       (2, 128, 128)])
def test_einsum_out_projection_shapes(n, h, block):
    d = 64
    args, exact = _out_proj_case(n, h, d, block=block, seed=80 + n)
    _compare(lambda *a: einsum_out_proj(*a, n=n, h=h, d=d), args, exact)


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
    dev = _metal()
    with jax.default_device(dev):
        moved = [jax.device_put(a, dev) for a in args]
        jf = jax.jit(f)
        got = np.asarray(jf(*moved)).astype(np.float32)
        got2 = np.asarray(jf(*moved)).astype(np.float32)
    np.testing.assert_array_equal(got, got2)
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)


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
    dev = _metal()
    f = jax.jit(lambda *a: einsum_out_proj(*a, n=n, h=h, d=d))
    outs = []
    with jax.default_device(dev):
        for args, _exact in parts:
            moved = [jax.device_put(a, dev) for a in args]
            outs.append(np.asarray(f(*moved)).astype(np.float32))
    for (args, exact), got in zip(parts, outs):
        np.testing.assert_allclose(got, exact, rtol=3e-2,
                                   atol=3e-2 * np.abs(exact).max())
    assert not np.allclose(outs[0], outs[1])


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
    _compare(lambda *a: dense_sub(*a, columns=cols), args, exact)


# --------------------------------------------------------------------------
# fallbacks: still correct, just not rewritten
# --------------------------------------------------------------------------


def _fallback_case(f, args, exact):
    """Recognized structurally, then rejected by a first-execute check --
    the literal chain must still be numerically right."""
    got = _run(f, args, _metal())
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
             x.astype(np.float32) @ wref)


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
    got = _run(f, args, _metal())
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=1e-1)


def test_new_weight_buffers_repack_without_stale_results():
    """A second set of weights through the SAME executable must not reuse
    the first pack (the packed arrays travel as compiled-graph inputs)."""
    rows, cols = 128, 64
    a = _quantize(rows, cols, 64, "bfloat16", seed=22)
    b = _quantize(rows, cols, 64, "bfloat16", seed=23)
    x = (np.random.default_rng(24).standard_normal((2, rows)) * 0.4).astype("bfloat16")
    dev = _metal()
    f = jax.jit(lambda *a_: dense_sub(*a_, columns=cols))
    outs = []
    with jax.default_device(dev):
        for part in (a, b):
            args = [jax.device_put(v, dev) for v in
                    (part[1], part[2], part[3], part[4], x)]
            outs.append(np.asarray(f(*args)).astype(np.float32))
    for part, got in zip((a, b), outs):
        exact = x.astype(np.float32) @ part[5]
        np.testing.assert_allclose(got, exact, rtol=2e-2,
                                   atol=2e-2 * np.abs(exact).max())
    assert not np.allclose(outs[0], outs[1])


# --------------------------------------------------------------------------
# the decode loop (the reason this module exists)
# --------------------------------------------------------------------------


def test_decode_loop_packs_once_and_matches_cpu():
    """The keras sampler shape: a data-dependent while whose body holds the
    quantized layer. The weights are loop-invariant free captures, so the
    pack must serve every step of the loop -- and a second whole call --
    without drifting."""
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
    dev = _metal()
    with jax.default_device(dev):
        # Place the weights once, as a real decode loop does, then step.
        moved = [jax.device_put(a, dev) for a in args]
        jf = jax.jit(f)
        got = np.asarray(jf(*moved)).astype(np.float32)
        got2 = np.asarray(jf(*moved)).astype(np.float32)
    np.testing.assert_array_equal(got, got2)
    want = _run(f, args, _cpu())
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)


# --------------------------------------------------------------------------
# the cross-executable build cache (numeric behavior only)
# --------------------------------------------------------------------------


def _on_device(weights):
    """`weights` placed ONCE: the cache keys on buffer identity, so a second
    `device_put` of the same array is a different weight as far as it is
    concerned (correctly -- it really is a different buffer)."""
    dev = _metal()
    with jax.default_device(dev):
        return [jax.device_put(v, dev) for v in weights]


def _scaled_sub(packed, scale, zero, g_idx, x, columns, const):
    """`dense_sub` with one extra constant folded into the scale map.

    The two variants differ in NOTHING but that constant's attribute, and
    they read the very same buffers -- so a fingerprint that skipped op
    attributes would hand the second graph the first one's weight.
    """
    w = _unpack_nibbles(packed, columns)
    g = g_idx.astype(jnp.int32)
    s = jnp.take(scale, g, axis=0) * jnp.asarray(const, x.dtype)
    z = jnp.take(zero, g, axis=0)
    return x @ ((w.astype(x.dtype) - z.astype(x.dtype)) * s)


def test_different_weight_content_misses_the_build_cache():
    """Same shapes, same dtypes, different bytes: identity has to catch it.

    The fingerprint cannot -- the two graphs ARE the same graph. What
    separates them is that the second execute hands the recognizer different
    buffers; a stale cache hit would come back as the wrong numbers, which
    are asserted per weight set.
    """
    rows, cols = 128, 64
    a = _quantize(rows, cols, 64, "float32", seed=42)
    b = _quantize(rows, cols, 64, "float32", seed=43)
    x = (np.random.default_rng(44).standard_normal((2, rows))
         * 0.4).astype("float32")
    dev = _metal()
    f = jax.jit(lambda *a_: dense_sub(*a_, columns=cols))
    outs = []
    with jax.default_device(dev):
        for part in (a, b):
            moved = [jax.device_put(v, dev)
                     for v in (part[1], part[2], part[3], part[4], x)]
            outs.append(np.asarray(f(*moved)))
    for part, got in zip((a, b), outs):
        exact = x @ part[5]
        np.testing.assert_allclose(got, exact, rtol=1e-5,
                                   atol=1e-5 * np.abs(exact).max())
    assert not np.allclose(outs[0], outs[1])


def test_one_different_constant_misses_the_build_cache():
    """Same buffers, same shapes, one attribute apart.

    A wrong hit would be silent: both graphs pack to a legal weight, so
    only the numbers say which one came back. The first executable is HELD
    so any pack it built stays live while the second variant runs over the
    very same buffers; each variant's output is asserted against its own
    exact reference.
    """
    rows, cols = 128, 64
    part = _quantize(rows, cols, 64, "float32", seed=45)
    moved = _on_device(part[1:5])
    x = (np.random.default_rng(46).standard_normal((2, rows))
         * 0.4).astype("float32")
    dev = _metal()
    held = []
    with jax.default_device(dev):
        xd = jax.device_put(x, dev)
        for const in (2.0, 3.0):
            f = jax.jit(lambda *a, c=const: _scaled_sub(*a, columns=cols,
                                                        const=c))
            held.append(f)
            out = np.asarray(f(*moved, xd))
            exact = x @ (part[5] * const)
            np.testing.assert_allclose(out, exact, rtol=1e-5,
                                       atol=1e-5 * np.abs(exact).max())
