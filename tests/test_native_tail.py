"""Stage 2: the decline tail on the native tape, differential.

The families here are the ones the M2-era op set left out — pad, the bit
counters, IEEE totalOrder compare, general reduce bodies, reduce_window,
rng_bit_generator, complex64. Each is a transliteration of the Python
handler in src/metaljax/ops/, so the differential is over BYTES: same
inputs, same MLX calls, same bits. Where an engine is itself
nondeterministic (a GPU scatter's summation order) the case says so.

Cases are hand-written StableHLO where the op is easier to say than to
coax out of jax (pad's interior dilation, a variadic reduce body) and
jax-lowered where jax's own lowering is the thing under test (jnp.fft,
jax.random, jnp.pad's negative edges).
"""
import numpy as np
import pytest

from test_native_tape import (  # noqa: F401  (imports gate the native build)
    _f32,
    _mod,
    _native_engine,
    _specials,
    check,
    engine,
    native,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from helpers import lower_bytes  # noqa: E402

RNG = np.random.default_rng(20260810)


def _diff(f, *args, lowered=True, compiled=False):
    """Both engines on a jitted function; returns the native executable."""
    mod = lower_bytes(f, *args)
    flat = [np.asarray(x) for x in jax.tree.leaves(args)]
    return check(mod, flat, lowered=lowered, compiled=compiled)


# --------------------------------------------------------------------------
# stablehlo.pad
# --------------------------------------------------------------------------


def test_pad_edges():
    t = "tensor<3x4xf32>"
    out = "tensor<6x5xf32>"
    mod = _mod([("a", t), ("v", "tensor<f32>")], [out], f"""
    %0 = stablehlo.pad %a, %v, low = [1, 0], high = [2, 1],
         interior = [0, 0] : ({t}, tensor<f32>) -> {out}
    return %0 : {out}""")
    check(mod, [_f32(), np.float32(-1.5)])


def test_pad_interior_dilation():
    t = "tensor<3x4xf32>"
    out = "tensor<5x10xf32>"
    mod = _mod([("a", t), ("v", "tensor<f32>")], [out], f"""
    %0 = stablehlo.pad %a, %v, low = [0, 0], high = [0, 0],
         interior = [1, 2] : ({t}, tensor<f32>) -> {out}
    return %0 : {out}""")
    check(mod, [_f32(), np.float32(7.25)])


def test_pad_interior_and_edges_together():
    t = "tensor<3x4xf32>"
    out = "tensor<8x11xf32>"
    mod = _mod([("a", t), ("v", "tensor<f32>")], [out], f"""
    %0 = stablehlo.pad %a, %v, low = [2, 1], high = [1, 0],
         interior = [1, 2] : ({t}, tensor<f32>) -> {out}
    return %0 : {out}""")
    check(mod, [_f32(), np.float32(0.0)])


def test_pad_negative_edges_crop():
    t = "tensor<4x5xi32>"
    out = "tensor<2x6xi32>"
    mod = _mod([("a", t), ("v", "tensor<i32>")], [out], f"""
    %0 = stablehlo.pad %a, %v, low = [-1, 2], high = [-1, -1],
         interior = [0, 0] : ({t}, tensor<i32>) -> {out}
    return %0 : {out}""")
    a = RNG.integers(-50, 50, (4, 5)).astype(np.int32)
    check(mod, [a, np.int32(9)])


def test_pad_negative_with_interior():
    t = "tensor<3x3xf32>"
    out = "tensor<4x5xf32>"
    mod = _mod([("a", t), ("v", "tensor<f32>")], [out], f"""
    %0 = stablehlo.pad %a, %v, low = [-1, 0], high = [0, 0],
         interior = [1, 1] : ({t}, tensor<f32>) -> {out}
    return %0 : {out}""")
    check(mod, [_f32((3, 3)), np.float32(2.0)])


def test_pad_zero_size_operand():
    t = "tensor<0x4xf32>"
    out = "tensor<3x4xf32>"
    mod = _mod([("a", t), ("v", "tensor<f32>")], [out], f"""
    %0 = stablehlo.pad %a, %v, low = [1, 0], high = [2, 0],
         interior = [1, 0] : ({t}, tensor<f32>) -> {out}
    return %0 : {out}""")
    check(mod, [np.zeros((0, 4), np.float32), np.float32(3.5)])


def test_pad_specials_keep_their_bits():
    t = "tensor<12xf32>"
    out = "tensor<25xf32>"
    mod = _mod([("a", t), ("v", "tensor<f32>")], [out], f"""
    %0 = stablehlo.pad %a, %v, low = [1], high = [1],
         interior = [1] : ({t}, tensor<f32>) -> {out}
    return %0 : {out}""")
    check(mod, [_specials(), np.float32(-0.0)])


def test_pad_from_jax():
    def f(x):
        return jnp.pad(x, ((2, 1), (0, 3)), constant_values=1.5)
    _diff(f, _f32((4, 5)))


# --------------------------------------------------------------------------
# popcnt / count_leading_zeros
# --------------------------------------------------------------------------

_BIT_DTYPES = {
    "i8": ("i8", np.int8),
    "i32": ("i32", np.int32),
    "i64": ("i64", np.int64),
    "ui8": ("ui8", np.uint8),
    "ui32": ("ui32", np.uint32),
    "ui64": ("ui64", np.uint64),
}


def _bit_values(npdt):
    info = np.iinfo(npdt)
    vals = [0, 1, 2, 3, 7, 8, 255, info.max, info.min, -1 if info.min else 0]
    with np.errstate(over="ignore"):
        return np.array([np.array(v).astype(npdt) for v in vals], dtype=npdt)


@pytest.mark.parametrize("name", sorted(_BIT_DTYPES))
@pytest.mark.parametrize("op", ["popcnt", "count_leading_zeros"])
def test_bit_counters(name, op):
    el, npdt = _BIT_DTYPES[name]
    t = f"tensor<10x{el}>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.{op} %a : {t}
    return %0 : {t}""")
    check(mod, [_bit_values(npdt)])


def test_bit_counters_from_jax():
    def f(x):
        return jax.lax.population_count(x), jax.lax.clz(x)
    _diff(f, RNG.integers(0, 1 << 30, (3, 4)).astype(np.uint32))


# --------------------------------------------------------------------------
# TOTALORDER compare
# --------------------------------------------------------------------------


@pytest.mark.parametrize("el,npdt", [("f32", np.float32), ("f16", np.float16)])
@pytest.mark.parametrize("d", ["LT", "GT", "EQ"])
def test_totalorder_compare(el, npdt, d):
    t = f"tensor<12x{el}>"
    mod = _mod([("a", t), ("b", t)], ["tensor<12xi1>"], f"""
    %0 = stablehlo.compare {d}, %a, %b, TOTALORDER : ({t}, {t})
         -> tensor<12xi1>
    return %0 : tensor<12xi1>""")
    a = _specials(npdt)
    b = np.roll(a, 3)
    check(mod, [a, b])


def test_generic_reduce_bitwise():
    """A body neither reducer table names: the pairwise halving path."""
    x = RNG.integers(0, 1 << 20, 9).astype(np.int32)
    for fn in (jnp.bitwise_and, jnp.bitwise_or, jnp.bitwise_xor):
        _diff(lambda v, fn=fn: fn.reduce(v), x)


def test_generic_reduce_odd_extent_pads_with_the_init():
    """An odd extent concatenates the init before halving — the round that
    reads the init is exactly where an off-by-one would show up."""
    for n in (1, 2, 3, 5, 7, 8, 17):
        _diff(lambda v: jnp.bitwise_xor.reduce(v),
              RNG.integers(0, 1 << 20, n).astype(np.int32))


def test_generic_reduce_variadic_min_with_index():
    def min_with_index(v):
        def body(a, b):
            (av, ai), (bv, bi) = a, b
            pick = (av < bv) | ((av == bv) & (ai < bi))
            return jnp.where(pick, av, bv), jnp.where(pick, ai, bi)
        return jax.lax.reduce(
            (v, jnp.arange(v.shape[0], dtype=jnp.int32)),
            (jnp.inf, jnp.int32(2**31 - 1)), body, (0,))
    _diff(min_with_index, np.array([3.0, 1.0, 2.0, 1.0], np.float32))
    _diff(min_with_index, np.array([3.0, 1.0, 2.0, 1.0, 0.5], np.float32))


def test_generic_reduce_several_axes_and_batch():
    x = RNG.integers(0, 1 << 20, (3, 4, 5)).astype(np.int32)
    _diff(lambda v: jnp.bitwise_or.reduce(v, axis=(0, 2)), x)
    _diff(lambda v: jnp.bitwise_and.reduce(v, axis=1), x)


def test_generic_reduce_empty_axis_returns_the_init():
    _diff(lambda v: jnp.bitwise_or.reduce(v, axis=0),
          np.zeros((0, 3), np.int32))


def test_generic_reduce_body_reading_a_capture():
    """The body reads a value from the enclosing block, which the region
    carries as a trailing argument.

    jax refuses to build one of these ("Reduction computations can't close
    over Tracers"), so the module is written by hand — the capture path is
    shared with while/if and this is the only way to reach it here.
    """
    t = "tensor<6xi32>"
    mod = _mod([("a", t), ("m", "tensor<i32>")], ["tensor<i32>"], f"""
    %init = stablehlo.constant dense<0> : tensor<i32>
    %0 = "stablehlo.reduce"(%a, %init) <{{dimensions = array<i64: 0>}}> ({{
    ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
      %x = stablehlo.and %lhs, %m : tensor<i32>
      %y = stablehlo.or %x, %rhs : tensor<i32>
      stablehlo.return %y : tensor<i32>
    }}) : ({t}, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>""")
    check(mod, [RNG.integers(0, 1 << 20, 6).astype(np.int32), np.int32(0x0F0F)])


# --------------------------------------------------------------------------
# rng_bit_generator (Philox / ThreeFry)
# --------------------------------------------------------------------------

_ALGS = {
    "default": jax.lax.RandomAlgorithm.RNG_DEFAULT,
    "philox": jax.lax.RandomAlgorithm.RNG_PHILOX,
    "threefry": jax.lax.RandomAlgorithm.RNG_THREE_FRY,
}

_KEY = np.array([1, 2, 3, 4], np.uint32)


@pytest.mark.parametrize("alg", sorted(_ALGS))
@pytest.mark.parametrize("shape", [(8,), (7,), (3, 5), (1,), (2, 3, 4)])
def test_rng_shapes(alg, shape):
    """The block/half schedule is the whole family: an odd count, a shape
    with no even dim, a rank-3 output all take different arms."""
    def f(k):
        return jax.lax.rng_bit_generator(
            k, shape, dtype=jnp.uint32, algorithm=_ALGS[alg])
    _diff(f, _KEY)


@pytest.mark.parametrize("alg", sorted(_ALGS))
@pytest.mark.parametrize("dt", ["uint8", "uint16", "uint32"])
def test_rng_narrow_outputs(alg, dt):
    def f(k):
        return jax.lax.rng_bit_generator(
            k, (9,), dtype=jnp.dtype(dt), algorithm=_ALGS[alg])
    _diff(f, _KEY)


@pytest.mark.parametrize("state", ["ui64", "ui32"])
def test_rng_philox_64_bit(state):
    """The 64-bit arm pairs two u32 words per element. Hand-written: jax
    refuses a u64 output without x64, and the state form (four u32 words
    vs two u64 ones) is the other axis the handler branches on."""
    st = "tensor<2xui64>" if state == "ui64" else "tensor<4xui32>"
    mod = _mod([("s", st)], [st, "tensor<5xui64>"], f"""
    %state, %out = stablehlo.rng_bit_generator %s, algorithm = PHILOX :
        ({st}) -> ({st}, tensor<5xui64>)
    return %state, %out : {st}, tensor<5xui64>""")
    key = (np.array([7, 11], np.uint64) if state == "ui64"
           else np.array([1, 2, 3, 4], np.uint32))
    check(mod, [key])


@pytest.mark.parametrize("alg", sorted(_ALGS))
@pytest.mark.parametrize("shape", [(0,), (0, 5)])
def test_rng_empty_output_returns_the_state(alg, shape):
    def f(k):
        return jax.lax.rng_bit_generator(
            k, shape, dtype=jnp.uint32, algorithm=_ALGS[alg])
    _diff(f, _KEY)


def test_rng_scalar_output():
    def f(k):
        return jax.lax.rng_bit_generator(
            k, (), dtype=jnp.uint32,
            algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY)
    _diff(f, _KEY)


def test_rng_through_jax_random():
    """The consumer that matters: jax.random on an rbg key, whose bits feed
    a float conversion where one wrong word is visible everywhere."""
    k = jax.random.key_data(jax.random.key(42, impl="rbg"))

    def uni(kd):
        return jax.random.uniform(
            jax.random.wrap_key_data(kd, impl="rbg"), (16,))

    def norm(kd):
        return jax.random.normal(
            jax.random.wrap_key_data(kd, impl="rbg"), (4, 4))
    _diff(uni, np.asarray(k))
    _diff(norm, np.asarray(k))


def test_rng_state_advances_across_calls():
    """Two draws in one program: the second reads the first's state, so a
    counter that advanced by the wrong amount shows up in the bits."""
    def f(k):
        s1, b1 = jax.lax.rng_bit_generator(k, (5,), dtype=jnp.uint32)
        s2, b2 = jax.lax.rng_bit_generator(s1, (5,), dtype=jnp.uint32)
        return s2, b1, b2
    _diff(f, _KEY)


# --------------------------------------------------------------------------
# reverse
# --------------------------------------------------------------------------


def test_reverse_axes():
    x = _f32((3, 4, 5))
    _diff(lambda v: v[::-1], x)
    _diff(lambda v: jnp.flip(v, axis=(0, 2)), x)
    _diff(lambda v: jnp.flip(v), x)


def test_reverse_unit_and_empty_dims():
    """Extent 0 or 1 is identity — the Python handler skips the take, and
    mx.take on an empty axis would raise."""
    _diff(lambda v: jnp.flip(v, axis=(0, 1)), _f32((1, 4)))
    _diff(lambda v: jnp.flip(v, axis=0), np.zeros((0, 3), np.float32))


# --------------------------------------------------------------------------
# reduce_window
# --------------------------------------------------------------------------


def test_reduce_window_cumulative_peephole():
    v = _f32((9,))
    _diff(lambda x: jnp.cumsum(x), v)
    _diff(lambda x: jnp.cumprod(x), v)
    _diff(lambda x: jax.lax.cummax(x), v)
    _diff(lambda x: jax.lax.cummin(x), v)
    _diff(lambda x: jax.lax.cumsum(x, reverse=True), v)
    _diff(lambda x: jnp.cumsum(x, axis=1), _f32((3, 4)))


def test_reduce_window_pooling():
    from jax import lax
    x = _f32((2, 1, 8, 8))
    _diff(lambda v: lax.reduce_window(v, -jnp.inf, lax.max,
                                      (1, 1, 2, 2), (1, 1, 2, 2), "VALID"), x)
    _diff(lambda v: lax.reduce_window(v, 0.0, lax.add,
                                      (1, 1, 3, 3), (1, 1, 1, 1), "SAME"), x)


def test_reduce_window_dilated():
    """window_dilation folds the window block into a non-unit-stride axis —
    the layout MLX reads stale memory from unless it is materialized."""
    from jax import lax
    x2 = _f32((4, 6))
    for op, iv in ((jax.lax.max, -np.inf), (jax.lax.min, np.inf),
                   (jax.lax.add, 0.0)):
        _diff(lambda v, op=op, iv=iv: lax.reduce_window(
            v, jnp.float32(iv), op, (1, 2), (1, 1), "SAME", (1, 1), (1, 2)),
            x2)
    # base dilation writes the operand into an init-valued array
    _diff(lambda v: lax.reduce_window(
        v, jnp.float32(0.0), jax.lax.add, (1, 2), (1, 1),
        ((0, 1), (1, 0)), (2, 3), (1, 2)), x2)


def test_reduce_window_generic_body():
    """A body that is neither a monoid nor a single compare: the pairwise
    fold over the window axis, with the body as a sub-Program."""
    from jax import lax
    _diff(lambda v: lax.reduce_window(
        v, jnp.float32(-np.inf), jnp.logaddexp, (1, 2), (1, 1), "SAME",
        (1, 1), (1, 2)), _f32((4, 6)))
    _diff(lambda v: jax.lax.cumlogsumexp(v), _f32((9,)))


def test_reduce_window_variadic_select():
    """select_and_gather_add: a variadic window whose one compare picks the
    element every output is read at. jax emits it for the JVP of a max
    pool (its transpose is select_and_scatter, which is a different op and
    still declines)."""
    from jax import lax
    x, t = _f32((2, 1, 6, 6)), _f32((2, 1, 6, 6))

    def f(v, tan):
        return jax.jvp(lambda a: lax.reduce_window(
            a, -jnp.inf, lax.max, (1, 1, 3, 3), (1, 1, 1, 1), "SAME"),
            (v,), (tan,))
    _diff(f, x, t)


def test_select_and_scatter_still_declines():
    """A pool BACKWARD lowers to stablehlo.select_and_scatter, which the
    tape does not carry: its scatter-add over overlapping windows is
    order-nondeterministic on the GPU, so a byte differential could not
    hold it to the Python engine anyway."""
    from jax import lax
    _diff(jax.grad(lambda v: lax.reduce_window(
        v, -jnp.inf, lax.max, (1, 1, 3, 3), (1, 1, 1, 1), "SAME").sum()),
        _f32((2, 1, 6, 6)), lowered=False)


# --------------------------------------------------------------------------
# complex64
# --------------------------------------------------------------------------


def _c64(n=6):
    return (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)).astype(
        np.complex64)


def _c_specials():
    """Every pair of interesting real parts with interesting imaginary ones.

    The C99 rearrangements (scaled hypot, Kahan's csqrt, the exp/expm1
    zero-sin rule, tan's pole) exist FOR these values, so a differential
    that only sees ordinary numbers would prove nothing about them.
    """
    parts = [0.0, -0.0, 1.0, -1.0, 1e-30, 1e20, np.inf, -np.inf, np.nan]
    vals = [complex(re, im) for re in parts for im in parts]
    return np.array(vals, np.complex64)


def test_complex_parts_and_construction():
    def f(re, im):
        z = jax.lax.complex(re, im)
        return z, jnp.real(z), jnp.imag(z), jnp.conj(z)
    re = np.array([0.0, -0.0, 1.0, np.inf, np.nan, -2.5], np.float32)
    im = np.array([-0.0, 0.0, np.inf, 1.0, -0.0, 3.5], np.float32)
    _diff(f, re, im)


@pytest.mark.parametrize("fn", ["abs", "sign", "exp", "expm1", "sqrt",
                                "rsqrt", "tan", "log", "negate", "square"])
def test_complex_unary_specials(fn):
    table = {
        "abs": jnp.abs, "sign": jax.lax.sign, "exp": jnp.exp,
        "expm1": jnp.expm1, "sqrt": jnp.sqrt, "rsqrt": jax.lax.rsqrt,
        "tan": jnp.tan, "log": jnp.log, "negate": jnp.negative,
        "square": jnp.square,
    }
    _diff(lambda z: table[fn](z), _c_specials())


def test_complex_binary_and_compare():
    def f(a, b):
        return (a + b, a * b, a - b, a / b, a == b, a != b)
    _diff(f, _c64(), _c64())


def test_complex_convert_both_ways():
    _diff(lambda z: jnp.real(z).astype(jnp.float32), _c_specials())
    _diff(lambda x: x.astype(jnp.complex64), _specials())
    _diff(lambda z: z.astype(jnp.complex64), _c64())


def test_complex_iota_and_constants():
    def f(z):
        return z + jnp.arange(6, dtype=jnp.complex64) + jnp.complex64(2 - 3j)
    _diff(f, _c64())


def test_complex_shape_ops():
    def f(z):
        w = jnp.concatenate([z, z[::-1]], axis=0).reshape(2, 6)
        return jnp.pad(w, ((1, 1), (0, 2)),
                       constant_values=jnp.complex64(1 + 1j)), w.T
    _diff(f, _c64(6))


def test_complex_select_and_dot():
    def f(z, w, m):
        return jnp.where(m, z, w), z @ w
    z = _c64(9).reshape(3, 3)
    w = _c64(9).reshape(3, 3)
    _diff(f, z, w, np.array([[True, False, True]] * 3))


def test_complex_reduce_and_cumsum():
    _diff(lambda z: (jnp.sum(z), jnp.prod(z)), _c64(5))
    _diff(lambda z: jnp.cumsum(z), _c64(5))


def test_complex_gather():
    idx = np.array([2, 0, 4, 4], np.int32)
    _diff(lambda z, i: z[i], _c64(6), idx)


def test_complex_scatter_declines():
    """MLX has no complex scatter kernels, so the Python handler works on
    the parts — a composition this entry does not carry."""
    def f(z, u):
        return z.at[np.array([0, 2])].set(u)
    _diff(f, _c64(5), _c64(2), lowered=False)


# --------------------------------------------------------------------------
# fft
# --------------------------------------------------------------------------


def test_fft_complex_forward_and_inverse():
    z = _c64(8)
    _diff(lambda x: jnp.fft.fft(x), z)
    _diff(lambda x: jnp.fft.ifft(x), z)
    _diff(lambda x: jnp.fft.fft2(x), _c64(16).reshape(4, 4))
    _diff(lambda x: jnp.fft.ifftn(x), _c64(24).reshape(2, 3, 4))


def test_fft_real_transforms():
    x = _f32((8,))
    _diff(lambda v: jnp.fft.rfft(v), x)
    _diff(lambda v: jnp.fft.irfft(jnp.fft.rfft(v)), x)
    _diff(lambda v: jnp.fft.rfft2(v), _f32((4, 6)))


def test_fft_cropped_and_padded_lengths():
    """`fft_length` shorter or longer than the axis: jax crops or pads in
    its own executable, which is where the input barrier matters."""
    z = _c64(8)
    _diff(lambda x: jnp.fft.fft(x, n=4), z)
    _diff(lambda x: jnp.fft.fft(x, n=12), z)
    _diff(lambda v: jnp.fft.rfft(v, n=5), _f32((8,)))


def test_fft_unit_last_length_rewrite():
    """MLX's rfftn/irfftn silently drop the transforms over the remaining
    axes when the last length is 1; the handler spells that case out."""
    _diff(lambda v: jnp.fft.rfftn(v, s=(4, 1)), _f32((4, 6)))
    _diff(lambda v: jnp.fft.irfftn(v, s=(4, 1)), _c64(24).reshape(4, 6))
    _diff(lambda v: jnp.fft.rfft(v, n=1), _f32((6,)))


def test_fft_empty_input():
    _diff(lambda x: jnp.fft.fft(x), np.zeros((0,), np.complex64))


def test_complex_fma_chain_matches_bitwise():
    """A longer expression: any operand-order or weak-type slip inside the
    complex arms shows up as a byte difference here."""
    def f(a, b, c):
        return jnp.exp(a * b + c) / (jnp.sqrt(a) + 1e-3)
    _diff(f, _c64(8), _c64(8), _c64(8))


# --------------------------------------------------------------------------
# the same families through mx::compile
# --------------------------------------------------------------------------
#
# Everything above runs the tape op by op. A pure program normally traces
# through mx::compile instead, where a region is walked with `in_trace` set
# and nothing may be evaluated — which is a different path through the same
# handlers (the fft barrier and the generic reduce's sub-Program calls both
# ask that flag). Both engines compile here, so the comparison is
# fused-graph against fused-graph.


def _diff_compiled(f, *args):
    mod = lower_bytes(f, *args)
    flat = [np.asarray(x) for x in jax.tree.leaves(args)]
    return check(mod, flat, compiled=True)


def test_compiled_pad_and_reverse():
    _diff_compiled(lambda v: jnp.pad(v, ((1, 2), (0, 1)))[::-1], _f32((3, 4)))


def test_compiled_bit_counters():
    _diff_compiled(lambda v: jax.lax.population_count(v) + jax.lax.clz(v),
                   RNG.integers(0, 1 << 30, (3, 4)).astype(np.uint32))


def test_compiled_rng():
    def f(k):
        s, b = jax.lax.rng_bit_generator(k, (7,), dtype=jnp.uint32)
        return s, b
    _diff_compiled(f, _KEY)


def test_compiled_generic_reduce():
    _diff_compiled(lambda v: jnp.bitwise_xor.reduce(v),
                   RNG.integers(0, 1 << 20, 7).astype(np.int32))


def test_compiled_reduce_window():
    from jax import lax
    _diff_compiled(lambda v: jnp.cumsum(v), _f32((9,)))
    _diff_compiled(lambda v: lax.reduce_window(
        v, jnp.float32(-np.inf), jnp.logaddexp, (1, 2), (1, 1), "SAME",
        (1, 1), (1, 2)), _f32((4, 6)))


def test_compiled_complex_and_fft():
    _diff_compiled(lambda z: jnp.sqrt(z) + jnp.exp(z), _c64(8))
    _diff_compiled(lambda z: jnp.fft.fft(z), _c64(8))


def test_totalorder_compare_on_integers_is_ordinary():
    """TOTALORDER on an int operand is a plain compare (the Python handler
    asks `is_float` first), which is what the tape must lower too."""
    t = "tensor<6xi32>"
    mod = _mod([("a", t), ("b", t)], ["tensor<6xi1>"], f"""
    %0 = stablehlo.compare LT, %a, %b, TOTALORDER : ({t}, {t})
         -> tensor<6xi1>
    return %0 : tensor<6xi1>""")
    a = RNG.integers(-9, 9, 6).astype(np.int32)
    b = RNG.integers(-9, 9, 6).astype(np.int32)
    check(mod, [a, b])
