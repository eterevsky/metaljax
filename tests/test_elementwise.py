import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import check

F = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5], np.float32)
P = np.array([0.1, 0.7, 1.3, 2.9, 4.2], np.float32)  # positive
I = np.array([-7, -3, -1, 0, 1, 3, 7], np.int32)
B = np.array([True, False, True, True, False])


@pytest.mark.parametrize("fn", [
    jnp.exp, jnp.expm1, jnp.tanh, jnp.sin, jnp.cos, jnp.abs,
    jnp.floor, jnp.ceil, jnp.sign, jax.nn.sigmoid, jnp.negative, jnp.round,
])
def test_unary_float(fn):
    check(fn, F)


@pytest.mark.parametrize("fn", [jnp.log, jnp.log1p, jnp.sqrt, jax.lax.rsqrt, jnp.cbrt])
def test_unary_positive(fn):
    check(fn, P)


def test_isfinite():
    check(jnp.isfinite, np.array([1.0, np.inf, -np.inf, np.nan], np.float32))


def test_erf():
    check(jax.scipy.special.erf, F)


@pytest.mark.parametrize("fn", [
    jnp.add, jnp.subtract, jnp.multiply, jnp.maximum, jnp.minimum, jnp.arctan2,
])
def test_binary_float(fn):
    check(fn, F, np.roll(F, 1))


def test_divide_float():
    check(jnp.divide, F, np.roll(F, 1) + 3.1)


def test_power():
    check(jnp.power, P, np.array([2.0, 0.5, -1.0, 3.0, 0.0], np.float32))


def test_remainder_float():
    check(jax.lax.rem, F, np.array([2.0, 2.0, -2.0, 3.0, -3.0, 2.0, 2.0], np.float32))


def test_int_arith():
    check(lambda a, b: a + b * a - b, I, np.roll(I, 2))


def test_int_div_rem_trunc():
    d = np.array([3, -3, 2, 5, -2, 4, -4], np.int32)
    check(jax.lax.div, I, d)
    check(jax.lax.rem, I, d)


def test_int_bitwise():
    check(lambda a, b: (a & b) ^ (a | b), I, np.roll(I, 1))
    check(jnp.invert, I)


def test_shifts():
    a = np.array([1, 2, -4, 8, -16], np.int32)
    s = np.array([0, 1, 2, 3, 4], np.int32)
    check(jax.lax.shift_left, a, s)
    check(jax.lax.shift_right_arithmetic, a, s)
    check(jax.lax.shift_right_logical, a, s)
    u = a.astype(np.uint32)
    check(jax.lax.shift_right_logical, u, s.astype(np.uint32))


def test_bool_logic():
    check(lambda a, b: jnp.logical_and(a, b) | jnp.logical_xor(a, ~b), B, np.roll(B, 1))


@pytest.mark.parametrize("fn", [
    jnp.less, jnp.less_equal, jnp.greater, jnp.greater_equal, jnp.equal, jnp.not_equal,
])
def test_compare(fn):
    check(fn, F, np.roll(F, 1))
    check(fn, I, np.roll(I, 1))


def test_where():
    check(jnp.where, B, F[:5], np.roll(F[:5], 1))


def test_clip():
    check(lambda x: jnp.clip(x, -1.0, 1.0), F)


def test_scalar_promotion():
    check(lambda x: 2 * x + 1.5, F)


@pytest.mark.parametrize("dt", [np.float16, jnp.bfloat16])
def test_half_precision(dt):
    x = np.linspace(-2, 2, 8).astype(np.float32)
    check(lambda a: jnp.tanh(a) * a + 1, x.astype(dt), rtol=2e-2, atol=2e-2)


def test_bf16_hex_splat_constant():
    # xla-translate emits bf16 specials as hex splats (dense<0xFF80> = -inf);
    # the MLIR bindings mis-decode those as float(hex) if allowed to.
    import mlx.core as mx
    from metaljax.interpreter import Interpreter

    mod = """
module {
  func.func @main() -> (tensor<2xbf16>, tensor<bf16>) {
    %c = stablehlo.constant dense<0xFF80> : tensor<2xbf16>
    %s = stablehlo.constant dense<0x3F80> : tensor<bf16>
    return %c, %s : tensor<2xbf16>, tensor<bf16>
  }
}
"""
    a, b = Interpreter(mod)()
    assert np.all(np.isneginf(np.array(a.astype(mx.float32))))
    assert float(np.array(b.astype(mx.float32))) == 1.0


def test_emulated_dtypes_i4_f8():
    # int4/uint4 in i8/u8 storage with 4-bit wraparound; float8 as exact
    # values in f16 with grid quantization on convert.
    check(lambda x: x.astype(jnp.float8_e4m3fn).astype(jnp.float32),
          np.array([1.7, 300.0, 1e-6, -2.5], np.float32))
    check(lambda x: x.astype(jnp.float8_e5m2).astype(jnp.float32),
          np.array([1e6, np.nan, 0.1], np.float32))
    check(lambda a, b: (a.astype(jnp.int4) + b.astype(jnp.int4))
          .astype(jnp.int32),
          np.array([7, -8], np.int32), np.array([1, -1], np.int32))
    check(lambda x: x.astype(jnp.uint4).astype(jnp.int32),
          np.array([3, 17], np.int32))


def test_expm1_accuracy():
    # MLX's Metal expm1 kernel is fast-math (worst rel 2e-5); the generic
    # comparison grid is too coarse to catch it.
    x = np.concatenate([np.linspace(-10, 10, 501),
                        np.logspace(-8, 1, 300),
                        -np.logspace(-8, 1, 300)]).astype(np.float32)
    check(lambda v: jnp.expm1(v), x, rtol=5e-7, atol=1e-8)


def _jit_on(platform, f, *args):
    """Run f through the REAL backend on `platform` (the bare Interpreter
    used by `check` never calls mx.compile, where this bug lives)."""
    with jax.default_device(jax.devices(platform)[0]):
        return np.asarray(jax.jit(f)(*[jnp.asarray(a) for a in args]))


def test_rank0_constant_is_not_a_lossy_literal():
    # mx.compile inlines RANK-0 constants into generated Metal source as
    # %.7g decimal literals -- one digit short of float32's 9-digit round
    # trip, so 2/3 of constants come back 1 ULP off. A single f32 multiply
    # is exactly rounded on any IEEE backend, so metal must match CPU bit
    # for bit here.
    consts = np.array([np.pi, np.pi / 2, 1 / 3, 12345.6789, 0.7, 8.5e-9,
                       1e-7, 2.0, 0.1, 0.5], np.float32)
    # rank-0 operands in a CHAIN: a constant that first feeds
    # broadcast_in_dim rides in memory anyway, and a lone binary op uses
    # MLX's prebuilt kernel (scalar as a kernel argument) -- only a fused
    # multi-op kernel bakes the literal.
    xs = np.array([0.995, -3.25, 7.125, 1.0000001], np.float32)

    def f(v):
        return jnp.stack([jnp.float32(c) * v * v for c in consts])

    for x in xs:
        got, want = _jit_on("metal", f, x), _jit_on("cpu", f, x)
        bad = np.argwhere(got != want).ravel()
        assert not len(bad), (
            f"x={x}: {len(bad)}/{got.size} products differ: "
            f"{[(float(consts[i]), float(got[i]), float(want[i])) for i in bad[:4]]}")


def test_ill_conditioned_constant_expression_matches_cpu():
    # One ULP on a constant is invisible until something ill-conditioned
    # amplifies it: jax's scipy.stats.cauchy.isf is tan(pi/2 - pi*q), whose
    # condition number is ~64 at the ends of the test's clipped q range
    # (scipy_stats_test's testCauchyIsf, which compares the jitted result
    # against the op-by-op one, saw 3.8e-5 there against a 3e-4 tolerance,
    # and up to 4.7e-3 closer to the pole).
    def f(v):
        return jnp.tan(jnp.float32(np.pi / 2) - jnp.float32(np.pi) * v)

    for q in (0.995, 0.005, 0.99, 0.01, 0.75):
        got = _jit_on("metal", f, np.float32(q))
        want = _jit_on("cpu", f, np.float32(q))
        rel = abs(float(got) - float(want)) / abs(float(want))
        assert rel < 1e-6, f"q={q}: {got} vs {want} (rel {rel:.3e})"
