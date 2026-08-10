"""Sub-byte float emulation: float4_e2m1fn, float6_e2m3fn, float6_e3m2fn.

Like the float8 family these have no MLX storage type: metaljax keeps
their exact values in float16 (every grid point is representable there)
and quantizes onto the format's grid at stablehlo.convert and after
elementwise arithmetic.

What sets the OCP microscaling FP4/FP6 formats apart from the float8s is
that *every* bit pattern is a finite number -- no infinities, no NaN --
so a converting overflow saturates to +-max instead of producing NaN.

How far the CPU backend can be used as the reference differs per format:

  float4_e2m1fn  fully supported by XLA:CPU -- converts both ways and
                 arithmetic, so plain check() works.
  float6_*       XLA:CPU can only convert f32 -> f6. Converting back out
                 fails to compile ("Failed to translate module to LLVM
                 IR"), f6 arithmetic fails to legalize (arith.addf), and
                 an f6 *constant* aborts the process inside
                 CreateLiteralFromAttribute. So the fp6 references here
                 are taken from ml_dtypes -- which agrees bit for bit
                 with XLA's fp4 implementation and with the OCP spec --
                 plus a direct CPU comparison for the one thing CPU can
                 do. CPU's fp6 convert also returns 0 instead of
                 saturating on overflow, which fp4 and the spec both
                 contradict; test_overflow_saturates_to_max pins the
                 spec behaviour.
"""

import ml_dtypes
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from helpers import check, run_metal, run_metal_device

F4 = jnp.float4_e2m1fn
F6E2M3 = jnp.float6_e2m3fn
F6E3M2 = jnp.float6_e3m2fn

SUBBYTE = [F4, F6E2M3, F6E3M2]
FP6 = [F6E2M3, F6E3M2]

# Every value each format can hold, as its own dtype.
GRID = {
    F4: np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn),
    F6E2M3: np.arange(64, dtype=np.uint8).view(ml_dtypes.float6_e2m3fn),
    F6E3M2: np.arange(64, dtype=np.uint8).view(ml_dtypes.float6_e3m2fn),
}


def metal_pack(f, x, dtype):
    """Run f on metal and read the result the way a caller does: the engine
    computes in the f16 storage and `engine.to_host` casts it to the narrow
    dtype, which is what `run_metal` returns. The `astype` is therefore a
    no-op here and is kept only so the requested dtype is asserted."""
    (got,) = run_metal(f, x)
    assert got.dtype == np.dtype(dtype), (got.dtype, dtype)
    return got.astype(dtype)


def assert_same_bits(got, want, dtype):
    bits = ml_dtypes.finfo(dtype).bits
    mask = np.uint8((1 << bits) - 1)
    g, w = got.view(np.uint8) & mask, want.view(np.uint8) & mask
    bad = np.nonzero(g != w)[0]
    assert not len(bad), (
        f"{len(bad)} of {len(g)} differ, first at {bad[:5]}: "
        f"{got.astype(np.float32)[bad[:5]]} != {want.astype(np.float32)[bad[:5]]}"
    )


# ---------------------------------------------------------------- fp4

def test_float4_roundtrip_grid_values():
    check(lambda v: v.astype(F4).astype(jnp.float32),
          GRID[F4].astype(np.float32))


def test_float4_convert_matches_cpu():
    hi = float(ml_dtypes.finfo(F4).max)
    x = np.concatenate([
        np.linspace(-4 * hi, 4 * hi, 601),
        np.linspace(-1.0, 1.0, 101),
        [0.0, -0.0, np.inf, -np.inf, 1e30, -1e30],
    ]).astype(np.float32)
    check(lambda v: v.astype(F4).astype(jnp.float32), x)


def test_float4_ties_round_to_even():
    g = np.unique(GRID[F4].astype(np.float32))
    mids = ((g[:-1] + g[1:]) / 2).astype(np.float32)
    check(lambda v: v.astype(F4).astype(jnp.float32), mids)


def test_float4_convert_from_float16_and_int():
    check(lambda v: v.astype(F4).astype(jnp.float32),
          np.array([1.0, 1.5, -2.0, 0.5, -0.0], np.float16))
    check(lambda v: v.astype(F4).astype(jnp.int32),
          np.array([0, 1, 2, -1, -2], np.int32))
    check(lambda v: v.astype(F4).astype(jnp.float16),
          np.array([0.0, 1.0, -1.0, 2.0, 6.0], np.float32))


def test_float4_arithmetic_stays_on_grid():
    # XLA rounds every result back onto the 16-point grid: 4 + 4 is 6.
    g = GRID[F4].astype(np.float32)
    check(lambda v: (v.astype(F4) + v.astype(F4)).astype(jnp.float32), g)
    check(lambda v: (v.astype(F4) * v.astype(F4)).astype(jnp.float32), g)
    check(lambda v: (v.astype(F4) - v.astype(F4) * v.astype(F4))
          .astype(jnp.float32), g)
    check(lambda v: jnp.negative(v.astype(F4)).astype(jnp.float32), g)


def test_float4_constants():
    check(lambda v: (jnp.zeros(4, F4) + v.astype(F4)).astype(jnp.float32),
          np.array([0.0, 1.5, -3.0, 6.0], np.float32))
    check(lambda v: (jnp.full(4, 1.5, F4) * v.astype(F4)).astype(jnp.float32),
          np.array([0.0, 1.0, -2.0, 4.0], np.float32))


# ---------------------------------------------------------------- fp6

@pytest.mark.parametrize("dtype", FP6)
def test_fp6_convert_matches_cpu_in_range(dtype):
    # The one fp6 operation XLA:CPU implements: f32 -> f6. Out-of-range
    # inputs are excluded (CPU returns 0 there; see the module docstring).
    fi = ml_dtypes.finfo(dtype)
    hi, lo = float(fi.max), float(fi.smallest_subnormal)
    x = np.concatenate([
        np.linspace(-hi, hi, 501),
        np.linspace(-4 * lo, 4 * lo, 33),
        [0.0, -0.0, lo, -lo, hi, -hi],
    ]).astype(np.float32)
    got = metal_pack(lambda v: v.astype(dtype), x, dtype)
    with jax.default_device(jax.devices("cpu")[0]):
        want = np.asarray(jax.jit(lambda v: v.astype(dtype))(x))
    assert_same_bits(got, want, dtype)


@pytest.mark.parametrize("dtype", FP6)
def test_fp6_roundtrip_grid_values(dtype):
    # f6 -> f32 has no CPU implementation; ml_dtypes is the reference.
    x = GRID[dtype].astype(np.float32)
    (got,) = run_metal(lambda v: v.astype(dtype).astype(jnp.float32), x)
    np.testing.assert_array_equal(got, x)


@pytest.mark.parametrize("dtype", FP6)
def test_fp6_ties_round_to_even(dtype):
    g = np.unique(GRID[dtype].astype(np.float32))
    mids = ((g[:-1] + g[1:]) / 2).astype(np.float32)
    got = metal_pack(lambda v: v.astype(dtype), mids, dtype)
    assert_same_bits(got, mids.astype(dtype), dtype)


@pytest.mark.parametrize("dtype", FP6)
def test_fp6_convert_from_float16_and_int(dtype):
    got = metal_pack(lambda v: v.astype(dtype),
                     np.array([1.0, 1.5, -2.0, 0.5, -0.0], np.float16), dtype)
    assert_same_bits(got, np.array([1.0, 1.5, -2.0, 0.5, -0.0],
                                   np.float16).astype(dtype), dtype)
    (ints,) = run_metal(lambda v: v.astype(dtype).astype(jnp.int32),
                        np.array([0, 1, 2, -1, -2], np.int32))
    np.testing.assert_array_equal(ints, np.array([0, 1, 2, -1, -2], np.int32))


@pytest.mark.parametrize("dtype", FP6)
def test_fp6_arithmetic_stays_on_grid(dtype):
    # No CPU reference exists (arith.addf on f6 fails to legalize), so
    # this pins the grid property itself: every result is representable.
    g = GRID[dtype].astype(np.float32)
    for f in (lambda v: v.astype(dtype) + v.astype(dtype),
              lambda v: v.astype(dtype) * v.astype(dtype),
              lambda v: jnp.negative(v.astype(dtype))):
        packed = metal_pack(f, g, dtype)
        # The DEVICE storage, before to_host narrows it: the claim is that
        # narrowing loses nothing, so the two have to be read differently or
        # the comparison is an array against itself.
        (storage,) = run_metal_device(f, g)
        np.testing.assert_array_equal(
            packed.astype(np.float32), storage.astype(np.float32),
            err_msg="arithmetic result left the value grid")


# ------------------------------------------------------- all three

@pytest.mark.parametrize("dtype", SUBBYTE)
def test_overflow_saturates_to_max(dtype):
    # No inf and no NaN encoding: XLA saturates.
    fi = ml_dtypes.finfo(dtype)
    big = float(fi.max) * 4
    x = np.array([big, -big, np.inf, -np.inf, 1e30, -1e30], np.float32)
    got = metal_pack(lambda v: v.astype(dtype), x, dtype)
    assert_same_bits(got, x.astype(dtype), dtype)
    assert np.all(np.abs(got.astype(np.float32)) == float(fi.max))


@pytest.mark.parametrize("dtype", SUBBYTE)
def test_nan_has_no_encoding(dtype):
    # NaN cannot be represented; the host cast lands on a zero.
    got = metal_pack(lambda v: v.astype(dtype),
                     np.array([np.nan, 1.0], np.float32), dtype)
    assert_same_bits(got, np.array([np.nan, 1.0], np.float32).astype(dtype),
                     dtype)
    assert float(got.astype(np.float32)[0]) == 0.0


@pytest.fixture
def metal_default():
    with jax.default_device(jax.devices("metal")[0]):
        yield


@pytest.mark.parametrize("dtype", SUBBYTE)
def test_promotion_lattice(dtype, metal_default):
    # jax refuses implicit promotion for the sub-byte floats; explicit
    # casts are the only way out, and the dtype survives a device
    # round trip unchanged. Pinned to metal: XLA:CPU cannot even
    # materialize an fp6 constant (RET_CHECK in fusion_compiler.cc), so
    # on the CPU default device this dies before the promotion check.
    x = jnp.array(1, dtype=dtype)
    assert x.dtype == np.dtype(dtype)
    assert jnp.result_type(dtype) == np.dtype(dtype)
    with pytest.raises(jax.dtypes.TypePromotionError):
        x + jnp.float32(1)
    with pytest.raises(jax.dtypes.TypePromotionError):
        jnp.promote_types(dtype, np.float32)
    assert x.astype(jnp.float32).dtype == np.dtype(np.float32)
    assert float(x.astype(jnp.float32)) == 1.0


@pytest.mark.parametrize("dtype", SUBBYTE)
def test_pjrt_roundtrip(dtype):
    # Through the REAL backend: the PJRT buffer types (F4E2M1FN /
    # F6E2M3FN / F6E3M2FN) must carry these host arrays both ways.
    metal = jax.devices("metal")[0]
    host = GRID[dtype]
    with jax.default_device(metal):
        dev = jnp.asarray(host)
        assert dev.dtype == np.dtype(dtype)
        assert dev.devices() == {metal}
        assert_same_bits(np.asarray(dev), host, dtype)
        out = jax.jit(lambda v: v.astype(jnp.float32) + 1.0)(dev)
        np.testing.assert_array_equal(
            np.asarray(out), host.astype(np.float32) + 1.0)
