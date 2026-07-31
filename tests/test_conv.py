"""stablehlo.convolution -> mx.conv_general, vs the CPU backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import check

rng = np.random.default_rng(3)

X = rng.standard_normal((2, 3, 12, 12)).astype(np.float32)  # NCHW
K = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)    # OIHW


def test_conv2d_basic():
    check(lambda x, k: jax.lax.conv(x, k, (1, 1), "SAME"), X, K,
          rtol=1e-5, atol=1e-5)
    check(lambda x, k: jax.lax.conv(x, k, (2, 2), "VALID"), X, K,
          rtol=1e-5, atol=1e-5)


def test_conv1d_and_correlate():
    a = rng.standard_normal(20).astype(np.float32)
    b = rng.standard_normal(5).astype(np.float32)
    check(lambda a, b: jnp.convolve(a, b), a, b, rtol=1e-5, atol=1e-5)
    check(lambda a, b: jnp.correlate(a, b, mode="full"), a, b,
          rtol=1e-5, atol=1e-5)


def test_conv_groups_dilation_transpose():
    xg = rng.standard_normal((2, 8, 10, 10)).astype(np.float32)
    kg = rng.standard_normal((8, 2, 3, 3)).astype(np.float32)
    check(lambda x, k: jax.lax.conv_general_dilated(
        x, k, (1, 1), "SAME", feature_group_count=4), xg, kg,
        rtol=1e-5, atol=1e-5)
    check(lambda x, k: jax.lax.conv_general_dilated(
        x, k, (1, 1), "SAME", rhs_dilation=(2, 2)), X, K,
        rtol=1e-5, atol=1e-5)
    check(lambda x, k: jax.lax.conv_transpose(
        x, k, (2, 2), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")),
        X, K, rtol=1e-5, atol=1e-5)


def test_conv_grads():
    # grad wrt kernel exercises batch_group_count; wrt input exercises
    # window_reversal + lhs_dilation.
    def loss_k(k):
        return jnp.sum(jax.lax.conv(X, k, (1, 1), "SAME") ** 2)

    def loss_x(x):
        return jnp.sum(jax.lax.conv(x, K, (1, 1), "SAME") ** 2)

    check(jax.grad(loss_k), K, rtol=1e-4, atol=1e-4)
    check(jax.grad(loss_x), X, rtol=1e-4, atol=1e-4)


def test_conv_int_complex():
    rng2 = np.random.default_rng(7)
    xi = rng2.integers(-50, 50, (2, 3, 8, 8)).astype(np.int32)
    ki = rng2.integers(-5, 5, (4, 3, 3, 3)).astype(np.int32)
    check(lambda x, k: jax.lax.conv(x, k, (1, 1), "SAME"), xi, ki)
    check(lambda a, b: jnp.convolve(a, b),
          rng2.integers(-9, 9, 20).astype(np.int32),
          rng2.integers(-9, 9, 5).astype(np.int32))
    z = (rng2.standard_normal(12)
         + 1j * rng2.standard_normal(12)).astype(np.complex64)
    zk = (rng2.standard_normal(4)
          + 1j * rng2.standard_normal(4)).astype(np.complex64)
    check(lambda a, b: jnp.convolve(a, b), z, zk, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Regressions from jax's lax_test.py ConvGeneralDilated{,PatchesNonOverlapping}
# (configs copied verbatim from the failing cases).

def test_conv_int_feature_groups():
    """lax_test testConvGeneralDilated3: int32, NCHW/OIHW/NCHW, fgc=2."""
    r = np.random.default_rng(11)
    x = r.integers(-5, 5, (2, 6, 9, 10)).astype(np.int32)
    k = r.integers(-5, 5, (6, 3, 4, 5)).astype(np.int32)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (2, 1), ((10, 8), (7, 13)), (1, 2), (1, 2),
        ("NCHW", "OIHW", "NCHW"), feature_group_count=2), x, k)


def test_conv_int_batch_groups():
    """lax_test testConvGeneralDilated5: uint16, NCHW/HWIO/NHWC, bgc=2."""
    r = np.random.default_rng(12)
    x = r.integers(0, 40, (6, 3, 9, 10)).astype(np.uint16)
    k = r.integers(0, 6, (4, 5, 3, 4)).astype(np.uint16)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (2, 1), ((1, 2), (2, 0)), (1, 2), (1, 2),
        ("NCHW", "HWIO", "NHWC"), batch_group_count=2), x, k)


def test_conv_int_feature_groups_permuted_dims():
    """lax_test PatchesNonOverlapping0/3: 1-D and 2-D uint16, fgc=2, with
    fully permuted dimension numbers."""
    r = np.random.default_rng(13)
    dn1 = jax.lax.ConvDimensionNumbers(lhs_spec=(1, 0, 2), rhs_spec=(0, 2, 1),
                                       out_spec=(1, 2, 0))
    x1 = r.integers(0, 40, (2, 3, 4)).astype(np.uint16)
    k1 = r.integers(0, 6, (4, 2, 1)).astype(np.uint16)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (2,), [(0, 2)], dimension_numbers=dn1,
        feature_group_count=2), x1, k1)

    dn2 = jax.lax.ConvDimensionNumbers(lhs_spec=(0, 1, 3, 2),
                                       rhs_spec=(0, 3, 1, 2),
                                       out_spec=(1, 0, 2, 3))
    x2 = r.integers(-40, 40, (1, 2, 3, 4)).astype(np.int32)
    k2 = r.integers(-6, 6, (2, 1, 1, 1)).astype(np.int32)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1), [(0, 0), (0, 0)], dimension_numbers=dn2,
        feature_group_count=2), x2, k2)


def test_conv_bool_feature_groups():
    r = np.random.default_rng(14)
    x = r.integers(0, 2, (2, 4, 6, 6)).astype(bool)
    k = r.integers(0, 2, (4, 2, 3, 3)).astype(bool)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=2), x, k)


def test_conv3d_feature_groups():
    """MLX implements native conv groups only in 1-D/2-D; 3-D must split.
    Second case is lax_test PatchesNonOverlapping8's permuted layout."""
    r = np.random.default_rng(15)
    x = r.standard_normal((2, 6, 5, 6, 7)).astype(np.float32)   # NCDHW
    k = r.standard_normal((12, 2, 2, 3, 3)).astype(np.float32)  # OIDHW
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (2, 1, 2), ((1, 1), (2, 0), (0, 2)),
        dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
        feature_group_count=3), x, k, rtol=1e-5, atol=1e-5)

    dn = jax.lax.ConvDimensionNumbers(lhs_spec=(0, 4, 1, 3, 2),
                                      rhs_spec=(4, 2, 0, 1, 3),
                                      out_spec=(3, 1, 4, 0, 2))
    xp = r.standard_normal((2, 3, 4, 5, 6)).astype(np.float32)
    kp = r.standard_normal((2, 1, 1, 3, 36)).astype(np.float32)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (2, 1, 3), [(1, 2), (5, 3), (3, 5)], dimension_numbers=dn,
        feature_group_count=6), xp, kp, rtol=1e-5, atol=1e-5)


def test_conv3d_int_feature_groups():
    r = np.random.default_rng(16)
    x = r.integers(-5, 5, (2, 4, 4, 5, 5)).astype(np.int32)
    k = r.integers(-4, 4, (6, 2, 2, 2, 2)).astype(np.int32)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1, 1), "SAME",
        dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
        feature_group_count=2), x, k)


def test_conv_empty_operand():
    """lax_test testConvGeneralDilated2: a zero-size *spatial* input dim with
    lhs_dilation > 1. MLX's dilated extent for size 0 is (0-1)*d+1, not XLA's
    0, so it returns a narrower array than the declared output shape."""
    r = np.random.default_rng(17)
    x = np.zeros((2, 9, 0, 4), dtype=jnp.bfloat16)
    k = r.standard_normal((4, 5, 2, 4)).astype(jnp.bfloat16)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1), ((10, 8), (7, 13)), (1, 2), (1, 2),
        ("NHWC", "HWIO", "NHWC"), feature_group_count=2), x, k,
        rtol=1e-2, atol=1e-2)

    xi = np.zeros((2, 3, 9, 0), dtype=np.int32)          # NCHW
    ki = r.integers(-4, 4, (4, 3, 2, 2)).astype(np.int32)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1), ((2, 2), (5, 5)), (1, 3), (1, 1),
        ("NCHW", "OIHW", "NCHW")), xi, ki)

    # zero-size input *feature* dim: the output keeps rhs's O features and
    # every element is an empty sum. (A zero-size kernel *window* is not
    # representable -- stablehlo.convolution requires positive window dims.)
    xf = np.zeros((2, 0, 6, 6), dtype=np.float32)
    kf = np.zeros((4, 0, 3, 3), dtype=np.float32)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1), ((1, 1), (1, 1)),
        dimension_numbers=("NCHW", "OIHW", "NCHW")), xf, kf)


def test_conv_negative_padding():
    """XLA pads after lhs dilation, so negative padding crops the *dilated*
    array; MLX crops the operand instead. lax_vmap_test
    testConvGeneralDilatedBatching0/7 hit this through vmap."""
    r = np.random.default_rng(18)
    x = r.standard_normal((6, 7, 4, 1)).astype(np.float32)   # [0, 1, f, b]
    k = r.standard_normal((1, 2, 2, 2)).astype(np.float32)   # [0, 1, i, o]
    dn = jax.lax.ConvDimensionNumbers(lhs_spec=(3, 2, 0, 1),
                                      rhs_spec=(3, 2, 0, 1),
                                      out_spec=(3, 2, 0, 1))
    # lhs_dilation 2 with a -1 crop: MLX would return one *dilation step*
    # short (9 instead of 10 rows).
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1), ((0, -1), (0, 0)), (2, 1), (1, 1),
        dimension_numbers=dn, feature_group_count=2), x, k,
        rtol=1e-5, atol=1e-5)

    x2 = r.standard_normal((2, 2, 6, 7)).astype(np.float32)
    k2 = r.standard_normal((4, 1, 1, 2)).astype(np.float32)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 2), ((0, -1), (0, 0)), (2, 1), (2, 2),
        ("NCHW", "OIHW", "NCHW"), feature_group_count=2), x2, k2,
        rtol=1e-5, atol=1e-5)

    # crops on both sides / both spatial axes, undilated and dilated,
    # and one that is not a whole multiple of the dilation factor
    x3 = r.standard_normal((2, 3, 9, 9)).astype(np.float32)
    k3 = r.standard_normal((4, 3, 3, 3)).astype(np.float32)
    for ldil in ((1, 1), (2, 3), (3, 2)):
        check(lambda a, b: jax.lax.conv_general_dilated(
            a, b, (1, 1), ((-2, -3), (-4, -1)), ldil, (1, 1),
            ("NCHW", "OIHW", "NCHW")), x3, k3, rtol=1e-5, atol=1e-5)

    xi = r.integers(-5, 5, (2, 3, 9, 9)).astype(np.int32)
    ki = r.integers(-4, 4, (4, 3, 3, 3)).astype(np.int32)
    check(lambda a, b: jax.lax.conv_general_dilated(
        a, b, (1, 1), ((-2, -3), (-4, -1)), (2, 3), (1, 1),
        ("NCHW", "OIHW", "NCHW")), xi, ki)
