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
