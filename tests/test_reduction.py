import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import check

rng = np.random.default_rng(1)
X = rng.standard_normal((3, 4, 5)).astype(np.float32)


@pytest.mark.parametrize("fn", [jnp.sum, jnp.max, jnp.min, jnp.prod, jnp.mean])
@pytest.mark.parametrize("axis", [None, 0, 2, (0, 1), (1, 2)])
def test_reduce(fn, axis):
    check(lambda x: fn(x, axis=axis), X, rtol=1e-4, atol=1e-5)


def test_keepdims():
    check(lambda x: jnp.sum(x, axis=1, keepdims=True), X, rtol=1e-4, atol=1e-5)


def test_bool_reduce():
    b = X > 0
    check(lambda x: jnp.all(x, axis=1), b)
    check(lambda x: jnp.any(x, axis=(0, 2)), b)


def test_softmax():
    check(lambda x: jax.nn.softmax(x, axis=-1), X, rtol=1e-5, atol=1e-6)


def test_logsumexp():
    check(lambda x: jax.scipy.special.logsumexp(x, axis=1), X, rtol=1e-5, atol=1e-6)


def test_norm():
    check(jnp.linalg.norm, X, rtol=1e-4, atol=1e-5)


def test_var_std():
    check(lambda x: jnp.var(x, axis=0), X, rtol=1e-4, atol=1e-5)


def test_argmax_argmin():
    check(lambda x: jnp.argmax(x, axis=1), X)
    check(lambda x: jnp.argmin(x, axis=0), X)
    check(lambda x: jnp.argmax(x, axis=-1), X)
    # ties resolve to the lowest index
    T = np.zeros((4, 6), np.float32)
    T[:, 2] = T[:, 4] = 7.0
    check(lambda x: jnp.argmax(x, axis=1), T)
    check(lambda x: jnp.argmin(x, axis=1), T)
    XI = (X * 100).astype(np.int32)
    check(lambda x: jnp.argmax(x, axis=2), XI)
    # max+argmax pair as sampling code uses it
    check(lambda x: (jnp.max(x, axis=1), jnp.argmax(x, axis=1)), X)


def test_reduce_precision():
    check(lambda x: jax.lax.reduce_precision(x, 8, 7), X)     # bf16 grid
    check(lambda x: jax.lax.reduce_precision(x, 5, 10), X)    # f16 grid
    check(lambda x: jax.lax.reduce_precision(x, 8, 12), X)    # generic f32


def test_empty_axis_reduce_returns_init():
    # MLX reducers crash on zero-size inputs; an empty fold = init value.
    check(lambda x: jnp.max(x, axis=1, initial=-1.0),
          np.zeros((3, 0), np.float32))
    check(lambda x: jnp.min(x, axis=1, initial=5.0),
          np.zeros((2, 0), np.float32))
    check(lambda x: jnp.sum(x, axis=0), np.zeros((3, 0), np.uint32))
    check(lambda x: jax.nn.softmax(x, axis=-1), np.zeros((2, 0), np.float32))


def test_argmax_argmin_nan_wins():
    # XLA/numpy: NaN wins argmax/argmin; first NaN's index is returned.
    x = np.array([0.0, np.nan], np.float32)
    check(lambda x: jnp.argmax(x), x)
    check(lambda x: jnp.argmin(x), np.array([3.0, np.nan, 1.0], np.float32))
    check(lambda x: (jnp.argmax(x, axis=1), jnp.argmin(x, axis=1)),
          np.array([[1.0, np.nan, 0.5], [2.0, 1.0, 3.0]], np.float32))
