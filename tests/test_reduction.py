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
