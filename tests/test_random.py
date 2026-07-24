import jax
import numpy as np

from helpers import check


def test_key_split():
    check(lambda k: jax.random.split(k, 4), jax.random.PRNGKey(42))


def test_uniform():
    check(lambda k: jax.random.uniform(k, (16,)), jax.random.PRNGKey(0))


def test_normal():
    check(lambda k: jax.random.normal(k, (4, 8)), jax.random.PRNGKey(7))


def test_randint():
    check(lambda k: jax.random.randint(k, (32,), 0, 100), jax.random.PRNGKey(3))


def test_choice_weighted():
    p = np.ones(10, np.float32) / 10
    check(lambda k, p: jax.random.choice(k, 10, p=p), jax.random.PRNGKey(1), p)
