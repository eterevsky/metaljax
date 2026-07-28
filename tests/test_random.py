import jax
import jax.numpy as jnp
import pytest
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


def test_rng_bit_generator_bit_exact():
    # Philox4x32 with XLA's state mapping — bit-exact vs the CPU backend.
    try:
        metal = jax.devices("metal")[0]
        cpu = jax.devices("cpu")[0]
    except RuntimeError:
        pytest.skip("metal plugin not available")
    key = jnp.array([1, 2, 3, 4], dtype=jnp.uint32)
    for shape, dt in (((8,), jnp.uint32), ((7,), jnp.uint32),
                      ((3, 5), jnp.uint32), ((6,), jnp.uint16)):
        with jax.default_device(metal):
            sm, bm = jax.lax.rng_bit_generator(key, shape, dtype=dt)
        with jax.default_device(cpu):
            sc, bc = jax.lax.rng_bit_generator(key, shape, dtype=dt)
        np.testing.assert_array_equal(np.asarray(sm), np.asarray(sc))
        np.testing.assert_array_equal(np.asarray(bm), np.asarray(bc))
    k = jax.random.key(42, impl="rbg")
    with jax.default_device(metal):
        um = np.asarray(jax.random.uniform(k, (16,)))
    with jax.default_device(cpu):
        uc = np.asarray(jax.random.uniform(k, (16,)))
    np.testing.assert_array_equal(um, uc)
