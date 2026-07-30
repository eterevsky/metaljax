"""Buffer donation: the caller's promise not to reuse donated arguments.

jax marks donated args in the lowered module (tf.aliasing_output /
jax.buffer_donor); the plugin invalidates those buffers after execute.
These run through the REAL backend (not the bare Interpreter) because
donation is a PJRT-boundary contract.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp


def _metal():
    return jax.devices("metal")[0]


def test_donated_input_is_deleted():
    with jax.default_device(_metal()):
        move = jax.jit(lambda x: x + x - x, donate_argnums=(0,))
        x = jnp.ones([])
        y = move(x)
        assert float(y) == 1.0
        assert x.is_deleted()
        with pytest.raises(RuntimeError):
            np.asarray(x)


def test_non_donated_inputs_survive():
    with jax.default_device(_metal()):
        step = jax.jit(lambda w, g: w - 0.1 * g, donate_argnums=(0,))
        w = jnp.arange(8.0)
        g = jnp.ones(8)
        step(w, g)
        assert w.is_deleted()
        assert not g.is_deleted()
        np.testing.assert_allclose(np.asarray(g), np.ones(8))


def test_chained_donation_matches_plain():
    # w = step(w, ...) training-loop shape: the donated input of step k+1
    # is the (possibly still-pending) output of step k. Must be
    # bit-identical to the non-donating loop.
    def loop(donate):
        with jax.default_device(_metal()):
            kw = dict(donate_argnums=(0,)) if donate else {}
            step = jax.jit(lambda w, g: w * 0.999 - 0.1 * g, **kw)
            w = jnp.arange(8.0)
            g = jnp.linspace(0, 1, 8)
            for _ in range(50):
                w = step(w, g)
            return np.asarray(w)

    np.testing.assert_array_equal(loop(True), loop(False))


def test_no_unusable_donation_warning():
    import warnings
    with jax.default_device(_metal()):
        step = jax.jit(lambda w: w * 2.0, donate_argnums=(0,))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            step(jnp.ones(4))
        assert not [w for w in rec if "not usable" in str(w.message)]
