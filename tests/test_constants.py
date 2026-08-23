"""Splat constants: broadcasting one must not change what it holds.

A splat is a dense constant whose every element is the same value -- MLIR
writes it as `dense<1.5> : tensor<151936x1024xf32>`, four bytes of IR for a
622 MB tensor.  jax's own lowerings fold whole-shape splats into programs
(jax.random.normal carries 23 of them), so the value path here is hot in
every real workload.  The Stage-1 memory-accounting tests (per-executable
splat retention, splat read cost) were engine-internal and were removed at
the Stage-1 retirement; the value tests below run through the PJRT plugin.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import check


@pytest.mark.parametrize("dtype,value", [
    (jnp.float32, 1.2345678901234),
    (jnp.float32, float("inf")),
    (jnp.float32, float("-inf")),
    (jnp.float16, 2.5),
    (jnp.bfloat16, 1.5),
    (jnp.int32, -7),
    (jnp.uint32, 9),
    (jnp.int8, -3),
    (jnp.bool_, True),
])
def test_splat_values(dtype, value):
    """Broadcasting a splat must not change what it holds."""
    x = np.arange(12, dtype=np.float32).reshape(3, 4)

    def f(a):
        c = jnp.full((3, 4), value, dtype)
        if dtype == jnp.bool_:
            return jnp.logical_and(a > 2, c), c
        return a.astype(dtype) + c, c

    check(f, x)


def test_splat_nan():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)

    def f(a):
        c = jnp.full((2, 3), np.nan, jnp.float32)
        return jnp.isnan(a + c), jnp.isnan(c)

    check(f, x)


def test_splat_consumers():
    """Broadcast views must survive the ops that need contiguous inputs
    (MLX has had strided-view bugs in reductions, sort and conv)."""
    x = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)

    def f(a):
        c = jnp.full((8, 16), 0.25, jnp.float32)
        return (jnp.sum(a * c, axis=0),
                jnp.sort(jnp.where(a > 0, c, a), axis=-1),
                jnp.cumsum(a * c, axis=1),
                jnp.argmax(a + c, axis=1),
                jax.lax.top_k(a * c, 3)[0],
                jnp.concatenate([a, c], axis=0),
                (a @ c.T).sum(),
                jnp.transpose(c)[:4].sum())

    check(f, x, rtol=1e-5, atol=1e-5)


def test_splat_reduce_window():
    x = np.random.default_rng(1).standard_normal((1, 8, 8, 1)).astype(np.float32)

    def f(a):
        c = jnp.full((1, 8, 8, 1), 0.5, jnp.float32)
        return jax.lax.reduce_window(a * c, -np.inf, jax.lax.max,
                                     (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    check(f, x)


def test_nonsplat_constant_still_exact():
    """The elementwise (non-splat) path is untouched."""
    vals = np.linspace(-3, 3, 24, dtype=np.float32).reshape(4, 6)

    def f(a):
        return a + jnp.asarray(vals)

    check(f, np.zeros((4, 6), np.float32))
