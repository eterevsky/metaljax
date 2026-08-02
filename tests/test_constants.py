"""Splat constants: values, and the memory they are allowed to cost.

A splat is a dense constant whose every element is the same value -- MLIR
writes it as `dense<1.5> : tensor<151936x1024xf32>`, four bytes of IR for a
622 MB tensor. XLA broadcasts one; materializing one costs the whole shape,
and (because mx.compile keeps a compiled graph's constants alive for as long
as the executable) keeps costing it. jax's random.normal lowering carries 23
full-shape splat coefficients, so a keras model whose weights are created by
an initializer retained 23 copies of every weight tensor it ever built.
"""

import gc

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest

from helpers import check
from metaljax import Interpreter

SHAPE = (2048, 2048)
CONST_BYTES = 4 * SHAPE[0] * SHAPE[1]  # 16 MiB as f32


def _settled_active() -> int:
    """Device memory in use once everything droppable has been dropped."""
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory()


def _splat_program(value: float) -> bytes:
    """max(arg * dense<value>) over a big tensor: small in, small out, one
    whole-shape splat constant in between.

    Written as IR because jax lowers `jnp.full` as a rank-0 constant plus a
    broadcast; the whole-shape splats that cost the memory come out of
    StableHLO folding inside jax's own lowerings (jax.random.normal carries
    23 of them).
    """
    n, m = SHAPE
    return f"""
    module {{
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {{
        %c = stablehlo.constant dense<{value:e}> : tensor<{n}x{m}xf32>
        %b = stablehlo.broadcast_in_dim %arg0, dims = []
             : (tensor<f32>) -> tensor<{n}x{m}xf32>
        %p = stablehlo.multiply %b, %c : tensor<{n}x{m}xf32>
        %i = stablehlo.constant dense<0xFF800000> : tensor<f32>
        %r = stablehlo.reduce(%p init: %i) applies stablehlo.maximum
             across dimensions = [0, 1]
             : (tensor<{n}x{m}xf32>, tensor<f32>) -> tensor<f32>
        return %r : tensor<f32>
      }}
    }}
    """.encode()


def test_splat_constant_is_not_retained_per_executable():
    """Distinct executables carrying big splat constants must not each pay
    for the whole shape. mx.compile keeps a compiled graph's constants
    alive, so pre-fix every executable held CONST_BYTES until it died."""
    from metaljax import engine

    n = 4
    arg = engine.buffer_from_host(np.float32(2.0).tobytes(),
                                  engine._ENUM["F32"], [], None)
    exes = []
    base = _settled_active()
    for i in range(n):
        value = 1.0 + i
        ex = engine.compile_program(_splat_program(value), "mlir")
        (out,) = engine.execute(ex, [arg])
        got = np.frombuffer(engine.to_host(out), np.float32)[0]
        assert got == np.float32(2.0 * value), f"{got} != {2.0 * value}"
        exes.append(ex)  # keep them alive: that is the leak under test
    grew = _settled_active() - base
    assert grew < CONST_BYTES // 2, (
        f"{n} executables with a {CONST_BYTES} B splat constant each "
        f"retained {grew} B of device memory (materialized splats cost "
        f"~{n * CONST_BYTES} B)")


def test_splat_constant_costs_nothing_to_read():
    """Reading a splat allocated the full shape -- on the host (the MLIR
    bindings expand splats into dense arrays) and then on the device."""
    src = """
    module {
      func.func @main() -> tensor<2048x2048xf32> {
        %0 = stablehlo.constant dense<1.5> : tensor<2048x2048xf32>
        return %0 : tensor<2048x2048xf32>
      }
    }
    """
    interp = Interpreter(src)
    base = _settled_active()
    (out,) = interp()
    mx.eval(out)
    assert out.shape == SHAPE
    assert float(out[0, 0]) == 1.5 and float(out[-1, -1]) == 1.5
    grew = mx.get_active_memory() - base
    assert grew < CONST_BYTES // 2, (
        f"reading a {CONST_BYTES} B splat constant allocated {grew} B")


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
