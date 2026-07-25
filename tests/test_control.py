import jax
import jax.numpy as jnp
import numpy as np

from helpers import check

rng = np.random.default_rng(2)


def test_fori_loop():
    def f(x):
        return jax.lax.fori_loop(0, 10, lambda i, c: c + i * x, x)
    check(f, np.float32(1.5))


def test_while_loop():
    def f(x):
        return jax.lax.while_loop(lambda c: c[0] < 100.0,
                                  lambda c: (c[0] * 2, c[1] + 1),
                                  (x, jnp.int32(0)))
    check(f, np.float32(3.0))


def test_cond():
    def f(p, x):
        return jax.lax.cond(p, lambda a: a * 2.0, lambda a: a - 1.0, x)
    check(f, np.bool_(True), np.float32(5.0))
    check(f, np.bool_(False), np.float32(5.0))


def test_scan_cumsum():
    def f(xs):
        return jax.lax.scan(lambda c, x: (c + x, c + x), jnp.float32(0.0), xs)
    check(f, rng.standard_normal(16).astype(np.float32), rtol=1e-4, atol=1e-5)


def test_scan_matmul_carry():
    w = rng.standard_normal((4, 4)).astype(np.float32) * 0.1

    def f(xs):
        def step(h, x):
            h = jnp.tanh(h @ w + x)
            return h, h
        return jax.lax.scan(step, jnp.zeros(4, jnp.float32), xs)
    check(f, rng.standard_normal((8, 4)).astype(np.float32), rtol=1e-4, atol=1e-5)


def test_gelu():
    x = rng.standard_normal(16).astype(np.float32)
    check(jax.nn.gelu, x, rtol=1e-5, atol=1e-6)
    check(lambda a: jax.nn.gelu(a, approximate=False), x, rtol=1e-5, atol=1e-6)


def test_nested_jit():
    g = jax.jit(lambda x: x * 3.0)

    def f(x):
        return g(x) + g(x * 2)
    check(f, np.float32(2.0))


def test_remat():
    f = jax.checkpoint(lambda x: jnp.sin(x) * jnp.cos(x))
    check(f, rng.standard_normal(8).astype(np.float32))


def test_equal_constant_outputs_compile():
    # mx.compile dies (unordered_map::at) when two outputs bake to equal
    # constants; the engine anchors non-input-derived outputs so the whole
    # executable still compiles. Exercise the real runner path.
    import mlx.core as mx
    from metaljax import engine

    mod = """
module {
  func.func @main(%x: tensor<2xf32>) -> (tensor<2xf32>, tensor<f32>, tensor<f32>) {
    %a = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %b = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %y = stablehlo.add %x, %x : tensor<2xf32>
    return %y, %a, %b : tensor<2xf32>, tensor<f32>, tensor<f32>
  }
}
"""
    ex = engine.compile_program(mod.encode(), "mlir")
    x = mx.array(np.array([1.0, 2.0], np.float32))
    outs = list(ex.runner()(x))
    mx.eval(*outs)
    assert ex._can_compile  # must have taken the compiled path
    np.testing.assert_array_equal(np.array(outs[0]), [2.0, 4.0])
    assert float(outs[1].item()) == float(np.float32(0.999))
    assert float(outs[2].item()) == float(np.float32(0.999))


def test_scan_carrying_equal_constants():
    # A scan whose carries include two equal scalar constants each step —
    # the compiled-body path must not trip MLX's equal-constant folding.
    def f(x):
        def cell(c, xt):
            h, u, v = c
            return (jnp.tanh(h + xt), jnp.float32(0.999) * 1.0,
                    jnp.float32(0.999) * 1.0), h
        (h, u, v), ys = jax.lax.scan(
            cell, (x, jnp.float32(0.0), jnp.float32(0.0)),
            jnp.ones((70, 4), jnp.float32))
        return h + u + v, ys
    check(f, np.ones((4,), np.float32))
