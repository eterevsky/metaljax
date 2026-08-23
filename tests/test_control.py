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


# A data-dependent trip count says nothing about the body, so the body of a
# dynamic while is compiled like any other. LLM decode is exactly this loop
# (one transformer step per token, stopping on an end-of-sequence token):
# interpreting the body op by op made it Python-dispatch-bound.
_W = rng.standard_normal((8, 8)).astype(np.float32) * 0.3


def _dynamic_while(x):
    def cond(c):
        h, i = c
        # Not the counted pattern (an `and`, not a bare LT on the counter),
        # yet bounded, so the test cannot hang.
        return jnp.logical_and(i < 25, jnp.max(jnp.abs(h)) < 50.0)

    def body(c):
        h, i = c
        return (jnp.tanh(h @ _W) * 3.0 + 0.5, i + 1)

    return jax.lax.while_loop(cond, body, (x, jnp.int32(0)))


def test_dynamic_while_matches_cpu():
    check(_dynamic_while, rng.standard_normal(8).astype(np.float32),
          rtol=1e-5, atol=1e-5)


# A compiled body's kernels are GENERATED AT EVAL, not by the call that
# builds the graph. MLX fuses a wide elementwise body into a single Metal
# kernel whose argument buffers are (non-constant inputs + outputs + up to
# three for strides/shape/ndim), and Metal caps those at 31 -- so a body
# reading ~29 distinct arrays dies with "Too many inputs/outputs fused in
# the Metal Compiled primitive". Below: a wide carry combined through
# selects, which is the shape of jax's rejection samplers (random.binomial,
# random.multinomial) and of the hyp2f1 series -- the real programs this
# came from. The failure surfaces at the next sync point, by which time the
# carry has advanced past the iteration a redo could repair, so the body
# runner probes each freshly compiled body with one synchronous eval and
# falls back to the uncompiled body. Without that probe this dies at the
# loop flush and the whole execute fails.
_NWIDE = 20


def _wide_dynamic_while(x):
    def cond(c):
        xs, i = c
        # Not the counted pattern -> the dynamic path, whose body compiles.
        return jnp.logical_and(i < 3, xs[0][0] < 1e6)

    def body(c):
        xs, i = c
        terms = [jnp.where(xs[j] > 0, xs[j] * np.float32(1.0 + j),
                           xs[j] - np.float32(0.5)) for j in range(_NWIDE)]
        # Combined as a TREE: MLX stops fusing at depth 11, so a linear
        # chain of the same width would never reach one kernel.
        while len(terms) > 1:
            nxt = [terms[k] + terms[k + 1] for k in range(0, len(terms) - 1, 2)]
            if len(terms) % 2:
                nxt.append(terms[-1])
            terms = nxt
        acc = terms[0]
        return (tuple(xs[j] * 0.5 + acc * 0.001 for j in range(_NWIDE)),
                i + 1)

    xs = tuple(x + np.float32(j) for j in range(_NWIDE))
    return jax.lax.while_loop(cond, body, (xs, jnp.int32(0)))


def test_wide_dynamic_while_body():
    check(_wide_dynamic_while, rng.standard_normal(8).astype(np.float32),
          rtol=1e-5, atol=1e-5)


def test_binomial_sampler_wide_body():
    # The reported case: jax's btrs rejection loop is a dynamic while whose
    # body carries 16 values and captures 12 more.
    def f(seed):
        key = jax.random.key(seed)
        return jax.random.binomial(key, jnp.float32(20.0), jnp.float32(0.3),
                                   shape=(3,))
    check(f, np.uint32(0))


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
    # mx::compile dies (unordered_map::at) when two outputs bake to equal
    # constants; the engine must anchor non-input-derived outputs so the
    # whole executable still compiles and the values survive exactly.
    from helpers import run_module

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
    y, a, b = run_module(mod, (np.array([1.0, 2.0], np.float32),))
    np.testing.assert_array_equal(y, [2.0, 4.0])
    assert float(a) == float(np.float32(0.999))
    assert float(b) == float(np.float32(0.999))


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
