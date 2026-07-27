import jax.numpy as jnp
import numpy as np
import pytest

from helpers import check

rng = np.random.default_rng(0)


def test_matmul_2d():
    a = rng.standard_normal((7, 5), dtype=np.float32)
    b = rng.standard_normal((5, 9), dtype=np.float32)
    check(jnp.matmul, a, b, rtol=1e-4, atol=1e-5)


def test_vector_dot():
    a = rng.standard_normal(64, dtype=np.float32)
    check(jnp.dot, a, a, rtol=1e-4, atol=1e-4)


def test_matvec():
    a = rng.standard_normal((6, 4), dtype=np.float32)
    v = rng.standard_normal(4, dtype=np.float32)
    check(jnp.dot, a, v, rtol=1e-4, atol=1e-5)


def test_batched_matmul():
    a = rng.standard_normal((3, 7, 5), dtype=np.float32)
    b = rng.standard_normal((3, 5, 2), dtype=np.float32)
    check(jnp.matmul, a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("spec,shapes", [
    ("bij,bj->bi", [(3, 4, 5), (3, 5)]),
    ("bhtd,bhsd->bhts", [(2, 3, 4, 5), (2, 3, 6, 5)]),
    ("bhts,bhsd->bhtd", [(2, 3, 4, 6), (2, 3, 6, 5)]),
    ("...kc,kcd->...kd", [(2, 3, 4), (3, 4, 5)]),
    ("hd,hde->he", [(3, 4), (3, 4, 5)]),
    ("ij,kj->ik", [(4, 5), (6, 5)]),
])
def test_einsum(spec, shapes):
    args = [rng.standard_normal(s, dtype=np.float32) for s in shapes]
    check(lambda *xs: jnp.einsum(spec, *xs), *args, rtol=1e-4, atol=1e-5)


def test_outer():
    a = rng.standard_normal(5, dtype=np.float32)
    b = rng.standard_normal(7, dtype=np.float32)
    check(jnp.outer, a, b, rtol=1e-5, atol=1e-6)


def test_int_matmul():
    a = rng.integers(-5, 5, (4, 3)).astype(np.int32)
    b = rng.integers(-5, 5, (3, 6)).astype(np.int32)
    check(jnp.matmul, a, b)


def test_mixed_precision_matmul():
    a = rng.standard_normal((4, 5)).astype(jnp.bfloat16)
    b = rng.standard_normal((5, 6)).astype(jnp.bfloat16)
    check(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32),
          a, b, rtol=2e-2, atol=2e-2)


def test_plain_stablehlo_dot():
    # stablehlo.dot never comes out of jax; feed the interpreter a module
    # directly (HLO-imported benchmarks contain it).
    from metaljax import _ir
    from metaljax.interpreter import Interpreter
    import mlx.core as mx

    mod = """
module {
  func.func @main(%a: tensor<3x4xf32>, %b: tensor<4x5xf32>) -> tensor<3x5xf32> {
    %0 = stablehlo.dot %a, %b : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}
"""
    interp = Interpreter(mod)
    a = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((4, 5)).astype(np.float32)
    (out,) = interp(mx.array(a), mx.array(b))
    np.testing.assert_allclose(np.array(out), a @ b, rtol=1e-5, atol=1e-6)


def test_lapack_qr_eigh_svd():
    # host-computed custom_call targets (Qr/orgqr, syevd, gesdd)
    rng2 = np.random.default_rng(5)
    x = rng2.standard_normal((4, 4)).astype(np.float32)
    r53 = rng2.standard_normal((5, 3)).astype(np.float32)
    sym = (x + x.T).astype(np.float32)
    # eigenvector/singular-vector signs are arbitrary: compare via
    # reconstruction identities instead of raw factors.
    check(lambda r: tuple(jnp.abs(v) for v in jnp.linalg.qr(r)), r53,
          rtol=1e-4, atol=1e-5)
    check(lambda s: jnp.sort(jnp.linalg.eigh(s)[0]), sym,
          rtol=1e-4, atol=1e-5)
    check(lambda x: jnp.linalg.svd(x, compute_uv=False), x,
          rtol=1e-4, atol=1e-5)
    check(lambda x: jnp.linalg.pinv(x), x, rtol=1e-3, atol=1e-4)
    check(lambda r: jnp.linalg.lstsq(r, jnp.ones(5))[0], r53,
          rtol=1e-3, atol=1e-4)
