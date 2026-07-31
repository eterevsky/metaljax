import jax
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


def test_linalg_half_precision():
    # bf16/f16 factorizations upcast to f32 on the host — these FAIL on
    # jax-CPU (LAPACK has no half routines); compare against the f32
    # factorization of the same values instead.
    try:
        metal = jax.devices("metal")[0]
    except RuntimeError:
        pytest.skip("metal plugin not available")
    rng3 = np.random.default_rng(6)
    x = rng3.standard_normal((4, 4)).astype(np.float32)
    sym = x + x.T
    with jax.default_device(metal):
        for dt in (jnp.bfloat16, jnp.float16):
            w16, _ = jax.jit(jnp.linalg.eigh)(jnp.asarray(sym, dt))
            w32, _ = jax.jit(jnp.linalg.eigh)(sym)
            np.testing.assert_allclose(np.asarray(w16, np.float32),
                                       np.asarray(w32), rtol=2e-2, atol=2e-2)
            s16 = jax.jit(lambda a: jnp.linalg.svd(a, compute_uv=False))(
                jnp.asarray(x, dt))
            s32 = jnp.linalg.svd(x, compute_uv=False)
            np.testing.assert_allclose(np.asarray(s16, np.float32),
                                       np.asarray(s32), rtol=2e-2, atol=2e-2)


def _metal_device():
    try:
        return jax.devices("metal")[0]
    except RuntimeError:
        pytest.skip("metal plugin not available")


def test_singular_triangular_solve_is_nonfinite():
    # XLA never fails on a singular triangular solve: the zero pivot
    # divides through to +-inf/nan.  scipy raises LinAlgError, which used
    # to surface as a backend crash.  Only "does not raise" and "not all
    # finite" are checked -- the inf/nan placement is implementation
    # defined, so no value comparison against CPU.
    metal = _metal_device()
    a = np.array([[1.0, 1.0], [0.0, 0.0]], np.float32)
    b = np.array([[1.0], [1.0]], np.float32)
    with jax.default_device(metal):
        out = np.asarray(jax.jit(
            lambda x, y: jax.lax.linalg.triangular_solve(
                x, y, left_side=True))(jnp.asarray(a), jnp.asarray(b)))
        assert not np.all(np.isfinite(out)), out
        # batched (the vmapped/batched lowering takes the same host path)
        outb = np.asarray(jax.jit(
            lambda x, y: jax.lax.linalg.triangular_solve(
                x, y, left_side=True))(jnp.asarray(a)[None],
                                       jnp.asarray(b)[None]))
        assert not np.all(np.isfinite(outb)), outb


def test_singular_det_grad_is_finite_after_filtering():
    # jnp.linalg.det's JVP does the singular triangular solve
    # unconditionally and filters the non-finite results with a where;
    # it only works if the solve returns instead of raising.
    metal = _metal_device()
    with jax.default_device(metal):
        for shape in ((3, 3), (5, 7, 7)):
            z = jnp.zeros(shape, jnp.float32)
            g = np.asarray(jax.jit(jax.grad(
                lambda a: jnp.linalg.det(a).sum()))(z))
            assert g.shape == shape
            assert np.all(np.isfinite(g)), g


@pytest.mark.parametrize("dl,d,du,b", [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [[1.0], [1.0], [1.0]]),
    ([0.0, 1.0], [1.0, 3.0], [3.0, 0.0], [[1.0], [4.0]]),
])
def test_singular_tridiagonal_solve_is_nonfinite(dl, d, du, b):
    # Same contract for the tridiagonal solve: np.linalg.solve raises on
    # a singular matrix, XLA's sweep divides through the zero pivot.
    metal = _metal_device()
    args = [np.array(v, np.float32) for v in (dl, d, du, b)]
    with jax.default_device(metal):
        x = np.asarray(jax.jit(jax.lax.linalg.tridiagonal_solve)(
            *[jnp.asarray(v) for v in args]))
    assert x.shape == args[3].shape
    assert np.any(np.isnan(x)) or np.any(np.isinf(x)), x


def test_nonsingular_tridiagonal_solve_matches_cpu():
    # The singular fallback must not disturb the normal path.
    metal = _metal_device()
    cpu = jax.devices("cpu")[0]
    r = np.random.default_rng(11)
    dl = np.concatenate([[0.0], r.standard_normal(4)]).astype(np.float32)
    d = (r.standard_normal(5) + 4.0).astype(np.float32)
    du = np.concatenate([r.standard_normal(4), [0.0]]).astype(np.float32)
    b = r.standard_normal((5, 2)).astype(np.float32)
    outs = []
    for dev in (metal, cpu):
        with jax.default_device(dev):
            outs.append(np.asarray(jax.jit(jax.lax.linalg.tridiagonal_solve)(
                *[jnp.asarray(v) for v in (dl, d, du, b)])))
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-5, atol=1e-6)
