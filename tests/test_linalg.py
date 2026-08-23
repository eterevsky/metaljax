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


def _idot(a, b, out=jnp.int32):
    return jax.lax.dot_general(a, b, (((a.ndim - 1,), (0,)), ((), ())),
                               preferred_element_type=out)


# 8-bit operands take the chunked-f32 dot (MLX has no integer matmul); the
# chunk is sized so no partial sum can leave f32's exact-integer range, so
# every case below must match the CPU backend BITWISE, not approximately.
# K is swept across the i8 chunk boundary (1024) and well past it.
@pytest.mark.parametrize("K", [7, 256, 1024, 1040, 4096, 12288])
@pytest.mark.parametrize("M", [1, 64])
def test_int8_dot_exact(K, M):
    a = rng.integers(-128, 128, (M, K)).astype(np.int8)
    b = rng.integers(-128, 128, (K, 24)).astype(np.int8)
    check(_idot, a, b)


@pytest.mark.parametrize("K", [1024, 1040, 4096, 12288])
def test_int8_dot_adversarial(K):
    # Every element at an extreme, so the true |sum| is K * 128 * 127 or
    # K * 128**2 -- far past 2**24, where a single unchunked f32 matmul
    # starts losing whole integers.
    for av, bv in ((-128, -128), (127, -128), (-128, 127), (127, 127)):
        check(_idot, np.full((64, K), av, np.int8),
              np.full((K, 8), bv, np.int8))


def test_int8_dot_s32_wraparound():
    # True sum 3,288,334,336 > 2**31: XLA's integer dot is defined modulo
    # the result width, and the chunk accumulator has to wrap the same way.
    K = 200704
    a = np.full((4, K), -128, np.int8)
    b = np.full((K, 3), -128, np.int8)
    check(_idot, a, b)


def test_int8_dot_narrow_result():
    # preferred_element_type narrower than the accumulator: the result
    # wraps to 8 bits.
    a = rng.integers(-128, 128, (6, 2048)).astype(np.int8)
    b = rng.integers(-128, 128, (2048, 5)).astype(np.int8)
    check(lambda x, y: _idot(x, y, jnp.int8), a, b)
    check(lambda x, y: _idot(x, y, jnp.int16), a, b)


@pytest.mark.parametrize("K", [255, 256, 257, 12288])
def test_uint8_dot_exact(K):
    # u8 x u8 peaks at 255**2 per product, so the chunk is 256; the
    # all-255 case is the worst input there is.
    a = rng.integers(0, 256, (16, K)).astype(np.uint8)
    b = rng.integers(0, 256, (K, 9)).astype(np.uint8)
    check(lambda x, y: _idot(x, y, jnp.uint32), a, b)
    check(lambda x, y: _idot(x, y, jnp.uint32),
          np.full((8, K), 255, np.uint8), np.full((K, 4), 255, np.uint8))


def test_mixed_sign_int8_dot():
    # jax converts mixed-signedness operands to i32 before the dot, so the
    # i8 x u8 chunk (512) is only reachable from a hand-written module.
    from helpers import run_module

    mod = """
module {
  func.func @main(%a: tensor<3x2048xi8>, %b: tensor<2048x5xui8>) -> tensor<3x5xi32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<3x2048xi8>, tensor<2048x5xui8>) -> tensor<3x5xi32>
    return %0 : tensor<3x5xi32>
  }
}
"""
    for a, b in (
        (np.full((3, 2048), -128, np.int8), np.full((2048, 5), 255, np.uint8)),
        (rng.integers(-128, 128, (3, 2048)).astype(np.int8),
         rng.integers(0, 256, (2048, 5)).astype(np.uint8)),
    ):
        (out,) = run_module(mod, (a, b))
        want = (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
        np.testing.assert_array_equal(out, want)


def test_int8_dot_batched_and_multi_contracting():
    # The chunk slicing happens on the canonicalized [B, M, K] x [B, K, N]
    # form, so batch dims and a multi-dimension contraction (qwix's output
    # projection contracts [2,3] x [0,1]) must slice correctly too.
    a = rng.integers(-128, 128, (3, 5, 2048)).astype(np.int8)
    b = rng.integers(-128, 128, (3, 2048, 7)).astype(np.int8)
    check(lambda x, y: jax.lax.dot_general(
        x, y, (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.int32), a, b)
    a2 = np.full((1, 4, 16, 128), -128, np.int8)
    b2 = np.full((16, 128, 64), -128, np.int8)
    check(lambda x, y: jax.lax.dot_general(
        x, y, (((2, 3), (0, 1)), ((), ())),
        preferred_element_type=jnp.int32), a2, b2)


def test_int16_dot_stays_exact():
    # i16 products (up to 2**30) are not f32-representable, so these keep
    # the int64 outer-product path -- which must still be exact.
    a = rng.integers(-32768, 32768, (5, 300)).astype(np.int16)
    b = rng.integers(-32768, 32768, (300, 4)).astype(np.int16)
    check(_idot, a, b)
    check(_idot, np.full((5, 300), -32768, np.int16),
          np.full((300, 4), -32768, np.int16))


def test_mixed_precision_matmul():
    a = rng.standard_normal((4, 5)).astype(jnp.bfloat16)
    b = rng.standard_normal((5, 6)).astype(jnp.bfloat16)
    check(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32),
          a, b, rtol=2e-2, atol=2e-2)


@pytest.mark.xfail(strict=True, reason=(
    "the native plugin declines `op stablehlo.dot` (it lowers dot_general "
    "only). jax never emits the plain form, so this bites only "
    "hand-written / xla-translate'd modules -- the Stage-1 interpreter "
    "handled it, and the gap is named in the Stage-1 retirement report."))
def test_plain_stablehlo_dot():
    # stablehlo.dot never comes out of jax; feed the plugin a module
    # directly (HLO-imported benchmarks contain it).
    from helpers import run_module

    mod = """
module {
  func.func @main(%a: tensor<3x4xf32>, %b: tensor<4x5xf32>) -> tensor<3x5xf32> {
    %0 = stablehlo.dot %a, %b : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}
"""
    a = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((4, 5)).astype(np.float32)
    (out,) = run_module(mod, (a, b))
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


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


def _export_both(f, poly, dtype, x):
    """Export f for metal and for cpu under the same symbolic shape and
    run both on x."""
    from jax import export
    _metal_device()
    outs = []
    for name in ("metal", "cpu"):
        with jax.default_device(jax.devices(name)[0]):
            e = export.export(jax.jit(f), platforms=[name])(
                jax.ShapeDtypeStruct(export.symbolic_shape(poly), dtype))
            outs.append(jax.tree.map(np.asarray, e.call(x)))
    return outs


@pytest.mark.parametrize("poly,shape,dtype", [
    ("b, 4", (5, 4), np.float32),
    ("m, n", (5, 4), np.float32),
    ("b1, b2, m, n", (2, 3, 4, 5), np.float32),
    ("b1, b2, m, n", (2, 3, 8, 4), np.complex64),
    ("b, m, 0", (2, 4, 0), np.float32),
    ("b, 0, n", (2, 0, 4), np.float32),
    # symbolic BATCH only: must stay on jax's generic device algorithm,
    # whose rounding the host getrf does not reproduce at low precision
    ("b1, b2, 4, 5", (2, 3, 4, 5), np.float32),
])
def test_shape_polymorphic_lu_matches_cpu(poly, shape, dtype):
    # jax's generic LU lowering is a python loop over min(m, n), which a
    # symbolic dimension cannot drive; metal routes those to a host getrf.
    r = np.random.default_rng(3)
    x = r.standard_normal(shape).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * r.standard_normal(shape).astype(np.float32)
    got, want = _export_both(jax.lax.linalg.lu, poly, dtype, x)
    for g, w in zip(got, want):
        assert g.shape == w.shape and g.dtype == w.dtype
        if np.issubdtype(g.dtype, np.inexact):
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(g, w)


def test_shape_polymorphic_inv_matches_cpu():
    r = np.random.default_rng(4)
    x = (r.standard_normal((5, 5)) + 5 * np.eye(5)).astype(np.float32)
    got, want = _export_both(jnp.linalg.inv, "b, b", np.float32, x)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_static_shape_lu_still_matches_cpu():
    # The host path must not disturb the on-device factorization.
    metal = _metal_device()
    x = np.random.default_rng(5).standard_normal((6, 4)).astype(np.float32)
    outs = []
    for dev in (metal, jax.devices("cpu")[0]):
        with jax.default_device(dev):
            outs.append([np.asarray(t)
                         for t in jax.jit(jax.lax.linalg.lu)(jnp.asarray(x))])
    np.testing.assert_allclose(outs[0][0], outs[1][0], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][2], outs[1][2])
