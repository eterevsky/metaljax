"""Persistent-kernel scan codegen: correctness vs the CPU backend.

These scans qualify for msl_scan's generated kernels (elementwise bodies);
`check` compares against jax-CPU, so any codegen bug shows up as a mismatch,
and any recognizer gap silently falls back to the interpreter (still passing,
but covered by the explicit fallback test below).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import check

rng = np.random.default_rng(7)
L, B, H = 12, 3, 5
A = rng.random((L, B, H)).astype(np.float32) * 0.9
X = (rng.standard_normal((L, B, H)) * 0.3).astype(np.float32)
H0 = rng.standard_normal((B, H)).astype(np.float32)


def affine_scan(a, x, h0):
    def cell(h, ax):
        h = ax[0] * h + ax[1]
        return h, h
    return jax.lax.scan(cell, h0, (a, x))


def test_affine_forward():
    check(affine_scan, A, X, H0)


def test_affine_grad():
    def loss(a, x, h0):
        return affine_scan(a, x, h0)[1].sum()
    check(jax.value_and_grad(loss, argnums=(0, 1, 2)), A, X, H0)


def test_scalar_bool_carry_and_select():
    def f(a, x, h0):
        def cell(carry, ax):
            h, first = carry
            mult = jnp.where(first, 1.0, jnp.sqrt(1.0 - ax[0] ** 2))
            nh = ax[0] * h + mult * ax[1]
            return (nh, jnp.zeros((), bool)), nh
        return jax.lax.scan(cell, (h0, jnp.ones((), bool)), (a, x))[1]
    check(f, A, X, H0)
    check(jax.value_and_grad(lambda a, x, h: f(a, x, h).sum(), argnums=(0, 1)),
          A, X, H0)


def test_mingru_flavor_with_bias():
    bias = (rng.standard_normal(H) * 0.1).astype(np.float32)

    def f(z_in, c_in, h0, b):
        def cell(h, zc):
            z = jax.nn.sigmoid(zc[0] + b)
            c = jnp.tanh(zc[1])
            nh = (1 - z) * h + z * c
            return nh, nh
        return jax.lax.scan(cell, h0, (z_in, c_in))
    check(f, A, X, H0, bias)
    check(lambda a, x, h, b: jax.value_and_grad(
        lambda *v: f(*v)[1].sum(), argnums=(0, 1, 2))(a, x, h, b),
        A, X, H0, bias)


def test_transcendental_cell():
    def f(a, x, h0):
        def cell(h, ax):
            nh = jnp.exp(-jnp.abs(ax[0])) * h + jnp.log1p(ax[1] ** 2)
            return nh, jnp.minimum(nh, 3.0)
        return jax.lax.scan(cell, h0, (a, x))
    check(f, A, X, H0, rtol=1e-5, atol=1e-5)


def test_matvec_cell_falls_back():
    # h @ W inside the body: not elementwise, must fall back and still be
    # correct through the regular loop path.
    W = (rng.standard_normal((H, H)) * 0.2).astype(np.float32)

    def f(x, h0, w):
        def cell(h, xt):
            nh = jnp.tanh(xt + h @ w)
            return nh, nh
        return jax.lax.scan(cell, h0, x)
    check(f, X, H0, W, rtol=1e-4, atol=1e-5)


def test_int_carry_data():
    # non-affine integer state (min-tracking): falls back or runs — either
    # way must match CPU.
    XI = rng.integers(-50, 50, (L, B, H)).astype(np.int32)

    def f(xs):
        def cell(m, x):
            nm = jnp.minimum(m, x)
            return nm, nm
        return jax.lax.scan(cell, jnp.full((B, H), 100, jnp.int32), xs)
    check(f, XI)


def test_small_matvec_cell_vector_mode():
    W = (rng.standard_normal((H, H)) * 0.3).astype(np.float32)

    def f(xs, h0, w):
        def cell(h, x):
            g = jax.nn.sigmoid(x + h @ w)
            nh = (1 - g) * h + g * jnp.tanh(x)
            return nh, nh
        return jax.lax.scan(cell, h0, xs)
    check(f, X, H0, W)
    check(jax.value_and_grad(lambda a, b, c: f(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), X, H0, W)


def test_block_diagonal_cell_vector_mode():
    K, C = 2, 4
    W = (rng.standard_normal((K, C, C)) * 0.3).astype(np.float32)
    Xb = (rng.standard_normal((L, B, K, C)) * 0.3).astype(np.float32)
    hb = rng.standard_normal((B, K, C)).astype(np.float32)

    def f(xs, h0, w):
        def cell(h, x):
            nh = jnp.tanh(jnp.einsum('bkc,kcd->bkd', h, w) + x)
            return nh, nh
        return jax.lax.scan(cell, h0, xs)
    check(f, Xb, hb, W)
    check(jax.value_and_grad(lambda a, b, c: f(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), Xb, hb, W)


def test_concat_cell():
    # split.cat-style composition: two sub-cells concatenated on features
    # (concat lowers to summed zero-pads; the AD reverse loop broadcasts
    # through the pads).
    def f(xs, h0):
        def cell(h, x):
            a = jnp.tanh(h[..., :3] + x[..., :3])
            b = jax.nn.sigmoid(h[..., 3:] * x[..., 3:])
            nh = jnp.concatenate([a, b], axis=-1)
            return nh, nh
        return jax.lax.scan(cell, h0, xs)
    check(f, X, H0)
    check(jax.value_and_grad(lambda a, b: f(a, b)[1].sum(), argnums=(0, 1)),
          X, H0)


def test_sliced_fused_dot_gates():
    # One fused matvec whose output is gate-split by slicing (slice of a
    # SymDot shifts the weight window).
    W = (rng.standard_normal((H, 2 * H)) * 0.3).astype(np.float32)

    def f(xs, h0, w):
        def cell(h, x):
            d = h @ w
            z = jax.nn.sigmoid(d[..., :H] + x)
            n = jnp.tanh(d[..., H:])
            nh = (1 - z) * h + z * n
            return nh, nh
        return jax.lax.scan(cell, h0, xs)
    check(f, X, H0, W)
    check(jax.value_and_grad(lambda a, b, c: f(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), X, H0, W)


def test_i64_scalar_carry():
    # texmo runs under jax_enable_x64: position/suffix indices are i64
    # scalars carried through the scan (msl counters must accept them).
    jax.config.update("jax_enable_x64", True)
    try:
        def f(xs):
            def cell(c, x):
                h, k = c
                return (h * 0.9 + x.astype(h.dtype).sum() * 0.01, k + 1), k
            (h, k), ks = jax.lax.scan(
                cell, (jnp.float32(1.0), jnp.int64(3)), xs)
            return h, k, ks

        xs = np.ones((32, 4), np.float32)
        got = jax.jit(f, backend="metal")(jnp.array(xs))
        want = jax.jit(f, backend="cpu")(jnp.array(xs))
        for g, w in zip(got, want):
            np.testing.assert_allclose(np.array(g), np.array(w),
                                       rtol=1e-5, atol=1e-6)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_mul_reduce_contraction_cell():
    # lrnn-style: the block matvec written as broadcast-multiply + reduce
    # (texmo's lowering) instead of einsum/dot_general.
    K, C = 2, 4
    W = (rng.standard_normal((K, C, C)) * 0.3).astype(np.float32)
    Xb = (rng.standard_normal((L, B, K, C)) * 0.3).astype(np.float32)
    hb = rng.standard_normal((B, K, C)).astype(np.float32)

    def f(xs, h0, w):
        def cell(h, x):
            nh = jnp.tanh((h[:, :, None, :] * w[None]).sum(-1) + x)
            return nh, nh
        return jax.lax.scan(cell, h0, xs)
    check(f, Xb, hb, W)
    check(jax.value_and_grad(lambda a, b, c: f(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), Xb, hb, W)


def test_register_reduce_readout():
    # lrnn-style readout: contraction of register-resident products via
    # reduce over the register dim (no stored weight on the reduce path).
    W = (rng.standard_normal((H, H)) * 0.3).astype(np.float32)

    def f(xs, h0, w):
        def cell(h, x):
            d = h @ w
            y = (d * x).sum(-1)          # in-lane register reduce
            nh = jnp.tanh(d + y[..., None] * 0.1)
            return nh, y
        return jax.lax.scan(cell, h0, xs)
    check(f, X, H0, W)
    check(jax.value_and_grad(lambda a, b, c: f(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), X, H0, W)


def test_rectangular_coop_dots():
    # mullstm-class cell at full width: fused 3-gate weights (F x 3F) and
    # a wide input projection (4F -> F) force rectangular coop dots.
    F = 32
    Wg = (rng.standard_normal((F, 3 * F)) * 0.1).astype(np.float32)
    Wp = (rng.standard_normal((4 * F, F)) * 0.1).astype(np.float32)
    Xw = (rng.standard_normal((10, 3, 4 * F)) * 0.3).astype(np.float32)
    h0w = rng.standard_normal((3, F)).astype(np.float32)

    def f(xs, h0, wg, wp):
        def cell(h, x):
            xp = x @ wp                     # (B, 4F) @ (4F, F)
            g = h @ wg                      # (B, F) @ (F, 3F)
            i = jax.nn.sigmoid(g[..., :F] + xp)
            o = jax.nn.sigmoid(g[..., F:2 * F])
            c = jnp.tanh(g[..., 2 * F:])
            nh = o * jnp.tanh(i * c + (1 - i) * h)
            return nh, nh
        return jax.lax.scan(cell, h0, xs)
    check(f, Xw, h0w, Wg, Wp, rtol=1e-4, atol=1e-5)
    check(jax.value_and_grad(lambda a, b, c, d: f(a, b, c, d)[1].sum(),
                             argnums=(0, 1, 2, 3)), Xw, h0w, Wg, Wp,
          rtol=1e-3, atol=1e-4)
