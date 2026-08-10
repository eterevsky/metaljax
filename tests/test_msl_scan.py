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


# ---- coop preference + input packing (0.4.3) ----

F16 = 16
W16 = (rng.standard_normal((F16, F16)) * 0.25).astype(np.float32)
X16 = (rng.standard_normal((L, B, F16)) * 0.3).astype(np.float32)
H16 = rng.standard_normal((B, F16)).astype(np.float32)


def _mgru16(xs, h0, w):
    def cell(h, x):
        g = jax.nn.sigmoid(x + h @ w)
        nh = (1 - g) * h + g * jnp.tanh(x)
        return nh, nh
    return jax.lax.scan(cell, h0, xs)


def _plan_modes(f, *args):
    """Run through a fresh executable; return the MSL plan modes built."""
    from helpers import execute_module, from_buffer, lower_bytes

    ex, outs = execute_module(lower_bytes(f, *args), jax.tree.leaves(args))
    for o in outs:
        from_buffer(o)          # settle: plans are built as the loop runs
    return [p.mode for p in ex.interpreter._msl_cache.values()
            if p is not None]


def test_coop_preferred_for_square_16wide_cell():
    # F=16 square dots used to pick vector mode (one thread per batch
    # lane, no latency hiding — 5-8x slower); coop must be preferred and
    # match CPU.
    check(_mgru16, X16, H16, W16, rtol=1e-4, atol=1e-5)
    check(jax.value_and_grad(lambda a, b, c: _mgru16(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), X16, H16, W16,
          rtol=1e-3, atol=1e-4)
    modes = _plan_modes(_mgru16, X16, H16, W16)
    assert "coop" in modes, modes


def test_coop_pref_off_restores_vector(monkeypatch):
    from metaljax import msl_scan
    monkeypatch.setattr(msl_scan, "_COOP_PREF", False)
    check(_mgru16, X16, H16, W16, rtol=1e-4, atol=1e-5)
    modes = _plan_modes(_mgru16, X16, H16, W16)
    assert modes and "coop" not in modes, modes


def test_coop_flip_falls_back_to_vector(monkeypatch):
    # If the preferred coop build dies in emission, build_plan must retry
    # the old vector pick rather than losing the plan entirely.
    from metaljax import msl_scan

    def boom(self):
        raise msl_scan._Unsupported("synthetic coop emitter gap")

    monkeypatch.setattr(msl_scan.Plan, "_emit_coop", boom)
    check(_mgru16, X16, H16, W16, rtol=1e-4, atol=1e-5)
    modes = _plan_modes(_mgru16, X16, H16, W16)
    assert modes and all(m == "vector" for m in modes), modes


def test_input_packing_low_trigger(monkeypatch):
    # Force packing on ordinary plans: results must be unchanged (packing
    # only relocates buffers; offsets are static per plan).
    from metaljax import msl_scan
    monkeypatch.setattr(msl_scan, "_PACK_TRIGGER", 2)
    check(affine_scan, A, X, H0)
    check(jax.value_and_grad(
        lambda a, x, h: affine_scan(a, x, h)[1].sum(),
        argnums=(0, 1, 2)), A, X, H0)
    check(_mgru16, X16, H16, W16, rtol=1e-4, atol=1e-5)
    check(jax.value_and_grad(lambda a, b, c: _mgru16(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), X16, H16, W16,
          rtol=1e-3, atol=1e-4)


def test_many_input_cell_packs_over_limit():
    # A body with enough distinct captured tensors to blow Metal's 31-
    # binding cap without packing; must still take the MSL path and match.
    ws = [(rng.standard_normal((H,)) * 0.1
           + (1.0 if i % 2 else 0.9)).astype(np.float32)
          for i in range(30)]

    def f(xs, h0, *ws):
        def cell(h, x):
            acc = x
            for i, w in enumerate(ws):
                acc = acc * w if i % 2 else acc + w * 0.5
            nh = 0.9 * h + jnp.tanh(acc)
            return nh, nh
        return jax.lax.scan(cell, h0, xs)

    check(f, X, H0, *ws, rtol=1e-4, atol=1e-5)
    modes = _plan_modes(f, X, H0, *ws)
    assert modes, "expected an MSL plan (packing should rescue the bindings)"


def test_packing_with_sliced_weight_window(monkeypatch):
    # A single sliced gate makes the dot read a non-canonical WINDOW of
    # the fused weight; weight normalization materializes just the view
    # (numel 25 of a 50-element buffer here), and run() concatenates
    # that view into the pool. Pool slots must be sized by the VIEW —
    # sizing by the source buffer shifts every later member (silent
    # corruption, found in review).
    from metaljax import msl_scan
    monkeypatch.setattr(msl_scan, "_PACK_TRIGGER", 2)
    Wf = (rng.standard_normal((H, 2 * H)) * 0.3).astype(np.float32)

    def f(xs, h0, w):
        def cell(h, x):
            n = jnp.tanh((h @ w)[..., H:] + x)
            nh = 0.8 * h + 0.2 * n
            return nh, nh
        return jax.lax.scan(cell, h0, xs)

    check(f, X, H0, Wf, rtol=1e-4, atol=1e-5)
    check(jax.value_and_grad(lambda a, b, c: f(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), X, H0, Wf,
          rtol=1e-3, atol=1e-4)


def test_matrix_state_outer_and_matvec():
    # matlstm-family building blocks: a matrix state updated with an
    # outer product of loop-computed vectors, and an in-body matvec of
    # the state (dot_generals with NO invariant operand — the in-lane
    # rewrite must turn them into broadcast-multiplies / register
    # reduces rather than illegal fission accumulators).
    D = 2
    Q = (rng.standard_normal((L, B, D)) * 0.4).astype(np.float32)
    K2 = (rng.standard_normal((L, B, D)) * 0.4).astype(np.float32)
    C0 = (rng.standard_normal((B, D, D)) * 0.3).astype(np.float32)

    def f(qs, ks, c0):
        def cell(c, qk):
            q, k = qk
            nc = 0.9 * c + jnp.einsum('bi,bj->bij', q, k)  # outer accum
            y = jnp.einsum('bij,bj->bi', nc, q)     # state matvec
            out = nc * (1.0 + 0.1 * y[:, :, None])
            return nc, out
        return jax.lax.scan(cell, c0, (qs, ks))
    check(f, Q, K2, C0, rtol=1e-4, atol=1e-5)
    check(jax.value_and_grad(lambda a, b, c: f(a, b, c)[1].sum(),
                             argnums=(0, 1, 2)), Q, K2, C0,
          rtol=1e-3, atol=1e-4)
    assert _plan_modes(f, Q, K2, C0), \
        "matrix-state cell fell off the MSL path"


def test_vmapped_scan_scalar_stacked_output():
    # A per-step output that is ONE SCALAR PER LANE has a shape equal to
    # the lane shape; the emitter used to read that trailing dim as a
    # register width, so every lane broadcast its own value across all
    # lanes' output slots (wrong since v0.2.0, silently).
    d = rng.standard_normal(2).astype(np.float32)
    c0 = rng.standard_normal(4).astype(np.float32)

    def cell(c, a):
        b = jnp.cos(jnp.sum(jnp.sin(a)) + jnp.sum(jnp.cos(c)) + jnp.sum(d))
        return jnp.sin(c * b), b

    for bdim in (0, 1, 2):
        shape = [5, 3]
        shape.insert(bdim, 7)
        xs = rng.standard_normal(shape).astype(np.float32)
        f = jax.vmap(lambda a: jax.lax.scan(cell, jnp.asarray(c0), a),
                     in_axes=bdim)
        check(f, xs)
        check(jax.value_and_grad(lambda a: jax.vmap(
            lambda v: jax.lax.scan(cell, jnp.asarray(c0), v)[1].sum(),
            in_axes=bdim)(a).sum()), xs, rtol=1e-4, atol=1e-5)


def test_partially_unrolled_scan():
    # unroll=k makes one loop body do k steps and concatenate their
    # per-step outputs. Each concat part was padded with the concat TOTAL
    # width instead of its own, so the first part claimed every slot: a
    # scalar part landed in all of them and the sum double-counted it.
    c0 = rng.standard_normal(3).astype(np.float32)
    xs = rng.standard_normal((8, 3)).astype(np.float32)

    def cell(c, x):
        nc = jnp.tanh(c * 0.5 + x)
        return nc, jnp.sum(nc)

    for u in (1, 2, 3, 4, 8):
        check(lambda a, b, u=u: jax.lax.scan(cell, a, b, unroll=u), c0, xs)
        check(jax.value_and_grad(
            lambda a, b, u=u: jax.lax.scan(cell, a, b, unroll=u)[1].sum(),
            argnums=(0, 1)), c0, xs, rtol=1e-4, atol=1e-5)


def test_carry_permutation_is_a_parallel_assignment():
    # Per-iteration state updates are emitted as `st0 = new0; st1 = new1;
    # ...`. A computed update lands in its own temporary first, but an
    # update that IS verbatim another state register (a carry rotation
    # like `[c[-1], *c[:-1]]`) used to read the already-overwritten slot,
    # so every carry collapsed onto one element -- silently, since the
    # AD-transposed loop is a different program and stayed correct
    # (jax api_test's dce_jaxpr_scan_nontrivial_fixedpoint_extensive_output
    # caught it as a *gradient* mismatch). All three emission modes.

    # scalar mode: 4 rotating scalar carries
    def rot_scalars(a, b, c, d):
        def body(cr, _):
            return [cr[-1], *cr[:-1]], cr[-1]
        return jax.lax.scan(body, [a, b, c, d], None, length=4)[1]

    s = [np.float32(v) for v in (1.0, 2.0, 3.0, 4.0)]
    check(rot_scalars, *s)
    assert _plan_modes(rot_scalars, *s) == ["scalar"]

    # scalar mode: two array carries, one a bare move of the other
    def swap_arrays(xs, h0, h1):
        def cell(c, x):
            p, q = c
            return (q, jnp.tanh(p + x)), p
        return jax.lax.scan(cell, (h0, h1), xs)

    H1 = rng.standard_normal((B, H)).astype(np.float32)
    check(swap_arrays, X, H0, H1)
    assert _plan_modes(swap_arrays, X, H0, H1) == ["scalar"]

    # vector / coop modes: same move alongside a matvec cell
    def swap_matvec(xs, h0, h1, w):
        def cell(c, x):
            p, q = c
            nh = jnp.tanh(x + p @ w)
            return (q, nh), nh
        return jax.lax.scan(cell, (h0, h1), xs)

    for width, mode in ((4, "vector"), (F16, "coop")):
        w = (rng.standard_normal((width, width)) * 0.25).astype(np.float32)
        xs = (rng.standard_normal((L, B, width)) * 0.3).astype(np.float32)
        a0 = rng.standard_normal((B, width)).astype(np.float32)
        b0 = rng.standard_normal((B, width)).astype(np.float32)
        check(swap_matvec, xs, a0, b0, w, rtol=1e-4, atol=1e-5)
        assert _plan_modes(swap_matvec, xs, a0, b0, w) == [mode]


def test_mixed_rank_carry_stabilizer_cell():
    # matlstm/mLSTM shape: a matrix carry (B,D,D), a vector carry (B,D)
    # and a per-batch SCALAR carry (B,) (the exponential stabilizer). In
    # vector mode the (B,) carry is register-resident (all B values in
    # every lane), yet the body views it as (B,1)/(B,D)/(B,D,D) — its
    # register axis mapped onto a LANE coordinate. The emitter used to
    # take those uses at face value and write `v{n}[r]` with no `r` in
    # scope: "use of undeclared identifier 'r'" at Metal build time,
    # killing the process mid-training. Such carries must be rejected so
    # the loop falls back to the compiled-graph path.
    D, Bm, Lm = 2, 2, 6
    Q = (rng.standard_normal((Lm, Bm, D)) * 0.4).astype(np.float32)
    K2 = (rng.standard_normal((Lm, Bm, D)) * 0.4).astype(np.float32)
    V = (rng.standard_normal((Lm, Bm, D)) * 0.4).astype(np.float32)
    IG = (rng.standard_normal((Lm, Bm)) * 0.5).astype(np.float32)
    FG = (rng.standard_normal((Lm, Bm)) * 0.5).astype(np.float32)
    C0 = (rng.standard_normal((Bm, D, D)) * 0.3).astype(np.float32)
    N0 = (rng.standard_normal((Bm, D)) * 0.3).astype(np.float32)
    M0 = np.zeros(Bm, np.float32)

    def f(qs, ks, vs, igs, fgs, c0, n0, m0):
        def cell(carry, step):
            c, n, m = carry
            q, k, v, ig, fg = step
            nm = jnp.maximum(fg + m, ig)             # (B,)
            i_ = jnp.exp(ig - nm)
            f_ = jnp.exp(fg + m - nm)
            # in-cell contractions as broadcast-multiply + reduce (how
            # texmo's matlstm lowers): those stay inside the kernel's
            # register lanes, so the plan reaches the emitter
            o = v[:, :, None] * k[:, None, :]
            nc = f_[:, None, None] * c + i_[:, None, None] * o
            nn = f_[:, None] * n + i_[:, None] * k
            num = jnp.sum(nc * q[:, None, :], axis=-1)
            den = jnp.maximum(jnp.abs(jnp.sum(nn * q, axis=-1)), 1.0)
            return (nc, nn, nm), num / den[:, None]
        return jax.lax.scan(cell, (c0, n0, m0), (qs, ks, vs, igs, fgs))

    args = (Q, K2, V, IG, FG, C0, N0, M0)
    check(f, *args, rtol=1e-4, atol=1e-5)
    check(jax.value_and_grad(lambda *a: f(*a)[1].sum(),
                             argnums=tuple(range(len(args)))), *args,
          rtol=1e-3, atol=1e-4)


def test_metal_build_failure_falls_back(monkeypatch):
    # Any generated kernel Metal's compiler rejects must degrade to the
    # compiled-graph path instead of killing the program: engine.execute
    # disables msl_scan for that executable and re-runs it (the program is
    # pure). Runs through the real backend — the recovery lives there.
    monkeypatch.setenv("METALJAX_MSL_FORCE_BUILD_FAIL", "1")

    def f(a, x, h0):
        return affine_scan(a, x, h0)

    metal = jax.devices("metal")[0]
    jf = jax.jit(f)
    with jax.default_device(metal):
        got = jax.tree.leaves(jf(A, X, H0))
        # ...and again on the same executable: the blacklisted plan must
        # stay blacklisted instead of being rebuilt (and re-failing).
        again = jax.tree.leaves(jf(A, X, H0))
    with jax.default_device(jax.devices("cpu")[0]):
        want = jax.tree.leaves(jax.jit(f)(A, X, H0))
    for g, g2, w in zip(got, again, want):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w),
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(g2), np.asarray(w),
                                   rtol=1e-5, atol=1e-6)


def test_register_write_width_guard():
    # Unit-level cover for the invalid-MSL form of the same bug: the
    # register loop variable `r` exists only when the TARGET is wider
    # than one register, so an R>1 value may only be written into an
    # R-wide target ("use of undeclared identifier 'r'" otherwise).
    from metaljax.msl_scan import _Unsupported, _reg_src

    assert _reg_src("v0", 2, 2, "w") == "v0[r]"
    assert _reg_src("v0", 1, 4, "w") == "v0"    # scalar broadcast is fine
    assert _reg_src("v0", 1, 1, "w") == "v0"
    for R, tgt in ((2, 1), (2, 4), (4, 2)):
        with pytest.raises(_Unsupported):
            _reg_src("v0", R, tgt, "w")
