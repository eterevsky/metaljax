"""Expert-gather recognizer: dense MoE dispatch -> mx.gather_mm/gather_qmm.

The layer graph below is keras-hub's `GptOssSparseMoeBlock` written in plain
`jax.numpy` (keras is not installed in this venv): a top-k router whose
one-hot-weighted sum produces a `[tokens, experts]` score tensor with exact
zeros off the selection, expert weights applied to EVERY token by two
einsums, and a final `sum_e score * expert_out`. That is what XLA lowers
densely and what this recognizer turns into `K` gathered experts per token.

Three layers of testing:

* the rewrite end to end THROUGH THE REAL BACKEND (analysis and the eager
  verification prologue live in engine.execute, so `helpers.check`'s bare
  Interpreter never exercises them), against jax-CPU;
* the MXFP4 variant, where the expert dots were already packed by
  `metaljax.qmm` and the gather reads those packs (`gather_qmm`);
* the negative cases -- everything that must FAIL CLOSED and leave the dense
  region alone: expert outputs consumed twice, routing weights that are not
  a top-k selection, a non-sum root.

`0 * inf` is covered explicitly: the dense path multiplies an unrouted
expert's output by an exact zero and turns an overflow into NaN, while the
gathered path never evaluates that expert. That divergence is intended (it
is towards correctness) and is asserted here so it cannot change silently.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from metaljax import moe, qmm

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

needs_moe = pytest.mark.skipif(not moe.ENABLED, reason="METALJAX_MOE=0")


def _metal():
    return jax.devices("metal")[0]


def _cpu():
    return jax.devices("cpu")[0]


def _run(f, args, device):
    with jax.default_device(device):
        moved = [jax.device_put(a, device) for a in args]
        out = jax.jit(f)(*moved)
        return jax.tree.map(lambda a: np.asarray(a).astype(np.float32), out)


# --------------------------------------------------------------------------
# the layer, exactly as keras-hub emits it
# --------------------------------------------------------------------------


def router_scores(x, wr, br, k):
    """[tokens, experts], zero off the top-k -- keras' GptOssTopKRouter."""
    logits = x @ wr + br
    vals, idx = jax.lax.top_k(logits, k)
    w = jax.nn.softmax(vals.astype(jnp.float32), axis=-1).astype(vals.dtype)
    hot = jax.nn.one_hot(idx, logits.shape[-1], dtype=w.dtype)
    return jnp.sum(hot * w[..., None], axis=1)


def experts(x, wgu, bgu, wd, bd, limit=7.0):
    """[experts, tokens, hidden] -- every expert on every token."""
    gu = jnp.einsum("th,ehm->etm", x, wgu) + bgu[:, None, :]
    gate = jnp.clip(gu[..., ::2], -1e9, limit)
    up = jnp.clip(gu[..., 1::2], -limit, limit)
    glu = gate * jax.nn.sigmoid(gate * 1.702)
    return jnp.einsum("etm,emh->eth", (up + 1) * glu, wd) + bd[:, None, :]


def moe_block(x, wr, br, wgu, bgu, wd, bd, k):
    s = router_scores(x, wr, br, k)
    y = experts(x, wgu, bgu, wd, bd)
    return jnp.sum(y * s.T[..., None], axis=0)


def make_weights(E, T, H, I, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)

    def n(*shape):
        return (rng.standard_normal(shape) / np.sqrt(H)).astype(dtype)

    return dict(
        x=n(T, H), wr=n(H, E), br=n(E),
        wgu=n(E, H, 2 * I), bgu=n(E, 2 * I),
        wd=n(E, I, H), bd=n(E, H))


def _args(w):
    return [w["x"], w["wr"], w["br"], w["wgu"], w["bgu"], w["wd"], w["bd"]]


# --------------------------------------------------------------------------
# float experts -> gather_mm
# --------------------------------------------------------------------------


@needs_moe
@pytest.mark.parametrize("E,k", [(4, 1), (4, 2), (8, 2), (8, 4), (32, 4)])
@pytest.mark.parametrize("T", [1, 5])
def test_float_experts_gathered(E, k, T):
    w = make_weights(E, T, 64, 32, seed=E * 10 + k)
    f = lambda *a: moe_block(*a, k=k)
    moe.reset_stats()
    got = _run(f, _args(w), _metal())
    st = moe.stats()
    assert st["recognized"] == 1, st
    assert st["verified"] == 1, st
    assert st["gather_mm"] >= 2, st        # both expert projections
    assert st["gather_qmm"] == 0, st
    want = _run(f, _args(w), _cpu())
    assert got.shape == want.shape == (T, 64)
    # f32 reduction-order difference only: the gathered sum is over k terms,
    # the dense one over E (of which E-k are exact zeros).
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)


@needs_moe
def test_gathered_matches_dense_on_the_same_backend():
    """metal-gathered vs metal-dense (METALJAX_MOE=0): same graph, one knob."""
    w = make_weights(8, 4, 64, 32, seed=7)
    f = lambda *a: moe_block(*a, k=2)
    moe.reset_stats()
    got = _run(f, _args(w), _metal())
    assert moe.stats()["recognized"] == 1
    try:
        moe.ENABLED = False
        dense = _run(f, _args(w), _metal())
    finally:
        moe.ENABLED = True
    np.testing.assert_allclose(got, dense, rtol=2e-5, atol=2e-6)


@needs_moe
@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
def test_dtypes(dtype):
    import ml_dtypes
    dt = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.float32
    w = make_weights(8, 3, 64, 32, seed=3, dtype=dt)
    f = lambda *a: moe_block(*a, k=2)
    moe.reset_stats()
    got = _run(f, _args(w), _metal())
    assert moe.stats()["recognized"] == 1
    want = _run(f, _args(w), _cpu())
    tol = 4e-2 if dt is not np.float32 else 2e-5
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


@needs_moe
def test_shared_expert_outside_the_sum_still_fuses():
    """`moe(x) + shared(x)`: the add is outside the root, so it is untouched."""
    w = make_weights(8, 4, 64, 32, seed=11)
    ws = (np.random.default_rng(1).standard_normal((64, 64)) / 8).astype(
        np.float32)

    def f(x, wr, br, wgu, bgu, wd, bd, ws_):
        return moe_block(x, wr, br, wgu, bgu, wd, bd, k=2) + jnp.tanh(x @ ws_)

    moe.reset_stats()
    got = _run(f, _args(w) + [ws], _metal())
    assert moe.stats()["recognized"] == 1, moe.stats()
    want = _run(f, _args(w) + [ws], _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)


@needs_moe
def test_router_scores_also_returned():
    """keras' block returns the scores too; the dense score tensor stays."""
    w = make_weights(8, 4, 64, 32, seed=13)

    def f(x, wr, br, wgu, bgu, wd, bd):
        s = router_scores(x, wr, br, 2)
        y = experts(x, wgu, bgu, wd, bd)
        return jnp.sum(y * s.T[..., None], axis=0), s

    moe.reset_stats()
    got = _run(f, _args(w), _metal())
    assert moe.stats()["recognized"] == 1, moe.stats()
    want = _run(f, _args(w), _cpu())
    for g, wn in zip(got, want):
        np.testing.assert_allclose(g, wn, rtol=2e-5, atol=2e-6)


# --------------------------------------------------------------------------
# fail-closed cases
# --------------------------------------------------------------------------


def _expect_no_fusion(f, args):
    moe.reset_stats()
    got = _run(f, args, _metal())
    assert moe.stats()["recognized"] == 0, moe.stats()
    want = _run(f, args, _cpu())
    for g, wn in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        np.testing.assert_allclose(g, wn, rtol=2e-5, atol=2e-6)


@needs_moe
def test_expert_outputs_used_twice_does_not_fuse():
    """An aux head on the dense expert outputs: the region is not private."""
    w = make_weights(8, 4, 64, 32, seed=17)

    def f(x, wr, br, wgu, bgu, wd, bd):
        s = router_scores(x, wr, br, 2)
        y = experts(x, wgu, bgu, wd, bd)
        return jnp.sum(y * s.T[..., None], axis=0), jnp.sum(y * y)

    _expect_no_fusion(f, _args(w))


@needs_moe
def test_dense_softmax_router_does_not_fuse():
    """No top-k: every expert is really routed, nothing may be skipped."""
    w = make_weights(8, 4, 64, 32, seed=19)

    def f(x, wr, br, wgu, bgu, wd, bd):
        s = jax.nn.softmax(x @ wr + br, axis=-1)
        y = experts(x, wgu, bgu, wd, bd)
        return jnp.sum(y * s.T[..., None], axis=0)

    _expect_no_fusion(f, _args(w))


@needs_moe
def test_capacity_style_mask_does_not_fuse():
    """Scores masked by a threshold rather than scattered from a top-k."""
    w = make_weights(8, 4, 64, 32, seed=23)

    def f(x, wr, br, wgu, bgu, wd, bd):
        p = jax.nn.softmax(x @ wr + br, axis=-1)
        s = jnp.where(p > 0.2, p, 0.0)
        y = experts(x, wgu, bgu, wd, bd)
        return jnp.sum(y * s.T[..., None], axis=0)

    _expect_no_fusion(f, _args(w))


@needs_moe
def test_max_over_experts_does_not_fuse():
    """The root must be a SUM; a max over experts is a different operation."""
    w = make_weights(8, 4, 64, 32, seed=29)

    def f(x, wr, br, wgu, bgu, wd, bd):
        s = router_scores(x, wr, br, 2)
        y = experts(x, wgu, bgu, wd, bd)
        return jnp.max(y * s.T[..., None], axis=0)

    _expect_no_fusion(f, _args(w))


@needs_moe
def test_ordinary_model_is_untouched():
    """No false positives: a plain stack that sums over a leading axis.

    (The whole-model gate is the real evidence -- 104 texmo training
    programs, zero recognized -- but this keeps a cheap version in the
    suite.)"""
    rng = np.random.default_rng(41)
    x = (rng.standard_normal((4, 3, 64)) / 8).astype(np.float32)
    w1 = (rng.standard_normal((64, 128)) / 8).astype(np.float32)
    w2 = (rng.standard_normal((128, 64)) / 8).astype(np.float32)
    s = (rng.standard_normal((4, 3)) / 8).astype(np.float32)

    def f(x_, w1_, w2_, s_):
        h = jax.nn.gelu(x_ @ w1_) @ w2_
        return jnp.sum(h * s_[..., None], axis=0)

    _expect_no_fusion(f, [x, w1, w2, s])


@needs_moe
def test_disabled_by_env_flag(monkeypatch):
    w = make_weights(8, 3, 64, 32, seed=31)
    f = lambda *a: moe_block(*a, k=2)
    moe.reset_stats()
    try:
        moe.ENABLED = False
        got = _run(f, _args(w), _metal())
    finally:
        moe.ENABLED = True
    assert moe.stats()["recognized"] == 0
    want = _run(f, _args(w), _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)


# --------------------------------------------------------------------------
# the 0 * inf corner
# --------------------------------------------------------------------------


@needs_moe
def test_unrouted_overflow_is_not_propagated():
    """The one intended divergence, pinned.

    An expert whose weights overflow produces inf; the dense path multiplies
    it by an EXACT zero routing weight and gets NaN in a token that never
    selected it. The gathered path never evaluates that expert, so the token
    keeps its correct value. jax-CPU agrees with the dense path -- this test
    asserts metal deliberately does not.
    """
    E, T, H, I = 4, 2, 64, 32
    w = make_weights(E, T, H, I, seed=37)
    # Expert 3's output is not finite (an overflow upstream, or an inf that
    # reached the checkpoint); its router logit is pushed far down so that
    # no token ever selects it.
    w["bd"][3] = np.inf
    w["br"][3] = -1e4
    f = lambda *a: moe_block(*a, k=2)
    moe.reset_stats()
    got = _run(f, _args(w), _metal())
    assert moe.stats()["recognized"] == 1, moe.stats()
    want = _run(f, _args(w), _cpu())
    assert np.isnan(want).any(), "the dense reference should carry the NaN"
    assert np.isfinite(got).all(), "the gathered path should stay finite"
    # And the finite values are the ones the dense path would have had if
    # `0 * inf` were 0: expert 3 contributes nothing to any token.
    w2 = dict(w)
    w2["bd"] = w["bd"].copy()
    w2["bd"][3] = 0.0
    ref = _run(f, _args(w2), _cpu())
    np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-6)


# --------------------------------------------------------------------------
# MXFP4 experts -> gather_qmm (the gpt-oss shape)
# --------------------------------------------------------------------------

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def _tables(dtype):
    table = np.ldexp(np.ones(256), np.arange(256) - 127)
    table[255] = np.nan
    return jnp.asarray(E2M1, dtype=dtype), jnp.asarray(table.astype(np.float32))


def mxfp4_weight(blocks, scale_bytes, k, dtype):
    """The in-graph dequantization a packed loader emits (see qmm)."""
    vtable, stable = _tables(dtype)
    lead = tuple(blocks.shape[:-1])
    lo = jnp.bitwise_and(blocks, jnp.uint8(0x0F))
    hi = jnp.right_shift(blocks, jnp.uint8(4))
    nib = jnp.reshape(jnp.stack([lo, hi], axis=-1), lead + (k,))
    vals = jnp.take(vtable, nib.astype(jnp.int32), axis=0)
    scale = jnp.take(stable, scale_bytes.astype(jnp.int32), axis=0)
    w = (jnp.reshape(vals, lead + (k // 32, 32)) * scale[..., None].astype(dtype))
    return jnp.reshape(w, lead + (k,))


def make_mxfp4(shape_out, k, E, seed):
    """([E, out, k/2] codes, [E, out, k/32] E8M0 bytes, dequantized weight)."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(E, shape_out, k)).astype(np.uint8)
    exps = rng.integers(120, 128, size=(E, shape_out, k // 32)).astype(np.uint8)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    vals = E2M1[codes]
    scale = np.exp2(exps.astype(np.float32) - 127.0)
    ref = (vals.reshape(E, shape_out, k // 32, 32)
           * scale[..., None]).reshape(E, shape_out, k)
    return packed, exps, ref.astype(np.float32)


def moe_block_mxfp4(x, wr, br, gu_c, gu_s, d_c, d_s, bgu, bd, H, I, k):
    """gpt-oss' block: `[E, out, in]` packed weights, contraction last."""
    dt = x.dtype
    wgu = mxfp4_weight(gu_c, gu_s, H, dt)          # [E, 2I, H]
    gu = jnp.einsum("th,emh->etm", x, wgu) + bgu[:, None, :]
    gate = jnp.clip(gu[..., ::2], -1e9, 7.0)
    up = jnp.clip(gu[..., 1::2], -7.0, 7.0)
    act = (up + 1) * (gate * jax.nn.sigmoid(gate * 1.702))
    wd = mxfp4_weight(d_c, d_s, I, dt)             # [E, H, I]
    y = jnp.einsum("etm,ehm->eth", act, wd) + bd[:, None, :]
    s = router_scores(x, wr, br, k)
    return jnp.sum(y * s.T[..., None], axis=0)


@needs_moe
@pytest.mark.parametrize("E,k", [(4, 2), (8, 4)])
def test_mxfp4_experts_gathered(E, k):
    T, H, I = 3, 64, 32
    rng = np.random.default_rng(E)
    gu_c, gu_s, gu_ref = make_mxfp4(2 * I, H, E, seed=E)
    d_c, d_s, d_ref = make_mxfp4(H, I, E, seed=E + 1)
    x = (rng.standard_normal((T, H)) / 8).astype(np.float32)
    wr = (rng.standard_normal((H, E)) / 8).astype(np.float32)
    br = rng.standard_normal(E).astype(np.float32)
    bgu = (rng.standard_normal((E, 2 * I)) / 8).astype(np.float32)
    bd = (rng.standard_normal((E, H)) / 8).astype(np.float32)
    args = [x, wr, br, gu_c, gu_s, d_c, d_s, bgu, bd]
    f = lambda *a: moe_block_mxfp4(*a, H=H, I=I, k=k)

    moe.reset_stats()
    qmm.reset_stats()
    got = _run(f, args, _metal())
    st, qst = moe.stats(), qmm.stats()
    assert st["recognized"] == 1, st
    if qmm.QMM_ENABLED:
        assert st["gather_qmm"] >= 2, st      # both projections read packs
        assert st["gather_mm"] == 0, st
        assert qst["mxfp4"] == 2, qst         # both were packed by qmm
    else:
        # Nothing packed the weights, so the gather reads the dense
        # reconstruction -- correct, just not a saving.
        assert st["gather_mm"] >= 2, st

    # Reference: the exactly dequantized weights, dense, on CPU.
    def ref(x_, wr_, br_, bgu_, bd_):
        gu = jnp.einsum("th,emh->etm", x_, jnp.asarray(gu_ref)) + bgu_[:, None, :]
        gate = jnp.clip(gu[..., ::2], -1e9, 7.0)
        up = jnp.clip(gu[..., 1::2], -7.0, 7.0)
        act = (up + 1) * (gate * jax.nn.sigmoid(gate * 1.702))
        y = jnp.einsum("etm,ehm->eth", act, jnp.asarray(d_ref)) + bd_[:, None, :]
        s = router_scores(x_, wr_, br_, k)
        return jnp.sum(y * s.T[..., None], axis=0)

    want = _run(ref, [x, wr, br, bgu, bd], _cpu())
    np.testing.assert_allclose(got, want, rtol=3e-5, atol=3e-6)


@needs_moe
def test_offgrid_weights_fall_back_to_dense_and_stay_correct():
    """qmm cannot pack these, so the gather must not claim their packs."""
    T, H, I, E, k = 3, 64, 32, 4, 2
    rng = np.random.default_rng(101)
    w = make_weights(E, T, H, I, seed=101)
    # A float MoE whose contraction is not a multiple of 32 rejects mxfp4
    # packing; the gather itself is unaffected (it is a gather_mm then).
    f = lambda *a: moe_block(*a, k=k)
    moe.reset_stats()
    got = _run(f, _args(w), _metal())
    assert moe.stats()["gather_mm"] >= 2
    want = _run(f, _args(w), _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)


# --------------------------------------------------------------------------
# the dead sweep
# --------------------------------------------------------------------------


def _fixpoint_sweep(block, skip, protect, root_key):
    """The pre-worklist `_dead_sweep`, verbatim, as the reference.

    Kept here rather than in the module: it is what the worklist has to
    agree with, and agreeing with a rescan-to-fixpoint is the whole
    correctness argument for not rescanning.
    """
    changed = True
    while changed:
        changed = False
        for op in block.operations:
            o = op.operation
            k = moe._okey(o)
            if (k is None or k in skip or k == root_key
                    or o.name in moe._NEVER_SWEEP):
                continue
            if any(r in protect for r in o.results):
                continue
            uses = list(moe._users(o))
            if not uses:
                continue
            if all(moe._okey(u) in skip or moe._is_key(u, root_key)
                   for u in uses):
                skip.add(k)
                changed = True


@needs_moe
def test_dead_sweep_worklist_agrees_with_the_fixpoint(monkeypatch):
    """Same skip set, op for op, on the real MoE block.

    The worklist only re-tests the ops that define a newly-skipped op's
    operands; that is sound because nothing else can have its
    every-use-is-skipped test turn true. This asserts it.
    """
    seen = []
    orig = moe._dead_sweep

    def spy(block, skip, protect, root_key):
        before = set(skip)
        orig(block, skip, protect, root_key)
        ref = set(before)
        _fixpoint_sweep(block, ref, protect, root_key)
        seen.append((len(before), len(skip)))
        assert skip == ref, (skip - ref, ref - skip)

    monkeypatch.setattr(moe, "_dead_sweep", spy)
    w = make_weights(4, 5, 64, 32, seed=55)
    got = _run(lambda *a: moe_block(*a, k=2), _args(w), _metal())
    assert seen, "the recognizer never reached the sweep"
    # ...and the sweep is not vacuous: it really adds the dense router work.
    assert any(after > before for before, after in seen), seen
    want = _run(lambda *a: moe_block(*a, k=2), _args(w), _cpu())
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
