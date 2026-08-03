"""Plan-aware liveness pruning: the eager env drops what the block stops using.

`Interpreter.eager_plan` computes, per block, the last use of every value so
the eager loop can let go of it (notes/eager-memory-2026-08.md). Read off the
literal IR that analysis is WRONG for any block a recognizer rewrites: a
quantized matmul's activation, or an attention's Q/K/V, is last used by an op
the recognizer ABSORBS, and the fused `emit` -- which runs further down the
block, in the root's place -- reads it back afterwards. The first fix was to
turn pruning off on such blocks, which disabled it exactly where it matters
most: a diffusion sampler body is 16.5k ops with an attention in every layer,
and it retained ~5000 live values and ramped +15 GB a sample.

So the plan is an INPUT to the analysis now. Three things are tested here:

* the ANALYSIS -- absorbed ops use nothing, produce nothing and cost nothing;
  a root's uses are what its `emit` reads; and the map is re-derived when the
  plan changes (qmm disables a match whose packing failed) rather than served
  stale, which would prune ops the new plan still executes;

* the completeness of the three `emit_reads` sets, which is the part that can
  only be got right by reading the `emit` implementations. Under
  METALJAX_PRUNE_VERIFY=1 the environment remembers what pruning dropped and
  raises `PrunedValueError` when anything reads it back, so running each
  recognizer's own fixtures under it IS the proof -- with a negative control
  that shows the check has teeth;

* the PEAK, on a synthetic attention stack: flat in the chain length with
  pruning, linear without.
"""

import gc

import numpy as np
import pytest

import mlx.core as mx

import jax
import jax.numpy as jnp

import helpers
from metaljax import Interpreter, interpreter, moe, qmm, sdpa
from metaljax import dtypes as mdt
from metaljax.interpreter import _SKIP, PrunedValueError

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

needs_sdpa = pytest.mark.skipif(not sdpa.ENABLED, reason="METALJAX_SDPA=0")
needs_qmm = pytest.mark.skipif(not qmm.QMM_ENABLED, reason="METALJAX_QMM=0")
needs_moe = pytest.mark.skipif(not moe.ENABLED, reason="METALJAX_MOE=0")
# METALJAX_ENV_PRUNE=0 restores the retain-everything behaviour, so nothing is
# ever dropped and the sentinel environment has nothing to catch.
needs_prune = pytest.mark.skipif(not interpreter._ENV_PRUNE,
                                 reason="METALJAX_ENV_PRUNE=0")

MB = 2 ** 20


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def _plan_of(interp, block=None):
    """(block, rewrite plan, drops, out_bytes) for a block."""
    with interp.context:
        blk = interp._main_block() if block is None else block
        plan = interp._rewrite_plan(blk)
        drops, out_bytes = interp.eager_plan(blk, plan)
    return blk, plan, drops, out_bytes


def _roots(plan):
    """{op index: match} for the plan's rewritten ops."""
    return {i: e[1] for i, e in plan if e is not _SKIP}


def _skipped(plan):
    return {i for i, e in plan if e is _SKIP}


def _reads(plan):
    return {i: e[2] for i, e in plan if e is not _SKIP}


def _live_profile(interp, block, plan, drops, prune=True):
    """Replay the eager loop's bookkeeping: the largest `env` it ever holds.

    Exactly what `run_block` does to `env` -- absorbed ops bind nothing, every
    other op binds its results, and `drops[i]` is applied after op i -- so this
    is the live set the interpreter really carries, counted without running a
    single kernel.
    """
    with interp.context:
        skipped = _skipped(plan)
        live = set(block.arguments)
        tcache, peak_n, peak_b = {}, 0, 0
        for i, op in enumerate(block.operations):
            o = op.operation
            if i not in skipped:
                for r in o.results:
                    live.add(r)
            if prune:
                for v in drops[i]:
                    live.discard(v)
            peak_n = max(peak_n, len(live))
            peak_b = max(peak_b,
                         sum(interpreter.value_bytes(v, tcache) for v in live))
    return peak_n, peak_b


def _literal_users(block, value):
    """Indices of the ops in `block` that take `value` as an operand."""
    out = []
    for i, op in enumerate(block.operations):
        if any(v == value for v in op.operation.operands):
            out.append(i)
    return out


def _cpu():
    return jax.devices("cpu")[0]


def _metal():
    return jax.devices("metal")[0]


def _run_eager(interp, args):
    """The bare Interpreter, which is always the op-by-op path."""
    outs = interp(*[mdt.to_mx(np.asarray(a)) for a in args])
    mx.eval(*outs)
    return [mdt.to_np(o) for o in outs]


def _through_engine(f, args):
    """Compile+execute once through the engine, which runs the packing and
    verification prologues qmm/moe need before their plans go live."""
    from metaljax import engine

    ex = engine.compile_program(helpers.lower_bytes(f, *args), "mlir")
    bufs = []
    for a in jax.tree.leaves(args):
        arr = np.asarray(a)
        bufs.append(engine.MetalBuffer(mdt.to_mx(arr),
                                       engine._NP_TO_ENUM[arr.dtype],
                                       list(arr.shape)))
    outs = engine.execute(ex, bufs)
    mx.eval(*[o.data for o in outs])
    return ex.interpreter, [mdt.to_np(o.data) for o in outs]


def _prune_verify(on=True):
    """METALJAX_PRUNE_VERIFY for the duration of a `with` block."""

    class _Ctx:
        def __enter__(self):
            self.old = interpreter._PRUNE_VERIFY
            interpreter._PRUNE_VERIFY = on

        def __exit__(self, *a):
            interpreter._PRUNE_VERIFY = self.old

    return _Ctx()


# --------------------------------------------------------------------------
# the fixtures
# --------------------------------------------------------------------------

B, H, T, D = 2, 4, 8, 16


def attn_computed(q0, k0, v0, m0):
    """Attention over COMPUTED Q/K/V and a computed mask.

    The point of the `tanh`/`*` prologue: each of Q, K, V and the mask base is
    then a value the block's liveness analysis owns, whose only literal uses
    are inside the chain the recognizer absorbs. Handed straight in as block
    arguments they would never be dropped either way (an argument with no use
    at all is simply never entered in the last-use map), and the test would
    prove nothing.
    """
    q, k, v = jnp.tanh(q0), jnp.tanh(k0), jnp.tanh(v0)
    m = m0 * 2.0
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5) + m
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)


def _attn_args(seed=0):
    rng = np.random.default_rng(seed)
    out = [jnp.asarray(rng.standard_normal((B, H, T, D)), jnp.float32)
           for _ in range(3)]
    out.append(jnp.asarray(rng.standard_normal((B, H, T, T)), jnp.float32))
    return out


def _unpack_nibbles(packed, columns):
    """keras' int4 storage -> [rows, columns] int8 codes (test_qmm's copy)."""
    lo = jnp.left_shift(packed, 4).astype(jnp.int8) / 16
    hi = jnp.right_shift(packed, 4).astype(jnp.int8)
    w = jnp.stack([lo.astype(jnp.int8), hi], axis=-1)
    return jnp.reshape(w, (packed.shape[0], columns))


def dense_int4(packed, scale, zero, g_idx, x, columns):
    """keras `Dense.quantize("int4")`, sub-channel -- one quantized matmul."""
    w = _unpack_nibbles(packed, columns)
    g = g_idx.astype(jnp.int32)
    s = jnp.take(scale, g, axis=0)
    z = jnp.take(zero, g, axis=0)
    wf = (w.astype(x.dtype) - z.astype(x.dtype)) * s
    return x @ wf


def _qmm_fixture(rows=128, cols=64, block=64, seed=7):
    """(f, args): a quantized dense over a COMPUTED activation."""
    rng = np.random.default_rng(seed)
    q = rng.integers(-8, 8, size=(rows, cols)).astype(np.int8)
    ng = rows // block
    scale = ((rng.random((ng, cols)).astype(np.float32) + 0.5) * 0.05)
    zero = rng.integers(-3, 4, size=(ng, cols)).astype(np.int8)
    g_idx = (np.arange(rows) // block).astype(np.float32)
    packed = ((q[:, 0::2] & 0x0F) | (q[:, 1::2] << 4)).astype(np.int8)
    x = (rng.standard_normal((4, rows)) * 0.5).astype(np.float32)

    def f(packed_, scale_, zero_, g_, x_):
        # tanh(x) is the activation the fused dot reads: its only literal use
        # is the dot the rewrite replaces.
        return dense_int4(packed_, scale_, zero_, g_, jnp.tanh(x_), cols)

    args = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(zero),
            jnp.asarray(g_idx), jnp.asarray(x))
    return f, args


def _moe_fixture(E=4, k=2, T_=5, Hd=32, I=16, seed=3):
    """keras-hub's GptOssSparseMoeBlock, as tests/test_moe.py writes it."""
    from test_moe import make_weights, moe_block, _args

    w = make_weights(E, T_, Hd, I, seed=seed)
    return (lambda *a: moe_block(*a, k=k)), [jnp.asarray(x) for x in _args(w)]


# --------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------


def test_empty_plan_leaves_the_analysis_exactly_as_it_was():
    """No rewrite, no change: last use is the last consumer, a result nothing
    reads goes as soon as it exists."""
    interp = Interpreter("""
      func.func @main(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
        %0 = stablehlo.multiply %a, %a : tensor<8x8xf32>
        %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
        %2 = stablehlo.subtract %0, %0 : tensor<8x8xf32>
        return %1 : tensor<8x8xf32>
      }
    """)
    blk, plan, drops, out_bytes = _plan_of(interp)
    assert plan == ()
    with interp.context:
        ops = [o.operation for o in blk.operations]
        mul, add, sub = ops[0], ops[1], ops[2]
        assert drops[0] == (interp._main_block().arguments[0],)
        assert set(drops[1]) == set()          # %0 still feeds the subtract
        assert set(drops[2]) == {mul.results[0], sub.results[0]}
        assert out_bytes == [8 * 8 * 4] * 3 + [0]


def attn_causal(q, k, v):
    """A `select` mask, so the absorbed set includes the softmax's reduces."""
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
    mask = jnp.tril(jnp.ones((T, T), bool))
    lg = jnp.where(mask, lg, jnp.finfo(lg.dtype).min)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)


@needs_sdpa
def test_absorbed_ops_can_carry_regions():
    """Not an assumption that can be asserted away: a softmax's max and sum
    are `stablehlo.reduce`s, each with a body region, and sdpa absorbs both.
    That is why an absorbed op contributes neither operands NOR region
    captures -- an op that never runs reads nothing at all."""
    args = _attn_args()[:3]
    interp = Interpreter(helpers.lower_bytes(attn_causal, *args))
    blk, plan, drops, out_bytes = _plan_of(interp)
    with interp.context:
        ops = [o.operation for o in blk.operations]
        regioned = [i for i, e in plan if e is _SKIP and ops[i].regions]
        assert len(regioned) == 2, [ops[i].name for i in regioned]
        assert all(ops[i].name == "stablehlo.reduce" for i in regioned)
        for i in regioned:
            assert drops[i] == () and out_bytes[i] == 0


@needs_sdpa
def test_absorbed_ops_use_nothing_produce_nothing_and_cost_nothing():
    args = _attn_args()
    interp = Interpreter(helpers.lower_bytes(attn_computed, *args))
    blk, plan, drops, out_bytes = _plan_of(interp)
    skipped = _skipped(plan)
    assert skipped, "the recognizer absorbed nothing"
    with interp.context:
        ops = [o.operation for o in blk.operations]
        for i in skipped:
            # never executed: no drop entry may land on it (a value named
            # there would be retained for the whole block instead)...
            assert drops[i] == (), (i, ops[i].name)
            # ... and it materializes nothing, exactly as
            # ops.control._block_bytes charges absorbed ops nothing.
            assert out_bytes[i] == 0, (i, ops[i].name)
            # its own results are never bound, so nothing may drop them
            for r in ops[i].results:
                assert all(r not in d for d in drops)


@needs_sdpa
def test_a_root_uses_what_emit_reads_not_what_the_op_reads():
    """Q/K/V/mask are last used by an ABSORBED op and read back by `emit`.

    This is the whole bug in one assertion: off the literal IR each of them
    dies inside the chain, so a plan-blind analysis either drops them before
    the fused kernel runs (use-after-free) or -- as the first fix did -- gives
    up and retains the entire block.
    """
    args = _attn_args()
    interp = Interpreter(helpers.lower_bytes(attn_computed, *args))
    blk, plan, drops, _ = _plan_of(interp)
    roots = _roots(plan)
    assert len(roots) == 1
    (i_root, m), = roots.items()
    reads = _reads(plan)[i_root]
    assert set(reads) == {m.q, m.k, m.v, m.mask[1]}

    with interp.context:
        skipped = _skipped(plan)
        for v in reads:
            users = _literal_users(blk, v)
            assert users, "fixture bug: the value has no literal user"
            # every literal consumer is absorbed -- including the last one
            assert all(u in skipped or u == i_root for u in users), users
            assert max(users) < i_root or i_root in users
            # so liveness has to come from the plan, and it ends at the root
            assert v in drops[i_root], v
            assert all(v not in drops[j] for j in range(i_root)), v


@needs_sdpa
def test_pruned_live_set_is_flat_in_the_chain_length():
    """The claim the SD3.5 sampler body needs: the retained set tracks the
    LIVE set, not the number of ops in the block."""
    rng = np.random.default_rng(0)
    args = [jnp.asarray(rng.standard_normal((B, H, T, D)) * 0.3, jnp.float32)
            for _ in range(3)]

    def stack(n):
        def f(q, k, v):
            h = q
            for i in range(n):
                lg = jnp.einsum("bhqd,bhkd->bhqk", h, k) * (D ** -0.5)
                h = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)
                h = jnp.tanh(h * (1.0 + 0.01 * i)) + 0.5
            return h
        return f

    got = {}
    for n in (2, 8):
        interp = Interpreter(helpers.lower_bytes(stack(n), *args))
        blk, plan, drops, _ = _plan_of(interp)
        assert len(_roots(plan)) == n
        got[n] = (_live_profile(interp, blk, plan, drops, prune=True),
                  _live_profile(interp, blk, plan, drops, prune=False),
                  len(list(blk.operations)))

    (p2, r2, ops2), (p8, r8, ops8) = got[2], got[8]
    assert ops8 > 3 * ops2                    # 4x the block
    assert p8 == p2, (p2, p8)                 # ... same live set
    assert r8[0] > 3 * r2[0], (r2, r8)        # ... 4x retained without it
    # `prune=False` is also what the shipped policy did on any block with a
    # rewrite in it, so this ratio is the size of the regression being fixed:
    # >20x the values, measured 643 vs 4 on the 32-layer version.
    assert r8[0] > 20 * p8[0], (p8, r8)


@needs_sdpa
def test_a_changed_plan_re_derives_the_map_instead_of_serving_it_stale():
    """qmm disables a match whose packing failed and `State.rebuild` replaces
    `skip`/`roots` wholesale. A liveness map cached against the previous plan
    would prune ops the new one executes -- the same use-after-free by another
    route -- so the cache is keyed on the plan's IDENTITY."""
    args = _attn_args()
    interp = Interpreter(helpers.lower_bytes(attn_computed, *args))
    fused = _run_eager(interp, args)

    blk, plan, drops, out_bytes = _plan_of(interp)
    assert plan and interp._live_cache[blk][0] is plan
    (i_root, m), = _roots(plan).items()

    # Drop the match, exactly as a failed prologue would.
    with interp.context:
        st = sdpa.analyze(interp)
        st.matches = []
        st.rebuild()
        plan2 = interp._rewrite_plan(blk)
        drops2, out_bytes2 = interp.eager_plan(blk, plan2)
    assert plan2 == ()
    assert plan2 is not plan
    assert interp._live_cache[blk][0] is plan2
    assert drops2 != drops
    assert sum(out_bytes2) > sum(out_bytes)   # the chain is charged again
    # Q now dies where the IR says it does, well before the (literal) root.
    assert m.q not in drops2[i_root]

    literal = _run_eager(interp, args)
    with jax.default_device(_cpu()):
        want = np.asarray(jax.jit(attn_computed)(*args))
    for got in (fused[0], literal[0]):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


@needs_qmm
def test_qmm_root_reads_only_the_activation():
    """The packed weight, scales, biases and permutation live in
    `qmm.State.values`, never in `env`; the dequantized weight the literal dot
    would have read is produced by absorbed ops and never exists."""
    f, args = _qmm_fixture()
    qmm.reset_stats()
    interp, _ = _through_engine(f, args)
    blk, plan, drops, _ = _plan_of(interp)
    roots = _roots(plan)
    assert len(roots) == 1, roots
    (i_root, m), = roots.items()
    assert _reads(plan)[i_root] == (m.lhs,)

    with interp.context:
        users = _literal_users(blk, m.lhs)
        assert users == [i_root]              # only the dot reads it
        assert m.lhs in drops[i_root]
        # the quantized side is absorbed: the dot's other operand is produced
        # by a skipped op, so nothing ever binds it
        skipped = _skipped(plan)
        root_op = list(blk.operations)[i_root].operation
        others = [v for v in root_op.operands if v != m.lhs]
        assert others
        for v in others:
            owner = v.owner.operation
            oi = [i for i, o in enumerate(blk.operations)
                  if o.operation == owner]
            assert oi and oi[0] in skipped, owner.name


@needs_moe
def test_moe_root_reads_the_router_and_the_gathered_values():
    f, args = _moe_fixture()
    interp, _ = _through_engine(f, args)
    blk, plan, drops, _ = _plan_of(interp)
    roots = _roots(plan)
    assert len(roots) == 1, roots
    (i_root, m), = roots.items()
    reads = _reads(plan)[i_root]
    assert set(reads) == set(m.reads) | {m.router.indices, m.router.weights}
    assert m.router.indices in reads and m.router.weights in reads
    with interp.context:
        for v in reads:
            assert v in drops[i_root] or any(
                v in drops[j] for j in range(i_root, len(drops))), v
            assert all(v not in drops[j] for j in range(i_root)), v


# --------------------------------------------------------------------------
# emit_reads completeness (the sentinel environment)
# --------------------------------------------------------------------------


def attn_computed_unmasked(q0, k0, v0):
    """`attn_computed` without the mask: `emit_reads` is then three values."""
    q, k, v = jnp.tanh(q0), jnp.tanh(k0), jnp.tanh(v0)
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)


@needs_sdpa
@pytest.mark.parametrize("masked", [False, True])
def test_emit_reads_complete_for_sdpa(masked):
    f = attn_computed if masked else attn_computed_unmasked
    args = _attn_args()[:4 if masked else 3]
    code = helpers.lower_bytes(f, *args)
    with _prune_verify():
        interp = Interpreter(code)
        got = _run_eager(interp, args)
        with interp.context:
            assert len(sdpa.analyze(interp).matches) == 1
    with jax.default_device(_cpu()):
        want = np.asarray(jax.jit(f)(*args))
    np.testing.assert_allclose(got[0], want, rtol=2e-5, atol=2e-5)


@needs_sdpa
def test_emit_reads_complete_across_every_recognized_spelling():
    """Every attention shape tests/test_sdpa.py recognizes, run under the
    sentinel environment. This is the breadth behind the claim -- the mask
    shapes in particular decide whether `_mask_array`'s base is read."""
    import test_sdpa as ts

    cases = [
        (ts.attn_bhtd, [(B, H, T, D)] * 3),
        (ts.attn_bthd, [(B, T, H, D)] * 3),
        (ts.attn_causal, [(B, H, T, D)] * 3),
        (ts.attn_bias, [(B, H, T, D)] * 3 + [(B, H, T, T)]),
        (ts.attn_gqa, [(B, H, T, D), (B, H // 2, T, D), (B, H // 2, T, D)]),
    ]
    for fn, shapes in cases:
        args = ts._arrays(shapes, jnp.float32, seed=1)
        code = helpers.lower_bytes(fn, *args)
        with _prune_verify():
            interp = Interpreter(code)
            got = _run_eager(interp, args)
            with interp.context:
                assert len(sdpa.analyze(interp).matches) == 1, fn.__name__
        with jax.default_device(_cpu()):
            want = np.asarray(jax.jit(fn)(*args))
        np.testing.assert_allclose(got[0], want, rtol=2e-5, atol=2e-5,
                                   err_msg=fn.__name__)


@needs_qmm
def test_emit_reads_complete_for_qmm():
    f, args = _qmm_fixture()
    interp, _ = _through_engine(f, args)   # packs, so the plan is live
    with _prune_verify():
        got = _run_eager(interp, args)
    with interp.context:
        assert _roots(interp._rewrite_plan(interp._main_block()))
    with jax.default_device(_cpu()):
        want = np.asarray(jax.jit(f)(*args))
    np.testing.assert_allclose(got[0], want, rtol=2e-2, atol=2e-2)


@needs_moe
def test_emit_reads_complete_for_moe():
    f, args = _moe_fixture()
    interp, _ = _through_engine(f, args)   # verifies, so the plan is live
    with _prune_verify():
        got = _run_eager(interp, args)
    with interp.context:
        assert _roots(interp._rewrite_plan(interp._main_block()))
    with jax.default_device(_cpu()):
        want = np.asarray(jax.jit(f)(*args))
    np.testing.assert_allclose(got[0], want, rtol=2e-3, atol=2e-3)


@needs_sdpa
@needs_prune
@pytest.mark.parametrize("drop", [0, 1, 2, 3])
def test_the_completeness_check_has_teeth(drop):
    """Negative control. Remove one entry from `sdpa.emit_reads` and the
    sentinel environment must catch the read -- otherwise the four tests
    above assert nothing."""
    args = _attn_args()
    code = helpers.lower_bytes(attn_computed, *args)
    orig = sdpa.emit_reads

    def broken(m):
        r = list(orig(m))
        del r[drop]
        return tuple(r)

    sdpa.emit_reads = broken
    try:
        with _prune_verify():
            interp = Interpreter(code)
            with pytest.raises(PrunedValueError):
                _run_eager(interp, args)
    finally:
        sdpa.emit_reads = orig


def test_the_sentinel_environment_is_off_by_default():
    """It puts a Python `__getitem__` on the interpreter's hottest lookup."""
    import os
    assert interpreter._PRUNE_VERIFY == (
        os.environ.get("METALJAX_PRUNE_VERIFY", "") == "1")


# --------------------------------------------------------------------------
# the peak
# --------------------------------------------------------------------------


_PB, _PH, _PT, _PD = 2, 8, 512, 64


def _peak_stack(n, prune, flush_mb, args, cache={}):
    """Peak MLX memory for an n-layer attention stack, run op by op.

    Measured net of the baseline: this shares a process with the rest of the
    suite, so what is alive when the run starts is not zero.
    """
    def f(q, k, v):
        h = q
        for i in range(n):
            lg = jnp.einsum("bhqd,bhkd->bhqk", h, k) * (_PD ** -0.5)
            h = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)
            h = jnp.tanh(h * (1.0 + 0.01 * i)) + 0.5
        return h

    if n not in cache:
        cache[n] = helpers.lower_bytes(f, *args)
    old, oldf = interpreter._ENV_PRUNE, interpreter.FLUSH_BYTES
    interpreter._ENV_PRUNE = prune
    interpreter.FLUSH_BYTES = int(flush_mb * MB)
    try:
        interp = Interpreter(cache[n])
        mxargs = [mdt.to_mx(np.asarray(a)) for a in args]
        mx.eval(*mxargs)
        gc.collect()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        outs = interp(*mxargs)
        mx.eval(*outs)
        peak = mx.get_peak_memory()
        out = mdt.to_np(outs[0])
        del outs, interp, mxargs
        gc.collect()
        mx.clear_cache()
        return (peak - base) / MB, out
    finally:
        interpreter._ENV_PRUNE, interpreter.FLUSH_BYTES = old, oldf


@needs_sdpa
def test_peak_is_flat_in_the_chain_length_and_the_answer_does_not_change():
    """The measurement behind the analysis, on a block big enough to see.

    Pruning and the byte-denominated flush only work TOGETHER
    (notes/eager-memory-2026-08.md): pruning decides what a flush has to
    settle, and without a flush the lazy graph still grows. With both, the
    peak is flat in the chain length; with the flush alone it is linear.
    Measured here (B2/H8/T512/D64, 32 MB budget), MB net of baseline:

        layers            2      4      8
        pruned          6.0    6.0    6.0
        retained       24.0   48.0   96.0

    and at the shipped 1024 MB budget (which never fires at this size):
    6/10/18 pruned versus the same 24/48/96.
    """
    rng = np.random.default_rng(0)
    args = [jnp.asarray(rng.standard_normal((_PB, _PH, _PT, _PD)) * 0.3,
                        jnp.float32) for _ in range(3)]

    small, big = 2, 8
    # Every layer carries a recognized attention, so this is precisely the
    # block class the shipped policy refused to prune.
    probe = Interpreter(helpers.lower_bytes(
        lambda q, k, v: jnp.einsum(
            "bhqk,bhkd->bhqd",
            jax.nn.softmax(jnp.einsum("bhqd,bhkd->bhqk", q, k)
                           * (_PD ** -0.5), -1), v), *args))
    with probe.context:
        assert len(sdpa.analyze(probe).matches) == 1
    del probe

    p_s, out_s = _peak_stack(small, True, 32, args)
    p_b, out_b = _peak_stack(big, True, 32, args)
    r_s, ref_s = _peak_stack(small, False, 32, args)
    r_b, ref_b = _peak_stack(big, False, 32, args)

    # Pruning changes when a buffer is released, never what is computed.
    np.testing.assert_array_equal(out_s, ref_s)
    np.testing.assert_array_equal(out_b, ref_b)

    # 4x the chain: retained tracks it, pruned does not.
    assert r_b > 3 * r_s, (r_s, r_b)
    assert p_b < 1.5 * p_s, (p_s, p_b)
    assert p_b < r_b / 4, (p_b, r_b)
