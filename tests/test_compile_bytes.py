"""The compile-decision byte gate.

`_block_cost` bounds how many OPS one `mx.compile` trace may hold, which
bounds the Metal live-buffer count; it says nothing about how much those
buffers hold. MLX retains a compiled graph's intermediates through the first
call, so a program comfortably inside the op budget can still make the tracer
materialize tens of gigabytes -- which is how big-model load, warmup and
conversion programs (few ops, enormous tensors) took this machine down.
`_block_bytes` estimates that, and every compile decision consults it.

Two things are tested here:

* the ESTIMATOR, on hand-built modules where the right answer can be counted
  by hand -- including the two places a naive result-size sum would lie: a
  splat constant (one value with a whole-tensor type, broadcast from a
  one-element buffer, so charging it its nominal size would invent gigabytes
  that never exist) and ops a recognizer absorbs (never executed at all);
* the GATE, on a program whose traced footprint is real: it must decline to
  compile, still produce CPU-identical results through the eager path, and
  peak far below what the compiled path peaks at.
"""

import numpy as np
import pytest

import mlx.core as mx

import jax
import jax.numpy as jnp

import helpers
from metaljax import Interpreter, moe, qmm, sdpa
from metaljax import dtypes as mdt
from metaljax.interpreter import BYTES_UNKNOWN, op_bytes, value_bytes
from metaljax.ops import control

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

MB = 2 ** 20


def _bytes(module_text):
    """The estimator's answer for a module, and the interpreter it read.

    Interpreter-direct throughout this section on purpose: the estimator is
    a question about parsed IR, and none of the tests below it EXECUTE
    anything -- so there is no engine path to take, and nothing that could
    show up in (or be missing from) the native-tape census. The tests that
    do run programs use `_run_program`, which goes through the engine.
    """
    interp = Interpreter(module_text)
    with interp.context:
        return control._block_bytes(interp, interp._main_block()), interp


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


def test_straight_line_block_sums_its_results():
    nb, _ = _bytes("""
      func.func @main(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
        %0 = stablehlo.multiply %a, %a : tensor<8x8xf32>
        %1 = stablehlo.add %0, %a : tensor<8x8xf32>
        return %1 : tensor<8x8xf32>
      }
    """)
    assert nb == 2 * 8 * 8 * 4


def test_splat_constant_costs_one_element_not_its_shape():
    """A splat is one value with a whole-tensor type; ops.elementwise
    broadcasts it from a one-element buffer. Charging 1 MB for
    `dense<1.0> : tensor<512x512xf32>` would gate programs that touch no
    real data -- jax lowers a single random.normal with 23 of these."""
    text = """
      func.func @main(%a: tensor<512x512xf32>) -> tensor<512x512xf32> {
        %c = stablehlo.constant dense<1.000000e+00> : tensor<512x512xf32>
        %0 = stablehlo.add %a, %c : tensor<512x512xf32>
        return %0 : tensor<512x512xf32>
      }
    """
    nb, interp = _bytes(text)
    assert nb == 4 + 512 * 512 * 4  # the constant is 4 bytes, the add is 1 MB
    with interp.context:
        ops = list(interp._main_block().operations)
        const = ops[0].operation
        assert op_bytes(const) == 4
        # ... which is exactly where op_bytes and value_bytes disagree.
        assert value_bytes(const.results[0]) == 512 * 512 * 4


def test_non_splat_constant_is_charged_in_full():
    nb, _ = _bytes("""
      func.func @main() -> tensor<2x2xf32> {
        %c = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
        return %c : tensor<2x2xf32>
      }
    """)
    assert nb == 2 * 2 * 4


def test_calls_are_recursed_into():
    """A func.call is charged its callee's whole block, once per call site --
    the same convention as _block_cost, since a trace inlines both."""
    nb, _ = _bytes("""
      func.func private @twice(%x: tensor<16xf32>) -> tensor<16xf32> {
        %0 = stablehlo.add %x, %x : tensor<16xf32>
        return %0 : tensor<16xf32>
      }
      func.func @main(%a: tensor<16xf32>) -> tensor<16xf32> {
        %0 = func.call @twice(%a) : (tensor<16xf32>) -> tensor<16xf32>
        %1 = func.call @twice(%0) : (tensor<16xf32>) -> tensor<16xf32>
        return %1 : tensor<16xf32>
      }
    """)
    # two call results (64 each) + two callee bodies (64 each)
    assert nb == 4 * 16 * 4


COUNTED = """
  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %z = stablehlo.constant dense<0> : tensor<i32>
    %n = stablehlo.constant dense<10> : tensor<i32>
    %r:2 = "stablehlo.while"(%z, %a) ({
      ^cond(%i: tensor<i32>, %c: tensor<4xf32>):
        %p = stablehlo.compare LT, %i, %n : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %p : tensor<i1>
    }, {
      ^body(%i: tensor<i32>, %c: tensor<4xf32>):
        %one = stablehlo.constant dense<1> : tensor<i32>
        %ni = stablehlo.add %i, %one : tensor<i32>
        %nc = stablehlo.add %c, %c : tensor<4xf32>
        stablehlo.return %ni, %nc : tensor<i32>, tensor<4xf32>
    }) : (tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
    return %r#1 : tensor<4xf32>
  }
"""

DYNAMIC = COUNTED.replace(
    "%p = stablehlo.compare LT, %i, %n :",
    "%p = stablehlo.compare LT, %i, %b :").replace(
    "func.func @main(%a: tensor<4xf32>)",
    "func.func @main(%a: tensor<4xf32>, %b: tensor<i32>)")


def _body_bytes():
    # constant 1 (rank 0) + increment + the f32 add
    return 4 + 4 + 4 * 4


def test_counted_loop_multiplies_the_body_by_its_trip():
    from metaljax import msl_scan
    old = msl_scan.ENABLED
    try:
        msl_scan.ENABLED = False
        nb, _ = _bytes(COUNTED)
    finally:
        msl_scan.ENABLED = old
    outer = 4 + 4 + 4 + 4 * 4  # two constants + the while's own results
    assert nb == outer + 10 * _body_bytes()


def test_generated_kernel_loop_is_charged_only_its_outputs():
    """A loop msl_scan turns into ONE persistent kernel keeps its state in
    registers: no per-iteration buffers exist, so charging trip x body would
    gate exactly the loops that need no gating."""
    from metaljax import msl_scan
    if not msl_scan.ENABLED:
        pytest.skip("METALJAX_MSL=0")
    interp = Interpreter(COUNTED)
    with interp.context:
        blk = interp._main_block()
        wh = list(blk.operations)[2].operation
        assert control._msl_plan_for(interp, wh) is not None
        assert control._block_bytes(interp, blk) == 4 + 4 + 4 + 4 * 4


def test_unknown_trip_is_pessimistic():
    """An unknown bound counts as 1024 iterations -- _block_cost's own
    convention, and for the same reason: a dynamic-bound loop once counted
    as ONE body, so its enclosing block passed the budget and mx.compile
    tried to trace the whole thing."""
    nb, _ = _bytes(DYNAMIC)
    outer = 4 + 4 + 4 + 4 * 4
    assert nb == outer + 1024 * _body_bytes()


def test_absorbed_attention_is_not_charged():
    """Ops a recognizer absorbs never execute, so they never materialize.
    The literal chain builds the [B,H,T,K] logits five times over; the fused
    kernel produces one output. (One skip set covers both recognizers, so
    this exercises the quantized-matmul rewrite's path too -- that one needs
    the engine's packing prologue to be active, which tests/test_qmm.py
    drives.)"""
    if not sdpa.ENABLED:
        pytest.skip("METALJAX_SDPA=0")
    B, H, T, D = 2, 4, 8, 16

    def attn(q, k, v):
        lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)

    rng = np.random.default_rng(0)
    args = [jnp.asarray(rng.standard_normal((B, H, T, D)), jnp.float32)] * 3
    code = helpers.lower_bytes(attn, *args)
    old = sdpa.ENABLED
    try:
        sdpa.ENABLED = True
        fused, _ = _bytes(code)
        sdpa.ENABLED = False
        literal, _ = _bytes(code)
    finally:
        sdpa.ENABLED = old
    assert fused < literal / 2, (fused, literal)


# --------------------------------------------------------------------------
# what a fused ROOT materializes
# --------------------------------------------------------------------------
#
# Charging absorbed ops nothing is right; charging their ROOT only its own
# declared result is not, and on an expert-gather rewrite it is not close.
# The dense region a MoE match absorbs is essentially the whole layer, so the
# estimate collapsed onto the [tokens, hidden] output and reported ~nothing:
# measured on gpt-oss-20b's decode program, 156 of 783 ops are absorbed and
# they carry 99.97% of the block's declared result bytes (28.9 of 28.9 GB) --
# estimate 10.4 MB against a first-call trace that really peaked at 55.8 MB,
# and 71.1 MB against 107.1 MB once the prompt was 64 tokens rather than 11.
# What actually runs is the gathered emission, in PAIR space (P = T*K rows),
# and that is what `moe.emit_bytes` charges: 34.1 MB and 214.0 MB for those
# same two programs, back on the safe side of measurement.


def _moe_program(T, E=8, K=2, H=64, I=32):
    """(estimate, literal bytes, recognizer state) for a MoE block. Static:
    the recognizer pass and the estimator read IR, nothing runs (see
    `_bytes`)."""
    import test_moe as tm

    w = tm.make_weights(E, T, H, I, seed=T)
    code = helpers.lower_bytes(lambda *a: tm.moe_block(*a, k=K), *tm._args(w))
    interp = Interpreter(code)
    with interp.context:
        st = qmm.analyze(interp)   # the recognizer pass, IR only
        blk = interp._main_block()
        skip = st.skip if st.active else frozenset()
        literal = sum(op_bytes(o.operation) for o in blk.operations
                      if not (len(o.operation.results)
                              and o.operation.results[0] in skip))
        return control._block_bytes(interp, blk), literal, st


@pytest.mark.skipif(not moe.ENABLED, reason="METALJAX_MOE=0")
def test_absorbed_moe_root_is_charged_what_its_emission_builds():
    nb, literal, st = _moe_program(T=8)
    assert len(st.moe) == 1 and st.active
    assert st.skip, "nothing absorbed: the recognizer did not fire"
    # `literal` is what this block was worth before emit_bytes existed: every
    # unabsorbed op's result, and nothing at all for the rewrite. (It is a
    # few bytes under _block_bytes' own walk, which also recurses into the
    # reduce/top_k region bodies -- hence the size-scaled bound below rather
    # than a bare inequality.)
    P, I = 8 * 2, 32
    assert nb - literal >= P * 2 * I * 4, (nb, literal)


@pytest.mark.skipif(not moe.ENABLED, reason="METALJAX_MOE=0")
def test_moe_estimate_grows_with_the_pair_space():
    """The absorbed region's cost is per (token, selected expert) pair, so
    the estimate has to move with the token count. Charging the root its own
    result made it nearly flat -- which is how a prefill 60x the size of a
    decode step looked identical to the gate."""
    small, lit_s, _ = _moe_program(T=4)
    big, lit_b, _ = _moe_program(T=64)
    grew = (big - lit_b) / max(small - lit_s, 1)
    assert grew > 8, (small, lit_s, big, lit_b)


# --------------------------------------------------------------------------
# outputs no operation produces
# --------------------------------------------------------------------------


_PASSTHROUGH = """
  func.func @main(%a: tensor<256x256xf32>) -> tensor<256x256xf32> {
    return %a : tensor<256x256xf32>
  }
"""


def test_passthrough_output_is_charged_at_program_level():
    """`func.return %arg0` has no op to charge, but XLA's no-alias contract
    makes engine.execute copy the result -- keras' streaming loader lowers
    every variable move to exactly this, and each one really did copy 1.1 GB
    of embedding while the estimate read 0.0 MB."""
    interp = Interpreter(_PASSTHROUGH)
    with interp.context:
        blk = interp._main_block()
        assert control._block_bytes(interp, blk) == 0   # no ops to charge
        assert control.program_bytes(interp, blk) == 256 * 256 * 4


# jax does not lower a while_loop's closed-over constants as region captures:
# they become carries the body returns UNCHANGED (see qmm._hoist). A decode
# loop's weights arrive that way, so a body's pass-throughs can be the whole
# parameter set.
_CARRIED = """
  func.func @main(%a: tensor<4xf32>, %w: tensor<1024x1024xf32>)
      -> tensor<4xf32> {
    %z = stablehlo.constant dense<0> : tensor<i32>
    %n = stablehlo.constant dense<10> : tensor<i32>
    %r:3 = "stablehlo.while"(%z, %a, %w) ({
      ^cond(%i: tensor<i32>, %c: tensor<4xf32>, %cw: tensor<1024x1024xf32>):
        %p = stablehlo.compare LT, %i, %n : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %p : tensor<i1>
    }, {
      ^body(%i: tensor<i32>, %c: tensor<4xf32>, %cw: tensor<1024x1024xf32>):
        %one = stablehlo.constant dense<1> : tensor<i32>
        %ni = stablehlo.add %i, %one : tensor<i32>
        %nc = stablehlo.add %c, %c : tensor<4xf32>
        stablehlo.return %ni, %nc, %cw
            : tensor<i32>, tensor<4xf32>, tensor<1024x1024xf32>
    }) : (tensor<i32>, tensor<4xf32>, tensor<1024x1024xf32>)
       -> (tensor<i32>, tensor<4xf32>, tensor<1024x1024xf32>)
    return %r#1 : tensor<4xf32>
  }
"""


def test_passthrough_charge_stays_out_of_loop_bodies():
    """A while body returns its carries, and jax turns a loop's closed-over
    constants into exactly that. Charging those x trip would put 40 GB on a
    ten-iteration loop over one 4 MB weight -- it would inflate every scan in
    texmo and move the chunk sizing the command-buffer lottery is pinned to.
    So the pass-through term is a whole-PROGRAM term only."""
    from metaljax import msl_scan
    old = msl_scan.ENABLED
    try:
        msl_scan.ENABLED = False
        interp = Interpreter(_CARRIED)
        with interp.context:
            blk = interp._main_block()
            body = list(blk.operations)[2].operation.regions[1].blocks[0]
            # the carried weight IS a pass-through of the body's block args...
            assert control._passthrough_bytes(body) == 1024 * 1024 * 4
            # ...and the estimate the loop sites use does not see it.
            whole = control._block_bytes(interp, blk)
            assert whole < 10 * 1024 * 1024 * 4, whole
            # main returns none of its own arguments, so nothing is added.
            assert control.program_bytes(interp, blk) == whole
    finally:
        msl_scan.ENABLED = old


# --------------------------------------------------------------------------
# values the estimator cannot size
# --------------------------------------------------------------------------


def test_tuple_type_is_sized_through_its_leaves():
    """A type the sizing code does not handle used to return 0 from a bare
    `except`, which is indistinguishable from "carries no data"."""
    interp = Interpreter(_PASSTHROUGH)
    with interp.context:
        from jaxlib.mlir import ir
        from metaljax.interpreter import _type_bytes

        with ir.Location.unknown():
            f32 = ir.F32Type.get()
            t = ir.TupleType.get_tuple([
                ir.RankedTensorType.get([4, 4], f32),
                ir.RankedTensorType.get([8], f32),
            ])
        assert _type_bytes(t) == (16 + 8) * 4


def test_unsized_type_is_counted_not_silently_zero():
    interp = Interpreter(_PASSTHROUGH)
    with interp.context:
        from jaxlib.mlir import ir
        from metaljax.interpreter import _type_bytes

        before = BYTES_UNKNOWN["unsized"]
        assert _type_bytes(ir.Type.parse("tensor<*xf32>")) == 0
        assert BYTES_UNKNOWN["unsized"] == before + 1


def test_token_and_dynamic_zero_are_not_counted_as_unsized():
    """The two zeros that are correct: a token really carries nothing, and a
    symbolic dimension is a deliberate abstention (guessing it would invent
    numbers on every shape-poly export)."""
    interp = Interpreter(_PASSTHROUGH)
    with interp.context:
        from jaxlib.mlir import ir
        from metaljax.interpreter import _type_bytes

        before = dict(BYTES_UNKNOWN)
        assert _type_bytes(ir.Type.parse("!stablehlo.token")) == 0
        assert BYTES_UNKNOWN["unsized"] == before["unsized"]
        assert _type_bytes(ir.Type.parse("tensor<?x4xf32>")) == 0
        assert BYTES_UNKNOWN["unsized"] == before["unsized"]
        assert BYTES_UNKNOWN["dynamic"] == before["dynamic"] + 1


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def _budget(mb):
    """Set METALJAX_COMPILE_BYTES_MB for the duration of a `with` block."""

    class _Ctx:
        def __enter__(self):
            self.old = (control._COMPILE_BYTES_MB, control._COMPILE_BYTES)
            control._COMPILE_BYTES_MB = mb
            control._COMPILE_BYTES = max(mb, 0) * MB

        def __exit__(self, *a):
            control._COMPILE_BYTES_MB, control._COMPILE_BYTES = self.old

    return _Ctx()


def _run_program(f, args):
    """Through the engine, so the test sees the executable's own decision."""
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
    return ex, [mdt.to_np(o.data) for o in outs]


# The shape of program this gate exists for: FEW ops, BIG tensors, no loop
# for the op budget to catch. A jitted parameter init is the canonical one --
# jax lowers `random.normal` into ~200 full-shape intermediates, so 64 MB of
# weights is ~15 GB of traced traffic at 365 ops (2% of the op budget). This
# is the program class that took the machine down twice: maxtext's
# `create_sharded_state`, keras' per-variable initializers, checkpoint
# conversion (notes/eager-memory-2026-08.md).
_SEED = jnp.asarray(np.array([0, 1], np.uint32))


def _init(n):
    def f(s):
        return jax.random.normal(jax.random.wrap_key_data(s), (n, n))

    return f


def _peak_of(f, args, budget_mb):
    mx.clear_cache()
    mx.reset_peak_memory()
    with _budget(budget_mb):
        ex, outs = _run_program(f, args)
    return ex, outs, mx.get_peak_memory()


def test_small_program_does_not_gate():
    """Nothing this size may be pushed off the compiled path: the default
    budget sits ~1.5x above the largest program metaljax compiles today."""
    ex, _, _ = _peak_of(_init(512), (_SEED,), 65536)
    assert ex._can_compile is True
    assert not ex.interpreter._bytes_gated


def test_big_bytes_program_gates_stays_correct_and_is_bounded():
    """Over the budget the program runs op by op, where 03ae8bb's last-use
    pruning and byte-denominated flush bound it -- and the answer does not
    change.

    The SCALING is what is asserted, not a fixed number. Measured one
    program per process, this initializer at 4/16/64/256 MB of output peaks
    at 0.18/0.72/2.87/10.0 GB compiled -- it tracks the program -- while
    refused it peaks at 0.80/1.19/3.25 GB, held down by the eager flush
    budget. (The 256 MB point is the one notes/eager-memory-2026-08.md had
    to leave at "9.75 GB compiled either way".)"""
    small, big = _init(2048), _init(4096)
    ex_c_s, _, peak_c_s = _peak_of(small, (_SEED,), 0)     # gate off
    ex_c_b, outs_c, peak_c_b = _peak_of(big, (_SEED,), 0)
    ex_g_s, _, peak_g_s = _peak_of(small, (_SEED,), 1024)  # gate on
    ex_g_b, outs_g, peak_g_b = _peak_of(big, (_SEED,), 1024)

    assert ex_c_s._can_compile and ex_c_b._can_compile
    assert not ex_g_s._can_compile and not ex_g_b._can_compile
    assert ex_g_b.interpreter._bytes_gated  # the block is marked

    with jax.default_device(jax.devices("cpu")[0]):
        want = np.asarray(jax.jit(big)(_SEED))
    # The gate changes which path evaluates the graph, not what the graph
    # computes. (Against CPU it is a few ULP: the threefry BITS are exact
    # either way, the erf_inv that turns them into normals is not.)
    np.testing.assert_allclose(outs_g[0], want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outs_c[0], want, rtol=1e-5, atol=1e-5)

    # 4x the data: the compiled peak follows it (measured 4.0x), the gated
    # one does not (1.5x). Bounds are loose -- this shares a process with
    # the rest of the suite, so the floor under both is whatever else is
    # alive; the ORDER of growth is the claim.
    assert peak_c_b > 2.5 * peak_c_s, (peak_c_s, peak_c_b)
    assert peak_g_b < 2.0 * peak_g_s, (peak_g_s, peak_g_b)
    assert peak_g_b < peak_c_b, (peak_g_b, peak_c_b)


def test_gate_off_by_env_restores_compilation():
    """METALJAX_COMPILE_BYTES_MB=0 is the escape hatch."""
    gated, _, _ = _peak_of(_init(4096), (_SEED,), 1024)
    free, _, _ = _peak_of(_init(4096), (_SEED,), 0)
    assert gated._can_compile is False
    assert free._can_compile is True


# 2048 wide on purpose: at 1024 the same loop becomes one msl kernel (which
# is right -- a persistent kernel keeps its state in registers and has no
# wave to bound), and this test would then be about msl, not about the gate.
_LOOP_W, _LOOP_N = 2048, 4


def _loop(x, w):
    return jax.lax.fori_loop(0, _LOOP_N, lambda _, h: jnp.tanh(h @ w) * 1.5, x)


def _loop_args():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((_LOOP_W, _LOOP_W)) * 0.05,
                        jnp.float32),
            jnp.asarray(rng.standard_normal((_LOOP_W, _LOOP_W)) * 0.02,
                        jnp.float32))


def _run_loop(budget_mb):
    """Bare interpreter, deliberately: BODY-level chunking is only reachable
    when the enclosing main is not itself one compiled trace, and at the
    budgets this test needs (0 = gate off) the engine compiles the whole
    main -- which would leave `_body_cache` empty and make the assertions
    below unobservable. `test_byte_budget_through_the_engine` runs the same
    program the ordinary way, so the ops still reach engine.execute (and the
    native tape) from this file.

    Returns the result, the interpreter, and the chunk sizes it compiled.
    """
    args = _loop_args()
    interp = Interpreter(helpers.lower_bytes(_loop, *args))
    with _budget(budget_mb):
        outs = interp(*[mdt.to_mx(np.asarray(a)) for a in jax.tree.leaves(args)])
        mx.eval(*outs)
    repeats = sorted({k[2] for k, v in interp._body_cache.items() if v[2]})
    return mdt.to_np(outs[0]), interp, repeats


def test_byte_budget_through_the_engine():
    """The same loop, executed as a real program: the budget decides how the
    engine runs it (one traced main, a chunked body, op by op) and none of
    the three may change the answer."""
    args = _loop_args()
    with jax.default_device(jax.devices("cpu")[0]):
        want = np.asarray(jax.jit(_loop)(*args))

    compiled = []
    for budget_mb in (0, 256, 64):
        with _budget(budget_mb):
            ex, outs = helpers.execute_module(
                helpers.lower_bytes(_loop, *args), jax.tree.leaves(args))
            got = helpers.from_buffer(outs[0])
        compiled.append(bool(ex._can_compile))
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5,
                                   err_msg=f"budget {budget_mb} MB")
    assert compiled == [True, False, False], compiled


def test_byte_budget_sizes_the_chunked_replay_and_can_refuse_it():
    """Body-level gating end to end: the same loop chunked freely, chunked
    to what the budget allows, and (below one iteration) not compiled at
    all. All three must agree with CPU."""
    args = _loop_args()
    with jax.default_device(jax.devices("cpu")[0]):
        want = np.asarray(jax.jit(_loop)(*args))

    # ~80 MB per iteration, four iterations.
    free, i_free, k_free = _run_loop(0)
    part, i_part, k_part = _run_loop(256)
    none, i_none, k_none = _run_loop(64)

    assert k_free == [_LOOP_N]        # one trace for the whole loop
    assert k_part and max(k_part) < _LOOP_N   # byte-limited chunk
    assert not k_none                 # nothing compiled: op by op
    assert not i_free._bytes_gated
    assert i_none._bytes_gated
    for got in (free, part, none):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def test_gate_scales_a_chunked_replay_instead_of_refusing_it():
    """A body small enough per iteration keeps its compiled chunk; the byte
    budget only decides how many iterations ride in one trace."""
    interp = Interpreter(COUNTED)
    with interp.context:
        body = list(interp._main_block().operations)[2].operation.regions[1].blocks[0]
        per = control._block_bytes(interp, body)
        with _budget(1):  # 1 MB / 24 B per iteration
            assert control._bytes_chunks(interp, body) == MB // per
            assert control._bytes_ok(interp, body, 1, "t") is True
        with _budget(0):
            assert control._bytes_chunks(interp, body) > 1 << 20
