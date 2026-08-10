"""Stage 2 M3: control flow, compiled tapes, and the runtime disciplines.

Same differential contract as tests/test_native_tape.py, whose harness this
file reuses: every case runs the SAME executable on both engines and compares
output BYTES. What is new here is that the native engine now has an
mx::compile of its own, so a case can be run compiled on BOTH sides
(`check(..., compiled=True)`) as well as eagerly on both.

Which of the two is the fair reference is not a matter of taste:

* EAGER vs EAGER is the default, and it is bit-exact. Both engines dispatch
  the same MLX kernels in the same order.
* COMPILED vs COMPILED is bit-exact too, and for the same reason -- the two
  traces build the same graph, so MLX fuses and bakes identically.
* COMPILED vs EAGER is NOT: MLX inlines rank-0 constants into fused kernels
  as %.7g literals, which is 1 ULP on most of them (CLAUDE.md item 20).
  `stablehlo.cbrt` shows it -- its 1/3 exponent survives the round trip in
  one path and not the other. So a compiled tape is compared against the
  compiled Python engine, never against the eager one.

The loops here run with msl_scan OFF on both engines. That is not a
workaround: a counted loop with an elementwise body compiles to ONE
generated Metal kernel (src/metaljax/msl_scan.py), which since M5b both
engines run instead of the loop (tests/test_native_msl.py), and a
differential test of the loop machinery has to be about the loop machinery.
"""
import contextlib

import numpy as np
import pytest

from test_native_tape import (  # noqa: F401  (imports gate the native build)
    _buffers,
    _f32,
    _mod,
    _native_engine,
    _run,
    check,
    engine,
    fresh_outputs,
    native,
)

pytestmark = pytest.mark.skipif(
    not hasattr(native, "Program"),
    reason="native extension predates the M2 tape")


@contextlib.contextmanager
def _no_msl():
    from metaljax import msl_scan
    saved = msl_scan.ENABLED
    msl_scan.ENABLED = False
    try:
        yield
    finally:
        msl_scan.ENABLED = saved


@contextlib.contextmanager
def _patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def _counted_mod(n, body, start=0):
    """A counted loop of the shape jax lowers scan/fori_loop to.

    The carry is seeded with a COMPUTED value rather than the argument
    itself: a loop that may forward its initial carry to an output declines
    (see the aliasing tests below), and that is not what most of these cases
    are about.
    """
    t = "tensor<4xf32>"
    return _mod([("x", t)], [t], f"""
    %ci = stablehlo.constant dense<{start}> : tensor<i32>
    %cn = stablehlo.constant dense<{n}> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %init = stablehlo.add %x, %x : {t}
    %r:2 = stablehlo.while(%i = %ci, %v = %init) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %cn : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
{body}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")


# A body msl_scan would happily turn into one kernel, and the tape has to
# interpret instead. Bounded (tanh) so a long trip stays finite.
_STEP = """       %h = stablehlo.multiply %v, %v : tensor<4xf32>
       %nv = stablehlo.tanh %h : tensor<4xf32>"""


def _x():
    return (_f32((4,)) * 0.25).astype(np.float32)


# --------------------------------------------------------------------------
# while: counted loops
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 5, 17])
def test_while_counted_static_bound(n):
    with _no_msl():
        check(_counted_mod(n, _STEP), [_x()])


def test_while_counted_start_offset():
    # A non-zero initial counter: the trip count is bound - start.
    with _no_msl():
        check(_counted_mod(9, _STEP, start=4), [_x()])


def test_while_zero_trip_returns_the_initial_carry():
    with _no_msl():
        check(_counted_mod(0, _STEP), [_x()])


@pytest.mark.parametrize("n", [63, 64, 65, 129])
@pytest.mark.parametrize("chunked", [True, False])
def test_while_flush_cadence_lengths(n, chunked):
    """Trip counts that straddle the loop's sync points.

    The eager loop flushes every `period` iterations (single-step path) or
    every `sync_every` chunks (chunked replay), and both end with a partial
    window. Lengths either side of those boundaries are where a mis-ported
    cadence shows up: as an unevaluated graph if the remainder handling is
    off, and -- the reason the cadences are pinned at all -- as a WRONG
    answer if a sync point lands where MLX's command-buffer split bites
    (notes/mlx-command-buffer-split.md).
    """
    from metaljax.ops import control
    with _no_msl():
        if chunked:
            check(_counted_mod(n, _STEP), [_x()])
        else:
            # A zero chunk-cost ceiling bans chunking, which hands the loop
            # to the single-step path and its own `period` cadence.
            with _patched(control, "_CHUNK_MAX_COST", 0):
                check(_counted_mod(n, _STEP), [_x()])


def test_while_chunked_replay_remainder():
    # 100 iterations = six chunks of 16 plus a remainder of 4, which the
    # chunked path runs through the single-step body.
    with _no_msl():
        ex = check(_counted_mod(100, _STEP), [_x()])
    assert ex._native_prog is not False


def test_while_bound_carried_in_the_loop_state():
    """The lbfgs shape: the bound rides in the carry, forwarded unchanged."""
    t = "tensor<4xf32>"
    mod = _mod([("x", t), ("n", "tensor<i32>")], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %init = stablehlo.add %x, %x : {t}
    %r:3 = stablehlo.while(%i = %ci, %m = %n, %v = %init) : tensor<i32>, tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %m : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %nv = stablehlo.tanh %v : {t}
       stablehlo.return %ni, %m, %nv : tensor<i32>, tensor<i32>, {t}
     }}
    return %r#2 : {t}""")
    with _no_msl():
        check(mod, [_x(), np.int32(6)])


def test_while_bound_captured_from_the_enclosing_scope():
    t = "tensor<4xf32>"
    mod = _mod([("x", t), ("n", "tensor<i32>")], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %init = stablehlo.add %x, %x : {t}
    %r:2 = stablehlo.while(%i = %ci, %v = %init) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %n : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %nv = stablehlo.tanh %v : {t}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
    with _no_msl():
        check(mod, [_x(), np.int32(7)])


def test_while_body_captures_a_value():
    t = "tensor<4xf32>"
    mod = _mod([("x", t), ("w", t)], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %cn = stablehlo.constant dense<6> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %init = stablehlo.add %x, %x : {t}
    %r:2 = stablehlo.while(%i = %ci, %v = %init) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %cn : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %h = stablehlo.multiply %v, %w : {t}
       %nv = stablehlo.tanh %h : {t}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
    with _no_msl():
        check(mod, [_x(), _x()])


def test_while_nested():
    """An inner loop inside an outer one, reading the outer carry.

    The inner counter is seeded by a constant declared INSIDE the body,
    which is how jax emits nested scans -- and is load-bearing: a counter
    captured from an enclosing scope becomes a tracer when the outer body
    is compiled, and reading it on the host is then an error (see the note
    in the module docstring of this file's sibling finding).
    """
    t = "tensor<4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %c3 = stablehlo.constant dense<3> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %init = stablehlo.add %x, %x : {t}
    %r:2 = stablehlo.while(%i = %ci, %v = %init) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %c3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %zero = stablehlo.constant dense<0> : tensor<i32>
       %two = stablehlo.constant dense<2> : tensor<i32>
       %one = stablehlo.constant dense<1> : tensor<i32>
       %s:2 = stablehlo.while(%j = %zero, %u = %v) : tensor<i32>, {t}
        cond {{
          %q = stablehlo.compare LT, %j, %two : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %q : tensor<i1>
        }} do {{
          %nj = stablehlo.add %j, %one : tensor<i32>
          %nu = stablehlo.tanh %u : {t}
          stablehlo.return %nj, %nu : tensor<i32>, {t}
        }}
       %nv = stablehlo.multiply %s#1, %v : {t}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
    with _no_msl():
        check(mod, [_x()])


# --------------------------------------------------------------------------
# while: dynamic loops
# --------------------------------------------------------------------------


def test_while_dynamic_bound():
    """A trip count that is data: the cond runs every iteration."""
    t = "tensor<i32>"
    mod = _mod([("x", t)], [t], f"""
    %lim = stablehlo.constant dense<1000> : {t}
    %c1 = stablehlo.constant dense<1> : {t}
    %seed = stablehlo.add %x, %c1 : {t}
    %r = stablehlo.while(%v = %seed) : {t}
     cond {{
       %p = stablehlo.compare LT, %v, %lim : ({t}, {t}) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %d = stablehlo.multiply %v, %v : {t}
       %nv = stablehlo.add %d, %c1 : {t}
       stablehlo.return %nv : {t}
     }}
    return %r : {t}""")
    with _no_msl():
        check(mod, [np.int32(2)])


def test_while_dynamic_never_runs():
    """The cond is false on entry: the carries come straight back out."""
    t = "tensor<i32>"
    mod = _mod([("x", t)], [t], f"""
    %lim = stablehlo.constant dense<0> : {t}
    %c1 = stablehlo.constant dense<1> : {t}
    %seed = stablehlo.add %x, %c1 : {t}
    %r = stablehlo.while(%v = %seed) : {t}
     cond {{
       %p = stablehlo.compare LT, %v, %lim : ({t}, {t}) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %nv = stablehlo.add %v, %c1 : {t}
       stablehlo.return %nv : {t}
     }}
    return %r : {t}""")
    with _no_msl():
        check(mod, [np.int32(5)])


# --------------------------------------------------------------------------
# while: the aliasing a loop can smuggle into an output
# --------------------------------------------------------------------------


def test_while_carry_forwarding_an_argument_is_copied():
    """A zero-trip loop returns its initial carry -- which here IS main's
    argument, so the output would alias an input across calls. The Python
    engine catches that by id() at the end of execute; nanobind's fresh
    wrappers make it invisible, so the tape marks the output and C++
    copies."""
    t = "tensor<4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %cn = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %r:2 = stablehlo.while(%i = %ci, %v = %x) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %cn : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %nv = stablehlo.tanh %v : {t}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
    with _no_msl():
        check(mod, [_x()])
        fresh_outputs(mod, [_x()], [0], args_too=[0])


def test_while_carry_that_may_forward_a_constant_is_copied():
    """The same for a constant, which the Program holds for the life of the
    executable. A data-dependent trip count cannot rule the pass-through
    out, so the possibility alone is the copy."""
    t = "tensor<4xf32>"
    mod = _mod([("x", t), ("n", "tensor<i32>")], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %k = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : {t}
    %r:2 = stablehlo.while(%i = %ci, %v = %k) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %n : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %nv = stablehlo.multiply %v, %v : {t}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
    with _no_msl():
        check(mod, [_x(), np.int32(0)])
        # Zero iterations: the output IS the constant, and two calls must
        # not hand out the same buffer.
        fresh_outputs(mod, [_x(), np.int32(0)], [0])


def test_while_with_a_statically_known_trip_may_return_its_carry():
    """...and the counterpart: a loop that certainly RUNS produces its
    carry from the body, so the same output is not an alias at all. Getting
    this wrong declines every scan over an argument."""
    with _no_msl():
        t = "tensor<4xf32>"
        mod = _mod([("x", t)], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %cn = stablehlo.constant dense<3> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %r:2 = stablehlo.while(%i = %ci, %v = %x) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %cn : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %nv = stablehlo.tanh %v : {t}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
        check(mod, [_x()])


def test_while_with_a_real_msl_plan_takes_the_kernel():
    """Why every other loop test here turns msl off: this loop really does
    get a plan, and since M5b the tape takes it (tests/test_native_msl.py
    is where the kernels themselves are pinned). What belongs HERE is the
    other half of that entry — it still carries the interpreted loop, which
    is what a rejected kernel falls back to.
    """
    from metaljax.ops import control
    with _native_engine():
        ex = engine.compile_program(_counted_mod(8, _STEP), "mlir")
        with ex.interpreter.context:
            wop = next(o.operation for o in ex.interpreter._main_block()
                       .operations if o.operation.name == "stablehlo.while")
            assert control._msl_plan_for(ex.interpreter, wop) is not None
        prog = ex.native_program()
        assert prog is not None
        ops = native.opcodes()
        hist = prog.op_histogram()
        assert hist.get(ops["metaljax.msl_scan"], 0) == 1
        assert hist.get(ops["stablehlo.while"], 0) == 0
        # The fallback regions are on the tape: the body's ops and the
        # cond's compare are counted through them.
        assert hist.get(ops["stablehlo.tanh"], 0) == 1
        assert hist.get(ops["stablehlo.compare"], 0) == 1


# --------------------------------------------------------------------------
# if / case
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pred", [True, False])
def test_if_branches(pred):
    t = "tensor<4xf32>"
    mod = _mod([("x", t), ("p", "tensor<i1>")], [t], f"""
    %c = stablehlo.constant dense<2.000000e+00> : {t}
    %0 = "stablehlo.if"(%p) ({{
      %a = stablehlo.multiply %x, %c : {t}
      stablehlo.return %a : {t}
    }}, {{
      %b = stablehlo.tanh %x : {t}
      stablehlo.return %b : {t}
    }}) : (tensor<i1>) -> {t}
    return %0 : {t}""")
    check(mod, [_x(), np.array(pred)])


@pytest.mark.parametrize("index", [-1, 0, 1, 2, 5])
def test_case_index_is_clamped(index):
    """An out-of-range case index picks the last branch (ops/control._case);
    both engines take the same branch for the same reason."""
    t = "tensor<4xf32>"
    mod = _mod([("x", t), ("i", "tensor<i32>")], [t], f"""
    %0 = "stablehlo.case"(%i) ({{
      %a = stablehlo.negate %x : {t}
      stablehlo.return %a : {t}
    }}, {{
      %b = stablehlo.tanh %x : {t}
      stablehlo.return %b : {t}
    }}, {{
      %c = stablehlo.abs %x : {t}
      stablehlo.return %c : {t}
    }}) : (tensor<i32>) -> {t}
    return %0 : {t}""")
    check(mod, [_x(), np.int32(index)])


@pytest.mark.parametrize("pred", [True, False])
def test_if_multiple_results_and_captures(pred):
    t = "tensor<4xf32>"
    mod = _mod([("x", t), ("y", t), ("p", "tensor<i1>")], [t, t], f"""
    %s = stablehlo.add %x, %y : {t}
    %0:2 = "stablehlo.if"(%p) ({{
      %a = stablehlo.multiply %s, %x : {t}
      stablehlo.return %a, %s : {t}, {t}
    }}, {{
      %b = stablehlo.subtract %s, %y : {t}
      %c = stablehlo.tanh %x : {t}
      stablehlo.return %b, %c : {t}, {t}
    }}) : (tensor<i1>) -> ({t}, {t})
    return %0#0, %0#1 : {t}, {t}""")
    check(mod, [_x(), _x(), np.array(pred)])


def test_if_branch_forwarding_an_argument_is_copied():
    """One branch returns main's argument untouched: the result may BE that
    array, and which branch ran is data -- so the output is copied whichever
    branch actually runs."""
    t = "tensor<4xf32>"
    mod = _mod([("x", t), ("p", "tensor<i1>")], [t], f"""
    %0 = "stablehlo.if"(%p) ({{
      %a = stablehlo.tanh %x : {t}
      stablehlo.return %a : {t}
    }}, {{
      stablehlo.return %x : {t}
    }}) : (tensor<i1>) -> {t}
    return %0 : {t}""")
    check(mod, [_x(), np.array(False)])
    fresh_outputs(mod, [_x(), np.array(False)], [0], args_too=[0])


def test_if_inside_a_while():
    t = "tensor<4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %cn = stablehlo.constant dense<5> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %init = stablehlo.add %x, %x : {t}
    %r:2 = stablehlo.while(%i = %ci, %v = %init) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %cn : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %q = stablehlo.compare LT, %i, %c2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
       %nv = "stablehlo.if"(%q) ({{
         %a = stablehlo.multiply %v, %v : {t}
         stablehlo.return %a : {t}
       }}, {{
         %b = stablehlo.tanh %v : {t}
         stablehlo.return %b : {t}
       }}) : (tensor<i1>) -> {t}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
    with _no_msl():
        check(mod, [_x()])


# --------------------------------------------------------------------------
# pass-through ops
# --------------------------------------------------------------------------


def test_alias_ops_are_lowered_by_aliasing_slots():
    t = "tensor<3x4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %0 = stablehlo.tanh %x : {t}
    %1 = "stablehlo.optimization_barrier"(%0) : ({t}) -> {t}
    %2 = stablehlo.multiply %1, %1 : {t}
    return %2 : {t}""")
    ex = check(mod, [_f32()])
    # Aliased, not executed: two ops on the tape, not three.
    assert ex._native_prog.num_ops == 2


def test_alias_op_forwarding_an_argument_is_copied():
    t = "tensor<3x4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %0 = "stablehlo.optimization_barrier"(%x) : ({t}) -> {t}
    return %0 : {t}""")
    check(mod, [_f32()])
    fresh_outputs(mod, [_f32()], [0], args_too=[0])


# --------------------------------------------------------------------------
# the compiled tape
# --------------------------------------------------------------------------


def test_compiled_tape_matches_the_compiled_python_engine():
    """Both engines trace the same ops through mx.compile, so the fused
    kernels are the same kernels and the bytes must agree exactly."""
    t = "tensor<8x8xf32>"
    mod = _mod([("a", t), ("b", t)], [t], f"""
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : ({t}, {t}) -> {t}
    %1 = stablehlo.tanh %0 : {t}
    %2 = stablehlo.add %1, %a : {t}
    %3 = stablehlo.dot_general %2, %b, contracting_dims = [1] x [0] : ({t}, {t}) -> {t}
    return %3 : {t}""")
    ex = check(mod, [_f32((8, 8)), _f32((8, 8))], compiled=True)
    assert ex._can_compile is True


def test_compiled_tape_unrolls_a_small_counted_loop():
    with _no_msl():
        before = native.stats()["unrolls"]
        ex = check(_counted_mod(4, _STEP), [_x()], compiled=True)
        assert native.stats()["unrolls"] > before
    assert ex._can_compile is True


def test_compiled_tape_anchors_equal_constant_outputs():
    """MLX's compiler bakes an output no input feeds into a constants table
    KEYED BY VALUE, and two equal-valued ones collide (unordered_map::at,
    CLAUDE.md item 11). ops.control._underived_outputs finds them; the tape
    anchors them with where(x == x, out, out), which is bitwise exact."""
    t = "tensor<4xf32>"
    st = "tensor<f32>"
    mod = _mod([("x", t)], [t, st, st], f"""
    %c = stablehlo.constant dense<3.000000e-01> : {st}
    %d = stablehlo.multiply %c, %c : {st}
    %e = stablehlo.multiply %c, %c : {st}
    %0 = stablehlo.tanh %x : {t}
    return %0, %d, %e : {t}, {st}, {st}""")
    with _native_engine():
        ex = engine.compile_program(mod, "mlir")
        outs = engine.execute(ex, _buffers([_x()]))
        assert ex._native_prog is not False
        assert ex._can_compile is True
        got = [engine.to_host(o) for o in outs]
    assert np.frombuffer(got[1], np.float32)[0] == np.float32(0.3) ** 2
    assert got[1] == got[2]


def test_compiled_tape_falls_back_when_a_trace_is_refused():
    """A counted loop the cost model calls traceable but whose trip count is
    past what one trace may hold: the unroll refuses, and the program
    finishes on the native EAGER path -- without handing itself back to
    Python, which is what a run-time failure used to mean."""
    with _no_msl():
        mod = _counted_mod(100, _STEP)
        x = _x()
        with _native_engine():
            before = engine.NATIVE_STATS["run_failures"]
            ref, _ = _run(mod, [x], False)
            ex = engine.compile_program(mod, "mlir")
            outs = engine.execute(ex, _buffers([x]))
            got = [engine.to_host(o) for o in outs]
            assert ex._native_prog is not False
            assert engine.NATIVE_STATS["run_failures"] == before
            assert ex._native_prog.compiled_dropped
    assert got == ref


def test_compiled_tape_is_reused_across_calls():
    t = "tensor<4x4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.tanh %a : {t}
    %1 = stablehlo.multiply %0, %a : {t}
    return %1 : {t}""")
    with _native_engine():
        ex = engine.compile_program(mod, "mlir")
        before = native.stats()["compiles"]
        seen = []
        for _ in range(3):
            x = _f32((4, 4))
            outs = engine.execute(ex, _buffers([x]))
            seen.append((x, engine.to_host(outs[0])))
        assert ex._native_prog is not False
        # One trace for three calls...
        assert native.stats()["compiles"] == before + 1
        # ...and every one of them right.
        for x, got in seen:
            ref, _ = _run(mod, [x], False, compiled=True)
            assert got == ref[0]


# --------------------------------------------------------------------------
# flush disciplines
# --------------------------------------------------------------------------


def test_eager_flush_fires_on_the_byte_cadence():
    """interpreter.FLUSH_BYTES, metered by the tape's own per-op byte
    estimates. The cadence lives in Python; tape.configure copies it in."""
    from metaljax import interpreter
    t = "tensor<256x256xf32>"
    body = [f"    %v0 = stablehlo.tanh %a : {t}"]
    for i in range(1, 12):
        body.append(f"    %v{i} = stablehlo.tanh %v{i - 1} : {t}")
    body.append(f"    return %v11 : {t}")
    mod = _mod([("a", t)], [t], "\n".join(body))
    x = _f32((256, 256))
    with _native_engine():
        ref, _ = _run(mod, [x], False)
        with _patched(interpreter, "FLUSH_BYTES", 256 * 1024):
            before = native.stats()["flushes"]
            got, ex = _run(mod, [x], True)
            assert ex._native_prog is not False
            assert native.stats()["flushes"] > before
        # ...and the same bytes whatever the cadence.
        assert got == ref


def test_eager_flush_is_off_when_the_budget_is_zero():
    from metaljax import interpreter
    t = "tensor<64x64xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.tanh %a : {t}
    return %0 : {t}""")
    with _native_engine():
        with _patched(interpreter, "FLUSH_BYTES", 0):
            before = native.stats()["flushes"]
            _run(mod, [_f32((64, 64))], True)
            assert native.stats()["flushes"] == before


def test_loop_flush_runs_and_clears_on_the_op_unit_cadence():
    """ops.control._LOOP_CLEAR_COST: loop sync points return MLX's cache to
    the OS every so many op-units, because Metal caps live buffers by COUNT
    while the cache is bounded by BYTES (CLAUDE.md item 14)."""
    from metaljax.ops import control
    with _no_msl():
        with _patched(control, "_LOOP_CLEAR_COST", 1):
            before = native.stats()
            check(_counted_mod(40, _STEP), [_x()])
            after = native.stats()
    assert after["loop_flushes"] > before["loop_flushes"]
    assert after["loop_clears"] > before["loop_clears"]


def test_compiled_bodies_are_used_for_long_loops():
    """A chunked replay is a compiled graph of K iterations, and the
    single-step path compiles the body too: either way the loop must not be
    walking op by op, which is what this counter says."""
    with _no_msl():
        before = native.stats()["compiled_calls"]
        check(_counted_mod(40, _STEP), [_x()])
        assert native.stats()["compiled_calls"] > before


def test_a_body_over_the_byte_budget_is_not_compiled():
    """The compile-bytes gate, solved for `repeat`.

    ops.control._body_fn refuses to compile a body whose SINGLE iteration
    would trace more than METALJAX_COMPILE_BYTES_MB, and that refusal is
    load-bearing: a compiled body holds every intermediate of an iteration,
    while an interpreted one flushes inside it on the byte cadence. Reading
    the limit off `_bytes_chunks` (which never returns less than 1, because
    its own callers ask a different question) compiled bodies the Python
    engine refuses -- measured as 2.38 GB of peak against 1.19 on the
    byte-gated random.normal init.
    """
    from metaljax.ops import control
    with _no_msl():
        with _patched(control, "_COMPILE_BYTES", 1):
            before = native.stats()["compiled_calls"]
            ex = check(_counted_mod(6, _STEP), [_x()])
            assert native.stats()["compiled_calls"] == before
    assert ex._native_prog is not False


def test_body_compilation_can_be_turned_off():
    """METALJAX_BODY_COMPILE=0, the targeted mitigation for the command-
    buffer corruption in REPLAYED compiled bodies: bodies then run
    uncompiled while everything else keeps its path."""
    from metaljax.ops import control
    with _no_msl():
        with _patched(control, "_BODY_COMPILE", False):
            with _patched(control, "_CHUNK_MAX_COST", 0):
                before = native.stats()["compiled_calls"]
                ex = check(_counted_mod(12, _STEP), [_x()])
                assert native.stats()["compiled_calls"] == before
    assert ex._native_prog is not False


# --------------------------------------------------------------------------
# M5a: pipelined dynamic loops, and the carry donation that rides with them
# --------------------------------------------------------------------------
#
# M4's real-model verdict was that the mlx-lm decode gap is not Python
# dispatch but the PER-TOKEN pipeline stall: a dynamic while used to submit
# its condition and wait, then submit its body and wait, so the device sat
# idle across both decisions. The native engine now builds the body and the
# next condition BEFORE reading the current one back, which is only sound
# because building an MLX graph is lazy and pure -- and only sound for a
# body that reads nothing back to the host itself (a nested dynamic while
# would be RUN, not built, and need not even terminate at a carry the outer
# loop is about to abandon).
#
# The same rewrite is what lets MLX donate a KV cache: a second handle on a
# carry that is still alive when the next iteration's update evaluates makes
# mx::array::is_donatable false, and the whole cache gets copied per token.
#
# The differential contract is unchanged -- `check` runs the Python engine,
# which keeps the serial shape, against the native one -- so the cases below
# are statements about the pipeline holding the SAME values, not just about
# it being fast.


def _dyn_mod(limit, carry_t="tensor<4xf32>", step=None, seed=None):
    """A dynamic while with an exact trip count.

    Dynamic, not counted: the comparison is written with the bound on the
    LEFT (`limit > i`), which ops.control._analyze_counted does not
    recognise -- it wants `LT` with a block argument as its left operand.
    Same trip count as the counted form, none of the counted fast path.
    """
    step = step or """       %h = stablehlo.multiply %v, %v : tensor<4xf32>
       %nv = stablehlo.tanh %h : tensor<4xf32>"""
    seed = seed or f"    %init = stablehlo.add %x, %x : {carry_t}"
    return _mod([("x", carry_t)], [carry_t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %cn = stablehlo.constant dense<{limit}> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
{seed}
    %r:2 = stablehlo.while(%i = %ci, %v = %init) : tensor<i32>, {carry_t}
     cond {{
       %p = stablehlo.compare GT, %cn, %i : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
{step}
       stablehlo.return %ni, %nv : tensor<i32>, {carry_t}
     }}
    return %r#1 : {carry_t}""")


def _pipelined(on):
    from metaljax.ops import control
    return _patched(control, "_WHILE_PIPELINE", 1 if on else 0)


# 0 and 1 straddle the unpipelined warm-up iteration; 2 and 3 are the first
# and second turns of the pipelined loop proper; 63/64/65 straddle the flush
# cadence, and 129 runs long enough that a carry left behind would have
# compounded.
@pytest.mark.parametrize("trip", [0, 1, 2, 3, 5, 63, 64, 65, 129])
def test_dynamic_while_pipelined_matches_serial(trip):
    """Bit-for-bit, at every trip count that straddles a boundary."""
    mod = _dyn_mod(trip)
    args = [_x()]
    with _no_msl(), _native_engine():
        with _pipelined(False):
            ref, ex0 = _run(mod, args, True)
            assert ex0._native_prog is not False
        with _pipelined(True):
            got, ex1 = _run(mod, args, True)
            assert ex1._native_prog is not False
    assert got == ref, "the pipelined loop returned different bytes"


@pytest.mark.parametrize("trip", [0, 1, 2, 5, 65])
def test_dynamic_while_pipelined_matches_the_python_engine(trip):
    """...and the Python engine, which keeps the serial shape, agrees too."""
    with _no_msl(), _pipelined(True):
        check(_dyn_mod(trip), [_x()])


def test_pipelined_dynamic_while_says_so():
    with _no_msl(), _native_engine():
        with _pipelined(True):
            before = native.stats()
            _run(_dyn_mod(6), [_x()], True)
            after = native.stats()
    assert after["pipelined_loops"] == before["pipelined_loops"] + 1
    assert after["serial_loops"] == before["serial_loops"]
    # Six iterations, one of them the unpipelined warm-up.
    assert after["pipelined_steps"] == before["pipelined_steps"] + 5


def test_the_pipeline_can_be_turned_off():
    with _no_msl(), _native_engine():
        with _pipelined(False):
            before = native.stats()
            _run(_dyn_mod(6), [_x()], True)
            after = native.stats()
    assert after["serial_loops"] == before["serial_loops"] + 1
    assert after["pipelined_loops"] == before["pipelined_loops"]


def test_a_body_that_reads_the_host_is_not_pipelined():
    """A nested while makes "build the body" mean "run the body", and a
    speculative run of a nested DYNAMIC loop is not even guaranteed to
    terminate at a carry the outer loop is about to abandon. Such a body
    keeps the serial shape."""
    t = "tensor<4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %cn = stablehlo.constant dense<3> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %cz = stablehlo.constant dense<0> : tensor<i32>
    %init = stablehlo.add %x, %x : {t}
    %r:2 = stablehlo.while(%i = %ci, %v = %init) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare GT, %cn, %i : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %s:2 = stablehlo.while(%j = %cz, %u = %v) : tensor<i32>, {t}
        cond {{
          %q = stablehlo.compare GT, %c2, %j : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %q : tensor<i1>
        }} do {{
          %nj = stablehlo.add %j, %c1 : tensor<i32>
          %nu = stablehlo.tanh %u : {t}
          stablehlo.return %nj, %nu : tensor<i32>, {t}
        }}
       stablehlo.return %ni, %s#1 : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
    with _no_msl(), _native_engine():
        with _pipelined(True):
            before = native.stats()
            _run(mod, [_x()], True)
            after = native.stats()
    # The OUTER loop declines: one serial loop, run once. The INNER one is
    # dynamic too and its own body reads nothing back, so it pipelines --
    # three times, once per outer iteration. The rule is per loop, not per
    # program.
    assert after["serial_loops"] == before["serial_loops"] + 1
    assert after["pipelined_loops"] == before["pipelined_loops"] + 3
    with _no_msl():
        check(mod, [_x()])


def test_pipelined_loop_keeps_the_loop_clear_cadence():
    """The pipelined loop's blocking point is a host read of the CONDITION
    rather than an eval of the carry, but it is still a loop sync point: the
    op-unit budget that returns MLX's cache to the OS (CLAUDE.md item 14)
    has to be charged there, or a long decode accumulates tiny buffers until
    metal::malloc dies."""
    from metaljax.ops import control
    with _no_msl(), _native_engine():
        with _pipelined(True), _patched(control, "_LOOP_CLEAR_COST", 1):
            before = native.stats()
            _run(_dyn_mod(40), [_x()], True)
            after = native.stats()
    assert after["loop_flushes"] > before["loop_flushes"]
    assert after["loop_clears"] > before["loop_clears"]


# --- the carry a loop updates in place -------------------------------------


def _dus_step(t, n):
    return (
        "       %one = stablehlo.constant dense<1.0> : tensor<f32>\n"
        "       %u = stablehlo.broadcast_in_dim %one, dims = [] : "
        "(tensor<f32>) -> tensor<1xf32>\n"
        f"       %nv = stablehlo.dynamic_update_slice %v, %u, %i : "
        f"({t}, tensor<1xf32>, tensor<i32>) -> {t}")


@pytest.mark.parametrize("trip", [0, 1, 2, 3, 8])
def test_a_speculatively_built_body_never_writes_the_carry(trip):
    """The exit carry, element by element.

    This is the case donation makes dangerous: the body writes INTO its
    carry, and MLX reuses that buffer when nothing else holds it. The
    pipeline builds iteration t's body before knowing whether iteration t
    happens -- so if it ever let that body EVALUATE before the condition
    said yes, the loop would come back with one write too many. `trip` is
    exactly how many ones the answer may hold.
    """
    t = "tensor<8xf32>"
    mod = _dyn_mod(trip, carry_t=t, step=_dus_step(t, 8))
    x = np.zeros(8, np.float32)
    with _no_msl(), _native_engine():
        with _pipelined(True):
            got, ex = _run(mod, [x], True)
            assert ex._native_prog is not False
    out = np.frombuffer(got[0], np.float32)
    want = np.zeros(8, np.float32)
    want[:trip] = 1.0
    np.testing.assert_array_equal(out, want)


def test_an_in_place_carry_update_does_not_copy_the_whole_cache():
    """Donation, measured the only way it shows: by not costing anything.

    A `dynamic_update_slice` whose operand is a carry can reuse that
    carry's buffer -- mx::array::is_donatable is a use_count test, so it
    happens exactly when the engine has let go of the old carry by the time
    the update evaluates. Nothing observable distinguishes the two answers
    except bandwidth: copying an N-megabyte cache per iteration measured
    4.6 us per megabyte on this machine, so a 4x bigger cache cost 4x more
    per step. Donated, the per-step cost is FLAT.

    Measured through this harness when it landed: 32 MB 143 us/step, 128 MB
    204 (ratio 1.43 -- the residual is per-execute cost the slope does not
    quite cancel at these sizes). With the bug deliberately put back: 271
    and 746, ratio 2.75, and 5/5 runs fail. The threshold sits between.
    """
    import time

    import mlx.core as mx

    def per_step(mbytes):
        n = (mbytes << 20) // 4
        t = f"tensor<{n}xf32>"
        best = {}
        for trip in (4, 36):
            mod = _dyn_mod(trip, carry_t=t, step=_dus_step(t, n))
            ex = engine.compile_program(mod, "mlir")
            bufs = _buffers([np.zeros(n, np.float32)])
            # Warm up: the first execute of a process pays for kernel
            # builds and a cold allocator, and it lands on whichever
            # measurement happens to run first -- which is enough to hide
            # the copy one time in five (measured, on code with the bug
            # deliberately put back).
            outs = engine.execute(ex, bufs)
            mx.eval(*[o.data for o in outs])
            del outs
            best[trip] = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                outs = engine.execute(ex, bufs)
                mx.eval(*[o.data for o in outs])
                best[trip] = min(best[trip], time.perf_counter() - t0)
                del outs
            assert ex._native_prog is not False
        return (best[36] - best[4]) / 32

    with _no_msl(), _native_engine(), _pipelined(True):
        small = per_step(32)
        big = per_step(128)
    assert big < 2.0 * small, (
        f"a 4x bigger carry cost {big / small:.2f}x more per iteration "
        f"({small * 1e6:.0f} -> {big * 1e6:.0f} us): the update is copying "
        f"the whole cache instead of donating it")
