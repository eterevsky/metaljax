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
generated Metal kernel (src/metaljax/msl_scan.py), the native tape declines
any loop that has such a plan (native msl is M5 -- see the gate test), and a
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


def test_while_with_an_msl_plan_declines():
    """The M5 boundary: a loop msl_scan can turn into one generated kernel
    is worth more than anything this milestone would do with it."""
    from metaljax.ops import control
    mod = _counted_mod(8, _STEP)
    with _patched(control, "_msl_plan_for", lambda interp, op: object()):
        with _native_engine():
            ex = engine.compile_program(mod, "mlir")
            assert ex.native_program() is None


def test_while_with_a_real_msl_plan_declines():
    """The same boundary without the patch: this loop really does get a
    plan, which is why every other loop test here turns msl off."""
    from metaljax.ops import control
    with _native_engine():
        ex = engine.compile_program(_counted_mod(8, _STEP), "mlir")
        with ex.interpreter.context:
            wop = next(o.operation for o in ex.interpreter._main_block()
                       .operations if o.operation.name == "stablehlo.while")
            assert control._msl_plan_for(ex.interpreter, wop) is not None
        assert ex.native_program() is None


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
