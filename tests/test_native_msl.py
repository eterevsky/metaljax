"""Stage 2 M5b: generated kernels and host ops on the native tape.

Two features, one contract, and it is the contract of every file in this
family: the SAME executable runs on both engines and the output BYTES must
be equal.

* **msl_scan kernels.** A counted loop msl_scan can turn into one generated
  Metal kernel takes that kernel on whichever engine runs it — the tape asks
  `ops.control._msl_plan_for`, the same question through the same cache the
  Python engine asks, because a loop that took a kernel on one engine and an
  interpreted loop on the other would compute its carries by different
  arithmetic and no byte comparison could hold. So these cases are also a
  test that the two engines agree about WHICH loops get kernels.
  The families here are the texmo shapes msl_scan was built for, in all
  three of its emitter modes: elementwise cells (scalar), small in-lane
  matvecs (vector), full-width cells through threadgroup memory (coop),
  forward and — where the interesting machinery is — through AD, whose
  weight-gradient accumulations leave the kernel as stacked outputs and
  come back as one post-kernel matmul (loop fission).

* **Host ops.** LAPACK through numpy/scipy and the in-process callbacks
  jax.debug.print / pure_callback lower to are not portable to C++ and never
  will be. The tape marks the site; the native engine reacquires the GIL
  there and nowhere else, so the rest of an impure program still runs
  natively. Their tests check the values, the side effects, and that the
  program really did lower.
"""
import contextlib
import os

import numpy as np
import pytest

from test_native_tape import (  # noqa: F401  (imports gate the native build)
    _buffers,
    _f32,
    _mod,
    _native_engine,
    _run,
    check,
    declines,
    engine,
    native,
)

pytestmark = pytest.mark.skipif(
    not hasattr(native, "MslPlan"),
    reason="native extension predates the M5b kernels")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from helpers import lower_bytes  # noqa: E402

# Opcodes, as native/tape.cc numbers them. Read from the extension so a new
# op in the middle of the enum cannot silently re-point these.
OPS = native.opcodes()
MSL = OPS["metaljax.msl_scan"]
HOST = OPS["metaljax.host_call"]
WHILE = OPS["stablehlo.while"]
TOKEN = OPS["stablehlo.create_token"]


def _lowered(f, *args, compiled=False):
    """Both engines on a jitted function; returns the native Program.

    The module is what jax lowers, which is the point: msl_scan's
    recognizers match the IR jax's scan and its AD transpose produce, and a
    hand-written loop would not exercise the same shapes.
    """
    mod = lower_bytes(f, *args)
    flat = [np.asarray(x) for x in jax.tree.leaves(args)]
    return check(mod, flat, compiled=compiled)


def _histogram(ex):
    return ex._native_prog.op_histogram()


def _msl_count(ex):
    return _histogram(ex).get(MSL, 0)


@contextlib.contextmanager
def _patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


# --------------------------------------------------------------------------
# the loop families
# --------------------------------------------------------------------------

RNG = np.random.default_rng(20260810)
L, B, H = 12, 3, 5
A = (RNG.random((L, B, H)) * 0.9).astype(np.float32)
X = (RNG.standard_normal((L, B, H)) * 0.3).astype(np.float32)
H0 = RNG.standard_normal((B, H)).astype(np.float32)


def affine_scan(a, x, h0):
    """The elementwise cell: h' = a*h + x, one kernel, scalar mode."""
    def cell(h, ax):
        h = ax[0] * h + ax[1]
        return h, h
    return jax.lax.scan(cell, h0, (a, x))


def mingru(z_in, c_in, h0, b):
    def cell(h, zc):
        z = jax.nn.sigmoid(zc[0] + b)
        c = jnp.tanh(zc[1])
        nh = (1 - z) * h + z * c
        return nh, nh
    return jax.lax.scan(cell, h0, (z_in, c_in))


def matvec(w, x, h0):
    """An in-lane matvec cell: vector mode, and its AD transpose is where
    the weight-gradient accumulation (loop fission) lives."""
    def cell(h, x_):
        return jnp.tanh(h @ w + x_), h
    return jax.lax.scan(cell, h0, x)


def _sum_grad(f, argnums=(0, 1, 2)):
    return jax.value_and_grad(lambda *a: f(*a)[1].sum(), argnums=argnums)


W = (RNG.standard_normal((H, H)) * 0.3).astype(np.float32)
BIAS = (RNG.standard_normal(H) * 0.1).astype(np.float32)

# Coop mode: a full-width cell, one threadgroup per batch element.
FC = 32
WC = (RNG.standard_normal((FC, FC)) * 0.1).astype(np.float32)
XC = (RNG.standard_normal((8, 4, FC)) * 0.3).astype(np.float32)
HC = (RNG.standard_normal((4, FC)) * 0.3).astype(np.float32)


@pytest.mark.parametrize("compiled", [False, True])
def test_affine_cell(compiled):
    ex = _lowered(affine_scan, A, X, H0, compiled=compiled)
    assert _msl_count(ex) == 1
    # ...and the loop is the kernel, not an interpreted while beside it.
    assert _histogram(ex).get(WHILE, 0) == 0


@pytest.mark.parametrize("compiled", [False, True])
def test_affine_cell_grad(compiled):
    ex = _lowered(_sum_grad(affine_scan), A, X, H0, compiled=compiled)
    # Forward and backward are separate loops, and both get kernels.
    assert _msl_count(ex) == 2


def test_mingru_cell():
    assert _msl_count(_lowered(mingru, A, X, H0, BIAS)) == 1


def test_mingru_cell_grad():
    ex = _lowered(_sum_grad(mingru, argnums=(0, 1, 2, 3)), A, X, H0, BIAS)
    assert _msl_count(ex) == 2


@pytest.mark.parametrize("compiled", [False, True])
def test_matvec_cell_vector_mode(compiled):
    ex = _lowered(matvec, W, X[:, 0], H0[0], compiled=compiled)
    assert _msl_count(ex) == 1


@pytest.mark.parametrize("compiled", [False, True])
def test_matvec_cell_grad(compiled):
    """The loop-fission path: a cross-lane weight gradient leaves the kernel
    as a stacked output and comes back as one batched matmul, which is the
    accumulator recipe the tape encodes as a little tree."""
    ex = _lowered(_sum_grad(matvec), W, X[:, 0], H0[0], compiled=compiled)
    assert _msl_count(ex) == 2


def test_coop_cell():
    assert _msl_count(_lowered(matvec, WC, XC, HC)) == 1


def test_coop_cell_grad():
    assert _msl_count(_lowered(_sum_grad(matvec), WC, XC, HC)) == 2


def test_packed_inputs():
    """Input pooling (0.4.3): with the trigger dropped to nothing, the plan
    pools its same-dtype inputs into one buffer per dtype and bakes the
    element offsets into the source. The tape has to concatenate them in
    exactly the order those offsets assume."""
    from metaljax import msl_scan
    with _patched(msl_scan, "_PACK_TRIGGER", 0):
        ex = _lowered(mingru, A, X, H0, BIAS)
    assert _msl_count(ex) == 1


def test_carry_that_is_also_read():
    """A pass-through carry the kernel reads as a buffer: it is handed back
    as the very array it came in as, which is also the one carry an msl loop
    can return unchanged (and so the only one whose aliasing the lowering
    has to charge)."""
    def f(a, x, h0):
        h, ys = affine_scan(a, x, h0)
        return h, ys, a
    ex = _lowered(f, A, X, H0)
    assert _msl_count(ex) == 1


def test_int_carry_and_stacked_output():
    """A counter carry and a stacked write, which the kernel does not
    compute at all: the counter is an affine step the tape applies
    afterwards, the stack is a kernel output."""
    def f(x, h0):
        def cell(carry, x_):
            h, n = carry
            return (jnp.tanh(h + x_), n + 2), h * 2.0
        return jax.lax.scan(cell, (h0, jnp.int32(0)), x)
    ex = _lowered(f, X[:, 0], H0[0])
    assert _msl_count(ex) == 1


def test_kernel_reads_a_captured_value():
    """A source that is not a carry but a value from the enclosing block.

    Written by hand because jax does not emit one: `lax.scan` threads its
    closed-over constants through the carry list, so every source of a
    jax-lowered plan is a carry. A region capture is what the tape has to
    resolve to a slot of the FRAME it is lowering, which is the same
    question `Plan.run`'s `hoisted` answers out of the environment.
    """
    t = "tensor<4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %ci = stablehlo.constant dense<0> : tensor<i32>
    %cn = stablehlo.constant dense<6> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %k = stablehlo.tanh %x : {t}
    %init = stablehlo.add %x, %x : {t}
    %r:2 = stablehlo.while(%i = %ci, %v = %init) : tensor<i32>, {t}
     cond {{
       %p = stablehlo.compare LT, %i, %cn : (tensor<i32>, tensor<i32>) -> tensor<i1>
       stablehlo.return %p : tensor<i1>
     }} do {{
       %ni = stablehlo.add %i, %c1 : tensor<i32>
       %m = stablehlo.multiply %v, %k : {t}
       %nv = stablehlo.tanh %m : {t}
       stablehlo.return %ni, %nv : tensor<i32>, {t}
     }}
    return %r#1 : {t}""")
    ex = check(mod, [(_f32((4,)) * 0.25).astype(np.float32)])
    assert _msl_count(ex) == 1


def test_msl_off_lowers_an_ordinary_loop():
    """The agreement runs both ways: with msl_scan off, neither engine
    plans a kernel and the loop is an ordinary while on the tape."""
    from metaljax import msl_scan
    with _patched(msl_scan, "ENABLED", False):
        ex = _lowered(affine_scan, A, X, H0)
    assert _msl_count(ex) == 0
    assert _histogram(ex).get(WHILE, 0) == 1


def test_two_loops_one_program():
    """Two scans in one program: two entries, two kernels, one tape."""
    def f(a, x, h0):
        h1, y1 = affine_scan(a, x, h0)
        h2, y2 = affine_scan(a, y1, h1)
        return h2, y2
    ex = _lowered(f, A, X, H0)
    assert _msl_count(ex) == 2


# --------------------------------------------------------------------------
# the failure path
# --------------------------------------------------------------------------


def test_metal_build_failure_falls_back(monkeypatch):
    """Metal's shader compiler rejecting a generated source must cost the
    kernel, not the program: the plan dies, the entry runs the interpreted
    loop it carries alongside, and the answer is still right. The C++ half
    of engine.MetalExecutable.disable_msl.
    """
    monkeypatch.setenv("METALJAX_MSL_FORCE_BUILD_FAIL", "1")
    before = native.stats()["msl_failures"]
    mod = lower_bytes(affine_scan, A, X, H0)
    flat = [np.asarray(x) for x in jax.tree.leaves((A, X, H0))]
    with _native_engine():
        ref, _ = _run(mod, flat, False)
        ex = engine.compile_program(mod, "mlir")
        ex._can_compile = False
        got = [engine.to_host(o) for o in engine.execute(ex, _buffers(flat))]
        assert got == ref
        assert ex._native_prog is not False, "the tape was retired"
        after = native.stats()["msl_failures"]
        assert after > before
        # ...and the dead plan stays dead: a second call neither rebuilds
        # the kernel nor fails again.
        again = [engine.to_host(o) for o in engine.execute(ex, _buffers(flat))]
        assert again == ref
        assert native.stats()["msl_failures"] == after


def test_build_failure_inside_a_compiled_trace(monkeypatch):
    """The same failure from a kernel that was TRACED into a compiled graph,
    where it cannot be proven at the point it is built. The call is settled
    synchronously while any such plan is unproven (an async worker raising a
    Metal build error aborts the process), and the program is rerun with the
    plan retired."""
    monkeypatch.setenv("METALJAX_MSL_FORCE_BUILD_FAIL", "1")
    mod = lower_bytes(affine_scan, A, X, H0)
    flat = [np.asarray(x) for x in jax.tree.leaves((A, X, H0))]
    with _native_engine():
        ref, _ = _run(mod, flat, False)
        ex = engine.compile_program(mod, "mlir")
        got = [engine.to_host(o) for o in engine.execute(ex, _buffers(flat))]
        assert got == ref
        assert ex._native_prog is not False


# --------------------------------------------------------------------------
# host ops
# --------------------------------------------------------------------------


def _spd(n=4, seed=3):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n)).astype(np.float32)
    return (m @ m.T + n * np.eye(n)).astype(np.float32), m


SPD, MAT = _spd()


def test_cholesky_runs_on_the_host():
    t = "tensor<4x4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %s = stablehlo.add %a, %a : {t}
    %0 = "stablehlo.cholesky"(%s) {{lower = true}} : ({t}) -> {t}
    %1 = stablehlo.multiply %0, %0 : {t}
    return %1 : {t}""")
    ex = check(mod, [SPD])
    hist = _histogram(ex)
    assert hist.get(HOST, 0) == 1
    # The arithmetic around it stayed native — that is the whole point.
    assert sum(hist.values()) > 1


def test_triangular_solve_runs_on_the_host():
    t, tv = "tensor<4x4xf32>", "tensor<4x1xf32>"
    mod = _mod([("a", t), ("b", tv)], [tv], f"""
    %0 = "stablehlo.triangular_solve"(%a, %b) {{
      left_side = true, lower = true,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>,
      unit_diagonal = false}} : ({t}, {tv}) -> {tv}
    %1 = stablehlo.add %0, %0 : {tv}
    return %1 : {tv}""")
    v = RNG.standard_normal((4, 1)).astype(np.float32)
    assert _histogram(check(mod, [SPD, v])).get(HOST, 0) == 1


@pytest.mark.parametrize("name", ["eigh", "qr", "svd"])
def test_lapack_families(name):
    fns = {
        "eigh": lambda a: jnp.linalg.eigh(a)[0],
        "qr": lambda a: jnp.linalg.qr(a)[1],
        "svd": lambda a: jnp.linalg.svd(a, compute_uv=False),
    }
    args = (SPD,) if name == "eigh" else (MAT,)
    ex = _lowered(fns[name], *args)
    assert _histogram(ex).get(HOST, 0) >= 1


def test_callback_with_results():
    """pure_callback / io_callback: the handler runs in Python, in program
    order, and its result feeds the native ops after it."""
    from metaljax.ops import callbacks
    seen = []

    def fn(x):
        seen.append(float(np.asarray(x).sum()))
        return np.asarray(x) * 2.0

    idx = callbacks.register_callback(fn)
    t = "tensor<4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %h = stablehlo.multiply %x, %x : {t}
    %0 = stablehlo.custom_call @metaljax_callback(%h) {{
      backend_config = "{idx}", has_side_effect = true
    }} : ({t}) -> {t}
    %1 = stablehlo.add %0, %h : {t}
    return %1 : {t}""")
    x = _f32((4,))
    ex = check(mod, [x])
    assert _histogram(ex).get(HOST, 0) == 1
    # Twice: once per engine, and each ran the callback exactly once.
    assert len(seen) == 2 and seen[0] == seen[1]


def test_ordered_callback_and_tokens():
    """jax.debug.print's shape: a callback with no results at all, and the
    ordered-effect tokens threaded around it — which carry no data and exist
    only to keep the order the tape already runs them in."""
    from metaljax.ops import callbacks
    order = []
    a = callbacks.register_callback(lambda x: order.append(("a", float(x[0]))))
    b = callbacks.register_callback(lambda x: order.append(("b", float(x[0]))))
    t = "tensor<4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %t0 = stablehlo.create_token : !stablehlo.token
    %h = stablehlo.multiply %x, %x : {t}
    stablehlo.custom_call @metaljax_callback(%h) {{
      backend_config = "{a}", has_side_effect = true }} : ({t}) -> ()
    %g = stablehlo.add %h, %h : {t}
    stablehlo.custom_call @metaljax_callback(%g) {{
      backend_config = "{b}", has_side_effect = true }} : ({t}) -> ()
    %t1 = "stablehlo.after_all"(%t0) : (!stablehlo.token) -> !stablehlo.token
    %o = stablehlo.optimization_barrier %g : {t}
    return %o : {t}""")
    x = _f32((4,))
    ex = check(mod, [x])
    hist = _histogram(ex)
    assert hist.get(HOST, 0) == 2 and hist.get(TOKEN, 0) == 2
    assert [k for k, _ in order] == ["a", "b", "a", "b"]


def test_callback_that_raises_comes_back_as_an_exception():
    """A host handler is Python, so it can raise — and the native run has
    the GIL released everywhere around it. The message is formatted where
    the GIL is held and travels as a plain string, because the engine's
    recovery paths ask `what()` of whatever they catch."""
    from metaljax.ops import callbacks

    def boom(x):
        raise ValueError("callback exploded")

    idx = callbacks.register_callback(boom)
    t = "tensor<4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %h = stablehlo.multiply %x, %x : {t}
    %0 = stablehlo.custom_call @metaljax_callback(%h) {{
      backend_config = "{idx}", has_side_effect = true
    }} : ({t}) -> {t}
    return %0 : {t}""")
    with _native_engine():
        ex = engine.compile_program(mod, "mlir")
        assert ex.native_program() is not None
        with pytest.raises(Exception) as info:
            engine.execute(ex, _buffers([_f32((4,))]))
        assert "callback exploded" in str(info.value)


def test_sharding_custom_call_is_an_identity():
    t = "tensor<3x4xf32>"
    mod = _mod([("x", t)], [t], f"""
    %0 = stablehlo.custom_call @Sharding(%x) : ({t}) -> {t}
    %1 = stablehlo.tanh %0 : {t}
    return %1 : {t}""")
    ex = check(mod, [_f32()])
    assert _histogram(ex).get(HOST, 0) == 0


def test_unknown_custom_call_declines():
    t = "tensor<3x4xf32>"
    declines(_mod([("x", t)], [t], f"""
    %0 = stablehlo.custom_call @not_a_real_target(%x) : ({t}) -> {t}
    return %0 : {t}"""))
