"""Stage 2 M2: the native tape, differential against the Python engine.

Every case runs the SAME executable twice — once with the tape lowered
(native/, program.h) and once with it forced off — and compares output BYTES.
Both engines call the same MLX kernels, so a difference is a mis-ported
handler, never a tolerance question.

TWO TRAPS the harness pins down, both worth knowing before reading a
failure here:

* The reference is the Python EAGER path, not its default one. A pure
  program normally runs through mx.compile, and a fused MLX kernel does
  not always agree bit-for-bit with the same ops dispatched one at a time
  — MLX inlines rank-0 constants as %.7g literals, which is 1 ULP on most
  of them (CLAUDE.md item 20). `stablehlo.cbrt` shows it: compiled and
  eager differ, eager and native are identical. The M2 tape is an eager
  interpreter, so eager is what it must equal.
* `engine.NATIVE` stays set for both runs, so the C++ buffer path (M1)
  carries the data either way and only the tape varies.
"""
import contextlib
import os
import sys

import ml_dtypes
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "native", "build"))

native = pytest.importorskip(
    "metaljax_native",
    reason="native engine not built (native/build.sh)")

import mlx.core as mx  # noqa: E402

from metaljax import engine, tape  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(native, "Program"),
    reason="native extension predates the M2 tape")


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def _mod(params, results, body, name="tape_test"):
    """A single-function StableHLO module, as text."""
    args = ", ".join(f"%{n}: {t}" for n, t in params)
    outs = ", ".join(results)
    return (f"module @{name} {{\n"
            f"  func.func public @main({args}) -> ({outs}) {{\n"
            f"{body}\n"
            f"  }}\n"
            f"}}\n").encode()


@contextlib.contextmanager
def _native_engine():
    saved = engine.NATIVE
    engine.NATIVE = native
    try:
        yield
    finally:
        engine.NATIVE = saved


def _buffers(arrays):
    bufs = []
    for a in arrays:
        arr = np.ascontiguousarray(a)
        bufs.append(engine.buffer_from_host(
            arr.tobytes(), engine._NP_TO_ENUM[arr.dtype], list(arr.shape),
            None, 0))
    return bufs


def _run(mod, arrays, tape_on, compiled=False):
    ex = engine.compile_program(mod, "mlir")
    if not tape_on:
        ex._native_prog = False  # the Python engine
    if not compiled:
        # Op by op on WHICHEVER engine runs it; see the module docstring on
        # why eager is the reference. Set before the first execute, so the
        # tape is lowered with the same answer the Python engine acts on.
        ex._can_compile = False
    outs = engine.execute(ex, _buffers(arrays))
    return [engine.to_host(o) for o in outs], ex


def check(mod, arrays, lowered=True, compiled=False):
    """Run both engines on `mod`; assert identical bytes and that the tape
    did (or did not) take the program. Returns the native executable."""
    with _native_engine():
        before = dict(engine.NATIVE_STATS)
        ref, _ = _run(mod, arrays, False, compiled)
        got, ex = _run(mod, arrays, True, compiled)
    assert len(ref) == len(got), f"{len(got)} outputs vs {len(ref)}"
    for i, (r, g) in enumerate(zip(ref, got)):
        assert r == g, f"output {i}: native bytes differ from the Python engine"
    took = ex._native_prog is not False
    assert took == lowered, (
        f"expected the tape to {'lower' if lowered else 'decline'} this "
        f"program, it did the opposite")
    key = "lowered" if lowered else "declined"
    assert engine.NATIVE_STATS[key] == before[key] + 1
    return ex


def fresh_outputs(mod, arrays, positions, args_too=()):
    """Assert the named outputs are buffers of their own.

    Two calls are kept alive at once on purpose: an output that reads a
    constant the Program holds (or an argument's array) would come back at
    the SAME address twice, and with both alive the allocator cannot make
    two distinct copies look equal by reusing a freed buffer.
    """
    with _native_engine():
        ex = engine.compile_program(mod, "mlir")
        in1 = _buffers(arrays)
        out1 = engine.execute(ex, in1)
        in2 = _buffers(arrays)
        out2 = engine.execute(ex, in2)
        assert ex._native_prog is not False, "the tape declined this program"
        for j in positions:
            assert engine.buffer_pointer(out1[j]) != \
                engine.buffer_pointer(out2[j]), \
                f"output {j} is the same buffer on two calls"
            for i in args_too:
                assert engine.buffer_pointer(out1[j]) != \
                    engine.buffer_pointer(in1[i]), \
                    f"output {j} shares argument {i}'s buffer"
        return ex


def declines(mod):
    """Assert the tape refuses `mod` without executing it.

    For programs the PYTHON engine cannot run either — a recursive callee
    diverges — so `check` has no reference to compare against.
    """
    with _native_engine():
        ex = engine.compile_program(mod, "mlir")
        assert ex.native_program() is None, "the tape lowered this program"


RNG = np.random.default_rng(20260806)


def _f32(shape=(3, 4)):
    return RNG.standard_normal(shape).astype(np.float32)


def _specials(npdt=np.float32):
    """The values that separate a faithful port from an approximate one.

    3.4e38 overflows to inf in the halves, which is the point: both engines
    see the same input bytes whatever numpy warns about on the way in.
    """
    x = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 1e-30, 3.4e38,
                  np.inf, -np.inf, np.nan, -np.nan], np.float32)
    if npdt is np.float32:
        return x
    with np.errstate(over="ignore"):
        return x.astype(npdt)


# --------------------------------------------------------------------------
# elementwise
# --------------------------------------------------------------------------

_CHAIN_DTYPES = {
    "f32": ("f32", np.float32),
    "f16": ("f16", np.float16),
    "bf16": ("bf16", ml_dtypes.bfloat16),
    "i32": ("i32", np.int32),
    "i8": ("i8", np.int8),
    "ui16": ("ui16", np.uint16),
}


@pytest.mark.parametrize("name", sorted(_CHAIN_DTYPES))
def test_elementwise_chain(name):
    el, npdt = _CHAIN_DTYPES[name]
    t = f"tensor<3x4x{el}>"
    mod = _mod([("a", t), ("b", t)], [t], f"""
    %0 = stablehlo.add %a, %b : {t}
    %1 = stablehlo.multiply %0, %a : {t}
    %2 = stablehlo.subtract %1, %b : {t}
    %3 = stablehlo.maximum %2, %a : {t}
    %4 = stablehlo.minimum %3, %b : {t}
    return %4 : {t}""")
    if np.issubdtype(np.dtype(npdt), np.integer):
        info = np.iinfo(npdt)
        a = RNG.integers(info.min // 2, info.max // 2, (3, 4)).astype(npdt)
        b = RNG.integers(info.min // 2, info.max // 2, (3, 4)).astype(npdt)
    else:
        a = _f32().astype(npdt)
        b = _f32().astype(npdt)
        a.reshape(-1)[:3] = np.array([np.nan, np.inf, -0.0]).astype(npdt)
    check(mod, [a, b])


def test_bool_chain():
    t = "tensor<3x4xi1>"
    mod = _mod([("a", t), ("b", t)], [t, t, t, t, t], f"""
    %0 = stablehlo.add %a, %b : {t}
    %1 = stablehlo.multiply %a, %b : {t}
    %2 = stablehlo.and %a, %b : {t}
    %3 = stablehlo.or %a, %b : {t}
    %4 = stablehlo.xor %a, %b : {t}
    return %0, %1, %2, %3, %4 : {t}, {t}, {t}, {t}, {t}""")
    a = RNG.integers(0, 2, (3, 4)) > 0
    b = RNG.integers(0, 2, (3, 4)) > 0
    check(mod, [a, b])


def test_int_bitwise():
    t = "tensor<3x4xi32>"
    mod = _mod([("a", t), ("b", t)], [t, t, t], f"""
    %0 = stablehlo.and %a, %b : {t}
    %1 = stablehlo.or %a, %b : {t}
    %2 = stablehlo.xor %a, %b : {t}
    return %0, %1, %2 : {t}, {t}, {t}""")
    a = RNG.integers(-1 << 30, 1 << 30, (3, 4)).astype(np.int32)
    b = RNG.integers(-1 << 30, 1 << 30, (3, 4)).astype(np.int32)
    check(mod, [a, b])


# Every unary in the native set, on values that reach its corners. The
# domain-restricted ones get an input the Python handler is happy with;
# NaN/inf go through the ones that must propagate them.
_UNARY = [
    ("abs", "f32", "f32"),
    ("cbrt", "f32", "f32"),
    ("ceil", "f32", "f32"),
    ("cosine", "f32", "f32"),
    ("exponential", "f32", "f32"),
    ("floor", "f32", "f32"),
    ("is_finite", "f32", "i1"),
    ("log", "f32", "f32"),
    ("log_plus_one", "f32", "f32"),
    ("logistic", "f32", "f32"),
    ("negate", "f32", "f32"),
    ("round_nearest_afz", "f32", "f32"),
    ("round_nearest_even", "f32", "f32"),
    ("rsqrt", "f32", "f32"),
    ("sign", "f32", "f32"),
    ("sine", "f32", "f32"),
    ("sqrt", "f32", "f32"),
    ("tan", "f32", "f32"),
    ("tanh", "f32", "f32"),
]


@pytest.mark.parametrize("op,el,out_el", _UNARY)
def test_unary(op, el, out_el):
    t = f"tensor<12x{el}>"
    ot = f"tensor<12x{out_el}>"
    sig = f": {t}" if el == out_el else f": ({t}) -> {ot}"
    mod = _mod([("a", t)], [ot], f"""
    %0 = stablehlo.{op} %a {sig}
    return %0 : {ot}""")
    check(mod, [_specials()])


@pytest.mark.parametrize("op", ["erf", "erf_inv", "square"])
def test_chlo_unary(op):
    # erf/erf_inv live only in chlo at StableHLO v1.18 (the stablehlo.*
    # spellings the Python registry also carries do not parse); square has
    # no stablehlo spelling at all.
    t = "tensor<12xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = chlo.{op} %a : {t} -> {t}
    return %0 : {t}""")
    x = (np.linspace(-0.99, 0.99, 12).astype(np.float32) if op == "erf_inv"
         else _specials())
    check(mod, [x])


def test_unary_sign_on_ints():
    # _sign's NaN rule is float-only; integers must keep plain mx.sign.
    t = "tensor<7xi32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.sign %a : {t}
    return %0 : {t}""")
    check(mod, [np.array([-5, -1, 0, 1, 7, -2147483648, 2147483647],
                         np.int32)])


def test_unary_abs_int_and_half():
    for el, npdt in (("i16", np.int16), ("f16", np.float16),
                     ("bf16", ml_dtypes.bfloat16)):
        t = f"tensor<6x{el}>"
        mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.abs %a : {t}
    return %0 : {t}""")
        x = np.array([-3, -1, 0, 1, 2, -7]).astype(npdt)
        check(mod, [x])


_DIRECTIONS = ["EQ", "NE", "LT", "LE", "GT", "GE"]


@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_compare(direction):
    t = "tensor<12xf32>"
    ot = "tensor<12xi1>"
    mod = _mod([("a", t), ("b", t)], [ot], f"""
    %0 = stablehlo.compare {direction}, %a, %b : ({t}, {t}) -> {ot}
    return %0 : {ot}""")
    x = _specials()
    y = np.roll(x, 3)
    check(mod, [x, y])


def test_compare_ints_and_bools():
    for el, x, y in (
            ("i32", np.array([-2, 0, 5], np.int32),
             np.array([-2, 3, 1], np.int32)),
            ("i1", np.array([True, False, True]),
             np.array([True, True, False]))):
        t = f"tensor<3x{el}>"
        ot = "tensor<3xi1>"
        mod = _mod([("a", t), ("b", t)], [ot, ot], f"""
    %0 = stablehlo.compare EQ, %a, %b : ({t}, {t}) -> {ot}
    %1 = stablehlo.compare LT, %a, %b : ({t}, {t}) -> {ot}
    return %0, %1 : {ot}, {ot}""")
        check(mod, [x, y])


def test_select_and_clamp():
    t = "tensor<12xf32>"
    pt = "tensor<12xi1>"
    mod = _mod([("p", pt), ("a", t), ("b", t)], [t, t], f"""
    %0 = stablehlo.select %p, %a, %b : {pt}, {t}
    %1 = stablehlo.clamp %b, %a, %a : {t}
    return %0, %1 : {t}, {t}""")
    p = RNG.integers(0, 2, 12) > 0
    check(mod, [p, _specials(), np.roll(_specials(), 5)])


def test_clamp_operand_order():
    # clamp is (min, operand, max): a swapped port still passes on
    # symmetric data, so pin it with bounds the operand leaves.
    t = "tensor<5xf32>"
    mod = _mod([("lo", t), ("x", t), ("hi", t)], [t], f"""
    %0 = stablehlo.clamp %lo, %x, %hi : {t}
    return %0 : {t}""")
    lo = np.array([-1.0, -1.0, 0.0, 2.0, -5.0], np.float32)
    x = np.array([-9.0, 0.5, 9.0, 1.0, 0.0], np.float32)
    hi = np.array([1.0, 1.0, 3.0, 8.0, 5.0], np.float32)
    check(mod, [lo, x, hi])


# --------------------------------------------------------------------------
# wrapped semantics: the binaries whose StableHLO meaning is not an MLX op
# --------------------------------------------------------------------------


def _binary_mod(op, el, n=12):
    t = f"tensor<{n}x{el}>"
    return _mod([("a", t), ("b", t)], [t], f"""
    %0 = stablehlo.{op} %a, %b : {t}
    return %0 : {t}""")


@pytest.mark.parametrize("el,npdt", [("f32", np.float32), ("f16", np.float16),
                                     ("bf16", ml_dtypes.bfloat16)])
def test_divide_and_remainder_floats(el, npdt):
    # Float remainder is `a - trunc(a/b)*b` (sign of the DIVIDEND), which is
    # neither mx.remainder nor floor_divide; zero and infinite divisors are
    # what separate the three.
    a = _specials(npdt)
    b = np.roll(_specials(npdt), 5)
    for op in ("divide", "remainder"):
        check(_binary_mod(op, el), [a, b])
    # The specials alone cannot tell trunc from floor: every pair of them
    # divides to 0, +-inf or NaN. Ordinary mixed-sign values can — 1 % -2.5
    # is 1.0 truncated and -1.5 floored.
    ra = np.array([7.5, -7.5, 7.5, -7.5, 1.0, -1.0, 0.3, -0.3, 5.0, -5.0,
                   2.5, -2.5], np.float32).astype(npdt)
    rb = np.array([2.0, 2.0, -2.0, -2.0, -2.5, 2.5, 0.1, -0.1, 5.0, -5.0,
                   -1.25, 1.25], np.float32).astype(npdt)
    for op in ("divide", "remainder"):
        check(_binary_mod(op, el), [ra, rb])


_INT_DIV = [("i8", np.int8), ("i16", np.int16), ("i32", np.int32),
            ("i64", np.int64), ("ui8", np.uint8), ("ui32", np.uint32)]


@pytest.mark.parametrize("el,npdt", _INT_DIV)
def test_divide_and_remainder_ints(el, npdt):
    # StableHLO divides toward ZERO; MLX only has floor_divide, so every
    # sign combination has to come out right (-7/2 == -3, not -4).
    signed = np.issubdtype(np.dtype(npdt), np.signedinteger)
    if signed:
        # INT_MIN is the one dividend where the handler's abs()/sign dance
        # is observable: abs(INT_MIN) wraps back to itself, so the two
        # spellings of truncated division part company there. Whatever the
        # Python engine answers, the tape has to answer the same.
        imin = int(np.iinfo(npdt).min)
        a = np.array([-7, 7, -7, 7, -1, imin, imin, -8, 8, imin, 100, -3],
                     npdt)
        b = np.array([2, 2, -2, -2, 3, 2, -2, 8, -8, 1, -7, 1], npdt)
    else:
        a = np.array([7, 9, 15, 1, 0, 5, 200, 3, 12, 99, 8, 6], npdt)
        b = np.array([2, 4, 4, 3, 5, 5, 7, 8, 12, 10, 3, 1], npdt)
    for op in ("divide", "remainder"):
        check(_binary_mod(op, el), [a, b])


def test_power_and_atan2():
    a = _specials()
    b = np.roll(_specials(), 4)
    check(_binary_mod("power", "f32"), [a, b])
    check(_binary_mod("atan2", "f32"), [a, b])
    # integer power: MLX's own integer exponentiation, exponents >= 0
    ia = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], np.int32)
    ib = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], np.int32)
    check(_binary_mod("power", "i32"), [ia, ib])


@pytest.mark.parametrize("el,npdt", [("i1", np.bool_), ("i8", np.int8),
                                     ("i32", np.int32), ("i64", np.int64),
                                     ("ui8", np.uint8),
                                     ("ui32", np.uint32)])
def test_not(el, npdt):
    t = f"tensor<6x{el}>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.not %a : {t}
    return %0 : {t}""")
    if npdt is np.bool_:
        x = np.array([True, False, True, True, False, False])
    elif np.issubdtype(np.dtype(npdt), np.signedinteger):
        info = np.iinfo(npdt)
        x = np.array([0, -1, 1, 7, info.min, info.max], npdt)
    else:
        info = np.iinfo(npdt)
        x = np.array([0, 1, 2, 7, info.max, info.max // 2], npdt)
    check(mod, [x])


_SHIFTS = ["shift_left", "shift_right_logical", "shift_right_arithmetic"]


@pytest.mark.parametrize("op", _SHIFTS)
@pytest.mark.parametrize("el,npdt", [("i8", np.int8), ("i32", np.int32),
                                     ("i64", np.int64), ("ui8", np.uint8),
                                     ("ui32", np.uint32)])
def test_shift_dynamic_amount(op, el, npdt):
    # XLA saturates a shift by >= the bit width (0, or the sign fill for the
    # arithmetic one); Metal's shifts are mod-width. With a non-constant
    # amount the guard is a compare + select, and negative amounts fall
    # through it deliberately.
    signed = np.issubdtype(np.dtype(npdt), np.signedinteger)
    info = np.iinfo(npdt)
    big = min(200, int(info.max))          # past every width tested here
    if signed:
        a = np.array([-1, 1, -128 if npdt is np.int8 else -12345, 0, 7,
                      info.min, info.max, -3, 42, -42], npdt)
        # -1 is deliberate: _shift_guard leaves negative amounts to
        # whatever the underlying shift does, and both engines must agree.
        amounts = [0, 1, 3, 7, 8, 31, 63, 64, big, -1]
    else:
        a = np.array([0, 1, 3, 255 if npdt is np.uint8 else 65535, 7,
                      info.max, info.max // 3, 9, 42, 128], npdt)
        amounts = [0, 1, 3, 7, 8, 31, 63, 64, big, 5]
    b = np.array(amounts, npdt)
    check(_binary_mod(op, el, n=10), [a, b])


@pytest.mark.parametrize("op", _SHIFTS)
@pytest.mark.parametrize("amount", [0, 1, 7, 31, 32, 100])
def test_shift_static_amount(op, amount):
    # A splat constant amount: tape.py resolves the guard at lowering and
    # emits ONE arm, so both the in-range and the saturated answer have to
    # come out of a program with no select in it.
    t = "tensor<8xi32>"
    mod = _mod([("a", t)], [t], f"""
    %c = stablehlo.constant dense<{amount}> : {t}
    %0 = stablehlo.{op} %a, %c : {t}
    return %0 : {t}""")
    x = np.array([-1, 1, 0, 7, -2147483648, 2147483647, -9999, 12345],
                 np.int32)
    check(mod, [x])


def test_shift_static_amount_broadcast():
    # keras' int4 unpack shifts by broadcast_in_dim(constant): the peephole
    # has to see through the broadcast, and the answer must not change if it
    # does not.
    t = "tensor<2x4xi32>"
    mod = _mod([("a", t)], [t], f"""
    %c = stablehlo.constant dense<4> : tensor<i32>
    %b = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> {t}
    %0 = stablehlo.shift_right_logical %a, %b : {t}
    return %0 : {t}""")
    check(mod, [np.array([[-1, 1, 0, 255], [-256, 4096, 7, -7]], np.int32)])


@pytest.mark.parametrize("el,npdt", [("f32", np.float32), ("f16", np.float16),
                                     ("bf16", ml_dtypes.bfloat16)])
def test_expm1(el, npdt):
    # f32 splits at |x| < 0.25 (MLX's Metal expm1 is fast-math outside it);
    # halves keep mx.expm1 outright. The split point itself is the test.
    t = f"tensor<16x{el}>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.exponential_minus_one %a : {t}
    return %0 : {t}""")
    x = np.array([0.0, -0.0, 1e-8, -1e-8, 0.1, -0.1, 0.2499, -0.2499,
                  0.25, -0.25, 0.2501, 1.0, -1.0, 20.0, -20.0, 100.0],
                 np.float32).astype(npdt)
    check(mod, [x])
    check(mod, [_specials(npdt).repeat(2)[:16]])


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------

_CONVERT = ["f32", "f16", "bf16", "i32", "i8", "ui8", "i64", "i1"]
_CONVERT_NP = {"f32": np.float32, "f16": np.float16,
               "bf16": ml_dtypes.bfloat16, "i32": np.int32, "i8": np.int8,
               "ui8": np.uint8, "i64": np.int64, "i1": np.bool_}


@pytest.mark.parametrize("src", _CONVERT)
def test_convert_lattice(src):
    t = f"tensor<6x{src}>"
    body = []
    results = []
    types = []
    for i, dst in enumerate(_CONVERT):
        ot = f"tensor<6x{dst}>"
        # The add is not decoration: a convert to its own dtype is an MLX
        # no-op that may return the operand itself, and the tape declines
        # such a value in an output position (see the alias tests).
        body.append(f"    %c{i} = stablehlo.convert %a : ({t}) -> {ot}")
        body.append(f"    %{i} = stablehlo.add %c{i}, %c{i} : {ot}")
        results.append(f"%{i}")
        types.append(ot)
    body.append(f"    return {', '.join(results)} : {', '.join(types)}")
    mod = _mod([("a", t)], types, "\n".join(body))
    npdt = _CONVERT_NP[src]
    if src == "i1":
        x = np.array([True, False, True, True, False, False])
    elif np.issubdtype(np.dtype(npdt), np.integer):
        x = np.array([-3, 0, 1, 7, 100, -1]).astype(npdt)
    else:
        x = np.array([-2.5, 0.0, 1.5, 100.25, -0.0, 3.75]).astype(npdt)
    check(mod, [x])


# --------------------------------------------------------------------------
# shape ops
# --------------------------------------------------------------------------


def test_shape_ops():
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t), ("b", t)], [
        "tensor<12xf32>", "tensor<4x3xf32>", "tensor<2x3x4xf32>",
        "tensor<2x2xf32>", "tensor<6x4xf32>"], f"""
    %0 = stablehlo.reshape %a : ({t}) -> tensor<12xf32>
    %1 = stablehlo.transpose %a, dims = [1, 0] : ({t}) -> tensor<4x3xf32>
    %2 = stablehlo.broadcast_in_dim %a, dims = [1, 2] : ({t}) -> tensor<2x3x4xf32>
    %3 = stablehlo.slice %a [0:3:2, 1:4:2] : ({t}) -> tensor<2x2xf32>
    %4 = stablehlo.concatenate %a, %b, dim = 0 : ({t}, {t}) -> tensor<6x4xf32>
    return %0, %1, %2, %3, %4 : tensor<12xf32>, tensor<4x3xf32>, tensor<2x3x4xf32>, tensor<2x2xf32>, tensor<6x4xf32>""")
    check(mod, [_f32(), _f32()])


def test_broadcast_unsorted_dims():
    # broadcast_dimensions out of order: the handler transposes first, and
    # the interim shape is built from the TRANSPOSED operand.
    mod = _mod([("a", "tensor<3x4xf32>")], ["tensor<5x4x3xf32>"], """
    %0 = stablehlo.broadcast_in_dim %a, dims = [2, 1] : (tensor<3x4xf32>) -> tensor<5x4x3xf32>
    return %0 : tensor<5x4x3xf32>""")
    check(mod, [_f32()])


def test_broadcast_scalar_and_size_one():
    mod = _mod([("s", "tensor<f32>"), ("v", "tensor<1x4xf32>")],
               ["tensor<2x3xf32>", "tensor<3x4xf32>"], """
    %0 = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %1 = stablehlo.broadcast_in_dim %v, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<3x4xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<3x4xf32>""")
    check(mod, [np.float32(2.5), _f32((1, 4))])


def test_concatenate_many_and_middle_axis():
    t = "tensor<2x3xf32>"
    mod = _mod([("a", t), ("b", t), ("c", t)], ["tensor<2x9xf32>"], f"""
    %0 = stablehlo.concatenate %a, %b, %c, dim = 1 : ({t}, {t}, {t}) -> tensor<2x9xf32>
    return %0 : tensor<2x9xf32>""")
    check(mod, [_f32((2, 3)), _f32((2, 3)), _f32((2, 3))])


_IOTA = ["f32", "f16", "bf16", "i32", "i64", "ui8"]  # i1 iota is illegal


@pytest.mark.parametrize("el", _IOTA)
def test_iota(el):
    for dim in (0, 1):
        ot = f"tensor<5x3x{el}>"
        mod = _mod([("a", "tensor<f32>")], [ot], f"""
    %0 = stablehlo.iota dim = {dim} : {ot}
    return %0 : {ot}""")
        check(mod, [np.float32(0.0)])


def test_slice_rank0_and_full():
    mod = _mod([("a", "tensor<6xf32>")], ["tensor<1xf32>", "tensor<3xf32>"], """
    %0 = stablehlo.slice %a [5:6] : (tensor<6xf32>) -> tensor<1xf32>
    %1 = stablehlo.slice %a [0:6:2] : (tensor<6xf32>) -> tensor<3xf32>
    return %0, %1 : tensor<1xf32>, tensor<3xf32>""")
    check(mod, [_f32((6,))])


# --------------------------------------------------------------------------
# bitcast / dynamic slice — the kv-cache and RNG shapes
# --------------------------------------------------------------------------


_BITCAST_SAME = [("f32", "i32", np.float32), ("i32", "f32", np.int32),
                 ("ui32", "f32", np.uint32), ("i16", "ui16", np.int16),
                 ("f16", "ui16", np.float16), ("bf16", "ui16",
                                               ml_dtypes.bfloat16)]


@pytest.mark.parametrize("src,dst,npdt", _BITCAST_SAME)
def test_bitcast_same_width(src, dst, npdt):
    t, ot = f"tensor<6x{src}>", f"tensor<6x{dst}>"
    mod = _mod([("a", t)], [ot], f"""
    %0 = stablehlo.bitcast_convert %a : ({t}) -> {ot}
    return %0 : {ot}""")
    if np.issubdtype(np.dtype(npdt), np.integer):
        info = np.iinfo(npdt)
        x = np.array([0, 1, -1 if info.min else 2, info.max, info.min, 7],
                     npdt)
    else:
        x = _specials(npdt)[:6]
    check(mod, [x])


def test_bitcast_narrowing_and_widening():
    # i32 -> i8 adds a trailing dim of 4 (XLA packs the minor-most axis),
    # and the reverse collapses it.
    mod = _mod([("a", "tensor<3xi32>")],
               ["tensor<3x4xi8>", "tensor<3xi32>"], """
    %0 = stablehlo.bitcast_convert %a : (tensor<3xi32>) -> tensor<3x4xi8>
    %1 = stablehlo.bitcast_convert %0 : (tensor<3x4xi8>) -> tensor<3xi32>
    return %0, %1 : tensor<3x4xi8>, tensor<3xi32>""")
    check(mod, [np.array([0, -1, 305419896], np.int32)])


def test_bitcast_rank0_narrowing():
    mod = _mod([("a", "tensor<f32>")], ["tensor<4xui8>"], """
    %0 = stablehlo.bitcast_convert %a : (tensor<f32>) -> tensor<4xui8>
    return %0 : tensor<4xui8>""")
    check(mod, [np.float32(-1.5)])


def test_bitcast_of_a_constant_is_copied():
    # mx.view shares storage, so a bitcast reaching an output would hand
    # out the Program's own constant.
    mod = _mod([("a", "tensor<4xi32>")], ["tensor<4xf32>", "tensor<4xi32>"], """
    %c = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    %0 = stablehlo.bitcast_convert %c : (tensor<4xi32>) -> tensor<4xf32>
    %1 = stablehlo.add %a, %c : tensor<4xi32>
    return %0, %1 : tensor<4xf32>, tensor<4xi32>""")
    check(mod, [np.array([5, 6, 7, 8], np.int32)])
    fresh_outputs(mod, [np.array([5, 6, 7, 8], np.int32)], [0])


@pytest.mark.parametrize("start", [0, 2, 5, -3])
def test_dynamic_slice(start):
    """XLA clamps the start so the window stays inside the operand — the
    negative and past-the-end cases are the whole point."""
    t = "tensor<6x4xf32>"
    ot = "tensor<2x4xf32>"
    mod = _mod([("a", t), ("i", "tensor<i32>")], [ot], f"""
    %z = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %a, %i, %z, sizes = [2, 4] : ({t}, tensor<i32>, tensor<i32>) -> {ot}
    return %0 : {ot}""")
    check(mod, [_f32((6, 4)), np.int32(start)])


@pytest.mark.parametrize("start", [0, 3, 9, -2])
def test_dynamic_update_slice(start):
    t = "tensor<6x4xf32>"
    ut = "tensor<2x4xf32>"
    mod = _mod([("a", t), ("u", ut), ("i", "tensor<i32>")], [t], f"""
    %z = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %a, %u, %i, %z : ({t}, {ut}, tensor<i32>, tensor<i32>) -> {t}
    return %0 : {t}""")
    check(mod, [_f32((6, 4)), _f32((2, 4)), np.int32(start)])


def test_dynamic_slice_unsigned_and_i64_indices():
    t = "tensor<8xf32>"
    ot = "tensor<3xf32>"
    mod = _mod([("a", t), ("i", "tensor<ui32>"), ("j", "tensor<i64>")],
               [ot, ot], f"""
    %0 = stablehlo.dynamic_slice %a, %i, sizes = [3] : ({t}, tensor<ui32>) -> {ot}
    %1 = stablehlo.dynamic_slice %a, %j, sizes = [3] : ({t}, tensor<i64>) -> {ot}
    return %0, %1 : {ot}, {ot}""")
    check(mod, [_f32((8,)), np.uint32(6), np.int64(1)])


def test_dynamic_slice_rank0_operand_passes_through():
    """No index operands: there is nothing to slice, and the handler hands
    the operand back — so the output is a copy of the argument."""
    t = "tensor<f32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.dynamic_slice %a, sizes = [] : ({t}) -> {t}
    return %0 : {t}""")
    check(mod, [np.float32(2.5)])
    fresh_outputs(mod, [np.float32(2.5)], [0], args_too=[0])


def test_dynamic_update_slice_rank0_operand_passes_through():
    t = "tensor<f32>"
    mod = _mod([("a", t), ("u", t)], [t], f"""
    %0 = stablehlo.dynamic_update_slice %a, %u : ({t}, {t}) -> {t}
    return %0 : {t}""")
    check(mod, [np.float32(2.5), np.float32(-1.0)])
    fresh_outputs(mod, [np.float32(2.5), np.float32(-1.0)], [0],
                  args_too=[1])


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------


def test_constants():
    mod = _mod([("a", "tensor<2x3xf32>")], [
        "tensor<2x3xf32>", "tensor<2x3xf32>", "tensor<f32>"], """
    %splat = stablehlo.constant dense<2.500000e+00> : tensor<2x3xf32>
    %dense = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    %scalar = stablehlo.constant dense<0.100000001> : tensor<f32>
    %0 = stablehlo.multiply %a, %splat : tensor<2x3xf32>
    %1 = stablehlo.add %a, %dense : tensor<2x3xf32>
    %2 = stablehlo.multiply %scalar, %scalar : tensor<f32>
    return %0, %1, %2 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<f32>""")
    check(mod, [_f32((2, 3))])


def test_bf16_constants():
    # bf16 constants cross the MLIR bindings only through the text/hex
    # decoder (_ir.dense_to_np); the tape reuses that handler verbatim, and
    # the hex splat is the form that once decoded -inf as 65536.
    mod = _mod([("a", "tensor<2x3xbf16>")], [
        "tensor<2x3xbf16>", "tensor<2x3xbf16>", "tensor<2x3xbf16>"], """
    %splat = stablehlo.constant dense<2.500000e+00> : tensor<2x3xbf16>
    %hex = stablehlo.constant dense<0xFF80> : tensor<2x3xbf16>
    %dense = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xbf16>
    %0 = stablehlo.multiply %a, %splat : tensor<2x3xbf16>
    %1 = stablehlo.maximum %a, %hex : tensor<2x3xbf16>
    %2 = stablehlo.add %a, %dense : tensor<2x3xbf16>
    return %0, %1, %2 : tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>""")
    check(mod, [_f32((2, 3)).astype(ml_dtypes.bfloat16)])


def test_integer_and_bool_constants():
    mod = _mod([("a", "tensor<4xi32>")], ["tensor<4xi32>", "tensor<4xi1>"], """
    %c = stablehlo.constant dense<[1, -2, 3, -4]> : tensor<4xi32>
    %t = stablehlo.constant dense<true> : tensor<4xi1>
    %0 = stablehlo.add %a, %c : tensor<4xi32>
    %1 = stablehlo.compare LT, %a, %c : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %2 = stablehlo.and %1, %t : tensor<4xi1>
    return %0, %2 : tensor<4xi32>, tensor<4xi1>""")
    check(mod, [np.array([5, -5, 0, 9], np.int32)])


# --------------------------------------------------------------------------
# reduce
# --------------------------------------------------------------------------

_MONOIDS = [("stablehlo.add", "f32"), ("stablehlo.multiply", "f32"),
            ("stablehlo.maximum", "f32"), ("stablehlo.minimum", "f32"),
            ("stablehlo.add", "i32"), ("stablehlo.maximum", "i32"),
            ("stablehlo.or", "i1"), ("stablehlo.and", "i1"),
            ("stablehlo.add", "i1")]


@pytest.mark.parametrize("body,el", _MONOIDS)
def test_reduce_monoids(body, el):
    t = f"tensor<3x4x{el}>"
    it = f"tensor<{el}>"
    ot = f"tensor<3x{el}>"
    mod = _mod([("a", t), ("init", it)], [ot], f"""
    %0 = stablehlo.reduce(%a init: %init) applies {body} across dimensions = [1] : ({t}, {it}) -> {ot}
    return %0 : {ot}""")
    if el == "f32":
        x = _f32()
        init = np.float32(0.5)  # deliberately NOT the monoid identity
    elif el == "i32":
        x = RNG.integers(-9, 9, (3, 4)).astype(np.int32)
        init = np.int32(3)
    else:
        x = RNG.integers(0, 2, (3, 4)) > 0
        init = np.bool_(True)
    check(mod, [x, init])


def test_reduce_multiple_and_all_dims():
    t = "tensor<2x3x4xf32>"
    mod = _mod([("a", t), ("init", "tensor<f32>")],
               ["tensor<3xf32>", "tensor<f32>"], f"""
    %0 = stablehlo.reduce(%a init: %init) applies stablehlo.add across dimensions = [0, 2] : ({t}, tensor<f32>) -> tensor<3xf32>
    %1 = stablehlo.reduce(%a init: %init) applies stablehlo.maximum across dimensions = [0, 1, 2] : ({t}, tensor<f32>) -> tensor<f32>
    return %0, %1 : tensor<3xf32>, tensor<f32>""")
    check(mod, [_f32((2, 3, 4)), np.float32(1.25)])


def test_reduce_no_dimensions():
    # dimensions = [] is the identity fold: combine with the init and stop.
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t), ("init", "tensor<f32>")], [t], f"""
    %0 = stablehlo.reduce(%a init: %init) applies stablehlo.add across dimensions = [] : ({t}, tensor<f32>) -> {t}
    return %0 : {t}""")
    check(mod, [_f32(), np.float32(2.0)])


def test_reduce_zero_size():
    # MLX's reducers crash on empty inputs; both engines must take the
    # "an empty fold is the init" path (and the zeros path when only a
    # KEPT dim is empty).
    mod = _mod([("a", "tensor<3x0xf32>"), ("b", "tensor<0x4xf32>"),
                ("init", "tensor<f32>")],
               ["tensor<3xf32>", "tensor<0xf32>"], """
    %0 = stablehlo.reduce(%a init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<3x0xf32>, tensor<f32>) -> tensor<3xf32>
    %1 = stablehlo.reduce(%b init: %init) applies stablehlo.add across dimensions = [1] : (tensor<0x4xf32>, tensor<f32>) -> tensor<0xf32>
    return %0, %1 : tensor<3xf32>, tensor<0xf32>""")
    check(mod, [np.zeros((3, 0), np.float32), np.zeros((0, 4), np.float32),
                np.float32(-7.5)])


def test_reduce_nan_and_inf():
    t = "tensor<2x6xf32>"
    mod = _mod([("a", t), ("init", "tensor<f32>")],
               ["tensor<2xf32>", "tensor<2xf32>"], f"""
    %0 = stablehlo.reduce(%a init: %init) applies stablehlo.maximum across dimensions = [1] : ({t}, tensor<f32>) -> tensor<2xf32>
    %1 = stablehlo.reduce(%a init: %init) applies stablehlo.add across dimensions = [1] : ({t}, tensor<f32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>""")
    x = _specials().reshape(2, 6)
    check(mod, [x, np.float32(0.0)])


# --------------------------------------------------------------------------
# the (values, indices) reduce jax lowers argmax/argmin to
# --------------------------------------------------------------------------


def _argpair_mod(direction, el="f32", shape=(3, 4), dim=1, idx_el="i32"):
    """The exact body shape jax emits for argmax/argmin."""
    dims = list(shape)
    out = [d for i, d in enumerate(dims) if i != dim]
    t = "tensor<" + "x".join(str(d) for d in dims) + f"x{el}>"
    it = "tensor<" + "x".join(str(d) for d in dims) + f"x{idx_el}>"
    ot = "tensor<" + "".join(f"{d}x" for d in out) + f"{el}>"
    oit = "tensor<" + "".join(f"{d}x" for d in out) + f"{idx_el}>"
    if el == "f32":
        init = "0xFF800000" if direction in ("GT", "GE") else "0x7F800000"
    elif el == "i1":
        init = "false" if direction in ("GT", "GE") else "true"
    else:
        init = "0"
    return _mod([("a", t), ("idx", it)], [ot, oit], f"""
    %vi = stablehlo.constant dense<{init}> : tensor<{el}>
    %ii = stablehlo.constant dense<0> : tensor<{idx_el}>
    %0:2 = "stablehlo.reduce"(%a, %idx, %vi, %ii) <{{dimensions = array<i64: {dim}>}}> ({{
    ^bb0(%av: tensor<{el}>, %ai: tensor<{idx_el}>, %bv: tensor<{el}>, %bi: tensor<{idx_el}>):
      %p = stablehlo.compare {direction}, %av, %bv : (tensor<{el}>, tensor<{el}>) -> tensor<i1>
      %v = stablehlo.select %p, %av, %bv : tensor<i1>, tensor<{el}>
      %i = stablehlo.select %p, %ai, %bi : tensor<i1>, tensor<{idx_el}>
      stablehlo.return %v, %i : tensor<{el}>, tensor<{idx_el}>
    }}) : ({t}, {it}, tensor<{el}>, tensor<{idx_el}>) -> ({ot}, {oit})
    return %0#0, %0#1 : {ot}, {oit}"""), t, it


def _iota_idx(shape, dim, dtype=np.int32):
    return np.broadcast_to(
        np.arange(shape[dim], dtype=dtype).reshape(
            [shape[dim] if i == dim else 1 for i in range(len(shape))]),
        shape).astype(dtype)


@pytest.mark.parametrize("direction", ["GT", "GE", "LT", "LE"])
def test_argpair_reduce(direction):
    mod, _, _ = _argpair_mod(direction)
    x = _f32((3, 4))
    check(mod, [x, _iota_idx((3, 4), 1)])


def test_argpair_reduce_ties_and_nans():
    # XLA/numpy: a NaN wins argmax AND argmin and the FIRST NaN's index is
    # the answer; MLX's own argmax skips NaNs. Ties go to the lowest index.
    mod, _, _ = _argpair_mod("GT", shape=(4, 5))
    x = np.array([
        [1.0, 3.0, 3.0, 2.0, 0.0],          # tie at the max
        [np.nan, 1.0, np.nan, 5.0, -1.0],   # two NaNs
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.0, -0.0, 0.0, -0.0, 7.0],
    ], np.float32)
    check(mod, [x, _iota_idx((4, 5), 1)])
    modmin, _, _ = _argpair_mod("LT", shape=(4, 5))
    check(modmin, [x, _iota_idx((4, 5), 1)])


def test_argpair_reduce_dim0_and_ints():
    mod, _, _ = _argpair_mod("GE", shape=(4, 3), dim=0)
    check(mod, [_f32((4, 3)), _iota_idx((4, 3), 0)])
    # An integer values operand skips the NaN machinery entirely.
    modi, _, _ = _argpair_mod("GT", el="i32", shape=(3, 5))
    x = RNG.integers(-9, 9, (3, 5)).astype(np.int32)
    check(modi, [x, _iota_idx((3, 5), 1)])


def test_argpair_reduce_zero_size_batch():
    # jax forbids argmax over an empty reduced axis, so only a batch dim can
    # be zero — and MLX's reducers crash on empties, so both engines take
    # the short-circuit.
    mod, _, _ = _argpair_mod("GT", shape=(0, 4))
    check(mod, [np.zeros((0, 4), np.float32), np.zeros((0, 4), np.int32)])


def test_argpair_reduce_i64_indices():
    mod, _, _ = _argpair_mod("GT", idx_el="i64")
    check(mod, [_f32((3, 4)), _iota_idx((3, 4), 1, np.int64)])


def test_argpair_on_bools_takes_the_generic_path():
    # dtypes.is_bool(values) sends the Python handler to _generic_reduce,
    # which walks the body block; the tape must not shortcut past it into
    # the argmax pair, and now lowers the body as a sub-Program instead.
    mod, _, _ = _argpair_mod("GT", el="i1", shape=(3, 4))
    x = RNG.integers(0, 2, (3, 4)) > 0
    check(mod, [x, _iota_idx((3, 4), 1)])


def test_pair_reduce_without_a_compare_is_generic():
    # A two-operand reduce whose body is arithmetic is a generic variadic
    # fold, not an argmax — the same distinction on both engines.
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t), ("b", t)], ["tensor<3xf32>", "tensor<3xf32>"], f"""
    %i0 = stablehlo.constant dense<0.0> : tensor<f32>
    %i1 = stablehlo.constant dense<1.0> : tensor<f32>
    %0:2 = "stablehlo.reduce"(%a, %b, %i0, %i1) <{{dimensions = array<i64: 1>}}> ({{
    ^bb0(%av: tensor<f32>, %ai: tensor<f32>, %bv: tensor<f32>, %bi: tensor<f32>):
      %s = stablehlo.add %av, %bv : tensor<f32>
      %m = stablehlo.multiply %ai, %bi : tensor<f32>
      stablehlo.return %s, %m : tensor<f32>, tensor<f32>
    }}) : ({t}, {t}, tensor<f32>, tensor<f32>) -> (tensor<3xf32>, tensor<3xf32>)
    return %0#0, %0#1 : tensor<3xf32>, tensor<3xf32>""")
    check(mod, [_f32(), _f32()])


# --------------------------------------------------------------------------
# dot_general
# --------------------------------------------------------------------------


def test_dot_plain_matmul():
    mod = _mod([("a", "tensor<3x4xf32>"), ("b", "tensor<4x5xf32>")],
               ["tensor<3x5xf32>"], """
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %0 : tensor<3x5xf32>""")
    check(mod, [_f32((3, 4)), _f32((4, 5))])


def test_dot_batched():
    mod = _mod([("a", "tensor<2x3x4xf32>"), ("b", "tensor<2x4x5xf32>")],
               ["tensor<2x3x5xf32>"], """
    %0 = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
    return %0 : tensor<2x3x5xf32>""")
    check(mod, [_f32((2, 3, 4)), _f32((2, 4, 5))])


def test_dot_contracting_only():
    mod = _mod([("a", "tensor<3x4xf32>"), ("b", "tensor<3x4xf32>")],
               ["tensor<f32>"], """
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [0, 1] x [0, 1] : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<f32>
    return %0 : tensor<f32>""")
    check(mod, [_f32(), _f32()])


def test_dot_high_rank_and_transposed_dims():
    # Free dims on both sides, a contraction that is not the last axis:
    # exercises the transpose/reshape bookkeeping rather than the matmul.
    mod = _mod([("a", "tensor<2x3x4x5xf32>"), ("b", "tensor<4x2x6xf32>")],
               ["tensor<2x3x5x6xf32>"], """
    %0 = stablehlo.dot_general %a, %b, batching_dims = [0] x [1], contracting_dims = [2] x [0] : (tensor<2x3x4x5xf32>, tensor<4x2x6xf32>) -> tensor<2x3x5x6xf32>
    return %0 : tensor<2x3x5x6xf32>""")
    check(mod, [_f32((2, 3, 4, 5)), _f32((4, 2, 6))])


def test_dot_no_contraction():
    # An outer product: K == 1 after the reshape, nothing to sum.
    mod = _mod([("a", "tensor<3xf32>"), ("b", "tensor<4xf32>")],
               ["tensor<3x4xf32>"], """
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [] x [] : (tensor<3xf32>, tensor<4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>""")
    check(mod, [_f32((3,)), _f32((4,))])


def test_dot_zero_sized():
    # mx.matmul with an empty M/N segfaults on host copy; both engines
    # short-circuit to zeros.
    mod = _mod([("a", "tensor<0x4xf32>"), ("b", "tensor<4x5xf32>")],
               ["tensor<0x5xf32>"], """
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<0x4xf32>, tensor<4x5xf32>) -> tensor<0x5xf32>
    return %0 : tensor<0x5xf32>""")
    check(mod, [np.zeros((0, 4), np.float32), _f32((4, 5))])


@pytest.mark.parametrize("el,npdt", [("f16", np.float16),
                                     ("bf16", ml_dtypes.bfloat16)])
def test_dot_halves(el, npdt):
    mod = _mod([("a", f"tensor<3x4x{el}>"), ("b", f"tensor<4x5x{el}>")],
               [f"tensor<3x5x{el}>"], f"""
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<3x4x{el}>, tensor<4x5x{el}>) -> tensor<3x5x{el}>
    return %0 : tensor<3x5x{el}>""")
    check(mod, [_f32((3, 4)).astype(npdt), _f32((4, 5)).astype(npdt)])


def _int_dot_mod(el, out_el, m=3, k=4, n=5, rel=None):
    rel = rel or el
    return _mod([("a", f"tensor<{m}x{k}x{el}>"),
                 ("b", f"tensor<{k}x{n}x{rel}>")],
                [f"tensor<{m}x{n}x{out_el}>"], f"""
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<{m}x{k}x{el}>, tensor<{k}x{n}x{rel}>) -> tensor<{m}x{n}x{out_el}>
    return %0 : tensor<{m}x{n}x{out_el}>""")


@pytest.mark.parametrize("el,out_el,npdt", [
    ("i8", "i32", np.int8),      # the exact-f32 chunk path (chunk 1024)
    ("i8", "i8", np.int8),       # ...wrapping to 8 bits at the end
    ("ui8", "i32", np.uint8),    # chunk 256
    ("i32", "i32", np.int32),    # no chunk: the int64 outer product
    ("i64", "i64", np.int64),
    ("i16", "i32", np.int16),
])
def test_int_dot(el, out_el, npdt):
    info = np.iinfo(npdt)
    lo, hi = max(info.min, -50), min(info.max, 50)
    a = RNG.integers(lo, hi, (3, 4)).astype(npdt)
    b = RNG.integers(lo, hi, (4, 5)).astype(npdt)
    check(_int_dot_mod(el, out_el), [a, b])


def test_int_dot_mixed_operand_signs():
    # i8 x ui8 halves the exact chunk (512): _exact_f32_chunk reads BOTH
    # operand dtypes, not the result's.
    mod = _int_dot_mod("i8", "i32", rel="ui8")
    a = RNG.integers(-128, 128, (3, 4)).astype(np.int8)
    b = RNG.integers(0, 256, (4, 5)).astype(np.uint8)
    check(mod, [a, b])


def test_int_dot_multiple_chunks():
    """K past the exact chunk size: three f32 matmuls, summed as integers.

    The data is deliberately irregular. Extreme-but-uniform operands make
    every partial sum a large power of two times something small, which f32
    happens to hold exactly — so an accumulator that wrongly stayed in f32
    would still agree. These totals (~3.5e7, odd) do not survive f32.
    """
    k = 2600
    mod = _int_dot_mod("i8", "i32", m=2, k=k, n=3)
    a = np.full((2, k), 127, np.int8)
    a[1] = -128
    a[0, ::13] = -128
    b = np.full((k, 3), 127, np.int8)
    b[::7, 0] = 126
    b[::5, 1] = -128
    b[3::11, 2] = 1
    check(mod, [a, b])


def test_int_dot_zero_contraction_and_batched():
    # K == 0 keeps the outer-product arm (chunking needs prod(k) != 0).
    mod = _mod([("a", "tensor<3x0xi32>"), ("b", "tensor<0x5xi32>")],
               ["tensor<3x5xi32>"], """
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<3x0xi32>, tensor<0x5xi32>) -> tensor<3x5xi32>
    return %0 : tensor<3x5xi32>""")
    check(mod, [np.zeros((3, 0), np.int32), np.zeros((0, 5), np.int32)])
    batched = _mod([("a", "tensor<2x3x4xi8>"), ("b", "tensor<2x4x5xi8>")],
                   ["tensor<2x3x5xi32>"], """
    %0 = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xi8>, tensor<2x4x5xi8>) -> tensor<2x3x5xi32>
    return %0 : tensor<2x3x5xi32>""")
    check(batched, [RNG.integers(-50, 50, (2, 3, 4)).astype(np.int8),
                    RNG.integers(-50, 50, (2, 4, 5)).astype(np.int8)])


def test_dot_mixed_operand_dtype():
    # f16 operands, f32 accumulator: the handler casts both to the RESULT
    # dtype before the matmul.
    mod = _mod([("a", "tensor<3x4xf16>"), ("b", "tensor<4x5xf16>")],
               ["tensor<3x5xf32>"], """
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<3x4xf16>, tensor<4x5xf16>) -> tensor<3x5xf32>
    return %0 : tensor<3x5xf32>""")
    check(mod, [_f32((3, 4)).astype(np.float16),
                _f32((4, 5)).astype(np.float16)])


# --------------------------------------------------------------------------
# func.call / stablehlo.composite inlining
# --------------------------------------------------------------------------
#
# jax 0.11 wraps jnp.where / jnp.clip / jnp.round in private helper
# functions, so a call in main is not exotic — it was the single biggest
# decline family before this batch. The callee's block is spliced into the
# tape at lowering time: its block arguments alias the call's operand slots
# and its results alias whatever it returned, so the C++ side never sees a
# call at all.


def _with_funcs(main_params, main_results, main_body, funcs, name="calls"):
    args = ", ".join(f"%{n}: {t}" for n, t in main_params)
    outs = ", ".join(main_results)
    return (f"module @{name} {{\n"
            f"  func.func public @main({args}) -> ({outs}) {{\n"
            f"{main_body}\n  }}\n"
            + "\n".join(funcs) + "\n}\n").encode()


def _helper(name, params, result, body):
    args = ", ".join(f"%{n}: {t}" for n, t in params)
    return (f"  func.func private @{name}({args}) -> {result} {{\n"
            f"{body}\n  }}")


def test_func_call_inlined():
    t = "tensor<3x4xf32>"
    mod = _with_funcs(
        [("a", t)], [t],
        f"    %0 = call @helper(%a) : ({t}) -> {t}\n"
        f"    return %0 : {t}",
        [_helper("helper", [("a", t)], t,
                 f"    %0 = stablehlo.tanh %a : {t}\n"
                 f"    %1 = stablehlo.sine %0 : {t}\n"
                 f"    return %1 : {t}")])
    ex = check(mod, [_f32()])
    # The callee's ops are the tape's ops; the call itself costs nothing.
    assert ex._native_prog.num_ops == 2


def test_func_call_nested_and_shared_operands():
    t = "tensor<3x4xf32>"
    mod = _with_funcs(
        [("a", t), ("b", t)], [t, t],
        f"    %0 = call @outer(%a, %b) : ({t}, {t}) -> {t}\n"
        f"    %1 = stablehlo.multiply %0, %a : {t}\n"
        f"    return %0, %1 : {t}, {t}",
        [_helper("outer", [("x", t), ("y", t)], t,
                 f"    %s = stablehlo.add %x, %y : {t}\n"
                 f"    %0 = call @inner(%s) : ({t}) -> {t}\n"
                 f"    return %0 : {t}"),
         _helper("inner", [("z", t)], t,
                 f"    %0 = stablehlo.tanh %z : {t}\n"
                 f"    return %0 : {t}")])
    ex = check(mod, [_f32(), _f32()])
    assert ex._native_prog.num_ops == 3


def test_func_call_same_callee_twice():
    """A callee inlined twice re-binds its OWN values a second time.

    Slot numbers used to be `len(self.slots)`, which does not grow when a
    key is overwritten — so the second inline handed the callee's values
    slots the first one had already used. The helper reads %p again AFTER
    %q exists, which is the shape that makes such a collision visible
    (a collision between a value and its immediate consumer is benign).
    """
    t = "tensor<4xf32>"
    mod = _with_funcs(
        [("a", t), ("b", t)], [t],
        f"    %0 = call @h(%a) : ({t}) -> {t}\n"
        f"    %1 = stablehlo.multiply %0, %b : {t}\n"
        f"    %2 = call @h(%1) : ({t}) -> {t}\n"
        f"    %3 = stablehlo.subtract %2, %0 : {t}\n"
        f"    %4 = call @h(%3) : ({t}) -> {t}\n"
        f"    return %4 : {t}",
        [_helper("h", [("x", t)], t,
                 f"    %p = stablehlo.multiply %x, %x : {t}\n"
                 f"    %q = stablehlo.add %p, %x : {t}\n"
                 f"    %r = stablehlo.subtract %q, %p : {t}\n"
                 f"    return %r : {t}")])
    ex = check(mod, [_f32((4,)), _f32((4,))])
    assert ex._native_prog.num_ops == 3 * 3 + 2


def test_func_call_multiple_results():
    t = "tensor<4xf32>"
    mod = _with_funcs(
        [("a", t), ("b", t)], [t, t],
        f"    %0:2 = call @two(%a, %b) : ({t}, {t}) -> ({t}, {t})\n"
        f"    %s = stablehlo.add %0#0, %0#1 : {t}\n"
        f"    return %s, %0#1 : {t}, {t}",
        [(f"  func.func private @two(%x: {t}, %y: {t}) -> ({t}, {t}) {{\n"
          f"    %p = stablehlo.multiply %x, %y : {t}\n"
          f"    %d = stablehlo.subtract %x, %y : {t}\n"
          f"    return %p, %d : {t}, {t}\n  }}")])
    check(mod, [_f32((4,)), _f32((4,))])


def test_func_call_constant_inside_callee():
    # A constant defined in the callee still belongs to the Program for the
    # executable's life: returning it (even through a broadcast) aliases.
    t = "tensor<3xf32>"
    mod = _with_funcs(
        [("a", t)], [t],
        f"    %0 = call @h(%a) : ({t}) -> {t}\n"
        f"    return %0 : {t}",
        [_helper("h", [("x", t)], t,
                 f"    %c = stablehlo.constant dense<[1.0, 2.0, 3.0]> : {t}\n"
                 f"    %0 = stablehlo.add %x, %c : {t}\n"
                 f"    return %0 : {t}")])
    check(mod, [_f32((3,))])


def test_composite_inlined():
    # stablehlo.composite runs its `decomposition` symbol, exactly as
    # func.call runs its callee (ops/control.py), so it inlines the same way.
    t = "tensor<3x4xf32>"
    mod = (f"module @comp {{\n"
           f"  func.func public @main(%a: {t}) -> ({t}) {{\n"
           f'    %0 = stablehlo.composite "test.gelu" %a '
           f"{{decomposition = @dec}} : ({t}) -> {t}\n"
           f"    return %0 : {t}\n  }}\n"
           f"  func.func private @dec(%x: {t}) -> {t} {{\n"
           f"    %0 = stablehlo.tanh %x : {t}\n"
           f"    return %0 : {t}\n  }}\n}}\n").encode()
    check(mod, [_f32()])


def test_call_forwarding_an_argument_is_copied():
    """`call @identity(%a)` hands back main's own argument array.

    engine.execute copies statically-forwarded outputs, but its notion of
    "forwarded" is main's terminator naming a block argument — an OpResult
    that happens to hold the same array is invisible to it, and nanobind
    defeats the dynamic id() pass. So the tape marks it and C++ copies.
    """
    t = "tensor<3x4xf32>"
    mod = _with_funcs(
        [("a", t)], [t, t],
        f"    %0 = call @ident(%a) : ({t}) -> {t}\n"
        f"    %1 = stablehlo.tanh %a : {t}\n"
        f"    return %1, %0 : {t}, {t}",
        [_helper("ident", [("x", t)], t, f"    return %x : {t}")])
    check(mod, [_f32()])
    fresh_outputs(mod, [_f32()], [1], args_too=[0])


def test_decline_call_with_an_unsupported_body():
    # The decline is WHOLESALE: one unported op anywhere, callee included,
    # and the whole program keeps the Python engine. (Convolution is the
    # stand-in for "an op with no opcode"; popcnt was, and then fft, until
    # each was ported.)
    t = "tensor<1x1x5xf32>"
    mod = _with_funcs(
        [("a", t)], [t],
        f"    %0 = call @h(%a) : ({t}) -> {t}\n"
        f"    return %0 : {t}",
        [_helper("h", [("x", t)], t,
                 "    %k = stablehlo.constant dense<1.0> : "
                 "tensor<1x1x3xf32>\n"
                 f"    %0 = {_CONV}\n"
                 f"    return %0 : {t}")])
    check(mod, [_f32((1, 1, 5))], lowered=False)


def test_decline_recursive_call():
    # Inlining has no fixed point here; the Python engine's own recursive
    # run_func keeps it (and would diverge too, so this checks the LOWERING
    # only).
    t = "tensor<4xf32>"
    mod = _with_funcs(
        [("a", t)], [t],
        f"    %0 = call @rec(%a) : ({t}) -> {t}\n"
        f"    return %0 : {t}",
        [_helper("rec", [("x", t)], t,
                 f"    %0 = stablehlo.tanh %x : {t}\n"
                 f"    %1 = call @rec(%0) : ({t}) -> {t}\n"
                 f"    return %1 : {t}")])
    declines(mod)


# --------------------------------------------------------------------------
# declines — one per family. Each must still compute the right answer.
# --------------------------------------------------------------------------


# An op the Python engine runs and the tape has no opcode for. Written as
# a same-shape convolution against a unit kernel so the reference value is
# obvious; what is under test is the opcode registry as the gate.
_CONV = (
    "stablehlo.convolution(%x, %k) "
    "dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], "
    "window = {stride = [1], pad = [[1, 1]], lhs_dilate = [1], "
    "rhs_dilate = [1], reverse = [false]} "
    "{batch_group_count = 1 : i64, feature_group_count = 1 : i64} : "
    "(tensor<1x1x5xf32>, tensor<1x1x3xf32>) -> tensor<1x1x5xf32>")


def test_decline_unsupported_op():
    t = "tensor<1x1x5xf32>"
    mod = _mod([("x", t)], [t], f"""
    %k = stablehlo.constant dense<1.0> : tensor<1x1x3xf32>
    %0 = {_CONV}
    return %0 : {t}""")
    check(mod, [_f32((1, 1, 5))], lowered=False)


def test_non_monoid_reduce_body_takes_the_generic_path():
    # `subtract` is in neither reducer table, so ops/reduction.py folds the
    # reduced axis pairwise with the body itself; the tape lowers the body
    # into a sub-Program and calls it once per halving round.
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], ["tensor<3xf32>"], f"""
    %init = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = "stablehlo.reduce"(%a, %init) <{{dimensions = array<i64: 1>}}> ({{
    ^bb0(%x: tensor<f32>, %y: tensor<f32>):
      %s = stablehlo.subtract %x, %y : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }}) : ({t}, tensor<f32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>""")
    check(mod, [_f32()])


def test_control_flow_lowers_but_a_forwarding_branch_is_copied():
    # M3 lowers if/case; this one's branches BOTH hand back a value the
    # Program owns (main's argument, and a constant), so the result is
    # copied. tests/test_native_control.py covers control flow properly.
    t = "tensor<f32>"
    mod = _mod([("a", t)], [t], f"""
    %c = stablehlo.constant dense<0.0> : tensor<f32>
    %p = stablehlo.compare GT, %a, %c : ({t}, {t}) -> tensor<i1>
    %0 = "stablehlo.if"(%p) ({{
      stablehlo.return %a : {t}
    }}, {{
      stablehlo.return %c : {t}
    }}) : (tensor<i1>) -> {t}
    return %0 : {t}""")
    check(mod, [np.float32(1.5)])
    fresh_outputs(mod, [np.float32(1.5)], [0], args_too=[0])


def test_complex_add():
    # complex64 joined the dtype table with the tail sweep.
    t = "tensor<4xcomplex<f32>>"
    mod = _mod([("a", t), ("b", t)], [t], f"""
    %0 = stablehlo.add %a, %b : {t}
    return %0 : {t}""")
    z = (RNG.standard_normal(4) + 1j * RNG.standard_normal(4)).astype(
        np.complex64)
    check(mod, [z, z * 2])


def test_decline_f64_passthrough():
    # f64 pass-through is legal (stored f32) but never native: the tape's
    # dtype table is the bit-exact storage set.
    mod = _mod([("a", "tensor<4xf64>")], ["tensor<4xf32>"], """
    %0 = stablehlo.convert %a : (tensor<4xf64>) -> tensor<4xf32>
    return %0 : tensor<4xf32>""")
    check(mod, [np.array([1.5, -2.25, 3e300, 0.1], np.float64)],
          lowered=False)


def test_decline_emulated_dtype():
    mod = _mod([("a", "tensor<6xf32>")], ["tensor<6xf32>"], """
    %0 = stablehlo.convert %a : (tensor<6xf32>) -> tensor<6xf8E4M3FN>
    %1 = stablehlo.convert %0 : (tensor<6xf8E4M3FN>) -> tensor<6xf32>
    return %1 : tensor<6xf32>""")
    check(mod, [np.array([0.5, -1.25, 300.0, 1e-9, 0.0, 7.0], np.float32)],
          lowered=False)


def test_variadic_reduce_is_generic():
    # Three operands: past the argmax pair, into _generic_reduce's pairwise
    # halving, which the tape runs as a sub-Program per round.
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t), ("b", t), ("c", t)],
               ["tensor<3xf32>", "tensor<3xf32>", "tensor<3xf32>"], f"""
    %i = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "stablehlo.reduce"(%a, %b, %c, %i, %i, %i) <{{dimensions = array<i64: 1>}}> ({{
    ^bb0(%a0: tensor<f32>, %a1: tensor<f32>, %a2: tensor<f32>, %b0: tensor<f32>, %b1: tensor<f32>, %b2: tensor<f32>):
      %s0 = stablehlo.add %a0, %b0 : tensor<f32>
      %s1 = stablehlo.add %a1, %b1 : tensor<f32>
      %s2 = stablehlo.maximum %a2, %b2 : tensor<f32>
      stablehlo.return %s0, %s1, %s2 : tensor<f32>, tensor<f32>, tensor<f32>
    }}) : ({t}, {t}, {t}, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>)
    return %0#0, %0#1, %0#2 : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>""")
    check(mod, [_f32(), _f32(), _f32()])


def test_totalorder_compare():
    t = "tensor<6xf32>"
    ot = "tensor<6xi1>"
    mod = _mod([("a", t), ("b", t)], [ot], f"""
    %0 = stablehlo.compare LT, %a, %b, TOTALORDER : ({t}, {t}) -> {ot}
    return %0 : {ot}""")
    x = np.array([np.nan, -np.nan, 0.0, -0.0, np.inf, -np.inf], np.float32)
    check(mod, [x, np.roll(x, 2)])


def test_constant_output_is_copied():
    # A constant returned directly would alias across calls: the Program
    # holds it for the executable's life. The tape says so statically and
    # native/program.cc hands out a copy (XLA's no-alias contract).
    t = "tensor<3xf32>"
    mod = _mod([("a", t)], [t, t], f"""
    %c = stablehlo.constant dense<[1.0, 2.0, 3.0]> : {t}
    %0 = stablehlo.add %a, %c : {t}
    return %0, %c : {t}, {t}""")
    check(mod, [_f32((3,))])
    fresh_outputs(mod, [_f32((3,))], [1])


def test_constant_view_output_is_copied():
    # A splat constant is a ONE-element buffer the tape holds forever;
    # broadcasting it to an output would hand every call a view of that
    # same storage.
    mod = _mod([("a", "tensor<3xf32>")],
               ["tensor<3xf32>", "tensor<2x3xf32>"], """
    %c = stablehlo.constant dense<4.000000e+00> : tensor<3xf32>
    %b = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
    %0 = stablehlo.add %a, %c : tensor<3xf32>
    return %0, %b : tensor<3xf32>, tensor<2x3xf32>""")
    check(mod, [_f32((3,))])
    fresh_outputs(mod, [_f32((3,))], [1])


def test_identity_forwarded_argument_is_copied():
    # A reshape to the SAME shape may hand back the operand object itself;
    # engine.execute catches that by id() on the Python path only, and
    # nanobind's fresh wrappers make it invisible there.
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.reshape %a : ({t}) -> {t}
    return %0 : {t}""")
    check(mod, [_f32()])
    fresh_outputs(mod, [_f32()], [0], args_too=[0])


def test_argument_forwarded_directly_still_lowers():
    # main returning an argument is caught STATICALLY (forwarded_outputs),
    # engine-independently — so it must NOT decline.
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], [t, t], f"""
    %0 = stablehlo.tanh %a : {t}
    return %0, %a : {t}, {t}""")
    ex = check(mod, [_f32()])
    assert ex.forwarded_outputs == [(1, 0)]


def test_duplicate_outputs_are_distinct_buffers():
    # One slot read twice: nanobind hands out two wrappers of one array, so
    # the C++ side copies (XLA forbids two outputs sharing a buffer).
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], [t, t], f"""
    %0 = stablehlo.tanh %a : {t}
    return %0, %0 : {t}, {t}""")
    with _native_engine():
        ex = engine.compile_program(mod, "mlir")
        outs = engine.execute(ex, _buffers([_f32()]))
        assert ex._native_prog is not False
        assert engine.buffer_pointer(outs[0]) != engine.buffer_pointer(outs[1])
        assert engine.to_host(outs[0]) == engine.to_host(outs[1])


# --------------------------------------------------------------------------
# fallback and liveness
# --------------------------------------------------------------------------


def test_run_time_failure_falls_back(monkeypatch):
    """A tape that explodes mid-call hands the program back, once."""

    class _Boom:
        # `num_args` is part of the Program contract engine.execute checks
        # before it calls (the packed-weight arity guard, M4).
        num_args = 1

        def run(self, inputs):
            raise RuntimeError("boom")

    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.tanh %a : {t}
    return %0 : {t}""")
    x = _f32()
    with _native_engine():
        ref, _ = _run(mod, [x], False)
        monkeypatch.setattr(tape, "lower", lambda interp, **kw: _Boom())
        before = engine.NATIVE_STATS["run_failures"]
        ex = engine.compile_program(mod, "mlir")
        outs = engine.execute(ex, _buffers([x]))
        got = [engine.to_host(o) for o in outs]
        assert got == ref
        assert engine.NATIVE_STATS["run_failures"] == before + 1
        assert ex._native_prog is False
        # ...and the tape is retired: a second call does not try again.
        engine.execute(ex, _buffers([x]))
        assert engine.NATIVE_STATS["run_failures"] == before + 1


def test_native_errors_raise_rather_than_crash():
    """The interpreter's own guards: a malformed tape must come back as a
    Python exception with the GIL intact, which is what lets engine.execute
    fall back instead of taking the process with it."""
    ops = native.opcodes()
    p = native.Program(num_slots=3, num_args=1)
    p.add(opcode=ops["stablehlo.tanh"], operands=[0], results=[1], attrs=[],
          payload=None, drops=[0])
    p.add(opcode=ops["stablehlo.add"], operands=[0, 1], results=[2],
          attrs=[], payload=None, drops=[1])
    p.set_outputs(slots=[2])
    with pytest.raises(RuntimeError, match="dropped slot"):
        p.run(inputs=[mx.array([1.0, 2.0])])
    with pytest.raises(ValueError, match="wrong number of inputs"):
        p.run(inputs=[])
    with pytest.raises(ValueError, match="out of range"):
        p.add(opcode=ops["stablehlo.tanh"], operands=[99], results=[1],
              attrs=[], payload=None, drops=[])
    # ...and the extension still works afterwards.
    q = native.Program(num_slots=2, num_args=1)
    q.add(opcode=ops["stablehlo.tanh"], operands=[0], results=[1], attrs=[],
          payload=None, drops=[0])
    q.set_outputs(slots=[1])
    out = q.run(inputs=[mx.array(np.array([0.0, 1.0], np.float32))])[0]
    np.testing.assert_allclose(np.array(out),
                               np.tanh([0.0, 1.0]).astype(np.float32),
                               rtol=1e-6)


def test_lowering_error_is_not_fatal(monkeypatch):
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.tanh %a : {t}
    return %0 : {t}""")

    def _explode(interp, **kw):
        raise ValueError("lowering bug")

    with _native_engine():
        ref, _ = _run(mod, [_f32()], False)
        monkeypatch.setattr(tape, "lower", _explode)
        before = engine.NATIVE_STATS["lower_errors"]
        ex = engine.compile_program(mod, "mlir")
        outs = engine.execute(ex, _buffers([_f32()]))
        assert engine.NATIVE_STATS["lower_errors"] == before + 1
        assert ex._native_prog is False
        assert len(outs) == 1


def test_drop_lists_bound_the_live_set():
    """A long chain must not accumulate: liveness pruning is what keeps the
    C++ env (and the Metal buffers it pins) flat, exactly as
    Interpreter.eager_plan does for the Python engine."""
    t = "tensor<4xf32>"
    n = 60
    body = ["    %v0 = stablehlo.tanh %a : " + t]
    for i in range(1, n):
        body.append(f"    %v{i} = stablehlo.tanh %v{i - 1} : {t}")
    body.append(f"    return %v{n - 1} : {t}")
    mod = _mod([("a", t)], [t], "\n".join(body))
    ex = check(mod, [_f32((4,))])
    prog = ex._native_prog
    assert prog.num_ops == n
    assert prog.num_slots == n + 1
    # One argument plus one intermediate at a time — never the whole chain.
    assert prog.max_live == 2


def test_drop_lists_keep_multiply_used_values():
    t = "tensor<4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.tanh %a : {t}
    %1 = stablehlo.sine %0 : {t}
    %2 = stablehlo.cosine %0 : {t}
    %3 = stablehlo.add %1, %2 : {t}
    %4 = stablehlo.multiply %3, %0 : {t}
    return %4 : {t}""")
    ex = check(mod, [_f32((4,))])
    # %0 survives to its last use at %4: arg + %0 + the pair being built.
    assert ex._native_prog.max_live == 4


def test_repeated_execution_is_stable():
    """The tape is reused; a second call must not read a dropped slot or
    reuse a constant's storage as an output."""
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %c = stablehlo.constant dense<1.500000e+00> : {t}
    %0 = stablehlo.multiply %a, %c : {t}
    %1 = stablehlo.tanh %0 : {t}
    return %1 : {t}""")
    with _native_engine():
        ex = engine.compile_program(mod, "mlir")
        seen = []
        for _ in range(3):
            x = _f32()
            outs = engine.execute(ex, _buffers([x]))
            seen.append((x, engine.to_host(outs[0])))
        assert ex._native_prog is not False
        for x, got in seen:
            ref, _ = _run(mod, [x], False)
            assert got == ref[0]


def test_opcode_registry_covers_the_lowering_table():
    """C++ owns the enum: every name tape.py knows how to lower must be in
    it, or the handler is dead code."""
    ops = native.opcodes()
    for name in tape._HANDLERS:
        assert name in ops, f"{name} has a lowering but no opcode"
    for name in tape._IDENTITY_CHECKS:
        assert name in ops


def test_compile_eligible_programs_take_the_native_path():
    """M3 retired the eager-only production gate.

    Until the native engine had its own mx::compile, a compile-eligible
    program had to keep the Python fused-graph replay (the M2 tape was 5x
    slower than one). It now compiles natively, so the gate would only cost
    the dispatch win it exists to deliver — and the compiled tape really is
    what runs, which `compiled_calls` proves.
    """
    t = "tensor<4xf32>"
    mod = _mod([("a", t), ("b", t)], [t], f"""
    %0 = stablehlo.add %a, %b : {t}
    %1 = stablehlo.multiply %0, %a : {t}
    return %1 : {t}""")
    with _native_engine():
        before = dict(engine.NATIVE_STATS)
        calls = native.stats()["compiled_calls"]
        ex = engine.compile_program(mod, "mlir")
        engine.execute(ex, _buffers([np.ones(4, np.float32)] * 2))
        assert ex._native_prog is not False
        assert engine.NATIVE_STATS["lowered"] == before["lowered"] + 1
        assert ex._can_compile is True
        assert native.stats()["compiled_calls"] > calls


# --------------------------------------------------------------------------
# M4: the recognizer emits
# --------------------------------------------------------------------------
#
# The recognizer suites (test_qmm, test_qmm_mxfp4, test_moe, test_sdpa)
# already drive these programs through engine.execute and check them against
# jax-CPU, so run under METALJAX_ENGINE=native they are this milestone's
# end-to-end harness. What they cannot say is that the two ENGINES agree bit
# for bit on the SAME rewritten program — the recognizer suites compare
# against CPU with tolerances, and a tape that quietly declined would pass
# them unchanged. These do both: identical output bytes, AND the fused op
# really on the tape.
#
# The graphs come from the recognizer suites themselves rather than being
# rewritten here, so the differential tests can never drift onto a program
# the recognizers no longer match.

import jax                              # noqa: E402
import jax.numpy as jnp                 # noqa: E402

from helpers import lower_bytes         # noqa: E402
from metaljax import interpreter, moe, qmm, sdpa  # noqa: E402

OPS = native.opcodes()

needs_qmm = pytest.mark.skipif(not qmm.QMM_ENABLED, reason="METALJAX_QMM=0")
needs_moe = pytest.mark.skipif(not moe.ENABLED, reason="METALJAX_MOE=0")
needs_sdpa = pytest.mark.skipif(not sdpa.ENABLED, reason="METALJAX_SDPA=0")


def _count(ex, name):
    """How many `name` entries the lowered Program holds (regions too)."""
    return ex._native_prog.op_histogram().get(OPS[name], 0)


def _fused(f, args, name, n=1, **kw):
    """Run `f` on both engines, and assert `n` fused ops on the tape."""
    mod = lower_bytes(f, *args)
    ex = check(mod, [np.asarray(a) for a in args], **kw)
    got = _count(ex, name)
    assert got == n, f"tape holds {got} {name} entries, expected {n}"
    return ex


# --- quantized matmul -----------------------------------------------------


@needs_qmm
def test_qmm_emit_matches_the_python_engine():
    """keras' sub-channel int4 Dense: affine mode, a bias table, no perm."""
    from test_qmm import _quantize, dense_sub
    rows, cols = 256, 64
    _q, packed, scale, zero, g_idx, _ref = _quantize(rows, cols, 128,
                                                     np.float32)
    x = RNG.standard_normal((3, rows)).astype(np.float32)
    ex = _fused(lambda *a: dense_sub(*a, columns=cols),
                [packed, scale, zero, g_idx, x], "metaljax.qmm")
    m = ex.interpreter._qmm.matches[0]
    assert (m.mode, m.has_perm, m.bshape) == ("affine", False, [])


@needs_qmm
def test_qmm_emit_with_a_group_permutation():
    """keras' attention output projection: the reversed contracting-dim
    pairing interleaves the groups, so the weight is packed with its K axis
    permuted and `emit` takes the same permutation of the activations."""
    from test_qmm import _out_proj_case, einsum_out_proj
    n, h, d = 8, 32, 96
    args, _exact = _out_proj_case(n, h, d)
    ex = _fused(lambda *a: einsum_out_proj(*a, n=n, h=h, d=d), args,
                "metaljax.qmm")
    assert ex.interpreter._qmm.matches[0].has_perm


@needs_qmm
def test_qmm_emit_mxfp4_and_batched():
    """MXFP4 mode (no bias table) over a stack of per-expert weights, which
    is also the batched arm: [B, M, K] x [B, N, K]."""
    from test_qmm_mxfp4 import _random_mxfp4, experts_mxfp4
    e, h, m, t = 4, 32, 64, 2
    _codes, blocks, sb, _w = _random_mxfp4((e, h, m), np.random.default_rng(2))
    x = (RNG.standard_normal((e, t, m)) * 0.4).astype(np.float32)
    ex = _fused(lambda *a: experts_mxfp4(*a, k=m), [blocks, sb, x],
                "metaljax.qmm")
    match = ex.interpreter._qmm.matches[0]
    assert match.mode == "mxfp4" and match.bshape == [e]


@needs_qmm
def test_qmm_emit_with_the_weight_on_the_left():
    """`th,emh->etm` — the expert gate/up projection, which jax lowers with
    the QUANTIZED operand as the dot's LHS. `emit` then has to swap the last
    two axes of the product back."""
    from test_qmm import _quantize, _unpack_nibbles
    E, H, M, T = 4, 64, 32, 3
    cols = E * M
    _q, packed, scale, zero, g_idx, _ref = _quantize(H, cols, 64, np.float32)
    x = (RNG.standard_normal((T, H)) * 0.2).astype(np.float32)

    def f(packed_, scale_, zero_, g_idx_, x_):
        w = _unpack_nibbles(packed_, cols)
        g = g_idx_.astype(jnp.int32)
        wf = ((w.astype(x_.dtype) - jnp.take(zero_, g, axis=0).astype(x_.dtype))
              * jnp.take(scale_, g, axis=0))
        wf = jnp.transpose(jnp.reshape(wf, (H, E, M)), (1, 2, 0))
        return jnp.einsum("th,emh->etm", x_, wf)

    ex = _fused(f, [packed, scale, zero, g_idx, x], "metaljax.qmm")
    assert ex.interpreter._qmm.matches[0].swapped


@needs_qmm
def test_qmm_emit_inside_a_decode_loop():
    """The weight is loop-carried state, so the root lives in the while
    BODY: its Program takes the packed arrays as extra captures."""
    from test_qmm import _quantize, dense_sub
    rows, steps = 128, 3
    _q, packed, scale, zero, g_idx, _ref = _quantize(rows, rows, 128,
                                                     np.float32)
    x = (RNG.standard_normal((1, rows)) * 0.1).astype(np.float32)

    def f(packed_, scale_, zero_, g_idx_, x_):
        def body(state):
            i, acc = state
            y = dense_sub(packed_, scale_, zero_, g_idx_, acc, columns=rows)
            return i + 1, acc + y

        _i, out = jax.lax.while_loop(lambda s: s[0] < steps, body, (0, x_))
        return out

    ex = _fused(f, [packed, scale, zero, g_idx, x], "metaljax.qmm")
    # ...and the body Program really did get them: its own arity grew by the
    # pack's arrays, not by a capture of a constant.
    assert ex._native_prog.num_args == 5 + len(qmm.values(ex.interpreter))


@needs_qmm
def test_qmm_repack_relowers_the_tape():
    """A repack that changes the pack's STRUCTURE invalidates the Program.

    The permutation is the case that bites hardest: interleaved groups make
    the pack carry a fourth array, so the fused op's operand list — and the
    Program's arity — are different from the call before. Same buffers'
    SHAPES throughout, only the grouping changes, which is exactly what a
    second weight set can do to a program compiled for the first.
    `prologue` reports it through `changed`; this is the test that the
    native side acts on it rather than replaying a stale plan.
    """
    from test_qmm import _quantize, dense_sub
    rows, cols = 256, 64
    x = RNG.standard_normal((3, rows)).astype(np.float32)
    ramp = _quantize(rows, cols, 128, np.float32)
    woven = _quantize(rows, cols, 128, np.float32, seed=3,
                      g_idx=(np.arange(rows) % 2).astype(np.float32))
    mod = lower_bytes(lambda *a: dense_sub(*a, columns=cols),
                      ramp[1], ramp[2], ramp[3], ramp[4], x)

    with _native_engine():
        # NB a snapshot, not zero: NATIVE_STATS is process-wide, and
        # test_run_time_failure_falls_back deliberately fails a run.
        failures = engine.NATIVE_STATS["run_failures"]
        ex = engine.compile_program(mod, "mlir")
        seen, perms = [], []
        for w in (ramp, woven, ramp):
            args = [w[1], w[2], w[3], w[4], x]
            outs = engine.execute(ex, _buffers(args))
            seen.append((args, [engine.to_host(o) for o in outs]))
            assert ex._native_prog is not False, "the tape was retired"
            m = ex.interpreter._qmm.matches[0]
            perms.append(m.has_perm)
            assert ex._native_prog.num_args == 5 + m.nvals
        assert engine.NATIVE_STATS["run_failures"] == failures
    assert perms == [False, True, False], perms
    # Each call still agrees with the Python engine on its own weights.
    for args, got in seen:
        ref, _ = _run(mod, args, False)
        assert got == ref


# --- attention ------------------------------------------------------------


@needs_sdpa
@pytest.mark.parametrize("name,fn,shapes", [
    (s[0], s[1], s[2]) for s in __import__("test_sdpa").SPELLINGS])
def test_sdpa_emit_matches_the_python_engine(name, fn, shapes):
    """Every spelling test_sdpa recognizes, through both engines."""
    from test_sdpa import _arrays
    args = _arrays(shapes, jnp.float32)
    _fused(fn, args, "metaljax.sdpa")


@needs_sdpa
def test_sdpa_emit_scales_the_mask():
    """`(QK + mask) * s`: the scale is applied AFTER the mask, so the mask
    the fused kernel is handed has to carry it (`Match.mask`'s multiplier —
    the one branch of `_mask_array` no spelling in test_sdpa reaches)."""
    from test_sdpa import B, D, H, T, _arrays
    q, k, v, m = _arrays([(B, H, T, D)] * 3 + [(B, 1, T, T)], jnp.float32)

    def attn(q_, k_, v_, m_):
        lg = (jnp.einsum("bhqd,bhkd->bhqk", q_, k_) + m_) * (D ** -0.5)
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v_)

    ex = _fused(attn, [q, k, v, m], "metaljax.sdpa")
    assert ex.interpreter._sdpa.matches[0].mask[3] != 1.0


@needs_sdpa
def test_sdpa_mask_is_built_once_for_the_whole_block():
    """`_mask_array`'s cache, as a tape entry: two attentions sharing one
    mask build the additive form ONCE (it is a full [.., Tq, Tk] tensor, and
    a transformer shares one mask across every layer, so recomputing it per
    root would allocate half a gigabyte per token on a 60-layer model)."""
    from test_sdpa import B, D, H, T, attn_bias, _arrays
    q, k, v, m = _arrays([(B, H, T, D)] * 3 + [(B, 1, T, T)], jnp.float32)

    def two_layers(q_, k_, v_, m_):
        first = attn_bias(q_, k_, v_, m_)
        return attn_bias(first, k_, v_, m_)

    ex = _fused(two_layers, [q, k, v, m], "metaljax.sdpa", n=2)
    assert _count(ex, "metaljax.sdpa.mask") == 1


# --- the MoE expert gather ------------------------------------------------


@needs_moe
@pytest.mark.parametrize("T", [1, 5])
def test_moe_emit_matches_the_python_engine(T):
    """Float experts: gather_mm, the pair-space plan, the weighted sum."""
    from test_moe import _args, make_weights, moe_block
    w = make_weights(4, T, 64, 32, seed=T)
    ex = _fused(lambda *a: moe_block(*a, k=2), _args(w), "metaljax.moe.tail")
    assert _count(ex, "metaljax.moe.dot") == 2      # both projections
    assert _count(ex, "metaljax.qmm") == 0          # nothing was packed


@needs_moe
def test_moe_emit_pair_space_views():
    """The plan nodes the keras-hub and gpt-oss blocks never produce.

    `extra` carries BOTH the expert and the token axis and is computed
    outside the region, so it is gathered on the (e, t) grid rather than
    along one axis; the activation is CONCATENATED in pair space; and the
    expert output is RESHAPED over its trailing axes. Each is a branch of
    `emit` with no other test.
    """
    from test_moe import router_scores
    E, T, H, I, k = 4, 5, 64, 32, 2
    rng = np.random.default_rng(11)

    def n(*shape):
        return (rng.standard_normal(shape) / np.sqrt(H)).astype(np.float32)

    args = [n(T, H), n(H, E), n(E), n(E, H, 2 * I), n(E, 2 * I),
            n(E, 2 * I, H), n(E, H), n(E, T, H)]

    def f(x, wr, br, wgu, bgu, wd, bd, extra):
        s = router_scores(x, wr, br, k)
        gu = jnp.einsum("th,ehm->etm", x, wgu) + bgu[:, None, :]
        act = jnp.concatenate([jax.nn.sigmoid(gu[..., :I]),
                               jnp.tanh(gu[..., I:])], axis=-1)
        y = jnp.einsum("etm,emh->eth", act, wd) + bd[:, None, :]
        y = jnp.reshape(y + extra, (E, T, H // 2, 2))
        return jnp.sum(y * s.T[..., None, None], axis=0)

    ex = _fused(f, args, "metaljax.moe.tail")
    assert _count(ex, "metaljax.moe.dot") == 2


@needs_moe
@needs_qmm
def test_moe_emit_reads_the_packed_experts():
    """gpt-oss' MXFP4 block: the expert dots were packed by metaljax.qmm and
    the gather reads those packs (gather_qmm) instead of a dense weight."""
    from test_moe import make_mxfp4, moe_block_mxfp4, router_scores  # noqa
    E, k, T, H, I = 4, 2, 3, 64, 32
    rng = np.random.default_rng(E)
    gu_c, gu_s, _gu = make_mxfp4(2 * I, H, E, seed=E)
    d_c, d_s, _d = make_mxfp4(H, I, E, seed=E + 1)
    args = [(rng.standard_normal((T, H)) / 8).astype(np.float32),
            (rng.standard_normal((H, E)) / 8).astype(np.float32),
            rng.standard_normal(E).astype(np.float32),
            gu_c, gu_s, d_c, d_s,
            (rng.standard_normal((E, 2 * I)) / 8).astype(np.float32),
            (rng.standard_normal((E, H)) / 8).astype(np.float32)]
    ex = _fused(lambda *a: moe_block_mxfp4(*a, H=H, I=I, k=k), args,
                "metaljax.moe.tail")
    assert _count(ex, "metaljax.moe.dot") == 2
    # The dense quantized_matmul is NOT emitted: moe took those dots over.
    assert _count(ex, "metaljax.qmm") == 0
    assert all(m.absorbed for m in ex.interpreter._qmm.matches)


@pytest.mark.skipif(not interpreter.COMPILE_ENABLED,
                    reason="METALJAX_COMPILE=0")
@pytest.mark.parametrize("which", ["qmm", "sdpa", "moe"])
def test_emits_are_identical_through_mx_compile(which):
    """All three rewrites again, with the tape traced through mx::compile.

    Everything above compares the two EAGER interpreters (see the module
    docstring on why eager is the reference). This is the other path, and
    the one a real program takes: both engines compile, so the fused
    kernels are the same on both sides and the bytes must still agree.
    """
    if which == "qmm":
        if not qmm.QMM_ENABLED:
            pytest.skip("METALJAX_QMM=0")
        from test_qmm import _quantize, dense_sub
        rows, cols = 256, 64
        _q, packed, scale, zero, g_idx, _ref = _quantize(rows, cols, 128,
                                                         np.float32)
        x = RNG.standard_normal((3, rows)).astype(np.float32)
        ex = _fused(lambda *a: dense_sub(*a, columns=cols),
                    [packed, scale, zero, g_idx, x], "metaljax.qmm",
                    compiled=True)
    elif which == "sdpa":
        if not sdpa.ENABLED:
            pytest.skip("METALJAX_SDPA=0")
        from test_sdpa import B, D, H, T, attn_causal, _arrays
        args = _arrays([(B, H, T, D)] * 3, jnp.float32)
        ex = _fused(attn_causal, args, "metaljax.sdpa", compiled=True)
    else:
        if not moe.ENABLED:
            pytest.skip("METALJAX_MOE=0")
        from test_moe import _args, make_weights, moe_block
        w = make_weights(4, 5, 64, 32, seed=5)
        ex = _fused(lambda *a: moe_block(*a, k=2), _args(w),
                    "metaljax.moe.tail", compiled=True)
    assert ex._can_compile, "this program was supposed to compile"


# --- the ordering ops the router needs ------------------------------------
#
# A top_k is what makes an MoE router a router, so M4 had to lower it or
# every MoE program would decline on the way in. It arrives two ways — as
# chlo.top_k from a direct lowering, as the sort it decomposes to through a
# portable artifact — and both are here, with the key kinds the handler
# branches on.


@pytest.mark.parametrize("dt", [np.float32, np.int32])
def test_top_k_matches_the_python_engine(dt):
    x = (RNG.standard_normal((3, 8)) * 4).astype(dt)
    mod = lower_bytes(lambda a: jax.lax.top_k(a, 3), x)
    ex = check(mod, [x])
    assert _count(ex, "chlo.top_k") == 1


@pytest.mark.parametrize("dt", [np.int32, np.uint16, np.bool_])
def test_sort_of_an_ordered_key_matches_the_python_engine(dt):
    """A comparator that IS a compare: no canonicalization chain, because
    integers and bools are already in total order."""
    x = (RNG.standard_normal((3, 8)) * 4).astype(dt)
    mod = lower_bytes(lambda a: jnp.sort(a, axis=-1), x)
    ex = check(mod, [x])
    assert _count(ex, "stablehlo.sort") == 1


def test_descending_totalorder_sort_matches_the_python_engine():
    """chlo.top_k's decomposition, which is what the PLUGIN receives: a
    portable artifact has legalized the chlo op away, leaving
    `sort(values, iota)` under a strict TOTALORDER GT. Written out here
    because a module holding chlo.top_k cannot be serialized as one.

    The input carries the values that separate a totalOrder key from a
    plain compare: NaN sorts above +inf, and -0 below +0.
    """
    t, ti = "tensor<2x6xf32>", "tensor<2x6xi32>"
    mod = _mod([("a", t)], [t, ti], f"""
    %0 = stablehlo.iota dim = 1 : {ti}
    %1:2 = "stablehlo.sort"(%a, %0) <{{dimension = 1 : i64, is_stable = true}}> ({{
    ^bb0(%al: tensor<f32>, %ar: tensor<f32>, %bl: tensor<i32>, %br: tensor<i32>):
      %2 = stablehlo.compare GT, %al, %ar, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }}) : ({t}, {ti}) -> ({t}, {ti})
    return %1#0, %1#1 : {t}, {ti}""")
    x = np.array([[0.0, -0.0, np.inf, -np.inf, np.nan, 1.5],
                  [2.0, 2.0, -1.0, 0.5, -0.0, 0.0]], np.float32)
    ex = check(mod, [x])
    assert _count(ex, "stablehlo.sort") == 1


# --- what does not lower --------------------------------------------------


def test_sort_with_a_key_chain_declines():
    """The sort opcode carries a plan, not a program: it can only run a
    comparator that compares an operand pair directly (what every top_k
    lowers to). jnp.sort's float comparator canonicalizes -0 and NaN first,
    which means running block code on arrays — that declines, wholesale."""
    x = RNG.standard_normal((3, 8)).astype(np.float32)
    with _native_engine():
        ex = engine.compile_program(
            lower_bytes(lambda a: jnp.sort(a, axis=-1), x), "mlir")
        assert ex.native_program() is None
