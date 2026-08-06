"""Stage 2 M2: the native tape, differential against the Python engine.

Every case runs the SAME executable twice — once with the tape lowered
(native/tape.cc) and once with it forced off — and compares output BYTES.
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
    saved_eager = engine._TAPE_EAGER_ONLY
    engine.NATIVE = native
    # The differential programs here are small enough to be compile-
    # eligible; lift the eager-only production gate so the tape actually
    # runs them (its own test below covers the gate).
    engine._TAPE_EAGER_ONLY = False
    try:
        yield
    finally:
        engine.NATIVE = saved
        engine._TAPE_EAGER_ONLY = saved_eager


def _buffers(arrays):
    bufs = []
    for a in arrays:
        arr = np.ascontiguousarray(a)
        bufs.append(engine.buffer_from_host(
            arr.tobytes(), engine._NP_TO_ENUM[arr.dtype], list(arr.shape),
            None, 0))
    return bufs


def _run(mod, arrays, tape_on):
    ex = engine.compile_program(mod, "mlir")
    if not tape_on:
        ex._native_prog = False  # the Python engine...
        ex._can_compile = False  # ...op by op; see the module docstring
    outs = engine.execute(ex, _buffers(arrays))
    return [engine.to_host(o) for o in outs], ex


def check(mod, arrays, lowered=True):
    """Run both engines on `mod`; assert identical bytes and that the tape
    did (or did not) take the program. Returns the native executable."""
    with _native_engine():
        before = dict(engine.NATIVE_STATS)
        ref, _ = _run(mod, arrays, False)
        got, ex = _run(mod, arrays, True)
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


RNG = np.random.default_rng(20260806)


def _f32(shape=(3, 4)):
    return RNG.standard_normal(shape).astype(np.float32)


def _specials():
    """The values that separate a faithful port from an approximate one."""
    return np.array([0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 1e-30, 3.4e38,
                     np.inf, -np.inf, np.nan, -np.nan], np.float32)


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
# declines — one per family. Each must still compute the right answer.
# --------------------------------------------------------------------------


def test_decline_unsupported_op():
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t), ("b", t)], [t], f"""
    %0 = stablehlo.divide %a, %b : {t}
    return %0 : {t}""")
    check(mod, [_f32(), _f32() + 3.0], lowered=False)


def test_decline_region_op():
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], ["tensor<3xf32>"], f"""
    %init = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = "stablehlo.reduce"(%a, %init) <{{dimensions = array<i64: 1>}}> ({{
    ^bb0(%x: tensor<f32>, %y: tensor<f32>):
      %s = stablehlo.subtract %x, %y : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }}) : ({t}, tensor<f32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>""")
    check(mod, [_f32()], lowered=False)


def test_decline_control_flow():
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
    check(mod, [np.float32(1.5)], lowered=False)


def test_decline_complex():
    t = "tensor<4xcomplex<f32>>"
    mod = _mod([("a", t), ("b", t)], [t], f"""
    %0 = stablehlo.add %a, %b : {t}
    return %0 : {t}""")
    z = (RNG.standard_normal(4) + 1j * RNG.standard_normal(4)).astype(
        np.complex64)
    check(mod, [z, z * 2], lowered=False)


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


def test_decline_integer_dot():
    mod = _mod([("a", "tensor<3x4xi32>"), ("b", "tensor<4x5xi32>")],
               ["tensor<3x5xi32>"], """
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<3x4xi32>, tensor<4x5xi32>) -> tensor<3x5xi32>
    return %0 : tensor<3x5xi32>""")
    check(mod, [RNG.integers(-9, 9, (3, 4)).astype(np.int32),
                RNG.integers(-9, 9, (4, 5)).astype(np.int32)], lowered=False)


def test_decline_variadic_reduce():
    # The (values, indices) argmax pair: two operands, two results.
    t = "tensor<3x4xf32>"
    it = "tensor<3x4xi32>"
    mod = _mod([("a", t), ("idx", it)], ["tensor<3xf32>", "tensor<3xi32>"], f"""
    %vi = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %ii = stablehlo.constant dense<0> : tensor<i32>
    %0:2 = "stablehlo.reduce"(%a, %idx, %vi, %ii) <{{dimensions = array<i64: 1>}}> ({{
    ^bb0(%av: tensor<f32>, %ai: tensor<i32>, %bv: tensor<f32>, %bi: tensor<i32>):
      %p = stablehlo.compare GT, %av, %bv : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %v = stablehlo.select %p, %av, %bv : tensor<i1>, tensor<f32>
      %i = stablehlo.select %p, %ai, %bi : tensor<i1>, tensor<i32>
      stablehlo.return %v, %i : tensor<f32>, tensor<i32>
    }}) : ({t}, {it}, tensor<f32>, tensor<i32>) -> (tensor<3xf32>, tensor<3xi32>)
    return %0#0, %0#1 : tensor<3xf32>, tensor<3xi32>""")
    idx = np.tile(np.arange(4, dtype=np.int32), (3, 1))
    check(mod, [_f32(), idx], lowered=False)


def test_decline_totalorder_compare():
    t = "tensor<6xf32>"
    ot = "tensor<6xi1>"
    mod = _mod([("a", t), ("b", t)], [ot], f"""
    %0 = stablehlo.compare LT, %a, %b, TOTALORDER : ({t}, {t}) -> {ot}
    return %0 : {ot}""")
    x = np.array([np.nan, -np.nan, 0.0, -0.0, np.inf, -np.inf], np.float32)
    check(mod, [x, np.roll(x, 2)], lowered=False)


def test_decline_func_call():
    # jax wraps jnp.where / jnp.clip / jnp.round in private helpers, so a
    # call in main is not exotic — it is the biggest single decline family
    # today (M3 territory).
    t = "tensor<3x4xf32>"
    mod = (f"module @calls {{\n"
           f"  func.func public @main(%a: {t}) -> ({t}) {{\n"
           f"    %0 = call @helper(%a) : ({t}) -> {t}\n"
           f"    return %0 : {t}\n"
           f"  }}\n"
           f"  func.func private @helper(%a: {t}) -> {t} {{\n"
           f"    %0 = stablehlo.tanh %a : {t}\n"
           f"    return %0 : {t}\n"
           f"  }}\n"
           f"}}\n").encode()
    check(mod, [_f32()], lowered=False)


def test_decline_constant_output():
    # A constant returned directly would alias across calls: the Program
    # holds it for the executable's life. Declining keeps XLA's contract.
    t = "tensor<3xf32>"
    mod = _mod([("a", t)], [t, t], f"""
    %c = stablehlo.constant dense<[1.0, 2.0, 3.0]> : {t}
    %0 = stablehlo.add %a, %c : {t}
    return %0, %c : {t}, {t}""")
    check(mod, [_f32((3,))], lowered=False)


def test_decline_constant_view_output():
    # A splat constant is a ONE-element buffer the tape holds forever;
    # broadcasting it to an output would hand every call a view of that
    # same storage.
    mod = _mod([("a", "tensor<3xf32>")],
               ["tensor<3xf32>", "tensor<2x3xf32>"], """
    %c = stablehlo.constant dense<4.000000e+00> : tensor<3xf32>
    %b = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
    %0 = stablehlo.add %a, %c : tensor<3xf32>
    return %0, %b : tensor<3xf32>, tensor<2x3xf32>""")
    check(mod, [_f32((3,))], lowered=False)


def test_decline_identity_forwarded_argument():
    # A reshape to the SAME shape may hand back the operand object itself;
    # engine.execute catches that by id() on the Python path only.
    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.reshape %a : ({t}) -> {t}
    return %0 : {t}""")
    check(mod, [_f32()], lowered=False)


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
        def run(self, inputs):
            raise RuntimeError("boom")

    t = "tensor<3x4xf32>"
    mod = _mod([("a", t)], [t], f"""
    %0 = stablehlo.tanh %a : {t}
    return %0 : {t}""")
    x = _f32()
    with _native_engine():
        ref, _ = _run(mod, [x], False)
        monkeypatch.setattr(tape, "lower", lambda interp: _Boom())
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

    def _explode(interp):
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


def test_eager_only_gate_declines_compiled_programs():
    """The production gate: compile-eligible programs keep mx.compile.

    Until M3, the tape only replaces the eager path (it is 5x slower
    than a fused-graph replay and 2.4x faster than Python eager); the
    flag must never regress a program the default engine would compile.
    """
    t = "tensor<4xf32>"
    mod = _mod([("a", t), ("b", t)], [t], f"""
    %0 = stablehlo.add %a, %b : {t}
    %1 = stablehlo.multiply %0, %a : {t}
    return %1 : {t}""")
    with _native_engine():
        engine._TAPE_EAGER_ONLY = True
        before = dict(engine.NATIVE_STATS)
        ex = engine.compile_program(mod, "mlir")
        engine.execute(ex, _buffers([np.ones(4, np.float32)] * 2))
        assert ex._native_prog is False
        assert engine.NATIVE_STATS["declined"] == before["declined"] + 1
        assert ex._can_compile is True
