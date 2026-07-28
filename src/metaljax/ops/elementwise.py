"""Elementwise StableHLO ops (unary, binary, compare, select, clamp)."""

import re

import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes
from metaljax.interpreter import register, UnsupportedOpError


def _trunc(x):
    return mx.where(x < 0, mx.ceil(x), mx.floor(x))


def _round_afz(x):
    return mx.sign(x) * mx.floor(mx.abs(x) + 0.5)


def _round_even(x):
    f = mx.floor(x)
    d = x - f
    f_is_even = (f % 2) == 0
    up = f + 1
    return mx.where(d > 0.5, up, mx.where(d < 0.5, f, mx.where(f_is_even, f, up)))


def _cbrt(x):
    return mx.sign(x) * mx.power(mx.abs(x), 1.0 / 3.0)


_UNARY = {
    "abs": mx.abs,
    "cbrt": _cbrt,
    "ceil": mx.ceil,
    "cosine": mx.cos,
    "erf": mx.erf,
    "erf_inv": mx.erfinv,
    "exponential": mx.exp,
    # MLX has no complex expm1 GPU kernel
    "exponential_minus_one": lambda x: (mx.exp(x) - 1 if x.dtype == mx.complex64 else mx.expm1(x)),
    "floor": mx.floor,
    "is_finite": mx.isfinite,
    "log": mx.log,
    "log_plus_one": mx.log1p,
    "logistic": mx.sigmoid,
    "negate": mx.negative,
    "round_nearest_afz": _round_afz,
    "round_nearest_even": _round_even,
    "rsqrt": mx.rsqrt,
    # mx.sign returns 0 for NaN; stablehlo.sign must propagate it
    "sign": lambda x: (mx.where(mx.isnan(x), x, mx.sign(x))
                       if x.dtype in (mx.float32, mx.float16, mx.bfloat16)
                       else mx.sign(x)),
    "sine": mx.sin,
    "sqrt": mx.sqrt,
    "tan": mx.tan,
    "tanh": mx.tanh,
}

for _name, _fn in _UNARY.items():
    def _h(interp, op, ins, env, _fn=_fn):
        return _fn(ins[0])
    register(f"stablehlo.{_name}")(_h)
# chlo variants that sometimes survive lowering
register("chlo.erf")(lambda i, o, ins, e: mx.erf(ins[0]))
register("chlo.erf_inv")(lambda i, o, ins, e: mx.erfinv(ins[0]))
register("chlo.square")(lambda i, o, ins, e: mx.square(ins[0]))
register("chlo.erfc")(lambda i, o, ins, e: 1.0 - mx.erf(ins[0]))
register("stablehlo.erfc")(lambda i, o, ins, e: 1.0 - mx.erf(ins[0]))


_UNSIGNED_VIEW = {mx.int8: mx.uint8, mx.int16: mx.uint16,
                  mx.int32: mx.uint32, mx.int64: mx.uint64}


def _as_unsigned(x):
    if dtypes.is_bool(x.dtype):
        return x.astype(mx.uint8)
    if x.dtype in _UNSIGNED_VIEW:
        return mx.view(x, _UNSIGNED_VIEW[x.dtype])
    return x


def _popcount(u):
    """SWAR popcount on an unsigned array (any width, computed in u32/u64)."""
    if u.dtype == mx.uint64:
        c1, c2, c4, m = (0x5555555555555555, 0x3333333333333333,
                         0x0F0F0F0F0F0F0F0F, 0x0101010101010101)
        shift = 56
    else:
        u = u.astype(mx.uint32)
        c1, c2, c4, m = 0x55555555, 0x33333333, 0x0F0F0F0F, 0x01010101
        shift = 24
    u = u - ((u >> 1) & c1)
    u = (u & c2) + ((u >> 2) & c2)
    u = (u + (u >> 4)) & c4
    return (u * m) >> shift


@register("stablehlo.popcnt")
def _popcnt(interp, op, ins, env):
    (x,) = ins
    return _popcount(_as_unsigned(x)).astype(x.dtype)


@register("stablehlo.count_leading_zeros")
def _clz(interp, op, ins, env):
    (x,) = ins
    w = x.dtype.size * 8
    u = _as_unsigned(x)
    s = 1
    while s < w:
        u = u | (u >> s)
        s *= 2
    return (w - _popcount(u)).astype(x.dtype)


@register("stablehlo.real")
def _real(interp, op, ins, env):
    return mx.real(ins[0])


@register("stablehlo.imag")
def _imag(interp, op, ins, env):
    x = ins[0]
    if x.dtype != mx.complex64:
        return mx.zeros_like(x)
    return mx.imag(x)


@register("stablehlo.complex")
def _complex(interp, op, ins, env):
    re, im = ins
    return re.astype(mx.complex64) + im.astype(mx.complex64) * mx.array(1j)


@register("stablehlo.fft")
def _fft(interp, op, ins, env):
    (x,) = ins
    kind = str(op.attributes["fft_type"])
    length = _ir.i64_list(op, "fft_length")
    axes = list(range(len(x.shape) - len(length), len(x.shape)))
    s = [int(v) for v in length]
    if "IRFFT" in kind:
        return mx.fft.irfftn(x, s=s, axes=axes)
    if "RFFT" in kind:
        return mx.fft.rfftn(x, s=s, axes=axes)
    if "IFFT" in kind:
        return mx.fft.ifftn(x, s=s, axes=axes)
    return mx.fft.fftn(x, s=s, axes=axes)


@register("stablehlo.not")
def _not(interp, op, ins, env):
    (x,) = ins
    if dtypes.is_bool(x.dtype):
        return mx.logical_not(x)
    if hasattr(mx, "bitwise_invert"):
        return mx.bitwise_invert(x)
    return mx.bitwise_xor(x, mx.array(-1, dtype=mx.int64).astype(x.dtype))


def _int_trunc_div(a, b):
    if dtypes.is_unsigned(a.dtype):
        return mx.floor_divide(a, b)
    q = mx.floor_divide(mx.abs(a), mx.abs(b))
    neg = (a < 0) != (b < 0)
    return mx.where(neg, -q, q).astype(a.dtype)


def _divide(a, b):
    if dtypes.is_int(a.dtype):
        return _int_trunc_div(a, b)
    return mx.divide(a, b)


def _remainder(a, b):
    # StableHLO remainder: truncated division remainder, sign of the dividend.
    if dtypes.is_int(a.dtype):
        return (a - _int_trunc_div(a, b) * b).astype(a.dtype)
    return a - _trunc(a / b) * b


def _logical_or_bitwise(logical, bitwise):
    def fn(a, b):
        if dtypes.is_bool(a.dtype):
            return logical(a, b)
        return bitwise(a, b)
    return fn


def _shift_guard(a, b, fn, overflow):
    # XLA: shifting by >= bit width yields 0 (logical/left) or the sign
    # fill (arithmetic); Metal shifts are mod-width (x86-style).
    w = a.dtype.size * 8
    over = b.astype(mx.int32) >= w
    base = fn(a, b)
    if overflow == "sign":
        fill = fn(a, mx.array(w - 1, dtype=b.dtype).astype(b.dtype))
    else:
        fill = mx.zeros_like(a)
    return mx.where(over, fill, base)


def _shift_right_logical(a, b):
    if dtypes.is_unsigned(a.dtype) or dtypes.is_bool(a.dtype):
        return mx.right_shift(a, b)
    u = dtypes.unsigned_of(a.dtype)
    return mx.right_shift(a.astype(u), b.astype(u)).astype(a.dtype)


_BINARY = {
    "add": lambda a, b: mx.logical_or(a, b) if dtypes.is_bool(a.dtype) else mx.add(a, b),
    "atan2": mx.arctan2,
    "divide": _divide,
    "maximum": mx.maximum,
    "minimum": mx.minimum,
    "multiply": lambda a, b: mx.logical_and(a, b) if dtypes.is_bool(a.dtype) else mx.multiply(a, b),
    "power": mx.power,
    "remainder": _remainder,
    "subtract": mx.subtract,
    "and": _logical_or_bitwise(mx.logical_and, mx.bitwise_and),
    "or": _logical_or_bitwise(mx.logical_or, mx.bitwise_or),
    "xor": _logical_or_bitwise(
        lambda a, b: mx.not_equal(a, b), mx.bitwise_xor
    ),
    "shift_left": lambda a, b: _shift_guard(a, b, mx.left_shift, "zero"),
    "shift_right_logical": lambda a, b: _shift_guard(
        a, b, _shift_right_logical, "zero"),
    "shift_right_arithmetic": lambda a, b: _shift_guard(
        a, b, mx.right_shift, "sign"),
}

_WRAP4 = {"add", "multiply", "subtract"}


def _maybe_wrap4(op, ins, out):
    # Emulated int4/uint4 live in i8/u8 storage: arithmetic must wrap
    # to 4 bits like XLA. Only consulted when the storage type matches.
    if out.dtype == mx.int8:
        if str(ir.RankedTensorType(op.results[0].type).element_type) == "i4":
            return (((out.astype(mx.int32) + 8) % 16) - 8).astype(mx.int8)
    elif out.dtype == mx.uint8:
        el = str(ir.RankedTensorType(op.results[0].type).element_type)
        if el == "ui4":
            return (out.astype(mx.int32) % 16).astype(mx.uint8)
    return out


for _name, _fn in _BINARY.items():
    def _h(interp, op, ins, env, _fn=_fn, _w=_name in _WRAP4):
        out = _fn(ins[0], ins[1])
        if _w and out.dtype in (mx.int8, mx.uint8):
            out = _maybe_wrap4(op, ins, out)
        return out
    register(f"stablehlo.{_name}")(_h)


_COMPARE = {
    "EQ": mx.equal,
    "NE": mx.not_equal,
    "LT": mx.less,
    "LE": mx.less_equal,
    "GT": mx.greater,
    "GE": mx.greater_equal,
}


def _comparison_direction(op) -> str:
    attr = op.attributes["comparison_direction"]
    s = str(attr)
    m = re.search(r"comparison_direction\s+(\w+)", s)
    if m:
        return m.group(1)
    return s


@register("stablehlo.compare")
def _compare(interp, op, ins, env):
    a, b = ins
    direction = _comparison_direction(op)
    try:
        fn = _COMPARE[direction]
    except KeyError:
        raise UnsupportedOpError(f"compare direction {direction!r}") from None
    if ("compare_type" in op.attributes
            and "TOTALORDER" in str(op.attributes["compare_type"])
            and dtypes.is_float(a.dtype)):
        # IEEE totalOrder (searchsorted/sort NaN handling): compare the
        # order-preserving integer keys instead of the raw floats.
        return fn(dtypes.total_order_key(a), dtypes.total_order_key(b))
    return fn(a, b)


@register("stablehlo.select")
def _select(interp, op, ins, env):
    pred, on_true, on_false = ins
    return mx.where(pred, on_true, on_false)


@register("stablehlo.clamp")
def _clamp(interp, op, ins, env):
    lo, x, hi = ins
    return mx.minimum(mx.maximum(x, lo), hi)


@register("stablehlo.constant")
def _constant(interp, op, ins, env):
    from metaljax import _ir
    t = ir.RankedTensorType(op.results[0].type)
    if str(t.element_type).startswith("complex"):
        # op.attributes["value"] itself raises for complex dense attrs
        # (the bindings can't cast DenseTypedElementsAttr): parse the
        # op's text form instead.
        arr = _ir.complex_dense_from_text(str(op), tuple(t.shape))
        return dtypes.to_mx(arr)
    arr = _ir.dense_to_np(op.attributes["value"], t)
    out = dtypes.to_mx(arr)
    want = dtypes.mx_dtype_for(t.element_type)
    if out.dtype != want:
        out = out.astype(want)
    return out


@register("stablehlo.iota")
def _iota(interp, op, ins, env):
    t = ir.RankedTensorType(op.results[0].type)
    shape = list(t.shape)
    dim = ir.IntegerAttr(op.attributes["iota_dimension"]).value
    dtype = dtypes.mx_dtype_for(t.element_type)
    # MLX has no complex/bool arange kernels: ramp in a real dtype, cast.
    ramp_dt = (mx.int32 if dtype in (mx.bool_, mx.complex64)
               else dtype)
    ramp = mx.arange(shape[dim], dtype=ramp_dt)
    view = [1] * len(shape)
    view[dim] = shape[dim]
    out = mx.broadcast_to(mx.reshape(ramp, view), shape)
    return out.astype(dtype)


@register("stablehlo.reduce_precision")
def _reduce_precision(interp, op, ins, env):
    from metaljax import _ir
    exp = _ir.int_attr(op, "exponent_bits")
    man = _ir.int_attr(op, "mantissa_bits")
    x = ins[0]
    if exp >= 8 and man >= 23:
        return x  # no-op for f32-or-narrower storage
    if (exp, man) == (8, 7):
        return x.astype(mx.bfloat16).astype(x.dtype)
    if (exp, man) == (5, 10):
        return x.astype(mx.float16).astype(x.dtype)
    orig = x.dtype
    if orig in (mx.float16, mx.bfloat16):
        x = x.astype(mx.float32)
    if 0 <= man < 23 and 2 <= exp <= 8 and x.dtype == mx.float32:
        # Round f32 mantissa to `man` bits (round-to-nearest-even), then
        # clamp to the e-bit exponent range: overflow -> inf,
        # underflow -> 0 (XLA reduce_precision has no subnormals).
        isnan = mx.isnan(x)
        u = mx.view(x, mx.uint32)
        if man < 23:
            shift = 23 - man
            half = mx.array((1 << (shift - 1)) - 1, dtype=mx.uint32)
            lsb = (u >> shift) & mx.array(1, dtype=mx.uint32)
            u = (u + half + lsb) & mx.array(
                ~((1 << shift) - 1) & 0xFFFFFFFF, dtype=mx.uint32)
        r = mx.view(u, mx.float32)
        if exp < 8:
            biased = ((u >> 23) & 0xFF).astype(mx.int32)
            max_e = (1 << (exp - 1)) - 1
            min_e = 2 - (1 << (exp - 1))
            sign = mx.where(r < 0, mx.array(-1.0, mx.float32),
                            mx.array(1.0, mx.float32))
            over = biased > 127 + max_e
            under = biased < 127 + min_e
            r = mx.where(over, sign * mx.array(float("inf"), mx.float32), r)
            r = mx.where(under, sign * mx.array(0.0, mx.float32), r)
        r = mx.where(isnan, x, r)
        return r.astype(orig)
    raise UnsupportedOpError(
        f"reduce_precision e{exp}m{man} on {orig} not implemented")
