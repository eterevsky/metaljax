"""MLIR element type <-> MLX dtype <-> numpy dtype mappings."""

import os
import warnings

import ml_dtypes
import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir


class UnsupportedDtypeError(TypeError):
    pass


# Metal has no float64. Default policy (strict): f64 values may pass
# *through* the device (jax_enable_x64 wraps python scalars as f64 buffers
# that programs immediately convert to f32 — storing them as f32 rounds
# exactly once, bit-identical to CPU), but any op that *computes* in f64
# fails at compile time. METALJAX_F64=downcast opts into full f32 emulation.
F64_DOWNCAST = os.environ.get("METALJAX_F64", "error") == "downcast"
_warned_f64 = False


def _warn_f64():
    global _warned_f64
    if not _warned_f64:
        _warned_f64 = True
        warnings.warn(
            "metaljax: float64 is not supported on Metal; computing in float32 "
            "(set METALJAX_F64=error to fail instead)",
            stacklevel=3,
        )


_MLIR_TO_MX = {
    "complex<f32>": mx.complex64,
    "f16": mx.float16,
    "f32": mx.float32,
    "bf16": mx.bfloat16,
    "i1": mx.bool_,
    "i8": mx.int8,
    "i16": mx.int16,
    "i32": mx.int32,
    "i64": mx.int64,
    "ui8": mx.uint8,
    "ui16": mx.uint16,
    "ui32": mx.uint32,
    "ui64": mx.uint64,
}


# numpy dtypes as JAX should see them (f64 stays f64 in metadata).
_MLIR_TO_NP = {
    "complex<f32>": np.dtype(np.complex64),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "bf16": np.dtype(ml_dtypes.bfloat16),
    "i1": np.dtype(np.bool_),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "ui8": np.dtype(np.uint8),
    "ui16": np.dtype(np.uint16),
    "ui32": np.dtype(np.uint32),
    "ui64": np.dtype(np.uint64),
}


# Emulated sub-byte / float8 dtypes (no MLX storage): values live EXACTLY
# in a wider storage dtype (every f8 grid embeds in f16; E8M0's huge
# exponent range needs f32; i4/u4 fit i8/u8). Host transfer converts
# via ml_dtypes; stablehlo.convert quantizes onto the target grid.
import ml_dtypes as _mld
EMULATED = {
    "f8E4M3FN":       (mx.float16, np.dtype(_mld.float8_e4m3fn)),
    "f8E5M2":         (mx.float16, np.dtype(_mld.float8_e5m2)),
    "f8E4M3":         (mx.float16, np.dtype(_mld.float8_e4m3)),
    "f8E3M4":         (mx.float16, np.dtype(_mld.float8_e3m4)),
    "f8E8M0FNU":      (mx.float32, np.dtype(_mld.float8_e8m0fnu)),
    "f8E4M3B11FNUZ":  (mx.float16, np.dtype(_mld.float8_e4m3b11fnuz)),
    "f8E5M2FNUZ":     (mx.float16, np.dtype(_mld.float8_e5m2fnuz)),
    "f8E4M3FNUZ":     (mx.float16, np.dtype(_mld.float8_e4m3fnuz)),
    "f4E2M1FN":       (mx.float16, np.dtype(_mld.float4_e2m1fn)),
    "i4":             (mx.int8, np.dtype(_mld.int4)),
    "ui4":            (mx.uint8, np.dtype(_mld.uint4)),
}
for _name, (_mxdt, _npdt) in EMULATED.items():
    _MLIR_TO_MX[_name] = _mxdt
    _MLIR_TO_NP[_name] = _npdt
_EMULATED_BY_NP = {npdt: (name, mxdt) for name, (mxdt, npdt) in EMULATED.items()}


# jax carries ordered-effect tokens as bool[0] buffers at the runtime
# boundary (core.get_token_aval() is ShapedArray((0,), bool); dispatch
# seeds them with np.zeros(0, bool)), so that is what the interpreter
# reports for !stablehlo.token avals and hands back as a token value.
TOKEN_SHAPE = (0,)
TOKEN_NP_DTYPE = np.dtype(np.bool_)


def token_value() -> "mx.array":
    return mx.zeros(TOKEN_SHAPE, mx.bool_)

# LOGICAL bit width of an MLIR element type -- what XLA's memory layout uses,
# which for the emulated types above is NOT the storage width (an i4 lives in
# a whole int8, an f8 in a float16). Only bit-level ops (bitcast_convert)
# care; everything else works on values. i1 is deliberately absent: XLA's
# PRED occupies a full byte, like mx.bool_.
_LOGICAL_BITS = {
    "i4": 4, "ui4": 4, "f4E2M1FN": 4,
    "f8E4M3FN": 8, "f8E5M2": 8, "f8E4M3": 8, "f8E3M4": 8, "f8E8M0FNU": 8,
    "f8E4M3B11FNUZ": 8, "f8E5M2FNUZ": 8, "f8E4M3FNUZ": 8,
    "f64": 64,
}


def bit_width(t: ir.Type) -> int:
    """Bits one value of this element type occupies in XLA's memory layout."""
    b = _LOGICAL_BITS.get(str(t))
    return b if b is not None else mx_dtype_for(t).size * 8


def storage_is_bit_exact(t: ir.Type) -> bool:
    """True if our MLX storage has the same bit width as the logical type,
    i.e. the raw bytes of the storage ARE the type's bit pattern."""
    return bit_width(t) == mx_dtype_for(t).size * 8


def np_dtype_for_mlir(t: ir.Type) -> np.dtype:
    """The numpy dtype JAX expects for this element type (f64 reported as-is)."""
    s = str(t)
    try:
        return _MLIR_TO_NP[s]
    except KeyError:
        raise UnsupportedDtypeError(
            f"MLIR element type '{s}' is not supported on Metal"
        ) from None

_MX_TO_NP = {
    mx.complex64: np.dtype(np.complex64),
    mx.float16: np.dtype(np.float16),
    mx.float32: np.dtype(np.float32),
    mx.bfloat16: np.dtype(ml_dtypes.bfloat16),
    mx.bool_: np.dtype(np.bool_),
    mx.int8: np.dtype(np.int8),
    mx.int16: np.dtype(np.int16),
    mx.int32: np.dtype(np.int32),
    mx.int64: np.dtype(np.int64),
    mx.uint8: np.dtype(np.uint8),
    mx.uint16: np.dtype(np.uint16),
    mx.uint32: np.dtype(np.uint32),
    mx.uint64: np.dtype(np.uint64),
}

_NP_TO_MX = {v: k for k, v in _MX_TO_NP.items()}


def mx_dtype_for(t: ir.Type) -> mx.Dtype:
    """MLX dtype for an MLIR tensor *element* type."""
    s = str(t)
    if s == "f64":
        # Storage dtype. In strict mode the interpreter's compile-time check
        # already rejected any op that would *compute* in f64.
        if F64_DOWNCAST:
            _warn_f64()
        return mx.float32
    try:
        return _MLIR_TO_MX[s]
    except KeyError:
        raise UnsupportedDtypeError(
            f"MLIR element type '{s}' is not supported on Metal "
            f"(no float64/complex128 on this backend)"
        ) from None


def mx_result_dtype(value: ir.Value) -> mx.Dtype:
    return mx_dtype_for(ir.RankedTensorType(value.type).element_type)


def np_dtype_for(d: mx.Dtype) -> np.dtype:
    return _MX_TO_NP[d]


def mx_dtype_for_np(d: np.dtype) -> mx.Dtype:
    try:
        return _NP_TO_MX[np.dtype(d)]
    except KeyError:
        raise UnsupportedDtypeError(f"numpy dtype {d} unsupported on Metal") from None


def to_mx(arr: np.ndarray) -> mx.array:
    """numpy -> mx.array, handling ml_dtypes.bfloat16 which MLX can't ingest directly."""
    arr = np.asarray(arr)
    if not arr.flags.c_contiguous:
        # np.ascontiguousarray promotes 0-d to (1,); restore the shape.
        arr = np.ascontiguousarray(arr).reshape(arr.shape)
    em = _EMULATED_BY_NP.get(arr.dtype)
    if em is not None:
        name, mxdt = em
        wide = np.int8 if name == "i4" else (
            np.uint8 if name == "ui4" else np.float32)
        return mx.array(arr.astype(wide)).astype(mxdt)
    if arr.dtype == ml_dtypes.bfloat16:
        return mx.array(arr.astype(np.float32)).astype(mx.bfloat16)
    if arr.dtype == np.float64:
        if F64_DOWNCAST:
            _warn_f64()
        return mx.array(arr.astype(np.float32))
    return mx.array(arr)


def to_np(arr: mx.array) -> np.ndarray:
    if arr.size == 0:
        # Never touch an empty array's buffer: some MLX ops (matmul with
        # an empty output dim, MLX 0.32) produce 0-size arrays whose host
        # conversion segfaults on a null data pointer.
        return np.empty(arr.shape, dtype=_MX_TO_NP[arr.dtype])
    if arr.dtype == mx.bfloat16:
        return np.array(arr.astype(mx.float32)).astype(ml_dtypes.bfloat16)
    return np.array(arr)


def is_bool(d: mx.Dtype) -> bool:
    return d == mx.bool_


def is_float(d: mx.Dtype) -> bool:
    return d in (mx.float32, mx.float16, mx.bfloat16)


_TO_UINT = {mx.float32: (mx.uint32, mx.int32),
            mx.float16: (mx.uint16, mx.int16),
            mx.bfloat16: (mx.uint16, mx.int16)}


def total_order_key(x: mx.array) -> mx.array:
    """Unsigned-int key whose ascending order is IEEE-754 totalOrder
    (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN)."""
    ut, it = _TO_UINT[x.dtype]
    u = mx.view(x, ut)
    neg = mx.view(x, it) < 0
    top = mx.array(1 << (u.dtype.size * 8 - 1), dtype=ut)
    return mx.where(neg, ~u, u | top)


def is_int(d: mx.Dtype) -> bool:
    return d in (
        mx.int8, mx.int16, mx.int32, mx.int64,
        mx.uint8, mx.uint16, mx.uint32, mx.uint64,
    )


def is_unsigned(d: mx.Dtype) -> bool:
    return d in (mx.uint8, mx.uint16, mx.uint32, mx.uint64)


def make_complex(re: mx.array, im: mx.array) -> mx.array:
    """complex64 from its parts by bit interleave.

    Building the value arithmetically (re + im*1j) runs a complex multiply
    that destroys exactly the values XLA/C99 specify: (1, inf) becomes
    (nan, inf) because inf*0 is nan, and -0 + 0 collapses to +0. Writing
    the halves into place keeps every bit.
    """
    re = re.astype(mx.float32)
    im = im.astype(mx.float32)
    if re.shape != im.shape:
        re, im = mx.broadcast_arrays(re, im)
    pair = mx.stack([re, im], axis=-1)
    return mx.reshape(mx.view(pair, mx.complex64), re.shape)


_UNSIGNED_OF = {
    mx.int8: mx.uint8, mx.int16: mx.uint16, mx.int32: mx.uint32, mx.int64: mx.uint64,
}


def unsigned_of(d: mx.Dtype) -> mx.Dtype:
    return _UNSIGNED_OF.get(d, d)


def quantize_emulated(x: mx.array, mlir_name: str) -> mx.array:
    """Round values onto an emulated dtype's grid (storage dtype stays
    wide). Normal range: RNE mantissa rounding in f32 bits; subnormal
    range: uniform-grid rounding; overflow saturates (ml_dtypes
    convention); NaN passes through."""
    storage, npdt = EMULATED[mlir_name]
    if mlir_name in ("i4",):
        v = ((x.astype(mx.int32) + 8) % 16) - 8
        return v.astype(mx.int8)
    if mlir_name in ("ui4",):
        return (x.astype(mx.int32) % 16).astype(mx.uint8)
    fi = _mld.finfo(npdt)
    man = fi.nmant
    if mlir_name == "f8E8M0FNU":
        # exponent-only log-scale format: nearest power of two
        f = mx.maximum(x.astype(mx.float32), mx.array(1e-45, mx.float32))
        e = mx.round(mx.log2(f))
        e = mx.clip(e, float(fi.minexp), float(fi.maxexp))
        return mx.power(mx.array(2.0, mx.float32), e)
    f = x.astype(mx.float32)
    isnan = mx.isnan(f)
    # RNE mantissa rounding in f32 bit-space
    u = mx.view(f, mx.uint32)
    shift = 23 - man
    half = mx.array((1 << (shift - 1)) - 1, dtype=mx.uint32)
    lsb = (u >> shift) & mx.array(1, dtype=mx.uint32)
    u = (u + half + lsb) & mx.array(~((1 << shift) - 1) & 0xFFFFFFFF,
                                    dtype=mx.uint32)
    rounded = mx.view(u, mx.float32)
    # subnormal range: uniform grid of spacing tiny * 2^-man
    tiny = float(fi.tiny)
    step = tiny / (1 << man)
    sub = mx.round(f / step) * step
    q = mx.where(mx.abs(f) < tiny, sub, rounded)
    # overflow: formats with inf produce +-inf, finite-only (FN/FNUZ)
    # formats produce NaN (XLA convert semantics; ml_dtypes saturates,
    # but the CPU backend does not)
    mx_max = float(fi.max)
    over = mx.abs(q) > mx_max
    has_inf = mlir_name in ("f8E5M2", "f8E4M3", "f8E3M4")
    if has_inf:
        q = mx.where(over, mx.sign(q) * mx.array(float("inf"), mx.float32),
                     q)
    else:
        q = mx.where(over, mx.array(float("nan"), mx.float32), q)
    q = mx.where(isnan, f, q)
    return q.astype(storage)
