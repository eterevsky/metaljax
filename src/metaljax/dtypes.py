"""MLIR element type <-> MLX dtype <-> numpy dtype mappings."""

import os
import warnings

import ml_dtypes
import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir


class UnsupportedDtypeError(TypeError):
    pass


# Metal has no float64. Default policy: compute in f32 while reporting f64
# avals/buffers back to JAX (texmo runs with jax_enable_x64=True).
F64_DOWNCAST = os.environ.get("METALJAX_F64", "downcast") == "downcast"
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


def np_dtype_for_mlir(t: ir.Type) -> np.dtype:
    """The numpy dtype JAX expects for this element type (f64 reported as-is)."""
    s = str(t)
    if s == "f64" and not F64_DOWNCAST:
        raise UnsupportedDtypeError("float64 unsupported on Metal (METALJAX_F64=error)")
    try:
        return _MLIR_TO_NP[s]
    except KeyError:
        raise UnsupportedDtypeError(
            f"MLIR element type '{s}' is not supported on Metal"
        ) from None

_MX_TO_NP = {
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
        if F64_DOWNCAST:
            _warn_f64()
            return mx.float32
        raise UnsupportedDtypeError("float64 unsupported on Metal (METALJAX_F64=error)")
    try:
        return _MLIR_TO_MX[s]
    except KeyError:
        raise UnsupportedDtypeError(
            f"MLIR element type '{s}' is not supported on Metal "
            f"(no float64/complex on this backend)"
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
    if arr.dtype == ml_dtypes.bfloat16:
        return mx.array(arr.astype(np.float32)).astype(mx.bfloat16)
    if arr.dtype == np.float64:
        if not F64_DOWNCAST:
            raise UnsupportedDtypeError("float64 unsupported on Metal")
        _warn_f64()
        return mx.array(arr.astype(np.float32))
    return mx.array(arr)


def to_np(arr: mx.array) -> np.ndarray:
    if arr.dtype == mx.bfloat16:
        return np.array(arr.astype(mx.float32)).astype(ml_dtypes.bfloat16)
    return np.array(arr)


def is_bool(d: mx.Dtype) -> bool:
    return d == mx.bool_


def is_int(d: mx.Dtype) -> bool:
    return d in (
        mx.int8, mx.int16, mx.int32, mx.int64,
        mx.uint8, mx.uint16, mx.uint32, mx.uint64,
    )


def is_unsigned(d: mx.Dtype) -> bool:
    return d in (mx.uint8, mx.uint16, mx.uint32, mx.uint64)


_UNSIGNED_OF = {
    mx.int8: mx.uint8, mx.int16: mx.uint16, mx.int32: mx.uint32, mx.int64: mx.uint64,
}


def unsigned_of(d: mx.Dtype) -> mx.Dtype:
    return _UNSIGNED_OF.get(d, d)
