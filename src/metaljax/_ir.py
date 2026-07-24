"""MLIR context construction and attribute helpers.

Deliberately depends only on jaxlib (not jax): this module gets imported
inside the PJRT plugin trampoline while jax itself may still be mid-import.
"""

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import chlo, stablehlo
from jaxlib.mlir._mlir_libs import _jax_mlir_ext as jax_mlir_ext

_upstream = ir.DialectRegistry()
jax_mlir_ext.register_dialects(_upstream)


def make_context() -> ir.Context:
    ctx = ir.Context()
    ctx.append_dialect_registry(_upstream)
    ctx.load_all_available_dialects()
    stablehlo.register_dialect(ctx)
    chlo.register_dialect(ctx)
    # jax modules may carry sharding (sdy/mpmd) attrs even on one device.
    for name in ("sdy", "mpmd"):
        try:
            mod = __import__(f"jaxlib.mlir.dialects.{name}", fromlist=[name])
            mod.register_dialect(ctx)
        except Exception:
            pass
    return ctx


def i64_list(op: ir.Operation, name: str) -> list[int]:
    """Decode an i64 array-ish attribute (DenseI64ArrayAttr or DenseIntElementsAttr)."""
    attr = op.attributes[name]
    try:
        return list(ir.DenseI64ArrayAttr(attr))
    except ValueError:
        return [int(x) for x in ir.DenseIntElementsAttr(attr)]


def int_attr(op: ir.Operation, name: str) -> int:
    return ir.IntegerAttr(op.attributes[name]).value


def str_attr(op: ir.Operation, name: str) -> str:
    return ir.StringAttr(op.attributes[name]).value


def tensor_type(value: ir.Value) -> ir.RankedTensorType:
    return ir.RankedTensorType(value.type)


_TEXT_NP_DTYPES = {
    "bf16": "bfloat16",  # resolved via ml_dtypes below
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
    "ui8": "uint8", "ui16": "uint16", "ui32": "uint32", "ui64": "uint64",
}


def dense_to_np(attr, ttype: ir.RankedTensorType):
    """DenseElementsAttr -> numpy array, with a text-parsing fallback for
    element types the bindings can't export via the buffer protocol (bf16, i1...)."""
    import numpy as np

    shape = tuple(ttype.shape)
    try:
        arr = np.array(ir.DenseElementsAttr(attr))
        if arr.dtype != object:
            if arr.shape != shape:
                arr = np.broadcast_to(arr, shape)
            return arr
    except Exception:
        pass

    import ml_dtypes
    el = str(ttype.element_type)
    name = _TEXT_NP_DTYPES.get(el)
    if name is None:
        raise TypeError(f"cannot decode dense constant of element type {el}")
    np_dtype = np.dtype(ml_dtypes.bfloat16) if name == "bfloat16" else np.dtype(name)

    s = str(attr)
    if not s.startswith("dense<"):
        raise TypeError(f"unexpected constant attribute form: {s[:60]}")
    body = s[len("dense<"):s.index("> : ")]

    if body.startswith('"0x'):
        raw = bytes.fromhex(body[3:-1])
        return np.frombuffer(raw, np_dtype).reshape(shape)

    import re
    toks = [t for t in re.split(r"[\s,\[\]]+", body) if t]

    def decode(tok):
        if tok == "true":
            return True
        if tok == "false":
            return False
        if tok.startswith("0x") or tok.startswith("-0x"):
            neg = tok.startswith("-")
            bits = int(tok.lstrip("-"), 16)
            if np_dtype.kind == "f":
                v = np.array([bits], dtype=f"u{np_dtype.itemsize}").view(np_dtype)[0]
                return -v if neg else v
            return -bits if neg else bits
        if np_dtype.kind in "iub":
            return int(tok)
        return float(tok)

    vals = [decode(t) for t in toks]
    if len(vals) == 1:
        return np.broadcast_to(np.array(vals[0], dtype=np_dtype), shape)
    return np.array(vals, dtype=np_dtype).reshape(shape)
