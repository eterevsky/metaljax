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


def is_token(t: ir.Type) -> bool:
    """True for !stablehlo.token — the value jax threads through function
    signatures to sequence ordered effects (jax.debug.print(ordered=True),
    io_callback(ordered=True)). Not a tensor type: everything that assumes
    RankedTensorType must check this first."""
    try:
        return stablehlo.TokenType.isinstance(t)
    except Exception:
        return str(t) == "!stablehlo.token"


_TEXT_NP_DTYPES = {
    "bf16": "bfloat16",  # resolved via ml_dtypes below
    "i4": "int4", "ui4": "uint4",
    "f8E4M3FN": "float8_e4m3fn", "f8E5M2": "float8_e5m2",
    "f8E4M3": "float8_e4m3", "f8E3M4": "float8_e3m4",
    "f8E8M0FNU": "float8_e8m0fnu",
    "f8E4M3B11FNUZ": "float8_e4m3b11fnuz",
    "f8E5M2FNUZ": "float8_e5m2fnuz", "f8E4M3FNUZ": "float8_e4m3fnuz",
    "f4E2M1FN": "float4_e2m1fn",
    "f6E2M3FN": "float6_e2m3fn", "f6E3M2FN": "float6_e3m2fn",
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
    el = str(ttype.element_type)
    # bf16 must NOT go through the buffer-protocol branch: the bindings
    # "succeed" on hex-form attrs (dense<0xFF80>) by converting the hex
    # integer to float instead of reinterpreting the bits (-inf became
    # 65536.0). The text fallback decodes both forms correctly.
    if el != "bf16":
        try:
            arr = np.array(ir.DenseElementsAttr(attr))
            if arr.dtype != object:
                if arr.shape != shape:
                    arr = np.broadcast_to(arr, shape)
                return arr
        except Exception:
            pass

    if el.startswith("complex"):
        return complex_dense_from_text(
            str(attr), shape,
            np.complex128 if el == "complex<f64>" else np.complex64)

    import ml_dtypes
    name = _TEXT_NP_DTYPES.get(el)
    if name is None:
        raise TypeError(f"cannot decode dense constant of element type {el}")
    np_dtype = (np.dtype(getattr(ml_dtypes, name))
                if hasattr(ml_dtypes, name) else np.dtype(name))

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
            # Hex float literals are bit patterns. NB ml_dtypes' bfloat16
            # has dtype kind 'V', so test the MLIR type, not np_dtype.kind.
            if el.startswith("f") or el == "bf16":
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


def complex_dense_from_text(text: str, shape: tuple,
                            np_dtype=None) -> "np.ndarray":
    """Decode a complex dense constant from MLIR text: dense<(re,im)>,
    dense<[(re,im), ...]>, or the hex-blob form. Needed because the
    python bindings cannot even *cast* complex dense attributes
    (op.attributes["value"] raises). np_dtype selects the component
    width (complex<f64> constants are pass-through data, decoded at
    their declared width and narrowed by dtypes.to_mx)."""
    import re

    import numpy as np

    if np_dtype is None:
        np_dtype = np.complex64
    np_dtype = np.dtype(np_dtype)
    part = np.dtype(f"f{np_dtype.itemsize // 2}")
    m = re.search(r"dense<(.*)> : ", text, re.S)
    if not m:
        raise TypeError(f"no dense constant in: {text[:80]}")
    body = m.group(1)
    if body.startswith('"0x'):
        return np.frombuffer(
            bytes.fromhex(body[3:-1]), np_dtype).reshape(shape)

    def num(tok):
        tok = tok.strip()
        if tok.startswith("0x"):
            bits = int(tok, 16).to_bytes(part.itemsize, "little")
            return np.frombuffer(bits, part)[0]
        return float(tok)

    pairs = re.findall(r"\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)", body)
    arr = np.array([complex(num(a), num(b)) for a, b in pairs], np_dtype)
    if arr.size == 1 and shape != (1,) * len(shape or (1,)):
        return np.broadcast_to(arr.reshape(()), shape).copy()
    return arr.reshape(shape)
