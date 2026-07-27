"""Python side of the PJRT trampoline.

The native plugin (plugin/metal_pjrt.cc) calls the module-level functions here
via the CPython C API. Depends only on jaxlib/mlx/numpy — never on jax itself,
since these run while jax may still be mid-initialization.
"""

from __future__ import annotations

import hashlib
import os

import ml_dtypes
import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes
from metaljax.interpreter import Interpreter

# PJRT_Buffer_Type enum values (pjrt_c_api.h, PJRT API 0.114).
_PJRT_TYPES = [
    "INVALID", "PRED", "S8", "S16", "S32", "S64", "U8", "U16", "U32", "U64",
    "F16", "F32", "F64", "BF16", "C64", "C128",
    "F8E5M2", "F8E4M3FN", "F8E4M3B11FNUZ", "F8E5M2FNUZ", "F8E4M3FNUZ",
    "S4", "U4", "TOKEN", "S2", "U2", "F8E4M3", "F8E3M4", "F8E8M0FNU",
    "F4E2M1FN", "S1", "U1", "F6E2M3FN", "F6E3M2FN",
]
_ENUM = {name: i for i, name in enumerate(_PJRT_TYPES)}

_ENUM_TO_NP = {
    _ENUM["PRED"]: np.dtype(np.bool_),
    _ENUM["S8"]: np.dtype(np.int8),
    _ENUM["S16"]: np.dtype(np.int16),
    _ENUM["S32"]: np.dtype(np.int32),
    _ENUM["S64"]: np.dtype(np.int64),
    _ENUM["U8"]: np.dtype(np.uint8),
    _ENUM["U16"]: np.dtype(np.uint16),
    _ENUM["U32"]: np.dtype(np.uint32),
    _ENUM["U64"]: np.dtype(np.uint64),
    _ENUM["F16"]: np.dtype(np.float16),
    _ENUM["F32"]: np.dtype(np.float32),
    _ENUM["F64"]: np.dtype(np.float64),  # stored as f32 on device (downcast)
    _ENUM["BF16"]: np.dtype(ml_dtypes.bfloat16),
}
_NP_TO_ENUM = {v: k for k, v in _ENUM_TO_NP.items()}


class MetalBuffer:
    """A device buffer: an mx.array plus PJRT-visible metadata."""

    __slots__ = ("data", "type_enum", "dims", "nbytes")

    def __init__(self, data: mx.array, type_enum: int, dims: list[int]):
        self.data = data
        self.type_enum = int(type_enum)
        self.dims = [int(d) for d in dims]
        n = 1
        for d in self.dims:
            n *= d
        self.nbytes = n * _ENUM_TO_NP[self.type_enum].itemsize


def buffer_from_host(data, type_enum: int, dims, byte_strides) -> MetalBuffer:
    dims = list(dims)
    np_dtype = _ENUM_TO_NP.get(type_enum)
    if np_dtype is None:
        raise TypeError(
            f"PJRT buffer type {_PJRT_TYPES[type_enum]} not supported on metal"
        )
    if data is None:
        arr = np.zeros(dims, np_dtype)
    elif byte_strides is None:
        arr = np.frombuffer(data, np_dtype).reshape(dims)
    else:
        arr = np.ndarray(shape=dims, dtype=np_dtype, buffer=data,
                         strides=list(byte_strides))
        arr = np.ascontiguousarray(arr).reshape(dims)
    return MetalBuffer(dtypes.to_mx(arr), type_enum, dims)


def to_host(buf: MetalBuffer) -> bytes:
    arr = dtypes.to_np(buf.data)
    want = _ENUM_TO_NP[buf.type_enum]
    if arr.dtype != want:  # e.g. f64 buffers stored as f32 on device
        arr = arr.astype(want)
    return arr.tobytes()


class MetalExecutable:
    def __init__(self, interp: Interpreter, name: str, fingerprint: str):
        self.interpreter = interp
        self.name = name
        self.fingerprint = fingerprint
        outs = interp.out_avals
        ins = interp.in_avals
        self.num_params = len(ins)
        self.num_outputs = len(outs)
        self.out_types = [_NP_TO_ENUM[np.dtype(dt)] for _, dt in outs]
        self.out_dims = [list(shape) for shape, _ in outs]
        self._compiled = None
        self._can_compile = None  # resolved lazily on first execute

    def runner(self):
        """The callable executing this program, mx.compile'd when possible."""
        from metaljax.interpreter import COMPILE_ENABLED
        from metaljax.ops import control

        interp = self.interpreter
        if self._can_compile is None:
            with interp.context:
                self._can_compile = (
                    COMPILE_ENABLED
                    and interp.main_pure
                    and control._block_cost(interp, interp._main_block())
                    <= control._TRACE_BUDGET
                )
                if control._DEBUG:
                    print(f"[metaljax] exec {self.name}: pure="
                          f"{interp.main_pure} cost="
                          f"{control._block_cost(interp, interp._main_block())} "
                          f"compile={self._can_compile}", flush=True)
        if not self._can_compile:
            return interp
        if self._compiled is None:
            with interp.context:
                underived = control._underived_outputs(
                    interp._main_block(), [])

            def traced(*a):
                prev = interp._in_trace
                interp._in_trace = True
                try:
                    outs = tuple(interp(*a))
                    return control._anchor_outputs(outs, a, underived)
                finally:
                    interp._in_trace = prev

            self._compiled = mx.compile(traced)
        return self._compiled


def compile_program(code: bytes, fmt: str) -> MetalExecutable:
    if fmt not in ("mlir",):
        raise ValueError(f"unsupported program format {fmt!r}")
    # New program = config boundary: cached buffers from prior shapes are
    # dead weight against the Metal buffer-count limit.
    mx.clear_cache()
    from jaxlib.mlir.dialects import stablehlo

    ctx = _ir.make_context()
    # JAX serializes plugin programs as StableHLO portable artifacts (VHLO
    # bytecode); a raw parse "succeeds" on those but yields vhlo.* ops, so try
    # proper deserialization first.
    try:
        module = stablehlo.deserialize_portable_artifact(ctx, code)
    except Exception:
        with ctx:
            module = ir.Module.parse(code)

    name = "metaljax_exec"
    try:
        with ctx:
            attr = module.operation.attributes["sym_name"]
            name = ir.StringAttr(attr).value
    except Exception:
        pass

    interp = Interpreter(module, context=ctx)
    fingerprint = hashlib.sha256(code).hexdigest()
    return MetalExecutable(interp, name, fingerprint)


# Blocking eval costs a full Metal command-buffer roundtrip (~190us) per
# execute — ruinous for eager per-primitive dispatch. async_eval submits the
# work (~17us) and lets device->host reads synchronize on demand.
# METALJAX_SYNC=1 restores blocking eval (errors then surface at the call).
_SYNC = os.environ.get("METALJAX_SYNC", "0") == "1"

# Metal caps LIVE MTLBuffers at ~499k (device_info()["resource_limit"]) and
# MLX's buffer cache is bounded by BYTES, not count — a long-lived process
# compiling many programs of different shapes (a texmo worker sweeping
# configurations) accumulates freed small buffers in the cache until the
# count limit kills an unrelated allocation. Clearing at compile boundaries
# (old shapes are garbage then) plus a periodic execute backstop keeps the
# count bounded. METALJAX_CLEAR_PERIOD=0 disables the backstop.
_CLEAR_PERIOD = int(os.environ.get("METALJAX_CLEAR_PERIOD", "50000"))
_exec_count = 0


def execute(ex: MetalExecutable, buffers) -> list[MetalBuffer]:
    global _exec_count
    _exec_count += 1
    if _CLEAR_PERIOD and _exec_count % _CLEAR_PERIOD == 0:
        mx.clear_cache()
    args = [b.data for b in buffers]
    try:
        outs = list(ex.runner()(*args))
        if outs:
            if _SYNC:
                mx.eval(*outs)
            else:
                mx.async_eval(*outs)
    except (RuntimeError, IndexError, ValueError) as e:
        if isinstance(e, RuntimeError) and "Resource limit" in str(e):
            # Metal buffer-handle exhaustion, not a graph problem: purge
            # MLX's cache and retry the same path once (programs are pure,
            # so re-running from unchanged inputs is safe).
            mx.clear_cache()
            outs = list(ex.runner()(*args))
            if outs:
                mx.eval(*outs)
        else:
            # MLX's compiler can reject certain traces (fused-kernel
            # argument-buffer exhaustion; unordered_map::at on graphs with
            # unused inputs surfaces as IndexError). Retry eagerly.
            if not ex._can_compile:
                raise
            ex._can_compile = False
            ex._compiled = None
            outs = list(ex.interpreter(*args))
            if outs:
                mx.eval(*outs)
    res = []
    for arr, type_enum, dims in zip(outs, ex.out_types, ex.out_dims):
        res.append(MetalBuffer(arr, type_enum, dims))
    return res


def device_kind() -> str:
    try:
        return mx.device_info()["device_name"]
    except Exception:
        return "Apple GPU"


def stablehlo_version() -> list[int]:
    from jaxlib.mlir.dialects import stablehlo
    return [int(x) for x in stablehlo.get_current_version().split(".")]
