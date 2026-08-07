"""Python side of the PJRT trampoline.

The native plugin (plugin/metal_pjrt.cc) calls the module-level functions here
via the CPython C API. Depends only on jaxlib/mlx/numpy — never on jax itself,
since these run while jax may still be mid-initialization.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import os
import time

import ml_dtypes
import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes, qmm
from metaljax.interpreter import Interpreter

# --- Stage 2: the native replay engine (notes/cpp-migration-plan.md) ---
# METALJAX_ENGINE=native loads the C++ extension (native/build.sh) and,
# milestone by milestone, moves the runtime path there. Python remains
# the reference engine and the fallback for any program the native side
# does not support; "py" (default) never touches the extension.
ENGINE = os.environ.get("METALJAX_ENGINE", "py")
NATIVE = None
if ENGINE == "native":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                     "..", "..", "native", "build"))
    import metaljax_native as NATIVE  # noqa: N812
    # libmlx has no stable ABI: refuse a skewed native engine loudly at
    # import instead of corrupting at runtime. All three must agree —
    # the headers the extension compiled against, the libmlx it linked,
    # and the mlx the Python side imports (version lives on mlx.core;
    # the bare mlx package does not export it).
    _vers = (NATIVE.compiled_mlx_version(), NATIVE.linked_mlx_version(),
             mx.__version__)
    if len(set(_vers)) != 1:
        raise ImportError(
            f"metaljax native engine: mlx version skew {_vers} "
            f"(compiled/linked/python) — rebuild native/build.sh against "
            f"the installed mlx wheel")

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
    _ENUM["C64"]: np.dtype(np.complex64),
    _ENUM["C128"]: np.dtype(np.complex128),  # stored as c64 on device
    _ENUM["F8E5M2"]: np.dtype(ml_dtypes.float8_e5m2),
    _ENUM["F8E4M3FN"]: np.dtype(ml_dtypes.float8_e4m3fn),
    _ENUM["F8E4M3B11FNUZ"]: np.dtype(ml_dtypes.float8_e4m3b11fnuz),
    _ENUM["F8E5M2FNUZ"]: np.dtype(ml_dtypes.float8_e5m2fnuz),
    _ENUM["F8E4M3FNUZ"]: np.dtype(ml_dtypes.float8_e4m3fnuz),
    _ENUM["F8E4M3"]: np.dtype(ml_dtypes.float8_e4m3),
    _ENUM["F8E3M4"]: np.dtype(ml_dtypes.float8_e3m4),
    _ENUM["F8E8M0FNU"]: np.dtype(ml_dtypes.float8_e8m0fnu),
    _ENUM["F4E2M1FN"]: np.dtype(ml_dtypes.float4_e2m1fn),
    _ENUM["F6E2M3FN"]: np.dtype(ml_dtypes.float6_e2m3fn),
    _ENUM["F6E3M2FN"]: np.dtype(ml_dtypes.float6_e3m2fn),
    _ENUM["S4"]: np.dtype(ml_dtypes.int4),
    _ENUM["U4"]: np.dtype(ml_dtypes.uint4),
}
_NP_TO_ENUM = {v: k for k, v in _ENUM_TO_NP.items()}


# --------------------------------------------------------------------------
# heap reclamation
# --------------------------------------------------------------------------

# Reclamation points (compile boundaries, the execute backstop, buffer-limit
# recovery) collect Python cycles before clearing MLX's cache: dead managers
# and executables routinely sit in reference cycles, and clear_cache cannot
# free a buffer something still references (CLAUDE.md item 19).
#
# gc.collect() walks the WHOLE live heap, so it costs what the process holds,
# not what the boundary it guards left behind: ~20 ms into a fresh jax
# process, 130 ms once its caches have grown, ~100 ms on an LLM-sized heap.
# Programs arrive at wildly different rates -- a training worker compiles
# once per configuration, but eager per-primitive dispatch compiles one
# program per operation, and then the cost lands on every one of them
# (jax's sparse_bcoo_bcsr_test: 2,574 compiles in 60 tests, 83% of the
# file's wall time inside gc.collect, growing with the heap the whole way:
# the 3600 s release-gate timeout).
#
# So amortize by measured cost, not by a period: a collection here may take
# at most 1/_GC_DUTY of the wall time since the last one, which bounds the
# overhead at any heap size and still collects often in absolute terms
# (130 ms collection -> at most one every 6.5 s). Recovery paths force a
# collection regardless -- those run because memory has actually run out.
# METALJAX_COMPILE_GC_DUTY=0 restores a collection at every boundary.
_GC_DUTY = float(os.environ.get("METALJAX_COMPILE_GC_DUTY", "50"))
_gc_deadline = 0.0

GC_STATS = {"collections": 0, "skipped": 0, "seconds": 0.0}

# Stage 2 M2 observability: how many executables the native tape took, how
# many it declined, and — the number that must stay zero — how many blew up
# at run time and fell back mid-flight. Tests read it; METALJAX_DEBUG
# narrates the individual decisions.
NATIVE_STATS = {"lowered": 0, "declined": 0, "lower_errors": 0,
                "runs": 0, "run_failures": 0}


def reclaim(force: bool = False) -> bool:
    """Collect cycles (duty-limited unless `force`) and clear MLX's cache."""
    global _gc_deadline
    now = time.monotonic()
    if force or not _GC_DUTY or now >= _gc_deadline:
        gc.collect()
        done = time.monotonic()
        GC_STATS["collections"] += 1
        GC_STATS["seconds"] += done - now
        _gc_deadline = done + _GC_DUTY * (done - now)
        collected = True
    else:
        GC_STATS["skipped"] += 1
        collected = False
    # Cheap and heap-independent: always worth doing at a reclamation point.
    mx.clear_cache()
    return collected


# Device dtype the native to_host expects per PJRT enum (F64/C128 pass
# through stored narrow). Anything else falls back to the numpy path.
_EXPECTED_MX = {
    1: mx.bool_, 2: mx.int8, 3: mx.int16, 4: mx.int32, 5: mx.int64,
    6: mx.uint8, 7: mx.uint16, 8: mx.uint32, 9: mx.uint64,
    10: mx.float16, 11: mx.float32, 12: mx.float32, 13: mx.bfloat16,
    14: mx.complex64, 15: mx.complex64,
}


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


def buffer_from_host(data, type_enum: int, dims, byte_strides,
                     offset: int = 0) -> MetalBuffer:
    dims = list(dims)
    np_dtype = _ENUM_TO_NP.get(type_enum)
    if np_dtype is None:
        raise TypeError(
            f"PJRT buffer type {_PJRT_TYPES[type_enum]} not supported on metal"
        )
    if NATIVE is not None and NATIVE.native_type(type_enum):
        # M1: the C++ path — one staging copy straight into the array's own
        # storage, raw-byte bf16, in-place f64/c128 narrowing. The emulated
        # dtypes (i4/f8*/f6...) keep the numpy path below.
        arr = NATIVE.buffer_from_host(
            data, type_enum, [int(d) for d in dims],
            None if byte_strides is None else [int(s) for s in byte_strides],
            int(offset))
        return MetalBuffer(arr, type_enum, dims)
    if data is None:
        arr = np.zeros(dims, np_dtype)
    elif byte_strides is None:
        arr = np.frombuffer(data, np_dtype).reshape(dims)
    else:
        # offset: byte position of the logical [0, 0, ...] element within
        # `data` — nonzero when some stride is negative (flipped views).
        arr = np.ndarray(shape=dims, dtype=np_dtype, buffer=data,
                         offset=offset, strides=list(byte_strides))
        arr = np.ascontiguousarray(arr).reshape(dims)
    return MetalBuffer(dtypes.to_mx(arr), type_enum, dims)


def to_host(buf: MetalBuffer) -> bytes:
    if (NATIVE is not None and NATIVE.native_type(buf.type_enum)
            and buf.data.dtype == _EXPECTED_MX.get(buf.type_enum)):
        # The dtype check guards results whose device storage differs from
        # the enum's expected form (host-op products, dtype quirks); those
        # keep the numpy path's astype semantics.
        return bytes(NATIVE.to_host(buf.data, buf.type_enum))
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
        with interp.context:
            self.donated_params = interp.donated_args
            self.forwarded_outputs = interp.forwarded_outputs
        self._compiled = None
        self._can_compile = None  # resolved lazily on first execute
        self._sync_inputs = None  # ditto (a whole-module walk)
        self._msl_build_failed = False  # a generated kernel didn't compile
        # M2 native tape: None = not tried, False = declined or disabled,
        # else the C++ Program. Resolved on first execute, never retried.
        self._native_prog = None

    def disable_msl(self, plans=()):
        """Drop msl_scan persistent kernels after Metal's shader compiler
        rejected one ("Unable to build metal library").

        `plans` are the loops whose kernels were still unproven when the
        failure surfaced — the only ones that can be at fault, since a
        kernel that has evaluated once is built. Blacklisting just those
        keeps the rest of the program on the fast path; with no candidates
        (or on a repeat failure) the whole program stops planning kernels.
        Either way every cache that could hand out — or re-trace — the
        offending kernel is dropped. Per executable, not per process: a
        worker sweeping configurations keeps the fast path for the
        programs whose kernels do build."""
        interp = self.interpreter
        bad = [k for k, p in interp._msl_cache.items()
               if any(p is q for q in plans)]
        if bad:
            for k in bad:
                interp._msl_cache[k] = None   # cached "not eligible"
        else:
            interp._no_msl = True
            interp._msl_cache.clear()
        interp._msl_pending.clear()
        interp._body_cache.clear()      # compiled bodies embed the kernel
        interp._traceable_cache.clear()  # an msl plan made a loop traceable
        interp._cost_cache.clear()      # ...and cost it as one kernel (8)
        interp._bytes_cache.clear()     # ...and charged only its outputs
        interp._bytes_gated.clear()     # decisions taken under the old plans
        self._compiled = None
        self._can_compile = None        # re-decide with the new costs

    @property
    def sync_inputs(self) -> bool:
        """Whether execute must settle this program's inputs first.

        MLX's FFT kernels can run ahead of a pending async evaluation of
        their input (jax.numpy.fft crops/pads to `s` in a separate
        executable, and every execute ends in async_eval, so the transform
        read the padded buffer mid-copy). Programs that transform need
        their inputs materialized; ops.elementwise._fft repeats the barrier
        for values produced inside an eagerly interpreted program.
        """
        if self._sync_inputs is None:
            with self.interpreter.context:
                self._sync_inputs = self.interpreter.main_has_fft
        return self._sync_inputs

    def invalidate_traces(self):
        """Drop everything derived from the program's rewritten structure.

        Called when the quantized-matmul recognizer changes its mind at
        execute time (a first-execute check failed, or a repack picked a
        different group size): op costs, compiled bodies and the compiled
        main were all built around the previous rewrite.
        """
        interp = self.interpreter
        interp._body_cache.clear()
        interp._cost_cache.clear()
        # Absorbed ops are charged nothing, so a changed rewrite changes the
        # byte estimate exactly as it changes the op count.
        interp._bytes_cache.clear()
        interp._bytes_gated.clear()
        interp._traceable_cache.clear()
        self._compiled = None
        self._can_compile = None

    def native_program(self):
        """This program as a native tape (Stage 2 M2), or None.

        Resolved once, on the first execute, and cached either way: the
        answer is a static property of the module and re-asking would mean
        walking it again on every call. None means the Python engine runs
        the program, which is always correct — the native op set is a
        subset and grows monotonically (see metaljax.tape).

        The recognizer gate is here rather than in the lowering because it
        is about the ANALYSIS, not the ops: a rewritten program's roots
        execute `emit` instead of themselves and its absorbed ops do not
        execute at all, so the literal op list the tape would walk is not
        what the Python engine runs. qmm's prologue has already analyzed by
        the time execute asks; sdpa analyzes lazily.
        """
        if NATIVE is None or self._native_prog is False:
            return None
        if self._native_prog is None:
            self._native_prog = False  # decided now, whatever happens below
            interp = self.interpreter
            try:
                # M3: a compile-eligible program is no longer held back —
                # the native tape traces through mx::compile itself, so
                # taking it costs nothing the fused-graph replay was
                # providing. The DECISION stays Python's (same estimators,
                # one answer for both engines); the tape merely records it.
                prog = self._lower_native(interp, self.can_compile())
                if prog is not None:
                    self._native_prog = prog
            except Exception as e:  # a lowering bug must not break a program
                NATIVE_STATS["lower_errors"] += 1
                if os.environ.get("METALJAX_DEBUG", "") == "1":
                    print(f"[metaljax] native tape lowering failed: {e!r}",
                          flush=True)
            NATIVE_STATS["lowered" if self._native_prog is not False
                         else "declined"] += 1
        return None if self._native_prog is False else self._native_prog

    def _lower_native(self, interp, compile_main):
        qst = interp._qmm
        if qst is not None and (qst.matches or qst.moe):
            return None
        # NB no purity gate. Until M3 the tape declined every impure
        # program wholesale, which was really a decline of control flow;
        # now while/if/case lower, and everything else impure — host
        # callbacks, LAPACK on the host, tokens — is still declined where
        # it belongs, by the op set and the dtype table.
        from metaljax import sdpa as _sdpa, tape
        prog = tape.lower(interp, compile_main=compile_main)
        if prog is None:
            return None
        # The sdpa question comes LAST, after a program has proved itself
        # lowerable, because asking it is not free of consequences: sdpa's
        # analysis consults qmm's candidate list, and control._block_cost
        # reads `interp._qmm` directly and discounts the ops a candidate
        # absorbs — so merely ASKING can change the Python engine's compile
        # decision for a program that was never going to be rewritten
        # (visible with qmm's packing prologue turned off, where nothing
        # else would have analyzed). Asking only about programs the tape
        # can already run keeps the two engines' decisions independent.
        with interp.context:
            if _sdpa.analyze(interp).matches:
                return None
        return prog

    def can_compile(self) -> bool:
        """Whether the whole main compiles (resolves the lazy decision)."""
        self.runner()
        return bool(self._can_compile)

    def runner(self):
        """The callable executing this program, mx.compile'd when possible."""
        from metaljax.interpreter import COMPILE_ENABLED
        from metaljax.ops import control

        interp = self.interpreter
        if self._can_compile is None:
            with interp.context:
                blk = interp._main_block()
                cost = control._block_cost(interp, blk)
                # Two independent budgets. `cost` bounds how many ops one
                # mx.compile trace may hold (and so the Metal live-buffer
                # count); `bytes` bounds how much those buffers hold, which
                # op count says nothing about — a jitted parameter init is
                # 365 ops and 15 GB of traffic. Over either budget the
                # program runs op by op, where last-use pruning and the
                # byte-denominated flush bound it, which the compiled path
                # does not (notes/eager-memory-2026-08.md).
                nbytes = control.program_bytes(interp, blk)
                self._can_compile = (
                    COMPILE_ENABLED
                    and interp.main_pure
                    and cost <= control._TRACE_BUDGET
                    and control._bytes_ok(interp, blk, 1, f"main {self.name}",
                                          whole=True)
                )
                if control._DEBUG:
                    # `bytes` is this program's estimated traced result data
                    # (control.program_bytes: loops unrolled, splats
                    # broadcast, absorbed ops free but their fused roots
                    # charged what `emit` builds, plus pass-through outputs)
                    # — the quantity the gate acts on, and it is traffic, so
                    # it reads ~2.8x what the run peaks at. `eagerbytes` is
                    # the flat per-op sum the eager flush meters (no loop
                    # unrolling), kept because it says which programs that
                    # safety net can act on. `unsized` counts values no rule
                    # could size: nonzero means `bytes` is a floor, not a
                    # bound, and wants looking at.
                    from metaljax.interpreter import BYTES_UNKNOWN
                    print(f"[metaljax] exec {self.name}: pure="
                          f"{interp.main_pure} cost={cost} "
                          f"bytes={nbytes / 2**20:.1f}MB "
                          f"eagerbytes="
                          f"{sum(interp.eager_plan(blk, interp._rewrite_plan(blk))[1]) / 2**20:.1f}MB "
                          f"unsized={BYTES_UNKNOWN['unsized']} "
                          f"compile={self._can_compile}", flush=True)
        if not self._can_compile:
            return interp
        if self._compiled is None:
            with interp.context:
                underived = control._underived_outputs(
                    interp._main_block(), [])
            nargs = self.num_params

            def traced(*flat):
                a = flat[:nargs]
                # Packed quantized weights arrive as real inputs, not as
                # constants captured from the prologue: mx.compile bakes
                # captured arrays by value and would never see a repack.
                token = qmm.push(interp, flat[nargs:])
                prev = interp._in_trace
                interp._in_trace = True
                try:
                    outs = tuple(interp(*a))
                    return control._anchor_outputs(outs, a, underived)
                finally:
                    interp._in_trace = prev
                    qmm.pop(interp, token)

            compiled = mx.compile(traced)

            def call(*a):
                return compiled(*a, *qmm.values(interp))

            self._compiled = call
        return self._compiled


def compile_program(code: bytes, fmt: str,
                    options: bytes | None = None) -> MetalExecutable:
    if fmt not in ("mlir",):
        raise ValueError(f"unsupported program format {fmt!r}")
    # `options` is the serialized CompileOptionsProto. metaljax applies none
    # of it, but an option XLA would refuse has to be refused here too --
    # accepting a misspelled flag silently is how a caller ends up believing
    # a dump/debug switch took effect. See metaljax.compile_options.
    from metaljax import compile_options as _copts
    _copts.validate(options)
    # New program = config boundary: cached buffers from prior shapes are
    # dead weight against the Metal buffer-count limit, and the managers and
    # executables that pinned them often sit in reference cycles (Python's
    # gc triggers on allocation counts, which array workloads barely tick).
    # `reclaim` collects those, duty-limited: see its comment — a compile
    # boundary is where reclamation is CHEAPEST to reach, not where it is
    # urgent, and one compiled program is not evidence that anything died.
    reclaim()
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
        # 50k executes is evidence of real work, not of a cheap boundary:
        # collect whatever the duty cycle would have deferred.
        reclaim(force=True)
    args = [b.data for b in buffers]
    if args and ex.sync_inputs:
        mx.eval(*args)
    # Quantized weights are repacked here, eagerly, on concrete buffers:
    # packing cannot happen inside an mx.compile trace, and the result is
    # cached on the source buffers' identity so this is a no-op from the
    # second execute on.
    if qmm.ENABLED and qmm.prologue(ex.interpreter, args):
        ex.invalidate_traces()
    # NB retries happen AFTER the except block: the in-flight exception's
    # traceback pins every array of the failed attempt (a failed
    # whole-main trace can hold hundreds of thousands of buffers), so a
    # retry inside the handler inherits the exhaustion it is retrying.
    retry = None  # "same" | "eager" | "no_msl"
    pending = ex.interpreter._msl_pending
    pending.clear()
    # Stage 2 M2: the native tape, when this program lowered. It returns
    # LAZY arrays like every other path — flush policy stays here — and any
    # failure hands the call back to the Python engine and retires the tape
    # for this executable (the fallback is what makes the native op set
    # safe to grow).
    outs = None
    prog = ex.native_program()
    if prog is not None:
        try:
            outs = list(prog.run(inputs=args))
            NATIVE_STATS["runs"] += 1
            if outs:
                if _SYNC:
                    mx.eval(*outs)
                else:
                    mx.async_eval(*outs)
        except Exception as e:
            NATIVE_STATS["run_failures"] += 1
            ex._native_prog = False
            outs = None
            print(f"[metaljax] native tape failed at run time ({e!r}); "
                  f"falling back to the Python engine for {ex.name}",
                  flush=True)
    if outs is not None:
        return _finish(ex, args, outs)
    try:
        outs = list(ex.runner()(*args))
        if outs:
            # An msl_scan kernel that was traced into a compiled graph has
            # never been built: settle synchronously so a Metal build error
            # raises here (in an async worker it would abort the process).
            if _SYNC or pending:
                mx.eval(*outs)
            else:
                mx.async_eval(*outs)
        for plan in pending:
            plan._validated = True
        pending.clear()
    except (RuntimeError, IndexError, ValueError) as e:
        if isinstance(e, RuntimeError) and "Unable to build metal library" in str(e):
            # A generated msl_scan kernel did not compile. The program is
            # pure, so re-run it without the kernels that could be at
            # fault. BOUNDED: the second failure disables the whole
            # persistent-kernel path for this program, after which the
            # error can no longer come from a generated kernel.
            if ex.interpreter._no_msl:
                raise
            # note the suspects here, act on them after the handler
            suspect = list(pending) if not ex._msl_build_failed else []
            ex._msl_build_failed = True
            retry = "no_msl"
        elif isinstance(e, RuntimeError) and "Resource limit" in str(e):
            # Metal buffer-handle exhaustion: if the failure came from a
            # compile attempt, drop to eager; else retry the same path
            # once (programs are pure — re-running is safe).
            if ex._can_compile:
                ex._can_compile = False
                ex._compiled = None
                retry = "eager"
            else:
                retry = "same"
        else:
            # MLX's compiler can reject certain traces (fused-kernel
            # argument-buffer exhaustion; unordered_map::at on graphs with
            # unused inputs surfaces as IndexError). Retry eagerly.
            if not ex._can_compile:
                raise
            ex._can_compile = False
            ex._compiled = None
            retry = "eager"
    if retry is not None:
        if retry == "no_msl":
            # NB outside the handler: the in-flight traceback pins the
            # failed attempt's arrays, and disable_msl drops the caches
            # holding them.
            from metaljax.ops import control
            if control._DEBUG:
                print(f"[metaljax] msl_scan kernel failed to build; "
                      f"dropping {len(suspect) or 'all'} generated "
                      f"kernel(s) for this program and retrying", flush=True)
            ex.disable_msl(suspect)
            retry = "same"
        # A retry only happens after something failed for want of resources:
        # force the collection the duty cycle may be holding back (dead
        # refcycles pin executables clear_cache cannot free).
        reclaim(force=True)
        runner = ex.interpreter if retry == "eager" else ex.runner()
        outs = list(runner(*args))
        if outs:
            mx.eval(*outs)
    return _finish(ex, args, outs)


def _finish(ex: MetalExecutable, args, outs) -> list[MetalBuffer]:
    """Turn a run's arrays into PJRT buffers. Engine-independent."""
    if os.environ.get("METALJAX_MEMDBG", "") == "1":
        print(f"[metaljax-mem] gc: {GC_STATS['collections']} collections "
              f"({GC_STATS['seconds']:.1f}s), "
              f"{GC_STATS['skipped']} deferred by the duty cycle", flush=True)
        print(f"[metaljax-mem] execute done: "
              f"active={mx.get_active_memory()}B "
              f"cache={mx.get_cache_memory()}B", flush=True)
    # XLA's no-alias contract: outputs may not share a buffer with a
    # non-donated input or with another output (jit identity must return
    # a FRESH buffer; jax asserts this via unsafe_buffer_pointer). The
    # interpreter naturally forwards pass-through values, so copy any
    # output that IS an input array — unless that input was donated, in
    # which case aliasing is exactly what donation licenses.
    donated = set(getattr(ex, "donated_params", ()) or ())
    # Statically-known forwards (main returns an argument): these alias
    # even through mx.compile, whose outputs are fresh objects sharing
    # the input's buffer — object identity cannot see that.
    for j, i in getattr(ex, "forwarded_outputs", ()) or ():
        if i not in donated and j < len(outs):
            outs[j] = mx.array(outs[j])
    # Dynamic pass: eager paths can also forward objects (and duplicate
    # an output); catch those by identity.
    in_ids = {id(a) for i, a in enumerate(args) if i not in donated}
    seen_out: set = set()
    fixed = []
    for arr in outs:
        if id(arr) in in_ids or id(arr) in seen_out:
            arr = mx.array(arr)
        seen_out.add(id(arr))
        fixed.append(arr)
    outs = fixed
    res = []
    for arr, type_enum, dims in zip(outs, ex.out_types, ex.out_dims):
        res.append(MetalBuffer(arr, type_enum, dims))
    return res


class _PyBuffer(ctypes.Structure):
    """CPython's Py_buffer (documented, stable-ABI layout)."""

    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),  # borrowed here: ctypes must not refcount
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_PyObject_GetBuffer = ctypes.pythonapi.PyObject_GetBuffer
_PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer),
                                ctypes.c_int]
_PyObject_GetBuffer.restype = ctypes.c_int
_PyBuffer_Release = ctypes.pythonapi.PyBuffer_Release
_PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]
_PyBuffer_Release.restype = None
_PyBUF_STRIDES = 0x0018  # PyBUF_ND | PyBUF_STRIDES


def _strided_base_address(obj) -> int:
    """Py_buffer.buf of a strided export: the address of element [0, ...]."""
    view = _PyBuffer()
    if _PyObject_GetBuffer(obj, ctypes.byref(view), _PyBUF_STRIDES) != 0:
        raise BufferError("object does not export a strided buffer")
    try:
        return int(view.buf or 0)
    finally:
        _PyBuffer_Release(ctypes.byref(view))


def buffer_pointer(buf: MetalBuffer) -> int:
    """Unified-memory address of the buffer's data (PJRT UnsafePointer).

    Apple silicon MTLBuffers in shared mode are CPU-addressable, and MLX
    exports them zero-copy through the buffer protocol (all dtypes,
    including bf16 — raw bytes). The buffer protocol CANNOT copy, unlike
    np.array(copy=False) (silently converts non-native dtypes) and
    __dlpack__ (stages a GPU-side copy), both of which returned transient
    addresses that made distinct buffers compare equal and equal buffers
    compare distinct. "Unsafe" per the PJRT contract: aliasing checks,
    not access.

    An mx.array need not be contiguous: broadcasts export stride 0 (every
    `jnp.ones` lands here), slices export a step. What we report for those
    is the address their leading element [0, 0, ...] occupies — the base
    of the buffer as this handle sees it, the same thing a contiguous array
    reports, and the only answer that keeps aliasing checks meaningful (two
    handles on one buffer must compare equal, distinct buffers must not).
    Neither numpy route reaches it: np.frombuffer refuses a non-C-contiguous
    memoryview outright, and np.asarray(mv) chokes on bfloat16 (MLX exports
    it as PEP-3118 'B' with itemsize 2, which numpy calls a mismatch); nor
    does memoryview indexing, which cannot take a sub-view of a >1-D buffer
    at all. Read Py_buffer.buf, which is that address by definition.
    """
    mx.eval(buf.data)
    mv = memoryview(buf.data)
    if mv.nbytes == 0:
        return 0
    if mv.c_contiguous:
        return np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
    return _strided_base_address(buf.data)


def device_kind() -> str:
    try:
        return mx.device_info()["device_name"]
    except Exception:
        return "Apple GPU"


def stablehlo_version() -> list[int]:
    from jaxlib.mlir.dialects import stablehlo
    return [int(x) for x in stablehlo.get_current_version().split(".")]
