"""Core StableHLO interpreter: walks MLIR and evaluates ops onto mx.arrays."""

from __future__ import annotations

import gc
import os
from typing import Callable

import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes, qmm as _qmm, sdpa as _sdpa

# When enabled, pure programs/loop bodies are traced through mx.compile so
# repeat executions replay a fused Metal graph instead of re-dispatching
# op by op from Python. Disable for debugging.
COMPILE_ENABLED = os.environ.get("METALJAX_COMPILE", "1") != "0"

# --- eager-phase memory safety net -----------------------------------------
# An eagerly interpreted block used to keep EVERY intermediate it produced in
# `env` until the block returned. MLX frees an intermediate as soon as nothing
# references it -- measured on a 20-op chain over 128 MB arrays: peak 0.25 GB
# with only the result referenced, 2.62 GB with every step held. That
# retention, not laziness, is what multiplied an eagerly interpreted
# program's footprint by the length of its op chain. Two mechanisms bound it:
#
#   * liveness pruning -- a value leaves `env` after its last use in the block
#     (METALJAX_ENV_PRUNE=0 restores the old retain-everything behaviour);
#   * a byte-denominated flush -- once this much estimated result data has
#     been produced with no sync point, settle what is still live so the
#     pending graph, and the Metal buffers it pins, stay bounded.
#
# The two only work TOGETHER: flushing without pruning materializes every
# retained intermediate at once instead of letting MLX free them as it walks
# the graph (measured on the 0.6B maxtext load: 75 GB and still climbing,
# watchdog-killed, versus 17 GB with neither and 3.5 GB with both).
#
# NB the deliberately-omitted third mechanism: denominating a LOOP's flush
# cadence in bytes as well. It measured no better than the shipped op-count
# cadence here (6.63 GB either way) and it changes where existing sync points
# fall, which reshuffles the MLX command-buffer lottery -- it produced a
# stable but WRONG token stream on qwen3-0.6B decode. See
# notes/mlx-command-buffer-split.md; loop cadences stay as they are until MLX
# is fixed upstream.
#
# The default is deliberately generous: it must never fire on workloads that
# are already fast (a texmo train step moves kilobytes to a few megabytes per
# block). METALJAX_EAGER_FLUSH_MB=0 disables the flush entirely.
FLUSH_MB = int(os.environ.get("METALJAX_EAGER_FLUSH_MB", "1024"))
FLUSH_BYTES = max(FLUSH_MB, 0) * 2**20
_ENV_PRUNE = os.environ.get("METALJAX_ENV_PRUNE", "1") != "0"
# Blocking eval costs a full command-buffer roundtrip (~190us); async_eval
# only submits. Checkpoints block by default so the budget means what it
# says: with N async checkpoints in flight the peak is N x the budget
# (measured 4.0 GB at budget 1024 MB and N=4, 1.1 GB at N=1). The blocking
# cost is one roundtrip per budget's worth of data produced -- unmeasurable
# next to producing it, and the flush never fires on small workloads anyway.
# METALJAX_EAGER_FLUSH_SYNC=N blocks at every Nth checkpoint instead.
_FLUSH_SYNC_EVERY = int(os.environ.get("METALJAX_EAGER_FLUSH_SYNC", "1"))
_DEBUG = os.environ.get("METALJAX_DEBUG", "") == "1"

# Observability, for tests and for the "did this fire?" question.
FLUSH_STATS = {"flushes": 0, "bytes": 0}


def value_bytes(v: ir.Value, cache: dict | None = None) -> int:
    """Estimated device bytes of an SSA value.

    Static, straight from the IR: element count x dtype size. mx.compile
    fuses elementwise chains, so summing this over a block overestimates what
    is ever simultaneously live (~2x, measured on the qwen3 prefill repro).
    That is fine for a flush cadence -- it only makes the safety net fire
    earlier -- and it is exactly why such an estimate must never be used as a
    correctness gate (notes/mlx-command-buffer-split.md addendum).

    `cache` keys results by ir.Type. MLIR uniques types within a context, so
    one dict per block collapses thousands of these to a handful of real
    computations -- and each one is several Python-level MLIR calls, which
    measured as ~200 ms of extra first-call latency on a texmo chunk before
    it was memoized.
    """
    t = v.type
    if cache is not None:
        got = cache.get(t)
        if got is not None:
            return got
    nb = 0
    if not _ir.is_token(t):
        try:
            rt = ir.RankedTensorType(t)
            n = 1
            for d in rt.shape:
                if d < 0:
                    n = 0  # dynamic dim (shape-poly export): don't guess
                    break
                n *= d
            nb = n * dtypes.np_dtype_for_mlir(rt.element_type).itemsize
        except Exception:
            nb = 0
    if cache is not None:
        cache[t] = nb
    return nb


def flush_eval(arrays, hard: bool):
    """Settle `arrays`, recovering once from Metal buffer exhaustion.

    Same contract as ops.control._loop_flush: programs are pure, so clearing
    the cache and retrying is safe. gc first -- dead refcycles pin buffers
    that clear_cache cannot free (CLAUDE.md item 19).
    """
    try:
        if hard:
            mx.eval(*arrays)
        else:
            mx.async_eval(*arrays)
    except RuntimeError as e:
        if "Resource limit" not in str(e):
            raise
        if _DEBUG:
            print("[metaljax] Metal buffer limit hit at eager flush; "
                  "clearing cache and retrying", flush=True)
        gc.collect()
        mx.clear_cache()
        mx.eval(*arrays)


class UnsupportedOpError(NotImplementedError):
    pass


def free_values(block: ir.Block) -> list[ir.Value]:
    """SSA values used inside `block` but defined outside it (captures)."""
    defined: set = set()
    free: dict = {}

    def walk(blk: ir.Block):
        for a in blk.arguments:
            defined.add(a)
        for op in blk.operations:
            o = op.operation
            for v in o.operands:
                if v not in defined and v not in free:
                    free[v] = None
            for region in o.regions:
                for b in region.blocks:
                    walk(b)
            for r in o.results:
                defined.add(r)

    walk(block)
    return list(free)


def _op_captures(o: ir.Operation) -> list[ir.Value]:
    """Values an op's nested regions read from enclosing scopes. Such a value
    is live for as long as the op runs, even though it is not an operand."""
    caps: dict = {}
    for region in o.regions:
        for b in region.blocks:
            for v in free_values(b):
                caps[v] = None
    return list(caps)


# op name -> handler(interp, op: ir.Operation, ins: list[mx.array], env) -> list[mx.array] | mx.array
REGISTRY: dict[str, Callable] = {}

_TERMINATORS = ("func.return", "stablehlo.return")

# Marks an op a recognizer absorbed: it is skipped, not executed and not
# emitted. Roots carry an (emit, match) pair instead. See _rewrite_plan.
_SKIP = object()

# Ops through which f64 values may flow without any arithmetic: since f64
# device storage is f32, these are bit-identical to CPU as long as every
# consumer eventually converts to <= f32. Anything not listed that touches
# an f64 tensor *computes* in f64 and is rejected in strict mode.
_F64_DATA_MOVEMENT = {
    "func.func", "func.call", "func.return",
    "stablehlo.return", "stablehlo.constant", "stablehlo.convert",
    "stablehlo.reshape", "stablehlo.broadcast_in_dim", "stablehlo.transpose",
    "stablehlo.slice", "stablehlo.dynamic_slice",
    "stablehlo.dynamic_update_slice", "stablehlo.concatenate",
    "stablehlo.reverse", "stablehlo.gather", "stablehlo.scatter",
    "stablehlo.select", "stablehlo.optimization_barrier", "stablehlo.iota",
    "stablehlo.pad", "stablehlo.while", "stablehlo.if", "stablehlo.case",
    "stablehlo.composite", "stablehlo.custom_call",
    "sdy.sharding_constraint", "sdy.reshard",
}


# The double-width element types: neither exists on Metal, and both follow
# the same policy (pass-through stored narrowed, compute rejected).
_F64_ELEMENT_TYPES = ("f64", "complex<f64>")


def _has_f64(types) -> bool:
    for t in types:
        try:
            if str(ir.RankedTensorType(t).element_type) in _F64_ELEMENT_TYPES:
                return True
        except Exception:
            pass
    return False


def _check_no_f64_compute(module_op: ir.Operation):
    """Raise if any non-data-movement op touches an f64/complex128 tensor."""

    def visit(op: ir.Operation):
        name = op.name
        if name not in _F64_DATA_MOVEMENT and name != "builtin.module":
            if _has_f64(r.type for r in op.results) or _has_f64(
                o.type for o in op.operands
            ):
                raise dtypes.UnsupportedDtypeError(
                    f"program computes in float64/complex128, unsupported on "
                    f"Metal (set METALJAX_F64=downcast to compute in "
                    f"float32/complex64):\n"
                    f"  {str(op).splitlines()[0]}"
                )
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    visit(inner.operation)

    visit(module_op)


def register(*names: str):
    def deco(fn):
        for n in names:
            REGISTRY[n] = fn
        return fn
    return deco


class Interpreter:
    """Interprets a StableHLO module's @main function on MLX arrays.

    Accepts MLIR bytecode bytes, module text, or an ir.Module.
    """

    # Ops whose handlers synchronize with the host (.item()) and therefore
    # cannot be traced through mx.compile.
    _IMPURE_OPS = ("stablehlo.while", "stablehlo.if", "stablehlo.case",
                   # computed on the host via numpy (see ops.lapack)
                   "stablehlo.triangular_solve", "stablehlo.cholesky")

    # Set by ops.control: hook(interp, while_op) -> bool, True when the loop
    # has a small static trip count and can be unrolled inside a trace.
    while_traceable_hook = None
    # Set by ops.lapack: hook(op) -> bool, True when a custom_call target
    # computes on the host (numpy/scipy) and must stay out of traces.
    custom_call_host_hook = None

    def __init__(self, module: bytes | str | ir.Module, context: ir.Context | None = None):
        # Caches keyed by ir.Block (pointer-stable identity across traversals).
        self._body_cache: dict = {}    # while-body block -> (fn, free_values, nvals)
        self._counted_cache: dict = {}  # while cond block -> counted-loop info | None
        self._pure_cache: dict = {}    # block -> bool
        self._traceable_cache: dict = {}  # while body block -> bool
        self._cost_cache: dict = {}    # block -> approx op count when traced
        self._no_chunk: set = set()    # body blocks where chunking failed
        self._no_body_compile: set = set()  # bodies whose compiled fn failed
        self._msl_cache: dict = {}     # (body block, trip, start) -> Plan | None
        # Set by engine.execute after a generated kernel failed to build:
        # this program then plans no more msl_scan kernels (loops take the
        # compiled-graph path, which is correct).
        self._no_msl = False
        # msl_scan plans traced into an mx.compile graph and never yet
        # evaluated; engine.execute settles those executes synchronously so
        # a Metal build error raises here instead of in an async worker.
        self._msl_pending: list = []
        self._fft_cache: dict = {}     # block -> contains a stablehlo.fft?
        self._live_cache: dict = {}    # block -> (drop-after lists, result bytes)
        self._in_trace = False  # True while mx.compile is tracing our code
        # metaljax.qmm.State: recognized quantized matmuls (None until the
        # engine analyzes the program; stays empty when nothing matches).
        self._qmm = None
        # metaljax.sdpa.State: recognized fused attentions. Analyzed lazily
        # on first use -- unlike qmm this rewrite needs nothing from the
        # concrete buffers, so there is no packing prologue to hang it off.
        self._sdpa = None
        # (gate, merged rewrite table, per-block plans); see _rewrite_plan.
        self._rewrite_cache = None
        if isinstance(module, ir.Module):
            if context is None:
                raise ValueError("pass the ir.Context that owns the module")
            self.context = context
            self.module = module
        else:
            self.context = _ir.make_context()
            with self.context:
                self.module = ir.Module.parse(module)
        self.funcs: dict[str, ir.Operation] = {}
        public = []
        for op in self.module.body.operations:
            o = op.operation
            if o.name == "func.func":
                name = _ir.str_attr(o, "sym_name")
                self.funcs[name] = o
                try:
                    vis = _ir.str_attr(o, "sym_visibility")
                except KeyError:
                    vis = "public"
                if vis == "public":
                    public.append(o)
        if not dtypes.F64_DOWNCAST:
            with self.context:
                _check_no_f64_compute(self.module.operation)
        if "main" in self.funcs:
            self.main = self.funcs["main"]
        elif len(public) == 1:
            self.main = public[0]
        elif len(self.funcs) == 1:
            self.main = next(iter(self.funcs.values()))
        else:
            raise ValueError(
                f"cannot determine entry function among {sorted(self.funcs)}"
            )

    # --- shape/dtype metadata (used by tests and the PJRT layer) ---

    def _main_block(self) -> ir.Block:
        return self.main.regions[0].blocks[0]

    @staticmethod
    def _aval_for(t: ir.Type) -> tuple[tuple[int, ...], np.dtype]:
        """Shape/dtype JAX expects for an MLIR type. Ordered-effect tokens
        are not tensors — they cross the PJRT boundary as bool[0]."""
        if _ir.is_token(t):
            return (dtypes.TOKEN_SHAPE, dtypes.TOKEN_NP_DTYPE)
        rt = ir.RankedTensorType(t)
        return (tuple(rt.shape), dtypes.np_dtype_for_mlir(rt.element_type))

    @property
    def in_avals(self) -> list[tuple[tuple[int, ...], np.dtype]]:
        return [self._aval_for(a.type) for a in self._main_block().arguments]

    @property
    def out_avals(self) -> list[tuple[tuple[int, ...], np.dtype]]:
        ftype = ir.FunctionType(ir.TypeAttr(self.main.attributes["function_type"]).value)
        return [self._aval_for(t) for t in ftype.results]

    @property
    def forwarded_outputs(self) -> list[tuple[int, int]]:
        """(output_index, argument_index) pairs where main returns one of
        its arguments directly. XLA's no-alias contract requires such
        outputs to be COPIES unless the argument was donated; the engine
        inserts the copy. (Buffer sharing through views of arguments is
        not detected — nothing asserts on it today.)"""
        blk = self._main_block()
        args = list(blk.arguments)
        ret = list(blk.operations)[-1].operation
        out = []
        for j, v in enumerate(ret.operands):
            if isinstance(v, ir.BlockArgument) and v in args:
                out.append((j, args.index(v)))
        return out

    @property
    def donated_args(self) -> list[int]:
        """Main-function argument indices the caller donated: jax marks
        them with tf.aliasing_output (aliased to an output) or
        jax.buffer_donor (donated without a specific alias)."""
        out = []
        try:
            arg_attrs = self.main.attributes["arg_attrs"]
        except KeyError:
            return out
        for i, a in enumerate(ir.ArrayAttr(arg_attrs)):
            d = ir.DictAttr(a)
            if "tf.aliasing_output" in d or "jax.buffer_donor" in d:
                out.append(i)
        return out

    @property
    def in_tokens(self) -> list[bool]:
        return [_ir.is_token(a.type) for a in self._main_block().arguments]

    @property
    def out_tokens(self) -> list[bool]:
        ftype = ir.FunctionType(ir.TypeAttr(self.main.attributes["function_type"]).value)
        return [_ir.is_token(t) for t in ftype.results]

    # --- purity / capture analysis (used for mx.compile) ---

    def block_is_pure(self, block: ir.Block) -> bool:
        """True if executing the block never synchronizes with the host."""
        cached = self._pure_cache.get(block)
        if cached is not None:
            return cached
        # Token-carrying code sequences host-visible effects: keep it off
        # every tracing path (whole-main compile, counted-loop replay,
        # compiled while bodies). Such programs sync with the host anyway,
        # so nothing is lost.
        if any(_ir.is_token(a.type) for a in block.arguments):
            self._pure_cache[block] = False
            return False
        self._pure_cache[block] = True  # optimistic; no recursion in jax IR
        pure = True
        for op in block.operations:
            o = op.operation
            name = o.name
            if any(_ir.is_token(r.type) for r in o.results):
                pure = False
                break
            if name in self._IMPURE_OPS:
                hook = type(self).while_traceable_hook
                if (
                    name == "stablehlo.while"
                    and hook is not None
                    and hook(self, o)
                ):
                    continue  # statically-counted small loop: unrollable
                pure = False
                break
            if name == "stablehlo.custom_call":
                hook = type(self).custom_call_host_hook
                if hook is not None and hook(o):
                    pure = False
                    break
            if name in ("func.call", "stablehlo.composite"):
                attr = "callee" if name == "func.call" else "decomposition"
                callee = ir.FlatSymbolRefAttr(o.attributes[attr]).value
                fn = self.funcs.get(callee)
                if fn is not None and not self.block_is_pure(
                    fn.regions[0].blocks[0]
                ):
                    pure = False
                    break
            stop = False
            for region in o.regions:
                for b in region.blocks:
                    if not self.block_is_pure(b):
                        pure = False
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        self._pure_cache[block] = pure
        return pure

    @property
    def main_pure(self) -> bool:
        with self.context:
            return self.block_is_pure(self._main_block())

    def block_has_fft(self, block: ir.Block) -> bool:
        """True if the block, or anything it calls, holds a stablehlo.fft.

        MLX's FFT kernels can start before a pending async evaluation of
        their input has landed, so such programs need their inputs
        materialized first — see engine.execute and ops.elementwise._fft.
        """
        cached = self._fft_cache.get(block)
        if cached is not None:
            return cached
        self._fft_cache[block] = False  # break cycles defensively
        found = False
        for op in block.operations:
            o = op.operation
            if o.name == "stablehlo.fft":
                found = True
                break
            if o.name in ("func.call", "stablehlo.composite"):
                attr = "callee" if o.name == "func.call" else "decomposition"
                callee = ir.FlatSymbolRefAttr(o.attributes[attr]).value
                fn = self.funcs.get(callee)
                if fn is not None and self.block_has_fft(
                    fn.regions[0].blocks[0]
                ):
                    found = True
                    break
            if any(self.block_has_fft(b)
                   for region in o.regions for b in region.blocks):
                found = True
                break
        self._fft_cache[block] = found
        return found

    @property
    def main_has_fft(self) -> bool:
        with self.context:
            return self.block_has_fft(self._main_block())

    # --- eager execution plan (liveness + result bytes) ---

    def eager_plan(self, block: ir.Block):
        """(drop_after, out_bytes) for op-by-op execution of `block`.

        drop_after[i] lists the values whose last use is operation i, so the
        eager loop can let go of them; out_bytes[i] is the estimated device
        data operation i produces. Cached per block (ir.Block identity is
        pointer-stable across traversals).

        A value's uses are its consumers' operands PLUS the captures of any
        nested region, which is live for the whole of the op owning it. A
        result nothing reads is dropped immediately. Values inherited from an
        enclosing block may be dropped too: `run_block` works on a copy of
        the parent's environment, so only this block's reference goes away.
        """
        plan = self._live_cache.get(block)
        if plan is not None:
            return plan
        ops = [o.operation for o in block.operations]
        last: dict = {}
        out_bytes: list[int] = []
        tcache: dict = {}
        for i, o in enumerate(ops):
            for v in o.operands:
                last[v] = i
            if o.regions:
                for v in _op_captures(o):
                    last[v] = i
            out_bytes.append(sum(value_bytes(r, tcache) for r in o.results))
        for i, o in enumerate(ops):
            for r in o.results:
                if r not in last:
                    last[r] = i  # never read: let go as soon as it exists
        drops: list[tuple] = [()] * len(ops)
        by_index: dict = {}
        for v, i in last.items():
            by_index.setdefault(i, []).append(v)
        for i, vs in by_index.items():
            drops[i] = tuple(vs)
        plan = (drops, out_bytes)
        self._live_cache[block] = plan
        return plan

    def _eager_flush(self, env: dict):
        """Byte-denominated sync point: settle everything still live.

        Only what `env` still holds needs to survive -- pruned intermediates
        are already unreferenced, so MLX frees them as the evaluation walks
        the graph instead of materializing the whole chain at once.
        """
        arrays = [a for a in env.values() if isinstance(a, mx.array)]
        if not arrays:
            return
        FLUSH_STATS["flushes"] += 1
        hard = (_FLUSH_SYNC_EVERY > 0
                and FLUSH_STATS["flushes"] % _FLUSH_SYNC_EVERY == 0)
        if _DEBUG:
            print(f"[metaljax] eager flush #{FLUSH_STATS['flushes']}: "
                  f"{len(arrays)} live values, hard={hard}", flush=True)
        flush_eval(arrays, hard)

    # --- execution ---

    def __call__(self, *args: mx.array) -> list[mx.array]:
        with self.context:
            return self.run_func(self.main, list(args))

    def run_func(self, func_op: ir.Operation, args: list[mx.array]) -> list[mx.array]:
        return self.run_block(func_op.regions[0].blocks[0], args)

    def _rewrite_plan(self, block: ir.Block):
        """Ops of `block` a recognizer rewrites or absorbs: `((index, entry),)`.

        Both recognizers answer the same per-op question, and they are
        disjoint by construction (sdpa drops any candidate overlapping a
        quantized matmul), so they merge into ONE table. That table is then
        resolved against a block once and replayed BY POSITION, which keeps
        the hot loop from touching `op.results` for the ~99% of ops that are
        neither a root nor absorbed: a decode program is thousands of ops per
        layer and far too big to compile, so `run_block` is its execute path.
        """
        qst = self._qmm
        # `qst.moe` (expert-gather rewrites, metaljax.moe) needs no packed
        # weights of its own -- a float MoE has an empty `values`.
        if qst is not None and not (qst.active and (qst.values or qst.moe)):
            # qmm needs its prologue to have packed (the bare Interpreter has
            # none); without that, run every op literally.
            qst = None
        sst = self._sdpa
        if sst is None:
            sst = _sdpa.analyze(self)
        if not sst.active:
            sst = None
        # Compared by IDENTITY, not equality: `State.rebuild` replaces `skip`
        # and `roots` wholesale (qmm disables a match whose pack failed), so a
        # new frozenset is exactly the signal that the plan is stale.
        gate = (qst, None if qst is None else qst.skip,
                sst, None if sst is None else sst.skip)
        cache = self._rewrite_cache
        if cache is None or not all(a is b for a, b in zip(cache[0], gate)):
            table: dict = {}
            for st, fuse in ((qst, _qmm.emit), (sst, _sdpa.emit)):
                if st is None:
                    continue
                for k in st.skip:
                    table.setdefault(k, _SKIP)
                for k, match in st.roots.items():
                    table.setdefault(k, (fuse, match))
            cache = (gate, table, {})
            self._rewrite_cache = cache
        table, plans = cache[1], cache[2]
        if not table:
            return ()
        plan = plans.get(block)
        if plan is None:
            found = []
            for i, op in enumerate(block.operations):
                res = op.operation.results
                if len(res):
                    entry = table.get(res[0])
                    if entry is not None:
                        found.append((i, entry))
            plan = plans[block] = tuple(found)
        return plan

    def run_block(
        self,
        block: ir.Block,
        args: list[mx.array],
        parent_env: dict | None = None,
    ) -> list[mx.array]:
        # StableHLO regions may capture values from enclosing scopes, so seed
        # the environment with the parent's bindings.
        env: dict = dict(parent_env) if parent_env else {}
        block_args = list(block.arguments)
        if len(block_args) != len(args):
            raise ValueError(f"block expects {len(block_args)} args, got {len(args)}")
        for ba, v in zip(block_args, args):
            env[ba] = v

        # Recognized rewrites, resolved against this block ONCE (see
        # _rewrite_plan): absorbed ops are not executed at all and each root
        # becomes one fused op.
        plan = self._rewrite_plan(block)
        pi, nxt = 0, (plan[0][0] if plan else -1)

        # Pruning/flushing are for eagerly interpreted programs only. Inside
        # an mx.compile trace the values are tracers -- there is nothing to
        # evaluate and nothing to free, MLX manages the traced tape itself --
        # so a program that compiles never even builds the eager plan, and
        # pays nothing for any of this.
        eager = not self._in_trace
        drops = out_bytes = None
        pruning = flushing = False
        if eager:
            drops, out_bytes = self.eager_plan(block)
            flushing = FLUSH_BYTES > 0
            # Liveness is computed from the literal IR, but recognizer
            # rewrites (qmm/sdpa) read the operands of ops they SKIP when
            # they emit the fused op further down the block -- a static
            # last use can fall on a skipped op. Retain everything on any
            # block with an active rewrite plan.
            pruning = _ENV_PRUNE and not plan
        acc = 0

        for i, op in enumerate(block.operations):
            o = op.operation
            name = o.name
            if name in _TERMINATORS:
                return [env[v] for v in o.operands]
            if i == nxt:
                entry = plan[pi][1]
                pi += 1
                nxt = plan[pi][0] if pi < len(plan) else -1
                if entry is not _SKIP:
                    fuse, match = entry
                    for r, v in zip(o.results, fuse(self, match, env)):
                        env[r] = v
                continue
            handler = REGISTRY.get(name)
            if handler is None:
                raise UnsupportedOpError(
                    f"op '{name}' not implemented by metaljax.\n  {str(o).splitlines()[0]}"
                )
            ins = [env[v] for v in o.operands]
            out = handler(self, o, ins, env)
            results = list(o.results)
            if isinstance(out, mx.array):
                out = [out]
            elif out is None:
                out = []
            if len(out) != len(results):
                raise RuntimeError(
                    f"handler for '{name}' returned {len(out)} values, op has {len(results)} results"
                )
            for r, v in zip(results, out):
                env[r] = v
            if pruning:
                for v in drops[i]:
                    env.pop(v, None)
            if flushing:
                acc += out_bytes[i]
                if acc >= FLUSH_BYTES:
                    FLUSH_STATS["bytes"] += acc
                    acc = 0
                    self._eager_flush(env)
        raise RuntimeError("block ended without a terminator")
