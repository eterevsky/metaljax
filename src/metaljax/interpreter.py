"""Core StableHLO interpreter: walks MLIR and evaluates ops onto mx.arrays."""

from __future__ import annotations

import os
from typing import Callable

import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes

# When enabled, pure programs/loop bodies are traced through mx.compile so
# repeat executions replay a fused Metal graph instead of re-dispatching
# op by op from Python. Disable for debugging.
COMPILE_ENABLED = os.environ.get("METALJAX_COMPILE", "1") != "0"


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


# op name -> handler(interp, op: ir.Operation, ins: list[mx.array], env) -> list[mx.array] | mx.array
REGISTRY: dict[str, Callable] = {}

_TERMINATORS = ("func.return", "stablehlo.return")

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
        self._fft_cache: dict = {}     # block -> contains a stablehlo.fft?
        self._in_trace = False  # True while mx.compile is tracing our code
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

    # --- execution ---

    def __call__(self, *args: mx.array) -> list[mx.array]:
        with self.context:
            return self.run_func(self.main, list(args))

    def run_func(self, func_op: ir.Operation, args: list[mx.array]) -> list[mx.array]:
        return self.run_block(func_op.regions[0].blocks[0], args)

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

        for op in block.operations:
            o = op.operation
            name = o.name
            if name in _TERMINATORS:
                return [env[v] for v in o.operands]
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
        raise RuntimeError("block ended without a terminator")
