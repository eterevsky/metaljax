"""Core StableHLO interpreter: walks MLIR and evaluates ops onto mx.arrays."""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes


class UnsupportedOpError(NotImplementedError):
    pass


# op name -> handler(interp, op: ir.Operation, ins: list[mx.array], env) -> list[mx.array] | mx.array
REGISTRY: dict[str, Callable] = {}

_TERMINATORS = ("func.return", "stablehlo.return")


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

    def __init__(self, module: bytes | str | ir.Module, context: ir.Context | None = None):
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

    @property
    def in_avals(self) -> list[tuple[tuple[int, ...], np.dtype]]:
        out = []
        for a in self._main_block().arguments:
            t = ir.RankedTensorType(a.type)
            out.append((tuple(t.shape), dtypes.np_dtype_for(dtypes.mx_dtype_for(t.element_type))))
        return out

    @property
    def out_avals(self) -> list[tuple[tuple[int, ...], np.dtype]]:
        ftype = ir.FunctionType(ir.TypeAttr(self.main.attributes["function_type"]).value)
        out = []
        for t in ftype.results:
            rt = ir.RankedTensorType(t)
            out.append((tuple(rt.shape), dtypes.np_dtype_for(dtypes.mx_dtype_for(rt.element_type))))
        return out

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
