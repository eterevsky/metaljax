"""Control flow, calls, composites, and pass-through ops."""

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir, dtypes
from metaljax.interpreter import (
    COMPILE_ENABLED,
    UnsupportedOpError,
    free_values,
    register,
)


def _callee_name(op, attr_name):
    return ir.FlatSymbolRefAttr(op.attributes[attr_name]).value


@register("func.call")
def _call(interp, op, ins, env):
    return interp.run_func(interp.funcs[_callee_name(op, "callee")], ins)


@register("stablehlo.composite")
def _composite(interp, op, ins, env):
    # Composites (e.g. jax.nn ops) carry a reference to a decomposition func.
    return interp.run_func(interp.funcs[_callee_name(op, "decomposition")], ins)


def _splat_int(op) -> int | None:
    """Value of a scalar integer constant op, else None."""
    if op.name != "stablehlo.constant":
        return None
    t = ir.RankedTensorType(op.results[0].type)
    if list(t.shape):
        return None
    try:
        return int(_ir.dense_to_np(op.attributes["value"], t))
    except Exception:
        return None


def _analyze_counted(interp, op):
    """Detect the canonical counted loop JAX emits for scan/fori_loop.

    cond:  %n = constant N            (or N captured from outside)
           %p = compare LT, %arg_k, %n
           return %p
    body:  ... return ..., add(%arg_k, 1), ...

    Returns (k, bound) where bound is an int or the ir.Value holding N,
    or None if the loop doesn't match.
    """
    cond_block = op.regions[0].blocks[0]
    cached = interp._counted_cache.get(cond_block, "miss")
    if cached != "miss":
        return cached

    result = None
    try:
        cond_ops = [o.operation for o in cond_block.operations]
        ret = cond_ops[-1]
        cmp_val = ret.operands[0]
        if isinstance(cmp_val, ir.OpResult):
            cmp = cmp_val.owner.operation
            if (
                cmp.name == "stablehlo.compare"
                and "LT" in str(cmp.attributes["comparison_direction"])
            ):
                lhs, rhs = cmp.operands
                args = list(cond_block.arguments)
                if isinstance(lhs, ir.BlockArgument) and lhs in args:
                    k = args.index(lhs)
                    bound = None
                    if isinstance(rhs, ir.OpResult):
                        n = _splat_int(rhs.owner.operation)
                        if n is not None:
                            bound = n
                    if bound is None and not isinstance(rhs, ir.OpResult):
                        bound = rhs  # captured from an enclosing scope
                    if bound is not None:
                        # body must return arg_k + 1 at position k
                        body_block = op.regions[1].blocks[0]
                        body_ret = list(body_block.operations)[-1].operation
                        inc = body_ret.operands[k]
                        ok = False
                        if isinstance(inc, ir.OpResult):
                            add = inc.owner.operation
                            if add.name == "stablehlo.add":
                                a, b = add.operands
                                barg = list(body_block.arguments)[k]
                                for x, y in ((a, b), (b, a)):
                                    if x == barg and isinstance(y, ir.OpResult):
                                        if _splat_int(y.owner.operation) == 1:
                                            ok = True
                        if ok:
                            result = (k, bound)
    except Exception:
        result = None
    interp._counted_cache[cond_block] = result
    return result


def _body_fn(interp, body_block):
    """Cached (compiled) executor for a while body: fn(*vals, *captures)."""
    entry = interp._body_cache.get(body_block)
    if entry is None:
        free = free_values(body_block)
        nvals = len(list(body_block.arguments))

        def raw(*flat):
            vals = list(flat[:nvals])
            captures = dict(zip(free, flat[nvals:]))
            return tuple(interp.run_block(body_block, vals, captures))

        fn = raw
        if COMPILE_ENABLED and interp.block_is_pure(body_block):
            fn = mx.compile(raw)
        entry = (fn, free)
        interp._body_cache[body_block] = entry
    return entry


@register("stablehlo.while")
def _while(interp, op, ins, env):
    cond_block = op.regions[0].blocks[0]
    body_block = op.regions[1].blocks[0]

    counted = _analyze_counted(interp, op)
    if counted is not None:
        k, bound = counted
        n = bound if isinstance(bound, int) else int(env[bound].item())
        start = int(ins[k].item())
        trip = max(n - start, 0)
        fn, free = _body_fn(interp, body_block)
        captures = [env[v] for v in free]
        vals = list(ins)
        for _ in range(trip):
            vals = list(fn(*vals, *captures))
        return vals

    vals = list(ins)
    while True:
        (pred,) = interp.run_block(cond_block, vals, env)
        if not bool(pred.item()):
            return vals
        vals = interp.run_block(body_block, vals, env)


@register("stablehlo.if")
def _if(interp, op, ins, env):
    (pred,) = ins
    region = op.regions[0] if bool(pred.item()) else op.regions[1]
    return interp.run_block(region.blocks[0], [], env)


@register("stablehlo.case")
def _case(interp, op, ins, env):
    (index,) = ins
    i = int(index.item())
    i = min(max(i, 0), len(op.regions) - 1)
    return interp.run_block(op.regions[i].blocks[0], [], env)


@register("stablehlo.optimization_barrier")
def _optimization_barrier(interp, op, ins, env):
    return list(ins)


# Single-device: sharding ops are identities.
@register("sdy.sharding_constraint", "sdy.reshard")
def _sharding_identity(interp, op, ins, env):
    return list(ins)


@register("stablehlo.custom_call")
def _custom_call(interp, op, ins, env):
    target = _ir.str_attr(op, "call_target_name")
    if target in ("Sharding", "annotate_device_placement"):
        return list(ins)
    if target == "shape_assertion":
        return []
    raise UnsupportedOpError(f"custom_call target {target!r} not implemented")
