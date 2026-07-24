"""Control flow, calls, composites, and pass-through ops."""

from jaxlib.mlir import ir

from metaljax import _ir
from metaljax.interpreter import register, UnsupportedOpError


def _callee_name(op, attr_name):
    return ir.FlatSymbolRefAttr(op.attributes[attr_name]).value


@register("func.call")
def _call(interp, op, ins, env):
    return interp.run_func(interp.funcs[_callee_name(op, "callee")], ins)


@register("stablehlo.composite")
def _composite(interp, op, ins, env):
    # Composites (e.g. jax.nn ops) carry a reference to a decomposition func.
    return interp.run_func(interp.funcs[_callee_name(op, "decomposition")], ins)


@register("stablehlo.while")
def _while(interp, op, ins, env):
    cond_block = op.regions[0].blocks[0]
    body_block = op.regions[1].blocks[0]
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
