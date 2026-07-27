"""stablehlo.sort — the comparator patterns JAX emits.

JAX lowers every sort (jnp.sort/argsort/partition/median/percentile/
unique/searchsorted-with-sorter, ...) to stablehlo.sort whose comparator
region computes a KEY from one operand pair and compares it with a strict
LT/GT — for floats the key runs through a canonicalization chain
(-0 -> +0, NaN -> canonical qNaN) followed by a TOTALORDER compare.

We recognize that shape generically: trace the final compare's two sides
to their block arguments, check both sides compute the *same* function
(structural DAG equality with lhs/rhs arg roles swapped), evaluate the
chain once on the full operand array (every op in it is elementwise, so
scalar block code runs unchanged on arrays), and stable-argsort a
total-order integer key. All operands are then gathered with the sorted
indices; mx.argsort is stable, satisfying is_stable = true.
"""

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir, dtypes
from metaljax.interpreter import REGISTRY, register, UnsupportedOpError

def _arg_deps(val, args, local):
    """Transitive comparator-block-argument dependencies of an SSA value.
    Values defined outside the comparator block (hoisted constants) are
    opaque leaves."""
    seen, deps, stack = set(), set(), [val]
    while stack:
        v = stack.pop()
        if isinstance(v, ir.BlockArgument):
            if v in args:
                deps.add(args.index(v))
            continue
        if v not in local:
            continue
        o = v.owner.operation
        if id(o) in seen:
            continue
        seen.add(id(o))
        stack.extend(o.operands)
    return deps


def _serialize(val, arg_map, local):
    """Canonical form of the value's def DAG with block args renamed."""
    if isinstance(val, ir.BlockArgument):
        return arg_map.get(val, "OTHER_ARG")
    if val not in local:
        return ("EXT", str(val))
    o = val.owner.operation
    attrs = sorted((n, str(o.attributes[n])) for n in o.attributes)
    return (o.name, tuple(attrs),
            tuple(_serialize(w, arg_map, local) for w in o.operands))


@register("stablehlo.sort")
def _sort(interp, op, ins, env):
    dim = _ir.int_attr(op, "dimension")
    block = op.regions[0].blocks[0]
    body = [o.operation for o in block.operations]
    ret = body[-1]
    if ret.name != "stablehlo.return" or len(ret.operands) != 1:
        raise UnsupportedOpError("sort: comparator must return one value")
    cmp_val = ret.operands[0]
    if isinstance(cmp_val, ir.BlockArgument):
        raise UnsupportedOpError("sort: comparator returns an argument")
    cmp = cmp_val.owner.operation
    if cmp.name != "stablehlo.compare":
        raise UnsupportedOpError(
            f"sort: comparator ends in {cmp.name}, not compare")
    direction = str(cmp.attributes["comparison_direction"])
    if "LT" in direction:
        descending = False
    elif "GT" in direction:
        descending = True
    else:
        raise UnsupportedOpError(f"sort: non-strict compare {direction}")

    args = list(block.arguments)
    local = {r for o in body for r in o.results}
    lhs, rhs = cmp.operands
    ldeps = _arg_deps(lhs, args, local)
    rdeps = _arg_deps(rhs, args, local)
    if len(ldeps) != 1 or len(rdeps) != 1:
        raise UnsupportedOpError(
            f"sort: comparator mixes operands (deps {ldeps} vs {rdeps})")
    li, ri = ldeps.pop(), rdeps.pop()
    if ri != li + 1 or li % 2 != 0:
        raise UnsupportedOpError(
            f"sort: comparator args ({li}, {ri}) are not an (lhs, rhs) pair")
    k = li // 2
    # Both sides must compute the same key function of their argument.
    if (_serialize(lhs, {args[li]: "KEY"}, local)
            != _serialize(rhs, {args[ri]: "KEY"}, local)):
        raise UnsupportedOpError("sort: asymmetric comparator")

    # Evaluate the comparator's key chain on the full operand: every op
    # in it is elementwise, so the scalar block code runs on arrays.
    # Seed with the enclosing env (jax hoists comparator constants out).
    benv = dict(env) if env else {}
    for j, x in enumerate(ins):
        if 2 * j < len(args):
            benv[args[2 * j]] = x
        if 2 * j + 1 < len(args):
            benv[args[2 * j + 1]] = x
    for o in body[:-1]:
        handler = REGISTRY.get(o.name)
        if handler is None:
            raise UnsupportedOpError(f"sort: comparator op {o.name}")
        vals = [benv[w] for w in o.operands]
        res = handler(interp, o, vals, benv)
        if isinstance(res, mx.array):
            res = [res]
        for r, v in zip(o.results, res or []):
            benv[r] = v
    key = benv[lhs] if not isinstance(lhs, ir.BlockArgument) else ins[k]

    if dtypes.is_float(key.dtype):
        key = dtypes.total_order_key(key)
    elif dtypes.is_bool(key.dtype):
        key = key.astype(mx.uint8)
    if descending:
        key = ~key  # bitwise NOT reverses order for signed and unsigned
    idx = mx.argsort(key, axis=dim)
    outs = [mx.take_along_axis(x, idx, axis=dim) for x in ins]
    return outs if len(outs) > 1 else outs[0]
