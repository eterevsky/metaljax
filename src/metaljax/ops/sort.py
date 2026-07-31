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
    # NB dedup by the ir.Value (stable hash), never by id() of the
    # transient .owner wrapper — CPython reuses freed wrapper addresses,
    # which silently truncated the walk.
    seen, deps, stack = set(), set(), [val]
    while stack:
        v = stack.pop()
        if isinstance(v, ir.BlockArgument):
            if v in args:
                deps.add(args.index(v))
            continue
        if v in seen or v not in local:
            continue
        seen.add(v)
        stack.extend(v.owner.operation.operands)
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
    args = list(block.arguments)
    local = {r for o in body for r in o.results}

    if cmp.name != "stablehlo.compare":
        # Compare/select trees: jax emits exactly two shapes here — the
        # complex lexicographic (re, im) comparator over ONE operand
        # pair, and the multi-key lexicographic comparator over the
        # first num_keys operand pairs (lax.sort num_keys > 1, lexsort,
        # unique over rows). Both are ascending.
        deps = _arg_deps(cmp_val, args, local)
        ks = {d // 2 for d in deps}
        if len(ks) == 1:
            k = ks.pop()
            if k < len(ins) and ins[k].dtype == mx.complex64:
                return _gather_sorted(ins, ins[k], False, dim)
        if ks and ks == set(range(len(ks))) and len(ks) <= len(ins):
            return _lex_sorted(ins, len(ks), dim)
        raise UnsupportedOpError(
            f"sort: comparator ends in {cmp.name}, not compare")
    direction = str(cmp.attributes["comparison_direction"])
    if "LT" in direction:
        descending = False
    elif "GT" in direction:
        descending = True
    else:
        raise UnsupportedOpError(f"sort: non-strict compare {direction}")

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
    return _gather_sorted(ins, key, descending, dim)


def _sort_key(x):
    """Ascending sort key with jax's canonicalization: -0 == +0, all
    NaNs equal and largest; complex lexicographic by (re, im)."""
    if x.dtype == mx.complex64:
        re_k = dtypes.total_order_key(_canon_float(mx.real(x))).astype(
            mx.uint64)
        im_k = dtypes.total_order_key(_canon_float(mx.imag(x))).astype(
            mx.uint64)
        return (re_k << 32) | im_k
    if dtypes.is_float(x.dtype):
        return dtypes.total_order_key(_canon_float(x))
    if dtypes.is_bool(x.dtype):
        return x.astype(mx.uint8)
    return x


def _canon_float(x):
    x = x + mx.array(0, dtype=x.dtype)  # -0 + +0 == +0
    return mx.where(mx.isnan(x), mx.array(float("nan"), dtype=x.dtype), x)


def _argsort(key, axis):
    """mx.argsort/mx.sort read the WRONG elements from a non-contiguous
    input (MLX 0.32): jax lowers a non-last-axis sort/top_k as
    transpose -> sort -> transpose, so the operand is routinely a strided
    view. Materialize first; a no-op when the input is already dense."""
    return mx.argsort(mx.contiguous(key), axis=axis)


def _lex_sorted(ins, num_keys, dim):
    """Stable lexicographic sort by operands[0..num_keys-1], ascending:
    successive stable argsorts from the last key to the first."""
    perm = None
    for j in reversed(range(num_keys)):
        key = _sort_key(ins[j])
        if perm is not None:
            key = mx.take_along_axis(key, perm, axis=dim)
        idx = _argsort(key, axis=dim)
        perm = idx if perm is None else mx.take_along_axis(perm, idx,
                                                           axis=dim)
    outs = [mx.take_along_axis(x, perm, axis=dim) for x in ins]
    return outs if len(outs) > 1 else outs[0]


def _gather_sorted(ins, key, descending, dim):

    if key.dtype == mx.complex64:
        # numpy/XLA order complex lexicographically by (real, imag), on
        # CANONICALIZED components: -0 ties with +0 and every NaN with
        # every other NaN.  The tie matters here (unlike a real sort,
        # where nothing can be ordered between -0 and +0): a -0 real
        # part must not split a group whose imaginary parts then decide
        # the order.  jax emits the canonicalization inside the
        # comparator for real keys, but for complex keys it emits a
        # select tree we recognize structurally instead of evaluating,
        # so we must apply it here.  (jnp.unique lexsorts complex rows
        # and then compares neighbours -- unsorted ties => extra
        # "uniques".)
        key = _sort_key(key)
    elif dtypes.is_float(key.dtype):
        key = dtypes.total_order_key(key)
    elif dtypes.is_bool(key.dtype):
        key = key.astype(mx.uint8)
    if descending:
        key = ~key  # bitwise NOT reverses order for signed and unsigned
    idx = _argsort(key, axis=dim)
    outs = [mx.take_along_axis(x, idx, axis=dim) for x in ins]
    return outs if len(outs) > 1 else outs[0]


@register("chlo.top_k")
def _top_k(interp, op, ins, env):
    (x,) = ins
    k = _ir.int_attr(op, "k")
    key = dtypes.total_order_key(x) if dtypes.is_float(x.dtype) else x
    if dtypes.is_bool(key.dtype):
        key = key.astype(mx.uint8)
    idx = _argsort(~key, axis=-1)[..., :k]  # stable descending
    return [mx.take_along_axis(x, idx, axis=-1), idx.astype(mx.int32)]


def approx_top_k(op, ins):
    """ApproxTopK custom call: exact top-k is a valid implementation
    (recall 1.0 >= any recall_target). Operands (values, indices,
    init_val, init_idx); results (values, indices) with top_k along
    reduction_dim."""
    import re as _re
    cfg = str(op.attributes["mhlo.backend_config"])
    k = int(_re.search(r"top_k = (\d+)", cfg).group(1))
    dim = int(_re.search(r"reduction_dim = (\d+)", cfg).group(1))
    comp = str(op.attributes["called_computations"])
    is_max = "_gt_" in comp or "_ge_" in comp
    vals, idxs = ins[0], ins[1]
    try:
        out_n = int(ir.RankedTensorType(op.results[0].type).shape[dim])
    except Exception:
        out_n = -1
    if out_n > 0:
        # aggregate_to_topk=False: the result is wider than backend_config's
        # top_k (XLA returns an unsorted approximate set of that width).
        # Slicing to top_k under-fills the result buffer, leaving the tail
        # as uninitialised device memory.
        k = max(k, min(out_n, vals.shape[dim]))
    key = _sort_key(vals)
    if is_max:
        key = ~key
    order = _argsort(key, axis=dim)
    sl = [slice(None)] * len(vals.shape)
    sl[dim] = slice(0, k)
    top = order[tuple(sl)]
    return [mx.take_along_axis(vals, top, axis=dim),
            mx.take_along_axis(idxs, top, axis=dim)]
