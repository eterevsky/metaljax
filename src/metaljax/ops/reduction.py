"""stablehlo.reduce with canonical single-op bodies."""

import mlx.core as mx

from metaljax import _ir, dtypes
from metaljax.interpreter import REGISTRY, register, UnsupportedOpError

# body op name -> (reduction fn, combine-with-init fn)
_REDUCERS = {
    "stablehlo.add": (mx.sum, mx.add),
    "stablehlo.multiply": (mx.prod, mx.multiply),
    "stablehlo.maximum": (mx.max, mx.maximum),
    "stablehlo.minimum": (mx.min, mx.minimum),
}
_BOOL_REDUCERS = {
    "stablehlo.or": (mx.any, mx.logical_or),
    "stablehlo.and": (mx.all, mx.logical_and),
    "stablehlo.add": (mx.any, mx.logical_or),  # or on i1 sometimes lowers as add
}


def _eval_body(interp, body, bindings, env):
    """Run a reduce-body block elementwise on full arrays; returns the
    values bound to the terminator's operands."""
    benv = dict(env) if env else {}
    benv.update(bindings)
    ops_ = [o.operation for o in body.operations]
    for o in ops_[:-1]:
        handler = REGISTRY.get(o.name)
        if handler is None:
            raise UnsupportedOpError(f"reduce body op {o.name}")
        res = handler(interp, o, [benv[w] for w in o.operands], benv)
        if isinstance(res, mx.array):
            res = [res]
        for r, v in zip(o.results, res or []):
            benv[r] = v
    return [benv[w] for w in ops_[-1].operands]


def _generic_reduce(interp, op, inputs, inits, dims, env):
    """Any associative body, any number of operands: move the reduced
    dims to one trailing axis and combine pairwise (log2 rounds), padding
    odd extents with the init values."""
    body = op.regions[0].blocks[0]
    args = list(body.arguments)
    n = len(inputs)
    rank = len(inputs[0].shape)
    keep = [i for i in range(rank) if i not in dims]
    xs = []
    for x in inputs:
        x = mx.transpose(x, keep + list(dims))
        r = 1
        for d_ in dims:
            r *= inputs[0].shape[d_]
        xs.append(mx.reshape(x, [inputs[0].shape[i] for i in keep] + [r]))
    r = xs[0].shape[-1]
    if r == 0:
        out_shape = [inputs[0].shape[i] for i in keep]
        return [mx.broadcast_to(init, out_shape) for init in inits]
    while r > 1:
        if r % 2:
            xs = [mx.concatenate(
                [x, mx.broadcast_to(init.astype(x.dtype),
                                    x.shape[:-1] + (1,))], axis=-1)
                for x, init in zip(xs, inits)]
            r += 1
        bindings = {}
        for j in range(n):
            bindings[args[j]] = xs[j][..., 0::2]
            bindings[args[n + j]] = xs[j][..., 1::2]
        xs = _eval_body(interp, body, bindings, env)
        r //= 2
    xs = [mx.squeeze(x, axis=-1) for x in xs]
    # fold in the init once (XLA folds from init; order is unspecified)
    bindings = {}
    for j in range(n):
        bindings[args[j]] = inits[j]
        bindings[args[n + j]] = xs[j]
    return _eval_body(interp, op.regions[0].blocks[0], bindings, env)


@register("stablehlo.reduce")
def _reduce(interp, op, ins, env):
    n = len(ins) // 2
    inputs, inits = ins[:n], ins[n:]
    dims = _ir.i64_list(op, "dimensions")
    body_ops = [o.operation for o in op.regions[0].blocks[0].operations]

    if n == 1 and len(body_ops) == 2:
        body_name = body_ops[0].name
        table = _BOOL_REDUCERS if dtypes.is_bool(inputs[0].dtype) else _REDUCERS
        if body_name in table:
            fn, combine = table[body_name]
            x = inputs[0]
            if any(s == 0 for s in x.shape):
                # MLX reducers crash on zero-size inputs (mx.max raises;
                # zero-size uint32 sum aborts in a missing Metal kernel).
                # An empty fold is well-defined: the init value.
                out_shape = [s for i, s in enumerate(x.shape)
                             if i not in dims]
                if any(x.shape[d] == 0 for d in dims):
                    return [mx.broadcast_to(inits[0].astype(x.dtype),
                                            out_shape)]
                return [mx.zeros(out_shape, dtype=x.dtype)]
            out = fn(x, axis=tuple(dims)) if dims else x
            return [combine(out, inits[0])]

    if n == 2 and len(dims) == 1 and not dtypes.is_bool(inputs[0].dtype):
        # (values, indices) reduce as lowered by jax argmax/argmin: the first
        # value comparison's direction identifies max vs min; ties resolve to
        # the lowest index, matching MLX's first-occurrence argmax/argmin.
        from metaljax.ops.elementwise import _comparison_direction
        first = next((_comparison_direction(o) for o in body_ops
                      if o.name == "stablehlo.compare"), None)
        if first in ("GT", "GE", "LT", "LE"):
            d = dims[0]
            is_max = first in ("GT", "GE")
            if any(s == 0 for s in inputs[0].shape):
                # zero-size batch dims: empty results (jax forbids
                # argmax over an empty reduced axis, so only batch dims
                # can be 0 here); MLX reducers crash on empties.
                out_shape = [s for i, s in enumerate(inputs[0].shape)
                             if i != d]
                return [mx.zeros(out_shape, dtype=inputs[0].dtype),
                        mx.zeros(out_shape, dtype=inputs[1].dtype)]
            val = (mx.max if is_max else mx.min)(inputs[0], axis=d)
            arg = (mx.argmax if is_max else mx.argmin)(inputs[0], axis=d)
            if dtypes.is_float(inputs[0].dtype):
                # XLA/numpy semantics: NaN wins argmax/argmin, first NaN's
                # index is returned (MLX skips NaNs).
                isnan = mx.isnan(inputs[0])
                has_nan = mx.any(isnan, axis=d)
                first_nan = mx.argmax(isnan, axis=d)
                arg = mx.where(has_nan, first_nan, arg)
                val = mx.where(
                    has_nan, mx.array(float("nan"), dtype=val.dtype), val)
            idx = mx.take_along_axis(inputs[1], mx.expand_dims(arg, d), axis=d)
            return [val, mx.squeeze(idx, axis=d)]

    # Anything else (bitwise and/or/xor, variadic min-with-index chains,
    # arbitrary combiner compositions): generic pairwise reduction.
    res = _generic_reduce(interp, op, inputs, inits, list(dims), env)
    return res if len(res) > 1 else res[0]


_CUM_OPS = {
    "stablehlo.add": mx.cumsum,
    "stablehlo.maximum": mx.cummax,
    "stablehlo.minimum": mx.cummin,
    "stablehlo.multiply": mx.cumprod,
}


def _opt_i64_list(op, name, default):
    if name in op.attributes:
        return _ir.i64_list(op, name)
    return default


def _extract_windows(x, init, wd, strides, pad, rank):
    """Pad with init and extract sliding windows: result shape
    out_sizes + (prod(window),)."""
    if (pad != 0).any():
        widths = [(int(lo), int(hi)) for lo, hi in pad]
        x = mx.pad(x, widths, constant_values=init.astype(x.dtype))
    x = mx.contiguous(x)
    es = []
    acc = 1
    for s in reversed(x.shape):
        es.append(acc)
        acc *= s
    es = es[::-1]
    out_sizes = [max(0, (x.shape[i] - int(wd[i])) // int(strides[i]) + 1)
                 for i in range(rank)]
    wflat = 1
    for w_ in wd:
        wflat *= int(w_)
    win = mx.as_strided(
        x, tuple(out_sizes) + tuple(int(w) for w in wd),
        tuple(es[i] * int(strides[i]) for i in range(rank)) + tuple(es), 0)
    return mx.reshape(win, tuple(out_sizes) + (wflat,))


@register("stablehlo.reduce_window")
def _reduce_window(interp, op, ins, env):
    import numpy as np
    from jaxlib.mlir import ir

    n = len(ins) // 2
    inputs = ins[:n]
    x = inputs[0]
    rank = len(x.shape)
    wd = _opt_i64_list(op, "window_dimensions", [1] * rank)
    strides = _opt_i64_list(op, "window_strides", [1] * rank)
    bdil = _opt_i64_list(op, "base_dilations", [1] * rank)
    wdil = _opt_i64_list(op, "window_dilations", [1] * rank)
    if "padding" in op.attributes:
        pad = np.array(ir.DenseIntElementsAttr(op.attributes["padding"])).reshape(rank, 2)
    else:
        pad = np.zeros((rank, 2), np.int64)
    body_ops = [o.operation for o in op.regions[0].blocks[0].operations]

    # Recognize the cumulative-reduction pattern JAX emits (cumsum et al.):
    # full-size window along one axis, prefix (or suffix) padding, unit strides.
    if (
        n == 1
        and len(body_ops) == 2
        and body_ops[0].name in _CUM_OPS
        and all(s == 1 for s in strides)
        and all(d == 1 for d in bdil + wdil)
    ):
        big = [i for i, w in enumerate(wd) if w > 1]
        if len(big) == 1:
            ax = big[0]
            size = x.shape[ax]
            others_ok = all(
                (wd[i] == 1 and pad[i, 0] == 0 and pad[i, 1] == 0)
                for i in range(rank) if i != ax
            )
            fn = _CUM_OPS[body_ops[0].name]
            if others_ok and wd[ax] == size:
                if pad[ax, 0] == size - 1 and pad[ax, 1] == 0:
                    return [fn(x, axis=ax)]
                if pad[ax, 0] == 0 and pad[ax, 1] == size - 1:
                    return [fn(x, axis=ax, reverse=True)]

    if n >= 2:
        if any(b != 1 for b in bdil + wdil):
            raise UnsupportedOpError(
                "variadic reduce_window with dilations not implemented")
        inits_v = ins[n:]
        wins = [_extract_windows(xi, initi, wd, strides, pad, rank)
                for xi, initi in zip(inputs, inits_v)]
        out_rank = len(wins[0].shape) - 1
        if wins[0].shape[-1] == 0 or any(
                s == 0 for s in wins[0].shape[:-1]):
            return [mx.broadcast_to(i.astype(x.dtype),
                                    wins[0].shape[:-1])
                    for i, x in zip(inits_v, inputs)]
        # Fast path — select_and_gather_add: one compare on the first
        # pair + selects picks a single window element for all outputs.
        from metaljax.ops.elementwise import _comparison_direction
        first_cmp = next((o for o in body_ops
                          if o.name == "stablehlo.compare"), None)
        n_selects = sum(o.name == "stablehlo.select" for o in body_ops)
        n_cmps = sum(o.name == "stablehlo.compare" for o in body_ops)
        if first_cmp is not None and n_cmps == 1 and n_selects >= n:
            direction = _comparison_direction(first_cmp)
            is_max = direction in ("GE", "GT")
            arg = (mx.argmax if is_max else mx.argmin)(wins[0], axis=-1)
            return [mx.squeeze(
                mx.take_along_axis(wv, mx.expand_dims(arg, -1), axis=-1),
                axis=-1) for wv in wins]
        # General variadic body: pairwise reduction over the window axis.
        return _generic_reduce(interp, op, wins, list(inits_v),
                               [out_rank], env)
    (init,) = ins[n:]

    # General path: pad with the init value, extract windows with
    # mx.as_strided, then reduce the flattened window axis (monoid table
    # fast path, generic pairwise body otherwise).
    if any(b != 1 for b in bdil):
        # base dilation inserts init-valued holes between elements
        for ax, b in enumerate(bdil):
            if b == 1:
                continue
            shape = list(x.shape)
            shape[ax] = shape[ax] * b
            holes = mx.broadcast_to(init.astype(x.dtype), tuple(shape))
            holes = mx.contiguous(holes)
            sl = [slice(None)] * rank
            sl[ax] = slice(0, None, b)
            holes[tuple(sl)] = x
            sl[ax] = slice(0, shape[ax] - (b - 1))
            x = holes[tuple(sl)]
    if (pad != 0).any():
        if (pad < 0).any():
            raise UnsupportedOpError("reduce_window negative padding")
        widths = [(int(lo), int(hi)) for lo, hi in pad]
        x = mx.pad(x, widths, constant_values=init.astype(x.dtype))

    x = mx.contiguous(x)
    out_sizes = []
    for i in range(rank):
        span = (wd[i] - 1) * wdil[i] + 1
        out_sizes.append(max(0, (x.shape[i] - span) // strides[i] + 1))
    est = 1
    for o_, w_ in zip(out_sizes, wd):
        est *= o_ * w_
    if est > 200_000_000:
        raise UnsupportedOpError(
            f"reduce_window materialization too large ({est} elements)")
    elem_strides = []
    acc = 1
    for s in reversed(x.shape):
        elem_strides.append(acc)
        acc *= s
    elem_strides = elem_strides[::-1]
    view_shape = out_sizes + [int(w) for w in wd]
    view_strides = ([elem_strides[i] * int(strides[i]) for i in range(rank)]
                    + [elem_strides[i] * int(wdil[i]) for i in range(rank)])
    win = mx.as_strided(x, tuple(view_shape), tuple(view_strides), 0)
    wflat = 1
    for w_ in wd:
        wflat *= int(w_)
    win = mx.reshape(win, tuple(out_sizes) + (wflat,))
    if any(o == 0 for o in out_sizes) or wflat == 0:
        return [mx.broadcast_to(init.astype(x.dtype), tuple(out_sizes))]

    if len(body_ops) == 2:
        table = _BOOL_REDUCERS if dtypes.is_bool(win.dtype) else _REDUCERS
        if body_ops[0].name in table:
            fn, combine = table[body_ops[0].name]
            if wflat == 0:
                return [mx.broadcast_to(init.astype(x.dtype),
                                        tuple(out_sizes))]
            return [combine(fn(win, axis=-1), init)]
    res = _generic_reduce(interp, op, [win], [init],
                          [len(out_sizes)], env)
    return res


@register("stablehlo.select_and_scatter")
def _select_and_scatter(interp, op, ins, env):
    """Max/min-pool backward: route each source element to the window
    position its select (GE/LE compare) chose, combining with add."""
    import numpy as np
    from jaxlib.mlir import ir
    from metaljax.ops.elementwise import _comparison_direction

    operand, source, init = ins
    rank = len(operand.shape)
    wd = _opt_i64_list(op, "window_dimensions", [1] * rank)
    strides = _opt_i64_list(op, "window_strides", [1] * rank)
    if "padding" in op.attributes:
        pad = np.array(
            ir.DenseIntElementsAttr(op.attributes["padding"])).reshape(rank, 2)
    else:
        pad = np.zeros((rank, 2), np.int64)

    sel_ops = [o.operation for o in op.regions[0].blocks[0].operations]
    sct_ops = [o.operation for o in op.regions[1].blocks[0].operations]
    if not (len(sel_ops) == 2 and sel_ops[0].name == "stablehlo.compare"
            and len(sct_ops) == 2 and sct_ops[0].name == "stablehlo.add"):
        raise UnsupportedOpError(
            f"select_and_scatter bodies "
            f"{[o.name for o in sel_ops]} / {[o.name for o in sct_ops]}")
    direction = _comparison_direction(sel_ops[0])
    if direction in ("GE", "GT"):
        is_max, fill = True, float("-inf")
    elif direction in ("LE", "LT"):
        is_max, fill = False, float("inf")
    else:
        raise UnsupportedOpError(f"select_and_scatter compare {direction}")

    x = operand
    if (pad != 0).any():
        widths = [(int(lo), int(hi)) for lo, hi in pad]
        x = mx.pad(x, widths, constant_values=mx.array(fill, dtype=x.dtype))
    elem_strides = []
    acc = 1
    for s in reversed(x.shape):
        elem_strides.append(acc)
        acc *= s
    elem_strides = elem_strides[::-1]
    out_sizes = [max(0, (x.shape[i] - int(wd[i])) // int(strides[i]) + 1)
                 for i in range(rank)]
    win = mx.as_strided(
        x, tuple(out_sizes) + tuple(int(w) for w in wd),
        tuple(elem_strides[i] * int(strides[i]) for i in range(rank))
        + tuple(elem_strides), 0)
    wflat = 1
    for w in wd:
        wflat *= int(w)
    win = mx.reshape(win, tuple(out_sizes) + (wflat,))
    arg = (mx.argmax if is_max else mx.argmin)(win, axis=-1)  # first hit,
    # matching XLA's in-order GE/LE select

    # absolute flat position in the padded array of each window's winner
    flat = mx.zeros(tuple(out_sizes), dtype=mx.int64)
    rem = arg.astype(mx.int64)
    for i in reversed(range(rank)):
        off = rem % int(wd[i])
        rem = rem // int(wd[i])
        pos = mx.reshape(
            mx.arange(out_sizes[i], dtype=mx.int64) * int(strides[i]),
            [-1 if j == i else 1 for j in range(rank)])
        flat = flat + (pos + off) * elem_strides[i]

    total = 1
    for s in x.shape:
        total *= s
    base = mx.zeros((total,), dtype=operand.dtype)
    base = base.at[mx.reshape(flat, (-1,))].add(
        mx.reshape(source.astype(operand.dtype), (-1,)))
    base = mx.reshape(base, x.shape)
    if (pad != 0).any():
        sl = tuple(slice(int(pad[i, 0]), int(pad[i, 0]) + operand.shape[i])
                   for i in range(rank))
        base = base[sl]
    return base + init.astype(operand.dtype)
