"""stablehlo.reduce with canonical single-op bodies."""

import mlx.core as mx

from metaljax import _ir, dtypes
from metaljax.interpreter import register, UnsupportedOpError

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

    raise UnsupportedOpError(
        f"stablehlo.reduce with body {[o.name for o in body_ops]} "
        f"({n} inputs) not implemented"
    )


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

    raise UnsupportedOpError(
        f"stablehlo.reduce_window (window={wd}, strides={strides}, pad={pad.tolist()}, "
        f"body={[o.name for o in body_ops]}) not implemented"
    )
