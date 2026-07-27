"""dot_general and friends."""

import re

import mlx.core as mx

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from metaljax import dtypes
from metaljax.interpreter import register


def _dot_dims(op):
    attr = op.attributes["dot_dimension_numbers"]
    try:
        dn = stablehlo.DotDimensionNumbers(attr)
        return (
            list(dn.lhs_batching_dimensions),
            list(dn.rhs_batching_dimensions),
            list(dn.lhs_contracting_dimensions),
            list(dn.rhs_contracting_dimensions),
        )
    except Exception:
        s = str(attr)
        def grab(name):
            m = re.search(rf"{name}\s*=\s*\[([^\]]*)\]", s)
            return [int(x) for x in m.group(1).split(",") if x.strip()] if m else []
        return (
            grab("lhs_batching_dimensions"),
            grab("rhs_batching_dimensions"),
            grab("lhs_contracting_dimensions"),
            grab("rhs_contracting_dimensions"),
        )


@register("stablehlo.dot_general")
def _dot_general(interp, op, ins, env):
    lhs, rhs = ins
    lb, rb, lc, rc = _dot_dims(op)
    out_dtype = dtypes.mx_result_dtype(op.results[0])

    lfree = [d for d in range(len(lhs.shape)) if d not in lb and d not in lc]
    rfree = [d for d in range(len(rhs.shape)) if d not in rb and d not in rc]

    l = mx.transpose(lhs, lb + lfree + lc)
    r = mx.transpose(rhs, rb + rc + rfree)

    def prod(xs):
        p = 1
        for x in xs:
            p *= x
        return p

    batch = [lhs.shape[d] for d in lb]
    m = [lhs.shape[d] for d in lfree]
    k = [lhs.shape[d] for d in lc]
    n = [rhs.shape[d] for d in rfree]

    l3 = mx.reshape(l, (prod(batch), prod(m), prod(k)))
    r3 = mx.reshape(r, (prod(batch), prod(k), prod(n)))

    if prod(batch) * prod(m) * prod(n) == 0:
        # mx.matmul with an empty M/N output yields an array whose host
        # conversion segfaults (null data pointer, MLX 0.32); the result
        # carries no data anyway, so construct it directly.
        return mx.zeros(batch + m + n, dtype=out_dtype)

    if dtypes.is_int(out_dtype) or dtypes.is_bool(out_dtype):
        # MLX matmul is float-only; do an explicit multiply-accumulate.
        acc = l3.astype(mx.int64) if not dtypes.is_bool(out_dtype) else l3
        prod_ = acc[:, :, :, None] * r3.astype(acc.dtype)[:, None, :, :]
        o3 = mx.sum(prod_, axis=2)
        o3 = o3.astype(out_dtype)
    else:
        if l3.dtype != out_dtype:
            l3 = l3.astype(out_dtype)
        if r3.dtype != out_dtype:
            r3 = r3.astype(out_dtype)
        o3 = mx.matmul(l3, r3)

    return mx.reshape(o3, batch + m + n)


@register("stablehlo.dot")
def _dot(interp, op, ins, env):
    # Plain rank<=2 dot (HLO-imported modules; jax itself emits dot_general):
    # contracts the last dim of lhs with the first dim of rhs, numpy-style.
    t = ir.RankedTensorType(op.results[0].type)
    if any(d == 0 for d in t.shape):
        # see _dot_general: empty matmul outputs segfault on host copy
        return mx.zeros(list(t.shape),
                        dtype=dtypes.mx_result_dtype(op.results[0]))
    return mx.matmul(ins[0], ins[1])
