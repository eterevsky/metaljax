"""dot_general and friends."""

import re

import mlx.core as mx

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from metaljax import dtypes
from metaljax.interpreter import register


# f32 holds every integer up to 2**24 exactly, so a sum of integer products
# is exact -- in ANY accumulation order -- as long as the sum of the products'
# magnitudes stays within that range (every partial sum is bounded by it too).
_F32_EXACT_INT = 1 << 24

# Largest magnitude a value of this dtype can have. The emulated i4/ui4 live
# in i8/ui8 storage and are covered by their storage type's (larger) bound.
_INT_MAX_ABS = {mx.int8: 128, mx.uint8: 255}


def _exact_f32_chunk(ld, rd):
    """Contraction-dim chunk size that keeps an integer dot exact under f32
    matmul, or None if these operands cannot use the f32 path.

    MLX has no integer matmul at all, so the general fallback materializes a
    whole [B, M, K, N] int64 outer product and reduces it in a second kernel
    -- 8 bytes of traffic per multiply-accumulate. For 8-bit operands we can
    instead run the real f32 matmul over K-slices short enough that no
    partial sum can leave f32's exact-integer range, then accumulate the
    (exact) per-chunk results in integer arithmetic. Wider operands have
    products f32 cannot represent at all, so they keep the int64 path.
    """
    ml = _INT_MAX_ABS.get(ld)
    mr = _INT_MAX_ABS.get(rd)
    if ml is None or mr is None:
        return None
    # Round down to a power of two: i8xi8 -> 1024, i8xu8 -> 512, u8xu8 -> 256.
    return 1 << ((_F32_EXACT_INT // (ml * mr)).bit_length() - 1)


def _int_dot_via_f32(l3, r3, chunk, out_dtype):
    """[B, M, K] x [B, K, N] integer dot as exact f32 matmuls over K-chunks.

    Each chunk's f32 result is an exact integer; casting it to the integer
    accumulator and adding wraps in two's complement exactly like XLA's
    integer dot, which is defined modulo the result width.
    """
    k = l3.shape[2]
    # 64-bit results must not wrap at 32 bits; everything narrower is
    # congruent mod 2**32 to the true sum, and the final astype narrows it.
    acc_dtype = mx.int64 if out_dtype.size == 8 else mx.int32
    o3 = None
    for s in range(0, k, chunk):
        if k <= chunk:
            lp, rp = l3, r3
        else:
            lp = l3[:, :, s:s + chunk]
            rp = r3[:, s:s + chunk, :]
        part = mx.matmul(lp.astype(mx.float32), rp.astype(mx.float32))
        part = part.astype(acc_dtype)
        o3 = part if o3 is None else o3 + part
    return o3.astype(out_dtype)


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

    chunk = (_exact_f32_chunk(l3.dtype, r3.dtype)
             if dtypes.is_int(out_dtype) and prod(k) != 0 else None)
    if chunk is not None:
        # Narrow integer operands: exact, and orders of magnitude cheaper
        # than materializing the outer product below.
        o3 = _int_dot_via_f32(l3, r3, chunk, out_dtype)
    elif dtypes.is_int(out_dtype) or dtypes.is_bool(out_dtype):
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
