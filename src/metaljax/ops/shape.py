"""Shape-manipulation StableHLO ops."""

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir, dtypes
from metaljax.interpreter import UnsupportedOpError, register


def _result_shape(op, i=0):
    return list(ir.RankedTensorType(op.results[i].type).shape)


@register("stablehlo.broadcast_in_dim")
def _broadcast_in_dim(interp, op, ins, env):
    (x,) = ins
    dims = _ir.i64_list(op, "broadcast_dimensions")
    out_shape = _result_shape(op)
    if dims != sorted(dims):
        perm = sorted(range(len(dims)), key=lambda i: dims[i])
        x = mx.transpose(x, perm)
        dims = sorted(dims)
    interim = [1] * len(out_shape)
    for i, d in enumerate(dims):
        interim[d] = x.shape[i]
    return mx.broadcast_to(mx.reshape(x, interim), out_shape)


@register("stablehlo.reshape")
def _reshape(interp, op, ins, env):
    return mx.reshape(ins[0], _result_shape(op))


@register("stablehlo.transpose")
def _transpose(interp, op, ins, env):
    return mx.transpose(ins[0], _ir.i64_list(op, "permutation"))


@register("stablehlo.slice")
def _slice(interp, op, ins, env):
    starts = _ir.i64_list(op, "start_indices")
    limits = _ir.i64_list(op, "limit_indices")
    strides = _ir.i64_list(op, "strides")
    idx = tuple(slice(b, e, s) for b, e, s in zip(starts, limits, strides))
    return ins[0][idx]


@register("stablehlo.reverse")
def _reverse(interp, op, ins, env):
    (x,) = ins
    for d in _ir.i64_list(op, "dimensions"):
        n = x.shape[d]
        if n > 1:  # size 0/1 dims: identity (mx.take chokes on empties)
            x = mx.take(x, mx.arange(n - 1, -1, -1), axis=d)
    return x


@register("stablehlo.concatenate")
def _concatenate(interp, op, ins, env):
    return mx.concatenate(ins, axis=_ir.int_attr(op, "dimension"))


@register("stablehlo.convert")
def _convert(interp, op, ins, env):
    el = str(ir.RankedTensorType(op.results[0].type).element_type)
    if el in dtypes.EMULATED:
        # quantize onto the emulated dtype's value grid
        return dtypes.quantize_emulated(ins[0], el)
    dt = dtypes.mx_result_dtype(op.results[0])
    x = ins[0]
    if x.dtype == mx.complex64 and dt != mx.complex64:
        x = mx.real(x)  # XLA convert complex->real keeps the real part
    return x.astype(dt)


def _to_bytes(x):
    """Flat little-endian byte buffer of an array's storage (rank-0 ok)."""
    return mx.reshape(mx.view(mx.reshape(x, (-1, 1)), mx.uint8), (-1,))


def _nibbles(x):
    """i4/ui4 storage (one value per byte, i4 sign-extended) -> flat uint8
    nibbles in [0, 15]."""
    return mx.bitwise_and(mx.reshape(x, (-1,)).astype(mx.uint8),
                          mx.array(0x0F, mx.uint8))


def _from_nibbles(n, name):
    """Flat nibbles -> the emulated dtype's storage values. quantize_emulated
    maps i4 -> ((n+8)%16)-8 (sign extension) and ui4 -> n%16 (identity)."""
    return dtypes.quantize_emulated(n, name)


@register("stablehlo.bitcast_convert")
def _bitcast_convert(interp, op, ins, env):
    t = ir.RankedTensorType(op.results[0].type)
    in_el = _ir.tensor_type(op.operands[0]).element_type
    out_el = t.element_type
    dt = dtypes.mx_dtype_for(out_el)
    out_shape = list(t.shape)
    x = ins[0]

    # bitcast reads BITS, so the storage must be the logical bit pattern.
    # i4/ui4 are the one exception we can reconstruct (whole nibbles);
    # f8/f4 hold VALUES in a wider float and f64 is truncated to f32, so
    # their bit patterns simply do not exist on this device.
    for el in (in_el, out_el):
        if not dtypes.storage_is_bit_exact(el) and str(el) not in ("i4", "ui4"):
            raise UnsupportedOpError(
                f"bitcast_convert on {el}: metaljax stores it in a wider "
                f"dtype, so its bit pattern is unavailable"
            )

    ib, ob = dtypes.bit_width(in_el), dtypes.bit_width(out_el)
    if ib != 4 and ob != 4:
        # Byte-multiple types: MLX storage is already the XLA layout.
        if dt.size == x.dtype.size:
            return mx.view(x, dt)
        if dt.size < x.dtype.size:
            # Narrowing: the result gains a trailing dim of ratio; mx.view
            # rescales the LAST axis, so give it a fresh unit axis to split
            # (also makes rank-0 inputs legal).
            return mx.view(mx.expand_dims(x, -1), dt)
        # Widening: the input's trailing ratio-sized dim collapses to 1.
        return mx.squeeze(mx.view(x, dt), axis=-1)

    # 4-bit end: XLA packs two values per byte along the minor-most
    # dimension, low nibble first, and a width-changing bitcast adds (or
    # removes) exactly that trailing ratio dim -- so a row-major flatten
    # makes the packed byte stream contiguous and the whole thing is one
    # linear reinterpretation.
    if x.size == 0:
        return mx.zeros(out_shape, dtype=dt)
    if ib == 4 and ob == 4:  # i4 <-> ui4: reinterpret the nibble in place
        return mx.reshape(_from_nibbles(_nibbles(x), str(out_el)), out_shape)
    if ib == 4:  # pack pairs into bytes, low nibble first
        n = _nibbles(x)
        b = n[0::2] | (n[1::2] << 4)
    else:
        b = _to_bytes(x)
    if ob == 4:  # unpack each byte into (low, high)
        lo = mx.bitwise_and(b, mx.array(0x0F, mx.uint8))
        n = mx.reshape(mx.stack([lo, b >> 4], axis=-1), (-1,))
        return mx.reshape(_from_nibbles(n, str(out_el)), out_shape)
    return mx.reshape(mx.view(b, dt), out_shape)


@register("stablehlo.pad")
def _pad(interp, op, ins, env):
    x, pad_val = ins
    low = _ir.i64_list(op, "edge_padding_low")
    high = _ir.i64_list(op, "edge_padding_high")
    interior = _ir.i64_list(op, "interior_padding")
    rank = len(x.shape)

    if any(i > 0 for i in interior):
        exp_shape = [
            (s - 1) * (i + 1) + 1 if s > 0 else 0
            for s, i in zip(x.shape, interior)
        ]
        exp = mx.broadcast_to(pad_val.astype(x.dtype), exp_shape)
        idx = tuple(slice(None, None, i + 1) for i in interior)
        exp = mx.array(exp)  # materialize before setitem
        exp[idx] = x
        x = exp

    pos = [(max(0, l), max(0, h)) for l, h in zip(low, high)]
    if any(p != (0, 0) for p in pos):
        x = mx.pad(x, pos, constant_values=pad_val.astype(x.dtype))

    if any(l < 0 or h < 0 for l, h in zip(low, high)):
        idx = []
        for d in range(rank):
            b = -low[d] if low[d] < 0 else 0
            e = x.shape[d] + high[d] if high[d] < 0 else x.shape[d]
            idx.append(slice(b, e))
        x = x[tuple(idx)]
    return x


def _clamped_starts(start_arrs, x_shape, sizes):
    starts = mx.stack([s.astype(mx.int32).reshape(()) for s in start_arrs])
    bounds = mx.array([d - s for d, s in zip(x_shape, sizes)], dtype=mx.int32)
    return mx.clip(starts, mx.array(0, dtype=mx.int32), bounds)


@register("stablehlo.dynamic_slice")
def _dynamic_slice(interp, op, ins, env):
    x, *start_arrs = ins
    sizes = _ir.i64_list(op, "slice_sizes")
    starts = _clamped_starts(start_arrs, x.shape, sizes)
    return mx.slice(x, starts, tuple(range(len(sizes))), sizes)


@register("stablehlo.dynamic_update_slice")
def _dynamic_update_slice(interp, op, ins, env):
    x, update, *start_arrs = ins
    sizes = update.shape
    starts = _clamped_starts(start_arrs, x.shape, sizes)
    return mx.slice_update(x, update, starts, tuple(range(len(x.shape))))
