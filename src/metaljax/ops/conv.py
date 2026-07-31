"""stablehlo.convolution -> mx.conv_general."""

import re

import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from metaljax import _ir, dtypes
from metaljax.interpreter import register, UnsupportedOpError


def _conv_dims(op):
    attr = op.attributes["dimension_numbers"]
    try:
        dn = stablehlo.ConvDimensionNumbers(attr)
        return {
            "ib": dn.input_batch_dimension,
            "if": dn.input_feature_dimension,
            "is": list(dn.input_spatial_dimensions),
            "ko": dn.kernel_output_feature_dimension,
            "ki": dn.kernel_input_feature_dimension,
            "ks": list(dn.kernel_spatial_dimensions),
            "ob": dn.output_batch_dimension,
            "of": dn.output_feature_dimension,
            "os": list(dn.output_spatial_dimensions),
        }
    except Exception:
        # Text form: [b, f, 1, 0]x[o, i, 1, 0]->[b, f, 1, 0]
        text = str(attr)
        m = re.search(r"\[([^\]]*)\]x\[([^\]]*)\]->\[([^\]]*)\]", text)
        if not m:
            raise UnsupportedOpError(f"conv dim numbers: {text}")

        def parse(part, batch, feat):
            toks = [t.strip() for t in part.split(",")]
            spatial = {}
            out = {"s": []}
            for pos, t in enumerate(toks):
                if t == batch:
                    out["b"] = pos
                elif t == feat:
                    out["f"] = pos
                else:
                    spatial[int(t)] = pos
            out["s"] = [spatial[k] for k in sorted(spatial)]
            return out

        i = parse(m.group(1), "b", "f")
        k = parse(m.group(2), "o", "i")
        o = parse(m.group(3), "b", "f")
        return {"ib": i["b"], "if": i["f"], "is": i["s"],
                "ko": k["b"], "ki": k["f"], "ks": k["s"],
                "ob": o["b"], "of": o["f"], "os": o["s"]}


def _opt_list(op, name, default):
    if name in op.attributes:
        return _ir.i64_list(op, name)
    return list(default)


def _int_conv(x, w, strides, ldil, rdil, pad, flip, out_dtype):
    """Exact integer convolution: im2col patches + int64 multiply-sum
    (mx.conv_general is float-only; f32 emulation would round)."""
    rank = len(x.shape) - 2
    if any(b != 1 for b in ldil):
        # input dilation: insert zero holes
        for ax, b in enumerate(ldil):
            if b == 1:
                continue
            shape = list(x.shape)
            shape[ax + 1] = shape[ax + 1] * b
            holes = mx.zeros(tuple(shape), dtype=x.dtype)
            sl = [slice(None)] * len(shape)
            sl[ax + 1] = slice(0, None, b)
            holes[tuple(sl)] = x
            sl[ax + 1] = slice(0, shape[ax + 1] - (b - 1))
            x = holes[tuple(sl)]
    widths = [(0, 0)] + [(int(l), int(h)) for l, h in pad] + [(0, 0)]
    if any(v for p_ in widths for v in p_):
        x = mx.pad(x, widths)
    x = mx.contiguous(x)
    wd = list(w.shape[1:-1])
    C = x.shape[-1]
    out_sizes = [
        max(0, (x.shape[i + 1] - ((wd[i] - 1) * int(rdil[i]) + 1))
            // int(strides[i]) + 1)
        for i in range(rank)]
    es = []
    acc = 1
    for s in reversed(x.shape):
        es.append(acc)
        acc *= s
    es = es[::-1]
    view_shape = [x.shape[0]] + out_sizes + wd + [C]
    view_strides = ([es[0]]
                    + [es[i + 1] * int(strides[i]) for i in range(rank)]
                    + [es[i + 1] * int(rdil[i]) for i in range(rank)]
                    + [es[-1]])
    patches = mx.as_strided(x, tuple(view_shape), tuple(view_strides), 0)
    if flip:
        for ax in range(rank):
            n = wd[ax]
            if n > 1:
                w = mx.take(w, mx.arange(n - 1, -1, -1), axis=ax + 1)
    K = C
    for v in wd:
        K *= v
    p2 = mx.reshape(patches, tuple([x.shape[0]] + out_sizes + [K]))
    w2 = mx.reshape(w, (w.shape[0], K))
    acc_t = mx.bool_ if x.dtype == mx.bool_ else mx.int64
    prod = p2[..., None, :].astype(acc_t) * w2.astype(acc_t)
    out = (mx.any(prod, axis=-1) if acc_t == mx.bool_
           else mx.sum(prod, axis=-1))
    return out.astype(out_dtype)


@register("stablehlo.convolution")
def _convolution(interp, op, ins, env):
    lhs, rhs = ins
    out_t = ir.RankedTensorType(op.results[0].type)
    if any(s == 0 for s in out_t.shape) or lhs.size == 0 or rhs.size == 0:
        # An empty operand means every output element sums an empty set of
        # products, so the result is all zeros at the XLA-declared shape.
        # (Never hand this to MLX: for a zero-size *spatial* input dim it
        # computes the dilated extent as (0-1)*d+1 instead of XLA's 0, so
        # mx.conv_general returns a narrower array and the engine then reads
        # uninitialized memory past its end.)
        return mx.zeros(list(out_t.shape),
                        dtype=dtypes.mx_dtype_for(out_t.element_type))
    d = _conv_dims(op)
    rank = len(d["is"])
    if rank == 0:
        # No spatial dims: a (grouped) matmul over features.
        fgc0 = (_ir.int_attr(op, "feature_group_count")
                if "feature_group_count" in op.attributes else 1)
        bgc0 = (_ir.int_attr(op, "batch_group_count")
                if "batch_group_count" in op.attributes else 1)
        x0 = mx.transpose(lhs, [d["ib"], d["if"]]).astype(mx.float32)
        w0 = mx.transpose(rhs, [d["ko"], d["ki"]]).astype(mx.float32)

        def mm(xg, wg):
            return mx.matmul(xg, mx.transpose(wg))

        if bgc0 > 1:
            out = mx.concatenate(
                [mm(xg, wg) for xg, wg in zip(mx.split(x0, bgc0, axis=0),
                                              mx.split(w0, bgc0, axis=0))],
                axis=-1)
        elif fgc0 > 1:
            out = mx.concatenate(
                [mm(xg, wg) for xg, wg in zip(mx.split(x0, fgc0, axis=1),
                                              mx.split(w0, fgc0, axis=0))],
                axis=-1)
        else:
            out = mm(x0, w0)
        out = out.astype(dtypes.mx_dtype_for(out_t.element_type))
        axes = [0, 0]
        axes[d["ob"]] = 0
        axes[d["of"]] = 1
        return mx.transpose(out, axes)
    strides = _opt_list(op, "window_strides", [1] * rank)
    ldil = _opt_list(op, "lhs_dilation", [1] * rank)
    rdil = _opt_list(op, "rhs_dilation", [1] * rank)
    if "padding" in op.attributes:
        pad = np.array(
            ir.DenseIntElementsAttr(op.attributes["padding"])).reshape(rank, 2)
    else:
        pad = np.zeros((rank, 2), np.int64)
    fgc = (_ir.int_attr(op, "feature_group_count")
           if "feature_group_count" in op.attributes else 1)
    bgc = (_ir.int_attr(op, "batch_group_count")
           if "batch_group_count" in op.attributes else 1)
    flip = False
    if "window_reversal" in op.attributes:
        a = op.attributes["window_reversal"]
        try:
            rev = [bool(b) for b in ir.DenseBoolArrayAttr(a)]
        except Exception:
            rev = [bool(b) for b in np.array(ir.DenseElementsAttr(a))]
        if any(rev):
            if not all(rev):
                raise UnsupportedOpError("conv: mixed window_reversal")
            flip = True

    out_dtype = dtypes.mx_result_dtype(op.results[0])

    # MLX layouts: input (N, *spatial, C_in), weight (C_out, *spatial, C_in).
    x = mx.transpose(lhs, [d["ib"]] + d["is"] + [d["if"]])
    w = mx.transpose(rhs, [d["ko"]] + d["ks"] + [d["ki"]])

    if (pad < 0).any():
        # XLA pads *after* lhs dilation, so negative padding crops the
        # dilated array. MLX crops the undilated operand instead (its output
        # is a whole dilation step short per cropped element), and mx.pad on
        # the integer path rejects negative widths. Rewrite each crop of k
        # elements as dropping q = ceil(k / dilation) operand elements plus a
        # non-negative pad of q*dilation - k: dropping q operand elements
        # removes q*dilation entries from the dilated array, and the excess
        # is exactly that many interior holes, i.e. zeros, on that side.
        pad = pad.copy()
        sl = [slice(None)] * (rank + 2)
        for a in range(rank):
            dl = int(ldil[a])
            k0 = max(0, int(-pad[a, 0]))
            k1 = max(0, int(-pad[a, 1]))
            if not k0 and not k1:
                continue
            q0, q1 = -(-k0 // dl), -(-k1 // dl)
            n = x.shape[a + 1]
            start, stop = min(q0, n), max(n - q1, 0)
            sl[a + 1] = slice(start, max(stop, start))
            if k0:
                pad[a, 0] = q0 * dl - k0
            if k1:
                pad[a, 1] = q1 * dl - k1
        x = mx.contiguous(x[tuple(sl)])
        if x.size == 0:
            return mx.zeros(list(out_t.shape),
                            dtype=dtypes.mx_dtype_for(out_t.element_type))

    lo = [int(p) for p in pad[:, 0]]
    hi = [int(p) for p in pad[:, 1]]

    # Integers go through the exact im2col path (mx.conv_general is
    # float-only and f32 emulation would round); complex is four real
    # convolutions, so it uses mx.conv_general too.
    mx_conv = dtypes.is_float(out_dtype) or out_dtype == mx.complex64
    # ...and mx.conv_general implements feature groups only for 1-D and 2-D.
    native_groups = mx_conv and rank <= 2

    def run(xi, wi, groups=1):
        if not mx_conv:
            return _int_conv(xi, wi, strides, ldil, rdil, pad, flip, out_dtype)
        return mx.conv_general(
            xi, wi, stride=[int(s) for s in strides], padding=(lo, hi),
            kernel_dilation=[int(v) for v in rdil],
            input_dilation=[int(v) for v in ldil],
            groups=groups, flip=flip)

    def feature_groups(xi, wi):
        if fgc == 1:
            return run(xi, wi)
        if native_groups:
            return run(xi, wi, fgc)
        # XLA feature groups: input features split into fgc groups, output
        # feature block g computed from input group g. Expanded into one
        # ungrouped convolution per group for the cases MLX won't take.
        return mx.concatenate(
            [run(a, b) for a, b in zip(mx.split(xi, fgc, axis=-1),
                                       mx.split(wi, fgc, axis=0))],
            axis=-1)

    def conv_all(xi, wi):
        if bgc > 1:
            # XLA batch groups: batch and kernel output features split
            # into bgc groups, group i convolved with kernel group i,
            # outputs concatenated along features (grad-of-weights).
            return mx.concatenate(
                [feature_groups(a, b)
                 for a, b in zip(mx.split(xi, bgc, axis=0),
                                 mx.split(wi, bgc, axis=0))],
                axis=-1)
        return feature_groups(xi, wi)

    if out_dtype == mx.complex64:
        # (ar + i*ai) conv (br + i*bi): four real convolutions.
        ar, ai = mx.real(x.astype(mx.complex64)), mx.imag(
            x.astype(mx.complex64))
        br, bi = mx.real(w.astype(mx.complex64)), mx.imag(
            w.astype(mx.complex64))
        re = conv_all(ar, br) - conv_all(ai, bi)
        im = conv_all(ar, bi) + conv_all(ai, br)
        out = dtypes.make_complex(re, im)
    elif not mx_conv:
        out = conv_all(x, w)
    else:
        if x.dtype != out_dtype:
            x = x.astype(out_dtype)
        if w.dtype != out_dtype:
            w = w.astype(out_dtype)
        out = conv_all(x, w)

    # out is (N, *spatial, C): place dims where the output layout wants.
    axes = [0] * (rank + 2)
    axes[d["ob"]] = 0
    axes[d["of"]] = rank + 1
    for k, s in enumerate(d["os"]):
        axes[s] = 1 + k
    # Guard against MLX sizing a window differently from XLA: silently
    # returning a wrong-shaped buffer makes the caller read past its end.
    want = ([out_t.shape[d["ob"]]] + [out_t.shape[s] for s in d["os"]]
            + [out_t.shape[d["of"]]])
    if list(out.shape) != want:
        raise UnsupportedOpError(
            f"conv: produced {list(out.shape)}, XLA declares {want} "
            f"(strides={list(strides)} pad={pad.tolist()} "
            f"lhs_dilation={list(ldil)} rhs_dilation={list(rdil)})")
    return mx.transpose(out, axes)
