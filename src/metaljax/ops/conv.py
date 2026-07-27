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


@register("stablehlo.convolution")
def _convolution(interp, op, ins, env):
    lhs, rhs = ins
    d = _conv_dims(op)
    rank = len(d["is"])
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
        rev = [bool(b) for b in
               np.array(ir.DenseElementsAttr(op.attributes["window_reversal"]))]
        if any(rev):
            if not all(rev):
                raise UnsupportedOpError("conv: mixed window_reversal")
            flip = True

    out_dtype = dtypes.mx_result_dtype(op.results[0])
    if not dtypes.is_float(out_dtype):
        raise UnsupportedOpError(
            f"conv: non-float dtype {out_dtype} (MLX conv is float-only)")

    # MLX layouts: input (N, *spatial, C_in), weight (C_out, *spatial, C_in).
    x = mx.transpose(lhs, [d["ib"]] + d["is"] + [d["if"]])
    w = mx.transpose(rhs, [d["ko"]] + d["ks"] + [d["ki"]])
    if x.dtype != out_dtype:
        x = x.astype(out_dtype)
    if w.dtype != out_dtype:
        w = w.astype(out_dtype)
    lo = [int(p) for p in pad[:, 0]]
    hi = [int(p) for p in pad[:, 1]]

    def run(xi, wi):
        return mx.conv_general(
            xi, wi, stride=[int(s) for s in strides], padding=(lo, hi),
            kernel_dilation=[int(v) for v in rdil],
            input_dilation=[int(v) for v in ldil],
            groups=fgc, flip=flip)

    if bgc > 1:
        # XLA batch groups: batch and kernel output features split into
        # bgc groups, group i convolved with kernel group i, outputs
        # concatenated along the feature axis (used by grad-of-weights).
        xs = mx.split(x, bgc, axis=0)
        ws = mx.split(w, bgc, axis=0)
        out = mx.concatenate([run(xi, wi) for xi, wi in zip(xs, ws)],
                             axis=-1)
    else:
        out = run(x, w)

    # out is (N, *spatial, C): place dims where the output layout wants.
    axes = [0] * (rank + 2)
    axes[d["ob"]] = 0
    axes[d["of"]] = rank + 1
    for k, s in enumerate(d["os"]):
        axes[s] = 1 + k
    return mx.transpose(out, axes)
