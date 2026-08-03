"""Run openai/gpt-oss-20b from its NATIVE MXFP4 weights.

The stock keras-hub loader (`keras_hub.src.utils.transformers.
convert_gpt_oss`) dequantizes the checkpoint's MXFP4 expert tensors to
bfloat16 while porting them, which turns 10.2 GB of codes and scales into
38.2 GB of weights and leaves the device graph a plain float matmul. The
row therefore measured 41.8 GB resident and 222 ms/tok, against mlx-lm's
13.8 GB and 8.8 ms/tok on the same checkpoint.

This module keeps the checkpoint packed:

  * `GptOssExperts` stores the HF `_blocks` (two E2M1 nibbles per byte) and
    `_scales` (one E8M0 byte per 32 elements) tensors verbatim, as uint8
    variables -- 10.2 GB instead of 38.2 GB;
  * its `call` reconstructs the weight IN THE GRAPH: nibble unpack, a
    16-entry E2M1 table gather, and a multiply by the per-group
    power-of-two scale;
  * metaljax's quantized-matmul recognizer (`metaljax.qmm`) matches that
    chain, verifies the values really are on the E2M1 grid with
    power-of-two group scales, repacks once into MLX's layout and replaces
    the whole chain with `mx.quantized_matmul(mode="mxfp4")`.

Nothing here is metaljax-specific: on any other backend the chain simply
executes, and the row stays correct (just slow, and without the fused
kernel). The weights are still stored packed either way.

WHY THE EINSUMS CHANGED. The stock layer transposes each dequantized weight
to `[experts, in, out]` and contracts `ehm`/`emh`. Keeping the checkpoint's
own `[experts, out, in]` layout and contracting the LAST axis instead makes
the quantization groups contiguous along the recognizer's canonical
contraction axis, so the pack is a reinterpretation with no data movement at
all. The two forms compute the same thing.

Peak load memory stays bounded: the expert tensors are streamed one at a
time straight into their (uint8) variables, and the full bfloat16 weight set
is never materialized -- on the host or on the device.

Enabled for the `gpt-oss-20b` row through the manifest's `native_quant`
field; METALJAX_BENCH_NATIVE_QUANT=0 forces the old dequantize-at-load path
for A/B comparison.
"""

import os
import re

import numpy as np

# The OCP E2M1 grid, indexed by the 4-bit code (bit 3 is the sign). This is
# the same table keras-hub's loader computes arithmetically.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)

GROUP = 32                      # MXFP4's only group size
_EXPERT_KEY = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)"
    r"_(blocks|scales)$")

_installed = False
_orig = {}


def enabled():
    return os.environ.get("METALJAX_BENCH_NATIVE_QUANT", "1") != "0"


# --------------------------------------------------------------------------
# the in-graph dequantization
# --------------------------------------------------------------------------


def e8m0_table():
    """`2**(byte - 127)` for every E8M0 byte, as float32.

    Byte 0 is 2**-127 (an f32 subnormal, which Metal may flush) and byte 255
    is NaN by the OCP spec; neither occurs in a real checkpoint --
    `check_scale_range` fails loudly rather than let one through silently.
    """
    table = np.ldexp(np.ones(256), np.arange(256) - 127)   # f64: no overflow
    table[255] = np.nan
    return table.astype(np.float32)


def check_scale_range(scale_bytes, what=""):
    lo, hi = int(scale_bytes.min()), int(scale_bytes.max())
    if lo < 1 or hi > 254:
        raise ValueError(f"{what}: E8M0 bytes span [{lo}, {hi}]; 0 and 255 "
                         f"are not usable scales")


def mxfp4_weight(codes, scale_bytes, out_dim, k, dtype):
    """`[E, out, k//2]` codes + `[E, out, k//32]` E8M0 bytes -> `[E, out, k]`.

    The low nibble of byte j is element 2j and the high nibble is element
    2j+1 -- the HF `_blocks` convention, which is also the order MLX's
    packed uint32 stream uses, so the recognizer's repack is a
    reinterpretation.
    """
    from keras import ops

    lead = (codes.shape[0], out_dim)
    lo = ops.bitwise_and(codes, np.uint8(0x0F))
    hi = ops.right_shift(codes, np.uint8(4))
    nib = ops.reshape(ops.stack([lo, hi], axis=-1), lead + (k,))
    values = ops.take(ops.cast(ops.convert_to_tensor(E2M1), dtype),
                      ops.cast(nib, "int32"), axis=0)
    scale = ops.take(ops.convert_to_tensor(e8m0_table()),
                     ops.cast(scale_bytes, "int32"), axis=0)
    w = (ops.reshape(values, lead + (k // GROUP, GROUP))
         * ops.expand_dims(ops.cast(scale, dtype), -1))
    return ops.reshape(w, lead + (k,))


# --------------------------------------------------------------------------
# the patched keras-hub layer
# --------------------------------------------------------------------------


def _build(self, _):
    """Packed replacements for `gate_up_proj` / `down_proj`.

    Layouts are the checkpoint's own: `[experts, out, in]`, with `in`
    packed two nibbles per byte and one E8M0 byte per 32 inputs.
    """
    e, h, i = self.num_experts, self.hidden_dim, self.intermediate_dim
    for name, out, k in (("gate_up", 2 * i, h), ("down", h, i)):
        if k % (2 * GROUP):
            raise ValueError(f"{name}: contraction {k} is not a multiple "
                             f"of {2 * GROUP}")
        setattr(self, f"{name}_codes", self.add_weight(
            shape=(e, out, k // 2), initializer="zeros", dtype="uint8",
            trainable=False, name=f"{name}_codes"))
        setattr(self, f"{name}_scale_bytes", self.add_weight(
            shape=(e, out, k // GROUP), initializer="zeros", dtype="uint8",
            trainable=False, name=f"{name}_scale_bytes"))
    self.gate_up_proj_bias = self.add_weight(
        shape=(e, 2 * i), initializer="zeros", name="gate_up_proj_bias")
    self.down_proj_bias = self.add_weight(
        shape=(e, h), initializer="zeros", name="down_proj_bias")
    self.built = True


def _call(self, hidden_states):
    """The stock GptOssExperts.call with the two weights reconstructed.

    Everything between the projections is copied verbatim from keras-hub
    3.15.1 (`_check_stock_layer` fails the run if that body moves).
    """
    from keras import ops

    dt = self.compute_dtype
    gate_up_w = mxfp4_weight(self.gate_up_codes, self.gate_up_scale_bytes,
                             2 * self.intermediate_dim, self.hidden_dim, dt)
    # `th,emh->etm` == the stock `th,ehm->etm` on the transposed weight.
    gate_up = ops.einsum("th,emh->etm", hidden_states, gate_up_w)
    gate_up = gate_up + self.gate_up_proj_bias[:, None, :]

    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]

    gate = ops.clip(gate, -1e9, self.limit)
    up = ops.clip(up, -self.limit, self.limit)

    glu = gate * ops.sigmoid(gate * self.alpha)
    gated_output = (up + 1) * glu

    down_w = mxfp4_weight(self.down_codes, self.down_scale_bytes,
                          self.hidden_dim, self.intermediate_dim, dt)
    out = ops.einsum("etm,ehm->eth", gated_output, down_w)
    out = out + self.down_proj_bias[:, None, :]
    return out


_STOCK_MARKERS = ('ops.einsum("th,ehm->etm"', 'ops.einsum("etm,emh->eth"',
                  "gate_up[..., ::2]", "ops.sigmoid(gate * self.alpha)")


def _check_stock_layer(cls):
    """Refuse to patch a layer whose body is not the one we mirrored."""
    import inspect
    src = inspect.getsource(cls.call)
    missing = [m for m in _STOCK_MARKERS if m not in src]
    if missing:
        raise RuntimeError(
            "keras-hub's GptOssExperts.call has changed; mxfp4_gpt_oss._call "
            f"mirrors it and must be updated (missing: {missing})")


# --------------------------------------------------------------------------
# the patched checkpoint conversion
# --------------------------------------------------------------------------


class _Sink:
    """Absorbs the stock converter's dequantized `assign`.

    The stock `convert_weights` is still what ports the ~40 non-expert
    tensors per layer; only its four expert assignments are redirected here,
    and they are handed a single output row (see `_get_tensor`) so the
    dequantization it insists on doing costs nothing.
    """

    __slots__ = ("assigned",)

    def __init__(self):
        self.assigned = 0

    def assign(self, value):
        self.assigned += 1


def _convert_weights(backbone, loader, transformers_config):
    layers = backbone.transformer_layers
    seen = set()
    orig_get_tensor = loader.get_tensor

    def _get_tensor(key):
        m = _EXPERT_KEY.match(key)
        if m is None:
            return orig_get_tensor(key)
        idx, which, kind = int(m.group(1)), m.group(2), m.group(3)
        tensor = orig_get_tensor(key)
        experts = layers[idx].sparse_moe_block.experts
        stem = "gate_up" if which == "gate_up_proj" else "down"
        var = getattr(experts, stem + ("_codes" if kind == "blocks"
                                       else "_scale_bytes"))
        # blocks arrive as [E, out, k/64, 16] bytes and scales as
        # [E, out, k/32]; both flatten to [E, out, -1] with no copy.
        flat = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        if tuple(flat.shape) != tuple(var.shape):
            raise ValueError(f"{key}: checkpoint gives {flat.shape}, layer "
                             f"wants {tuple(var.shape)}")
        if kind == "scales":
            check_scale_range(flat, key)
        var.assign(flat)
        seen.add((idx, which, kind))
        # One output row is enough for the stock converter to dequantize and
        # hand to a sink: [E, 1, ...] costs ~5 KB instead of ~2 GB.
        return tensor[:, :1]

    loader.get_tensor = _get_tensor
    for layer in layers:
        experts = layer.sparse_moe_block.experts
        experts.gate_up_proj = _Sink()
        experts.down_proj = _Sink()
    try:
        _orig["convert_weights"](backbone, loader, transformers_config)
    finally:
        loader.get_tensor = orig_get_tensor

    want = {(i, w, k) for i in range(len(layers))
            for w in ("gate_up_proj", "down_proj") for k in ("blocks",
                                                             "scales")}
    if seen != want:
        raise RuntimeError("native-MXFP4 loading missed "
                           f"{sorted(want - seen)}: the checkpoint is not "
                           "the MXFP4 gpt-oss repo, or its tensor names "
                           "changed")
    return backbone


# --------------------------------------------------------------------------
# install / uninstall
# --------------------------------------------------------------------------


def install():
    """Patch keras-hub in place. Idempotent; returns True if now active."""
    global _installed
    if _installed:
        return True
    if not enabled():
        return False
    from keras_hub.src.models.gpt_oss import gpt_oss_decoder
    from keras_hub.src.utils.transformers import convert_gpt_oss

    _check_stock_layer(gpt_oss_decoder.GptOssExperts)
    _orig["build"] = gpt_oss_decoder.GptOssExperts.build
    _orig["call"] = gpt_oss_decoder.GptOssExperts.call
    _orig["convert_weights"] = convert_gpt_oss.convert_weights
    gpt_oss_decoder.GptOssExperts.build = _build
    gpt_oss_decoder.GptOssExperts.call = _call
    convert_gpt_oss.convert_weights = _convert_weights
    _installed = True
    return True


def uninstall():
    global _installed
    if not _installed:
        return
    from keras_hub.src.models.gpt_oss import gpt_oss_decoder
    from keras_hub.src.utils.transformers import convert_gpt_oss

    gpt_oss_decoder.GptOssExperts.build = _orig["build"]
    gpt_oss_decoder.GptOssExperts.call = _orig["call"]
    convert_gpt_oss.convert_weights = _orig["convert_weights"]
    _installed = False


# --------------------------------------------------------------------------
# host-side reference (used by the checkpoint spot check)
# --------------------------------------------------------------------------


def np_dequant(blocks, scale_bytes):
    """The numpy MXFP4 dequantization, on `[..., k/2]` / `[..., k/32]`."""
    lo = np.bitwise_and(blocks, 0x0F)
    hi = np.right_shift(blocks, 4)
    nib = np.stack([lo, hi], axis=-1).reshape(blocks.shape[:-1] + (-1,))
    values = E2M1[nib]
    k = values.shape[-1]
    grouped = values.reshape(values.shape[:-1] + (k // GROUP, GROUP))
    scale = np.exp2(scale_bytes.astype(np.float32) - 127.0)
    return (grouped * scale[..., None]).reshape(values.shape)
