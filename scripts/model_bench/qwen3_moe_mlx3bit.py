"""Load an mlx-community 3-bit Qwen3-MoE checkpoint into keras-hub, PACKED.

This is row 20 (`aspirational-235b-3bit`): Qwen3-235B-A22B-Instruct-2507 at 3
bits, 102.9 GB on disk.  It is the tightest row we have -- the weights alone
are ~96 GiB of a 128 GiB machine -- so the whole design is about never holding
a second copy of anything.

The shape of it, and why:

* **The checkpoint is MLX-native affine.**  `config.json` declares
  `{"group_size": 64, "bits": 3}`; every linear ships `<name>.weight` as uint32
  words, plus bf16 `<name>.scales` and `<name>.biases`, and dequantizes as
  `w = scales * q + biases` with an unsigned code `q` in [0, 7].  Note that is
  an ARBITRARY float bias, not an integer zero point: `scale*(q - zero)` cannot
  express it (matching slopes forces `zero = -bias/scale`, which is not an
  integer, and rounding it costs up to half a quantization step).  The engine
  learned this form for row 20; see metal_qmm.cc's `add_bias`.

* **The packing is a bit STREAM, not a word-at-a-time affair.**  At 3 bits an
  element straddles uint32 boundaries -- 32 codes fill exactly 3 words -- so
  `unpack_codes` stitches one code in three from two words.  This is the exact
  inverse of what `mx.quantize` writes, which is what lets the engine repack it
  to bytes identical to the file and then ALIAS the argument instead of
  allocating a second 96 GB.

* **Nothing is ever dequantized on the host.**  The packed bytes go straight
  into uint32/bf16 variables and the reconstruction is graph code, exactly as
  row 7's MXFP4 loader does.  What the recognizer sees, it verifies and
  deletes; what it cannot verify still runs, literally and correctly.

* **The einsums are flipped.**  keras-hub stores expert weights `[E, in, out]`
  and contracts the MIDDLE axis; the checkpoint stores `[E, out, in]` with its
  quantization groups along the last.  Groups must lie along the contraction
  axis for the recognizer to fuse the dot, so the weight keeps the file's
  layout and the equation moves instead.  Same reason the attention
  projections become plain `[out, in]` matmuls with a reshape after.

Everything here is a monkey-patch installed before `from_preset`, and
`_check_stock` refuses to patch a keras-hub whose bodies have drifted.
"""

import os
import re
import sys

import numpy as np

GROUP = 64
BITS = 3
_WORDS_PER_32 = BITS  # 32 codes -> 3 uint32 words

_orig = {}
_installed = False
_active = False

# Version tripwire: substrings that must appear in the bodies we replace or
# depend on.  Row 7's `_check_stock_layer`, same idea -- copying a body you did
# not write is only safe if you notice when it changes.
_STOCK_MARKERS = {
    "experts_call": ('"th,ehm->etm"', '"eti,eih->eth"', "ops.split"),
    "attention_build": ('"bqm,muh->bquh"', '"bkm,mvh->bkvh"',
                        '"bquh,uhm->bqm"', "rotary_embedding_layer"),
    "sparse_build": ("sparse_feedforward_gate_dense", "expert_bank"),
}


def enabled():
    return os.environ.get("METALJAX_BENCH_NATIVE_QUANT", "1") == "1"


def _check_stock(fn, key):
    import inspect
    src = inspect.getsource(fn)
    missing = [m for m in _STOCK_MARKERS[key] if m not in src]
    if missing:
        raise RuntimeError(
            f"keras-hub's {key} has drifted; missing {missing}. "
            "Re-derive the patch in qwen3_moe_mlx3bit.py before trusting it.")


# --------------------------------------------------------------------------
# the in-graph reconstruction
# --------------------------------------------------------------------------

def unpack_codes(ops, words, k):
    """uint32 words `[..., k*3/32]` -> unsigned codes `[..., k]`.

    Element i occupies bits [3i, 3i+3) of the row's flat little-endian stream.
    Because 3 does not divide 32, one code in three spans two words and has to
    be stitched; the loop below is the exact inverse of MLX's packing.
    """
    lead = tuple(words.shape[:-1])
    w = ops.reshape(words, lead + (k // 32, _WORDS_PER_32))
    outs = []
    for i in range(32):
        bit = i * BITS
        w0, off = bit // 32, bit % 32
        v = ops.right_shift(w[..., w0], np.uint32(off))
        if off + BITS > 32:
            v = ops.bitwise_or(
                v, ops.left_shift(w[..., w0 + 1], np.uint32(32 - off)))
        outs.append(ops.bitwise_and(v, np.uint32((1 << BITS) - 1)))
    return ops.reshape(ops.stack(outs, axis=-1), lead + (k,))


def dequant(ops, words, scales, biases, k, dtype):
    """`scales * q + biases`, at FULL contraction width.

    The maps are broadcast out to `[..., k]` rather than left on a grouped
    `[..., k/64, 64]` axis on purpose: the recognizer reads its scale and bias
    maps along the contraction axis, and handing it the weight's real shape is
    what lets it MEASURE the group size rather than be told.  It also keeps the
    maps bit-identical to the file, so they alias too.
    """
    q = ops.cast(unpack_codes(ops, words, k), dtype)

    def full(m):
        m = ops.cast(m, dtype)
        return ops.reshape(
            ops.broadcast_to(ops.expand_dims(m, -1),
                             tuple(m.shape) + (GROUP,)),
            tuple(m.shape[:-1]) + (k,))

    return q * full(scales) + full(biases)


# --------------------------------------------------------------------------
# a packed [out, in] linear
# --------------------------------------------------------------------------

def _make_packed(layer, stem, out_features, in_features):
    """Create the (codes, scales, biases) triple for one `[out, in]` weight."""
    if in_features % 32:
        raise ValueError(f"{stem}: contraction {in_features} is not a "
                         "multiple of 32 (the 3-bit packing unit)")
    if in_features % GROUP:
        raise ValueError(f"{stem}: contraction {in_features} is not a "
                         f"multiple of the group size {GROUP}")
    lead = () if isinstance(out_features, int) else tuple(out_features[:-1])
    out = out_features if isinstance(out_features, int) else out_features[-1]
    mk = lambda name, tail, dt: layer.add_weight(  # noqa: E731
        shape=lead + (out,) + tail, initializer="zeros", dtype=dt,
        trainable=False, name=name)
    return (mk(f"{stem}_codes", (in_features * BITS // 32,), "uint32"),
            mk(f"{stem}_scales", (in_features // GROUP,), "bfloat16"),
            mk(f"{stem}_biases", (in_features // GROUP,), "bfloat16"))


def _packed_linear_class():
    import keras
    from keras import ops

    class PackedLinear(keras.layers.Layer):
        """`y = x @ dequant(codes, scales, biases).T`, reshaped.

        The weight keeps the checkpoint's `[out, in]` layout so its groups run
        along the contraction axis; the output is reshaped afterwards, which is
        free and is what lets one packed form stand in for keras' EinsumDense
        in all four attention projections and the router.
        """

        def __init__(self, out_features, in_features, out_shape=None,
                     contract_rank=1, **kw):
            super().__init__(**kw)
            self.out_features = out_features
            self.in_features = in_features
            self.out_shape = tuple(out_shape) if out_shape else (out_features,)
            self.contract_rank = contract_rank

        def build(self, input_shape):
            self.codes, self.scales, self.biases = _make_packed(
                self, "w", self.out_features, self.in_features)
            self.built = True

        def call(self, x):
            lead = tuple(ops.shape(x))[:len(x.shape) - self.contract_rank]
            xf = ops.reshape(x, (-1, self.in_features))
            w = dequant(ops, self.codes, self.scales, self.biases,
                        self.in_features, self.compute_dtype)
            y = ops.einsum("ti,oi->to", xf, w)
            return ops.reshape(y, lead + self.out_shape)

        def compute_output_shape(self, input_shape):
            return (tuple(input_shape)[:len(input_shape) - self.contract_rank]
                    + self.out_shape)

    return PackedLinear


# --------------------------------------------------------------------------
# the patched layer bodies
# --------------------------------------------------------------------------

def _experts_build(self, _):
    """Packed expert bank, in the checkpoint's `[E, out, in]` layout.

    gate and up are concatenated along `out` -- which for a packed weight is
    just a row concat, the bit stream runs along `in` -- so one dot serves
    both, exactly as the stock layer intends with its `ops.split`.
    """
    e, h, i = self.num_experts, self.hidden_dim, self.intermediate_dim
    self.gate_up_codes, self.gate_up_scales, self.gate_up_biases = (
        _make_packed(self, "gate_up", (e, 2 * i), h))
    self.down_codes, self.down_scales, self.down_biases = (
        _make_packed(self, "down", (e, h), i))
    self.built = True


def _experts_call(self, hidden_states):
    from keras import ops
    dt = self.compute_dtype
    gate_up = ops.einsum(
        "th,emh->etm", hidden_states,
        dequant(ops, self.gate_up_codes, self.gate_up_scales,
                self.gate_up_biases, self.hidden_dim, dt))
    gate, up = ops.split(gate_up, 2, axis=-1)
    hidden = up * self.activation(gate)
    return ops.einsum(
        "etm,ehm->eth", hidden,
        dequant(ops, self.down_codes, self.down_scales, self.down_biases,
                self.intermediate_dim, dt))


def _attention_build(self, inputs_shape):
    """Stock body with the four EinsumDense projections replaced.

    Copied rather than wrapped: calling the stock build first would create the
    float kernels, and under the streaming loader an unassigned variable is
    randomly initialized at finalize -- 94 layers' worth of q/k/v/o would be
    ~12 GB of random weights we then have to throw away.
    """
    import keras
    import math
    stock = _orig["attention_module"]
    PackedLinear = _orig["PackedLinear"]

    hidden_dim = inputs_shape[-1]
    if not self.head_dim:
        self.head_dim = hidden_dim // self.num_query_heads
    self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
    u, v, hd = self.num_query_heads, self.num_key_value_heads, self.head_dim

    def proj(name, heads, in_features=hidden_dim, contract_rank=1,
             out_shape=None, out_features=None):
        p = PackedLinear(
            out_features=out_features if out_features is not None
            else heads * hd,
            in_features=in_features, out_shape=out_shape,
            contract_rank=contract_rank, dtype=self.dtype_policy, name=name)
        p.build(inputs_shape)
        return p

    self._query_dense = proj("query", u, out_shape=(u, hd))
    self._query_dense_layer_norm = stock.Qwen3MoeLayerNorm(
        epsilon=self.layer_norm_epsilon, dtype=self.dtype_policy,
        head_dim=self.head_dim, name="query_dense_layernorm")
    self._query_dense_layer_norm.build(inputs_shape)

    self._key_dense = proj("key", v, out_shape=(v, hd))
    self._key_dense_layer_norm = stock.Qwen3MoeLayerNorm(
        epsilon=self.layer_norm_epsilon, dtype=self.dtype_policy,
        head_dim=self.head_dim, name="key_dense_layernorm")
    self._key_dense_layer_norm.build(inputs_shape)

    self._value_dense = proj("value", v, out_shape=(v, hd))

    self._softmax = keras.layers.Softmax(axis=-1, dtype="float32",
                                         name="attention_softmax")
    self._dropout_layer = keras.layers.Dropout(rate=self.dropout,
                                               dtype=self.dtype_policy)
    # `bquh,uhm->bqm`: the contraction is the flattened (u, h) pair.
    self._output_dense = proj("attention_output", None, in_features=u * hd,
                              contract_rank=2, out_shape=(hidden_dim,),
                              out_features=hidden_dim)

    self.rotary_embedding_layer = stock.RotaryEmbedding(
        max_wavelength=self.rope_max_wavelength,
        scaling_factor=self.rope_scaling_factor, dtype=self.dtype_policy)
    self._dot_product_equation = "bquh,bkuh->buqk"
    self._combine_equation = "buqk,bkuh->bquh"
    self.built = True


def _sparse_build(self, decoder_sequence_shape):
    """Stock body with the router Dense replaced (same reason as above)."""
    stock = _orig["decoder_module"]
    PackedLinear = _orig["PackedLinear"]
    self._sparse_feedforward_gate_dense = PackedLinear(
        out_features=self.num_experts, in_features=self.hidden_dim,
        dtype=self.dtype_policy, name="sparse_feedforward_gate_dense")
    self._sparse_feedforward_gate_dense.build(decoder_sequence_shape)
    self.expert_bank = stock.Qwen3MoeExperts(
        num_experts=self.num_experts, hidden_dim=self.hidden_dim,
        intermediate_dim=self.intermediate_dim,
        kernel_initializer=self.kernel_initializer, name="experts",
        dtype=self.dtype_policy)
    self.expert_bank.build(decoder_sequence_shape)
    self.built = True


def _embedding_build(self, inputs_shape=None):
    """Packed token embedding, and the untied LM head that reads it back.

    The forward direction is a GATHER, so only the selected rows are ever
    dequantized -- a handful per step, not the 151936-row table.  The reverse
    direction is an ordinary `[vocab, hidden]` matmul and fuses like any other.
    """
    self.embed_codes, self.embed_scales, self.embed_biases = _make_packed(
        self, "embed", self.input_dim, self.output_dim)
    if not self.tie_weights:
        self.head_codes, self.head_scales, self.head_biases = _make_packed(
            self, "head", self.input_dim, self.output_dim)
    self.built = True


def _embedding_call(self, inputs, reverse=False):
    from keras import ops
    dt = self.compute_dtype
    if reverse:
        codes, scales, biases = (self.head_codes, self.head_scales,
                                 self.head_biases) if not self.tie_weights \
            else (self.embed_codes, self.embed_scales, self.embed_biases)
        w = dequant(ops, codes, scales, biases, self.output_dim, dt)
        return ops.einsum("...h,vh->...v", inputs, w)
    idx = ops.cast(inputs, "int32")
    rows = dequant(ops, ops.take(self.embed_codes, idx, axis=0),
                   ops.take(self.embed_scales, idx, axis=0),
                   ops.take(self.embed_biases, idx, axis=0),
                   self.output_dim, dt)
    return rows


# --------------------------------------------------------------------------
# the loader
# --------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _convert_backbone_config(cfg):
    """Stock mapping, plus the two things the MLX repo needs.

    bfloat16 is not cosmetic: the scale and bias maps are bf16 in the file, and
    the engine can only alias them (7 GB apiece at this size) if the graph
    keeps them at that width instead of widening to f32.
    """
    out = _orig["convert_backbone_config"](cfg)
    out["dtype"] = "bfloat16"
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if q.get("bits") != BITS or q.get("group_size") != GROUP:
        raise ValueError(
            f"expected {BITS}-bit / group {GROUP}, got {q!r}; this loader is "
            "written for the mlx-community affine layout only")
    return out


def _assign_packed(triple, tensors, key, expect_words, expect_groups):
    """Assign one (codes, scales, biases) triple straight from the file."""
    codes, scales, biases = triple
    w, s, b = tensors
    if w.dtype != np.uint32:
        raise ValueError(f"{key}: codes are {w.dtype}, expected uint32")
    if tuple(w.shape)[-1] != expect_words:
        raise ValueError(f"{key}: {w.shape[-1]} words, expected {expect_words}")
    if tuple(s.shape)[-1] != expect_groups:
        raise ValueError(f"{key}: {s.shape[-1]} groups, "
                         f"expected {expect_groups}")
    for var, val in ((codes, w), (scales, s), (biases, b)):
        if tuple(var.shape) != tuple(val.shape):
            raise ValueError(f"{key}: {tuple(val.shape)} does not fit "
                             f"{var.path} {tuple(var.shape)}")
        var.assign(val)


def _convert_weights(backbone, loader, transformers_config):
    """A full replacement: the stock converter cannot read this checkpoint.

    keras-hub's `convert_qwen3_moe` reads per-expert keys
    (`...mlp.experts.{j}.gate_proj.weight`) which do not exist here -- MLX
    ships the experts batched as `mlp.switch_mlp.*` -- and every linear in the
    file carries `.scales`/`.biases` siblings it knows nothing about.  Only the
    norm key names survive, so those are all we borrow.
    """
    get = loader.get_tensor
    hidden = backbone.token_embedding.output_dim

    def trio(stem):
        return (get(f"{stem}.weight"), get(f"{stem}.scales"),
                get(f"{stem}.biases"))

    emb = backbone.token_embedding
    _assign_packed((emb.embed_codes, emb.embed_scales, emb.embed_biases),
                   trio("model.embed_tokens"), "model.embed_tokens",
                   hidden * BITS // 32, hidden // GROUP)
    if not emb.tie_weights:
        _assign_packed((emb.head_codes, emb.head_scales, emb.head_biases),
                       trio("lm_head"), "lm_head",
                       hidden * BITS // 32, hidden // GROUP)
    backbone.get_layer("sequence_output_layernorm").scale.assign(
        get("model.norm.weight"))

    n_seen = 0
    for i, layer in enumerate(backbone.transformer_layers):
        p = f"model.layers.{i}"
        attn = layer._self_attention_layer
        layer._self_attention_layernorm.scale.assign(
            get(f"{p}.input_layernorm.weight"))
        layer._feedforward_layernorm.scale.assign(
            get(f"{p}.post_attention_layernorm.weight"))
        attn._query_dense_layer_norm.scale.assign(
            get(f"{p}.self_attn.q_norm.weight"))
        attn._key_dense_layer_norm.scale.assign(
            get(f"{p}.self_attn.k_norm.weight"))

        for name, dense in (("q_proj", attn._query_dense),
                            ("k_proj", attn._key_dense),
                            ("v_proj", attn._value_dense),
                            ("o_proj", attn._output_dense)):
            key = f"{p}.self_attn.{name}"
            _assign_packed((dense.codes, dense.scales, dense.biases),
                           trio(key), key,
                           dense.in_features * BITS // 32,
                           dense.in_features // GROUP)
            n_seen += 1

        router = layer.mlp._sparse_feedforward_gate_dense
        key = f"{p}.mlp.gate"
        _assign_packed((router.codes, router.scales, router.biases),
                       trio(key), key, hidden * BITS // 32, hidden // GROUP)
        n_seen += 1

        ex = layer.mlp.expert_bank
        sm = f"{p}.mlp.switch_mlp"
        # gate and up concatenate along the OUTPUT axis, which for a weight
        # packed along its input axis is a plain row concat -- no re-packing,
        # and it matches the stock layer's `ops.split(..., 2, axis=-1)`.
        gu = [np.concatenate([get(f"{sm}.gate_proj.{k}"),
                              get(f"{sm}.up_proj.{k}")], axis=1)
              for k in ("weight", "scales", "biases")]
        _assign_packed((ex.gate_up_codes, ex.gate_up_scales, ex.gate_up_biases),
                       gu, f"{sm}.gate_up",
                       ex.hidden_dim * BITS // 32, ex.hidden_dim // GROUP)
        _assign_packed((ex.down_codes, ex.down_scales, ex.down_biases),
                       trio(f"{sm}.down_proj"), f"{sm}.down_proj",
                       ex.intermediate_dim * BITS // 32,
                       ex.intermediate_dim // GROUP)
        n_seen += 2
        del gu
    print(f"[mlx3bit] assigned {n_seen} packed weights over "
          f"{len(backbone.transformer_layers)} layers", flush=True)


# --------------------------------------------------------------------------
# install / uninstall
# --------------------------------------------------------------------------

def install():
    global _installed, _active
    if _installed or not enabled():
        return False
    from keras_hub.src.models.qwen3_moe import (qwen3_moe_attention,
                                                qwen3_moe_decoder)
    from keras_hub.src.utils.transformers import convert_qwen3_moe
    # keras 3.15 moved ReversibleEmbedding into core keras; keras-hub re-exports
    # its own for older versions.  Patch the class the backbone actually
    # instantiates, not whichever name happens to import.
    from keras_hub.src.models.qwen3_moe import qwen3_moe_backbone
    reversible_embedding = sys.modules[
        qwen3_moe_backbone.ReversibleEmbedding.__module__]

    _check_stock(qwen3_moe_decoder.Qwen3MoeExperts.call, "experts_call")
    _check_stock(qwen3_moe_attention.Qwen3MoeAttention.build,
                 "attention_build")
    _check_stock(qwen3_moe_decoder.Qwen3SparseMoeBlock.build, "sparse_build")

    _orig.update(
        attention_module=qwen3_moe_attention,
        decoder_module=qwen3_moe_decoder,
        PackedLinear=_packed_linear_class(),
        experts_build=qwen3_moe_decoder.Qwen3MoeExperts.build,
        experts_call=qwen3_moe_decoder.Qwen3MoeExperts.call,
        attention_build=qwen3_moe_attention.Qwen3MoeAttention.build,
        sparse_build=qwen3_moe_decoder.Qwen3SparseMoeBlock.build,
        emb_build=reversible_embedding.ReversibleEmbedding.build,
        emb_call=reversible_embedding.ReversibleEmbedding.call,
        convert_weights=convert_qwen3_moe.convert_weights,
        convert_backbone_config=convert_qwen3_moe.convert_backbone_config,
        reversible_embedding=reversible_embedding,
        convert_qwen3_moe=convert_qwen3_moe,
    )

    qwen3_moe_decoder.Qwen3MoeExperts.build = _experts_build
    qwen3_moe_decoder.Qwen3MoeExperts.call = _experts_call
    qwen3_moe_attention.Qwen3MoeAttention.build = _attention_build
    qwen3_moe_decoder.Qwen3SparseMoeBlock.build = _sparse_build
    reversible_embedding.ReversibleEmbedding.build = _embedding_build
    reversible_embedding.ReversibleEmbedding.call = _embedding_call
    convert_qwen3_moe.convert_weights = _convert_weights
    convert_qwen3_moe.convert_backbone_config = _convert_backbone_config
    _installed = True
    _active = True
    return True


def uninstall():
    global _installed, _active
    if not _installed:
        return
    _orig["decoder_module"].Qwen3MoeExperts.build = _orig["experts_build"]
    _orig["decoder_module"].Qwen3MoeExperts.call = _orig["experts_call"]
    _orig["attention_module"].Qwen3MoeAttention.build = _orig[
        "attention_build"]
    _orig["decoder_module"].Qwen3SparseMoeBlock.build = _orig["sparse_build"]
    _orig["reversible_embedding"].ReversibleEmbedding.build = _orig[
        "emb_build"]
    _orig["reversible_embedding"].ReversibleEmbedding.call = _orig["emb_call"]
    _orig["convert_qwen3_moe"].convert_weights = _orig["convert_weights"]
    _orig["convert_qwen3_moe"].convert_backbone_config = _orig[
        "convert_backbone_config"]
    _installed = False
    _active = False
