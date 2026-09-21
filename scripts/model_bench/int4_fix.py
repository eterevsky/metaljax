"""Row-13 harness variants for the keras int4 route (`keras_lm_quant`).

Two independent, opt-out monkeypatches installed by `run_bench.py` on the
int4 route only.  Both live here, never in the venv's keras/keras-hub.

1. `install_ffn_layer_call()` -- METALJAX_BENCH_INT4_FIX (default 1)

   keras-hub 0.30.0's `Gemma4TextDecoderBlock.call` does not call its three
   FFN `EinsumDense` layers; it reads `layer.kernel` and multiplies by hand:

       # HOTFIX: Replacing EinsumDense with direct matmul to bypass graph
       # tracer bugs.
       x1 = ops.matmul(normalized_x, self.gating_ffw.kernel)
       x2 = ops.matmul(normalized_x, self.gating_ffw_2.kernel)
       x = keras.activations.gelu(x1, approximate=True) * x2
       x = ops.matmul(x, self.ffw_linear.kernel)

   On an int4-quantized `EinsumDense` the `kernel` property
   (keras/src/layers/core/einsum_dense.py, int4 branch) returns
   `reshape(unpack_int4(self._kernel))`: the raw signed nibble CODES in
   [-8, 7], with no `kernel_scale` and no `kernel_zero` -- the
   dequantization lives only in `_int4_call`, which this spelling never
   reaches.  So every FFN of the 35-layer model multiplies activations by
   integer codes: the model decodes garbage on EVERY backend (token 5182
   repeated), and the 105 dots/token are not quantized matmuls in the graph
   either, so metaljax's qmm recognizer correctly declines them and the
   unpack chain runs literally (~14 GB/token).

   The variant replaces those statements -- and the identical pair in the
   MoE twin -- with a call to `_ffw_dense()` below, which CALLS the three
   layers (`self.gating_ffw(normalized_x)`, ...) when they are int4
   quantized, and otherwise runs the original `.kernel` matmul spelling
   verbatim.  The float path (row 4, gemma4-e2b-bf16) is therefore
   bit-unchanged: for an unquantized `EinsumDense`, `.kernel` is the real
   float kernel and the original statements execute.

   The rewrite is done on `call`'s own source so that everything else in
   that 120-line method stays byte-identical; the spans it replaces are
   matched statement by statement and the install fails loudly if
   keras-hub's source ever drifts.

2. `install_embedding_gather_first()` -- METALJAX_BENCH_INT4_EMB (default 1)

   keras's int4 `Embedding._int4_call` unpacks the WHOLE packed table and
   then gathers the rows it needs:

       unpacked = unpack_int4(self._embeddings, orig_output_dim, axis=-1)
       outputs = ops.take(unpacked, inputs, axis=0)

   For gemma4-E2B that is the 1.17 GB per-layer-embedding table plus the
   0.20 GB token table, expanded in full for ONE row, every token
   (measured: 28-37 ms/token of the row's 69.4).  `unpack_int4(.., axis=-1)`
   is elementwise plus transposes/reshapes/slices along the PACKED axis
   only, so it commutes with a gather along axis 0:

       take(unpack(E), i, axis=0) == unpack(take(E, i, axis=0), axis=-1)

   bit-identically (same codes, same scales, same rows -- the scale/zero
   gathers are unchanged).  The variant gathers the packed rows first and
   unpacks only those.  Everything after the unpack is the stock body.
"""

import inspect
import os
import textwrap

_FFN_INSTALLED = False
_EMB_INSTALLED = False


def _flag(name, default="1"):
    return os.environ.get(name, default) == "1"


# --------------------------------------------------------------- 1. the FFN

def _ffw_dense(block, normalized_x):
    """gate/up/down FFN of one Gemma4 decoder block.

    int4-quantized layers: CALL the layers (keras's `_int4_call`, the
    dequantizing path q/k/v/o already use).  Anything else: the original
    keras-hub HOTFIX statements, verbatim.
    """
    import keras
    from keras import ops

    layers = (block.gating_ffw, block.gating_ffw_2, block.ffw_linear)
    if all(getattr(lyr, "quantization_mode", None) == "int4"
           for lyr in layers):
        x1 = block.gating_ffw(normalized_x)
        x2 = block.gating_ffw_2(normalized_x)
        x = keras.activations.gelu(x1, approximate=True) * x2
        return block.ffw_linear(x)

    x1 = ops.matmul(normalized_x, block.gating_ffw.kernel)
    x2 = ops.matmul(normalized_x, block.gating_ffw_2.kernel)
    x = keras.activations.gelu(x1, approximate=True) * x2
    return ops.matmul(x, block.ffw_linear.kernel)


_HEAD = "x1 = ops.matmul(normalized_x, self.gating_ffw.kernel)"
_MID = ("x2 = ops.matmul(normalized_x, self.gating_ffw_2.kernel)",
        "keras.activations.gelu(x1, approximate=True) * x2")
_TAIL = "ops.matmul({v}, self.ffw_linear.kernel)"


def _rewrite_call_source(src):
    """Replace each `.kernel`-matmul FFN span with a `_ffw_dense` call."""
    lines = src.split("\n")
    out, i, spans = [], 0, 0
    while i < len(lines):
        if lines[i].strip() != _HEAD:
            out.append(lines[i])
            i += 1
            continue
        # A span is exactly: x1=, x2=, (blank), <v> = gelu(x1)*x2,
        # <v> = matmul(<v>, ffw_linear.kernel).  Anything else -> refuse.
        body = [ln for ln in lines[i:i + 6] if ln.strip()]
        if (len(body) < 4
                or body[1].strip() != _MID[0]
                or _MID[1] not in body[2]
                or "=" not in body[3]):
            raise RuntimeError(
                "int4_fix: unexpected FFN span in Gemma4TextDecoderBlock."
                f"call:\n{chr(10).join(lines[i:i + 6])}")
        var = body[3].split("=")[0].strip()
        if body[3].strip() != f"{var} = " + _TAIL.format(v=var):
            raise RuntimeError(
                f"int4_fix: unexpected FFN tail statement: {body[3]!r}")
        if body[2].split("=")[0].strip() != var:
            raise RuntimeError(
                f"int4_fix: unexpected FFN gelu statement: {body[2]!r}")
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        out.append(f"{indent}{var} = _MJ_FFW_DENSE(self, normalized_x)")
        i += lines[i:i + 6].index(body[3]) + 1
        spans += 1
    if spans != 2:
        raise RuntimeError(f"int4_fix: expected 2 FFN spans, found {spans}")
    return "\n".join(out)


def install_ffn_layer_call():
    """Patch Gemma4TextDecoderBlock.call.  Returns True if installed."""
    global _FFN_INSTALLED
    if not _flag("METALJAX_BENCH_INT4_FIX"):
        print("[bench] int4 FFN fix: OFF (METALJAX_BENCH_INT4_FIX=0) -- "
              "the keras-hub HOTFIX multiplies by unscaled int4 codes",
              flush=True)
        return False
    from keras_hub.src.models.gemma4 import gemma4_decoder_block as mod

    cls = mod.Gemma4TextDecoderBlock
    src = textwrap.dedent(inspect.getsource(cls.call))
    ns = dict(mod.__dict__)
    ns["_MJ_FFW_DENSE"] = _ffw_dense
    exec(compile(_rewrite_call_source(src),
                 f"<int4_fix:{mod.__file__}>", "exec"), ns)
    cls.call = ns["call"]
    _FFN_INSTALLED = True
    print("[bench] int4 FFN fix: ON -- Gemma4TextDecoderBlock.call now "
          "CALLS gating_ffw/gating_ffw_2/ffw_linear on int4 layers "
          "(keras-hub HOTFIX bypassed the scale)", flush=True)
    return True


# --------------------------------------------------------- 2. the embedding

def install_embedding_gather_first():
    """Patch keras Embedding._int4_call.  Returns True if installed."""
    global _EMB_INSTALLED
    if not _flag("METALJAX_BENCH_INT4_EMB"):
        print("[bench] int4 embedding fix: OFF "
              "(METALJAX_BENCH_INT4_EMB=0) -- whole-table unpack per token",
              flush=True)
        return False
    from keras.src.layers.core import embedding as mod

    ops, quantizers, backend = mod.ops, mod.quantizers, mod.backend
    dequantize_with_sz_map = mod.dequantize_with_sz_map
    stock = mod.Embedding._int4_call

    def _int4_call(self, inputs, training=None):
        """Stock `_int4_call` with the gather sunk under the unpack."""
        if backend.standardize_dtype(inputs.dtype) not in ("int32", "int64"):
            inputs = ops.cast(inputs, "int32")

        # THE VARIANT: gather the PACKED rows, then unpack just those.
        # unpack_int4(.., axis=-1) never mixes rows, so this is bitwise
        # identical to unpacking the whole table and gathering after.
        packed_rows = ops.take(self._embeddings, inputs, axis=0)
        outputs = quantizers.unpack_int4(
            packed_rows, self._orig_output_dim, axis=-1
        )

        block_size = getattr(self, "_int4_block_size", None)

        if block_size is None or block_size == -1:
            embeddings_scale = ops.take(self.embeddings_scale, inputs, axis=0)
            outputs = ops.divide(
                ops.cast(outputs, dtype=self.compute_dtype),
                ops.expand_dims(embeddings_scale, axis=-1),
            )
        else:
            embeddings_scale = ops.take(self.embeddings_scale, inputs, axis=0)
            embeddings_zero = ops.take(self.embeddings_zero, inputs, axis=0)
            outputs = dequantize_with_sz_map(
                ops.cast(outputs, dtype=self.compute_dtype),
                embeddings_scale,
                embeddings_zero,
                self.g_idx,
                group_axis=-1,
            )

        if self.lora_enabled:
            lora_outputs = ops.take(self.lora_embeddings_a, inputs, axis=0)
            lora_outputs = ops.matmul(lora_outputs, self.lora_embeddings_b)
            outputs = ops.add(
                outputs, (self.lora_alpha / self.lora_rank) * lora_outputs
            )
            outputs = ops.cast(outputs, dtype=self.compute_dtype)
        return outputs

    _int4_call._mj_stock = stock
    mod.Embedding._int4_call = _int4_call
    _EMB_INSTALLED = True
    print("[bench] int4 embedding fix: ON -- Embedding._int4_call gathers "
          "the packed rows before unpacking (token table + the per-layer "
          "embedding table)", flush=True)
    return True


# ------------------------------------------------------------------- driver

def install(quant):
    """Install the variants that apply to `quant`; return the record tag."""
    if quant != "int4":
        return "original"
    parts = []
    if install_ffn_layer_call():
        parts.append("int4_ffn_layer_call")
    if install_embedding_gather_first():
        parts.append("int4_emb_gather_first")
    return "+".join(parts) if parts else "original"
