# Gemma4: the FFN "HOTFIX" matmul bypasses int4/int8 dequantization — `quantize("int4")` produces a numerically broken model

**Repo:** keras-team/keras-hub
**Versions:** keras-hub 0.30.0, keras 3.15.1, backend JAX (reproduces on any
backend — the bug is in the graph, not the backend).
**Model:** `google/gemma-4-E2B-it` via `Gemma4CausalLM.from_preset("hf://google/gemma-4-E2B-it")`,
but the code path is architectural and affects every Gemma4 text model.

## Summary

`Gemma4TextDecoderBlock.call` does not call its three feed-forward
`EinsumDense` layers; it reads `layer.kernel` and multiplies by hand:

`keras_hub/src/models/gemma4/gemma4_decoder_block.py`, dense path (L552-559)
and the MoE dense twin (L525-531):

```python
# HOTFIX: Replacing EinsumDense with direct matmul to bypass graph tracer bugs.
normalized_x = self.pre_ffw_norm(x)
x1 = ops.matmul(normalized_x, self.gating_ffw.kernel)
x2 = ops.matmul(normalized_x, self.gating_ffw_2.kernel)
x = keras.activations.gelu(x1, approximate=True) * x2
x = ops.matmul(x, self.ffw_linear.kernel)
```

After `model.quantize("int4")`, `EinsumDense.kernel`
(`keras/src/layers/core/einsum_dense.py`, the `is_int4` branch of the
`kernel` property) returns

```python
kernel = quantizers.unpack_int4(self._kernel, self._int4_unpacked_column_size, axis=-1)
kernel = ops.reshape(kernel, self.original_kernel_shape)
```

i.e. **the raw signed 4-bit codes in [-8, 7]** — `kernel_scale`,
`kernel_zero` and `g_idx` are applied only inside `_int4_call`, which this
spelling never reaches. So every FFN of every layer multiplies activations
by integer codes instead of `(q - z) * s`, and the residual stream is wrong
from layer 0 on. The attention projections (q/k/v/o) and the per-layer
gate/up projections DO call their layers, so they are correct — only the FFN
is bypassed, which is enough to destroy the output.

`int8` is affected the same way (`kernel` returns `self._kernel`, the int8
codes, with `kernel_scale` unapplied); `gptq`/`awq` likewise return
unpacked codes for the 4-bit cases. Unquantized models are unaffected —
there `.kernel` is the real float kernel and the matmul is equivalent to the
layer call (modulo bias, which Gemma4's FFN layers do not use).

## Repro

```python
import numpy as np, keras
from keras import ops
keras.config.set_dtype_policy("bfloat16")

x = np.random.default_rng(0).standard_normal((1, 1, 1536)).astype("float32") * 0.5
w = np.random.default_rng(1).standard_normal((1536, 6144)).astype("float32") * 0.02

layer = keras.layers.EinsumDense("btd,dh->bth", output_shape=(None, 6144))
layer.build((1, 1, 1536)); layer.kernel.assign(w)
ref = np.asarray(ops.convert_to_numpy(layer(ops.array(x, "bfloat16"))), "float32")

layer.quantize("int4")
via_call   = np.asarray(ops.convert_to_numpy(layer(ops.array(x, "bfloat16"))), "float32")
via_kernel = np.asarray(ops.convert_to_numpy(                       # keras-hub's spelling
    ops.matmul(ops.array(x, "bfloat16"), ops.cast(layer.kernel, "bfloat16"))), "float32")

rel = lambda y: np.linalg.norm(y - ref) / np.linalg.norm(ref)
print(rel(via_call), rel(via_kernel))     # 0.0998   152.7
print(np.abs(ref).mean(), np.abs(via_kernel).mean())   # 0.308   47.0
```

The layer call is ~10 % off the float reference (ordinary 4-bit block
quantization error on random weights; 6.4 % on the real checkpoint); the
`.kernel` matmul is 150-270x off, and its outputs are ~150x too large.

End to end:

```python
lm = keras_hub.models.Gemma4CausalLM.from_preset("hf://google/gemma-4-E2B-it")
lm.quantize("int4"); lm.compile(sampler="greedy")
print(lm.generate("<any prompt>", max_length=len(prompt_ids) + 16))
```

generates a single token repeated for the whole continuation (token id 5182,
`' క'`, on our prompt). The same model in bfloat16 answers the prompt
normally, and the same checkpoint quantized to affine 4-bit by another
toolchain answers normally too, so this is not 4-bit quality loss — the
scales are simply not applied.

## Second effect: performance

Because the dequantization is missing, the FFN's right-hand side is a bare
`unpack_int4(...) -> convert` chain rather than the `multiply(subtract(codes,
zero), scale)` pattern, so no backend can recognize it as a quantized
matmul. On our stack the 105 FFN dots per token (35 layers x gate/up/down)
materialize full bfloat16 weights every token: ~14 GB of extra traffic and
~30 ms of the 69 ms/token. Restoring the layer call makes them ordinary
quantized matmuls.

## Suggested fix

Call the layers (the MoE twin at L525-531 needs the same change):

```python
normalized_x = self.pre_ffw_norm(x)
x1 = self.gating_ffw(normalized_x)
x2 = self.gating_ffw_2(normalized_x)
x = keras.activations.gelu(x1, approximate=True) * x2
x = self.ffw_linear(x)
```

We have been running exactly this (installed as a monkeypatch over
`Gemma4TextDecoderBlock.call`, with the `.kernel` spelling kept for
unquantized layers) on both the JAX-CPU backend and our own Metal PJRT
backend: it traces and jit-compiles on both, with no tracer error, and it
restores sane generation. If the original "graph tracer bug" is still real
on some backend/version, it would be better to guard the workaround on
`quantization_mode is None` than to silently drop the scales — or to make
`EinsumDense.kernel` return a dequantized kernel for quantized layers, which
would fix every consumer of that property at once.

## Notes

* `grep -rn HOTFIX keras_hub/src/models/` finds this file only, so gemma4 is
  the only architecture with the bypass.
* The same class of bug exists wherever library code reads `.kernel` off a
  quantized layer instead of calling it; the `kernel` property returning raw
  codes (rather than raising, or dequantizing) is what makes it silent.
