# keras-hub extra benchmarks: hosting, auth and environment findings

Companion notes for `adapter_keras_extra.py` (SigLIP 2 vision, Stable
Diffusion 3.5 Large, Gemma 4 E2B LoRA training, plus a cheap converter
smoke used for DeepSeek-R1-Distill-Qwen-32B).  Everything below was
measured on this machine on 2026-08-02 with the benchmark venv
(python 3.13.5, keras 3.15.1, keras-hub 0.30.0, jax/jaxlib 0.11.0,
tensorflow 2.20.0, tensorflow-text 2.20.1, sentencepiece 0.2.1).

`adapter_keras_extra.py` deliberately does not touch `run_bench.py` or
`manifest.json`.  It exposes an `ADAPTERS` dict with the same calling
convention, so wiring it in is one line in `run_bench.py`:

```python
import adapter_keras_extra
ADAPTERS.update(adapter_keras_extra.ADAPTERS)
```

It is also runnable standalone, writing the same jsonl record shape:

```
bench-venv/bin/python scripts/model_bench/adapter_keras_extra.py \
    siglip2-so400m cpu --out results.jsonl
```

---

## 1. Preset hosting / auth

**Every Kaggle-hosted keras preset checked here is anonymously
downloadable.  No Kaggle account, no API token, no consent click, no
`~/.kaggle/kaggle.json`** (there is none on this machine, and
`kagglehub` 1.0.2 never prompted).

Mechanism, verified without downloading weights: the public models API

```
GET https://www.kaggle.com/api/v1/models/<owner>/<model>/<fw>/<variation>/<ver>/files
```

returns `200` with the full file listing unauthenticated, and

```
GET .../download/<file>
```

302s to a pre-signed `storage.googleapis.com/kagglesdsdata/...` URL that
also serves anonymously.  Probing `metadata.json` (160 bytes) is enough
to prove access without pulling the payload — that is how the 93 GB
Mixtral preset was checked.

| preset | handle | total size | anon access |
|---|---|---|---|
| `siglip2_so400m_patch14_384` | `keras/siglip/keras/siglip2_so400m_patch14_384/1` | 4.55 GB (`model.weights.h5` 4.546 GB, f32) | yes — downloaded in full, 136 s |
| `stable_diffusion_3.5_large` | `keras/stablediffusion-3.5/keras/stable_diffusion_3.5_large/3` | 18.10 GB (2 shards: 10.725 + 7.376 GB) | yes — downloaded in full, 323 s |
| `stable_diffusion_3_medium` | `keras/stablediffusion3/keras/stable_diffusion_3_medium/5` | 5.98 GB | yes (metadata probe; not needed, 3.5-large worked) |
| `mixtral_8_instruct_7b_en` | `keras/mixtral/keras/mixtral_8_instruct_7b_en/4` | **93.41 GB** (9 shards, 8.6–10.7 GB each) | yes — **metadata probe only, weights NOT downloaded** |
| `mixtral_8_7b_en` | `keras/mixtral/keras/mixtral_8_7b_en/4` | 93.41 GB | yes (same probe) |

Reproduce the probe with `scripts/model_bench/` + this snippet (no
keras, no downloads beyond a few hundred bytes):

```python
import json, urllib.request
h = "keras/mixtral/keras/mixtral_8_instruct_7b_en/4"
d = json.load(urllib.request.urlopen(
        f"https://www.kaggle.com/api/v1/models/{h}/files"))
print(sum(f["size"] for f in d["files"]) / 1e9, "GB")
```

### `hf://` is NOT an alternative route for SigLIP or SD3

`keras_hub.src.utils.transformers.preset_loader.TransformersPresetLoader`
dispatches on the HF `config.json` `model_type`.  In keras-hub 0.30.0 the
supported set is: albert, bart, bert, blip-2, deit, distilbert, dinov2,
dinov3_vit, esm, gemma/gemma2/gemma3/gemma3_text/gemma3n/gemma4/
gemma4_text/gemma4_assistant, gpt2, gpt_oss, llama, metaclip_2, mistral,
mixtral, paligemma, vit, qwen2, qwen2_moe, qwen3, qwen3_moe, qwen3_5,
qwen3_5_moe, sam3_video, smollm3, t5gemma, t5gemma2, xlm-roberta.

There is **no `siglip` / `siglip2` converter and no stable-diffusion-3
converter**, so `from_preset("hf://google/siglip2-so400m-patch14-384")`
fails with

```
ValueError: KerasHub has no converter for huggingface/transformers models
with model type `'siglip'`.
```

The Kaggle presets are the only keras-hub route for those two, and since
they need no auth that is not a problem.  Gemma 4 and Qwen 2 do have
converters, so `hf://` is used for those rows.

---

## 2. ENVIRONMENT BUG: `import tensorflow` breaks `sentencepiece`

This one silently deletes benchmark rows, so it is worth stating plainly.

**Symptom.** Any keras-hub model with a SentencePiece tokenizer dies at
`from_preset` with

```
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
libc++abi: terminating due to uncaught exception of type
  std::__1::system_error: mutex lock failed: Invalid argument
```

This is a `SIGABRT`, not a python exception — `run_bench.py`'s
`except Exception` never runs, no `RESULT` line is written, and the row
just vanishes from `results.jsonl`.  **This is why the two
`gemma4-e2b-bf16` rows are missing from the 2026-08-02 suite output**
(the block ran qwen3-8b, llama31-8b, gemma4-e2b, gpt-oss-20b × 2
backends and produced 5 rows, not 8).

**Root cause.** `tensorflow` 2.20.0 and `sentencepiece` 0.2.1 ship
colliding copies of the SentencePiece/protobuf C++ symbols; macOS's flat
dyld namespace merges them.  Minimal repro (nothing keras involved):

```python
import tensorflow                      # <- poisons it
import sentencepiece
sp = sentencepiece.SentencePieceProcessor()
sp.Init(model_proto=open("vocabulary.spm", "rb").read())   # SIGABRT
```

Measured variants:

| order | result |
|---|---|
| sentencepiece only | works (`vocab 256000`) |
| `import tensorflow` then spm `Init` | **abort** |
| `import tensorflow_text` then spm `Init` | **abort** |
| `import sentencepiece` (module only), then tf_text, then `Init` | **abort** |
| spm `Init` first, **then** `import tensorflow_text` | **hangs forever** (process wedged 30+ min, had to be `kill -9`ed) |

So blocking `tensorflow_text` does not help — plain `tensorflow` is
already fatal, and keras-hub imports it unconditionally for its tf.data
preprocessing.

**Why it hits every SentencePiece preset.** keras-hub 0.30.0's
`SentencePieceTokenizer.set_proto` builds the tf-text tokenizer *and
then always* calls `_set_proto_spm` (native sentencepiece), because it
reads the vocabulary metadata from it:

```python
try:
    self._set_proto_tf(proto_bytes)     # tf-text: fine
except ImportError:
    pass
...
self._set_proto_spm(proto_bytes)        # native spm: ABORTS
```

Affected: Gemma (all), SigLIP/SigLIP 2, Mixtral, T5Gemma, PaliGemma —
i.e. `gemma4-e2b-bf16`, `gemma4-26b-a4b`, `lora-gemma4-e2b`,
`siglip2-so400m` and `mixtral-8x7b` in the manifest.  Qwen/Llama/GPT-OSS
are BPE and unaffected.

**Workaround shipped here.** `adapter_keras_extra.patch_sentencepiece_native()`
replaces `_set_proto_spm` with a shim over the tf-text tokenizer that
already exists (`vocab_size()` / `id_to_string()` give everything
`set_proto` needs) and sets `_allow_python_workflow = False` so the
native processor is never reached.  Tokenize, detokenize and tf.data
preprocessing all keep working.  `BENCH_SPM_PATCH=0` opts out.

Verified: `hf://google/gemma-4-E2B-it` now loads and trains on both
backends (§5).  **If run_bench.py adopts this one call, the missing
gemma4 rows come back.**  The real fixes would be to rebuild
sentencepiece with hidden visibility, or for keras-hub to skip
`_set_proto_spm` when tf-text is available.

Separate consequence for SigLIP: its Kaggle preset also ships no
`preprocessor.json`, so `SigLIPPreprocessor.from_preset` cannot be used
either.  `run_keras_vision` therefore tokenizes in a clean child process
(`_spm_tokenize`, a faithful re-implementation of SigLIP's canonicalize
→ encode → truncate → `</s>` pad) and uses
`keras_hub.layers.SigLIPImageConverter` (note: `layers`, not `models`)
for the image half.  Tokenization is outside every timed region.

---

## 3. SigLIP 2 so400m/14 @384 — `run_keras_vision`

Metric: one image+text embedding forward through `SigLIPBackbone`
(`predict_on_batch`, which ends in `np.array(...)` — the only real
barrier on the metal backend), at batch 1 and batch 32.

```
preset  siglip2_so400m_patch14_384      params 1,136,009,291
```

`params` matches the Kaggle metadata exactly, and the trained head is
present (a randomly initialised `SigLIPHead` starts at
`logit_scale = log(1) = 0`, `logit_bias = 0`):

```
head_logit_scale 4.688      head_logit_bias -15.938
```

| backend | load_s | warmup_s (b1) | step_ms b1 | step_ms b32 | img/s b32 |
|---|---|---|---|---|---|
| cpu | 3.89 | 2.34 | **597.3** | **17757.5** | 1.80 |
| metaljax | 10.90 | 2.24 | **91.0** | **2662.3** | 12.02 |

Metal is 6.6x faster at batch 1 and 6.7x at batch 32.  Cross-backend
agreement (bf16 granularity), image/text logit-matrix diagonal:

```
b32 cpu    [-9.5, -10.1875, -12.75, -9.75]
b32 metal  [-9.5, -10.125,  -12.75, -9.75]
b1  cpu -9.5      b1 metal -9.5625
```

All outputs finite; shapes `vision_logits`/`text_logits` = `(B, B)`.

---

## 4. Stable Diffusion 3.5 Large — `run_keras_diffusion`

```
preset  stable_diffusion_3.5_large      params 9,048,410,595
```

Runs on metal.  1024x1024, 20 steps:

| backend | load_s | warmup_s (4 steps) | generate_ms (20 steps) | ms/diffusion step | marginal ms/step |
|---|---|---|---|---|---|
| metaljax | 53.5 | 40.7 | 171543.9 | **8577.2** | 8544.2 |

The marginal figure ((t20 − t4)/16) matches the naive average, i.e. text
encoding and VAE decode are noise next to the MMDiT loop.  At 512x512 /
4 steps the same path gives 5223.6 ms/step.

### Two blockers found

**(a) jax-CPU cannot run this preset at all.**  The preset config bakes
per-submodel dtypes — MMDiT and VAE `bfloat16`, but `clip_l` and
`clip_g` **`float16`** — and jax's CPU backend rejects the resulting
matmul precision:

```
ValueError: The precision 'F16_F16_F32' is not supported by dot_general on CPU
```

So there is no CPU reference row for SD 3.5 Large, and no CPU
cross-check of its output.  (`stable_diffusion_3_medium` was not tried;
it is 5.98 GB and anonymously available if a CPU-runnable fallback is
wanted, but its config would need checking for the same f16 CLIP.)

**(b) the generated image is entirely black on metal.**  Timings are
real (the sampler runs, the wall time scales with `num_steps`), but

```
out_shape [1024,1024,3]  out_dtype uint8
pixel_min 0  pixel_max 0  pixel_mean 0.0  pixel_std 0.0
```

— identical at 512x512/4 steps.  Uniform zero across every pixel and
channel is the signature of non-finite latents being clipped, not of a
merely-bad sample.  With no CPU reference (blocker (a)) this has not
been attributed to metaljax versus the f16 CLIP encoders versus keras'
SD3 sampler.  **Treat `sd35-large` as a performance-only row and do not
publish it as a correctness result until the black image is explained.**
`BENCH_IMAGE_OUT=<path>.png` dumps the image for inspection.

---

## 5. Gemma 4 E2B + LoRA rank 4 — `run_keras_lora_train`

One `.fit()` step, synthetic batch, sequence length 256, batch size 1.
`load_s` is `from_preset`; `compile_s` is the first train batch (trace +
compile); `step_ms` is the median of the warm batches.  Timing uses a
callback that reads `float(logs["loss"])` at batch end — that read is
the host barrier.

Gemma 4 is multimodal: the backbone takes eleven inputs
(`audio_indices, audio_mask, audio_mel, audio_mel_mask, padding_mask,
pixel_position_ids, pixel_values, position_ids, token_ids,
vision_indices, vision_mask`).  The adapter runs the model's own
`Gemma4CausalLMPreprocessor` eagerly on `{"prompts": ..., "responses":
...}`, then detaches it and fits on the resulting arrays, so the timed
step is pure device work.

```
backbone.enable_lora(4)  ->  trainable 1,873,920 / 5,106,172,387 params (0.037 %)
```

| backend | load_s | compile_s | step_ms (median) | step samples (ms) |
|---|---|---|---|---|
| cpu | 57.2 | 9.42 | **3286.6** | 3165.8, 3016.5, 3399.7, 11727.2\*, 4219.7, 3173.5, 3043.3, 3515.4 |
| metaljax | 450.2 | 2.24 | **416.6** | 419.8, 411.9, 415.2, 417.9 |

\* one outlier under concurrent load; the median is unaffected.

Metal is **7.9x** faster per step.  Losses agree across backends, which
is the correctness evidence for the training path:

```
cpu    2.9264  2.7834  2.6649  2.8744  2.9425 ...
metal  2.9171  2.7778  2.6613  2.8702  2.9374
```

Note `load_s = 450 s` on metal versus 57 s on CPU — pushing 5.1 B
parameters through the plugin's host->device path dominates, and is
worth a look if LoRA rows ever go into a regular sweep.

---

## 6. DeepSeek-R1-Distill-Qwen-32B converter smoke — `run_keras_lm_smoke`

### The class mapping is correct

`config.json` of `hf://deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` reports
`"architectures": ["Qwen2ForCausalLM"]`, `"model_type": "qwen2"`,
`hidden_size 5120`, `num_hidden_layers 64`, `num_attention_heads 40`,
`num_key_value_heads 8`, `vocab_size 152064`, `torch_dtype bfloat16`.
`TransformersPresetLoader` maps `qwen2 -> convert_qwen`, whose
`backbone_cls` is `QwenBackbone`, so the task class is
**`keras_hub.models.QwenCausalLM`** — the `arch` already in
`manifest.json` is right, no remapping needed.  `Qwen2CausalLM` /
`Qwen2Backbone` / `Qwen2Tokenizer` exist in the namespace but they are
the *same objects* re-exported under a second name
(`@keras_hub_export([... "QwenTokenizer", ... "Qwen2Tokenizer"])`), so
there is no alternative class to try.

The backbone conversion itself is fine: a full run got all the way
through `QwenBackbone` weight loading (RSS peaked around 94 GB with the
safetensors mmapped, settling near 70 GB) before failing on the
tokenizer.

### The blocker: keras-hub cannot load DeepSeek's tokenizer

```
ValueError: Cannot find special token `'<|endoftext|>'` in the provided
vocabulary for `QwenTokenizer`. Please ensure `'<|endoftext|>'` is in the
provided vocabulary when creating the Tokenizer.
```

`QwenTokenizer.__init__` hardcodes `eos_token = "<|endoftext|>"`.  The
R1 distills keep Qwen 2.5's *architecture* but ship DeepSeek's own
tokenizer: `<|endoftext|>`, `<|im_start|>` and `<|im_end|>` are all
**absent**, and the 22 added tokens start at

```
151643 '<｜end▁of▁sentence｜>'   151644 '<｜User｜>'
151645 '<｜Assistant｜>'          151646 '<｜begin▁of▁sentence｜>'
151647 '<|EOT|>'                151648 '<think>'  151649 '</think>'
```

with `tokenizer_config.json` declaring
`eos_token = pad_token = '<｜end▁of▁sentence｜>'` (full-width bars).

Worse, this fires *after* `load_task` has already materialised the 65 GB
backbone, so the failure costs a full load.

### Workaround, and what is verified

`_bpe_tokenizer_with_eos_alias` repeats `convert_qwen.convert_tokenizer`'s
vocabulary assembly and builds a subclass whose `__init__` registers the
checkpoint's real `eos_token` instead of the hardcoded one (calling
`BytePairTokenizer.__init__` directly).  Aliasing inside the vocabulary
does **not** work — keras-hub's id->token map is a tf `HashTable` and
rejects two names for one id:

```
FailedPreconditionError: HashTable has different value for same key.
Key 151643 has <｜end▁of▁sentence｜> and trying to add value <|endoftext|>
```

`run_keras_lm_smoke` now builds the preprocessor first (cheap) and only
then loads the backbone, so the retry costs nothing.

**Verified standalone** (no weights, seconds):

```
stock from_preset FAILS: ValueError Cannot find special token `'<|endoftext|>'` ...
alias for <|endoftext|> -> '<｜end▁of▁sentence｜>'
end_token_id = 151643   vocab = 151665
tokenize:   [785, 6722, 315, 9625, 374]        # "The capital of France is"
detokenize: 'The capital of France is'
preprocessor OK, seq len 1024
```

`end_token_id` comes out as the checkpoint's real EOS, and the BPE
round-trip is exact.

**NOT verified: the end-to-end load + generate.**  Two attempts ran:
the first died on the tokenizer (78.6 s, before the fix); the second got
past the tokenizer, started the single backbone load, and was killed by
the machine-wide stand-down at 16:03:15 (`Terminated: 15`, rc 143).  So
**"does it generate sane tokens" is still open** — everything up to and
including the tokenizer is proven, the weight load has been proven once
(under the pre-fix code path), and only `generate()` is unwitnessed.

Command to finish it (serialised, ~65 GB, holds the lock):

```
SP=/private/tmp/claude-501/-Users-oleg-metaljax/43351818-f4b5-4809-87c8-2909e6e0e70e/scratchpad
KMP_DUPLICATE_LIB_OK=TRUE JAX_PLATFORMS=cpu KERAS_BACKEND=jax \
  $SP/bench-venv/bin/python scripts/model_bench/adapter_keras_extra.py \
    r1-distill-32b cpu --path keras_lm_smoke \
    --manifest scripts/model_bench/manifest.json --out results.jsonl
```

Expect ~4 min for the backbone load plus jit-compile of a 64-layer
decode loop on CPU; `n_decode` is 5.  `BENCH_SMOKE_WEIGHTS=0` gives a
weightless variant, but note it still *allocates* the 32 B parameters
from initializers, so it is not cheap either.

---

## 7. Still open / commands for a serialised run

1. **SD 3.5 Large black image** (§4).  Stage-by-stage finite check,
   written but never run — loads the model once on metal at 512x512 and
   prints shape/finite/nan/min/max/mean for the CLIP text embeddings,
   two denoise steps and the VAE decode:

   ```
   SP=/private/tmp/.../scratchpad
   KMP_DUPLICATE_LIB_OK=TRUE KERAS_BACKEND=jax JAX_PLATFORMS=metal SIZE=512 \
     $SP/bench-venv/bin/python $SP/t_sd35_diag.py
   ```

   (script at `<scratchpad>/t_sd35_diag.py`; ~18 GB, needs the lock).
   If the CLIP embeddings are already non-finite the f16 encoders are
   the cause, not metaljax; if they are clean and the latents go bad
   during `denoise_step`, it is worth reducing to a metaljax repro.

2. **DeepSeek-R1 generate** — command in §6.

3. Optional: a CPU reference for SD3 is only possible on a preset whose
   CLIP towers are not f16 (`stable_diffusion_3_medium`, 5.98 GB,
   anonymously available — its config was not inspected).

Nothing here changes the adapter code; all three are runs, not fixes.

---

## 8. Files added

* `scripts/model_bench/adapter_keras_extra.py` — the adapters (new file).
* `scripts/model_bench/README_keras_extra.md` — this file (new file).

Raw records from the runs above are appended to
`<scratchpad>/extra_results.jsonl` (same shape as `results.jsonl`).  Note
that file also contains two early `lora-gemma4-e2b` rows with `ok:false`
(`Missing data for input "audio_indices"` and the `'prompts'` indexing
`TypeError`) — those are iterations while the adapter was being written,
not results; the final `ok:true` rows supersede them.

Nothing else in `scripts/model_bench/` was touched; `run_bench.py` and
`manifest.json` are unmodified.  No new python packages were installed
into the benchmark venv.

Environment knobs used by the adapters:

| var | default | meaning |
|---|---|---|
| `BENCH_WARM_STEPS` | 5 | warm forwards per batch size (vision) |
| `BENCH_DIFFUSION_STEPS` | 20 | diffusion steps |
| `BENCH_IMAGE_SIZE` | 1024 | diffusion image side |
| `BENCH_IMAGE_OUT` | unset | path to dump the generated PNG |
| `BENCH_LORA_STEPS` | 8 | warm train steps |
| `BENCH_SMOKE_WEIGHTS` | 1 | set 0 to check class/config mapping only |
| `BENCH_SPM_PATCH` | 1 | set 0 to disable the sentencepiece workaround |

### Suggested `manifest.json` corrections (NOT applied — read-only here)

* `siglip2-so400m`, `sd35-large`, `lora-gemma4-e2b`: drop
  `"status": "needs-adapter"`; all three now run.
* `sd35-large`: `"backends": ["metaljax"]` is right — jax-CPU cannot run
  this preset at all (§4a).
* `siglip2-so400m` `size_gb: 2.3` is the bf16 figure; the Kaggle download
  is 4.55 GB (f32 `model.weights.h5`).
* `r1-distill-32b` `"note": "converter path unverified"` should become
  "arch confirmed (qwen2 -> QwenBackbone); needs the EOS-alias tokenizer
  workaround, see README_keras_extra.md §6".
