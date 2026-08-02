# torch-MPS: the three non-decode rows

`adapter_torch_mps.py` covers STATUS rows 1–15 (causal-LM decode) with its
original flat CLI.  This addendum documents the three subcommands added for
the rows that are *not* decode, each mirroring the metaljax/keras cell it is
compared against in `adapter_keras_extra.py`:

| STATUS row | subcommand | model | headline metric |
|---|---|---|---|
| 16 SigLIP 2 so400m | `vision` | `google/siglip2-so400m-patch14-384` | image+text forward ms, b1 and b32 |
| 17 SD 3.5 Large | `diffusion` | SD 3.5 Large (ungated mirror) | ms per diffusion step |
| 18 LoRA E2B train | `lora` | `google/gemma-4-E2B-it` | ms per train step |

The legacy CLI is unchanged — `adapter_torch_mps.py --model … --smoke 8`
still works and is now also reachable as the explicit `decode` subcommand.

## Running them

```sh
V=~/.cache/metaljax-bench/venvs/torch/bin/python
$V scripts/model_bench/adapter_torch_mps.py vision
$V scripts/model_bench/adapter_torch_mps.py lora
$V scripts/model_bench/adapter_torch_mps.py diffusion --image-size 512
```

Each appends one JSONL record to `~/.cache/metaljax-bench/logs/
results_new.jsonl`, takes the suite-wide `/tmp/metaljax-bench.lock`
(`--no-lock` to skip, never for a published timing) and waits for swap to
fall below `--max-swap-gb` (default 20) before it starts.  The lock is
released from an `atexit` hook *and* a SIGINT/SIGTERM handler — a `with`
block alone leaks the directory when a run is killed, and a leaked lock
wedges every later job in the suite.

Extra packages this needed on top of the existing `torch` venv
(torch 2.13.0, transformers 5.14.1): `peft`, `diffusers`, `pillow`,
`sentencepiece`, `protobuf`.

## Method notes

**Timing.** `torch.mps.synchronize()` is a real barrier, unlike
`jax.block_until_ready` on the metaljax backend (CLAUDE.md), so it is what
closes each timed region; training steps additionally pull `loss.item()`.

**SigLIP 2 (`vision`).** Inputs reproduce `_siglip_preprocess`: uniform
random pixels in 0..255 from `default_rng(0)` at 384², the same three
captions cycled to fill the batch, seq 64.  The HF image processor is
`rescale=1/255` then `normalize(0.5, 0.5)` — arithmetically identical to
the `x/127.5 - 1` the keras converter applies — so `pixel_values` is built
directly, which also keeps host preprocessing out of the timed region.
Warm median of 5 forwards per batch size, after one warm-up forward.

**LoRA on Gemma 4 E2B (`lora`).** rank 4, seq 256, batch 1, bf16, Adam
1e-4, 1 warm-up step + 8 timed steps (the keras cell's shape: its first
`.fit()` step is trace+compile, ours is just the first step, reported as
`compile_s` so the columns line up).  The synthetic word-salad batch is
generated with the same vocabulary and `default_rng(0)`, and the keras
prompt/response `sample_weight` masking is reproduced with `labels = -100`
over the first half of the window, so both stacks backprop the same span.

## Measured, 2026-08-03 (M5 Max 128 GB, torch 2.13.0, bf16)

Records in `~/.cache/metaljax-bench/logs/results_new.jsonl`; `jax CPU` and
`metaljax` columns are the existing STATUS values for the same cells.

| row | metric | jax CPU | metaljax | **torch-MPS** |
|---|---|---|---|---|
| 16 SigLIP 2 | fwd b1 (ms) | 964.7 | 248.0 | **29.8** |
| 16 SigLIP 2 | fwd b32 (ms) | 23324.1 | 5287.0 | **591.4** |
| 17 SD 3.5 L | ms/step @512² | ✗ | ✗ blocked | **654.2** |
| 17 SD 3.5 L | ms/step @1024² | ✗ | ✗ blocked | **2997.5** |
| 18 LoRA E2B | ms/train step | 2141.3 | 417 | **135.6** |

torch-MPS is 8.3× (b1) / 8.9× (b32) faster than metaljax on SigLIP 2 and
3.1× on the LoRA step.  Row 17 is the one with no metaljax number at all:
both resolutions ran, including the 1024² the metaljax cell could not
reach, at 29.2 GB / 34.0 GB device memory.

**Row 16 correctness** — bf16-on-MPS against the same checkpoint in float32
on torch-CPU, batch 4: `cos_logits` 0.999996, `cos_image_embeds` 0.99995,
`cos_text_embeds` 0.999971, max abs logit diff 0.118, argmax 4/4.  Also
worth recording as a cross-stack check: the trained SigLIP head's
`logit_scale`/`logit_bias` come out 4.6875 / −15.9375 here and 4.6875 /
−15.9375 in the keras cell — bit-identical, i.e. the same checkpoint
reached by two entirely different loaders (HF safetensors vs Kaggle
preset).

Re-measured back-to-back the row reproduces to 0.6 %: 29.77 / 29.95 ms at
b1 and 591.4 / 591.2 ms at b32 across two runs.

**Row 17 correctness** — the gate the metaljax cell failed.  Both images are
non-black and photographically coherent (512²: pixel mean 58.0, std 69.3;
1024²: mean 111.4, std 94.9), saved to
`~/.cache/metaljax-bench/logs/sd35_torch_mps_{512,1024}.png`.  A coherent
astronaut-on-Mars render is also the strongest available evidence that the
ungated mirror carries genuine SD 3.5 L weights.

**Row 18** — 135.6 ms median over 8 warm steps (samples 135.2–166.1; the
first step is 1893 ms), 669,696 trainable params over 50 adapted modules,
0.0131 % of 5.10 B.

### Numbers that are NOT comparable across stacks

*Row 18 loss series.*  torch sits near 7.4, the keras/CPU cell near 2.9.
Both stacks see word salad from the same generator, but keras feeds it
through the Gemma instruction preprocessor (prompt/response, its own
packing) while we tokenise the raw string and mask the first half, so the
targets differ.  Step *cost* is the same shape — batch 1, 256 tokens,
rank-4 LoRA on q/v — which is what the row compares; the absolute loss is
not a cross-stack quantity.  Within a stack the series is still the sanity
check that a real backward ran.

*Row 16 `logit_diag`.*  −12.125 here vs −9.5625 in the keras cell, from
text preprocessing, not from the model: keras canonicalises and pads with
`</s>` (id 1), the HF processor pads with id 0.  Each stack is gated
against its own CPU reference, which is the comparison that means
something.

## Findings worth keeping

**MPS has no fused-SDPA backward kernel** (STATUS footnote 10) — now with
evidence, and with a dead end recorded so nobody repeats it.

The obvious probe does not work: pinning a backend with
`torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])` and watching
the backward succeed proves nothing, because those flags are CUDA-scoped
and do not constrain dispatch on MPS — all three "succeed" here whatever
runs underneath.  The first version of this probe reported exactly that
and was worthless.

What is real evidence is the autograd node the forward records.  On MPS no
dtype produces a fused SDPA node:

| device / dtype | recorded `grad_fn` |
|---|---|
| mps float32 | `UnsafeViewBackward0` |
| mps float16 | `ToCopyBackward0` → `UnsafeViewBackward0` |
| mps bfloat16 | `ToCopyBackward0` → `UnsafeViewBackward0` |
| cpu (control) | `ScaledDotProductFlashAttentionForCpuBackward0` |

A view/copy node at the top of the graph is the tail of a decomposed
matmul-softmax-matmul; CPU, by contrast, names a genuine fused SDPA
backward.  Same result at realistic attention shape (1×8×256×64, causal).
The `lora` row therefore records `attn_backward="math"` and ships the
probe in `sdpa_backward_probe`.

**peft cannot target Gemma 4's tower projections.**  `AutoModelForCausalLM`
on `gemma-4-E2B-it` returns the whole multimodal
`Gemma4ForConditionalGeneration`, whose vision/audio `q_proj`/`v_proj` are
`Gemma4ClippableLinear` wrappers; peft 0.20 rejects them ("only
torch.nn.Linear … supported").  A bare `target_modules=["q_proj","v_proj"]`
therefore fails outright.  The adapter targets
`.*language_model.*\.(q_proj|v_proj)$` — plain `nn.Linear`, and the right
thing anyway: it is what `lm.backbone.enable_lora` adapts on the keras
side, and a text-only train step would never send gradient to tower
adapters.

**SD 3.5 Large is gated on Hugging Face.**
`stabilityai/stable-diffusion-3.5-large` answers HTTP 401 without an
accepted licence and a token (`gated: auto` — instant approval, but still
an account).  No token was requested or used.  The row runs against
`adamo1139/stable-diffusion-3.5-large-ungated`, an ungated copy of the same
diffusers tree (uploaded 2024-10-29, a week after release; 5.7k downloads),
pinned at revision `5d868ff`.  Only the diffusers subtrees are fetched
(~26 GB): the duplicate `*.fp16.*` variants, the ComfyUI-style
`text_encoders/` single files and the combined `sd3.5_large.safetensors`
are skipped.  Being a third-party mirror, the record carries both `repo`
and `upstream_repo`, and the run is gated on producing a non-black image.

## Machine-discipline notes

**A killed run leaks the lock, and a leaked lock wedges the suite.**  Seen
live: a `--lock-once` llama.cpp sweep was killed mid-flight, its lock
directory survived, and every later job polled against a lock whose owner
was long dead (the queued `vision` run sat on it for 440 s until the
directory was removed by hand).  Diagnosis is `stat` the directory for its
creation time and check whether the creating PID still exists.  The
subcommands here release from `atexit` **and** from a SIGINT/SIGTERM
handler for that reason; `with MachineLock(...)` alone is not enough.

**`results_new.jsonl` had two records glued onto one line** — a previous
writer appended without a trailing newline, so the next record landed on
the same line and the file no longer parsed as JSONL.  Repaired by
re-splitting with `json.JSONDecoder().raw_decode` (backup at
`results_new.jsonl.bak`).  Every writer must end its record with `\n`.
