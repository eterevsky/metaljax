# Comparison stacks for the model benchmark suite

The suite's point is *metaljax runs real models on Metal*. A number on its
own proves nothing, so every row is measured against the stacks a Mac user
would otherwise reach for:

| leg | what it is | venv | adapter |
| --- | --- | --- | --- |
| `metaljax` | JAX through our PJRT plugin, `JAX_PLATFORMS=metal` | `bench-venv` | `run_bench.py` |
| `cpu` | jax's own CPU backend, same program | `bench-venv` | `run_bench.py` |
| `mlx` | mlx-lm on the same checkpoint — same Metal library, hand-written kernels | `bench-venv` | `run_bench.py` (`run_mlx`) |
| `torch-mps` | PyTorch + transformers on Apple's MPS backend | **`torch-venv`** | `adapter_torch_mps.py` |

torch gets its **own interpreter**. torch and Metal-backed JAX/MLX must not
share a process — two Metal device queues, two allocators, one 128 GiB pool.
The runner already sequences one benchmark per process; the torch leg just
uses a different python.

---

## 1. torch-MPS leg

### Setup

```sh
uv venv torch-venv --python 3.13
uv pip install --python torch-venv/bin/python \
    -r scripts/model_bench/requirements-torch-mps.txt
torch-venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

Plain PyPI `torch` on macOS/arm64 ships MPS; there is no separate wheel index.

### Use

```sh
# functional check, no timings
HF_HUB_OFFLINE=1 torch-venv/bin/python scripts/model_bench/adapter_torch_mps.py \
    --model Qwen/Qwen3-8B --device mps --smoke 32 --out ids_mps.json

# CPU reference for the same checkpoint + agreement gate
HF_HUB_OFFLINE=1 torch-venv/bin/python scripts/model_bench/adapter_torch_mps.py \
    --model Qwen/Qwen3-8B --device cpu --smoke 32 --compare ids_mps.json

# timed row (matches run_bench.py's metric definitions)
torch-venv/bin/python scripts/model_bench/adapter_torch_mps.py \
    --bench-id qwen3-8b-bf16 --device mps --decode-tokens 128
```

`run_torch_mps(bench, prompt, n_decode)` returns the same keys the other
adapters do — `load_s`, `warmup_s`, `prefill_ms` (warm, 1 new token),
`decode_ms_tok` ((full generate − prefill)/new tokens), `token_ids` (first
64 greedy ids), `out_tokens`, `prompt_tokens` — plus `prompt_sha` (sha256
of the rendered prompt, so a template change is visible) and `mem_gb_torch`
from
`torch.mps.driver_allocated_memory()`. On a unified-memory box RSS
double-counts wired GPU pages, so the driver number is the honest one; note
it is *driver-allocated*, i.e. the high-water pool, not live tensors.

### Traps (respect these; they are not stylistic)

**`PYTORCH_ENABLE_MPS_FALLBACK` must be UNSET.** With it set, any op the MPS
backend lacks is silently relocated to the CPU. The run still emits tokens
and still produces a plausible ms/token — it is simply not a Metal
measurement, and nothing in the output says so. `adapter_torch_mps.py`
raises at *import* if the variable is present. The correct response to a
model that needs it is to run once without it, let the op raise, and report
the op by name as an MPS gap. **Enumerated on this stack so far: none** —
Qwen3-8B bf16 (sdpa, greedy, 32 tokens) completes with the fallback unset
and no `NotImplementedError`. Any future row that trips one goes in this
list with its exact error text.

**`device_map` takes the plain string `"mps"`.** `torch.device("mps", 0)`
and `"mps:0"` trip accelerate's device-map bookkeeping inside transformers
and fail to dispatch. Indexed MPS devices are a known transformers issue;
the string form is the supported spelling.

**MPS SDPA has had correctness bugs on non-contiguous q/k/v.** They are
silent — plausible text, wrong logits. Every model row must be gated
against the same checkpoint on torch-CPU before its numbers are published:
greedy token ids must match, or the first divergence index must be
explained. `--attn eager` swaps the fused kernel out for a row that
disagrees; a row that only agrees under `eager` must say so in its results
line, because that is a different kernel path than the one users get.

**`torch_dtype` → `dtype`.** transformers 5.x renamed the argument;
`load_model` tries `dtype=` and falls back, so the adapter works on both.

### Functional evidence (2026-08-02, M5 Max, macOS 26.5.2)

`Qwen/Qwen3-8B` bf16, cached snapshot, `attn_implementation="sdpa"`,
greedy, the manifest prompt (59 prompt tokens after chat-template render):

```
MPS: {"repo": "Qwen/Qwen3-8B", "device": "mps", "attn": "sdpa",
      "prompt_tokens": 59,
      "token_ids": [151667, 198, 32313, 11, 279, 1196, 374, 10161, 911, 3170,
                    4938, 33394, 374, 803, 2989, 1091, 7112, 12564, 2337, 279,
                    16895, 10262, 315, 42578, 44378, 11, 5310, 448, 8162, 594,
                    42690, 4938],
      "text": "<think>\nOkay, the user is asking about why memory bandwidth is
               more important than raw compute during the decode phase of
               transformer inference, especially with Apple's unified memory",
      "mem_gb_torch": 16.49}

CPU  (sdpa):  identical token_ids; compare -> {"agree": true, "compared": 32}
MPS  (eager): identical token_ids; compare -> {"agree": true, "compared": 32}
```

Three-way agreement, 32/32 greedy tokens: MPS·sdpa ≡ MPS·eager ≡ CPU·sdpa.
The CPU leg rules out a wrong-kernel result; the eager leg rules out the
fused-SDPA path specifically, which is where the known non-contiguous
q/k/v bugs live. Memory 16.49 GB driver-allocated
for a 16.4 GB bf16 checkpoint — i.e. weights plus a small KV cache, no
hidden host copy. No timing claims are made here: timed runs belong in the
suite's own process, not in a smoke test.

---

## 2. Pinned quantized / aspirational repos

The bf16 rows measure the frameworks. The 4-bit rows measure *what actually
fits*, and only mlx-lm can run them today (metaljax has no packed quantized
storage — the 235B row is tracked as `blocked-metaljax` in the manifest).

Selection rule: **uniform quantization, official `mlx-community` conversion,
highest download count**. Uniform matters because the comparison is
cross-stack — a repo whose per-layer bit allocation was tuned by a
calibration search is measuring the search, not the runtime. `OptiQ`
variants (per-layer mixed bits + a `kv_config.json`) are consistently
larger for the same nominal bit width (e.g. gemma-4-31b: 23.5 GB OptiQ vs
18.4 GB plain 4-bit) and are excluded for that reason. All recommended
repos are `mode: affine, bits: 4, group_size: 64`; the two MoE rows carry
8-bit *router/gate* projections, which is mlx-lm's standard conversion for
MoE (routing is numerically fragile at 4 bits) and not an ad-hoc mixed
scheme.

Sizes are summed safetensors bytes from the HF API (decimal GB), verified
2026-08-02.

| manifest row | pinned 4-bit repo | revision | weights | last modified | DLs | status |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma4-12b-bf16` | `mlx-community/gemma-4-12B-it-4bit` | `73bcf09092aa` | 6.74 GB | 2026-06-08 | 7.4k | not downloaded |
| `gemma4-31b-bf16` | `mlx-community/gemma-4-31b-it-4bit` | `696d436c4047` | 18.41 GB | 2026-07-05 | 49k | not downloaded |
| `gemma4-26b-a4b` | `mlx-community/gemma-4-26b-a4b-it-4bit` | `0d77464eeb23` | 15.34 GB | 2026-07-05 | 30k | not downloaded |
| `gemma4-e2b-bf16` | `mlx-community/gemma-4-e2b-it-4bit` | `238767527555` | 3.55 GB | 2026-07-06 | 68k | not downloaded |
| `qwen3-8b-bf16` | `mlx-community/Qwen3-8B-4bit` | `545dc4251c05` | 4.61 GB | 2025-04-28 | 36k | **downloaded** |
| `llama31-8b-bf16` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | `241a666dad6c` | 4.52 GB | 2024-11-26 | 17k | **downloaded** |
| `qwen36-35b-a3b` | `mlx-community/Qwen3.6-35B-A3B-4bit` | `38740b847e4c` | 20.40 GB | 2026-04-16 | 62k | not downloaded |
| `aspirational-235b-4bit` | see below | | | | | **manifest needs a change** |

Repo-name notes: the mlx-community ids are **lowercase** where the Google
ids are not (`gemma-4-31b-it-4bit`, `gemma-4-26b-a4b-it-4bit`,
`gemma-4-e2b-it-4bit`), while the 12B one keeps the capital B
(`gemma-4-12B-it-4bit`). The Llama row's bf16 checkpoint is the
`unsloth/` mirror (the `meta-llama/` original is gated); the 4-bit
conversion is from the original Meta weights and its tokenizer/chat
template is byte-identical to the unsloth mirror's, so the rows are
comparable.

### The aspirational 235B row: 4-bit does NOT fit, 3-bit does

This box has 137.4 GB (128 GiB) of unified memory.

| repo | bits | weights | fits 128 GiB? |
| --- | --- | --- | --- |
| `mlx-community/Qwen3-235B-A22B-4bit` | 4 | **132.24 GB** | **no** — exceeds physical RAM before the OS gets a page |
| `mlx-community/Qwen3-235B-A22B-mixed-3-4bit` | 3/4 mixed | 107.58 GB | yes, ~30 GB headroom, but mixed |
| `mlx-community/Qwen3-235B-A22B-3bit` | 3 | **102.86 GB** | **yes**, ~34 GB headroom |
| `mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit` | 3 | 102.86 GB | yes — newer Instruct-2507 weights |

**Recommendation: `mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit`**
(revision `8df14918a672`, 102.86 GB, uniform `bits: 3, group_size: 64`,
2025-07-22). Uniform quant, and the Instruct-2507 weights are the ones
people actually run; it ships a `chat_template.jinja` where the original
2025-04 conversion has none. If a same-weights comparison against the older
row matters more than currency, take `mlx-community/Qwen3-235B-A22B-3bit`
(`aac8fe6015d8`) — same size, same quant settings.

The manifest currently pins `mlx-community/Qwen3-235B-A22B-4bit` with
`size_gb: 125`. Both numbers are wrong: the repo is 132.24 GB and will not
load on this machine. That row's model/size needs updating (I did not touch
`manifest.json`). Neither 235B repo was downloaded — 103 GB is a deliberate
decision, not a default.

**Not downloaded, on purpose:** everything except the two ~4.5 GB rows. The
cache is already 366 GB.

---

## 3. Chat templates: the same rendered prompt for every stack

**The finding.** Community MLX conversions snapshot the upstream chat
template at conversion time and do not track it. Every `mlx-community`
gemma-4 mirror — bf16 *and* 4-bit — carries the same older template:

| repo | template sha256 (first 16) | bytes |
| --- | --- | --- |
| `google/gemma-4-{12B,31B,26B-A4B}-it` | `ae53464bf3be2580` | 18681 |
| `mlx-community/gemma-4-{12B,31b,26b-a4b}-it-{bf16,4bit}` | `36e3a42e5cf14cd0` | 17466 |
| `google/gemma-4-E2B-it` | `0a2c8073c878ab1d` | 18567 |
| `mlx-community/gemma-4-e2b-it-4bit` | `2f1b4d75d067bae3` | 17336 |
| `Qwen/Qwen3-8B` | `a55ee1b1660128b7` | 4168 |
| `mlx-community/Qwen3-8B-4bit` | `87a2728cb8dc9fe4` | 4116 |
| `unsloth/Meta-Llama-3.1-8B-Instruct` ≡ `mlx-community/…-4bit` | `e10ca381b1ccc5cf` | 4614 |
| `Qwen/Qwen3.6-35B-A3B` ≡ `mlx-community/…-4bit` | `e84f32a23fdda276` | 7764 |

Google's current template carries its own provenance header — *"Google
Gemma 4 Canonical Chat Template … Published: 2026-07-09 … Fixed tool-calling
loops, turn closures, and thinking content-ordering"* — which the mirrors
predate. The substantive diffs are: `null` argument rendering in tool
calls; an O(1) tracked-state vs O(n) backward-scan detection of continued
model turns; `enable_thinking` / `preserve_thinking` defaulting; empty-
`messages` guards; and thinking-content gating around `tool_calls`.

**The good news, measured, not assumed.** For this suite's prompt shape —
one user message, no system message, no tools, thinking off — upstream and
mirror templates render **byte-identical text and identical token ids**:

```
gemma-4-12B  google vs mlx-community : same_text=true same_ids=true (63 tok)
gemma-4-E2B  google vs mlx-community : same_text=true same_ids=true (59 tok)
Qwen3-8B     Qwen   vs mlx-community : same_text=true same_ids=true (59 tok)
```

Every divergence lives in a branch this prompt never enters. So today's
numbers are comparable. That is a property of the prompt, not of the
repos — add a system message, a tool schema, a second turn, or
`enable_thinking=True` and the stacks start seeing different prompts, with
no error and no warning. It must therefore be pinned rather than trusted.

**The mechanism.** Render once, upstream; hand token ids to everyone.

`adapter_torch_mps.py` implements the first half:
`render_prompt(prompt, tokenizer, template_source=None)` takes the
*vocabulary* from the benchmarked repo and, when `template_source` (or the
env var `METALJAX_BENCH_CHAT_TEMPLATE`) names an upstream repo, the
*template* from there — pulling only `tokenizer_config.json` /
`chat_template.jinja`, not weights. It returns `(text, ids)`.

For the harness the concrete proposal is:

1. Add `"template_source"` to each manifest row (`google/gemma-4-12B-it`,
   `Qwen/Qwen3-8B`, …; absent = use the row's own repo).
2. In `run_bench.py`, render *once* per benchmark before dispatching to any
   adapter, and put the resulting `prompt_ids` in the record. Every adapter
   takes ids, not text: mlx-lm's `generate` accepts `prompt=` as a token
   list, transformers' `generate` takes a tensor of ids, and the JAX/Keras
   paths already tokenize explicitly.
3. Record the template sha256 alongside the ids in each result line. A
   mirror that re-syncs its template mid-campaign then shows up as a
   changed hash instead of as an unexplained token-count shift.

Feeding ids is strictly better than the alternative of pinning a
`--chat-template` per stack: it removes the template from the runtime
entirely, so a stack cannot silently disagree about `add_generation_prompt`,
BOS handling, or trailing whitespace. `mlx_lm.generate` does guard the
obvious double-BOS case, but by *heuristic* — `add_special_tokens =
tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)`
(`mlx_lm/generate.py`) — so a template that emits BOS anywhere other than
offset 0 still gets a second one, and the prompt length shifts by a token
with no error. Handing it a list of ids skips that branch entirely.

### Already-live discrepancy in `run_bench.py`

Worth flagging before any cross-stack numbers are published (I did not
modify `run_bench.py`): **the existing legs do not agree about chat
templating today.**

- `run_gemma_lib` calls `sampler.chat(prompt, …)` — the gemma library
  applies the chat template.
- `run_mlx` calls `generate(model, tok, prompt=prompt, …)` with the **raw**
  manifest prompt — no template.
- `run_keras_lm` calls `lm.generate(prompt, …)` with the **raw** prompt —
  no template.

So `gemma4-12b-bf16` (gemma_lib) and `gemma4-e2b-bf16` (keras_lm) are not
prompted the same way, and no mlx row is prompted the same way as any
gemma_lib row. The `token_ids` field exists precisely to catch cross-backend
disagreement, and it will fire on this — as a false alarm about the backend
when the real cause is the prompt. Rendering once in the runner and passing
ids (above) fixes the harness-level problem and the mirror-drift problem in
one change.

Until that lands, all rows use single-turn no-tool prompts, which the
measurements above show are template-invariant *given* that each stack
applies (or skips) the template consistently.
