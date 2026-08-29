# Comparator-fill campaign — discovery record (2026-08-29)

Task: fill STATUS.md cross-framework cells that are empty despite a published
same-precision implementation existing (fn-20 like-for-like rule: same
precision as the metaljax cell; custom kernels fair, different quantization
not; published artifacts only — nothing converted or authored locally).

## Toolchain

llama.cpp REBUILT at the pinned commit (the 2026-08-03 build lived in a tmp
scratchpad and was wiped):

- commit `221f0f6356efe2260023208365705ec5d5a7c8f5` (= b10235, the pinned
  build of README_llamacpp.md), ggml 0.18.0, AppleClang 21.0.0, METAL+BLAS.
- location: `~/.cache/metaljax-bench/llamacpp/llama.cpp-221f0f63/build/bin`
  (durable, so the next campaign does not lose it again).
- NB `--version` prints `version: 1 (221f0f6)` — depth-1 fetch artifact;
  the hash is the identifier (same caveat as the original `version: 200`).
- build log: build-llamacpp.log here.

Protocol: identical to README_llamacpp.md — `llama-bench -p 51 -n 128 -r 5
-o json -v`, all-Metal, f16 KV, machine lock `/tmp/metaljax-bench.lock`
taken by the adapter per row; greedy `llama-completion --jinja` coherence
check. Driver: scratchpad `drive_bf16.py` (registers new rows into
`adapter_llamacpp.py` at runtime; repo untouched). Results JSONL:
`results_llamacpp_bf16.jsonl` here.

## BF16 GGUF discovery (HF API, 2026-08-29)

Per-row provider sweep; sha = repo main at query time = the pin used.

| STATUS row | pick | file(s) | GB | repo sha |
|---|---|---|---|---|
| 3 gemma4-26B-A4B | ggml-org/gemma-4-26B-A4B-it-GGUF (single file, llama.cpp's own conversion) | gemma-4-26B-A4B-it-BF16.gguf | 50.51 | bb4531cd |
| 5 Qwen3-8B | unsloth/Qwen3-8B-GGUF (upstream Qwen has Q4–Q8 only) | Qwen3-8B-BF16.gguf | 16.39 | a6adef13 |
| 6 Llama-3.1-8B | legraphista/Meta-Llama-3.1-8B-Instruct-IMat-GGUF | Meta-Llama-3.1-8B-Instruct.BF16.gguf | 16.07 | cf2a95d3 |
| 8 Qwen3.6-35B-A3B | unsloth/Qwen3.6-35B-A3B-GGUF (non-MTP; upstream Qwen GGUF repo 401) | BF16/…-BF16-0000{1,2}-of-00002.gguf | 69.37 | a483e9e6 |
| 9 R1-Distill-32B | bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF (unsloth publishes F16 only) | …-bf16-0000{1,2}-of-00002.gguf | 65.54 | 1dc8cf9f |
| 10 DeepSeek-V2-Lite | legraphista/DeepSeek-V2-Lite-Chat-IMat-GGUF | DeepSeek-V2-Lite-Chat.BF16.gguf | 31.42 | b6350f4c |
| 11 Qwen3-0.6B | unsloth/Qwen3-0.6B-GGUF (primary; ggml-org as control) | Qwen3-0.6B-BF16.gguf | 1.20 | 50968a44 |
| 12 Mixtral 8x7B | **NONE EXISTS** — see below | — | — | — |

fn 20's "no bf16 GGUF exists" claim is STALE for rows 5/6 (and 3): published
BF16 conversions exist today for all three.

Rejected-provider notes:
- Row 6: bartowski (the pinned Q8/Q4 provider), MaziyarPanahi,
  lmstudio-community carry no 16-bit file; mradermacher is **F16** (not
  bf16); unsloth/Meta-Llama-3.1-8B-Instruct-GGUF returns 401 (gated).
- Row 9: unsloth = F16 only; mmnga = IQ quants only; ggml-org = Q8_0 only.
- Row 11: ggml-org/bartowski files are 1.51 GB vs unsloth 1.20 GB — the
  difference is a duplicated (untied) bf16 output tensor, NOT an f32
  embedding; all three are all-bf16 weights. unsloth = pure conversion.

## Row 12 Mixtral-8x7B-Instruct-v0.1: no 16-bit GGUF exists (search evidence)

Tree-listed 9 providers: TheBloke, mradermacher, second-state,
RichardErkhov, Artefact2, MaziyarPanahi, billborkowski, shenberg1, HenryJJ —
all quants only (Q2K…Q8_0, IQ*). Name-level sweep of the top-100
`search=Mixtral-8x7B` repos: zero GGUF repos with f16/bf16/fp16 in the name.
The llama.cpp cell therefore stays legitimately empty at bf16 (the mlx-lm
52.8 cell remains the only same-precision cross-framework point).

## Rows 16/18 precision paper trail (no re-measurement needed)

Both recorded torch-MPS cells carry explicit dtype evidence and match our
cells' precision — like-for-like rule SATISFIED:

- Row 16 SigLIP 2: torch record `{"id": "siglip2-so400m", "backend":
  "torch-mps", "dtype": "bfloat16", "step_ms": 29.95, "step_ms_b32": 591.2}`
  (~/.cache/metaljax-bench/logs/results_new.jsonl, 2026-08-03); gated
  against torch-CPU float32 (cos 0.99999x). Our cell:
  `adapter_keras_extra.py::run_keras_vision_forward` runs under
  `_keras_bf16()` = plain `set_dtype_policy("bfloat16")` (variables AND
  compute bf16, not mixed). Both sides bf16.
- Row 18 LoRA E2B: torch record `{"id": "lora-gemma4-e2b", "dtype":
  "bfloat16", "step_ms": 135.598, "attn_backward": "math"}`. Our cell:
  `run_keras_lora_train` under the same bf16 policy, rank 4 / seq 256 / b1
  on both stacks. Both sides bf16. The disclosed asymmetry (torch MPS has
  no fused SDPA backward — math fallback) is kernel-level, which fn 20
  explicitly allows ("custom kernels fair").

## Dtype verification

Every measured gguf's tensor-type census (gguf-py from the pinned checkout,
header parse — filenames not trusted): dtype-evidence.log here. Bar: all
weight matrices BF16; F32 confined to 1-D norm/bias vectors.

## Measured cells (running record; full JSONL: results_llamacpp_bf16.jsonl)

llama-bench -p 51 -n 128 -r 5, all-Metal, build 221f0f63, machine lock held
per row; every gguf's tensor census in dtype-evidence.log (all weights BF16,
F32 confined to 1-D norm vectors); greedy coherence checks all passed.

| STATUS row | gguf | decode ms/tok | tg tok/s (sd) | prefill ms | mem GB |
|---|---|---:|---:|---:|---:|
| 11 Qwen3-0.6B | unsloth BF16 1.20 GB | **3.41** | 293.6 (2.9) | 11.7 | 1.3 |
| 11 control | ggml-org BF16 1.51 GB (untied output copy) | 3.38 | 296.3 (5.7) | 11.7 | 1.6 |
| 5 Qwen3-8B | unsloth BF16 16.39 GB | **29.57** | 33.8 (0.21) | 68.1 | 15.3 |
| 6 Llama-3.1-8B | legraphista BF16 16.07 GB | **29.16** | 34.3 (0.21) | 67.8 | 16.2 |
| 10 DeepSeek-V2-Lite | legraphista BF16 31.42 GB | **10.72** | 93.3 (0.24) | 68.9 | 31.5 |
| 3 gemma4-26B-A4B | ggml-org BF16 50.51 GB | **16.85** | 59.4 (1.1) | 86.5 | 50.7 |

Consistency: dense rows sit at 551-554 GB/s effective bandwidth (the
439-555 band of README_llamacpp.md); both MoE rows show the impossible
nominal rate (~3 TB/s) that proves gathered-expert dispatch.

Row 9 R1-Distill-32B: bartowski BF16 2-shard 65.5 GB — decode **114.94**
ms/tok (tg 8.70 sd 0.04), prefill 260.6 ms, coherent. NB the JSONL's
mem_gb=38.5 and gguf_gb=39.88 describe SHARD 1 only: adapter_llamacpp's
metal_mem_gb takes max per buffer class, and split models allocate one
model buffer per shard (39.9 + 25.7). True resident ~= 65.5 GB weights +
~1 GB KV/compute. Timings unaffected (they come from llama-bench itself).

Row 8 Qwen3.6-35B-A3B: unsloth BF16 2-shard 69.3 GB — decode **15.34**
ms/tok (tg 65.19 sd 0.20), prefill 92.8 ms, coherent. Same split-gguf
caveat: JSONL gguf_gb/mem_gb describe shard 1 only; true resident ~= 69.3
GB weights. The one row where mlx-lm (13.7) leads llama.cpp (1.12x).

## Campaign complete — 2026-08-29

8/8 records ok, all on build 221f0f6, all greedy coherence checks passed,
machine lock acquired cleanly for every run (0 s wait each time — no
contention with the decode-optimization agent observed) and verified
released at the end.

Disk actions: downloaded ~250 GB of pinned BF16 ggufs; after measurement
DELETED the two wave-2 split sets (bartowski R1-Distill-32B-bf16 65.5 GB,
unsloth Qwen3.6-35B-A3B BF16 69.4 GB) to restore headroom — free went
311 GB (start) -> 75 GB (peak usage) -> 201 GB (after cleanup). Kept in
~/.cache/huggingface: the 0.6B pair, Qwen3-8B, Llama-3.1-8B, V2-Lite-Chat
and gemma4-26B-A4B BF16 ggufs (~116 GB) for cheap re-measurement.
