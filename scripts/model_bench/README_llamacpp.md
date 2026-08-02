# llama.cpp comparison column

The llama.cpp column of [STATUS.md](../../STATUS.md) is **informational**: it
is the answer to "what does the fastest widely-used local-inference stack do
with these models on this machine", not a like-for-like backend comparison.
Read the [caveats](#what-this-column-does-and-does-not-measure) before
quoting any ratio — most of these rows run **quantized** weights against
metaljax's bf16.

Everything here is produced by
[`adapter_llamacpp.py`](adapter_llamacpp.py):

```
python scripts/model_bench/adapter_llamacpp.py --list          # rows
python scripts/model_bench/adapter_llamacpp.py <row-id> ...    # measure
python scripts/model_bench/adapter_llamacpp.py --all
```

It downloads the GGUF (pinned revision) into the shared
`~/.cache/huggingface`, takes the suite-wide `/tmp/metaljax-bench.lock` for
the timed section, runs `llama-bench`, then runs one greedy
`llama-completion` generation as a coherence check, and appends a
`backend="llamacpp"` record in the same JSONL shape `run_bench.py` emits.

Result records from the measurement run live outside the repo, in
`~/.cache/metaljax-bench/llamacpp/`.

## Results

Measured 2026-08-03, M5 Max / 128 GB / macOS 26.5.2, build `221f0f6`
(b10235). Every row: `-p 51 -n 128 -r 5`, all layers on Metal, f16 KV
cache. All 11 rows completed and produced coherent greedy text.

| row | quant | file GB | **decode ms/tok** | tg tok/s | prefill ms | pp tok/s | load s | mem GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-12b-bf16 | BF16 | 23.83 | **44.21** | 22.62 | 104.4 | 488.7 | 0.12 | 24.0 |
| gemma4-12b-q4 | Q4_0 QAT | 6.98 | **15.90** | 62.88 | 89.8 | 568.0 | 0.10 | 7.2 |
| gemma4-31b-bf16 | BF16 | 61.41 | **111.21** | 8.99 | 241.0 | 211.6 | 0.25 | 61.7 |
| gemma4-31b-q8 | Q8_0 | 32.64 | **63.95** | 15.64 | 203.6 | 250.4 | 0.22 | 33.0 |
| gemma4-31b-q4 | Q4_0 QAT | 17.65 | **37.29** | 26.82 | 192.5 | 265.0 | 0.22 | 18.0 |
| gemma4-26b-a4b-q4 | Q4_0 QAT | 14.44 | **7.86** | 127.15 | 49.3 | 1035.2 | 0.07 | 14.6 |
| qwen3-8b-q8 | Q8_0 | 8.71 | **15.71** | 63.64 | 57.9 | 881.1 | 0.07 | 8.8 |
| qwen3-8b-q4 | Q4_K_M | 5.03 | **10.22** | 97.83 | 57.0 | 894.0 | 0.08 | 5.1 |
| llama31-8b-q8 | Q8_0 | 8.54 | **15.38** | 65.04 | 57.1 | 893.0 | 0.08 | 8.6 |
| llama31-8b-q4 | Q4_K_M | 4.92 | **9.91** | 100.87 | 56.6 | 900.9 | 0.07 | 5.0 |
| gpt-oss-20b-mxfp4 | MXFP4 native | 12.11 | **6.65** | 150.31 | 46.6 | 1093.8 | 0.08 | 12.2 |

Run-to-run spread is small: an independent earlier pass over the same 11
files reproduced every decode figure within 4% (most within 1%), and
llama-bench's own inter-repetition stddev is under 1% except gpt-oss
(2.9 tok/s on 150, ~2%).

### Against the other columns

Only the two BF16 rows are a like-for-like precision match with the
metaljax / mlx-lm cells. Decode ms/token, lower is better:

| STATUS row | jax CPU | metaljax | mlx-lm | **llama.cpp** | precision match? |
|---|---:|---:|---:|---:|---|
| gemma4-12B bf16 | 346 (f32) | 101 | 58.3 | **44.21** | ✅ all bf16 |
| gemma4-31B bf16 | ✗ | 363 | 137 | **111.21** | ✅ all bf16 |
| gpt-oss-20b | — | 220.4 | 8.8 | **6.65** | ✅ mlx-lm + llama.cpp both native MXFP4; metaljax dequantizes to bf16 |
| gemma4-26B-A4B (MoE) | ✗ | 473 ⚠ | 17.0 | **7.86** | ❌ Q4_0 vs bf16 |
| Qwen3-8B | 219 | 60.3 | 30.4 | **15.71** (Q8) / 10.22 (Q4) | ❌ 8-/4-bit vs bf16 |
| Llama-3.1-8B | 206 | 58.6 | 29.4 | **15.38** (Q8) / 9.91 (Q4) | ❌ 8-/4-bit vs bf16 |

On the honest bf16 comparisons llama.cpp is **2.3× metaljax at 12B and
3.3× at 31B**, and it is ahead of mlx-lm by 1.32× and 1.23× — so the
"same Metal library underneath" gap band in STATUS.md is not the floor;
llama.cpp's hand-written Metal kernels beat MLX's on both bf16 rows.

Two rows deserve separate emphasis:

* **gpt-oss-20b** is the one row where llama.cpp and mlx-lm run the
  *identical* numeric format (native MXFP4, no requantization). llama.cpp
  wins 6.65 vs 8.8 (1.32×). The metaljax cell is 33× slower, but that is
  the dequantize-to-bf16 penalty (41.8 GB resident vs 12.2), not a kernel
  gap — it is the quantized-matmul roadmap item, priced.
* **gemma4-26B-A4B** at 7.86 ms/tok is 60× the metaljax cell (473) and
  2.2× mlx-lm. It corroborates STATUS footnote 16: the dense-expert
  lowering, not the kernels, is what costs metaljax this row.

### Effective bandwidth — a consistency check

Dividing file size by decode time gives the memory traffic decode
sustains. For the dense rows this should pin to the machine's practical
bandwidth ceiling, and it does:

| row | GB/s | | row | GB/s |
|---|---:|---|---|---:|
| qwen3-8b-q4 | 492 | | gemma4-12b-bf16 | 539 |
| qwen3-8b-q8 | 554 | | gemma4-31b-q4 | 473 |
| llama31-8b-q4 | 496 | | gemma4-31b-q8 | 510 |
| llama31-8b-q8 | 555 | | gemma4-31b-bf16 | 552 |
| gemma4-12b-q4 | 439 | | | |

Every dense row lands in a 439–555 GB/s band regardless of model or
quantization — the textbook signature of bandwidth-bound decode, and
evidence the numbers are sound rather than accidentally fast.

The two MoE rows break the pattern in the informative direction:
**gpt-oss-20b 1821 GB/s** and **gemma4-26B-A4B 1837 GB/s**. Those rates
are physically impossible on this machine, which proves llama.cpp touches
only the active experts (~1/3 of the file) per token. That is exactly the
gathered-dispatch behaviour metaljax lacks.

### Depth sensitivity

Caveat 2 below (llama-bench decodes from an empty context; the suite
decodes after a 51-token prompt) is measurable and negligible. Re-running
with `-d 51`:

| row | default (d=0) | d=51 | delta |
|---|---:|---:|---:|
| qwen3-8b-q4 | 10.22 | 10.18 | −0.4% |
| gemma4-12b-bf16 | 44.21 | 45.05 | +1.9% |
| gemma4-31b-q4 | 37.29 | 37.30 | +0.0% |

At these context lengths the KV cache is negligible next to the weights,
so the two framings agree inside run-to-run noise. Reproduce with
`--depth 51`.

### Suggested STATUS.md cells

bf16 where a bf16 GGUF exists, otherwise the nearest-precision quant,
always labelled — never an unlabelled quantized number next to bf16:

| STATUS row | cell |
|---|---|
| 1 gemma4-31B bf16 | `111` (bf16) |
| 2 gemma4-12B bf16 | `44.2` (bf16) |
| 3 gemma4-26B-A4B | `7.9` (Q4_0 QAT) |
| 5 Qwen3-8B | `15.7` (Q8_0) |
| 6 Llama-3.1-8B | `15.4` (Q8_0) |
| 7 gpt-oss-20b | `6.7` (MXFP4 native) |



## Build

Homebrew is not installed on this machine, so llama.cpp was built from
source with the Xcode toolchain and a `pip`-provided cmake:

```
git clone --depth 200 https://github.com/ggml-org/llama.cpp.git
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j 8 \
      --target llama-bench llama-completion llama-tokenize
```

`llama-completion` is required, not optional. In current builds `llama-cli`
is a full-screen interactive app: it renders a TUI, blocks on stdin, and
never prints the perf block — piping it gives you ASCII-art banners instead
of model output. The scriptable completion path is its own binary.

| | |
|---|---|
| commit | `221f0f6356efe2260023208365705ec5d5a7c8f5` |
| tag | **b10235** (well past the b9493 Gemma-4-12B-unified floor) |
| commit date | 2026-08-02 |
| ggml | 0.18.0 |
| compiler | AppleClang 21.0.0.21000101, `Darwin arm64` |
| backends | METAL + BLAS (Accelerate) |
| `llama-cli --version` | `version: 200 (221f0f6)` |

**The `version: 200` string is a shallow-clone artefact, not the build
number.** llama.cpp derives it from `git rev-list --count HEAD`, and the
clone was `--depth 200`. The tag at that commit is `b10235`; use the commit
hash, not the printed version, to identify this build.

Machine: Apple M5 Max, 128 GB unified memory, macOS 26.5.2 (25F84).
`llama-bench --list-devices` reports `MTL0: Apple M5 Max (110100 MiB,
110099 MiB free)` — that ~107.5 GiB working-set ceiling is what bounds the
largest row here.

## Method

Timed numbers come from `llama-bench` with the tool's default settings
(`-ngl -1` = all layers on Metal, `-fa auto`, `-b 2048 -ub 512`, `f16` KV
cache, mmap load). Exact invocation per row:

```
llama-bench -m <gguf> -p 51 -n 128 -r 5 -o json
```

* `-p 51` — the standard suite prompt in `manifest.json` tokenizes to **51**
  tokens under the Qwen3 and Llama-3.1 tokenizers and **50** under Gemma 4,
  so one prefill length covers every row.
* `-n 128` — `manifest.json`'s `decode_tokens`.
* `-r 5` — llama-bench's default repetition count; the JSONL records the
  reported standard deviation for both phases.

llama-bench reports throughput; the suite reports latency, so the adapter
converts:

```
prefill_ms    = 1000 * 51 / pp_tok_s        # whole 51-token prefill
decode_ms_tok = 1000 / tg_tok_s             # warm per-token decode
```

`load_s` is `llama-cli`'s reported `load time` with the file warm in the
page cache (mmap; llama.cpp does not eagerly copy weights, which is why
these numbers are so far under the JAX rows'). `mem_gb` is the sum of the
Metal buffers llama.cpp reports (weights + KV + compute) — the closest
analogue of the metaljax column's `mx.get_active_memory()`.

`warmup_s` is null for every row: llama-bench does its own internal warmup
and llama.cpp has no compile step to amortise, unlike the JAX rows.

### Coherence check

After the timed section each row runs

```
llama-completion -m <gguf> -st -p "<manifest prompt>" -n 128 --temp 0 \
                 --no-warmup --no-display-prompt --jinja -cnv
```

(`--temp 0` = greedy, `-st` = single turn, `-cnv` = the model's own chat
template) and stores the first 600 characters in `sanity_text`.

`--jinja` is load-bearing: **all six Gemma 4 rows abort without it** with
`std::runtime_error: this custom template is not supported, try using
--jinja` — before emitting a token. The first pass of this column silently
recorded empty coherence checks for every Gemma row because of it.

`load_s` also comes from this run (`common_perf_print: load time`), which is
why it is quoted with the chat template and tokenizer initialisation
included.

Greedy
*token-id* agreement with the JAX rows is not checked and cannot be: these
are different weights (quantized) reached through a different tokenizer and
chat template.

## Model provenance

One provider per model family, pinned by commit:

| row | quant | repo (revision) | file | GB |
|---|---|---|---|---|
| `gemma4-12b-bf16` | BF16 | `ggml-org/gemma-4-12B-it-GGUF` (`7e0fbb82`) | `gemma-4-12B-it-BF16.gguf` | 23.83 |
| `gemma4-12b-q4` | Q4_0 QAT | `google/gemma-4-12B-it-qat-q4_0-gguf` (`29d09777`) | `gemma-4-12b-it-qat-q4_0.gguf` | 6.98 |
| `gemma4-31b-bf16` | BF16 | `ggml-org/gemma-4-31B-it-GGUF` (`4fa4fdf3`) | `gemma-4-31B-it-BF16.gguf` | 61.41 |
| `gemma4-31b-q8` | Q8_0 | `ggml-org/gemma-4-31B-it-GGUF` (`4fa4fdf3`) | `gemma-4-31B-it-Q8_0.gguf` | 32.64 |
| `gemma4-31b-q4` | Q4_0 QAT | `google/gemma-4-31B-it-qat-q4_0-gguf` (`59dde245`) | `gemma-4-31B_q4_0-it.gguf` | 17.65 |
| `gemma4-26b-a4b-q4` | Q4_0 QAT | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` (`d1c082be`) | `gemma-4-26B_q4_0-it.gguf` | 14.44 |
| `qwen3-8b-q8` | Q8_0 | `Qwen/Qwen3-8B-GGUF` (`7c41481f`) | `Qwen3-8B-Q8_0.gguf` | 8.71 |
| `qwen3-8b-q4` | Q4_K_M | `Qwen/Qwen3-8B-GGUF` (`7c41481f`) | `Qwen3-8B-Q4_K_M.gguf` | 5.03 |
| `llama31-8b-q8` | Q8_0 | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (`bf5b95e9`) | `…-Q8_0.gguf` | 8.54 |
| `llama31-8b-q4` | Q4_K_M | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (`bf5b95e9`) | `…-Q4_K_M.gguf` | 4.92 |
| `gpt-oss-20b-mxfp4` | MXFP4 (native) | `ggml-org/gpt-oss-20b-GGUF` (`ef9b12f2`) | `gpt-oss-20b-MXFP4.gguf` | 12.11 |

Sizes are decimal GB of the file on disk.

### Why these providers

* **Gemma 4 non-QAT — `ggml-org`.** llama.cpp's own conversions, so the
  quantization is the reference implementation of the format rather than a
  third party's recipe, and it is the only provider publishing 31B BF16 as a
  **single** file (57.2 GiB); unsloth splits it across two shards. Both
  providers' Q8_0 are byte-for-byte the same size (30.39 GiB), so nothing is
  lost by preferring ggml-org.
* **Gemma 4 Q4 — `google`, the QAT releases.** These are the quantization-
  aware-trained q4_0 weights named in the task, not post-training quants;
  they are the ones Google intends people to run at 4 bits.
* **Qwen3-8B — `Qwen` (upstream).** The model authors' own GGUFs, and they
  publish both quants this column needs.

  The provider spread that earlier research flagged is **not** in the plain
  quants. Checked at measurement time, `Q4_K_M` is 5,027,783,488 B (Qwen),
  5,027,784,512 (unsloth), 5,027,784,224 (bartowski), 5,027,783,968
  (lmstudio-community) — a **1 KB** window, i.e. GGUF metadata only. Same
  story for `Q8_0` (8,709,518,112–8,709,519,168). The GB-scale differences
  live in the *dynamic* recipes (unsloth's `UD-*`, imatrix variants), which
  are a different quantization and are deliberately not used here. Picking
  upstream costs nothing and removes the question.
* **Llama-3.1-8B-Instruct — `bartowski`.** Meta publishes no GGUF; bartowski
  is the most-downloaded conversion and carries a consistent Q8_0/Q4_K_M
  pair from one recipe (and again the plain quants agree with
  lmstudio-community and MaziyarPanahi to within 5 KB). Note the JAX rows
  load `unsloth/Meta-Llama-3.1-8B-Instruct` — same upstream weights,
  different packager.
* **gpt-oss-20b — `ggml-org`, native MXFP4.** The task's preferred case:
  llama.cpp runs the checkpoint's own MXFP4 blocks, with no requantization.
  This is the one row where llama.cpp and the mlx-lm column run the *same*
  numeric format, and where the metaljax cell is a dequantized-to-bf16
  41.8 GB monster.

## What this column does and does not measure

1. **Quantization.** Only `gemma4-12b-bf16` and `gemma4-31b-bf16` run the
   same numeric precision as the metaljax/mlx-lm cells. Every other row is
   4- or 8-bit and should never be quoted as "llama.cpp is N× metaljax"
   without the quant attached. The bf16 rows are the honest head-to-head.
2. **Decode context depth.** `llama-bench`'s `tg` test generates from an
   empty context (KV grows 0 → 128), while the suite's `decode_ms_tok`
   decodes after the 51-token prompt (51 → 179). See
   [depth sensitivity](#depth-sensitivity) for the measured size of this
   effect.
3. **Prompt content.** `-p 51` prefills 51 *synthetic* tokens, not the
   manifest text; prefill cost at this length is content-independent.
4. **No chat template in the timed path.** The JAX rows generate through a
   chat-templated sampler. At a 51-token prompt the handful of template
   tokens are noise, but they are not zero.
5. **`load_s` is an mmap, not a load.** llama.cpp maps the file and lets the
   GPU fault pages in; the JAX rows materialise every tensor. Comparing the
   two as "load time" flatters llama.cpp by construction.
