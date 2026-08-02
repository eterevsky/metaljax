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
the timed section, runs `llama-bench`, then runs one greedy `llama-cli`
generation as a coherence check, and appends a `backend="llamacpp"` record in
the same JSONL shape `run_bench.py` emits.

<!-- RESULTS -->

## Build

Homebrew is not installed on this machine, so llama.cpp was built from
source with the Xcode toolchain and a `pip`-provided cmake:

```
git clone --depth 200 https://github.com/ggml-org/llama.cpp.git
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j 8 \
      --target llama-bench llama-cli llama-tokenize
```

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
llama-cli -m <gguf> -st -p "<manifest prompt>" -n 128 --temp 0 --no-warmup -cnv
```

(`--temp 0` = greedy, `-st` = single turn, `-cnv` = the model's own chat
template) and stores the first 600 characters in `sanity_text`. Greedy
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
