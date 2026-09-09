# Model benchmark suite — tracking over time

> Convention (Oleg, 2026-08-16): the rightmost column always tracks
> **HEAD** — updated opportunistically whenever a row is measured, no
> forced re-runs; missing or semi-stale cells are acceptable and marked.
> Release columns are frozen snapshots and follow release rule 1
> (CLAUDE.md): every release cell must come from the release binary.

*One column per tracked run of the model suite (scripts/model_bench/).
Cells: metaljax warm decode ms/token (or the row's noted metric);
✗ = blocked (the measured reason is in that release's gate record,
notes/release-gates-<version>.md). Comparators, footnotes and the current
release's cells: STATUS.md. Append a column per release / major
optimization.*

| # | benchmark | 0.11.1 | 0.11.2 | 0.11.3 | 0.11.4 | 0.11.5 | 0.11.6 | 0.11.7 | HEAD | goal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | gemma4-31B | 363 | 350 | 237.5 | 301.6 | 235.5 | 235.2 | **126.1** | 125.5 ʰ | 111.2 ˡ |
| 2 | gemma4-12B | 101 | 97.1 | 92.5 | 92.9 ᴾ²⁷ | 92.3 | 92.1 | **57.3** | 56.6 ʰ | 44.2 ˡ |
| 3 | gemma4-26B-A4B (MoE) | 473 | 284 | 44.3 | 43.4 | 43.5 | 43.3 | **33.4** | 28.9 ʰ | 16.9 ˡ |
| 4 | gemma4-E2B | 28.9 | 29.5 | 27.5 | 27.0 | 27.2 | 27.2 | **24.0** | 17.0 ʰ | 10.5 ˣ |
| 5 | Qwen3-8B | 60.3 | 60.4 | 57.8 | 58.1 | 57.9 | 57.6 | **42.0** | 40.3 ʰ | 29.6 ˡ |
| 6 | Llama-3.1-8B | 58.6 | 57.3 | 54.2 | 54.7 | 54.5 | 54.3 | **42.2** | 37.6 ʰ | 29.2 ˡ |
| 7 | gpt-oss-20b | 220 | 222 | 22.2 | 22.0 | 21.7 | 21.3 | **19.8** | 13.2 ʰ | 8.8 ˣ |
| 8 | Qwen3.6-35B-A3B | ✗ | ✗ | ✗ | ✗ | 29.7 ᴳ | 29.4 ᴳ | **28.5** ᴳ | 21.3 ʰ | 13.7 ˣ |
| 9 | R1-Distill-32B | ✗ | ✗ | 217.7 | 214.4 | 210.3 ᴳ | 211.0 ᴳ | **190.8** ᴳ | 188.0 ʰ | 114.9 ˡ |
| 10 | DeepSeek-V2-Lite ᵖ | ✗ | ✗ | ✗ | ✗ | 1871.1 ᴳ | 1948.2 ᴳ | **25.9** ᵖ | 22.8 ʰ | 10.5 ˣ |
| 11 | Qwen3-0.6B decode ᵐ | ✗ | 16.0 ᵐ | 15.8 ᵐ | 16.63 ᵐ | 16.35 ᵐ | 16.35 ᵐ | **12.33** ᵐ | 5.2 ʰ | 3.2 ˣ |
| 12 | Mixtral 8×7B | ✗ | ✗ | ✗ | ✗ | ✗ | 91.3 ᴳ | **85.6** ᴳ | | |
| 13 | gemma4-E2B keras-int4 | 340 | 336 | 81.1 | 80.3 ᴾ²⁷ | 78.0 | 78.0 | **77.0** | | |
| 14 | Qwen3-0.6B qwix-int8 | 48.3 | 48.5 | 32.5 | 35.0 | 31.77 | 31.85 | **29.88** | 26.7 ʰ | |
| 15 | Qwen3-8B qwix-int8 | ✗ | ✗ | ✗ | ✗ | 401.4 ᵛ | 381.7 ᵛ | **388.4** ᵛ | | |
| 16 | SigLIP 2 (fwd ms) | 248 | 93.4 | 82.9 | 87.9 | 88.37 | 88.31 | **86.68** | 42.0 ʰ | 29.8 ᵗ |
| 17 | SD3.5 (ms/step, 512² / 1024²) | ✗ | ✗ | 1389 / 5141 | 1234.8 / 5781.6 | 1231.3 / 5696.8 | 1234.7 / 4974.9 | **1249.3 / 4961.6** | | 553 / 3078 ᵗ |
| 18 | LoRA gemma4-E2B (ms/step) | 417 | 407 | 407 | 360.2 ᴾ²⁷ | 370.7 | 369.2 | **362.1** | 113.2 ʰ | 135.6 ᵗ |
| 19 | Qwen3-0.6B maxtext train (ms/step) | ✗ | 440 | 440 | 469.7 ᴾ²⁷ | 460.2 | 463.4 | **444.6** | 376.1 ʰ | |
| 20 | Qwen3-235B-A22B 3-bit (mlx quant) | ✗ | ✗ | ✗ | ✗ | ✗ | 66.3 ᴳ | **56.2** ᴳ | | 28.0 ˣ |
| 21 | Qwen3.8-27B bf16 | — | — | — | — | — | — | **154.9** | 145.5 ʰ | 98.2 ˡ |

Notes:

- **goal** = the best non-metaljax cell for the row in STATUS.md's table at
  the same precision and workload (STATUS.md's like-for-like rule, fn 10):
  ˣ mlx-lm, ˡ llama.cpp, ᵗ torch-MPS; jax-CPU is never a goal and rows with
  no other framework stay empty. It sits last, beside HEAD, so a row's
  current status is its last two cells. Standing caveats (STATUS.md has the
  detail): row 4's goal is a dated mlx-lm cell whose run record is lost
  (fn 7); row 12's goal is empty until mlx-lm is measured on a bf16
  checkpoint (the mirror is float16, fn 17); row 7's goal is mlx-lm because
  the llama.cpp GGUF quantizes attention/embeddings/head to Q8_0 (fn 16);
  row 17's torch goal runs with the T5 encoder off, matching our context
  (fn 18); the llama.cpp cells carry a 4 % two-pass band, wider than their
  lead over mlx-lm on rows 3 and 6.
- ʰ = HEAD-column cells: rerun-first medians of ≥ 2 runs on a frozen build
  of main, token streams identical to the row's release record unless
  stated. As of 2026-09-09 (late): rows 16/18/19 and the sentinels 4/7/11 are on
  the precision-default build (METALJAX_MATMUL_PRECISION=high: f32 stays
  exact, bf16/f16 and quantized matmuls use the M5 accelerators, as mlx-lm
  and torch-MPS do); row 10 on the ragged-decode build; every other row on
  the projection-pack build ad9507e, all with a 60 s cool-down before each
  row.  Decode sentinels on the precision build read in band (row 11 5.1,
  row 4 16.7, row 7 13.2) with the first 64 token ids identical; rows 4 and
  7 end generation a few tokens earlier past that window (a bf16
  accumulation-order tie).  Rows 1/2/3 read within run spread of their
  week-old cells; knob-off arms cleared every merge.  Row 12 has no cell:
  the governor refused cleanly at 106.4 GB of claimed memory against its
  105 GB ceiling (the desktop held 18 GB); it needs a lighter machine or a
  raised ceiling with a matching guard.
- ᵐ **Row 11 changed benchmark implementation after 0.11.7** (the
  best-available-implementation rule): every cell through the 0.11.7 column
  is the maxtext decode harness and is NOT comparable to the keras-hub cells
  that follow — the harness switch is worth ~1.36× by itself (same-session
  control: maxtext 12.22 vs keras 9.0 on one binary), so no release-over-
  release comparison may span it. The same applies to the row's jax-CPU
  history (maxtext 89.7 → keras-hub 29.4).
- ᵖ **Row 10 changed workload on 2026-09-06**: the cell now decodes the
  manifest prompt (50 tokens in a 64-slot prefill) for 128 tokens like the
  comparators (25.9 on the 0.11.7 release binary); every earlier cell used
  the adapter's 5-token default prompt for 8 tokens (24.8 on the same
  binary) and is not comparable across the switch. The harness's per-step
  RNG split is worth ~0.5 ms/token on this row; the cell keeps the original
  harness (MAXTEXT_RNG_PRESPLIT=1 is the reported variant).
- ᴳ = measured under the memory governor (from 2026-08-17): the original jax
  implementation with no benchmark-code changes; rows that previously
  panicked or were guard-killed run under the no-panic contract.
- ᴾ²⁷ (0.11.4 column) = re-measured on the 2026-08-16 flush-pressure build
  (notes/cpp-p27-flush-pressure.md) rather than the release-gate run.
- ᵛ (row 15) = a timing cell; before 2026-08-18 the row produced wrong
  output (MLX's dropped command-buffer fence, fixed in the vendored MLX,
  notes/mlx-patch-diagnosis.md) and its earlier cells were not timings.
