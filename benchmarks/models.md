# Model benchmark suite — tracking over time

*One row per tracked run of the model suite (scripts/model_bench/).
Cells: metaljax warm decode ms/token (or noted metric). Full
per-run tables live in STATUS.md at the referenced commit; raw JSONL
in notes/data/. Append a row per release / major optimization.*

*Rows = benchmarks (STATUS.md numbering); one column per tracked
run; cells = metaljax warm decode ms/token unless noted; ✗ = blocked
(measured reason in STATUS footnotes at that date's commit).*

| # | benchmark | 0.11.1+fixes (2026-08-02)¹ | **0.11.2 gate** (2026-08-03) |
|---|---|---:|---:|
| 1 | gemma4-31B | 363 | **350** |
| 2 | gemma4-12B | 101 | **97.1** |
| 3 | gemma4-26B-A4B (MoE) | 473 | **284** |
| 4 | gemma4-E2B | 28.9 | **29.5** |
| 5 | Qwen3-8B | 60.3 | **60.4** |
| 6 | Llama-3.1-8B | 58.6 | **57.3** |
| 7 | gpt-oss-20b | 220 | **222** |
| 8 | Qwen3.6-35B-A3B | ✗ | ✗ |
| 9 | R1-Distill-32B | ✗ | ✗ |
| 10 | DeepSeek-V2-Lite | ✗ | ✗ |
| 11 | Qwen3-0.6B maxtext decode | ✗² | **16.0** |
| 12 | Mixtral 8×7B | ✗ | ✗ |
| 13 | E2B keras-int4 | 340 | **336** |
| 14 | qwix-int8 0.6B | 48.3 | **48.5** |
| 15 | qwix-int8 8B | ✗ | ✗ |
| 16 | SigLIP 2 (fwd ms) | 248³ | **93.4** |
| 17 | SD3.5 (ms/diff-step) | ✗ | ✗ |
| 18 | LoRA E2B (ms/step) | 417 | **407** |
| 19 | maxtext train 0.6B (ms/step) | ✗² | **440** |
| 20 | 235B-A22B 3-bit (mlx-only) | ✗ | ✗ |

¹ pre-buffer-fix era, mixed command-buffer caps; raw data lost to the
panic reboots — values from STATUS history.
² MLX command-buffer corruption (garbage output / loss mismatch),
fixed in 0.11.2. ³ measured under machine contention.

Comparison stacks (2026-08-03, versions in
scripts/model_bench/versions.lock.md — re-measure alongside major
metaljax changes): mlx-lm 12B 58.3 / 31B 137 / MoE 17.0;
torch-MPS 12B 67.6 / 31B 148.7 / SigLIP 29.8 / LoRA 135.6;
llama.cpp 12B-bf16 44.2 / 31B-bf16 111.2 (the bandwidth roofline,
439–555 GB/s effective on every dense row).
