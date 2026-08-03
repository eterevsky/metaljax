# Model benchmark suite — tracking over time

*One row per tracked run of the model suite (scripts/model_bench/).
Cells: metaljax warm decode ms/token (or noted metric). Full
per-run tables live in STATUS.md at the referenced commit; raw JSONL
in notes/data/. Append a row per release / major optimization.*

*Columns are STATUS.md row numbers (legend below); cells = metaljax
warm decode ms/token unless noted; ✗ = blocked (measured reason in
STATUS footnotes at that date's commit).*

| date | version | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---:|---:|---|---:|---|---:|---:|---|
| 2026-08-02 | 0.11.1+fixes¹ | 363 | 101 | 473 | 28.9 | 60.3 | 58.6 | 220 | ✗ | ✗ | ✗ | ✗² | ✗ | 340 | 48.3 | ✗ | 248³ | ✗ | 417 | ✗² | ✗ |
| 2026-08-03 | **0.11.2 gate** | **350** | **97.1** | **284** | **29.5** | **60.4** | **57.3** | **222** | ✗ | ✗ | ✗ | **16.0** | ✗ | **336** | **48.5** | ✗ | **93.4** | ✗ | **407** | **440** | ✗ |

Legend: 1 gemma4-31B · 2 gemma4-12B · 3 gemma4-26B-A4B (MoE) ·
4 gemma4-E2B · 5 Qwen3-8B · 6 Llama-3.1-8B · 7 gpt-oss-20b ·
8 Qwen3.6-35B-A3B · 9 R1-Distill-32B · 10 DeepSeek-V2-Lite ·
11 Qwen3-0.6B maxtext decode · 12 Mixtral 8×7B · 13 E2B keras-int4 ·
14 qwix-int8 0.6B · 15 qwix-int8 8B · 16 SigLIP 2 fwd ms ·
17 SD3.5 ms/diff-step · 18 LoRA E2B ms/step ·
19 maxtext train 0.6B ms/step · 20 235B-A22B 3-bit (mlx-only row).

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
