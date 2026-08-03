# Model benchmark suite — tracking over time

*One row per tracked run of the model suite (scripts/model_bench/).
Cells: metaljax warm decode ms/token (or noted metric). Full
per-run tables live in STATUS.md at the referenced commit; raw JSONL
in notes/data/. Append a row per release / major optimization.*

| date | version | 12B | 31B | Qwen3-8B | E2B | gpt-oss | 26B-A4B (MoE) | E2B-int4 | SigLIP fwd | LoRA step | maxtext 0.6B | raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-02 | 0.11.1 + splat/while fixes (pre-buffer-fix era, mixed caps) | 101 | 363 | 60.3 | 28.9 | 220 | 473 | 340 | 248² | 417 | garbage³ | (lost to panic) |
| 2026-08-03 | **0.11.2 release gate** (ops=800, 512 MB) | **97.1** | **350** | **60.4** | **29.5** | **222** | **284** | **336** | **93.4** | **407** | **16.0** | notes/data/model-bench-0.11.2-final.jsonl |

² measured under machine contention; ³ MLX command-buffer corruption,
fixed in 0.11.2.

Comparison stacks (2026-08-03, versions in
scripts/model_bench/versions.lock.md — re-measure alongside major
metaljax changes): mlx-lm 12B 58.3 / 31B 137 / MoE 17.0;
torch-MPS 12B 67.6 / 31B 148.7 / SigLIP 29.8 / LoRA 135.6;
llama.cpp 12B-bf16 44.2 / 31B-bf16 111.2 (the bandwidth roofline,
439–555 GB/s effective on every dense row).
