# Model benchmark suite — tracking over time

> Convention (Oleg, 2026-08-16): the rightmost column always tracks
> **HEAD** — updated opportunistically whenever a row is measured, no
> forced re-runs; missing or semi-stale cells are acceptable and marked.
> Release columns are frozen snapshots and follow release rule 1
> (CLAUDE.md): every release cell must come from the release binary.

*One column per tracked run of the model suite (scripts/model_bench/).
Cells: metaljax warm decode ms/token (or the row's noted metric);
✗ = blocked, with the measured reason in STATUS.md footnotes at that
version's commit. Full per-run tables live in STATUS.md; raw JSONL in
notes/data/. Append a column per release / major optimization.*

| # | benchmark | 0.11.1 | 0.11.2 | 0.11.3 | 0.11.4 |
|---|---|---:|---:|---:|---:|
| 1 | gemma4-31B | 363 | 350 | 237.5 | 301.6 |
| 2 | gemma4-12B | 101 | 97.1 | 92.5 | 98.6 |
| 3 | gemma4-26B-A4B (MoE) | 473 | 284 | 44.3 | 43.4 |
| 4 | gemma4-E2B | 28.9 | 29.5 | 27.5 | 27.0 |
| 5 | Qwen3-8B | 60.3 | 60.4 | 57.8 | 58.1 |
| 6 | Llama-3.1-8B | 58.6 | 57.3 | 54.2 | 54.7 |
| 7 | gpt-oss-20b | 220 | 222 | 22.2 | 22.0 |
| 8 | Qwen3.6-35B-A3B | ✗ | ✗ | ✗ | ✗ |
| 9 | R1-Distill-32B | ✗ | ✗ | 217.7 | 214.4 |
| 10 | DeepSeek-V2-Lite | ✗ | ✗ | ✗ | ✗ |
| 11 | Qwen3-0.6B maxtext decode | ✗ | 16.0 | 15.8 | 16.63 |
| 12 | Mixtral 8×7B | ✗ | ✗ | ✗ | ✗ |
| 13 | E2B keras-int4 | 340 | 336 | 81.1 | 79.7 |
| 14 | qwix-int8 0.6B | 48.3 | 48.5 | 32.5 | 35.0 |
| 15 | qwix-int8 8B | ✗ | ✗ | ✗ | ✗ |
| 16 | SigLIP 2 (fwd ms) | 248 | 93.4 | 82.9 | 87.9 |
| 17 | SD3.5 (ms/step, 512² / 1024²) | ✗ | ✗ | 1389 / 5141 | 1234.8 / 5781.6 |
| 18 | LoRA E2B (ms/step) | 417 | 407 | 407 | 394.7 |
| 19 | maxtext train 0.6B (ms/step) | ✗ | 440 | 440 | 1006.2 |
| 20 | 235B-A22B 3-bit (mlx-only) | ✗ | ✗ | ✗ | ✗ |

Notes:

- **0.11.1** (2026-08-02): pre-buffer-fix era, mixed command-buffer
  caps; raw data lost to the panic reboots — values from STATUS
  history. Row 16 measured under machine contention.
- **0.11.2** (2026-08-03): sequential release-gate run at shipped
  defaults; token agreement audited.
- **0.11.3** (2026-08-05 release gate, tree 0012813): steady-state
  harness — prefill/decode absorb per-shape executable builds before
  timing (one-time cost reported as build_s; materially this moved
  only the quantized rows). What landed: qmm quantized-matmul
  recognizer incl. native MXFP4 (rows 7/13/14), MoE expert gather
  incl. the T=1 decode fix (rows 3/7), sdpa fused attention + the
  eager memory stack (row 17 first-ever), streaming keras load +
  the retry double-residency fix (row 9 first-ever). Rows 18/19
  carried from 0.11.2 (not re-measured this gate). Rows 8/12 paused
  after kernel panic #7 (machine-wedge class; size-ladder protocol in
  TASKS.md); rows 10/15 blocked on the maxtext 8B class (MLX
  command-buffer bug / phase-2 load transient); row 20 deferred.
- **0.11.4** = the **fully native PJRT plugin** (`plugin-native/`; wheel
  variant `METALJAX_WHEEL_PLUGIN=native` — C++ StableHLO parse + lowering,
  no Python engine in the loop). Sourced entirely from
  benchmarks/perf-2026-08-native-baseline.md's Table 3 and
  notes/rc-gates-2026-08-16.md, no new numbers invented for this column.
  Latest pass per row: P18 (rows 3, 6, 17), P19 (row 9, footnote 29 in
  STATUS.md), P20 (rows 13, 14, 16, 19), the 2026-08-16 RC
  spot-check (rows 5, 7, 18 — 58.1/22.0/394.7, each within ±1% of its
  P18/P20 cell; `METALJAX_DEBUG=1` confirmed 0 msl_scan plans on all
  three), and **P24, the same day** (rows 1, 2, 4, 11 — the four cells
  that had been carried at their P16 value since 2026-08-12). P24
  re-measured all four on the RC binary with a same-day Stage-1 control
  each, since their Stage-1 numbers were P16-era too:
  **301.6 / 98.6 / 27.0 / 16.63** against P16's 301.6 / 98.8 / 27.2 /
  16.67 — every cell reproduced inside 0.7 %, so the ceiling caveat is
  withdrawn and these are current numbers, not upper bounds.
  Their same-day Stage-1 controls read 242.4 / 93.8 / 27.0 / 16.39, so
  the native/Stage-1 ratios are 1.24 / 1.05 / 1.00 / 1.02 — unchanged
  from P16, and the row-1 gap is *not* an sdpa miss: on the 31B row the
  recognizer never fires (0 sdpa emits, 0 msl plans), while on the 12B
  row it does (8 fused attentions) and the timing is P16's to 0.2 %.
  Data: `notes/data/p24-stale-rows-2026-08-16.{json,csv}`.
  Rows 8/10/12/15 stay
  ✗ under the same embargoes as the metaljax column (kernel-panic /
  MLX command-buffer classes); row 20 is mlx-only and has never run on
  either metaljax stack. **Row 19's 1006.2 carries a known, shared,
  unfixed regression**: both stacks read ~2.2× their 0.11.3 anchor here
  because of the eager flush's `mx::clear_cache()` (root-caused, not
  the plugin under test — both recover to ~470 on the
  `METALJAX_FLUSH_CLEAR_MB` knob; `mx::set_cache_limit` is the
  candidate fix and awaits Oleg's sign-off). Rows 5 and 7's timings are
  reproducible to ±1%, but their greedy-token streams are not: the
  fused-attention recognizer emits are run-to-run nondeterministic
  (RC gate 1 finding, `METALJAX_RECOGNIZE=0` restores determinism) —
  the numbers above are timing only, not a token-identity claim.
  That nondeterminism does **not** extend to the four P24 rows: rows 2
  and 4 are token-identical to Stage 1 *and* to their P16 runs (64/64),
  and rows 1 and 11, which diverge from Stage 1 (at token 34 and token
  ~3), reproduce their own P16 native stream exactly — four days apart
  for row 1, and twice in a row for row 11. Row 1 takes no sdpa emit at
  all, so its divergence is the plain lowering's arithmetic order, not
  the fused path.
- Comparison stacks (2026-08-03, versions in
  scripts/model_bench/versions.lock.md — re-measure alongside major
  metaljax changes): mlx-lm 12B 58.3 / 31B 137 / MoE 17.0 / gpt-oss
  8.8 / V2-Lite 10.6 / Mixtral 52.8 / R1-32B 131.8; torch-MPS 12B
  67.6 / 31B 148.7 / SigLIP 29.8 / LoRA 135.6 / SD3.5 654 @512²,
  2998 @1024²; llama.cpp 12B-bf16 44.2 / 31B-bf16 111.2 (the
  bandwidth roofline, 439–555 GB/s effective on every dense row).
