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
| 2 | gemma4-12B | 101 | 97.1 | 92.5 | 92.9 ᴾ²⁷ |
| 3 | gemma4-26B-A4B (MoE) | 473 | 284 | 44.3 | 43.4 |
| 4 | gemma4-E2B | 28.9 | 29.5 | 27.5 | 27.0 |
| 5 | Qwen3-8B | 60.3 | 60.4 | 57.8 | 58.1 |
| 6 | Llama-3.1-8B | 58.6 | 57.3 | 54.2 | 54.7 |
| 7 | gpt-oss-20b | 220 | 222 | 22.2 | 22.0 |
| 8 | Qwen3.6-35B-A3B | ✗ | ✗ | ✗ | **29.6** ᴳ |
| 9 | R1-Distill-32B | ✗ | ✗ | 217.7 | 214.4 |
| 10 | DeepSeek-V2-Lite | ✗ | ✗ | ✗ | **completes** ᴳ (88 GB peak) |
| 11 | Qwen3-0.6B maxtext decode | ✗ | 16.0 | 15.8 | 16.63 |
| 12 | Mixtral 8×7B | ✗ | ✗ | ✗ | ✗ |
| 13 | E2B keras-int4 | 340 | 336 | 81.1 | 80.3 ᴾ²⁷ |
| 14 | qwix-int8 0.6B | 48.3 | 48.5 | 32.5 | 35.0 |
| 15 | qwix-int8 8B | ✗ | ✗ | ✗ | ✗ |
| 16 | SigLIP 2 (fwd ms) | 248 | 93.4 | 82.9 | 87.9 |
| 17 | SD3.5 (ms/step, 512² / 1024²) | ✗ | ✗ | 1389 / 5141 | 1234.8 / 5781.6 |
| 18 | LoRA E2B (ms/step) | 417 | 407 | 407 | 360.2 ᴾ²⁷ |
| 19 | maxtext train 0.6B (ms/step) | ✗ | 440 | 440 | 469.7 ᴾ²⁷ |
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
  either metaljax stack. **Row 19 = P25 (2026-08-16), 1006.2 → 833.9**:
  the eager flush now TRIMS MLX's buffer pool back to
  `METALJAX_FLUSH_CLEAR_MB` instead of dumping the whole pool to the OS
  (`runtime.cc::trim_cache`; notes/cpp-p25-cache-limit.md). Measured in
  one hold beside a same-day control on the RC binary, which reads
  **975.4** — so the mechanism is worth 1.17× and the cell's remaining
  1.9× of the 440 anchor is the WATERMARK, not the dumping: the same row
  reads 685.6 at an 8 GB watermark and 464.1 at 32 GB, and 32 GB is where
  the LoRA row (18) blows through its 70 GB guard. The default is left at
  2048 MB, which is strictly better than the shipped dump at the same
  peak (20-21 GB here); raising it is Oleg's call off that table.
  **Rows 2/13/18/19 = P27 (2026-08-16), and they retire that table**
  (notes/cpp-p27-flush-pressure.md): the watermark is no longer one number
  for every program, because the LoRA row's blowout turned out not to be a
  pool. Measured with a footprint meter inside the dylib, its live set goes
  19.6 → 46.5 GB in about a second during the keras build/convert phase —
  identically on both binaries — and the watermark decides only how much
  DEAD pool is standing beside that spike (1.5 GB at 2048, 16.2 at 32768,
  which is the whole difference between 48.5 and 63.2 GB of footprint).
  So the cap moves to 32768 and two rules spend it: a program must have
  taken 8 hard flushes to count as an eager main, and even then the pool may
  claim only what a 48 GB (3/8 of RAM) footprint target has left after its
  own live set. P25's 2048 is the FLOOR under both, so nothing is trimmed
  harder than it was. **Row 19 833.9 → 469.7** (five runs 460-478, peak
  25 GB under a 48 GB budget, against 811.6 for the same binary with the
  policy off) and **row 18 394.7 → 360.2** (five runs, peak unchanged: its
  meter reads 56.7-57.5 GB either way). Rows 13 and 2 are the controls —
  80.3 vs 80.2 and 92.9 vs 93.2 with the policy off, peaks unmoved — and
  row 2's distance from its 98.6 P24 cell is the tree, not this policy.
  Suite-106 same-binary policy-on/off geomean **0.9983** over 106,
  `texmo_gate` 106/106 three times, 0 buffer-limit recoveries in a
  106-config sweep on either policy.
  **Stage 1 still dumps** — its copies of the flush are frozen, so the
  backport is a separate decision and any same-day native/Stage-1 ratio
  on an eager-main row now has this in it. Rows 5 and 7's timings are
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

- ᴳ = first measured under the memory governor (2026-08-17, frozen-gov7 ebe56e71): ORIGINAL jax implementations, no benchmark-code modifications; the rows that previously panicked (#7) or guard-killed (122 GB) now run under the no-panic contract. Row 10's per-token number pending the release-gate sweep; the governor campaign recorded the completion + peak.
