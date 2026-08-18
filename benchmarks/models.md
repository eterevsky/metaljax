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

| # | benchmark | 0.11.1 | 0.11.2 | 0.11.3 | 0.11.4 | 0.11.5 |
|---|---|---:|---:|---:|---:|---:|
| 1 | gemma4-31B | 363 | 350 | 237.5 | 301.6 | **235.5** |
| 2 | gemma4-12B | 101 | 97.1 | 92.5 | 92.9 ᴾ²⁷ | **92.3** |
| 3 | gemma4-26B-A4B (MoE) | 473 | 284 | 44.3 | 43.4 | **43.5** |
| 4 | gemma4-E2B | 28.9 | 29.5 | 27.5 | 27.0 | **27.2** |
| 5 | Qwen3-8B | 60.3 | 60.4 | 57.8 | 58.1 | **57.9** |
| 6 | Llama-3.1-8B | 58.6 | 57.3 | 54.2 | 54.7 | **54.5** |
| 7 | gpt-oss-20b | 220 | 222 | 22.2 | 22.0 | **21.7** |
| 8 | Qwen3.6-35B-A3B | ✗ | ✗ | ✗ | ✗ | **29.7** ᴳ |
| 9 | R1-Distill-32B | ✗ | ✗ | 217.7 | 214.4 | **210.3** ᴳ |
| 10 | DeepSeek-V2-Lite | ✗ | ✗ | ✗ | ✗ | **1871.1** ᴳ |
| 11 | Qwen3-0.6B maxtext decode | ✗ | 16.0 | 15.8 | 16.63 | **16.35** |
| 12 | Mixtral 8×7B | ✗ | ✗ | ✗ | ✗ | ✗ |
| 13 | E2B keras-int4 | 340 | 336 | 81.1 | 80.3 ᴾ²⁷ | **78.0** |
| 14 | qwix-int8 0.6B | 48.3 | 48.5 | 32.5 | 35.0 | **31.77** |
| 15 | qwix-int8 8B | ✗ | ✗ | ✗ | ✗ | **401.4** ᵛ |
| 16 | SigLIP 2 (fwd ms) | 248 | 93.4 | 82.9 | 87.9 | **88.37** |
| 17 | SD3.5 (ms/step, 512² / 1024²) | ✗ | ✗ | 1389 / 5141 | 1234.8 / 5781.6 | **1231.3 / 5696.8** |
| 18 | LoRA E2B (ms/step) | 417 | 407 | 407 | 360.2 ᴾ²⁷ | **370.7** |
| 19 | maxtext train 0.6B (ms/step) | ✗ | 440 | 440 | 469.7 ᴾ²⁷ | **460.2** |
| 20 | 235B-A22B 3-bit (mlx-only) | ✗ | ✗ | ✗ | ✗ | ✗ |

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

- ᴳ (0.11.5 column; first attested in the rc-era governor campaign) = measured under the memory governor (2026-08-17, frozen-gov7 ebe56e71): ORIGINAL jax implementations, no benchmark-code modifications; the rows that previously panicked (#7) or guard-killed (122 GB) now run under the no-panic contract. Row 10's per-token number is the governor campaign's own (1865 ms/tok, 88 GB peak, exit 0).

- **Every unmarked 0.11.5rc cell** was measured by the release gate on
  2026-08-17 on the frozen release dylib `ebe56e71…` — the SAME binary the ᴳ
  cells were measured on (the release build reproduces `frozen-gov7` byte for
  byte), one guarded process per row, machine lock held, `METALJAX_DEBUG=1`.
  Details and per-row ratios: `notes/release-gates-0.11.5.md` gate 5.

- ᶜ (rc-era note, column since overwritten by the 0.11.5 re-gate) **Row 7 was a bracketed cell.** Its first two samples of the evening read
  24.2 / 23.9 ms/tok; a governor-off arm read 22.1 and a fourth arm with the
  governor back ON read **21.9**, so the pair was the suite-context trap
  (CLAUDE.md item 12), not a governor cost. The cell is the bracketed value;
  the spread (21.9–24.2) is recorded in the gate document.

- ᶠ (rc-era note, column since overwritten by the 0.11.5 re-gate) **Rows 11, 14 and 19 were the P28 re-measure, at the HISTORICAL budgets.**
  The release gate found both decode rows guard-killing at the budgets every
  previous campaign used (22 > 20 GB, 26 > 25 GB) and attributed it, one
  variable at a time, to **P27's flush watermark** rather than the governor —
  their checkpoint load takes 134 hard flushes in one call, so P27 reads it as
  an eager main and lets the 14 GB it frees at its last flush stand in the
  pool for the rest of the process. **P28's benefit gate**
  (`METALJAX_FLUSH_EARN_MULT`, default 2, `notes/cpp-p28-benefit-gate.md`)
  bounds a program's pool by the live set it has demonstrated it CYCLES: that
  load earns 3.6 GB, the decode step earns the floor, and row 19's training
  step — which genuinely swings 13.6 GB a flush — keeps everything P27 gave
  it. All three cells above are **shipped defaults at the rows' own historical
  budgets** (20 / 25 / 48 GB), medians of three or more guarded runs:

  | row | budget | P27 (0.11.5 as gated) | **P28** | P25 semantics (the control) |
  |---|---:|---|---|---|
  | 11 | 20 GB | **0 of 6 complete**, 21–25 GB | **9 of 9**, 16.61–16.83 ms/tok, 9.6–19 GB | 9 of 9, 16.52–17.07, 7.6–17 GB |
  | 14 | 25 GB | guard kill at 26 GB | **4 of 4**, 31.82–32.13 ms/tok, 9.1–17 GB | completes, 32.14, 7.7–15 GB |
  | 19 | 48 GB | 456.1 ms/step, 25 GB | **459.2 / 458.4 / 462.5**, 25 GB | 811–834 ms/step |

  Row 11's peak is a sub-second LIVE transient in the orbax restore (~17 GB
  under *every* policy, P25 included) with whatever the pool holds standing
  beside it, and `mem_guard.sh` samples at 2 Hz — so single peak readings
  scatter and the completion counts, not the peaks, are the statement. Row
  19's `loss` / `loss_first` are identical to P27's to thirteen digits in all
  three runs.

  **Re-spotted on the combined build** (2026-08-18, `frozen-vendor-d651add3` —
  the same plugin linked against the vendored patched `libmlx_metaljax.dylib`),
  same historical budgets, one guarded process per row: **16.60** ms/tok
  (16 GB), **31.94** ms/tok (9.2 GB), **463.5** ms/step (25 GB), all exit 0 —
  single runs against the three-run spreads above and inside their noise (row
  14 inside, row 11 0.01 ms under, row 19 1.0 ms over its trio and inside the
  456–470 class), with row 19's loss bit-identical across all eight runs of the
  campaign. The cells stand on either library.

- ʷ **Row 15 is a WRONG-OUTPUT row, not a timing row.** Its memory blocker is
  gone (it completes, 79 GB peak, 0 governor refusals) and it decodes at
  369.7 ms/tok, but the text is `" fragment!!!!!!!"` = token ids
  `[12289, 0, 0, 0, 0, 0, 0, 0]`, and `!` is Qwen3's token 0 — i.e. the logits
  have collapsed to a constant and greedy `argmax` returns index 0. The number
  is therefore **not published as a cell**: timing a program that computes the
  wrong answer measures nothing. Row 14 is the same adapter, the same qwix
  int8 overrides and the same emits at 0.6B and is coherent (31.995 ms/tok,
  the gate-5 run of the same day); the two differ by ~10× in traffic per
  compiled unit and by tied-vs-untied logits. The "known MLX-quantization bug"
  label the governor campaign gave it does not exist and has been withdrawn
  (2026-08-03 exonerated the quantized dots, `7932b4d`).

  **Mechanism established 2026-08-17 evening** (`notes/row15-wrong-output-2026-08-17.md`
  §8, STATUS fn 34): **nondeterministic MLX command-buffer corruption at 8B
  traffic, on BOTH engines**, amplified into the collapse by qwix's per-tensor
  `absmax` scale — which clamps a zero scale but not a NaN one, so one bad
  element turns a whole tensor NaN. Ten prefills of the same loaded parameters
  in one process, on identical inputs, return **8 distinct first tokens and 2
  full collapses** on the native engine and **10 distinct** on Stage 1, while
  **row 14 returns the same token 10/10**. It is not our compiled path
  (`METALJAX_COMPILE=0` is worse), not the recognizer, not the chunked replay,
  and not the int8 arithmetic (every row-14/row-15 s8×s8→s32 contraction is
  bit-exact vs numpy on both stacks). The committed 8B **bf16** canary
  `notes/data/qwen3_8b_prefill_36layer.mlir` — no quantization, no checkpoint —
  still corrupts at today's shipped budgets, bit-identically on both engines.
  No fix at our level: this is the upstream MLX report. **The cell stays
  unpublished and the row stays ✗.** *(Superseded 2026-08-18 — the level
  became ours; see footnote ᵛ.)*

- ᵛ **Row 15 is FIXED as of 2026-08-18 — the level became ours.** Footnote ʷ
  above is the history. The wrong output was MLX's own command-buffer fence
  drop (`slicing.cpp:62`), fixed in our vendored patched MLX inside the
  native wheel; 10/10 deterministic first tokens and a coherent decode on
  the release binary, and the 401.4 ms/tok cell is the row's first honest
  timing. Mechanism + attestation: STATUS.md footnotes 35/36,
  `notes/mlx-patch-diagnosis.md`.

- **0.11.5** (2026-08-18 consolidated re-gate): every cell measured in one
  campaign on the release binary `frozen-vendor-d651add3` (native plugin +
  vendored patched `libmlx_metaljax`, tree `29bb8eb`), one guarded process
  per cell, historical budgets, token agreement PASS. Named items (row 1
  variance disposition, rows 3/8 suite-context brackets, row 10 spread,
  row 18 drift, row 15 first timing) in STATUS.md footnote 36; full report
  `~/.cache/metaljax-bench/logs/regate-0.11.5/models/`.
