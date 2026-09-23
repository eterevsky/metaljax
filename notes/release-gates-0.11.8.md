# metaljax 0.11.8 — RELEASE GATE REPORT

Re-gated 2026-09-23 on jax 0.11.2.  This report supersedes the 2026-09-21 gate (binary
`frozen-m2main-7434ae86`, jax 0.11.0; commit f34ac28): the pinned jax release moved to 0.11.2 and its
suite exposed a float-remainder bug, so the binary changed and every number below was re-measured on
the new one (release rule 1).

## 0. Provenance (release rule 1)

- Tree: main `6b142b7` (the jax 0.11.2 pin) = code `e96f861` (the fixes); clean at launch.
- Release binary: `~/.cache/metaljax-bench/frozen-j112main-12c5a8d1.dylib`, sha256 `715332a5ddcc139e…`,
  built under Xcode 27.0; `plugin-native/bazel-bin` and `src/metaljax/lib/` hold the same bytes.
- Vendored MLX: fork `vendor/0.32.0 @ 661b2e38` (unchanged since 2026-09-10).  Plugin XLA: `131bf41`
  (jax 0.11.0's) — jaxlib 0.11.2 negotiates PJRT minor 114 and VHLO 1.18 (`notes/jax-0.11.2-pin.md`).
- jax/jaxlib 0.11.2 in main's `.venv` and the bench and gemma benchmark venvs; the maxtext venv stays on
  jax 0.11.0 because no released flax imports on 0.11.2 (rows 10, 11's maxtext arm, 14, 15, 19).
- Every gate pins `METALJAX_PLUGIN_PATH`; the supplemental stage asserts the sha.  Drivers:
  `scripts/release/run_gates.sh` (GATE_DATE=2026-09-23) then `logs/gate-0.11.8b/stageC3_driver.sh`.

## Stage A — pinned jax test suite (jax-v0.11.2) → **PASS**

- 163 files, 23 min: **28,679 passed / 135 failed** / 6,120 skipped → 99.53 %, id-identical to the new
  whitelist `notes/data/pinned-0.11.2-failures.txt` (0 new, 0 fixed).
- Against 0.11.7's 0.11.0 set (129 ids): 5 no longer fail (4 in files jax removed, 1 changed test) and 11
  are added, all classified in `notes/jax-0.11.2-pin.md` — 9 complex-plane accuracy tests that jax-CPU
  fails identically, one f64-policy test, one TPU-only layout test.  The two genuine new failures (float
  remainder, int2) were fixed in e96f861, not listed.

## Stage B — texmo release anchors → **PASS**

- Correctness: **106/106** (25 via sensitivity scaling), 0 decline, 0 FAIL.
- Perf sweep (223 configs): **geomean 1.117×** vs the standing anchor (was 1.091× on 09-21); 118
  configs improved > 5 %; the worst reads 0.950× — the bimodal dispatch-floor config
  (`split.add(split.add(split.add(rnn.1.gelu…` b4 l1024) that has sampled ~0.89 / ~1.00 ms on every binary
  since 0.11.7.  No config regressed against 0.11.7.

## Stage C1 — the main model battery → **PASS**

`final_run.sh` (unmodified this run), 22.5 min.  Metaljax cells against the 0.11.7 release column, jax-CPU
on jax 0.11.2:

| row | benchmark | 0.11.7 | **0.11.8** | Δ | jax-CPU (0.11.2) |
|---|---|---:|---:|---:|---:|
| 1 | gemma4-31B | 126.1 | **123.1** | −2 % | ✗ |
| 2 | gemma4-12B | 57.3 | **56.2** | −2 % | 314.1 |
| 3 | gemma4-26B-A4B | 33.4 | **31.1** | −7 % | ✗ |
| 4 | gemma4-E2B | 24.0 | **16.9** | −30 % | 67.4 |
| 5 | Qwen3-8B | 42.0 | **36.0** | −14 % | 205.0 |
| 6 | Llama-3.1-8B | 42.2 | **37.3** | −12 % | 203.9 |
| 7 | gpt-oss-20b | 19.8 | **13.2** | −33 % | ✗ |
| 13 | gemma4-E2B keras-int4 (harness fix) | 77.0 ᵇ | **6.0** | | 67.4 |
| 14 | Qwen3-0.6B qwix-int8 | 29.88 | **26.37** | −12 % | 144.4 |
| 16 | SigLIP 2 fwd b1 | 86.68 | **41.86** | −52 % | 358.5 |
| 18 | LoRA gemma4-E2B | 362.1 | **119.8** | −67 % | 1360.2 |
| 19 | Qwen3-0.6B maxtext train | 444.6 | **362.8** | −18 % | 1406.4 |
| 11 (maxtext arm, not the headline) | Qwen3-0.6B | 12.33 | 8.03 | | 91.1 |

Token agreement vs jax-CPU: rows 2, 5, 6 AGREE 64/64; row 13 now AGREES 64/64 (jax-CPU 0.11.2 moved to
the metaljax stream); row 4 diverges at 51/64 (the certified-benign entry).  The runner's model step
reads FAIL for two reasons fixed in the harness, neither a regression: its row-11 map pointed at the
maxtext arm (now the keras headline; `final_run.sh` gains the keras row for the next gate) and the
keras cell is therefore "missing" from this battery (it is in stage C3).

## Stage C3 — the remaining model rows → **PASS**

| row | benchmark | 0.11.7 | **0.11.8** | ratio | note |
|---|---|---:|---:|---:|---|
| 15t | qwix-int8 8B | 388.4 | **267.7** | 0.69× | 8-token protocol as before; 15d forensics clean |
| 9 | R1-Distill-32B | 190.8 | **199.7** | 1.05× | the row's ±4 % scatter (185.4–199.7 across six draws on two binaries; 188.0 under the pre-T1 pin); stream identical to 0.11.7's |
| 8 | Qwen3.6-35B-A3B | 28.5 | **23.3** | 0.82× | |
| 17a | SD3.5 512² | 1249.3 | **462.5** | 0.37× | |
| 17b | SD3.5 1024² | 4961.6 | **2061.8** ˢ | 0.42× | in sequence 2081.0; ˢ standalone |
| 12 | Mixtral 8×7B | 85.6 | **70.8** | 0.83× | 110 GiB envelope, exit 0 |
| 20 | 235B-A22B 3-bit | 56.2 | **45.2** | 0.80× | envelope 110/114; the in-sequence attempt was refused cleanly at execute (114.3 vs 114.0 GB claimed, 12.5 GB of page cache after the Mixtral row); retried after a quiet settle: exit 0, pack-wave peak 103.4 GB, stream identical to 0.11.7's (the 09-21 cell on the previous binary read 43.3) |
| 10 | DeepSeek-V2-Lite (manifest workload) | 25.9 | **19.8** | 0.76× | presplit harness variant 19.1 |
| 11 | Qwen3-0.6B keras-hub (headline) | 9.0 ᵐ | **5.1** | | |
| 21 | Qwen3.8-27B bf16 | 154.9 (first cell) | **142.5** | 0.92× | |

Guard fires in the model stages: 0 during decode.  Panics: 0.

## Standalone re-draws and attributions (rerun-first rule)

| row | in-battery | standalone | verdict |
|---|---:|---:|---|
| 7 gpt-oss-20b | 13.2 | 13.2 / 13.2 | no suite-context spread this run |
| 3 gemma4-26B-A4B | 31.1 | 28.2 / 28.3 | suite context; the release cell stays the battery's, as in 0.11.7 |
| 18 LoRA gemma4-E2B | 119.8 | new binary 113.8 / 113.2 / 113.4 (jax 0.11.2) and 113.0 (jax 0.11.0); previous binary 113.7 / 113.0 (jax 0.11.2) | in-battery context: neither the binary nor jax 0.11.2 moves it; the release cell stays the battery's 119.8 with the standalone 113–114 named |

## Cross-release token streams

Every metaljax stream is **identical to the 2026-09-21 gate's** (battery rows 1–7, 13; supplemental rows
8, 9, 11, 12, 21; plus row 20 when measured): the remainder fix, the int2/i4 work and jax 0.11.2 moved no
generated token.  Hence the verdicts against 0.11.7 stand as probed on 09-21: rows 4/5/6/7/20/21
identical; row 2 flipped back to the 0.11.6 / jax-CPU stream; row 1 an exact 0-ULP tie at generated 45;
row 3 an undecided preamble position at generated 3 (precision-default perturbation 39 ULP median vs a
15-ULP margin; `highest` reproduces 0.11.7) — `logs/gate-0.11.8/probes/findings.md`.

## Consolidated disclosure list

1. **Precision default** (`METALJAX_MATMUL_PRECISION=high`, 44fa042): f32 exact, bf16/f16/quantized on
   the M5 accelerators (as mlx-lm and torch-MPS do); bf16 within ~1 ULP of the pinned kernels; `highest`
   restores the whole-arch pin.  Behind the row 1 and row 3 stream changes (items 4–5).
2. **Silent-wrongness fix: float remainder** (e96f861).  `stablehlo.remainder` on floats was inexact in
   every float dtype in every release (f32 1.2e-6, bf16 2.2, f16 inf where a/b overflows); now bit-exact
   vs fmod/jax-CPU.  No model row contains a float remainder, so no table number could move through it.
3. **int2/uint2 added; i4/ui4 semantics corrected to jax-CPU** (e96f861): float → sub-byte int saturates
   (it wrapped: 100.0 → int4 gave 4, CPU gives 7) and negate/not/shift/divide/abs/power wrap.  No model
   row uses the i4 dtype.
4. **Row 1**: exact 0-ULP tie at generated 45; accepted class.
5. **Row 3**: tie flip on an undecided position (no chat template), a consequence of item 1; no jax-CPU
   reference exists for the row.
6. **Row 13's history was a broken model** (keras-hub Gemma4 int4 "HOTFIX", still upstream): fixed in the
   harness (`scripts/model_bench/int4_fix.py`); earlier cells carry ᵇ; metaljax and jax-CPU now agree 64/64.
7. **Row 2** flipped back to the 0.11.6 / jax-CPU stream (0.11.7 disclosure 1 resolved).
8. **jax 0.11.2 behaviour changes**: half-precision linalg is refused by jax itself (new lax.linalg dtype
   rules), so the plugin's bf16/f16 linalg only serves jax 0.11.0/0.11.1; the effort compile options left
   jax's build options (refused by jax-CPU and metaljax alike); fori_loop bodies are no longer wrapped in
   `closed_call`.  The jax-CPU column moved with jax 0.11.2 (LoRA 2316 → 1360, SigLIP 537 → 359).
9. **Maxtext rows on jax 0.11.0**: flax ≤ 0.12.9 cannot import on jax 0.11.2 (hijax rename).
10. **Plugin XLA at jax 0.11.0's commit**; moving it is its own cycle.
11. **Suite-context spreads**, named in the tables: rows 3 and 18 read higher inside the battery than alone.
12. **Row 11 maxtext prefill +1.9 ms** (B3, environmental) — carried.
13. **msl_scan fixes since 0.11.7** (lane-scalar af7f140, invariant carry 7adb350) — number-neutral.
14. **Vendored MLX fork** 65cb64b8 → 661b2e38: 0005 donation, 0006 empty-command-buffer skip (wall-neutral),
    0007 transient byte cadence (opt-in, off).
15. **Not fixed, documented**: row 10's checkpoint load peaks 99–111 GB vs its 105 GB guard (cells die at
    load, never in decode); row 20 sits within a few hundred MB of its 114 GB ceiling (tonight's first cell was refused cleanly at execute; the retry on a settled machine fit);
    unsigned `shift_right_arithmetic` runs as a logical shift (pre-existing, found this cycle, spawned as
    its own task); keras rows' timed generate includes a padded-window prefill.

## Verdict

**PASS** (release rule 2: no regression on any suite or benchmark against 0.11.7; every disclosure is
stated above).  Every number in the 0.11.8 release table comes from `frozen-j112main-12c5a8d1` (release
rule 1), with jax 0.11.2 wherever the benchmark stack allows it.  All 21 rows have a cell; no row is at or
above 2× its like-for-like goal.  Awaiting Oleg: the greenlight, the upload, the push and tag, the fork
push, the keras-hub issue.
