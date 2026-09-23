# metaljax 0.11.8 — RELEASE GATE REPORT

Re-gated 2026-09-23 evening on jax 0.11.2.  This report supersedes the early-morning gate of the same
day (binary `frozen-j112main-12c5a8d1`, commit fcf27ce): the plugin's XLA moved to jax 0.11.2's commit
(09b5f01) and the integer shift / popcnt / clz fix landed (a66dbc9), so the binary changed and every
number below was re-measured on the new one (release rule 1).

## 0. Provenance (release rule 1)

- Tree: main `a66dbc9` (clean at launch) = fcf27ce + the XLA move (09b5f01) + the shift fix (a66dbc9).
- Release binary: `~/.cache/metaljax-bench/frozen-x918main-e2ffcc75.dylib`, sha256 `e2ffcc75806e15fe…`,
  built under Xcode 27.0; `plugin-native/bazel-bin` and `src/metaljax/lib/` hold the same bytes.
- Plugin XLA: `91888df6` (jax 0.11.2's pin, checkout `metaljax/xla-91888df6`, WORKSPACE build).  The
  plugin reports PJRT C API 0.115 and VHLO 1.20.0, so jaxlib 0.11.2 no longer downgrades programs to
  VHLO 1.18 (`notes/jax-0.11.2-pin.md`).  Vendored MLX: fork `vendor/0.32.0 @ 661b2e38` (unchanged
  since 2026-09-10).
- jax/jaxlib 0.11.2 in main's `.venv` and the bench and gemma benchmark venvs; the maxtext venv stays on
  jax 0.11.0 because no released flax imports on 0.11.2 (rows 10, 11's maxtext arm, 14, 15, 19).
- Every stage pins `METALJAX_PLUGIN_PATH` and asserts the sha.  Drivers: `logs/gate-0.11.8c/chain.sh`
  (stage V, `scripts/release/run_gates.sh` with GATE_DATE=2026-09-23c, stage C3, row 20 on a settled
  machine, row 18 standalone) and `logs/gate-0.11.8c/redraw.sh` (rows 6 and 12 standalone).

## Stage V — the combined tree before the gate → **PASS**

`execute_test.py` and `ingest_test.py` pass; `pytest tests/` **559 passed + 1 xfailed** (526 before the
shift fix's 33 new tests); `texmo_gate.py` 106/106.  Old (131bf41) vs new (91888df6) XLA on the
pre-shift tree had already matched on all four suites with an unchanged compile cost (`logs/xla918/`).

## Stage A — pinned jax test suite (jax-v0.11.2) → **PASS**

- 163 files, 22.5 min: **28,679 passed / 135 failed** / 6,120 skipped → 99.53 %, id-identical to the
  whitelist `notes/data/pinned-0.11.2-failures.txt` (0 new, 0 fixed).  The XLA move and the shift fix
  moved no test.
- Against 0.11.7's 0.11.0 set (129 ids): 5 no longer fail (4 in files jax removed, 1 changed test) and 11
  are added, all classified in `notes/jax-0.11.2-pin.md` — 9 complex-plane accuracy tests that jax-CPU
  fails identically, one f64-policy test, one TPU-only layout test.  The two genuine new failures (float
  remainder, int2) were fixed in e96f861, not listed.

## Stage B — texmo release anchors → **PASS**

- Correctness: **106/106** (18 via sensitivity scaling), 0 decline, 0 FAIL.
- Perf sweep (223 configs): **geomean 1.115×** vs the standing anchor; 116 configs improved > 5 %; the
  one config past −5 % reads 0.936× — the bimodal dispatch-floor config
  (`split.add(split.add(split.add(rnn.1.gelu…` b4 l1024) that has sampled ~0.89 / ~1.00 ms on every binary
  since 0.11.7.  Config for config against the morning gate's binary: 0.998×, none beyond ±5 %.  No
  config regressed against 0.11.7.

## Stage C1 — the main model battery → **PASS**

`final_run.sh` (now including row 11's keras arm), 22.9 min.  Metaljax cells against the 0.11.7 release
column, jax-CPU on jax 0.11.2:

| row | benchmark | 0.11.7 | **0.11.8** | Δ | jax-CPU (0.11.2) |
|---|---|---:|---:|---:|---:|
| 1 | gemma4-31B | 126.1 | **123.1** | −2 % | ✗ |
| 2 | gemma4-12B | 57.3 | **56.2** | −2 % | 314.2 |
| 3 | gemma4-26B-A4B | 33.4 | **31.1** | −7 % | ✗ |
| 4 | gemma4-E2B | 24.0 | **16.9** | −30 % | 67.5 |
| 5 | Qwen3-8B | 42.0 | **36.0** | −14 % | 212.5 |
| 6 | Llama-3.1-8B | 42.2 | **38.6** | −9 % | 208.0 |
| 7 | gpt-oss-20b | 19.8 | **13.2** | −33 % | ✗ |
| 11 | Qwen3-0.6B keras-hub (headline) | 9.0 ᵐ | **5.1** | | 28.6 |
| 13 | gemma4-E2B keras-int4 (harness fix) | 77.0 ᵇ | **6.0** | | 67.8 |
| 14 | Qwen3-0.6B qwix-int8 | 29.88 | **26.32** | −12 % | 143.0 |
| 16 | SigLIP 2 fwd b1 | 86.68 | **41.77** | −52 % | 361.5 |
| 18 | LoRA gemma4-E2B | 362.1 | **126.1** | −65 % | 1358.1 |
| 19 | Qwen3-0.6B maxtext train | 444.6 | **362.8** | −18 % | 1413.6 |
| 11 (maxtext arm, not the headline) | Qwen3-0.6B | 12.33 | 8.06 | | 94.2 |

Token agreement vs jax-CPU: rows 2, 5, 6, 11, 13 AGREE 64/64; row 4 diverges at 51/64 (the
certified-benign entry, `MODEL_TOKEN_KNOWN`).  The runner reads WARN for that entry alone.

## Stage C3 — the remaining model rows → **PASS**

| row | benchmark | 0.11.7 | **0.11.8** | ratio | note |
|---|---|---:|---:|---:|---|
| 15t | qwix-int8 8B | 388.4 | **265.3** | 0.68× | 8-token protocol as before; 15d forensics clean (first token " Paris", no bad layer) |
| 9 | R1-Distill-32B | 190.8 | **199.9** | 1.05× | the row's ±4 % scatter (185.4–199.9 across draws on three binaries; 188.0 under the pre-T1 pin); stream identical to 0.11.7's |
| 8 | Qwen3.6-35B-A3B | 28.5 | **23.4** | 0.82× | |
| 17a | SD3.5 512² | 1249.3 | **460.6** | 0.37× | |
| 17b | SD3.5 1024² | 4961.6 | **2090.4** | 0.42× | standalone 2094.7 |
| 12 | Mixtral 8×7B | 85.6 | **69.0** ˢ | 0.81× | ˢ standalone; 73.3 in sequence; 110 GiB envelope, exit 0 |
| 10 | DeepSeek-V2-Lite (manifest workload) | 25.9 | **19.6** | 0.76× | presplit harness variant 19.0 |
| 21 | Qwen3.8-27B bf16 | 154.9 (first cell) | **142.3** | 0.92× | |
| 20 | 235B-A22B 3-bit | 56.2 | **45.2** | 0.80× | own stage on a settled machine (8 GB claimed); envelope 110/114, peak 111.8 GB claimed, guard never fired, exit 0 |

Guard fires in the model stages: 0 during decode.  Panics: 0.

## Standalone re-draws and attributions (rerun-first rule)

| row | in sequence | standalone | verdict |
|---|---:|---:|---|
| 7 gpt-oss-20b | 13.2 | 13.1 / 13.1 | no suite-context spread |
| 3 gemma4-26B-A4B | 31.1 | 28.2 / 28.2 | suite context; the release cell stays the battery's, as in 0.11.7 |
| 6 Llama-3.1-8B | 38.6 | 34.5 / 34.6 | suite context (the morning gate's battery read 37.3); the release cell stays the battery's |
| 18 LoRA gemma4-E2B | 126.1 | 113.3 / 113.9 (morning: 113.0–113.8 on either binary and either jax) | suite context; the release cell stays the battery's |
| 12 Mixtral 8×7B | 73.3 (stage C3) | 69.0 | resolved alone; the table carries 69.0 ˢ, the supplemental-row convention |
| 17b SD3.5 1024² | 2090.4 (stage C3) | 2094.7 | no spread; the in-sequence cell stands |

## Cross-release token streams

Every metaljax stream is **identical to the morning gate's** (battery rows 1–7, 11, 13; supplemental rows
8, 9, 10 both arms, 12, 15t, 20, 21; the standalone draws of rows 3, 6, 7, 12): the XLA move and the
shift fix moved no generated token.  Hence the verdicts against 0.11.7 stand as probed on 09-21: rows
4/5/6/7/20/21 identical; row 2 flipped back to the 0.11.6 / jax-CPU stream; row 1 an exact 0-ULP tie at
generated 45; row 3 an undecided preamble position at generated 3 (precision-default perturbation 39 ULP
median vs a 15-ULP margin; `highest` reproduces 0.11.7) — `logs/gate-0.11.8/probes/findings.md`.

## Consolidated disclosure list

1. **Precision default** (`METALJAX_MATMUL_PRECISION=high`, 44fa042): f32 exact, bf16/f16/quantized on
   the M5 accelerators (as mlx-lm and torch-MPS do); bf16 within ~1 ULP of the pinned kernels; `highest`
   restores the whole-arch pin.  Behind the row 1 and row 3 stream changes (items 5–6).
2. **Silent-wrongness fix: float remainder** (e96f861).  `stablehlo.remainder` on floats was inexact in
   every float dtype in every release (f32 1.2e-6, bf16 2.2, f16 inf where a/b overflows); now bit-exact
   vs fmod/jax-CPU.  No model row contains a float remainder, so no table number could move through it.
3. **Silent-wrongness fix: integer shifts, popcnt, clz** (a66dbc9).  `shift_right_arithmetic` on
   unsigned integers ran as a logical shift (uint8 0x80 >> 1 gave 0x40, XLA 0xC0); the out-of-range
   guard compared the amount as signed int32, so negative, uint32 ≥ 2^31 and 64-bit ≥ 2^32 amounts took
   Metal's mod-width shift; the emulated i2/ui2/i4/ui4 shifted, counted and clz'd their 8-bit storage.
   Now exact vs jax-CPU (379 exhaustive forms), including XLA's int4-popcnt widening quirk.  The RNG's
   static-amount shifts are op-for-op unchanged and every model stream is identical.
4. **int2/uint2 added; i4/ui4 semantics corrected to jax-CPU** (e96f861): float → sub-byte int saturates
   (it wrapped: 100.0 → int4 gave 4, CPU gives 7) and negate/not/shift/divide/abs/power wrap.  No model
   row uses the i4 dtype.
5. **Row 1**: exact 0-ULP tie at generated 45; accepted class.
6. **Row 3**: tie flip on an undecided position (no chat template), a consequence of item 1; no jax-CPU
   reference exists for the row.
7. **Row 13's history was a broken model** (keras-hub Gemma4 int4 "HOTFIX", still upstream): fixed in the
   harness (`scripts/model_bench/int4_fix.py`); earlier cells carry ᵇ; metaljax and jax-CPU agree 64/64.
8. **Row 2** flipped back to the 0.11.6 / jax-CPU stream (0.11.7 disclosure 1 resolved).
9. **jax 0.11.2 behaviour changes**: half-precision linalg is refused by jax itself (new lax.linalg dtype
   rules), so the plugin's bf16/f16 linalg only serves jax 0.11.0/0.11.1; the effort compile options left
   jax's build options (refused by jax-CPU and metaljax alike); fori_loop bodies are no longer wrapped in
   `closed_call`.  The jax-CPU column moved with jax 0.11.2 (LoRA 2316 → 1358, SigLIP 537 → 362).
10. **Maxtext rows on jax 0.11.0**: flax ≤ 0.12.9 cannot import on jax 0.11.2 (hijax rename).
11. **Plugin XLA at jax 0.11.2's commit** (91888df6, 09b5f01): PJRT C API 0.115, VHLO 1.20 —
    `collective_reduce` (VHLO 1.19, not emitted by jax) now reaches the plugin and declines by name;
    two-operand `collective_broadcast` (1.20) compiles.  jaxlib 0.11.0 still loads the plugin.  Still a
    WORKSPACE build (XLA marks that mode "to be removed"; the next pin move is likely the bzlmod migration).
12. **Suite-context spreads**, named in the tables: rows 3, 6 and 18 read higher inside the battery than
    alone (the battery cells are the table's); row 12 read higher in the stage-C3 sequence (the table
    carries its standalone cell, ˢ).
13. **Row 11 maxtext prefill +1.9 ms** (B3, environmental) — carried.
14. **msl_scan fixes since 0.11.7** (lane-scalar af7f140, invariant carry 7adb350) — number-neutral.
15. **Vendored MLX fork** 65cb64b8 → 661b2e38: 0005 donation, 0006 empty-command-buffer skip (wall-neutral),
    0007 transient byte cadence (opt-in, off).
16. **Not fixed, documented**: row 10's checkpoint load peaks 99–111 GB vs its 105 GB guard (cells die at
    load, never in decode); row 20 runs within ~2 GB of its 114 GB ceiling (peak 111.8 GB), so the gate
    runs it as its own stage on a settled machine — in sequence it can be refused cleanly at execute;
    keras rows' timed generate includes a padded-window prefill.

## Verdict

**PASS** (release rule 2: no regression on any suite or benchmark against 0.11.7; every disclosure is
stated above).  Every number in the 0.11.8 release table comes from `frozen-x918main-e2ffcc75` (release
rule 1), with jax 0.11.2 wherever the benchmark stack allows it.  All 21 rows have a cell; no row is at or
above 2× its like-for-like goal.  Awaiting Oleg: the greenlight, the upload, the push and tag, the fork
push, the keras-hub issue.
