# metaljax 0.11.8 — RELEASE GATE REPORT

## 0. Provenance (release rule 1)

- Tree: main `3eb6cc5` (tables) = code `e46bc94`; working tree clean (0 dirty files at launch).
- Release binary: `~/.cache/metaljax-bench/frozen-m2main-7434ae86.dylib`, sha256 `61abb742e52bfbd5…`,
  built from that tree under Xcode 27.0 (the 09-21 upgrade; every earlier frozen dylib is ABI-stale);
  `plugin-native/bazel-bin` and the dev copy `src/metaljax/lib/libmetal_pjrt_native.dylib` are the same bytes.
- Vendored MLX: fork `vendor/0.32.0 @ 661b2e38` (restaged 2026-09-10; patches 0005 donation, 0006
  skip-empty-finalize, 0007 transient byte cadence on top of 0.11.7's 65cb64b8).
- Every gate pins `METALJAX_PLUGIN_PATH` at the frozen path; the supplemental stage asserts the sha.
- Driver: `scripts/release/run_gates.sh` (GATE_DATE=2026-09-21) then `logs/gate-0.11.8/stageC3_driver.sh`.

## Stage A — pinned jax test suite → **PASS** (id-identical failure set)

- `--jobs 1`, 164 files, 31.5 min: **28,073 passed / 129 failed** / 6,161 skipped → 99.54 % — the same
  totals as 0.11.7 and 0.11.6.
- The runner's whitelist is still `notes/data/pinned-0.11.0-failures.txt` (130 ids), so it reports
  "12 new / 13 fixed" exactly as the 0.11.7 gate did; against `notes/data/pinned-0.11.6-failures.txt`
  (the 129 ids 0.11.7 shipped with) the set is **id-identical: 0 new, 0 fixed**.
- The precision default (T1: f32 exact, bf16/f16/quantized on NAX) moved no jax test.
- Housekeeping for Oleg: re-point `scripts/release/jax_suite.sh` at the 0.11.6 list (or refresh the
  0.11.0 whitelist) so the step stops reporting the same stale 12/13.

## Stage B — texmo release anchors → **PASS**

- Correctness (`plugin-native/texmo_gate.py`, 106 whole-model configs vs jax-CPU, 1-ULP sensitivity
  tolerance): **106 ok** (22 via sensitivity scaling), 0 decline, 0 FAIL, 0 error.
- Perf sweep (`scripts/bench_texmo_pjrt.py`, the 223-config top_confs set, 17.8 min):
  **geomean 1.091x** vs the standing anchor `topconfs16k-metal-2026-08-22.jsonl` (>1 = faster);
  82 configs improved > 5 %, 8 configs read below 0.97x in sequence (worst 0.886x:
  `bits.1+bp|split.add(split.add(split.add(rnn.1.gelu…` fp32 b4 l1024, 0.887 -> 1.001 ms), thresholds
  geomean >= 0.97 / no config > 1.3x slower both met.  The 8 are sub-2 ms rows of the documented
  suite-context class; they are re-run standalone after the supplemental model rows
  (`gate-0.11.8/texmo_rerun.sh`, two passes) and the standalone readings replace the in-sequence
  ones here:

| config (fp32) | anchor 08-22 | in-sequence | standalone ×2 | 0.11.7 gate's own record | verdict |
|---|---:|---:|---:|---:|---|
| `tokens…emb.8\|rnn.16.gelu-mingru.8…` b64 l256 | 0.794 | 0.830 | 0.795 / 0.827 | — | context; standalone = anchor |
| `tokens…oh\|rnn.32.gelu-split.add…` b64 l128 (#1) | 1.853 | 1.959 | 1.863 / 1.862 | — | context; standalone = anchor |
| `tokens…oh\|rnn.32.gelu-split.add…` b64 l128 (#2) | 1.999 | 2.059 | 1.989 / 1.985 | — | context; standalone = anchor |
| `tokens…oh\|rnn.32.gelu-dense.32.gelu` b128 l128 | 0.513 | 0.530 | 0.525 / 0.527 | — | −2.3 %, inside the band |
| `tokens…oh\|rnn.32.gelu-split.add…` b256 l64 | 3.697 | 3.879 | 3.865 / 3.866 | — | −4.3 % vs the 0.11.6-era anchor |
| `bits.1+bp\|rnn.1.tanh-suffix.4-dense.1.tanh` b16 l256 | 0.313 | 0.336 | 0.334 / 0.335 | 0.334 / 0.337 / 0.337 | unchanged since 0.11.7 |
| `bits.1+bp\|split.cat(rnn.1.tanh…)` b16 l512 (tc010-w17) | 0.567 | 0.612 | 0.612 / 0.610 | 0.615 / 0.615 / 0.616 | unchanged since 0.11.7 (its accepted micro-regression) |
| `bits.1+bp\|split.add(split.add(split.add(rnn.1.gelu…` b4 l1024 | 0.887 | 1.001 | 1.000 / 1.007 | 0.937 / 0.897 / 1.014 (standalone 1.003) | unchanged since 0.11.7 |

No texmo config regressed against 0.11.7; the four that sit below the 2026-08-22 anchor were there at
0.11.7 (three sub-ms dispatch-floor rows and one 3.9 ms row).  Knob attribution arms for them (loop
specialization off / the pre-T1 precision pin / both) ran in stage C6: every arm is within ±1 % of the
plain standalone reading on seven of the eight configs (no knob moves them); the eighth,
`split.add(split.add(split.add(rnn.1.gelu…` b4 l1024, is bimodal at ~0.89 / ~1.00 ms on every binary
that has measured it (0.11.7's three runs read 0.937 / 0.897 / 1.014; tonight's arms 1.011 / 0.892 /
0.995 / 1.002 with no pattern across knobs) — a dispatch-floor row sampling two modes, not a knob effect.
- Record: `notes/data/texmo-topconfs-2026-09-21.jsonl`.

## Stage C1 — the main model battery → **PASS** (0 regressions, 13 rows improved)

`scripts/model_bench/final_run.sh` unmodified, one guarded process per cell, lock per cell, 22 min
(the machine was free).  Timing vs the 0.11.7 release column (the runner's first table compared
against the goal column by mistake -- fixed in d4626f6 and regenerated; the goal-column table is kept
as `model_gate.goalcol.md`):

| row | benchmark | 0.11.7 | **0.11.8** | Δ | CPU (0.11.8) |
|---|---|---:|---:|---:|---:|
| 1 | gemma4-31B | 126.1 | **123.1** | −2 % | ✗ |
| 2 | gemma4-12B | 57.3 | **56.5** | −1 % | 316.1 |
| 3 | gemma4-26B-A4B | 33.4 | **31.1** | −7 % | ✗ |
| 4 | gemma4-E2B | 24.0 | **16.9** | −30 % | 67.6 |
| 5 | Qwen3-8B | 42.0 | **36.0** | −14 % | 211.0 |
| 6 | Llama-3.1-8B | 42.2 | **37.5** | −11 % | 206.3 |
| 7 | gpt-oss-20b | 19.8 | **15.6** | −21 % | ✗ |
| 11 | Qwen3-0.6B maxtext arm (not the headline) | 12.33 | 8.12 | −34 % | 90.5 |
| 13 | gemma4-E2B keras-int4 (harness fix; earlier cells were a broken model) | 77.0 ᵇ | **6.0** | | 67.9 |
| 14 | Qwen3-0.6B qwix-int8 | 29.88 | **26.39** | −12 % | 142.6 |
| 16 | SigLIP 2 fwd b1 | 86.68 | **41.67** | −52 % | 536.9 |
| 18 | LoRA gemma4-E2B | 362.1 | **112.1** | −69 % | 2316.3 |
| 19 | Qwen3-0.6B maxtext train | 444.6 | **362.6** | −18 % | 1396.8 |

Token agreement vs jax-CPU (`compare_tokens.py`): rows 2, 5, 6 AGREE 64/64; row 4 diverges at 51/64
(the certified-benign entry, MODEL_TOKEN_KNOWN; the same stream 0.11.7 recorded); row 13 diverges at
51/64 (quantized class; see the 1-ULP tie in fn 8).  Guard fires: 0.  Governor refusals: 0.  Panics: 0.
Rows 7 and 3 read above their HEAD cells in sequence (15.6 vs 13.2, 31.1 vs 28.9): stage C4 re-draws
them standalone (rerun-first rule, stage C4, same binary, 128 tokens):

| row | in-battery (C1) | standalone (C4) | HEAD (2026-09-21) | 0.11.7 (in-battery) | verdict |
|---|---:|---:|---:|---:|---|
| 7 gpt-oss-20b | 15.6 | **13.2 / 13.2** | 13.2 | 19.8 (standalone 18.1) | suite context; the release cell stays the battery's 15.6 as in 0.11.7, with the standalone 13.2 named |
| 3 gemma4-26B-A4B | 31.1 | **28.3 / 28.3** | 28.9 | 33.4 | suite context; the same convention (31.1, standalone 28.3) |
| 1 gemma4-31B | 123.1 | 123.6 | 125.5 | 126.1 | in band |
Same-binary reproducibility: the C4 draws are token-identical to the C1 cells on rows 1, 3 and 7
(5 of 5 draws, separate processes).

## Stage C3/C4 — the remaining model rows → **PASS** (0 regressions; every row measured, rows 12 and 20 included)

Driven by the 0.11.7 gate's own scripts re-pointed at the frozen binary (`gate-0.11.8/models/stageC3.sh`
= 0.11.7's with the new pin and sha assert, plus rows 10 at the manifest workload, 11 keras and 21),
historical budgets and envelopes, one guarded process per row, settle precheck inside every row.

| row | benchmark | 0.11.7 | **0.11.8** | ratio | peak | note |
|---|---|---:|---:|---:|---:|---|
| 15t | qwix-int8 8B | 388.4 | **268.4** | 0.69× | 52 GB | 8-token protocol as before |
| 15d | forensics | — | first token 12095 on 10/10 draws, logits healthy | | | identical to 0.11.6/0.11.7 |
| 9 | R1-Distill-32B | 190.8 | **196.0** | 1.03× | 69 GB | standalone draws 199.2 / 185.4; 188.0 under the pre-T1 pin — a ±4 % spread, not a change (item 10) |
| 8 | Qwen3.6-35B-A3B | 28.5 | **23.3** | 0.82× | 73 GB | |
| 17a | SD3.5 512² | 1249.3 | **456.5** | 0.37× | 21 GB | |
| 17b | SD3.5 1024² | 4961.6 | **2076.0** ˢ | 0.42× | 25 GB | in-sequence 2152.1 (HEAD 2056); ˢ = standalone re-run (stage C5), as 0.11.7 did |
| 12 | Mixtral 8×7B | 85.6 | **72.0** | 0.84× | 93 GB | 110 GiB envelope, exit 0 |
| 20 | 235B-A22B 3-bit | 56.2 | **43.3** | 0.77× | 103 GB pack wave | envelope 110/114, exit 0; load 321 s + 57 min warm-up (the pack wave, unchanged from 0.11.7: 1716 s) |
| 10 | DeepSeek-V2-Lite (manifest workload, 51-token prompt, 128 tokens) | 25.9 | **19.8** | 0.76× | 90 GB | presplit harness variant 19.1 (fn 11); text identical to every prior cell |
| 11 | Qwen3-0.6B keras-hub (the headline since 2026-09-02) | — | **5.1** | | 4 GB | first release cell; stream identical to the HEAD record |
| 21 | Qwen3.8-27B bf16 | — (154.9 first cell, 2026-09-02) | **142.7** | 0.92× | 57 GB | first release cell; stream IDENTICAL 64/64 to the row's 2026-09-01 record (the 154.9 cell); the 2026-09-03 GDN-era HEAD stream (which parted from it at 56) has flipped back |
Guard fires: 0.  Governor refusals: 0.  Panics: 0.

## Cross-release token streams (0.11.8 vs the 0.11.7 gate's records)

| row | verdict |
|---|---|
| 4 gemma4-e2b-bf16 | IDENTICAL 64/64 |
| 5 qwen3-8b-bf16 | IDENTICAL 64/64 (and CPU-exact) |
| 6 llama31-8b-bf16 | IDENTICAL 64/64 (and CPU-exact) |
| 7 gpt-oss-20b | IDENTICAL 64/64 |
| 2 gemma4-12b-bf16 | diverges from 0.11.7 at 16 and is IDENTICAL to the 0.11.6 record and to jax-CPU: the accepted tie (disclosure 1 of 0.11.7) flipped back |
| 1 gemma4-31b-bf16 | diverges from 0.11.7 at 45 (from 0.11.6 at 34, as 0.11.7 did): an exact tie — teacher-forcing the 0.11.7 prefix, ' massive' (12566) and ' weights' (18710) attain the identical bf16 logit 24.75 (0 ULP), the only two ids at the row max; the branches rejoin at offset −2 and share the next 8 ids (probes/findings.md) |
| 3 gemma4-26b-a4b | diverges from 0.11.7 at record index 53 = generated index 3 (this route records prompt+generation): margin 15 ULP at a position with no signal (p(top1) 0.11, 132 ids for 90 % of the mass — the preamble the missing chat template leaves undecided); the precision default's per-step perturbation there is 39 ULP median, and the release binary under `highest` reproduces the 0.11.7 stream exactly (0.11.7 was gated on pre-accelerator numerics). No CPU reference: XLA:CPU widens the 51.6 GB checkpoint to ~103 GB (two clean guard kills) — no gate has ever had a row-3 CPU cell |
| 13 gemma4-e2b-int4 | diverges at 50: the earlier records are the broken model (fn 8) |
Every CPU cell is within 3 % of its 0.11.7 record and its stream identical (rows 2, 4, 5, 6, 13 CPU).

## Consolidated disclosure list

1. **Precision default changed** (`METALJAX_MATMUL_PRECISION=high`, commit 44fa042): f32 GEMM/attention
   stay exact (2.1e-6 vs f64, bit-identical to the pinned kernels on 9 of 10 swept GEMM shapes and every
   attention shape); bf16/f16 and quantized matmuls move to the M5 accelerators, as mlx-lm and torch-MPS
   do — bf16 outputs within ~1 ULP of the pinned kernels (accumulation order).  Stream effects on the
   release records: none on rows 4/5/6/7/20/21 (identical); rows 1, 2, 3 carry tie flips (items 3–5).
   `highest` restores the whole-arch pin; f32 is never silently degraded (CLAUDE.md rule, 2026-09-09).
2. **Row 13's history was a broken model** (keras-hub Gemma4 int4 "HOTFIX", upstream still): every cell
   through 0.11.7, metaljax and jax-CPU, decoded one repeated token.  The harness variant
   (`scripts/model_bench/int4_fix.py`, default on for the int4 route, knobs to reproduce the original)
   restores the computation; the row's cells carry the ᵇ marker; its two backends part at generated
   token 1 on a 1-bf16-ULP tie (top-2 sets identical, margin 0.125 both sides).
3. **Row 2 gemma4-12B**: the stream flipped back to the 0.11.6 record and is jax-CPU-exact again
   (0.11.7's accepted disclosure 1 resolved).
4. **Row 1 gemma4-31B**: diverges from the 0.11.7 record at generated index 45 (0.11.7 itself diverged
   from 0.11.6 at 34): an exact 0-ULP tie between the two candidates (both 24.75 in bf16, the only
   two ids at the row max); the pre-T1 pin reopens a 1-ULP gap toward the 0.11.7 choice.  Accepted
   class (0.11.7 disclosure 3); no CPU reference exists for this row.
5. **Row 3 gemma4-26B-A4B**: diverges from the 0.11.7 record at 53 (agrees with the 0.11.6 record
   through 60); the row's prompt has no chat template, so its preamble region is undecided
   (0.11.7 disclosure 10).  The flip is at generated index 3 (record index 53), margin 15 ULP on a
   position where 132 ids share 90 % of the mass; the precision default perturbs that step by 39 ULP
   (median) and `highest` reproduces the 0.11.7 stream — a consequence of item 1 on an undecided
   position, not a lowering change.  Row 3 has no jax-CPU cell (the checkpoint widens to ~103 GB on
   XLA:CPU).
6. **Row 4 gemma4-E2B vs jax-CPU**: diverges at token 51/64 — the certified-benign entry, the same
   stream 0.11.7 recorded (identical 64/64 cross-release).
7. **Row 21 Qwen3.8-27B**: first release cell; identical to the row's 2026-09-01 record; the GDN-era
   HEAD stream (2026-09-03, parting at 56) has flipped back.
8. **Row 11 maxtext prefill +1.9 ms** (B3, 2026-09-06, environmental: pool leftovers of the decode
   program) — unchanged, carried.
9. **texmo**: four sub-2 ms configs sit 4–11 % below the 2026-08-22 anchor exactly where 0.11.7's
   own records had them (tc010-w17's accepted micro-regression among them); one bimodal config.  No
   config regressed against 0.11.7; geomean 1.091× over 223.
10. **Suite-context spreads**: rows 7 and 3 read 15.6 / 31.1 in the battery and 13.2 / 28.3
    standalone (release cells stay the battery's, as in 0.11.7); row 17 at 1024² takes its standalone
    2076 (ˢ); row 9 read 196.0 in sequence, then 199.2 and 185.4 in two standalone draws on the same binary, and 188.0 under the pre-T1 precision pin (stage C7): a ±4 % spread on a 67 GB model whose cells have always scattered (217.7 / 214.4 / 210.3 / 211.0 / 190.8 / 188.0 across releases), with the pinned arm inside it — no attributable change; the release cell is the in-sequence 196.0 with the spread named.
11. **msl_scan silent-wrongness fixes since 0.11.7** (both shipped-number-neutral, both gated 106/106):
    the lane-scalar misclassification (af7f140) and the invariant-carry-as-counter misread (7adb350).
12. **Vendored MLX fork** moved from 65cb64b8 to 661b2e38: donate-through-stream-pins (0005),
    skip-empty-finalize (0006, structural: 178 → 112 command buffers per row-10 token, wall-neutral),
    transient byte cadence (0007, opt-in, off).  The plugin is built under Xcode 27.0; the vendored
    MLX was built under 26.6 (2026-09-10) — the suites above ran on exactly that pair.
13. **Not fixed, documented**: row 10's checkpoint load peaks 99–111 GB against the 105 GB projected
    guard (about a third of its cells die at load, never during decode, never a panic); row 20's 57-min
    pack wave at first execute (unchanged since 0.11.6); the harness's timed generate on keras rows
    includes a padded-window prefill (~1.7 ms/tok of the row-7 cell frame) — a metric-definition
    question, unchanged for comparability.

## Verdict

**PASS** (release rule 2: no regression on any suite or benchmark against 0.11.7; every disclosure
above is stated here).  Every number in the 0.11.8 release table comes from `frozen-m2main-7434ae86`
(release rule 1); rows 12 and 20 have real cells for the first time since 0.11.7; no row is at or
above 2× its like-for-like goal.  Awaiting Oleg: the version number (0.11.8 assumed), the greenlight,
the upstream keras-hub issue, the fork push.
