# Release gates — 0.11.6 (2026-08-27)

**The release binary**: `frozen-0.11.6-dde2d668.dylib`, sha256
`dde2d6680194589fae52e53aec139b224391075dde9dfc2d94598fb45df178a8`, built
from tree `83ff94f` (0 dirty files), DYLD-verified in every phase to load
the vendored `libmlx_metaljax.dylib` (sha256 `06da3cfd…`) and no pip mlx.
Every number below is from this binary (release rule 1). Three sequential
lock-held phases, 2026-08-27; artifacts under
`~/.cache/metaljax-bench/logs/gate-0.11.6/{models,jax-suite,texmo-tests,readme}/`.

## Verdicts

| # | gate | verdict |
|---|---|---|
| 1 | Freeze (build reproduces sha; tree clean; version 0.11.6 both files) | **PASS** |
| 2 | Pinned jax suite (164 files, `--jobs 1`, vs the banked 0.11.5 whitelist) | **PASS** — 28,073 / 129 = 99.54 %, failing set **id-identical**, zero new, zero gone; 35.3 min |
| 3 | Model rows, ALL 20 | **PASS** — 19 comparable rows 0.87–1.04× their 0.11.5 cells, zero regressions standing; rows 12 and 20 produce **first-ever release cells** (91.3 and 66.3 ms/tok, documented envelopes) |
| 4 | texmo | **PASS** — `texmo_gate` **106/106**; suite-106 **1.026× faster** than the 0.11.5 anchor; top_confs-223 fp32 **1.035× faster**; bf16 leg (first release recording) 1.028× faster than its baseline, **bf16/fp32 cross-geomean 1.005 (parity)** |
| 5 | tests/ + contract suites | **PASS** — pytest **485 + 1 xfail** exact; execute_test ~566 checks all match CPU; ingest 13/13; smoke; decline_census 35/35; bazel test |
| 6 | The no-panic contract | **PASS** — zero panics, zero wedges, zero guard kills across the whole gate; **two clean governor refusals** (rows 10 and 20 attempt-1), both documented, budgets never raised after a breach |
| 7 | Wheel | see the build record below (built from the clean tree carrying this binary; twine PASSED; fresh-venv Metal smoke) |
| 8 | Finale | **GO — pending Oleg's review** (his hold: numbers + regressions review before TestPyPI) |

## The model table (vs 0.11.5)

See `models.md` (0.11.6 column) and `STATUS.md` footnote 37 for
the cells and per-row detail. Headlines: all 19 comparable rows within
noise (worst +4.1 % = row 10 with cause named; best 0.87× = row 17@1024²
with spread named); row 12 = **91.3 ms/tok** and row 20 = **66.3 ms/tok**,
the first release in which **all 20 rows produce numbers**.

## Disclosures (rule 2 — none blocking, all named)

1. **Rows 12/20 memory envelopes**: both run above the shipped 96 GB
   governor default (110 GiB and 110/114 GiB respectively), per Oleg's
   explicit approval; envelopes chosen before the runs from measured
   need, recorded in every flight log; shipped defaults unchanged.
2. **Two clean refusals**: row 10 attempt 1 (16.7 GB machine baseline;
   needs ~13) and row 20 attempt 1 (118.8 GB claimed vs the 114 line
   after the full pack wave — the campaign's own success had 0.3 GB of
   margin there). Both refused with clean RESOURCE_EXHAUSTED, retried
   once on clean baselines, completed. The contract's preferred mode.
3. **Row 1 variance** (standing since 0.11.5): canonical 235.2 (0.999×),
   standalone confirm 292.3 — inside the known 261–297 machine-state
   band. Bimodal, machine-side, not a regression.
4. **The 0.11.5 release-note correction**: the "fused-attention
   recognizer nondeterminism on rows 5/7" item is **refuted** on this
   binary — gpt-oss 3/3 independent-process greedy draws identical,
   Qwen3-8B 2/2 and CPU-exact, recognizer ON. The 0.11.5 evidence was
   top-k(5) sampling with a per-process random seed (harness bug, fixed
   at 9f69ee8). Release notes carry the correction.
5. **Token agreement strengthened**: Qwen3-8B and Llama-3.1-8B now EXACT
   vs CPU (64/64); the certified-benign list tightens to
   {gemma4-E2B-bf16}. gemma4-E2B-int4's agreement is vacuous (both
   backends emit the same degenerate loop — keras int4 PTQ artifact).
6. **Whitelist data refresh**: `notes/data/pinned-0.11.6-failures.txt`
   (129 ids, identical to the banked 0.11.5 set) is now the tracked
   comparison file; the pre-0.11.5 `pinned-0.11.0-failures.txt` (130
   ids) stays as history — the 12-new/13-fixed delta between them was
   reviewed and approved at the 0.11.5 release.
7. **Two README-table drifts, named for Oleg's review** (both reproducible
   ×2, neither in the release gate's own suites): bench_compare
   transformer-d512 metal **+3.6 %** (159.4 vs the historical 153.9;
   d256 and GRU.256 at baseline), and the xla-suite maxtext-2.5B train
   step **+9–10 %** (11 606 vs July's 10 618 — that cell predates the
   memory governor and the entire 0.11.x engine line, so the comparison
   spans far more than this release). Also from that re-measure: the
   gemma2/gemma4 xla rows were VACUOUS all along (zero-iteration
   generate loops — the old 17.5/16.9 ms cells measured Stage-1 dispatch
   on an empty program), now footnoted honestly in the README; and the
   gemma3 rows needed a one-op plain-dot→dot_general rewrite (the plugin
   declines plain `stablehlo.dot` by design), validated against pristine
   CPU references.
8. **Attributed wins**: every texmo row faster than 1.10× (21 fp32 rows,
   the db05 suite pair, the bf16 win family) is the F=4 coop flip
   (14c6068), each within ~2 % of its pre-measured ratio. The one
   unattributed candidate (db02 suite row) was disproven by dual-binary
   rerun (anchor-recording artifact). Sub-ms excursions (tc002/003/005
   class, 9–14 % in-suite) adjudicated machine-side by dual-binary
   standalone reruns — release-vs-anchor binary parity within ±2.5 %.

## What ships in 0.11.6 (release-notes basis)

- **Stage 1 retired**: the Python engine, the pre-PJRT C++ engine and the
  trampoline (~26k lines) are deleted; the native plugin is the only
  engine, the wheel builds native-only from the real tree, `mlx` is no
  longer a dependency. tests/ re-pointed onto the plugin (recovering ~134
  genuine parity tests).
- **Correctness**: the msl window-keying P0 fixed (silent wrong values
  for nested scans / multi-slice bodies on the kernel path; shipped in
  0.11.5, census showed no released output affected); the Event::wait
  wedge fixed (kernel-build failures now surface as clean errors, never
  hangs); 16-bit scatter-add accumulates in f32 (33× contended-atomics
  fix); keras bench rows decode greedy (honest token evidence).
- **bf16**: msl fast paths (the 25× cliff → parity with fp32; wins up to
  1.35× on bandwidth-bound rows); bf16 dots with XLA semantics.
- **Rows 12 and 20**: Mixtral 8×7B and Qwen3-235B-A22B-3bit run for the
  first time (packed sub-byte quantized storage landed for the latter).
- **F=4 coop flip**: 20+ texmo configs up to 2.9× faster.
- Full detail: notes/topconfs16k-sweep-2026-08-22.md,
  notes/stage1-retirement-inventory-2026-08-23.md, and the commit log
  `9822054..83ff94f`.
