# texmo suite — tracking over time

*One row per tracked run of `scripts/texmo_topconfs.py` (163 top
configurations; headline = geomean of per-config metal ms/step ratios
vs the PREVIOUS row, >1 = faster; "abs" columns are medians by weight
class, metal ms/step). Raw data in notes/data/. Append a row for every
run that gates a release or an optimization.*

| date | version/commit | geomean vs prev | <100 | 100–500 | 500–1500 | 1500–3000 | checks | raw |
|---|---|---|---:|---:|---:|---:|---|---|
| 2026-08-02 | 0.11.1+fixes, 256-step chunks, 512 MB buffer cap (baseline)¹ | — | 0.97 | 0.66 | 0.73 | 0.80 | 163/163 | notes/data/texmo-topconfs-2026-08-02{,b}.jsonl |
| 2026-08-03 | 0.11.2 release gate (ops=800) | 1.008² | — | — | — | — | 163/163 | notes/data/texmo-topconfs-final.jsonl |
| 2026-08-05 | 0.11.3 release gate | 1.002³ | 0.98 | 0.66 | 0.77 | 0.83 | 163/163 | notes/data/texmo-topconfs-2026-08-05.jsonl |
| 2026-08-15 | P22 release measurement, Stage 1 (engine route) | 1.071⁴ | 0.97 | 0.58 | 0.66 | 0.72 | 163/163 | ~/.cache/metaljax-bench/logs/p22-release-measure/topconfs-stage1-engine.jsonl |
| 2026-08-15 | **P22, phase-2 native plugin** (PJRT route, frozen dylib) | 1.002⁵ | 0.97 | 0.58 | 0.65 | 0.71 | texmo_gate 106/106 | notes/data/p22-release-measure-2026-08-15.{csv,json} |

¹ Two same-day runs merged: the buffer-cap correctness fix between
them cost geomean 0.980 (−0.5% small configs → −4% largest class,
partly ambient drift).
² CPU control drifted +2.2% (quiet machine); the ops-800 cost (+2–3%
expected) is absorbed within drift.
³ CPU control 0.992; one config improved >5%, zero regressed >5%.
The 0.11.3 feature work (recognizers, memory stack, body probe) is
texmo-neutral by design — the win target for this suite remains the
C++ replay engine. This row supersedes 0.11.2 as the C++-era anchor.

Suite-wide context (0.11.3 anchor): metal wins 53/163 configs vs
jax-CPU; crossover ~500–700 weights; the sub-crossover dispatch floor
(~0.7–1.0 ms/step flat across sizes) is the native-replay-engine
target.

⁴ CPU control 0.990 (machine 1% slower), 163/163 checks ok. The +7.1% over
the 0.11.3 anchor is a real shared drift of metaljax's own code across the
C++-migration commits; it is measured on the anchor's own runner and route.
⁵ vs the PREVIOUS row (Stage 1 today), i.e. **the two stacks are at parity**;
against the 0.11.3 anchor the native plugin is **1.073x faster**. The native
column is a different route (jax/PJRT, 64-step chunks — `texmo_topconfs.py`
drives `metaljax.engine` and can only ever measure Stage 1), and that route's
own factor was measured in the same campaign at **1.002** over 163
configurations, so the columns are comparable. Frozen dylib
`frozen-release-208ca0d1`; the correctness column is the phase-2 gate
(`plugin-native/texmo_gate.py`, whole-model vs jax-CPU) rather than
`texmo_topconfs.py`'s per-config check, which is a Stage-1-route harness.

**Native-plugin baseline (2026-08-12, tree 845ab89):** Stage 1 re-measured on
this tree is **1.046x faster** than the 0.11.3 anchor (163/163 checks ok, CPU
control 0.986), so the anchor stands. The phase-2 plugin, same configurations
through jax/PJRT, is **36.5x slower** on top_confs (median 46x, worst 175x)
and **4.24x** on the 106-config suite (`db` class 14.6x, `big` class 1.34x,
five `big`/`mid` rows *faster* than Stage 1) - it has no msl_scan in its
lowering, and where no msl plan fires the two stacks are equal to 1%. Details
and per-row data: benchmarks/perf-2026-08-native-baseline.md.

**RELEASE MEASUREMENT (2026-08-15, P22).** The gap above is closed. With
`msl_scan` ported (P21) and the coop width cap landed (P22), the phase-2
plugin is at **0.998x of Stage 1** on top_confs (geomean over 163, median
0.998, every row within 1.2x, native faster on 101) and **1.011x** on the
106-config suite (median 1.000, 103 of 106 within 1.2x, none above 10x). It
beats jax-CPU on **59 of 163** configurations where the P16 native plugin beat
it on none, and where Stage 1 wins 55 on the anchor's route. One qualifier
survives: three `db*-b256l512` rows are genuinely 1.31-1.77x slower natively
(identical plans and identical `METALJAX_MSL=0` timings, so it is the kernel
LAUNCH, not the recognizer). Full tables:
benchmarks/perf-2026-08-native-baseline.md ("THE RELEASE TABLE"), narrative
notes/cpp-p22-release.md.
