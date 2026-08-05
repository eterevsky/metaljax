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
