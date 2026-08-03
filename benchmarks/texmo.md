# texmo suite — tracking over time

*One row per tracked run of `scripts/texmo_topconfs.py` (163 top
configurations; headline = geomean of per-config metal ms/step ratios
vs the PREVIOUS row, >1 = faster; "abs" columns are medians by weight
class, metal ms/step). Raw data in notes/data/. Append a row for every
run that gates a release or an optimization.*

| date | version/commit | geomean vs prev | <100 | 100–500 | 500–1500 | 1500–3000 | checks | raw |
|---|---|---|---:|---:|---:|---:|---|---|
| 2026-08-02 | 0.11.1+fixes (8-step chunks, superseded methodology) | — | 1.10 | 0.74 | 0.91 | 1.01 | 163/163 | texmo-topconfs (overwritten) |
| 2026-08-02 | 0.11.1+fixes, 256-step chunks (methodology baseline) | — | 0.97 | 0.66 | 0.73 | 0.80 | 163/163 | notes/data/texmo-topconfs-2026-08-02.jsonl |
| 2026-08-02b | + buffer-cap 512 MB (correctness) | 0.980 | ~same | ~same | ~same | −4% | 163/163 | notes/data/texmo-topconfs-2026-08-02b.jsonl |
| 2026-08-03 | 0.11.2 release gate (ops=800) | 1.008¹ | — | — | — | — | 163/163 | notes/data/texmo-topconfs-final.jsonl |

¹ CPU control drifted +2.2% (quiet machine); the ops-800 cost (+2–3%
expected) is absorbed within drift. This row is the C++-era anchor.

Suite-wide context (0.11.2 anchor): metal wins 50/163 configs vs
jax-CPU; crossover ~500–700 weights; the sub-crossover dispatch floor
(~0.7–1.0 ms/step flat across sizes) is the native-replay-engine
target.
