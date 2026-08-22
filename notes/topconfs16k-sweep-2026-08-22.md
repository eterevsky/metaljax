# top_confs_16k sweep — 223 configs, three legs + the bf16 cliff (2026-08-22)

The new top-configurations set (`top_confs.jsonl` @ 112ae10, replacing the
163-row Aug-2 set): 221 fp32 + 2 bf16, weights 5–14,625 (median 511),
lengths 32–32,768, batch 1–1024; first appearance of the `split` (73 rows),
`latent`, `conv` and `mingru` families in this suite.

Three legs, `scripts/bench_texmo_pjrt.py` (PJRT route — measures the shipped
native engine), release binary `frozen-vendor-d651add3`, machine lock held,
sequential, zero errors ×3:

| leg | file | wall |
|---|---|---|
| metal, native precisions | `notes/data/topconfs16k-metal-2026-08-22.jsonl` | ~15 min |
| metal, all forced bf16 | `notes/data/topconfs16k-metal-bf16-2026-08-22.jsonl` | ~89 min |
| cpu, native precisions | `notes/data/topconfs16k-cpu-2026-08-22.jsonl` | ~42 min |

Logs + probes: `~/.cache/metaljax-bench/logs/topconfs16k/`.

## 1. Metal vs CPU (native precisions)

Geomean metal/cpu **0.964** over 223; metal faster on 122. The split by size
is the whole story:

| slice | n | geomean metal/cpu | metal faster |
|---|---|---|---|
| weights ≥ 1000 | 92 | **0.213** (≈4.7× faster) | **92/92** |
| weights < 1000 | 131 | 2.78 (slower) | 30/131 |

Every config with ≥1000 weights beats CPU. The sub-1000 losses are
dominated by the documented dispatch floor (sub-ms CPU steps, batch 1–4,
long scans of minuscule cells — worst: `tc027-w47` b1 l4096 at 41.6×,
CPU 0.28 ms/step). Search-relevance note: the search optimizes for
thousands-of-weights models, where metal sweeps.

**Unexpected regressions** (metal >15 % slower AND cpu ≥ 1 ms/step — i.e.
NOT dispatch floor): 11 rows, two families:

1. **`lrnn.8.x` under `split.add` composites, batch 32–64, len 64–128**
   (tc112–tc121, weights 515–659): metal 3.0–4.2× slower than CPU. The
   0.3.0 lrnn work (SymRedReg) covered the bare cells; these wrap the cell
   in `split.add(...)` composites, which likely breaks the msl recognizer
   pattern → compiled-graph path at shapes where it loses. THE optimization
   target from this sweep.
2. `mgru.4`/`rnn.4` stacks at batch 4–64 (tc038 3.2×, tc044 1.6×,
   tc046 1.3×) — same class, milder.

## 2. bf16 vs fp32 (metal) — a 25× CLIFF, root-caused

Forcing the 221 fp32 configs to bf16: geomean bf16/fp32 = **25.5**, slower
on 219/221, faster on exactly 1 (`dense.1.gelu` — the only spec with no
recurrent cell). 53 rows land on a plateau of **~62 µs × sequence length**
per training step (l2048 → ~128 ms/step vs 0.85 fp32), warmups 33–65 s vs
0.4 s. The two already-bf16 rows repeat within 5 % (measurement sound).

**Root cause (static sweep + confirmed by GPU probe on tc013):**

- `plugin-native/metal/metal_msl.cc:155` `MslDtypes()` has **no bf16 entry**
  (MSL has `bfloat` since Metal 3.1; the table maps f32/f16/ints only).
  First bf16 carry → `MslDecline("dtype bf16")` → no msl plan.
  **Inherited from Stage 1** (`src/metaljax/msl_scan.py:262` — identical
  table); invisible until now because suite-106 and the old top_confs set
  are essentially all-fp32.
- The missing plan **cascades**: the plan-less scan is charged trip×cost in
  the cost/bytes walks → the 256-step training-chunk body blows the trace
  budget (probe: `while gate: cost=170979 … budget=20000 by_cost=0 pure=0
  body_compile=0 chunkable=0 kmax=1 period=1`) → outer body uncompilable,
  `period=1` = blocking flush EVERY timestep → ~62 µs/timestep interpreted
  dispatch.
- **Probe confirmation** (`METALJAX_DEBUG=1`): bf16 tc013 prints
  `msl_scan: not eligible (dtype bf16)` ×3 (fwd+bwd) and runs 127.2 ms/step;
  fp32 with `METALJAX_MSL=0` reproduces the identical gate collapse and
  **128.4 ms/step** — the entire regression flows through the missing msl
  plan; nothing else in the engine is bf16-specific. (Recognizers qmm/sdpa/
  moe are bf16-clean; no astype storm; bf16 storage honest end-to-end.)

**Staged fix plan** (post-release queue):

1. Add `bf16` → `bfloat` to `MslDtypes()` (both emit sites; f32
   accumulation care where carries accumulate).
2. Relax the four deeper f32-only msl gates behind it: dot recognition
   (`metal_msl.cc:1496`), `InlaneDot` (1564), SymRedReg (1310), accumulator
   fission (2506) — with f32 accumulate.
3. Independent cascade hardening (helps any plan-less loop, incl. the
   MSL=0 escape hatch): (a) the lowering/runtime unroll mismatch —
   `WhileTraceable` approves trip×cost ≤ 20000 but `run_while` refuses
   trip>64 inside a trace, so approved-then-failed bodies retire compile
   permanently; (b) `period=1` flush-every-timestep when cost is
   pessimistic — the flush cadence shouldn't collapse below the chunk size.

## 3. Anchor status

This sweep's metal leg is the first recording for the new set (the old
anchor `notes/data/texmo-topconfs-final.jsonl` covers the retired 163-row
set). Gate-harness anchor swap folds into the post-release cleanup
(task #48); until then `topconfs16k-metal-2026-08-22.jsonl` is the
comparison file.

## 4. FIX LANDED (same day): bf16 fast paths in the native engine

bf16/fp32 geomean **24.7 → 1.086** on the 223-config set (22.8× faster
than the pre-fix bf16 leg, 0/223 rows regressed vs it; warmup total
2603 s → 146 s). fp32 no-regression: geomean 0.9983 with all excursions
disproven standalone (binary-equal); suite-106 0.9673 vs the release arm,
six in-suite excursions all binary-equal standalone (the suite-context
trap); texmo_gate 106/106; execute/ingest/bazel green. What landed:

- **msl bf16**: `MslDtypes()` maps bf16 → MLX's `bfloat16_t`; typed bf16
  literals (Metal has NO implicit float→bfloat conversion — measured, an
  11-case kernel experiment in the logs); `CastTo` on every value landing
  in a bf16 register (all three emitters); `sign` computed via float.
- **bf16 dots** (ReduceAsDot / DotGeneral / InlaneDot / accumulator
  fission): XLA bf16-dot semantics — operands widened to f32, f32
  accumulate, one rounding at the typed result store; result dtype honors
  `preferred_element_type`; hidden weight-grad stacks stay f32 with the
  total astype'd back to the carry dtype.
- **Cascade hardening**: `kUnrollMax=64` aligns `WhileTraceable` with
  `run_while`'s in-trace refusal (the approve-then-fail path that retired
  compiled bodies permanently); the eager counted loop's BLOCKING flush
  floors at 8 iterations (submission cadence + op-unit accounting
  unchanged — bit-identical for period ≥ 8).
- **Tests**: 4 bf16 msl differential cases (forward bit-exact vs CPU) +
  a "bf16 msl plans build in all three modes / no dtype-bf16 declines"
  contract in execute_test; texmo_gate gains JSONL configs + bf16-aware
  checking (uint16-view ULP bumps, f32 references).

Measured answer to "why didn't chunking save the plateau": it was
working — K=16 chunked replays cost ~60 µs/timestep intrinsically at b1
(encode-bound, ~8–10 launches/t; CHUNK_MAX 16→256 changes nothing). The
graph path cannot rescue these shapes; only the msl kernel (0.4 µs/t) can.

Remaining, stated: 8 bf16 rows at 2–4.2× fp32 (small-F matvec cells,
per-element bf16 weight-load conversion — optimization remainder, kernels
correct); f16 dots still declined (no workload); a LATENT WEDGE seen once
on an intermediate broken build — repeated Metal kernel-build failures
inside chunked replays left an MLX shared event unsignaled, process hung
in `Event::wait()` (stack on file). Unreachable on the final build but a
no-panic-contract concern → follow-up task filed.

Logs: `~/.cache/metaljax-bench/logs/bf16-msl/`.
