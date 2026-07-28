# 0.4.3: coop-over-vector mode pick + kernel input packing (2026-07)

Trigger: Oleg's texmo search data (0.3.2-era) listing configs slowest vs
CPU. Re-measured at 0.4.2 HEAD on the M5: the catastrophic rows (190x,
23x, and the >=10ms/step rows with real batch) were 0.3.2 pathologies
already fixed by the 0.4.x accdot/retry repairs. What still reproduced
were the mid-table 4-9x rows — all small-feature recurrent cells — plus
the batch=1 tiny-model class (launch-overhead physics, deprioritized per
Oleg: search optimizes for models from thousands of weights up).

## Diagnosis 1: vector-mode occupancy collapse (F<=16 cells)

Cells whose square dots have min dim <= _REG_LIMIT(16) — rnn.16,
mgru.16, mullstm.8, gru.16 — picked *vector* mode, whose lane count is
just batch(*blocks): 16-32 GPU threads total, ONE simdgroup on a 40-core
GPU, and every lane re-reads the full weight matrices from device memory
each timestep as a dependent load->fma chain. ~60us/timestep, linear in
L, insensitive to batch. Forcing coop (threads = batch*F, dot data
staged through threadgroup memory) was faster at EVERY batch size:

  rnn.16.gelu-mgru.16 l128 (ms/step, vector vs coop vs cpu):
    b8    7.49   0.90   1.19
    b32   8.16   1.09   2.54
    b128  8.63   1.52   7.11
    b512  10.37  1.77   23.2
    b2048 14.01  6.16   78.4
  mullstm.8 l128: b16 5.89/1.48/1.17  b128 6.80/1.55/6.10
                  b1024 10.82/4.58/47.9

No crossover: occupancy was only half the disease — coop's threadgroup
staging of weights is the other half. Fix: prefer coop whenever the dots
are square multiples of the state width and F >= 8
(METALJAX_MSL_COOP_MIN_F; METALJAX_MSL_COOP_PREF=0 restores old pick).
F<=4 pockets (mgru.4 etc.) keep vector — coop threadgroups of 4 threads
waste 7/8 of a simdgroup and were not measured to win.

Safety: the flip only touches the coop-eligible AND vector-eligible
intersection; asymmetric (lrnn) and batched dots can't flip. The coop
emitter covers fewer leaf kinds than vector's, so build_plan() retries
without the flip when the coop build raises _Unsupported after flipping
(module global _last_flipped; single-threaded engine).

## Diagnosis 2: binding-limit cascade (lmgu-class deep bodies)

lmgu.4.2-rmsnorm-mingru.4 b64 l64: the AD backward scan kernel needed 38
buffer bindings (fwd stacks 19 residual streams; Metal caps kernels at
31). The 0.3.1 guard rejected it -> inner loop cost counted at 64x ->
train-step body blew the trace budget -> whole chunk ran EAGER: 11.2
ms/step vs CPU 1.68 (6.7x), all from one rejected kernel.

Fix: input packing. When a plan would need > METALJAX_MSL_PACK_TRIGGER
(30) bindings, same-dtype non-0-dim inputs are pooled into one buffer
per dtype; run() concatenates the flattened buffers, and the generated
source addresses them through static pointer offsets ("(pk0 + 1234u)")
baked per plan — no runtime indirection tables. 0-dim inputs stay
separate (MLX passes those by value). Packing changes nothing
numerically (same values, same order) and engages only over the
threshold, so already-working kernels are byte-identical.

lmgu after: bwd kernel builds (packed=23), chunk compiles, 2.10 ms/step
(5.3x; within 25% of CPU from 6.7x behind).

## After (automatic, no env)

  rnn.16.gelu-mgru.16 b32 l128:  7.96 -> 0.95 ms/step (cpu 2.53)
  mullstm.8 b16 l128:            5.89 -> 1.09         (cpu ~1.2)
  lmgu.4.2-rmsnorm-mingru.4 b64: 11.2 -> 2.10         (cpu 1.68)

## Validation

- pytest: 199 passed (7 new: coop-pref assert + pref-off compare +
  synthetic coop-emitter-failure fallback + low-trigger packing +
  natural >31-input body packs and stays on the MSL path + the
  sliced-weight-window packing regression below).
- texmo_check whole-model gate (m5-metal.csv): 104 ok, 0 FAIL — plus
  the three new-path configs (lmgu/rnn16/mullstm8, not in the suite)
  checked explicitly: 3 ok, worst 2.2e-05 vs sensitivity-scaled tol.
- Suite A/B, same session, new features env-disabled vs enabled
  (METALJAX_MSL_COOP_PREF=0 + METALJAX_MSL_PACK_TRIGGER=999 == 0.4.2
  behavior): F<=16 cell rows improved db17 7.1x/11.7x, db11 5.1x/6.2x,
  db15 5.4x/3.7x, db14 3.4x/3.9x, db18-b4 3.0x, db12 1.1x; ALL other
  rows 0.96-1.05x (noise band; mid/big uniformly 1.00x). No structural
  regressions.

## Review (adversarial workflow, 4 lenses + verify)

Caught one CRITICAL bug before commit: pool slot sizes used the SOURCE
buffer numel, but run() concatenates the weight-normalized WINDOW view
(gate-split windows of fused weights are strict slices with different
numel) — every later pool member would read shifted garbage, silently.
Fixed: slot = numel(_weight_norms[sid][0]) when normed. Regression
test test_packing_with_sliced_weight_window constructs the single-
sliced-gate cell (view 25 of a 50-elem buffer, packed) and FAILS on
the unfixed code. Also from review: build_plan retries on any
exception (not just _Unsupported) so a flipped coop crash can't lose a
working vector plan; WRONG-plan npz dump includes pk pools. Flagged
pre-existing (not this diff): sid in both _reads and _weights isn't
rejected though run()'s norm swap would misaddress the reads.

## Leftovers / future

- batch=1 long-L rows: recurrence runs serially on ONE GPU lane
  (lanes=1). Only algorithmic escape is a parallel (Blelloch) scan over
  time for affine recurrences. Deprioritized (CPU's home turf).
- F<=4 square cells: unmeasured coop-vs-vector pocket; revisit if a
  search pass surfaces mgru.4-class configs as gaps.
- Output packing (stacked/hidden/fin) if a plan ever exceeds 31 via
  outputs alone.
- Vector mode could stage invariant weights in threadgroup memory for
  the coop-ineligible small-lane cases (lrnn at tiny batch).
