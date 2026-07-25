# texmo benchmark analysis (2026-07, suite in ~/texmo/benchmarks/)

208 common configs: m5-metal vs m5-cpu vs linux-4090. Headline: metal beats
cpu only above ~1M weights; user expected metal between cpu and 4090.

## The three regimes in the data

1. **noscan mode (all sizes): metal 10-40x slower than cpu.**
   Instrumented db00-b16l128 noscan: **~55 engine executes per training
   step** — optax's optimizer update + apply_updates run *eagerly* in
   texmo's per-step path, one PJRT executable per primitive (jit_add,
   jit_multiply, jit_square, jit_sqrt, ...). Each execute averaged 317us,
   dominated by the blocking mx.eval in engine.execute (Metal
   command-buffer submit+wait ~165-190us). CPU pays ~3us per eager
   primitive. 55 x 0.3ms ~= the 9ms/step observed.

2. **scan mode, recurrent specs with length 512-1024: metal 30-700x
   slower than cpu** (worst rows: db02-b4l1024 90ms vs 0.12ms cpu).
   Cause A: cell_cost x length > METALJAX_TRACE_BUDGET (20k) so the time
   loop is NOT unrolled; each timestep is a separate compiled-body replay
   in a chain, and each chained call evaluates its inputs first:
   **~52us/iteration** (measured; async_eval does not help this).
   Cause B (the floor): even fully-unrolled traces replay at
   **~25us/timestep for a tiny cell** = 5-6 kernels x ~4.5us Metal launch
   cost. XLA:CPU compiles the whole scan into native code: 0.2us/step.
   Sharp budget cliff visible in data: db01-b256l512 (no recurrence,
   whole chunk = one trace) 1.56ms/step vs db02-b256l512 (recurrent,
   chained) 51.8ms/step.

3. **Non-recurrent / attention specs: metal already wins big** (mid01/05/07,
   big12/14: 5-10x faster than cpu, 0.1-0.2x ratios). Whole-main
   mx.compile works as designed; disparity is exclusively in loop paths.

## Reality check vs the 4090

Even the 4090 loses to M5-CPU below ~1M weights (db02-b4l1024: 13.8ms vs
0.12ms — 115x). Tiny sequential models are simply a CPU regime (XLA:CPU
fuses the scan into a native loop; any GPU pays per-kernel launches on a
sequential dependency chain). The *closable* gap is metal vs 4090: 2-7x,
made of (a) the blocking-eval dispatch floor, (b) the 52us replay-chain
overhead, (c) 25us vs ~13us per-step launch floor.

## Measured floors (M5 Max, mlx 0.32)

- mx.eval blocking roundtrip: ~165-190us (regardless of graph size)
- mx.async_eval dispatch: ~17us; 100-call async stream: 13.4us/call
- compiled-fn lazy call: 0.6us; chained replay (input eval): ~52us/iter
- kernel launch inside a replayed trace: ~4.5us; tiny RNN cell = 5-6
  kernels -> ~25us/timestep at any length (L64-L2048 measured flat)
- jax-level dispatch floor per jitted call: metal 199us vs cpu 2us
  (becomes ~40-60us with async_eval)

## Fix plan (prioritized, expected effect)

1. **async_eval in engine.execute** (blocking eval -> mx.async_eval;
   correctness: to_host/np.array forces evaluation; errors surface at
   sync points). Expect noscan rows ~4-6x better (9ms -> ~2ms for tiny),
   and general dispatch floor 199 -> ~40-60us. Guard w/ METALJAX_SYNC=1.
2. **Chunked unrolling for long counted loops**: when trip x cost >
   budget, unroll K = budget // cost iterations per compiled chunk, replay
   trip/K times (+ remainder). Kills most of the 52us/iter chain overhead
   for l512/l1024 recurrent rows: expect ~2x (db02-b4l1024 90 -> ~35ms).
3. **Kernel-count reduction for cells** (the remaining floor): fuse
   gates/activations via custom Metal kernels or mx.fast.*; or hoist
   loop-invariant input projections (x @ W precomputed for all t as one
   GEMM before the loop — torch's fused-GRU trick, also applicable at
   interpreter level by detecting scan-invariant matmul operands). This is
   the only path toward 4090-class per-step cost for small cells.

Items 1-2 are interpreter/engine changes; item 3 starts pattern-matching
(loop-invariant hoisting) and shades into Stage 2.

## Outcome of implementing 1-2 (same session)

async_eval (1) delivered as measured: dispatch floor 199 -> 33us/call;
noscan rows 1.4-5.8x faster (db00 9.26 -> 1.85ms, db01-b256l512 17.5 ->
3.0ms, db09 39.7 -> 20.0, big16 80.9 -> 54.5); bench_compare transformer
31.2 -> 29.9ms (== torch-MPS), GRU 112 -> 95ms; texmo mgru.4 chunk 8.7 ->
7.8s, mgru.256 5.1 -> 4.1s.

Chunked unrolling (2) was ~neutral on real recurrent cells: they are
kernel-launch-bound (matmuls break MLX fusion), not chain-overhead-bound —
the microbench body over-fused vs reality. Kept (small wins on fusible
bodies) but restricted to cost <= METALJAX_CHUNK_MAX_COST (1500) and
K <= METALJAX_CHUNK_MAX (16): big-body chunking inflates traces, loosens
flush cadence (GPU fault via pending-buffer overrun: "command buffer
Ignored" — much nastier than the old clean malloc error), and can hit
MLX's fused-kernel argument-buffer limit on long elementwise chains
(linear RNNs like db07/lrnn: "Too many inputs/outputs fused"). Both
failure modes now have graceful fallbacks (chunk -> single-step; engine
compiled -> eager).

Remaining gap = item 3 (kernel count per timestep). db02-b4l1024 stays
~100ms vs cpu 0.12ms vs 4090 13.8ms: ~30k launches x 4.5us. Next lever:
hoist loop-invariant input projections out of scan bodies (x@W over all
timesteps as one pre-loop GEMM), then fused cell kernels (mx.fast /
custom Metal) — that is the path to the small-model "Metal > 4090" goal.

## Post-pass-3 full-suite rerun (same session, scratchpad m5-metal-opt.csv)

208 configs vs stored m5-cpu/4090 CSVs. noscan median 1.43x faster than
pre-opt metal (max 5.9x); scan median 0.93x = flat within the ~15% thermal
drift band (same-config reruns of unchanged code drifted 90->110ms).

Crossover vs CPU (scan mode): consistent metal wins from ~700k weights
(near-universal >1M: 33/34); scattered wins from ~60k on non-recurrent
specs; below 100k recurrent CPU is 9-25x ahead. vs 4090: scan median
6-9x behind in every bucket; noscan median ~2.4-2.9x behind but metal
already BEATS the 4090 on 7/24 of the tiniest noscan configs (the
dispatch-overhead war: our 33us floor vs jax-GPU's ~4ms noscan steps).
Worst rows remain b4l1024 recurrents (db02/03/07/08: 100-800ms vs cpu
0.12-5ms vs 4090 13-131ms) — precisely the persistent-kernel targets;
prototype's 17x would put db02-b4l1024 at ~6ms, under the 4090's 13.8.

## msl_scan persistent-kernel codegen (implemented same session)

Generic recognizer+codegen (src/metaljax/msl_scan.py): counted loops with
elementwise bodies -> one generated Metal kernel, thread per lane, state
in registers. Handles fwd and AD-generated bwd loops (reverse affine
indexing), scalar bool carries, broadcasts, hoisted-constant folding.
Isolated affine scan H256/B16/L256: 5.35 -> 0.08ms (cpu 0.22); fwd+grad
15.4 -> 0.07ms (cpu 0.73). Suite: db03-b4l1024 108.7 -> 4.8ms (beats
4090's 13.6; cpu 0.44 still ahead at 33 weights), db09-b128l128 12.1 ->
1.00ms (fastest of all three systems), db06-b64l256 19.4 -> 2.1ms.
rglru texmo training converges (loss 5.18 vs cpu 5.03, different seeds).
Remaining for pass C: matvec cells (rnn/gru/mgru/lstm) need the
cooperative threadgroup-per-batch-element variant.
