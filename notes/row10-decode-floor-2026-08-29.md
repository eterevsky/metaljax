# Rows 10/11: the decode floor decomposed, and the stacked-weight fixes

**2026-08-29.  Post-ragged campaign phase 2** (row 10 was 113.9 ms/tok after
`notes/row10-ragged-dot-2026-08-28.md`; mlx-lm's verified bf16 cell is 10.5).
Logs: `~/.cache/metaljax-bench/logs/row10-opt2/`.

## The floor, isolated (row 11 = qwen3-0.6b through the same harness)

New comparator provenance: **mlx-lm on Qwen/Qwen3-0.6B decodes at 3.0
ms/tok** (bench venv, mlx-lm 0.31.3 / mlx 0.32.0, greedy `temp=0.0`, 64
tokens, `mx.get_active_memory()` = 1.19 GB — the full bf16 model resident).
Our row-11 cell was 16.35 (0.11.6) / 16.86 measured here — a 5.6x gap on a
0.6B dense model.

Instrumentation (this session, dev-only, env-gated): `METALJAX_TIMING=1`
prints a per-execute phase report from `RunOnce` (lock/args/tape/run —
split into graph building vs blocking at sync points via two new `g_stats`
wait counters — /eval/contig/wrap) plus a to-host transfer timer; a
scratchpad COPY of adapter_maxtext.py brackets the decode loop per token
(split / generate / token-pull).  Diagnosis tooling, not a benchmark tweak.

Where a 16.86 ms row-11 token went (probe + timing, decode 64):

| component | ms |
|---|---|
| `jax.random.split` (own execute, threefry_split 208 entries) | 0.58 |
| `engine.generate` execute, total | 16.19 |
| — jax-side dispatch above PJRT | ~0.08 |
| — RunOnce: args wrap + tape pick + buffer wrap | ~0.04 |
| — `Program::run` (graph build; ONE compiled-main replay, 0 flushes) | 1.53 |
| — **`mx::eval`** (device + MLX schedule/encode) | **14.54** |
| token pull (`np.asarray`, 12-byte to_host) | 0.07 |

So the maxtext-harness "floor" is NOT protocol/dispatch cost: PJRT entry,
buffer wrap, donation bookkeeping and jax dispatch total well under 0.2
ms/token.  The generate main is already ONE mx::compile graph replayed once
per token, with zero eager flushes.  The 14.5 ms is the graph itself.

What the graph was doing (standalone repro: the captured
`jit__generate_jit` module through `run_stablehlo_bench.py`, 18.7 ms/exec
with seeded random inputs):

* **~3.8 ms: per-layer weight-slice copies.**  maxtext (`scan_layers=true`,
  `param_scan_axis=1`) stacks every weight on a mid axis (`[1024, 28,
  3072]`), hoists layer-major transposes out of the scan, carries them as
  pass-through carries, and reads layer i in the body as
  `dynamic_index_in_dim(stack, i)` — jax outlines it as a
  `@dynamic_index_in_dim` helper whose body is
  `reshape(dynamic_slice(...))`.  MLX's dynamic slice is a COPY (the
  offset is data): ~31 MB of weights re-materialized per layer per token,
  ~1.7 GB/token of copy traffic on a 1.2 GB model.  Measured by a
  static-slice variant of the module (18.7 -> 14.6/15.1).
* **The rest: graph bulk.**  The layer body is ~200 stablehlo ops
  (attention with separate prefill/ar cache spans, segment masks, rope,
  norms, cache DUS bookkeeping), unrolled 28x into a ~5600-node trace ->
  ~800+ kernel launches plus MLX per-node schedule/encode inside `eval`.
  Discriminators: `no_fuse` costs only +1.5 ms (fusion is already doing
  what it can), eager (`METALJAX_COMPILE=0`) +9.6, command-buffer cadence
  (`MLX_MAX_OPS_PER_BUFFER` 800 -> 100k) ~0.  mlx-lm's equivalent step is
  ~300 launches.
* **NOT the GPU-arch pin**: `MLX_METAL_GPU_ARCH=applegpu_g16g` (the f32
  matmul accuracy default) costs a bf16 decode nothing — mlx-lm under
  g16g still decodes at 3.0 (row 11) / 10.5 (row 10), and metaljax under
  the native arch still takes 16.8.

## Engine fixes landed

1. **Stacked-weight dot recognizer** (`metal/metal_stacked.cc`, opcode
   `metaljax.stacked_dot`, handler in `runtime/emits.cc`;
   `METALJAX_STACKED_DOT=0` disables).  `dot_general(x,
   dynamic_index_in_dim(stack, layer))` — both the outlined-helper and the
   inline `reshape(dynamic_slice(...))` spellings, any slice axis — becomes
   ONE `mx::gather_mm(x_[1,M,K], stack_view_[L,K,N], [0], [layer])` reading
   the ORIGINAL contiguous stack in place: MLX's gather kernels take the
   batch and leading strides from the array (`ensure_batch_contiguous` /
   `check_transpose` accept any row stride with unit column stride, all
   four dispatch paths), so an `as_strided` view over the buffer needs no
   copy anywhere.  The geometry — the contracted / free stack axes must
   collapse to single strides with the free stride 1, the stack must hoist
   (through pass-through carries and at most one loop-invariant transpose)
   to a @main argument — is proven at analysis; anything else Bails to the
   ordinary chain (row 10's o-proj layout, whose contracted axes straddle
   the stack axis, correctly declines).  The dense chain's index clamp is
   reproduced exactly.  Matches on row 11: 6 of the 7 per-layer weight
   dots (mlp x3, q, k, v; per-layer copies drop from ~31 MB to ~4 MB).

2. **K=1 dots are broadcast multiplies** (`runtime/ops_linalg.cc`).  jax
   spells `x * vec` einsums as batching-only dot_generals (maxtext's norm
   scales and rope: 4 per layer, B=1024, M=N=K=1), and each one lowered to
   a 1024-batch 1x1x1 GEMM launch.  A K=1 contraction has no sum, so the
   handler now emits `mx::multiply` — bit-identical (the matmul path
   computes the exact product in its f32 accumulator and rounds once,
   which IS the elementwise multiply's RNE for bf16/f16/f32) and fusable
   by mx::compile.

3. **Bytes-gate audit** (`metal_lowering.cc`, METALJAX_DEBUG=1): when a
   compile bytes gate fires it now prints the estimate aggregated by op
   name, one level into while bodies — the tool that showed row 10's
   93.5 GB main estimate to be dominated by charged-at-full-size
   loop-invariant stack transposes (MLX views) plus stale per-iteration
   slice charges.

## Results so far (row-10 protocol where applicable)

* Row 11 (16.86 baseline this session; 16.35 = the 0.11.6 cell): **13.61
  ms/tok** with fix 1 alone (eval 14.5 -> 11.4; one compiled call, zero
  flushes, unchanged ~0.7 ms split+pull); 13.51 with fix 2 as well — the
  k=1 rewrite is noise-level HERE (the 112 tiny gemms sat inside the
  compiled graph), its win shows on the keras control row instead.  The
  CLEAN protocol cell (real adapter, no debug/timing): **13.28 ms/tok**.
* Standalone generate-21 module: 18.7 -> 15.9 ms median, `check PASS`
  vs a jax-CPU reference (max_norm_err 6.7e-3, bf16-level).
* execute_test: all cases match the CPU backend (incl. 6 new stacked
  cases: scan + inline spellings, bf16, clamped index, multi-axis free,
  non-collapsible decline); ingest_test 0 failed.
* **Token-stream note (report both cells)**: the gather_mm gemv reduces in
  a different order than slice-copy + matmul, so bf16 greedy decode can
  flip argmax on near-ties: row 11's text diverges from the 0.11.6 gate
  record at token ~9 (" Paris. The capital of France is also the capital
  of the French Republic..." vs "... The capital of Italy is Rome...").
  Same class as any reduction-order change; the standalone module check
  passes elementwise at bf16 tolerances.

## Row 10: the dispatch-shape theory measured dead

Row-10 protocol cells on the final binary (stacked + k=1), probe harness:

| cell | ms/tok | run (host) | loopwait | eval | text |
|---|---|---|---|---|---|
| ragged build (0828 confirm) | 113.9 / 113.4 | — | — | — | gate record |
| F1: stacked+k1, default gates (probe harness) | 107.8 | 104.2 | 42.9 | 1.1 | identical |
| F2: + METALJAX_COMPILE_BYTES_MB=262144 (whole main compiles) | 109.1 | 3.2 | 0 | 106.5 | identical |
| **H1: CLEAN protocol cell** (real adapter, no debug/timing) | **108.7** | — | — | — | identical |

F2 is the experiment that matters: ONE compiled graph per token — replay
3.2 ms, zero flushes, zero loop syncs — and the token costs the SAME.  The
row-10 gap is not dispatch shape, flush syncs or any "harness floor"
(protocol overhead measured < 1 ms/token); it is the graph's own
device+encode cost: ~20k nodes/token at ~5.4 us/node all-in, against row
11's 0.84 us/node — i.e. the ~78 expert gather_mms (~20 ms at the 0.26
ms/dot active-expert floor), ~15 ms of genuine streaming, and ~70 ms of
MLA-attention / router / shared-expert machinery whose per-layer op count
(~700 tape nodes/layer) is the wall.  Peak footprint 82-84 GB (was 91).

Getting meaningfully under 50 ms/tok is an op-count campaign inside the
layer body — fuse the maxtext MLA decode attention (two cache spans,
segment masks, joint softmax) into the sdpa fast path, collapse the router
bookkeeping — sized by these numbers at roughly 4-5 ms per 1000 nodes
removed.  Engine-side, well-scoped, and NOT attempted here on top of the
validation budget.

The compile bytes gate itself was still a real finding: the audit shows
the fused main priced at 89.1 GB = 59.9 GB from the two scans (whose
per-iteration charge is dominated by the layer callee) + 29.2 GB from 25
loop-invariant stack transposes that are zero-cost views at run time.  A
transpose-as-view repricing would let decode-shaped mains compile at the
default gate — worth landing only once a compiled main WINS something
(today it is cost-neutral).
* Three clean governor refusals before the row-10 cells ran, all the same
  knife's edge: the load's single 8.9 GiB expert-stack transfer arrives
  with ~90 GB already claimed machine-wide, and the governor's DEFAULT
  sys ceiling (96 GB = 75% of RAM) is 9 GB BELOW the row protocol's own
  stated budget (mem_guard 105, GUARD_SYS_GB 110).  The 0.11.5/0.11.6
  row-10 cells threaded it with ~0.2 GB to spare on a ~13 GB-clean
  machine; a parallel session's downloader (+2-5 GB baseline) tipped it
  over — RESOURCE_EXHAUSTED at transfer, no panic, machine fine, three
  times.  The reruns pin METALJAX_MEM_SYS_MB=107520 (105 GB, still under
  the 110 guard) — a run-env alignment of the governor with the row's
  documented budget, flagged here for Oleg.  Big-load settle gates now
  also REQUIRE claimed < 16 GB (abort, never "proceed on stuck").

## Validation (final binary: stacked + k=1)

* bazel test //... PASSED; execute_test 581 ok / all cases match the CPU
  backend (incl. the 6 stacked cases); ingest_test 0 failed.
* texmo_gate --limit 20: 20 ok / 0 decline / 0 FAIL (training graphs and
  their cadences unmoved).
* Row 14 (maxtext-qwix-int8, decode 64): 31.08 ms/tok (0.11.6 cell 31.85,
  ragged-session spot 31.40) — small improvement, and its token stream is
  BIT-IDENTICAL to the prior record (stacked declines on the s8 dequant
  chain by dtype rule; k=1 is bit-exact).
* Row 2/4-class dense control: gemma4-e2b-bf16 (keras route, shipped
  defaults) 25.8 ms/tok vs the 0.11.6 cell 27.2 — improved, no
  regression (prefill 19.6 vs cell-era ~comparable).
* Row 10's token stream: identical to the gate record in every cell.
* Row 11's token stream: CHANGED (documented above); the standalone
  module check against jax-CPU passes elementwise at bf16 tolerances.
