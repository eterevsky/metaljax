# Row 10 (DeepSeek-V2-Lite / maxtext): the ragged_dot cliff, diagnosed and fixed

**2026-08-28.  Post-0.11.6 optimization campaign, row 1 of N.**
Baseline: 1948.2 ms/tok (0.11.6 gate cell) vs mlx-lm's 10.6 — a 184× gap.
Logs: `~/.cache/metaljax-bench/logs/row10-opt/`.

## Comparator provenance (the ᵖ in STATUS.md)

Re-ran mlx-lm (bench venv, mlx-lm 0.31.3 / mlx 0.32.0) on the original
`deepseek-ai/DeepSeek-V2-Lite-Chat` repo: **10.5 ms/tok decode**, 64 tokens,
greedy, `mx.get_active_memory()` = **31.4 GB** — the full bf16 model
resident, no quantization.  The 10.6ᵖ cell is a legitimate bf16 comparison;
our maxtext row also computes in bf16 (`weight_dtype=bfloat16`, activations
bf16, `preferred_element_type=self.dtype`).

## Diagnosis: where 1948 ms/token went

The instrumented rerun reproduced the cell (1958.7 ms/tok, identical decode
text).  One decoded token = one `jit__generate_jit` execute: 89 tape
entries, 9 compiled calls, 5 flushes of which 2 trimmed the pool — not a
dispatch-count pathology.  The engine's own bytes model priced the execute
at **297 GB of movement**, 237 GB of it in the 26-iteration MoE layer scan
(9.1 GB/layer/token).

Root cause: maxtext's only non-Pallas sparse path (`sparse_matmul=true`)
calls `jax.lax.ragged_dot`, and jax's lowering for every backend without a
native ragged_dot (i.e. everyone but TPU/GPU-Triton) is
`_ragged_dot_general_impl` — "ragged_to_dense": broadcast the rows to
`[g, m, k]`, mask by the `cumsum(group_sizes)` intervals, contract against
the full `[g, k, n]` expert stack over BOTH g and k.  maxtext additionally
pads the rows to the ragged tiling (512).  For decode (1 token × top-6 = 6
real rows) each of the 26 MoE layers therefore ran three
`[64, 512, 2048] × [64, 2048, 1408]` dense GEMMs — **~910× the FLOPs of the
6 rows × 6 live experts actually needed** — plus the mask materialization,
plus a transpose-materialized copy of the weights from the emitter's
contracting-pair order, plus a 370 MB `dynamic_slice` COPY per weight per
layer (the scan carries the stack `[26, 64, 2048, 1408]` as a transposed
view of the checkpoint layout; MLX's dynamic slice is a copy because the
offset is data).

Microbenchmark (real shapes, `microbench_ragged.py`): one ragged dot =
**20.75 ms**, of which the pure dense GEMM (cell D) is 18.2 ms — the row is
FLOP-bound on a 910×-inflated GEMM, not bandwidth-bound.  78 dots/token ×
20.75 ms = 1619 ms ≈ the row.  The active-experts floor (cell E) is
0.26 ms/dot.  Mask tree alone: 1.3 ms.  Pair-order transpose copy:
1.8 ms.  Slice copy: +1.3 ms.

## Fixes (all engine-side; no benchmark-code changes)

1. **Ragged-dot recognizer** (`metal/metal_ragged.cc`, opcode
   `metaljax.ragged_dot`, `runtime/emits.cc`): matches the exact
   ragged_to_dense fingerprint — pad → broadcast → iota/cumsum interval
   mask → select → dot_general contracting `[2,0]×[1,0]` — and emits ONE
   `mx::gather_mm(sorted_indices=true)` over the real (pre-pad) rows, the
   per-row group index recovered from the same cumsum
   (`eid_i = #(ends <= i)`, nondecreasing by construction).  Rows the dense
   mask zeroes everywhere (tiling pad, anything past `cumsum[-1]`) come
   back as the exact zero rows the dense form computes.  Two documented
   deviations, both "implements ragged_dot's contract rather than the
   fallback's accidents": non-finite weights in never-selected experts
   (dense: 0×NaN pollutes; gather: never read), and non-partition
   group_sizes (nothing a bincount produces).  Mixed-dtype dots
   (`preferred_element_type` ≠ input dtype) decline — the gather would
   round the output.  The walk follows `func.call` symbols (maxtext outlines
   scan bodies; the region-only walk the moe recognizer uses never sees
   inside them — which is also why AnalyzeMoe's candidate scan never met
   this dispatch).  `METALJAX_RAGGED=0` disables.

2. **Stacked-weights extension** (same file): proves `w` is
   `dynamic_index_in_dim(stack, layer)` out of a pass-through carry whose
   init is a `[1,0,2,3]` transpose of the contiguous `[g, L, k, n]`
   checkpoint argument, absorbs the helper call (the 3 × 370 MB/layer
   copies), and gathers matrix `group·L + layer` straight out of the
   original buffer — the emit's transpose-back + flatten is a zero-copy
   view, and `gather_mm`'s kernel takes the expert axis stride from the
   array (`batch_stride_b = b.strides()[-3]`), so no copy anywhere.
   Needed one splice accommodation: `Inline` unbinds a callee argument
   whose operand a recognizer absorbed (the call-aware use-count rule
   proved nothing reads it).

3. **Unit-axis reshape as a view** (`runtime/ops_shape.cc`): a reshape that
   only inserts/removes size-1 axes now lowers as `mx::squeeze` /
   `mx::expand_dims` — views — where `mx::reshape` copies any
   non-row-contiguous operand.  This is jax's `dynamic_index_in_dim`
   (dynamic_slice + reshape) over any transposed stack.

4. **Contracting-pair-order canonicalization**
   (`metal/metal_lowering.cc::LowerDotGeneral`): the contraction pairs of a
   multi-contract dot are jointly reordered so the BIGGER operand's
   contracting dims come out ascending — its canonical transpose is then a
   no-op instead of a whole-operand materialized copy (jax's ragged
   fallback lists `(k, g)` against `[g, k, n]` weights).  Output-invariant
   (batch pairs untouched).

## Results (row-10 protocol: budget 105, sys 110, rss 112, decode 8)

| cell | 0.11.6 baseline | fixes 1+3+4 | + fix 2 (stacked) |
|---|---|---|---|
| decode ms/tok | 1948.2 / 1958.7ᵣ | 306.5 | **113.9** (confirm 113.4) |
| prefill ms | 2028 / 1960ᵣ | 555 | 186.0 (confirm 185.1) |
| warmup s | 20.8 | 6.3 | 4.3 |
| peak footprint GB | 89 / 95ᵣ | 85 | 91 |
| per-token profile | 5 flushes, 2 trims | 5 flushes, 2 trims | 5 flushes, **0 trims** |
| decode text | " Paris.\n\nThe official language is" | identical | identical |

ᵣ = instrumented rerun of the baseline binary, same protocol.
**17.1× on the row cell**; measurement binary frozen as
`~/.cache/metaljax-bench/frozen-row10-ragged-3debbc33.dylib` (tree 7c54b41
+ this diff).  One load-phase mem_guard slope kill on the first stacked
attempt (projected 114 GB at +13 GB/sample during checkpoint restore — the
same ramp every baseline run had; clean kill, no panic, rerun completed).

No-regression evidence, all on the final build: execute_test all-pass (575
ok lines; incl. 10 new ragged / scanned-stack / reshape-view / pair-order
cases), ingest_test 0 failed, texmo_gate --limit 20: 20 ok / 0 decline /
0 fail, row 11 = 16.15 ms/tok (0.11.6 cell 16.35), row 14 = 31.40 (cell
31.85), token streams identical to their gate records.

## Remaining gap (113.9 vs mlx-lm 10.5, ~11×) — attribution

The pathology is gone; what remains is dispatch density, not a wrong path:

* ~16 ms/token is the maxtext-harness floor at ANY size (row 11's 0.6B
  dense model decodes at 16.15 ms through the identical machinery).
* Genuine streaming is now ~5 GB/token (6 live experts × 3 × 26 layers
  ≈ 2.7 GB, shared experts ≈ 0.9, attention ≈ 0.8, logits 0.4) ≈ 10-15 ms.
* The rest is per-kernel dispatch across ~26 layers × ~25-30 launches
  (MLA attention chain, the router's top-k/sort/scatter/cumsum bookkeeping,
  3 gather_mms, shared-expert FFN) plus 5 flush syncs — the same
  launch-bound decode class as the sub-ms texmo rows and the 0.11.0
  gemma decode finding.  mlx-lm's layer is ~9-10 launches with none of the
  router bookkeeping.  Closing it is a decode-loop launch-amortization
  campaign (deeper fusion / persistent decode kernels), not a row-10 fix.

Prefill (186 ms for 64 tokens ≈ 2.9 ms/token-in-batch) no longer resembles
the decode cost — consistent with the launch-bound reading.
