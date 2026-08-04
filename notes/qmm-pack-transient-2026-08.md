# qmm pack transient: 15.9 GB -> 1.5 GB per pack (2026-08)

Trigger: TASKS.md "qmm pack transient (row 7 re-measure blocker)". The
per-pack ACCUMULATION was already fixed (02808b3 — `gc.collect()` +
`mx.clear_cache()` per pack in `qmm.prologue`), leaving base memory stable
at ~28-31 GB through the gpt-oss-20b row. What still killed the row was a
single pack's SPIKE: 9-15 GB inside one 5 s guard sample, which trips
mem_guard's trajectory rule (`f + 2*(f - prev) > budget`) at budgets 45
and 55. Oleg's directive: no budget creep — shrink the transient.

Measured standalone on the REAL checkpoint
(`~/.cache/huggingface/.../gpt-oss-20b`, layer 0), the packed loader's own
in-graph dequant chain (nibble unpack, 16-entry E2M1 table gather, per-32
power-of-two scale) driven through the recognizer with nothing else
running. Machine: M5 Max, base claimed ~13.5 GB.

## Diagnosis

`gate_up_proj`: codes `[32, 5760, 1440]` u8 -> weight `[32, 5760, 2880]`
= 530.8M values, 0.26 GB packed. Per-op trace of the pack:

| stage | array | GB |
|---|---|---|
| and / shift / concatenate / reshape | u8 chain | 0.25 x2, 0.49 x2 |
| convert to i32 (the gather's indices) | i32 | **1.98** |
| `func.call @_take` (jax's take wrapper) | own temporaries | **~4.45** |
| gathered E2M1 values | bf16 | 0.99 |
| scale operand (broadcast per 32) | bf16 | 0.99 |
| `mxfp4_codes` (8 grid steps x where/or) | u8 + bool chain | peak 5.9 |
| `pack_codes` (codes as u32 + 8 shifts) | u32 | peak 4.0 |

Three separate multipliers, none of them the weight itself:

1. **jax's `take` wrapper is three int32 copies wide.** `jnp.take(table,
   idx)` lowers to a private `@_take` function that clamps and range-checks
   the indices: `compare`, `add`, `select`, two more compares and a
   reduce — all at 530M elements, i32 at 4 bytes each. It alone peaked at
   4.45 GB above what was already live.
2. **`env` pins the whole subtree.** `_eval` memoized every intermediate
   and settled once at the root, so nothing could be freed until the last
   op finished.
3. **MLX frees into its own cache, not to the OS.** The cache limit
   defaults to the memory limit, so the pack's dead buffers stayed CLAIMED:
   live peak 7.68 GB, live+cache 15.93 GB, and `vm_stat` claimed (the
   guard's metric: wired+anon+compressor) went 14.0 -> 29.6 GB. That factor
   of two is the difference between "big" and "fatal" for the guard.

The MXFP4 pack really is a reinterpretation of bytes already in MLX's
layout — but only the RESULT is. Everything the verification has to see
(the decoded E2M1 values, the per-group scales) is reconstructed by the
graph first, and that reconstruction is ~16 bytes per weight element.

## Fix: pack one row block at a time

`qmm._Source` evaluates each operand subtree in slices of the weight's
LEADING axis and packs as it goes, so nothing bigger than a block is ever
live. Every op on the way must be provably row-local — the slice of the
result must BE the result of the slice:

* elementwise ops: free (`_ROW_LOCAL`);
* `reshape` / `broadcast_in_dim`: the evaluator rewrites the leading
  dimension of the declared result shape itself, after checking the op does
  not mix or expand that axis;
* `transpose` / `concatenate` / `slice` / `reduce`: allowed when their
  dimension attributes leave axis 0 alone;
* `gather`: only with the INDICES blocked and the table read whole (the
  handler reads `slice_sizes` off the op, so a blocked operand would have
  it reassemble the result at the full leading extent);
* `func.call`: evaluated INSIDE the callee — `interp.run_func` would run
  the body against the shapes it declares, and `@_take` is full of splat
  constants of the full leading extent that MLX would happily broadcast
  against a block instead of failing.

Blocked-ness is a DEMAND passed down from the weight, not a property read
off shapes: the gather's indices are blocked and its table is whole,
however the table's 16 rows happen to compare to the expert count. (A
bottom-up rule would slice a 16-entry decode table for a 16-expert model
and pack the wrong values — silently. There is a regression test.)

Anything not provably row-local raises `_NotBlockable`; `_build_pack`
retries with `_Source.unblock()`, where `blocks()` yields one block over
the whole weight and the pack code is unchanged. Small weights take that
path too (blocking a 4 MB reconstruction would only add syncs).

Two supporting changes:

* `_NoCache` turns MLX's buffer cache off for the duration of a pack
  (`set_cache_limit(0)`), so dead blocks go back to the OS instead of
  sitting in claimed memory. This is what makes claimed track the live set.
* `_eval` (the whole-evaluation fallback) now settles op by op and drops
  each intermediate as its last consumer reads it.

The verification is untouched and still covers every element: values are
read back off the E2M1 grid by exact integer bit comparison, group scales
must be exact powers of two, affine scale/zero maps must be constant within
their group — all per block, every block.

## Numbers (real gpt-oss-20b layer-0 tensors)

| tensor | metric | before | after |
|---|---|---|---|
| gate_up `[32,5760,2880]` | MLX peak live | 7.68 GB | **1.51 GB** |
| | peak live+cache | 15.93 GB | **1.51 GB** |
| | vm_stat claimed spike | +15.6 GB | **+1.5 GB** |
| | pack time | 0.48 s | 0.90 s |
| down `[32,2880,2880]` (batched dot) | MLX peak live | 4.09 GB | **0.76 GB** |
| | peak live+cache | 7.22 GB | **0.76 GB** |
| | claimed spike | +6.1 GB | **+0.8 GB** |
| | pack time | 0.27 s | 0.49 s |

Packed arrays (codes + E8M0 scale bytes) and the dot's output are BIT
IDENTICAL before and after, on both tensors, and also when the dot is
wrapped in a decode `while_loop` (weights arriving as loop carries).

Pack time roughly doubles: 32 blocks of one expert each, ~15 wide ops per
block, one sync per wide op (`_SETTLE_ELEMS`). Over gpt-oss-20b's 24
layers x 2 expert weights that is +16 s once, against a row that could not
complete at all. Dropping the per-op settle entirely measured 0.73 s with
the same peak at this block size, but the settle is what bounds a wider
chain or a larger `METALJAX_QMM_BLOCK`, so it stays.

The whole-evaluation fallback also improves, from the cache change: same
tensor, blocking disabled, claimed spike +15.6 -> +5.0 GB (live peak 7.68
-> 7.20; the staged `_eval` is worth only ~6% here because this chain's
peak sits INSIDE `@_take` and `mxfp4_codes`, not across ops).

## Knobs

* `METALJAX_QMM_BLOCK` (default `1 << 24` weight elements) — block size.
  One gpt-oss expert is 16.6M elements, so the default is one expert per
  block. Bigger = fewer syncs, proportionally bigger transient.
* `qmm.stats()["blocked"]` counts packs built block-wise (tests assert on
  it; `METALJAX_DEBUG=1` prints the reason when a weight falls back).

## Known gaps (not fixed)

* **Weights stored K-major do not block.** keras' `Dense.quantize("int4")`
  stores `[K, N]` and contracts dim 0, so the leading axis is the
  CONTRACTION: a block of it is a slice of every row, and the group scales
  and the packed nibble stream both run along K. `_blocking` returns None
  (`rperm != identity`, or `per % K`) and those weights pack whole — with
  the cache fix, at ~1/3 of the old claimed spike. Blocking along K is
  possible (concatenate along the last axis instead of axis 0, with blocks
  aligned to 128 columns) and is the obvious follow-up if a big keras-int4
  model ever needs it.
* **A gather whose OPERAND carries the blocked axis** (a per-group scale
  gathered by a `g_idx` ramp) falls back, because the shared gather handler
  reassembles its result from the op's `slice_sizes` attribute rather than
  from the array it was handed.
* **Full-size `stablehlo.constant` weights** fall back: a baked constant
  would have to be materialized whole for every block.
* `mxfp4_codes` (8 grid steps) and `pack_codes` (codes widened to u32)
  still peak at ~11x and ~8x their output when handed a whole weight. Only
  the fallback path sees that now; staging them would cost 16 more syncs
  per block on the path that matters.
