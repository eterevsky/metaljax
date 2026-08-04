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

# Pack WAVES: one build per weight per process (2026-08-04)

Follow-up to the same investigation. With the per-pack transient fixed, what
was left on gpt-oss-20b was the number of TIMES the wave ran. keras-hub
compiles a separate generate program per sequence-length shape, so a normal
benchmark run has three executables over one set of weights; each gets its
own `qmm.State` with its own fresh `Match`es, whose pack cache is empty, so
`_resolve` missed and `_build_pack` re-evaluated and re-verified all 94
weights -- ~0.9 s each, ~80 s per executable, three times
(`~/.cache/metaljax-bench/logs/qmm-transient/profile-defaults2.log`: warmup
95.7 s, and pack waves bleeding into timed windows). `_share` deduped the
STORAGE afterwards, which is exactly the evidence that the builds were
duplicates.

## The build cache

A pack is a deterministic pure function of two things: the argument buffers
its reconstruction reads, and the reconstruction itself. So a finished pack
can be reused iff BOTH are provably identical, and `_BUILT` keys on exactly
that pair:

1. **the buffers**, by identity -- `id()` is only the hash bucket, and every
   hit re-confirms with `is` against weak references (CPython recycles the
   addresses of freed objects; this project has shipped that bug twice).
2. **the reconstruction**, as a canonical serialization (`_Fingerprint`):
   op name, result types, and every attribute in full text sorted by name;
   operands in operand order; values named by the order the walk reaches
   them, never by SSA name; regions (reduce bodies) walked with their own
   numbering; and `func.call` serialized through the callee's BODY, not its
   symbol -- jax renumbers private helpers per program (`@_take`,
   `@_take_0`), and two modules can bind one name to different bodies.
   Leaves are numbered by the walk too, so two programs that take the model's
   weights in different argument orders still agree.

What the serialization covers on the match itself is exactly what
`_build_pack` reads: mode, K, N, bshape, rshape, rperm, nshape, recip,
sub_range, bcast_dims and the `post` shape ops. NOT `M`/`mshape`/`lperm` --
those describe the dot's ACTIVATION operand, they are precisely what differs
between two executables of one model, and no packed byte depends on them.

Anything that cannot be serialized exactly DECLINES (`_NoFingerprint`,
counted as `stats()["build_declines"]`, silent): a dense attribute over
`_FP_DENSE_ELEMS` (1024) elements, an attribute whose text runs past 64 KB, a
`dense_resource` (its blob name is not its contents), an unresolvable or
recursive callee, an unbound block argument. A decline costs nothing -- the
weight is built exactly as it was before this cache existed. Verification is
never weakened either: every miss runs the full build with every element
checked.

Entries are weak in BOTH directions -- they pin neither the weights nor the
packed arrays -- so a cached pack lives exactly as long as some `Match` holds
it, the same lifetime rule `_Pack` and `_SHARED` already follow. Dropping the
first executable therefore costs a rebuild, correctly.

`_share` stays for what the cache cannot prove: the same bytes arriving in
two different buffers (`test_identical_weights_from_two_buffers_share_a_pack`).
Two dots over one buffer set inside one program -- jax's prefill and its
decode-loop body -- are now answered by the build cache instead, so that pair
never builds twice at all.

Knobs: `METALJAX_QMM_BUILD_CACHE` (entries, default 512; **0 turns
cross-executable reuse off**), `stats()["build_hits"]` /
`["build_declines"]`, `METALJAX_DEBUG=1` names each reuse and each decline.

## Prologue cost trims that came with it

* **`gc.collect()` per fresh pack** (added when whole-eval packs left ~8 GB
  in the cache each) is ~100 ms on an LLM-sized heap -- 9.3 s of the 95 s
  wave, for 94 packs that with row-blocking leave ~1.5 GB rather than
  9-15 GB. Blocked builds now collect every `_GC_EVERY` (8,
  `METALJAX_QMM_GC_EVERY`); UNBLOCKED builds, the ones that actually spike,
  still collect per pack; and the end-of-prologue sweep now collects as well
  as clears, so the tail is caught either way.
* **`_hoist` memoized per `_Source`** (12.9 s of the same wave): it walks a
  while body to its terminator, and `list(body.operations)[-1]` costs the
  whole body -- a decode loop's body is the entire model. Every row block
  re-asked the same question about the same loop carries.
* **`moe._dead_sweep` on a worklist** (8.8 s per recognizer pass, i.e. per
  executable, and NOT something the build cache helps): it rescanned the
  whole main block to a fixpoint, once per MoE layer. An op's
  every-use-is-skipped test can only turn true when one of its users joins
  `skip`, so the only candidates worth re-testing are the defining ops of a
  newly-skipped op's operands. Seeded from the region and the root, that is
  linear in the region instead of quadratic in the program.
  `test_dead_sweep_worklist_agrees_with_the_fixpoint` runs the old fixpoint
  alongside the new sweep on the real MoE block and asserts the skip sets
  are equal, op for op.
* **`moe.emit_bytes` memoized on the match** (23.5k calls, 2.1 s on a
  gpt-oss decode profile): it reads only `m.order`/`m.P`/`m.out`, none of
  which change after analysis, and the cost estimator asks on every
  traversal of the block.

## Numbers (real gpt-oss-20b tensors, `pack_wave.py`)

Layers 0-1, gate_up + down = 4 MXFP4 weights (0.74 GB of codes), three
executables at token counts 2 / 5 / 3, prologue timed directly:

| wave | before (`METALJAX_QMM_BUILD_CACHE=0`) | after |
|---|---|---|
| 1 | 2.450 s, 4 packs | 2.450 s, 4 packs |
| 2 | 2.464 s, 4 more packs (4 `_share` aliases) | **0.0027 s**, 4 hits |
| 3 | 2.505 s, 4 more packs (4 more aliases) | **0.0028 s**, 4 hits |
| total prologue | 7.42 s | **2.46 s** |
| packs built | 12 | **4** |
| `_share` aliases | 8 | 0 (nothing duplicate is built) |
| MLX peak | 2.83 GB | 2.43 GB |

Extrapolated to the shape that motivated this (94 packs x 3 executables at
~0.9 s), the ~240 s of pack waves in a benchmark run becomes ~80 s, and only
the first one lands in a timed window.

gc.collect() calls the prologue makes, over 12 stacked blocked packs (the
synthetic `_affine_grouped` layout, `METALJAX_QMM_BLOCK=64`, two
executables):

| | wave 1 | wave 2 |
|---|---|---|
| `METALJAX_QMM_GC_EVERY=1` (the old per-pack rule) | 13 | 13 |
| amortized | 2 | 2 |
| amortized + build cache | 2 | **0** |

The wall-clock saving from the gc change does not show at this heap size --
a collect over a few hundred MB of arrays is milliseconds. It is the
LLM-sized heap that makes it ~100 ms a call, which is where the 9.3 s over
95 packs in the decode profile came from; what is measured here is the call
COUNT, which is what the rule controls.

## Known gaps (build cache)

* **Weak by design**: if the first executable is released before the second
  runs, the pack is gone and the second rebuilds. That is right (nothing
  else holds the arrays) but means the win depends on the caller keeping its
  shape variants alive -- keras-hub does.
* **A big baked constant declines.** A weight stored as a
  `stablehlo.constant` rather than an argument has no leaf buffer to key on
  and its value attribute is over the element cap, so it is rebuilt per
  executable. Those weights also do not block (see above), so they are the
  expensive case twice over.
* The fingerprint is recomputed per `Match`, i.e. per executable; on
  gpt-oss it is ~8 KB of text per weight and well under a millisecond, but
  it is O(subtree) and would not stay free for a subtree with thousands of
  ops.
