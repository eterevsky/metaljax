# P19: the two pack optimizations P17 deferred (2026-08-13)

P17 ported `src/metaljax/qmm.py`'s matching and its packing and deliberately
left out two things, naming them: `_Source`'s **row-blocked evaluation** and the
**cross-executable build cache**. Both are memory disciplines over an answer
that does not move — and P18 measured what leaving them out costs. Row 7
(gpt-oss-20b) recognizes everything (94 qmm, 47 gathered expert dispatches, 188
packs) and then dies building those packs: guard kill at 46 GB under the row's
historical 45 GB budget, 62 GB under 60, against Stage 1's 25 GB for the same
model. Row 13 (E2B keras-int4) peaks at 48 GB for a 3 GB model.

Both are ported here. `src/metaljax/qmm.py` is the specification, as it was for
P17; what is different in C++ is written down below.

## 1. Row-blocked evaluation

**What it is.** Packing has to see every element of the reconstructed weight —
the exactness checks are what the whole rewrite rests on — but it never has to
see them all at once, because what it derives from them (a 4-bit code, one
scale per group) is an eighth of their size. So the operand subtrees are
evaluated one slice of the weight's leading axis at a time and packed as they
go. Every op on the way must be row-local: the slice of the result must BE the
result of the slice.

**How it is done here, and why not the Python way.** Stage 1 walks the IR with
its own little evaluator (`_Source._ev` / `_op`), calling the interpreter's
handlers on block-shaped arrays. This plugin has no interpreter to call — the
only thing that can compute is a `Program`, built by the `Lowering`. So the
work is split:

* `RowSource` (metal_qmm.cc) does the ANALYSIS. It is `_Source`'s demand walk
  with `_op`'s rules and nothing else: which values carry the weight's row axis
  (`Demand`/`DemandOp`), and a `NotBlockable` for anything not provably
  row-local. It produces a SET of values.
* `Lowering::LowerConeBlocked` (metal_lowering.cc) builds the cone as usual but
  declares every value of that set with `c` rows instead of the full extent.
  One line does most of it: `Dims()` — the single place a declared shape enters
  a lowering — returns the narrowed shape, so every shape derived from it
  follows (a reshape's target, a broadcast's interim, a gather's batch shape
  and its whole index plan). Only `stablehlo.slice` needed a second touch,
  since its extent lives in an attribute rather than in a type.
* `RowSource::Rows` runs that Program once per block, with every blocked @main
  argument sliced to the same rows, and the two packers loop over `blocks()`.

This is worth stating plainly: **the blocked evaluator is the ordinary
lowering, run on a narrower graph.** No second interpreter exists, no handler
is written twice, and a block is computed by exactly the code a whole
evaluation would have used.

**The safety argument.** A wrongly narrowed value is the worst failure this
project can have: the pack's exactness checks pass perfectly happily on the
wrong rows, so the weight would be silently wrong. There are therefore two
locks on that door, and they are in different files:

1. `RowSource::DemandOp` decides, by qmm.py's rules, which values may be
   narrowed. Anything else raises `NotBlockable` and the weight packs whole.
2. `Lowering::LowerOp` asserts the consequence at emission time: an op that
   READS a block must hand back a block, and only an op from `RowLocalOps()`
   may be narrowed at all.

The second lock earned itself immediately. `func.call` was missing from the
row-local set, so every MXFP4 weight — jax lowers `jnp.take` as a call, and
that is how an E2M1 grid is read — fell back to a whole evaluation. The guard
said so by name. Without it the fallback would have been silent and row 7 would
have gained nothing.

**Where this port is narrower than Stage 1's, on purpose.** Stage 1 keys its
evaluation on (value, demand) and may legally read one value BOTH ways — whole
for a table, blocked for the indices into it. A tape has one slot per value, so
reading a value both ways raises `NotBlockable` here and the weight packs
whole. Two further conservatisms of the same kind: a callee invoked twice in
one subtree declines (the lowering aliases a callee's parameters to the call's
operands, so one parameter cannot be two things), and a callee body is swept
after the walk, because the lowering splices a callee WHOLE — dead ops included
— and an op the demand walk never reached could otherwise be handed a block.
Each of these is a fall-back to the P17 behaviour, never a wrong answer, and
each is reported under `METALJAX_DEBUG=1`.

Which weights can block at all is qmm.py's `_blocking` unchanged: the leading
axis must already be the row axis of the `[(B,) N, K]` matrix
`quantized_matmul` wants (no transpose, `rperm` identity) and whole rows must
live inside one block (`per % K`). gpt-oss's `[E, N, K]` expert projections
qualify; a keras `Dense`'s `[K, N]` weight does not, on either stack, and packs
whole.

## 2. The cross-executable build cache

A pack is a deterministic pure function of exactly two things: the argument
buffers its reconstruction reads, and the reconstruction itself. Two
EXECUTABLES over one model share the first and duplicate the second —
keras-hub compiles a separate generate program per sequence-length shape, and
each gets its own plan whose matches hold no packs. So the same weights were
verified and repacked from scratch once per shape, and a full pack set was live
per shape.

The key is a canonical serialization of the reconstruction (`Fingerprint`, a
transliteration of qmm.py's `_Fingerprint`: values numbered by the order the
walk reaches them and never by SSA name, operands in operand order, attributes
sorted and printed in full, callees serialized through their BODY because jax
renumbers private helpers per program) plus the identity of the buffers it
bottoms out on. Anything the serialization cannot cover exactly — a
`dense_resource`, a constant of more than 1024 elements — declines to be cached
and is built exactly as before.

**The one real difference from the Python.** Stage 1's entry is weak
throughout: a Python weakref hands the object back, so an entry keeps neither
the weights nor the pack alive. An `mx::array` cannot be rebuilt from a weak
handle, so here the PACK is held strongly and the LEAVES weakly, through
`data_shared_ptr()`. The consequences are handled explicitly:

* An entry whose source weight has been freed is swept at the next insertion —
  which is what an unloaded model looks like from inside the cache.
* The bound (`METALJAX_QMM_BUILD_CACHE`, 512, `0` turns reuse off) evicts the
  least recently used one at a time rather than clearing, so a steady-state
  model's packs stay resident while a config-sweeping worker cannot grow the
  cache forever.
* Leaf identity is checked three ways on every hit, because `mx::array::id()`
  is the address of a refcounted descriptor and address recycling applies to it
  exactly as it does to a Python `id()` (this project has twice shipped a bug
  from a set keyed on an address alone): the weak handle proves the buffer is
  still alive, the data pointer proves it is the SAME buffer, and the shape and
  dtype prove it is the same view of it.

A hit does not re-verify, and must not: the pack was verified against these
very buffers, under this very reconstruction, earlier in this process. A miss
runs the full build with every element checked.

Invalidation is unchanged and independent: `MetalLoadedExecutable::Tape` still
compares `pack_arg_ids` per call and re-fuses when a caller hands over
different weight buffers, bounded by `kMaxRepacks`.

## Diagnostics

`METALJAX_DEBUG=1` now distinguishes a build from a reuse and says how the
build was done:

    qmm: packed 32x5760x2880 mode=mxfp4 bits=4 group=32 in 32 row blocks
    qmm: packed 4x256x128 mode=affine bits=4 group=128 whole
    qmm: reused 7x256x128 mode=affine bits=4 group=128
    qmm: <name> packs from a whole evaluation (<the rule that declined>)
    qmm: <name> is not build-cached (<what the fingerprint could not cover>)
    qmm: pack wave peak 0.134 GB

The pack-wave peak is read from the plugin's own libmlx. The host process's
`mlx.core` is a different runtime and its counters read zero for the plugin —
the P16 campaign's finding, and the reason the memory test is written this way
rather than with RSS (host RSS never sees a Metal allocation at all).

## Knobs

| variable | default | what it does |
|---|---|---|
| `METALJAX_QMM_BLOCK` | `1<<24` | weight elements per block; a block costs ~16 bytes per element while it runs |
| `METALJAX_QMM_BUILD_CACHE` | `512` | packs the cross-executable cache retains; `0` turns reuse off |

Both are qmm.py's names and qmm.py's defaults.

## Measured

Cells: `notes/data/p19-packing-models-2026-08-13.jsonl`; every run under
`guarded_run.sh` with the machine lock, sequential.

### Row 7 (gpt-oss-20b, MXFP4) — unblocked

| | peak | ms/tok | prefill | outcome |
|---|---:|---:|---:|---|
| Stage 1, same day | 26 G | **21.9** | 140.8 | reproduces its anchor exactly |
| native P18 | 46 G @45, 62 G @60 | — | — | **guard-killed** |
| native P19 | **35 G** | **25.3** | 153.8 | ok, 128 tokens, **1.16×** |
| native P19, `METALJAX_TRACE_BUDGET=1e7` | 34 G | 25.3 | 152.4 | *(labelled secondary cell)* |

Four P19 samples: 25.3 / 25.5 / 25.3 / 25.3. The target was ~1.5× and P17's
micro proxy predicted 1.55×; the row lands at 1.16×.

**Which port did it is not what P18 predicted.** One knob at a time, same
budget:

| configuration | peak | outcome |
|---|---:|---|
| both (P19 default) | **35 G** | ok |
| cache off, blocking on | 46 G | **guard-killed** — P18's number to the gigabyte |
| blocking off, cache on | 36 G | ok |
| neither (P17/P18) | 46 G / 62 G | guard-killed at both budgets |

The **cache** is the load-bearing half: three executables were each building
their own ~10 GB pack set. Row-blocking is worth a further gigabyte here, which
means P17's argument — "the tape already stages op by op with last-use pruning"
— was substantially right about the per-weight TRANSIENT and simply did not
cover the same pack set being built three times. Only the pair clears 45 GB.

Mechanism, from the run's own log: **94 packs built, 188 reused** (a 100 % hit
rate on the second and third executables), every one of the 94 blocked (47 in 16
row blocks, 47 in 32), and the pack-wave peak **33.9 GB → 0.000 GB → 0.000 GB**
— the reuse waves allocate nothing at all. That 33.9 is a process-wide
high-water mark and so includes the resident model; the flight-log footprint is
the number to compare across rows.

Row 7 does **not** have row 13's compile problem. `METALJAX_TRACE_BUDGET=1e7`
gives the same 25.3 ms/tok with the compile decisions bit-identical — 16
compiles / 354 compiled calls either way — so its decode loop was already
compiling and item 6 of the P18 frontier does not touch this row.

### Row 13 (gemma4-E2B keras-int4) — neutral in time, 1 GB in steady state

| | peak | steady | ms/tok |
|---|---:|---:|---:|
| native P18 | 48 G | — | 249.0 (and 274.6 on its own byte-cap control) |
| native P19 | 46 / 48 G | **3.2 G** | 275.6, 276.3 |
| P19 OFF, same binary | 48 G | 4.2 G | 271.7 |

P19 is **timing-neutral** here: the off-control on the same binary reads 271.7
against 275.6, and P18's own byte-cap control — a configuration that behaves
identically — read 274.6, so 249.0 was the low end of this row's spread rather
than a mark P19 missed. What P19 changes is the steady state (4.2 → 3.2 GB) and
the work: **259 packs built, 518 reused** across three executables, 0
fingerprint declines, the two reuse waves peaking at 0.000 GB.

All 259 pack **whole**, on both stacks: a keras `Dense`'s `[K, N]` weight needs
a transpose to reach the `[(B,) N, K]` matrix `quantized_matmul` wants, which is
exactly the precondition `_blocking` tests. So this row was never blocking's to
win.

**And its 46 GB peak is now attributed rather than open.** The pack wave peaks
at 6.64 GB against a 46 GB flight peak, and Stage 1's own load transient on this
row is 44 GB (already recorded in the baseline's regression section). The peak
is the keras streaming load, not the packs.

### Scrutiny

Row 7's greedy tokens diverge from Stage 1 at index 52 of 64. P18 never
completed this row, so there is no prior native record and this is a first
observation rather than a change; it is the same late-divergence ladder class
already carried for rows 3, 5 and 11. The cache cannot be the cause — a hit
hands back the identical `mx::array` objects — and blocking is pinned byte-for-
byte by the execute_test row; but the ladder is still owed on this row.

## Battery

`notes/data/p19-packing-battery-2026-08-13.txt`. execute_test 520 -> **524**
checks (the four new rows are described there), `texmo_gate` **106 ok / 0
decline / 0 FAIL**, smoke, decline_census 35/35, ingest_test 8/8, coexist_test,
`bazel test //...`, and the native wheel from a fresh 3.13 venv.
