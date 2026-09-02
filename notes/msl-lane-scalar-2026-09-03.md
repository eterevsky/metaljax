# msl_scan: one scalar per lane is not a register vector (2026-09-03)

The last open member of the "carry/lane misclassification" family
(2125c96 guarded one shape of it; 7adb350 closed the counter half; the
2026-09-02 findings §6 named this one).  Vector-mode kernels read every
value's TRAILING dim as its register width.  A value that is one scalar
per lane -- a reduce over the feature axis, a vmapped scalar carry, a
per-step scalar input -- has no register dim at all, so its trailing dim
is a LANE dim, and the convention loaded, carried or wrote it as
`lane[-1]` registers: each lane broadcast its own scalar over a whole row
of the buffer.

## What was wrong (baseline binary 8833450a; probes and logs in `~/.cache/metaljax-bench/logs/msl-lane-scalar/`)

Exact-arithmetic probes (doublings of small integers, compared with no
tolerance) against jax-CPU:

| shape of the per-lane scalar | 1-D lane | 2-D lane |
|---|---|---|
| stacked output, bare reduce | declined (2125c96) | right by luck (width-1 write path) |
| stacked output, expression over a reduce | declined (2125c96) | **wrong** (88/96 elements) |
| scalar CARRY updated from a reduce | **wrong** (3/4 lanes) | **wrong** (15/16 lanes) |
| per-step scalar INPUT used bare with a reduce | **wrong** | **wrong** |

So the guard had covered one of six cells.  The carry and input cases
are the natural shapes of a vmapped scan that accumulates a per-example
loss or reads a per-example scalar per step.

## Why the 2-D case was left unguarded, and the real defect behind it

2125c96 kept 2-D lane spaces unguarded because "lrnn's (4,4) lane space
relies on the trailing lane dim doubling as the register axis".  The
(4,4) space was an artifact: lrnn's forward loop has a (B, F) state and
stacks the residuals of its unrolled inner loop as (reps, B, F)
`SymStack`s, and the lane-space computation counted the STACK AXIS as a
lane dim -- `reps`-fold redundant threads (db08-b4l1024's kernel ran 16
threads and never read `c0`).  On that space a (B, F) state with
B == F == reps is indistinguishable from one scalar per lane by shape,
which is exactly the ambiguity that blocked a general rule.

## The fix (plugin-native/metal/metal_msl.cc, metal_msl_emit.cc)

1. **Lane space by structure** (`StackedLaneDims`, `ValueLaneDims`): a
   `SymStack` contributes its parts' lane dims, never its stack axis; a
   stacked block contributes the dims between its step dim (if the squeeze
   left it) and its register dim; full-rank carries are folded first, and
   any value whose non-unit dims are already a suffix of the space is one
   scalar per lane and contributes nothing.  lrnn's forward loop now runs
   on lane (B,): db08-b4l1024 16 -> 4 threads, db07-b256l512 1024 -> 256,
   same kernel otherwise (`c1` -> `c0`).
2. **`MslLaneSuffix` / `LaneScalar` / `RegWidth`**: in vector mode a value
   whose non-unit dims equal a suffix of the lane space has register width
   1 and lane-positional offsets.  Applied at every site that derived a
   width or an offset from the trailing dim: `R`, `VecOff`, state
   load/update/final write, the carry-view check, iota, stacked and hidden
   writes.  (B, F) values are untouched -- one dim more than the lane
   space -- so every existing texmo kernel that was right stays
   byte-identical except for the lrnn lane shrink.
3. 2125c96's 1-D guard is subsumed: that loop takes a kernel now.
4. A pre-existing hole the guard had hidden: a rank-0 by-value input
   (a hoisted scalar reduce) broadcast to a register vector was declared
   `float w[4]` and never loaded -- NaN gradients on the 2125c96 cell the
   moment its loop planned.  Filled in both emitters.
5. Narration gains `lane=<dims>`; `texmo_gate.py` tags the narration per
   row under METALJAX_DEBUG so a plan census can be attributed.
6. Folded-in minor items from the 2026-09-02 findings: the nested-while
   inner counter takes its dtype from the block argument (was assumed
   i32); the stacked-write index acceptance documents that affine-index
   coverage of the buffer is guaranteed by jax's lowerings only.

## Tests

* `tests/test_msl_scan.py`: six new cases (carry / input / stacked
  expression, each in 1-D and 2-D lane spaces, square and 3x4), EXACT;
  the 2125c96 test now runs through a kernel (its gradient was the NaN).
* `plugin-native/execute_test.py`: five EXACT lane-scalar rows, the
  2125c96 gradient row, and the contract "one scalar per lane takes a
  kernel" -- a METALJAX_DEBUG child must plan all five in vector mode on
  lanes `4 / 3,4 / 4 / 4,4 / 3,4`, so a decline cannot pass them.

## Gates (after binary 71e24bb3, before 8833450a; machine lock held)

Logs, narration, bench jsonls and both frozen binaries:
`~/.cache/metaljax-bench/logs/msl-lane-scalar/`.

**Plan census** (`census.py` over the METALJAX_DEBUG narration of both
arms, one `### config` tag per row): 106 rows, **97 identical plan sets,
9 rows with the same plans on a smaller lane space, 0 plan-set
differences** -- no configuration lost a kernel, none gained one.  The
nine are the lrnn family, the reps-fold shrink: db07-b4l1024 and
db08-b4l1024 16 -> 4, db07-b256l512 and db12-b256l512 1024 -> 256,
db08-b128l128 512 -> 128, db10-b16l128 32 -> 16, db10-b64l256 128 -> 64,
db12-b4l1024 16 -> 4, synth-matlstm-a 256 -> 64.  (mid14's lrnn.512.2
was already on (B,): its residual stacks squeeze cleanly.)

**texmo_gate.py, 106 configs, 8 steps/chunk** (whole model vs jax-CPU):

| arm | result |
|---|---|
| before 8833450a | 106 ok (24 via sensitivity scaling), 0 decline, 0 FAIL, 0 error |
| after 71e24bb3, run 1 | 105 ok, **1 FAIL: mid03-b64l128** (`bits.4.oh+bp\|lstm.128`, worst 2.88e-02 vs tol 1.9e-02, out[1]) |
| after 71e24bb3, run 2 | 106 ok (23 via sensitivity scaling), 0 decline, 0 FAIL, 0 error |

The mid03-b64l128 failure is NOT this change.  The row is coop mode
(lane (64,128)), whose emitter this change does not touch; its plan
census is identical on both arms, and the generated MSL of both its
kernels is **byte-identical** between the two binaries (317 lines,
`mid03-before.msl` / `mid03-after.msl`).  Re-run three times per binary
it passed 6/6, worst 1.1e-04 .. 1.0e-03 against tolerances 1.6e-02 ..
1.0e-01 -- and its 1-ULP sensitivity itself varies run to run (3.2e-05
.. 2.1e-04), i.e. the row's Metal-side arithmetic is not bit-reproducible
(the gather/scatter path; item 10's GPU scatter-add order).  Run 1 hit a
~1.5x tolerance excursion of that noise.  Reported here rather than
smoothed over; a flaky-row watch item for the gate, not a regression.

**Perf, `scripts/bench_texmo_pjrt.py` (256-step chunk, best of N) on the
lrnn family plus two controls, both frozen binaries, lock held:**

| row | before ms/step | after | after/before |
|---|---|---|---|
| db07-b4l1024 | 24.218 | 24.185 | 0.999 |
| db07-b256l512 | 15.866 | 15.580 | 0.982 |
| db08-b4l1024 | 24.095 | 24.204 | 1.005 |
| db08-b128l128 | 3.630 | 3.590 | 0.989 |
| db10-b16l128 | 3.497 | 3.502 | 1.001 |
| db10-b64l256 | 7.406 | 7.333 | 0.990 |
| db12-b4l1024 | 135.538 | 124.972 | **0.922** |
| db12-b256l512 | 100.932 | 99.691 | 0.988 |
| mid14-b16l256 | 93.263 | 94.249 | 1.011 |
| mid14-b64l128 | 49.945 | 50.284 | 1.007 |
| synth-matlstm-a | 13.289 | 13.145 | 0.989 |
| db09-b128l128 (control) | 1.370 | 1.369 | 0.999 |
| big00-b32l128 (control) | 10.165 | 10.178 | 1.001 |

Within noise everywhere; the one mover (db12-b4l1024, -8%) is the
rnn.16-split-lrnn.16.4 row at b4, where the redundant threads were a
quarter of the kernel's traffic.

**Other suites on the after binary:** `execute_test.py` all cases match
the CPU backend, every contract ok (incl. the new "one scalar per lane
takes a kernel"); `ingest_test.py` 0 failed; `pytest tests/` 512 passed,
1 xfailed (485 + 1x baseline, +6 new, +21 the 2026-09-02 batch).  A first
run of each had died with a clean RESOURCE_EXHAUSTED from the memory
governor while a MoE battery held the machine at 99.5 GB -- the no-panic
contract, not the change -- and passed clean when re-run alone.

## Left open

* Coop mode keeps the trailing-dim convention (`CoopR`); a per-batch
  scalar there is a cross-thread reduce and declines as an accumulator
  in a stacked write (probe P8), so no wrong answer is reachable, but no
  kernel either.
* `VecOff` still drops unit dims before right-aligning: a (B, 1, F)
  carry beside a (B, G, F) one maps B onto G's lane coordinate when
  B == G (declines otherwise).  Same family, different shape; not
  reachable through jax's scan batching, which never leaves unit batch
  dims.

