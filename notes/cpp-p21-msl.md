# P21: msl_scan, natively (2026-08-14)

Item 1 of the P16 performance frontier -- the last and largest of them:
`msl_scan`, whose absence cost the native plugin a **36.5x geomean** on
`top_confs` and 14.6x on the texmo suite's `db` class. The plugin now
recognizes the same counted loops Stage 1 recognizes, generates the same three
kinds of Metal kernel, and hands them to the launch recipe the shared runtime
has been able to execute since M5b.

`src/metaljax/msl_scan.py` is the specification, class for class and check for
check; `src/metaljax/tape.py::_lower_msl` is the spec for the launch recipe;
`plugin-native/runtime/msl.cc` (M5b, unchanged bar one fix below) is the
executor.

## The shape of it

    plugin-native/metal/metal_msl.h        the Sym IR, the plan's products
    plugin-native/metal/metal_msl.cc       the analyzer, the canonicalizer,
                                           classification, mode choice, the
                                           weight-layout and packing passes
    plugin-native/metal/metal_msl_emit.cc  the three emitters (scalar/affine,
                                           vector, coop) -- MSL text
    plugin-native/metal/metal_lowering.cc  `MslPlanFor` (the cache the cost
                                           walk, the traceability question and
                                           `LowerWhile` all ask) and
                                           `LowerMslPlan` (the launch recipe)

**All three modes landed together**, because they are one recognizer: the
analysis, the classification and the fission machinery are shared, and the
modes differ only in the lane geometry each emitter assumes. Splitting the
milestone would have meant porting the analyzer three times.

**Pointer identity replaces `id(sym)`.** Every memo in the emitters, the
structural CSE of the canonicalizer, and the `is`-comparisons of the
accumulator resolver key on node identity. The nodes live in a `SymArena`
(`std::deque<Sym>`) owned by the plan, so a `Sym*` is stable for the life of
the plan -- which is *more* than CPython gives (CLAUDE.md item 16's trap is
exactly that `id()` is recyclable). Nothing here can hit that class of bug.

**Declines are exceptions, as they are in the Python.** `MslUnsupported` is
thrown from anywhere in analysis or emission and caught in exactly one place,
`BuildMslPlan` -- the transliteration of `build_plan` + `_msl_plan_for`'s
`except Exception`, including the retry that turns off the coop-over-vector
flip and the in-lane dot rewrite when either fired before the failure. It
derives from `std::exception` so an escape (there is none) would be an
`InternalError` rather than a `std::terminate`.

**The lowering asks once.** `LowerContext::msl` caches by (body block, trip,
start) exactly as `interp._msl_cache` does, because `BlockCost` (charging a
planned loop 8 units), `WhileTraceable` (a planned loop is traceable outright)
and `LowerWhile` must get the same answer or the compile decisions and the tape
would describe different programs.

**A planned loop still lowers its regions.** They are the FALLBACK: `kMslScan`
is a `kWhile` in every other respect, so a kernel Metal rejects falls back to
the interpreted loop in the same call. Only when the body is outside the op set
do the regions go missing, and then the taints come from the plan's own
pass-through list (`_lower_msl`'s rule).

## The census: identical to Stage 1's, plan for plan

The whole 106-configuration texmo suite through both engines with
`METALJAX_DEBUG=1` (native: `plugin-native/texmo_gate.py`; Stage 1:
`scripts/texmo_check.py`), counting the plans each engine built and the reasons
it declined:

| | native (P21) | Stage 1 |
|---|---:|---:|
| coop plans | 146 | 146 |
| vector plans | 52 | 52 |
| scalar (affine) plans | 12 | 12 |
| decline `op stablehlo.gather` | 104 | 104 |
| decline `coop: dot work N > 2200000` | 18 | 18 |
| decline `coop: width not multiple of F` | 4 | 4 |
| decline `coop emit type SymRedReg` | 4 | 4 |
| decline `concat on non-feature dim` | 2 | 2 |
| decline `acc-dot outside accumulator update` | 2 | 2 |
| decline `acc in stacked write` | 2 | 2 |

Every count matches, and every decline matches by REASON -- including the three
that name a Sym tree, which are the same trees. 192 kernels launch across the
suite. That equality is the strongest evidence available that the port is
faithful: the two recognizers agree about every loop in 106 real training
chunks, and they disagree about none.

## Measured (standalone, the same binary, `METALJAX_MSL` flipped)

Best of 20 executes of one 8-step training chunk, ms; `compile` is
`compile_and_load`, `first` the first execute, `steady` the best of the rest.
Machine lock held; the two arms interleaved, since the FIRST process of a
sequence pays MLX's cold kernel-library build (a 200 ms artifact that looked
like a regression until the arms were reversed).

| config | plans | msl=0 | msl=1 | |
|---|---|---:|---:|---|
| `db02-b4l1024` | 2 vector | 588.0 | **8.0** | **73x** |
| `db11-b64l256` | 2 coop | 230.7 | **7.0** | **33x** |
| `big09-b8l256` `rnn.1024` | 2 coop | 202.0 | 308.1 | **0.66x** |
| `db00-b16l128` | none | 1.73 | 1.76 | 1.00x |
| `db04-b128l128` | none | 4.03 | 3.97 | 1.00x |
| `big14-b32l128` | none | 143.2 | 143.5 | 1.00x |
| `big16-b32l128` | none | 480.8 | 483.5 | 1.00x |

Two things to read out of it.

**A program with no plan pays nothing.** Compile time does not move (31.4 vs
31.9 ms on `db00`), the first execute does not move, and the steady state does
not move. That is what the code says too -- with no plan the `while` entry is
byte-identical to the one P20 emitted -- and it is worth measuring anyway,
because the whole integration hangs off a question now asked of every counted
loop in every program.

**`big09` is the one row where the kernel LOSES, and it is Stage 1's policy
doing it.** `rnn.1024`'s single 1024x1024 dot is 1.05M elements per step, under
the `METALJAX_MSL_COOP_CAP` of 2.2M, so coop mode takes it -- and
per-threadgroup weight re-streaming loses 1.53x to the compiled matmul at that
width. The cap was tuned on `gru/lstm.1024` (3.1M and 11.5M, both rejected);
a square `rnn.1024` slips under it. `METALJAX_MSL_COOP_CAP=1000000` returns the
row to **201.6 ms** with nothing else changed. NOT applied: it would put the
native census out of step with Stage 1's, which is a policy question for Oleg
rather than a port decision. CLAUDE.md item 12e already records "F=1024 loses".

## The one runtime fix (`runtime/msl.cc`)

`settle_msl`'s recovery was written when a second failure could hand the
program back to the Python engine. In phase 2 there is nothing underneath, and
the second failure is REACHABLE: a loop whose kernel is retired runs its body
instead, and the body may hold a loop of its own -- so a nested scan produces a
second unproven plan on the very run that was recovering from the first. The
ladder is now bounded and terminates on its own: first failure retires the
pending plans and reruns; a second retires EVERY plan the program holds
(`Program::disable_msl_deep`) and reruns once more, after which no kernel can
fail because none is left. A program holding a host call still keeps the
failure, since a rerun repeats what the first attempt did to the world.

Found by the new `METALJAX_MSL_FORCE_BUILD_FAIL` arm of `execute_test`, on
`msl nested unrolled loop`. It is the only change outside the plugin's own
directory.

## Battery

| | |
|---|---|
| `plugin-native/execute_test.py` | **534 of 534** checks match jax-CPU (524 before, + 8 msl cases + 2 msl contracts) |
| `plugin-native/texmo_gate.py` | **106 ok / 0 decline / 0 FAIL / 0 error** (21 via sensitivity scaling) |
| `plugin-native/smoke_test.py` | all checkpoints passed |
| `plugin-native/decline_census.py` | 35 of 35 programs lower |
| `bazel test //...` | 1 test passes |

Final binary: `plugin-native/bazel-bin/metal/libmetal_pjrt_native.dylib`,
47,386,808 bytes, sha256 `f8ad74cc...c18c42e` -- every number above was
measured on it (the earlier 105/106 gate ran on a binary one fix older).

New in `execute_test`, and what each one is for:

* Eight cases, one per emitter plus the AD-generated backward passes: `msl
  affine cell (mingru)`, `msl affine cell, backward`, `msl vector matvec cell`,
  `msl vector cell, weight grad`, `msl coop matvec cell`, `msl coop cell,
  weight grad`, `msl gru cell (coop flip)`, `msl nested unrolled loop`. The
  weight-grad rows are the ones that exercise loop fission -- hidden per-step
  stacks out of the kernel and one batched matmul after it.
* **`msl kernels vs the interpreted loop`**: the same cases re-run in a child
  with `METALJAX_MSL=0` and compared with the parent's. **466 of 469
  bit-identical**, and the three that are not are exactly the three weight-grad
  rows (4.8e-07 .. 1.4e-06): a fissioned accumulation sums in a different order
  than the loop does. Every forward kernel is bit-for-bit the interpreted loop.
* **`a rejected kernel falls back to the loop`**: every case again, in a child
  with `METALJAX_MSL_FORCE_BUILD_FAIL=1`, so Metal rejects every generated
  source. Same 466 of 469. This is the recovery ladder end to end.
* **`msl covers its three modes`** and **`METALJAX_MSL=0 builds no kernel`**:
  the mode census as a test, read out of the plugin's own narration in a child
  run with `METALJAX_DEBUG`. A port that quietly stopped picking one mode would
  show up nowhere else -- every answer would still be right, just slower.

## Gate flake, attributed

The first full gate run of the milestone was 105/106: `big10-b8l256`
(`bits.4.oh+bp|gru.1024`, one of the ill-conditioned rows the sensitivity
scaling exists for) came back `inf`. It passes standalone 3/3, and it builds
**no msl plan at all** -- its three loops decline on the work cap (3.1M and
8.4M) and on `stablehlo.gather` -- so its tape is the one P20 shipped. It is
the in-suite lottery P4 recorded for this row's class ("a second run was
105/106"). Two full gate runs on the final binary are 106/106.

## Carried over verbatim, because they are correctness

* **The volatile loop counter** (CLAUDE.md item 12c). Apple's shader compiler
  miscompiles multi-iteration kernel time loops; only a volatile access at
  every use of `t` produces correct code. `METALJAX_MSL_VOLATILE` keeps the
  four insufficient variants for retesting a future macOS.
* **The first launch of an unproven plan evaluates synchronously** -- a Metal
  build error raised on an async worker aborts the process.
* **The 30-binding cap** and the per-dtype input pooling that keeps plans under
  it, with the pooled slot sized by the weight-norm WINDOW rather than the
  source buffer (0.4.3's silent-shift bug).
* **The three silent-wrongness fixes of the 2026-07 campaign**: the lane-scalar
  broadcast guard (a stacked output whose per-step shape IS the lane shape),
  the concat pad width (the part's own width, not the concat total), and the
  sequential carry assignment (`_aliased_state_moves`, in all three emitters).
* **A plan that cannot be expressed is a decline, never an error** -- and a
  plan the launch recipe cannot express declines the whole PROGRAM, because
  both engines' cost walks have already treated the loop as one kernel.

## Deliberately not ported

`METALJAX_MSL_VOLATILE=tmap` (the table-driven counter binds an extra input the
plan would build per call) declines, exactly as `tape.py` declines it; it is a
retest knob and never the shipped path.

## Open, for scrutiny

* ~~`big09`-class rows: the coop cap admits a square F=1024 cell that loses to
  the compiled matmul (above). One env knob away, one row measured, Oleg's
  call.~~ **RESOLVED 2026-08-15 (P22, `notes/cpp-p22-release.md`)** — and not
  by the knob that was proposed. Lowering `METALJAX_MSL_COOP_CAP` to 1e6 takes
  coop away from **22 of 106** configurations to fix 2, and costs up to
  **1.73x** (`mgru.512`) on the rows it hits; the loss mechanism is the
  per-lane re-streaming of weights, which scales with the FEATURE width, not
  with total work. A width cap (`METALJAX_MSL_COOP_MAX_F`, default 1024)
  changes exactly the two `big09` rows — census-verified, 4 plans of 210 —
  and is worth 1.52x on `big09-b8l256`. It is the phase-2 lowering's first
  deliberate divergence from `msl_scan.py`; `=0` restores Stage 1's policy.
* The suite-context trap (CLAUDE.md item 12) now has ~200 more Metal kernels to
  work with: sub-millisecond rows measured late in a 106-configuration sweep
  should be re-measured standalone before a regression is believed. `db00` and
  `db04` are the two that looked worst in-sweep and are exactly at parity
  standalone.
* A planned loop whose regions do NOT lower has no interpreted fallback, so a
  Metal build failure there fails the program (there is no Python engine to
  hand it to). No program in reach is in that state -- the native op set covers
  every texmo body -- and declining such a loop instead would be worse: the
  program would not run at all.
