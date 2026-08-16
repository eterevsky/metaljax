# P23: the last pre-RC gap — a planned loop the byte gate could not see (2026-08-16)

P22 shipped the release measurement with one qualifier: three `db*-b256l512`
rows ran **1.77x / 1.60x / 1.31x** slower natively, reproducibly, standalone,
on a frozen binary, with *identical* plans and *identical* generated kernels.
Its verdict was "identical kernels, dispatched differently", and it pointed at
`runtime/msl.cc`'s per-call launch work.

**It is not the launch.** Nothing in `runtime/msl.cc` changed. The gap was one
missing case in the byte estimate every COMPILE decision reads, which took the
compiled body away from the whole training step and left each step to be
dispatched op by op — around kernels that were, indeed, identical.

Raw data `~/.cache/metaljax-bench/logs/p23-dispatch/`, per-config table and
aggregates `notes/data/p23-dispatch-2026-08-16.{csv,json}`, RC table
`benchmarks/perf-2026-08-native-baseline.md`.

## The bug

`ops/control._block_bytes` (Stage 1) walks a block charging what it would
materialize if TRACED — and a `stablehlo.while` that became one generated msl
kernel is charged **its outputs only**:

```python
        if o.name == "stablehlo.while":
            if _msl_plan_for(interp, o) is not None:
                continue                      # <- charged op_bytes(o), no more
            ...
            total += trip * _block_bytes(interp, body)
```

The function's own docstring says why: *"a loop that became one generated msl
kernel charged only its outputs (its per-timestep state lives in registers, not
in buffers)"*. `BlockCost` has the same case one function earlier (`cost += 8;
continue;` — "a single generated kernel"), and P21 ported THAT one.
`BlockBytes` was ported without it, so a planned loop was charged `trip × body`:
the traffic of a loop that does not run.

What that costs, on `db16-b256l512` (b256, l512, two coop plans at trip=512,
64 training steps per chunk):

| | Stage 1 | native P22 | native P23 |
|---|---:|---:|---:|
| `main: … bytes=` for the chunk | 134,234.4 MB | **10,723,677.4 MB** | **134,234.4 MB** |
| ≈ per training step | 2.05 GB | 163 GB | 2.05 GB |
| vs `METALJAX_COMPILE_BYTES_MB` (64 GB) | under | **over** | under |
| eager flushes per chunk | — | **128** | **0** |
| compiled bodies / compiled calls | — | **0 / 0** | **1 / 4** (K=16) |

Three decisions flip together at that gate, all asking the same estimate
(`metal_lowering.cc`, `LowerWhile`): whether the loop BODY may be compiled
(`by_bytes = kCompileBytes / BlockBytes(body)`, which went to 0), how many
iterations one compiled chunk may hold (`BytesChunks`, which fell to 1), and on
smaller programs the whole-main compile. The step then runs on the single-step
eager path — where the byte-denominated safety net in `Program::interpret`
fires on the same inflated numbers, adding two blocking `mx::eval`s per step.

The fix is that one case, in `BlockBytes`:

```cpp
      if (MslPlanFor(ctx, &o) != nullptr) continue;
```

`MslPlanFor` is the cache `BlockCost`, `WhileTraceable` and `LowerWhile`
already ask, so the plan is built once and the walk costs nothing new. After
it the native estimate is Stage 1's number **exactly** — 134,234.4 MB against
134,234.4 MB, not approximately — which is the strongest available statement
that the two walks now agree.

## How it was found: no profiler was needed, and that is the lesson

P22's own probe logs already held it. The two stacks' narration for the same
chunk, side by side:

    native  main: pure=0 cost=39113 bytes=10723677.4MB compile=0
            jit_chunk: flushes=128 … compiles=0 compiled_calls=0
    Stage 1 exec jit_chunk: pure=False cost=39113 bytes=134234.4MB
            eagerbytes=8.1MB … compile=False

`cost` agrees to the unit (39113 — the walk that HAS the msl case); `bytes` is
80x apart (the walk that does not); and the native line says in the same breath
that it compiled nothing and flushed 128 times. Instrumenting the launch path
would have profiled a launch path that was never the problem. When two engines
are meant to be transliterations, **diff what they SAY about a program before
profiling what they do** — both stacks already narrate cost, bytes, purity, the
compile decision and the flush counters.

### The cost, split (`db16-b256l512`, ms/step, standalone, lock held)

The first three arms are the same program on P22's released binary; only the
environment changes.

| arm | ms/step | |
|---|---:|---|
| P22 released, as shipped | 7.923 | |
| … `METALJAX_EAGER_FLUSH_MB=0` | 7.185 | the 128 blocking flushes: **0.74 ms** |
| … `METALJAX_COMPILE_BYTES_MB=1048576` | **4.469** | the byte gate raised past the inflated estimate — the fix, by knob |
| P23 binary (the fix, by code) | **4.464** | |
| Stage 1 | 4.471 | |

So of the 3.45 ms/step gap, **0.74 ms was the flush cadence** and the remaining
**2.71 ms was op-by-op dispatch** of the step's ~611 ops (cost=39113 over 64
steps) instead of one compiled replay per 16 steps — ≈4.4 µs per op, which is
what an eager MLX graph build costs. The knob arm is the control that matters:
the *unmodified* P22 binary, with nothing but the gate moved, lands on the
fixed binary's number.

## Measured (standalone, one process per arm, arms interleaved, lock held)

Best chunk of a 64-step training chunk, ms/step. `old` is P22's frozen release
dylib, `new` P23's, `S1` the Python engine through the same PJRT route.

| config | S1 | old | new | old/S1 | **new/S1** |
|---|---:|---:|---:|---:|---:|
| `db16-b256l512` | 4.475 | 7.942 | **4.466** | 1.774 | **0.998** |
| `db17-b256l512` | 7.287 | 11.614 | **7.280** | 1.594 | **0.999** |
| `db11-b256l512` | 2.180 | 2.850 | **2.147** | 1.307 | **0.985** |
| `db11-b64l256` (the parity control) | 0.634 | 0.634 | 0.634 | 1.000 | **1.001** |
| `db02-b4l1024` (vector flagship) | 0.782 | 0.798 | 0.793 | 1.021 | 1.015 |

The three rows P22 qualified its parity claim with are gone. `db11-b64l256`,
already at exact parity, has not moved — which is also why the pocket was
`b256 × l512`-shaped: the bug needs a program big enough for a 512x overcharge
to cross 64 GB. `db02-b4l1024` was never in the pocket (1.02 on both binaries).

## suite-106 and top_confs (the RC pair)

Machine lock held throughout, one process per arm, `scripts/bench_texmo_pjrt.py`
(PJRT route, 64-step chunks) for every arm, frozen dylib
`~/.cache/metaljax-bench/frozen-rc-ed355691.dylib` (sha256 `ed355691…94a16`)
for the native arms. The analysis reproduces P22's published aggregates from
P22's own artifacts before being applied to the new ones (1.0111 whole, 1.0043
/ 0.9979 / 1.0301 / 0.950 by class, top_confs 1.0007) and refuses to report if
they do not come back.

| suite-106 aggregate | n | **P23** | P22 |
|---|---:|---:|---:|
| whole suite, geomean | 106 | **1.0050** | 1.011 |
| whole suite, median | 106 | **1.0012** | 1.000 |
| `big` | 34 | 1.0107 | 1.004 |
| `mid` | 30 | 1.0033 | 0.998 |
| **`db` (msl territory)** | 40 | **1.0013** | 1.030 |
| `synth` | 2 | 1.0062 | 0.950 |
| rows within 1.2x | 106 | **106** | 103 |
| rows at or above 10x | 106 | 0 | 0 |
| rows where native is faster | 106 | 42 | 52 |

| top_confs (163) | n | **P23** | P22 |
|---|---:|---:|---:|
| native / Stage 1, same PJRT route | 163 | **1.0016** | 1.001 |
| rows within 1.2x | 163 | **163** | 163 |
| native faster | 163 | 63 | — |
| beating jax-CPU | 163 | native **58**, Stage 1 59 | native 59 |
| native arm vs P22's native arm | 163 | **0.9999** | — |

**`top_confs` does not move, and should not**: no configuration in it is large
enough for the overcharge to cross the gate (native vs P22 native is 0.9999
geomean, one row outside ±10 % and that one in native's favour). The whole
effect is in `db`, whose class geomean goes 1.030 → 1.0013, and the suite is
now **106 of 106 within 1.2x** where P22 had exactly the three msl rows outside.

### Every outlier re-measured standalone — and all seven were the suite

| config | in-suite | standalone S1 | standalone new | standalone | verdict |
|---|---:|---:|---:|---:|---|
| `big11-b32l128` | 1.178 | 35.853 | 35.856 | **1.000** | in-suite artifact |
| `big14-b32l128` | 1.143 | 17.482 | 17.443 | **0.998** | in-suite artifact |
| `big14-b8l256` | 1.105 | 10.356 | 10.340 | **0.998** | in-suite artifact |
| `big16-b32l128` | 1.105 | 60.504 | 60.519 | **1.000** | in-suite artifact |
| `big12-b32l128` | 0.874 | 9.703 | 9.713 | **1.001** | in-suite artifact |
| `big09-b32l128` | 0.873 | 21.534 | 20.736 | **0.963** | REAL (P22's width cap) |
| `big09-b8l256` | 0.660 | 38.404ᴾ²² | 25.358ᴾ²² | **0.662** | REAL (P22's width cap) |

ᴾ²² `big09-b8l256`'s standalone pair is P22's (part 1's width-cap study
measured it twice on the same code); every other row here was measured today.
Substituting the six new standalone numbers moves the suite geomean 1.0050 →
**1.0024** (median 1.0009, native faster on 43 of 106) — the aggregate barely
moves; what changes is which rows one is allowed to name.

### A measurement lesson worth more than the rows: do not run the gate first

The suite pair was measured **twice**. The first pair ran immediately after a
263 s `texmo_gate` in the same lock hold (106 model-building subprocesses, tens
of GB of artifacts written and deleted), and it came back at geomean 1.047 with
**21 rows outside ±10 %** — every one of them an `l128` (large-batch) row, and
**7 of the 11 worst build no msl plan at all**, i.e. their tapes are
byte-identical to P22's. The Stage 1 arm of the same hold was inflated too
(its own drift against P22 was 0.9725 with a worst row at 0.72), just less,
because it ran second.

Re-run in a hold of its own with a 120 s settle and nothing before it, every
one of those rows is back on its P22 value (`big12-b32l128` 16.31 → **9.93**
against P22's 9.74; `big03-b32l128` 27.99 → **18.20** against 18.02;
`big16-b32l128` 88.28 → **67.09** against 66.76). Both pairs are kept
(`suite106-*.jsonl` and `suite106-*-r2.jsonl`); the clean pair is the RC table.
CLAUDE.md item 12's suite-context trap has a new and sharper form: **a heavy
gate run poisons the next few minutes of measurement in the same hold**, and it
poisons the arm that runs first.

## Battery

| | |
|---|---|
| `plugin-native/execute_test.py` | **536 of 536** on the frozen binary — P22's 535 plus the new contract; the log diff against P22's is exactly one row plus the plugin path |
| `plugin-native/texmo_gate.py` | **106 ok / 0 decline / 0 FAIL / 0 error**, twice on the fixed code (once on the frozen binary), and 106/106 on P22's binary in the same session |
| `plugin-native/smoke_test.py` | all checkpoints passed |
| `bazel test //…` | 1 test passes |
| census diff, whole suite | **EMPTY** |

**The census, in full.** The 106-configuration suite through the new binary
with `METALJAX_DEBUG=1`, diffed against P22's census on the released binary:
**568 narration lines, identical in content AND in order** — 142 coop / 52
vector / 12 scalar plans and every decline reason, count for count. This had to
be empty: the change is to a compile decision, and a plan decision that moved
with it would mean the byte walk was feeding the recognizer.

**The new contract, `msl loop charged as one kernel`.** No correctness test can
see a bug like this: every answer stays right, the program is merely dispatched
badly. So the contract pins the MECHANISM — the same gru cell is jitted twice,
once planned and once under `METALJAX_MSL=0`, and the byte estimate the compile
decisions read must be much smaller when the loop became a kernel. Measured
**3.1 MB planned vs 290.1 MB interpreted** (the bar is 3x); with the bug the two
arms report the SAME number, which is exactly the failure.

**One gate flake, attributed.** The first full gate on the fixed binary was
105/106: `mid03-b16l256` (`lstm.128`) came back `worst=4.3e-02` against
`tol=5.9e-03`. It passes standalone **3/3 on the fixed binary and 3/3 on P22's
released one** (worst 1.1e-04 … 9.4e-04 either way), and the two full gates that
followed — one per binary, interleaved — are **106/106 both**. The row's measured
sensitivity moves by an order of magnitude run to run (8.6e-05 in P22's gate,
1.2e-05 in the failing one), which is what the sensitivity scaling exists for
and what makes the row a lottery ticket in-suite; P21 recorded the same shape
for `big10-b8l256`.

## Open, for scrutiny

* **The byte gate now trusts a kernel that may still fall back.** A planned
  loop whose Metal build fails runs its interpreted body instead — under a
  compile decision made on the assumption it would not, so that program's
  compiled trace is sized for a loop that is now materializing per-step
  buffers. Stage 1 has had exactly this exposure since 0.2.0 (same code, same
  order), no configuration in reach exercises it, and the resource-limit ladder
  recovers; but it is the one place where this fix makes the estimate
  optimistic rather than conservative.
* **What else the two byte walks might disagree about.** `cost` and `bytes` now
  match Stage 1 on `db16` digit for digit, and the suite agrees within
  measurement everywhere — but only one program was checked against the Python
  by its digits. The cheap standing check is the census doctrine applied to the
  compile decisions: narrate `cost`/`bytes`/`compile` for every program in the
  gate and diff the two engines' lines across the suite. That would have caught
  this in P21 without a stopwatch.
* **`native faster on 42 of 106` (P22: 52).** Not a regression: P22's 52
  counted a run whose Stage 1 arm was itself noisy on ~5 rows (its own anomaly
  table says so), and the median is 1.0012 either way. The honest statement is
  that the two stacks are within a percent on this suite and the row-level sign
  is noise for anything under ~2 %.
* **The suite is now sensitive to what ran before it.** Documented above; the
  protocol change (a hold of its own, a settle, gate runs after rather than
  before) is worth adopting for every future release measurement.
