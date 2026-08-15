# P22: the coop width cap, and THE RELEASE MEASUREMENT (2026-08-15)

Two things, in this order: the one open policy question P21 left (`big09`'s
losing kernel), and then the measurement that the parity claim rests on --
both texmo suites, both stacks, a frozen binary, an exclusive machine.

Raw data: `~/.cache/metaljax-bench/logs/p22-release-measure/`; per-config
table and aggregates `notes/data/p22-release-measure-2026-08-15.{csv,json}`.

---

## Part 1: the coop cap, and why the free win is NOT the work cap

P21 left this open: `big09`'s `rnn.1024` cell is one 1024x1024 dot, 1.05M
elements per step, which slips under `METALJAX_MSL_COOP_CAP`'s 2.2M -- so
coop mode takes it and loses to the compiled matmul.
`METALJAX_MSL_COOP_CAP=1000000` fixed the row, and the question was whether
it costs anything elsewhere.

**It costs a great deal.** The census answers the first half without a
stopwatch. Every coop candidate narrates `coop dot work/lane/step` before the
cap decision, so one warm pass over the 106-configuration suite
(`census-suite106.log`, 58 s) enumerates every plan the cap can reach:

| work (elems/step) | plans | cells |
|---:|---:|---|
| 786,432 | 20 | `gru.512` forward |
| **1,048,576** | **6** | `mgru.512`, `lstm.512`, **`rnn.1024`** |
| **1,310,720** | **2** | `lrnn.512.2` |
| **2,097,152** | **18** | `gru.512` backward, `rnn.1024` |
| 2,883,584 .. 11,534,336 | 18 | already declined by the cap |

A cap of 1e6 takes coop away from **22 of 106 configurations** to fix 2 of
them. `top_confs` cannot be touched at all, and that is structural rather
than lucky: coop work is bounded by the weight count of the cell, and the
largest of the 163 configurations has 1,888 weights in the whole model.

### Measured (standalone, one process per arm, arms interleaved)

Suite protocol (64-step chunks), machine lock held, `part1-summary.txt`:

| config | cell | default | `COOP_CAP=1e6` | ratio | |
|---|---|---:|---:|---:|---|
| `big09-b8l256` | `rnn.1024` | 38.387 | **25.516** | 0.665 | the target, **1.50x win** |
| `big09-b32l128` | `rnn.1024` | 21.674 | **20.982** | 0.968 | 1.03x win |
| `big02-b8l256` | `gru.512` | **32.282** | 34.952 | 1.083 | LOSS |
| `big02-b32l128` | `gru.512` | **17.092** | 23.799 | 1.392 | LOSS |
| `big08-b8l256` | `lstm.512` | **43.115** | 49.756 | 1.154 | LOSS |
| `big00-b8l256` | `mgru.512` | **17.427** | 30.068 | 1.725 | **worst LOSS** |
| `big11-b8l256` | `gru.512` x2 | **65.181** | 75.359 | 1.156 | LOSS |
| `mid14-b16l256` | `lrnn.512.2` | 93.065 | 94.283 | 1.013 | neutral |
| `db11-b64l256` | coop flagship | 0.635 | 0.635 | 0.999 | control, inert |

So the premise of the free win is false, and CLAUDE.md item 12e is worth
correcting on one point: its "lstm.512 ties" was measured against the
*vector* mode, not against no kernel at all -- taking the kernel away costs
1.15x. `mgru.512` was never in that crossover study and loses 1.73x.

### What actually costs nothing: gate on the WIDTH

The loss mechanism is not total work, it is re-streaming: every threadgroup
reads each dot's weights from device memory every timestep, and the traffic
per lane scales with the FEATURE width F. At F=1024 a square cell loses even
at 1.05M elements; at F=512 the same 1.05M is a win. So the cap that costs
nothing is a cap on F, and Stage 1's work cap stays exactly where it is:

    // metal_msl.cc, after the work cap
    if (!dots.empty() && Flags().coop_max_f > 0 && F >= Flags().coop_max_f)
      MslDecline("coop: state width F=... >= ... (matmul path wins)");

`METALJAX_MSL_COOP_MAX_F` (default **1024**, `0` restores Stage 1's policy
exactly). Gated on there being a dot: with no weights to re-stream, a wide
elementwise cell has none of this cost.

**The collateral is provably two rows.** Re-running the census on the new
binary and diffing it plan for plan against the old one: **2 of 106
configurations change** (`big09-b32l128`, `big09-b8l256`, 4 plans), every
other decline reason is identical, and no other configuration's census moves
at all. That is by construction -- every other F>=1024 cell in reach
(`gru.1024` at 3.1M, `lstm.1024` at 4.2M and 11.5M) is already over the work
cap, so the new gate has nothing else to catch.

Re-measured, same nine configurations, `A` = the new default, `B` =
`METALJAX_MSL_COOP_MAX_F=0`:

| config | F-gated | Stage 1 policy | ratio |
|---|---:|---:|---:|
| `big09-b8l256` | **25.322** | 38.396 | **1.516x win** |
| `big09-b32l128` | **20.887** | 21.575 | 1.033x win |
| `big02-b8l256` | 32.181 | 32.187 | 1.000 |
| `big02-b32l128` | 17.351 | 18.066 | 1.041 ᶰ |
| `big08-b8l256` | 43.032 | 43.075 | 1.001 |
| `big00-b8l256` | 17.378 | 17.385 | 1.000 |
| `big11-b8l256` | 65.072 | 65.066 | 1.000 |
| `mid14-b16l256` | 92.449 | 91.794 | 0.993 |
| `db11-b64l256` | 0.633 | 0.633 | 1.000 |

ᶰ `big02-b32l128` runs the *identical* code under both arms (the census
proves no plan changed), so its 4 % is this row's ambient spread, not an
effect; it is the only row in the set whose two repetitions differ by more
than 1 %.

**Landed as the native default**, with the divergence recorded here, in the
ledger and in the plugin's own comment. It is the first place where the
phase-2 lowering deliberately decides something Stage 1 decides differently,
and the census is the receipt: 4 plans out of 210.

---

## Part 2: the release measurement

One machine-lock hold, **22:41:34 to 23:33:37**, strictly sequential, nothing
else on the machine. The native arms run a **frozen copy** of the dylib
(`~/.cache/metaljax-bench/frozen-release-208ca0d1.dylib`, sha256
`208ca0d1...558d61`) rather than the tree's build, which is what P21's halt
was about: a binary cannot move under a campaign it is not in.

| run | route | wall |
|---|---|---:|
| `execute_test` on the exact binary | — | 32 s |
| `suite106-native` | PJRT, 64-step chunks, frozen dylib | 488 s |
| `suite106-stage1` | PJRT, 64-step chunks | 489 s |
| `topconfs-stage1-engine` | `texmo_topconfs.py`, 256-step chunks (the anchor's own route) | 1104 s |
| `topconfs-native-pjrt` | PJRT, 64-step chunks, frozen dylib | 507 s |
| `topconfs-stage1-pjrt` | PJRT, 64-step chunks (this campaign's own route control) | 503 s |

Every run completed: **106 ok / 0 error** twice, **163 ok / 0 error** twice,
and the engine-route run **163 ok / 0 FAIL / 0 error** including its
correctness checks.

The analysis (`analyse.py`) reproduces P16's published aggregates from P16's
own artifacts before it is applied to the new ones -- 4.2377 vs 4.24, the four
suite classes to three digits, top_confs 36.459 vs 36.46 -- so the two
campaigns are computed identically.

### top_confs (163) -- the headline

`native/Stage 1` > 1 means native is slower; `anchor/today` > 1 means today is
faster.

| aggregate | n | value |
|---|---:|---:|
| **native (PJRT) / Stage 1 (engine route)** | 163 | **0.998** geomean, 0.998 median |
| native (PJRT) / Stage 1 (PJRT, same route) | 163 | **1.001** geomean, 1.000 median |
| — by weight class (0-100 / 100-500 / 500-1500 / 1500+) | 47/61/29/26 | 0.999 / 0.998 / 1.008 / 1.003 |
| route factor measured today (Stage 1 engine / Stage 1 PJRT) | 163 | **1.002** (P16 measured 1.009) |
| Stage 1 engine vs the 0.11.3 anchor | 163 | **1.071 faster** |
| native vs the 0.11.3 anchor | 163 | **1.073 faster** |
| jax-CPU control, anchor / today | 163 | 0.990 (machine 1 % slower) |
| **configurations beating jax-CPU** | 163 | **native 59**, Stage 1 55 (engine) / 59 (PJRT), anchor 53 |

Distribution: **163 of 163 within 1.2x**, none above, and native is faster on
**101 of 163**. Worst row either way: `tc029-w47` 0.892 (native ahead),
`tc000-w5` 1.079 (native behind, 0.159 -> 0.172 ms/step -- the smallest
configuration in the suite).

Three things to read out of it. **The msl_scan port closed the whole P16 gap**:
36.46x -> 0.998x, and the "native wins 0 of 163 against jax-CPU" line becomes
59 of 163, which is *more* than Stage 1 wins on the anchor's route.
**The cross-route pairing is honest**: the route factor was 1.002 today, so
the engine-vs-PJRT column and the same-route column agree to 3 parts in 1000.
And **both stacks are 7 % faster than the 0.11.3 anchor** with the CPU control
at 0.990, i.e. the drift is real and shared, not a measurement of one stack.

### suite-106

| aggregate | n | native/Stage 1 | P16 | P21 (WIP) |
|---|---:|---:|---:|---:|
| whole suite, geomean | 106 | **1.011** | 4.24 | 1.027 |
| whole suite, median | 106 | **1.000** | 3.37 | 1.000 |
| `big` | 34 | 1.004 | 1.34 | 1.010 |
| `mid` | 30 | 0.998 | 3.22 | 0.987 |
| `db` (msl territory) | 40 | 1.030 | 14.61 | 1.075 |
| `synth` | 2 | 0.950 | 1.52 | 0.983 |
| rows within 1.2x | 106 | **103** | 33 | 98 |
| rows at or above 10x | 106 | **0** | 32 | 0 |
| rows where native is faster | 106 | **52** | 18 | 54 |

Stage 1's own column is stable: **1.007** against P16 and **1.010** against
P21, geomean, on the same route.

### Every anomaly re-run standalone, and five of nine were the suite itself

The suite-context trap (CLAUDE.md item 12) is not a footnote here -- it is the
majority of the outliers. Each row outside +/-10 % was re-measured standalone,
one process per arm, both stacks interleaved:

| config | in-suite S1 | in-suite native | in-suite | standalone S1 | standalone native | standalone | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `db16-b256l512` | 4.472 | 7.946 | 1.777 | 4.472 | 7.935 | **1.774** | REAL |
| `db17-b256l512` | 7.282 | 11.646 | 1.599 | 7.281 | 11.618 | **1.596** | REAL |
| `db11-b256l512` | 2.176 | 2.857 | 1.313 | 2.180 | 2.859 | **1.311** | REAL |
| `big14-b32l128` | 17.881 | 21.419 | 1.198 | 17.463 | 17.456 | **1.000** | artifact |
| `big12-b8l256` | 5.190 | 6.014 | 1.159 | 5.191 | 5.180 | **0.998** | artifact |
| `big07-b8l256` | 33.353 | 37.759 | 1.132 | 33.021 | 33.139 | **1.004** | artifact |
| `big00-b32l128` | 12.444 | 10.041 | 0.807 | 10.248 | 10.189 | **0.994** | artifact |
| `mid11-b64l128` | 9.911 | 8.764 | 0.884 | 8.527 | 8.561 | **1.004** | artifact |
| `big09-b8l256` | 38.368 | 25.560 | 0.666 | 38.346 | 25.378 | **0.662** | REAL (part 1) |

The five artifacts are Stage-1-side in-suite variance, and they show up as
such independently: those are exactly the rows whose *Stage 1* column moved
against P16 (`big00-b32l128` +20 %, `mid11-b64l128` +15 %, `big14-b32l128`
-18 %) while their standalone numbers sit on their P16 values. They cut BOTH
ways -- two of them flattered native.

With the nine standalone numbers substituted the suite geomean is **1.010**
(median 0.9999, native faster on 53 of 106), so the aggregate barely moves;
what changes is which rows one is allowed to name.

### The one qualifier on the parity claim

**Three `db*-b256l512` rows are genuinely slower natively** -- `db16` 1.77x,
`db17` 1.60x, `db11` 1.31x -- reproducibly, standalone, on a frozen binary.
P21's preliminary column saw the same two rows (1.84x, 1.66x), so they are a
property of the port rather than of a day.

What is already ruled out, each measured:

* **Not the kernels' existence, and not the surrounding graph.** With
  `METALJAX_MSL=0` the two stacks are equal on these rows (`db16` native
  83.99 vs Stage 1 83.82, `db11` 60.21 vs 59.02). The whole gap lives on the
  msl path.
* **Not the plans.** The narration is identical, plan for plan: `db16` two
  coop plans `trip=512 lanes=8192 states=1 stacked=1/8 packed=0` on both
  stacks; `db11` two coop plans at `lanes=4096`, `stacked=1/7`. Same mode,
  same geometry, same trip -- and the kernel source is generated under the
  same name, so both engines share MLX's compiled library.
* **Not the flush cadence** (`METALJAX_EAGER_FLUSH_MB=1e6`: native 7.94 ->
  7.18, Stage 1 unmoved at 4.48) and **not the compile budget**
  (`METALJAX_TRACE_BUDGET=1e5`: 7.95 / 4.46, neither moves).

So it is the LAUNCH, not the plan: identical kernels, dispatched differently.
The suspects the evidence points at are `runtime/msl.cc`'s per-call work --
the weight-normalization recipe (`as_strided` -> transpose -> contiguous) and
the input pooling, both of which Stage 1's `Plan.run` performs in Python but
may amortize differently -- and they are worth an hour with a profiler. The
pocket is narrow and identifiable: it is the largest `db` configurations
(b256 x l512) only; `db11-b64l256`, the same spec at a smaller shape, is at
**exact** parity (0.633 vs 0.633).

**It does not move the headline.** Three rows of 106, none of them in
`top_confs`, with the suite geomean at 1.011 and the median at 1.000.

### Battery

| | |
|---|---|
| `plugin-native/execute_test.py` | **535 of 535** — P21's 534 plus the new width-cap contract; the delta was verified as exactly one check row by re-running the pre-change file on the same binary (536 -> 537 rows by the log's own line count, whose convention differs from P21's by two arm-summary lines) |
| `plugin-native/texmo_gate.py` | **106 ok / 0 decline / 0 FAIL / 0 error** (27 via sensitivity scaling) |
| census diff, whole suite | 2 configurations changed, 4 plans, every other decline reason identical |

The new contract, `msl coop width cap (F>=1024)`, runs one square F=1024 cell
in two children: under the default it must build **no** coop plan and narrate
the width decline; under `METALJAX_MSL_COOP_MAX_F=0` it must build one; and
the two answers must agree (measured 9.1e-06 on a 12,288-element sum -- the
same different-contraction-order effect the three fissioned weight-gradient
rows show).
