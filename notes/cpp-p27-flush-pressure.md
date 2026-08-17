# P27: the flush watermark is not one number (2026-08-16)

> **SUPERSEDED IN PART by P28 (`notes/cpp-p28-benefit-gate.md`, 2026-08-17).**
> The two rules below are intact and still ship; a THIRD joined them, because
> they turned out to grant a big pool to programs that do nothing with one.
> The 0.11.5 release gate measured the consequence on the two rows this
> battery never ran — the maxtext DECODE rows 11 and 14 — and scored gate 5
> as REGRESSION over it: their checkpoint load is one program taking 134 hard
> flushes in a single call, so it is an "eager main" by rule 1 and has
> footprint to spare by rule 2, and the 14 GB of weights it frees at its last
> flush then stand in the pool for the rest of the process. 17 GB and 11 GB
> of peak footprint for no speed, and a guard kill at budgets those rows had
> never come near. **Rule 3 (`METALJAX_FLUSH_EARN_MULT`, default 2) bounds a
> program's pool by the live set it has demonstrated it cycles**; it enters
> as one more `min`, so every number in this document that rule 3 does not
> lower still stands, and the row-19 and row-18 results below were both
> re-measured under it. Read this document for what rules 1 and 2 are and why;
> read P28 for what a program has to show before it is given a pool at all.

*P25 shipped the eager flush's pool TRIM and measured what the watermark
itself is worth, leaving Oleg a table with no good row in it: the maxtext
training row (STATUS 19) reaches its 0.11.3 anchor only at a 32 GB watermark,
and 32 GB is where the LoRA row (18) blew through its 70 GB guard. This pass
was asked to find a policy satisfying three conditions at once — no row OOMs
or guard kills, no regression anywhere, and row 19 fixed to its ~464-470 ms
class. It does, and the reason it can is that the two rows were never asking
for the same thing: **row 18's blowout is not a pool at all.***

*Code: `plugin-native/runtime/{runtime,program,config}.cc`, `program.h`,
`plugin-native/metal/{metal_client,runtime_gil_free_test}.cc`, four new
`execute_test.py` contracts. `src/` and `native/` are FROZEN and untouched.
Raw data, runners and the frozen binary:
`~/.cache/metaljax-bench/logs/p27-flush-pressure/` (`p27_diag.sh` — the
diagnosis; `p27_battery.sh` — the validation battery, one hold;
`p27_phase3.sh` — the same-binary controls, the buffer-count probe and the
standalone re-measures; `p27_final.sh` — rebuild and re-verify; `analyse.py`
— the suite aggregates, P22/P23's method, which reproduces P23's published
numbers from P23's artifacts before it reports). Aggregates:
`notes/data/p27-flush-pressure-2026-08-16.{json,csv}`.*

## 1. Where row 18's blowout actually lives — measured, not inferred

P25 recorded row 18's peak as "a LOAD transient, sampled at ~10 of ~60 guard
samples" and left it there, because the guard samples twice a second and the
event is shorter than that. So the flush meter grew a `foot=` field — the
process's **physical footprint**, read inside the dylib with
`task_info(TASK_VM_INFO)`, which is the same number `mem_guard.sh` kills on —
and the row was run again at both watermarks.

The last four flushes of the arm the guard killed (`METALJAX_FLUSH_CLEAR_MB=32768`,
no P27 policy):

```
flush #262: active=19597MB cache=16194MB bound=32768MB foot=36297MB
flush #263: active=19595MB cache=16196MB bound=32768MB foot=36297MB
flush #264: active=37515MB cache=16196MB bound=32768MB foot=54217MB
flush #265: active=46475MB cache=16196MB bound=32768MB foot=63177MB
mem_guard: projected 128.00 GB (+31.00 GB/sample) crosses budget 70 GB
```

and the same four in the shipped arm (`2048`), which survives:

```
flush #262: active=19597MB cache= 1536MB foot=21639MB
flush #264: active=37515MB cache= 1538MB foot=39559MB
flush #265: active=46475MB cache= 1538MB foot=48519MB
flush #266: active=19595MB cache=17920MB (was 19458MB) foot=56711MB
```

**The spike is the LIVE set: 19.6 → 37.5 → 46.5 GB in about a second, and it
is identical on both binaries.** It happens in the keras build/convert phase,
before the checkpoint's bulk transfers (the first ingest clear, at 8 GB
ingested, comes ~190 flushes later). The watermark decides one thing only:
how much **dead pool is standing beside that spike when it lands** — 1.5 GB
at 2048, 16.2 GB at 32768 — which is the whole difference between 48.5 GB of
footprint and 63.2 GB.

Two further findings from the same trace, both of which shaped the policy:

* **The pool that accumulates there is being reused, not leaked.** For 250
  flushes before the spike the footprint sits FLAT at 16.8 GB while active
  and cache trade places (1.7↔7.4 GB active, 8↔16 GB cached). Nothing about
  that phase looks wrong until the spike arrives, so no static watermark can
  be said to be "too high" for it in advance.
* **The shipped policy's own peak on this row is 56.7 GB**, in a run whose
  guard flight log records 39 — the watchdog samples twice a second and the
  crest is shorter than that. P25's "37-56 GB across runs" band was sampling
  luck, and the top of it is the truth.

The row 19 side of the same instrumentation: its training step holds an
**18.6 GB live set** and wants a **~26 GB pool** (`active=13754..19310MB
cache=20345..25900MB foot=40126MB`, 0 trims at a 32768 watermark). So the two
rows are not on one axis at all: row 19 wants a big pool ON TOP of a big live
set, and row 18 needs a small pool BESIDE a live set that is about to
double.

## 2. The policy

`runtime.cc::flush_bound(cache_now, program_flushes)` — the watermark is now
decided per flush, by two rules over P25's floor:

```c++
if (program_flushes < flush_main_flushes) return floor;      // (1) the gate
const int64_t live = phys_footprint() - cache_now;           // (2) the room
return min(cap, max(floor, footprint_target - live));
```

1. **The gate** (`METALJAX_FLUSH_MAIN_FLUSHES`, default 8). Only a program
   that has already taken 8 hard flushes in its life — an eager MAIN, whose
   traffic dwarfs its live set and which is therefore REUSING a pool rather
   than filling one — may go past the floor at all. maxtext's training step
   crosses inside its first call (410 hard flushes per step); a program that
   has just started has not. Counted per PROGRAM, not per call, so a main
   pays the introductory trims once in its life rather than at every step.
2. **The room** (`METALJAX_FLUSH_FOOTPRINT_MB`, default 3/8 of `hw.memsize`
   = 48 GB here). Even for a main, the pool may claim only what the
   footprint target has left after the program's own live set is paid for.
   `foot - cache` rather than `foot`, because the cache is what the bound is
   about to limit: charging a program for the pool it is being asked to
   shrink would make the bound collapse the moment it was granted.

The **floor** (`METALJAX_FLUSH_FLOOR_MB`, default 2048) is P25's shipped
watermark, and it is the no-regression anchor: it is what every program got
before, so no program can be trimmed harder than it was, and both rules can
only ever hand memory back. The **cap** is `METALJAX_FLUSH_CLEAR_MB`, whose
default moves 2048 → 32768 — the value P25 measured row 19's anchor at, now
safe to set because nothing else can spend it.

Nothing else about the cadence changes: same trigger, same `trim_cache`, same
untouched loop/ingest/recovery clears.

**Why the gate and the room are both needed** — each covers the other's
blind spot, and row 18 exercises both within four flushes of each other:

* the gate is what keeps the load's pool small: the spike lands inside a
  program on its 7th flush, still at the floor (`n=7 ... cache=1536MB`);
* the room is what catches a main that turns out to be huge: two flushes
  later the same program is past the gate at `n=9`, and the bound it gets is
  not the 32768 cap but **11129 MB**, because 38 GB of live set has already
  spent the target — then 2170 MB at `n=10`, and the 19.5 GB the spike frees
  is returned instead of held (`was=19458MB`).

## 3. The three conditions

All measured on `aa7bc0b6…` — `frozen-p27b.dylib`, and `frozen-p27c.dylib`
rebuilt from the final tree, which came out byte-identical — one machine-lock
hold, sequential, every model row through `p27_model.sh` (recovery precheck,
`mem_guard.sh` at the row's historical budget, durable logs).

### Condition 3 — row 19 fixed: **PASS**

| row 19 (maxtext train 0.6B, ms/step) | | peak | budget |
|---|---:|---:|---:|
| P27, five runs | **460.0 / 468.7 / 469.7 / 470.2 / 478.1** (median **469.7**) | 25-26 GB | 48 |
| same binary, P25 semantics (same hold) | 811.6 | 16 GB | 48 |
| P25 shipped binary (P25's own battery) | 833.9 | 20 GB | 48 |
| fixed watermark 32768, no P27 policy | 465.9 | 39 GB | 48 |
| the 0.11.3 anchor | 440 | | |

469.7 ms is **1.067× the anchor** — the class asked for — at **25 GB** of
peak, which is 14 GB *below* what the flat 32768 watermark needed for the
same speed. The difference is the seven introductory trims the gate charges
the program: the pool it then keeps settles at 18.6 GB instead of 26. The
spread across five runs is 460-478 (±2 %); the 478 is the last run of a
two-hour hold and the 460 the first.

### Condition 1 — no OOMs, no panics, every guard held: **PASS**

| row | runs | exit | peak (guard) | budget |
|---|---|---|---:|---:|
| 19 maxtext train | 5 | 0 | 25-26 GB | 48 |
| 18 LoRA E2B | 5 | 0 | 55-56 GB | 70 |
| 13 E2B keras-int4 | 1 | 0 | 48 GB | 70 |
| 2 gemma4-12B bf16 | 1 | 0 | 30 GB | 45 |

Row 18 is the one that had to be repeated, and the repetitions say something
P25's could not: its meter peak is **56712 / 57480 / 57479 MB** — the same
event, `n=11`, within 1.4 % — against **56711 MB** for the same binary at P25
semantics. The peak is the live-set spike, it is reproducible to the
megabyte, and this policy does not move it. What P25 saw as a 37-56 GB band
was the guard's 2 Hz sampling catching a sub-second event or missing it.

Row 2 is here because it has the **tightest budget of the four** (45 GB), and
a footprint target set too high is exactly the mistake that would show up
there first: 30 GB peak with the policy on, 30 GB with it off, decode 92.9 vs
93.2 ms/tok. Nothing moved, which is what a compiled decode row should do —
it barely reaches an eager flush at all.

### Condition 2 — no regressions: **PASS**

The suite pair moved (native/Stage 1 **0.9893** against P25's 0.9685) and the
two arms disagree about why: the native arm is FLAT against P25's native
(geomean 1.0010, median 0.9994) while the **Stage 1** arm — frozen code, on a
machine that cannot have changed — reads **0.9799** of P25's Stage 1, and
uniformly so across classes (big 0.971, mid 0.984, db 0.989). A cross-day
pair cannot separate a faster machine from a slower policy, so it was not
asked to: the same binary ran the suite a third time in the same hold with
the policy off (`METALJAX_FLUSH_CLEAR_MB=2048 METALJAX_FLUSH_FOOTPRINT_MB=0`,
which makes cap == floor and the gate a no-op — P25's semantics exactly).

| suite-106 | geomean | median |
|---|---:|---:|
| **P27 / P25-semantics, same binary, same hold** | **0.9983** | 1.0001 |
| …`big` (the eager mains the pool is for) | 0.9836 | 0.9989 |
| …`mid` / `db` / `synth` | 1.0092 / 1.0020 / 1.0153 | |
| P25-semantics / Stage 1 (today's pair, shipped policy) | 0.9910 | |
| P27 / Stage 1 (today's pair) | 0.9893 | |
| P25's pair, P25's day | 0.9685 | |

So on today's machine the shipped policy reads 0.9910 and P27 reads 0.9893 —
**P27 is the better of the two in the only comparison that holds the machine
fixed**, and the distance from 0.9685 is the day, not the change.

## 4. Battery

| gate | result |
|---|---|
| `texmo_gate` | **106 ok / 0 FAIL** of 106, three times (the battery's, and both arms of the retry probe below) — P25's run had the documented `mid03` flake, none of these did |
| `execute_test` | all cases match the CPU backend, incl. P25's four contracts and P27's four |
| `ingest_test` | 0 failed (8 checks) |
| `smoke_test` | pass |
| `bazel test //...` | pass |
| suite-106, both stacks, one hold | 106/106 within 1.2×, table above |
| model rows 19 ×5, 18 ×5, 13, 2 | every guard held, table above |

**The four new contracts** (`execute_test.py::_p27_flush_pressure`) run P25's
traffic program with one rule disabled at a time, so a failure names the rule:

| contract | reads |
|---|---|
| an eager main earns the pool | bound 256 → 4096 MB at flush 8, peak 3330 MB cached |
| the gate is what grants it | gate closed: every bound 256 MB, peak 305 MB, same checksum |
| the footprint target takes it back | target 1 MB: every bound 256 MB even past the gate |
| the bound is the target minus live | bound 383-1075 MB on 394 of 395 flushes, worst 79 MB off the footprint identity |

**The hazard a byte watermark cannot see.** Metal caps LIVE BUFFERS at ~499k
by COUNT, and a config sweep that accumulates freed small buffers is the shape
that hit it (CLAUDE.md item 11a) — raising a byte bound is exactly the change
that could let a count run away. Probe: the 106-config gate under
`METALJAX_DEBUG=1` on both policies. **0 buffer-limit recoveries on both**,
106/106 correct on both.

## 5. Scrutiny

* **The measured binary is the tree.** Two comments in `runtime.cc` were
  corrected after the battery (they described the LoRA load as "thousands of
  small programs" — which the meter disproves: one program takes 255 flushes
  there and the spike lands in the *next* one). The tree was rebuilt and the
  dylib came out **byte-identical** (`aa7bc0b6…` both times), so
  `frozen-p27b` and `frozen-p27c` are the same binary and every number above
  is the tree's. `execute_test`, `smoke_test`, row 19 and row 18 were re-run
  on it anyway.
* **The gate's margin on row 18 is one flush.** The spike lands at `n=7`; had
  it landed inside the long-lived build program (which is at `n=255` and past
  the gate), only rule 2 would have held it, and rule 2 acts one flush LATER —
  measured on the same row, it drops the bound to 11.1 GB at 38 GB of live set
  and 2.2 GB at 47. The expected peak in that case is ~54 GB rather than
  ~57: still inside the row's 70 GB budget, but it is the thinnest margin in
  the policy and the one to watch if this row's loader changes.
* **`flush_bound` reacts, it cannot predict.** Whatever a program allocates
  BETWEEN two flushes is unbounded by construction — on this row that window
  is 27 GB, because a keras convert step materializes a model's worth of
  arrays between two 1 GB-of-traffic flush points. Nothing in the eager
  cadence can see that coming; what the policy does is make sure the pool is
  not sitting next to it.
* **The footprint target is per PROCESS.** Two metaljax processes may each
  claim 48 GB. Under the machine-lock discipline that does not arise, and the
  guard budgets have the same property, but a user running two trainings at
  once has 96 GB of permission on a 128 GB machine.
* **Neither rule asks whether the pool is being USED, and that is the hole
  P28 closes.** Both are permissions — "has this program flushed enough to be
  a main" and "does the process have footprint to spare" — and a checkpoint
  load that sweeps 134 flushes through one call satisfies both while cycling
  ~3 GB. The gap was invisible here because this battery measured rows
  19/18/13/2 and the rows it cost are 11 and 14. See
  `notes/cpp-p28-benefit-gate.md`.
* **Two suite rows move more than the aggregate, and the suite-context trap
  applies to both directions.** `mid11-b64l128` reads 1.125 and
  `big14-b32l128` 0.902 against the same-hold control INSIDE the sweep;
  standalone, `mid11` is 1.027 and `big14` is 1.06-1.18 across four
  repetitions per arm whose ranges overlap (P27 20.85-24.49, control
  19.72-20.97). The suite says the opposite of standalone on `big14`, which
  is the trap CLAUDE.md item 12 documents; neither direction survives
  repetition, and the aggregate (0.9983 over 106) is the statement that does.
* **Row 18's loss series varies run to run under BOTH policies** — final
  losses 2.8469-2.8701 over today's nine runs, with the P27 runs
  (2.8533/2.8663/2.8669/2.8671) inside the band the shipped policy spans
  (2.8644-2.8701). Same nondeterminism class as the RC-gate-1 finding (the
  recognizer emits are not run-to-run stable); the first loss is identical to
  four decimals in every run, and the divergence grows with step, which is
  amplification rather than a different answer. `texmo_gate` (106 configs
  against jax-CPU, three runs) is the correctness statement.
* **What P25's table would have predicted, and why it was wrong.** Reading
  the sweep, 32768 "costs" row 18 its guard. It does not: the watermark never
  touches that row's live set, and with the pool held small beside the spike
  the same row runs 9 % FASTER (360 vs 389 ms/step) at its historical peak.
  The sweep measured a real trade, but between the wrong two things.
