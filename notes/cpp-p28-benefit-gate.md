# P28: the watermark has to be EARNED (2026-08-17)

*The 0.11.5 release gate scored gate 5 as REGRESSION over one thing: P27's
flush watermark costs the two maxtext DECODE rows 17 GB and 11 GB of peak
footprint, buys them no speed at all, and guard-kills both at the budgets
every previous campaign used (`notes/release-gates-0.11.5.md` gate 5,
"REGRESSION 1"). Oleg's option (b): benefit-gate the watermark so those rows
keep their historical footprint while row 19 keeps P27's 1.85x. This pass
does that. The rule is one more `min` over P27's two, and the counter it is
written in came out of the two rows' own flight logs: **the live set**.*

*Code: `plugin-native/runtime/{runtime,program,config}.cc`, `program.h`,
`plugin-native/metal/{metal_client,runtime_gil_free_test}.cc`, four new
`execute_test.py` contracts (and two pre-existing arms pinned to
`METALJAX_FLUSH_EARN_MULT=0`, see §5). `src/` and `native/` are FROZEN and
untouched. Raw data, runners and the frozen binaries:
`~/.cache/metaljax-bench/logs/p28-benefit-gate/` (`p28_diag.sh` — the
diagnosis; `p28_battery.sh` — every validation phase; `p28_suite.sh` — the
suite; `p28_close.sh` — the `texmo_gate` re-run and §7's combined-build
re-spot; `meter.py`, `evaluate.py` — the trace analysis, which replays
candidate rules over the MEASURED flush traces before any of them was
built).*

## 1. What the flight logs say, and what they refuse to say

Rows 11 and 14 had never been run with the flush meter on, so the first thing
this pass did was run them (`p28_diag.sh`, both policies, ~10 s a row). Per
PROGRAM, on the shipped 0.11.5 policy:

| program | hard flushes | pool p50 | pool max | live hi / lo |
|---|---:|---:|---:|---|
| 19 `jit_train_step` (5 calls) | 352 | **11,749 MB** | 21,456 MB | 20,463 / 6,843 |
| 19 `jit__lambda` (the load) | 142 | 2,375 | 17,833 | 3,561 / 0 |
| 11 `jit_create_sharded_state` ×2 | 134 | 2,375 | **19,586** | 3,561 / 0 |
| 14 `jit_create_sharded_state` ×2 | 137 | 2,968 | **20,667** | 3,575 / 14 |
| 14 `jit__generate_jit` (71 calls) | 71 | 2,071 | 2,291 | 1,197 / 1,197 |

**The 17 GB and 11 GB are one flush each.** Row 11's whole life sits at a
2.4-3.0 GB pool with the footprint flat at 4.7-6.3 GB for 133 flushes, and
then:

```
flush #267: active=311MB  cache=5491MB  bound=32768MB foot= 6338MB n=133
flush #268: active=1166MB cache=19586MB bound=32768MB foot=21289MB n=134
```

That is the checkpoint restore finishing: 14 GB of staging buffers are freed
into the pool at once, the bound is the cap so nothing trims them, the
program then ends — and the decode phase that follows **never reaches an eager
flush at all** (row 11's decode narrates `flushes=0`), so nothing ever takes
them out again or gives them back. Row 14 is the same event four flushes
wide, plus a decode program that flushes once per call and sits at the floor
throughout.

**Every counter P27 has says "eager main" for that load.** It takes 134 hard
flushes in a SINGLE call, so rule 1's gate opens at flush 8 and stays open for
the remaining 126; its live set is ~2 GB against a 48 GB target, so rule 2's
room is the whole cap. Flush count, call count, traffic per flush, trims at
the floor (row 14's decode trims on 62 of 71 calls — *more* often than the
training step) — none of them separates the row that needs the pool from the
rows that are only storing it.

**The live set does, and not by coincidence.** A trim can only ever cost a
program the memory it has to RE-ACQUIRE after the trim, and across a flush
point that is bounded by how far its own live set falls and rises. Whatever a
program churns BETWEEN two flushes is served out of the pool with no trim in
the way. So a program whose live set is flat at its flush points loses
nothing when the pool is taken — which is precisely rows 11/14 — and one
whose live set swings 13.6 GB every flush loses that much — which is row 19.

## 2. The rule

`runtime.cc::flush_bound` gains a third rule over P25's floor, and rules 2
and 3 are now both `min`s so that neither has to know about the other:

```c++
if (governor_squeezed())                    return floor;   // the veto
if (st.flushes < flush_main_flushes)        return floor;   // 1. the gate
int64_t bound = cap;
if (flush_footprint_bytes > 0)                              // 2. the room
  bound = min(bound, flush_footprint_bytes - (phys_footprint() - cache_now));
if (flush_earn_mult > 0)                                    // 3. the BENEFIT
  bound = min(bound, min(flush_earn_mult * st.swing(), st.peak_live()));
return min(cap, max(floor, bound));
```

`FlushState` (program.h) is the program's own flush history: P27's hard-flush
count, plus the high and low water of its live set sampled at each hard flush
(`mx::get_active_memory()`, read where `flush_eval` has just settled
everything). Both marks are monotone, so the grant they imply cannot
oscillate, and the live set is what the program HOLDS rather than what the
pool caches — so it is evidence independent of the bound being decided.

**Two clamps, and both were measured to be necessary.**

* `flush_earn_mult * swing` — what the program cycles. The multiplier is 2
  because a flush point lands at an arbitrary phase of the allocation cycle,
  so the swing seen between two of them is a lower bound: row 19's pool peaks
  at 1.58 swings under P27. `1` was measured and is too tight — **row 19 reads
  569.9 ms/step at mult 1** against 456-470.
* `peak_live` — the most the program has ever held at once. Twice the swing is
  still too generous for a checkpoint load, which cycles 2.8 GB of staging
  through a 3.1 GB live set and was handed 7.1 GB for it; that put row 11's
  guard at 22 GB against its 20 GB budget on one run in three. A pool larger
  than the program's own high-water cannot be one it is cycling: at that
  high-water it held that much and no more. Row 19 pays nothing for it (its
  high-water is 20,463 MB against a pool that peaks at 18,640).

`METALJAX_FLUSH_EARN_MULT` is the knob, **default 2**; `0` restores P27's
two-rule bound exactly. Because rule 3 enters as a `min`, every bound it
decides is ≤ P27's and ≥ P25's floor: **no program can be trimmed harder than
P25 trimmed it, and no program that was safe under P27 becomes unsafe.**

What the rule decides on the three programs, from the meter's own `earn=`:

| program | live hi / lo | 2 × swing | peak_live | **earn** | bound | which term binds |
|---|---|---:|---:|---:|---:|---|
| 19 train step | 20,463 / 6,843 | 27,240 | 20,463 | **20,463** | 20,463 | high-water |
| 11 / 14 load | 3,561 / 0 | 7,122 | 3,561 | **3,561** | 3,561 | high-water |
| 14 decode | 1,197 / 1,197 | 0 | 1,197 | **0** | 2,048 (floor) | **swing** |

**Both terms are load-bearing, and each has exactly one shape it catches.**
Since `live_lo >= 0`, the swing can never exceed the high-water, so `2*swing`
binds only for a program whose live set never falls below half its peak — a
program that HOLDS rather than cycles. That is row 14's decode step, whose
live set is the same 1,197 MB at all 71 of its flushes and which therefore
earns nothing at all. Everywhere else the high-water is the tighter of the
two, and it is what takes the load from 7.1 GB to 3.6. A single-term rule
would have missed one of the two rows: `2*swing` alone leaves the loads at
7.1 GB (row 11 killed 1 run in 3), `peak_live` alone hands the decode step
1.2 GB of pool it has no use for.

## 3. Row 11 at its own budget, six reps of each policy, interleaved

Row 11's peak is **not** the pool: its restore crest is a sub-second LIVE
transient of ~17 GB that is present under every policy including P25's, and
`mem_guard.sh` samples at 2 Hz, so a single peak reading is a coin flip (P25
itself reads 7.62-17 GB across runs). What the policy owns is how much pool
is standing beside that crest. Six reps of each arm, interleaved so the
machine drifts through all three equally, at the row's **historical 20 GB
budget**:

| policy | completions | decode ms/tok | peak footprints (GB) |
|---|---|---|---|
| **P28, shipped default** | **6 / 6** | 16.63–16.83 | 11.0, 9.6, 18.0, 16.0, 15.0, 13.0 |
| P25 semantics (the budget's own era) | 6 / 6 | 16.52–16.90 | 17.0, 17.0, 17.0, 7.6, 8.4, 12.0 |
| P27 (`METALJAX_FLUSH_EARN_MULT=0`) | **0 / 6 — killed every time** | — | 25, 21, 21, 25, 25, 21 |

The pool at the crest is 3.6 GB under P28, 2.0 GB under P25 and 7.1-19.6 GB
under P27, and the completions follow it. Counting the three-rep phase as
well, P28 is **9 of 9** at this budget (16.61-16.83 ms/tok) and P25 **9 of 9**
(16.52-17.07): the two arms are indistinguishable in speed, which is the
finding restated — this row was never paying for the pool.

## 4. The five rows

Every row through `p28_battery.sh` on `frozen-p28b-37060770.dylib` — recovery
precheck, `mem_guard.sh` at the row's HISTORICAL budget, one guarded process
per row, never chained, machine lock held per phase.

| row | budget | 0.11.5 shipped | **P28** | peak | verdict |
|---|---:|---|---|---|---|
| **11** Qwen3-0.6B decode | **20 GB** | guard kill (22-25 GB) | **16.61 / 16.81 / 16.75 ms/tok**, exit 0 ×3 (+6 more) | 19 / 18 / 11 GB | at budget, ≡ the 16.85-16.92 cell |
| **14** maxtext qwix-int8 | **25 GB** | guard kill (26 GB) | **31.95 / 32.13 / 31.82 ms/tok**, exit 0 ×3 | 9.2 / 9.1 / 13 GB | at budget, ≡ the 32.00-32.14 cell |
| **19** maxtext train 0.6B | 48 GB | 456.1 ms/step | **459.2 / 458.4 / 462.5 ms/step** | 25 GB ×3 | the 456-470 class holds; `loss` / `loss_first` **identical to P27's to 13 digits** |
| **18** LoRA E2B train | 70 GB | 359.2 ms/step | **361.8 ms/step**, exit 0 | 60 GB (guard) / **57,478 MB** (meter) | 1.007× the cell; §4.1 |
| suite-106 (native) | | the rc column | **0.9963** of it (106/106) | | no row regressed; §4.2 |

### 4.1 Row 18 — the spike is the same event, to the megabyte

361.8 ms/step against 359.2 (the gate's cell) and 360.2 (P27) — 1.007×, inside
the ±2 % this row has held all week — with the final loss 2.8648, inside the
2.8469-2.8701 band P27 measured across nine runs of BOTH policies.

The peak deserves its own line because the guard's number went UP (its flight
log reads 60 GB against the gate's 57) while the meter's went nowhere: the
`foot=` field inside the dylib — the same `task_info` reading P27 published —
maxes at **57,478 MB**, against P27's **56,712 / 57,480 / 57,479 MB** for the
same event, within 2 MB of two of them. Rule 3 cannot raise a bound (it
enters as a `min`), so a policy-caused increase is not available as an
explanation; what the two readings differ about is a 2 Hz sampler catching a
sub-second live-set spike at a different phase, which is exactly what P25's
"37-56 GB band" turned out to be. Rule 3 in fact LOWERS this row's bound as
well — its 250th flush reads `bound=6654MB earn=6654MB` against a 32 GB cap —
and the row is no slower for it.

### 4.2 Suite-106 — no regression against the recorded rc column

Per Oleg's scope ruling of 2026-08-18 the Python engine is out of release
scope, so the claim is **not** a native/Stage-1 pairing: it is this campaign's
native run against the **recorded rc numbers** for the same 106 configs (the
release gate's own native arm), plus P27's native arm as the second control.
Both arms are `ms_step`, so **> 1 means P28 is slower**.

| P28 native against | geomean | median | min / max | rows > 1.1× |
|---|---:|---:|---|---:|
| **the recorded rc column** | **0.9963** | 0.9997 | 0.826 / 1.094 | **0** |
| P27's native arm | 1.0004 | 0.9994 | 0.973 / 1.069 | **0** |

| class | vs rc | vs P27 |
|---|---:|---:|
| `big` (34) | 0.9795 | 1.0031 |
| `mid` (30) | 1.0052 | 0.9989 |
| `db` (40) | 1.0023 | 0.9991 |
| `synth` (2) | 1.0303 | 1.0025 |

**No row regressed past 1.1× against either control**, and exactly one row
falls outside ±10 % at all — `big12-b8l256` at **0.826**, i.e. faster. Summed
over the suite, 1995.4 ms against the rc column's 2003.2. Against P27 the
whole suite is flat to 0.04 %, which is the expected result for a rule that
enters as a `min` and that no suite config's live set is shaped to trip:
these are compiled training steps, not checkpoint loads.

## 5. Battery

Everything below is on `frozen-p28b-37060770` — the binary the rows were
measured on. That is not a formality: the battery had been run on the previous
build and §5.1 is what re-running it found.

| gate | result |
|---|---|
| `execute_test` | **553 `ok` rows**, all cases match the CPU backend (549 + P28's four) — after §5.1 |
| `ingest_test` | 0 failed, 13 checks |
| `smoke_test` | all checkpoints passed |
| `bazel test //...` | `runtime_gil_free_test` PASSED, uncached |
| `texmo_gate` | **106 ok / 0 FAIL / 0 decline / 0 error of 106** (24 via sensitivity scaling) |

**The four new contracts** (`execute_test.py::_p28_benefit_gate`) are two
programs run through one harness, so the difference is the program:

| contract | reads |
|---|---|
| a flat live set earns nothing | 545 flushes past the gate, bound ≤ 256 MB against a 4096 MB cap, worst live-set swing 79 MB, peak 309 MB cached |
| the earn rule is what denies it | same program, `EARN_MULT=0`: every bound the 4096 MB cap, peak 3379 MB — 11× the pool, same checksum |
| a cycling live set earns the pool | live-set swing 377 MB, bound up to 434 MB on 221 of 260 flushes, peak 786 MB cached |
| the bound is the multiplier times the swing | bound 419-434 MB on 221 of 260 flushes, **worst 0 MB off** the `min(cap, max(floor, earn))` identity; `earn` bound by the high-water on 221 flushes and by the swing on 46 |

### 5.1 The fourth contract was still testing the rule's first draft

Re-running the battery on the shipped binary — the contracts phase had been
run on `frozen-p28-f7f3c708`, the build BEFORE the `peak_live` clamp, while
every model row came from `frozen-p28b-37060770` after it — failed one case:

```
the bound is the multiplier times the swing
    FAIL: earn=419 MB where the sampled live set says 724 MB (hi 419, lo 57)
```

**The rule was right and the contract was stale.** Its second half asserted
`earn == mult * (hi - lo)`, which is §2's formula before measurement added the
high-water clamp; the swing probe drops to 57 MB from a 419 MB high-water, so
`2 * swing` is 724 MB and `peak_live` is what the shipped rule hands it — the
clamp doing exactly the job row 11 bought it for, on a probe whose shape
happens to match. The check is now the shipped identity
`min(mult * (hi - lo), hi)`, and it narrates which of the two terms bound each
flush, so a future divergence names the term rather than the rule (221
high-water, 46 swing on this probe). The first half — `bound` equals `earn`
clamped to `[floor, cap]` — was exact on both binaries and is untouched.

The battery above is the 11:36-11:38 run of 2026-08-18, with the fix in. One
edit landed after it — the success message now prints the total flush count
alongside the two term counts — and it is narration only: it runs after every
assertion in the case and formats a string. Under release rule 1 that is the
"provably cannot move a number" exemption, taken deliberately rather than by
queue-jumping the vendoring milestone's model rows for a re-run of a print
statement.

The three other contracts pass on both builds; what moved between them is the
*numbers* in the table above (the pool the swing probe keeps is 786 MB rather
than 964, because the clamp is what now bounds it). The lesson is P27's own
"the measured binary is the tree", one step further out: a battery split
across two builds attests neither, and the half that was not re-run is the
half that failed.

**Two pre-existing arms are now pinned to `METALJAX_FLUSH_EARN_MULT=0`, and
that is deliberate**: P27's four arms and the governor's "pressure takes the
pool back" all drive `_P25_TRAFFIC`, whose live set is flat by construction
("a scalar carry" — the comment is in the probe). Rule 3 alone would hold
every one of their bounds at the floor, so each would pass while proving
nothing about the rule it exists to test. Their design was already "one rule
disabled at a time, so a failure names the rule"; this is the third rule
taking its turn. `_p28_benefit_gate` turns it back on and owns its own arms.

**The meter gained two fields**, `live=` and `earn=`, after `n=`. `live=` is
the value rule 3 SAMPLED, not a re-read: the two differ by whatever the step
dropped in between — measured on the swing probe's phase-change flush, 399 MB
sampled against 95 MB printed — and a flight log has to show the rule's own
input. The `execute_test` regexes parse everything after `bound=` positionally
and were updated together.

## 6. Scrutiny

* **Row 11's budget was always marginal, and this does not fix that.** The
  17 GB crest is a live transient in the orbax restore; P28 puts 3.6 GB beside
  it instead of 19.6, which is enough for 20 GB and was enough 12 times out of
  12 across the reps here, but the row has no headroom to spare and a loader
  change would put it back. Same class of exposure as P27's own "the gate's
  margin on row 18 is one flush".
* **`peak_live` is a clamp, not a theory.** The swing term has an argument
  behind it (what a trim costs you is what you must re-acquire); the
  high-water clamp is an empirical ceiling with a plausible story, added
  because 2×swing measured too generous on exactly one program shape. If a
  future row wants a pool larger than its own live set, this is the term that
  will deny it, and the note to read is this bullet.
* **The multiplier was swept against two rows, not derived.** 1 breaks row 19
  (569.9 ms/step), 2 holds it (458-462) and leaves rows 11/14 inside their
  budgets. Nothing was measured between or above.
* **`get_active_memory` sees only what MLX allocated.** The transfer path
  adopts staging blocks through MLX's alien-buffer constructor, so a streaming
  load can be resident with MLX reporting almost none of it — the same caveat
  `phys_footprint`'s comment records for rule 2, and the reason rule 3 can
  only ever LOWER a bound rather than raise one.
* **Racy under `METALJAX_CONCURRENT_EXECUTE`** in exactly the way `g_stats`
  and P27's flush counter are, and as harmlessly: the water marks decide a
  watermark.
* **Row 14's decode program lands at the floor with `earn=0`** — its live set
  is 1,197 MB at all 71 of its flushes, which is the strongest form of the
  rule's premise. Its timing is unchanged (31.8-32.1 vs the 32.00 cell), which
  is the evidence that a floor-bounded pool costs a flat-live-set program
  nothing.
* **The battery spans two libmlx builds, and the split is dated, not hidden.**
  The model rows (§3, §4), the suite and the row-18 run are 2026-08-17, on
  pip's stock MLX 0.32.0. The contracts and `texmo_gate` above were re-run
  2026-08-18 11:29-11:39, by which time the concurrent vendoring work had
  staged our patched fork build into the venv MLX the plugin resolves through
  `@rpath` — so those two gates attest the rule against the *patched* library.
  Both are pass/fail correctness gates rather than timings, and both passed on
  both libraries (the 08-17 `texmo_gate` run reached 70 ok / 0 FAIL before it
  was interrupted, the 08-18 one 106/106), so nothing here rests on which was
  loaded. §7 is the re-spot that closes it for the numeric cells.
* **The earn expression exists twice** — `runtime.cc::flush_bound` decides it
  and `program.cc`'s meter re-derives it for `earn=`. They cannot be shared
  without either recomputing the bound or plumbing it out, and the fourth
  contract's first half (`bound == clamp(earn)`) is what would catch them
  drifting apart: it compares the two through the meter on every flush past
  the gate. Worth knowing before editing one of them.

## 7. The re-spot on the combined build

The vendoring milestone replaces the library this rule allocates out of, so
the three rows were re-measured on `frozen-vendor-d651add3` — the plugin
linked against the private patched `libmlx_metaljax.dylib` — at the same
historical budgets, one guarded process per row
(`p28_close.sh vspot`, 2026-08-18 11:40-11:43):

| row | budget | P28 on stock MLX | **P28 + vendored patched MLX** | peak |
|---|---:|---|---|---:|
| 11 | 20 GB | 16.61 / 16.81 / 16.75 ms/tok | **16.60**, exit 0 | 16 GB |
| 14 | 25 GB | 31.95 / 32.13 / 31.82 ms/tok | **31.94**, exit 0 | 9.2 GB |
| 19 | 48 GB | 459.2 / 458.4 / 462.5 ms/step | **463.5**, exit 0 | 25 GB |

Each row completes at its historical budget. Row 14 lands inside its P28
spread, row 11 0.01 ms under the bottom of its own, and row 19's 463.5 sits
1.0 ms above the top of its trio and inside the 456-470 class it must hold —
single runs against three-run spreads, all inside these rows' week-long noise.
Row 19's `loss` (**87.0428237915039**) and `loss_first`
(**228.39447021484375**) are **bit-identical across all eight runs of this
campaign** — seven on stock MLX, one on the patched library. Rule 3 reads
`mx::get_active_memory()`, which the fence fix does not touch, so the rule is
expected to be indifferent to the substitution; this is the measurement that
says it is.

The rest of the vendoring attestation (its own contracts, `texmo_gate`, suite
and model rows) belongs to that milestone's battery, not this one.
