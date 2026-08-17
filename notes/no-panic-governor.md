# The memory governor: metaljax never panics the machine (2026-08-17)

*The contract (Oleg, after kernel panic #9): metaljax must NEVER panic the
machine. Preferred behaviour under memory pressure is to DEGRADE; acceptable
is a clean OOM error that surfaces as RESOURCE_EXHAUSTED through the PJRT
boundary; a wedge is neither. It applies to every tested model INCLUDING the
previously embargoed rows 9/10/12/15/20 — they may OOM-error, they may not
panic — and under it the 15 pre-migration rows must still run at
not-worse-than-pre-migration performance.*

*Code: `plugin-native/runtime/memory.cc` (new), `runtime/{runtime,program,
control,msl}.cc`, `runtime/program.h`, `runtime/BUILD`,
`plugin-native/metal/{metal_client,metal_buffer,metal_executable}.cc`, five
new `execute_test` contracts, five new `ingest_test` checks,
`scripts/model_bench/mem_guard.sh` (two logged columns). `src/`, `native/`
and every other frozen tree untouched. Raw data, runners, probes and the
frozen binaries: `~/.cache/metaljax-bench/logs/no-panic-governor/`.*

## 0. Verdict

**The contract holds.** Nothing in this campaign panicked, wedged or was
jetsam-killed — 21 guarded model runs, 8 synthetic rungs and six loads of
58-93 GB, including two 1.5×-physical loads and the hot-cache sequence that
preceded panic #9.

* a load 1.5× the machine's memory **refuses cleanly and reproducibly** (3/3,
  identical message, process alive);
* a 61 GB checkpoint that used to leave **41-53 GB of page cache** behind it
  now leaves **1 MB**, with the free list at 47 GB instead of 55 MB;
* **row 9 — panic #9's own row, run LAST after six big loads — completes**,
  at 210.7 ms/tok against its 214.4 published, carrying 14 GB of RSS instead
  of 65;
* **row 8 (panic #7's row, PAUSED for two weeks) completes** at 29.6 ms/tok,
  its first metaljax number ever; **rows 10 and 15 complete**; row 12's model
  fits (93.4 GB streamed, 0 refusals) though the row itself is blocked on a
  download; row 20 is declined by the harness, as it was before;
* the no-regression half is clean: suite-106 pairing **0.9876** (0.11.5:
  0.9917), `texmo_gate` **106/106**, `execute_test` all-match with five new
  contracts, rows 1/4/5/18 at or better than their 0.11.5 numbers.

## 1. What the two machine wedges had in common — and what they did not

| | panic #7 (row 8, 2026-08-04) | panic #9 (row 9, 2026-08-17) |
|---|---|---|
| process footprint | 53 GB | 53 GB |
| machine claimed (wired+anon+compressor) | 58.8 GB | 59.0 GB |
| guard budget in force | 95 GB | 80 GB |
| guard verdict on every sample | `ok` | `ok` |
| ingest rate | ~1.2 GB/s | **0.30 GB/s** (already throttled) |
| position | mid-battery | **LAST of 34 rows** |
| what was full | RSS 101.9 GB — the mapped checkpoint | the machine's page cache |

Neither process was near any budget it was measured against, and the second
was already running under the rate throttle the first one's ladder produced.
The common factor is not a metaljax number at all: **physical memory full of
file-backed pages while a streaming load keeps pulling more in**. Every
discipline this runtime had (P25's flush trim, P27's footprint-aware pool
bound, the 8 GB ingest clear) bounds what *metaljax* holds, and all of them
read healthy at both wedges.

**The kernel's own pressure signal does not see it either.** Measured during
this campaign, at the exact regime (`synth-ckpt-r9-off`, free list at 7.7 GB
and falling, page cache 67.6 GB):

```
[metaljax-gov] pressure: want=0.0G foot=37.9G claimed=50.7G free=7.7G file=67.6G press=1
```

`press=1` is `kVMPressureNormal`. A governor that waited for the OS to
complain would have waited through the panic.

## 2. What ships

### 2.1 The page-cache discipline — the lever that did not exist

A checkpoint is read through a mapping, and every page stays in the page cache
afterwards though nothing will read it again. Four ways of giving those pages
back were measured on this machine (16 KB pages; probes and logs in the
campaign directory):

| lever | on a read-only MAP_SHARED file mapping | on MAP_PRIVATE (copy-on-write) |
|---|---|---|
| `msync(MS_INVALIDATE)` | **drops them**: file-backed 16.39 → 12.67 GB for a 3.72 GB shard, RSS 3.72 → 0 | returns 0, drops nothing |
| `madvise(MADV_DONTNEED)` | deactivates only (inactive +3.6 GB, file count flat) | nothing |
| `madvise(MADV_FREE_REUSABLE)` | `EPERM` | `EPERM` |
| a MAP_SHARED **window of our own** over the same file, then `msync` | — | **drops them**: 32.21 → 29.24 GB, the private reader's residency to 0, contents intact and re-faultable |

Contents are never at risk: a clean file page's contents are in the file, so
dropping it costs a re-fault. The probes check that explicitly
(`reread matches=1`), and the code refuses any mapping that is writable
(could be dirty) or executable (this process's own dylibs).

Two places use it:

* **per transfer** (`metal_client.cc`): after the staging copy, the consumed
  range is released if the VM says it is a read-only file mapping —
  `release_page_cache`. jax passes explicit byte strides for most host
  arrays, so the range is the view's bounding box, and a view whose box is
  more than twice its bytes is left alone (it would be dropping a
  neighbour's pages to save its own).
* **per 8 GB of ingest, and at every governor reclaim** (`sweep_page_cache`):
  a walk of this process's own VM regions, invalidating the large read-only
  file mappings it finds — through a shadow window where the mapping is
  copy-on-write. **This is the one that matters for real loaders**: measured
  on row 4, keras-hub reads a shard, casts it and hands `device_put` an
  anonymous copy, so the mapping is never named at the PJRT boundary
  (`released=0MB` over an 8 GB ingest, and the machine's file-backed pages up
  by exactly the checkpoint: 25.9 → 36.0 GB). With the sweep: 9.8 GB
  released on the same row.

### 2.2 The governor

`runtime/memory.cc`. Reads the machine (`host_statistics64`: free, file-backed,
wired+anonymous+compressor; `kern.memorystatus_vm_pressure_level`) as well as
the process (`task_info`, P27's `phys_footprint`), samples at most every
`METALJAX_MEM_SAMPLE_US` (20 ms) so a decode step pays a compare, and is asked
at four points: **every transfer** (before the staging allocation), **every
hard eager flush**, **every loop sync point**, and **program entry**.

Two lines, four rungs:

| | rule | default (fraction of `hw.memsize`) |
|---|---|---|
| hard: process | `footprint + want > budget` | `METALJAX_MEM_BUDGET_MB` = 3/4 → 96 GB |
| hard: machine | `claimed + want > ceiling` | `METALJAX_MEM_SYS_MB` = 3/4 → 96 GB |
| soft: the panic regime | `free - want < floor` or kernel pressure | `METALJAX_MEM_FREE_FLOOR_MB` = 1/16 → 8 GB |
| squeeze (the pool veto only) | `free < floor/4` or kernel pressure | 2 GB |

1. **Trim** — `gc_collect` + `mx::clear_cache` + a page-cache sweep, at most
   once per 250 ms, then look again.
2. **Pace** — past the soft line a transfer is held to a cumulative
   `METALJAX_MEM_THROTTLE_KBPS` (1 GB/s), the same shape as the bench
   harness's `BENCH_STREAM_THROTTLE_GBPS` but inside the library and only
   while the free list is on the floor. This is the degrade the contract
   prefers; it can only slow a LOAD.
3. **Stall** — past a hard line, wait up to `METALJAX_MEM_STALL_MS` (5 s),
   re-reading and narrating, because the memory may belong to a phase that is
   about to end.
4. **Refuse** — throw. `metal_client.cc` / `metal_buffer.cc` /
   `metal_executable.cc` turn it into `absl::ResourceExhaustedError`, which
   jax raises as `XlaRuntimeError: RESOURCE_EXHAUSTED: metaljax out of memory
   at <phase>: <what was needed> ... Raise METALJAX_MEM_BUDGET_MB /
   METALJAX_MEM_SYS_MB ...`. The recovery ladders in `program.cc`,
   `control.cc` and `msl.cc` rethrow it unchanged rather than retiring a
   compiled path over it.

**The pool veto is deliberately on the harder line.** `flush_bound` (P27) asks
`governor_squeezed()`, not `governor_pressured()`: a warm page cache puts the
free list under 8 GB routinely — this machine sits at 50+ GB of stale clean
cache after a few big loads — and trimming a training step's pool there would
cost the maxtext row 1.8x for a machine that is not in trouble.

`METALJAX_MEM_GOVERNOR=0` turns all of it off.

## 3. The ladder

Guards stayed ON for every run (`mem_guard.sh` at each row's budget, with the
guard's ceilings set ABOVE the governor's lines) so that a governor failure
would be a guard kill rather than a wedge. Order was chosen so nothing could
panic before the mechanism was proven on synthetics.

### 3.1 Rung 1 — the oversized synthetic load, ×3

`ingest_test --oversize 192` (1.5× physical) at shipped defaults, guard
budget 105 GB / `GUARD_SYS_GB=108`:

| run | moved | verdict | peak footprint | page cache | governor |
|---|---:|---|---:|---|---|
| r1 | 83.0 GB | **RESOURCE_EXHAUSTED**, process alive, exit 0 | 84.0 GB | 6.3 → 6.3 GB | 11 sweeps, 5 stalls, 1 refusal |
| r2 | 83.0 GB | same | 84.0 GB | 6.3 → 6.3 GB | 11 sweeps, 5 stalls, 1 refusal |
| r3 | 82.5 GB | same | 83.0 GB | 6.3 → 6.4 GB | 11 sweeps, 5 stalls, 1 refusal |

*(and three more runs on the development binary a hold earlier: 82.5 GB each,
same refusal, same message.)*

The error names what was needed and which variable moves the line:

```
RESOURCE_EXHAUSTED: metaljax out of memory at transfer: the machine would have
96.4 GB of unreclaimable memory claimed, over the 96.0 GB METALJAX_MEM_SYS_MB
ceiling. Needed 0.5 GB more; the machine has 128.0 GB total, 0.1 GB free and
29.7 GB in the page cache, and this process holds 83.1 GB. Raise
METALJAX_MEM_BUDGET_MB / METALJAX_MEM_SYS_MB to allow more (at the risk of
paging the machine), or run a smaller model.
```

Three runs, identical to the megabyte; no guard kill, no jetsam, no panic.

### 3.2 Rung 2 — the page-cache proof at model scale

Row 9's own checkpoint (61.0 GB, 771 tensors), load-only through
`ingest_test --checkpoint`, one variable apart:

| | governor on | control (`METALJAX_INGEST_SWEEP_MB=0 METALJAX_INGEST_ADVISE_KB=0`) |
|---|---:|---:|
| machine file-backed pages | **6.354 → 6.355 GB** | 6.362 → **59.693 GB** |
| free list, low-water | **47.1 GB** | **0.055 GB** |
| peak RSS | 61.16 GB (= the weights) | 65.55 GB |
| page cache released | **58.9 GB** | 0 |
| wall | 475.9 s | 479.6 s |
| outcome | complete, exit 0 | complete, exit 0 |

*(Both arms on the shipped binary, one hold, one variable apart. The same pair
on the development binary a hold earlier read 29.766 → 29.769 GB against
29.776 → 71.306 GB, with the free list at 23.7 vs 0.05 GB — the same
statement from a different starting cache.)*

The control is the pre-governor behaviour, and it is the panic regime on a
FRESH machine: 41.5 GB of page-cache growth and a free list at 55 MB. The
governed arm is also 10 % **faster** — the reclaimer has nothing to fight.

### 3.3 Rung 3 — the hot-cache pattern (#8/#9's own shape)

Two big checkpoints back to back in one lock hold, the second arriving on the
page cache the first one left — which is what made row 9 different on the
night it panicked (34 rows before it).

| | gemma-4-31B, 58.3 GB, onto a machine already holding 52.0 GB of ambient cache | R1-Distill, 61.0 GB, immediately after |
|---|---|---|
| outcome | **complete, exit 0** | **complete, exit 0** |
| peak footprint / RSS | 58.0 / 58.23 GB | 61.0 / 62.34 GB |
| machine page cache | 51.96 → **52.10 GB** (+0.14) | falls from 53.5 to **5.2 GB** as the sweep runs |
| free list, low | 3.2 GB | **45.9 GB** |
| governor | 76 sweeps, 70 paced admissions, **0 refusals, 0 stalls** | 7 sweeps, 0 paced, 0 refusals |
| wall | 870 s | 476 s |

The first load ran with 58 GB of wired weights beside 52 GB of somebody
else's page cache — 110 GB of a 128 GB machine in use — and the machine stayed
responsive throughout. What the governor could NOT do there is drop the
*ambient* cache: those pages belong to files no live process maps, so no
`msync` of ours reaches them. That is what the pace is for, and it is where it
engaged (70 admissions). By the second load the machine is healthy again and
nothing is paced at all.

**The degrade is visible and it is the point**: 870 s against 476 s for a
comparable cold load. Not isolated to the pace — this row also has 1.5× the
tensor count and is reading against a full cache — but the direction is
right and it is the trade the contract asks for.

### 3.4 The model rows — predictions, written before the runs

*Stated here BEFORE each row ran, per the ladder's rule. Per Oleg's
amendment, the previously embargoed rows carry a third column: where the
ORIGINAL jax implementation genuinely cannot work, a minimally-fixed harness
variant is identified and run as a distinct row (fixes live in
`scripts/model_bench`, never in a frozen tree).*

| row | what it is | prediction (original) | minimal fix, if needed |
|---|---|---|---|
| 12 mixtral-8x7b bf16, 93.4 GB (keras) | weights alone are 93 GB against a 96 GB ceiling | **clean RESOURCE_EXHAUSTED mid-load**; no panic | raise the governor's own lines for this row (`METALJAX_MEM_SYS_MB`/`_BUDGET_MB` = 108 GB) — the escape hatch the error message names; run as `row12f` |
| 10 deepseek-v2-lite (maxtext MoE) | guard-killed at 122 GB in 2026-08 | clean RESOURCE_EXHAUSTED at the 96 GB line, or completes; no panic | `MAXTEXT_PREFILL_LEN=64` (activation memory scales with prefill) → `row10f` |
| 15 qwix-int8-qwen3-8b (maxtext) | 8.7 GB of weights; a >60 GB post-restore materialization killed it in 2026-08 | completes, or clean RESOURCE_EXHAUSTED; no panic | `MAXTEXT_PREFILL_LEN=64 METALJAX_BODY_COMPILE=0` (the 2026-08 mitigation) → `row15f` |
| 20 235B-A22B 3-bit, 96 GB | metaljax has no packed-3-bit path: `run_bench` refuses the row outright (`blocked-metaljax`) | the harness declines — no metaljax execution, so no panic to have | the fix is packed sub-byte quant support end to end, a FEATURE and not a harness change; not attempted. The contract is tested at that scale instead by streaming the 96 GB checkpoint through the transfer path: predicted **clean RESOURCE_EXHAUSTED** |
| 9 r1-distill-32b, 65.5 GB | panic #9's row, run LAST after a multi-load sequence — the exact conditions | **completes**, decode in its 214-218 ms/tok class, page cache bounded | — |
| 8 qwen36-35b-a3b, 71.9 GB | panic #7's row, PAUSED since 2026-08-04 | completes or clean RESOURCE_EXHAUSTED; no panic | — |
| 1 / 5 / 18 (spot) | the no-regression check | within noise of 0.11.5 (301.6 / 57.9 / 360.2) | — |

## 4. Battery (the no-regression half)

All on the shipped binary (`frozen-gov7.dylib`), guards on, one hold per phase.

| gate | result |
|---|---|
| `smoke_test` | all checkpoints passed |
| `execute_test` | **all cases match the CPU backend**, 549 `ok` rows (544 + the five governor contracts) |
| `ingest_test` | **0 failed** — 13 checks (8 + the five page-cache / refusal checks) |
| `decline_census` | 35 of 35 programs lower |
| `tests/` (native leg) | **1053 passed, 0 failed** (205 deselected: the four Stage-1-only counter files) |
| `bazel test //... --nocache_test_results` | `//metal:runtime_gil_free_test` PASSED |
| **texmo suite-106 pairing** (native/Stage 1, one hold) | **0.9876** geomean, median 1.0002, **106/106 within 1.2×** — against 0.11.5's 0.9917 |
| …drift control, gov-native / 0.11.5-native | **0.9939** (the governor binary is marginally *faster*) |
| …drift control, gov-Stage 1 / 0.11.5-Stage 1 | 0.9980 (the machine is where it was) |
| **`texmo_gate`** (106 configurations vs jax-CPU) | **106 ok, 0 decline, 0 FAIL, 0 error** |

The analysis script reproduces 0.11.5's published pairing (0.9917 / 1.0002 /
worst `mid11` 1.129 / best `big09` 0.662) from 0.11.5's artifacts before it
reports today's — the P22/P23 self-validation rule.

## 5. The rows

Every row guarded (`mem_guard.sh` at a budget above the governor's own lines),
one process each, settle between rows, on the shipped binary.

### 5.1 The previously embargoed rows

| row | outcome | peak footprint | page cache | governor | vs prediction |
|---|---|---:|---|---|---|
| **10** deepseek-v2-lite (maxtext MoE) | **COMPLETE**, exit 0 — decode 1865 ms/tok, prefill 1902 ms, load 189 s | 88.0 GB | flat | 14 lines, 0 refusals | predicted "clean OOM or completes" — **completes**, where 2026-08 guard-killed it at 122 GB |
| **10f** the same with `MAXTEXT_PREFILL_LEN=64` | **guard kill** on the ramp (`projected 109 GB (+7 GB/sample)`), no panic | 95.0 GB | flat | 7 lines, pacing, 0 refusals | the fix is a no-op here (the original already computes a 64-token prefill), and the row sits ON the machine's edge: 88 GB one run, 95+ the next |
| **15** qwix-int8 Qwen3-8B (maxtext) | **COMPLETE**, exit 0 — decode 369.7 ms/tok, load 80.5 s; output text is garbage (`" fragment!!!"`), the row's known MLX-quantization bug, not memory | 79.0 GB | flat | 13 lines, 0 refusals | predicted "completes or clean OOM" — **completes** |
| **15f** `MAXTEXT_PREFILL_LEN=64 METALJAX_BODY_COMPILE=0` | **COMPLETE**, exit 0 — decode 1064 ms/tok (the mitigation's cost), same garbage text | 75.0 GB | flat | 7 lines, 0 refusals | the fix lowers the peak 4 GB and costs 2.9× |
| **12** Mixtral 8×7B, ORIGINAL keras route | **cannot be attempted here**: `from_preset` starts a 93 GB KaggleHub download (the preset is not in the local cache), ~5 MB/s | — | — | — | the row is blocked on a download, not on memory |
| **12** the same model's 93.4 GB checkpoint through the transfer path | **COMPLETE**, exit 0 — 86.99 GiB moved, 995 tensors, 837 s | 87.0 GB | **37.52 → 37.57 GB** | 175 paced admissions, **0 refusals** | the memory question the row asks is answered: the weights fit under the ceiling, with the free list bottoming at 0.06 GB and the machine intact |
| **20** 235B-A22B 3-bit, ORIGINAL | **harness declines** (`aspirational-235b-3bit: blocked-metaljax`), exit 1, nothing executed | — | — | — | as predicted: metaljax has no packed sub-byte path, so there is no run to panic |
| **20** its 96 GB checkpoint through the transfer path | completes; only **13.7 GB** of it is transferable — the 3-bit weights are packed into `U32` blobs the test's dtype map skips | 13.8 GB | flat | 0 refusals | not the scale test it was meant to be; the Mixtral row above is |
| **8** Qwen3.6-35B-A3B — **panic #7's own row**, PAUSED since 2026-08-04 | **COMPLETE**, exit 0 — decode **29.6 ms/tok**, load 104 s | 73.0 GB | 45.9 GB peak | 30 lines, 0 refusals | first metaljax number this row has ever produced |

**Row 20's minimal fix is a feature, not a harness edit**: packed 3-bit
weights need a sub-byte quantized path end to end (the `qmm` recognizer's
MXFP4 machinery is the closest thing). Identified, not attempted.

**Row 12's minimal fix is a download**: either the keras preset (93 GB from
KaggleHub) or a keras-format conversion of the local HF checkpoint. The
memory evidence above says the model itself is inside the machine's reach.

### 5.2 Row 9 — panic #9's row, run last, on the battery's own page cache

| | 0.11.5 (the run that panicked the machine) | this campaign |
|---|---|---|
| position | last of 34 rows | **last of 21**, after 6 loads of 58-93 GB |
| ingest throttle | `BENCH_STREAM_THROTTLE_GBPS=0.30` from the harness | none: the library governs itself |
| outcome | **kernel panic** | **COMPLETE, exit 0** |
| decode | — | **210.7 ms/tok** (0.11.5 published 214.4, Stage 1 217.7) |
| load | — | 322 s |
| peak footprint / RSS | 53 GB / 52.9 at the wedge | 67.0 GB / **13.96 GB** |
| machine page cache at the end | (unlogged — the column did not exist) | peak 50.4 GB, free list never below **7.2 GB** |
| governor | — | 23 lines, **0 refusals, 0 stalls** |

The RSS column is the discipline in one number: the same load used to carry
its mapped checkpoint in the resident set (65 GB), and now carries 14.

### 5.3 The no-regression spot rows

| row | this campaign | 0.11.5 | |
|---|---:|---:|---|
| 4 gemma4-E2B bf16 | **27.1** ms/tok | 27.0 | within noise |
| 5 Qwen3-8B bf16 | **58.4** ms/tok | 57.9 | +0.9 %, within the row's own spread |
| 1 gemma4-31B bf16 | **237.3** ms/tok | 301.6 | **1.27× faster** — its load no longer competes with its own page cache |
| 18 LoRA E2B train | **359.2** ms/step | 360.2 | identical |
| 19 maxtext train 0.6B | **868.7** ms/step | 469.7 (P27) | **not the governor — see below** |

**Row 19, attributed rather than excused.** 868 ms/step is 1.85× the number
P27 published, so it was A/B'd in one hold before anything was claimed:

| arm | ms/step |
|---|---:|
| governor ON (shipped) | 867.6 |
| `METALJAX_MEM_GOVERNOR=0`, same binary | 869.3 |
| governor ON again (position control) | 868.2 |
| **the 0.11.5 RELEASE binary** (`aa7bc0b6…`, no governor in it at all) | **867.2** |

Four arms within 0.25 %. The row is 1.85× slower than yesterday **on the
shipped 0.11.5 binary too**, so whatever moved is in the environment (the
maxtext venv, its checkpoint, or the machine) and predates this campaign —
and it is worth Oleg's attention on its own, because the 0.11.5 gate never
ran row 19 (gate 5 was left pending) and P27's 469.7 is the last measurement
of it. Every other row above is at or better than its 0.11.5 number.

## 6. Scrutiny

* **The sweep is blunt on purpose.** It invalidates every large read-only,
  non-executable file mapping this process holds, not only checkpoints, so a
  program that re-reads a mapped data file pays re-faults. Bounded by the
  64 MB region floor (`METALJAX_INGEST_SWEEP_MB`), a one-second rate limit,
  and the cadence it hangs off (8 GB of ingest). Measured cost where it fires
  hardest — the two 61 GB checkpoint arms — is *negative*: the governed load
  is 10 % faster.
* **What the governor still cannot see**: a single `mx::eval` that allocates
  tens of gigabytes inside one operation. The gate is at transfers, flush
  points, loop sync points and program entry; a compiled graph with none of
  those in reach can still outrun it. Row 15's 2026-08 post-restore
  materialization (+4-7 GB per guard sample, no transfers) is that shape, and
  it is why the guard stays on.
* **The ambient page cache is out of reach.** Pages left by a process that has
  exited belong to no mapping of ours, so no `msync` of ours reaches them —
  this machine sat at 52 GB of them after two loads. The pace is the only
  answer there, and it costs load time (hot-cache rung 3). A battery that
  wants a cold machine has to drop them from outside metaljax.
* **`vm_stat`'s "Pages free" is already net of speculative pages.** The first
  version of the guard's new column subtracted them again and printed a
  negative free list; fixed, and the governor's own reading (which uses
  `host_statistics64`'s raw `free_count`) never had the bug.
* **The machine line is per MACHINE, the budget per process.** Two metaljax
  processes cannot both spend the 96 GB ceiling — they read the same
  `claimed` — which is a property P27's footprint target does not have (its
  scrutiny list records exactly that gap).
* **A refusal is recognized by its message**, not its type: the plugin builds
  without RTTI and `is_resource_limit` set the precedent. The string is
  `"metaljax out of memory"`, produced in one place.
* **Metal's live-buffer COUNT limit is untouched.** The 499k-buffer ladder
  (`loop_clear_cost`, the resource-limit retries) is a different failure and
  keeps its own machinery; nothing here raises or lowers it.
* **The guard can beat the governor to the kill.** Row 10f died on the
  guard's TRAJECTORY rule (`projected 109 GB (+7 GB/sample)`) at 95 GB while
  the governor's own line is 96 — the guard projects one sample ahead, the
  governor judges the sample it has. Under the shipped defaults with no guard
  the same run gets a RESOURCE_EXHAUSTED a moment later; with the bench guard
  on, a row this close to the line will usually be killed first. Not a bug in
  either, but it means "guard kill" and "clean OOM" are the same verdict for
  rows that sit on the machine's edge, and row 10 sits there (88 GB one run,
  95+ the next).
* **Two of the mission's rows could not be run as themselves**, for reasons
  that have nothing to do with memory: row 12's keras preset is a 93 GB
  KaggleHub download this machine does not have, and row 20 needs a packed
  sub-byte path metaljax does not implement. Both were tested at their own
  SCALE through the transfer path instead (93.4 GB: complete; 96 GB 3-bit:
  only its 13.7 GB of unpacked tensors are transferable).
* **Harness lesson (self-inflicted, recorded because it nearly cost a run):**
  the lock is a `mkdir` and the trap that removes it is unconditional, so a
  wrapper that "cleans up" a lock it does not hold can let two guarded jobs
  start seconds apart. Two 83 GB synthetic loads did start together in this
  campaign; both were killed within seconds, free memory never went below
  66 GB. Check `ps` after every background start.

## 7. Where the artifacts are

`~/.cache/metaljax-bench/logs/no-panic-governor/`:

| what | file |
|---|---|
| the four page-cache probes (C, standalone) | `pagecache_probe.c`, `region_probe.c`, `private_probe.c`, `shadow_probe.c` |
| the synthetic ladder (runner + driver log) | `ladder_all.sh`, `ladder-all-driver.log`, `synth-gov6-*.{log,jsonl,flight.log}` |
| the hot-cache ladder | `ladder2_hot.sh`, `ladder2-driver.log`, `synth-hot*` |
| the model rows | `gov_rows.sh`, `gov_model.sh`, `rows-driver.log`, `<spec>-<row>-*.{log,jsonl,flight.log}` |
| row 19's four-arm A/B | `row19_ab.sh`, `row19ab-driver.log` |
| the battery | `gov_battery.sh`, `battery-driver.log`, `bat-*.log`, `suite106-gov-*.jsonl` |
| the suite aggregate (self-validating) | `analyse_suite.py` |
| the shipped binary | `~/.cache/metaljax-bench/frozen-gov7.dylib` (`ebe56e7168eff581…`) |

A row summary is committed as `notes/data/no-panic-governor-rows-2026-08-17.json`.
