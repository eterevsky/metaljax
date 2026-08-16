# P25: the eager flush trims MLX's pool instead of dumping it (2026-08-16)

*Oleg approved the P20 root cause's proposed fix —
`mx::set_cache_limit(flush_clear_bytes)` at plugin init. That literal shape was
implemented, measured, and **rejected on the numbers** (it costs 2.35x on a
compiled decode row); what ships is the same idea applied at the same cadence
the clear had. Both are reported below with their measurements, because the
choice between them, and the watermark's value, are Oleg's.*

*Code: `plugin-native/runtime/{runtime,program,config}.cc`, `program.h`,
`plugin-native/metal/metal_executable.cc`, four new `execute_test.py`
contracts. `src/` and `native/` are FROZEN and untouched — the Stage-1
backport question is stated at the end. Raw data, runners and both frozen
binaries: `~/.cache/metaljax-bench/logs/p25-cache-limit/`.*

## The change

A hard eager flush that found more than `METALJAX_FLUSH_CLEAR_MB` (2048) in
MLX's buffer cache used to call `mx::clear_cache()` — **the whole pool back to
the OS**, so the next couple of GB of allocations are cold Metal buffers. On
maxtext's training step that is 7 dumps a step at ~70 ms each: P20 measured the
row at 2.2x its anchor on both stacks and root-caused it here.

It now **trims** the pool back to the watermark instead:

```c++
// runtime.cc — MLX has no "trim to N": it has a cache LIMIT and reclaims down
// to it "on the next allocation" (mlx/memory.h).  So: set the limit, poke the
// allocator, put the limit back.
void trim_cache(int64_t bytes) {
  if (bytes < 0) return;
  const size_t prev = mx::set_cache_limit(static_cast<size_t>(bytes));
  mx::allocator::Buffer poke = mx::allocator::malloc(1);
  mx::allocator::free(poke);
  mx::set_cache_limit(prev);
}
```

The poke is an allocator call, not a kernel — microseconds, and it takes the
same mutex the reclaim needs anyway. Everything else about the cadence is
unchanged: same variable, same watermark, same hard-flush trigger, same
programs affected. `flushes=N(+clear M)` in the per-execute stats line is now
`flushes=N(+trim M)`, and `METALJAX_MEMDBG` gained the meter the ladder is
argued on — one line per hard flush with `active=`, `cache=`, `(was …)` and
`bound=`, printed from inside the dylib (an embedder can only sample *between*
executes; an eager program spends its whole life inside one).

**Not touched**, because they answer to different masters: the loop cadence's
clear (`loop_account` — Metal's ~499k live-buffer COUNT, which no byte limit
bounds), the ingest cadence (`ingest_account` — the transfer path, which
reaches no flush at all), and every resource-limit recovery (after an
exhaustion the point is to free *everything* freeable).

**No new hazard**: the trim frees a strict subset of what `clear_cache()` freed
at the same instant, so anything the shipped dump was safe to do, this is too.

## Why not at plugin init — the variant that was approved, measured, rejected

`mx::set_cache_limit(2048 MB)` once, at client construction, bounds the pool at
every instant — including on the paths that never reach an eager flush and were
therefore never bounded before. That is the half that costs:

| measurement (same day, same hold, adjacent arms) | global limit | shipped dump | scoped trim |
|---|---:|---:|---:|
| row 13, E2B keras-int4 decode, ms/tok (clean position) | **190.0** ⚠ | 80.7 | **80.8 / 80.9** |
| row 13, same pair inside the long campaign | 201.9 | 122.8 ᵃ | 82.1 |
| row 19, maxtext train, ms/step | 866.9 | 950.1 | **833.9** |
| row 18, LoRA train, ms/step | 411.3 | 403.0 | 393.98 |
| texmo suite-106, native/Stage 1 geomean | **1.0420** ⚠ | 1.0050 ᴾ²³ | **0.9685** |
| …its `mid` class | 1.1175 | 0.998 ᴾ²³ | 0.9848 |
| …native arm vs P23's native arm | 1.0588 | — | 0.9882 |

ᵃ that whole campaign position reads high — the control is itself 1.5x off the
row's anchor, which is why the pair was re-measured in a clean position (top
row). Both readings agree on the direction.

A compiled decode step whose transients exceed the bound re-allocates from the
OS every step; on this row that is **2.35x**. The suite says the same thing more
quietly (`mid` +11.8%, `mid07-b64l128` 1.89x against P23's native arm). And a
fourth arm of the row-13 repetition **died on the global binary**:

```
INTERNAL: metaljax-native: jit_compiled_generate_function failed:
[METAL] Command buffer execution failed: Caused GPU Address Fault Error
(0000000b:kIOGPUCommandBufferCallbackErrorPageFault)
```

one occurrence, on the arm that reclaims at *arbitrary allocation points* while
command buffers are in flight, after three clean runs of the same row on the
other two binaries. Not proof — but the mechanism is exactly the one an
allocation-time reclaim exposes and a flush-point reclaim does not, and it is a
second reason not to bound globally. The binary is kept
(`frozen-p25b.dylib`, sha256 `a1ae46ce…`) if Oleg wants it re-run.

## The memory ladder

Run first, before any stopwatch number was allowed to count.

| rung | what it shows | result |
|---|---|---|
| `execute_test` "the pool stays under its bound" | 18 GB of eager traffic through 64 distinct-size intermediates, 552 hard flushes, watermark 256 MB | **peak 255 MB** cached |
| …"the bound is a trim, not a dump" | the median flush must still find memory cached — a dump leaves 0 | **median 228 MB** (bound 256) |
| …"the bound is what bounds it" | control: `METALJAX_FLUSH_CLEAR_MB=-1` on the same program | pool runs to **4025 MB** |
| …"a long loop clears on its own cadence" | 20k interpreted iterations: the live-buffer COUNT discipline is untouched | **104 loop clears, 0 recoveries** |
| `ingest_test` (held / churn / off) | the transfer path's own cadence, 9.8 GB of churn | **8/8 checks**, cache peak 0 MB, RSS flat |
| row 18 (LoRA), the 81-GB-blowout row | peak footprint at the shipped watermark | **39 GB** (sweep arm), 56 GB in one run — see below |
| row 19, guarded | peak footprint | **20-21 GB**, both mechanisms |

**Row 18's peak is a LOAD transient, not a step property** (every peak lands at
sample ~10 of ~60, during the load): it read 37 GB (P20), 38, 39, 43 and 56 GB
across today's runs *on both binaries*, so the ±17 GB band is the row's, not the
mechanism's. What does move it is the watermark itself — see the sweep.

## Row 19, and what the watermark is worth

The mechanism swap is worth **1.17x** on the row it was aimed at (975.4 → 833.9,
same hold, adjacent arms; the rc control is P23's shipped binary). It is *not*
worth the 2.03x that "turn the clear off" promised — because what that A/B
turned off was the **bound**, not the dumping. The bound is the rest of it, and
it is a straight memory-for-speed trade:

| `METALJAX_FLUSH_CLEAR_MB` | row 19 ms/step | trims/step | row 19 peak | row 18 ms/step | row 18 peak |
|---:|---:|---:|---:|---:|---:|
| 512 | 1067.0 | 29-30 | 20 GB | — | — |
| **2048 (shipped)** | **833.9** | 15-16 | 21 GB | **395.6** | **39 GB** |
| 8192 | 685.6 | 5-9 | 25 GB | 364.5 | 57 GB |
| 32768 | 464.1 | 0 | 39 GB | — | **guard-killed at 68 GB** |
| unbounded (`1e6`) | 461.7 | 0 | 39 GB | (P20: 81 GB, killed) | |
| *shipped dump, same day* | *975.4* | *(6-8 dumps)* | *21 GB* | *400.0* | *37-43 GB* |

So row 19 reaches its anchor-era number (464.1 vs the 440 anchor, 461.7
unbounded) at a **32 GB** watermark — and 32 GB is exactly where **row 18 blows
through its 70 GB guard**, which is the wall P20 hit and the reason a bound
exists at all. 8192 is the interesting middle: row 19 1.42x better than the dump
and row 18 *faster* than either (364.5 vs 400.0), for +18 GB of peak on a 128 GB
machine.

**Shipped default left at 2048** — strictly better than the dump on every row
measured, at the same memory. Raising it is a one-variable decision with the
table above; it is Oleg's, not this pass's.

## No-regression battery (shipped configuration, `frozen-p25c.dylib`, sha256 `516e4b43…`)

| gate | result |
|---|---|
| `execute_test` | **all cases match the CPU backend**, incl. the 4 new P25 contracts |
| `ingest_test` | **0 failed** (8 checks) |
| `smoke_test` | pass |
| `texmo_gate` | 105 ok / 1 FAIL — `mid03-b64l128`, worst 1.95e-2 vs tol 1.8e-2. **P23's documented flake for this exact config**; re-run standalone **3/3 ok on this binary and 3/3 ok on the RC binary** |
| `bazel test //...` | pass |
| texmo suite-106, both stacks, one hold | geomean **0.9685**, median 0.9904, **106/106 within 1.2x**, 81 native-faster; native arm **0.9882** of P23's native while the Stage-1 control reads **1.0254** of P23's Stage 1 (i.e. the machine is ~2.5% slower today and the native arm is still 1.2% faster) |
| row 13 (compiled decode, qmm packs) | 80.8 / 80.9 vs 80.7 rc — parity |
| row 18 | 393.98 vs 400.01 rc |

The suite gain is not incidental, and it is not uniform either: it is
concentrated in the rows whose chunk main runs eagerly and flushes often.
Against P23's native arm (median 1.0012 — most rows do not move at all), the
eight that move are all `big`: `big13-b32l128` **72.5 → 55.3** (0.763),
`big10-b32l128` 61.5 → 49.2, `big15-b32l128` 75.9 → 60.8, `big14-b32l128`
22.4 → 18.2, `big13-b8l256` 98.2 → 82.6, then 0.875-0.890 for
`big12-b8l256` / `big11-b32l128` / `big14-b8l256`. That is why the `big` class
geomean goes 1.0107 → **0.9296** while its median stays at 0.9974.

Aggregates and the per-row table:
`notes/data/p25-cache-limit-2026-08-16.{json,csv}`.

## The Stage-1 backport — Oleg's call

`runtime/program.cc::eager_flush` is the transliteration of
`src/metaljax/interpreter.py::_eager_flush`, and Stage 1 has a second copy in
its frozen C++ tape (`native/program.cc:38`). Stage 1 therefore still dumps its
pool and still reads ~2.2x its anchor on row 19 (today: 975.4 through the RC
plugin, and Stage 1's own engine measured 969.1 at P20). The backport is three
edits and no new design:

* `src/metaljax/interpreter.py` — a `_trim_cache` beside `_eager_flush`
  (`mx.set_cache_limit` / a 1-byte `mx.array` or `mx.metal` allocation to poke
  the allocator / restore), replacing `mx.clear_cache()` at line 776;
* `native/runtime.cc` + `native/program.cc` — the two edits made here;
* the Stage 1 battery re-run, which is the real cost: reopening `src/` pulls
  the pytest suite, the jax-test suite and the wheel gates with it.

Until then the two stacks disagree about the eager path's memory discipline,
which is worth knowing when reading any same-day native/Stage-1 ratio on an
eager-main row (rows 14, 18, 19, and the texmo `big` class above).

## Artifacts

`~/.cache/metaljax-bench/logs/p25-cache-limit/`:

* `p25_battery.sh` (build → settle → suite pair → rows 19/18/13 → the test
  battery, one hold), `p25_model.sh` (one guarded model row: precheck, guard,
  durable logs), `p25_sweep.sh` (phase 2: mid03, row 13 global-vs-scoped, the
  watermark sweep), `analyse.py` (suite aggregates, P22/P23's method —
  reproduces P23's published numbers from P23's artifacts and refuses to report
  otherwise);
* `run1/` — the first battery. **Its Stage-1 suite arm is DIRTY**: another
  agent misread the lock and ran ~20 s of a 31B load at 10:06:17-10:06:40,
  inside that arm. Kept, not used. `run2/` — the global-limit binary. `run3/` —
  the shipped scoped-trim binary. Sweep artifacts sit at the top level;
* binaries: `frozen-p25b.dylib` (global limit, `a1ae46ce…`),
  `frozen-p25c.dylib` (**shipped**, `516e4b43…`), both byte-identical to a
  `bazel build` of the tree at their time.
