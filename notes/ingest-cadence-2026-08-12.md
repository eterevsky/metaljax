# The ingest cadence, and what panic #8 was not (2026-08-12)

Kernel panic #8 (03:33) killed the machine ~90 s into a 65 GB bf16 streaming
load of DeepSeek-R1-Distill-32B through the **native** plugin. The same
load through Stage 1 had completed 8 minutes earlier, and its telemetry
carried a number the native run had no analogue for: `"clears": 16`. This
note is (a) where Stage 1's clears come from, (b) the cadence the native
plugin now has, (c) what the measurements say the cadence does and does not
reclaim, and (d) why the panic is not attributable to its absence.

Data: `notes/data/ingest-cadence-2026-08-12.jsonl`,
`notes/data/ingest-qwen3-8b-native-flight-2026-08-12.log`; raw runs under
`~/.cache/metaljax-bench/logs/ingest-ladder/`.

## 1. Stage 1's load-phase mechanism is the HARNESS's, not the engine's

`scripts/model_bench/adapter_keras_extra.py`:

| what | where |
|---|---|
| `_STREAM["clears"]` / `clear_s` — the counter the telemetry reports | ~198-203 |
| `clear_every_gb`, from `BENCH_STREAM_CLEAR_GB`, **default 8 GB** | ~401-404 |
| the trigger, inside the patched `_direct_assign`: `if assigned_gb >= _next_clear: _next_clear += _clear_gb; _stream_clear()` | ~381-383 |
| `_stream_clear()` = `gc.collect()` + `mx.clear_cache()` | ~439-449 |

R1-Distill-32B, Stage 1, 2026-08-12 03:23 (the run that completed):
`assigned_gb` 122.05 (keras counts the f32-policy variable, twice the bf16
file), `clears` 16 = ⌈122.05 / 8⌉, `clear_s` 3.29 s over a 355 s load.
gemma4-26b-a4b's 48.07 GB gave 7; the 15 GB rows gave 2 — the cadence is
exactly 8 GB of assigned weights.

**The Stage 1 ENGINE has no transfer-denominated cadence at all.**
`engine.buffer_from_host` reclaims nothing; `engine.reclaim` fires at compile
boundaries (duty-limited `gc.collect` + `clear_cache`) and every 50 k executes
(`METALJAX_CLEAR_PERIOD`), `interpreter._eager_flush` above
`METALJAX_FLUSH_CLEAR_MB`, `ops/control._loop_flush` every
`METALJAX_LOOP_CLEAR_COST` op-units. All of them are reached by EXECUTING.
A model load executes almost nothing, so on Stage 1 the only thing standing
between a 65 GB ingest and the OS was the bench harness.

And the harness's clears were never inert for the native plugin either: the
plugin links the venv's own `libmlx.dylib` (`third_party/mlx/workspace.bzl`),
so it is the SAME image `mlx.core` imports. Measured: Python's
`mx.get_active_memory()` tracks the native plugin's transfers 1:1 (0.25 GB per
256 MB `device_put`, 16 of them → 4.0 GB).

## 2. The port

Runtime (`plugin-native/runtime/`), mirroring `loop_account` one for one:

* `program.h` — `Config::ingest_clear_bytes` (default 8 GiB),
  `Stats::ingest_bytes` / `ingest_clears`, `void ingest_account(int64_t)`.
* `config.cc` — the new `configure` parameter.
* `runtime.cc` — `ingest_account`: charge the bytes, and on crossing the
  budget `gc_collect()` (empty natively, by design) + `mx::clear_cache()`,
  narrating under `METALJAX_DEBUG`/`METALJAX_MEMDBG` with the cache size
  before and after, so a flight log can prove the cadence engaged.

Plugin (`plugin-native/metal/`):

* `metal_client.cc` — `METALJAX_INGEST_CLEAR_MB` (default 8192, 0 disables),
  and the charge inside `BufferFromHostBuffer`'s `wrap` lambda, which every
  transfer funnels through.
* `metal_buffer.cc` — `CopyToMemorySpace` charges its `mx::copy`; that one
  really does allocate through MLX.
* `metal_executable.cc` — `StatsDelta` grew `ingest=NNNMB(+clear N)`.

**No eval barrier is needed.** `wrap` already `eval`s every ingested array
before the buffer is handed out, so nothing lazy pins a staging block; the
flat resident set under 9.8 GB of churn (below) is the proof.

## 3. What the cadence reclaims — measured

**Not the transfers.** MLX's `array(void*, shape, dtype, deleter)` adopts the
staging block as the array's storage (the alien-buffer path), so ingest
memory never passes through MLX's buffer cache and a `clear_cache` has
nothing of it to return. `mx::get_cache_memory()` stays at **0 B** through:
8 000 × 4 KB, 8 000 × 256 KB, 100 × 256 MB (9.8 GB churned), and a real
15.26 GB checkpoint. Freed staging goes back to the allocator at once —
churning 9.8 GB peaks at 0.349 GB resident.

**The work AROUND the transfers, yes.** With a dtype round trip per tensor on
the device (`MJ_INGEST_CAST=1`, what a converter does), the same checkpoint
leaves **2.584 GB** in MLX's cache with the cadence off; with it on, each
clear returns what has accumulated (2646 MB → 0 at the 8 GB default;
272 MB → 0 at the seventh 2 GB clear) and the load ends at 2.318 GB.
The PEAK is not bounded by any cadence — one embedding cast (1.16 GB bf16 →
2.32 GB f32) is the peak, and it happens between two clears.

Qwen3-8B (399 tensors, 15.26 GB bf16, mmap'd shards, held to the end):

| run | clears | peak footprint | peak RSS | end cache |
|---|---:|---:|---:|---:|
| cadence 2048 MB | 7 | 15.00 GB | 17.226 GB | 0 |
| cadence off | 0 | 15.00 GB | 17.226 GB | 0 |
| + cast, cadence 2048 MB | 7 | — | 5.227 GB | 2.318 GB |
| + cast, cadence off | 0 | — | 5.227 GB | 2.584 GB |

Predicted peak = weights 15.26 + largest tensor 1.16 + base 0.12 = **16.54**;
measured **17.23** (+0.7 GB of jax/python overhead), within 4 %. The cadence
changes the peak by 0.4 MB and costs no measurable time — it is insurance
whose premium is zero, not a fix for a leak.

**The panic's own workload shape, at 10 GB** (`gemma4-e2b-bf16`, keras
streaming load through the native plugin, the harness's own clears disabled
with `BENCH_STREAM_CLEAR_GB=1000`, plugin cadence 2048 MB). It needed the
exported-symbols relink of the gotcha below, applied temporarily and
reverted; the dylib in `bazel-bin` is the tree's again.

    load ok, 20.9 s, mem_gb 10.2, decode 26.8 ms/tok (Stage 1's 0.11.3
    record: 10.5 s, 10.2, 27.5), peak footprint 12.00 GB, peak RSS 15.41
    plugin ingest clears: 3, at 5248 / 7305 / 9357 MB ingested, cache 0->0
    harness stream_load clears: 1 (finalize only -- its cadence was off)

So the load-phase reclamation point now comes from the plugin whether or not
the embedder provides one, which is what row 9's embargo asked for. The
`cache=0->0` says again that a keras assign of a bf16 checkpoint into a bf16
policy leaves nothing cached: the cadence has teeth only where a converter
computes (§3), and it is the clear's presence, not its yield, that the
embargo was about.

## 4. Panic #8 is not the missing cadence

Comparing the two flight logs at the same point of the same load:

| | Stage 1 (03:23, completed) | native (03:31, panicked) |
|---|---|---|
| at 51-54 GB footprint | RSS 51.5, claimed 62.7 G, t≈150 s | RSS 54.6, claimed 64.5 G, t≈128 s |
| fill rate | ~0.35 GB/s | ~0.42 GB/s |
| how far it got | 62-63 GB footprint, claimed 72.7 G, **twice in one process** | died at 54 |

Nothing in the native run's memory profile was worse at the moment it died —
it was ~20 % faster at filling memory, and the Stage 1 run passed through
that exact state twice. With the cache measurement above (ingest never
touches it), the "freed staging accumulates unboundedly" hypothesis is out.
Panic #8 sits with #4/#7 in TASKS.md's wedge class: watchdog starvation with
memory metrics healthy, whose only known leading indicator is the sustained
mmap read+copy rate. The 65 GB retry needs Oleg's go and should carry the
rate lever (`BENCH_STREAM_SYNC`, or a slower assign) as the variable under
test, not the cache.

## 5. Ladder + gotchas

* `plugin-native/ingest_test.py` — synthetic rung (8/8 checks: the cadence
  fires once per `METALJAX_INGEST_CLEAR_MB`, is disabled by 0, a held load
  costs its weights plus one transfer, a churned one is flat) and
  `--checkpoint <dir>`, the real-model rung.
  `~/.cache/metaljax-bench/logs/ingest-ladder/run_ckpt.sh` is its
  lock+precheck+`mem_guard.sh` runner (`guarded_run.sh` only knows manifest
  rows, `wedge_run.sh` only `wedge_repro.py`).
* **The static-protobuf/LLVM collision is in the tree, and it is why the real
  rung is a checkpoint stream.** On this tree `import tensorflow` SIGSEGVs in
  protobuf's `AddDescriptors` (crash reports 2026-08-12 09:08:58, 09:11:21)
  and the `gemma` venv's `gemma_lib` path aborts on `Option
  'info-output-file' registered more than once` — so every `keras_lm` row and
  every gemma-lib row dies within seconds, in either import order. P16's
  ledger entry has the diagnosis and the fix (an exported-symbols list of
  `_GetPjrtApi` + `_metaljax_native_set_callback_trampoline`, 166 -> 46 MB,
  measured to coexist with TF), and records that **the change is not in the
  tree**. Until it lands, no keras streaming row runs natively here, which is
  what the `--checkpoint` rung stands in for.
* zsh does not word-split unquoted variables: an A/B loop built with
  `set -- $pair` silently ran both cells with the default cadence (the two
  `cast-default*` rows in the data file are that mistake, kept because they
  are what produced the 2646 MB → 0 clear).
* The working tree was SHARED with the in-flight P17 recognizer work
  (`metal_qmm/moe/sdpa.cc`, `metal_recognize.h`, and their edits to
  `metal_lowering` / `metal_executable`). One `bazel test //...` failed with
  `undeclared inclusion of metal/metal_recognize.h` while that package's
  `srcs` were being edited underneath it, and passed on rerun; the one
  `texmo_gate` FAIL of the first battery likewise did not reproduce (two
  clean 106/106 runs after). Neither is attributable to this change, and
  neither would have been diagnosable without knowing the tree was moving.

## Battery (all green, tree 845ab89 + P17 in flight)

`bazel test //...` (rerun), `smoke_test`, `execute_test` (all cases match the
CPU backend), `decline_census` 35 of 35, `ingest_test` 8 of 8,
`texmo_gate` **106 ok / 0 decline / 0 FAIL, twice**, and the native wheel
built and run from a fresh 3.13 venv (`wheel_poc_test`).
