# Native plugin vs Stage 1 — the complete performance baseline (2026-08-12)

*Campaign: both texmo suites and every non-embargoed model row, measured on
the phase-2 native plugin (`plugin-native`) and on the frozen Stage 1
trampoline, on the tree at 845ab89. Sequential throughout, machine lock held
for every measured run, `guarded_run.sh` (precheck + `mem_guard.sh`) for every
model row.*

**The campaign was HALTED at 03:33 by kernel panic #8**, during the row-9
(R1-Distill-32B) *native* attempt — a watchdog wedge in the 65 GB streaming
load, the same signature as panic #7 (footprint 54 GB, RSS 54.6, system
64.5 GB, every guard sample "ok", flight log stops mid-line:
`r1-distill-32b-0812-033145-flight.log`). The native plugin has no equivalent
of the Stage 1 engine's load-phase cache-clear cadence; that fix is being
built separately. Everything below completed **before** the panic. Rows not
reached are marked *not run (campaign halted)*.

## What the two stacks are

| stack | selection | what it is |
|---|---|---|
| **Stage 1** | default env | `plugin/` trampoline → `metaljax.engine` (Python compile) → the C++ tape (`native/`, the M6 default engine) |
| **native** | `METALJAX_PLUGIN_PATH=…/libmetal_pjrt_native.dylib` | `plugin-native/`: C++ StableHLO parse + lowering + the same executor runtime. **No recognizer emits and no msl_scan in the lowering** — that is what this baseline measures |

## Protocol and route calibration

`scripts/texmo_topconfs.py` drives `metaljax.engine` directly, which is a
Stage-1-only route (the native plugin holds no Python engine), so a new runner
— `scripts/bench_texmo_pjrt.py` (untracked, written for this campaign) — runs
the same training chunks **through jax/PJRT**, where the environment's plugin
choice is the thing being measured. Results are forced through `np.asarray`
(`jax.block_until_ready` is a no-op on this backend, CLAUDE.md item 9).

Three routes were run on top_confs so the columns are comparable:

| comparison | geomean | reading |
|---|---:|---|
| anchor (0.11.3, engine route, 256-step chunks) / Stage 1 today (same route) | **1.046** | today is 4.6 % faster |
| jax-CPU control, anchor / today | 0.986 | machine 1.4 % slower — ambient |
| Stage 1 engine-route@256 / Stage 1 PJRT-route@64 | **1.009** | **the route and chunk-length change is worth 0.9 %** |

The third line is what licenses the tables: the PJRT-route, 64-step-chunk
numbers used for the native pairing sit within 1 % of the anchor's own route.

---

## Table 1 — texmo 106-config suite (`benchmarks/texmo-suite.csv`)

PJRT route, 64-step chunks, ms/step. No historical perf anchor exists for this
suite (it is the correctness gate's config list), so the Stage-1 column is the
reference. Full per-config data:
`~/.cache/metaljax-bench/logs/native-baseline/suite106-{stage1,native}.jsonl`.

| aggregate | n | Stage 1 anchor | Stage 1 today | native today | native/Stage 1 |
|---|---:|---|---|---|---:|
| whole suite (geomean) | 106 | *(none — new suite)* | 1.00× ref | — | **4.24×** |
| whole suite (median) | 106 | | | | 3.37× |
| `big*` (large cells: gru/lstm 512–1024, transformer) | 34 | | | | **1.34×** |
| `mid*` | 30 | | | | 3.22× |
| `db*` (small recurrent — msl_scan territory) | 40 | | | | **14.61×** |
| `synth*` | 2 | | | | 1.52× |

Distribution: 33 of 106 configs within 1.2×, 41 between 1.2× and 10×, **32 at
or above 10×**. Every config ran on both stacks (0 errors, 0 declines).

Worst rows (ms/step):

| config | Stage 1 | native | ratio |
|---|---:|---:|---:|
| `db02-b4l1024` | 0.790 | 72.7 | **92×** |
| `db03-b4l1024` | 0.818 | 74.1 | **91×** |
| `db13-b4l1024` | 1.877 | 109.5 | **58×** |
| `db14-b16l128` | 0.563 | 32.3 | **57×** |
| `db15-b16l128` | 0.627 | 33.4 | **53×** |
| `db17-b4l1024` | 5.747 | 305.7 | **53×** |
| `db15-b64l256` | 1.285 | 61.1 | **48×** |
| `db11-b64l256` | 0.647 | 30.2 | **47×** |

`db02-b4l1024` at 72.7 ms/step is the pre-msl number of CLAUDE.md item 8
(72.9 ms) to three digits: without msl_scan the native plugin computes exactly
what Stage 1 computed before that milestone.

Rows where **native is faster** — matmul-bound work with no msl plan, where the
C++ compile path wins on dispatch:

| config | Stage 1 | native | ratio |
|---|---:|---:|---:|
| `big09-b8l256` (`bytes.emb.1024\|rnn.1024.tanh`) | 38.59 | 26.07 | **0.68×** |
| `big14-b32l128` (transformer d512) | 21.89 | 17.91 | **0.82×** |
| `big14-b8l256` | 11.97 | 10.61 | 0.89× |
| `big13-b8l256` (`bytes\|lstm.1024`) | 95.47 | 85.43 | 0.89× |
| `mid13-b64l128` (`attn.512.8.64`) | 6.78 | 6.27 | 0.92× |

---

## Table 2 — top_confs (163 configurations)

Anchor: `notes/data/texmo-topconfs-2026-08-05.jsonl` (0.11.3 release gate).
Stage-1-today column is the **same runner and route as the anchor**
(`scripts/texmo_topconfs.py`, 256-step chunks, checks included: **163/163 ok,
0 FAIL**); the native pairing is the PJRT route at 64-step chunks.

| aggregate | n | Stage 1 anchor | Stage 1 today | native today | native/Stage 1 |
|---|---:|---:|---:|---:|---:|
| geomean (ratio vs anchor / vs Stage 1) | 163 | 1.000 | **1.046 faster** | 0.029 (34× slower than anchor) | **36.46×** |
| median | 163 | | | | 46.3× |
| weights 0–100 | 47 | | 1.023 | | 50.1× |
| weights 100–500 | 61 | | 1.058 | | 31.2× |
| weights 500–1500 | 29 | | 1.052 | | 27.6× |
| weights 1500+ | 26 | | 1.053 | | 40.4× |
| configs beating jax-CPU | 163 | 53 (recorded) | **54** | **0** | |

Distribution: 8 of 163 within 1.2×, 5 between 1.2× and 10×, **150 at or above
10×**. Stage 1 vs anchor: best +18.7 % (`tc003-w10`), worst −5.4 %
(`tc145-w1888`) — no regression outside noise.

Worst native rows (ms/step):

| config | spec | shape | Stage 1 | native | ratio |
|---|---|---|---:|---:|---:|
| `tc018-w27` | `bits.1+bp\|split.add(mingru.1, pass)-norm-…` | b1 l4096 | 1.585 | 277 | **175×** |
| `tc014-w23` | `…mingru.1…` | b1 l2048 | 0.868 | 143 | **165×** |
| `tc015-w24` | `…mingru.1…` | b1 l2048 | 0.886 | 142 | **160×** |
| `tc019-w28` | `…mingru.1…` | b1 l2048 | 0.908 | 142 | **156×** |
| `tc029-w47` | `…mingru.1-rglru.1…` | b1 l2048 | 1.677 | 259 | **155×** |

At parity (no msl plan in the loop): `tc000-w5` 0.98×, `tc001-w7` 1.00×,
`tc075-w229` 1.00× — i.e. where msl does not fire the two stacks are
indistinguishable, which is the P5 finding (`METALJAX_MSL=0` parity) confirmed
on the whole suite.

**The single sentence for this table**: the entire top_confs sweep is
msl_scan's home ground, so the native plugin loses the whole 0.3.0-era win —
and with it every one of the 54 configs where metaljax beats jax-CPU.

---

## Table 3 — model rows (STATUS.md numbering)

Headline metric per row as in STATUS.md (LLM = warm decode ms/token; vision =
forward ms; diffusion = ms/step; training = ms/step). "peak" = guard flight-log
footprint. Anchor column = `benchmarks/models.md` 0.11.3.

| # | model | metric | 0.11.3 anchor | Stage 1 today | native today | native/S1 | notes |
|---|---|---|---:|---:|---:|---:|---|
| 1 | gemma4-31B bf16 | ms/tok | 237.5 | **243.1** (peak 66 G) | **301.6** (peak 67 G) | **1.24×** | dense decode; no sdpa emit |
| 2 | gemma4-12B bf16 | ms/tok | 92.5 | **93.9** (29 G) | **98.8** (30 G) | **1.05×** | |
| 3 | gemma4-26B-A4B (MoE) | ms/tok | 44.3 | **43.7** (53 G) | **300.5** (54 G) | **6.88×** | no MoE expert-gather emit → dense dispatch; memory identical |
| 4 | gemma4-E2B bf16 | ms/tok | 27.5 | **27.2** (12 G) | **27.2** (12 G) | **1.00×** | exact parity |
| 5 | Qwen3-8B bf16 | ms/tok | 57.8 | **59.1** (18 G) | **61.5** (18 G) | **1.04×** | |
| 6 | Llama-3.1-8B bf16 | ms/tok | 54.2 | **55.5** (18 G) | **57.6** (18 G) | **1.04×** | |
| 7 | gpt-oss-20b (MXFP4) | ms/tok | 22.2 | **21.9** (26 G) | **9497.6** (21 G) | **434×** | no qmm + no MoE gather; measured at `--decode-tokens 8` after the 128-token run passed 33 min unfinished. Memory FITS (21 G) — this is compute, not the memory block that was predicted |
| 8 | Qwen3.6-35B-A3B | — | ✗ | *not run* | *not run* | — | PAUSED, kernel-panic embargo (TASKS.md) |
| 9 | R1-Distill-32B | ms/tok | 217.7 | **213.8** (67 G) | **PANIC #8** | — | Stage 1 clean; native wedged the machine mid-load (54 G, all samples ok). **Embargoed pending the native streaming-load cadence** |
| 10 | DeepSeek-V2-Lite | — | ✗ | *not run* | *not run* | — | embargoed (maxtext 8B class) |
| 11 | Qwen3-0.6B maxtext decode | ms/tok | 15.8 | **16.42** | **16.67** | **1.02×** | token stream diverges from Stage 1 at token ~3 (see incidents) |
| 12 | Mixtral 8×7B | — | ✗ | *not run* | *not run* | — | PAUSED, same wedge class as row 8 |
| 13 | gemma4-E2B keras-int4 | ms/tok | 81.1 | **80.6** (peak 44 G) | **454.6** (44 G) | **5.64×** | no qmm emit → in-graph unpack; packed storage survives (2.7 G active) |
| 14 | maxtext qwix-int8 0.6B | ms/tok | 32.5 | **60.1** ⚠ | **62.1** | **1.03×** | ⚠ **Stage-1 regression vs anchor, 1.85×** (see below) |
| 15 | qwix-int8 Qwen3-8B | — | ✗ | *not run* | *not run* | — | embargoed (MLX command-buffer bug) |
| 16 | SigLIP 2 fwd b1 | ms | 82.9 | **85.8** (7.6 G) | **108.7** (11 G) | **1.27×** | |
| 16b | SigLIP 2 fwd b32 | ms | — | **2501.6** | **4444.8** | **1.78×** | the sdpa gap opens with batch |
| 17 | SD 3.5 Large @1024² | ms/step | 5141 | **5107** (23 G) | **16016** (27 G) | **3.14×** | no sdpa emit; both produced real images (pixel_std 94.4 / 76.1). 512² cell *not run* |
| 18 | LoRA E2B train | ms/step | 407 | **402.1** (56 G) | **656.3** (49 G) | **1.63×** | |
| 19 | maxtext train 0.6B | ms/step | 440 | **956.5** ⚠ | **962.0** | **1.01×** | ⚠ **Stage-1 regression vs anchor, 2.17×**; losses bit-identical across stacks |
| 20 | 235B-A22B 3-bit | — | ✗ | — | — | — | mlx-only row |

**Unmeasured after the halt**: row 9 native (embargoed), rows 8/10/12/15
(pre-existing embargoes), the SD 3.5 512² cell, and CPU-column re-measurements
(not in scope — anchors used).

---

## Where the performance phases must act, ordered by measured gap

1. **`msl_scan` on the native lowering — 36× geomean (top_confs), 14.6× on the
   suite's `db` class, 175× worst case.** This is the largest *aggregate* gap
   and the one that decides whether metaljax beats jax-CPU at all on texmo:
   Stage 1 wins 54 of 163 top_confs, native wins 0. Parity is exact wherever no
   msl plan fires, so the port is additive — nothing else has to move.
2. **qmm (quantized matmul) emit — 434× on gpt-oss-20b, 5.6× on E2B-int4.**
   The largest single-row gap in the campaign. gpt-oss decodes at 9.5 s/token
   natively against 21.9 ms on Stage 1. Memory is *not* the blocker (21 GB peak,
   the packed weights survive) — the whole cost is in-graph MXFP4 dequant plus
   dense expert dispatch.
3. **MoE expert gather — 6.9× on gemma4-26B-A4B** (43.7 → 300.5 ms/tok, memory
   unchanged at 51.6 GB). This reproduces the pre-gather Stage 1 number (284)
   almost exactly.
4. **sdpa fusion — 3.1× on SD 3.5 @1024², 1.78× on SigLIP b32, 1.24× on the
   31B decode, 1.63× on LoRA training.** Also the memory term: SD 3.5's peak
   moves 23 → 27 GB without it.
5. **The load-phase memory cadence (correctness/safety, not throughput).**
   Panic #8 is the native plugin missing Stage 1's clear/flush discipline on a
   65 GB streaming load. Until it lands, no 60 GB+ row can be attempted
   natively.

Below the frontier, and worth recording because they are *already* good: dense
bf16 decode is within 4–5 % (rows 2/5/6), E2B is exact parity, maxtext decode
and training are within 1–2 %, and on large matmul-bound texmo configs the
native path is **up to 32 % faster** than Stage 1 (`big09-b8l256` 0.68×). The
dispatch argument for the C++ engine is settled — what is left is the emits.

---

## Stage-1-vs-anchor regressions (flagged per the campaign's charter)

| row | anchor | today | ratio | attribution |
|---|---:|---:|---:|---|
| 14, maxtext qwix-int8 0.6B | 32.5 | 60.1 | **1.85× slower** | `METALJAX_ENGINE=py` gives 42.3 → ~1.30× predates the tape, ~1.42× is the tape |
| 19, maxtext train 0.6B | 440 | 956.5 | **2.17× slower** | `METALJAX_ENGINE=py` gives 1043 — **not** the tape; regression is older or environmental |

Everything else is within ±4.5 % of its anchor (rows 1–6, 11, 13, 16, 17, 18),
and the texmo top_confs sweep is 4.6 % *faster* than its anchor with a 0.986
CPU control. Two Stage-1 memory observations, both new on this tree and both
independent of the plugin choice: the **LoRA row's load transient peaks at
56 GB** and the **E2B-int4 row's at 44 GB** (steady states 10.2 and 3.1 GB) —
both were guard-killed at a 45 GB budget before completing at 70.

## Incidents and findings

* **PANIC #8 (row 9 native)** — see the header. Evidence:
  `r1-distill-32b-0812-033145-flight.log`.
* **The native plugin cannot share a process with TensorFlow, array_record, or
  any other statically-linked protobuf/LLVM carrier.** Symmetric, load-order
  dependent, and fatal at `dlopen`: TF first → `EXC_BAD_ACCESS` inside
  `google::protobuf::internal::AddDescriptors` (the plugin's descriptor
  registration executing TF's `MergeFromImpl` — dyld weak-definition
  coalescing); plugin first → `CommandLine Error: Option 'info-output-file'
  registered more than once` when TF's LLVM loads. It reached every model row:
  the keras rows import TF directly, the gemma-lib rows through
  `kauldron`, the maxtext rows through `array_record_module.so`.
  **Fix, verified**: link the dylib with an exported-symbols list holding only
  `_GetPjrtApi` and `_metaljax_native_set_callback_trampoline`
  (`-Wl,-exported_symbols_list`). The dylib drops 166 MB → 46 MB and coexists
  with TF. *Every native model number in Table 3 was measured with that
  relinked build*; `plugin-native/metal/BUILD` was restored immediately
  afterwards and the tree carries no trace of it (the experimental dylib lived
  in session scratch and was wiped by the reboot). It is a one-line build
  change and it belongs in the plugin.
* **`mx.get_active_memory()` is unreliable under the native plugin** (0.0 on
  the gemma-venv rows): the Python `mlx` module and the plugin's linked libmlx
  can be two runtimes. Read memory from the guard flight log.
* **Row 11 token divergence**: Stage 1 and native produce different greedy
  continuations from token ~3 (`"The capital of Italy is Rome…"` vs `"The
  capital of France is also…"`). Consistent with the fused-vs-unfused attention
  rounding (footnote-22 ladder class), not established as benign — worth a
  logit-delta check when the emits land.
* **int4 run-to-run outlier**: one Stage-1 run of row 13 measured 439.0 ms/tok
  where three others give 80.4 / 80.6 (and the anchor 81.1). It was the run
  immediately following a five-model chain; treat sub-suite model timings taken
  straight after other big loads as contaminated (the suite-context trap of
  CLAUDE.md item 12, in the model harness).
* **gpt-oss native at 128 decode tokens never finished** (33 min, memory flat at
  21 GB, killed); the row was re-measured at 8 decode tokens.

## Provenance

Every number in the tables is sourced to a surviving artifact under
`~/.cache/metaljax-bench/logs/`:

| table | artifact |
|---|---|
| top_confs Stage 1 (engine, anchor-comparable) | `native-baseline/topconfs-stage1-engine.jsonl` |
| top_confs Stage 1 / native (PJRT) | `native-baseline/topconfs-{stage1,native}-pjrt.jsonl` |
| suite-106 both stacks | `native-baseline/suite106-{stage1,native}.jsonl` |
| model rows 1–7, 9, 13, 16, 17, 18 | `<bench-id>-0812-<stamp>.jsonl` + `-flight.log` |
| row 7 native (8 tokens) | `native-baseline/row7-native-0812-024351.jsonl` |
| model rows 11, 14, 19 (both stacks + py-engine controls) | `native-baseline/{qwen3-06b-maxtext,maxtext-qwix-int8,maxtext-train-06b}-{s1,native}-0812-*.log` |

**UNVERIFIED (session scratch wiped by the reboot; from the transcript only)**:
the relinked dylib's validation — `plugin-native/execute_test.py` 501 of 502
checks (the one failure being the known intermittent
`There is no Stream(gpu, N) in current thread` 8-thread row, P8.5/P15 open
item) and `smoke_test.py` clean — and its perf-neutrality check against the
pristine dylib on five suite configs (`db11-b256l512` 62.19 vs 63.24,
`db11-b64l256` 30.09 vs 30.19, `big05-b32l128` 32.08 vs 32.11, `big05-b8l256`
47.97 vs 47.57, `mid13-b64l128` 6.24 vs 6.27 — all within 1 %). The Table 3
native cells rest on that check; re-run it when the linker change lands.
