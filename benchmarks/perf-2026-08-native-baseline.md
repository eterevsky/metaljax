# Native plugin vs Stage 1 — the complete performance baseline (2026-08-12)

*Campaign: both texmo suites and every non-embargoed model row, measured on
the phase-2 native plugin (`plugin-native`) and on the frozen Stage 1
trampoline, on the tree at 845ab89. Sequential throughout, machine lock held
for every measured run, `guarded_run.sh` (precheck + `mem_guard.sh`) for every
model row.*

> **2026-08-16 update (P27) — the eager flush's watermark stops being one
> number, and rows 19/18 move with it.** P25 had shipped the flush-point TRIM
> (dump → trim, 1.17× on row 19) and left the watermark itself as a
> memory-for-speed table with no good row: 32 GB is where row 19 reaches its
> anchor and where row 18 was guard-killed at 68. P27 measured that conflict
> rather than trading it off — the flush meter now prints the process
> FOOTPRINT (`task_info(TASK_VM_INFO)`, `mem_guard.sh`'s own metric) — and
> **row 18's blowout is a live-set spike (19.6 → 46.5 GB in a second, during
> keras build/convert, identical on both binaries)**, not a pool; the
> watermark only decides how much dead pool stands beside it. `flush_bound`
> now spends a 32768 cap only on programs that have taken 8 hard flushes (an
> eager MAIN reusing a pool) and only up to what a 48 GB footprint target has
> left after their own live set, with P25's 2048 as the floor under both.
> **Row 19 1006.2 → 469.7 ms/step** (five runs, peak 25 GB / 48 budget),
> **row 18 397.5 → 360.2** (five runs, peak unchanged at 56.7-57.5 GB on the
> meter), rows 13/2 flat as controls (80.3 / 92.9). Suite-106 same-binary
> policy-on/off **0.9983** over 106 rows, gate 106/106 ×3, 0 buffer-limit
> recoveries in a 106-config sweep. Details and the three conditions
> Oleg set: `notes/cpp-p27-flush-pressure.md`.
>
> **2026-08-16 update (P24) — the last four P16 cells are closed.** Rows
> **1, 2, 4, 11** of Table 3 (the only ones still carrying a P16 native number,
> and whose Stage-1 columns were P16-era too) were re-measured on the frozen RC
> binary `frozen-rc-ed355691.dylib` — hash-verified byte-identical to this
> tree's `plugin-native/bazel-bin` build — with a fresh same-day Stage-1 control
> for each: native **301.6 / 98.6 / 27.0 / 16.63** ms/tok against P16's
> **301.6 / 98.8 / 27.2 / 16.67**, Stage 1 **242.4 / 93.8 / 27.0 / 16.39**
> against **243.1 / 93.9 / 27.2 / 16.42**. Every one of the eight numbers
> reproduces inside 0.7 %, so **no cell moved and no regression appeared**;
> ᴾ¹⁶ becomes ᴾ²⁴ and the "read these as a ceiling" caveat in
> `benchmarks/models.md` is withdrawn. Two things the re-measure settles rather
> than assumes: row 1's 1.24× is **not** a missing sdpa emit (the recognizer
> fires 0 times there, while row 2 takes 8 fused attentions and is timing-neutral
> with them), and rows 1/11's token divergence from Stage 1 is **deterministic** —
> both reproduce their own P16 native streams exactly, unlike rows 5/7. Raw:
> `notes/data/p24-stale-rows-2026-08-16.{json,csv}`,
> `~/.cache/metaljax-bench/logs/p24-stale-rows/`.
>
> **2026-08-13 update (P20).** The four named regressions, measured and
> dispositioned: row 13 **275.6 → 79.7** ms/tok and row 7 **25.3 → 22.2** (the
> fused compile gate now follows the rewrite plan), row 18 **656.3 → 397.5**
> ms/step (donated pass-through outputs are no longer copied), row 19 root-caused
> to the eager flush's cache clear (a SHARED mechanism, reported not fixed) and
> row 14's "1.85× regression" retired as a suite-context measurement. Table 3
> now carries **vs-anchor** ratio columns beside the same-day ones — row 19 read
> "1.01× of Stage 1" for two passes while both stacks sat 2.2× off the anchor.
>
> **2026-08-12 update (P18, tree 8c61e72).** The exported-symbols relink is in
> the tree as the default build and its validation was re-run for real; the
> model rows were then re-measured with the P17 recognizer emits. Table 3 now
> carries both native columns, and "Where the performance phases must act" has a
> status section. Everything above the tables is the original P16 campaign and
> is left as written.

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

Native columns by pass: **P16** is the baseline this campaign measured (no
recognizer emits, no msl_scan, no exported-symbols relink); **P18** re-measured
the rows on the P17 emits through the relinked plugin
(`notes/data/p18-relink-models-2026-08-12.jsonl`); **P19** unblocked row 7; and
**P20** (2026-08-13) is the regression campaign below — the fused compile gate,
the donated-output copies, and the eager-flush cache clear.

**Two ratio columns, deliberately.** `native/S1` is same-day and answers "is the
phase-2 plugin at parity with the trampoline"; `native/anchor` is against the
0.11.3 release column and answers "is this row as fast as it ever was". A row
where both stacks drifted together reads 1.00× in the first and shows the drift
only in the second — which is exactly how row 19 hid a 2.2× for two passes.
`S1/anchor` is the Stage 1 column's own drift, on the same principle.

| # | model | metric | 0.11.3 anchor | Stage 1 today | S1/anchor | native (latest) | native/S1 | native/anchor | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | gemma4-31B bf16 | ms/tok | 237.5 | **242.4** (peak 66 G) | 1.02× | **301.6** (67 G) ᴾ²⁴ | **1.24×** | **1.27×** | dense decode. P24 re-measured both stacks: native lands on its P16 cell **to the digit** and Stage 1 within 0.3 % (243.1 → 242.4). The gap is *not* a missing sdpa emit — on this row the recognizer fires **0** times (and 0 msl plans); it is the plain lowering |
| 2 | gemma4-12B bf16 | ms/tok | 92.5 | **93.8** (29 G) | 1.01× | **98.6** (30 G) ᴾ²⁴ | **1.05×** | 1.07× | P24: both stacks reproduce P16 inside 0.2 % (93.9 / 98.8). Here the sdpa emit **does** fire (8 fused attentions) and is timing-neutral — the P16 no-emit number and today's with-emit number are the same row |
| 3 | gemma4-26B-A4B (MoE) | ms/tok | 44.3 | **43.7** (53 G) | 0.99× | **43.4** (53 G) ᴾ¹⁸ | **0.99×** | **0.98×** | the MoE expert-gather emit closes 6.88× to parity; greedy tokens 64/64 identical to Stage 1 |
| 4 | gemma4-E2B bf16 | ms/tok | 27.5 | **27.0** (12 G) | 0.98× | **27.0** (12 G) ᴾ²⁴ | **1.00×** | **0.98×** | was already exact parity, and still is: P24 measured both stacks at 27.0 on the same peak (12 G), tokens 64/64 identical to each other and to both P16 runs |
| 5 | Qwen3-8B bf16 | ms/tok | 57.8 | **58.2** (18 G) ᶜ | 1.01× | **57.9** (18 G) ᴾ¹⁸ | **0.99×** | **1.00×** | ᶜ same-day control re-measure (P16 read 59.1 — the Stage 1 column is stable to 1.5 %) |
| 6 | Llama-3.1-8B bf16 | ms/tok | 54.2 | **55.5** (18 G) | 1.02× | **54.7** (18 G) ᴾ¹⁸ | **0.99×** | **1.01×** | |
| 7 | gpt-oss-20b (MXFP4) | ms/tok | 22.2 | **21.9** (26 G) ᵖ | 0.99× | **22.2** (34 G) ᴾ²⁰ | **1.01×** | **1.00×** | P16 9497.6 → P18 guard-killed → P19 25.3 (35 G) → **P20 22.2**: the fused compile gate was worth a further 1.16× here even though `METALJAX_TRACE_BUDGET=1e7` had said this row was not affected (it is the BYTE half that moved). ᵖ Stage 1 re-measured same-day, reproducing its anchor |
| 8 | Qwen3.6-35B-A3B | — | ✗ | *not run* | — | *not run* | — | — | PAUSED, kernel-panic embargo (TASKS.md) |
| 9 | R1-Distill-32B | ms/tok | 217.7 | **213.8** (67 G) | 0.98× | **214.4** (65.6 G) ᴾ¹⁹ | **1.00×** | **0.98×** | throttled load ladder (STATUS footnote 29) |
| 10 | DeepSeek-V2-Lite | — | ✗ | *not run* | — | *not run* | — | — | embargoed (maxtext 8B class) |
| 11 | Qwen3-0.6B maxtext decode | ms/tok | 15.8 | **16.39 / 15.99** | 1.04× | **16.63 / 16.86** ᴾ²⁴ | **1.02×** | 1.05× | two samples per stack (P24; the 16.86 carried `METALJAX_DEBUG=1`). Native reproduces its P16 cell inside 0.3 %. The token stream still diverges from Stage 1 at token ~3 — and it is **reproducible**: both P24 native runs are byte-identical to the P16 native text, both Stage-1 runs to the P16 Stage-1 text |
| 12 | Mixtral 8×7B | — | ✗ | *not run* | — | *not run* | — | — | PAUSED, same wedge class as row 8 |
| 13 | gemma4-E2B keras-int4 | ms/tok | 81.1 | **80.6** (peak 44 G) | 0.99× | **79.7** (49 G) ᴾ²⁰ | **0.99×** | **0.98×** | P16 454.6 → P18 275.6 → **P20 79.7**, and the whole 3.4× was the compile gate reading the UNFUSED IR: with cost and bytes following the rewrite plan the decode body compiles (`compiles=1 compiled_calls=127`, was `0/0`) with **no env override**. Two samples 79.7/79.7; greedy tokens 64/64 identical to the P19 run. Peak is the keras streaming LOAD transient (Stage 1's own is 44 G), not the packs |
| 14 | maxtext qwix-int8 0.6B | ms/tok | 32.5 | **32.9 / 32.7** ᵈ | **1.01×** | **35.0** ᴾ²⁰ | 1.06× | 1.08× | ᵈ **the 1.85× "Stage-1 regression" does not exist** — re-measured standalone twice today, the row sits on its anchor. P16's 60.1 was taken 12 minutes into a sequential campaign; treat it as the suite-context trap (CLAUDE.md item 12). Native's residual 1.06× is the eager-flush cache clear: with `METALJAX_FLUSH_CLEAR_MB` lifted it reads **32.1** (0.98×) |
| 15 | qwix-int8 Qwen3-8B | — | ✗ | *not run* | — | *not run* | — | — | embargoed (MLX command-buffer bug) |
| 16 | SigLIP 2 fwd b1 | ms | 82.9 | **85.8** (7.6 G) | 1.03× | **87.9** (8.2 G) ᴾ²⁰ | **1.02×** | 1.06× | re-run on the P20 binary as the sdpa-row regression check: 96.9 → 87.9, so the compile-gate change helps here too rather than shifting a cadence |
| 16b | SigLIP 2 fwd b32 | ms | — | **2501.6** | — | **2350.6** ᴾ²⁰ | **0.94×** | — | the sdpa emit closes 1.78× and goes past Stage 1; P18 read 2400.5 |
| 17 | SD 3.5 Large @1024² | ms/step | 5141 | **5107** (23 G) | 0.99× | **5781.6** (24 G) ᴾ¹⁸ | **1.13×** | 1.12× | sdpa emit: 3.14× → 1.13×; real image (pixel_std 68.0) |
| 17b | SD 3.5 Large @512² | ms/step | 1389 | **1520.7** (21 G) ᶜ | **1.09×** ⚠ | **1234.8** (21 G) ᴾ¹⁸ | **0.81×** | **0.89×** | ᶜ measured for this pairing. Native is 19 % faster than Stage 1 and 11 % faster than the anchor; the ⚠ is Stage 1's own 9 % drift, still unattributed |
| 18 | LoRA E2B train | ms/step | 407 | **398.9** (56 G) ᶜ | 0.98× | **397.5 / 396.2** (37 G) ᴾ²⁰ | **1.00×** | **0.98×** | P16 656.3 → **P20 397.5**. The gap was **1,952 output copies of donated pass-through parameters** (~10 GB/step) plus the eager-flush cadence; the copy list now exempts donated aliases as `engine.py::_dealias` does — `0 output copies, 2255 donated` — and the row's peak drops 55 → 37 G, below Stage 1's own 56 |
| 19 | maxtext train 0.6B | ms/step | 440 | **969.1** ⚠ | **2.20×** ⚠ | **833.9** ᴾ²⁵ | **0.86×** | 1.90× ⚠ | **P20's shared drift, half-fixed natively in P25**: the eager flush trims MLX's pool instead of dumping it (975.4 → 833.9 against a same-day RC control, 1.17×). The rest is the WATERMARK, not the dumping — 685.6 at 8 GB, 464.1 at 32 GB, 461.7 unbounded, and 32 GB is where row 18 blows its 70 GB guard. Stage 1 still dumps (frozen), which is what the 0.86× same-day ratio now measures |
| 20 | 235B-A22B 3-bit | — | ✗ | — | — | — | — | — | mlx-only row |

**Unmeasured**: rows 8/10/12/15 (pre-existing embargoes), row 9 native
(embargoed; the retry is the main agent's), and CPU-column re-measurements (not
in scope — anchors used). Rows 1/2/4/11 were the P16 leftovers of this list
until **P24 (2026-08-16)** closed them on the RC binary — table above, raw data
`notes/data/p24-stale-rows-2026-08-16.{json,csv}`, logs
`~/.cache/metaljax-bench/logs/p24-stale-rows/`. Rows 18/19 were closed by P20.

### The one blocked row, and the one gap the emits did not close

**Row 7 (gpt-oss-20b) is blocked on memory, not compute.** The emits fire, but
building those packs without qmm's row-blocked `_Source` evaluation and without
its cross-executable build cache (P17 left both out deliberately) keeps a full
pack set per compiled shape live at once. Measured: a steady climb to 46 GB at
the row's historical 45 GB budget, and an oscillating 49–62 GB plateau at 60.
Stage 1 runs the same row at 25 GB. No further escalation was attempted — 62 GB
is panic #7/#8 territory.

> **UNBLOCKED 2026-08-13 (P19).** Both optimizations are ported
> (`notes/cpp-p19-packing.md`) and the row completes at the historical 45 GB
> budget: **35 GB peak, 25.3 ms/tok, 128 tokens** — 1.16× of Stage 1's 21.9,
> inside the ~1.5× target and better than P17's micro proxy (1.55×). Raw cells:
> `notes/data/p19-packing-models-2026-08-13.jsonl`.
>
> **Which of the two ports did it is not what was expected.** An ablation at the
> same budget, one knob at a time:
>
> | configuration | peak | outcome |
> |---|---:|---|
> | both (P19 default) | **35 G** | ok, 25.3 ms/tok |
> | cache off, blocking on | 46 G | **guard-killed** — P18's number to the gigabyte |
> | blocking off, cache on | 36 G | ok, 25.5 ms/tok |
> | neither (P17/P18) | 46 G / 62 G | guard-killed at both budgets |
>
> So the **build cache is the load-bearing fix** (46 → 36 GB: three executables
> were each building their own ~10 GB pack set) and row-blocking is worth a
> further gigabyte on top. P17's argument that "the tape already stages the
> evaluation op by op with last-use pruning" was substantially right about the
> per-weight TRANSIENT; what it did not cover was the same pack set built three
> times over. Both are in, and only the pair clears the 45 GB line.
>
> Mechanism, from the run's own log: 94 packs built and **188 reused** (three
> executables × 94 weights, a 100 % hit rate on the second and third), every one
> of the 94 blocked (47 in 16 row blocks, 47 in 32), and the pack-wave peak
> reported by the plugin's own libmlx is **33.9 GB for the first wave and 0.000
> GB for the other two** — the reuse waves allocate nothing at all. (That 33.9
> is a process-wide high-water mark, so it includes the resident model; the
> flight-log footprint is the figure to compare across rows.)
>
> Row 7 does **not** have row 13's compile problem: `METALJAX_TRACE_BUDGET=1e7`
> gives 25.3 ms/tok, the same number, with the compile decisions bit-identical
> (16 compiles / 354 compiled calls either way). Its decode loop was already
> compiling.
>
> Scrutiny: the greedy tokens diverge from Stage 1 at index 52 of 64. There is
> no prior native token record for this row (P18 never completed it), so this is
> a first observation rather than a change, and it is the same late-divergence
> ladder class already carried for rows 3, 5 and 11.

> **Row 13 (P19).** Peak barely moves — 48 → 46 GB — and the reason is worth
> recording: that peak is the **keras streaming LOAD transient**, not the packs.
> Stage 1's own is 44 GB on this row (see the regression section below), and the
> plugin's pack wave peaks at 6.64 GB against a 46 GB flight peak. What P19 does
> change here is the steady state, **4.2 → 3.2 GB**, and the work: **259 packs
> built, 518 reused** (three executables × 259 weights, 0 fingerprint declines),
> with the two reuse waves peaking at 0.000 GB. All 259 pack whole, on both
> stacks — a keras `Dense`'s `[K, N]` weight needs a transpose to reach the
> `[(B,) N, K]` matrix `quantized_matmul` wants, which is exactly the
> precondition `_blocking` tests.

**Row 13's residual 3.09× is the compile gate reading the UNFUSED IR.** The
fused decode program reports `compiles=0 compiled_calls=0 serial_loops=1`: the
decode while body never compiles, so the loop runs op by op. It is not the byte
budget (`METALJAX_COMPILE_BYTES_MB=1e8` alone: 274.6 ms/tok, no change) and it
is not the packs (`METALJAX_RECOGNIZE=0` also reports `compiles=0`). It is the
**cost** term: `body_compile_max = min(kTraceBudget/cost, …)`, and `BlockCost`
walks the StableHLO block, still charging every op the qmm emit ABSORBS — the
whole int4 dequant chain. With `METALJAX_TRACE_BUDGET=1e7` the same row measures
**85.5 ms/tok = 1.06× of Stage 1**. Making the fused lowering's cost and byte
estimates follow the rewrite plan is worth 2.9× here, and is a candidate
explanation for any other emit row that lands short of parity.

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

### Status of that list after P17 + P18 (2026-08-12)

Items 2, 3 and 4 are **closed on the rows that could be re-run**: MoE 6.88× →
0.99×, sdpa 3.14× → 1.13× (@1024²), 1.78× → 0.96× (SigLIP b32), and at 512² SD
3.5 is now 0.81× of Stage 1. Item 1 (`msl_scan`) is untouched and remains the
whole texmo gap. Three new items join the list, all of them found by re-running
these rows:

6. ~~**The fused lowering's compile decisions read the unfused IR**~~ —
   **CLOSED 2026-08-13 (P20)**. `BlockCost` and `BlockBytes` now consult
   `ctx.plan` exactly as `ops/control.py`'s do: an absorbed op is charged
   nothing and is not recursed into, a qmm or moe root costs 2 units and an
   sdpa root 3, and a root's bytes are its own result plus what the emission
   really builds (`qmm.emit_bytes`'s activation copy, `moe.emit_bytes`'s
   pair-space plan; sdpa declares none, because the scores it absorbs are never
   written). Row 13 **275.6 → 79.7 ms/tok** with no env override — past the
   `METALJAX_TRACE_BUDGET=1e7` proxy (85.5) because the byte term moved too —
   and row 7 **25.3 → 22.2**, which P19's budget probe had said was unaffected:
   that probe only lifted the op-count budget. Rows 3/5/6/16/17 were at parity
   already and are not re-measured; they are the ones to watch for a cadence
   shift, since `cost` also sizes the loop flush period.

9. **The static output-copy rule ignored donation** — **CLOSED 2026-08-13
   (P20)**, worth **1.50×** on row 18. An output that may alias an argument was
   copied even when that argument was DONATED, which is the one case where
   aliasing is licensed (`engine.py::_dealias` has always exempted it). A LoRA
   training step donates 2,255 of its 2,262 arguments and threads the frozen
   parameters straight through: 1,952 copies, ~10 GB per step. Now `0 output
   copies`, 656.3 → **397.5 ms/step** and peak 55 → 37 GB. Donation is
   retractable per call, so the exempted outputs carry the arguments they alias
   and `RunOnce` copies the ones a call takes back through
   `non_donatable_input_indices`.
7. ~~**Pack building has no memory discipline**~~ — **CLOSED 2026-08-13 (P19)**.
   Both are ported; row 7 completes at its historical 45 GB budget (35 GB peak,
   25.3 ms/tok = 1.16× of Stage 1) and row 13's steady state drops 4.2 → 3.2 GB
   with 518 of its 777 pack builds eliminated. The ablation above says the
   **cache** was the load-bearing half. Row 13's 46 GB peak survives and is now
   attributed: it is the keras streaming load transient, which Stage 1 shares
   (44 GB), not the packs (6.6 GB).
8. **Row 5's greedy tokens now diverge from Stage 1 at token 61 of 64**
   (they agreed before the sdpa emit) — the footnote-21 tie-flip class, but it
   is a *new* divergence introduced by fusion and should be walked down the
   logit-delta ladder rather than assumed benign.

---

## Stage-1-vs-anchor regressions — RESOLVED 2026-08-13 (P20)

*(Raw data: `~/.cache/metaljax-bench/logs/p20-regressions/`.)*

| row | anchor | today | ratio | verdict |
|---|---:|---:|---:|---|
| 14, maxtext qwix-int8 0.6B | 32.5 | **32.9 / 32.7** | **1.01×** | **NOT A REGRESSION.** Re-measured standalone twice under the lock: the row is on its anchor. The 60.1 reading was taken inside the P16 sequential campaign |
| 19, maxtext train 0.6B | 440 | 969.1 | **2.20×** | **ROOT-CAUSED**: the eager flush's cache clear (`METALJAX_FLUSH_CLEAR_MB`, landed 4d34bff *after* the anchor was measured). Fires 7×/step here at ~77 ms each |

**Row 19, the bisect.** The anchor and the regression are both in the record and
one commit range apart: `final_run.jsonl.maxtext` of 2026-08-03 01:56 reads
**440.0** and its 2026-08-05 14:31 successor **964.2**, same losses to the last
digit (228.3945 → 87.0428), CPU 1402 → 1373. Today, on this machine and this
tree, the 0.11.2 `src/metaljax` re-run through the unchanged Stage 1 dylib
(`PYTHONPATH` shadowing the editable install) gives **448.2** against the
current tree's **969.1** — so the variable is metaljax's own code, not the
harness (untouched since 0.11.3), not the maxtext venv or checkout (both
2026-08-02, verified by mtime), and not the machine.

Inside that range the knobs answer it directly, all four measured today:

| configuration | Stage 1 | native |
|---|---:|---:|
| shipped defaults | 969.1 | 1006.2 |
| `METALJAX_FLUSH_CLEAR_MB=1e6` (flush, never clear) | **478.8** | **468.0** |
| `METALJAX_EAGER_FLUSH_SYNC=0` (async flush, so no clear) | 529.1 | — |
| `METALJAX_EAGER_FLUSH_MB=0` (no flush at all) | **446.2** | — |
| `METALJAX_COMPILE_BYTES_MB=0` / `METALJAX_ENV_PRUNE=0` (controls) | 970.1 / 984.5 | — |

**The mechanism.** This program's `@main` is over the trace budget (`cost=24870
> 20000`, unchanged since 0.11.2), so it runs op by op — and its eager traffic
is ~105 GB per step against a live set of a few hundred MB. The byte-denominated
flush therefore fires **82 times per step**, and every hard flush that finds more
than `METALJAX_FLUSH_CLEAR_MB` (2048) in MLX's buffer cache returns the whole
pool to the OS: **7 clears per step**, after each of which the next ~2 GB of
allocations are cold Metal buffers. 969 → 479 is those seven clears (~70 ms
each); the remaining 479 → 446 is the blocking eval of the flushes themselves.

It is **shared by construction**: `plugin-native/runtime/program.cc`'s
`eager_flush` is the transliteration of `interpreter._eager_flush`, and both
read the same environment budgets — which is why the native column read "1.01×
of Stage 1" for two passes while both were 2.2× off the anchor.

**Not fixed here, and why.** The clear is a real memory bound, not decoration:
with it disabled the LoRA row (18) blew through a 70 GB guard at 81 GB, and the
Stage 1 LoRA run with the flush off was killed on trajectory at a projected
95 GB. The shape of a fix that keeps the bound without the cliff is
`mx::set_cache_limit(flush_clear_bytes)` — MLX then reclaims only the EXCESS, on
the next allocation, so the pool stays bounded *at every instant* (a tighter
guarantee than clearing at flush points) while reuse below the limit survives.
That is a change to the shared runtime's memory discipline; it wants Oleg's
sign-off, a memory ladder of its own, and — for Stage 1, whose copy of
`_eager_flush` is frozen — an explicit decision to reopen `src/`.

Interim, for anyone measuring an eager-main row: `METALJAX_FLUSH_CLEAR_MB` is
the knob, and the row's memory must be watched when it is lifted.

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
  **LANDED 2026-08-12 (P18)** as `plugin-native/metal/exported_symbols.exp` +
  the `linkopts`/`additional_linker_inputs` pair in `plugin-native/metal/BUILD`,
  the DEFAULT build, with `plugin-native/coexist_test.py` as its standing
  contract. Re-verified from scratch, not from the transcript:
  `notes/data/p18-relink-battery-2026-08-12.txt`.
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
| **P18 native cells + the Stage 1 controls** | `p18-relink/<bench-id>-<tag>-0812-<stamp>.{jsonl,log,flight.log}`, summarised in `notes/data/p18-relink-models-2026-08-12.jsonl` |
| **P18 relink evidence** (coexistence, size, perf neutrality, battery) | `p18-relink/{coexist-*,execute_test-relinked,texmo_gate-relinked,wheel_poc-relinked,perfneutral-*}.log`, summarised in `notes/data/p18-relink-battery-2026-08-12.txt` |

**The formerly UNVERIFIED relink validation is now verified**, from a rebuilt
dylib rather than a transcript, and it came out better than the transcript said:
`execute_test.py` **520 of 520** (the intermittent 8-thread
`There is no Stream(gpu, N) in current thread` row passed this run),
`texmo_gate.py` 106/106, `smoke_test.py`, `decline_census.py` 35/35,
`ingest_test.py` 8/8, `bazel test //...`, and the wheel run from a fresh 3.13
venv. Perf neutrality was re-measured A/B/A against a kept copy of the pristine
no-list dylib on the same five suite configs (relinked mean / pristine: 0.998,
0.984, 0.976, 0.984, 1.001 — every delta inside the relinked passes' own 2.4 %
spread). The wheel drops 42.2 MB → 11.8 MB with the dylib.

---

## P21 re-measurement (2026-08-14) — HALTED after 3 of 5 runs

*Raw data: `~/.cache/metaljax-bench/logs/p21-texmo-suites/`; per-config table
and aggregates: `notes/data/p21-texmo-suites-2026-08-14.{csv,json}`. Machine
lock held 14:19:32–14:56, strictly sequential, same protocol and same runners as
P16 (`scripts/bench_texmo_pjrt.py` at 64-step chunks for the native pairing,
`scripts/texmo_topconfs.py` at 256 for the anchor-comparable Stage 1 column).
The analysis reproduces every published P16 aggregate from P16's own artifacts
before being applied to the new ones (4.238 vs 4.24, 36.459 vs 36.46, the four
suite classes to three digits), so the two campaigns are computed identically.*

**Why it halted.** The tree carried an **uncommitted `msl_scan` port of the
native lowering**: `plugin-native/metal/metal_msl.{cc,h}` and
`metal_msl_emit.cc` untracked, `metal_lowering.cc` + `BUILD` modified by 506
lines over HEAD (4d1e403). Those sources were written at **14:45–14:50**, the
dylib was **re-linked at 14:55:05**, and a concurrent
`plugin-native/texmo_gate.py` GPU run started at **14:55:40** — all inside the
campaign window. The native column was therefore measuring a *moving,
uncommitted binary under concurrent load*, and runs 3 and 5 would not have
shared a binary; the campaign was stopped rather than pair a 14:46 dylib against
a 14:55 one. The Stage 1 stack is unaffected by any of it (frozen dylib, frozen
`src/`), which is why its columns below are reported as measurements and the
native column only as an observation.

### Stage 1 — measured, and stable

| comparison | n | geomean | reading |
|---|---:|---:|---|
| top_confs: 0.11.3 anchor / Stage 1 today (engine route, 256-step chunks) | 163 | **1.056** | today 5.6 % faster than the anchor (P16 read 1.046) |
| — by weight class (0–100 / 100–500 / 500–1500 / 1500+) | 47/61/29/26 | 1.029 / 1.066 / 1.070 / 1.068 | |
| jax-CPU control, anchor / today | 163 | 0.985 | machine 1.5 % slower — the same ambient P16 measured (0.986) |
| suite-106: Stage 1 P16 / Stage 1 today (PJRT route) | 106 | 0.996 | no drift |

Checks **163/163 ok, 0 FAIL, 0 error**. Configs beating jax-CPU: **54 of 163**
(anchor 53, P16 54). Best vs anchor `tc005-w12` 1.197, worst `tc145-w1888`
0.947 — the same worst row P16 found (−5.4 %), so it is a property, not noise.
Two suite-106 rows sit 11 % under their P16 reading (`mid14-b16l256` 0.887,
`big11-b32l128` 0.888) and nothing else is outside ±5 %. They were **not**
re-run standalone: by the time the halt was called the machine had been taken
by the concurrent gate work, and a standalone re-check is precisely what the
suite-context trap (CLAUDE.md item 12) demands — it is owed before either is
called a regression.

### suite-106 native — PRELIMINARY (WIP binary), and the gap is essentially gone

Same day, same route, same chunk length, 106/106 on both stacks, 0 errors.
**Do not quote these as the native plugin's numbers**: they are the
work-in-progress msl port as it stood at 14:46.

| aggregate | n | native/Stage 1 (P16) | native/Stage 1 (today) |
|---|---:|---:|---:|
| whole suite, geomean | 106 | 4.24× | **1.027×** |
| whole suite, median | 106 | 3.37× | **1.000×** |
| `big` | 34 | 1.34× | 1.010× |
| `mid` | 30 | 3.22× | **0.987×** |
| `db` (msl territory) | 40 | 14.61× | **1.075×** |
| `synth` | 2 | 1.52× | 0.983× |
| rows where native is faster | 106 | 18 | **54** |
| rows within 1.2× | 106 | 33 | **98** |
| rows at or above 10× | 106 | 32 | **0** |

Against P16's native column the geomean improvement is **4.11×** (median 3.41,
best `db02-b4l1024` **91.7×**: 72.7 → 0.793 ms/step, which is Stage 1's 0.793 to
three digits). Run wall time fell 995 s → 507 s for the same 106 configs.

**Ten native rows are SLOWER than P16's native.** Eight are rows where the C++
compile path had been at or *ahead* of Stage 1 and is not any more:
`big09-b8l256` **0.679** (26.07 → 38.42 ms — P16's flagship 0.68× win now reads
1.00× of Stage 1), `big14-b32l128` 0.726 (transformer d512, 0.82× → **1.12×**),
`big16-b8l256` 0.726 (0.93× → **1.24×**), `big16-b32l128` 0.811,
`big15-b8l256` 0.816, `big14-b8l256` 0.879, `big13-b8l256` 0.882,
`big12-b8l256` 0.929 (0.96× → 1.00×). The other two were at parity in P16 and
are now the suite's worst rows: `db00-b16l128` 0.579 (0.155 → 0.267 ms, 1.04× →
**1.85×**) and `db04-b128l128` 0.743 (1.00× → 1.35×), both sub-0.5 ms
dispatch-floor configs. The remaining worst native/Stage 1 rows today are
`db16-b256l512` 1.84× and `db17-b256l512` 1.66×. All of this belongs to whoever
owns the port; on a clean tree it is the first thing to re-check.

### Not measured

`topconfs-stage1-pjrt` (killed at 45 of 163) and `topconfs-native-pjrt` (never
started), so **there is no top_confs native/Stage 1 geomean for this date**. The
pairing needs an exclusive machine and a frozen native dylib — a copy under a
different name is enough, `jax_plugins/metal` identifies a plugin by its exports
for exactly this reason.

---

# THE RELEASE TABLE — P22, 2026-08-15

*The measurement the parity claim rests on. One machine-lock hold,
**22:41:34–23:33:37**, strictly sequential, nothing else on the machine, and
the native arms run a **frozen copy** of the dylib
(`~/.cache/metaljax-bench/frozen-release-208ca0d1.dylib`, sha256
`208ca0d1…558d61`; tree 915d7e3 + the P22 coop width cap) so no rebuild can
move the binary under the campaign. Raw data
`~/.cache/metaljax-bench/logs/p22-release-measure/`, per-config table and
aggregates `notes/data/p22-release-measure-2026-08-15.{csv,json}`, narrative
`notes/cpp-p22-release.md`. The analysis reproduces every published P16
aggregate from P16's own artifacts before being applied to the new ones
(4.2377 vs 4.24, 36.459 vs 36.46, the four suite classes to three digits).*

## Table 4 — top_confs (163): the pairing that had never completed

| aggregate | n | value |
|---|---:|---:|
| **native (PJRT route) / Stage 1 (engine route — the anchor's own)** | 163 | **0.998** geomean, 0.998 median |
| native (PJRT) / Stage 1 (PJRT — same route) | 163 | **1.001** geomean, 1.000 median |
| — by weight class (0–100 / 100–500 / 500–1500 / 1500+) | 47/61/29/26 | 0.999 / 0.998 / 1.008 / 1.003 |
| **route factor measured today** (Stage 1 engine / Stage 1 PJRT) | 163 | **1.002** (P16 measured 1.009) |
| Stage 1 (engine) vs the 0.11.3 anchor | 163 | **1.071× faster** |
| **native vs the 0.11.3 anchor** | 163 | **1.073× faster** |
| jax-CPU control, anchor / today | 163 | 0.990 (machine 1 % slower — ambient) |
| **configurations beating jax-CPU** | 163 | **native 59** · Stage 1 55 (engine) / 59 (PJRT) · anchor 53 |
| distribution | 163 | **163 within 1.2×, 0 above**; native faster on **101** |

Checks on the engine-route run: **163/163 ok, 0 FAIL, 0 error**. Worst rows
either way: `tc029-w47` **0.892** (native ahead), `tc000-w5` **1.079** (native
behind — 0.159 → 0.172 ms/step, the smallest configuration in the suite).

**P16's item 1 is closed.** `msl_scan` on the native lowering was a 36.46×
geomean and cost every one of the configurations where metaljax beats jax-CPU
(native won 0 of 163). It is now 0.998× and native wins **59**, which is more
than Stage 1 wins on the anchor's route. Both stacks are 7 % faster than the
0.11.3 anchor against a 0.990 CPU control, so that drift is shared and real.

## Table 5 — texmo 106-config suite

| aggregate | n | native/Stage 1 (P22) | P16 | P21 (WIP binary) |
|---|---:|---:|---:|---:|
| whole suite, geomean | 106 | **1.011** | 4.24× | 1.027 |
| whole suite, median | 106 | **1.000** | 3.37× | 1.000 |
| `big` (gru/lstm 512–1024, transformer) | 34 | 1.004 | 1.34× | 1.010 |
| `mid` | 30 | 0.998 | 3.22× | 0.987 |
| `db` (small recurrent — msl territory) | 40 | 1.030 | 14.61× | 1.075 |
| `synth` | 2 | 0.950 | 1.52× | 0.983 |
| rows within 1.2× | 106 | **103** | 33 | 98 |
| rows at or above 10× | 106 | **0** | 32 | 0 |
| rows where native is faster | 106 | **52** | 18 | 54 |

Stage 1's own column is stable on the same route: **1.007** vs P16, **1.010**
vs P21. Both stacks: 106 ok, 0 error.

## Every anomaly re-run standalone — and five of nine were the suite itself

Each row outside ±10 % was re-measured standalone (one process per arm, stacks
interleaved, frozen dylib) *before* being reported:

| config | in-suite ratio | standalone S1 | standalone native | standalone ratio | verdict |
|---|---:|---:|---:|---:|---|
| `db16-b256l512` | 1.777 | 4.472 | 7.935 | **1.774** | REAL |
| `db17-b256l512` | 1.599 | 7.281 | 11.618 | **1.596** | REAL |
| `db11-b256l512` | 1.313 | 2.180 | 2.859 | **1.311** | REAL |
| `big09-b8l256` | 0.666 | 38.346 | 25.378 | **0.662** | REAL (the P22 width cap) |
| `big14-b32l128` | 1.198 | 17.463 | 17.456 | **1.000** | in-suite artifact |
| `big12-b8l256` | 1.159 | 5.191 | 5.180 | **0.998** | in-suite artifact |
| `big07-b8l256` | 1.132 | 33.021 | 33.139 | **1.004** | in-suite artifact |
| `big00-b32l128` | 0.807 | 10.248 | 10.189 | **0.994** | in-suite artifact |
| `mid11-b64l128` | 0.884 | 8.527 | 8.561 | **1.004** | in-suite artifact |

The five artifacts corroborate independently: they are exactly the rows whose
*Stage 1* column moved against P16 (`big00-b32l128` +20 %, `mid11-b64l128`
+15 %, `big14-b32l128` −18 %) while their standalone numbers sit on their P16
values — and they cut both ways, two of them flattering native. Substituting
the nine standalone numbers moves the suite geomean 1.011 → **1.010** (median
0.9999, native faster on 53 of 106).

## The one qualifier on the parity claim

**Three `db*-b256l512` rows are genuinely slower natively** — `db16` 1.77×,
`db17` 1.60×, `db11` 1.31× — reproducible standalone on the frozen binary, and
P21's preliminary column saw the same two rows (1.84×, 1.66×). Ruled out, each
measured: it is not the surrounding graph (`METALJAX_MSL=0` puts the stacks
level — `db16` 83.99 vs 83.82, `db11` 60.21 vs 59.02, so the entire gap is on
the msl path), not the plans (identical narration: `db16` two coop plans
`trip=512 lanes=8192 stacked=1/8`, `db11` `lanes=4096 stacked=1/7`, same
kernel names hence the same MLX library), not the flush cadence
(`EAGER_FLUSH_MB=1e6`: 7.94 → 7.18 native, Stage 1 unmoved) and not the
compile budget (`TRACE_BUDGET=1e5`: neither moves). **Identical kernels,
dispatched differently** — the per-call launch work in `runtime/msl.cc` (the
weight-normalization recipe and the input pooling) is where to look. The
pocket is the largest `db` shapes only: `db11-b64l256`, the same spec smaller,
is at exact parity (0.633 vs 0.633).

Three rows of 106, none in `top_confs`, against a suite geomean of 1.011 and a
median of 1.000 — it qualifies the claim, it does not overturn it.

> **RESOLVED 2026-08-16 (P23, `notes/cpp-p23-dispatch.md`) — and it was not
> the launch.** Nothing in `runtime/msl.cc` changed. `BlockBytes` (the byte
> estimate every compile decision reads) was ported without
> `_block_bytes`' msl case, so a loop that became one generated kernel was
> charged `trip × body` instead of its outputs: `db16-b256l512`'s step
> estimate came out at **163 GB instead of 2.05 GB**, over
> `METALJAX_COMPILE_BYTES_MB`, which took away the loop body's compile AND
> the chunked replay and left every training step to be dispatched op by op
> — with 128 blocking eager flushes per chunk. One line
> (`if (MslPlanFor(ctx, &o) != nullptr) continue;`) puts the native estimate
> on Stage 1's number exactly (134,234.4 MB, digit for digit) and the three
> rows at **0.998 / 0.999 / 0.985**. The census is unchanged, plan for plan.

## Not measured in this campaign

The **model rows** (Table 3) were not re-run: nothing in P22 touches their
paths (the coop width cap is a texmo-recurrent-cell decision, and no model row
builds an msl plan), and they were last measured at P20. The Stage-1-vs-anchor
regression on row 19 (the shared eager-flush cache clear) is unchanged and
still awaiting Oleg's call on `mx::set_cache_limit`.

---

# THE RC TABLE — P23, 2026-08-16

*P22's one qualifier, closed. Machine lock held for every run, one process per
arm, `scripts/bench_texmo_pjrt.py` (PJRT route, 64-step chunks) on both stacks,
native arms on the frozen release candidate
`~/.cache/metaljax-bench/frozen-rc-ed355691.dylib` (sha256 `ed355691…94a16`;
tree d70499b + the P23 byte-gate fix, `plugin-native/metal/metal_lowering.cc`).
Raw data `~/.cache/metaljax-bench/logs/p23-dispatch/`, aggregates
`notes/data/p23-dispatch-2026-08-16.{csv,json}`, narrative
`notes/cpp-p23-dispatch.md`. The analysis reproduces P22's published
aggregates from P22's own artifacts before it is applied to the new ones
(1.0111 whole, 1.0043 / 0.9979 / 1.0301 / 0.950 by class, top_confs 1.0007).*

**What changed in the binary**: `BlockBytes` was charging a loop that became one
generated msl kernel `trip × body` instead of its outputs (`_block_bytes`' msl
case, present in `BlockCost` since P21 and missing here). On the biggest `db`
shapes that put the per-step estimate at 163 GB instead of 2.05 GB — over
`METALJAX_COMPILE_BYTES_MB` — so the loop body's compile and the chunked replay
were both refused and each training step was dispatched op by op, with 128
blocking eager flushes per chunk. Nothing in `runtime/msl.cc` changed; the
plans and the kernels always were identical.

## Table 6 — the three qualifier rows (standalone, arms interleaved)

| config | Stage 1 | P22 native | **RC native** | P22 ratio | **RC ratio** |
|---|---:|---:|---:|---:|---:|
| `db16-b256l512` | 4.475 | 7.942 | **4.466** | 1.774 | **0.998** |
| `db17-b256l512` | 7.287 | 11.614 | **7.280** | 1.594 | **0.999** |
| `db11-b256l512` | 2.180 | 2.850 | **2.147** | 1.307 | **0.985** |
| `db11-b64l256` (control) | 0.634 | 0.634 | 0.634 | 1.000 | 1.001 |
| `db02-b4l1024` (control) | 0.782 | 0.798 | 0.793 | 1.021 | 1.015 |

Where the 3.45 ms/step went, all four arms the same program (the first three on
P22's *released* binary, one environment variable apart): shipped **7.923** →
`EAGER_FLUSH_MB=0` **7.185** (the flushes: 0.74 ms) → `COMPILE_BYTES_MB=1048576`
**4.469** (the gate raised past the inflated estimate — the fix, by knob) →
RC binary **4.464**, Stage 1 **4.471**. The remaining 2.71 ms is op-by-op
dispatch of ~611 ops per step (≈4.4 µs each) instead of one compiled replay
per 16 steps.

## Table 7 — texmo 106-config suite

| aggregate | n | **P23 (RC)** | P22 | P16 |
|---|---:|---:|---:|---:|
| whole suite, geomean | 106 | **1.0050** | 1.011 | 4.24× |
| whole suite, median | 106 | **1.0012** | 1.000 | 3.37× |
| `big` | 34 | 1.0107 | 1.004 | 1.34× |
| `mid` | 30 | 1.0033 | 0.998 | 3.22× |
| **`db` (msl territory)** | 40 | **1.0013** | 1.030 | 14.61× |
| `synth` | 2 | 1.0062 | 0.950 | 1.52× |
| **rows within 1.2×** | 106 | **106** | 103 | 33 |
| rows at or above 10× | 106 | **0** | 0 | 32 |
| rows where native is faster | 106 | 42 | 52 | 18 |

Every row outside ±10 % was re-measured standalone before being reported, and
**six of seven were the suite itself** (`big11-b32l128` 1.178 → 1.000,
`big14-b32l128` 1.143 → 0.998, `big14-b8l256` 1.105 → 0.998, `big16-b32l128`
1.105 → 1.000, `big12-b32l128` 0.874 → 1.001, `big09-b32l128` 0.873 → 0.963);
the seventh is `big09-b8l256` 0.660, P22's width-cap win. Substituting them:
geomean **1.0024**, median 1.0009, native faster on 43.

## Table 8 — top_confs (163), same PJRT route

| aggregate | n | **P23 (RC)** | P22 |
|---|---:|---:|---:|
| native / Stage 1 (same route) | 163 | **1.0016** geomean, 1.001 median | 1.001 |
| rows within 1.2× | 163 | **163** | 163 |
| rows where native is faster | 163 | 63 | — |
| configurations beating jax-CPU | 163 | native **58**, Stage 1 59 | native 59 |
| **native arm vs P22's native arm** | 163 | **0.9999** | — |
| Stage 1 arm vs P22's Stage 1 arm | 163 | 1.0008 | — |

`top_confs` does not move, and should not: no configuration in it is large
enough for the overcharge to cross the gate. One row outside ±10 %
(`tc009-w16` 0.888) and it is in native's favour.

## The measurement lesson: a gate run poisons the suite that follows it

The suite pair was measured twice. The first pair ran straight after a 263 s
`texmo_gate` in the same hold and came back at geomean **1.047 with 21 rows
outside ±10 %** — all `l128` (large-batch) rows, **7 of the 11 worst building
no msl plan at all**, i.e. on tapes byte-identical to P22's. The Stage 1 arm of
that hold was inflated too (0.9725 against P22, worst row 0.72), just less,
because it ran second. Re-run in a hold of its own with a settle and nothing
before it, every one of those rows returns to its P22 value (`big12-b32l128`
16.31 → 9.93 vs 9.74; `big16-b32l128` 88.28 → 67.09 vs 66.76). Both pairs are
kept; the clean pair is this table. **Future release measurements: the suite
first, the gate afterwards.**

## Battery (frozen RC binary)

`execute_test` **536 of 536** (P22's 535 plus one new contract — `msl loop
charged as one kernel`, 3.1 MB planned vs 290.1 MB interpreted; the log diff
against P22's is exactly that row plus the plugin path), `texmo_gate`
**106 ok / 0 decline / 0 FAIL / 0 error** twice, `smoke_test` passed,
`bazel test //…` passed, and the whole-suite plan census **identical to P22's,
568 narration lines in the same order**.

---

# P25 ADDENDUM — the eager flush stops dumping the pool (2026-08-16)

*Full write-up: `notes/cpp-p25-cache-limit.md`. Data:
`~/.cache/metaljax-bench/logs/p25-cache-limit/`. Binary under measurement:
`frozen-p25c.dylib`, sha256 `516e4b43…`, byte-identical to a `bazel build` of
the tree.*

P20 root-caused row 19's 2.2× to `mx::clear_cache()` at the eager flush and
proposed `mx::set_cache_limit`. Both shapes were built and measured. **The
literal one — a global cache limit at plugin init — is rejected**: it bounds
paths that never reach a flush and were never bounded, and a compiled decode
step whose transients exceed the bound then re-allocates from the OS every step
(row 13: **190.0 vs 80.7 ms/tok**, 2.35×; suite-106 geomean 1.0420 vs P23's
1.0050, `mid` class +11.8%; and one run of that binary died on a GPU address
fault). **What ships is the trim at the same cadence the clear had**
(`runtime.cc::trim_cache`): set the limit, poke the allocator, restore.

| row / gate | shipped dump (RC, same day) | **P25 trim** | note |
|---|---:|---:|---|
| 19, maxtext train, ms/step | 975.4 | **833.9** | 1.17×; peaks 21 / 20 GB |
| 18, LoRA train, ms/step | 400.0 | **394.0** | peak is a LOAD transient, 37-56 GB on both binaries |
| 13, E2B keras-int4, ms/tok | 80.7 | **80.8 / 80.9** | parity — the point of not bounding globally |
| texmo suite-106 native/Stage 1 | 1.0050 ᴾ²³ | **0.9685** | 106/106 within 1.2×, 81 native-faster; `big` class 1.0107 → 0.9296 |
| …native arm vs P23's native arm | — | **0.9882** | against a Stage-1 control reading 1.0254 of P23's (the machine is ~2.5% slower today) |

**The watermark is the rest of row 19**, and it is a memory trade, measured on
the two rows that care (`METALJAX_FLUSH_CLEAR_MB`, ms/step and peak footprint):

| watermark | row 19 | row 19 peak | row 18 | row 18 peak |
|---:|---:|---:|---:|---:|
| 512 | 1067.0 | 20 GB | — | — |
| **2048 (shipped)** | **833.9** | 21 GB | **395.6** | **39 GB** |
| 8192 | 685.6 | 25 GB | 364.5 | 57 GB |
| 32768 | 464.1 | 39 GB | — | **guard-killed, 68 GB** |
| unbounded | 461.7 | 39 GB | — | (P20: 81 GB) |

Battery on the shipped binary: `execute_test` all cases match CPU (plus four
new P25 contracts — the pool holds at 255 MB over 552 flushes with a 256 MB
watermark, median 228 MB cached, 4025 MB with the trim off, and the 20k-
iteration loop still clears on its own count cadence), `ingest_test` 0 failed,
`smoke_test`, `bazel test //…`, and `texmo_gate` 105 ok / 1 FAIL —
`mid03-b64l128`, P23's documented flake for that config, 3/3 ok standalone on
this binary and 3/3 on the RC one.
