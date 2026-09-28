# metaljax 0.11.9 — RELEASE GATE REPORT

Gated 2026-09-27/28 on jax 0.11.2.  0.11.9 is 0.11.8 plus two memory fixes from a texmo bug report
(a search worker on a 32 GB M4 died with RESOURCE_EXHAUSTED on a 10k-weight recurrent LM's training
chunk): the loop in-flight window (b583c2c) and the governor's hard-line reclaim (a946942).

## 0. Provenance (release rule 1)

- Tree: main `a946942` (clean at launch; untracked `scratch/` holds the bug report).
- Release binary: `~/.cache/metaljax-bench/frozen-0119main-c8577cab.dylib`, sha256 `c8577caba62378ec…`,
  byte-identical to main's `plugin-native/bazel-bin` build.  Plugin XLA `91888df6` (jax 0.11.2's), vendored
  MLX fork `vendor/0.32.0 @ 661b2e38`, both unchanged since 0.11.8.
- jax/jaxlib 0.11.2 in main's `.venv` and the bench/gemma venvs; the maxtext venv stays on jax 0.11.0 (no
  released flax imports on 0.11.2): rows 10, 11's maxtext arm, 14, 15, 19.
- Every stage pinned `METALJAX_PLUGIN_PATH` and asserted the sha.  Drivers: `logs/gate-0.11.9/chain.sh`
  (`scripts/release/run_gates.sh` with GATE_DATE=2026-09-27, stage C3, stage C9), then `row7ab.sh`,
  `row7battery.sh`, and the row-20 attempts (`row20quiet.sh`, `row20post.sh`).

## Stage V — the combined tree → **PASS**

On these exact bytes (the governor fix's review, `logs/governor-reclaim/suites/fix2-*`): `execute_test.py`
(including the new P54 loop-window and P55 hard-line contracts) and `ingest_test.py` pass; `pytest tests/`
**559 passed + 1 xfailed**; `texmo_gate.py` 106/106.

## The fixes, validated on the reported workload

- Attached train chunk (`logs/metal-mem/kit/`): MLX peak active 5.16 → 0.76 GB, footprint 5.50 → 1.09 GB,
  0.83 → 0.69 s per 256-step call.  Governor ceiling at this machine's baseline + 1.5 / 3 / 4.5 GB with the
  window off (`METALJAX_LOOP_INFLIGHT_MB=0`, the old spike): 0.11.8 refuses at all three; 0.11.9 completes
  every call (20-29 hard-line events per ceiling, each settled and cleared, none refused).
- texmo's own multi-model worker loop (`scratch/metal_mem/probe.py`, 4 models x 2 rounds, jax 0.11.0 as on
  the M4; `logs/metal-mem/probe/`): peak footprint 8.82 → 5.19 GB.  With the governor ceiling at baseline
  + 6 GB (the crowded-M4 emulation) 0.11.8 dies with the reporter's exact error in round 2; 0.11.9 trains all
  eight models (peak 4.91 GB) without the governor ever reaching its hard line.

## Stage A — pinned jax test suite (jax-v0.11.2) → **PASS**

163 files, 24.0 min: **28,679 passed / 135 failed** / 6,120 skipped → 99.53 %, id-identical to the whitelist
`notes/data/pinned-0.11.2-failures.txt` (0 new, 0 fixed).

## Stage B — texmo release anchors → **PASS**

- Correctness: **106/106** (19 via sensitivity scaling), 0 decline, 0 FAIL.
- Perf sweep (223 configs, `notes/data/texmo-topconfs-2026-09-27.jsonl`): **geomean 1.139× faster** than the
  standing anchor (0.11.8: 1.115×); 134 configs faster by > 5 %.  Config for config against the 0.11.8 gate
  run: **1.022× faster**, 21 configs faster by > 5 %, one slower by > 5 %: tc027-w47 (`bits.1+bp|rnn.4.gelu-
  norm-suffix.2-dense.1.tanh` b1 l4096), 3.26 → 3.75 ms — the 0.11.8 run's reading was the outlier: standalone
  on both binaries, 3 alternating draws each, 0.11.8 3.75 / 3.76 / 3.74, 0.11.9 3.74 / 3.73 / 3.73.

## Stage C1 — the main model battery → **PASS (with the row-7 attribution below)**

`final_run.sh`, 22.2 min, on 2026-09-27 22:36 — with a desktop browser holding ~7 GB of the machine (found
afterwards; closed for the later stages).  Metaljax against the 0.11.8 release column, jax-CPU on jax 0.11.2:

| row | benchmark | 0.11.8 | **0.11.9** | Δ | jax-CPU (0.11.2) |
|---|---|---:|---:|---:|---:|
| 1 | gemma4-31B | 123.1 | **123.2** | +0.1 % | ✗ |
| 2 | gemma4-12B | 56.2 | **58.1** | +3.4 % | 312.1 |
| 3 | gemma4-26B-A4B | 31.1 | **31.2** | +0.3 % | ✗ |
| 4 | gemma4-E2B | 16.9 | **16.9** | 0 | 67.2 |
| 5 | Qwen3-8B | 36.0 | **36.2** | +0.6 % | 204.2 |
| 6 | Llama-3.1-8B | 38.6 | **39.2** | +1.6 % | 207.9 |
| 7 | gpt-oss-20b | 13.2 | **13.2** ˢ | 0 | ✗ |
| 11 | Qwen3-0.6B keras-hub | 5.1 | **5.1** | 0 | 28.6 |
| 13 | gemma4-E2B keras-int4 | 6.0 | **6.0** | 0 | 67.3 |
| 14 | Qwen3-0.6B qwix-int8 | 26.32 | **26.33** | 0 | 143.2 |
| 16 | SigLIP 2 fwd b1 | 41.77 | **41.74** | −0.1 % | 363.6 |
| 18 | LoRA gemma4-E2B | 126.1 | **115.1** | −8.7 % | 1441.7 |
| 19 | Qwen3-0.6B maxtext train | 362.8 | **362.6** | −0.1 % | 1397.7 |
| 11 (maxtext arm) | Qwen3-0.6B | 8.06 | 8.05 | | 89.9 |

- **Row 7**: the battery read **16.6** (+25.8 %, prefill 146.7 vs 127.1), flagged REGRESSED by the gate
  tooling.  It is not the binary: 13.2 / 13.2 standalone in stage C3; an interleaved A/B read 13.2 / 13.1 on
  the 0.11.8 binary and 13.2 / 13.2 on 0.11.9; three re-runs through the battery's own invocation on the
  quiet machine (the browser closed) read 13.3 / 13.2 (0.11.9) and 13.2 (0.11.8); METALJAX_DEBUG shows no
  loop window engaging in any of gpt-oss's programs; every stream identical to 0.11.8's.  The cell is the
  standalone 13.2 (ˢ), the battery reading stated here (release rule 2: Oleg asked for the row-7 re-run,
  reviewed its result and continued the release, 2026-09-28).
- Token agreement vs jax-CPU: rows 2, 5, 6, 11, 13 AGREE 64/64; row 4 diverges at 51/64 (the certified-benign
  entry).  Every metaljax stream is identical to the 0.11.8 records.

## Stage C3 — the remaining model rows → **PASS**

| row | benchmark | 0.11.8 | **0.11.9** | note |
|---|---|---:|---:|---|
| 15t | qwix-int8 8B | 265.3 | **265.5** | 15d forensics clean |
| 9 | R1-Distill-32B | 199.9 | **198.3** | |
| 8 | Qwen3.6-35B-A3B | 23.4 | **23.5** | |
| 17a | SD3.5 512² | 460.6 | **457.2** | |
| 17b | SD3.5 1024² | 2090.4 | **2102.8** | standalone 2086.0 |
| 12 | Mixtral 8×7B | 69.0 ˢ | **70.5** ˢ | in sequence: refused cleanly — 91.6 GB live (its usual set) + ~19 GB of machine background (the browser) over its approved 110 GB envelope; standalone after the browser closed: 70.5 |
| 10 | DeepSeek-V2-Lite (manifest) | 19.6 | **20.3** | presplit variant 19.1 |
| 21 | Qwen3.8-27B bf16 | 142.3 | **142.6** | |

Guard fires during decode: 0.  Panics: 0.

## Standalone re-draws (rerun-first rule)

| row | in sequence | standalone | verdict |
|---|---:|---:|---|
| 3 gemma4-26B-A4B | 31.2 | 28.2 / 28.2 | suite context, as in 0.11.7/0.11.8; the battery cell stands |
| 6 Llama-3.1-8B | 39.2 | 34.5 / 34.6 | suite context; the battery cell stands |
| 18 LoRA gemma4-E2B | 115.1 | 113.8 / 113.6 | suite context; the battery cell stands |
| 7 gpt-oss-20b | 16.6 | 13.1-13.3 (x7 on 0.11.9, x3 on 0.11.8) | one-off battery reading; the standalone cell (ˢ) |
| 12 Mixtral | refused (browser) | 70.5 | environmental refusal; the standalone cell (ˢ) |
| 17b SD3.5 1024² | 2102.8 | 2086.0 | no spread; the in-sequence cell stands |

## Row 20 — **EXCLUDED from the suite** (Oleg, 2026-09-28)

Row 20 (Qwen3-235B-A22B 3-bit) needs ~104 GB of its own under an approved 114 GB machine ceiling, leaving
~10 GB for the rest of the machine; its earlier cells started with the machine at 8.7-9.2 GB claimed.  This
gate could not provide that: the in-chain cell started at 12.6 GB (the browser) and ran 77 min at the ceiling
— 622 governor stall-clears, qmm builds refused into their literal fallback, no crash (0.11.8 had refused the
same situation outright) — and was stopped as not a timing; after closing everything but the Claude desktop
app the machine idled at ~14 GB, and after a reboot at 11.5-12.6 GB, never under 10.  A run at a raised 119 GB
ceiling was approved by Oleg but blocked by the session's permission policy.  The row leaves the suite;
its cells through 0.11.8 remain as history.

## Consolidated disclosure list

1. **Loop in-flight window** (b583c2c, METALJAX_LOOP_INFLIGHT_MB=1024): counted loops bound the bytes of
   submitted-but-unfinished iterations by measurement and wait on the oldest; 0 restores the 0.11.8 cadence.
   First execute of a program that traces new msl kernels is 0.1-0.2 s slower (its submissions are
   synchronous until the kernels are proven, sized from the static estimate).  Details and limits:
   `notes/loop-inflight-window-2026-09.md`.
2. **Governor hard-line reclaim** (a946942): past a hard line the governor settles the device and clears
   MLX's cache before the stall and on every 10 ms tick (Python's gc once a second); refusals split the
   footprint into live arrays / cache / other.  The METALJAX_CONCURRENT_EXECUTE=1 path (poll-only settle)
   is unexercised.
3. **Row 7**: one battery reading of 16.6 against ten 13.1-13.3 re-runs on both binaries (above).
4. **Row 12**: refused cleanly in sequence while a browser held ~7 GB; the standalone cell.
5. **Row 20 excluded** from the suite (above).
6. **Suite-context spreads** named in the tables: rows 3, 6, 18 read higher in the battery than alone.
7. Carried from 0.11.8: the maxtext rows on jax 0.11.0; row 4's certified-benign tie; row 11's maxtext
   prefill; keras rows' timed generate includes a padded-window prefill; row 10's checkpoint load peaks
   99-111 GB vs its 105 GB guard.

## Verdict

**PASS** — release rule 2: no regression of the binary on any suite or benchmark against 0.11.8; the one
gate-tooling flag (row 7's battery reading) is attributed above and was reviewed by Oleg, and row 12's in-sequence
refusal is environmental.  Every number in the 0.11.9 release column comes from `frozen-0119main-c8577cab`
(release rule 1).  20 rows have a cell; row 20 is excluded.
