# 0.11.5 — Full Release Status (for Oleg's review, pre-TestPyPI)

Composed 2026-08-18. Tree at `9822054` + two uncommitted work sets (vendoring
wiring, benefit-gate) whose batteries are IN FLIGHT. Nothing uploaded anywhere.
Companion documents: `notes/release-gates-0.11.5.md` (the gate record),
`notes/mlx-patch-diagnosis.md`, `notes/no-panic-governor.md`,
`benchmarks/perf-2026-08-native-baseline.md`, `benchmarks/models.md`.

## What this release is

The completed C++ migration plus everything it forced us to learn:

1. **The fully native PJRT plugin** (`plugin-native/`): bazel-built against
   xla at the jax pin, PjRtClient behind the official C API wrapper, the
   whole StableHLO→tape lowering, recognizers (qmm/moe/sdpa), msl_scan
   codegen, compile decisions, Accelerate LAPACK, callbacks, donation,
   collectives, emulated dtypes — at 99.54 % of the pinned jax suite and at
   or above Stage-1 performance on both texmo suites and the model table.
2. **The no-panic contract** (after panic #9): a memory governor on leading
   indicators, page-cache discipline for checkpoint ingest, degrade →
   clean RESOURCE_EXHAUSTED. Rows 8/9/10 run for the first time ever; a
   192 GB load refuses cleanly ×3; the exact conditions of panics #7/#8/#9
   were deliberately recreated and survived.
3. **The owned MLX build**: our fork at v0.32.0 + the command-buffer fence
   fix (upstream's own unreleased 7e8b4ccc, cherry-picked) + our
   `end_encoding` hardening, vendored as `libmlx_metaljax.dylib`. Fixes the
   corruption class behind row 15, the canary lottery, and the historical
   command-buffer bands. (Battery in flight.)

## Gate table — what is BANKED vs IN FLIGHT

Rule 1 note: the vendored MLX and the benefit-gate change the binary, so the
final re-gate must re-attest on the one final build. "Banked" below = passed
on `ebe56e71` (pre-vendoring) and needs either re-run or a
provably-cannot-move argument; the plan is re-run for anything numeric.

| Gate | State | Detail |
|---|---|---|
| 1 Freeze | RE-FREEZE PENDING | final binary = vendored MLX + benefit-gate |
| 2 Pinned jax suite | banked 99.54 % zero-new (id-identical set) | **MUST RE-RUN on final binary** — the MLX substitution is numerics-relevant by design |
| 3 tests/ both legs | banked 1258/0 + 1187/71 | re-run native leg on final binary |
| 4 texmo | banked: suite 0.984, top_confs 0.997, 1.072× anchor | pairings re-run in the in-flight batteries |
| 5 Models | banked incl. governor rows | rows 11/14/19 (benefit-gate) + row 15 native (vendored) in flight; second-finisher re-spots on combined build |
| 5b No-panic contract | **PASS** (attested) | design unchanged by in-flight work; contract tests re-run in batteries |
| 6 Contract suites | banked | re-run in batteries |
| 7 Wheels | rebuild pending | now bundle vendored MLX; no-pip-mlx venv is a new capability under test |
| 8 Finale | pending | flips when 5's two lines close |

## The 20-row model table (current best, provenance-marked)

| Row | Status | Number (vs Stage 1 / anchor) |
|---|---|---|
| 1 gemma4-31B | ✅ | 239.8 — 0.989× S1 (callee-sdpa fix), tokens identical |
| 2 gemma4-12B | ✅ | 93.9 — 1.001× |
| 3 26B-A4B MoE | ✅ | 43.4 — 0.99× |
| 4 E2B | ✅ | 27.0 — 1.000× |
| 5 Qwen3-8B | ✅ | 58.1 — ~1.00×; nondeterminism watch (may die with fence fix — in flight) |
| 6 Llama-8B | ✅ | 54.7 — 0.99× |
| 7 gpt-oss MXFP4 | ✅ | 22.2 — 1.00× anchor; same watch as row 5 |
| 8 Qwen3.6-35B | ✅ **first ever** | 29.6 (torch 13.7 — perf-era target) |
| 9 R1-Distill-32B | ✅ under contract | 210.7–214.4 — 1.00× S1 |
| 10 DeepSeek-V2-Lite | ✅ **first ever** | completes, 88 GB peak (was killed @122); ms/tok in flight |
| 11 maxtext decode | benefit-gate in flight | 16.6–16.9 class; peak must return to ~7.6 GB |
| 12 Mixtral | blocked externally | 93 GB KaggleHub download; checkpoint passes transfer path |
| 13 E2B int4 | ✅ | 79.7–80.3 — 0.98× anchor |
| 14 qwix-int8 | benefit-gate in flight | 32.0–32.9 ≈ anchor; peak must return to ~15 GB |
| 15 qwix-int8 8B | **fix in flight** | patched-MLX Stage-1: 10/10 deterministic, coherent. Native attestation running. Was: never worked anywhere |
| 16 SigLIP | ✅ | 87.9 / 2350 — better than P18 |
| 17 SD3.5 | ✅ | 5782 @1024 (1.13×), 1235 @512 (**0.81× — native wins**) |
| 18 LoRA | ✅ | 394–400 — 0.98× anchor, peak 37–39 GB |
| 19 maxtext train | benefit-gate in flight | 456–470 = 1.06× anchor must hold |
| 20 235B-3bit | clean decline | needs packed sub-byte (feature, post-release) |

## Decisions on record (rule 2)

- **No-panic contract + amendment** — CLAUDE.md ground rules, verbatim.
- **Row 15 was release-blocking by your ruling; the mechanism proved to be
  MLX's fence drop** (slicing.cpp:62), fixed in our vendored build. Ships as
  FIXED if the native attestation passes; otherwise blocking again.
- **Watermark regression → option (b)** benefit-gate, in flight.
- **Rows 5/7 nondeterminism**: accepted as release-note item *if it
  survives the fence fix* — measurement in flight; may be retired.
- **`__version__` reports 0.11.3** inside wheels: accepted cosmetic; dies
  with the post-release `src/` retirement.
- **129-row jax-suite whitelist**: reviewed item-by-item (parity report);
  includes the 2 sparse rows = MLX fusion bug #8 (fork fix-branch
  candidate #2, needs an MLX-level scatter repro).

## For you, sequenced

1. **Now / anytime**: push the fork branches (`fix/command-buffer-split`,
   `fix/temporary-fence-tracking`, `vendor/0.32.0`, `diag/split-instrumentation`)
   to `eterevsky/mlx`; upstream actions with drafts ready in
   `notes/patches/`: the release request on ml-explore/mlx#4099 (their
   unreleased fix, our numbers + 20-line repro) and our hardening PR.
2. **After the two batteries + my consolidated re-gate**: review the flipped
   gate document (expect: zero open REGRESSION lines if rows 11/14/19 land
   and row 15 attests) → I upload to TestPyPI → verify from fresh venv →
   you publish public + push + tag.
3. **Post-release queue** (recorded in the ledger): Python-path retirement +
   docs cleanup (your call, first item), framework-gap fix list (dot_general
   middle-axis ~42 ms, KV in-place ~25, attn_vec ~23, mx.fast norms ~12 —
   the road from 1.82× to mlx-lm), fusion-#8 fork branch, row 12 download,
   row 20 sub-byte, row 8 perf (29.6 vs torch 13.7).

## Honest-open list (ships documented, not hidden)

- Row 15's jax-CPU reference cell was never re-measured (memory-blocked);
  its verdict rests on determinism + row-14 agreement + patched coherence.
- The canary's 4–7 % residual is attributed to bf16-vs-f32 reference depth,
  not corruption (budget-independent) — one independent re-check queued.
- Sub-ms dispatch floor on the tiniest texmo configs (documented since P22).
- Stage 1 (the default wheel) keeps every pre-fork behavior, including the
  watermark-less flush and the unpatched MLX — it is the frozen legacy
  artifact and ships as-is, one release before retirement.
