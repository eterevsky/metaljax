# 0.11.5 — Full Release Status (for Oleg's review, pre-TestPyPI)

Composed 2026-08-18; updated the same day as the two batteries (benefit gate,
vendoring) LANDED — both complete, results below. The consolidated re-gate on
the final binary is running; this document is superseded by the
plain-language final status once it lands. Nothing uploaded anywhere.
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
   command-buffer bands. (Battery complete — see the gate table.)

## Gate table — what is BANKED vs IN FLIGHT

Rule 1 note: the vendored MLX and the benefit-gate change the binary, so the
final re-gate must re-attest on the one final build. "Banked" below = passed
on `ebe56e71` (pre-vendoring) and needs either re-run or a
provably-cannot-move argument; the plan is re-run for anything numeric.

| Gate | State | Detail |
|---|---|---|
| 1 Freeze | **PASS** | final binary `frozen-vendor-d651add3` (vendored MLX + benefit-gate); rebuild reproduces the sha; wheel dylib byte-identical |
| 2 Pinned jax suite | banked 99.54 % zero-new (id-identical set) | **MUST RE-RUN on final binary** — the MLX substitution is numerics-relevant by design |
| 3 tests/ both legs | banked 1258/0 + 1187/71 | re-run native leg on final binary |
| 4 texmo | **re-run on the vendored binary** | suite-106 within run-to-run noise of the recorded native arms (0.9989 / 1.0026 geomean); `texmo_gate` **106/106**. Live Stage-1 pairing dropped per the 2026-08-18 scope change — the control is now the recorded native suite |
| 5 Models | banked incl. governor rows | **row 15 native: FIXED** (10/10 coherent, vendored); rows 11/14/19 re-measured on the vendored binary by the benefit-gate; **row 1 open — see the regression line below** |
| 5b No-panic contract | **PASS** (attested) | design unchanged by in-flight work; contract tests re-run in batteries |
| 6 Contract suites | **PASS on the vendored binary** | smoke / execute / ingest / decline / coexist ×2 / bazel test; `test_command_buffer.py` all 11 ran, 6 correctness PASS + 5 canaries correctly unable to find a corrupting budget |
| 7 Wheels | **BUILT, native-only** | 65 MB, 12 files, no Stage 1 module, carries the gated dylib byte-identically; installs and drives Metal on 3.12/3.13/3.14 **with no mlx in the venv**, and coexists with pip mlx |
| 8 Finale | pending | flips when the consolidated re-gate lands gates 2, 3 and 5 on the final binary and row 1 is dispositioned |

## The 20-row model table (current best, provenance-marked)

| Row | Status | Number (vs Stage 1 / anchor) |
|---|---|---|
| 1 gemma4-31B | ⚠️ cell not reproducing | vendoring-neutral: one source tree, two libraries, back to back — public 292.3 / **vendored 291.0** / public 285.7. But today's readings (256.8 and 275.6 standalone after a hard settle, 258.3 in-battery, up to 292.3 — drifting within one session on BOTH libraries) sit well above the published 237.3 cell. Not a fence-fix cost; an open question about that cell. The consolidated re-gate re-measures it on a settled machine |
| 2 gemma4-12B | ✅ | 93.9 — 1.001× |
| 3 26B-A4B MoE | ✅ | 43.4 — 0.99× |
| 4 E2B | ✅ | 27.0 — 1.000× |
| 5 Qwen3-8B | ✅ | 58.4–58.9 at its 58.5 cell; nondeterminism survives the fence fix (release-note item, see Decisions) |
| 6 Llama-8B | ✅ | 54.7 — 0.99× |
| 7 gpt-oss MXFP4 | ✅ | 22.2 — 1.00× anchor; same nondeterminism release-note item as row 5 |
| 8 Qwen3.6-35B | ✅ **first ever** | 29.6 (torch 13.7 — perf-era target) |
| 9 R1-Distill-32B | ✅ under contract | 210.7–214.4 — 1.00× S1 |
| 10 DeepSeek-V2-Lite | ✅ **first ever** | completes, 88 GB peak (was killed @122); ms/tok being measured by the consolidated re-gate |
| 11 maxtext decode | ✅ RESOLVED (benefit gate) | 16.60–16.83 ms/tok, 9 of 9 complete at the historical 20 GB budget; re-spotted on the vendored binary |
| 12 Mixtral | blocked externally | 93 GB KaggleHub download; checkpoint passes transfer path |
| 13 E2B int4 | ✅ | 79.7–80.3 — 0.98× anchor |
| 14 qwix-int8 | ✅ RESOLVED (benefit gate) | 31.82–32.13 ms/tok, 4 of 4 complete at the historical 25 GB budget (peak 9.2 GB); re-spotted on the vendored binary |
| 15 qwix-int8 8B | ✅ **FIXED** | native on the vendored MLX: first token 12095 **10/10**, 0 collapses, decode `" Paris. The capital"`, 76 GB. Was: never worked anywhere, on any binary, on either engine |
| 16 SigLIP | ✅ | 87.9 / 2350 — better than P18 |
| 17 SD3.5 | ✅ | 5782 @1024 (1.13×), 1235 @512 (**0.81× — native wins**) |
| 18 LoRA | ✅ | 359.2 cell, benefit-gate re-check 361.8 (1.007×); its live-set spike unmoved to 2 MB |
| 19 maxtext train | ✅ holds (benefit gate) | 458.4–463.5 across five vendored/stock runs, inside the 456–470 class; loss bit-identical across all nine runs of the campaign |
| 20 235B-3bit | clean decline | needs packed sub-byte (feature, post-release) |

## Decisions on record (rule 2)

- **No-panic contract + amendment** — CLAUDE.md ground rules, verbatim.
- **Row 15 was release-blocking by your ruling; the mechanism proved to be
  MLX's fence drop** (slicing.cpp:62), fixed in our vendored build. The
  native attestation **passed** (see the table) — ships as FIXED.
- **Watermark regression → option (b)** benefit gate: **landed and
  RESOLVED** (`notes/cpp-p28-benefit-gate.md`, commit 26a8941).
- **Rows 5/7 nondeterminism**: **it survives the fence fix — the release-note
  item stands.** Measured on the vendored binary: row 5 over **six** draws is
  5 identical + 1 divergent at token 51; row 7 diverges on every pair (50/51).
  Row 5's first three draws all agreed, which would have published a false
  retirement had the extra draws not been run — the gate document had already
  recorded a two-sample agreement on this row as luck. Different mechanism
  from the fence drop (fused-attention emit ordering), as the diagnosis
  predicted.
- **`__version__`**: fixed at **0.11.5** (f6fa849, the one-line frozen-src
  exception) — the release wheel reports it correctly, and
  `build_native_wheel.sh` now asserts `__version__` == pyproject's version
  so this class of drift cannot recur.
- **129-row jax-suite whitelist**: reviewed item-by-item (parity report);
  includes the 2 sparse rows = MLX fusion bug #8 (fork fix-branch
  candidate #2, needs an MLX-level scatter repro).

## For you, sequenced

1. **Now / anytime** — the fork branches are pushed (done 2026-08-18), so
   what is left is upstream, with drafts ready in `notes/patches/`: the
   **release request on ml-explore/mlx#4099** (their own unreleased fix; our
   numbers, and the 20-line reproducer — now verified wheel-side too: 1/20
   evaluations wrong on stock 0.32.0 in each of three fresh processes,
   0/20 on the patched build) and **our `end_encoding` hardening PR**
   (`fix/temporary-fence-tracking`, which fixes the class rather than the
   instance).
2. **After the consolidated re-gate** (both batteries landed; the re-gate is
   running — full model re-measure, pinned jax suite, tests/ native leg, all
   on the final binary): review the flipped gate document and the row-1
   disposition → I upload to TestPyPI → verify from fresh venv → you publish
   public + push + tag.
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
- Stage 1 no longer ships at all (your 2026-08-18 ruling: the release wheel
  is native-only). It stays in the repo as the frozen dev/reference
  implementation until the post-release retirement — and note it keeps the
  UNPATCHED public MLX: the fence fix lives only in the vendored library
  inside the native wheel, so anything run through Stage 1 (or pip's mlx)
  still has the row-15 corruption class until upstream releases #4099.
