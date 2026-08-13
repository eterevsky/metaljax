# Model benchmark suite — status

*(Harness and manifest: [scripts/model_bench/](scripts/model_bench/).)*

*Updated 2026-08-12: new `metaljax-native` column (phase-2 plugin) beside the
pre-migration `metaljax` one — footnotes 24–26 and
[benchmarks/perf-2026-08-native-baseline.md](benchmarks/perf-2026-08-native-baseline.md).
That campaign ended in kernel panic #8 (footnote 25); the empty native cells
were never attempted.*

*Updated 2026-08-12 (later): the native column is **re-measured with the P17
recognizer emits** on the relinked plugin — footnote 27. Rows 3/5/6/14/16/17
moved to 0.95–1.13× of Stage 1 (three of them past it); row 7 is now blocked on
pack-build MEMORY rather than compute; row 13's residual is one compile-gate
bug, itemised in footnote 27. Cells not re-run keep their P16 value and are
marked ᴾ¹⁶.*

*Last updated: 2026-08-04 — 0.11.2 baseline cells are FINAL (sequential
release-gate run at shipped defaults, token agreement audited per
footnotes 21/22); rows re-measured on the 0.11.3 tree are marked by
footnote (row 7 so far, footnote 23). As of the 2026-08-04 harness,
prefill/decode absorb per-shape executable builds first (steady-state
metric; one-time cost reported as build_s — footnote 23). Headline
metric per cell: LLM rows = warm decode ms/token; vision = forward ms;
diffusion = ms/step; training = ms/step. ✗ = established impossible
(with the measured reason).*

| # | benchmark | jax CPU | metaljax | metaljax-native ²⁴ | mlx-lm | torch-MPS | llama.cpp |
|---|---|---|---|---|---|---|---|
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **350** | 301.6 (1.24×) ᴾ¹⁶ | 137 | 148.7 | 111.2 ²⁰ |
| 2 | gemma4-12B bf16 | 315 (f32) | **97.1** | 98.8 (1.05×) ᴾ¹⁶ | 58.3 ¹⁵ | 67.6 | 44.2 ²⁰ |
| 3 | gemma4-26B-A4B (MoE) | ✗ guard-killed @34 GB ¹⁴ | **44.3** (51.6 GB) ¹⁶ | **43.4** (0.99× ²⁷) | **17.0** | — | 7.9 (Q4 QAT) ²⁰ |
| 4 | gemma4-E2B bf16 | 67.4 (bf16→f32) ¹³ | **29.5** ²¹ | 27.2 (1.00×) ᴾ¹⁶ | 10.5 ¹⁵ | — | — |
| 5 | Qwen3-8B bf16 | 209 (bf16→f32) ¹³ | **60.4** | **57.9** (0.99× ²⁷) | 30.4 | 38.1 | 15.7 (Q8) ²⁰ |
| 6 | Llama-3.1-8B bf16 | 200 (bf16→f32) ¹³ | **57.3** ²¹ | **54.7** (0.99× ²⁷) | 29.4 | 35.5 | 15.4 (Q8) ²⁰ |
| 7 | gpt-oss-20b | ✗ ⁴ | **22.2** (23.9 GB, MXFP4 + expert gather) ²³ | **25.3** (1.16×, 35 GB ²⁸) | **8.8** (13.8 GB, native MXFP4) | — | 6.7 (native MXFP4) ²⁰ |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | ✗ warmup transients ¹⁷ | not run (PAUSED ¹⁷) | **13.7** | — | — |
| 9 | R1-Distill-32B | ✗ 131 GB | **217.7** (65.5 GB) ¹⁷ | **214.4** (65.5 GB) ²⁹ | 131.8 | — | — |
| 10 | DeepSeek-V2-Lite (maxtext) | ✗ needs 50–105 GB ⁶ | ✗ guard-killed @122 GB ⁶ | not run (embargo) | **10.6** | — | — |
| 11 | Qwen3-0.6B (maxtext decode) | 89.7 | **16.0** ⁷ | 16.67 (1.02×) ᴾ¹⁶ | — | — | — |
| 12 | Mixtral 8×7B bf16 | ✗ | ✗ keras load ¹⁷ | not run (PAUSED ¹⁷) | **52.8** (93.4 GB) | — | — |
| 13 | gemma4-E2B keras-int4 (packed) | **67.8** ¹⁸ | 85.0 @ 2.7 GB ¹⁸ | **275.6** (qmm fires, decode body does not compile ²⁷; steady state 3.2 GB ²⁸) | — | — | — |
| 14 | maxtext qwix-int8 0.6B | 143.4 | **48.5** ²² | **56.8** (0.95× ²⁷) ²⁶ | — | — | — |
| 15 | *qwix-int8 Qwen3-8B* | 2118 | ✗ MLX command-buffer bug ⁸ | not run (embargo) | — | — | — |
| 16 | SigLIP 2 (fwd b1 ms) | 533 | **93.4** | **96.9** (1.13×; b32 0.96× ²⁷) | — | 29.8 (b32: 591) | — |
| 17 | SD 3.5 Large (ms/diff-step) | ✗ ¹² | 1389 @512², 5141 @1024² ⁹ | **1234.8** @512² (0.81×), **5781.6** @1024² (1.13×) ²⁷ | ✗ ¹⁹ | 654 @512², 2998 @1024² ¹⁹ | — |
| 18 | LoRA E2B train (ms/step) | 2048 | **407** | 656.3 (1.63×) ᴾ¹⁶ | — | 135.6 ¹⁰ | — |
| 19 | maxtext train 0.6B (ms/step) | 1402 | **440** ¹¹ | 962.0 (1.01×) ᴾ¹⁶ ²⁶ | — | — | — |
| 20 | *aspirational* 235B-A22B 3-bit | ✗ | ✗ needs packed-quant storage | — | **28.0** (102.9 GB, load 12 s) | — | — |

**Splat-fix before/after (measured today):** Qwen3-8B 268→60.3 ms/tok
(143.6→16.4 GB); Llama-8B 228→58.6 (127→16.1 GB); gpt-oss 2090→220.4
(224→41.8 GB). All three now beat jax-CPU 3.4–3.7×.

**mlx-lm gap band (same Metal library underneath — the C++-rewrite
target):** bf16 dense decode 1.7–2.6× (Qwen3-8B 60.4 vs 30.4; Llama
57.3 vs 29.4; 12B 97.1 vs 58.3; 31B 350 vs 137); gpt-oss 2.5×
(22.2 vs 8.8 — MXFP4 quantized_matmul + expert gather on both sides,
footnote 23); MoE 2.6× (44.3 vs 17.0, decode gather landed — footnote
16). llama.cpp leads mlx-lm a further ~1.25× on bf16 — the kernel
frontier. metaljax prefill trails ~6×; load ~20–30×.

## Footnotes

1. Both backends crashed on the sentencepiece SIGABRT (`import tensorflow`
   poisons pip sentencepiece — root-caused, shim landed in run_bench.py);
   re-run queued.
2. Pre-splat-fix numbers, contaminated by 23×-splat-constant retention
   (143.6 / 127 / 224 GB "active", swap thrash). Fix applied
   (splat constants broadcast from a 1-element buffer + dynamic-while
   bodies now compile); 0.6B evidence: decode 56 → 10.8 ms/tok, memory
   13.9× → 1.0×. These three rows re-run first after the gate.
3. torch-MPS adapter validated (greedy tokens ≡ torch-CPU, 32/32);
   timings deliberately deferred.
4. CPU: dequantized-bf16 working set projected ~126 GB (panic
   territory) — established infeasible; keras dequantizes the
   MXFP4-native repo to bf16 (~42 GB).
5. Tokenizer EOS fix verified (DeepSeek removed keras's hardcoded
   `<|endoftext|>`); first generation run pending, serialized.
6. maxtext memory model: sparse path still wants 50–83 GB for a 16B MoE
   prefill. Rescope decision pending (candidate for drop; Qwen3.6-35B
   covers the MoE class).
7. FIXED (28ad2eb): was MLX 0.32 corrupting compiled graphs split
   across Metal command buffers at the 40 MB byte default; byte cap
   raised. 3/3 runs byte-identical to CPU at 15.9 ms/tok.
8. int8 is functionally exact on both backends (tokens verified). The
   old int64 outer-product pessimization was fixed 2026-08-03 (chunked
   exact f32 dot, fdc7cde): 0.6B decode 48.5 → 31.8, prefill 1255 →
   37.7. The 8B row then ran for the first time (~330 ms/tok, no OOM)
   but produces garbage: the MLX 0.32 command-buffer split corruption
   at 8B scale. Single-variable proof: byte budget alone flips the
   outcome — 512 (default) corrupts decode replays (bf16 AND int8;
   first call clean, replays differ per process; correct under
   METALJAX_COMPILE=0), 2048 gives correct output with a benign KV
   curve — but 2048 has NO stability margin: an 8B load at 2048
   kernel-panicked the machine (watchdog timeout, wired-memory class)
   after an identical run had succeeded. Eager-mode mitigations at the
   default budget all ballooned (67–109 GB at load: uncompiled bodies
   pin the whole lazy load DAG) and the final attempt kernel-panicked
   the machine a second time — full attempt ledger in
   notes/mlx-command-buffer-split.md (2026-08-03 addendum). 8B-class
   maxtext is EMBARGOED on this machine; the row waits for either an
   engine-side eval-forcing mode (small-scale-validated first) or the
   MLX upstream fix. Repro:
   notes/data/qwen3_8b_prefill_36layer.mlir (0.3 s).
9. RESOLVED 2026-08-03, both walls. (a) MEMORY: attention now fuses
   (sdpa recognizer; 5–8 GB/block logits → ~0.15 GB) and the eager
   memory stack bounds the rest — 512²/20 steps peaks at 18.1 GB.
   Fusion measured worth 1.30× (1389 vs 1803 ms/step). (b) The
   "correctness wall" was MISATTRIBUTED to the command-buffer bug:
   every historical black image (1024², 20-step 512², incl.
   METALJAX_COMPILE=0) was a HARNESS bug — keras-hub's scheduler
   stores sigmas as a plain attribute, jit bakes them as constants,
   and the 4-step warmup executable was reused at 20 steps → OOB
   take → NaN sigmas → clip-to-zero pixels. jax-CPU reproduces the
   same black image. Fixed: per-step-count samplers + fail-closed
   image check + diffusion-appropriate prompt; backend exonerated
   (OOB-NaN propagation verified bit-matching CPU). Image verified
   against the torch-MPS reference. 1024² measured 2026-08-04 on the
   0.11.3 tree: 5141 ms/step (marginal 5100), 20 steps, real image
   (pixel_std 62.6), peak footprint 34.0 GB under a 70 GB guard —
   the earlier 55-GB-budget kill predated plan-aware pruning + the
   16 GB compile-bytes cap this run used (METALJAX_COMPILE_BYTES_MB=
   16384). 1.7× behind torch-MPS at 1024² (vs 2.1× at 512²).
10. torch MPS SDPA has no backward kernel — substantiated by autograd
    node inspection (math fallback), disclosed in the record. Loss
    series not comparable across stacks (different preprocessing);
    step cost is the comparison.
11. Correctness FIXED (52b90a2): the eager loss divergence was the
    command-buffer bug's THIRD face — ops=400 landed a buffer boundary
    that corrupted one RNG key in the init scan; ops now 800 (+2–3%).
    Validation: eager loss bitwise ≡ compiled (247.7775), 1.4e-4 vs
    CPU (247.8117). Step TIMING not yet captured — comes with the
    final sequential run.
12. keras's mixed-precision layers request the F16_F16_F32 dot
    algorithm, which XLA:CPU rejects (plain f16 dots work; the
    algorithm spec is an accelerator contract). A strip-workaround
    would enable a CPU reference; planned.
13. CPU cells run what XLA:CPU supports: weights load bf16, matmuls
    upcast per-op (bf16→f32); the 12B row is full f32 (gemma-lib path).
14. 26B-A4B CPU: the model cannot fit — f32 26B is ~104 GB of weights
    alone, and the observed keras-CPU load inflation (2.9×) projects a
    ~150 GB peak on a 128 GB machine. The guard killed the load as soon
    as the growth trajectory made that projection conclusive (34 GB and
    climbing); the alternative was the Qwen3.6-style swap-death (196 GB
    footprint) that froze the machine.
15. Released mlx-lm 0.31.3 cannot run gemma4_unified (12B) or the
    E-series KV-sharing layout (E2B) at all; these two cells measured
    on mlx-lm git main (2026-08-03 install). The 12B/31B gemma-lib
    decode improvements vs earlier entries (189→101, 374→363) come
    from the dynamic-while body-compile fix landing in the sampler's
    decode loop; old CPU 938 superseded by the uniform harness.
16. MoE DENSE-EXPERT GAP — CLOSED in two steps. Was the largest
    measured: keras/XLA lowers expert dispatch densely (streams all
    51.6 GB/token) → 473→284 ms/tok vs mlx-lm's gathered 17.0. The
    expert-gather recognizer (2310aa2) first moved only PREFILL here:
    its broadcast classifier tested "is the full expert/token axis"
    before "is a unit axis", and at T=1 — every decode step — the
    unit token axis bound as the real one, rejecting the per-expert
    scale chain, so decode stayed dense at 292 while prefill fused.
    Branch order swapped (3432dc0): 30 decode dispatches (E128/K8/T1)
    gather per executable, decode **44.3** ms/tok (6.4×), mem
    unchanged 51.6 GB, 2.6× behind mlx-lm (was 16.7×). Tokens vs the
    dense run agree 53/64 then flip — gathered sum runs K=8 terms
    where dense ran 128 with 120 exact zeros; footnote-22 ladder
    class. gpt-oss (row 7) re-verified neutral: 22.3 ms/tok, and the
    cross-executable pack-build cache (a3d25f0) cut its build_s
    200→8.5 s (wall 307→96 s; row 3 build_s 51→11 s).
17. The keras LOAD ceiling is fixed (streaming loader, 30c9717: init
    never materializes; E2B peak 25→9.5 GB; Qwen3.6-35B ports all
    1026 weights, ~70 GB resident). What remains lethal is the phase
    AFTER load on 60 GB+ models: warmup/compile transient ramps drove
    swap to 9 GB (R1, guard-killed at 95 GB budget) and a chained
    second load onto that degraded system caused kernel panic #6.
    mlx-lm holds 93.4 GB resident fine — the danger is our stack's
    allocation ramps, not static residency. RESOLVED for row 9
    (2026-08-04): the "warmup transient" was a HARNESS double
    residency — R1 takes the special-token retry path, and
    from_preset raises AFTER assigning the full 61 GB checkpoint;
    the retry ran INSIDE the except block, where the live traceback
    pins the dead first model while a second copy loads (93 GB and
    climbing at the guard kill; the CLAUDE.md item-17 trap in the
    harness this time). Retry moved out of the except scope +
    gc.collect: load runs flat at ~71 GB (phased probe), row 9
    completes at 65.5 GB peak under the 95 budget — **217.7**
    ms/tok, 1.65× behind mlx-lm, first metal number for this row.
    Rows 8/12 queued behind the same protocol (row 8 loads via the
    normal path — no retry — so its old kill was pure load ceiling,
    fixed by streaming).
18. Packed int4 memory saving IS real on metaljax (2.7 vs 10.2 GB —
    the only sub-byte JAX path that keeps it). XLA:CPU fuses the
    in-graph unpack into a small net WIN (67.5 vs 79.2 bf16). The
    mx.quantized_matmul recognizer (5fd6b2a) + interleaved-group
    K-permutation (f45bbbe) took metal 336 → 241 → 85.0 ms/tok
    (all 777 quantized dots fuse; decode body compiles). The residual
    1.25× vs CPU is the batch-1 GEMV kernel-launch floor (~2k Metal
    dispatches/token; XLA:CPU launches nothing) — C++-era.
19. torch SD3.5 via the ungated diffusers mirror
    adamo1139/stable-diffusion-3.5-large-ungated @5d868ff (official
    repo is gated); coherent images verified at both resolutions.
    MLX cell: mflux is Flux-only; DiffusionKit supports the SD3 family
    but 3.5-Large weights are gated in its formats — no ungated MLX
    path exists, cell closed as not-runnable.
20. llama.cpp build 221f0f63 (past the Gemma4 cutoff), llama-bench
    -p 51 -n 128 -r 5, all-Metal, reproduced within 4% on two passes;
    per-provider GGUF pins in README_llamacpp.md. Q8/Q4 marked where
    no bf16 GGUF exists. Dense rows all pin to 439–555 GB/s effective
    bandwidth (the bandwidth-bound signature); llama.cpp leads even
    mlx-lm ~1.25x on bf16 — the kernel frontier on this hardware.
    Deferred rows (35B/R1/Mixtral llama.cpp cells) dropped as
    redundant with the covered comparison classes.
21. Release-gate token audit: greedy streams metal-vs-CPU are exact on
    12B, Qwen3-8B, E2B-int4, and the maxtext rows; E2B-bf16 and
    Llama-8B diverge at their FIRST generated token via certified
    tie-flips — competing logits within 1–2 bf16 ULPs (Llama: exactly
    tied 11.875/11.875 on metal), the same benign class as the int8
    certification. gpt-oss/26B/31B have no CPU counterpart (recorded
    only).
22. CERTIFIED BENIGN (notes/int8-divergence-verdict.md): the token
    divergence vs int8-CPU is an exact logit tie on metal (14.5 vs
    14.5) at a step whose CPU margin is 7 bf16 ULPs = 1.3σ of the
    quantization noise; the s8 dot+dequant is bit-identical on real
    data; even CPU-int8 vs CPU-bf16 flips the same token. Timing valid
    (3.1× vs bf16 — the int64 cliff). NB token-stream equality is not
    a usable correctness criterion for quantized decode; use the
    logit-delta ladder.
23. Row 7 arc (2026-08-04): 222 (dense-dequant) → 39.6 (native MXFP4,
    0.11.2) → **22.2** with the MoE expert-gather engaged (47 dispatches)
    and TWO fixes that were prerequisites, both on the 0.11.3 tree:
    (a) qmm row-blocked pack evaluation (e04c7fc,
    notes/qmm-pack-transient-2026-08.md) — the 9–15 GB per-pack
    transients that trajectory-killed every earlier re-measure are now
    ~1.5 GB; run peaks at 25.0 GB under a 45 GB guard. (b) HARNESS
    metric fix: keras-hub compiles a separate generate executable per
    max_length, and each shape's first call was paying jit + engine
    compile + the whole qmm pack wave inside the TIMED window — row 7's
    "96 s prefill" was ~99% one-time build, and its measured decode
    varied 37–54 ms/tok with the wave's tail. run_bench.py now absorbs
    both timed shapes first (reported as build_s; 200 s here — the
    per-executable pack REBUILD it contains is the next cost to fall,
    packs are content-identical across executables); steady-state
    decode profiled flat at ~30 ms/tok under cProfile overhead, 22.2
    clean. Applies to all keras rows; materially moves only the
    quantized ones (7, 13). Greedy tokens vs the 39.6 run agree 55/64
    then flip once — benign reassociation (expert-gather sums 4 experts
    in a different order than dense-all-experts); footnote 22's ladder
    policy applies. Eager-discipline A/B (prune/flush/sdpa off) is
    IDENTICAL at steady state (29.9 vs 30.1 ms/tok) — the disciplines
    cost nothing where it counts; defaults unchanged.

24. **The metaljax-native column** (2026-08-12,
    [benchmarks/perf-2026-08-native-baseline.md](benchmarks/perf-2026-08-native-baseline.md)):
    the phase-2 plugin (`plugin-native`, tree 845ab89) measured against the
    frozen Stage 1 trampoline, one row at a time under the same
    `guarded_run.sh` protocol. Ratios in parentheses are native/Stage-1
    **measured on the same day**, not against the 0.11.3 anchor in the
    metaljax column — Stage 1 re-measured within ±4.5 % of every anchor
    except rows 14 and 19 (footnote 26). The native lowering has **no
    recognizer emits and no msl_scan**, which is what every gap above 1.3×
    is: qmm (rows 7, 13), MoE expert gather (rows 3, 7), sdpa (rows 1, 16,
    17, 18). Where none of them fire the two stacks are equal (rows 4, 11,
    19) or native is ahead (texmo `big09-b8l256` 0.68×). Two mechanical
    findings ride with the column: the native dylib **cannot share a process
    with TensorFlow / array_record** (static protobuf + LLVM symbol
    collision at dlopen — it reached every model row, through keras, through
    `kauldron`, and through `array_record_module.so`), fixed for these runs
    by relinking with an exported-symbols list holding only `_GetPjrtApi`
    and `_metaljax_native_set_callback_trampoline` (166 → 46 MB, verified
    perf-neutral); and `mx.get_active_memory()` cannot be trusted under the
    native plugin (two MLX runtimes) — memory here is the guard flight log's
    peak footprint.
25. **KERNEL PANIC #8** (2026-08-12, 03:33): the row-9 *native* attempt
    wedged the machine during the 65 GB streaming load — footprint 54 GB,
    RSS 54.6, system 64.5 GB, every guard sample "ok", flight log stops
    mid-line (`r1-distill-32b-0812-033145-flight.log`). Same watchdog-wedge
    class as panic #7, and the same blindness: no memory metric was
    unhealthy. Stage 1 had run the identical row clean 8 minutes earlier
    (213.8 ms/tok, peak 67 GB). Attributed to the native plugin lacking
    Stage 1's load-phase cache-clear cadence. **Row 9 native is embargoed
    until that cadence lands**, and the campaign was halted here — rows
    marked "not run" above were never attempted.
29. Row 9 native retry (2026-08-13): ladder-verified (16/31/65 GB rungs each within 2% of predicted peak), load throttled to 0.30 GB/s (BENCH_STREAM_THROTTLE_GBPS / MJ_INGEST_THROTTLE_GBPS -- the #4/#7/#8 wedge class is fill-rate-sensitive; the panicked run filled at 0.42, the surviving Stage 1 run at 0.35). decode 214.4 ms/tok = 1.003x Stage 1, tokens identical, peak RSS 65.6 GB, throttle_s 45.5, clears 16, machine clean after. Jsonl: r1-distill-32b-0813-020534.

26. Rows 14 and 19 are **Stage-1 regressions against their own 0.11.3
    anchors**, independent of the plugin under test: row 14 32.5 → 60.1
    ms/tok (1.85×; `METALJAX_ENGINE=py` gives 42.3, so ~1.42× of it is the
    C++ tape and ~1.30× predates it) and row 19 440 → 956.5 ms/step (2.17×;
    py-engine 1043, so *not* the tape). Their native/Stage-1 ratios are
    honest (both ~1.0×) but the Stage 1 baseline itself has moved. Also new
    on this tree, both plugin-independent: the LoRA row's load transient
    peaks at **56 GB** and the E2B-int4 row's at **44 GB** (steady states
    10.2 and 3.1 GB) — a 45 GB budget guard-kills both.

27. **The native column re-measured with the P17 emits** (2026-08-12, tree
    8c61e72; raw `~/.cache/metaljax-bench/logs/p18-relink/`, summary
    `notes/data/p18-relink-models-2026-08-12.jsonl`, ratios vs the Stage 1
    column re-measured the same day — a Qwen3-8B control gives 58.2 against the
    morning's 59.1, so that column is stable to 1.5 %). Cells still carrying a
    P16 number are marked ᴾ¹⁶.
    **Closed by the emits**: MoE 6.88× → **0.99×** (row 3, and its greedy tokens
    are 64/64 identical to Stage 1 — it is the P16 *dense* run that diverges at
    token 52, footnote 16's ladder class); sdpa 3.14× → **1.13×** at 1024² and
    **0.81×** at 512² (row 17), 1.78× → **0.96×** at SigLIP b32 (row 16); the
    dense rows and the int8 row went *past* Stage 1 (rows 5/6 0.99×, row 14
    0.95×).
    **Row 7 is now a MEMORY block, not a compute one.** The emits fire (94
    quantized matmuls recognised, 47 gathered expert dispatches, 188 packs) but
    P17 deliberately left out qmm's row-blocked `_Source` evaluation and its
    cross-executable pack cache, so a full pack set per compiled shape is live
    at once: guard-killed at 46 GB under the row's historical 45 GB budget and
    at 62 GB under 60, where Stage 1 runs the row at 25 GB. No further budget
    escalation was attempted (62 GB is panic #7/#8 territory).
    **Row 13's residual 3.09× is one bug, and it is not qmm.** All 777 quantized
    dots fuse (group 64/128, interleaved regrouping engaged) and prefill is
    already *ahead* of Stage 1 (218.3 vs 241.0 ms); what is left is that the
    fused program reports `compiles=0 compiled_calls=0` — the decode while body
    never compiles, so the loop runs op by op. The compile gate's `BlockCost`
    walks the StableHLO block and still charges every op the emit ABSORBS, so
    the body looks over `METALJAX_TRACE_BUDGET`. Proof: the byte budget alone
    changes nothing (274.6 ms/tok), `METALJAX_RECOGNIZE=0` reports `compiles=0`
    too, and `METALJAX_TRACE_BUDGET=1e7` measures **85.5 ms/tok = 1.06× of
    Stage 1**. Fix belongs in the fused lowering's cost/byte accounting.
    Also from this pass: row 5's greedy tokens now diverge from Stage 1 at token
    61 of 64 (they agreed before the sdpa emit) — footnote-21 tie-flip class but
    a *new* divergence, worth the logit-delta ladder rather than an assumption.
    Mechanically, all of it rides on the exported-symbols relink finally being
    in the tree (`plugin-native/metal/exported_symbols.exp`, default build, 166
    → 46 MB dylib and 42.2 → 11.8 MB wheel), without which none of these rows
    can even `dlopen` the plugin — `plugin-native/coexist_test.py` is its
    standing contract and `notes/data/p18-relink-battery-2026-08-12.txt` its
    evidence.
28. **Row 7 unblocked, and what actually did it** (2026-08-13, P19; raw
    `notes/data/p19-packing-models-2026-08-13.jsonl`, mechanism
    `notes/cpp-p19-packing.md`). The two optimizations P17 named and skipped —
    qmm's row-blocked `_Source` evaluation and its cross-executable build cache
    — are ported, and **gpt-oss-20b now completes at its historical 45 GB
    budget**: 35 GB peak, 128 tokens, **25.3 ms/tok = 1.16× of Stage 1's 21.9**
    (re-measured the same day and reproducing its anchor exactly), four samples
    inside 25.3–25.5.
    **The ablation says the CACHE was the load-bearing half**, which is not what
    the P18 diagnosis predicted: at the same budget, cache off / blocking on is
    guard-killed at 46 GB (P18's number to the gigabyte) while blocking off /
    cache on completes at 36 GB. Three executables were each building their own
    ~10 GB pack set; row-blocking is worth a further gigabyte on top, and only
    the pair clears the line. Evidence from the run's own log: 94 packs built,
    **188 reused** (100 % hit rate on the second and third executables), all 94
    blocked, and the plugin's pack-wave peak 33.9 GB → **0.000 GB** for the two
    reuse waves.
    Row 7 does *not* share row 13's compile bug: `METALJAX_TRACE_BUDGET=1e7`
    returns the same 25.3 ms/tok with bit-identical compile decisions (16
    compiles / 354 compiled calls either way).
    **Row 13 is timing-neutral under P19** — 275.6 against a P19-off control of
    271.7 on the same binary, and P18's own byte-cap control read 274.6, so the
    249.0 headline was the low end of this row's spread rather than a mark P19
    missed. What P19 changes there is the steady state (**4.2 → 3.2 GB**) and
    518 of the 777 pack builds. Its 46 GB peak is now *attributed*: it is the
    keras streaming LOAD transient, which Stage 1 shares at 44 GB, and not the
    packs — the pack wave peaks at 6.6 GB.
    Scrutiny: row 7's greedy tokens diverge from Stage 1 at index 52 of 64.
    There is no prior native token record for this row (P18 never completed it),
    so it is a first observation rather than a change, and it is the same
    late-divergence ladder class as rows 3, 5 and 11.

## Bug ledger (found by this suite)

- **Splat-constant retention** (FIXED, d9d774e, gated): whole-shape
  splat constants materialized + pinned per executable; jax
  `random.normal` carries 23 full-weight-shape splats → keras models
  retained ~9× their weights. Predicted 143.7 GB vs measured 143.6.
- **Dynamic-while bodies never compiled** (FIXED, d9d774e, gated): LLM
  decode loops interpreted op-by-op → Python-dispatch-bound (the reason
  8B decode lost to CPU pre-fix).
- **MLX command-buffer corruption, three faces** (all worked around:
  28ad2eb bytes-floor, 0da62c0 bytes-ceiling, 52b90a2 ops alignment;
  upstream report pending): footnotes 7, 9, 11. Every finite budget is
  a lottery draw until MLX fixes it; the command-buffer tests pin the
  shipped values.
- **SD3.5 black image** (RESOLVED by 28ad2eb, same MLX bug): footnote 9.
- **sentencepiece SIGABRT** (worked around): footnote 1.
- **int8 dot_general int64 cliff** (known, measured): footnote 8.
- **MoE dense-expert lowering** (FIXED: 2310aa2 expert gather +
  3432dc0 T=1 broadcast classification — row 7 39.6→22.2, row 3
  284→44.3): footnote 16.
- **Per-shape executable builds timed as prefill/decode** (FIXED in the
  harness, build-absorb + build_s column): footnote 23. Underlying
  per-executable pack rebuild (~0.9 s × 94/shape) open — build cache in
  progress.
- **keras load-path memory** (open, blocks ≥60 GB keras rows): footnote 17.
- **int4 unpack re-materialization on metal** (open, 11.7×): footnote 18.
- **native plugin: the fused lowering's compile gate reads the UNFUSED IR**
  (open, worth 2.9× on row 13): `BlockCost`/`BlockBytes` charge the ops a
  recognizer absorbs, so an emit-cheapened decode body still trips
  `METALJAX_TRACE_BUDGET` and runs uncompiled. Footnote 27.
- **native plugin: pack building has no memory discipline** (open, blocks row 7
  natively): no row-blocked `_Source` evaluation, no cross-executable pack
  cache — 49–62 GB where Stage 1 uses 25. Footnote 27.
- **native plugin dylib could not coexist with TensorFlow / array_record**
  (FIXED 2026-08-12, exported-symbols list, default build): static protobuf +
  LLVM weak-definition coalescing, SIGSEGV at `dlopen` in both load orders; it
  blocked every keras / gemma-lib / maxtext row. Contract:
  `plugin-native/coexist_test.py`. Footnote 27.
