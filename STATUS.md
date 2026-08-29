# Model benchmark suite — status

*(Harness and manifest: [scripts/model_bench/](scripts/model_bench/).)*

*Updated 2026-08-12: new `metaljax-native` column (phase-2 plugin) beside the
pre-migration `metaljax` one — footnotes 24–26 and
[benchmarks/perf-2026-08-native-baseline.md](benchmarks/perf-2026-08-native-baseline.md).
That campaign ended in kernel panic #8 (footnote 25); the empty native cells
were never attempted.*

*Updated 2026-08-13: the native column now carries **two** ratios — vs Stage 1
measured the same day, and vs the 0.11.3 anchor (`models.md`). A
drift both stacks share reads 1.0× in the first and shows up only in the second;
row 19 read "1.01×" through two passes while both stacks sat 2.2× off the
anchor. Rows 7/13/14/16/18/19 re-measured — footnote 30.*

*Updated 2026-08-28: the Stage-1 (Python engine) column is REMOVED — the
engine was retired in 0.11.6 (ef5774d); its cells live on in git history
and in models.md's 0.11.1–0.11.2 columns. The metaljax column below is
the native plugin at the 0.11.6 release gate.*

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

| # | benchmark | jax CPU | metaljax (current ³⁷) | mlx-lm | torch-MPS | llama.cpp |
|---|---|---|---|---|---|---|
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **235.2** ³⁷ | 137 | 148.7 | 111.2 ²⁰ |
| 2 | gemma4-12B bf16 | 315 (f32) | **92.1** ³⁷ | 58.3 ¹⁵ | 67.6 | 44.2 ²⁰ |
| 3 | gemma4-26B-A4B (MoE) | ✗ guard-killed @34 GB ¹⁴ | **43.3** ³⁷ (53 GB) | **17.0** | — | — ²⁰ |
| 4 | gemma4-E2B bf16 | 67.4 (bf16→f32) ¹³ | **25.8** ³⁸ | 10.5 ¹⁵ | — | — |
| 5 | Qwen3-8B bf16 | 209 (bf16→f32) ¹³ | **57.6** ³⁷ | 30.4 | 38.1 | — ²⁰ |
| 6 | Llama-3.1-8B bf16 | 200 (bf16→f32) ¹³ | **54.3** ³⁷ | 29.4 | 35.5 | 29.2 ²⁰ |
| 7 | gpt-oss-20b | ✗ ⁴ | **21.3** ³⁷ (34 GB) | **8.8** (13.8 GB, native MXFP4) | — | 6.7 (native MXFP4) ²⁰ |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | **29.4** ³³ ³⁷ (73 GB) | **13.7** | — | — |
| 9 | R1-Distill-32B | ✗ 131 GB | **211.0** ³⁷ (67 GB) | 131.8 | — | — |
| 10 | DeepSeek-V2-Lite (maxtext) | ✗ needs 50–105 GB ⁶ | **108.7** ³⁸ (83 GB) | 10.5 | — | 10.7 ²⁰ |
| 11 | Qwen3-0.6B (maxtext decode) | 89.7 | **13.28** ³⁸ | 3.0 | — | — |
| 12 | Mixtral 8×7B bf16 | ✗ | **91.3** ³⁷ (93 GB) | **52.8** (93.4 GB) | — | — |
| 13 | gemma4-E2B keras-int4 (packed) | **67.8** ¹⁸ | **78.0** ³⁷ | — | — | — |
| 14 | maxtext qwix-int8 0.6B | 143.4 | **31.08** ³⁸ (9.2 GB) | — | — | — |
| 15 | *qwix-int8 Qwen3-8B* | 2118 | **381.7** ³⁵ ³⁷ (73 GB) | — | — | — |
| 16 | SigLIP 2 (fwd b1 ms) | 533 | **88.31** ³⁷ | — | 29.8 (b32: 591) | — |
| 17 | SD 3.5 Large (ms/diff-step) | ✗ ¹² | **1234.7** @512², **4974.9** @1024² ³⁷ | ✗ ¹⁹ | 654 @512², 2998 @1024² ¹⁹ | — |
| 18 | LoRA E2B train (ms/step) | 2048 | **369.2** ³⁷ | — | 135.6 ¹⁰ | — |
| 19 | maxtext train 0.6B (ms/step) | 1402 | **463.4** ³⁷ | — | — | — |
| 20 | *aspirational* 235B-A22B 3-bit | ✗ | **66.3** ³⁷ (100 GB) | **28.0** (102.9 GB, load 12 s) | — | — |

**Splat-fix before/after (HISTORICAL, 2026-08-02, Stage-1 engine):** Qwen3-8B 268→60.3 ms/tok
(143.6→16.4 GB); Llama-8B 228→58.6 (127→16.1 GB); gpt-oss 2090→220.4
(224→41.8 GB). All three now beat jax-CPU 3.4–3.7×.

**mlx-lm gap band (same Metal library underneath — the optimization
target):** bf16 dense decode 1.6–1.9× (Qwen3-8B 57.6 vs 30.4; Llama
54.3 vs 29.4; 12B 92.1 vs 58.3; 31B 235.2 vs 137); gpt-oss 2.4×
(21.3 vs 8.8 — native MXFP4 both sides); MoE 2.5× (43.3 vs 17.0);
3-bit 2.4× (66.3 vs 28.0). llama.cpp leads mlx-lm a further ~1.25× on
bf16 — the kernel frontier. metaljax prefill trails ~6×; load ~20–30×.

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
    LIKE-FOR-LIKE RULE (Oleg, 2026-08-28): cross-framework cells appear
    only at the SAME precision as the metaljax cell — custom kernels are
    fair game, different quantization is not. The Q8/Q4-only rows
    (26B-A4B, Qwen3-8B, Llama-8B) therefore show no llama.cpp cell; the
    kept cells are BF16 (rows 1/2), native MXFP4 (row 7, both sides) and
    3-bit (row 20, both sides). Row 10's mlx-lm cell verified
    2026-08-28: mlx-lm 0.31.3 on the ORIGINAL bf16 repo, 10.5 ms/tok,
    31.4 GB active — like-for-like with our bf16 maxtext row.
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

30. **P20: the four named regressions** (2026-08-13; raw
    `~/.cache/metaljax-bench/logs/p20-regressions/`, tables in
    [benchmarks/perf-2026-08-native-baseline.md](benchmarks/perf-2026-08-native-baseline.md)).
    Ratios in the native column are now **both** same-day (vs Stage 1) and
    vs-anchor: a drift both stacks share reads 1.0× in the first and only the
    second catches it, which is how row 19 hid a 2.2× for two passes.
    (a) **Row 13 275.6 → 79.7 ms/tok, row 7 25.3 → 22.2** — footnote 27's
    compile-gate bug, fixed where it lives: `BlockCost`/`BlockBytes` in the
    native lowering now follow the rewrite plan as `ops/control.py` does
    (absorbed ops charged nothing, qmm/moe roots 2 units, sdpa 3, root bytes =
    own result + `emit_bytes`). Row 13's decode body compiles with no env
    override (`compiles=1 compiled_calls=127`, was `0/0`); row 7 gained 1.16×
    from the BYTE half, which P19's op-budget probe could not see.
    (b) **Row 18 656.3 → 397.5 ms/step, peak 55 → 37 GB** — the plugin copied
    every output that may alias an argument even when the argument was
    DONATED; a LoRA step donates 2,255 of 2,262 arguments and passes the frozen
    parameters through, so 1,952 outputs (~10 GB) were copied per step.
    `engine.py::_dealias` has always exempted donation; the plugin does now,
    with the per-call retraction (`non_donatable_input_indices`) handled in
    `RunOnce`.
    (c) **Row 19 is NOT the tape and NOT the harness**: 0.11.2's `src/metaljax`
    on today's machine and dylib measures **448.2** against the current tree's
    **969.1**, and the cause is the eager flush's `mx::clear_cache()`
    (`METALJAX_FLUSH_CLEAR_MB`, 4d34bff — landed *after* the 440 anchor was
    taken). This program's main is over the trace budget, so it runs eagerly at
    ~105 GB of traffic per step: 82 flushes and **7 pool-dumping clears** per
    step, ~70 ms each. Both stacks recover on the knob alone (Stage 1 478.8,
    native 468.0). NOT fixed — the clear is a real memory bound (without it the
    LoRA row blows an 81 GB peak) and the fix that keeps the bound without the
    cliff is `mx::set_cache_limit`, a shared-runtime memory-discipline change
    that needs Oleg's sign-off (and Stage 1's copy is frozen).
    (d) **Row 14 was never regressed**: standalone re-measurement gives 32.9 and
    32.7 against a 32.5 anchor. The 60.1 came 12 minutes into P16's sequential
    campaign — the suite-context trap of item 12, in the model harness.

31. **P24: the four ᴾ¹⁶ cells re-measured on the RC binary** (2026-08-16, raw
    `notes/data/p24-stale-rows-2026-08-16.{json,csv}`, logs
    `~/.cache/metaljax-bench/logs/p24-stale-rows/`, verdicts appended to
    `notes/rc-gates-2026-08-16.md` as the gate-1 scope correction). Rows 1, 2, 4
    and 11 were the last cells carried at their P16 value — measured before the
    P17 emits, the P18 relink and the P19/P20/P23 fixes — and their Stage-1
    controls were P16-era too, so both sides of every ratio were stale. Each row
    re-run Stage-1-control-first then native on the frozen RC dylib
    (`ed355691…94a16`, hash-verified against this tree's `bazel-bin`), one
    guarded process per row at its historical budget.
    **Native 301.6 / 98.6 / 27.0 / 16.63** against P16's 301.6 / 98.8 / 27.2 /
    16.67; **Stage 1 242.4 / 93.8 / 27.0 / 16.39** against 243.1 / 93.9 / 27.2 /
    16.42 — all eight inside 0.7 %, no cell moved, peaks on their recorded
    values (67 / 30 / 12 / 16 GB), 0 msl plans on all four.
    Two findings the pass settles: **row 1 takes no sdpa emit at all** (0
    recognized, `METALJAX_DEBUG=1`), so its 1.24× is the plain lowering and not a
    missing fusion, while row 2 *does* take 8 fused attentions and lands on its
    pre-emit number — sdpa is timing-neutral on T=1 gemma-lib decode. And the
    token divergence on rows 1/11 is **reproducible** (byte-identical to their
    P16 native streams; rows 2/4 are token-identical to Stage 1), i.e. distinct
    from the fused-path nondeterminism of rows 5/7.
    Guard note: row 1's load crest now sits at the old `GUARD_RSS_GB=110` cap —
    the first attempt was killed at rss 110.47 and the row was re-run at 115
    with the footprint (80) and system (100) budgets unchanged; measured crest
    113.8 / 113.6 GB, machine at 74–75 GB, crest collapses as the mmap'd
    checkpoint pages are released.
32. **P25 + P27: the eager flush's pool, and the watermark that is not one
    number** (2026-08-16, `notes/cpp-p25-cache-limit.md`,
    `notes/cpp-p27-flush-pressure.md`; raw
    `~/.cache/metaljax-bench/logs/p27-flush-pressure/`, aggregates
    `notes/data/p27-flush-pressure-2026-08-16.{json,csv}`).
    P25 replaced the flush-point `mx::clear_cache()` (a whole-pool dump, 7 a
    step at ~70 ms on row 19) with a TRIM back to `METALJAX_FLUSH_CLEAR_MB`,
    worth 1.17× there, and measured what the watermark itself is worth: row 19
    reaches its anchor only at 32 GB, where row 18 was guard-killed at 68.
    **P27 measured that conflict instead of trading it off**, with the process
    footprint (`task_info(TASK_VM_INFO)` — `mem_guard.sh`'s own metric) added
    to the flush meter: **row 18's blowout is a LIVE-SET spike**, 19.6 → 37.5
    → 46.5 GB in three flushes during keras build/convert, identical on both
    binaries, and the watermark only decides how much dead pool stands beside
    it. So `runtime.cc::flush_bound` now decides per flush — cap 32768, but
    only for a program that has taken 8 hard flushes (an eager MAIN reusing a
    pool, not a load filling one), and only up to what a 48 GB footprint
    target has left after that program's own live set; P25's 2048 is the FLOOR
    under both, so nothing is trimmed harder than it was.
    Cells: **row 19 1006.2 → 469.7** ms/step (five runs 460-478, peak 25 GB
    under its 48 GB budget), **row 18 397.5 → 360.2** (five runs, peak
    unchanged — 56.7-57.5 GB on the dylib's own meter with the policy either
    way, where the guard's 2 Hz sampling had reported 37-56), **row 13 80.3**
    and **row 2 92.9**, both controls.
    Ratios here are vs the 0.11.3 anchor and vs the SAME BINARY with the
    policy off (`METALJAX_FLUSH_CLEAR_MB=2048 METALJAX_FLUSH_FOOTPRINT_MB=0`),
    which is the only single-variable control for a policy change: 811.6 /
    389.0 / 80.2 / 93.2 against the four cells above. No same-day Stage-1
    control was taken for these rows — Stage 1's copy of the flush is frozen
    and still dumps, so its ratio on an eager-main row measures that, not this
    (footnote 30c); the suite pair covers the stack comparison.
    Suite-106 same-binary policy-on/off geomean **0.9983** (106 rows, median
    1.0001; `big` 0.9836), `texmo_gate` 106/106 three times, `execute_test`
    (P25's four contracts + P27's four), `ingest_test` 8/8, `bazel test`, and
    a buffer-COUNT probe — 0 buffer-limit recoveries in a 106-config sweep on
    either policy, which is the hazard raising a BYTE bound could have created
    (CLAUDE.md item 11a).
    Trap this pass adds to the ledger: **the guard's flight log is not a peak
    meter**. Row 18's true peak was 56.7 GB all along, on the shipped binary,
    sampled as 39-43 by a 2 Hz watchdog — a memory argument made from flight
    logs alone can be wrong by 17 GB.

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
  (FIXED 2026-08-13, footnote 30a; was worth 3.4× on row 13 and 1.16× on row 7):
  `BlockCost`/`BlockBytes` charged the ops a recognizer absorbs, so an
  emit-cheapened decode body still tripped `METALJAX_TRACE_BUDGET` and ran
  uncompiled. Footnote 27.
- **native plugin: output copies ignored donation** (FIXED 2026-08-13, footnote
  30b; 1.50× on row 18): an output aliasing a DONATED argument was copied, so a
  training step copied every frozen parameter it threads through.
- **eager flush dumps MLX's whole buffer pool** (OPEN, shared by Stage 1 and the
  plugin; 2.2× on row 19): every hard flush over `METALJAX_FLUSH_CLEAR_MB`
  clears the cache instead of bounding it, so an eager program with traffic ≫
  its live set re-allocates cold buffers all step. Footnote 30c.
- **native plugin: pack building has no memory discipline** (open, blocks row 7
  natively): no row-blocked `_Source` evaluation, no cross-executable pack
  cache — 49–62 GB where Stage 1 uses 25. Footnote 27.
- **native plugin dylib could not coexist with TensorFlow / array_record**
  (FIXED 2026-08-12, exported-symbols list, default build): static protobuf +
  LLVM weak-definition coalescing, SIGSEGV at `dlopen` in both load orders; it
  blocked every keras / gemma-lib / maxtext row. Contract:
  `plugin-native/coexist_test.py`. Footnote 27.

33. Rows 8/10 first native results (2026-08-17, memory-governor binary frozen-gov7): ORIGINAL implementations, no benchmark modifications. Row 8 = 29.6 ms/tok (panic #7's row; first metaljax number ever). Row 10 completes at 88 GB peak (previously guard-killed at 122 GB); ms/tok cell to be filled by the 0.11.5 release-gate sweep. notes/data/no-panic-governor-rows-2026-08-17.json.

34. **Row 15 runs and is WRONG — mechanism ESTABLISHED 2026-08-17 evening:
    nondeterministic MLX command-buffer corruption at 8B traffic, on BOTH
    engines, amplified to a logit collapse by qwix's per-tensor scale.**
    Full record and every arm: `notes/row15-wrong-output-2026-08-17.md` §8;
    raw `~/.cache/metaljax-bench/logs/row15-mechanism/` (driver
    `scripts/model_bench/row15_ladder.sh`, rungs `A1,B,C,D,Dp,Ref,K,Rate`).
    The governor campaign cleared this row's *memory* blocker — it completes,
    exit 0, 79 GB peak, flat page cache, 0 refusals, 369.7 ms/tok — and
    uncovered the one underneath: the output is `" fragment!!!!!!!"`, token ids
    `[12289, 0, 0, 0, 0, 0, 0, 0]`, and **`!` is Qwen3's token 0**. The timing
    is NOT published as a cell (a program computing the wrong answer has no
    meaningful ms/tok) and the row's status stays ✗.
    **The decisive measurement is a rate, not a verdict.** Ten prefills of the
    SAME loaded parameters, in ONE process, on identical inputs
    (`row15_forensics.py --prefill-reps 10`): row 15 native returns **8
    distinct first tokens and 2 full collapses**; row 15 on **Stage 1** returns
    **10 distinct**; row 14 (0.6B, same adapter, same qwix overrides, same
    script) returns **the same token 10/10** and decodes `" Paris. The
    capital"`. A correct implementation is deterministic, so this is
    self-proving — which matters, because the jax-CPU reference could not be
    re-measured (below).
    **Localization**: MaxEngine's prefill returns the layer-stacked KV cache,
    so the first non-finite layer is readable with no model surgery. Baseline:
    clean through layer 32, layers 33/34/35 **65536/65536 NaN, zero infs**,
    prefill logits 151936/151936 NaN. But the entry point is itself a draw —
    33, 6, 6, 11, or nowhere across arms; decode-side 27, then 1, then 1.
    **The amplifier is qwix's own code**: `qarray.compute_scale_zero_point`
    clamps a *zero* scale (`where(scale < sqrt(tiny), 1, scale)`) but not a
    non-finite one, and calibration is a per-tensor `absmax = max(|x|)`, so one
    bad element makes the whole tensor's scale NaN and every element of the
    dequantized result NaN. That is why it is always 100 % and never an inf.
    **What it is NOT** (single-variable arms, native release dylib, six
    different answers): `METALJAX_RECOGNIZE=0` still wrong; `METALJAX_COMPILE=0`
    *worse* (collapse at layer 11, so not our compiled path — it is the
    ops-alignment face); `METALJAX_CHUNK_MAX` 10/12/16 all wrong. The
    chunked-replay lead — 36 iterations at `kmax=16` replay as 16+16+4×1, and
    the first localization landed on layer 32, exactly the first remainder call
    — was **refuted by its own positive prediction** (`CHUNK_MAX=10` moves the
    remainder to layer 30 and predicted a break there; it broke nowhere, and so
    did a repeat of the unchanged baseline). H2 refuted too: all twelve row-14
    and row-15 s8×s8→s32 contractions are **bit-exact** against an int64 numpy
    reference on both stacks, and the qwix pattern at 8B widths matches CPU.
    H5 stays refuted (gate-5 probe, and the collapse is nowhere near the
    logits stage).
    **The witness that settles provenance — the 8B canary, re-run for the first
    time since 2026-08-03.** `notes/data/qwen3_8b_prefill_36layer.mlir` (36-layer
    8B **bf16** prefill, weights as arguments, no qwix, no int8, no checkpoint)
    still corrupts at today's shipped 800/512: **FAIL(5), max_abs_err 1.085e+04,
    max_norm_err 1.000e+00** — total signal loss on 5 of 15 outputs — and
    **bit-identically on the native and the Stage 1 engine**, reproduced across
    two passes. `METALJAX_COMPILE=0` drops it to FAIL(3) / 7.4e-02;
    `MLX_MAX_MB_PER_BUFFER=2048` to FAIL(3) / 4.1e-02 — the catastrophic face
    goes, the failure does not, so the raised budget is **not a fix and not a
    default** (2048 on a full 8B load is panic-#4 territory). This rung had been
    silently unrunnable: `scripts/run_stablehlo_bench.py` never registered the
    `chlo`/`sdy`/`mpmd` dialects, so the asset died at parse
    (`Dialect 'sdy' not found for custom op 'sdy.mesh'`) — fixed.
    **Provenance verdict: NOT a 0.11.3+ migration defect.** Stage 1's Python
    engine is wrong on this row today, on the same checkpoint through the same
    adapter, and more nondeterministically than the native one; its wrongness
    has the *shape* the 2026-08-03 note recorded (plausible-magnitude garbage,
    no collapse in ten draws), while native additionally reaches the collapsed
    face. The 0.11.2-src arm is **blocked by memory** (guard kill at 94 GB — it
    predates both `4d34bff` and the governor) and is no longer load-bearing,
    since the canary answers the same question for both engines at once.
    **Correction to the record**: `34f627c`'s "2118 (maxtext; coherent)" jax-CPU
    cell has **no artifact behind it** (a one-line STATUS edit; no log on disk
    carries jax-CPU text for this row), and re-measuring it tonight was
    guard-killed twice (60 GB, then 92 GB at an 89 GB footprint with the free
    list at 4.9 GB). Treat "jax-CPU is coherent on row 15" as an unverified
    2026-08-02 claim. Row 14's 10/10 determinism is what exonerates the
    harness, tokenizer, qwix path and conversion pipeline.
    **Open, both needing Oleg's go** (a full 8B maxtext load above row 15's
    79 GB envelope, i.e. the 67–109 GB panic-#5 band): the 0.11.2 arm and the
    jax-CPU reference. Also unattributed: a residual 4–7 % norm error on 3 of
    15 canary outputs that survives every arm including 2048 — possibly 36
    layers of legitimate bf16 drift against a 2 % tolerance, possibly a second
    smaller fault. No fix exists at our level; this goes to the upstream MLX
    report (`notes/mlx-command-buffer-upstream-issue.md`) as its strongest
    evidence yet.
35. **Row 15 is FIXED (2026-08-18) — the defect was MLX's, and MLX is now
    ours.** Footnote 34 is the investigation; this is the outcome. The
    corruption was located in MLX's own source: `compute_dynamic_offset`
    (`mlx/backend/metal/slicing.cpp:62`, v0.32.0) registers a **donated**
    dynamic-slice offset — an array that ALIASES a live graph buffer — as a
    command-encoder temporary, and `CommandEncoder::end_encoding()`
    (`device.cpp:442`) erases every temporary from `all_inputs_`/
    `all_outputs_`, which are the only input to MLX's cross-command-buffer
    fence bookkeeping. The erase deletes exactly the dependency that orders
    the producer of the start index against the kernel that reads it, so a
    `slice_update` lands at a **stale offset** whenever a command-buffer
    boundary falls between the two — a whole KV block, or a layer's
    parameters, written at the wrong position. Instrumented count on the 8B
    canary: 144 dropped fence waits in one run = 36 layers × 4 executions,
    with 0 write-after-read hazards (the other candidate hole never fires).
    Full derivation: `notes/mlx-patch-diagnosis.md`.
    **The fix ships.** metaljax vendors its own MLX, built from our fork at
    `vendor/0.32.0` (= upstream's own **unreleased** `7e8b4ccc` / PR #4099,
    cherry-picked, plus our generic `end_encoding` hardening), linked
    privately as `libmlx_metaljax.dylib` and carried inside the native
    wheel — so the fix does not wait on an MLX release.
    **Measured on the release binary `frozen-vendor-d651add3`**, row 15's
    own forensics, ten prefills of one loaded parameter set in one process
    plus a greedy decode (92 GB guard, 76 GB peak): **the same first token
    10 times out of 10** (12095, `" Paris"` — the token row 14 returns
    10/10), **zero collapses**, `logits_std` 2.292 where the collapse used
    to flatten the logits, and the decode reads **`" Paris. The capital"`**.
    Against the pre-vendoring native's 8 distinct first tokens and 2 full
    collapses. The row has emitted nothing but garbage since 2026-08-03, on
    every binary and both engines; it is now deterministic and coherent on
    the one that ships.
    The ms/tok cell stays unpublished: this was a correctness attestation,
    and a timing cell needs its own measured run (release rule 1). The two
    memory-blocked follow-ups in footnote 34 (the 0.11.2-src provenance arm
    and the jax-CPU reference) remain open and are no longer load-bearing —
    the verdict rests on determinism, row-14 agreement and coherent text.
    Corroborating, on the same build: `tests/test_command_buffer.py`'s five
    corruption canaries can no longer find a corrupting budget across 28
    budget settings and three sync-point layouts, and the 20-line pure-MLX
    reproducer `notes/data/mlx-cbuf-repro/repro_c.py` goes 0/20 wrong where
    the public wheel fails. Battery: `notes/mlx-vendoring-plan.md` §6.4.
36. **The 0.11.5 release column** (2026-08-18 consolidated re-gate; full
    report `~/.cache/metaljax-bench/logs/regate-0.11.5/models/`). Every
    cell measured in one campaign on the release binary
    `frozen-vendor-d651add3` (native plugin + vendored patched
    `libmlx_metaljax`, tree `29bb8eb`), 13:33–15:05, one guarded process
    per cell, machine lock held, historical budgets, no budget raised
    anywhere. Verdict: PASS — zero regressions >10 %, zero newly-failing
    rows. Token agreement PASS (12B / E2B-int4 / Qwen3-8B 64/64 vs CPU;
    gemma4-E2B and Llama-8B token-51 divergences are the pre-existing
    certified-benign bf16 tie-flips). Named items, per release rule 2:
    - **Row 1 variance**: the 237-class cell REPRODUCED (235.5, canonical
      chain), but a standalone confirm 23 min later read 296.9 → full
      disposition: 296.9 / 295.9 / 290.3, guard-at-5s 286.6 (guard
      sampling exonerated), unguarded 261.1 — falling monotonically toward
      fast over ~20 min of light load. Slow runs scale prefill AND decode
      by one uniform ~1.26× factor; recognizer emits (60 sdpa — the
      recognizer now FIRES on 31B, unlike the P24-era 0), peaks and loads
      identical fast vs slow; no OS thermal warnings, no background
      daemons. Best evidence: GPU thermal/DVFS machine state, oscillating
      inside the historical 237–302 span, vendoring-neutral (same-tree
      A/B public 292.3 / vendored 291.0 / public 285.7). A variance
      property of this row on this machine, not a regression; deeper
      attribution needs root (`powermetrics`).
    - **Rows 3 and 8 suite-context brackets**: +9–11 % inside the sweep,
      +0.2–0.3 % standalone (the CLAUDE.md item-12 trap). The standalone
      number is the cell; in-suite readings were 47.3 and 32.8.
    - **Row 10 spread + machine edge**: confirm run 1871.1; first
      completed sample 2017.9; one clean governor RESOURCE_EXHAUSTED at a
      13 GB-claimed baseline (the row needs a ~12 GB-clean machine — the
      no-panic contract's degrade-don't-panic behavior, working).
    - **Row 15 first honest timing**: 401.4 ms/tok (prefill 502.3 ms,
      load 79.3 s, 73 GB), decode coherent; 10/10 deterministic first
      tokens re-confirmed in the same campaign (footnote 35 has the fix).
    - **Row 18 named drift**: 370.7 vs the 359–362 cluster (+3.2 %),
      single in-suite sample inside the row's campaign spread — named,
      not absorbed.
    - **Row 19**: 460.2 ms/step with `loss` bit-identical to all eight
      prior campaign runs (the ninth identical run — the strongest
      numerics attestation the vendored MLX has).
    - **Row 20**: clean harness decline re-attested (exit 1, nothing
      executed, 0 GB touched).
37. **The 0.11.6 release column** (2026-08-27 release gate; full artifacts
    `~/.cache/metaljax-bench/logs/gate-0.11.6/`). Every cell from the
    release binary `frozen-0.11.6-dde2d668` (tree `83ff94f`), one guarded
    process per cell, machine lock held. Verdict: PASS — zero regressions
    standing (all in-sequence excursions adjudicated standalone), zero
    panics/wedges, two clean governor refusals (both documented below).
    **ALL 20 ROWS NOW PRODUCE NUMBERS.** Named items, per release rule 2:
    - **Rows 12 and 20, first release cells, at documented raised memory
      envelopes** (shipped defaults unchanged; Oleg-approved, set before
      the runs, never raised after a breach). Row 12: 91.3 ms/tok at
      `METALJAX_MEM_SYS_MB=METALJAX_MEM_BUDGET_MB=112640` (110 GiB), peak
      93 GB, 0 refusals. Row 20: 66.3 ms/tok at 110/114 GiB, load 320.5 s
      + ~104 min pack/build wave, peak 100 GB; its attempt 1 was a clean
      governor refusal at 118.8 GB claimed vs the 114 line — retried once
      on a 13 GB-clean baseline, envelope unchanged.
    - **Row 10**: attempt 1 clean refusal at a 16.7 GB machine baseline
      (the row needs ~13 GB-clean); sanctioned retry completed at 1948.2
      (+4.1% vs 0.11.5, attributed to governor pacing at the higher
      baseline; decode text identical to the 0.11.5 run).
    - **Row 1 variance** (standing footnote): canonical cell 235.2
      (0.999× the 0.11.5 cell, same measurement mode); the standalone
      confirm read 292.3, inside the known 261–297 machine-state band
      (fn 36). Bimodality unchanged; not a regression.
    - **Rows 3/8/17b**: cells are standalone values (43.3 / 29.4 /
      4974.9); in-sequence reads (46.7 / 32.2 / 6177.2) named as spread —
      the fn-36 suite-context class.
    - **Token agreement, first release under the GREEDY contract**
      (9f69ee8): **Qwen3-8B and Llama-3.1-8B are EXACT vs CPU, 64/64 —
      first release with exact matches**; the certified-benign divergence
      list tightens to {gemma4-E2B-bf16} (token 51). gemma4-E2B-int4's
      agreement is VACUOUS (both backends emit the same degenerate
      single-token loop — a keras int4 PTQ artifact, not a metaljax bug).
    - **The 0.11.5 release-note item "fused-attention recognizer
      nondeterminism on rows 5/7" is REFUTED on this binary**: gpt-oss-20b
      3/3 independent-process greedy draws token-identical, Qwen3-8B 2/2
      and CPU-exact, recognizer emits ON. The 0.11.5 evidence was top-k(5)
      sampling with a per-process random seed (see 9f69ee8); the
      `METALJAX_RECOGNIZE=0` workaround was unnecessary.
    - **Row 15**: 381.7 ms/tok (0.95× the 0.11.5 cell) with the full
      forensics re-pass: 10/10 first-token 12095, 0 collapses, coherent.
    - Row 17 @1024²: 4974.9 = 0.87× — an improvement with its 4975–6177
      spread named (fn 36 class).
38. **Row 10 post-0.11.6 fix (HEAD, ships next release)**: the 184× gap
    was jax's `lax.ragged_dot` DENSE fallback — maxtext's non-Pallas MoE
    path contracts every token against all 64 experts at padded width
    (~910× inflated FLOPs). A new engine-side recognizer
    (`metal_ragged.cc`, commit 6590c05) rewrites the fingerprint to one
    `gather_mm` over the real rows: decode **1948.2 → 113.9 ms/tok**
    (17.1×), prefill 2028 → 186 ms, decode text bit-identical, rows
    11/14 unregressed. The release column above keeps the 0.11.6-gated
    1948.2 per release rule 1; this table shows the current (HEAD)
    number. Remaining gap
    to mlx-lm's 10.5 is launch-bound decode (the ~16 ms maxtext-harness
    floor is row 11's entire cell), owned by the decode-loop fusion
    campaign. Diagnosis: notes/row10-ragged-dot-2026-08-28.md.
    PHASE 2 (2026-08-29, ff569eb): stacked-weight dots read the layer
    stack IN PLACE via gather_mm (zero-copy strided view) and K=1 dots
    become broadcast multiplies (bit-identical) — rows 10/11/14 and the
    gemma4-E2B control now 108.7 / 13.28 / 31.08 / 25.8. The floor is
    PROVEN to be graph op count (~20k nodes/token; whole-main
    compilation changes nothing), not protocol (<1 ms/token) — the <50
    target is an MLA-attention/router fusion campaign. Row 11's mlx-lm
    ground truth: 3.0 ms/tok bf16. Row 11's greedy stream moved at
    token ~9 with the reduction-order change (the documented tie-flip
    class; rows 10/14 streams unchanged). Row 10's protocol now pins
    METALJAX_MEM_SYS_MB=107520 (documented envelope; default 96 GB sits
    under the row's budget). notes/row10-decode-floor-2026-08-29.md.
