# Model benchmark suite — status

*(Harness and manifest: [scripts/model_bench/](scripts/model_bench/).)*

Headline metric per cell: LLM rows = warm decode ms/token; vision =
forward ms; diffusion = ms/step; training = ms/step. ✗ = established
impossible (with the measured reason). metaljax cells are the current
release's gate values (release rule 1: every release cell from that
release's binary).

Footnote numbering is stable across cleanups — gaps are deliberate
(historical footnotes were removed 2026-09-01; git history holds them
verbatim, and each release's full record lives in
`notes/release-gates-<version>.md`).

| # | benchmark | jax CPU | metaljax (current ³⁹) | mlx-lm | torch-MPS | llama.cpp |
|---|---|---|---|---|---|---|
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **126.1** ³⁹ (63 GB) | 133.1 ³⁹ | 148.7 | 111.2 ²⁰ |
| 2 | gemma4-12B bf16 | 315.2 (f32) ³⁹ | **57.3** ³⁹ (26 GB) | 58.3 ¹⁵ | 67.6 | 44.2 ²⁰ |
| 3 | gemma4-26B-A4B (MoE) | ✗ guard-killed @34 GB ¹⁴ | **33.4** ³⁹ | **17.0** | — | 16.9 ²⁰ |
| 4 | gemma4-E2B bf16 | 67.5 (bf16→f32) ¹³ ³⁹ | **24.0** ³⁹ | 10.5 ¹⁵ | — | — |
| 5 | Qwen3-8B bf16 | 207.0 (bf16→f32) ¹³ ³⁹ | **42.0** ³⁹ (17 GB) | 30.4 | 38.1 | 29.6 ²⁰ |
| 6 | Llama-3.1-8B bf16 | 203.6 (bf16→f32) ¹³ ³⁹ | **42.2** ³⁹ | 29.4 | 35.5 | 29.2 ²⁰ |
| 7 | gpt-oss-20b | ✗ ⁴ | **19.8** ³⁹ | **8.8** (13.8 GB, native MXFP4) | — | 6.7 (native MXFP4) ²⁰ |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | **28.5** ³⁹ (73 GB) | **13.7** | — | 15.3 ²⁰ |
| 9 | R1-Distill-32B | ✗ 131 GB | **190.8** ³⁹ (67 GB) | 131.8 | — | 114.9 ²⁰ |
| 10 | DeepSeek-V2-Lite (maxtext) | ✗ needs 50–105 GB ⁶ | **24.8** ³⁹ (92 GB) | 10.5 | — | 10.7 ²⁰ |
| 11 | Qwen3-0.6B (maxtext decode) | 89.7 | **12.33** ³⁹ | 3.0 | — | 3.4 ²⁰ |
| 12 | Mixtral 8×7B bf16 | ✗ | **85.6** ³⁹ (90 GB) | **52.8** (93.4 GB) | — | — |
| 13 | gemma4-E2B keras-int4 (packed) | **67.8** ¹⁸ | **77.0** ³⁹ | — | — | — |
| 14 | maxtext qwix-int8 0.6B | 143.4 | **29.88** ³⁹ | — | — | — |
| 15 | *qwix-int8 Qwen3-8B* | 2118 | **388.4** ³⁵ ³⁹ (73 GB) | — | — | — |
| 16 | SigLIP 2 (fwd b1 ms) | 533 | **86.68** ³⁹ | — | 29.8 (b32: 591) | — |
| 17 | SD 3.5 Large (ms/diff-step) | ✗ ¹² | **1249.3** @512², **4961.6** @1024² ³⁹ | ✗ ¹⁹ | 654 @512², 2998 @1024² ¹⁹ | — |
| 18 | LoRA E2B train (ms/step) | 2048 | **362.1** ³⁹ | — | 135.6 ¹⁰ | — |
| 19 | maxtext train 0.6B (ms/step) | 1402 | **444.6** ³⁹ | — | — | — |
| 20 | *aspirational* 235B-A22B 3-bit | ✗ | **56.2** ³⁹ (101 GB) | **28.0** (102.9 GB, load 12 s) | — | — |

**mlx-lm gap band (same Metal library underneath — the optimization
target):** the 0.11.7 dense-band campaign closed the flagship: 31B is
**0.95×** mlx-lm (126.1 vs a same-week 133.1) and 12B reads under the
dated mlx-lm cell (57.3 vs 58.3 of 2026-08-03; mlx-lm 0.31.3 refuses
the cached checkpoint, so that arm could not be re-measured — fn 39).
Remaining band: Qwen3-8B 1.4× (42.0 vs 30.4), Llama 1.4× (42.2 vs
29.4), gpt-oss 2.3× (19.8 vs 8.8 — native MXFP4 both sides), MoE 2.0×
(33.4 vs 17.0), 3-bit 2.0× (56.2 vs 28.0). llama.cpp leads mlx-lm a
further ~1.25× on bf16 — the kernel frontier. metaljax prefill trails
~5×; load ~20–30×.

## Footnotes

4. Row 7 CPU: keras dequantizes the MXFP4-native repo to bf16 (~42 GB
   weights); the working set projects ~126 GB — established infeasible
   on a 128 GB machine.
6. Row 10 CPU: maxtext's sparse MoE path wants 50–105 GB for the
   prefill on CPU — never completes inside this machine's budget.
10. torch-MPS LoRA: MPS has no SDPA backward kernel (math fallback,
    verified by autograd node inspection). Loss series are not
    comparable across stacks (different preprocessing); step cost is
    the comparison.
12. Row 17 CPU: keras's mixed-precision layers request the F16_F16_F32
    dot algorithm, which XLA:CPU rejects (an accelerator contract).
13. CPU cells run what XLA:CPU supports: weights load bf16, matmuls
    upcast per-op (bf16→f32); the 12B row is full f32 (gemma-lib path).
14. Row 3 CPU: f32 26B is ~104 GB of weights alone, and the observed
    keras-CPU load inflation (2.9×) projects a ~150 GB peak; the guard
    killed the load once the growth trajectory made that conclusive.
15. Released mlx-lm 0.31.3 cannot run gemma4_unified (12B) or the
    E-series KV-sharing layout (E2B); those two cells are mlx-lm git
    main (2026-08-03 install).
18. Row 13: packed int4 stays packed on metaljax (2.7 vs 10.2 GB — the
    only sub-byte JAX path that keeps it), while XLA:CPU fuses the
    in-graph unpack into a small net win (67.8 vs 79.2 bf16) — which is
    why the CPU cell leads this row.
19. Row 17 comparators: torch via the ungated diffusers mirror
    (adamo1139/stable-diffusion-3.5-large-ungated @5d868ff; images
    verified at both resolutions). No ungated MLX path exists for
    SD3.5-Large (mflux is Flux-only; DiffusionKit's formats are
    gated) — that cell is closed as not-runnable.
20. llama.cpp build 221f0f63, `llama-bench -p 51 -n 128 -r 5`,
    all-Metal, reproduced within 4 % on two passes; per-provider GGUF
    pins in scripts/model_bench/README_llamacpp.md. Dense rows pin to
    439–555 GB/s effective bandwidth — the machine's kernel frontier
    (llama.cpp leads even mlx-lm ~1.25× on bf16).
    LIKE-FOR-LIKE RULE (Oleg, 2026-08-28): cross-framework cells appear
    only at the SAME precision as the metaljax cell — custom kernels
    are fair game, different quantization is not. Kept cells: bf16
    (rows 1/2/3/5/6/8/9/10/11, dtype-verified 2026-08-29 survey),
    native MXFP4 (row 7, both sides), 3-bit (row 20, both sides).
    Mixtral (row 12) is proven quant-only across all 9 publishing
    providers, so its llama.cpp cell is legitimately empty.
35. Row 15 history: the row emitted garbage from 2026-08-03 through
    0.11.4 — an MLX bug, not ours: `compute_dynamic_offset`
    (mlx/backend/metal/slicing.cpp:62, v0.32.0) registers a donated
    dynamic-slice offset as an encoder temporary and `end_encoding`
    erases temporaries from the fence bookkeeping, so a `slice_update`
    lands at a stale offset whenever a command-buffer boundary falls
    between producer and reader. Fixed in OUR vendored MLX fork
    (vendor/0.32.0 = upstream's then-unreleased PR #4099 cherry-picked
    + our end_encoding hardening); 10/10 deterministic first tokens and
    coherent decode ever since. Derivation: notes/mlx-patch-diagnosis.md;
    investigation record: notes/row15-wrong-output-2026-08-17.md.
36. The 0.11.5 release column: notes/release-gates-0.11.5.md (full
    report `~/.cache/metaljax-bench/logs/regate-0.11.5/models/`).
    Standing item still relevant: row 1 has a machine-state variance
    band (237–302 ms/tok, uniform prefill+decode scaling, thermal/DVFS
    class) — the cell is the canonical-chain value.
37. The 0.11.6 release column: notes/release-gates-0.11.6.md (full
    artifacts `~/.cache/metaljax-bench/logs/gate-0.11.6/`). First
    release where all 20 rows produce numbers; rows 12/20 ran at
    documented raised memory envelopes (shipped defaults unchanged);
    the greedy token contract dates here (9f69ee8).
38. The post-0.11.6 row-10 decode-floor campaign (1948.2 → 25.4 ms/tok,
    shipped in 0.11.7): ragged_dot dense-fallback recognizer (6590c05),
    stacked-dot/K=1 fixes (ff569eb), CSE + region constant folding +
    MLA-sdpa + rms_norm recognizers (37b0cee). Full records:
    notes/row10-ragged-dot-2026-08-28.md,
    notes/row10-decode-floor-2026-08-29.md,
    notes/row10-opcount-2026-08-29.md. Protocol pin: row 10 runs with
    `METALJAX_MEM_SYS_MB=107520`.
39. **The 0.11.7 release column** (2026-08-31/09-01 gate; full artifacts
    ~/.cache/metaljax-bench/logs/gate-0.11.7/, per-fix records in
    .../dense-band-{diag,norm,dot,kv,combined,logits}/findings.md).
    Every cell from the release binary frozen-0117-combined-c0ed1a10
    (tree e9c0728; the binary reproduces from the tree byte for byte,
    3×), one guarded process per row, machine lock, settle prechecks.
    What landed: the dense-band campaign — (a) middle-contracted
    dot_general as batched matmul, both operand slots
    (METALJAX_DOT_BATCHED); (b) the MLX gemv occupancy floor, fixed in
    OUR FORK (fix/gemv-occupancy → vendor/0.32.0 @ d4967fa9, vendored;
    a 2.1× tiling cliff at K>=8192, notes/patches/
    mlx-gemv-occupancy.diff); (c) the cache-append scatter as ONE
    mx::slice_update (interval-analysis bounds proof + guarded arm,
    METALJAX_SCATTER_APPEND, bit-exact); (d) the rms_norm recognizer
    extended to the gemma/keras spellings incl. weightless norms
    (METALJAX_NORM; 0→11 of 11 dense-band spellings fused) — plus the
    post-0.11.6 row-10 decode-floor work (fn 38) shipping here.
    Gate: jax pinned suite 28,073/129 — failure set id-for-id identical
    to 0.11.6; texmo_gate 106/106; topconfs fp32/bf16 within ~1% of the
    0.11.6 gate; 17 of 20 rows improved, 3 flat (15/17a/17b, each
    dispositioned standalone — suite-context class), ZERO regressions;
    row 20 first-attempt clean at its documented envelope, stream
    identical. Zero guard fires, zero refusals, zero panics all night.
    NAMED ITEMS: (i) rows 1/2/3 carry ACCEPTED tie-flips (Oleg,
    2026-08-31, on logit evidence): each a 1-bf16-ULP adjacent-code
    near-tie (row 2 idx 16 p=0.515/0.484, row 1 idx 45 a three-way
    0.36/0.31/0.25), attributed to the norm recognizer by same-binary
    METALJAX_NORM=0 A/B; row 2 thereby loses its 0.11.6 CPU-exact
    64/64 status (takes the other side of a tie CPU resolves the old
    way) — candidate for MODEL_TOKEN_KNOWN; rows 5/6 stay CPU-exact
    64/64. (ii) tc010-w17, one topconfs config of 223: a real +8.4%
    fp32 / +7.3% bf16 micro-regression (0.046 ms), reproducible, NOT
    attributable to the three campaign knobs (all off doesn't recover
    it) — prime suspect the MLX restage; disclosed per release rule 2 and
    signed off by Oleg (2026-09-01). (iii) Historical frozen dylibs now resolve
    the RESTAGED vendored MLX via absolute rpath — pre-0.11.7 binaries
    can no longer be re-run as baselines; comparisons are to recorded
    numbers. (iv) Row 19's 4-step loss is no longer bit-identical
    (86.888 vs 87.043; step-1 loss CLOSER to CPU than 0.11.6's) —
    trajectory amplification of the ULP class, not a correctness
    signal. (v) Row 2's mlx-lm comparator arm is blocked (mlx-lm
    0.31.3 refuses the cached gemma4_unified checkpoint); its 58.3
    cell is dated 2026-08-03; row 1's mlx-lm 133.1 is the same-week
    diagnosis re-measure. (vi) The MLX slice_update donation patch was
    measured a wash (donate branch taken 95% of calls, 0.0 ms, 0.0 GB)
    and DROPPED; archived in logs/dense-band-kv/. (vii) Row-1 prefill
    still gates at cost=20839 compile=0 (BlockCost prices the MLIR
    block; recognizers collapse the tape) — deferred, latency-only.
    (viii) A pre-existing multi-thread Stream(gpu,N) execute flake
    surfaced 4× under concurrent GPU processes (clean 20/20 serially,
    both binaries) — ticketed, environmental.
