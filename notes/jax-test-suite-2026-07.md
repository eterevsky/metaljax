# Full jax/tests suite vs metaljax (2026-07-27)

Setup: all 164 files under `jax/tests/` (clone @ jax 0.11-era HEAD), one
pytest process per file, `JAX_PLATFORMS=metal,cpu` (metal default), f64
strict, x64 off. Logs + CSV: session scratchpad `jaxtests/`.

## Totals

**22,543 passed / 5,727 failed / 6,935 skipped / 36 errors** across
28,306 executed tests → **79.6% pass**. Every file runs to completion.
3 files can't import (clone-vs-installed version skew: `hijax_test`,
`hypothesis_test_util_test`, `state_test` need `jax._src` internals newer
than installed jax 0.11).

Two crash classes were found and fixed during the run (commit a57ca6a):
- MLX empty-matmul segfault at host copy (M=0/N=0 outputs, null data
  pointer; MLX 0.32 bug — worked around in dot_general/dot + to_np
  guard). Had killed 5 whole test processes.
- Counted-loop analyzer KeyError when the while bound is carried in loop
  state (lbfgs's maxiter); now recognized as counted iff the body
  forwards it unchanged, else dynamic fallback.

## Failure taxonomy (agent-classified over the 14 worst files, 4,055 of
5,727 failures; remainder spread thin across 100+ files, same classes)

| class | count | notes |
|---|---:|---|
| complex dtype | 1,594 | biggest by far: fft, complex linalg, complex variants everywhere. MLX has complex64 — a mapping project, not a bug |
| sort | 675 | argsort/partition/median/percentile/unique/setops all lower to stablehlo.sort |
| general reduce / reduce_window | 377 | non-standard reduce bodies (variadic min-with-index, and/or), windowed reductions, cum* via reduce_window |
| scatter windowed/exotic | 220 | window-on-indexed-dims, partial windows, non-add bodies |
| LAPACK custom_calls | 216 | Qr, Eigh, Lu, SVD, TriangularSolve, ApproxTopK |
| convolution | 214 | stablehlo.convolution unimplemented |
| numeric mismatch | 244 | most are REAL bugs, see below — not f32 noise |
| rng_bit_generator | 82 | DEFAULT algorithm path |
| popcnt / count_leading_zeros | 74 | trivial elementwise adds |
| version skew / misc harness | ~330 | test clone newer than installed jax; multi-device; PJRT edge APIs (UnsafePointer, executable serialization) |
| i4 / f8 dtypes | 31 | no MLX support |

## Genuine bugs found (consolidated root causes, prioritized)

1. **Scatter: out-of-bounds indices written (clamped) instead of
   dropped** — XLA semantics require dropping. One root cause explains
   ~90+ failures: nonzero/argwhere/triu_indices/where "last index comes
   back 0" cluster (~56 tests), bincount/place/histogram garbage in last
   slot (~13), sparse BCOO fromdense/todense losing an element, empty-
   sparse construction crash. Fix: mask OOB updates in the scatter
   handler.
2. **gather/scatter_add VJP wrong on the simplest 1-D case**
   (lax_autodiff testGatherGrad0) and sparsify duplicate-accumulation
   oddities (values halved). NB the plain case is fine — verified
   `zeros(3).at[[1,1]].add(5)` → [0,10,0] — so this is some other path
   (possibly the OOB root of (1), possibly a mode flag like
   unique_indices); needs a dedicated look.
3. **Reduce over zero-size axes crashes** instead of returning the init
   value (~45+ tests: jnp.max/min with initial=, empty linalg.norm,
   softmax on empty arrays, sparse empties). Guard: empty reduce dim →
   mx.full(init).
4. **Gather on zero-size slice results crashes** (shape_poly empty
   slices, x[-2:-4]).
5. **argmax/argmin NaN semantics**: NaN should win (MLX skips NaNs).
6. **searchsorted with NaNs**: comparisons don't follow XLA total-order;
   entries at/after NaN return len(a).
7. **bitcast_convert crashes** on size-changing element types and on
   rank-0 operands (shape.py mx.view); **stablehlo.reverse crashes on
   rank-0** (byteswap lowering).
8. **buffer_from_host rejects negative-stride numpy arrays** (flipped
   views) — should ascontiguousarray-copy. 10 ConvTranspose tests die at
   transfer.
9. Small special-value stuff: sign(NaN)→0 on bf16; expm1 ~5e-6 rel (over
   jax's 1e-6 rtol); sinc(±inf)→nan. Subnormal flush-to-zero
   (jnp.spacing) is Metal hardware FTZ — wontfix.
10. MLX bug: zero-size uint32 sum/prod aborts (missing
    init_reduce_sum uint32 kernel) — report upstream with the empty-
    matmul segfault.

## Suggested order of attack

Scatter OOB+duplicates (1,2) is the top correctness item — real training
code can hit it (embedding grads with OOB/padding tokens!). Then empty-
reduce/gather guards (3,4) — cheap, kill ~50 failures. Then the small
crashes (7,8) and NaN semantics (5,6). Complex dtype support is the
single biggest pass-rate lever (+~1,600) if ever worth doing; sort via
mx.sort/argsort covers another ~675.

## Follow-up fixes and features (same date, post-report)

Bugs fixed (order of severity, all with regression tests):
1. Scatter OOB drop semantics (the nonzero/bincount/sparse cluster).
   Arithmetic combiners neutralize dropped updates with the combiner
   identity; scatter-set redirects to a dummy slot (write-back-current
   would race genuine duplicate writes at the clamped slot).
2. Zero-size reduce -> init value; zero-size gather/scatter
   short-circuits.
3. bitcast_convert size-changing/rank-0; reverse rank-0/empty dims.
4. argmax/argmin NaN-wins (first-NaN index); TOTALORDER compare via
   radix key (fixes searchsorted with NaNs).
5. Plugin accepts negative-stride host buffers (flipped numpy views).

Features added:
- stablehlo.sort: generic comparator recognizer (key-chain evaluation +
  structural symmetry check) -> stable mx.argsort on a total-order key.
  chlo.top_k too.
- stablehlo.convolution -> mx.conv_general (all layouts, groups, batch
  groups, dilations, flip; float only).
- Scatter windows on indexed dims via index expansion (unique/polyadd/
  dus-style patterns; partial windows on free dims).

## 0.4.0 final (2026-07-27 evening)

Full 164-file rerun after the feature push:
**26,937 passed / 1,266 failed / 6,957 skipped -> 95.4% of executed**
(from 19,228/4,244 = 79.6% at the first run, 83.7% after the bug batch).
texmo gate 104/104; mid08/big15 perf at 0.3 baseline.

Remaining 1,266 by bucket: multi-device/pmap (~150), rng_bit_generator
+ rbg PRNG streams (~140), sparse deep semantics (~145), version-skew
+ export harnesses (~160), shape_poly symbolic harness residue (~114),
dtypes i4/f8 (~67), Schur/Hessenberg/tridiagonal + complex128 (~46),
misc exotics (>3-D convs, variadic corner cases). Everything above was
either declared out of scope (platform constraints) or measured as
poor ROI; nothing known-fixable remains at meaningful count.

## CPU-parity policy (Oleg, 2026-07-28)

Decision framework: JAX-CPU parity is the bar. Every metal-failing test
was rerun on the CPU backend (scratchpad jaxtests/cpu_parity.json):
**1,094 of 1,181 pass on CPU (parity targets); 87 fail on CPU too
(best effort)**. Explicit decisions:
- bf16: support EVERYTHING (beyond CPU parity — most common format).
  DONE for linalg: own metal lowerings for eigh/svd/eig emit
  metaljax_* custom calls accepting all dtypes; host handlers upcast
  halves to f32. bf16/f16 eigh/svd/qr/cholesky now work where CPU
  raises NotImplementedError.
- int4/uint4/float8: emulate (CPU supports create/add/convert; f8 even
  matmuls; i4 matmul fails on CPU too). Plan: i4/u4 as int8 with mod-16
  wrap; f8 as byte patterns, 256-entry table for up-conversion, RNE
  bit-twiddle for down-conversion. CPU fallback acceptable if needed.
- rng_bit_generator: REQUIRED (CPU implements it; 48+46 rbg tests are
  parity targets). Implement Philox4x32 matching XLA CPU.
- Single-device collectives (pmap 53), multi-key lexsort (~225),
  ApproxTopK (ann_test 32), debug callbacks (29), Schur/Hessenberg,
  export/shape_poly residue: parity targets, in the 0.5 campaign.
- The 87 cpu-fail tests (mostly random_lax distribution exactness,
  api_test internals): best effort / wontfix.

## 0.4.1 final (2026-07-28)

Full 164-file rerun: **27,304 passed / 898 failed / 6,191 skipped ->
96.7%** (0.4.0: 95.4%; first run: 79.6%). texmo gate 104/104;
mid08/big15 perf at baseline. Parity batch 1 zeroed the sparse family,
ann_test, array_extensibility, lax_scipy; pmap 53->3, setops 38->1.
Remaining ~898: rng_bit_generator/rbg (~190), i4/f8 dtypes (~100),
callbacks/tokens (~82), lax_test exotics (~137), eigh/lobpcg/
scipy_signal pockets (~65), export/version-skew/multi-device tails.

## Final audit + gap-closure campaign (2026-07-28/29, pre-0.4.2)

Projects landed since 0.4.1 (each verified vs CPU, most bit-exact):
Philox + ThreeFry rng_bit_generator; i4/u4/f8* dtype emulation;
host callbacks (debug.print/debug_callback/pure_callback/io_callback);
single-device collectives (pmap/shard_map); multi-key lexicographic
sorts; ApproxTopK; Schur/Hessenberg/tridiagonal(+solve incl.
perturb_singular); bf16/f16 linalg via own custom-call lowerings
(exceeds CPU); shape-polymorphic export of those custom calls
(result_shapes); XLA shift-overflow semantics; QR/SVD orthonormal
completion for full_matrices; generic reduce_precision; grouped 0-D
convs; dilated variadic reduce_window; and the 0.3.2 livelock triple
fix (bounded retries, pessimistic cost for unknown trips, accdot
unit-squeeze bug -> 67x on composite b1 specs).

Explicit dispositions for the remaining tail (16-agent audit over
every failing file, consolidated):
- intentional-unsupported (~33): f64/complex128 compute (lobpcg's 27
  are all f64), multi-device, denormals. CLOSED by policy.
- version-skew (~77) + harness-mismatch (~76): test checkout newer
  than jax 0.11 (hijax, gamma(method=), jax._src.hypothesis_test_util)
  and infrastructure that doesn't know the platform (skip-lists,
  export platform allowlists, compilation-cache/log internals,
  x64_context). CLOSED: not ours; shrink when the jax pin advances.
- best-effort residue (each with documented cause):
  * complex64 special-value semantics at inf/nan poles (~24): MLX
    kernel edge semantics; fixing means custom complex kernels.
  * token plumbing for ordered effects (~24): interpreter treats
    tokens as sentinels; full ordered-effect token threading through
    main signatures not implemented.
  * scatter-apply with duplicates AND windows (~13): CLOSED post-0.4.4.
    Only the update-batch axis has to be applied sequentially — within
    one update the window elements land on distinct operand slots, so
    the window axes stay vectorized (ops/gather.py builds the same
    start+arange index arrays the vectorized path uses, with the batch
    dims flattened to one leading axis). ops/control._scatter_cost now
    charges the nb_-fold expansion so the trace budget stays honest.
    Fixes lax_vmap_test testScatterApply 0/2/3/4/5/6/9 and lax_test
    testScatterApply 0/1/6/7/8/9. NB lax_test testScatter1 is flaky
    on metal at ~30% independent of this change (scatter-SET with
    duplicate overlapping windows is order-nondeterministic on GPU,
    same class as scatter-add).
  * int4/uint4 bitcast_convert (7): FIXED after 0.4.4 — the handler now
    packs/unpacks nibbles (2 per byte, low nibble first) around the
    byte-storage emulation, so i4/ui4 bitcasts to and from any
    byte-multiple type match XLA. Also cleared lax_numpy testView
    1/5/6/8. Bitcasts of f8*/f4/f64 now raise UnsupportedOpError
    (storage width != logical width) instead of silently producing
    wrong bits.
  * window-dilation numeric corners under vmap (~7).
  * PJRT surface APIs (~40): UnsafePointer (MLX exposes no device
    pointers), buffer donation, pinned_host memory space,
    executable-text retrieval, compile-options validation.
  * singular tri/tridiag solves returning inf/nan instead of raising
    (5), NaN-placement in approx_top_k padding (1), scan-grad
    fixedpoint corner (1), holomorphic-grad tolerance (1).

## 0.4.2 final (2026-07-29)

**27,779 passed / 418 failed / 6,191 skipped -> 98.40%** (0.4.1:
96.7%; 0.4.0: 95.4%; first run: 79.6%). texmo gate 104/104;
mid08/big15 perf at baseline. Remaining 418 = ~33 intentional
(f64/complex128/multi-device/denormals) + ~150 version-skew/harness +
~235 audited best-effort (dispositions above).

## Parity campaign final (2026-07-31, post-0.4.4 -> pre-next-release)

Pinned jax-v0.11.0 suite (the honest headline; scripts/run_jax_tests.py
--tests jax-v0.11.0/tests): **27,649 passed / 130 failed = 99.53%**,
zero files regressed across the campaign (213 -> 130 on this suite; the
retired HEAD-clone suite went 489 -> 328 before most fixes landed).
Export-harness sweep: 5,460/5,587 pass; every non-pass attributed
(107 = XLA:CPU's own f16/bf16 linalg lowering gaps blocking the joint
(cpu,metal) artifact, 2 harness-invalid, 1 real gap -> fixed).

Fix chronology (each commit gated 104/104): ordered-effect tokens (25);
complex constructor + C99 special values (15); reduce_window strided
views (2, silent); top_k non-last axis (7, silent); msl lane-scalar
guard (66, silent since 0.2.0); msl concat pad width (12, silent);
expm1 accuracy (8); donation (15+, 21 jax tests unskipped);
UnsafePointer + no-alias contract (15); windowed scatter-apply (13);
i4 bitcast (11); f6/f4 emulation (13); conv grouped-int/neg-pad/
zero-size (13, incl. an uninit-memory overread); small fixes (13:
rank-0 slice, empty fft, complex sort ties, rng state, singular
solves); fft unit-length + async barrier (4); numerics (4: carry
aliasing PRIMAL bug, Kahan csqrt, %.7g constants, + 1 wontfix with
numbers); mechanical tail (21: c128, pointers, options, LU).

Silent-wrongness bugs fixed in metaljax (7): msl lane-scalar broadcast;
msl concat pad-total; msl sequential carry assignment (rotating carries
collapsed, all 3 emitters, since ever); top_k non-last axis (all
dtypes); dilated reduce_window reading stale device memory; complex
sort -0/+0 tie splitting; conv short-buffer overread (uninitialized
memory, nondeterministic).

MLX 0.32 bugs found and worked around (7): reductions over strided
as_strided views read wrong elements; argsort/sort on non-contiguous
inputs; conv zero-size spatial dim returns a NARROWER buffer than
declared; rfftn/irfftn with unit last length silently drop leading-axis
transforms; FFT kernels race a pending async_eval of their input
(eager-after-jit stale reads; pure-MLX repro); mx.compile bakes rank-0
constants as %.7g literals (1 ULP on 67% of constants in every fused
kernel); complex sqrt textbook-formula cancellation at the negative
real axis. Plus: -0.0 literals lose their sign under mx.compile, and
f32->f16 astype canonicalizes NaN signs (both documented, worked
around where they mattered).

Remaining 130 + 35 collection errors, all classified: ~44 intentional
(f64/c128 compute, multi-device, denormals), ~44 export-harness
platform allowlist, ~35 collection (cache internals, import skew),
~4 better-than-CPU shape-poly cases, ~3 jax-side callback platform
allowlists, and singletons documented individually (sinc-at-inf CPU-
also-fails, FD-reference sign flip with the full number table,
OptimizedProgram debugging surface).
