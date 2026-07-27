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
