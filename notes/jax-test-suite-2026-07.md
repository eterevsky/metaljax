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
