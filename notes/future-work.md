# Future optimization ideas (as of 0.3 development, 2026-07)

Collected during the lrnn/mullstm passes; each is concrete and scoped.

## Codegen (msl_scan)

- **In-kernel gather** (db12's residual, ~7x vs CPU): a per-lane indexed
  read with a register-resident index — `inp[idx_reg * stride + ...]` is
  emittable in MSL; needs a SymGather node + bounds handling. Also shows
  up in one-hot embedding compositions (bits.4.oh inside fused loops).
- **Composite-model glue** (db17's residual, ~12x): with all four seq
  scans compiled, the per-step cost is inter-layer glue (concats,
  reshapes, dense layers) replayed in the compiled chunk graph. Options:
  fuse adjacent layer scans into one plan when trip counts match, or
  extend plans to cover elementwise pre/post-processing around the loop.
- **Batch-packing for small-F coop kernels**: at F=32/b4 a coop kernel
  occupies 128 threads; pack multiple batch elements per threadgroup
  (grid z) to fill the GPU.
- **Weight tiling for big-F coop kernels**: coop currently loses to the
  compiled matmul path above ~2.2M dot elements/step (each threadgroup
  re-streams the whole fused-gate weight matrix every timestep — gru/
  lstm.1024 measured 2-2.5x slower; hence METALJAX_MSL_COOP_CAP).
  Threadgroup-tiled weight reuse or simdgroup_matrix ops could push the
  crossover up and reclaim F>=1024 cells for the single-kernel path.
- **Suite-context timing noise for sub-ms models**: in a full-suite
  process, tiny configs (db03/db05 class, ~0.6 ms/step standalone) can
  measure ~2x slower after a hundred prior configs have run; standalone
  reruns match baseline. Suspects: kernel-cache growth (process-unique
  kernel names), buffer-pool state after clear_cache. Harmless for
  training, but worth understanding before trusting suite deltas <1 ms.
- **Hoisted-subgraph caching**: plan.run re-evaluates hoisted invariant
  IR per call; when the hoisted values depend only on weights (not on
  per-chunk inputs), cache by input buffer identity across calls.
- **Scalar-mode SymRedReg/lane reduces**: register reduces currently
  vector/coop-mode only.

## Engine

- **Prepared-closure interpreter rewrite**: kill MLIR-wrapper overhead on
  eager paths and trace time (biggest win for noscan mode and compile
  latency).
- **if/case branch compile**: branches still interpret eagerly.
- **CPU-side while for tiny trip counts**: sub-microsecond loops are
  cheaper unrolled on the host than replayed.

## Known upstream issues to track

- **Apple Metal shader compiler miscompiles multi-iteration loops**
  (worked around with a volatile loop counter, see notes 2026-07). The
  workaround costs ~1.4-1.7x on small-F kernels (db11/db14/db15 pay it;
  at coop F=1024 it is ~free) — the price of correctness, db10/db12 were
  order-1 wrong without it. Cheaper variants were tried and ALL still
  miscompile (verified with MJDBG_VERIFY_MSL): volatile per-iteration
  input loads only; t from a runtime identity table (opaque value); one
  volatile access copied to a plain register. Only a volatile access at
  every USE of t is correct — the bug is not (just) value-provenance
  reasoning. All variants remain selectable via METALJAX_MSL_VOLATILE
  (t/tmap/tv/load/0) for retesting on macOS/Metal updates.
- **Per-plan auto-verification to drop the volatile selectively**: build
  each kernel without the workaround, compare against the raw body on
  the first eager call (infrastructure exists in the MJDBG_VERIFY_MSL
  hook), and rebuild with volatile only on mismatch. Would reclaim the
  1.4x on the (majority of) bodies the compiler bug does not bite.
  Complication: plans first built inside an mx.compile trace cannot
  eval, so verification must defer to the first eager opportunity.
- **MLX mx.compile equal-constant-output collision** (worked around with
  where(x==x) anchoring): report upstream; minimal repro:
  `mx.compile(lambda x: (x+1, mx.array(.9), mx.array(.9)))`.
- **MLX buffer-count cache growth** (worked around with clear_cache at
  compile boundaries): a count-bounded cache upstream would be cleaner.

## Bigger arcs

- Stage 2: native engine (C++/Obj-C++, StableHLO parsed natively).
- sort, general reduce_window, partial-window scatter coverage.
- bf16 msl kernels (currently f32/f16/int only in generated kernels).

## Robustness

- **Kernel-build-failure fallback at run time**: the 31-binding limit is
  now guarded statically, but any *other* future "Unable to build metal
  library" error from a generated kernel inside an mx.compile'd main
  still surfaces only at execute time, where there is no fallback (the
  0.3.1 crash class). A catch there could blacklist the plan and
  re-trace the executable without it, like the compiled-while-body
  fallback.
- **MLX empty-matmul segfault** (worked around in linalg + to_np guards):
  `np.array(mx.matmul(mx.zeros((0,4)), mx.ones((4,2))))` segfaults on a
  null data pointer after successful eval (MLX 0.32; M=0 or N=0 outputs;
  zero-K is fine). Report upstream; found by jax/tests lax_numpy suite.
