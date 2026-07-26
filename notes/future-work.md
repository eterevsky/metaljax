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
  (worked around with a volatile loop counter, see notes 2026-07): if a
  future macOS/Metal update fixes it, drop the volatile (small perf
  upside, esp. big register-tail kernels).
- **MLX mx.compile equal-constant-output collision** (worked around with
  where(x==x) anchoring): report upstream; minimal repro:
  `mx.compile(lambda x: (x+1, mx.array(.9), mx.array(.9)))`.
- **MLX buffer-count cache growth** (worked around with clear_cache at
  compile boundaries): a count-bounded cache upstream would be cleaner.

## Bigger arcs

- Stage 2: native engine (C++/Obj-C++, StableHLO parsed natively).
- sort, general reduce_window, partial-window scatter coverage.
- bf16 msl kernels (currently f32/f16/int only in generated kernels).
