"""MLX 0.32.0 -- mx.compile raises "[compile] Too many inputs/outputs fused
in the Metal Compiled primitive" on a graph whose kernel would fit.

Run:  python repro.py            (needs only mlx + numpy, and this dir's
                                  binomial_body.mlxfn)

The function is jax's `random.binomial` BTRS rejection-sampler loop body as
metaljax lowers it: 16 carried arrays + 12 captured scalars in, 16 out, all
elementwise (compare / select / log / arithmetic). It evaluates fine eagerly
and dies under mx.compile.

Where it comes from (mlx/backend/metal/compiled.cpp):

  build_kernel() counts, per generated kernel variant,
      non-constant inputs
    + 1 if any input is non-scalar and the variant is strided (in_strides)
    + outputs
    + 1 (output_shape, or size for the contiguous variant)
    + 1 if dynamic_dims (the _strided_dynamic variants)
  and throws when that exceeds 31.

  Compiled::eval_gpu builds the WHOLE library up front -- _contiguous,
  _strided_1 .. _strided_7, _strided_dynamic, and the _large twins -- so the
  most argument-hungry variant decides, even when the call would dispatch to
  a cheaper one. Here: 28 inputs + 1 output.
    _contiguous     28 + 1 + 1          = 30   fits
    _strided_N      28 + 1 + 1 + 1      = 31   fits
    _strided_dynamic  ... + ndim        = 32   throws
  and _strided_dynamic is only ever DISPATCHED for ndim >= 8, while every
  array in this graph is rank <= 1. So the failure comes entirely from a
  variant that cannot run.

  The fusion pass (mlx/compile.cpp) means to bound this: max_compile_arrays
  = 24. The bound is on `input_set` during the first traversal, but the
  Compiled primitive's real inputs are recollected by `recurse_tape` in a
  second pass with no cap -- every node that returned early on the cap
  becomes an input there. That is how this graph reaches 28.

Two things would fix it upstream, either alone:
  - build the _strided_dynamic variants lazily (or skip them when
    outputs[0].ndim() < 8, which is knowable);
  - enforce max_compile_arrays on the second pass, i.e. on the input list
    the kernel is actually built from.

metaljax works around it by evaluating the first call of each freshly
compiled while body synchronously and falling back to the uncompiled body
when the kernels cannot be generated (src/metaljax/ops/control.py,
_BodyRunner).
"""

import os

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "binomial_body.mlxfn")

SHAPES = (
    [((3,), mx.float32)] * 8
    + [((10,), mx.float32)] * 4
    + [((), mx.int32), ((3,), mx.float32), ((3,), mx.bool_),
       ((2,), mx.uint32)]
    + [((), mx.float32)] * 9
    + [((), mx.int32)] * 3
)

rng = np.random.default_rng(0)
args = []
for shape, dt in SHAPES:
    if dt == mx.bool_:
        args.append(mx.array(rng.integers(0, 2, size=shape).astype(bool)))
    elif dt in (mx.int32, mx.uint32):
        args.append(mx.array(rng.integers(1, 5, size=shape)).astype(dt))
    else:
        args.append(mx.array(rng.random(shape).astype(np.float32)))
mx.eval(*args)

fn = mx.import_function(PATH)

outs = fn(*args)
mx.eval(*outs)
print(f"eager: ok, {len(outs)} outputs")

g = mx.compile(fn)
try:
    outs = g(*args)
    mx.eval(*outs)
    print("compiled: ok")
except RuntimeError as e:
    print("compiled: RuntimeError")
    print(str(e)[:200])
