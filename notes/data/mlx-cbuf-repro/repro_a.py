# repro_a.py -- only mlx and numpy
import os, sys
import mlx.core as mx
import numpy as np

MODE = os.environ.get("MODE", "compiled")     # "compiled" | "eager"
SHAPES = [((16,), "i"), ((1024,), "b"), ((1024, 8, 6144), "b"),
          ((1024, 8, 6144), "b"), ((6144, 8, 1024), "b"), ((1024, 8), "b"),
          ((1024, 8), "b"), ((1024, 8, 8, 128), "b"), ((128, 8), "b"),
          ((16, 8, 128, 1024), "b"), ((1024, 8, 16, 128), "b"),
          ((128, 8), "b"), ((1024, 8, 8, 128), "b"), ((2048, 1024), "b"),
          ((), "i")]

rng = np.random.default_rng(0)
args = []
for shape, kind in SHAPES:
    if kind == "i":
        args.append(mx.array(rng.integers(0, 4, size=shape).astype(np.int32)))
    else:
        x = (rng.standard_normal(size=shape) * 0.1).astype(np.float32)
        args.append(mx.array(x).astype(mx.bfloat16))
mx.eval(*args)

fn = mx.import_function("qwen3_prefill.mlxfn")
if MODE == "compiled":
    fn = mx.compile(fn)

def run():
    outs = fn(*args)
    mx.eval(*outs)
    return [np.array(o.astype(mx.float32)) for o in outs]

runs = [run() for _ in range(3)]
for i in (1, 2):
    for j, (a, b) in enumerate(zip(runs[0], runs[i])):
        if not np.array_equal(a, b):
            print(f"call 1 vs call {i+1}: output {j} {a.shape} differs, "
                  f"max |diff| {np.nanmax(np.abs(a - b)):.4g}")
