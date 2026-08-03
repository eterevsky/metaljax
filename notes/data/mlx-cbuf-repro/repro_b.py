# repro_b.py -- only mlx and numpy
import os, sys
import mlx.core as mx
import numpy as np

NCARRY = 27
SHAPES = [((28,), "u"), ((28, 2), "k"), ((28,), "u"), ((28, 2), "k"),
          ((28,), "u"), ((28, 2), "k"), ((), "i"),
          ((28, 1024, 3072), "f"), ((28, 1024, 3072), "f"),
          ((28, 3072, 1024), "f"), ((28, 1024), "f"), ((28, 1024), "f"),
          ((28, 1024, 8, 128), "f"), ((28, 128), "f"),
          ((28, 16, 128, 1024), "f"), ((28, 1024, 16, 128), "f"),
          ((28, 128), "f"), ((28, 1024, 8, 128), "f"),
          ((28,), "u"), ((28, 2), "u"), ((28,), "u"), ((28, 2), "u"),
          ((28,), "u"), ((28, 2), "u"), ((28,), "u"), ((28,), "u"),
          ((28,), "u"), ((), "one")]

rng = np.random.default_rng(0)
args = []
for shape, kind in SHAPES:
    if kind == "k":                      # PRNG keys: the only nonzero uint32 inputs
        args.append(mx.array(rng.integers(0, 2**32, size=shape, dtype=np.uint32)))
    elif kind == "u":
        args.append(mx.zeros(shape, dtype=mx.uint32))
    elif kind == "i":
        args.append(mx.array(np.int32(0)))
    elif kind == "one":
        args.append(mx.array(np.int32(1)))
    else:
        args.append(mx.zeros(shape, dtype=mx.float32))
mx.eval(*args)

body = mx.import_function("qwen3_init_scan.mlxfn")
captures = args[NCARRY:]

def run(flush_every):
    vals = list(args[:NCARRY])
    for i in range(1, 11):
        vals = list(body(*vals, *captures))
        if i % flush_every == 0:
            mx.eval(*vals)
    mx.eval(*vals)
    return vals

for j, (a, b) in enumerate(zip(run(1), run(5))):
    if not bool(mx.all(a == b).item()):
        d = mx.sum(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
        print(f"carry {j} {tuple(a.shape)} {a.dtype} differs: total |diff| {d}")
