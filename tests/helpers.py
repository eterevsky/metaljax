"""Test harness: run a jitted function through the metaljax interpreter and
compare against the JAX CPU backend."""

import io

import jax
import numpy as np

import ml_dtypes

from metaljax import Interpreter
from metaljax import dtypes as mdt


def lower_bytes(f, *args) -> bytes:
    lowered = jax.jit(f).lower(*args)
    buf = io.BytesIO()
    lowered.compiler_ir().operation.write_bytecode(buf)
    return buf.getvalue()


def run_metal(f, *args):
    interp = Interpreter(lower_bytes(f, *args))
    flat = jax.tree.leaves(args)
    mx_args = [mdt.to_mx(np.asarray(x)) for x in flat]
    outs = interp(*mx_args)
    return [mdt.to_np(o) for o in outs]


def check(f, *args, rtol=1e-5, atol=1e-6):
    got = run_metal(f, *args)
    want = [np.asarray(x) for x in jax.tree.leaves(jax.jit(f)(*args))]
    assert len(got) == len(want), f"{len(got)} outputs vs {len(want)} expected"
    for i, (g, w) in enumerate(zip(got, want)):
        assert g.shape == w.shape, f"out[{i}]: shape {g.shape} != {w.shape}"
        assert g.dtype == w.dtype, f"out[{i}]: dtype {g.dtype} != {w.dtype}"
        if g.dtype == ml_dtypes.bfloat16:
            g = g.astype(np.float32)
            w = w.astype(np.float32)
        if np.issubdtype(g.dtype, np.floating):
            np.testing.assert_allclose(g, w, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(g, w)
