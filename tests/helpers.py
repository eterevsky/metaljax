"""Test harness: run a jitted function through the metaljax PJRT plugin and
compare against the JAX CPU backend.

Everything here goes through the REAL backend -- `jax.jit` executed on the
metal device, or `client.compile_and_load` for hand-written StableHLO
modules.  Since the Stage-1 retirement (0.11.6) that is the only route: the
in-process Python engine these helpers used to drive is gone, so what a test
exercises here is exactly what a user's program exercises -- device buffers,
PJRT compile/execute, the native tape.

The suite needs no JAX_PLATFORMS: the plugin registers at priority -1 (CPU
stays the default backend) and `jax.devices("metal")` initializes it on
demand.  `check` pins its reference to the CPU backend explicitly either way.
"""

import io

import jax
import numpy as np

import ml_dtypes


def metal_device():
    return jax.devices("metal")[0]


def lower_bytes(f, *args) -> bytes:
    lowered = jax.jit(f).lower(*args)
    buf = io.BytesIO()
    lowered.compiler_ir().operation.write_bytecode(buf)
    return buf.getvalue()


def execute_module(code, arrays=()):
    """Compile and execute one StableHLO module through the plugin.

    Returns (loaded_executable, outputs as device arrays).  `code` may be
    module text or MLIR bytecode -- the same encodings PJRT itself hands the
    plugin.
    """
    from jax._src.lib import xla_client as xc

    dev = metal_device()
    exe = dev.client.compile_and_load(code, [dev], xc.CompileOptions())
    outs = exe.execute([jax.device_put(np.asarray(a), dev) for a in arrays])
    return exe, outs


def run_module(code, arrays=()):
    """Host outputs of one compile+execute of a StableHLO module."""
    _ex, outs = execute_module(code, arrays)
    return [np.asarray(o) for o in outs]


def run_metal(f, *args):
    """Host outputs of `f`, jitted and executed on the metal device."""
    with jax.default_device(metal_device()):
        out = jax.jit(f)(*args)
    return [np.asarray(x) for x in jax.tree.leaves(out)]


def check(f, *args, rtol=1e-5, atol=1e-6):
    got = run_metal(f, *args)
    # Pin the reference to the CPU backend: in a metal-default configuration
    # an unpinned reference would compare this backend against itself,
    # masking wrong values (found by the int4 subagent: its regression test
    # passed against broken code that way).
    with jax.default_device(jax.devices("cpu")[0]):
        want = [np.asarray(x) for x in jax.tree.leaves(jax.jit(f)(*args))]
    assert len(got) == len(want), f"{len(got)} outputs vs {len(want)} expected"
    for i, (g, w) in enumerate(zip(got, want)):
        assert g.shape == w.shape, f"out[{i}]: shape {g.shape} != {w.shape}"
        assert g.dtype == w.dtype, f"out[{i}]: dtype {g.dtype} != {w.dtype}"
        if g.dtype == ml_dtypes.bfloat16:
            g = g.astype(np.float32)
            w = w.astype(np.float32)
        if np.issubdtype(g.dtype, np.inexact):
            np.testing.assert_allclose(g, w, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(g, w)
