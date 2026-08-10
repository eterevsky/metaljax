"""Test harness: run a jitted function through the metaljax ENGINE and
compare against the JAX CPU backend.

Everything here goes through `metaljax.engine` -- compile_program,
buffer_from_host, execute, to_host -- which is the route the PJRT plugin
takes and the ONLY route the Stage 2 native tape hooks. A test that builds
an `Interpreter` and calls it directly runs the Python interpreter under
both engines: it never exercises the native path, and the ops it covers
never register as native declines, so the decline census reads clean for
ops nobody ever asked the tape about (`stablehlo.reverse` and the whole of
`convolution` hid in exactly that blind spot).
"""

import io

import jax
import numpy as np

import ml_dtypes

from metaljax import dtypes as mdt
from metaljax import engine


def lower_bytes(f, *args) -> bytes:
    lowered = jax.jit(f).lower(*args)
    buf = io.BytesIO()
    lowered.compiler_ir().operation.write_bytecode(buf)
    return buf.getvalue()


def to_buffer(x) -> engine.MetalBuffer:
    """`x` on the device, through the host-transfer path PJRT uses."""
    arr = np.asarray(x)
    if not arr.flags["C_CONTIGUOUS"]:
        # NB not `ascontiguousarray`: it promotes a 0-d array to rank 1, and
        # a rank-0 argument handed over as tensor<1xi32> changes what the
        # program computes (jnp.byteswap of a scalar then reverses a unit
        # axis and returns its input).
        arr = np.ascontiguousarray(arr)
    return engine.buffer_from_host(arr.tobytes(),
                                   engine._NP_TO_ENUM[arr.dtype],
                                   list(arr.shape), None, 0)


def from_buffer(buf: engine.MetalBuffer) -> np.ndarray:
    """`buf` on the host, in the dtype PJRT declares for it (the program's
    own result type -- an emulated dtype is narrowed here, exactly as it is
    when jax reads the buffer back)."""
    dt = engine._ENUM_TO_NP[buf.type_enum]
    return np.frombuffer(engine.to_host(buf), dt).reshape(buf.dims)


def execute_module(code, arrays=()):
    """Compile and execute one StableHLO module; returns (executable,
    output buffers). `code` may be module text or bytecode."""
    if isinstance(code, str):
        code = code.encode()
    ex = engine.compile_program(code, "mlir")
    return ex, engine.execute(ex, [to_buffer(a) for a in arrays])


def run_module(code, arrays=()):
    """Host outputs of one compile+execute of a StableHLO module."""
    _ex, outs = execute_module(code, arrays)
    return [from_buffer(o) for o in outs]


def run_metal(f, *args):
    return run_module(lower_bytes(f, *args), jax.tree.leaves(args))


def run_metal_device(f, *args):
    """Like `run_metal`, but the outputs as they sit on the DEVICE.

    The emulated dtypes (i4/f8/f6/f4) live in a wider storage dtype and only
    become their narrow selves in `to_host`; a test about the storage grid
    has to read them before that cast.
    """
    import mlx.core as mx

    _ex, outs = execute_module(lower_bytes(f, *args), jax.tree.leaves(args))
    mx.eval(*[o.data for o in outs])
    return [mdt.to_np(o.data) for o in outs]


def check(f, *args, rtol=1e-5, atol=1e-6):
    got = run_metal(f, *args)
    # Pin the reference to the CPU backend: under JAX_PLATFORMS=metal,cpu
    # the default backend is metal, and an unpinned reference would compare
    # this backend against itself — masking wrong values (found by the int4
    # subagent: its regression test passed against broken code that way).
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
