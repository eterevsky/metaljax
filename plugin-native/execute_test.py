#!/usr/bin/env python
"""Differential test for the native PJRT plugin's executor (phase 2).

Every expression below is evaluated twice: once through the bazel-built
native plugin (`JAX_PLATFORMS=metal`, `METALJAX_PLUGIN_PATH` pointing at the
dylib) and once through jax on the CPU backend, in a subprocess of its own.
The CPU answer is the bar -- this is the same doctrine the Stage 1 suite runs
under -- so a case that disagrees is a failure here whatever the two engines
have in common.

Four kinds of check, in this order: the jitted CASES below; hand-written
StableHLO MODULES, compiled through both clients, for encodings jax's own
lowerings never produce; the DECLINES, which must name the op that stopped
the program; and the CONTRACTS (no-alias, host round-trips, threading, the
f64 policy).

Run it from the repo venv:

    plugin-native/../.venv/bin/python plugin-native/execute_test.py

Exit status is the number of failing cases (0 = all good).  A `--reference
<path>` invocation is how the child process computes the CPU side; nothing
else needs it.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

import ml_dtypes
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_DYLIB = _HERE / "bazel-bin" / "metal" / "libmetal_pjrt_native.dylib"


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------
#
# (name, fn, args, rtol, atol).  Integer and boolean results are compared
# EXACTLY (the tolerances are ignored for them): a gather of bits that is one
# ULP out is a bug, not rounding.


def _rand(shape, seed, dtype=np.float32):
    return np.asarray(np.random.RandomState(seed).standard_normal(shape),
                      dtype=dtype)


def _randint(shape, seed, dtype=np.int32, lo=-5, hi=6):
    return np.asarray(np.random.RandomState(seed).randint(lo, hi, shape),
                      dtype=dtype)


def _crand(shape, seed):
    return (_rand(shape, seed) + 1j * _rand(shape, seed + 1000)).astype(
        np.complex64)


# Operands for the linalg family (P9).  Built rather than drawn so the
# conditioning is known: an eigendecomposition of a near-singular draw would
# measure the RNG's luck rather than the factorization.
def _spd(n, seed, dtype=np.float32):
    a = _rand((n, n), seed).astype(np.float64)
    return (a @ a.T + n * np.eye(n)).astype(dtype)


def _sym(n, seed):
    a = _rand((n, n), seed)
    return (a + a.T).astype(np.float32)


def _herm(n, seed):
    a = _rand((n, n), seed) + 1j * _rand((n, n), seed + 1)
    return (a + a.conj().T).astype(np.complex64)


def _cspd(n, seed):
    a = (_rand((n, n), seed) + 1j * _rand((n, n), seed + 1)).astype(
        np.complex128)
    return (a @ a.conj().T + n * np.eye(n)).astype(np.complex64)


def _triangular(n, seed, lower=False):
    a = _rand((n, n), seed) + 3 * np.eye(n, dtype=np.float32)
    return np.asarray(np.tril(a) if lower else np.triu(a), np.float32)


def _switch_case(i, x):
    """`lax.switch` over three branches, including two out-of-range indices.

    XLA clamps a case index into range; the executor does the same
    (native/control.cc), so 7 must run the last branch and -3 the first.
    A named function rather than a lambda only because the branch list would
    otherwise be spelled out once per index.
    """
    import jax
    import jax.numpy as jnp

    branches = [lambda a: a + 1, lambda a: a * 2, lambda a: -a]
    return tuple(jax.lax.switch(i[k], branches, x) for k in range(5))


# --------------------------------------------------------------------------
# msl_scan cases (P21)
# --------------------------------------------------------------------------
#
# One counted loop per generated-kernel MODE, plus the AD-generated backward
# passes -- which is where loop fission runs (hidden per-step stacks out of the
# kernel, one batched matmul after it).  Written as functions rather than
# lambdas so the body reads like the cell it is.


def _msl_mingru(h0, xs, wz, wh):
    """A pure elementwise cell: `scalar` (affine) mode, one thread per lane."""
    import jax
    import jax.numpy as jnp

    def step(h, x):
        z = jax.nn.sigmoid(x * wz)
        h = z * h + (1.0 - z) * jnp.tanh(x * wh)
        return h, h
    return jax.lax.scan(step, h0, xs)


def _msl_mingru_grad(h0, xs, wz, wh):
    """...and its backward pass, whose loop runs in reverse (idx a = -1)."""
    import jax
    import jax.numpy as jnp

    def loss(wz, wh):
        _, hs = _msl_mingru(h0, xs, wz, wh)
        return jnp.sum(hs * hs * 0.5)
    return jax.grad(loss, argnums=(0, 1))(wz, wh)


def _msl_rnn(h0, xs, w):
    """A matvec cell.  Narrow: `vector` mode holds the feature dim in
    registers and unrolls the matvec in lane."""
    import jax
    import jax.numpy as jnp

    def step(h, x):
        h = jnp.tanh(x + h @ w)
        return h, h
    return jax.lax.scan(step, h0, xs)


def _msl_rnn_grad(h0, xs, w):
    """The weight gradient: a cross-lane dot per step, which cannot run per
    lane -- the kernel stacks its operands and one batched matmul finishes the
    job (msl_scan's loop fission)."""
    import jax
    import jax.numpy as jnp

    def loss(w):
        _, hs = _msl_rnn(h0, xs, w)
        return jnp.sum(hs * hs * 0.5)
    return jax.grad(loss)(w)


def _msl_gru(h0, xs, wz, wr, wn):
    """Three gates over one state width: the coop-over-vector flip of 0.4.3
    (square dots, F >= 8) picks threadgroup mode for this."""
    import jax
    import jax.numpy as jnp

    def step(h, x):
        z = jax.nn.sigmoid(x + h @ wz)
        r = jax.nn.sigmoid(x + h @ wr)
        n = jnp.tanh(x + (r * h) @ wn)
        h = (1.0 - z) * n + z * h
        return h, h
    return jax.lax.scan(step, h0, xs)


def _msl_nested(h0, xs, w):
    """A statically-counted inner loop inside the scan: the analyzer unrolls
    it symbolically (trip <= 64) rather than declining."""
    import jax
    import jax.numpy as jnp

    def outer(h, x):
        h = jax.lax.fori_loop(0, 3, lambda i, c: jnp.tanh(c * 0.9 + x @ w), h)
        return h, h
    return jax.lax.scan(outer, h0, xs)


def _cases():
    import jax
    import jax.numpy as jnp

    f = np.float32
    # A tolerance band per class of arithmetic, not one global number.
    EXACT = (0.0, 0.0)
    F32 = (1e-6, 1e-6)
    # Contractions accumulate in a different order from the CPU's, so they get
    # a band of their own.  Not a WIDE one: the M5's low-precision matmul path
    # (CLAUDE.md's "M5 GPU MLX f32 matmul is low-precision", ~4e-3) is off
    # here, because src/jax_plugins/metal/__init__.py pins
    # MLX_METAL_GPU_ARCH before dlopening the plugin -- on the native branch
    # too.  Measured on 512x512: 7.6e-7 relative against an f64 reference,
    # where jax-CPU itself is 1.3e-6.
    DOT = (1e-5, 1e-5)
    HALF = (5e-3, 5e-3)

    # stablehlo.convolution's layouts, spelled the way jax spells them.  Every
    # one of them reaches the executor as three permutations and nothing else,
    # which is the whole point of testing more than one.
    conv = jax.lax.conv_general_dilated
    C1 = ("NCH", "OIH", "NCH")
    C2 = ("NCHW", "OIHW", "NCHW")
    C2L = ("NHWC", "HWIO", "NHWC")
    C3 = ("NCDHW", "OIDHW", "NCDHW")

    cases = [
        # milestone zero, exactly as CLAUDE.md states it
        ("2*x int32", lambda x: 2 * x, [np.array([1, 2, 3], np.int32)], *EXACT),
        ("elementwise chain f32", lambda x, y: jnp.tanh(x * y + x) - y / 3.0,
         [_rand((4, 5), 0), _rand((4, 5), 1)], *F32),
        ("unary mix f32",
         lambda x: (jnp.exp(x) + jnp.tanh(x) * jax.lax.rsqrt(jnp.abs(x) + 1.0)
                    + jnp.log1p(jnp.abs(x)) + jnp.floor(x) + jnp.sign(x)),
         [np.linspace(-3, 3, 32, dtype=f)], *F32),
        ("logistic/erf/sqrt f32",
         lambda x: jax.nn.sigmoid(x) + jax.scipy.special.erf(x) + jnp.sqrt(
             jnp.abs(x)),
         [np.linspace(-4, 4, 17, dtype=f)], *F32),
        ("elementwise chain f16", lambda x: jnp.tanh(x) * 2 - 1,
         [np.arange(12, dtype=np.float16).reshape(3, 4) / 8],
         *HALF),
        ("elementwise chain bf16",
         lambda x: (x * x + x).astype(jnp.float32),
         [np.arange(12, dtype=np.float32).reshape(3, 4).astype(jnp.bfloat16)],
         *HALF),
        ("integer arithmetic",
         lambda x, y: (x * y - x // 3 + y % 5, x & y, x | y, x ^ y, -x),
         [np.arange(-6, 6, dtype=np.int32),
          np.arange(1, 13, dtype=np.int32)], *EXACT),
        ("unsigned arithmetic", lambda x: x * 3 + 1,
         [np.arange(8, dtype=np.uint8)], *EXACT),
        ("bool logic", lambda a, b: (a & b, a | b, ~a, a ^ b),
         [np.array([True, False, True, False]),
          np.array([True, True, False, False])], *EXACT),

        # reductions
        ("sum over one axis", lambda x: x.sum(0),
         [np.arange(12, dtype=f).reshape(3, 4)], *F32),
        ("sum over two axes", lambda x: x.sum((0, 2)),
         [_rand((2, 3, 4), 2)], *F32),
        ("max/min/prod", lambda x: (x.max(1), x.min(0), jnp.prod(x, 1)),
         [_rand((3, 4), 3)], *F32),
        ("any/all", lambda x: ((x > 0).any(1), (x > 0).all(0), (x > 0).any()),
         [_rand((3, 4), 4)], *EXACT),
        ("sum of everything", lambda x: x.sum(),
         [_rand((5, 6), 5)], *F32),
        ("argmax / argmin", lambda x: (jnp.argmax(x, 1), jnp.argmin(x, 0)),
         [_rand((3, 4), 6)], *EXACT),
        ("argmax with a NaN", lambda x: jnp.argmax(x),
         [np.array([1.0, np.nan, 3.0], f)], *EXACT),
        ("integer sum", lambda x: x.sum(1),
         [np.arange(12, dtype=np.int32).reshape(3, 4)], *EXACT),

        # shape ops
        ("transpose + reshape", lambda x: x.T.reshape(4, 3),
         [np.arange(12, dtype=f).reshape(3, 4)], *EXACT),
        ("transpose rank 3", lambda x: jnp.transpose(x, (2, 0, 1)),
         [_rand((2, 3, 4), 7)], *EXACT),
        ("broadcast", lambda x, y: x + y[:, None],
         [_rand((3, 4), 8), _rand((3,), 9)], *F32),
        ("broadcast unsorted dims",
         lambda x: jnp.broadcast_to(x.T[None], (2, 4, 3)),
         [np.arange(12, dtype=f).reshape(3, 4)], *EXACT),
        ("strided slice", lambda x: x[1:4, ::2],
         [np.arange(24, dtype=f).reshape(4, 6)], *EXACT),
        ("concatenate", lambda x, y: jnp.concatenate([x, y], 0),
         [np.ones((2, 3), f), np.zeros((1, 3), f)], *EXACT),
        ("concatenate axis 1", lambda x, y: jnp.concatenate([x, y], 1),
         [np.ones((2, 3), f), np.zeros((2, 2), f)], *EXACT),
        ("iota", lambda x: x + jnp.arange(4, dtype=jnp.float32),
         [np.zeros((3, 4), f)], *EXACT),
        ("pad", lambda x: jnp.pad(x, ((1, 2), (0, 1))),
         [np.arange(6, dtype=f).reshape(2, 3)], *EXACT),
        ("pad with interior dilation",
         lambda x: jax.lax.pad(x, np.float32(0), ((0, 0, 1), (1, 1, 2))),
         [np.arange(6, dtype=f).reshape(2, 3)], *EXACT),
        ("pad with negative edges (a crop)",
         lambda x: jax.lax.pad(x, np.float32(9), ((-1, 0, 0), (0, -1, 0))),
         [np.arange(6, dtype=f).reshape(2, 3)], *EXACT),
        ("rank-0 scalar", lambda x: x * 2 + 1,
         [np.float32(2.5)], *F32),
        ("empty array", lambda x: x * 2,
         [np.zeros((0, 3), f)], *EXACT),

        # contractions
        ("matmul 2D", lambda a, b: a @ b,
         [_rand((4, 5), 10), _rand((5, 6), 11)], *DOT),
        ("batched matmul", lambda a, b: jnp.einsum("bij,bjk->bik", a, b),
         [_rand((2, 3, 4), 12), _rand((2, 4, 5), 13)], *DOT),
        ("dot with batch and free dims",
         lambda a, b: jnp.einsum("bik,bjk->bij", a, b),
         [_rand((2, 3, 7), 14), _rand((2, 5, 7), 15)], *DOT),
        ("vector dot", lambda a, b: a @ b,
         [_rand((16,), 16), _rand((16,), 17)], *DOT),
        ("matmul 128x128", lambda a, b: a @ b,
         [_rand((128, 128), 31), _rand((128, 128), 32)], *DOT),
        ("int32 matmul", lambda a, b: a @ b,
         [np.arange(12, dtype=np.int32).reshape(3, 4),
          np.arange(20, dtype=np.int32).reshape(4, 5)], *EXACT),
        ("int8 matmul", lambda a, b: (a.astype(jnp.int32)
                                      @ b.astype(jnp.int32)),
         [np.arange(-6, 6, dtype=np.int8).reshape(3, 4),
          np.arange(-10, 10, dtype=np.int8).reshape(4, 5)], *EXACT),

        # selection
        ("select / compare / clamp",
         lambda x: jnp.where(x > 0, jnp.clip(x, -1.0, 1.0), -x),
         [np.linspace(-3, 3, 21, dtype=f)], *F32),
        ("comparison ladder",
         lambda a, b: (a == b, a != b, a < b, a <= b, a > b, a >= b),
         [np.array([1.0, 2.0, 3.0, np.nan], f),
          np.array([1.0, 3.0, 2.0, 1.0], f)], *EXACT),
        ("maximum / minimum", lambda a, b: (jnp.maximum(a, b),
                                            jnp.minimum(a, b)),
         [_rand((7,), 18), _rand((7,), 19)], *F32),

        # constants and converts
        ("f32 constant", lambda x: x * jnp.array([1.5, 2.5, 3.5], jnp.float32),
         [np.ones(3, f)], *F32),
        ("bf16 constant",
         lambda x: (x * jnp.array([0.5, 1.25, -2.75], jnp.bfloat16)
                    ).astype(jnp.float32),
         [np.ones(3, np.float32).astype(jnp.bfloat16)], *HALF),
        ("splat constant", lambda x: x + 7.25,
         [np.zeros((3, 4), f)], *F32),
        ("bf16 splat constant",
         lambda x: (x + jnp.bfloat16(3.5)).astype(jnp.float32),
         [np.zeros((2, 3), np.float32).astype(jnp.bfloat16)], *HALF),
        ("bool constant",
         lambda x: x & jnp.array([True, False, True]),
         [np.array([True, True, True])], *EXACT),
        ("int constant", lambda x: x * jnp.array([2, 3, 4], jnp.int32),
         [np.ones(3, np.int32)], *EXACT),
        ("convert chain",
         lambda x: (x.astype(jnp.int32).astype(jnp.float32)
                    .astype(jnp.bfloat16).astype(jnp.float32)),
         [np.linspace(-5, 5, 9, dtype=f)], *EXACT),
        ("bool convert", lambda x: (x > 0).astype(jnp.float32),
         [np.linspace(-1, 1, 9, dtype=f)], *EXACT),

        # complex64: in the dtype table because its storage IS its bits
        ("complex arithmetic",
         lambda z: (z * z + z, jnp.abs(z), jnp.real(z), jnp.imag(z)),
         [np.array([1 + 1j, -2 - 3j, 0 + 0j, 4 - 0.5j], np.complex64)],
         1e-6, 1e-6),
        ("complex from parts", lambda a, b: jnp.abs(a + 1j * b),
         [_rand((5,), 20), _rand((5,), 21)], 1e-6, 1e-6),

        # program shapes
        ("several outputs", lambda x, y: (x + y, x - y, x * y),
         [_rand((3,), 22), _rand((3,), 23)], *F32),
        ("identity", lambda x: x, [_rand((4,), 24)], *EXACT),
        ("argument returned twice", lambda x: (x, x),
         [_rand((4,), 25)], *EXACT),
        ("constant output", lambda x: (x, jnp.arange(3, dtype=jnp.float32)),
         [_rand((4,), 26)], *EXACT),
        ("no argument", lambda: jnp.arange(5, dtype=jnp.float32) * 2,
         [], *EXACT),

        # control flow (P3).  Every region below becomes a sub-Program of the
        # tape and the executor enters it exactly as it enters a top-level
        # one; what is being checked here is the LOWERING -- the carry/capture
        # ordering, the counted-loop encoding, and the clamps.
        ("scan (cumulative)",
         lambda xs: jax.lax.scan(lambda c, x: (c + x, c * 2), np.float32(0),
                                 xs),
         [_rand((8,), 40)], *F32),
        ("scan (carry only)",
         lambda c0, xs: jax.lax.scan(lambda c, x: (c * 0.9 + x, None),
                                     c0, xs)[0],
         [_rand((4,), 41), _rand((6, 4), 42)], *F32),
        ("fori_loop",
         lambda x: jax.lax.fori_loop(
             0, 7, lambda i, c: c * 1.1 + i.astype(jnp.float32), x),
         [np.float32(1.0)], *F32),
        # A data-dependent trip count: the cond is evaluated on the host every
        # iteration (the dynamic arm of native/control.cc's run_while).
        ("while_loop (dynamic trip)",
         lambda x: jax.lax.while_loop(
             lambda s: s[0] < 100.0,
             lambda s: (s[0] * 2.0, s[1] + 1), (x, jnp.int32(0))),
         [np.float32(1.5)], *F32),
        # A loop whose bound is CAPTURED rather than constant: the counted
        # encoding's bound_kind 2, indexing the cond's capture list.
        ("fori_loop (captured bound)",
         lambda x, n: jax.lax.fori_loop(0, n, lambda i, c: c + 1.0, x),
         [np.float32(0.0), np.int32(9)], *EXACT),
        ("cond (both branches)",
         lambda p, x: (
             jax.lax.cond(p[0] > 0, lambda a: jnp.sin(a) * 2,
                          lambda a: jnp.cos(a) - 1, x),
             jax.lax.cond(p[1] > 0, lambda a: jnp.sin(a) * 2,
                          lambda a: jnp.cos(a) - 1, x)),
         [np.array([1.0, -1.0], f), _rand((4,), 43)], *F32),
        ("switch (every branch)", _switch_case,
         [np.array([0, 1, 2, 7, -3], np.int32), _rand((3,), 44)], *F32),
        # The GRU shape: a matmul inside the body, and a capture (the weights)
        # the region reads from the enclosing scope.
        ("scan over matmul",
         lambda h0, w, xs: jax.lax.scan(
             lambda h, x: (jnp.tanh(h @ w + x), None), h0, xs)[0],
         [_rand((4, 8), 45), _rand((8, 8), 46), _rand((5, 4, 8), 47)], *DOT),
        ("nested scan",
         lambda xs: jax.lax.scan(
             lambda c, row: (
                 jax.lax.scan(lambda d, v: (d + v * 0.5, None), c, row)[0],
                 None),
             np.float32(0), xs)[0],
         [_rand((4, 3), 48)], *F32),
        ("scan with a stacked output",
         lambda w, xs: jax.lax.scan(
             lambda c, x: (c + x @ w, c), jnp.zeros((3,), jnp.float32), xs)[1],
         [_rand((4, 3), 49), _rand((5, 4), 50)], *DOT),

        # dynamic_slice / dynamic_update_slice, including the CLAMPS.  XLA
        # clamps a start index so the window stays inside the operand; MLX
        # clamps nothing, so the tape carries the bounds and the handler
        # builds the clip.  An out-of-range index is silent wrongness if that
        # encoding is wrong, which is why it is tested in both directions.
        ("dynamic_slice", lambda x, i: jax.lax.dynamic_slice(x, (i,), (3,)),
         [np.arange(10, dtype=f), np.int32(4)], *EXACT),
        ("dynamic_slice (index past the end)",
         lambda x, i: jax.lax.dynamic_slice(x, (i,), (3,)),
         [np.arange(10, dtype=f), np.int32(100)], *EXACT),
        ("dynamic_slice (negative index)",
         lambda x, i: jax.lax.dynamic_slice(x, (i,), (3,)),
         [np.arange(10, dtype=f), np.int32(-5)], *EXACT),
        ("dynamic_slice 2d (both clamps)",
         lambda x, i, j: jax.lax.dynamic_slice(x, (i, j), (2, 2)),
         [np.arange(12, dtype=f).reshape(3, 4), np.int32(5), np.int32(-3)],
         *EXACT),
        ("dynamic_update_slice",
         lambda x, u, i: jax.lax.dynamic_update_slice(x, u, (i,)),
         [np.arange(8, dtype=f), np.array([-1, -2, -3], f), np.int32(2)],
         *EXACT),
        ("dynamic_update_slice (index past the end)",
         lambda x, u, i: jax.lax.dynamic_update_slice(x, u, (i,)),
         [np.arange(8, dtype=f), np.array([-1, -2, -3], f), np.int32(9)],
         *EXACT),
        ("dynamic_update_slice 2d (negative index)",
         lambda x, u, i, j: jax.lax.dynamic_update_slice(x, u, (i, j)),
         [np.arange(12, dtype=f).reshape(3, 4), np.full((2, 2), -1.0, f),
          np.int32(-4), np.int32(2)], *EXACT),
        # The carry-stacking shape scan lowers to: a dus into a buffer at a
        # loop-carried index, inside a counted loop.
        ("scan stacking through dus",
         lambda xs: jax.lax.scan(lambda c, x: (c + x, c), np.float32(0), xs)[1],
         [_rand((6,), 51)], *F32),
        # 10k iterations of a tiny body: the loop's flush cadence is what
        # keeps Metal's live-buffer count bounded, and a loop this long is
        # where its absence shows up (CLAUDE.md items 11/14).  Exact in f32:
        # every partial sum is an integer below 2**24.
        ("long counted loop (10k iterations)",
         lambda x: jax.lax.fori_loop(0, 10000, lambda i, c: c + 1.0, x),
         [np.float32(0.0)], *EXACT),
        # The CHUNKED replay (P5).  With the compile decisions on, a counted
        # loop whose body is pure and cheap enough replays kmax iterations per
        # compiled graph (native/control.cc `run_chunked`) instead of one --
        # a different sync-point layout, a different set of MLX kernels, and
        # the arm nothing exercised while `chunkable` was hard-wired to 0.
        # 512 steps of a matmul body: cost puts kmax at the 16 ceiling, so
        # this really does run 32 chunks and their remainder.
        ("chunked replay (512 x matmul body)",
         lambda h0, w, xs: jax.lax.scan(
             lambda h, x: (jnp.tanh(h @ w * 0.5 + x), None), h0, xs)[0],
         [_rand((4, 8), 52), _rand((8, 8), 53), _rand((512, 4, 8), 54)], *DOT),
        # A counted loop small enough to UNROLL into the enclosing trace
        # (ops/control._while_traceable): the whole main compiles, the loop
        # among it, so nothing here reaches run_while's eager arm at all.
        ("counted loop unrolled into a compiled main",
         lambda x: jax.lax.fori_loop(0, 6, lambda i, c: jnp.sin(c) + 0.25, x),
         [_rand((16,), 55)], *F32),

        # msl_scan (P21).  Every one of these takes a GENERATED METAL KERNEL
        # in place of the loop: the mode census below asserts that all three
        # emitters really run, and the "msl kernels" section re-runs them with
        # METALJAX_MSL=0 so the kernel is compared with the interpreted loop
        # as well as with the CPU.
        ("msl affine cell (mingru)", _msl_mingru,
         [_rand((4, 16), 60), _rand((24, 4, 16), 61), _rand((16,), 62),
          _rand((16,), 63)], *F32),
        ("msl affine cell, backward", _msl_mingru_grad,
         [_rand((4, 16), 64), _rand((24, 4, 16), 65), _rand((16,), 66),
          _rand((16,), 67)], *DOT),
        ("msl vector matvec cell", _msl_rnn,
         [_rand((4, 4), 68), _rand((16, 4, 4), 69),
          _rand((4, 4), 70) * f(0.3)], *DOT),
        ("msl vector cell, weight grad", _msl_rnn_grad,
         [_rand((4, 4), 71), _rand((16, 4, 4), 72),
          _rand((4, 4), 73) * f(0.3)], *DOT),
        ("msl coop matvec cell", _msl_rnn,
         [_rand((8, 64), 74), _rand((12, 8, 64), 75),
          _rand((64, 64), 76) * f(0.1)], *DOT),
        ("msl coop cell, weight grad", _msl_rnn_grad,
         [_rand((8, 32), 77), _rand((12, 8, 32), 78),
          _rand((32, 32), 79) * f(0.1)], *DOT),
        ("msl gru cell (coop flip)", _msl_gru,
         [_rand((8, 16), 80), _rand((12, 8, 16), 81),
          _rand((16, 16), 82) * f(0.2), _rand((16, 16), 83) * f(0.2),
          _rand((16, 16), 84) * f(0.2)], *DOT),
        ("msl nested unrolled loop", _msl_nested,
         [_rand((4, 8), 85), _rand((10, 4, 8), 86),
          _rand((8, 8), 87) * f(0.2)], *DOT),
        # The same decision, one size up: 200 iterations still fit the OP
        # budget, so the lowering calls the body traceable and compiles main
        # around it -- and the executor then refuses to unroll more than 64
        # into one trace and hands the program back to the eager path
        # (`run_recovering`).  Both engines take that route; what this row
        # checks is that the answer survives it.
        ("counted loop past the unroll ceiling",
         lambda x: jax.lax.fori_loop(0, 200, lambda i, c: c + 1.0, x),
         [np.float32(0.0)], *EXACT),

        # gather (P4).  StableHLO's gather goes straight to mx::gather, whose
        # index arrays, clamps, window sizes and offset_dims transpose are all
        # resolved at lowering.  MLX clamps NOTHING -- it wraps a negative
        # index like `take` and reads past the end otherwise -- so an
        # out-of-range index is silent wrongness if the bounds are wrong, and
        # the CPU comparison IS the test of XLA's clamp rule.
        ("take", lambda a, i: a[i],
         [np.arange(6, dtype=f), np.array([0, 5, 2], np.int32)], *EXACT),
        ("embedding lookup", lambda a, i: a[i],
         [np.arange(24, dtype=f).reshape(6, 4),
          np.array([0, 3, 5, 2], np.int32)], *EXACT),
        ("embedding lookup (2-D indices)", lambda a, i: a[i],
         [np.arange(24, dtype=f).reshape(6, 4),
          np.array([[0, 1], [2, 3]], np.int32)], *EXACT),
        ("gather (indices out of range)", lambda a, i: a[i],
         [np.arange(6, dtype=f), np.array([-3, 0, 9, 5], np.int32)], *EXACT),
        ("gather rows and columns", lambda a, i, j: a[i, j],
         [np.arange(24, dtype=f).reshape(6, 4),
          np.array([0, 2], np.int32), np.array([3, 1], np.int32)], *EXACT),
        # slice_sizes > 1 on an indexed dim: a WINDOW per index, which is the
        # arm that crosses verbatim into mx::gather.
        ("windowed gather",
         lambda a, i: jax.vmap(
             lambda k: jax.lax.dynamic_slice(a, (k,), (3,)))(i),
         [np.arange(10, dtype=f), np.array([0, 4, 8], np.int32)], *EXACT),
        # operand_batching_dims: the implicit iota index a vmapped gather
        # carries, paired with its operand dim.
        ("gather with batching dims",
         lambda a, i: jax.vmap(lambda r, k: r[k])(a, i),
         [np.arange(12, dtype=f).reshape(3, 4),
          np.array([[0, 3], [1, 1], [2, 0]], np.int32)], *EXACT),
        ("take_along_axis", lambda a, i: jnp.take_along_axis(a, i, -1),
         [np.arange(24, dtype=f).reshape(6, 4),
          np.array([[0], [1], [2], [3], [0], [1]], np.int32)], *EXACT),
        ("cross-entropy (gather of a log-softmax)",
         lambda logits, t: -jnp.take_along_axis(
             jax.nn.log_softmax(logits, -1), t[:, None], -1).mean(),
         [_rand((12, 32), 60), np.arange(12, dtype=np.int32) % 32], *F32),
        ("gather with an empty result", lambda a, i: a[i],
         [np.arange(6, dtype=f), np.zeros((0,), np.int32)], *EXACT),
        ("gather at a rank-0 index", lambda a: a[jnp.int32(2)],
         [np.arange(6, dtype=f)], *EXACT),

        # scatter (P4).  XLA DROPS an update whose start is out of bounds;
        # MLX has no such rule and does no bounds checking at all, so the
        # tape picks one of two drop strategies from the static sizes.  Every
        # case below is compared against the CPU backend, which is the only
        # honest statement of those semantics -- and the pairs with and
        # without an out-of-range index are what separates "dropped" from
        # "clamped onto a real slot", which is a wrong ANSWER, not an error.
        ("scatter set (a slice)", lambda x, u: x.at[2:5].set(u),
         [np.arange(8, dtype=f), np.full(3, -1.0, f)], *EXACT),
        ("scatter set (indices)", lambda x, i, u: x.at[i].set(u),
         [np.arange(8, dtype=f), np.array([1, 3, 6], np.int32),
          np.array([-1, -2, -3], f)], *EXACT),
        ("scatter set (out of bounds)", lambda x, i, u: x.at[i].set(u),
         [np.arange(8, dtype=f), np.array([1, 8, -2, 20], np.int32),
          np.array([-1, -2, -3, -4], f)], *EXACT),
        ("scatter add (duplicate indices)", lambda x, i, u: x.at[i].add(u),
         [np.zeros(5, f), np.array([0, 1, 1, 4], np.int32),
          np.array([1, 2, 3, 4], f)], *EXACT),
        ("scatter add (out of bounds)", lambda x, i, u: x.at[i].add(u),
         [np.zeros(5, f), np.array([0, 7, 1, -1], np.int32),
          np.array([1, 2, 3, 4], f)], *EXACT),
        # Updates bigger than the operand: the drop strategy flips to the
        # dummy pad, which is the arm "set" always takes.
        ("scatter add (updates > operand)", lambda x, i, u: x.at[i].add(u),
         [np.zeros(4, f), np.arange(64, dtype=np.int32) % 6,
          np.arange(64, dtype=f)], *F32),
        ("scatter multiply", lambda x, i, u: x.at[i].multiply(u),
         [np.ones(5, f), np.array([0, 1, 1, 9], np.int32),
          np.array([2, 3, 4, 5], f)], *EXACT),
        ("scatter max / min",
         lambda x, i, u: (x.at[i].max(u), x.at[i].min(u)),
         [np.zeros(5, f), np.array([0, 1, 1, 9], np.int32),
          np.array([2, -3, 4, 5], f)], *EXACT),
        ("scatter add (int32)", lambda x, i, u: x.at[i].add(u),
         [np.zeros(5, np.int32), np.array([0, 1, 1, 9], np.int32),
          np.array([2, -3, 4, 5], np.int32)], *EXACT),
        ("segment_sum", lambda x, s: jax.ops.segment_sum(x, s, 4),
         [np.arange(8, dtype=f), np.arange(8, dtype=np.int32) % 4], *EXACT),
        # bincount's overflow slot is the OOB-drop rule in production: an
        # index past the length must vanish, not land on the last bucket.
        ("bincount", lambda x: jnp.bincount(x, length=5),
         [np.array([0, 1, 1, 4, 9, 2], np.int32)], *EXACT),
        ("scatter whole rows", lambda x, i, u: x.at[i].set(u),
         [np.zeros((4, 3), f), np.array([0, 3], np.int32),
          np.ones((2, 3), f)], *EXACT),
        ("scatter along the middle axis", lambda x, i, u: x.at[:, i].set(u),
         [np.zeros((2, 4, 3), f), np.array([1, 3], np.int32),
          np.ones((2, 2, 3), f)], *EXACT),
        # A vmapped scatter: size-1 update windows on the mapped dims, which
        # is where the 0.4.1 expand-transpose bug lived.
        ("vmapped scatter set",
         lambda x, i, u: jax.vmap(lambda r, k, v: r.at[k].set(v))(x, i, u),
         [np.zeros((3, 4), f), np.array([[0], [2], [3]], np.int32),
          np.ones((3, 1), f)], *EXACT),
        ("vmapped scatter add (one lane out of bounds)",
         lambda x, i, u: jax.vmap(lambda r, k, v: r.at[k].add(v))(x, i, u),
         [np.zeros((3, 4), f), np.array([[0], [2], [5]], np.int32),
          np.full((3, 1), 2.0, f)], *EXACT),
        ("embedding gradient (a scatter-add through AD)",
         lambda e, t: jax.grad(lambda a: (a[t] ** 2).sum())(e),
         [np.arange(24, dtype=f).reshape(6, 4),
          np.array([0, 3, 5, 2, 3], np.int32)], *EXACT),
        # Empty updates: the handler returns the OPERAND array, so the tape
        # aliases the slot and the output-copy rule has to notice.
        ("scatter with empty updates", lambda x, i, u: x.at[i].add(u),
         [np.arange(4, dtype=f), np.zeros((0,), np.int32),
          np.zeros((0,), f)], *EXACT),

        # the small-op tail (P4)
        ("shift left", lambda x: x << 3, [np.arange(8, dtype=np.uint32)],
         *EXACT),
        # XLA defines a shift by >= the operand's bit width as 0 (logical and
        # left) or the sign fill (arithmetic); Metal's shifts are mod-width,
        # so the widths at and past 32 are the whole point of this row.
        ("shifts across the operand width",
         lambda x, s: (jax.lax.shift_left(x, s),
                       jax.lax.shift_right_logical(x, s),
                       jax.lax.shift_right_arithmetic(x, s)),
         [np.array([-8, -1, 1, 255, 1 << 30], np.int32),
          np.array([0, 1, 31, 32, 40], np.int32)], *EXACT),
        ("shifts on uint8 (overflow widths)",
         lambda x, s: (jax.lax.shift_left(x, s),
                       jax.lax.shift_right_logical(x, s)),
         [np.array([1, 128, 255, 7], np.uint8),
          np.array([1, 7, 8, 9], np.uint8)], *EXACT),
        # A constant amount past the width: the lowering resolves it and the
        # handler emits one arm instead of a compare and a select.
        ("shift by a static overflow amount", lambda x: (x << 40, x >> 40),
         [np.arange(4, dtype=np.int32)], *EXACT),
        ("reverse", lambda x: x[::-1], [np.arange(5, dtype=f)], *EXACT),
        ("reverse both axes", lambda x: x[::-1, ::-1],
         [np.arange(12, dtype=f).reshape(3, 4)], *EXACT),
        # An extent-1 dim is dropped at lowering (mx::take chokes on empties).
        ("reverse with a unit dim", lambda x: jax.lax.rev(x, (0, 1)),
         [np.arange(4, dtype=f).reshape(1, 4)], *EXACT),
        ("roll", lambda x: jnp.roll(x, 2), [np.arange(5, dtype=f)], *EXACT),
        ("bitcast f32 -> i32",
         lambda x: jax.lax.bitcast_convert_type(x, jnp.int32),
         [np.array([1.0, -2.5, 0.0], f)], *EXACT),
        ("bitcast i32 -> f32",
         lambda x: jax.lax.bitcast_convert_type(x, jnp.float32),
         [np.array([1065353216, -1, 0], np.int32)], *EXACT),
        ("bitcast widening (i16 -> i32)",
         lambda x: jax.lax.bitcast_convert_type(x, jnp.int32),
         [np.array([[1, 2], [3, 4]], np.int16)], *EXACT),
        ("bitcast narrowing (i32 -> i16)",
         lambda x: jax.lax.bitcast_convert_type(x, jnp.int16),
         [np.array([1, -2], np.int32)], *EXACT),
        ("popcount", lambda x: jax.lax.population_count(x),
         [np.array([0, 1, 255, 1 << 30, -1], np.int32)], *EXACT),
        ("count_leading_zeros", lambda x: jax.lax.clz(x),
         [np.array([0, 1, 255, 1 << 30, -1], np.int32)], *EXACT),

        # threefry (P4): with the shifts in the op set, jax's RNG is ordinary
        # elementwise arithmetic and must be BIT-exact against the CPU
        # backend, not merely close.  `bits`, `split` and `fold_in` are the
        # raw words, compared exactly; `normal` goes through erf_inv, whose
        # last ULP is MLX's rather than the CPU's, so it gets a tolerance.
        ("threefry bits",
         lambda k: jax.random.bits(jax.random.wrap_key_data(k), (16,)),
         [np.array([1, 2], np.uint32)], *EXACT),
        ("threefry split",
         lambda k: jax.random.key_data(
             jax.random.split(jax.random.wrap_key_data(k), 4)),
         [np.array([0, 7], np.uint32)], *EXACT),
        ("threefry fold_in",
         lambda k: jax.random.key_data(
             jax.random.fold_in(jax.random.wrap_key_data(k), 7)),
         [np.array([0, 7], np.uint32)], *EXACT),
        ("threefry uniform",
         lambda k: jax.random.uniform(jax.random.wrap_key_data(k), (3, 5)),
         [np.array([12345, 6789], np.uint32)], *EXACT),
        ("threefry randint",
         lambda k: jax.random.randint(jax.random.wrap_key_data(k), (6,), 0,
                                      10),
         [np.array([3, 4], np.uint32)], *EXACT),
        ("threefry normal",
         lambda k: jax.random.normal(jax.random.wrap_key_data(k), (4, 4)),
         [np.array([0, 7], np.uint32)], *F32),

        # --- P6: sort / top_k -------------------------------------------
        # jax lowers a float sort as a comparator that computes a KEY (-0 ->
        # +0, NaN -> canonical qNaN, then a TOTALORDER compare) and an integer
        # sort as a bare compare on the argument pair.  Both shapes are here,
        # and the values below are chosen so a wrong tie rule is visible: a
        # signed zero pair, repeated keys, and NaNs, which total order puts
        # last.
        ("sort f32", lambda x: jnp.sort(x, -1), [_rand((3, 5), 60)], *EXACT),
        ("sort f32 with ties, signed zeros and NaNs",
         lambda x: jnp.sort(x, -1),
         [np.array([[1.0, -0.0, 0.0, np.nan, -1.0, 1.0, np.inf, -np.inf]],
                   f)], *EXACT),
        ("sort i32", lambda x: jnp.sort(x, -1),
         [np.array([[5, -1, 5, 0, -7]], np.int32)], *EXACT),
        ("sort u8", lambda x: jnp.sort(x, -1),
         [np.array([[200, 1, 200, 0, 255]], np.uint8)], *EXACT),
        ("sort bool", lambda x: jnp.sort(x, -1),
         [np.array([[True, False, True, False]])], *EXACT),
        ("sort f16", lambda x: jnp.sort(x, -1),
         [np.arange(8, dtype=np.float16).reshape(2, 4) / 4 - 1], *EXACT),
        ("sort bf16", lambda x: jnp.sort(x, -1),
         [(np.arange(8, dtype=np.float32).reshape(2, 4) / 4 - 1).astype(
             jnp.bfloat16)], *EXACT),
        # A non-last axis arrives as transpose -> sort -> transpose, so the
        # sort's input is a strided VIEW -- the shape MLX 0.32's argsort reads
        # wrong elements from, which is why the handler materializes first.
        ("sort along axis 0", lambda x: jnp.sort(x, 0),
         [_rand((4, 3), 61)], *EXACT),
        ("sort a rank-3 array along the middle axis",
         lambda x: jnp.sort(x, 1), [_rand((2, 5, 3), 62)], *EXACT),
        ("argsort f32", lambda x: jnp.argsort(x, -1),
         [_rand((3, 5), 63)], *EXACT),
        ("argsort along axis 0", lambda x: jnp.argsort(x, 0),
         [_rand((4, 3), 64)], *EXACT),
        # Stability: every key is equal, so a stable sort must return the
        # identity permutation.  An unstable one is free to return anything,
        # which is exactly what this catches.
        ("argsort stability (all keys equal)", lambda x: jnp.argsort(x, -1),
         [np.zeros((2, 9), f)], *EXACT),
        ("argsort stability (repeated keys)", lambda x: jnp.argsort(x, -1),
         [np.array([[2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0]], f)], *EXACT),
        # -0 and +0 are numerically equal, so a values-only sort cannot see
        # whether the comparator's canonicalization ran.  argsort can: without
        # it total order puts every -0 BELOW every +0, and the indices move.
        # Same for NaNs, which must all tie with each other.
        ("argsort over signed zeros and NaNs", lambda x: jnp.argsort(x, -1),
         [np.array([[0.0, -0.0, 0.0, -0.0, np.nan, -1.0, np.nan]], f)],
         *EXACT),
        ("sort_key_val", lambda k, v: jax.lax.sort_key_val(k, v),
         [np.array([[3.0, 1.0, 2.0]], f),
          np.array([[10, 20, 30]], np.int32)], *EXACT),
        ("median (a sort under a slice)", lambda x: jnp.median(x, -1),
         [_rand((3, 7), 65)], *F32),
        ("percentile", lambda x: jnp.percentile(x, 40.0, axis=-1),
         [_rand((3, 7), 66)], *F32),
        ("partition", lambda x: jnp.partition(x, 2, axis=-1),
         [_rand((2, 8), 67)], *EXACT),
        ("top_k", lambda x: jax.lax.top_k(x, 3), [_rand((2, 16), 68)], *EXACT),
        ("top_k with ties", lambda x: jax.lax.top_k(x, 4),
         [np.array([[1.0, 1.0, 1.0, 0.0, 2.0, 2.0]], f)], *EXACT),
        ("top_k on integers", lambda x: jax.lax.top_k(x, 2),
         [np.array([[3, -1, 3, 7]], np.int32)], *EXACT),
        # lax.top_k is last-axis only; a top-k over another axis is a moveaxis
        # around it, which is the non-contiguous case again -- and the one
        # that was a silent-wrongness bug in 0.4.x.
        ("top_k over the leading axis",
         lambda x: jax.lax.top_k(jnp.moveaxis(x, 0, -1), 2),
         [_rand((5, 3), 69)], *EXACT),
        ("sort inside a scan body",
         lambda c, xs: jax.lax.scan(
             lambda a, r: (a + jnp.sort(r, -1)[0], None), c, xs)[0],
         [np.float32(0.0), _rand((4, 6), 70)], *F32),

        # --- P6: rng_bit_generator --------------------------------------
        # The bits must match the CPU backend EXACTLY, for both algorithms:
        # this family exists to be bit-compatible with XLA, and a tolerance
        # here would hide the only thing worth testing.  `_canonical` widens
        # unsigned words to int64, so EXACT really is a word-for-word compare.
        ("rng philox u32",
         lambda k: jax.lax.rng_bit_generator(
             k, (8,), dtype=jnp.uint32,
             algorithm=jax.lax.RandomAlgorithm.RNG_PHILOX),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        ("rng philox u32 (odd count)",
         lambda k: jax.lax.rng_bit_generator(
             k, (7,), dtype=jnp.uint32,
             algorithm=jax.lax.RandomAlgorithm.RNG_PHILOX),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        ("rng philox u8/u16 (narrow truncation)",
         lambda k: (*jax.lax.rng_bit_generator(
                        k, (9,), dtype=jnp.uint8,
                        algorithm=jax.lax.RandomAlgorithm.RNG_PHILOX),
                    *jax.lax.rng_bit_generator(
                        k, (9,), dtype=jnp.uint16,
                        algorithm=jax.lax.RandomAlgorithm.RNG_PHILOX)),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        ("rng philox rank-3",
         lambda k: jax.lax.rng_bit_generator(
             k, (2, 3, 4), dtype=jnp.uint32,
             algorithm=jax.lax.RandomAlgorithm.RNG_PHILOX),
         [np.array([9, 9, 9, 9], np.uint32)], *EXACT),
        ("rng threefry u32",
         lambda k: jax.lax.rng_bit_generator(
             k, (8,), dtype=jnp.uint32,
             algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        # The half-split: an odd extent rounds up and slices back, and a shape
        # with no even dim splits at the LARGEST one instead of the first.
        ("rng threefry u32 (odd count)",
         lambda k: jax.lax.rng_bit_generator(
             k, (7,), dtype=jnp.uint32,
             algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        ("rng threefry (3, 5): no even dim",
         lambda k: jax.lax.rng_bit_generator(
             k, (3, 5), dtype=jnp.uint32,
             algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        ("rng threefry (2, 3, 4)",
         lambda k: jax.lax.rng_bit_generator(
             k, (2, 3, 4), dtype=jnp.uint32,
             algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY),
         [np.array([5, 6, 7, 8], np.uint32)], *EXACT),
        ("rng threefry narrow outputs",
         lambda k: (*jax.lax.rng_bit_generator(
                        k, (9,), dtype=jnp.uint8,
                        algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY),
                    *jax.lax.rng_bit_generator(
                        k, (9,), dtype=jnp.uint16,
                        algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY)),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        ("rng threefry scalar output",
         lambda k: jax.lax.rng_bit_generator(
             k, (), dtype=jnp.uint32,
             algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        ("rng default algorithm",
         lambda k: jax.lax.rng_bit_generator(k, (6,), dtype=jnp.uint32),
         [np.array([4, 3, 2, 1], np.uint32)], *EXACT),
        # An empty output consumes no blocks, so the state comes back
        # unchanged -- and the handler hands the operand's own array back,
        # which is what the entry's taint rule is there for.
        ("rng empty output returns the state",
         lambda k: jax.lax.rng_bit_generator(k, (0,), dtype=jnp.uint32),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        ("rng state advances across calls",
         lambda k: (lambda s1, b1: (
             *jax.lax.rng_bit_generator(s1, (5,), dtype=jnp.uint32), b1))(
                 *jax.lax.rng_bit_generator(k, (5,), dtype=jnp.uint32)),
         [np.array([1, 2, 3, 4], np.uint32)], *EXACT),
        # The consumer that matters: one wrong word is visible everywhere.
        ("rbg uniform",
         lambda k: jax.random.uniform(
             jax.random.wrap_key_data(k, impl="rbg"), (16,)),
         [np.asarray(jax.random.key_data(jax.random.key(42, impl="rbg")))],
         *EXACT),
        ("rbg normal",
         lambda k: jax.random.normal(
             jax.random.wrap_key_data(k, impl="rbg"), (4, 4)),
         [np.asarray(jax.random.key_data(jax.random.key(7, impl="rbg")))],
         *F32),

        # --- P6: reduce_window ------------------------------------------
        # The cumulative peephole first: jax lowers cumsum and friends as a
        # full-width window with prefix (or suffix) padding, which the
        # lowering turns back into one MLX cum-op.
        ("cumsum", lambda x: jnp.cumsum(x, 0), [_rand((8, 3), 71)], *F32),
        ("cumsum on the last axis", lambda x: jnp.cumsum(x, -1),
         [_rand((3, 8), 72)], *F32),
        ("cumprod/cummax/cummin",
         lambda x: (jnp.cumprod(x, 0), jax.lax.cummax(x, 0),
                    jax.lax.cummin(x, 0)),
         [np.linspace(0.5, 2.0, 12, dtype=f).reshape(4, 3)], *F32),
        ("reverse cumsum", lambda x: jax.lax.cumsum(x, 0, reverse=True),
         [_rand((6, 2), 73)], *F32),
        ("max pooling", lambda x: jax.lax.reduce_window(
            x, -np.inf, jax.lax.max, (1, 2), (1, 2), "VALID"),
         [_rand((2, 8), 74)], *EXACT),
        ("sum pooling with SAME padding", lambda x: jax.lax.reduce_window(
            x, 0.0, jax.lax.add, (3, 3), (2, 2), "SAME"),
         [_rand((5, 7), 75)], *F32),
        ("min pooling, explicit padding", lambda x: jax.lax.reduce_window(
            x, np.inf, jax.lax.min, (3,), (1,), [(1, 2)]),
         [_rand((6,), 76)], *EXACT),
        ("window dilation", lambda x: jax.lax.reduce_window(
            x, -np.inf, jax.lax.max, (2,), (1,), [(0, 0)],
            base_dilation=(1,), window_dilation=(3,)),
         [_rand((9,), 77)], *EXACT),
        ("base dilation", lambda x: jax.lax.reduce_window(
            x, 0.0, jax.lax.add, (2,), (1,), [(0, 0)],
            base_dilation=(2,), window_dilation=(1,)),
         [_rand((5,), 78)], *F32),
        ("base and window dilation together", lambda x: jax.lax.reduce_window(
            x, 0.0, jax.lax.add, (2, 2), (1, 1), [(1, 1), (0, 0)],
            base_dilation=(2, 1), window_dilation=(1, 2)),
         [_rand((4, 5), 79)], *F32),
        # A window wider than its (padded) axis produces NO output elements:
        # the zero-size guard, which returns the init broadcast to an empty
        # shape rather than asking MLX for a view of nothing.
        ("a window wider than the axis (zero-size output)",
         lambda x: jax.lax.reduce_window(
             x, 0.0, jax.lax.add, (9,), (1,), "VALID"),
         [_rand((4,), 80)], *EXACT),
        ("bool reduce_window (any/all)",
         lambda x: (jax.lax.reduce_window(x, False, jax.lax.bitwise_or,
                                          (2,), (1,), "VALID"),
                    jax.lax.reduce_window(x, True, jax.lax.bitwise_and,
                                          (2,), (1,), "VALID")),
         [np.array([True, False, True, True, False])], *EXACT),
        ("integer max pooling", lambda x: jax.lax.reduce_window(
            x, np.int32(-2 ** 31), jax.lax.max, (2,), (2,), "VALID"),
         [np.array([3, -1, 4, -1, 5, 9], np.int32)], *EXACT),
        # The jvp of a max window is select_and_gather_add: one compare over
        # the window picks a single element and every output reads it there.
        ("jvp of a max pool (select_and_gather_add)",
         lambda x, t: tuple(jax.jvp(
             lambda v: jax.lax.reduce_window(
                 v, -np.inf, jax.lax.max, (2,), (2,), "VALID"),
             (x,), (t,))),
         [_rand((8,), 81), _rand((8,), 82)], *EXACT),
        ("reduce_window in a scan body",
         lambda xs: jax.lax.scan(
             lambda c, r: (c + jax.lax.reduce_window(
                 r, 0.0, jax.lax.add, (2,), (2,), "VALID"), None),
             np.zeros((3,), f), xs)[0],
         [_rand((4, 6), 83)], *F32),

        # --- P6: fft ----------------------------------------------------
        ("fft", lambda x: jnp.fft.fft(x), [_rand((8,), 84)], *F32),
        ("ifft", lambda x: jnp.fft.ifft(x + 0j), [_rand((8,), 85)], *F32),
        ("fft of an odd length", lambda x: jnp.fft.fft(x),
         [_rand((7,), 86)], *F32),
        ("rfft / irfft round trip",
         lambda x: (jnp.fft.rfft(x), jnp.fft.irfft(jnp.fft.rfft(x))),
         [_rand((16,), 87)], *F32),
        ("rfft of an odd length", lambda x: jnp.fft.rfft(x),
         [_rand((9,), 88)], *F32),
        ("fft2 over the trailing axes", lambda x: jnp.fft.fft2(x),
         [_rand((3, 4, 4), 89)], *F32),
        ("rfft2", lambda x: jnp.fft.rfft2(x), [_rand((4, 6), 90)], *F32),
        # The unit-length rewrite: MLX drops the transforms over the leading
        # axes when the real axis has length 1, so that case is spelled out as
        # the identity on the DC bin plus a complex transform of the rest.
        ("rfft with a unit last length", lambda x: jnp.fft.rfft(x, n=1),
         [_rand((5,), 91)], *F32),
        ("rfft2 with a unit last length",
         lambda x: jnp.fft.rfft2(x, s=(4, 1)), [_rand((4, 6), 92)], *F32),
        ("irfft with a unit last length",
         lambda x: jnp.fft.irfft(x, n=1),
         [(_rand((5,), 93) + 1j * _rand((5,), 94)).astype(np.complex64)],
         *F32),
        ("fft of a complex input",
         lambda x: (jnp.fft.fft(x), jnp.fft.ifft(x)),
         [(_rand((8,), 95) + 1j * _rand((8,), 96)).astype(np.complex64)],
         *F32),
        ("fft inside an elementwise chain",
         lambda x: jnp.abs(jnp.fft.fft(x * 2.0)) + 1.0,
         [_rand((16,), 97)], *F32),

        # --- P7: convolution --------------------------------------------
        # A convolution accumulates like a dot, so the float rows get the DOT
        # band; the integer rows are EXACT, because their arm exists to be
        # exact (im2col plus an int64 sum, where MLX's own convolution would
        # round through f32).
        ("conv 1d SAME", lambda x, k: conv(x, k, (1,), "SAME",
                                           dimension_numbers=C1),
         [_rand((2, 3, 8), 100), _rand((5, 3, 3), 101)], *DOT),
        ("conv 1d VALID, stride 2",
         lambda x, k: conv(x, k, (2,), "VALID", dimension_numbers=C1),
         [_rand((2, 3, 9), 102), _rand((4, 3, 3), 103)], *DOT),
        ("conv 1d explicit padding",
         lambda x, k: conv(x, k, (1,), [(2, 1)], dimension_numbers=C1),
         [_rand((1, 2, 6), 104), _rand((3, 2, 4), 105)], *DOT),
        ("conv 2d NCHW/OIHW",
         lambda x, k: conv(x, k, (1, 1), "SAME", dimension_numbers=C2),
         [_rand((2, 3, 7, 6), 106), _rand((4, 3, 3, 3), 107)], *DOT),
        # The other common layout: the same op with three different
        # permutations, which is what says the layout really is data here.
        ("conv 2d NHWC/HWIO",
         lambda x, k: conv(x, k, (2, 1), "VALID", dimension_numbers=C2L),
         [_rand((2, 7, 6, 3), 108), _rand((3, 3, 3, 4), 109)], *DOT),
        ("conv 3d", lambda x, k: conv(x, k, (1, 1, 1), "VALID",
                                      dimension_numbers=C3),
         [_rand((1, 2, 5, 5, 4), 110), _rand((3, 2, 2, 3, 2), 111)], *DOT),
        ("conv rhs dilation (atrous)",
         lambda x, k: conv(x, k, (1, 1), "VALID", rhs_dilation=(2, 2),
                           dimension_numbers=C2),
         [_rand((1, 2, 9, 9), 112), _rand((3, 2, 3, 3), 113)], *DOT),
        # lhs dilation is the transposed convolution, and it is what jax's
        # own backward pass emits -- the two grad rows below are its real
        # test, this one is the direct spelling.
        ("conv lhs dilation (transposed)",
         lambda x, k: conv(x, k, (1,), [(2, 2)], lhs_dilation=(2,),
                           dimension_numbers=C1),
         [_rand((1, 2, 5), 114), _rand((3, 2, 3), 115)], *DOT),
        ("conv both dilations",
         lambda x, k: conv(x, k, (1, 1), [(1, 1), (0, 0)],
                           lhs_dilation=(2, 1), rhs_dilation=(1, 2),
                           dimension_numbers=C2),
         [_rand((1, 2, 4, 6), 116), _rand((2, 2, 2, 2), 117)], *DOT),
        ("conv feature groups",
         lambda x, k: conv(x, k, (1,), "SAME", dimension_numbers=C1,
                           feature_group_count=2),
         [_rand((2, 4, 8), 118), _rand((6, 2, 3), 119)], *DOT),
        ("conv depthwise (groups == channels)",
         lambda x, k: conv(x, k, (1, 1), "SAME", dimension_numbers=C2,
                           feature_group_count=4),
         [_rand((1, 4, 6, 6), 120), _rand((8, 1, 3, 3), 121)], *DOT),
        # MLX implements `groups` for 1-D and 2-D only, so a 3-D grouped
        # convolution takes the expanded path: one ungrouped convolution per
        # group, concatenated along the features.
        ("conv 3d groups (the expanded path)",
         lambda x, k: conv(x, k, (1, 1, 1), "VALID", dimension_numbers=C3,
                           feature_group_count=2),
         [_rand((1, 4, 4, 4, 4), 122), _rand((6, 2, 2, 2, 2), 123)], *DOT),
        ("conv batch groups",
         lambda x, k: conv(x, k, (1,), "VALID", dimension_numbers=C1,
                           batch_group_count=2),
         [_rand((4, 2, 6), 124), _rand((6, 2, 3), 125)], *DOT),
        ("conv 2d batch groups",
         lambda x, k: conv(x, k, (1, 1), "SAME", dimension_numbers=C2,
                           batch_group_count=2),
         [_rand((4, 2, 5, 5), 126), _rand((6, 2, 3, 3), 127)], *DOT),
        # XLA pads AFTER lhs dilation, so a negative pad crops the DILATED
        # array -- the rewrite that turns it into an operand slice plus the
        # leftover holes.  Both spellings, since the second is the one the
        # dilation arithmetic can get wrong.
        ("conv negative padding",
         lambda x, k: conv(x, k, (1,), [(-1, -1)], dimension_numbers=C1),
         [_rand((1, 2, 8), 128), _rand((3, 2, 3), 129)], *DOT),
        ("conv negative padding with lhs dilation",
         lambda x, k: conv(x, k, (1,), [(-3, 1)], lhs_dilation=(2,),
                           dimension_numbers=C1),
         [_rand((1, 2, 6), 130), _rand((2, 2, 3), 131)], *DOT),
        ("conv negative padding that empties the operand",
         lambda x, k: conv(x, k, (1,), [(-6, 0)], dimension_numbers=C1),
         [_rand((1, 2, 6), 132), _rand((3, 2, 1), 133)], *EXACT),
        # A kernel wider than its axis produces no output elements at all --
        # the guard that keeps MLX from sizing a window its own way and
        # handing back a short buffer (CLAUDE.md item 20's conv overread).
        ("conv with a kernel wider than the axis",
         lambda x, k: conv(x, k, (1,), "VALID", dimension_numbers=C1),
         [_rand((1, 2, 2), 134), _rand((3, 2, 5), 135)], *EXACT),
        ("conv with a zero-size batch",
         lambda x, k: conv(x, k, (1,), "SAME", dimension_numbers=C1),
         [_rand((0, 2, 6), 136), _rand((3, 2, 3), 137)], *EXACT),
        ("conv with zero-size channels",
         lambda x, k: conv(x, k, (1,), "VALID", dimension_numbers=C1),
         [_rand((1, 0, 6), 138), _rand((3, 0, 3), 139)], *EXACT),
        ("conv int32 (exact)",
         lambda x, k: conv(x, k, (1,), "SAME", dimension_numbers=C1),
         [_randint((2, 3, 7), 140), _randint((4, 3, 3), 141)], *EXACT),
        ("conv int8 (exact)",
         lambda x, k: conv(x, k, (1,), "VALID", dimension_numbers=C1),
         [_randint((1, 2, 6), 142, np.int8),
          _randint((3, 2, 3), 143, np.int8)], *EXACT),
        ("conv uint8 (exact)",
         lambda x, k: conv(x, k, (1,), "VALID", dimension_numbers=C1),
         [_randint((1, 2, 6), 144, np.uint8, 0, 5),
          _randint((3, 2, 3), 145, np.uint8, 0, 5)], *EXACT),
        ("conv int with both dilations (exact)",
         lambda x, k: conv(x, k, (1,), [(0, 0)], lhs_dilation=(2,),
                           rhs_dilation=(2,), dimension_numbers=C1),
         [_randint((1, 2, 5), 146), _randint((3, 2, 2), 147)], *EXACT),
        ("conv int with feature groups (exact)",
         lambda x, k: conv(x, k, (1,), "SAME", dimension_numbers=C1,
                           feature_group_count=2),
         [_randint((1, 4, 6), 148), _randint((6, 2, 3), 149)], *EXACT),
        # The im2col view is one strided read over the whole padded operand,
        # so its stride arithmetic only has more than one spatial axis to get
        # wrong from 2-D up -- which the 1-D rows above cannot see.
        ("conv int 2d (exact)",
         lambda x, k: conv(x, k, (1, 1), "SAME", dimension_numbers=C2),
         [_randint((2, 3, 5, 6), 182), _randint((4, 3, 3, 3), 183)], *EXACT),
        ("conv int 2d strided, dilated (exact)",
         lambda x, k: conv(x, k, (2, 1), "VALID", rhs_dilation=(2, 1),
                           dimension_numbers=C2),
         [_randint((1, 2, 9, 7), 184), _randint((3, 2, 3, 2), 185)], *EXACT),
        ("conv int with negative padding (exact)",
         lambda x, k: conv(x, k, (1,), [(-1, -1)], dimension_numbers=C1),
         [_randint((1, 2, 8), 186), _randint((3, 2, 3), 187)], *EXACT),
        # complex is four real convolutions.
        ("conv complex64",
         lambda x, k: conv(x, k, (1,), "SAME", dimension_numbers=C1),
         [(_rand((1, 2, 6), 150) + 1j * _rand((1, 2, 6), 151)
           ).astype(np.complex64),
          (_rand((3, 2, 3), 152) + 1j * _rand((3, 2, 3), 153)
           ).astype(np.complex64)], *DOT),
        # No spatial dims at all: the convolution IS a contraction over the
        # features, and the grouped forms of it are a block-diagonal one.
        ("conv with no spatial dims",
         lambda x, k: conv(x, k, (), [],
                           dimension_numbers=("NC", "OI", "NC")),
         [_rand((3, 4), 154), _rand((5, 4), 155)], *DOT),
        ("conv with no spatial dims, feature groups",
         lambda x, k: conv(x, k, (), [],
                           dimension_numbers=("NC", "OI", "NC"),
                           feature_group_count=2),
         [_rand((3, 4), 156), _rand((6, 2), 157)], *DOT),
        ("conv with no spatial dims, batch groups",
         lambda x, k: conv(x, k, (), [],
                           dimension_numbers=("NC", "OI", "NC"),
                           batch_group_count=2),
         [_rand((4, 3), 158), _rand((6, 3), 159)], *DOT),
        # ...and the COMPLEX one, which is four real matmuls exactly as the
        # spatial arm is four real convolutions.  It used to decline, because
        # `ops/conv.py`'s matmul arm runs its operands through f32 and drops
        # the imaginary part -- and the Stage 1 engine SHIPS that: the jax test
        # that covers this shape (`lax_test::testConvGeneralDilated0D2`)
        # compares metal against metal and never saw it.  These rows compare
        # against jax-CPU, which is the whole difference.
        ("conv complex64 with no spatial dims",
         lambda x, k: conv(x, k, (), [],
                           dimension_numbers=("NC", "OI", "NC")),
         [_crand((3, 4), 350), _crand((5, 4), 351)], *DOT),
        ("conv complex64 with no spatial dims, feature groups",
         lambda x, k: conv(x, k, (), [],
                           dimension_numbers=("NC", "OI", "NC"),
                           feature_group_count=2),
         [_crand((3, 4), 352), _crand((6, 2), 353)], *DOT),
        ("conv f16", lambda x, k: conv(x, k, (1,), "SAME",
                                       dimension_numbers=C1),
         [_rand((1, 2, 6), 160, np.float16),
          _rand((3, 2, 3), 161, np.float16)], *HALF),
        ("conv bf16",
         lambda x, k: conv(x, k, (1,), "SAME",
                           dimension_numbers=C1).astype(jnp.float32),
         [np.asarray(_rand((1, 2, 6), 162)).astype(jnp.bfloat16),
          np.asarray(_rand((3, 2, 3), 163)).astype(jnp.bfloat16)], *HALF),
        # jax's own wrappers, which spell their own dimension numbers.
        ("jnp.convolve", lambda a, b: jnp.convolve(a, b),
         [_rand((7,), 164), _rand((3,), 165)], *DOT),
        ("jnp.correlate", lambda a, b: jnp.correlate(a, b, mode="full"),
         [_rand((7,), 166), _rand((3,), 167)], *DOT),
        ("lax.conv", lambda x, k: jax.lax.conv(x, k, (1, 1), "SAME"),
         [_rand((1, 2, 5, 5), 168), _rand((3, 2, 3, 3), 169)], *DOT),
        ("lax.conv_with_general_padding",
         lambda x, k: jax.lax.conv_with_general_padding(
             x, k, (1,), [(1, 1)], (1,), (2,)),
         [_rand((1, 2, 6), 170), _rand((3, 2, 2), 171)], *DOT),
        ("lax.conv_transpose",
         lambda x, k: jax.lax.conv_transpose(x, k, (2,), "SAME",
                                             dimension_numbers=C1),
         [_rand((1, 2, 4), 172), _rand((3, 2, 3), 173)], *DOT),
        # The backward pass of a strided convolution is a TRANSPOSED one (the
        # gradient wrt the input) plus a BATCH-GROUPED one (the gradient wrt
        # the weights), so one grad exercises the two arms jax's forward
        # spelling barely reaches.
        ("conv grad wrt input and weights",
         lambda x, k: jax.grad(
             lambda a, b: (conv(a, b, (2,), "SAME",
                                dimension_numbers=C1) ** 2).sum(),
             argnums=(0, 1))(x, k),
         [_rand((2, 3, 8), 174), _rand((4, 3, 3), 175)], *DOT),
        ("conv 2d grad",
         lambda x, k: jax.grad(
             lambda a, b: jnp.sum(jnp.tanh(
                 conv(a, b, (1, 1), "SAME", dimension_numbers=C2))),
             argnums=(0, 1))(x, k),
         [_rand((2, 3, 6, 6), 176), _rand((4, 3, 3, 3), 177)], *DOT),
        ("conv in a scan body",
         lambda xs, k: jax.lax.scan(
             lambda c, v: (c + conv(v[None], k, (1,), "SAME",
                                    dimension_numbers=C1)[0], None),
             np.zeros((3, 6), f), xs)[0],
         [_rand((4, 2, 6), 178), _rand((3, 2, 3), 179)], *DOT),
        ("conv in a fori_loop body",
         lambda x, k: jax.lax.fori_loop(
             0, 3, lambda i, c: c + conv(c, k, (1,), "SAME",
                                         dimension_numbers=C1), x),
         [_rand((1, 2, 6), 180), _rand((2, 2, 3), 181)], *DOT),

        # --- P8.5: the census's fix batch --------------------------------
        # A StableHLO reduce returns its OPERAND's element type, and MLX's
        # sum and prod accumulate wider than that for small integers (int8
        # -> int32, uint8 -> uint32), so the fold back is what makes the
        # result the declared type.  The values overflow on purpose: the
        # wrap is the answer XLA computes, and it is the same one whether
        # the truncation happens per step or once at the end.
        ("sum with an int8 accumulator", lambda x: jnp.sum(x, dtype=jnp.int8),
         [np.arange(40, 60, dtype=np.int8)], *EXACT),
        ("prod with a uint8 accumulator",
         lambda x: jnp.prod(x, dtype=jnp.uint8),
         [np.arange(2, 9, dtype=np.uint8)], *EXACT),
        ("int16 sum over one axis", lambda x: jnp.sum(x, 1, dtype=jnp.int16),
         [_randint((3, 4), 200, np.int16, -300, 300)], *EXACT),
        # The dtype has to be right INSIDE the tape, not just at the
        # boundary: this one wraps AFTER the reduce, which a widened
        # accumulator would carry through in full precision.
        ("an int8 sum feeding more int8 arithmetic",
         lambda x: jnp.sum(x, dtype=jnp.int8) * jnp.int8(3) + x,
         [np.arange(20, 30, dtype=np.int8)], *EXACT),
        ("sum pooling over int16", lambda x: jax.lax.reduce_window(
            x, np.int16(0), jax.lax.add, (3,), (1,), "VALID"),
         [_randint((8,), 201, np.int16, -20000, 20000)], *EXACT),
        # A zero-size constant: chlo's decompositions emit one whenever the
        # operand is empty, and MLIR stores it as a SPLAT holding one raw
        # element -- so the raw data is not the elements, and the decode has
        # nothing to read.
        ("sinh of an empty int8 array", lambda x: jnp.sinh(x),
         [np.zeros((0,), np.int8)], *F32),
        ("spacing of an empty f16 array", lambda x: jnp.spacing(x),
         [np.zeros((0,), np.float16)], *HALF),
        ("an empty array through a chlo composite",
         lambda x: (jnp.arcsin(x), jnp.cosh(x)),
         [np.zeros((0, 3), np.float32)], *F32),
        # MLX's compiler rejects some fused traces outright ("Too many
        # inputs/outputs fused in the Metal Compiled primitive": the
        # generated kernel's most argument-hungry variant would bind more
        # buffers than Metal allows, notes/data/mlx-fused-args-repro).
        # polygamma is 334 entries of elementwise chain and is refused
        # today; the answer below is the eager path's, computed after the
        # compiled one retires.
        ("polygamma (a trace MLX's compiler refuses)",
         lambda x: jax.scipy.special.polygamma(2, x),
         [np.array([0.5, 1.5, 2.5, 3.5], f)], *F32),

        # --- P10: the compiled-constant precision rule -------------------
        # mx::compile inlines a RANK-0 constant into generated Metal source
        # as a %.7g decimal literal, one digit short of float32's round trip,
        # so two thirds of constants come back a ULP off (CLAUDE.md item 20,
        # and tests/test_elementwise.py's two regression tests, which are
        # these rows).  The lowering hands the ones that do not round-trip to
        # a one-element buffer instead.  Rank-0 operands in a CHAIN: a lone
        # binary op passes the scalar as a kernel argument and a constant
        # that feeds a broadcast rides in memory anyway, so only a fused
        # multi-op kernel bakes the literal.
        ("rank-0 f32 constants through a fused chain",
         lambda x: jnp.stack([jnp.float32(c) * x * x for c in
                              (np.pi, np.pi / 2, 1 / 3, 12345.6789, 0.7,
                               8.5e-9, 1e-7, 2.0, 0.1, 0.5)]),
         [np.float32(0.995)], *EXACT),
        # An ill-conditioned consumer is what makes one ULP visible:
        # tan(pi/2 - pi*q) is scipy.stats.cauchy.isf, whose condition number
        # is ~64 at the ends of that test's clipped range (the census row
        # `scipy_stats_test::testCauchyIsf1` is exactly this expression).
        ("an ill-conditioned constant expression",
         lambda x: jnp.tan(jnp.float32(np.pi / 2) - jnp.float32(np.pi) * x),
         [np.array([0.995, 0.005, 0.99, 0.01, 0.75], f)], 2e-6, 0.0),
        # The rule is only for the constants that LOSE something: 0.5 and 2.0
        # round-trip through seven digits and stay literals, and the answer
        # must be the same either way.
        ("rank-0 constants that do round-trip",
         lambda x: jnp.stack([jnp.float32(c) * x * x for c in
                              (0.5, 2.0, 0.25, 1.0, 100.0)]),
         [np.float32(1.0000001)], *EXACT),

        # --- P10: complex scatter (by parts) -----------------------------
        # MLX has no complex scatter kernels, so the entry writes the real
        # and imaginary parts separately and recombines them -- which is
        # exact for the componentwise combiners and nothing else (the
        # lowering declines multiply, and complex has no order for max/min).
        ("complex scatter set", lambda x, i, u: x.at[i].set(u),
         [_crand((6, 4), 300), np.array([0, 5, 2], np.int32),
          _crand((3, 4), 301)], *EXACT),
        # Indices are UNIQUE in the arithmetic rows on purpose: two updates
        # summed into one slot is order-nondeterministic on this GPU (as it
        # is on jax-CUDA), so a duplicate would measure the scheduler.
        ("complex scatter add", lambda x, i, u: x.at[i].add(u),
         [_crand((6, 4), 302), np.array([1, 4, 3], np.int32),
          _crand((3, 4), 303)], *EXACT),
        ("complex scatter subtract", lambda x, i, u: x.at[i].add(-u),
         [_crand((6, 4), 304), np.array([4, 0, 2], np.int32),
          _crand((3, 4), 305)], *EXACT),
        # XLA DROPS an update whose window does not fit, and the two drop
        # strategies (neutral value, dummy pad) must both survive the split
        # into parts -- the pad grows each part, the neutral is the PART's
        # (0.0f, not a complex zero).
        ("complex scatter set, out of bounds", lambda x, i, u: x.at[i].set(u),
         [_crand((6, 4), 306), np.array([-1, 4, 99], np.int32),
          _crand((3, 4), 307)], *EXACT),
        ("complex scatter add, out of bounds", lambda x, i, u: x.at[i].add(u),
         [_crand((6, 4), 308), np.array([-3, 2, 7], np.int32),
          _crand((3, 4), 309)], *EXACT),
        # Signed zeros and NaN payloads: adding the parts must not go
        # through a complex multiply anywhere, and a dropped update must not
        # perturb the sign of a zero it lands on.
        ("complex scatter over signed zeros and NaNs",
         lambda x, i, u: x.at[i].set(u),
         [np.array([-0.0 + 0j, 0.0 - 0.0j, np.nan + 1j], np.complex64),
          np.array([0, 7], np.int32),
          np.array([-0.0 - 0.0j, 1 + np.nan * 1j], np.complex64)], *EXACT),
        # A single element (an inserted window dim), a partial window, and a
        # vmapped scatter (batching dims) -- the index-plan shapes P4 built.
        ("complex scatter into one column",
         lambda x, i, u: x.at[i, 1].set(u),
         [_crand((6, 4), 310), np.array([0, 3], np.int32),
          _crand((2,), 311)], *EXACT),
        # MULTIPLY is the combiner the decomposition cannot split, so it is
        # rewritten (gather the current values, multiply, set) under the op's
        # own `unique_indices` -- ops/gather.py's apply path, with the promise
        # checked rather than assumed.  Not EXACT: one complex multiply on
        # this GPU contracts to an FMA where the CPU's does not (~7e-8),
        # which is the same arithmetic the Python engine runs.
        ("complex scatter multiply (unique indices)",
         lambda x, i, u: x.at[i].multiply(u, unique_indices=True),
         [_crand((6, 4), 322), np.array([0, 5, 2], np.int32),
          _crand((3, 4), 323)], *F32),
        # ...and its dropped updates: the rewrite WRITES, so it takes the
        # dummy-pad drop rule a set takes, and the product of a clamped
        # gather never reaches the operand.  No NEGATIVE index here -- jax
        # wraps those before the scatter, which would make two updates land
        # on one slot and break the uniqueness the arm was given.
        ("complex scatter multiply, out of bounds",
         lambda x, i, u: x.at[i].multiply(u, unique_indices=True),
         [_crand((6, 4), 324), np.array([7, 5, 99], np.int32),
          _crand((3, 4), 325)], *F32),
        # WITHOUT the promise -- which is every plain `.at[i].multiply(u)`,
        # since jax sets `unique_indices = false` for all of them, literal
        # indices included.  Keying the arm on the flag refused programs whose
        # answer was right; the sequential apply arm answers them instead, one
        # update at a time in XLA's order, so a REPEATED index really does
        # multiply twice.  Both shapes are here, and the duplicate one is the
        # case the gather-multiply-set rewrite would get wrong.
        ("complex scatter multiply (no promise, distinct indices)",
         lambda x, i, u: x.at[i].multiply(u),
         [_crand((6, 4), 326), np.array([0, 5, 2], np.int32),
          _crand((3, 4), 327)], *F32),
        ("complex scatter multiply (no promise, duplicate indices)",
         lambda x, i, u: x.at[i].multiply(u),
         [_crand((6, 4), 328), np.array([1, 1, 4, 1], np.int32),
          _crand((4, 4), 329)], *F32),
        ("complex scatter multiply (no promise, out of bounds)",
         lambda x, i, u: x.at[i].multiply(u),
         [_crand((6, 4), 330), np.array([2, 99, 2], np.int32),
          _crand((3, 4), 331)], *F32),
        ("complex scatter over a partial window",
         lambda x, i, u: x.at[i, 0:2].set(u),
         [_crand((6, 4), 320), np.array([0, 3], np.int32),
          _crand((2, 2), 321)], *EXACT),
        ("complex scatter, vmapped",
         lambda x, u: jax.vmap(lambda r, w: r.at[1].add(w))(x, u),
         [_crand((3, 4), 312), _crand((3,), 313)], *EXACT),
        ("complex scatter inside a scan body",
         lambda c, xs: jax.lax.scan(
             lambda a, r: (a.at[jnp.array([0, 2])].add(r[:2]), None),
             c, xs)[0],
         [_crand((4,), 314), _crand((3, 4), 315)], *EXACT),

        # --- P10: lexicographic and complex sort -------------------------
        # A comparator that is a select TREE rather than one compare means
        # the other execution shape: successive stable argsorts threaded
        # through a permutation, from the last key to the first.  jnp.lexsort
        # is the plain form; jnp.unique over rows and every sparse index
        # canonicalization are the ones the suite is full of.
        ("lexsort, two keys",
         lambda a, b: jnp.lexsort((b, a)),
         [np.array([3, 1, 2, 1, 3, 1], np.int32),
          np.array([1.0, 2.0, -0.0, 0.0, np.nan, -1.0], f)], *EXACT),
        ("lexsort, three keys",
         lambda a, b, c: jnp.lexsort((c, b, a)),
         [np.array([3, 1, 2, 1, 3, 1], np.int32),
          np.array([1.0, 2.0, -0.0, 0.0, np.nan, -1.0], f),
          np.array([5, 4, 3, 2, 1, 0], np.int32)], *EXACT),
        # lax.sort with num_keys > 1 returns EVERY operand permuted by the
        # keys, which is where a permutation applied to the wrong operand
        # would show.
        ("lax.sort with two keys and a payload",
         lambda a, b, c: jax.lax.sort((a, b, c), num_keys=2),
         [np.array([3, 1, 2, 1, 3, 1], np.int32),
          np.array([1.0, 2.0, -0.0, 0.0, np.nan, -1.0], f),
          np.array([5, 4, 3, 2, 1, 0], np.int32)], *EXACT),
        # Stability is what makes successive argsorts equal one
        # lexicographic pass: with every secondary key equal, the primary
        # key's ties must keep their input order.
        ("lexsort stability (ties in every key)",
         lambda a, b: jnp.lexsort((b, a)),
         [np.zeros(7, np.int32), np.zeros(7, f)], *EXACT),
        ("lexsort over a leading axis",
         lambda a, b: jnp.lexsort((b, a), axis=0),
         [_randint((4, 3), 316, np.int32, 0, 2), _rand((4, 3), 317)], *EXACT),
        # The complex comparator is a tree too, over ONE operand pair: the
        # key is the (re, im) pair of canonicalized totalOrder keys packed
        # into a u64.  -0 must tie with +0 and every NaN with every other
        # NaN, or a real part splits a group the imaginary parts then order.
        ("sort complex", lambda x: jnp.sort(x),
         [np.array([3 + 1j, 1 - 2j, 2 + 0j, 1 + 1j], np.complex64)], *EXACT),
        ("argsort complex over signed zeros and NaNs",
         lambda x: jnp.argsort(x),
         [np.array([1 + 1j, 1 - 1j, np.nan * 1j, -0.0 + 0j, 0.0 - 0.0j,
                    2 + 3j, -0.0 + 1j], np.complex64)], *EXACT),
        ("sort complex along the leading axis", lambda x: jnp.sort(x, axis=0),
         [_crand((4, 3), 318)], *EXACT),
        ("unique over complex with NaN and -0 ties",
         lambda x: jnp.unique(x, size=6, fill_value=0),
         [np.array([1 + 1j, 1 - 1j, np.nan * 1j, -0.0 + 0j, 0.0 - 0.0j,
                    2 + 3j], np.complex64)], *EXACT),
        # jnp.unique over ROWS lexsorts the transposed rows and compares
        # neighbours, so an unsorted tie shows up as an extra "unique".
        ("unique rows (a lexsort under a diff)",
         lambda x: jnp.unique(x, axis=0, size=3, fill_value=0),
         [np.array([[1, 2], [1, 1], [1, 2], [0, 9]], np.int32)], *EXACT),
        ("lexsort with a complex key",
         lambda z, a: jnp.lexsort((a, z)),
         [np.array([1 + 1j, 1 - 1j, 1 + 1j, 0 + 0j], np.complex64),
          np.array([3, 2, 1, 0], np.int32)], *EXACT),
        ("lexsort inside a scan body",
         lambda c, xs: jax.lax.scan(
             lambda a, r: (a + jnp.lexsort((r, a.astype(jnp.int32)))[0],
                           None), c, xs)[0],
         [np.zeros(4, np.int32), _rand((3, 4), 319)], *EXACT),

        # a realistic little block: the shapes a model's forward pass has
        ("dense + norm + gelu",
         lambda x, w, b: jax.nn.gelu(
             (x @ w + b) / jnp.sqrt((x @ w + b).var(-1, keepdims=True) + 1e-5)),
         [_rand((8, 16), 27), _rand((16, 32), 28), _rand((32,), 29)], *DOT),
        ("softmax", lambda x: jax.nn.softmax(x, axis=-1),
         [_rand((4, 9), 30)], *F32),
    ]

    # ----------------------------------------------------------------------
    # linalg (P9): the family that computes on the HOST
    # ----------------------------------------------------------------------
    #
    # A factorization is not determined by its inputs the way a matmul is: an
    # eigenvector may be negated, a singular vector rotated inside a
    # degenerate subspace, a Q column's sign flipped, and every one of those
    # is a correct answer.  So most rows below hand back an INVARIANT -- a
    # reconstruction, a residual, an orthogonality product -- which is what
    # jax's own linalg_test asserts on, and only the quantities that really
    # are unique (a cholesky factor, the eigenvalues, the singular values)
    # are compared elementwise.  The tolerance is the band an f32
    # factorization earns, not the elementwise one.
    LIN = (2e-5, 2e-5)

    def adj(a):
        return jnp.swapaxes(a, -1, -2).conj()

    def qr_inv(x, mode="reduced"):
        q, r = jnp.linalg.qr(x, mode=mode)
        return q @ r, adj(q) @ q

    def eigh_inv(x):
        w, v = jnp.linalg.eigh(x)
        return w, (v * w) @ v.conj().T, v.conj().T @ v

    def svd_inv(x, full_matrices=True):
        u, s, vt = jnp.linalg.svd(x, full_matrices=full_matrices)
        k = min(x.shape[0], x.shape[1])
        return s, (u[:, :k] * s) @ vt[:k], u.conj().T @ u

    def eig_inv(x):
        w, v = jnp.linalg.eig(x)
        # The eigenvalues come back in LAPACK's order, which both backends
        # get from the same routine; sorting them makes the row independent
        # of that anyway.  The residual is the part that says the vectors go
        # with the values.
        order = jnp.argsort(w.real * 1e6 + w.imag)
        return w[order], jnp.abs(x @ v - v * w).max()

    tri = jax.lax.linalg.triangular_solve
    cases += [
        # --- cholesky: the factor is unique, so it compares elementwise ----
        ("cholesky f32", lambda x: jnp.linalg.cholesky(x), [_spd(5, 300)],
         *LIN),
        ("cholesky upper",
         lambda x: jax.scipy.linalg.cholesky(x, lower=False), [_spd(5, 301)],
         *LIN),
        ("cholesky c64", lambda x: jnp.linalg.cholesky(x), [_cspd(4, 302)],
         *LIN),
        ("cholesky batched", lambda x: jnp.linalg.cholesky(x),
         [np.stack([_spd(3, 303), _spd(3, 304), _spd(3, 305)])], *LIN),
        # Not positive definite: XLA fills the result with NaN rather than
        # failing, and `_compare` demands the NaNs land in the same places.
        ("cholesky of a singular matrix", lambda x: jnp.linalg.cholesky(x),
         [np.zeros((3, 3), np.float32)], *LIN),
        ("cholesky vmapped",
         lambda x: jax.vmap(jnp.linalg.cholesky)(x),
         [np.stack([_spd(4, 306), _spd(4, 307)])], *LIN),

        # --- qr: reconstruction and orthogonality --------------------------
        ("qr tall", qr_inv, [_rand((6, 3), 310)], *LIN),
        ("qr wide", qr_inv, [_rand((3, 6), 311)], *LIN),
        ("qr square", qr_inv, [_rand((4, 4), 312)], *LIN),
        # `complete` asks for more columns of Q than there are reflectors,
        # which is the zero-tau completion (`_householder_product`'s pad).
        ("qr tall complete", lambda x: qr_inv(x, "complete"),
         [_rand((6, 3), 313)], *LIN),
        ("qr complete orthonormal",
         lambda x: (lambda q: q.T @ q)(jnp.linalg.qr(x, mode="complete")[0]),
         [_rand((5, 2), 314)], *LIN),
        ("qr c64", qr_inv, [(_rand((5, 3), 315)
                             + 1j * _rand((5, 3), 316)).astype(np.complex64)],
         *LIN),
        ("qr batched", lambda x: qr_inv(x)[0], [_rand((3, 5, 4), 317)], *LIN),
        ("qr r factor",
         lambda x: jnp.abs(jnp.linalg.qr(x)[1]), [_rand((6, 3), 318)], *LIN),
        # jax.nn.initializers.orthogonal is a QR with a sign correction, and
        # is how a real program reaches this pair of targets.
        ("orthogonal initializer",
         lambda k: (lambda q: q.T @ q)(
             jax.nn.initializers.orthogonal()(k, (6, 3))),
         [jax.random.key(0)], *LIN),

        # --- eigh ----------------------------------------------------------
        ("eigh symmetric", eigh_inv, [_sym(5, 320)], *LIN),
        ("eigh upper triangle",
         lambda x: jnp.linalg.eigvalsh(x, UPLO="U"), [_sym(5, 321)], *LIN),
        ("eigh hermitian c64", eigh_inv, [_herm(4, 322)], *LIN),
        ("eigh batched", lambda x: jnp.linalg.eigh(x)[0],
         [np.stack([_sym(3, 323), _sym(3, 324)])], *LIN),
        # Degenerate spectrum: the eigenVECTORS are only defined up to a
        # rotation inside each repeated subspace, so nothing but the values
        # and the invariants can be compared at all.
        ("eigh with degenerate eigenvalues",
         lambda x: (jnp.linalg.eigh(x)[0],
                    (lambda w, v: (v * w) @ v.T)(*jnp.linalg.eigh(x))),
         [np.diag([2.0, 2.0, 2.0, 5.0]).astype(np.float32)], *LIN),
        ("eigh of the identity", lambda x: jnp.linalg.eigh(x)[0],
         [np.eye(4, dtype=np.float32)], *LIN),
        ("eigh grad",
         lambda x: jax.grad(lambda z: jnp.linalg.eigvalsh(z).sum())(x),
         [_sym(4, 325)], *LIN),

        # --- svd -----------------------------------------------------------
        ("svd values", lambda x: jnp.linalg.svd(x, compute_uv=False),
         [_rand((6, 4), 330)], *LIN),
        ("svd full", svd_inv, [_rand((6, 4), 331)], *LIN),
        ("svd thin", lambda x: svd_inv(x, False), [_rand((6, 4), 332)], *LIN),
        ("svd wide", svd_inv, [_rand((3, 7), 333)], *LIN),
        ("svd c64", lambda x: jnp.linalg.svd(x, compute_uv=False),
         [(_rand((5, 3), 334) + 1j * _rand((5, 3), 335)).astype(np.complex64)],
         *LIN),
        # Rank deficient: the trailing singular values are zero and their
        # vectors arbitrary, so only the values and the reconstruction hold.
        ("svd rank deficient",
         lambda x: (jnp.linalg.svd(x, compute_uv=False),
                    (lambda u, s, vt: (u[:, :3] * s) @ vt[:3])(
                        *jnp.linalg.svd(x))),
         [np.outer(np.arange(1, 6), np.arange(1, 4)).astype(np.float32)],
         *LIN),
        ("svd batched", lambda x: jnp.linalg.svd(x, compute_uv=False),
         [_rand((3, 4, 5), 336)], *LIN),
        ("pinv", lambda x: jnp.linalg.pinv(x), [_rand((6, 3), 337)], *LIN),
        ("matrix rank", lambda x: jnp.linalg.matrix_rank(x),
         [np.outer(np.arange(1, 6), np.arange(1, 4)).astype(np.float32)],
         *EXACT),
        ("2-norm and condition number",
         lambda x: (jnp.linalg.norm(x, 2), jnp.linalg.cond(x)),
         [_spd(4, 338)], *LIN),

        # --- eig (complex results from a real operand) ---------------------
        ("eig of a real matrix", eig_inv, [_rand((4, 4), 340)], *LIN),
        ("eig of a complex matrix", eig_inv,
         [(_rand((3, 3), 341) + 1j * _rand((3, 3), 342)).astype(np.complex64)],
         *LIN),
        # A rotation block: complex conjugate eigenvalues, which is where the
        # real geev's packed eigenvector columns get unpacked.
        ("eig with a conjugate pair", lambda x: jnp.sort(
            jnp.linalg.eigvals(x).imag),
         [np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
                   np.float32)], *LIN),
        ("eigvals batched", lambda x: jnp.sort(jnp.linalg.eigvals(x).real, -1),
         [np.stack([_sym(3, 343), _sym(3, 344)])], *LIN),

        # --- lu ------------------------------------------------------------
        ("lu factor", lambda x: jax.scipy.linalg.lu_factor(x),
         [_rand((4, 4), 350)], *LIN),
        ("lu reconstruction",
         lambda x: (lambda p, l, u: p @ l @ u)(*jax.scipy.linalg.lu(x)),
         [_rand((5, 3), 351)], *LIN),
        ("lu permutation", lambda x: jax.lax.linalg.lu(x)[2],
         [_rand((4, 4), 352)], *EXACT),

        # --- triangular solve: the four side/transpose combinations --------
        ("triangular solve left upper",
         lambda a, b: tri(a, b, left_side=True, lower=False),
         [_triangular(4, 360), _rand((4, 3), 361)], *LIN),
        ("triangular solve left lower",
         lambda a, b: tri(a, b, left_side=True, lower=True),
         [_triangular(4, 362, lower=True), _rand((4, 3), 363)], *LIN),
        ("triangular solve right upper",
         lambda a, b: tri(a, b, left_side=False, lower=False),
         [_triangular(4, 364), _rand((3, 4), 365)], *LIN),
        ("triangular solve right lower",
         lambda a, b: tri(a, b, left_side=False, lower=True),
         [_triangular(4, 366, lower=True), _rand((3, 4), 367)], *LIN),
        ("triangular solve transposed",
         lambda a, b: tri(a, b, left_side=True, lower=True,
                          transpose_a=True),
         [_triangular(4, 368, lower=True), _rand((4, 2), 369)], *LIN),
        ("triangular solve unit diagonal",
         lambda a, b: tri(a, b, left_side=True, lower=True,
                          unit_diagonal=True),
         [_triangular(4, 370, lower=True), _rand((4, 2), 371)], *LIN),
        ("triangular solve adjoint c64",
         lambda a, b: tri(a, b, left_side=True, lower=True,
                          transpose_a=True, conjugate_a=True),
         [np.tril(_rand((3, 3), 372) + 1j * _rand((3, 3), 373)
                  + 3 * np.eye(3)).astype(np.complex64),
          (_rand((3, 2), 374) + 1j * _rand((3, 2), 375)).astype(np.complex64)],
         *LIN),
        ("triangular solve batched",
         lambda a, b: tri(a, b, left_side=True, lower=True),
         [np.stack([_triangular(3, 376, lower=True),
                    _triangular(3, 377, lower=True)]),
          _rand((2, 3, 2), 378)], *LIN),
        # One matrix against a batch of right-hand sides: jax's batching rule
        # broadcasts `a`, which is the operand shape the handler has to
        # stretch to `b`'s batch (`np.broadcast_to` in the Python handler).
        ("triangular solve vmapped over b",
         lambda a, b: jax.vmap(
             lambda y: tri(a, y, left_side=True, lower=True))(b),
         [_triangular(3, 379, lower=True), _rand((4, 3, 2), 380)], *LIN),
        # A zero on the diagonal: XLA divides THROUGH it rather than failing,
        # and `_compare` demands the infinities land in the same places.
        ("triangular solve with a zero pivot",
         lambda a, b: tri(a, b, left_side=True, lower=True),
         [np.array([[1.0, 0.0], [2.0, 0.0]], np.float32),
          np.array([[1.0], [1.0]], np.float32)], *LIN),

        # --- the solvers built on the two above ----------------------------
        ("linalg.solve", lambda a, b: jnp.linalg.solve(a, b),
         [_spd(4, 390), _rand((4, 2), 391)], *LIN),
        ("linalg.inv", lambda x: jnp.linalg.inv(x), [_spd(4, 392)], *LIN),
        ("linalg.det", lambda x: jnp.linalg.det(x), [_spd(4, 393)],
         1e-5, 1e-4),
        ("linalg.slogdet", lambda x: jnp.linalg.slogdet(x), [_spd(4, 394)],
         *LIN),
        ("det grad", lambda x: jax.grad(jnp.linalg.det)(x), [_spd(3, 395)],
         1e-5, 1e-4),
        ("cho_solve", lambda a, b: jax.scipy.linalg.cho_solve(
            jax.scipy.linalg.cho_factor(a), b),
         [_spd(4, 396), _rand((4, 2), 397)], *LIN),
        ("lstsq", lambda a, b: jnp.linalg.lstsq(a, b)[0],
         [_rand((6, 3), 398), _rand((6,), 399)], *LIN),
        ("solve grad",
         lambda a, b: jax.grad(lambda z: jnp.linalg.solve(z, b).sum())(a),
         [_spd(4, 400), _rand((4, 2), 401)], *LIN),
        ("matrix_power", lambda x: jnp.linalg.matrix_power(x, -2),
         [_spd(3, 402)], *LIN),

        # --- schur / hessenberg / tridiagonal ------------------------------
        ("schur form", lambda x: jax.scipy.linalg.schur(x)[0],
         [_rand((4, 4), 410)], *LIN),
        ("schur reconstruction",
         lambda x: (lambda t, z: z @ t @ z.T)(*jax.scipy.linalg.schur(x)),
         [_rand((4, 4), 411)], *LIN),
        ("hessenberg", lambda x: jax.scipy.linalg.hessenberg(x),
         [_rand((4, 4), 412)], *LIN),
        ("tridiagonal", lambda x: jax.lax.linalg.tridiagonal(x)[1:3],
         [_sym(4, 413)], *LIN),
        ("tridiagonal solve",
         lambda dl, d, du, b: jax.lax.linalg.tridiagonal_solve(dl, d, du, b),
         [np.array([0.0, 1.0, 1.0, 1.0], np.float32),
          np.array([4.0, 4.0, 4.0, 4.0], np.float32),
          np.array([1.0, 1.0, 1.0, 0.0], np.float32),
          _rand((4, 2), 414)], *LIN),

        # --- ApproxTopK, answered exactly ----------------------------------
        ("approx_max_k", lambda x: jax.lax.approx_max_k(x, 3),
         [_rand((16,), 420)], *EXACT),
        ("approx_min_k", lambda x: jax.lax.approx_min_k(x, 4),
         [_rand((16,), 421)], *EXACT),
        ("approx_max_k over rows",
         lambda x: jax.lax.approx_max_k(x, 2, reduction_dimension=1),
         [_rand((3, 8), 422)], *EXACT),
        ("approx_max_k unaggregated",
         lambda x: jax.lax.approx_max_k(x, 2, aggregate_to_topk=False),
         [_rand((64,), 423)], *EXACT),

        # --- a host op inside control flow ---------------------------------
        # A block holding a host call is IMPURE, so neither the loop body nor
        # the main may be traced through mx::compile; that decision is what
        # this row exercises (a compiled trace would try to read a tracer on
        # the host and compute on nothing).
        ("cholesky inside a fori body",
         lambda x: jax.lax.fori_loop(
             0, 3, lambda i, c: jnp.linalg.cholesky(c @ c.T + 4 * jnp.eye(3)),
             x),
         [np.eye(3, dtype=np.float32)], *LIN),
        ("a solve inside a scan",
         lambda a, xs: jax.lax.scan(
             lambda c, y: (c + jnp.linalg.solve(a, y), c.sum()),
             jnp.zeros((3,), np.float32), xs)[0],
         [_spd(3, 430), _rand((4, 3), 431)], *LIN),
        ("eigh inside a cond",
         lambda p, x: jax.lax.cond(p, lambda z: jnp.linalg.eigvalsh(z),
                                   lambda z: jnp.diag(z), x),
         [np.bool_(True), _sym(3, 432)], *LIN),

        # ------------------------------------------------------------------
        # P11: the emulated grids, reduce_precision, and the scatter tail
        # ------------------------------------------------------------------
        #
        # The emulated element types (src/metaljax/dtypes.py `EMULATED`) hold
        # their VALUES in a wider storage dtype, so three things have to agree
        # with the CPU backend and are tested separately: the wire DECODE (a
        # host buffer of every canonical code, read as f32), the wire ENCODE
        # (an f32 array converted onto the grid and read back), and the
        # ROUND TRIP through both.  The round trip is exhaustive -- every bit
        # pattern the format has -- which is the only way to know that a NaN
        # or subnormal encoding nobody thought about survives.
        *_subbyte_cases(),
        # Arithmetic on a grid: the OCP FP4/FP6 formats re-round after every
        # operation (4 + 4 is 6 on f4E2M1FN, not 8) and i4/ui4 wrap to four
        # bits, which is what the entry's `regrid` field is for.  The float8
        # family deliberately does NOT re-round between ops, exactly as the
        # Python engine has not since the emulation landed.
        ("f4E2M1FN add re-grids", lambda a: a + a,
         [np.array([1.0, 2.0, 4.0, 3.0], ml_dtypes.float4_e2m1fn)], 0, 0),
        ("f4E2M1FN multiply re-grids", lambda a: a * a,
         [np.array([1.0, 2.0, 4.0, 3.0], ml_dtypes.float4_e2m1fn)], 0, 0),
        # The FP6 pair has no arithmetic row: XLA:CPU cannot compile an
        # `arith.subf` on one at all, so there would be no reference.  Their
        # grid is covered by the round-trip and encode rows above, and the
        # rounding they share with f4E2M1FN is one code path.
        ("f4E2M1FN subtract re-grids", lambda a: a - a[::-1],
         [np.array([1.0, 2.0, 4.0, 0.5], ml_dtypes.float4_e2m1fn)], 0, 0),
        ("f4E2M1FN maximum does not re-grid anything new",
         lambda a: jnp.maximum(a, a[::-1]),
         [np.array([1.0, 2.0, 4.0, 0.5], ml_dtypes.float4_e2m1fn)], 0, 0),
        ("i4 arithmetic wraps to four bits",
         lambda a: (a + a, a * a, a - a[::-1], jnp.maximum(a, a[::-1])),
         [np.array([-8, -1, 3, 7], ml_dtypes.int4)], 0, 0),
        ("ui4 arithmetic wraps to four bits",
         lambda a: (a + a, a * a),
         [np.array([0, 5, 9, 15], ml_dtypes.uint4)], 0, 0),
        ("f8E4M3FN arithmetic keeps its wide storage",
         lambda a: (a + a, a * a, (a + a).astype(jnp.float32)),
         [np.array([1.0, 2.0, 3.5, -0.5], ml_dtypes.float8_e4m3fn)], 0, 0),
        # The shape ops carry an emulated value without touching it, and a
        # constant of one arrives as the TYPE's encoding in the IR (bit-packed
        # for the sub-byte widths), which only the typed iterator can read.
        ("emulated values through the shape ops",
         lambda a: (a.reshape(2, 2).T, jnp.concatenate([a, a]),
                    jnp.where(jnp.arange(4) > 1, a, a[::-1]),
                    a > jnp.array(1.0, ml_dtypes.float8_e4m3fn)),
         [np.array([1.0, 2.0, 3.5, -0.5], ml_dtypes.float8_e4m3fn)], 0, 0),
        ("an f8E4M3FN constant, dense and splat",
         lambda a: (a + jnp.array([1.0, 0.5, 2.0, -1.0],
                                  ml_dtypes.float8_e4m3fn),
                    a * jnp.full((4,), 2.0, ml_dtypes.float8_e4m3fn)),
         [np.array([1.0, 2.0, 3.5, -0.5], ml_dtypes.float8_e4m3fn)], 0, 0),
        ("an i4 constant and a gather of one",
         lambda a, i: (a + jnp.array([1, 2, 3, 4], ml_dtypes.int4), a[i]),
         [np.array([-8, -1, 3, 7], ml_dtypes.int4),
          np.array([3, 0, 1], np.int32)], 0, 0),
        # bitcast_convert with a 4-bit end: XLA packs two nibbles per byte
        # along the minor-most dimension, low nibble first.
        ("i4 <-> ui4 bitcast reinterprets the nibble",
         lambda a: jax.lax.bitcast_convert_type(a, ml_dtypes.uint4),
         [np.array([-8, -1, 3, 7], ml_dtypes.int4)], 0, 0),
        ("i4 -> i8 bitcast packs pairs",
         lambda a: jax.lax.bitcast_convert_type(a.reshape(2, 2), jnp.int8),
         [np.array([-8, -1, 3, 7], ml_dtypes.int4)], 0, 0),
        ("i8 -> i4 bitcast unpacks bytes",
         lambda a: jax.lax.bitcast_convert_type(a, ml_dtypes.int4),
         [np.array([0x12, -0x7F, 0, -1], np.int8)], 0, 0),
        ("a zero-size 4-bit bitcast",
         lambda a: jax.lax.bitcast_convert_type(a, jnp.int8),
         [np.zeros((0, 2), ml_dtypes.int4)], 0, 0),
        ("converts between the grids and the real types",
         lambda a: (a.astype(jnp.float32), a.astype(jnp.int8),
                    a.astype(ml_dtypes.uint4)),
         [np.array([-8, -1, 3, 7], ml_dtypes.int4)], 0, 0),

        # reduce_precision: the four arms (identity, the bf16 and f16 grids,
        # and the general any-e/m rounding) over the three float storages.
        *[(f"reduce_precision e{e}m{m} on {nm}",
           (lambda v, e=e, m=m: jax.lax.reduce_precision(v, e, m)),
           [(np.arange(-8, 8, dtype=np.float32) * 0.3).astype(dt)], 0, 0)
          for e, m in [(8, 23), (8, 7), (5, 10), (5, 2), (4, 3), (3, 4),
                       (1, 0), (8, 0), (2, 5)]
          for nm, dt in [("f32", np.float32), ("f16", np.float16),
                         ("bf16", ml_dtypes.bfloat16)]],
        ("reduce_precision at the specials",
         lambda v: jax.lax.reduce_precision(v, 5, 2),
         [np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1e30, -1e-30,
                    65504.0], np.float32)], 0, 0),
        ("reduce_precision inside a counted loop",
         lambda x: jax.lax.fori_loop(
             0, 4, lambda i, c: jax.lax.reduce_precision(c * 1.5, 5, 10), x),
         [np.arange(4, dtype=np.float32)], 1e-6, 1e-6),

        # The scatter tail.  A computed body with no uniqueness promise is
        # applied ONE UPDATE AT A TIME, in row-major order, because a body
        # need be neither associative nor idempotent -- `_dup` below is the
        # row that says so (index 1 is written twice, and sin(sin(x)) is not
        # sin(x)).
        ("scatter_apply over distinct indices",
         lambda x, i: x.at[i].apply(jnp.sin),
         [np.arange(8, dtype=np.float32), np.array([0, 3, 5], np.int32)],
         1e-6, 1e-6),
        ("scatter_apply over DUPLICATE indices",
         lambda x, i: x.at[i].apply(jnp.sin),
         [np.arange(8, dtype=np.float32), np.array([1, 1, 5], np.int32)],
         1e-6, 1e-6),
        ("scatter_apply with an out-of-bounds index",
         lambda x, i: x.at[i].apply(jnp.sin),
         [np.arange(8, dtype=np.float32), np.array([0, 9, 5], np.int32)],
         1e-6, 1e-6),
        ("scatter_apply over rows",
         lambda x, i: x.at[i, :].apply(jnp.exp),
         [np.arange(12, dtype=np.float32).reshape(4, 3),
          np.array([0, 2], np.int32)], 1e-5, 1e-5),
        ("scatter_apply on integers",
         lambda x, i: x.at[i].apply(lambda z: z * 3),
         [np.arange(6, dtype=np.int32), np.array([1, 4], np.int32)], 0, 0),
        ("scatter_apply under vmap",
         jax.vmap(lambda x, i: x.at[i].apply(jnp.sin)),
         [np.arange(12, dtype=np.float32).reshape(3, 4),
          np.array([[0], [2], [3]], np.int32)], 1e-6, 1e-6),
        ("scatter_apply on a rank-0 operand",
         lambda x: x.at[()].apply(jnp.sin), [np.float32(2.0)], 1e-6, 1e-6),
        ("scatter_apply on complex",
         lambda x, i: x.at[i].apply(lambda z: z * z),
         [np.array([1 + 1j, 2 - 2j, 0j, 3j], np.complex64),
          np.array([0, 2], np.int32)], 1e-6, 1e-6),
        # A rank-0 operand with an empty coordinate vector: the scatter IS its
        # combiner.  jax reaches it through `jax.experimental.sparse` on a
        # 0-d array, whose updates are reduced before the scatter.
        ("a rank-0 scatter's combiner",
         lambda a, u: (a.at[()].set(u), a.at[()].add(u), a.at[()].max(u),
                       a.at[()].min(u), a.at[()].multiply(u)),
         [np.float32(3.0), np.float32(9.0)], 0, 0),
        ("a 0-d BCOO densifies through a rank-0 scatter",
         lambda d: _bcoo_0d(d),
         [np.array([1.0, 2.0, 4.0], np.float32)], 1e-6, 1e-6),

        # select_and_scatter: max/min-pool backward.  Its scatter-add lands on
        # overlapping windows, so the GPU's answer is order-nondeterministic
        # in the last bits -- these rows carry a tolerance rather than pinning
        # bytes, which is the disposition this family shipped with.
        ("max-pool backward (select_and_scatter)",
         jax.grad(lambda v: jnp.sum(jax.lax.reduce_window(
             v, -jnp.inf, jax.lax.max, (1, 2, 2), (1, 1, 1), "VALID") ** 2)),
         [np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.7],
         1e-6, 1e-6),
        ("min-pool backward, strided and padded",
         jax.grad(lambda v: jnp.sum(jax.lax.reduce_window(
             v, jnp.inf, jax.lax.min, (1, 2, 2), (1, 2, 2), "SAME") ** 2)),
         [np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.7],
         1e-6, 1e-6),
        ("max-pool backward with SAME padding",
         jax.grad(lambda v: jnp.sum(jax.lax.reduce_window(
             v, -jnp.inf, jax.lax.max, (1, 3, 3), (1, 2, 2), "SAME") ** 2)),
         [np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.7],
         1e-6, 1e-6),
        ("select_and_scatter_add directly",
         lambda o, s: _sas_add(s, o, jax.lax.ge_p, (2, 2, 2), (1, 1, 1),
                               [(0, 0), (0, 0), (0, 0)]),
         [np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.7,
          np.arange(6, dtype=np.float32).reshape(1, 2, 3) + 1.0], 1e-6, 1e-6),
        ("select_and_scatter_add with LE and padding",
         lambda o, s: _sas_add(s, o, jax.lax.le_p, (2, 2, 2), (1, 2, 2),
                               [(0, 0), (1, 1), (1, 1)]),
         [np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.7,
          np.arange(6, dtype=np.float32).reshape(1, 2, 3) + 1.0],
         1e-6, 1e-6),
        ("select_and_scatter_add under vmap",
         jax.vmap(lambda o, s: _sas_add(s, o, jax.lax.ge_p, (2, 2), (1, 1),
                                        [(0, 0), (0, 0)])),
         [np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.7,
          np.arange(12, dtype=np.float32).reshape(2, 2, 3)], 1e-6, 1e-6),
    ]
    cases += _recognizer_cases()
    return cases


# --------------------------------------------------------------------------
# the recognizer emits (P17)
# --------------------------------------------------------------------------
#
# Each of these graphs is one the native lowering REWRITES: a dequantize-and-
# matmul chain into `quantized_matmul`, a dense expert dispatch into
# `gather_mm`/`gather_qmm`, a softmax attention into
# `fast::scaled_dot_product_attention`.  The CPU backend runs the literal
# chain, so every row is the fused answer against the unfused one -- which is
# the only differential that can catch a misread axis, a wrong pack layout or
# a router the rewrite read backwards.
#
# The tolerances say what each rewrite is allowed to change.  A pack is EXACT
# (the reconstructed weight is bit-identical to a float32 dequantization), so
# what is left is the dot's own summation order, which is the `DOT` band; the
# gathered expert sum runs over K terms instead of E and the fused attention
# is a different kernel, so those get the same band their dtype earns.
#
# `tests/test_qmm.py`, `test_qmm_mxfp4.py`, `test_moe.py` and `test_sdpa.py`
# are the graphs' source: these are the same layer shapes, cut down to what a
# differential needs.  They live HERE as well because those files assert on
# Stage 1's Python counters, which a plugin with no interpreter in it cannot
# tick -- the numbers are the part that carries over.


def _quantize(rows, cols, block, dtype, seed=0, bits=4):
    """Codes + scale/zero maps in keras' storage layout, and the exact
    dequantized weight."""
    rng = np.random.RandomState(seed)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = rng.randint(lo, hi + 1, size=(rows, cols)).astype(np.int8)
    bsz = rows if block < 0 else block
    ng = rows // bsz
    scale = ((rng.rand(ng, cols).astype(np.float32) + 0.5) * 0.05)
    zero = (np.zeros((ng, cols), np.int8) if block < 0
            else rng.randint(-3, 4, size=(ng, cols)).astype(np.int8))
    g_idx = (np.arange(rows) // bsz).astype(np.float32)
    packed = ((q[:, 0::2] & 0x0F) | (q[:, 1::2] << 4)).astype(np.int8)
    return q, packed, scale.astype(dtype), zero, g_idx


def _mxfp4(shape_nk, seed, dtype, exp=(118, 133)):
    """Random on-grid MXFP4 blocks + E8M0 scale bytes for a [.., N, K] weight.

    `exp` is the E8M0 byte range, i.e. the per-group scale's exponent + 127.
    The default spans what a real checkpoint uses; a row that feeds its own
    output back wants a narrow band around 1.0, or three iterations of a
    64-wide contraction leave the differential comparing 1e10s.
    """
    rng = np.random.RandomState(seed)
    n, k = shape_nk[-2], shape_nk[-1]
    lead = tuple(shape_nk[:-2])
    codes = rng.randint(0, 16, size=lead + (n, k)).astype(np.uint8)
    blocks = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    sb = rng.randint(exp[0], exp[1], size=lead + (n, k // 32)).astype(np.uint8)
    return blocks, sb


def _recognizer_cases():
    import jax
    import jax.numpy as jnp

    DOT = (1e-5, 1e-5)
    HALF = (5e-3, 5e-3)
    E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    np.float32)
    # Byte 0 is 2**-127, a subnormal Metal may flush, and byte 255 is NaN by
    # the OCP spec; a real checkpoint lives at 117..126.
    _tab = np.ldexp(np.ones(256), np.arange(256) - 127)
    _tab[255] = np.nan
    SCALE_TABLE = _tab.astype(np.float32)

    def unpack(packed, columns):
        lo = jnp.bitwise_and(packed, jnp.int8(0x0F))
        lo = jnp.where(lo > 7, lo - 16, lo)
        hi = jnp.right_shift(packed, jnp.int8(4))
        w = jnp.reshape(jnp.stack([lo, hi], axis=-1),
                        packed.shape[:-1] + (columns,))
        return w

    def dense_sub(packed, scale, zero, g_idx, x, columns):
        """keras Dense._int4_call, sub-channel branch."""
        w = unpack(packed, columns)
        g = g_idx.astype(jnp.int32)
        s = jnp.take(scale, g, axis=0)
        z = jnp.take(zero, g, axis=0)
        return x @ ((w.astype(x.dtype) - z.astype(x.dtype)) * s)

    def dense_perchannel(packed, scale, x, columns):
        """...and its per-channel branch: the scale divides the OUTPUT."""
        return (x @ unpack(packed, columns).astype(x.dtype)) / scale

    def einsum_out(packed, scale, zero, g_idx, x, n, h, d):
        """keras EinsumDense, `btnh,nhd->btd`: the groups arrive interleaved
        along the canonical contraction axis, so the pack has to permute it."""
        w = unpack(packed, d)
        g = g_idx.astype(jnp.int32)
        wf = ((w.astype(x.dtype) - jnp.take(zero, g, axis=0).astype(x.dtype))
              * jnp.take(scale, g, axis=0))
        return jnp.einsum("btnh,nhd->btd", x, jnp.reshape(wf, (n, h, d)))

    def mxfp4_weight(blocks, sb, k, dtype):
        vt = jnp.asarray(E2M1, dtype=dtype)
        st = jnp.asarray(SCALE_TABLE)
        lead = tuple(blocks.shape[:-1])
        lo = jnp.bitwise_and(blocks, jnp.uint8(0x0F))
        hi = jnp.right_shift(blocks, jnp.uint8(4))
        nib = jnp.reshape(jnp.stack([lo, hi], axis=-1), lead + (k,))
        vals = jnp.take(vt, nib.astype(jnp.int32), axis=0)
        scale = jnp.take(st, sb.astype(jnp.int32), axis=0)
        w = (jnp.reshape(vals, lead + (k // 32, 32))
             * scale[..., None].astype(dtype))
        return jnp.reshape(w, lead + (k,))

    def moe_block(x, wg, wd, k):
        """The dense dispatch every jax MoE lowers to: all E experts, then
        the router's weights null the E - K that were never selected."""
        logits = x @ wg                                   # [T, E]
        vals, idx = jax.lax.top_k(logits, k)              # [T, K]
        w = jax.nn.softmax(vals, axis=-1)
        onehot = (idx[..., None] == jnp.arange(wg.shape[1])).astype(w.dtype)
        scores = jnp.sum(onehot * w[..., None], axis=1)   # [T, E]
        y = jnp.einsum("th,ehd->etd", x, wd)              # [E, T, D]
        return jnp.sum(y * scores.T[..., None], axis=0)

    def moe_mxfp4(x, wg, blocks, sb, k, kdim):
        wd = mxfp4_weight(blocks, sb, kdim, x.dtype)
        return moe_block(x, wg, jnp.swapaxes(wd, -1, -2), k)

    def attn(q, k, v, scale):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        p = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhqk,bkhd->bqhd", p, v)

    def attn_causal(q, k, v, scale):
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        mask = jnp.tril(jnp.ones((q.shape[2], k.shape[2]), bool))
        # `finfo.min`, which is what jax's own causal masks use: a sentinel
        # too small to be one is a `select` the rewrite must NOT read as a
        # mask (sdpa.py `_MASK_FRACTION`).
        logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, -1), v)

    def attn_additive(q, k, v, bias, scale):
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale + bias
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, -1), v)

    out = []

    # --- qmm ---------------------------------------------------------
    for dt, tol in (("float32", DOT), ("bfloat16", HALF)):
        rows, cols = 256, 128
        _q, packed, scale, zero, g_idx = _quantize(rows, cols, 128, dt)
        x = _rand((4, rows), 3).astype(dt) * 0.5
        out.append((f"qmm int4 sub-channel {dt}",
                    lambda p, s, z, g, a, c=cols: dense_sub(p, s, z, g, a, c),
                    [packed, scale, zero, g_idx, x], *tol))
    rows, cols = 128, 64
    _q, packed, scale, zero, g_idx = _quantize(rows, cols, 64, "float32",
                                               seed=5, bits=8)
    x = _rand((3, rows), 1).astype(np.float32) * 0.5
    out.append(("qmm int8 codes",
                lambda q, s, z, g, a: (a @ ((q.astype(a.dtype)
                                             - jnp.take(z, g.astype(jnp.int32),
                                                        axis=0).astype(a.dtype))
                                            * jnp.take(s,
                                                       g.astype(jnp.int32),
                                                       axis=0))),
                [_q, scale, zero, g_idx, x], *DOT))
    _q, packed, scale, zero, g_idx = _quantize(128, 64, -1, "float32", seed=7)
    # The per-channel form folds the divide into the weight scale: `1/s` is
    # computed once in f32 instead of dividing every output element, which is
    # a rounding CHANGE and not a rounding error (Stage 1 compares both sides
    # against an exactly dequantized reference and requires the fused one to
    # be no further from it; the band here is what that difference measures).
    out.append(("qmm per-channel (scale divides the output)",
                lambda p, s, a: dense_perchannel(p, s, a, 64),
                [packed, scale[0], _rand((5, 128), 2)], 1e-4, 1e-4))
    n, h, d = 8, 32, 256
    _q, packed, scale, zero, g_idx = _quantize(n * h, d, 128, "float32",
                                               seed=9)
    out.append(("qmm einsum projection (interleaved groups, regrouped)",
                lambda p, s, z, g, a: einsum_out(p, s, z, g, a, n, h, d),
                [packed, scale, zero, g_idx,
                 _rand((1, 2, n, h), 4) * 0.3], *DOT))
    for dt, tol in (("float32", DOT), ("bfloat16", HALF)):
        blocks, sb = _mxfp4((64, 128), 11, dt)
        out.append((f"qmm mxfp4 projection {dt}",
                    lambda b, s, a: jnp.einsum(
                        "th,nh->tn", a, mxfp4_weight(b, s, 128, a.dtype)),
                    [blocks, sb, _rand((6, 128), 12).astype(dt) * 0.4], *tol))
    blocks, sb = _mxfp4((4, 32, 64), 13, "float32")
    out.append(("qmm mxfp4 batched experts",
                lambda b, s, a: jnp.einsum(
                    "etm,ehm->eth", a, mxfp4_weight(b, s, 64, a.dtype)),
                [blocks, sb, _rand((4, 3, 64), 14) * 0.4], *DOT))

    def qmm_loop(packed, scale, zero, g_idx, x):
        def body(c):
            i, y = c
            return i + 1, jnp.tanh(dense_sub(packed, scale, zero, g_idx, y,
                                             128) * 0.3)
        return jax.lax.while_loop(lambda c: c[0] < 3, body, (0, x))[1]

    _q, packed, scale, zero, g_idx = _quantize(128, 128, 128, "float32",
                                               seed=15)
    out.append(("qmm inside a decode loop (packs cross the region)", qmm_loop,
                [packed, scale, zero, g_idx, _rand((2, 128), 16) * 0.3], *DOT))

    # --- moe ---------------------------------------------------------
    for E, K, T in ((8, 2, 5), (32, 4, 1)):
        wg = _rand((64, E), 20) * 0.5
        wd = _rand((E, 64, 32), 21) * 0.3
        out.append((f"moe gather E{E}/K{K}/T{T}",
                    lambda a, g, w, k=K: moe_block(a, g, w, k),
                    [_rand((T, 64), 22), wg, wd], *DOT))
    wg = _rand((64, 4), 23) * 0.5
    blocks, sb = _mxfp4((4, 32, 64), 24, "float32")
    out.append(("moe gather with mxfp4 experts (gather_qmm)",
                lambda a, g, b, s: moe_mxfp4(a, g, b, s, 2, 64),
                [_rand((3, 64), 25), wg, blocks, sb], *DOT))

    def moe_loop(x, wg, wd):
        pre = moe_block(x, wg, wd, 2)

        def body(c):
            i, tok = c
            return i + 1, moe_block(tok, wg, wd, 2)

        return jax.lax.while_loop(lambda c: c[0] < 3, body, (0, pre[-1:]))[1]

    # Three dispatches deep, each summing K terms where the dense graph sums
    # E: the reduction order differs at every step and the loop feeds its own
    # output back, so this row earns a band the single dispatches above do
    # not.
    out.append(("moe gather inside a decode loop", moe_loop,
                [_rand((5, 64), 26), _rand((64, 8), 27) * 0.5,
                 _rand((8, 64, 64), 28) * 0.3], 1e-4, 1e-4))

    def moe_q_loop(x, wg, blocks, sb):
        pre = moe_mxfp4(x, wg, blocks, sb, 2, 64)

        def body(c):
            i, tok = c
            return i + 1, moe_mxfp4(tok, wg, blocks, sb, 2, 64)

        return jax.lax.while_loop(lambda c: c[0] < 3, body, (0, pre[-1:]))[1]

    # gpt-oss in miniature: a QUANTIZED dispatch inside a decode loop, which is
    # the one shape that needs the packs threaded into a region as extra
    # captures AND the router verified on synthetic logits.
    blocks, sb = _mxfp4((4, 64, 64), 24, "float32", exp=(121, 125))
    out.append(("moe gather_qmm inside a decode loop", moe_q_loop,
                [_rand((3, 64), 25) * 0.1, _rand((64, 4), 23) * 0.5,
                 blocks, sb], 1e-4, 1e-4))

    # --- sdpa --------------------------------------------------------
    for dt, tol in (("float32", DOT), ("bfloat16", HALF)):
        q = _rand((2, 8, 4, 16), 30).astype(dt) * 0.5
        k = _rand((2, 8, 4, 16), 31).astype(dt) * 0.5
        v = _rand((2, 8, 4, 16), 32).astype(dt) * 0.5
        out.append((f"sdpa bqhd {dt}", lambda a, b, c: attn(a, b, c, 0.25),
                    [q, k, v], *tol))
    q = _rand((2, 4, 8, 16), 33) * 0.5
    k = _rand((2, 4, 8, 16), 34) * 0.5
    v = _rand((2, 4, 8, 16), 35) * 0.5
    out.append(("sdpa causal (boolean mask)",
                lambda a, b, c: attn_causal(a, b, c, 0.25), [q, k, v], *DOT))
    out.append(("sdpa additive mask",
                lambda a, b, c, m: attn_additive(a, b, c, m, 0.25),
                [q, k, v, _rand((2, 4, 8, 8), 36) * 0.1], *DOT))

    # P26: the same attention, rooted WHOLLY inside a `func.call` callee --
    # gemma-lib's sampler and maxtext both put a decode step there, and jax
    # gives that shape to any loop whose body calls a named function (a
    # `fori_loop` over `attn(...)` lowers to `func.call @attn` inside the while
    # body, non-inlined jit or not).  The recognizer walked @main and its
    # regions only, so a 60-layer decode step fused nothing and dispatched op
    # by op, per token.
    def attn_in_callee(q, k, v):
        step = jax.jit(lambda a: attn(a, k, v, 0.25), inline=False)
        return jax.lax.fori_loop(0, 3, lambda i, c: step(c), q)

    q = _rand((2, 8, 4, 16), 40) * 0.5
    k = _rand((2, 8, 4, 16), 41) * 0.5
    v = _rand((2, 8, 4, 16), 42) * 0.5
    out.append(("sdpa inside a callee", attn_in_callee, [q, k, v], *DOT))

    # ...and the hazard that scoping brings with it: ONE callee, two call
    # sites, two different masks.  Inlining lowers the callee's block twice and
    # binds its arguments to a different slot each time, so a mask cache keyed
    # by the IR value would hand the second attention the first one's mask --
    # the same answer everywhere except where the two masks differ.  Keyed by
    # the base's slot, the two are two entries.
    def attn_two_masks(q, k, v, m1, m2):
        one = jax.jit(lambda a, m: attn_additive(a, k, v, m, 0.25),
                      inline=False)
        return jax.lax.fori_loop(0, 2,
                                 lambda i, c: one(c, m1) + one(c, m2), q)

    q = _rand((2, 4, 8, 16), 43) * 0.5
    k = _rand((2, 4, 8, 16), 44) * 0.5
    v = _rand((2, 4, 8, 16), 45) * 0.5
    out.append(("sdpa two masks through one callee", attn_two_masks,
                [q, k, v, _rand((2, 4, 8, 8), 46) * 0.1,
                 _rand((2, 4, 8, 8), 47) * 0.1 - 3.0], *DOT))
    return out


def _sas_add(source, operand, select_prim, window_dimensions, window_strides,
             padding):
    """`stablehlo.select_and_scatter` without going through a pooling
    gradient.  jax has no public wrapper (only the primitive), and this is
    the entry point its own `lax_vmap_test` uses."""
    from jax._src.lax import windowed_reductions

    return windowed_reductions._select_and_scatter_add(
        source, operand, select_prim, window_dimensions, window_strides,
        padding)


def _bcoo_0d(data):
    """`jax.experimental.sparse` on a 0-d array, which is where a rank-0
    scatter comes from in the wild: the updates are reduced first, and what
    reaches the scatter is one value at an EMPTY coordinate vector."""
    import jax.numpy as jnp
    from jax.experimental import sparse

    return sparse.BCOO((data, jnp.zeros((3, 0), jnp.int32)),
                       shape=()).todense()


def _subbyte_cases():
    """One decode / encode / round-trip triple per emulated element type.

    The round trip walks every bit pattern the format has, so a NaN or
    subnormal encoding nobody thought about is covered by construction; the
    decode reads the same codes as f32, and the encode converts an f32 array
    (including the specials) onto the grid.  `_canonical` widens both sides to
    f64, so a NaN's PAYLOAD is not compared -- it survives neither engine's
    float16 storage, and the CPU backend is the only one that keeps it.
    """
    import jax.numpy as jnp

    grids = [
        ("f8E4M3FN", ml_dtypes.float8_e4m3fn, 8),
        ("f8E5M2", ml_dtypes.float8_e5m2, 8),
        ("f8E4M3", ml_dtypes.float8_e4m3, 8),
        ("f8E3M4", ml_dtypes.float8_e3m4, 8),
        ("f8E8M0FNU", ml_dtypes.float8_e8m0fnu, 8),
        ("f8E4M3B11FNUZ", ml_dtypes.float8_e4m3b11fnuz, 8),
        ("f8E5M2FNUZ", ml_dtypes.float8_e5m2fnuz, 8),
        ("f8E4M3FNUZ", ml_dtypes.float8_e4m3fnuz, 8),
        ("f6E2M3FN", ml_dtypes.float6_e2m3fn, 6),
        ("f6E3M2FN", ml_dtypes.float6_e3m2fn, 6),
        ("f4E2M1FN", ml_dtypes.float4_e2m1fn, 4),
        ("i4", ml_dtypes.int4, 4),
        ("ui4", ml_dtypes.uint4, 4),
    ]
    # The specials the encode has to place: a zero of each sign, values on and
    # off the grid, both overflows and a NaN.  Which of an infinity, a NaN or
    # a saturation each format produces is its own rule.
    src = np.array([0.0, -0.0, 1.0, -1.0, 0.5, 3.7, -2.25, 1e5, -1e5,
                    np.nan, np.inf, -np.inf, 1e-8], np.float32)
    # ...minus the OVERFLOWS, for the two families where this engine and
    # XLA:CPU knowingly disagree about them.  Neither is a defect of this
    # milestone; both are Stage 1's answers, unchanged:
    #   * FP6 -- XLA:CPU maps an overflow to a ZERO where ml_dtypes (the
    #     reference for these formats, CLAUDE.md item 20: "XLA:CPU's fp6 is
    #     itself broken") saturates to the largest finite value, which is
    #     what both metaljax engines produce;
    #   * i4/ui4 -- XLA saturates a float->4-bit convert where the emulation
    #     WRAPS (`((v + 8) mod 16) - 8`, dtypes.py `quantize_emulated`), so
    #     1e5 is 7 there and 0 here.
    # Everything in range, both signed zeros and the NaN are still compared.
    in_range = np.array([0.0, -0.0, 1.0, -1.0, 0.5, 3.7, -2.25, np.nan],
                        np.float32)
    # ui4 has no negatives either, and they are the same disagreement: XLA
    # clamps them to 0, the emulation's `v mod 16` wraps -1 to 15.
    unsigned = np.array([0.0, -0.0, 1.0, 0.5, 3.7, 12.0, np.nan], np.float32)
    out = []
    for name, dt, bits in grids:
        codes = np.arange(1 << bits, dtype=np.uint8).view(np.dtype(dt))
        out.append((f"{name}: every code round-trips through the device",
                    lambda a: a, [codes], 0, 0))
        # XLA:CPU cannot compile an FP6 convert at all (it crashes inside its
        # own fusion compiler), so those two have no decode reference; the
        # round trip above still covers them, since it needs no convert.
        if bits != 6:
            out.append((f"{name}: every code decodes",
                        lambda a: a.astype(jnp.float32), [codes], 0, 0))
        s = (unsigned if name == "ui4"
             else src if bits == 8 and name != "i4" else in_range)
        out.append((f"{name}: an f32 array encodes onto the grid",
                    (lambda a, dt=dt: a.astype(dt)), [s], 0, 0))
    return out


# --------------------------------------------------------------------------
# hand-written StableHLO
# --------------------------------------------------------------------------
#
# The same module text through both clients' `compile_and_load`, for encodings
# jax's own lowerings do not produce.  jax threads every value a while-cond
# closes over into the CARRY, so the counted encoding's third `bound_kind` --
# the bound as a CAPTURE of the cond region -- is unreachable from
# `lax.fori_loop`; getting it wrong would be a wrong trip count, which is the
# quietest way a loop can be wrong, so it is written out by hand here.

_WHILE_CAPTURED_BOUND = """
module @captured_bound {
  func.func public @main(%acc: tensor<f32>, %n: tensor<i32>) -> tensor<f32> {
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %one_i = stablehlo.constant dense<1> : tensor<i32>
    %one_f = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %r:2 = stablehlo.while(%i = %zero, %a = %acc) : tensor<i32>, tensor<f32>
     cond {
      %p = stablehlo.compare LT, %i, %n : (tensor<i32>, tensor<i32>)
          -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %i2 = stablehlo.add %i, %one_i : tensor<i32>
      %a2 = stablehlo.add %a, %one_f : tensor<f32>
      stablehlo.return %i2, %a2 : tensor<i32>, tensor<f32>
    }
    return %r#1 : tensor<f32>
  }
}
"""


# jax lowers `lax.cond` to stablehlo.CASE even for a two-way branch, so the
# IF encoding -- whose predicate is a bool read on the host, and whose FIRST
# region is the true branch -- has no other coverage.
_IF_BRANCHES = """
module @if_branches {
  func.func public @main(%p: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32> {
    %c = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %cb = stablehlo.broadcast_in_dim %c, dims = []
        : (tensor<f32>) -> tensor<4xf32>
    %r = "stablehlo.if"(%p) ({
      %t = stablehlo.multiply %x, %cb : tensor<4xf32>
      stablehlo.return %t : tensor<4xf32>
    }, {
      %f = stablehlo.subtract %x, %cb : tensor<4xf32>
      stablehlo.return %f : tensor<4xf32>
    }) : (tensor<i1>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
"""


# The 64-bit philox arm pairs two u32 words per element, and the state may
# arrive as four u32 words or two u64 ones -- the other axis the handler
# branches on.  jax refuses a u64 output without x64 and never emits the u64
# state form, so both are written out here.  The u64 state is BUILT inside the
# module from a u32 argument, because a u64 host buffer cannot cross
# `device_put` without x64 either.
_RNG_PHILOX_U64 = """
module @rng_philox_u64 {
  func.func public @main(%w: tensor<4xui32>)
      -> (tensor<4xui32>, tensor<5xui64>) {
    %p = stablehlo.reshape %w : (tensor<4xui32>) -> tensor<2x2xui32>
    %s = stablehlo.bitcast_convert %p
        : (tensor<2x2xui32>) -> tensor<2xui64>
    %state, %out = stablehlo.rng_bit_generator %s, algorithm = PHILOX
        : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui64>)
    %q = stablehlo.bitcast_convert %state
        : (tensor<2xui64>) -> tensor<2x2xui32>
    %r = stablehlo.reshape %q : (tensor<2x2xui32>) -> tensor<4xui32>
    return %r, %out : tensor<4xui32>, tensor<5xui64>
  }
}
"""

# (A u32 STATE with a u64 output has no case: XLA's own CPU backend fails to
# compile that combination -- "Binary op shift-right-logical with different
# element types: u32[] and u64[]" out of its rng expander -- so there is no
# reference to compare against.  The lowering covers it; nothing here can
# prove it does.)

# A reduce_window whose body is neither a monoid nor select_and_gather_add:
# the window axis is folded by the BODY itself, pairwise, which makes the body
# a sub-Program the entry carries.  jax's own lowerings only ever emit the
# monoid forms, so this encoding has no other coverage.
_REDUCE_WINDOW_GENERIC = """
module @reduce_window_generic {
  func.func public @main(%x: tensor<6xi32>) -> tensor<3xi32> {
    %init = stablehlo.constant dense<0> : tensor<i32>
    %r = "stablehlo.reduce_window"(%x, %init) <{
        window_dimensions = array<i64: 2>,
        window_strides = array<i64: 2>}> ({
      ^bb0(%a: tensor<i32>, %b: tensor<i32>):
        %o = stablehlo.or %a, %b : tensor<i32>
        stablehlo.return %o : tensor<i32>
    }) : (tensor<6xi32>, tensor<i32>) -> tensor<3xi32>
    return %r : tensor<3xi32>
  }
}
"""

# The same, variadic: two inputs folded by one body, which is the arity the
# generic path exists for.  (`stablehlo.reduce` with a general body takes the
# same route, so this covers both.)
_REDUCE_GENERIC_BODY = """
module @reduce_generic_body {
  func.func public @main(%x: tensor<2x4xi32>) -> tensor<2xi32> {
    %init = stablehlo.constant dense<-1> : tensor<i32>
    %r = stablehlo.reduce(%x init: %init) applies stablehlo.and
        across dimensions = [1] : (tensor<2x4xi32>, tensor<i32>)
        -> tensor<2xi32>
    return %r : tensor<2xi32>
  }
}
"""


# `window_reversal` flips the kernel, turning a correlation into a true
# convolution.  jax's `conv_general_dilated` has no parameter for it -- it
# flips the kernel itself when it wants one -- so the only way to reach the
# executor's `flip` (mx::conv_general's on the float path, an index reversal
# of the weights on the integer one) is to write the op out.
_CONV_REVERSAL = """
module @conv_reversal {
  func.func public @main(%x: tensor<1x2x6xf32>, %k: tensor<3x2x3xf32>)
      -> tensor<1x3x6xf32> {
    %r = stablehlo.convolution(%x, %k)
      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
      window = {stride = [1], pad = [[1, 1]], lhs_dilate = [1],
                rhs_dilate = [1], reverse = [1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x2x6xf32>, tensor<3x2x3xf32>) -> tensor<1x3x6xf32>
    return %r : tensor<1x3x6xf32>
  }
}
"""

_CONV_REVERSAL_INT = """
module @conv_reversal_int {
  func.func public @main(%x: tensor<1x2x6xi32>, %k: tensor<3x2x3xi32>)
      -> tensor<1x3x4xi32> {
    %r = stablehlo.convolution(%x, %k)
      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
      window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1],
                rhs_dilate = [1], reverse = [1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x2x6xi32>, tensor<3x2x3xi32>) -> tensor<1x3x4xi32>
    return %r : tensor<1x3x4xi32>
  }
}
"""

# ...and a MIXED reversal, which must decline: MLX's flip is all-or-nothing
# and so is the Python handler, so reversing one spatial axis and not the
# other has no spelling on either engine.
_CONV_MIXED_REVERSAL = """
module @conv_mixed_reversal {
  func.func public @main(%x: tensor<1x1x4x4xf32>, %k: tensor<1x1x2x2xf32>)
      -> tensor<1x1x3x3xf32> {
    %r = stablehlo.convolution(%x, %k)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
                lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [1, 0]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x3x3xf32>
    return %r : tensor<1x1x3x3xf32>
  }
}
"""


# Zero-size constants of the types jax's own lowerings never hand us empty:
# a bool one (whose elements are BIT-packed, so the decode reads them through
# the typed iterator) and an integer one.  Both are splats holding a single
# raw element under a shape with no elements at all.
_ZERO_SIZE_CONSTANTS = """
module @zero_size_constants {
  func.func public @main(%x: tensor<0xf32>)
      -> (tensor<0xf32>, tensor<0xi32>, tensor<0xi1>) {
    %cf = stablehlo.constant dense<1.500000e+00> : tensor<0xf32>
    %ci = stablehlo.constant dense<7> : tensor<0xi32>
    %cb = stablehlo.constant dense<true> : tensor<0xi1>
    %a = stablehlo.add %x, %cf : tensor<0xf32>
    return %a, %ci, %cb : tensor<0xf32>, tensor<0xi32>, tensor<0xi1>
  }
}
"""


# P12: the cross-replica collectives, on the one device this plugin has.  jax
# emits them only from pmap/shard_map, which cannot be nested in the jitted
# CASES above, so they are written out -- and the CPU backend, with the same
# one replica, is the reference for every one of them.  `all_reduce` carries
# its reduction as a REGION, which is what made it decline before the port:
# with a group of one there is nothing for that region to reduce.
_COLLECTIVES = """
module @collectives {
  func.func public @main(%x: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
          tensor<ui32>, tensor<ui32>) {
    %ar = "stablehlo.all_reduce"(%x) <{
        replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }) : (tensor<4xf32>) -> tensor<4xf32>
    %ag = "stablehlo.all_gather"(%x) <{all_gather_dim = 0 : i64,
        replica_groups = dense<0> : tensor<1x1xi64>}>
        : (tensor<4xf32>) -> tensor<4xf32>
    %cp = "stablehlo.collective_permute"(%x) <{
        source_target_pairs = dense<[[0, 0]]> : tensor<1x2xi64>}>
        : (tensor<4xf32>) -> tensor<4xf32>
    %rs = "stablehlo.reduce_scatter"(%x) <{scatter_dimension = 0 : i64,
        replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%c: tensor<f32>, %d: tensor<f32>):
      %t = stablehlo.add %c, %d : tensor<f32>
      stablehlo.return %t : tensor<f32>
    }) : (tensor<4xf32>) -> tensor<4xf32>
    %rid = stablehlo.replica_id : tensor<ui32>
    %pid = stablehlo.partition_id : tensor<ui32>
    return %ar, %ag, %cp, %rs, %rid, %pid
        : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
          tensor<ui32>, tensor<ui32>
  }
}
"""


# `collective_permute` with an EMPTY pair list: this replica receives nothing,
# which XLA fills with zeros -- the one arm of the family that is not an
# identity, and the one a single-device engine could get silently wrong.
_COLLECTIVE_PERMUTE_EMPTY = """
module @collective_permute_empty {
  func.func public @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %r = "stablehlo.collective_permute"(%x) <{
        source_target_pairs = dense<> : tensor<0x2xi64>}>
        : (tensor<4xf32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
"""


# The ASYNC wrapper (`jax.experimental.parallel`'s `psum_start(...).done()`
# family).  `async_start` hands its operands to a region holding one
# collective and yields a `!stablehlo.future`; `async_done` awaits it.  On one
# device there is no asynchrony to emulate -- the collective inside is one of
# the identities above -- so the pair lowers to the region's ops plus two
# aliases, and the CPU backend with its one replica is again the reference.
# The dot is here because jax puts one there: a real async collective needs
# something to overlap with, and a program that is only the pair would not say
# whether the ORDER survived.
_ASYNC_COLLECTIVES = """
module @async_collectives {
  func.func public @main(%x: tensor<4xf32>, %a: tensor<4x4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
    %d = stablehlo.dot_general %a, %a, contracting_dims = [1] x [0]
        : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %s = "stablehlo.async_start"(%x) ({
    ^bb0(%arg: tensor<4xf32>):
      %r = "stablehlo.all_reduce"(%arg) <{
          replica_groups = dense<0> : tensor<1x1xi64>}> ({
      ^bb0(%p: tensor<f32>, %q: tensor<f32>):
        %t = stablehlo.add %p, %q : tensor<f32>
        stablehlo.return %t : tensor<f32>
      }) : (tensor<4xf32>) -> tensor<4xf32>
      stablehlo.return %r : tensor<4xf32>
    }) : (tensor<4xf32>) -> !stablehlo.future<tensor<4xf32>>
    %done = "stablehlo.async_done"(%s)
        : (!stablehlo.future<tensor<4xf32>>) -> tensor<4xf32>
    %g = "stablehlo.async_start"(%x) ({
    ^bb0(%arg2: tensor<4xf32>):
      %h = "stablehlo.all_gather"(%arg2) <{all_gather_dim = 0 : i64,
          replica_groups = dense<0> : tensor<1x1xi64>}>
          : (tensor<4xf32>) -> tensor<4xf32>
      stablehlo.return %h : tensor<4xf32>
    }) : (tensor<4xf32>) -> !stablehlo.future<tensor<4xf32>>
    %gdone = "stablehlo.async_done"(%g)
        : (!stablehlo.future<tensor<4xf32>>) -> tensor<4xf32>
    return %done, %gdone, %d
        : tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>
  }
}
"""


def _module_cases():
    return [
        ("while with a captured bound", _WHILE_CAPTURED_BOUND,
         [np.float32(1.5), np.int32(6)]),
        ("rng philox 64-bit output (u64 state)", _RNG_PHILOX_U64,
         [np.array([1, 2, 3, 4], np.uint32)]),
        ("reduce_window with a general body", _REDUCE_WINDOW_GENERIC,
         [np.array([1, 2, 4, 8, 16, 32], np.int32)]),
        ("reduce with a general body", _REDUCE_GENERIC_BODY,
         [np.array([[7, 3, 5, 6], [15, 14, 12, 8]], np.int32)]),
        ("stablehlo.if (true)", _IF_BRANCHES,
         [np.bool_(True), np.arange(4, dtype=np.float32)]),
        ("stablehlo.if (false)", _IF_BRANCHES,
         [np.bool_(False), np.arange(4, dtype=np.float32)]),
        ("while with a captured bound (zero trip)", _WHILE_CAPTURED_BOUND,
         [np.float32(1.5), np.int32(0)]),
        ("while with a captured bound (negative)", _WHILE_CAPTURED_BOUND,
         [np.float32(1.5), np.int32(-3)]),
        ("convolution with window reversal", _CONV_REVERSAL,
         [_rand((1, 2, 6), 190), _rand((3, 2, 3), 191)]),
        ("integer convolution with reversal", _CONV_REVERSAL_INT,
         [_randint((1, 2, 6), 192), _randint((3, 2, 3), 193)]),
        ("zero-size constants", _ZERO_SIZE_CONSTANTS,
         [np.zeros((0,), np.float32)]),
        ("single-device collectives", _COLLECTIVES,
         [_rand((4,), 260)]),
        ("collective_permute with no pairs", _COLLECTIVE_PERMUTE_EMPTY,
         [_rand((4,), 261)]),
        ("async collectives (start/done)", _ASYNC_COLLECTIVES,
         [_rand((4,), 262), _rand((4, 4), 263)]),
    ]


def _run_module(text, args):
    """Compile and run one module on the default client of this process."""
    import jax
    from jax._src.lib import xla_client as xc

    dev = jax.devices()[0]
    exe = dev.client.compile_and_load(text, [dev], xc.CompileOptions())
    outs = exe.execute([jax.device_put(a, dev) for a in args])
    return [np.asarray(o) for o in outs]


# The two sort comparators jax's own lowerings cannot produce, both of which
# must decline: one where the sides compute DIFFERENT functions of their
# arguments (so there is no key array at all), and one whose chain leaves
# scalar elementwise code (so running it on the whole operand would compute
# something else entirely).
_SORT_ASYMMETRIC = """
module @sort_asymmetric {
  func.func public @main(%x: tensor<3xf32>) -> tensor<3xf32> {
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %r = "stablehlo.sort"(%x) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %s = stablehlo.add %a, %one : tensor<f32>
      %p = stablehlo.compare LT, %s, %b, FLOAT
          : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    }) : (tensor<3xf32>) -> tensor<3xf32>
    return %r : tensor<3xf32>
  }
}
"""

_SORT_NONSCALAR = """
module @sort_nonscalar {
  func.func public @main(%x: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %r = "stablehlo.sort"(%x) <{dimension = 1 : i64, is_stable = true}> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %ab = stablehlo.broadcast_in_dim %a, dims = []
          : (tensor<f32>) -> tensor<2xf32>
      %bb = stablehlo.broadcast_in_dim %b, dims = []
          : (tensor<f32>) -> tensor<2xf32>
      %as = stablehlo.reduce(%ab init: %a) applies stablehlo.maximum
          across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
      %bs = stablehlo.reduce(%bb init: %b) applies stablehlo.maximum
          across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
      %p = stablehlo.compare LT, %as, %bs, FLOAT
          : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    }) : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %r : tensor<1x3xf32>
  }
}
"""


# Two comparator TREES outside the vocabulary P10's recognizer reads (jax
# emits neither: its multi-key comparator is `or(lt k0, and(eq k0, lt k1))`
# and every direction in it is LT or EQ).  The first is that comparator with
# the decisions turned around -- a descending lexicographic sort, which the
# ascending execution shape would answer wrongly and silently.
_SORT_TREE_DESCENDING = """
module @sort_tree_descending {
  func.func public @main(%a: tensor<3xi32>, %b: tensor<3xf32>)
      -> (tensor<3xi32>, tensor<3xf32>) {
    %r:2 = "stablehlo.sort"(%a, %b) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%a0: tensor<i32>, %a1: tensor<i32>, %b0: tensor<f32>,
         %b1: tensor<f32>):
      %gt = stablehlo.compare GT, %a0, %a1, SIGNED
          : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %eq = stablehlo.compare EQ, %a0, %a1, SIGNED
          : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %lt = stablehlo.compare LT, %b0, %b1, FLOAT
          : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %and = stablehlo.and %eq, %lt : tensor<i1>
      %or = stablehlo.or %gt, %and : tensor<i1>
      stablehlo.return %or : tensor<i1>
    }) : (tensor<3xi32>, tensor<3xf32>) -> (tensor<3xi32>, tensor<3xf32>)
    return %r#0, %r#1 : tensor<3xi32>, tensor<3xf32>
  }
}
"""

# ...and the second computes with a key rather than deciding on the operands,
# so the sort it means is not "by operand 0, then operand 1" at all.
_SORT_TREE_ARITHMETIC = """
module @sort_tree_arithmetic {
  func.func public @main(%a: tensor<3xi32>, %b: tensor<3xf32>)
      -> (tensor<3xi32>, tensor<3xf32>) {
    %one = stablehlo.constant dense<1> : tensor<i32>
    %r:2 = "stablehlo.sort"(%a, %b) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%a0: tensor<i32>, %a1: tensor<i32>, %b0: tensor<f32>,
         %b1: tensor<f32>):
      %s = stablehlo.add %a0, %one : tensor<i32>
      %lt = stablehlo.compare LT, %s, %a1, SIGNED
          : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %eq = stablehlo.compare EQ, %a0, %a1, SIGNED
          : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %flt = stablehlo.compare LT, %b0, %b1, FLOAT
          : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %and = stablehlo.and %eq, %flt : tensor<i1>
      %or = stablehlo.or %lt, %and : tensor<i1>
      stablehlo.return %or : tensor<i1>
    }) : (tensor<3xi32>, tensor<3xf32>) -> (tensor<3xi32>, tensor<3xf32>)
    return %r#0, %r#1 : tensor<3xi32>, tensor<3xf32>
  }
}
"""


_UNKNOWN_CUSTOM_CALL = """
module @unknown_custom_call {
  func.func public @main(%x: tensor<3xf32>) -> tensor<3xf32> {
    %r = stablehlo.custom_call @no_such_op(%x) {backend_config = ""}
        : (tensor<3xf32>) -> tensor<3xf32>
    return %r : tensor<3xf32>
  }
}
"""


# Programs that must DECLINE, with the op the message has to name.  A decline
# is a feature here: the plugin refuses whole programs it cannot lower, and it
# says which op stopped it.
def _declines():
    import jax
    import jax.numpy as jnp

    return [
        # P10 gave the two select-tree comparators their execution shapes, and
        # what stays declined is a tree that is not one of them.  The
        # lexicographic reading is INFERRED from which operand pairs the tree
        # decides on -- it is never evaluated -- so its vocabulary is checked:
        # a GT anywhere means the tree orders the other way somewhere, and
        # running it ascending would be silently wrong rather than loud.
        ("a descending lexicographic comparator", _SORT_TREE_DESCENDING,
         [np.array([3, 1, 2], np.int32), np.array([1.0, 2.0, 3.0], np.float32)],
         "sort: comparator tree compares GT"),
        # ...and an arithmetic op inside the tree is not a decision at all:
        # the keys it would sort by are not the operands.
        ("a comparator tree holding arithmetic", _SORT_TREE_ARITHMETIC,
         [np.array([3, 1, 2], np.int32), np.array([1.0, 2.0, 3.0], np.float32)],
         "sort: comparator tree holds stablehlo.add"),
        # P7 gave convolution an executor, and two of its corners stay
        # declined.  A MIXED window reversal is one: MLX's flip is
        # all-or-nothing and so is the Python handler, so one axis reversed
        # and the other not has no spelling on either engine.
        ("convolution with a mixed window reversal", _CONV_MIXED_REVERSAL,
         [np.zeros((1, 1, 4, 4), np.float32),
          np.zeros((1, 1, 2, 2), np.float32)],
         "conv: mixed window_reversal"),
        # An ASYMMETRIC comparator is not a sort by a key: the two sides
        # compute different functions of their arguments, so no single key
        # array orders the operand the way the comparator does.  It must
        # decline, and this is the case where getting it wrong would be
        # SILENT -- the sort would run, on the left side's key.
        ("sort with an asymmetric comparator", _SORT_ASYMMETRIC,
         [np.array([3.0, 1.0, 2.0], np.float32)],
         "sort: asymmetric comparator"),
        # ...and one whose key chain holds an op that is not elementwise
        # scalar code: running THAT on the whole operand is what the rank-0
        # check exists to refuse.
        ("sort with a non-scalar key chain", _SORT_NONSCALAR,
         [np.array([[3.0, 1.0, 2.0]], np.float32)],
         "sort: comparator op"),
        # mx::pad has no negative widths, and the Python handler raises on one
        # too -- so a cropping window declines on both engines rather than
        # being rewritten into a slice the reference never computed.
        ("reduce_window with negative padding",
         lambda x: jax.lax.reduce_window(
             x, 0.0, jax.lax.add, (2,), (1,), [(-1, 0)]),
         [np.arange(6, dtype=np.float32)],
         "reduce_window negative padding"),
        # A complex scatter multiply without a uniqueness promise runs one
        # update at a time, which is exact -- but only up to the cap the
        # sequential arm carries.  Above it the op declines by name rather
        # than emitting thousands of MLX ops per call.
        ("complex scatter multiply above the sequential cap",
         lambda x, i, u: x.at[i].multiply(u),
         [np.ones(4096, np.complex64),
          np.arange(2048, dtype=np.int32),
          np.full(2048, 2 + 0j, np.complex64)],
         "complex scatter multiply with duplicates: 2048 updates"),
        # A loop whose BODY holds an op outside the set declines the whole
        # program, naming that op -- the region is lowered by the same
        # `Lowering` as main, so its declines are main's.  (Convolution used
        # to be the op here, then the LAPACK targets, then reduce_precision;
        # all three compute now, so `stablehlo.rng` -- XLA's
        # non-deterministic RNG, which NEITHER engine implements -- stands
        # in.)
        ("while loop over an unlowered op",
         lambda x: jax.lax.fori_loop(
             0, 4, lambda i, c: c + jax.lax.rng_uniform(
                 np.float32(0.0), np.float32(1.0), (4,)), x),
         [np.arange(4, dtype=np.float32)], "stablehlo.rng"),
        # A scatter whose computed body has no uniqueness promise runs one
        # update at a time, which is only affordable for the small shapes the
        # pattern shows up in: past ops/gather.py's own cap the program
        # declines rather than emitting thousands of entries' worth of work.
        ("scatter with a computed body over too many updates",
         lambda x, i: x.at[i].apply(jnp.sin),
         [np.arange(4096, dtype=np.float32),
          np.arange(2048, dtype=np.int32)],
         "scatter computed-body with duplicates"),
        # A bitcast whose end is an emulated FLOAT grid: those hold values in
        # a wider dtype, so the bits the op wants to read do not exist on the
        # device.  i4/ui4 are the exception (whole nibbles) and lower.
        ("bitcast_convert on an emulated float grid",
         lambda x: jax.lax.bitcast_convert_type(x, jnp.uint8),
         [np.array([1.0, 2.0], ml_dtypes.float8_e4m3fn)],
         "bitcast_convert on f8E4M3FN"),
        # A custom call whose target has no handler declines by NAME, so a
        # program that reaches an unknown external is a missing feature and
        # never a wrong answer.  (Written by hand: jax emits no such call.)
        ("an unknown custom call", _UNKNOWN_CUSTOM_CALL,
         [np.arange(3, dtype=np.float32)], "custom call target 'no_such_op'"),
    ]


# --------------------------------------------------------------------------
# the P13 surface: callbacks, ordered effects, donation, buffer identity
# --------------------------------------------------------------------------
#
# None of these is a NUMBER the CPU backend could be asked for -- they are
# contracts about what the runtime does around the numbers -- so they sit here
# rather than in the differential cases: what a callback printed and in which
# order, whether a donated buffer is gone, whether two buffers are the same
# memory.  Each returns (ok, detail).


# --------------------------------------------------------------------------
# P19: the row-blocked packer and the cross-executable build cache
# --------------------------------------------------------------------------
#
# Both are memory disciplines over an answer that must not move, so all four
# probes below are about SAMENESS and the plugin's own account of what it did.
# They run in subprocesses because what they vary -- METALJAX_QMM_BLOCK,
# METALJAX_QMM_BUILD_CACHE -- the dylib reads once, at load.
#
# The graphs are the two real reconstruction shapes: an MXFP4 weight whose
# rows CAN be blocked (its canonical `[N, K]` layout needs no transpose, which
# is the whole precondition -- gpt-oss-20b's projections are this shape), and
# a keras sub-channel int4 weight whose `[K, N]` layout cannot be, so it packs
# whole on both stacks and is here to prove the fallback is silent and exact.

_P19_GRAPHS = r'''
import os, sys
import numpy as np
import jax, jax.numpy as jnp

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
_T = np.ldexp(np.ones(256), np.arange(256) - 127); _T[255] = np.nan
SCALE_TABLE = _T.astype(np.float32)


def mxfp4_weight(blocks, sb, k, dtype):
    vt = jnp.asarray(E2M1, dtype=dtype)
    st = jnp.asarray(SCALE_TABLE)
    lead = tuple(blocks.shape[:-1])
    lo = jnp.bitwise_and(blocks, jnp.uint8(0x0F))
    hi = jnp.right_shift(blocks, jnp.uint8(4))
    nib = jnp.reshape(jnp.stack([lo, hi], axis=-1), lead + (k,))
    vals = jnp.take(vt, nib.astype(jnp.int32), axis=0)
    scale = jnp.take(st, sb.astype(jnp.int32), axis=0)
    w = jnp.reshape(vals, lead + (k // 32, 32)) * scale[..., None].astype(dtype)
    return jnp.reshape(w, lead + (k,))


def mxfp4(shape_nk, seed):
    rng = np.random.RandomState(seed)
    n, k = shape_nk[-2], shape_nk[-1]
    lead = tuple(shape_nk[:-2])
    codes = rng.randint(0, 16, size=lead + (n, k)).astype(np.uint8)
    blocks = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    sb = rng.randint(118, 133, size=lead + (n, k // 32)).astype(np.uint8)
    return blocks, sb


def int4(rows, cols, block, seed):
    rng = np.random.RandomState(seed)
    q = rng.randint(-8, 8, size=(rows, cols)).astype(np.int8)
    packed = ((q[:, 0::2] & 0x0F) | (q[:, 1::2] << 4)).astype(np.int8)
    ng = rows // block
    scale = (rng.rand(ng, cols).astype(np.float32) + 0.5) * 0.05
    zero = rng.randint(-3, 4, size=(ng, cols)).astype(np.int8)
    g_idx = (np.arange(rows) // block).astype(np.float32)
    return packed, scale, zero, g_idx


def dense_sub(packed, scale, zero, g_idx, x, columns):
    lo = jnp.bitwise_and(packed, jnp.int8(0x0F))
    lo = jnp.where(lo > 7, lo - 16, lo)
    hi = jnp.right_shift(packed, jnp.int8(4))
    w = jnp.reshape(jnp.stack([lo, hi], axis=-1),
                    packed.shape[:-1] + (columns,))
    g = g_idx.astype(jnp.int32)
    return x @ ((w.astype(x.dtype) - jnp.take(zero, g, axis=0).astype(x.dtype))
                * jnp.take(scale, g, axis=0))
'''

# Every quantized shape the plugin can pack, answered once.  The caller runs
# this twice with different block sizes and compares the BYTES.
_P19_ANSWERS = _P19_GRAPHS + r'''
outs = []
b, s = mxfp4((256, 256), 11)
x = (np.random.RandomState(12).rand(6, 256).astype(np.float32) - 0.5) * 0.8
outs.append(np.asarray(jax.jit(lambda b, s, a: jnp.einsum(
    "th,nh->tn", a, mxfp4_weight(b, s, 256, a.dtype)))(b, s, x)))
b, s = mxfp4((256, 256), 11)
xb = x.astype(jnp.bfloat16)
outs.append(np.asarray(jax.jit(lambda b, s, a: jnp.einsum(
    "th,nh->tn", a, mxfp4_weight(b, s, 256, a.dtype)))(b, s, xb)
    ).astype(np.float32))
b, s = mxfp4((4, 64, 128), 13)
xe = np.random.RandomState(14).rand(4, 3, 128).astype(np.float32) * 0.4
outs.append(np.asarray(jax.jit(lambda b, s, a: jnp.einsum(
    "etm,ehm->eth", a, mxfp4_weight(b, s, 128, a.dtype)))(b, s, xe)))
p, sc, z, g = int4(256, 128, 128, 5)
xi = np.random.RandomState(6).rand(4, 256).astype(np.float32) - 0.5
outs.append(np.asarray(jax.jit(
    lambda p, s, z, g, a: dense_sub(p, s, z, g, a, 128))(p, sc, z, g, xi)))
# ...and one inside a decode loop, where the weight arrives as a loop-carried
# block argument: the blocked walk and the fingerprint both have to follow it
# out to the value the loop was handed (qmm.py `_hoist`).
b, s = mxfp4((128, 128), 31)
xl = np.random.RandomState(32).rand(2, 128).astype(np.float32) * 0.2


def loop(b, s, x):
    def body(c):
        i, y = c
        return i + 1, jnp.tanh(jnp.einsum(
            "th,nh->tn", y, mxfp4_weight(b, s, 128, y.dtype)) * 0.3)
    return jax.lax.while_loop(lambda c: c[0] < 3, body, (0, x))[1]


outs.append(np.asarray(jax.jit(loop)(b, s, xl)))
np.save(sys.argv[1], np.concatenate([o.ravel().view(np.uint8) for o in outs]))
'''

# Two executables over ONE weight set, then a third over another: what
# keras-hub's per-sequence-length sampler does, and the reason the build cache
# exists.
_P19_CACHE = _P19_GRAPHS + r'''
b, s = mxfp4((128, 128), 3)
b, s = jax.device_put(b), jax.device_put(s)
f = jax.jit(lambda b, s, a: jnp.einsum(
    "th,nh->tn", a, mxfp4_weight(b, s, 128, a.dtype)))
for t in (4, 7):
    x = jax.device_put(np.random.RandomState(t).rand(t, 128).astype(np.float32))
    print("[probe] T=%d %.6f" % (t, float(np.asarray(f(b, s, x)).sum())))
b2, s2 = mxfp4((128, 128), 4)
x = jax.device_put(np.random.RandomState(9).rand(5, 128).astype(np.float32))
print("[probe] other %.6f" % float(np.asarray(f(b2, s2, x)).sum()))
'''

# A weight big enough that the whole reconstruction is worth measuring: 8192 x
# 4096 MXFP4 values are 33.5 M elements, and jax's `take` wrapper alone carries
# three int32 copies of the index tensor.  What the arms are compared on is the
# plugin's own "pack wave peak", i.e. `mx::get_peak_memory()` inside the dylib
# -- host RSS does not see a Metal allocation at all, and `mlx.core` in THIS
# process is a different runtime whose counters read zero for the plugin.
_P19_PEAK = _P19_GRAPHS + r'''
b, s = mxfp4((8192, 4096), 21)
x = np.random.RandomState(22).rand(2, 4096).astype(np.float32) * 0.1
out = np.asarray(jax.jit(lambda b, s, a: jnp.einsum(
    "th,nh->tn", a, mxfp4_weight(b, s, 4096, a.dtype)))(b, s, x))
print("[probe] checksum %.6f" % float(out.sum()))
'''


def _p19_packing(subprocess, tempfile, pathlib):
    """(label, check) pairs for the row-blocked packer and the build cache."""

    def run(source, extra_env, *args):
        env = dict(os.environ)
        env["METALJAX_DEBUG"] = "1"
        env.update(extra_env)
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(source)
            script = fh.name
        try:
            return subprocess.run([sys.executable, script, *args], env=env,
                                  capture_output=True, text=True)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass

    def packlog(proc):
        return [ln.split("qmm: ", 1)[1]
                for ln in (proc.stderr or "").splitlines()
                if "-native] qmm: " in ln]

    def blocked_answers_are_the_whole_ones():
        """The pack is the same pack however many pieces it was built from.

        Bit equality, not a tolerance: blocking changes WHEN the verification
        sees a value and nothing about what is derived from it, so a single
        differing bit would mean a block was read from the wrong rows.
        """
        with tempfile.TemporaryDirectory() as tmp:
            whole = str(pathlib.Path(tmp) / "whole.npy")
            small = str(pathlib.Path(tmp) / "small.npy")
            a = run(_P19_ANSWERS, {"METALJAX_QMM_BLOCK": str(1 << 30)}, whole)
            if a.returncode:
                return False, (a.stderr or a.stdout).strip()[-120:]
            b = run(_P19_ANSWERS, {"METALJAX_QMM_BLOCK": "4096"}, small)
            if b.returncode:
                return False, (b.stderr or b.stdout).strip()[-120:]
            if not np.array_equal(np.load(whole), np.load(small)):
                return False, "a blocked pack computes different bytes"
            # ...and the small-block arm really did block: three of the four
            # weights are `[N, K]`-shaped and must report several blocks, the
            # keras one cannot be blocked at all and must say so.
            got = [ln for ln in packlog(b) if ln.startswith("packed ")]
            many = [ln for ln in got if " row blocks" in ln]
            if len(got) != 5 or len(many) != 4:
                return False, f"{len(many)} of {len(got)} packs blocked"
            if not any(ln.endswith(" whole") for ln in got):
                return False, "the un-blockable weight did not pack whole"
        return True, ""

    def a_weight_packs_once_for_two_executables():
        proc = run(_P19_CACHE, {})
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-120:]
        log = packlog(proc)
        built = [ln for ln in log if ln.startswith("packed ")]
        reused = [ln for ln in log if ln.startswith("reused ")]
        # Two executables over one weight set: one build, one reuse.  A third
        # over DIFFERENT buffers must build again -- the cache is keyed on the
        # buffers the reconstruction reads, not on the graph alone.
        if len(built) != 2 or len(reused) != 1:
            return False, f"{len(built)} built / {len(reused)} reused"
        return True, ""

    def the_cache_can_be_turned_off():
        proc = run(_P19_CACHE, {"METALJAX_QMM_BUILD_CACHE": "0"})
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-120:]
        log = packlog(proc)
        if len([ln for ln in log if ln.startswith("reused ")]):
            return False, "a pack was reused with the cache off"
        if len([ln for ln in log if ln.startswith("packed ")]) != 3:
            return False, "not every executable rebuilt"
        return True, ""

    def a_blocked_pack_bounds_its_peak():
        """The point of the whole exercise, in the units a watchdog reads.

        The same 8192x4096 MXFP4 weight packed whole and packed in blocks;
        what must fall is the process's PEAK resident size, since MLX returns
        a dead intermediate to its own cache and a watchdog counts it as
        claimed either way (which is why the packer runs with that cache off).
        """
        whole = run(_P19_PEAK, {"METALJAX_QMM_BLOCK": str(1 << 30)})
        if whole.returncode:
            return False, (whole.stderr or whole.stdout).strip()[-120:]
        small = run(_P19_PEAK, {"METALJAX_QMM_BLOCK": str(1 << 22)})
        if small.returncode:
            return False, (small.stderr or small.stdout).strip()[-120:]

        def read(proc):
            peak = None
            for ln in (proc.stderr or "").splitlines():
                if "qmm: pack wave peak " in ln:
                    peak = float(ln.split("pack wave peak ")[1].split()[0])
            checksum = None
            for ln in proc.stdout.splitlines():
                if ln.startswith("[probe] checksum "):
                    checksum = float(ln.split()[2])
            return peak, checksum

        wg, wsum = read(whole)
        sg, ssum = read(small)
        if wg is None or sg is None:
            return False, "no pack-wave peak reported"
        if wsum != ssum:
            return False, f"the two arms disagree ({wsum} vs {ssum})"
        # The reconstruction is several times the packed weight and a block is
        # a fraction of it, so the saving is most of a gigabyte; three quarters
        # of the whole arm's peak is a bar a noisy allocator cannot cross by
        # accident.
        if not sg < wg * 0.75:
            return False, f"peak {sg:.2f} GB blocked vs {wg:.2f} GB whole"
        return True, f"ok ({sg:.2f} GB blocked vs {wg:.2f} GB whole)"

    return [
        ("blocked pack == whole pack", blocked_answers_are_the_whole_ones),
        ("one pack for two executables", a_weight_packs_once_for_two_executables),
        ("the build cache has an off switch", the_cache_can_be_turned_off),
        ("a blocked pack bounds its peak", a_blocked_pack_bounds_its_peak),
    ]


# --------------------------------------------------------------------------
# P25: the eager flush trims MLX's pool instead of dumping it
# --------------------------------------------------------------------------
#
# A hard flush that finds MLX's pool over `METALJAX_FLUSH_CLEAR_MB` used to
# DUMP it (`mx::clear_cache()`); it TRIMS it back to the watermark now
# (`runtime.cc::trim_cache`).  What has to be shown is that the swap kept the
# BOUND -- the clear was never decoration, it is what stops an eager program
# whose traffic dwarfs its live set from claiming the traffic -- so the arms
# below run one such program and read the pool out of the DYLIB's own meter
# (`METALJAX_MEMDBG`, runtime/program.cc's flush line), which is the only
# reading taken at a flush point at all.
#
# The program is eager by construction (`METALJAX_COMPILE=0`, the same arm
# P3/P4 use) with a 64 MB flush budget so the sync points really fall inside
# it, and every intermediate it produces is a DIFFERENT SIZE -- which is what
# makes a pool grow at all.  MLX reuses a cached buffer only for a request
# within `min(2 * size, size + 2 * page)` of it (mlx/backend/common/
# buffer_cache.h `reuse_from_cache`), i.e. essentially an exact match at these
# widths, so 64 distinct 16-80 MB results per call accumulate ~3 GB of freed
# buffers -- the synthetic shape of what a real over-budget main (maxtext's
# training step: ~105 GB of traffic, hundreds of distinct shapes, a few
# hundred MB live) does to the cache.  A constant-shape chain would prove
# nothing: every free would be reused exactly and no pool would ever grow.

_P25_TRAFFIC = r'''
import os, sys
import numpy as np
import jax, jax.numpy as jnp

BASE = 4 * 1024 * 1024       # 16 MB of f32
STEP = 256 * 1024            # ...growing by 1 MB per op
ROUNDS, DEPTH = 3, 64


def chain(x):
    acc = jnp.float32(0)
    for i in range(DEPTH):
        y = x[:BASE + i * STEP] * jnp.float32(1.0000001) + jnp.float32(0.5)
        acc = acc + jnp.sum(y)   # a scalar carry: the live set stays flat
    return acc


n = BASE + DEPTH * STEP
x = jax.device_put(np.random.RandomState(7).rand(n).astype(np.float32))
f = jax.jit(chain)
total = 0.0
for _ in range(ROUNDS):
    total += float(np.asarray(f(x)))
print("[probe] checksum %.6f" % total)
print("[probe] traffic_gb %.2f" % (
    ROUNDS * 2 * sum(BASE + i * STEP for i in range(DEPTH)) * 4 / (1 << 30)))
'''

# The loop discipline, on the row that exists to exercise it (the differential
# suite's "long counted loop"): a tiny body, run tens of thousands of times.
# What keeps Metal's live-buffer COUNT bounded there is the op-unit loop-clear
# cadence (`METALJAX_LOOP_CLEAR_COST`), which is NOT what the pool bound
# replaces -- a byte limit says nothing about a count -- so this arm is here to
# prove that swapping the flush clear left it alone.  The loop is forced onto
# the interpreted path (no compiled chunks, no generated kernel), because that
# is the arm the cadence exists for; the shipped paths run the same loop in the
# differential suite above.
_P25_LONGLOOP = r'''
import os
import numpy as np
import jax

n = int(os.environ.get("MJ_P25_ITERS", "20000"))
out = float(np.asarray(jax.jit(
    lambda x: jax.lax.fori_loop(0, n, lambda i, c: c + 1.0, x))(
        np.float32(0.0))))
print("[probe] loop %.1f of %d" % (out, n))
'''


def _p25_cache_limit(subprocess, tempfile, pathlib, re):
    """(label, check) pairs for the pool bound that replaced the flush clear."""

    def run(source, extra_env):
        env = dict(os.environ)
        env["METALJAX_DEBUG"] = "1"
        env["METALJAX_MEMDBG"] = "1"
        env.update(extra_env)
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(source)
            script = fh.name
        try:
            return subprocess.run([sys.executable, script], env=env,
                                  capture_output=True, text=True)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass

    # "... active=123MB cache=45MB (was 678MB) bound=256MB" -- the `(was ...)`
    # half is there only when that flush trimmed.
    _FLUSH = re.compile(r"\[metaljax-mem\] flush #\d+: active=(\d+)MB "
                        r"cache=(\d+)MB(?: \(was (\d+)MB\))? bound=(-?\d+)MB")

    def traffic(bound_mb):
        """One eager traffic run; returns (proc, cache samples MB, checksum).

        The meter lines come out of `debug_line`, which writes to STDOUT (the
        per-execute `[metaljax-native]` stats line is the one on stderr), so
        both streams are searched rather than the wrong one guessed at.
        """
        proc = run(_P25_TRAFFIC, {"METALJAX_COMPILE": "0",
                                  "METALJAX_EAGER_FLUSH_MB": "64",
                                  "METALJAX_FLUSH_CLEAR_MB": str(bound_mb)})
        caches = [int(m.group(2)) for m in
                  _FLUSH.finditer((proc.stdout or "") + (proc.stderr or ""))]
        checksum = None
        for ln in proc.stdout.splitlines():
            if ln.startswith("[probe] checksum "):
                checksum = ln.split()[2]
        return proc, caches, checksum

    state = {}

    def the_pool_stays_under_its_bound():
        """~18 GB of eager traffic, and the cache never passes 256 MB.

        The trim happens at the NEXT allocation, so a reading can sit one
        allocation over the line -- 80 MB at the widest here, and 128 MB of
        slack covers it.  What the bound is really being separated from is
        the traffic: an unbounded pool reads gigabytes (the arm below).
        """
        proc, caches, checksum = traffic(256)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        state["bounded"] = (caches, checksum)
        if len(caches) < 20:
            return False, (f"only {len(caches)} hard flushes narrated; last "
                           f"output: {(proc.stdout or '').strip()[-90:]!r}")
        if max(caches) > 256 + 128:
            return False, f"peak cache {max(caches)} MB over a 256 MB bound"
        return True, (f"ok (peak {max(caches)} MB cached over "
                      f"{len(caches)} flushes)")

    def the_bound_is_not_a_dump():
        """...and the buffers UNDER the bound survive to be reused.

        This is the whole difference between a limit and a clear, and the
        2.2x on the maxtext training row: after a dump every allocation is a
        cold Metal buffer.  A pool that is being trimmed rather than emptied
        sits NEAR its bound, so the median flush must find real memory
        cached.
        """
        caches = state.get("bounded", ([], None))[0]
        if not caches:
            return False, "the bounded arm did not run"
        mid = sorted(caches)[len(caches) // 2]
        if mid <= 0:
            return False, "the pool is empty at half the flushes (a dump)"
        return True, f"ok (median {mid} MB cached, bound 256 MB)"

    def the_bound_is_what_bounds_it():
        """The control: with no limit, the same program's pool runs away.

        `METALJAX_FLUSH_CLEAR_MB=-1` leaves MLX's default cache limit (its
        memory limit) alone.  If the bounded arm above were bounded by
        something else -- the flush itself, the allocator's own pressure
        rules -- this arm would read the same numbers.
        """
        proc, caches, checksum = traffic(-1)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        if not caches:
            return False, "no flushes narrated"
        bounded, bsum = state.get("bounded", ([], None))
        if checksum != bsum:
            return False, f"the arms disagree ({checksum} vs {bsum})"
        if max(caches) <= 2 * max(bounded or [0]):
            return False, (f"unbounded peak {max(caches)} MB vs bounded "
                           f"{max(bounded or [0])} MB")
        return True, (f"ok (unbounded peak {max(caches)} MB vs bounded "
                      f"{max(bounded or [0])} MB)")

    def a_long_loop_still_clears_on_the_count_cadence():
        """20k interpreted iterations: the op-unit cadence, bound or no bound.

        Read out of the plugin's own stats line -- the loop clears must fire
        and no execute may have needed a buffer-limit recovery, which is the
        live-buffer COUNT staying bounded stated in the only terms a caller
        can see.  `METALJAX_LOOP_CLEAR_COST` is turned down to 1000 op units
        so the cadence is crossed in seconds rather than in the half-million
        units the shipped default spends (this loop's body is worth ~2).
        """
        proc = run(_P25_LONGLOOP, {"METALJAX_FLUSH_CLEAR_MB": "256",
                                   "METALJAX_COMPILE": "0",
                                   "METALJAX_MSL": "0",
                                   "METALJAX_LOOP_CLEAR_COST": "1000",
                                   "MJ_P25_ITERS": "20000"})
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        if "[probe] loop 20000.0 of 20000" not in proc.stdout:
            return False, f"wrong answer: {proc.stdout.strip()[-80:]}"
        stats = [ln for ln in (proc.stderr or "").splitlines()
                 if "[metaljax-native] " in ln and "loop_flushes=" in ln]
        if not stats:
            return False, "no stats line"
        clears = retries = 0
        for ln in stats:
            m = re.search(r"loop_flushes=(\d+)\(\+clear (\d+)\).*?"
                          r"limit_retries=(\d+)", ln)
            if m:
                clears += int(m.group(2))
                retries += int(m.group(3))
        if clears < 1:
            return False, "the loop-clear cadence never fired"
        if retries:
            return False, f"{retries} buffer-limit recoveries"
        return True, f"ok ({clears} loop clears, 0 recoveries)"

    return [
        ("the pool stays under its bound", the_pool_stays_under_its_bound),
        ("the bound is a trim, not a dump", the_bound_is_not_a_dump),
        ("the bound is what bounds it", the_bound_is_what_bounds_it),
        ("a long loop clears on its own cadence",
         a_long_loop_still_clears_on_the_count_cadence),
    ]


# --------------------------------------------------------------------------
# P27: the watermark is not one number
# --------------------------------------------------------------------------
#
# P25's watermark had to be one value for every program, and the sweep found
# no value that worked: the maxtext training row needs a ~26 GB pool to reach
# its anchor, and handing that to every program guard-kills the LoRA row's
# load.  `runtime.cc::flush_bound` now decides per flush, from two things --
# whether the program has flushed enough times to BE an eager main
# (METALJAX_FLUSH_MAIN_FLUSHES), and whether the process footprint has room
# for the pool (METALJAX_FLUSH_FOOTPRINT_MB) -- with P25's shipped watermark
# as the floor under both.
#
# The arms below run P25's traffic program (one program, ~550 hard flushes,
# every intermediate a different size: an eager main by construction) and read
# the dylib's own meter, which now prints the `bound=` it chose, the `foot=`
# it chose it from and the program's own flush count `n=`.  Each arm turns
# exactly one of the three rules off, so a failure names which one broke.
def _p27_flush_pressure(subprocess, tempfile, pathlib, re):
    """(label, check) pairs for the footprint-aware flush bound."""

    _METER = re.compile(
        r"\[metaljax-mem\] flush #\d+: active=(\d+)MB cache=(\d+)MB"
        r"(?: \(was (\d+)MB\))? bound=(-?\d+)MB foot=(-?\d+)MB "
        r"cap=(-?\d+)MB n=(\d+)")

    def traffic(**extra):
        """One eager-traffic run; returns (proc, meter rows, checksum)."""
        env = dict(os.environ)
        env["METALJAX_DEBUG"] = "1"
        env["METALJAX_MEMDBG"] = "1"
        env["METALJAX_COMPILE"] = "0"
        env["METALJAX_EAGER_FLUSH_MB"] = "64"
        # P28's benefit gate OFF for these four arms, deliberately.  They test
        # P27's two rules, one disabled at a time, and this program is exactly
        # the shape P28 denies -- "a scalar carry: the live set stays flat", so
        # its swing is zero and rule 3 alone would hold every bound at the
        # floor, hiding whichever of rules 1 and 2 an arm is about to break.
        # `_p28_benefit_gate` below turns it back on and owns its own arms.
        env["METALJAX_FLUSH_EARN_MULT"] = "0"
        env.update({k: str(v) for k, v in extra.items()})
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(_P25_TRAFFIC)
            script = fh.name
        try:
            proc = subprocess.run([sys.executable, script], env=env,
                                  capture_output=True, text=True)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass
        rows = [tuple(int(g) if g else 0 for g in m.groups())
                for m in _METER.finditer((proc.stdout or "") +
                                         (proc.stderr or ""))]
        checksum = None
        for ln in (proc.stdout or "").splitlines():
            if ln.startswith("[probe] checksum "):
                checksum = ln.split()[2]
        return proc, rows, checksum

    # cap 4096 / floor 256: far enough apart that "which rule chose this
    # bound" is legible in the meter, and both under any machine's footprint.
    CAP, FLOOR, GATE = 4096, 256, 8
    state = {}

    def an_eager_main_earns_the_pool():
        """A program that keeps flushing is allowed past the floor.

        With the footprint target out of the way (a target no machine can
        reach), the only rule left is the main gate: the first `GATE` hard
        flushes of the program's life are bounded at the FLOOR, everything
        after at the CAP, and the pool really does grow past the floor -- the
        1.10x P25 measured on the maxtext row is exactly this pool surviving.
        """
        proc, rows, checksum = traffic(
            METALJAX_FLUSH_CLEAR_MB=CAP, METALJAX_FLUSH_FLOOR_MB=FLOOR,
            METALJAX_FLUSH_MAIN_FLUSHES=GATE,
            METALJAX_FLUSH_FOOTPRINT_MB=1 << 22)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        state["main"] = (rows, checksum)
        if len(rows) < 20:
            return False, f"only {len(rows)} hard flushes narrated"
        # `n` counts this flush, and the gate opens ON the `GATE`-th one.
        early = [r for r in rows if r[6] < GATE]
        late = [r for r in rows if r[6] >= GATE]
        if not early or not late:
            return False, f"{len(early)} early / {len(late)} late flushes"
        if any(r[3] != FLOOR for r in early):
            return False, (f"a flush inside the gate was bounded at "
                           f"{max(r[3] for r in early)} MB, not the floor")
        if any(r[3] != CAP for r in late):
            return False, (f"a flush past the gate was bounded at "
                           f"{min(r[3] for r in late)} MB, not the cap")
        peak = max(r[1] for r in rows)
        if peak <= FLOOR + 128:
            return False, f"the pool never grew past the floor ({peak} MB)"
        if peak > CAP + 128:
            return False, f"peak cache {peak} MB over a {CAP} MB cap"
        return True, (f"ok (bound {FLOOR}->{CAP} MB at flush {GATE}, peak "
                      f"{peak} MB cached)")

    def the_gate_is_what_grants_it():
        """The control: a program that never becomes a main never gets it.

        Same run, same answers, with the gate set past any flush count this
        program can reach -- so every bound is the floor and the pool stays
        there.  This is the LOAD phase's arm: thousands of small programs,
        one or two flushes each, none of which may leave a 16 GB pool
        standing where a live-set spike is about to land (P27's row 18).
        """
        proc, rows, checksum = traffic(
            METALJAX_FLUSH_CLEAR_MB=CAP, METALJAX_FLUSH_FLOOR_MB=FLOOR,
            METALJAX_FLUSH_MAIN_FLUSHES=1 << 30,
            METALJAX_FLUSH_FOOTPRINT_MB=1 << 22)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        if not rows:
            return False, "no flushes narrated"
        want = state.get("main", ([], None))[1]
        if checksum != want:
            return False, f"the arms disagree ({checksum} vs {want})"
        if any(r[3] != FLOOR for r in rows):
            return False, (f"bound reached {max(r[3] for r in rows)} MB with "
                           f"the gate closed")
        peak = max(r[1] for r in rows)
        if peak > FLOOR + 128:
            return False, f"peak cache {peak} MB over the {FLOOR} MB floor"
        return True, f"ok (every bound {FLOOR} MB, peak {peak} MB cached)"

    def the_footprint_target_takes_it_back():
        """...and so does the footprint, for a main that has spent it.

        A target of one megabyte is the arithmetic limit of "this process has
        no room": the room term goes negative at every flush, and the bound
        collapses to the floor for a program the gate has already let
        through.  That is the rule that keeps a 65 GB checkpoint stream, or a
        model whose live set is already the whole target, from being handed a
        32 GB pool because it happened to flush a lot.
        """
        proc, rows, checksum = traffic(
            METALJAX_FLUSH_CLEAR_MB=CAP, METALJAX_FLUSH_FLOOR_MB=FLOOR,
            METALJAX_FLUSH_MAIN_FLUSHES=GATE,
            METALJAX_FLUSH_FOOTPRINT_MB=1)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        if not rows:
            return False, "no flushes narrated"
        want = state.get("main", ([], None))[1]
        if checksum != want:
            return False, f"the arms disagree ({checksum} vs {want})"
        late = [r for r in rows if r[6] >= GATE]
        if not late:
            return False, "the gate was never crossed"
        if any(r[3] != FLOOR for r in late):
            return False, (f"bound reached {max(r[3] for r in late)} MB with "
                           f"no footprint to spare")
        peak = max(r[1] for r in rows)
        if peak > FLOOR + 128:
            return False, f"peak cache {peak} MB over the {FLOOR} MB floor"
        return True, f"ok (every bound {FLOOR} MB, peak {peak} MB cached)"

    def the_bound_is_the_target_minus_the_live_set():
        """The formula itself, on an arm where all three terms bind.

        A target a little above the program's own live set puts the bound
        strictly between the floor and the cap, where it must track
        `target - (foot - cache)` flush by flush -- which is the claim that
        the process footprint (not MLX's accounting, and not a constant) is
        what the pool is being charged against.  Slack: `foot` and `cache`
        are read after the trim this line describes, the bound before it, and
        the two differ by whatever was freed in between.
        """
        # The live set this program actually has, read off the arm that was
        # allowed to keep everything, plus a gigabyte of room -- so the arm
        # lands between the clamps on any machine rather than at a constant
        # that happens to work on this one.
        main_rows = state.get("main", ([], None))[0]
        if not main_rows:
            return False, "the main arm did not run"
        lives = sorted(r[4] - r[1] for r in main_rows)
        target = lives[len(lives) // 2] + 1024
        proc, rows, checksum = traffic(
            METALJAX_FLUSH_CLEAR_MB=CAP, METALJAX_FLUSH_FLOOR_MB=FLOOR,
            METALJAX_FLUSH_MAIN_FLUSHES=GATE,
            METALJAX_FLUSH_FOOTPRINT_MB=target)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        want = state.get("main", ([], None))[1]
        if checksum != want:
            return False, f"the arms disagree ({checksum} vs {want})"
        late = [r for r in rows if r[6] >= GATE and not r[2]]
        if len(late) < 10:
            return False, f"only {len(late)} untrimmed flushes past the gate"
        worst, worst_row = 0, None
        for active, cache, _was, bound, foot, _cap, _n in late:
            want_bound = min(CAP, max(FLOOR, target - (foot - cache)))
            if abs(want_bound - bound) > worst:
                worst, worst_row = abs(want_bound - bound), (bound, want_bound,
                                                             foot, cache)
        if worst > 128:
            return False, (f"bound {worst_row[0]} MB where the footprint says "
                           f"{worst_row[1]} MB (foot {worst_row[2]}, cache "
                           f"{worst_row[3]})")
        # The arm proves nothing unless the footprint term is what is
        # actually choosing the bound on a fair share of the flushes: a run
        # that sat on a clamp throughout would satisfy the identity above
        # trivially.  (Individual flushes DO clamp -- this program's live set
        # swings by more than the gigabyte of room the target leaves.)
        inside = [r for r in late if FLOOR < r[3] < CAP]
        if len(inside) < len(late) // 4:
            span = (min(r[3] for r in late), max(r[3] for r in late))
            return False, (f"only {len(inside)} of {len(late)} bounds left "
                           f"their clamps ({span[0]}-{span[1]} MB)")
        span = (min(r[3] for r in inside), max(r[3] for r in inside))
        return True, (f"ok (bound {span[0]}-{span[1]} MB on {len(inside)} of "
                      f"{len(late)} flushes, worst {worst} MB off)")

    return [
        ("an eager main earns the pool", an_eager_main_earns_the_pool),
        ("the gate is what grants it", the_gate_is_what_grants_it),
        ("the footprint target takes it back",
         the_footprint_target_takes_it_back),
        ("the bound is the target minus live",
         the_bound_is_the_target_minus_the_live_set),
    ]


# --------------------------------------------------------------------------
# P28: the benefit gate -- the pool has to be EARNED
# --------------------------------------------------------------------------
#
# P27's two rules ask whether a program MAY have a big pool.  Neither asks
# whether it is doing anything with one, and the two maxtext DECODE rows are
# where that shows: their checkpoint load is a single program taking 134 hard
# flushes in one call, so it is an "eager main" by rule 1 and has footprint to
# spare by rule 2 -- and the 14 GB of weights it frees at its LAST flush then
# stands in the pool for the rest of the process.  17 GB and 11 GB of extra
# peak footprint, no speed, and a guard kill at budgets those rows had never
# come near (notes/release-gates-0.11.5.md gate 5).
#
# `runtime.cc::flush_bound` now asks a third question, and the quantity it
# asks it about is the program's own LIVE set: a trim can only ever cost a
# program the memory it has to re-acquire AFTER the trim, and across a flush
# point that is bounded by how far its live set falls and rises.  So a program
# may keep METALJAX_FLUSH_EARN_MULT times the live-set SWING it has
# demonstrated, and no more.
#
# The two arms below are the two shapes, run through the SAME harness so the
# difference is the program and nothing else: P25's traffic program, whose
# live set is flat by construction ("a scalar carry" -- the decode rows'
# shape), and a program that genuinely cycles a large tensor in and out.
_P28_SWING = r'''
import numpy as np
import jax, jax.numpy as jnp

BASE = 4 * 1024 * 1024        # 16 MB of f32
STEP = 256 * 1024
DEPTH, ROUNDS = 24, 3
WIDE = 8                      # phase A holds WIDE copies live


def churn(x):
    """Traffic in two phases: one with a large tensor LIVE across the flush
    points, one without it.  The live set therefore falls and rises between
    flushes, which is exactly what a buffer pool is for -- and exactly what
    the flat-live-set program has none of."""
    acc = jnp.float32(0)
    big = jnp.concatenate([x] * WIDE)          # phase A: ~8x live
    for i in range(DEPTH):
        y = big[:BASE + i * STEP] * jnp.float32(1.0000001) + jnp.float32(0.5)
        acc = acc + jnp.sum(y)
    acc = acc + jnp.sum(big) * jnp.float32(0.0)   # last use of `big`
    for i in range(DEPTH):                     # phase B: `big` is dead
        y = x[:BASE + i * STEP] * jnp.float32(1.0000001) + jnp.float32(0.5)
        acc = acc + jnp.sum(y)
    return acc


n = BASE + DEPTH * STEP
x = jax.device_put(np.random.RandomState(11).rand(n).astype(np.float32))
f = jax.jit(churn)
total = 0.0
for _ in range(ROUNDS):
    total += float(np.asarray(f(x)))
print("[probe] checksum %.6f" % total)
'''


def _p28_benefit_gate(subprocess, tempfile, pathlib, re):
    """(label, check) pairs for the earned-pool rule over P27's two."""

    _METER = re.compile(
        r"\[metaljax-mem\] flush #\d+: active=(\d+)MB cache=(\d+)MB"
        r"(?: \(was (\d+)MB\))? bound=(-?\d+)MB foot=(-?\d+)MB "
        r"cap=(-?\d+)MB n=(\d+) live=(-?\d+)MB earn=(-?\d+)MB")

    # Same clamps as P27's arms, and a footprint target no machine can reach,
    # so rules 1 and 2 are out of the way and rule 3 is the only thing left
    # that can hold a bound down.
    CAP, FLOOR, GATE, MULT = 4096, 256, 8, 2
    TARGET = 1 << 22

    def run(source, **extra):
        env = dict(os.environ)
        env["METALJAX_DEBUG"] = "1"
        env["METALJAX_MEMDBG"] = "1"
        env["METALJAX_COMPILE"] = "0"
        env["METALJAX_EAGER_FLUSH_MB"] = "64"
        env["METALJAX_FLUSH_CLEAR_MB"] = str(CAP)
        env["METALJAX_FLUSH_FLOOR_MB"] = str(FLOOR)
        env["METALJAX_FLUSH_MAIN_FLUSHES"] = str(GATE)
        env["METALJAX_FLUSH_FOOTPRINT_MB"] = str(TARGET)
        env["METALJAX_FLUSH_EARN_MULT"] = str(MULT)
        env.update({k: str(v) for k, v in extra.items()})
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(source)
            script = fh.name
        try:
            proc = subprocess.run([sys.executable, script], env=env,
                                  capture_output=True, text=True)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass
        rows = [tuple(int(g) if g else 0 for g in m.groups())
                for m in _METER.finditer((proc.stdout or "") +
                                         (proc.stderr or ""))]
        checksum = None
        for ln in (proc.stdout or "").splitlines():
            if ln.startswith("[probe] checksum "):
                checksum = ln.split()[2]
        return proc, rows, checksum

    def by_program(rows):
        """Split meter rows into the PROGRAMS that produced them.

        `n=` is the program's own hard-flush count, so it increases by one
        within a program and drops when a different program flushes -- and the
        water marks the rule keeps are per program, so anything that replays
        the rule has to segment the same way.
        """
        progs, cur, prev = [], [], None
        for r in rows:
            if prev is not None and r[6] <= prev:
                progs.append(cur)
                cur = []
            cur.append(r)
            prev = r[6]
        if cur:
            progs.append(cur)
        return progs

    state = {}

    def a_flat_live_set_earns_nothing():
        """The rows-11/14 shape: past the gate, room to spare, no pool.

        P25's traffic program carries a scalar between its chain steps, so its
        live set barely moves at its flush points -- it never hands back a
        pool's worth of memory that it then has to take out again, and a trim
        therefore costs it almost nothing.  Rules 1 and 2 both wave it through
        (it takes hundreds of hard flushes and its live set is a rounding
        error against the target); rule 3 is the only reason its bound stays
        down near the floor, and that is exactly why the two maxtext decode
        rows stop carrying 17 GB and 11 GB of pool nothing reads.

        The claim is not "the bound is the floor" -- a program whose live set
        swings a little has earned a little -- but that what it earns tracks
        its own swing instead of the cap.
        """
        proc, rows, checksum = run(_P25_TRAFFIC)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        if len(rows) < 20:
            return False, f"only {len(rows)} hard flushes narrated"
        late = [r for r in rows if r[6] >= GATE]
        if not late:
            return False, "the gate was never crossed"
        peak = max(r[1] for r in rows)
        state["flat"] = (checksum, peak)
        loose = CAP // 4
        if max(r[3] for r in late) >= loose:
            return False, (f"bound reached {max(r[3] for r in late)} MB for a "
                           f"program that cycles nothing")
        if peak >= loose:
            return False, f"peak cache {peak} MB for a program that cycles nothing"
        swing = max(max(p[7] for p in prog) - min(p[7] for p in prog)
                    for prog in by_program(rows))
        return True, (f"ok ({len(late)} flushes past the gate, bound <= "
                      f"{max(r[3] for r in late)} MB against a {CAP} MB cap, "
                      f"worst live-set swing {swing} MB, peak {peak} MB cached)")

    def the_earn_rule_is_what_denies_it():
        """The control that names the rule: the SAME program, rule off.

        With `METALJAX_FLUSH_EARN_MULT=0` the bound is P27's again and the
        flat-live-set program is handed the whole cap -- the behaviour the two
        decode rows were guard-killed by, reproduced in a contract so that a
        change which quietly re-enables it fails here rather than on a 25 GB
        model row.  Same answers either way, and the pool it keeps is the
        measurement: several times what the same program keeps with the rule
        on, for a program that does nothing with either.
        """
        proc, rows, checksum = run(_P25_TRAFFIC, METALJAX_FLUSH_EARN_MULT=0)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        if not rows:
            return False, "no flushes narrated"
        want, on_peak = state.get("flat", (None, 0))
        if checksum != want:
            return False, f"the arms disagree ({checksum} vs {want})"
        late = [r for r in rows if r[6] >= GATE]
        if not late or any(r[3] != CAP for r in late):
            return False, (f"bound {max((r[3] for r in late), default=0)} MB "
                           f"with the earn rule off; expected the {CAP} MB cap")
        # `live=` is sampled either way -- it is one counter read and it keeps
        # the meter uniform -- so the rule being OFF shows up as `earn=-1`.
        if any(r[8] != -1 for r in rows):
            return False, "the meter still reported an earn with the rule off"
        peak = max(r[1] for r in rows)
        if peak < 4 * max(on_peak, 1):
            return False, (f"the rule saved nothing: peak {peak} MB with it "
                           f"off against {on_peak} MB with it on")
        return True, (f"ok (every bound {CAP} MB with the rule off, peak "
                      f"{peak} MB cached against {on_peak} MB with it on)")

    def a_cycling_live_set_earns_the_pool():
        """...and the program the pool exists for still gets one.

        This one holds a large tensor live across one stretch of flushes and
        drops it across the next, so its live set genuinely falls and rises --
        the maxtext training row's shape, whose 1.85x is the whole reason the
        watermark rises above the floor at all.  Its bound must leave the
        floor, and the pool must actually grow past it.
        """
        proc, rows, checksum = run(_P28_SWING)
        if proc.returncode:
            return False, (proc.stderr or proc.stdout).strip()[-140:]
        if len(rows) < 20:
            return False, f"only {len(rows)} hard flushes narrated"
        state["swing"] = (rows, checksum)
        late = [r for r in rows if r[6] >= GATE]
        if not late:
            return False, "the gate was never crossed"
        earned = [r for r in late if r[3] > FLOOR]
        if len(earned) < len(late) // 2:
            return False, (f"only {len(earned)} of {len(late)} bounds past "
                           f"the gate left the floor")
        swing = max(max(p[7] for p in prog) - min(p[7] for p in prog)
                    for prog in by_program(rows))
        if swing < 128:
            return False, (f"the probe's live set only swung {swing} MB -- it "
                           f"is not exercising the rule")
        peak = max(r[1] for r in rows)
        if peak <= FLOOR + 64:
            return False, f"the pool never grew past the floor ({peak} MB)"
        return True, (f"ok (live-set swing {swing} MB, bound up to "
                      f"{max(r[3] for r in late)} MB on {len(earned)} of "
                      f"{len(late)} flushes, peak {peak} MB cached)")

    def the_bound_is_the_multiplier_times_the_swing():
        """The formula itself, in two halves the meter reports separately.

        `earn=` is the rule's own product and `bound=` what the flush trimmed
        to, so the first half is EXACT: past the gate, and with the other two
        rules out of reach, the bound must be `earn` clamped to [floor, cap]
        and nothing else.

        The second half is that `earn` really is the live-set swing: it must
        equal `mult * (hi - lo)` over that PROGRAM's own `live=` readings --
        the values the rule sampled, which is why the meter prints them
        separately from `active=` (the two differ by whatever the step dropped
        between the sample and the print; on this probe's phase-change flush
        they read 399 MB and 95 MB).  Per program, because that is where the
        water marks live: a replay pooling every program's readings would be
        checking a rule nothing implements.
        """
        rows = state.get("swing", ([], None))[0]
        if not rows:
            return False, "the swinging arm did not run"
        worst, worst_row, checked = 0, None, 0
        drift, drift_row = 0, None
        for prog in by_program(rows):
            hi = lo = None
            for _active, _cache, _was, bound, _foot, _cap, n, live, earn in prog:
                if earn == -1 or live == -1:
                    return False, "the meter reported no earn with the rule on"
                hi = live if hi is None else max(hi, live)
                lo = live if lo is None else min(lo, live)
                if abs(earn - MULT * (hi - lo)) > drift:
                    drift = abs(earn - MULT * (hi - lo))
                    drift_row = (earn, MULT * (hi - lo), hi, lo)
                if n < GATE:
                    continue
                checked += 1
                want = min(CAP, max(FLOOR, earn))
                if abs(want - bound) > worst:
                    worst, worst_row = abs(want - bound), (bound, want, earn)
        if checked < 10:
            return False, f"only {checked} flushes past the gate"
        if worst > 0:
            return False, (f"bound {worst_row[0]} MB where earn={worst_row[2]} "
                           f"MB clamps to {worst_row[1]} MB")
        # Byte state, megabyte narration: each water mark can lose a megabyte
        # to the shift, so the product can be two out and no more.
        if drift > 2 * MULT:
            return False, (f"earn={drift_row[0]} MB where the sampled live set "
                           f"says {drift_row[1]} MB (hi {drift_row[2]}, lo "
                           f"{drift_row[3]})")
        # The identity is vacuous unless the rule is what actually chose the
        # bound on a fair share of the flushes rather than a clamp.
        inside = [r for r in rows if r[6] >= GATE and FLOOR < r[3] < CAP]
        if len(inside) < checked // 4:
            return False, (f"only {len(inside)} of {checked} bounds left "
                           f"their clamps")
        span = (min(r[3] for r in inside), max(r[3] for r in inside))
        return True, (f"ok (bound {span[0]}-{span[1]} MB on {len(inside)} of "
                      f"{checked} flushes, worst {worst} MB off)")

    return [
        ("a flat live set earns nothing", a_flat_live_set_earns_nothing),
        ("the earn rule is what denies it", the_earn_rule_is_what_denies_it),
        ("a cycling live set earns the pool", a_cycling_live_set_earns_the_pool),
        ("the bound is the multiplier times the swing",
         the_bound_is_the_multiplier_times_the_swing),
    ]


# The no-panic contract's own program: one transfer, one execute, one print,
# so that whichever of the two the governor is being asked about is the only
# thing that can fail.  Every arm below runs THIS, with one variable moved.
_GOV_PROGRAM = r'''
import numpy as np, jax, jax.numpy as jnp
try:
    x = jax.device_put(np.arange(1 << 20, dtype=np.float32))
    print("[probe] transferred")
    print("[probe] checksum %.6f" % float(np.asarray(jax.jit(
        lambda a: jnp.sum(a * 2.0))(x))))
except BaseException as exc:                                    # noqa: BLE001
    print("[probe] raised %s: %s" % (type(exc).__name__,
                                     " ".join(str(exc).split())[:400]))
'''


# ...and the same for a program that GROWS: 512 MB a step, kept, with no
# transfer after the first one -- the shape of a materialization phase, which
# is the failure mode a transfer gate cannot see.
_GOV_GROWTH = r'''
import numpy as np, jax, jax.numpy as jnp

x = jax.device_put(np.zeros(1 << 27, np.float32))          # 512 MB
# Every scalar the loop needs, transferred BEFORE it starts: a `device_put`
# inside the loop would be refused by the transfer gate first, and this arm is
# about the other one.
step = [jax.device_put(np.float32(i)) for i in range(32)]
f = jax.jit(lambda a, i: a + i)
held = []
for i in range(32):
    try:
        y = f(x, step[i])
        held.append(y)   # this plugin executes synchronously: y is real
    except BaseException as exc:                            # noqa: BLE001
        print("[probe] refused at %d: %s"
              % (i, " ".join(str(exc).split())[:300]))
        break
else:
    print("[probe] never refused")
print("[probe] alive")
'''


def _governor(subprocess, tempfile, pathlib, re):
    """(label, check) pairs for the memory governor (the no-panic contract).

    The governor's job is to make a machine wedge impossible, and the thing it
    must never do to earn that is refuse work the machine can do.  So each arm
    below moves ONE of its numbers to a value the running process is already
    on the wrong side of -- which is how a threshold whose real trigger takes
    a 65 GB checkpoint gets a contract that runs in a second.
    """

    _METER = re.compile(
        r"\[metaljax-mem\] flush #\d+: active=(\d+)MB cache=(\d+)MB"
        r"(?: \(was (\d+)MB\))? bound=(-?\d+)MB foot=(-?\d+)MB "
        r"cap=(-?\d+)MB n=(\d+)")

    def run(source, **extra):
        env = dict(os.environ)
        env["METALJAX_MEM_STALL_MS"] = "0"    # arms assert, they do not wait
        env["METALJAX_MEM_SAMPLE_US"] = "0"   # ...and never on a stale sample
        env.update({k: str(v) for k, v in extra.items()})
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(source)
            script = fh.name
        try:
            proc = subprocess.run([sys.executable, script], env=env,
                                  capture_output=True, text=True)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass
        return proc, (proc.stdout or "") + (proc.stderr or "")

    def a_transfer_past_the_hard_line_is_refused():
        """The load path's answer to a model that cannot fit.

        A budget of one megabyte is the arithmetic limit of "this process is
        already over": the transfer is refused BEFORE its staging block is
        allocated, and what the caller gets is a status -- RESOURCE_EXHAUSTED,
        the code XLA's own backends use -- rather than a machine that spends
        the next minute in a reclaim storm.  The message has to name the
        variable that moves the line, because the alternative is a user
        guessing.
        """
        proc, out = run(_GOV_PROGRAM, METALJAX_MEM_BUDGET_MB=1)
        if proc.returncode:
            return False, out.strip()[-140:]
        if "[probe] transferred" in out:
            return False, "the transfer was admitted"
        if "RESOURCE_EXHAUSTED" not in out and "RESOURCE EXHAUSTED" not in out:
            return False, f"not a resource-exhausted error: {out.strip()[-140:]}"
        if "METALJAX_MEM_BUDGET_MB" not in out:
            return False, "the error does not name the variable"
        if "metaljax out of memory at transfer" not in out:
            return False, "the error does not name the transfer path"
        return True, "ok (RESOURCE_EXHAUSTED, names the budget)"

    def the_machine_ceiling_refuses_too():
        """...and the same for the machine's own memory, not just ours.

        Both wedges happened at ~55 GB of process footprint, well inside any
        budget this process would set for itself: what was full was the
        MACHINE.  `METALJAX_MEM_SYS_MB` is that second line, read from
        `host_statistics64` rather than from `task_info`, and this arm is what
        proves the two are separately wired.
        """
        proc, out = run(_GOV_PROGRAM, METALJAX_MEM_SYS_MB=1)
        if proc.returncode:
            return False, out.strip()[-140:]
        if "[probe] transferred" in out:
            return False, "the transfer was admitted"
        if "METALJAX_MEM_SYS_MB" not in out:
            return False, f"wrong reason: {out.strip()[-140:]}"
        return True, "ok (RESOURCE_EXHAUSTED, names the ceiling)"

    def an_execute_that_grows_is_stopped():
        """A program is not entered when the process is already over.

        The transfer path is where a LOAD is stopped; this is the other half,
        for the row that grows inside its own executes (row 15's post-restore
        materialization is the measured case: +4-7 GB per guard sample with no
        transfer in sight).  512 MB a step against a 4 GB budget, and what
        this asserts is not only the refusal but what the process does after
        it: it is still running, and it says so.
        """
        proc, out = run(_GOV_GROWTH, METALJAX_MEM_BUDGET_MB=4096)
        if proc.returncode:
            return False, out.strip()[-140:]
        if "[probe] never refused" in out:
            return False, "16 GB of live buffers were admitted under a 4 GB "\
                          "budget"
        if "RESOURCE_EXHAUSTED" not in out:
            return False, f"nothing was refused: {out.strip()[-200:]}"
        if "[probe] alive" not in out:
            return False, "the process did not survive its own refusal"
        where = ("execute" if "out of memory at execute" in out else
                 "flush" if "out of memory at flush" in out else
                 "transfer" if "out of memory at transfer" in out else "?")
        if where not in ("execute", "flush"):
            return False, f"refused at {where}, not inside the program"
        step = [ln for ln in out.splitlines() if "[probe] refused at" in ln]
        return True, f"ok (refused at {where}, {step[0].split()[3][:-1]} steps in)"

    def the_governor_can_be_turned_off():
        """The control, and the escape hatch.

        Same impossible budget, `METALJAX_MEM_GOVERNOR=0`: the program runs to
        completion.  A user who would rather take the risk than the refusal
        has one variable to set, and this arm is what says the refusals above
        are the governor's doing and not something else breaking.
        """
        proc, out = run(_GOV_PROGRAM, METALJAX_MEM_BUDGET_MB=1,
                        METALJAX_MEM_SYS_MB=1, METALJAX_MEM_GOVERNOR=0)
        if proc.returncode:
            return False, out.strip()[-140:]
        if "[probe] checksum" not in out:
            return False, f"the program did not run: {out.strip()[-140:]}"
        return True, "ok (same program completes with the governor off)"

    def pressure_takes_the_pool_back():
        """The DEGRADE path, which is the one the contract prefers.

        P27 lets an eager main keep a big buffer pool because that is worth
        1.9x on the maxtext training row.  Under machine pressure it may not:
        `flush_bound` asks the governor first, and every bound collapses to
        the floor -- the same program, the same answers, a pool that is not
        standing beside somebody else's page cache.  Forced here by a free
        floor no machine can be above, which is the honest way to test a
        threshold whose real trigger is a full machine.

        P28's benefit gate is off for BOTH arms, deliberately: this program's
        live set is flat by construction, so rule 3 alone would hold both of
        them at the floor and the contract would pass while proving nothing
        about the governor.  The unpressured arm reaching the cap is this
        test's precondition, not its claim.
        """
        CAP, FLOOR, GATE = 4096, 256, 8
        env = dict(METALJAX_DEBUG=1, METALJAX_MEMDBG=1, METALJAX_COMPILE=0,
                   METALJAX_EAGER_FLUSH_MB=64, METALJAX_FLUSH_CLEAR_MB=CAP,
                   METALJAX_FLUSH_FLOOR_MB=FLOOR,
                   METALJAX_FLUSH_MAIN_FLUSHES=GATE,
                   METALJAX_FLUSH_EARN_MULT=0,
                   METALJAX_FLUSH_FOOTPRINT_MB=1 << 22)
        proc, out = run(_P25_TRAFFIC, **env)
        if proc.returncode:
            return False, out.strip()[-140:]
        free = [tuple(int(g) if g else 0 for g in m.groups())
                for m in _METER.finditer(out)]
        base = [ln for ln in out.splitlines() if ln.startswith("[probe] checksum")]
        # ...and the same run with the floor above the machine's memory, so
        # the governor is pressured at every flush.
        proc2, out2 = run(_P25_TRAFFIC, METALJAX_MEM_FREE_FLOOR_MB=1 << 22,
                          **env)
        if proc2.returncode:
            return False, out2.strip()[-140:]
        held = [tuple(int(g) if g else 0 for g in m.groups())
                for m in _METER.finditer(out2)]
        got = [ln for ln in out2.splitlines()
               if ln.startswith("[probe] checksum")]
        if not free or not held:
            return False, f"{len(free)} / {len(held)} flushes narrated"
        if base != got:
            return False, f"the arms disagree ({base} vs {got})"
        late = [r for r in free if r[6] >= GATE]
        if not late or any(r[3] != CAP for r in late):
            return False, "the unpressured arm never reached the cap"
        if any(r[3] != FLOOR for r in held):
            return False, (f"a pressured flush was bounded at "
                           f"{max(r[3] for r in held)} MB, not the floor")
        peak_free = max(r[1] for r in free)
        peak_held = max(r[1] for r in held)
        if peak_held > FLOOR + 128:
            return False, f"pool reached {peak_held} MB under pressure"
        return True, (f"ok (pool {peak_free} -> {peak_held} MB cached, every "
                      f"bound {FLOOR} MB)")

    return [
        ("a transfer past the hard line is refused",
         a_transfer_past_the_hard_line_is_refused),
        ("the machine ceiling refuses too", the_machine_ceiling_refuses_too),
        ("the governor can be turned off", the_governor_can_be_turned_off),
        ("an execute that grows is stopped",
         an_execute_that_grows_is_stopped),
        ("pressure takes the pool back", pressure_takes_the_pool_back),
    ]


# P26: an attention rooted inside a `func.call` callee, in a DYNAMICALLY
# bounded loop -- the shape gemma-lib's sampler and maxtext both emit, and the
# one jax gives any loop whose body calls a named function.  Two layers inside
# the callee, so the recognizer's count means something.
_P26_CALLEE = r'''
import numpy as np, jax, jax.numpy as jnp

def attn(q, k, v):
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * 0.25
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), v)

def run(q, k1, v1, k2, v2, n):
    # A while body with a PYTHON body inlines; the callee comes from the jit
    # inside it, which is how the gemma sampler's decode step gets its own
    # `@closed_call` (`moe.py::analyze`'s docstring names the same asymmetry).
    block = jax.jit(lambda c: attn(attn(c, k1, v1), k2, v2), inline=False)
    return jax.lax.while_loop(
        lambda s: s[0] < n, lambda s: (s[0] + 1, block(s[1])), (0, q))[1]

rng = np.random.RandomState(26)
a = [rng.rand(2, 8, 4, 16).astype(np.float32) * 0.5 for _ in range(5)]
print(f"[probe] checksum {float(np.asarray(jax.jit(run)(*a, 3)).sum()):.9e}")
'''


def _p26_callee_sdpa(subprocess, tempfile, pathlib, re):
    """(label, check) pairs for the callee-scoped attention recognizer.

    Two things need proving that no answer can show.  That an attention living
    wholly inside a callee is FOUND -- an unfused one computes the same thing,
    only slower, so the numeric rows pass either way.  And that finding it
    moves the COMPILE GATE, which is the whole of P26: `by_cost =
    METALJAX_TRACE_BUDGET / BlockCost(body)` is integer division, so the 31B
    decode body, 388 units over the budget without the discount its 60
    attentions are worth, replayed nothing and dispatched ~20000 tape entries
    per token for it.

    Neither check hard-codes a cost.  Both arms are the same binary and the
    same program under `METALJAX_SDPA=1/0`, and the budget the second one
    tests with is derived from the two costs the runs themselves narrate.
    """

    def run(extra_env):
        env = dict(os.environ)
        env["METALJAX_DEBUG"] = "1"
        env.update(extra_env)
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(_P26_CALLEE)
            script = fh.name
        try:
            return subprocess.run([sys.executable, script], env=env,
                                  capture_output=True, text=True)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass

    _FUSED = re.compile(r"sdpa: (\d+) fused attention\(s\) recognized")
    # The gate's own line: the two integers the decision is made of, for the
    # one `stablehlo.while` this program has.  The LAST one is the one that
    # matters -- a recognized program is lowered TWICE (P17's two-phase
    # compile: a plain tape at `CompileAndLoad`, then a fused one at the first
    # execute, which is the tape that runs), so the first line is always the
    # undiscounted cost even in the fused arm.
    _GATE = re.compile(r"while gate: cost=(\d+) .*? body_compile=(\d+)")

    def arm(extra_env):
        proc = run(extra_env)
        err = proc.stderr or ""
        gates = [(int(m.group(1)), int(m.group(2)))
                 for m in _GATE.finditer(err)]
        return proc, {
            "fused": sum(int(m.group(1)) for m in _FUSED.finditer(err)),
            "cost": gates[-1][0] if gates else 0,
            "compile": gates[-1][1] if gates else 0,
            "gates": gates,
            "sum": next((float(ln.split()[2])
                         for ln in proc.stdout.splitlines()
                         if ln.startswith("[probe] checksum ")), None),
        }

    def same(a, b):
        """The fused attention is a different KERNEL, not a different
        function: it may not be bit-identical to the literal chain (it is not
        -- ~6e-8 relative here), so the arms are compared the way every other
        row in this file is, against a tolerance."""
        if a is None or b is None:
            return False
        return abs(a - b) <= 1e-5 * max(abs(a), abs(b), 1.0)

    state = {}

    def the_callee_rooted_attention_is_found():
        on_proc, on = arm({})
        off_proc, off = arm({"METALJAX_SDPA": "0"})
        if on_proc.returncode or off_proc.returncode:
            return False, ((on_proc.stderr or off_proc.stderr)
                           or "").strip()[-140:]
        state["on"], state["off"] = on, off
        if not same(on["sum"], off["sum"]):
            return False, f"answers differ: {on['sum']} vs {off['sum']}"
        if off["fused"]:
            return False, f"{off['fused']} fused with METALJAX_SDPA=0"
        return on["fused"] == 2, (f"{on['fused']} fused (want 2), 0 with "
                                  f"METALJAX_SDPA=0, answers agree")

    def the_discount_reaches_the_compile_gate():
        on, off = state.get("on"), state.get("off")
        if on is None or off is None:
            return False, "the run above did not complete"
        if not on["cost"] or not off["cost"]:
            return False, "no `while gate` line narrated"
        if on["cost"] >= off["cost"]:
            return False, (f"the fused body costs {on['cost']}, the unfused "
                           f"{off['cost']} -- no discount")
        # A budget strictly between the two costs: the fused body clears it
        # (by_cost >= 1), the unfused one does not (by_cost == 0).  Nothing
        # else about the two runs differs.
        budget = (on["cost"] + off["cost"]) // 2
        env = {"METALJAX_TRACE_BUDGET": str(budget)}
        gated_proc, gated = arm(env)
        ungated_proc, ungated = arm(dict(env, METALJAX_SDPA="0"))
        if gated_proc.returncode or ungated_proc.returncode:
            return False, ((gated_proc.stderr or ungated_proc.stderr)
                           or "").strip()[-140:]
        if not same(gated["sum"], ungated["sum"]):
            return False, (f"the gate changed the answer: {gated['sum']} vs "
                           f"{ungated['sum']}")
        ok = gated["compile"] > 0 and ungated["compile"] == 0
        return ok, (f"cost {on['cost']} fused / {off['cost']} unfused; at "
                    f"budget {budget} body_compile="
                    f"{gated['compile']} / {ungated['compile']}")

    return [("an attention in a callee fuses",
             the_callee_rooted_attention_is_found),
            ("the callee discount moves the gate",
             the_discount_reaches_the_compile_gate)]


def _p13_contracts(jax, jnp):
    import jax.experimental   # io_callback

    # `jax.debug.print` itself cannot be exercised here: its impl device_puts
    # the operands onto a local CPU device, and this process sees the metal
    # platform alone (which is what stops a case comparing metal against
    # itself).  `io_callback` reaches the same `metaljax_callback` custom call
    # by the same lowering, and recording into a list says more about ORDER
    # than captured text would.
    def callbacks_run_in_order():
        """The callback trampoline, in the tape's order.

        The loop body holds the callback, so the whole program is impure and
        runs entry by entry -- which is exactly the shape that would run a
        pipelined loop's condition twice (the Stage 1 bug P8.5 found), so the
        count is as much the point as the values.
        """
        seen = []

        @jax.jit
        def f(x):
            def body(i, c):
                jax.experimental.io_callback(
                    lambda a, b: seen.append((int(a), float(b))), None, i, c)
                return c + 1.0
            return jax.lax.fori_loop(0, 3, body, x)

        out = float(np.asarray(f(jnp.float32(10.0))))
        jax.effects_barrier()
        want = [(0, 10.0), (1, 11.0), (2, 12.0)]
        if seen != want:
            return False, f"saw {seen}"
        return (out == 13.0), f"loop returned {out}"

    def ordered_effects_thread_tokens():
        """An ORDERED callback gives main a `!stablehlo.token` parameter and
        a token result, which is the whole of P12's token work: without them
        the program declines on a value that is not a ranked tensor."""
        seen = []

        @jax.jit
        def f(x):
            jax.experimental.io_callback(
                lambda v: seen.append(float(v)), None, x, ordered=True)
            return x * 2

        np.asarray(f(jnp.float32(3.0)))
        out = float(np.asarray(f(jnp.float32(4.0))))
        jax.effects_barrier()
        if seen != [3.0, 4.0]:
            return False, f"saw {seen}"
        return out == 8.0, f"returned {out}"

    def pure_callback_values():
        def host(v):
            return np.sin(v).astype(np.float32)

        x = _rand((5,), 262)
        got = np.asarray(jax.jit(lambda v: jax.pure_callback(
            host, jax.ShapeDtypeStruct((5,), np.float32), v) + 1.0)(x))
        want = np.sin(x) + 1.0
        err = float(np.max(np.abs(got - want)))
        return err < 1e-6, f"max error {err:.3e}"

    def callback_error_propagates():
        def host(v):
            raise ValueError("deliberate")

        try:
            np.asarray(jax.jit(lambda v: jax.pure_callback(
                host, jax.ShapeDtypeStruct((3,), np.float32), v))(
                    jnp.arange(3, dtype=jnp.float32)))
        except BaseException as exc:  # noqa: BLE001
            msg = str(exc)
            return "deliberate" in msg, f"raised {msg.splitlines()[0][:70]}"
        return False, "no error raised"

    def donation_invalidates():
        """XLA's donation contract: a donated argument is gone afterwards, and
        an argument that is not donated is untouched."""
        a = jax.device_put(np.arange(3, dtype=np.float32))
        b = jax.device_put(np.ones(3, np.float32))
        out = jax.jit(lambda x, y: (x + y, y), donate_argnums=0)(a, b)
        if not np.allclose(np.asarray(out[0]), np.arange(3) + 1):
            return False, "wrong result"
        if not a.is_deleted():
            return False, "the donated buffer survived"
        if b.is_deleted():
            return False, "a buffer that was not donated was deleted"
        try:
            np.asarray(a)
        except BaseException:  # noqa: BLE001 - this is the contract
            return True, ""
        return False, "the donated buffer is still readable"

    def buffer_pointer_is_stable():
        """`unsafe_buffer_pointer` is the buffer's IDENTITY, so it may not
        move between calls -- jax asserts on it (`testArrayCopy`).  A value
        held as a broadcast VIEW is the case that used to gather afresh, and
        hand out a new address, on every read."""
        x = jnp.ones(10, jnp.float32)          # a broadcast of one element
        ptrs = {x.unsafe_buffer_pointer() for _ in range(3)}
        if len(ptrs) != 1:
            return False, f"{len(ptrs)} different addresses in 3 calls"
        if jnp.copy(x).unsafe_buffer_pointer() in ptrs:
            return False, "a copy shares the original's buffer"
        return True, ""

    def default_layout_is_answered():
        x = jax.device_put(np.zeros((2, 3), np.float32))
        fmt = x.format
        mtm = tuple(fmt.layout.major_to_minor)
        return mtm == (0, 1), f"major_to_minor {mtm}"

    def cost_analysis_is_answered():
        info = jax.jit(lambda v: v + 1).lower(
            jnp.arange(3, dtype=jnp.float32)).compile().cost_analysis()
        if info is None:
            return False, "None"
        return "metaljax_tape_entries" in info, f"{sorted(info)[:3]}"

    def optimized_program_is_answered():
        """PJRT's `OptimizedProgram`, which jax turns into
        `compiled.as_text()`.  What comes back is the program this executable
        RUNS, as HLO -- unoptimized, because nothing here optimizes at the HLO
        level -- and answering nothing at all is worse: jax turns a refusal
        into `None`, and every caller that greps the text then fails on a
        `NoneType` (five `async_collectives_test` rows did)."""
        text = jax.jit(lambda v: jnp.sin(v) + 1.0).lower(
            jnp.arange(3, dtype=jnp.float32)).compile().as_text()
        if not isinstance(text, str):
            return False, f"{type(text).__name__}"
        return ("sine" in text and "add" in text), text.splitlines()[0][:70]

    def compile_options_are_validated():
        lowered = jax.jit(lambda v: v + 1).lower(
            jnp.arange(3, dtype=jnp.float32))
        try:
            lowered.compile(compiler_options={"invalid_key": "v"})
        except BaseException as exc:  # noqa: BLE001
            if "No such compile option" not in str(exc):
                return False, f"raised {str(exc).splitlines()[0][:70]}"
        else:
            return False, "an unknown compile option was accepted"
        # ...and a known one still compiles and runs.
        exe = lowered.compile(
            compiler_options={"xla_embed_ir_in_executable": True})
        got = np.asarray(exe(jnp.arange(3, dtype=jnp.float32)))
        return np.array_equal(got, np.arange(3) + 1), f"{got}"

    def double_donation_raises():
        """The donation contract is per CALL: the same buffer in a donated
        and in a plain position asks for it to be consumed and read at once.
        Every PjRtClient refuses that; this one used to delete the buffer out
        from under the second use."""
        x = jax.device_put(np.ones(3, np.float32))
        try:
            jax.jit(lambda a, b: a + b, donate_argnums=(0,))(x, x)
        except BaseException as exc:  # noqa: BLE001 - this is the contract
            msg = str(exc)
            if "donated" not in msg:
                return False, f"raised {msg.splitlines()[0][:70]}"
            if x.is_deleted():
                return False, "the buffer was consumed by the refused call"
            return True, ""
        return False, "double donation was accepted"

    def host_memory_space_is_honoured():
        """Apple silicon's memory is unified, so a host placement costs
        nothing -- but the KIND is something jax asks about and reports back,
        and answering `device` to every request made the annotation vanish.
        The client carries both spaces; a buffer points at the one asked
        for."""
        kinds = {m.kind for m in jax.devices()[0].addressable_memories()}
        if not {"device", "pinned_host"} <= kinds:
            return False, f"memory kinds {sorted(kinds)}"
        dev = jax.devices()[0]
        place = lambda kind: jax.sharding.SingleDeviceSharding(  # noqa: E731
            dev, memory_kind=kind)
        x = jax.device_put(np.arange(4, dtype=np.float32),
                           place("pinned_host"))
        if x.sharding.memory_kind != "pinned_host":
            return False, f"device_put gave {x.sharding.memory_kind}"
        if not np.array_equal(np.asarray(x), np.arange(4)):
            return False, "the values did not survive the placement"
        back = jax.device_put(x, place("device"))
        if back.sharding.memory_kind != "device":
            return False, f"copy back gave {back.sharding.memory_kind}"
        # ...and the kind a PROGRAM asks for, which is the annotation the
        # module carries on main's result (`mhlo.memory_kind`).
        out = jax.jit(lambda v: v * 2.0,
                      out_shardings=place("pinned_host"))(x)
        if out.sharding.memory_kind != "pinned_host":
            return False, f"out_shardings gave {out.sharding.memory_kind}"
        return np.array_equal(np.asarray(out), np.arange(4) * 2.0), \
            "the values did not survive the annotation"

    def outputs_own_their_bytes():
        """A buffer this plugin hands out must own its bytes.  A tape output
        can be an MLX view over a SMALLER buffer -- a broadcast is 4 bytes
        pretending to be N -- and jax passes that straight back as the next
        executable's argument, where a consumer reading it as dense memory
        computes on whatever follows those 4 bytes."""
        big = jax.jit(lambda v: jnp.broadcast_to(v, (64, 64)))(
            jnp.float32(2.5))
        # The pointer identifies the buffer; a real one is far apart from its
        # neighbour, and the read below is what would fault or read garbage on
        # a short one.
        got = np.asarray(big)
        if not np.array_equal(got, np.full((64, 64), 2.5, np.float32)):
            return False, "the broadcast did not read back"
        # ...and as an ARGUMENT of a program whose kernel reads it densely.
        rolled = np.asarray(jax.jit(lambda v: jnp.roll(v, 3, axis=1))(big))
        return np.array_equal(rolled, np.full((64, 64), 2.5, np.float32)), \
            "a dense consumer read the broadcast wrongly"

    return [
        ("callbacks run in order", callbacks_run_in_order),
        ("ordered effects thread tokens", ordered_effects_thread_tokens),
        ("pure_callback computes", pure_callback_values),
        ("a callback's error propagates", callback_error_propagates),
        ("donation invalidates its input", donation_invalidates),
        ("double donation raises", double_donation_raises),
        ("buffer pointers are stable", buffer_pointer_is_stable),
        ("outputs own their bytes", outputs_own_their_bytes),
        ("the host memory space is honoured", host_memory_space_is_honoured),
        ("the default layout is answered", default_layout_is_answered),
        ("cost analysis is answered", cost_analysis_is_answered),
        ("the optimized program is answered", optimized_program_is_answered),
        ("compile options are validated", compile_options_are_validated),
    ]


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def _canonical(value):
    """An output as (dtype-class, widened array), comparable across backends.

    Widening to f64/c128 is exact for every dtype this backend has, so the
    comparison never loses a bit that the tolerance then has to hide; the
    class is what decides whether a tolerance applies at all.
    """
    arr = np.asarray(value)
    if arr.dtype.kind == "c":
        return "c", arr.astype(np.complex128)
    if arr.dtype.kind in "fV":       # 'V' is how ml_dtypes' bfloat16 presents
        return "f", arr.astype(np.float64)
    if arr.dtype.kind == "b":
        return "b", arr.astype(np.int64)
    return "i", arr.astype(np.int64)


def _flatten(out):
    if isinstance(out, (tuple, list)):
        return [np.asarray(v) for v in out]
    return [np.asarray(out)]


def _compare(name, got, want, rtol, atol):
    """Largest error, or a message.  Returns (ok, detail)."""
    if len(got) != len(want):
        return False, f"{len(got)} outputs, the CPU backend gave {len(want)}"
    worst = 0.0
    for j, (g, (wk, wa)) in enumerate(zip(got, want)):
        gk, ga = _canonical(g)
        if gk != wk:
            return False, f"output {j} is {gk!r}, the CPU backend gave {wk!r}"
        if ga.shape != wa.shape:
            return False, f"output {j} is {ga.shape}, CPU gave {wa.shape}"
        if gk in "ib":
            if not np.array_equal(ga, wa):
                bad = int(np.count_nonzero(ga != wa))
                return False, f"output {j}: {bad} of {ga.size} elements differ"
            continue
        # NaNs must land in the same places, then the finite parts compare.
        if not np.array_equal(np.isnan(ga), np.isnan(wa)):
            return False, f"output {j}: NaNs are in different places"
        finite = ~np.isnan(ga)
        if not np.array_equal(np.isinf(ga[finite]), np.isinf(wa[finite])):
            return False, f"output {j}: infinities are in different places"
        both = finite & ~np.isinf(ga)
        if not np.any(both):
            continue
        err = np.abs(ga[both] - wa[both])
        scale = atol + rtol * np.abs(wa[both])
        rel = np.max(err / np.maximum(scale, 1e-300))
        worst = max(worst, float(np.max(err)))
        if rel > 1.0:
            return False, (f"output {j}: max |error| {float(np.max(err)):.3e} "
                           f"exceeds atol {atol} + rtol {rtol}")
    return True, worst


# --------------------------------------------------------------------------
# the two halves
# --------------------------------------------------------------------------


def write_reference(path):
    """Run every case on the CPU backend and save the answers."""
    import jax

    saved = {}
    for i, (name, fn, args, _rtol, _atol) in enumerate(_cases()):
        outs = _flatten(jax.jit(fn)(*args))
        saved[f"n{i}"] = np.asarray(len(outs))
        for j, out in enumerate(outs):
            kind, arr = _canonical(out)
            saved[f"k{i}_{j}"] = np.asarray(kind)
            saved[f"v{i}_{j}"] = arr
    for i, (name, text, args) in enumerate(_module_cases()):
        outs = _run_module(text, args)
        saved[f"mn{i}"] = np.asarray(len(outs))
        for j, out in enumerate(outs):
            kind, arr = _canonical(out)
            saved[f"mk{i}_{j}"] = np.asarray(kind)
            saved[f"mv{i}_{j}"] = arr
    np.savez(path, **saved)


def write_eager_arm(path):
    """Run every case through the PLUGIN with the compile decisions off.

    The caller's environment already holds METALJAX_COMPILE=0, which the dylib
    reads once at load: `chunkable`, `body_compile_max` and the whole-main
    `set_compile` all go to zero, which is the all-eager plugin P3 and P4
    measured.  The parent compares these answers with its own compiled ones --
    see the "eager vs compiled" section for what that comparison demands.
    """
    import jax

    saved = {}
    for i, (name, fn, args, _rtol, _atol) in enumerate(_cases()):
        outs = _flatten(jax.jit(fn)(*args))
        saved[f"n{i}"] = np.asarray(len(outs))
        for j, out in enumerate(outs):
            kind, arr = _canonical(out)
            saved[f"k{i}_{j}"] = np.asarray(kind)
            saved[f"v{i}_{j}"] = arr
    np.savez(path, **saved)


def read_module_reference(path, index):
    data = np.load(path)
    n = int(data[f"mn{index}"])
    return [(str(data[f"mk{index}_{j}"]), data[f"mv{index}_{j}"])
            for j in range(n)]


def read_reference(path, index):
    """The CPU answers for one case, as (kind, widened array) pairs.

    The kind is stored rather than re-derived: canonicalization widens a bool
    to an integer array, so reading it back would say "integer" for a case
    whose result really is boolean, and a dtype disagreement between the
    backends would then go unnoticed.
    """
    data = np.load(path)
    n = int(data[f"n{index}"])
    return [(str(data[f"k{index}_{j}"]), data[f"v{index}_{j}"])
            for j in range(n)]


def _p21_msl(subprocess, pathlib, re):
    """The msl_scan contracts: which emitters really run, and the knob.

    The three modes exist because they are three different lane geometries,
    and a port that quietly stopped picking one would show up nowhere else --
    every case would still be right, just slower.  So the census is a test:
    the plugin's own narration, read out of a child run with METALJAX_DEBUG.
    """

    def modes_are_covered():
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--msl-modes"], env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).splitlines()[-1][:80]
        modes = re.findall(r"msl_scan: compiled plan .*?mode=(\w+)",
                           proc.stderr)
        seen = sorted(set(modes))
        missing = {"scalar", "vector", "coop"} - set(seen)
        if missing:
            return False, f"no plan in {sorted(missing)} mode (saw {seen})"
        return True, f"{len(modes)} kernels: {', '.join(seen)}"

    def the_kill_switch_kills():
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        child["METALJAX_MSL"] = "0"
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--msl-modes"], env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).splitlines()[-1][:80]
        n = len(re.findall(r"msl_scan: compiled plan", proc.stderr))
        return n == 0, f"{n} kernels with METALJAX_MSL=0"

    def the_width_cap_holds():
        """P22's deliberate divergence from `msl_scan.py`: a coop plan is
        refused at state width F >= 1024 even when its total dot work is
        under `METALJAX_MSL_COOP_CAP`.

        Stage 1 has the work cap only, and a square `rnn.1024` cell slips
        under it (1.05M elems/step) to run 1.5x SLOWER than the compiled
        matmul -- the re-streaming that cap is about is per FEATURE width,
        not per total.  Both halves are pinned here: the cap fires by
        default, `METALJAX_MSL_COOP_MAX_F=0` restores Stage 1's policy, and
        the two paths agree on the answer.
        """
        def run(env_extra):
            child = dict(os.environ)
            child["METALJAX_DEBUG"] = "1"
            child.update(env_extra)
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()),
                 "--msl-wide-coop"], env=child, capture_output=True, text=True)
            if proc.returncode != 0:
                return None, None, (proc.stderr or proc.stdout
                                    ).splitlines()[-1][:80]
            plans = len(re.findall(r"msl_scan: compiled plan", proc.stderr))
            declines = re.findall(r"not eligible \(coop: state width F=(\d+)",
                                  proc.stderr)
            out = re.search(r"WIDE COOP CHECKSUM ([-\d.e+]+)", proc.stdout)
            return (plans, declines, out.group(1) if out else None), proc, None

        gated, _, err = run({})
        if err: return False, err
        stage1, _, err = run({"METALJAX_MSL_COOP_MAX_F": "0"})
        if err: return False, err
        if gated[0] != 0 or not gated[1]:
            return False, (f"default built {gated[0]} plan(s) at F=1024 "
                           f"(declines seen: {gated[1]})")
        if stage1[0] == 0:
            return False, "COOP_MAX_F=0 did not restore the coop plan"
        d = abs(float(gated[2]) - float(stage1[2]))
        rel = d / max(abs(float(stage1[2])), 1e-30)
        # A sum over 12,288 elements, so the bar is loose on purpose: the two
        # paths contract in different orders (the same reason the three
        # fissioned weight-gradient rows are not bit-identical either), and
        # the summation amplifies it.  Measured 9.1e-06; the bar is here to
        # catch a WRONG fallback, not to pin an accumulation order.
        if rel > 1e-4:
            return False, f"gated vs Stage-1-policy answers differ by {rel:.2e}"
        return True, (f"F=1024 declined by width (was {stage1[0]} coop plan), "
                      f"answers agree to {rel:.1e}")

    def a_planned_loop_is_charged_as_one_kernel():
        """P23: the byte estimate the COMPILE decisions are made on must
        notice the kernel.

        `ops/control._block_bytes` charges a loop that became one generated
        msl kernel its OUTPUTS only -- the per-timestep state lives in
        registers, not in buffers -- while an interpreted loop is charged
        trip x body.  The port had that case in `BlockCost` and not in
        `BlockBytes`, so a planned loop was charged as if it ran: on
        `db16-b256l512` the step estimate came out at 163 GB instead of 2 GB,
        over `METALJAX_COMPILE_BYTES_MB`, which took away the body compile
        AND the chunked replay and left every step to be dispatched op by op
        (1.77x slower than Stage 1, with identical kernels -- P23).

        No correctness test can see this: every answer stays right.  What is
        pinned here is the MECHANISM rather than a threshold -- the same
        program, planned and unplanned, must not be charged the same bytes.
        """
        def probe(env_extra):
            child = dict(os.environ)
            child["METALJAX_DEBUG"] = "1"
            child.update(env_extra)
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()),
                 "--msl-bytes"], env=child, capture_output=True, text=True)
            if proc.returncode != 0:
                return None, None, (proc.stderr or proc.stdout
                                    ).splitlines()[-1][:80]
            mb = [float(x) for x in re.findall(
                r"main: pure=\d+ cost=\d+ bytes=([\d.]+)MB", proc.stderr)]
            plans = len(re.findall(r"msl_scan: compiled plan", proc.stderr))
            if not mb:
                return None, None, "no byte narration"
            return max(mb), plans, None

        on, plans_on, err = probe({})
        if err: return False, err
        off, plans_off, err = probe({"METALJAX_MSL": "0"})
        if err: return False, err
        if plans_on < 1:
            return False, "no kernel planned for the probe program"
        if plans_off != 0:
            return False, f"METALJAX_MSL=0 still planned {plans_off}"
        # The probe cell is ~13 arrays of body traffic per step against one
        # stacked output, so the honest estimate is several times smaller.
        # The bug made the two arms report the SAME number.
        if not (on * 3 <= off):
            return False, (f"planned {on:.1f} MB vs interpreted {off:.1f} MB "
                           "-- the byte gate does not see the kernel")
        return True, f"{on:.1f} MB planned vs {off:.1f} MB interpreted"

    return [("msl covers its three modes", modes_are_covered),
            ("METALJAX_MSL=0 builds no kernel", the_kill_switch_kills),
            ("msl coop width cap (F>=1024)", the_width_cap_holds),
            ("msl loop charged as one kernel",
             a_planned_loop_is_charged_as_one_kernel)]


def _arm_section(title, env_extra, tag, ref_path, compiled_arm, failures):
    """Re-run every case through the SAME dylib under `env_extra` and compare.

    The child writes its answers with `--eager-arm` (the entry point is the
    plugin arm generally: what changes between the two callers is the
    environment the dylib reads at load).  Bit-identity is reported, not
    demanded: a compile decision or a generated kernel changes which MLX
    kernels run, and the bar is each case's own CPU tolerance.
    """
    print(f"\n{title}")
    print("-" * 62)
    path = ref_path.with_name(ref_path.name.replace("reference",
                                                    "arm-" + tag))
    child = dict(os.environ)
    child.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--eager-arm",
         str(path)],
        env=child, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        print(f"{'the arm':<32} {'-':>12}  FAIL: the child failed")
        failures.append(tag)
        return
    arm = np.load(path)
    inexact, bad = [], []
    for i, (name, _fn, _args, rtol, atol) in enumerate(_cases()):
        if i not in compiled_arm:
            continue
        got = compiled_arm[i]
        want = [(str(arm[f"k{i}_{j}"]), arm[f"v{i}_{j}"])
                for j in range(int(arm[f"n{i}"]))]
        identical = len(got) == len(want) and all(
            gk == wk and ga.shape == wa.shape and
            (np.array_equal(ga, wa) or
             (gk in "fc" and
              np.array_equal(np.isnan(ga), np.isnan(wa)) and
              np.array_equal(ga[~np.isnan(ga)], wa[~np.isnan(wa)])))
            for (gk, ga), (wk, wa) in zip(got, want))
        if identical:
            continue
        ok, detail = _compare(name, [ga for _k, ga in got], want, rtol, atol)
        inexact.append(f"{name} ({detail:.1e})" if ok else name)
        if not ok:
            bad.append(f"{name}: {detail}")
    n = len(compiled_arm)
    label = f"{n - len(inexact)} of {n} bit-identical"
    if bad:
        print(f"{label:<32} {'-':>12}  FAIL: {'; '.join(bad[:3])}")
        failures.append(tag)
    else:
        print(f"{label:<32} {'':>12}  ok"
              + (f" (within tolerance: {', '.join(inexact)})"
                 if inexact else ""))
    path.unlink(missing_ok=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--reference":
        write_reference(sys.argv[2])
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--msl-modes":
        # The mode census: run the msl cases through the plugin with
        # METALJAX_DEBUG on, so the parent can read which emitters really ran
        # out of the plugin's own narration.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        for name, fn, args, _rtol, _atol in _cases():
            if name.startswith("msl "):
                jax.jit(fn)(*args)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--msl-wide-coop":
        # One square matvec cell at the width the P22 cap is about (F=1024,
        # 1.05M dot elems/step -- under METALJAX_MSL_COOP_CAP, which is why
        # Stage 1 takes it).  The parent reads the plan census out of the
        # narration and the checksum out of stdout.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        h0 = _rand((2, 1024), 921) * np.float32(0.1)
        xs = _rand((6, 2, 1024), 922) * np.float32(0.1)
        w = _rand((1024, 1024), 923) * np.float32(0.02)
        _, hs = jax.jit(_msl_rnn)(h0, xs, w)
        print(f"WIDE COOP CHECKSUM {float(np.asarray(hs).sum()):.9e}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--msl-bytes":
        # P23: one gru cell fat enough that its body traffic dwarfs its
        # stacked output, planned into a kernel by default and interpreted
        # under METALJAX_MSL=0.  The parent reads the byte estimate the
        # compile decisions are made on out of the narration.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        f32 = np.float32
        h0 = _rand((32, 64), 931) * f32(0.1)
        xs = _rand((128, 32, 64), 932) * f32(0.1)
        ws = [_rand((64, 64), 933 + i) * f32(0.1) for i in range(3)]
        jax.jit(_msl_gru)(h0, xs, *ws)
        return 0
    if len(sys.argv) > 2 and sys.argv[1] == "--eager-arm":
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        write_eager_arm(sys.argv[2])
        return 0

    os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
    dylib = pathlib.Path(os.environ["METALJAX_PLUGIN_PATH"])
    if not dylib.exists():
        sys.exit(f"plugin dylib not found: {dylib}")
    os.environ["JAX_PLATFORMS"] = "metal"
    print(f"plugin: {dylib} ({dylib.stat().st_size / 1e6:.1f} MB)")

    # The CPU answers come from a subprocess: in this one, jax sees only the
    # metal platform, and pointing it at both would let a case silently
    # compare the metal backend against itself.
    ref_path = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"metaljax-native-reference-{os.getpid()}.npz")
    child = dict(os.environ)
    child["JAX_PLATFORMS"] = "cpu"
    child.pop("METALJAX_PLUGIN_PATH", None)
    print("computing CPU references ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--reference",
         str(ref_path)],
        env=child, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.exit("the CPU reference run failed")

    import jax  # noqa: E402  (after JAX_PLATFORMS is fixed)
    import jax.numpy as jnp  # noqa: E402

    failures = []
    compiled_arm = {}
    print(f"\n{'case':<32} {'max error':>12}  result")
    print("-" * 62)
    for i, (name, fn, args, rtol, atol) in enumerate(_cases()):
        try:
            got = _flatten(jax.jit(fn)(*args))
            compiled_arm[i] = [_canonical(g) for g in got]
            want = read_reference(ref_path, i)
            ok, detail = _compare(name, got, want, rtol, atol)
        except BaseException as exc:  # noqa: BLE001 - report and continue
            ok, detail = False, f"{type(exc).__name__}: " \
                                f"{str(exc).splitlines()[0][:110]}"
        if ok:
            print(f"{name:<32} {detail:>12.3e}  ok")
        else:
            print(f"{name:<32} {'-':>12}  FAIL: {detail}")
            failures.append(name)

    # The compile decisions change which MLX KERNELS run -- mx::compile fuses
    # elementwise chains -- and where a loop's sync points fall.  The child
    # runs the same cases through the same dylib with METALJAX_COMPILE=0, the
    # all-eager plugin of P3/P4, and every case must still land inside its own
    # CPU tolerance.  Most are bit-identical; the ones that are not are named,
    # because a fused kernel evaluating a transcendental differently is a fact
    # about MLX worth seeing rather than a threshold to hide.
    _arm_section("eager vs compiled (METALJAX_COMPILE=0 in a child)",
                 {"METALJAX_COMPILE": "0"}, "eager-vs-compiled",
                 ref_path, compiled_arm, failures)

    # The same shape for the generated kernels: with METALJAX_MSL=0 not one
    # loop takes a kernel, so the child computes every msl case through the
    # INTERPRETED loop the entry still carries.  A kernel accumulates a dot in
    # its own order, so bit-identity is reported rather than demanded and the
    # bar is each case's own CPU tolerance -- but a case that takes no kernel
    # must be identical, which is what makes the count mean something.
    _arm_section("msl kernels vs the interpreted loop "
                 "(METALJAX_MSL=0 in a child)",
                 {"METALJAX_MSL": "0"}, "msl-off", ref_path, compiled_arm,
                 failures)

    # ...and the recovery: with every generated source made invalid, Metal
    # rejects the kernel at its first EVAL and the executor must retire the
    # plan and run the interpreted loop in the same call (`Program::run_msl`,
    # `Program::settle_msl`).  Every case, not just the msl ones: what is
    # being proved is that a build failure costs an answer nowhere.
    _arm_section("a rejected kernel falls back to the loop "
                 "(METALJAX_MSL_FORCE_BUILD_FAIL=1 in a child)",
                 {"METALJAX_MSL_FORCE_BUILD_FAIL": "1"}, "msl-build-failure",
                 ref_path, compiled_arm, failures)

    print("\nhand-written StableHLO (the same text through both clients)")
    print("-" * 62)
    for i, (name, text, margs) in enumerate(_module_cases()):
        try:
            got = _run_module(text, margs)
            want = read_module_reference(ref_path, i)
            ok, detail = _compare(name, got, want, 1e-6, 1e-6)
        except BaseException as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: " \
                                f"{str(exc).splitlines()[0][:110]}"
        if ok:
            print(f"{name:<32} {detail:>12.3e}  ok")
        else:
            print(f"{name:<32} {'-':>12}  FAIL: {detail}")
            failures.append(name)

    print("\ndeclines (the message must name the op)")
    print("-" * 62)
    for name, fn, args, op in _declines():
        try:
            # A str stands for a hand-written module: some declines are
            # encodings jax's own lowerings cannot produce.
            if isinstance(fn, str):
                _run_module(fn, args)
            else:
                jax.jit(fn)(*args)
        except BaseException as exc:  # noqa: BLE001
            msg = str(exc)
            if "UNIMPLEMENTED" in msg and op in msg:
                print(f"{name:<32} {'':>12}  ok ({op})")
                continue
            print(f"{name:<32} {'':>12}  FAIL: {msg.splitlines()[0][:90]}")
            failures.append(f"decline {name}")
            continue
        print(f"{name:<32} {'':>12}  FAIL: it did not decline")
        failures.append(f"decline {name}")

    # XLA's no-alias contract: an output may not share a buffer with an input.
    # The tape works this out statically (metal_lowering.cc's copy rule), and
    # a jitted identity is the case that exercises it.
    print("\ncontracts")
    print("-" * 62)
    try:
        src = jax.device_put(np.arange(4, dtype=np.float32))
        out = jax.jit(lambda a: a)(src)
        same = src.unsafe_buffer_pointer() == out.unsafe_buffer_pointer()
        print(f"{'identity returns a fresh buffer':<32} {'':>12}  "
              f"{'FAIL: aliased' if same else 'ok'}")
        if same:
            failures.append("no-alias contract")
    except BaseException as exc:  # noqa: BLE001
        print(f"{'identity returns a fresh buffer':<32} {'':>12}  "
              f"FAIL: {exc}")
        failures.append("no-alias contract")

    # The same contract THROUGH a region: a carry the body forwards untouched
    # is still the caller's array on the way out, and the tape has to see that
    # across the frame boundary (metal_lowering.cc's MapTaint).  A loop whose
    # trip count is not statically zero is charged the taint of its init too,
    # which is what this exercises.
    label = "a forwarded carry is copied"
    try:
        src = jax.device_put(np.arange(4, dtype=np.float32))
        out = jax.jit(lambda x, w: jax.lax.fori_loop(
            0, 3, lambda i, s: (s[0] + w, s[1]), (x, w)))(
                jax.device_put(np.zeros(4, np.float32)), src)
        same = src.unsafe_buffer_pointer() == out[1].unsafe_buffer_pointer()
        ok = not same and np.array_equal(np.asarray(out[1]),
                                         np.arange(4, dtype=np.float32))
        detail = "ok" if ok else ("FAIL: aliased" if same else "FAIL: wrong")
    except BaseException as exc:  # noqa: BLE001
        ok, detail = False, f"FAIL: {str(exc).splitlines()[0][:90]}"
    print(f"{label:<32} {'':>12}  {detail}")
    if not ok:
        failures.append(label)

    for label, check in _p13_contracts(jax, jnp):
        try:
            ok, detail = check()
        except BaseException as exc:  # noqa: BLE001 - report and continue
            ok, detail = False, f"{type(exc).__name__}: " \
                                f"{str(exc).splitlines()[0][:90]}"
        print(f"{label:<32} {'':>12}  {'ok' if ok else f'FAIL: {detail}'}")
        if not ok:
            failures.append(label)

    for label, check in (_p19_packing(subprocess, tempfile, pathlib)
                         + _p21_msl(subprocess, pathlib, __import__("re"))
                         + _p25_cache_limit(subprocess, tempfile, pathlib,
                                            __import__("re"))
                         + _p27_flush_pressure(subprocess, tempfile, pathlib,
                                               __import__("re"))
                         + _p28_benefit_gate(subprocess, tempfile, pathlib,
                                             __import__("re"))
                         + _p26_callee_sdpa(subprocess, tempfile, pathlib,
                                            __import__("re"))
                         + _governor(subprocess, tempfile, pathlib,
                                     __import__("re"))):
        try:
            ok, detail = check()
        except BaseException as exc:  # noqa: BLE001 - report and continue
            ok, detail = False, f"{type(exc).__name__}: " \
                                f"{str(exc).splitlines()[0][:90]}"
        print(f"{label:<32} {'':>12}  "
              f"{(detail or 'ok') if ok else f'FAIL: {detail}'}")
        if not ok:
            failures.append(label)

    # Every dtype the transfer path claims, host -> device -> host, bit exact,
    # plus a negative-stride view (whose logical first element is not its
    # lowest address).
    dtypes = [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8,
              np.uint16, np.uint32, np.uint64, np.float16, np.float32,
              np.complex64]
    for dt in dtypes:
        src = (np.arange(6) % 2).astype(dt) if dt is np.bool_ else \
            np.arange(6).astype(dt)
        # With x64 off jax canonicalizes 64-bit integers down to 32 on the way
        # in, so the dtype to expect back is jax's answer, not numpy's.
        want_dtype = np.dtype(jax.dtypes.canonicalize_dtype(src.dtype))
        try:
            back = np.asarray(jax.device_put(src))
            ok = back.dtype == want_dtype and np.array_equal(
                back, src.astype(want_dtype))
        except BaseException as exc:  # noqa: BLE001
            ok, back = False, exc
        label = f"round-trip {np.dtype(dt).name}"
        print(f"{label:<32} {'':>12}  {'ok' if ok else f'FAIL: {back}'}")
        if not ok:
            failures.append(label)
    try:
        import ml_dtypes
        src = np.arange(6, dtype=np.float32).astype(ml_dtypes.bfloat16)
        back = np.asarray(jax.device_put(src))
        ok = back.dtype == src.dtype and np.array_equal(
            back.astype(np.float32), src.astype(np.float32))
    except BaseException as exc:  # noqa: BLE001
        ok, back = False, exc
    print(f"{'round-trip bfloat16':<32} {'':>12}  "
          f"{'ok' if ok else f'FAIL: {back}'}")
    if not ok:
        failures.append("round-trip bfloat16")
    try:
        src = np.arange(12, dtype=np.float32).reshape(3, 4)[::-1, ::-2]
        back = np.asarray(jax.device_put(src))
        ok = np.array_equal(back, src)
    except BaseException as exc:  # noqa: BLE001
        ok, back = False, exc
    print(f"{'round-trip negative strides':<32} {'':>12}  "
          f"{'ok' if ok else f'FAIL: {back}'}")
    if not ok:
        failures.append("round-trip negative strides")

    # One executable, called from many threads at once.  MLX streams are
    # thread-bound; the plugin's answer is a cross-thread-evaluable stream per
    # entering thread (metal_stream.cc), and this is what exercises it.
    #
    # Truly concurrent evals used to SEGFAULT ~5 % of runs inside
    # mlx::core::metal::get_command_encoder (a `new_thread_unsafe_stream`
    # routes through one process-wide encoder map; 4 crashes in 74 runs at
    # the pinned command-buffer budgets, 0 through the GIL-serialized Stage 1
    # plugin).  metal_stream.cc's SubmissionLock now serializes submission --
    # 0 crashes in 30 full-suite runs -- so a segfault here means the lock
    # was lifted (METALJAX_CONCURRENT_EXECUTE=1) or regressed; see
    # notes/cpp-p4-gather-scatter.md.
    import concurrent.futures

    try:
        # An all-positive sum on purpose: a cancelling one would measure f32
        # summation order, not the threading.
        fn = jax.jit(lambda x: jnp.abs(jnp.tanh(x * 2)).sum())
        xs = [_rand((64, 64), 100 + i) for i in range(32)]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda x: float(np.asarray(fn(x))), xs))
        want = [float(np.abs(np.tanh(x.astype(np.float64) * 2)).sum())
                for x in xs]
        worst = max(abs(a - b) / max(abs(b), 1.0) for a, b in zip(got, want))
        ok = worst < 1e-5
        detail = "ok" if ok else f"FAIL: max relative error {worst:.2e}"
    except BaseException as exc:  # noqa: BLE001
        ok, detail = False, f"FAIL: {str(exc).splitlines()[0][:90]}"
    print(f"{'32 executes on 8 threads':<32} {'':>12}  {detail}")
    if not ok:
        failures.append("threaded execute")

    # The f64 policy, STRICT exactly as Stage 1 defines it: an f64 buffer
    # passes through (stored f32) and an f64 COMPUTATION declines naming the
    # element type.  In a subprocess because enabling x64 is a global switch.
    probe = (
        "import os, numpy as np, jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "x = np.arange(4, dtype=np.float64)\n"
        "back = np.asarray(jax.device_put(x))\n"
        "assert back.dtype == np.float64 and np.array_equal(back, x), back\n"
        "try:\n"
        "    jax.jit(lambda a: a * 2)(x)\n"
        "except Exception as e:\n"
        "    assert 'element type f64' in str(e), str(e)\n"
        "    print('ok')\n"
        "else:\n"
        "    raise SystemExit('an f64 computation did not decline')\n")
    proc = subprocess.run([sys.executable, "-c", probe], env=dict(os.environ),
                          capture_output=True, text=True)
    ok = proc.returncode == 0 and "ok" in proc.stdout
    print(f"{'f64 passes through, f64 math not':<32} {'':>12}  "
          f"{'ok' if ok else 'FAIL: ' + (proc.stderr or proc.stdout).strip()[-90:]}")
    if not ok:
        failures.append("f64 policy")

    # Half-precision linalg, which has NO CPU answer to compare with: jax's
    # own rules reject bf16/f16 there outright (its LAPACK tables have no
    # half-precision entry), while metaljax's lowerings accept every float
    # dtype and the host handlers compute in f32 and cast back.  So the check
    # is the one a reference cannot give: that it runs, and that what comes
    # back really is a factorization of the operand it was handed.
    for label, dt in (("bfloat16", jnp.bfloat16), ("float16", jnp.float16)):
        try:
            a = _spd(4, 500)
            x = jax.device_put(jnp.asarray(a).astype(dt))
            # The operand the factorization actually saw: the half-rounded
            # matrix, which is what its invariants are invariants OF.
            rounded = np.asarray(x.astype(jnp.float32), np.float64)
            w, v = jax.jit(jnp.linalg.eigh)(x)
            w = np.asarray(jnp.asarray(w).astype(jnp.float32), np.float64)
            v = np.asarray(jnp.asarray(v).astype(jnp.float32), np.float64)
            recon = np.abs((v * w) @ v.T - rounded).max() / np.abs(a).max()
            orth = np.abs(v.T @ v - np.eye(4)).max()
            s = np.asarray(jnp.asarray(jax.jit(
                lambda z: jnp.linalg.svd(z, compute_uv=False))(x)
            ).astype(jnp.float32), np.float64)
            sval = np.abs(np.sort(s)[::-1]
                          - np.linalg.svd(rounded, compute_uv=False)).max()
            sval /= np.abs(a).max()
            c = np.asarray(jnp.asarray(jax.jit(jnp.linalg.cholesky)(x)
                                       ).astype(jnp.float32), np.float64)
            chol = np.abs(c @ c.T - rounded).max() / np.abs(a).max()
            # bf16 keeps 8 mantissa bits, so 1 % of the operand's norm is the
            # band its own rounding earns; f16 is inside it comfortably.
            worst = max(recon, orth, sval, chol)
            ok = worst < 1e-2
            detail = f"ok (worst invariant {worst:.1e})" if ok else \
                f"FAIL: worst invariant {worst:.1e}"
        except BaseException as exc:  # noqa: BLE001
            ok, detail = False, f"FAIL: {str(exc).splitlines()[0][:90]}"
        print(f"{label + ' linalg (no CPU rule)':<32} {'':>12}  {detail}")
        if not ok:
            failures.append(f"{label} linalg")

    try:
        ref_path.unlink()
    except OSError:
        pass

    if failures:
        print(f"\n{len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("\nall cases match the CPU backend")
    return 0


if __name__ == "__main__":
    # jax is imported only inside main(), after JAX_PLATFORMS is fixed: an
    # import up here would pin the platform before the child process could
    # choose the CPU one.
    sys.exit(main())
