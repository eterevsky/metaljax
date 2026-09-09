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


def _msl_rnn_rect(h0, xs, w, u):
    """A matvec cell with a RECTANGULAR input projection (10 % 4 != 0): the
    coop flip requires every dot dim to be a multiple of the state width, so
    this cell stays in `vector` mode even now that the flip admits F=4
    (2026-08-26) -- the census's vector coverage lives here."""
    import jax
    import jax.numpy as jnp

    def step(h, x):
        h = jnp.tanh(x @ u + h @ w)
        return h, h
    return jax.lax.scan(step, h0, xs)


def _msl_rnn_rect_grad(h0, xs, w, u):
    """...and its weight gradients, so vector-mode loop fission keeps a case
    after the F=4 square cells moved to coop."""
    import jax
    import jax.numpy as jnp

    def loss(w, u):
        _, hs = _msl_rnn_rect(h0, xs, w, u)
        return jnp.sum(hs * hs * 0.5)
    return jax.grad(loss, argnums=(0, 1))(w, u)


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


# --------------------------------------------------------------------------
# msl_scan carries that are NOT the loop counter (2026-09-02)
# --------------------------------------------------------------------------
#
# THE BUG (silent wrongness, present since msl_scan existed and carried
# through the Stage 2 port): `MslAnalyzer::Analyze` seeded a SymCounter for
# `i == counter_pos || (int scalar)`.  A SymCounter denotes "the induction
# variable", and the emitters bake it in as `a*(t+start)+b` -- a value read
# out of the ITERATION INDEX, never out of the carry.  That is true of the
# carry `_analyze_counted` proved is the counter, and of nothing else: any
# other i32/i64 scalar carry silently became the loop index, and never even
# became a kernel input.  An invariant `int32` position held across a 28-step
# scan produced sin(0), sin(1), ... instead of sin(7).
#
# The fix identifies the counter by DATAFLOW alone (the carry the cond
# compares and the body returns as `arg + 1`), so these five shapes pin the
# classification from both sides: an invariant integer carry must read back
# as ITSELF, and a carry that merely LOOKS like a counter must not be given
# the counter's arithmetic.
#
# Every body here is exact in f32 on purpose -- doublings and small integers,
# nothing above 2**16 -- so the comparison is EXACT and no tolerance can hide
# an off-by-a-few-iterations read.  Each returns the final integer carries
# too, not just the float state: the old code got the final value RIGHT
# (post-kernel `carry + delta*trip`) and only the in-body reads wrong, so a
# case that checked the carry alone would have passed on the bug.


def _msl_invariant_int(a0, q, xs):
    """An invariant int32 SCALAR carry beside the counter -- the row-11 rope
    shape, minimised.  `q` is returned unchanged forever, so the kernel must
    load it as an input and read 7 (or whatever it is) on every iteration."""
    import jax
    import jax.numpy as jnp

    def step(c, x):
        a, q = c
        return (a * 2.0 + q.astype(jnp.float32) + x, q), a
    (a, q), ys = jax.lax.scan(step, (a0, q), xs)
    return a, q, ys


def _msl_invariant_int_first(q, a0, b0, xs):
    """The same invariant carry, moved to the FIRST scan carry -- while-carry
    index 1, immediately after the counter, which is where a classification
    that went by POSITION rather than by dataflow would hide.  Two float
    states around it so the position really differs from the case above."""
    import jax
    import jax.numpy as jnp

    def step(c, x):
        q, a, b = c
        fq = q.astype(jnp.float32)
        return (q, a * 2.0 + fq + x, b - fq), (a, b)
    (q, a, b), ys = jax.lax.scan(step, (q, a0, b0), xs)
    return q, a, b, ys[0], ys[1]


def _msl_invariant_float(a0, s, xs):
    """An invariant FLOAT scalar carry.  This one was always right -- only
    integer scalars were seeded as counters -- and it is here so that the
    guard is "the counter is the carry the cond compares", not "the counter
    is the integer one"."""
    import jax

    def step(c, x):
        a, s = c
        return (a * 2.0 + s + x, s), a
    (a, s), ys = jax.lax.scan(step, (a0, s), xs)
    return a, s, ys


def _msl_counter_lookalike(a0, j, b0, xs):
    """Two carries that look exactly like counters; only one feeds the cond.

    `j` starts at 100 and the body returns `j + 1`: structurally identical to
    the induction variable, distinguishable ONLY by the cond's compare (and
    by its start).  The old code called it a counter and read `t + start` for
    it -- 0, 1, 2, ... instead of 100, 101, 102 -- while still handing back
    the right final `j`.  Middle position, to move it again."""
    import jax
    import jax.numpy as jnp

    def step(c, x):
        a, j, b = c
        fj = j.astype(jnp.float32)
        return (a * 2.0 + fj + x, j + 1, b - fj), (a, b)
    (a, j, b), ys = jax.lax.scan(step, (a0, j, b0), xs)
    return a, j, b, ys[0], ys[1]


def _msl_strided_int_carry(a0, j, xs):
    """An integer carry that IS incremented but does not feed the cond, and
    by 3 rather than 1.  The old code read `t + start` for it; the fix makes
    it an ordinary kernel state, so the kernel keeps `j` in a register and
    steps it itself."""
    import jax
    import jax.numpy as jnp

    def step(c, x):
        a, j = c
        return (a * 2.0 + j.astype(jnp.float32) + x, j + 3), a
    (a, j), ys = jax.lax.scan(step, (a0, j), xs)
    return a, j, ys


def _lane_ints(shape, seed):
    """Exact-in-f32 operands for the one-scalar-per-lane cases: small
    integers, every lane different, so a row broadcast cannot pass."""
    n = int(np.prod(shape))
    return ((np.arange(n, dtype=np.float32) * 7 + 3 * seed) % 5 - 2).reshape(
        shape)


# One scalar per lane (2026-09-02).  A value shaped like the trailing dims of
# the lane space has no register dim -- a reduce over the feature axis
# yields (B,) or (B1, B2), a vmapped scalar carry is (B,), a per-step scalar
# input is (B,) -- and vector-mode kernels read a value's trailing dim as its
# register width, so each lane broadcast its own scalar over a whole row of
# the buffer: wrong for carries and per-step reads in every lane rank, and
# for stacked outputs in 2-D lane spaces (2125c96 declined the 1-D stacked
# case only).  Doublings of small integers, compared EXACTLY; the "one
# scalar per lane takes a kernel" contract asserts these loops plan.

def _msl_lane_scalar_carry(c0, s0, xs):
    import jax
    import jax.numpy as jnp

    def scan(c0, s0, xs):
        def cell(carry, a):
            c, s = carry
            c2 = c * 2.0 + a
            return (c2, s + jnp.sum(c2)), c2
        (c, s), ys = jax.lax.scan(cell, (c0, s0), xs)
        return c, s, ys
    return jax.vmap(scan)(c0, s0, xs)


def _msl_lane_scalar_carry_2d(c0, s0, xs):
    import jax
    import jax.numpy as jnp

    def scan(c0, s0, xs):
        def cell(carry, a):
            c, s = carry
            c2 = c * 2.0 + a
            return (c2, s + jnp.sum(c2)), c2
        (c, s), ys = jax.lax.scan(cell, (c0, s0), xs)
        return c, s, ys
    return jax.vmap(jax.vmap(scan))(c0, s0, xs)


def _msl_lane_scalar_input(c0, s0, xs, zs):
    import jax
    import jax.numpy as jnp

    def scan(c0, s0, xs, zs):
        def cell(carry, x):
            a, z = x
            c, s = carry
            c2 = c * 2.0 + a
            return (c2, s + jnp.sum(c2) * z), c2
        (c, s), ys = jax.lax.scan(cell, (c0, s0), (xs, zs))
        return c, s, ys
    return jax.vmap(scan)(c0, s0, xs, zs)


def _msl_lane_scalar_input_2d(c0, s0, xs, zs):
    import jax
    import jax.numpy as jnp

    def scan(c0, s0, xs, zs):
        def cell(carry, x):
            a, z = x
            c, s = carry
            c2 = c * 2.0 + a
            return (c2, s + jnp.sum(c2) * z), c2
        (c, s), ys = jax.lax.scan(cell, (c0, s0), (xs, zs))
        return c, s, ys
    return jax.vmap(jax.vmap(scan))(c0, s0, xs, zs)


def _msl_lane_scalar_output_2d(c0, xs, zs):
    import jax
    import jax.numpy as jnp

    def scan(c0, xs, zs):
        def cell(c, x):
            a, z = x
            c2 = c * 2.0 + a
            return c2, jnp.sum(c2) * z
        return jax.lax.scan(cell, c0, (xs, zs))
    return jax.vmap(jax.vmap(scan))(c0, xs, zs)


def _msl_scalar_out_grad_2125c96(xs):
    """2125c96's cell -- a vmapped scan whose per-step output is one scalar
    per lane -- under value_and_grad.  The forward loop declined under the
    old guard; now that it plans, its AD forward pass is the loop that
    reached the rank-0 by-value input hole (a hoisted `sum(d)` broadcast to
    a register vector was declared and never loaded: NaN gradients,
    2026-09-03).  Transcendentals, so DOT tolerance."""
    import jax
    import jax.numpy as jnp
    d = np.array([0.3, -0.7], np.float32)
    c0 = (np.arange(4, dtype=np.float32) % 3) * 0.5 - 0.5

    def cell(c, a):
        b = jnp.cos(jnp.sum(jnp.sin(a)) + jnp.sum(jnp.cos(c)) + jnp.sum(d))
        return jnp.sin(c * b), b

    def loss(a):
        return jax.vmap(lambda v: jax.lax.scan(cell, jnp.asarray(c0), v)[1]
                        .sum())(a).sum()
    return jax.value_and_grad(loss)(xs)


def _carry_args():
    """Exact-in-f32 operands for the carry cases: small integers, 8 steps.

    Built rather than drawn, because the whole point is an EXACT comparison:
    `a` doubles each step, so the largest value any of these reaches is
    about 2**8 * 4 + 255 * 103 -- far below the 2**24 where f32 stops
    counting by ones.
    """
    a0 = (np.arange(32, dtype=np.float32).reshape(4, 8) % 5) - 2
    b0 = (np.arange(32, dtype=np.float32).reshape(4, 8) % 3) - 1
    xs = (np.arange(8 * 32, dtype=np.float32).reshape(8, 4, 8) % 7) - 3
    return a0, b0, xs


def _msl_chunked(h0, xs, us, wz, wh):
    """The 0.11.6 WEDGE SHAPE: a generated kernel traced into a CHUNKED
    replay.

    The outer scan is too long to unroll (trip 256 > kUnrollMax), so it runs
    eagerly and -- being pure and cheap -- replays kmax iterations per
    compiled graph; the inner scan is an msl cell, so each chunk graph holds
    a kernel MLX has not built yet.  Chunk 0 was the one compiled call
    nothing settled before submitting: under METALJAX_MSL_FORCE_BUILD_FAIL
    the build failed inside `mx::async_eval`, which abandons the per-stream
    events it had already attached, and the next blocking eval waited on one
    of them forever (`Event::wait`, no timeout).  `wzp`/`whp` are lazy
    captures AND results on purpose: they are what the failed walk visits
    first and what a caller reads back last.

    Driven by the "a forced kernel-build failure never wedges" contract, not
    by `_cases()`: what it pins is the wedge, and the shape's arithmetic is
    covered by the two "nested scan ..." cases below.  Those exist because
    this shape ALSO carried the nested-scan P0 -- an msl scan inside an outer
    scan read one timestep of the captured sequence for every step -- which
    was a collision between invariant loads keyed by source rather than by
    window, and is fixed.
    """
    import jax
    import jax.numpy as jnp

    wzp = jnp.tanh(wz) * 0.5
    whp = jnp.tanh(wh) * 0.5

    def outer(h, x):
        def cell(c, u):
            z = jax.nn.sigmoid(u * wzp)
            return z * c + (1.0 - z) * jnp.tanh(u * whp), None
        return jax.lax.scan(cell, h + x, us)[0], None

    return jax.lax.scan(outer, h0, xs)[0], wzp, whp


def _wedge_args():
    """The wedge shape's inputs, identical in every arm that computes it."""
    return [_rand((4, 16), 88), _rand((256, 4, 16), 89) * np.float32(0.1),
            _rand((12, 4, 16), 90) * np.float32(0.1), _rand((16,), 91),
            _rand((16,), 92)]


def run_wedge_probe(path):
    """Compute the wedge shape and write the answers (a child entry point)."""
    import jax

    out = jax.jit(_msl_chunked)(*_wedge_args())
    flat = [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]
    np.savez(path, **{f"a{i}": a for i, a in enumerate(flat)})


# --------------------------------------------------------------------------
# the cache-append scatter (METALJAX_SCATTER_APPEND)
# --------------------------------------------------------------------------
#
# A SET whose index batch holds exactly ONE coordinate writes exactly one
# contiguous window, and the lowering emits it as `mx::slice_update` instead of
# the dummy pad's concatenate/scatter/slice.  Two arms, and both are here: the
# index is either provably in bounds (nothing else needed) or it is not, and
# then the drop is a window-sized read-back.  A SET moves bits, so every case
# below is compared EXACTLY -- a fast path that changed a value would not be a
# tolerance question.


def _kv_append(cache, upd, end_index):
    """The decode KV-cache append, as gemma's sampler spells it.

    `cache.at[b, end_index % T].set(u)`: jnp's Python-semantics modulo, then
    jnp's negative-index normalization, then a two-component index vector --
    which is the exact `stablehlo.scatter` the row-2 decode program contains
    (3 per layer per token), and the shape the analysis has to bound to prove
    the drop can never fire.
    """
    import jax.numpy as jnp

    b = jnp.arange(cache.shape[0])[:, None]
    slot = (end_index % cache.shape[1])[:, None]
    return cache.at[b, slot].set(upd)


def _kv_append_vmap(cache, upd, end_index):
    """The same append written as a vmapped `dynamic_update_slice`.

    jax's batching rule turns that into a scatter under
    `GatherScatterMode.CLIP` -- an explicit `clamp(0, i, hi)` over the index
    vector, which is the other spelling the bound analysis has to read.
    """
    import jax

    T = cache.shape[1]
    return jax.vmap(lambda c, u, e: jax.lax.dynamic_update_slice(
        c, u, (e % T, 0, 0)))(cache, upd, end_index)


def _window_set(x, i, u):
    """One window SET at a traced index nothing bounds: the GUARDED arm.

    jnp normalizes a negative index (`i + n`) but does not clamp, so an index
    past either end stays out of range and XLA DROPS the update.  That is what
    the read-back guard has to reproduce exactly.
    """
    return x.at[i].set(u)


def _window_set_skewed(x, n, u):
    """`select(f < -5, f + 8, f)` over `f = rem(n, 8)`: the normalization's
    SHAPE without its rule, written as a RAW `lax.scatter` so nothing
    normalizes the index afterwards.

    The bound analysis reads the select's predicate.  A predicate that merely
    IMPLIES `f < 0` leaves the else-arm free to be negative -- at n = -3 the
    else-arm is taken and the index is -3 -- so claiming this in bounds would
    turn XLA's DROP into a write at the CLAMPED slot 0, which is a wrong
    answer rather than a slow one.  Measured: an `is_zero` relaxed to
    `hi <= 0` writes `[-1, -1, -1, -1]` into row 0 here.

    Two spelling details, both load-bearing: jnp's `.at[]` would normalize -3
    to 5 and hide the drop (hence the raw `lax.scatter`), and `jnp.where`
    lowers through jax's `@_where` helper, which puts the select's arms behind
    block arguments where the rule never fires (hence `lax.select`).
    """
    import jax
    import jax.numpy as jnp

    f = jax.lax.rem(n, jnp.int32(8))
    j = jax.lax.select(f < jnp.int32(-5), f + jnp.int32(8), f)
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))
    return jax.lax.scatter(x, j.reshape(1, 1), u.reshape(1, -1), dnums)


# A dynamic-while body of ~400 tape entries -- past `control.cc`'s 256-entry
# eager threshold, so a body that COMPILES is the only way this loop pipelines.
# Every step adds a small integer to an f32 that stays well under 2**24, which
# is what lets the case be compared EXACTLY: the trip count is then a fact
# about the loop, not a rounding accident.  (Each iteration adds 799; from 1.0
# the loop runs 7 times and stops at 5594.)
_DYNAMIC_BIG_STEPS = 400


def _dynamic_big_body(s):
    import jax.numpy as jnp

    v, n = s
    for k in range(_DYNAMIC_BIG_STEPS):
        v = v + jnp.float32(k % 3 + 1)
    return (v, n + 1)


# The KV in-place rewrite (metal_lowering.cc `kv::`): keras-hub's decode loop
# carries ONE stacked cache [B, layers, 2, T, H, D] and rebuilds it every
# token -- each layer slices its slab out, `dynamic_update_slice`s the new
# row in, reads attention off the result, and the body re-stacks the slabs.
# The rewrite turns that into a chain of window writes ON the carry.  These
# bodies spell the pattern the way jax lowers keras-hub's (slice + reshape,
# DUS, stack = broadcast_in_dim + concatenate), with the corners the rewrite
# has to get right: a wrapping position (`i % T`, a cache that fills and
# wraps), a layer that reads its slab BEFORE its own update (gemma4's
# `where(mask, new, dynamic_slice(slab))`), a layer that keeps its slab as
# it is (gemma4's KV-sharing layers), and a bf16 cache.
_KV_L, _KV_T, _KV_H, _KV_D, _KV_B = 3, 8, 2, 4, 1


def _kv_layer(cache, l, pos, x, wk, wv, pre_read, read_after=False):
    import jax
    import jax.numpy as jnp
    from jax import lax
    B, T, H, D = _KV_B, _KV_T, _KV_H, _KV_D
    slab = cache[:, l]                     # [B, 2, T, H, D]
    k, v = slab[:, 0], slab[:, 1]          # [B, T, H, D]
    new_k = (x @ wk).reshape(B, 1, H, D).astype(cache.dtype)
    new_v = (x @ wv).reshape(B, 1, H, D).astype(cache.dtype)
    if pre_read:
        old = lax.dynamic_slice(k, (0, pos, 0, 0), (B, 1, H, D))
        mask = (jnp.arange(D) % 2 == 0).reshape(1, 1, 1, D)
        new_k = jnp.where(mask, new_k, old)
    k2 = lax.dynamic_update_slice(k, new_k, (0, pos, 0, 0))
    v2 = lax.dynamic_update_slice(v, new_v, (0, pos, 0, 0))
    q = (x @ wk).reshape(B, 1, H, D)
    scores = jnp.einsum("bqhd,bthd->bhqt", q, k2.astype(jnp.float32))
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqt,bthd->bqhd", probs,
                     v2.astype(jnp.float32)).reshape(B, H * D)
    if read_after:
        # The slab BEFORE its update, read after it: an in-place write
        # would change this value, so the rewrite must decline the carry.
        out = out + jnp.sum(k.astype(jnp.float32)) * 1e-3
    return jnp.stack([k2, v2], axis=1), out


def _kv_body(s, wrap=True, identity_layer=True, read_after=False):
    import jax.numpy as jnp
    cache, x, i, acc, (wk, wv) = s
    pos = (i % _KV_T) if wrap else i
    parts = []
    for l in range(_KV_L):
        if identity_layer and l == _KV_L - 1:
            parts.append(cache[:, l])
            continue
        slab, out = _kv_layer(cache, l, pos, x, wk[l], wv[l],
                              pre_read=(l == 1), read_after=read_after)
        parts.append(slab)
        x = x + 0.5 * out
    acc = acc + jnp.sum(x) * 1e-3 + 1.0
    return (jnp.stack(parts, axis=1), x, i + 1, acc, (wk, wv))


def _kv_args(dtype, seed):
    import jax.numpy as jnp
    B, L, T, H, D = _KV_B, _KV_L, _KV_T, _KV_H, _KV_D
    return [np.zeros((B, L, 2, T, H, D), dtype),
            _rand((B, H * D), seed),
            np.int32(0), np.float32(0.0),
            (_rand((L, H * D, H * D), seed + 1) * np.float32(0.3),
             _rand((L, H * D, H * D), seed + 2) * np.float32(0.3))]


def _kv_dynamic(steps, **kw):
    """`steps` iterations behind a data-dependent stop (the pipelined
    dynamic while), the cache wrapping at T=8."""
    import jax
    from jax import lax

    def fn(cache, x, i, acc, ws):
        return lax.while_loop(lambda s: s[3] < (steps - 0.5),
                              lambda s: _kv_body(s, **kw),
                              (cache, x, i, acc, ws))
    return fn


def _kv_counted(steps, **kw):
    """The same body under fori_loop: the counted (chunked) path."""
    from jax import lax

    def fn(cache, x, i, acc, ws):
        return lax.fori_loop(0, steps,
                             lambda _t, s: _kv_body(s, **kw),
                             (cache, x, i, acc, ws))
    return fn


# The rotate-half rope apply as a view (metal_rope.cc): every spelling the
# model table applies its table with, at decode's [1, 1, H, D] and prefill's
# [1, L, H, D], eager at the top level and inside a scan body (the compiled
# path, the table computed from the position carry as keras does).  The
# contract `_p42_rope_view` pins that the rewrite FIRES, that it removes
# dispatches, and that every answer is bit-identical with
# METALJAX_ROPE_VIEW=0: only data movement changes.
_ROPE_H, _ROPE_D = 4, 32


def _rope_keras(x, cos, sin):
    """keras-hub `RotaryEmbedding._apply_rotary_pos_emb`: the pair stack."""
    import jax.numpy as jnp
    x1, x2 = jnp.split(x, 2, axis=-1)
    rot = jnp.stack((-x2, x1), axis=-2).reshape(x.shape)
    return x * cos + rot * sin


def _rope_concat(x, cos, sin):
    """maxtext / mlx-style `rotate_half`: the last-axis concatenate."""
    import jax.numpy as jnp
    x1, x2 = jnp.split(x, 2, axis=-1)
    rot = jnp.concatenate((-x2, x1), axis=-1)
    return x * cos + rot * sin


# Every fixture applies ONE table to two arrays (q and k), as every
# transformer does: with a single consumer MLX fuses the whole cos/sin table
# into the literal apply's kernel and the rewrite is count-neutral (its
# tables sit behind a reshape, which MLX does not fuse through); with two,
# the table is materialized on both arms and each apply's negate and two
# concatenate copies are the whole difference.
def _rope_top(name, form, dtype):
    import jax.numpy as jnp

    def fn(x, theta):
        cos = jnp.cos(theta).astype(dtype)
        sin = jnp.sin(theta).astype(dtype)
        q = x.astype(dtype)
        k = (x * 0.5).astype(dtype)
        return form(q, cos, sin), form(k, cos, sin)
    fn.__name__ = name
    return fn


def _rope_scan(name, form, dtype):
    import jax.numpy as jnp
    from jax import lax

    def fn(xs, theta):
        def body(carry, xt):
            p, acc = carry
            ang = theta * (1.0 + p.astype(jnp.float32))
            cos = jnp.cos(ang).astype(dtype)
            sin = jnp.sin(ang).astype(dtype)
            yq = form(xt.astype(dtype), cos, sin)
            yk = form((xt * 0.5).astype(dtype), cos, sin)
            # The carry keeps the loop honest (the position feeds the next
            # step's table; the trip count comes back exactly); the compared
            # outputs are the per-step applies themselves.  An f32 sum of
            # the bf16 applies is NOT compared: XLA:CPU keeps excess f32
            # precision across the fused multiply-add where the tape rounds
            # each op, and twelve such terms drift past the half band while
            # both arms of the plugin agree with each other bit for bit.
            return (p + 1, acc + 1.0), (yq, yk)
        init = (jnp.int32(0), jnp.float32(0.0))
        (p, acc), (yq, yk) = lax.scan(body, init, xs)
        return p, acc, yq, yk
    fn.__name__ = name
    return fn


def _rope_args(seed, steps=None, L=1):
    shape = (1, L, _ROPE_H, _ROPE_D)
    if steps is not None:
        shape = (steps,) + shape
    return [_rand(shape, seed) * np.float32(3.0),
            _rand((1, L, 1, _ROPE_D), seed + 1) * np.float32(2.0)]


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
    # bf16 contractions/backward passes: reordered accumulation at 2^-8 ULP.
    BF16DOT = (2e-2, 2e-2)
    # The fissioned bf16 weight gradient sums ~100 bf16-rounded products, and
    # the CPU rounds the scan carry to bf16 every step where a fused MLX
    # chain (and the kernel's f32 registers) do not: measured 6.25e-02 on an
    # O(1) element -- a few bf16 ULP -- and the METALJAX_MSL=0 arm computes
    # the SAME value, so the spread is the graph path's, not the kernel's.
    BF16GRAD = (5e-2, 1e-1)
    bf = lambda a: np.asarray(a).astype(jnp.bfloat16)  # noqa: E731

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

        # The MIDDLE-contracted operand (metal_lowering.cc's batched arm).
        # When an operand's contracted axes have free axes on BOTH sides, the
        # plain [B, K, N] merge is not a view and MLX copies the whole thing;
        # the batched arm reads it where it lies instead.  Both slots, because
        # jax puts the weight in whichever one the einsum's output names
        # first: `BTD,NDH` gives dot(x, w), `BSD,CKDH` gives dot(w, x).
        ("middle-contracted rhs (q einsum)",
         lambda x, w: jnp.einsum("BTD,NDH->BTNH", x, w),
         [_rand((1, 1, 24), 80), _rand((5, 24, 7), 81)], *DOT),
        ("middle-contracted lhs (kv einsum)",
         lambda x, w: jnp.einsum("BSD,CKDH->CBSKH", x, w),
         [_rand((1, 1, 24), 82), _rand((2, 3, 24, 7), 83)], *DOT),
        # M > 1: the rhs arm owes an output permute here, the lhs arm does not.
        ("middle-contracted rhs, M > 1",
         lambda x, w: jnp.einsum("BTD,NDH->BTNH", x, w),
         [_rand((2, 6, 24), 84), _rand((5, 24, 7), 85)], *DOT),
        ("middle-contracted lhs, M > 1",
         lambda x, w: jnp.einsum("BSD,CKDH->CBSKH", x, w),
         [_rand((3, 4, 24), 86), _rand((2, 3, 24, 7), 87)], *DOT),
        ("middle-contracted bf16",
         lambda x, w: jnp.einsum("BTD,NDH->BTNH", x, w),
         [bf(_rand((1, 1, 24), 88)), bf(_rand((5, 24, 7), 89))], *BF16DOT),
        ("middle-contracted lhs bf16",
         lambda x, w: jnp.einsum("BSD,CKDH->CBSKH", x, w),
         [bf(_rand((1, 1, 24), 90)), bf(_rand((2, 3, 24, 7), 91))], *BF16DOT),
        # A real batching dim in front of the middle-contracted axes.
        ("middle-contracted with batch dim",
         lambda x, w: jnp.einsum("GBD,GNDH->GBNH", x, w),
         [_rand((3, 1, 24), 92), _rand((3, 5, 24, 7), 93)], *DOT),
        # Two contracted axes, adjacent and in the middle: still one merge.
        ("middle-contracted, 2 contracting dims",
         lambda x, w: jnp.einsum("BTDE,NDEH->BTNH", x, w),
         [_rand((1, 1, 4, 6), 94), _rand((5, 4, 6, 7), 95)], *DOT),
        # ...and NON-adjacent, which the arm must decline (the weight cannot
        # be viewed as [G, K, Ntail] and the plain copy stands).
        ("split contracting dims decline",
         lambda x, w: jnp.einsum("BTDE,NDMEH->BTNMH", x, w),
         [_rand((1, 1, 4, 6), 96), _rand((5, 4, 3, 6, 7), 97)], *DOT),
        # The shapes that must keep the plain arm: contracted axes LEADING
        # (attn_vec) and LAST (gating) are views already, and a unit leading
        # free axis has nothing to batch over.
        ("leading-contracted stays plain",
         lambda x, w: jnp.einsum("BTNH,NHD->BTD", x, w),
         [_rand((1, 1, 5, 7), 98), _rand((5, 7, 24), 99)], *DOT),
        ("last-contracted stays plain",
         lambda x, w: jnp.einsum("BTF,NHF->BTNH", x, w),
         [_rand((1, 1, 24), 100), _rand((2, 9, 24), 101)], *DOT),
        ("middle-contracted, one group",
         lambda x, w: jnp.einsum("BTD,NDH->BTNH", x, w),
         [_rand((1, 1, 24), 102), _rand((1, 24, 7), 103)], *DOT),
        # Integer dots take the exact-f32-chunk arm, never the batched one.
        ("middle-contracted int32",
         lambda x, w: jnp.einsum("BTD,NDH->BTNH", x, w),
         [np.arange(-12, 12, dtype=np.int32).reshape(1, 1, 24),
          (np.arange(5 * 24 * 7, dtype=np.int32).reshape(5, 24, 7) % 9) - 4],
         *EXACT),

        # lax.ragged_dot: jax's dense fallback is the ragged-dot recognizer's
        # match (metal_ragged.cc -> one gather_mm); the CPU runs the dense
        # chain literally, so this compares the rewrite against its spec.
        ("ragged_dot f32",
         lambda x, w, gs: jax.lax.ragged_dot(x, w, gs),
         [_rand((10, 8), 60), _rand((4, 8, 6), 61),
          np.array([3, 2, 0, 5], np.int32)], *DOT),
        ("ragged_dot bf16",
         lambda x, w, gs: jax.lax.ragged_dot(
             x, w, gs, preferred_element_type=jnp.bfloat16),
         [bf(_rand((10, 8), 62)), bf(_rand((4, 8, 6), 63)),
          np.array([1, 4, 5, 0], np.int32)], *BF16DOT),
        # The maxtext idiom: rows padded to the ragged tiling, group sizes
        # covering only the real ones.  The pad is absorbed and the padded
        # rows must come back as the exact zeros the dense mask computes.
        ("ragged_dot padded rows",
         lambda x, w, gs: jax.lax.ragged_dot(
             jnp.pad(x, ((0, 6), (0, 0))), w, gs),
         [_rand((10, 8), 64), _rand((4, 8, 6), 65),
          np.array([2, 3, 4, 1], np.int32)], *DOT),
        # Group sizes that cover only SOME rows: the uncovered tail is zero.
        ("ragged_dot uncovered tail",
         lambda x, w, gs: jax.lax.ragged_dot(x, w, gs),
         [_rand((10, 8), 66), _rand((3, 8, 6), 67),
          np.array([2, 0, 3], np.int32)], *DOT),
        # bf16 inputs accumulating to f32: the recognizer must DECLINE (the
        # gather would round the output to bf16) and the dense chain run.
        ("ragged_dot mixed dtypes declines",
         lambda x, w, gs: jax.lax.ragged_dot(
             x, w, gs, preferred_element_type=jnp.float32),
         [bf(_rand((6, 8), 68)), bf(_rand((4, 8, 6), 69)),
          np.array([1, 2, 2, 1], np.int32)], *HALF),
        # The ragged contraction WITHOUT the mask chain: not a match, and the
        # canonicalized contracting-pair order must still contract correctly
        # in both listed orders.
        ("dot 2 contracting dims (k,g)",
         lambda a, b: jax.lax.dot_general(a, b, (((2, 0), (1, 0)), ((), ()))),
         [_rand((3, 4, 5), 70), _rand((3, 5, 6), 71)], *DOT),
        ("dot 2 contracting dims (g,k)",
         lambda a, b: jax.lax.dot_general(a, b, (((0, 2), (0, 1)), ((), ()))),
         [_rand((3, 4, 5), 72), _rand((3, 5, 6), 73)], *DOT),
        # The maxtext scan-over-layers form: ragged_dot inside lax.scan over
        # a transposed weight stack.  The recognizer's STACKED extension
        # absorbs the per-layer dynamic-slice copy and gathers matrix
        # `group * L + layer` straight out of the original buffer.
        ("ragged_dot scanned stack",
         lambda x, Wt, gs: jax.lax.scan(
             lambda h, w: (jax.lax.ragged_dot(
                 h, w, gs, preferred_element_type=h.dtype), None),
             x, jnp.transpose(Wt, (1, 0, 2, 3)))[0],
         [_rand((10, 8), 75), _rand((4, 5, 8, 8), 76) * 0.1,
          np.array([3, 2, 0, 5], np.int32)], *DOT),
        # dynamic_index_in_dim of a transposed stack: the trailing reshape
        # only drops the unit axis and must lower as a view-preserving
        # squeeze (ops_shape.cc kReshape), not a copy -- and stay correct.
        ("unit reshape of strided slice",
         lambda a, i: jax.lax.dynamic_index_in_dim(
             jnp.transpose(a, (1, 0, 2)), i, 0, keepdims=False) * 2.0,
         [_rand((4, 5, 3), 74), np.array(2, np.int32)], *F32),

        # The stacked-weight dot (metal_stacked.cc): jax's scanned-layer
        # weight read -- dot(x, dynamic_index_in_dim(stack, i)) over a
        # mid-axis layer stack -- becomes one gather_mm reading an [L, K, N]
        # strided view of the original buffer.  The CPU runs the slice chain
        # literally, so these compare the rewrite against its spec.
        ("stacked dot scan (mid-axis stack)",
         lambda x, W: jax.lax.scan(
             lambda h, w: (h @ w, None), x,
             jnp.transpose(W, (1, 0, 2)))[0],
         [_rand((2, 128), 80), _rand((128, 4, 128), 81) * 0.1], *DOT),
        ("stacked dot scan bf16",
         lambda x, W: jax.lax.scan(
             lambda h, w: (h @ w, None), x,
             jnp.transpose(W, (1, 0, 2)))[0],
         [bf(_rand((2, 128), 82)), bf(_rand((128, 4, 128), 83) * 0.1)],
         *BF16DOT),
        # The inline spelling (no scan, no helper call), with multi-axis
        # free dims that must collapse in the strided view.
        ("stacked dot inline multi-axis free",
         lambda x, W, i: jnp.einsum(
             "bk,kcd->bcd", x, jax.lax.dynamic_index_in_dim(
                 jnp.transpose(W, (1, 0, 2, 3)), i, 0, keepdims=False)),
         [_rand((3, 128), 84), _rand((128, 4, 2, 64), 85),
          np.array(2, np.int32)], *DOT),
        # ...and slicing the middle axis directly, no transpose at all.
        ("stacked dot inline mid-axis slice",
         lambda x, W, i: x @ jax.lax.dynamic_index_in_dim(
             W, i, 1, keepdims=False),
         [_rand((3, 130), 86), _rand((130, 4, 128), 87),
          np.array(1, np.int32)], *DOT),
        # An out-of-range layer index: dynamic_slice clamps, and so must the
        # gather.
        ("stacked dot clamped index",
         lambda x, W, i: jnp.einsum(
             "bk,kn->bn", x, jax.lax.dynamic_index_in_dim(
                 jnp.transpose(W, (1, 0, 2)), i, 0, keepdims=False)),
         [_rand((2, 128), 88), _rand((128, 4, 128), 89),
          np.array(9, np.int32)], *DOT),
        # Contracted axes that straddle the stack axis (maxtext's o-proj
        # layout): no view of the ORIGINAL buffer exists.  At M = 3 the
        # RELAYOUT form (B3) declines too -- it is decode-only, M = 1, where
        # `_p41_stacked_relayout` pins it -- so the slice chain runs here
        # and the answer is the CPU's either way.
        ("stacked dot o-proj layout (M=3 declines)",
         lambda x, W, i: jax.lax.dot_general(
             x, jax.lax.dynamic_index_in_dim(
                 jnp.transpose(W, (1, 0, 2, 3)), i, 0, keepdims=False),
             (((1, 2), (0, 1)), ((), ()))),
         # Small magnitudes: the DECLINE is the point, and the fallback's
         # canonicalized pair order reduces K = 512 in a different order
         # than the CPU, which at unit variance exceeds DOT's per-element
         # atol on f32.
         [_rand((3, 8, 64), 90) * 0.1, _rand((8, 4, 64, 64), 91) * 0.1,
          np.array(3, np.int32)], *DOT),

        # P30: the tape post-passes (metal_lowering.cc CsePass /
        # FoldConstants / DcePass).  The CPU backend runs the literal chains,
        # so these compare the merged/folded tape against its spec.
        ("cse: repeated subexpression",
         lambda x: jnp.sin(x) * jnp.cos(x) + jnp.sin(x) - jnp.cos(x),
         [_rand((4, 5), 92)], *F32),
        ("cse: duplicate embedded constants",
         lambda x: (x + np.linspace(0, 1, 12, dtype=f).reshape(3, 4))
         * (np.linspace(0, 1, 12, dtype=f).reshape(3, 4) + 1.0),
         [_rand((3, 4), 93)], *F32),
        # The maxtext decode shape: a loop body that rebuilds an
        # argument-independent table from iota and gathers one row by a
        # dynamic index.  The fold pass evaluates the table once at lower
        # time; the gather stays dynamic.
        ("fold: invariant table in a scan body",
         lambda h, idx: jax.lax.scan(
             lambda c, i: (c + jnp.take(
                 jnp.sin(jnp.arange(64.0, dtype=f).reshape(8, 8) * 0.1) ** 2,
                 i, axis=0), None),
             h, idx)[0],
         [_rand((8,), 94), np.array([0, 3, 7, 2, 5], np.int32)], *F32),
        # ...the rope spelling: a complex frequency table, one row applied
        # per step (complex fold path: make_complex, complex exp, real/imag).
        ("fold: complex rope table in a scan body",
         lambda h, idx: jax.lax.scan(
             lambda c, i: (c * jnp.real(jnp.take(
                 jnp.exp(1j * (jnp.arange(64.0, dtype=f)[:, None]
                               / (10000.0 ** (jnp.arange(8.0, dtype=f)
                                              / 8.0))[None, :])),
                 i, axis=0)) + 0.1, None),
             h, idx)[0],
         [_rand((8,), 95), np.array([1, 9, 33], np.int32)], *F32),
        # A rank-0 f32 whose value does not round-trip through %.7g: the
        # folded payload must take the one-element-buffer form so a compiled
        # graph cannot bake it as a lossy literal.
        ("fold: rank-0 f32 in a loop body",
         lambda h, idx: jax.lax.scan(
             lambda c, i: (c * (jnp.sin(jnp.float32(0.3)) + 1.5)
                           + i.astype(f), None), h, idx)[0],
         [_rand((6,), 96), np.arange(4, dtype=np.int32)], *F32),
        # A body whose CARRY is purely constant: the payload crosses the loop
        # and reaches @main's outputs, where the no-alias copy rule must see
        # the constant-view taint behind it.
        ("fold: constant carry crosses the loop",
         lambda h, idx: jax.lax.scan(
             lambda c, i: (jnp.cos(jnp.arange(6.0, dtype=f)) * 2.0, None),
             h, idx)[0],
         [_rand((6,), 97), np.arange(3, dtype=np.int32)], *F32),
        # A multi-result const entry (top_k) with only one result read across
        # the fold boundary.
        ("fold: multi-result const top_k in a body",
         lambda h, idx: jax.lax.scan(
             lambda c, i: (c + jax.lax.top_k(
                 jnp.sin(jnp.arange(16.0, dtype=f)), 4)[0][i % 4], None),
             h, idx)[0],
         [_rand((2,), 98), np.arange(3, dtype=np.int32)], *F32),

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
        # The same arm with a body too big for the eager entry-count rule
        # (`_dynamic_big_body` is ~400 entries against control.cc's 256), and
        # therefore the first case that runs a COMPILED body through the
        # pipelined dynamic loop -- LLM decode's shape, where the whole model
        # is one graph replayed per token behind a data-dependent stop.
        #
        # It is a trip-count test, not an arithmetic one.  Speculation builds
        # iteration t+1 before iteration t's condition is read, so a bug that
        # committed the mis-speculated last one would return one iteration too
        # many; the carry reports BOTH the value and the count, and every
        # quantity in it is a small integer in f32, so EXACT means the trip
        # count is pinned exactly.
        ("while_loop (dynamic trip, big compiled body)",
         lambda x: jax.lax.while_loop(
             lambda s: s[0] < 5000.0, _dynamic_big_body, (x, jnp.int32(0))),
         [np.float32(1.0)], *EXACT),
        # The KV in-place rewrite (see `_kv_body`): every case here must match
        # the CPU AND -- the contract `_p39_kv_inplace` checks -- be
        # bit-identical with METALJAX_KV_INPLACE=0, since only data movement
        # changes.  The band is the attention's contractions', not the
        # cache's: the cache holds matmul rows.
        ("kv cache in place (dynamic loop, wraps)",
         _kv_dynamic(12), _kv_args(np.float32, 700), *DOT),
        # fori_loop: jax wraps the body in a `func.call @closed_call`, and the
        # rewrite matches a rebuild INSIDE the while body's own block (keras-
        # hub's generate loop is a while_loop with an inline body), so this
        # case runs the literal tape -- a differential row for the counted
        # path either way, and the contract pins that it is NOT rewritten.
        ("kv cache in place (counted loop, fills)",
         _kv_counted(8, wrap=False), _kv_args(np.float32, 710), *DOT),
        # The rotate-half rope apply as a view (see `_rope_keras`): each row
        # must match the CPU AND -- the contract `_p42_rope_view` checks --
        # be bit-identical with METALJAX_ROPE_VIEW=0.  bf16 rows carry the
        # half band because XLA:CPU keeps excess f32 precision across the
        # fused multiply-add where the tape rounds each op (both arms of the
        # plugin agree with each other exactly).
        ("rope view (keras pair-stack, bf16)",
         _rope_top("rope_keras_bf16", _rope_keras, jnp.bfloat16),
         _rope_args(800), *HALF),
        ("rope view (last-axis concat, f32)",
         _rope_top("rope_concat_f32", _rope_concat, jnp.float32),
         _rope_args(810), *F32),
        ("rope view (prefill shape, bf16)",
         _rope_top("rope_prefill_bf16", _rope_keras, jnp.bfloat16),
         _rope_args(820, L=5), *HALF),
        ("rope view (scan body, bf16)",
         _rope_scan("rope_scan_bf16", _rope_keras, jnp.bfloat16),
         _rope_args(830, steps=6), *HALF),
        ("kv cache in place (bf16, no identity layer)",
         _kv_dynamic(10, identity_layer=False),
         _kv_args(ml_dtypes.bfloat16, 720), *BF16DOT),
        ("kv cache in place (slab read after update: declined)",
         _kv_dynamic(6, read_after=True), _kv_args(np.float32, 730), *DOT),
        # B4 (submit-ahead): the loop stops after 6 steps with T=8 rows and
        # no wrap, so rows 6 and 7 of every slab must come back ZERO.  The
        # pipelined loop builds -- and, from the third step on, SUBMITS --
        # iteration 7 before it reads iteration 6's condition; that
        # iteration's update writes row 6.  It runs on the device and is
        # dropped, and the only thing that keeps its write out of the
        # returned cache is that the loop holds the carry across the
        # submission (a held array is never donated, so the chain's root
        # copies).  A speculation that donated would return row 6 filled.
        ("kv cache in place (dynamic loop, no wrap: rows past the stop stay zero)",
         _kv_dynamic(6, wrap=False), _kv_args(np.float32, 740), *DOT),
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

        # THE NESTED-SCAN P0.  An inner counted scan over a sequence CAPTURED
        # from the enclosing scope, inside an outer scan: the msl analyzer
        # unrolls the inner loop symbolically into the outer loop's kernel,
        # which turns `us[j]` into one loop-INVARIANT load per unrolled step,
        # each addressing a different window of the same buffer.  Invariant
        # loads used to be keyed by SOURCE alone, so all twelve collapsed onto
        # one window and every step read the same timestep -- full-magnitude
        # wrong answers, no error, and only when nested (a standalone msl scan
        # was always bit-exact, which is what hid it).
        #
        # The first form is the decodable one and is deliberately EXACT: with
        # us[t] = 2**t the carry is literally the bitmask of the timesteps the
        # loop read, so a stuck index shows up as the wrong POWER OF TWO and
        # not as drift.  Correct = 3 * (2**12 - 1) = 12285; the unfixed engine
        # returned 36, having read us[0] twelve times per outer iteration.
        ("nested scan reads every timestep",
         lambda h0, xs, us: jax.lax.scan(
             lambda h, x: (jax.lax.scan(lambda c, u: (c + u, None),
                                        h + x, us)[0], None),
             h0, xs)[0],
         [np.zeros((2, 8), f), np.zeros((3, 2, 8), f),
          np.tile((f(2.0) ** np.arange(12, dtype=f))[:, None, None],
                  (1, 2, 8))], *EXACT),
        # ...and the shape it was found in: a gated cell over a captured
        # sequence, whose wrongness was a plausible-looking 4.2e-2 on values
        # of scale 6.3e-2.  Wrong on the FIRST outer iteration already, so one
        # outer step is enough to catch it.
        ("nested scan over a captured sequence",
         lambda h0, xs, us, wz, wh: jax.lax.scan(
             lambda h, x: (jax.lax.scan(
                 lambda c, u: (jax.nn.sigmoid(u * wz) * c
                               + (1.0 - jax.nn.sigmoid(u * wz))
                               * jnp.tanh(u * wh), None),
                 h + x, us)[0], None),
             h0, xs)[0],
         [_rand((4, 16), 93), _rand((2, 4, 16), 94) * f(0.1),
          _rand((12, 4, 16), 95) * f(0.1), _rand((16,), 96) * f(0.3),
          _rand((16,), 97) * f(0.3)], *F32),
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
        # A trip that is NOT a multiple of kmax (90 = 5 x 16 + 10): the ten
        # single-step replays of the remainder lead the five chunks and each
        # one is submitted (control.cc `run_chunked`, T4).  Long enough not
        # to unroll into the main (kUnrollMax = 64), and the batch mean is a
        # cross-lane reduce msl_scan declines -- the matmul body above is a
        # coop cell, and a generated kernel replaces the loop that would
        # otherwise be chunked.  The chunk-plan contract below pins the
        # schedule and its bit-exactness across K; this row is the plain CPU
        # comparison.
        ("chunked replay with a remainder (90 x matmul body)",
         lambda h0, w, xs: jax.lax.scan(
             lambda h, x: (jnp.tanh(h @ w * 0.5 + x)
                           - jnp.mean(h, axis=0, keepdims=True), None),
             h0, xs)[0],
         [_rand((4, 8), 56), _rand((8, 8), 57), _rand((90, 4, 8), 58)], *DOT),
        # The same remainder loop with a STACKED OUTPUT: scan's `ys` is a
        # carried accumulator written one slab per step by a
        # dynamic_update_slice chain (metal_lowering.cc `AccumulatorBytes`).
        # 90 x [4, 4096] f32 = 5.6 MB of accumulator, big enough for the
        # chunk-plan contract below to see it in the gate narration and to
        # veto the byte bound on it with a small METALJAX_CHUNK_ACC_MB.  The
        # slab is tanh(h) @ w2, bounded: the carry itself grows to ~90 here
        # and h @ w2 cancelled to 1.5e-4 absolute on small elements.
        ("chunked replay with a stacked output (90 x matmul body)",
         lambda h0, w, w2, xs: jax.lax.scan(
             lambda h, x: (jnp.tanh(h @ w * 0.5 + x)
                           - jnp.mean(h, axis=0, keepdims=True),
                           jnp.tanh(h) @ w2),
             h0, xs)[1],
         [_rand((4, 8), 56), _rand((8, 8), 57), _rand((8, 4096), 59),
          _rand((90, 4, 8), 58)], *DOT),
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
        # Square F=4 cells: vector mode until 2026-08-26, when the coop
        # flip's bound came down to 4 (measured 1.8-2.4x on the topconfs16k
        # tc038/tc044/tc046 regressions).  The "msl coop flip at F=4"
        # contract below asserts the mode; these rows pin the answer.
        ("msl square F=4 cell (coop flip)", _msl_rnn,
         [_rand((4, 4), 68), _rand((16, 4, 4), 69),
          _rand((4, 4), 70) * f(0.3)], *DOT),
        ("msl square F=4 cell, weight grad", _msl_rnn_grad,
         [_rand((4, 4), 71), _rand((16, 4, 4), 72),
          _rand((4, 4), 73) * f(0.3)], *DOT),
        # ...and the rectangular cell that CANNOT flip (10 % 4 != 0), which
        # is where the census's vector coverage now lives.
        ("msl vector matvec cell (rect dot)", _msl_rnn_rect,
         [_rand((4, 4), 940), _rand((16, 4, 10), 941),
          _rand((4, 4), 942) * f(0.3), _rand((10, 4), 943) * f(0.3)], *DOT),
        ("msl vector cell, weight grads (rect dot)", _msl_rnn_rect_grad,
         [_rand((4, 4), 944), _rand((16, 4, 10), 945),
          _rand((4, 4), 946) * f(0.3), _rand((10, 4), 947) * f(0.3)], *DOT),
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

        # Carries that are NOT the loop counter (2026-09-02).  See the
        # helpers: every i32/i64 SCALAR carry used to be seeded as the
        # induction variable, so an invariant integer carry read back inside
        # the kernel as 0, 1, 2, ...  All five are EXACT by construction, and
        # the "only the induction variable is an msl counter" contract below
        # asserts that the msl path really claims them -- without it these
        # would pass by falling back to the interpreted loop.
        ("msl invariant int32 carry", _msl_invariant_int,
         [_carry_args()[0], np.int32(7), _carry_args()[2]], *EXACT),
        ("msl invariant int32 carry, first position",
         _msl_invariant_int_first,
         [np.int32(-3), _carry_args()[0], _carry_args()[1],
          _carry_args()[2]], *EXACT),
        ("msl invariant float carry", _msl_invariant_float,
         [_carry_args()[0], np.float32(2.5), _carry_args()[2]], *EXACT),
        ("msl counter-lookalike carry (+1, not in the cond)",
         _msl_counter_lookalike,
         [_carry_args()[0], np.int32(100), _carry_args()[1],
          _carry_args()[2]], *EXACT),
        ("msl incremented int32 carry (+3)", _msl_strided_int_carry,
         [_carry_args()[0], np.int32(5), _carry_args()[2]], *EXACT),

        # One scalar per lane (2026-09-02); see the helpers.  Lane spaces in
        # this order: 4 / 3,4 / 4 / 4,4 / 3,4 -- the contract reads them back.
        ("msl lane scalar state, 1-D lane", _msl_lane_scalar_carry,
         [_lane_ints((4, 8), 1), _lane_ints((4,), 2),
          _lane_ints((4, 6, 8), 3)], *EXACT),
        ("msl lane scalar state, 2-D lane (3,4)", _msl_lane_scalar_carry_2d,
         [_lane_ints((3, 4, 8), 4), _lane_ints((3, 4), 5),
          _lane_ints((3, 4, 6, 8), 6)], *EXACT),
        ("msl lane scalar input read bare, 1-D lane (F == B)",
         _msl_lane_scalar_input,
         [_lane_ints((4, 4), 7), _lane_ints((4,), 8),
          _lane_ints((4, 6, 4), 9), _lane_ints((4, 6), 10)], *EXACT),
        ("msl lane scalar input read bare, 2-D lane (4,4) (F == B)",
         _msl_lane_scalar_input_2d,
         [_lane_ints((4, 4, 4), 11), _lane_ints((4, 4), 12),
          _lane_ints((4, 4, 6, 4), 13), _lane_ints((4, 4, 6), 14)], *EXACT),
        ("msl lane scalar stacked expression, 2-D lane (3,4)",
         _msl_lane_scalar_output_2d,
         [_lane_ints((3, 4, 8), 15), _lane_ints((3, 4, 6, 8), 16),
          _lane_ints((3, 4, 6), 17)], *EXACT),
        ("msl per-lane scalar stacked output, grad (2125c96 cell)",
         _msl_scalar_out_grad_2125c96,
         [_rand((7, 5, 3), 18) * f(0.5)], *DOT),

        # msl in bf16 (the topconfs16k cliff, 2026-08-22): the dtype table
        # maps bf16 to MLX's bfloat16_t, so all three modes must plan and
        # agree with the CPU's bf16 arithmetic; the "bf16 msl plans build"
        # contract below reads the census.  Forward passes round per op on
        # both engines (HALF); the fissioned backward passes contract the
        # weight gradient in a different order, and one bf16 ULP is 2^-8, so
        # their band is wider.
        ("msl bf16 affine cell (mingru)", _msl_mingru,
         [bf(_rand((4, 16), 460)), bf(_rand((24, 4, 16), 461)),
          bf(_rand((16,), 462)), bf(_rand((16,), 463))], *HALF),
        ("msl bf16 affine cell, backward", _msl_mingru_grad,
         [bf(_rand((4, 16), 464)), bf(_rand((24, 4, 16), 465)),
          bf(_rand((16,), 466)), bf(_rand((16,), 467))], *BF16DOT),
        ("msl bf16 square F=4 cell (coop flip)", _msl_rnn,
         [bf(_rand((4, 4), 468)), bf(_rand((16, 4, 4), 469)),
          bf(_rand((4, 4), 470) * f(0.3))], *BF16DOT),
        ("msl bf16 vector matvec cell (rect dot)", _msl_rnn_rect,
         [bf(_rand((4, 4), 948)), bf(_rand((16, 4, 10), 949)),
          bf(_rand((4, 4), 950) * f(0.3)), bf(_rand((10, 4), 951) * f(0.3))],
         *BF16DOT),
        ("msl bf16 coop cell, weight grad", _msl_rnn_grad,
         [bf(_rand((8, 32), 471)), bf(_rand((12, 8, 32), 472)),
          bf(_rand((32, 32), 473) * f(0.1))], *BF16GRAD),

        # The same decision, one size up: 200 iterations still fit the OP
        # budget, but the executor refuses to unroll more than 64 into one
        # trace -- and since the topconfs16k cascade fix the LOWERING carries
        # the same bound (metal_lowering.cc kUnrollMax), so the loop is never
        # called traceable and main takes the eager loop path outright
        # instead of compiling, failing at trace time and retiring its
        # compiled path.  What this row checks is that the answer survives
        # that route.
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
        # CONTENDED 16-bit scatter-add (the topconfs16k bf16 residual,
        # 2026-08-22): MLX's Metal atomics are f32-only, and the emulated
        # 16-bit path is ~33x slower under contention -- an embedding
        # backward is exactly that shape.  The engine accumulates these in
        # f32 with one rounding at the end (order-nondeterministic on GPU
        # anyway), so the CPU's per-update bf16 rounding differs by a few
        # ULP of the colliding sums; positive updates keep that relative.
        ("scatter add bf16 (contended)", lambda x, i, u: x.at[i].add(u),
         [bf(np.zeros((4, 3), f)), np.arange(64, dtype=np.int32) % 4,
          bf(np.abs(_rand((64, 3), 60)))], *BF16DOT),
        ("scatter add f16 (contended)", lambda x, i, u: x.at[i].add(u),
         [np.zeros((4, 3), np.float16), np.arange(64, dtype=np.int32) % 4,
          np.abs(_rand((64, 3), 61)).astype(np.float16)], *BF16DOT),
        ("embedding gradient bf16 (contended scatter-add through AD)",
         lambda e, t: jax.grad(lambda a: (a[t] ** 2).sum())(e),
         [bf(np.arange(24, dtype=f).reshape(6, 4) / 24.0),
          np.arange(64, dtype=np.int32) % 6], *BF16DOT),
        # Empty updates: the handler returns the OPERAND array, so the tape
        # aliases the slot and the output-copy rule has to notice.
        ("scatter with empty updates", lambda x, i, u: x.at[i].add(u),
         [np.arange(4, dtype=f), np.zeros((0,), np.int32),
          np.zeros((0,), f)], *EXACT),

        # the cache-append fast path (METALJAX_SCATTER_APPEND).  Run the whole
        # suite at 0, 1 and 2 to cover the three lowerings of these rows: the
        # dummy pad, the provably-in-bounds `slice_update`, and the guarded
        # one.  EXACT throughout -- a SET copies bits.
        ("kv cache append (bf16, first slot)", _kv_append,
         [bf(_rand((1, 16, 4, 8), 71)), bf(_rand((1, 1, 4, 8), 72)),
          np.array([0], np.int32)], *EXACT),
        ("kv cache append (bf16, last slot)", _kv_append,
         [bf(_rand((1, 16, 4, 8), 71)), bf(_rand((1, 1, 4, 8), 72)),
          np.array([15], np.int32)], *EXACT),
        # end_index past the cache: the modulo wraps it, which is the whole
        # reason the append is in bounds at all.
        ("kv cache append (bf16, wrapped)", _kv_append,
         [bf(_rand((1, 16, 4, 8), 71)), bf(_rand((1, 1, 4, 8), 72)),
          np.array([16 * 5 + 3], np.int32)], *EXACT),
        ("kv cache append (f32)", _kv_append,
         [_rand((1, 16, 4, 8), 73), _rand((1, 1, 4, 8), 74),
          np.array([7], np.int32)], *EXACT),
        # Batch 3: THREE update windows, so the fast path declines and the
        # dummy pad runs -- the row that says the batch-size guard holds.
        ("kv cache append (batch 3, declines)", _kv_append,
         [_rand((3, 16, 4, 8), 75), _rand((3, 1, 4, 8), 76),
          np.array([2, 15, 33], np.int32)], *EXACT),
        ("kv cache append (vmapped dynamic_update_slice)", _kv_append_vmap,
         [bf(_rand((1, 16, 4, 8), 77)), bf(_rand((1, 1, 4, 8), 78)),
          np.array([9], np.int32)], *EXACT),
        # The guarded arm: nothing in the graph bounds these indices.
        ("window set at a traced index", _window_set,
         [np.arange(32, dtype=f).reshape(8, 4), np.int32(3),
          np.full(4, -1.0, f)], *EXACT),
        ("window set at a normalized negative index", _window_set,
         [np.arange(32, dtype=f).reshape(8, 4), np.int32(-3),
          np.full(4, -1.0, f)], *EXACT),
        ("window set past the end (must DROP)", _window_set,
         [np.arange(32, dtype=f).reshape(8, 4), np.int32(11),
          np.full(4, -1.0, f)], *EXACT),
        ("window set far below zero (must DROP)", _window_set,
         [np.arange(32, dtype=f).reshape(8, 4), np.int32(-100),
          np.full(4, -1.0, f)], *EXACT),
        ("window set bf16 past the end (must DROP)", _window_set,
         [bf(_rand((8, 4), 79)), np.int32(9), bf(np.full(4, -1.0, f))],
         *EXACT),
        # A window that does not span the un-indexed axes: the update starts
        # at 0 on those, which is XLA's rule and MLX's alike.
        ("window set of a partial row", lambda x, i, u: x.at[i, 1:3].set(u),
         [np.arange(32, dtype=f).reshape(8, 4), np.int32(5),
          np.full(2, -1.0, f)], *EXACT),
        ("window set of a partial row (out of bounds)",
         lambda x, i, u: x.at[i, 1:3].set(u),
         [np.arange(32, dtype=f).reshape(8, 4), np.int32(12),
          np.full(2, -1.0, f)], *EXACT),
        # A single-window SET of a COMPLEX operand: the handler writes the two
        # parts, so the fast path has to survive being called twice.
        ("window set complex64", _window_set,
         [_crand((8, 4), 80), np.int32(6),
          np.full(4, -1.0 - 2.0j, np.complex64)], *EXACT),
        ("window set complex64 (out of bounds)", _window_set,
         [_crand((8, 4), 80), np.int32(19),
          np.full(4, -1.0 - 2.0j, np.complex64)], *EXACT),
        # int32, where a one-ULP story could not hide anything.
        ("window set int32", _window_set,
         [np.arange(32, dtype=np.int32).reshape(8, 4), np.int32(2),
          np.full(4, -7, np.int32)], *EXACT),
        # The normalization's shape under a predicate that only IMPLIES
        # negativity: must not be claimed in bounds.  n = -3 takes the
        # ELSE arm and lands at -3, which XLA drops; n = -7 takes the then
        # arm and lands at 1, which it writes.
        ("raw scatter under a skewed normalization (drops)",
         _window_set_skewed,
         [np.arange(32, dtype=f).reshape(8, 4), np.int32(-3),
          np.full(4, -1.0, f)], *EXACT),
        ("raw scatter under a skewed normalization (writes)",
         _window_set_skewed,
         [np.arange(32, dtype=f).reshape(8, 4), np.int32(-7),
          np.full(4, -1.0, f)], *EXACT),

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

    # --- the two keras-hub decode spellings, transcribed ----------------
    #
    # Both are `keras_hub 0.30.0` as the model_bench rows 7 and 3 run it, cut
    # down to tiny dims.  They are written out rather than imported so the
    # suite does not depend on keras-hub, and each line is the line it copies:
    # the divergences from the qwen3/llama spelling above (which has fused
    # since 0.2.x) are what these rows exist to cover.

    def attn_gptoss(q, kk, vv, sinks, mslide, mcausal, scale, groups):
        """`gpt_oss_attention.py::_compute_attention` (row 7).

        Four things differ from `attn` above: the KV heads are lifted by an
        explicit `ops.repeat`; the mask is a `where` onto a "large negative
        number" (`-1e4`, which bf16 rounds to -9984) rather than `finfo.min`;
        a learned per-head SINK logit is concatenated onto the key axis,
        softmaxed, and sliced back off; and the softmax is preceded by its own
        explicit max-subtract, so the graph carries TWO max reductions.  A
        sliding-window layer (every other one) masks TWICE.
        """
        k = jnp.repeat(kk, groups, axis=2)
        v = jnp.repeat(vv, groups, axis=2)
        logits = jnp.einsum("bquh,bkuh->buqk", q, k)
        logits = logits * jnp.asarray(scale, logits.dtype)
        adder = jnp.asarray(-1e4, logits.dtype)
        if mslide is not None:
            logits = jnp.where(mslide[None, None, :, :], logits, adder)
        logits = jnp.where(mcausal[None, None, :, :], logits, adder)
        s = jnp.broadcast_to(sinks.reshape(1, -1, 1, 1),
                             logits.shape[:3] + (1,))
        combined = jnp.concatenate([logits, s], axis=-1)
        combined = combined - jnp.max(combined, axis=-1, keepdims=True)
        probs = jax.nn.softmax(combined, axis=-1)[..., :-1]
        return jnp.einsum("buqk,bkuh->bquh", probs.astype(v.dtype), v)

    def attn_gemma4(q, k, v, mask):
        """`gemma4_attention.py::_compute_attention` (row 3).

        Grouped query attention WITHOUT a repeat -- a 5-D `(b, kv, g, t, s)`
        layout and two `ops.matmul`s; no scale at all (`query_normalization =
        1.0`, Q/K norm replaces it); and `keras.layers.Softmax(dtype="f32",
        mask=)`, which upcasts, selects `-1e9` onto the logits, softmaxes, and
        then selects 0 onto the PROBABILITIES with the same predicate.
        """
        b, t, n, h = q.shape
        kvh = k.shape[2]
        g = n // kvh
        qt = jnp.transpose(q.reshape(b, t, kvh, g, h), (0, 2, 3, 1, 4))
        ke = jnp.expand_dims(jnp.transpose(k, (0, 2, 3, 1)), 2)
        logits = jnp.matmul(qt, ke).astype(jnp.float32)
        m = mask[:, None, None, :, :]
        logits = jnp.where(m, logits, jnp.asarray(-1e9, jnp.float32))
        p = jax.nn.softmax(logits, axis=-1)
        p = jnp.where(m, p, jnp.asarray(0.0, jnp.float32)).astype(v.dtype)
        ve = jnp.expand_dims(jnp.transpose(v, (0, 2, 1, 3)), 2)
        r = jnp.transpose(jnp.matmul(p, ve), (0, 3, 1, 2, 4))
        return r.reshape(b, t, n, h)

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

    # --- the keras decode rows -----------------------------------------
    # Decode shapes: one query, a filled cache, GQA 4-over-2.  The causal
    # mask is the one a decode step really has (`i + start >= j`), which
    # always keeps at least the current position -- so no row is fully
    # masked, which is the condition under which the additive rewrite of a
    # "large negative number" select is exact (metal_sdpa.cc kMaskFloor).
    klen, heads, kvh, hd = 8, 4, 2, 16
    causal = np.zeros((1, klen), bool)
    causal[0, :5] = True                     # decode step 4 of an 8-slot cache
    slide = np.zeros((1, klen), bool)
    slide[0, 1:5] = True                     # ...with a 4-wide window
    for dt, tol in (("float32", DOT), ("bfloat16", HALF)):
        q = _rand((2, 1, heads, hd), 50).astype(dt) * 0.5
        kk = _rand((2, klen, kvh, hd), 51).astype(dt) * 0.5
        vv = _rand((2, klen, kvh, hd), 52).astype(dt) * 0.5
        sinks = _rand((heads,), 53).astype(dt)
        out.append((f"sdpa gpt-oss sink decode {dt}",
                    lambda a, b, c, s, mc=causal, g=heads // kvh: attn_gptoss(
                        a, b, c, s, None, jnp.asarray(mc), 0.25, g),
                    [q, kk, vv, sinks], *tol))
        out.append((f"sdpa gpt-oss sink sliding decode {dt}",
                    lambda a, b, c, s, ms=slide, mc=causal,
                    g=heads // kvh: attn_gptoss(
                        a, b, c, s, jnp.asarray(ms), jnp.asarray(mc), 0.25, g),
                    [q, kk, vv, sinks], *tol))
        gq = _rand((2, 1, heads, hd), 54).astype(dt) * 0.5
        gk = _rand((2, klen, kvh, hd), 55).astype(dt) * 0.5
        gv = _rand((2, klen, kvh, hd), 56).astype(dt) * 0.5
        gmask = np.repeat(causal[None], 2, axis=0)          # [b, t, s]
        out.append((f"sdpa gemma4 grouped decode {dt}",
                    lambda a, b, c, mm=gmask: attn_gemma4(
                        a, b, c, jnp.asarray(mm)),
                    [gq, gk, gv], *tol))

    # The negative the widened sentinel rule must keep: a `select` whose
    # false branch is a SMALL constant is not a mask, and fusing it as one
    # would change the answer outright.  -8 is under kMaskFloor, so this must
    # run literally -- the row passes either way, but `_keras_attn_tags`
    # checks that nothing fused.
    def attn_small_select(q, k, v, mask, scale):
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        logits = jnp.where(mask, logits, jnp.asarray(-8.0, logits.dtype))
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, -1), v)

    q = _rand((2, 4, 8, 16), 60) * 0.5
    k = _rand((2, 4, 8, 16), 61) * 0.5
    v = _rand((2, 4, 8, 16), 62) * 0.5
    smask = np.zeros((2, 4, 8, 8), bool)
    smask[..., :5] = True
    out.append(("sdpa small select constant is not a mask",
                lambda a, b, c, mm=smask: attn_small_select(
                    a, b, c, jnp.asarray(mm), 0.25), [q, k, v], *DOT))
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
        # P30: the fused-recognizer fingerprints, exactly as maxtext spells
        # them (metal_norm.cc / metal_mla.cc); the CPU runs the literal
        # chains.  f32, so the fused-vs-literal difference stays at the
        # 1e-6 band this section compares at.
        ("rms norm fingerprint", _RMS_NORM_FORM,
         [_rand((2, 1, 8), 264), _rand((8,), 265)]),
        # P34: every branch of the dynamic-slice START PLAN
        # (metal_lowering.cc `AppendStartPlan`), written as raw modules
        # because jax canonicalises an all-constant dynamic_slice into a
        # static `stablehlo.slice` and never emits the mixed spellings at
        # all.  The plan decides which axes reach MLX and with what start,
        # so a wrong one is silent wrongness: a window read from the wrong
        # offset.  Each case is compared against jax-CPU running the same
        # module, and each start is chosen to need CLAMPING in at least one
        # direction somewhere in the set.
        ("dynamic_slice: one dynamic axis of four", _DS_PLAN_ONE_DYNAMIC,
         [np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5),
          np.int32(1)]),
        ("dynamic_slice: one dynamic axis, clamped high",
         _DS_PLAN_ONE_DYNAMIC,
         [np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5),
          np.int32(9)]),
        ("dynamic_slice: one dynamic axis, clamped low", _DS_PLAN_ONE_DYNAMIC,
         [np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5),
          np.int32(-4)]),
        ("dynamic_slice: constant non-zero starts", _DS_PLAN_MIXED,
         [np.arange(20, dtype=np.float32).reshape(4, 5), np.int32(2)]),
        ("dynamic_slice: constant starts past the end", _DS_PLAN_MIXED,
         [np.arange(20, dtype=np.float32).reshape(4, 5), np.int32(-7)]),
        ("dynamic_slice: every start constant", _DS_PLAN_ALL_CONST,
         [np.arange(20, dtype=np.float32).reshape(4, 5)]),
        ("dynamic_slice: every start a constant zero", _DS_PLAN_ALL_ZERO,
         [np.arange(20, dtype=np.float32).reshape(4, 5)]),
        ("dynamic_update_slice: one dynamic axis of four",
         _DUS_PLAN_ONE_DYNAMIC,
         [np.zeros((2, 3, 4, 5), np.float32),
          np.full((1, 3, 4, 5), 7.0, np.float32), np.int32(1)]),
        ("dynamic_update_slice: one dynamic axis, clamped",
         _DUS_PLAN_ONE_DYNAMIC,
         [np.zeros((2, 3, 4, 5), np.float32),
          np.full((1, 3, 4, 5), 7.0, np.float32), np.int32(6)]),
        ("dynamic_update_slice: constant non-zero starts", _DUS_PLAN_MIXED,
         [np.zeros((4, 5), np.float32), np.full((2, 2), -3.0, np.float32),
          np.int32(1)]),
        ("dynamic_update_slice: every start constant", _DUS_PLAN_ALL_CONST,
         [np.zeros((4, 5), np.float32), np.full((2, 2), -3.0, np.float32)]),
        ("dynamic_update_slice: every start a constant zero",
         _DUS_PLAN_ALL_ZERO,
         [np.zeros((4, 5), np.float32), np.full((2, 2), -3.0, np.float32)]),
        ("mla two-span decode attention", _MLA_TWO_SPAN,
         [_rand((1, 1, 2, 4), 266) * 0.5, _rand((1, 4, 2, 4), 267) * 0.5,
          _rand((1, 4, 2, 4), 268) * 0.5,
          np.array([[1, 0, 1, 1]], np.int32),
          _rand((1, 2, 2, 4), 269) * 0.5, _rand((1, 2, 2, 4), 270) * 0.5,
          np.array([[1, 1]], np.int32)]),
        # The same chain in maxtext's GROUPED-query spelling (row 11: 16
        # query heads over 8 KV heads), which declined until the matcher
        # read the grouping off the masked-scores reshape.
        ("gqa two-span decode attention", _MLA_GQA_TWO_SPAN_F32,
         [_rand((1, 1, 4, 4), 271) * 0.5, _rand((1, 4, 2, 4), 272) * 0.5,
          _rand((1, 4, 2, 4), 273) * 0.5,
          np.array([[1, 0, 1, 1]], np.int32),
          _rand((1, 2, 2, 4), 274) * 0.5, _rand((1, 2, 2, 4), 275) * 0.5,
          np.array([[1, 1]], np.int32)]),
        # keras-hub's grouped-query decode attention with the KV head repeat
        # spelled out (rows 7 / 20), both f32 forms: the composite path and
        # the multi-query unit-axis spelling.  `_p37_gqa` pins that the
        # repeat is ABSORBED and that the answer is bit-identical to the
        # repeated fusion's; this row pins the answer against jax-CPU.
    ] + [
        (f"keras gqa repeat {label}", text, _gqa_inputs(text))
        for label, text, _absorbed, dt, _why in _gqa_forms() if dt == "f32"
    ]


# P34: the dynamic-slice START PLAN's branches.
#
# `AppendStartPlan` (metal_lowering.cc) decides which axes reach MLX's
# dynamic slice at all and which starts it resolves at lowering, so what
# these pin is the OFFSET the window is read from -- the one thing a wrong
# plan gets wrong silently.  Four shapes of plan, each in a slice and an
# update flavour:
#
#   ONE_DYNAMIC   one data start, the rest constant zeros -- every scanned
#                 layer's weight read and every KV cache write, and the
#                 case the plan exists for (rank 4: three zeros dropped)
#   MIXED         a data start on one axis and a NON-ZERO constant on
#                 another, so the plan must carry a resolved value
#   ALL_CONST     no data start at all, all of them non-zero
#   ALL_ZERO      no data start and nothing non-zero: the plan's
#                 degenerate case, which keeps one constant-zero axis
#
# jax emits none of these but the first: it folds an all-constant
# dynamic_slice into `stablehlo.slice`.  Hence raw modules.
_DS_PLAN_ONE_DYNAMIC = """
module @ds_plan_one_dynamic {
  func.func public @main(%x: tensor<2x3x4x5xf32>, %i: tensor<i32>)
      -> tensor<1x3x4x5xf32> {
    %z = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %x, %i, %z, %z, %z, sizes = [1, 3, 4, 5]
        : (tensor<2x3x4x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>,
           tensor<i32>) -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>
  }
}
"""

_DS_PLAN_MIXED = """
module @ds_plan_mixed {
  func.func public @main(%x: tensor<4x5xf32>, %i: tensor<i32>)
      -> tensor<2x2xf32> {
    %c = stablehlo.constant dense<3> : tensor<i32>
    %0 = stablehlo.dynamic_slice %x, %i, %c, sizes = [2, 2]
        : (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
"""

_DS_PLAN_ALL_CONST = """
module @ds_plan_all_const {
  func.func public @main(%x: tensor<4x5xf32>) -> tensor<2x2xf32> {
    %a = stablehlo.constant dense<2> : tensor<i32>
    %b = stablehlo.constant dense<7> : tensor<i32>
    %0 = stablehlo.dynamic_slice %x, %a, %b, sizes = [2, 2]
        : (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
"""

_DS_PLAN_ALL_ZERO = """
module @ds_plan_all_zero {
  func.func public @main(%x: tensor<4x5xf32>) -> tensor<2x2xf32> {
    %z = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %x, %z, %z, sizes = [2, 2]
        : (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
"""

_DUS_PLAN_ONE_DYNAMIC = """
module @dus_plan_one_dynamic {
  func.func public @main(%x: tensor<2x3x4x5xf32>, %u: tensor<1x3x4x5xf32>,
      %i: tensor<i32>) -> tensor<2x3x4x5xf32> {
    %z = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %x, %u, %i, %z, %z, %z
        : (tensor<2x3x4x5xf32>, tensor<1x3x4x5xf32>, tensor<i32>,
           tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x3x4x5xf32>
    return %0 : tensor<2x3x4x5xf32>
  }
}
"""

_DUS_PLAN_MIXED = """
module @dus_plan_mixed {
  func.func public @main(%x: tensor<4x5xf32>, %u: tensor<2x2xf32>,
      %i: tensor<i32>) -> tensor<4x5xf32> {
    %c = stablehlo.constant dense<3> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %x, %u, %i, %c
        : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>)
        -> tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
}
"""

_DUS_PLAN_ALL_CONST = """
module @dus_plan_all_const {
  func.func public @main(%x: tensor<4x5xf32>, %u: tensor<2x2xf32>)
      -> tensor<4x5xf32> {
    %a = stablehlo.constant dense<1> : tensor<i32>
    %b = stablehlo.constant dense<9> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %x, %u, %a, %b
        : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>)
        -> tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
}
"""

_DUS_PLAN_ALL_ZERO = """
module @dus_plan_all_zero {
  func.func public @main(%x: tensor<4x5xf32>, %u: tensor<2x2xf32>)
      -> tensor<4x5xf32> {
    %z = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %x, %u, %z, %z
        : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>)
        -> tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
}
"""


def _stacked_relayout_forms():
    """(label, jax function, inputs) for `_p41_stacked_relayout`.

    Both spell maxtext's attention out-projection read: the ORIGINAL stack
    is [heads, L, head_dim, model] (param_scan_axis = 1), the graph hoists a
    [1, 0, 2, 3] transpose out of the loop, and the layer's dot contracts
    (heads, head_dim) -- axes that straddle L in the buffer, so no [L, K, N]
    view of it exists and the recognizer DECLINED until B3.  The inline form
    slices in @main; the scanned form is the helper-call spelling inside a
    while loop, the way maxtext's decode body reads it.

    Both are M = 1 -- one decoded token, the row-10 geometry -- because that
    is where the bit contract lives: at M = 1 the pack is read by the same
    gemv `gather_mm` dispatches for every other stacked read.  (At M > 1
    `gather_mm` and the slice chain's `matmul` pick different steel kernels
    and accumulate K in different orders -- measured 9.6e-7 relative on
    K = 512 -- which is the stacked recognizer's own, pre-B3 class; the
    M = 3 differential case above covers it at DOT tolerance.)  The dots are
    spelled with `dot_general` directly (`jnp.einsum` folds the head axes
    through a reshape the matcher does not read through), and the inline
    slice with `lax.dynamic_slice` (`dynamic_index_in_dim`'s helper carries
    jax's negative-index wrap -- compare/add/select -- which the helper-call
    matcher has never accepted; the scan lowering's helper is the bare
    slice).
    """
    import jax
    import jax.numpy as jnp

    dn = (((1, 2), (0, 1)), ((), ()))
    # B3_RELAYOUT_M (test-only): the activation row count, so the same
    # forms can show the decode-only rule at M > 1.
    m = int(os.environ.get("B3_RELAYOUT_M", "1"))
    x = _rand((m, 8, 64), 90) * 0.1
    W = _rand((8, 4, 64, 64), 91) * 0.1
    Ws = _rand((8, 4, 64, 512), 92) * 0.05
    h0 = _rand((m, 8, 64), 93) * 0.1

    def inline(x, W, i):
        w = jax.lax.dynamic_slice(jnp.transpose(W, (1, 0, 2, 3)),
                                  (i, 0, 0, 0), (1, 8, 64, 64))
        return jax.lax.dot_general(x, w.reshape(8, 64, 64), dn)

    def scanned(h, W):
        def body(h, w):
            return jax.lax.dot_general(h, w, dn).reshape(m, 8, 64), None
        return jax.lax.scan(body, h, jnp.transpose(W, (1, 0, 2, 3)))[0]

    return [
        ("o-proj inline", inline, [x, W, np.array(3, np.int32)]),
        ("o-proj scanned", scanned, [h0, Ws]),
    ]


def _ragged_decode_forms():
    """(label, jax function, args) for `_p44_ragged_decode`.

    Row 10 rewrite 1: the ragged DECODE form (metal_ragged.cc, "the decode
    form").  One MoE layer stack spelled the way maxtext's tape has it --
    softmax router, `lax.top_k`, ids flattened, `jnp.argsort` (jax's
    `@argsort` callee), rows = `repeat(x, topk)[perm]` (a clamping gather),
    `jnp.bincount` (`@clip` callee + scatter-add), rows padded to the
    tiling, three `lax.ragged_dot` (jax's dense fallback, whose cumsum is
    the `@cumsum` callee) + swiglu, `[:m]` slices, unpermute by
    `argsort(perm)`, the f32 weighted combine, two shared experts and the
    residual add -- inside a `lax.scan` over transposed [g, L, k, n]
    weight stacks (the stacked-weights extension) unless said otherwise.
    Small: g = 8, top-k 2, k = 32, n = 16, tile 16, L = 3.

      D1   decode (T = 1), bf16: fires, `decode nopad` on the gate/up dots
           (one replicated activation) and `decode-rows nopad` on the down
           dot (real rows).
      D1f  D1 in f32.
      D2   prefill (T = 4, m = 8 of a 16-row tile): declines -- the ids
           come from a top-k over 4 tokens -- and keeps the pad.
      D3   decode over PLAIN [g, k, n] weights (no scan): fires with L = 1.
      D4   decode whose ids are a program INPUT: declines, range unproven.
      D5   decode whose gate root has a SECOND reader (a sum over the whole
           tile added to the output): fires, but that dot keeps its pad
           (`decode` without `nopad`).
      D6   decode with `argsort(descending=True)`: declines, the sort is not
           ascending (the rows then sit in descending expert order and the
           dense form computes what it computes; both arms agree with it).
      D7   decode with top-k 1 (m = 1).
      D8   decode (top-k 4) whose rows come off a broadcast RESHAPED to
           [4, 48] instead of flattened to [m, k] -- rows 1 and 3 are x0
           rotated by 16, not x0 (a hand-written gather; jax's `repeat`
           never spells it).  The replicated-activation proof declines on
           the reshape width and the dots run as `decode-rows` over the
           real rows: still fewer dispatches, still bit-identical.
      D9   decode whose "argsort" is `sort((V, [1, 0]))` -- an in-range
           constant payload that is NOT the iota: declines, the payload is
           not the iota (`take(V, perm)` would name the wrong expert; the
           unfixed recognizer accepted any in-range constant and swapped
           the rows).
    Each function's answer is [final activation, the down dot's rows before
    the unpermute (per layer), the shared-expert output (per layer)].

    `_ragged_shared_ends_module` adds the SHARED raw module: two dots over
    ONE `ends` value (jax spells one `@cumsum` call per `ragged_dot`; the
    module is jax's own text with the second call removed) where the
    second root has an extra real row and declines the decode form on its
    bincount -- the base root reads the cumsum the decode root absorbed,
    which the joint fixpoint must keep lowered (before the fix the whole
    executable fell back to the plain tape).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    G, K, N, TILE, L = 8, 32, 16, 16, 3

    def rnd(shape, seed, scale=1.0, dtype=ml_dtypes.bfloat16):
        return (np.random.RandomState(seed).standard_normal(shape)
                * scale).astype(dtype)

    def make(tag, T=1, topk=2, dtype=jnp.bfloat16, stacked=True,
             ids_input=False, second_reader=False, descending=False,
             rows_rotated=False, perm_const=False):
        m = T * topk
        M = -(-m // TILE) * TILE

        def layer(h, wr, wg, wu, wd, s1, s2, s3, ids_in=None):
            x2d = h.reshape(-1, K)
            probs = jax.nn.softmax((x2d @ wr).astype(jnp.float32), axis=-1)
            if ids_in is None:
                vals, ids = lax.top_k(probs, topk)
            else:
                ids = ids_in
                vals = jnp.take_along_axis(probs, ids.reshape(T, topk), -1)
            V = ids.reshape(-1)
            if perm_const:
                # An in-range payload that is not the iota: result 1 is
                # the payload sorted along with V, not V's argsort.
                payload = jnp.asarray(np.arange(m)[::-1].astype(np.int32))
                perm = lax.sort((V, payload), num_keys=1)[1]
            else:
                perm = jnp.argsort(V, descending=descending)
            if rows_rotated:
                # [1, 6, K] -> [4, 48]: a k-wide row of it is x0 only for
                # rows 0 and 2; the gather geometry is repeat's exactly.
                assert m == 4 and T == 1
                R = lax.broadcast_in_dim(x2d, (1, 6, K), (0, 2)).reshape(4, 48)
                rows = lax.gather(
                    R, perm[:, None],
                    lax.GatherDimensionNumbers(offset_dims=(1,),
                                               collapsed_slice_dims=(0,),
                                               start_index_map=(0,)),
                    slice_sizes=(1, K), mode="clip")
            else:
                rows = jnp.repeat(x2d, topk, axis=0)[perm]
            gs = jnp.bincount(V, length=G)
            zero = jnp.zeros((), dtype)
            rp = lax.pad(rows, zero, ((0, M - m, 0), (0, 0, 0)))
            a_full = lax.ragged_dot(rp, wg, gs, preferred_element_type=dtype)
            a = a_full[:m]
            b = lax.ragged_dot(rp, wu, gs, preferred_element_type=dtype)[:m]
            hm = (a * jax.nn.sigmoid(a)) * b
            hp = lax.pad(hm, zero, ((0, M - m, 0), (0, 0, 0)))
            o = lax.ragged_dot(hp, wd, gs, preferred_element_type=dtype)[:m]
            ou = o[jnp.argsort(perm)]
            comb = (ou.astype(jnp.float32).reshape(-1, topk, K)
                    * vals.reshape(-1, topk)[..., None]).sum(1)
            sg = x2d @ s1
            sh = ((sg * jax.nn.sigmoid(sg)) * (x2d @ s2)) @ s3
            out = (h.astype(jnp.float32) + comb.reshape(h.shape)
                   + sh.astype(jnp.float32).reshape(h.shape))
            if second_reader:
                out = out + a_full.astype(jnp.float32).sum() * 1e-3
            return out.astype(dtype), (o, sh)

        if stacked:
            def fn(x, Wr, Wg, Wu, Wd, S1, S2, S3, *rest):
                def body(h, ws):
                    return layer(h, *ws, *rest)
                return lax.scan(body, x, (Wr, jnp.transpose(Wg, (1, 0, 2, 3)),
                                          jnp.transpose(Wu, (1, 0, 2, 3)),
                                          jnp.transpose(Wd, (1, 0, 2, 3)),
                                          S1, S2, S3))
        else:
            def fn(x, wr, wg, wu, wd, s1, s2, s3, *rest):
                out, (o, sh) = layer(x, wr, wg, wu, wd, s1, s2, s3, *rest)
                return out, (o[None], sh[None])
        fn.__name__ = f"rgd_{tag}"
        npdt = np.float32 if dtype == jnp.float32 else ml_dtypes.bfloat16
        seed = sum(map(ord, tag))
        if stacked:
            args = [rnd((1, T, K), seed, 1.0, npdt),
                    rnd((L, K, G), seed + 1, 0.3, npdt),
                    rnd((G, L, K, N), seed + 2, 0.2, npdt),
                    rnd((G, L, K, N), seed + 3, 0.2, npdt),
                    rnd((G, L, N, K), seed + 4, 0.2, npdt),
                    rnd((L, K, N), seed + 5, 0.2, npdt),
                    rnd((L, K, N), seed + 6, 0.2, npdt),
                    rnd((L, N, K), seed + 7, 0.2, npdt)]
        else:
            args = [rnd((1, T, K), seed, 1.0, npdt),
                    rnd((K, G), seed + 1, 0.3, npdt),
                    rnd((G, K, N), seed + 2, 0.2, npdt),
                    rnd((G, K, N), seed + 3, 0.2, npdt),
                    rnd((G, N, K), seed + 4, 0.2, npdt),
                    rnd((K, N), seed + 5, 0.2, npdt),
                    rnd((K, N), seed + 6, 0.2, npdt),
                    rnd((N, K), seed + 7, 0.2, npdt)]
        if ids_input:
            args.append(np.array([5, 1], np.int32)[:m].reshape(T, topk))
        return (tag.upper(), fn, args)

    return [make("d1"),
            make("d1f", dtype=jnp.float32),
            make("d2", T=4),
            make("d3", stacked=False),
            make("d4", ids_input=True),
            make("d5", second_reader=True),
            make("d6", descending=True),
            make("d7", topk=1),
            make("d8", topk=4, rows_rotated=True),
            make("d9", perm_const=True)]


def _ragged_shared_ends_module():
    """(module text, args) for the SHARED case of `_p44_ragged_decode`.

    Two `lax.ragged_dot`s over one `gs`: the first over the two routed
    rows, the second over those rows plus one extra real row (m = 3 against
    a bincount of 2, so its decode proof declines on the row count).  jax
    spells a separate `call @cumsum(counts)` per dot; the second call is
    removed from jax's own text and its result replaced by the first's, so
    both roots read ONE `ends`.  Built at run time so the spelling is the
    installed jax's; an unexpected spelling raises instead of testing
    nothing.
    """
    import re
    import jax
    import jax.numpy as jnp
    from jax import lax
    G, K, N, TILE = 8, 32, 16, 16

    def rgd_shared(x2d, wr, wg, wg2, extra):
        probs = jax.nn.softmax((x2d @ wr).astype(jnp.float32), axis=-1)
        _vals, ids = lax.top_k(probs, 2)
        V = ids.reshape(-1)
        perm = jnp.argsort(V)
        rows = jnp.repeat(x2d, 2, axis=0)[perm]
        gs = jnp.bincount(V, length=G)
        zero = jnp.zeros((), jnp.bfloat16)
        rp = lax.pad(rows, zero, ((0, TILE - 2, 0), (0, 0, 0)))
        a = lax.ragged_dot(rp, wg, gs, preferred_element_type=jnp.bfloat16)[:2]
        rp3 = lax.pad(jnp.concatenate([rows, extra], 0), zero,
                      ((0, TILE - 3, 0), (0, 0, 0)))
        b = lax.ragged_dot(rp3, wg2, gs, preferred_element_type=jnp.bfloat16)[:3]
        return a, b

    def rnd(shape, seed, scale=1.0):
        return (np.random.RandomState(seed).standard_normal(shape)
                * scale).astype(ml_dtypes.bfloat16)

    args = [rnd((1, K), 901), rnd((K, G), 902, 0.3), rnd((G, K, N), 903, 0.2),
            rnd((G, K, N), 904, 0.2), rnd((1, K), 905)]
    lines = jax.jit(rgd_shared).lower(*args).as_text().splitlines()
    calls = [i for i, ln in enumerate(lines) if "call @cumsum(" in ln]
    if len(calls) != 2:
        raise RuntimeError(f"expected two @cumsum calls, found {len(calls)}")
    first = re.match(r"\s*(%\d+) = call @cumsum", lines[calls[0]]).group(1)
    second = re.match(r"\s*(%\d+) = call @cumsum", lines[calls[1]]).group(1)
    del lines[calls[1]]
    text = re.sub(re.escape(second) + r"\b", first, "\n".join(lines) + "\n")
    if text.count("call @cumsum(") != 1:
        raise RuntimeError("the shared-ends rewrite did not take")
    return text, args


def _proj_pack_forms():
    """(label, jax function, inputs, calls) for `_p43_proj_pack`.

    B6: the projection PACK (metal_proj.cc).  Sibling decode projections over
    one activation become one dot over the concatenated weights.  The forms
    are the shapes the model table runs, cut down to what a contract needs:

      F1  keras-shaped gpt-oss decode (row 7): x[1,1,2880] against
          W_q[2880,64,64] / W_k[2880,8,64] / W_v[2880,8,64], biases, the
          rotate-half rope on q and k, the K/V cache updates -- inside a
          counted while whose bound is an ARGUMENT (so the body is a region,
          never unrolled) with the weights as carries, the way the real
          decode program reads them.  Policy `kv` packs K+V and leaves q.
      F2  row 11's shape at the top level, no bias, q/k norms over the head
          dim: x[1,1,1024] . W[1024,16,128] / [1024,8,128] / [1024,8,128].
          Called FOUR times with fresh weights (the repack contract: two
          repacks, then the executable re-lowers without the projection
          packs and keeps its norms).
      F3  F1's projections at prefill (M = 59): declines, decode-only.
      F4  mismatched head widths (row 8's gated q): [2048,4,512] + [2048,2,256]
          pack on the merged [K, n] view.
      F5  two tiny f32 members (GDN's b/a): [2048,32] + [2048,32].
      F6  two CONSTANT weights (no pack argument, never repacks).
      F7  two affine-quantized members: the qmm recognizer owns the dots.
      F8  DONATED weights, returned updated in place: declines (a pack would
          repack every call), called three times with fresh weights.
      F9  K = 8192 with the packed N crossing 2048: a different gemv_t
          summation tree, tolerance-level.
      F10 weights stored [N, K] (`x @ W.T`, an Equinox/PyTorch port) in
          f32: the members run MLX's non-T `gemv`, so the pack is laid out
          [n_total, K] BY STORAGE and stays bit-identical (f32 exposes a
          reorder bf16 rounding would hide).
      F11 the weights behind ONE tuple `optimization_barrier` (jax's usual
          way to pin a pytree), one of them returned as is: looked
          through, packed, the pass-through is not an update.
      F12 weights the program UPDATES (`w - lr * g`, not donated: texmo's
          training-step shape): declines at analysis, no repack storm.
    `calls` is how many times the child runs the form (each with fresh
    inputs); a label with a call suffix is one answer.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    bf16 = jnp.bfloat16

    def rnd(shape, seed, scale=1.0, dtype=ml_dtypes.bfloat16):
        return (np.random.RandomState(seed).standard_normal(shape)
                * scale).astype(dtype)

    # --- F1 -----------------------------------------------------------
    D, HQ, HKV, HD, T = 2880, 64, 8, 64, 4

    def f1_step(s):
        i, n, x, ck, cv, wq, wk, wv, wo, bq, bk, bv = s
        pos = i % T
        ang = (pos.astype(jnp.float32)
               * jnp.arange(HD // 2, dtype=jnp.float32) * 0.01)
        ang = jnp.concatenate([ang, ang])[None, None, None, :]
        cos = jnp.cos(ang).astype(bf16)
        sin = jnp.sin(ang).astype(bf16)
        q = jnp.einsum("bqm,muh->bquh", x, wq) + bq
        k = jnp.einsum("bqm,muh->bquh", x, wk) + bk
        v = jnp.einsum("bqm,muh->bquh", x, wv) + bv
        q = _rope_keras(q, cos, sin)
        k = _rope_keras(k, cos, sin)
        ck = lax.dynamic_update_slice(ck, k, (0, pos, 0, 0))
        cv = lax.dynamic_update_slice(cv, v, (0, pos, 0, 0))
        qg = q.reshape(1, 1, HKV, HQ // HKV, HD).sum(3)
        sc = jnp.einsum("bqhd,bthd->bqht", qg, ck)
        o = jnp.einsum("bqht,bthd->bqhd", sc, cv).reshape(1, 1, HKV * HD)
        x = (x.astype(jnp.float32)
             + 0.01 * jnp.einsum("bqn,nm->bqm", o, wo).astype(jnp.float32)
             ).astype(bf16)
        return (i + 1, n, x, ck, cv, wq, wk, wv, wo, bq, bk, bv)

    def proj_f1(n, x, ck, cv, wq, wk, wv, wo, bq, bk, bv):
        s = lax.while_loop(lambda s: s[0] < s[1], f1_step,
                           (jnp.int32(0), n, x, ck, cv, wq, wk, wv, wo,
                            bq, bk, bv))
        return s[2], s[3], s[4]

    def f1_args(seed):
        zeros = np.zeros((1, T, HKV, HD), ml_dtypes.bfloat16)
        return [np.int32(3), rnd((1, 1, D), seed), zeros, zeros.copy(),
                rnd((D, HQ, HD), seed + 1, 0.02),
                rnd((D, HKV, HD), seed + 2, 0.02),
                rnd((D, HKV, HD), seed + 3, 0.02),
                rnd((HKV * HD, D), seed + 4, 0.02),
                rnd((HQ, HD), seed + 5, 0.1), rnd((HKV, HD), seed + 6, 0.1),
                rnd((HKV, HD), seed + 7, 0.1)]

    # --- F2 -----------------------------------------------------------
    def head_norm(x, w):
        xf = x.astype(jnp.float32)
        var = jnp.mean(xf * xf, axis=-1, keepdims=True)
        return (xf * lax.rsqrt(var + 1e-6)).astype(bf16) * w

    def proj_f2(x, wq, wk, wv, nq, nk):
        q = jnp.einsum("bqm,muh->bquh", x, wq)
        k = jnp.einsum("bqm,muh->bquh", x, wk)
        v = jnp.einsum("bqm,muh->bquh", x, wv)
        return head_norm(q, nq), head_norm(k, nk), v

    def f2_args(seed):
        return [rnd((1, 1, 1024), seed), rnd((1024, 16, 128), seed + 1, 0.03),
                rnd((1024, 8, 128), seed + 2, 0.03),
                rnd((1024, 8, 128), seed + 3, 0.03),
                (1.0 + np.random.RandomState(seed + 4).standard_normal((128,))
                 * 0.1).astype(ml_dtypes.bfloat16),
                (1.0 + np.random.RandomState(seed + 5).standard_normal((128,))
                 * 0.1).astype(ml_dtypes.bfloat16)]

    # --- F3 -----------------------------------------------------------
    def proj_f3(x, wq, wk, wv, bq, bk, bv):
        q = jnp.einsum("bqm,muh->bquh", x, wq) + bq
        k = jnp.einsum("bqm,muh->bquh", x, wk) + bk
        v = jnp.einsum("bqm,muh->bquh", x, wv) + bv
        return q[:, :, :2], k, v

    def f3_args(seed):
        return [rnd((1, 59, D), seed, 0.5), rnd((D, HQ, HD), seed + 1, 0.02),
                rnd((D, HKV, HD), seed + 2, 0.02),
                rnd((D, HKV, HD), seed + 3, 0.02),
                rnd((HQ, HD), seed + 5, 0.1), rnd((HKV, HD), seed + 6, 0.1),
                rnd((HKV, HD), seed + 7, 0.1)]

    # --- F4 -----------------------------------------------------------
    def proj_f4(x, w1, w2):
        return (jnp.einsum("bqm,muh->bquh", x, w1),
                jnp.einsum("bqm,muh->bquh", x, w2))

    def f4_args(seed):
        return [rnd((1, 1, 2048), seed), rnd((2048, 4, 512), seed + 1, 0.02),
                rnd((2048, 2, 256), seed + 2, 0.02)]

    # --- F5 -----------------------------------------------------------
    def proj_f5(x, wb, wa):
        return x @ wb, x @ wa

    def f5_args(seed):
        return [rnd((1, 2048), seed, 1.0, np.float32),
                rnd((2048, 32), seed + 1, 0.02, np.float32),
                rnd((2048, 32), seed + 2, 0.02, np.float32)]

    # --- F6 -----------------------------------------------------------
    c1 = rnd((256, 128), 601, 0.05, np.float32)
    c2 = rnd((256, 128), 602, 0.05, np.float32)

    def proj_f6(x):
        return x @ jnp.asarray(c1), x @ jnp.asarray(c2)

    def f6_args(seed):
        return [rnd((1, 256), seed, 1.0, np.float32)]

    # --- F7 -----------------------------------------------------------
    def unpack(packed, columns):
        lo = jnp.bitwise_and(packed, jnp.int8(0x0F))
        lo = jnp.where(lo > 7, lo - 16, lo)
        hi = jnp.right_shift(packed, jnp.int8(4))
        return jnp.reshape(jnp.stack([lo, hi], axis=-1),
                           packed.shape[:-1] + (columns,))

    def dense_sub(packed, scale, zero, g_idx, x, columns):
        w = unpack(packed, columns)
        g = g_idx.astype(jnp.int32)
        s_ = jnp.take(scale, g, axis=0)
        z = jnp.take(zero, g, axis=0)
        return x @ ((w.astype(x.dtype) - z.astype(x.dtype)) * s_)

    def proj_f7(x, p1, s1, z1, g1, p2, s2, z2, g2):
        return (dense_sub(p1, s1, z1, g1, x, 64),
                dense_sub(p2, s2, z2, g2, x, 64))

    def f7_args(seed):
        _q1, p1, s1, z1, g1 = _quantize(512, 64, 128, "float32", seed=seed)
        _q2, p2, s2, z2, g2 = _quantize(512, 64, 128, "float32",
                                        seed=seed + 1)
        return [rnd((1, 512), seed, 0.5, np.float32), p1, s1, z1, g1,
                p2, s2, z2, g2]

    # --- F8 -----------------------------------------------------------
    def proj_f8(x, w1, w2):
        # The donated weights come back updated in place (the training
        # shape): jax aliases each to its output, and only a USABLE donation
        # reaches the plugin -- an unusable one is dropped from the module.
        return x @ w1, x @ w2, w1 * 0.5, w2 * 0.5

    def f8_args(seed):
        return [rnd((1, 512), seed), rnd((512, 256), seed + 1, 0.05),
                rnd((512, 256), seed + 2, 0.05)]

    # --- F9 -----------------------------------------------------------
    def proj_f9(x, w1, w2):
        return x @ w1, x @ w2

    def f9_args(seed):
        return [rnd((1, 8192), seed), rnd((8192, 1536), seed + 1, 0.01),
                rnd((8192, 1024), seed + 2, 0.01)]

    # --- F10 ----------------------------------------------------------
    def proj_f10(x, w1, w2):
        return (jnp.einsum("bqm,hm->bqh", x, w1),
                jnp.einsum("bqm,hm->bqh", x, w2))

    def f10_args(seed):
        return [rnd((1, 1, 2560), seed, 1.0, np.float32),
                rnd((384, 2560), seed + 1, 0.02, np.float32),
                rnd((384, 2560), seed + 2, 0.02, np.float32)]

    # --- F11 ----------------------------------------------------------
    def proj_f11(x, w1, w2):
        w1b, w2b = lax.optimization_barrier((w1, w2))
        return x @ w1b, x @ w2b, w1b

    def f11_args(seed):
        return [rnd((1, 640), seed), rnd((640, 192), seed + 1, 0.05),
                rnd((640, 192), seed + 2, 0.05)]

    # --- F12 ----------------------------------------------------------
    def proj_f12(x, w1, w2, g1, g2):
        return x @ w1, x @ w2, w1 - 0.1 * g1, w2 - 0.1 * g2

    def f12_args(seed):
        return [rnd((1, 512), seed), rnd((512, 320), seed + 1, 0.05),
                rnd((512, 320), seed + 2, 0.05),
                rnd((512, 320), seed + 3, 0.05),
                rnd((512, 320), seed + 4, 0.05)]

    # The executables are named jit_<function name>: the parent reads each
    # program's narration by that name.
    f8 = jax.jit(proj_f8, donate_argnums=(1, 2))
    return [
        ("F1 keras decode loop", jax.jit(proj_f1), f1_args, 1),
        ("F2 row11 norms", jax.jit(proj_f2), f2_args, 4),
        ("F3 prefill M59", jax.jit(proj_f3), f3_args, 1),
        ("F4 mismatched heads", jax.jit(proj_f4), f4_args, 1),
        ("F5 tiny f32 pair", jax.jit(proj_f5), f5_args, 1),
        ("F6 constant pair", jax.jit(proj_f6), f6_args, 1),
        ("F7 quantized pair", jax.jit(proj_f7), f7_args, 1),
        ("F8 donated pair", f8, f8_args, 3),
        ("F9 K8192 crossing 2048", jax.jit(proj_f9), f9_args, 1),
        ("F10 NK-stored f32 pair", jax.jit(proj_f10), f10_args, 1),
        ("F11 tuple barrier pair", jax.jit(proj_f11), f11_args, 1),
        ("F12 updated pair", jax.jit(proj_f12), f12_args, 1),
    ]


def _gemv_band(K, N, nk):
    """The tile MLX's M = 1 dispatch picks for a [K -> N] dot, reduced to
    what fixes the per-output K summation order (mlx-src/mlx/backend/metal/
    matmul.cpp 1085-1153, gemv.h): `gemv_t` for a [K, N] weight orders by
    (BM, SM, TM) with the fork's occupancy floor on (sm, sn); the non-T
    `gemv` for an [N, K] weight by (BN, SN, TN).  Two dots with equal bands
    run the same instruction sequence per output; a different band is a
    different PSO (expected identical, compiler-dependent) or, past the
    floor, a different tree."""
    if not nk:
        sm, sn = (4, 8) if (K >= 8192 and N >= 2048) else (8, 4)
        bn = 16 if N >= 2048 else 4 if N >= 512 else 2
        tn = 1 if N < 4 else 4
        while sn > 4 and (N + bn * sn * tn - 1) // (bn * sn * tn) < 32:
            sn //= 2
            sm *= 2
        return ("gemv_t", 1, sm, 4, bn, sn, tn)
    bn, sm, sn = 1, 1, 32
    if K <= 64:
        sm, sn = 8, 4
    elif K >= 16 * N:
        bn = 8
    tn = 4
    return ("gemv", bn, sn, tn, sm)


# Per form: K, the members' n_i under the `kv` policy, the storage class;
# `all` adds the q member the policy leaves alone.  From these the contract
# decides which forms must be BIT-identical to the literal dots (every
# member in the pack's band) and which report their bits and hold a ULP.
_PROJ_FORM_BANDS = {
    "F1": (2880, [512, 512], False), "F2": (1024, [2048, 1024, 1024], False),
    "F4": (2048, [2048, 512], False), "F5": (2048, [32, 32], False),
    "F6": (256, [128, 128], False), "F9": (8192, [1536, 1024], False),
    "F10": (2560, [384, 384], True), "F11": (640, [192, 192], False),
}
_PROJ_FORM_BANDS_ALL = dict(_PROJ_FORM_BANDS)
_PROJ_FORM_BANDS_ALL["F1"] = (2880, [4096, 512, 512], False)
_PROJ_FORM_BANDS_ALL["F2"] = (1024, [2048, 1024, 1024], False)


def _proj_same_band(form, bands=_PROJ_FORM_BANDS):
    K, ns, nk = bands[form]
    pack = _gemv_band(K, sum(ns), nk)
    return all(_gemv_band(K, n, nk) == pack for n in ns)


def _start_plan_forms():
    """(label, module, inputs) for `_p34_start_plan`'s narration arm.

    The same modules `_module_cases` compares against jax-CPU, gathered so a
    child can run them all in ONE process and the parent can read a single
    `ds plan:` tally per program off its stderr.  The rank-4 pair is the
    shape the plan exists for -- one data start, three constant zeros.
    """
    x4 = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    u4 = np.full((1, 3, 4, 5), 7.0, np.float32)
    x2 = np.arange(20, dtype=np.float32).reshape(4, 5)
    u2 = np.full((2, 2), -3.0, np.float32)
    return [
        ("ds rank4 one dynamic", _DS_PLAN_ONE_DYNAMIC, [x4, np.int32(1)]),
        ("ds mixed", _DS_PLAN_MIXED, [x2, np.int32(2)]),
        ("ds all const", _DS_PLAN_ALL_CONST, [x2]),
        ("ds all zero", _DS_PLAN_ALL_ZERO, [x2]),
        ("dus rank4 one dynamic", _DUS_PLAN_ONE_DYNAMIC,
         [np.zeros((2, 3, 4, 5), np.float32), u4, np.int32(1)]),
        ("dus mixed", _DUS_PLAN_MIXED, [np.zeros((4, 5), np.float32), u2,
                                        np.int32(1)]),
        ("dus all const", _DUS_PLAN_ALL_CONST,
         [np.zeros((4, 5), np.float32), u2]),
        ("dus all zero", _DUS_PLAN_ALL_ZERO,
         [np.zeros((4, 5), np.float32), u2]),
    ]


# P30: the RMS-norm fingerprint exactly as maxtext/flax spell it (upcast
# omitted: the f32 form), through metal_norm.cc's rewrite on metal and the
# literal chain on CPU.
_RMS_NORM_FORM = """
module @rms_norm_form {
  func.func public @main(%x: tensor<2x1x8xf32>, %w: tensor<8xf32>)
      -> tensor<2x1x8xf32> {
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<2x1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<2x1x8xf32>, tensor<f32>) -> tensor<2x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %b5 = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<2x1x1xf32>
    %div = stablehlo.divide %b4, %b5 : tensor<2x1x1xf32>
    %b7 = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<2x1x1xf32>
    %pe = stablehlo.add %div, %b7 : tensor<2x1x1xf32>
    %rs = stablehlo.rsqrt %pe : tensor<2x1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<2x1x1xf32>) -> tensor<2x1x8xf32>
    %y = stablehlo.multiply %x, %b10 : tensor<2x1x8xf32>
    %zv = stablehlo.broadcast_in_dim %zero, dims = []
        : (tensor<f32>) -> tensor<8xf32>
    %wp = stablehlo.add %w, %zv : tensor<8xf32>
    %dot = stablehlo.dot_general %wp, %y,
        batching_dims = [0] x [2], contracting_dims = [] x []
        : (tensor<8xf32>, tensor<2x1x8xf32>) -> tensor<8x2x1xf32>
    %out = stablehlo.transpose %dot, dims = [1, 2, 0]
        : (tensor<8x2x1xf32>) -> tensor<2x1x8xf32>
    return %out : tensor<2x1x8xf32>
  }
}
"""


# ---------------------------------------------------------------------------
# The dense band's RMS-norm spellings (metal_norm.cc)
# ---------------------------------------------------------------------------
#
# P30 fingerprinted maxtext's norm (above) and matched none of the dense
# rows.  Each module below is a REAL library's lowering, captured at toy dims
# from the harness the model table actually runs and saved with its
# provenance under ~/.cache/metaljax-bench/logs/dense-band-norm/:
#
#   gemma_lib  rows 1/2   gemma.gm.nn.gemma4._layers.RMSNorm
#   keras_lm   rows 5/6/9/12 (and 3/4/7/8/20)
#              keras_hub {Qwen3,Llama,Mixtral,Qwen}LayerNorm -- one spelling,
#              all four families
#   gemma 2/3  gemma.gm.nn._layers.RMSNorm, the `(1 + scale)` form.  No
#              bench row runs it; it is here because the offset fold is
#              general and this is the evidence it was built from.
#
# The MoE band (rows 3/7/8/10) turned out NOT to be that one keras spelling:
# its models carry their own norm classes, and both of them were declined
# whole.  Captured the same way, under the bench harness's own dtype policy,
# and saved under ~/.cache/metaljax-bench/logs/norm-coverage/:
#
#   keras_hub  row 8   qwen3_5_layers.Qwen3_5LayerNorm -- `(1 + w)` formed at
#              [N] width AND in f32 over a bf16 parameter (717 declines per
#              generate shape against 60 matches); Qwen3_5RMSNormGated is the
#              60 that already matched
#   keras_hub  row 3   gemma4_layers.{RMSNormalization, Gemma4VNorm,
#              Gemma4FrozenNorm} -- `ops.power(var + eps, -0.5)` where every
#              other library writes `rsqrt`, so the module contains no
#              `stablehlo.rsqrt` at all and a 30-layer model recognized
#              nothing, silently
#
# The text is the module the PLUGIN is handed, not the one jax prints: XLA's
# parse legalizes chlo (gemma's `chlo.square` arrives as a multiply), CSEs,
# and hoists the constants to the top.  What the matcher walks is this.
#
# `_p31_norm` runs them: each must FIRE with its own form tag, must agree
# with the same binary under METALJAX_NORM=0, and must agree with jax-CPU.

# gemma_lib, bf16 -- rows 1 and 2.  Squares in bf16 and upcasts the RESULT;
# rounds the mean back to bf16, so the eps add and the rsqrt run there too;
# applies the weight as a broadcast multiply.  The root is that multiply.
_NORM_GEMMA_MUL_BF16 = """
module @norm_gemma_mul_bf16 {
  func.func public @main(%w: tensor<8xbf16>, %x: tensor<1x1x8xbf16>)
      -> tensor<1x1x8xbf16> {
    %eps = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<1x1x8xbf16>
    %up = stablehlo.convert %sq : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %red = stablehlo.reduce(%up init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %dn = stablehlo.convert %div : (tensor<1x1x1xf32>) -> tensor<1x1x1xbf16>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<bf16>) -> tensor<1x1x1xbf16>
    %pe = stablehlo.add %dn, %be : tensor<1x1x1xbf16>
    %rs = stablehlo.rsqrt %pe : tensor<1x1x1xbf16>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xbf16>) -> tensor<1x1x8xbf16>
    %y = stablehlo.multiply %x, %b10 : tensor<1x1x8xbf16>
    %bw = stablehlo.broadcast_in_dim %w, dims = [2]
        : (tensor<8xbf16>) -> tensor<1x1x8xbf16>
    %out = stablehlo.multiply %y, %bw : tensor<1x1x8xbf16>
    return %out : tensor<1x1x8xbf16>
  }
}
"""

# The same shape in f32: no converts anywhere, so the whole chain -- square,
# mean, eps, rsqrt -- runs in one dtype.
_NORM_GEMMA_MUL_F32 = """
module @norm_gemma_mul_f32 {
  func.func public @main(%w: tensor<8xf32>, %x: tensor<1x1x8xf32>)
      -> tensor<1x1x8xf32> {
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<1x1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1x1xf32>
    %rs = stablehlo.rsqrt %pe : tensor<1x1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %y = stablehlo.multiply %x, %b10 : tensor<1x1x8xf32>
    %bw = stablehlo.broadcast_in_dim %w, dims = [2]
        : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %out = stablehlo.multiply %y, %bw : tensor<1x1x8xf32>
    return %out : tensor<1x1x8xf32>
  }
}
"""

# gemma_lib `RMSNorm(with_scale=False)` -- the value/router/qk norms rows 1
# and 2 run per layer per token.  There is no weight to bind at all, and
# MLX's weight argument is optional; the root here IS the normalize multiply,
# which is why it must not end up in its own absorbed list.
_NORM_GEMMA_NOSCALE_BF16 = """
module @norm_gemma_noscale_bf16 {
  func.func public @main(%x: tensor<1x1x8xbf16>) -> tensor<1x1x8xbf16> {
    %eps = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<1x1x8xbf16>
    %up = stablehlo.convert %sq : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %red = stablehlo.reduce(%up init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %dn = stablehlo.convert %div : (tensor<1x1x1xf32>) -> tensor<1x1x1xbf16>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<bf16>) -> tensor<1x1x1xbf16>
    %pe = stablehlo.add %dn, %be : tensor<1x1x1xbf16>
    %rs = stablehlo.rsqrt %pe : tensor<1x1x1xbf16>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xbf16>) -> tensor<1x1x8xbf16>
    %out = stablehlo.multiply %x, %b10 : tensor<1x1x8xbf16>
    return %out : tensor<1x1x8xbf16>
  }
}
"""

# gemma 2/3: `normed * (1 + scale)`, the add at FULL rank and in the weight's
# own dtype.  The match folds the 1 off and the emit re-forms it [N] wide,
# which is the same arithmetic because the dtypes agree -- the matcher
# declines the fold when they do not.
_NORM_GEMMA_OFFSET_BF16 = """
module @norm_gemma_offset_bf16 {
  func.func public @main(%w: tensor<8xbf16>, %x: tensor<1x1x8xbf16>)
      -> tensor<1x1x8xbf16> {
    %one = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %eps = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<1x1x8xbf16>
    %up = stablehlo.convert %sq : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %red = stablehlo.reduce(%up init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %dn = stablehlo.convert %div : (tensor<1x1x1xf32>) -> tensor<1x1x1xbf16>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<bf16>) -> tensor<1x1x1xbf16>
    %pe = stablehlo.add %dn, %be : tensor<1x1x1xbf16>
    %rs = stablehlo.rsqrt %pe : tensor<1x1x1xbf16>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xbf16>) -> tensor<1x1x8xbf16>
    %y = stablehlo.multiply %x, %b10 : tensor<1x1x8xbf16>
    %bw = stablehlo.broadcast_in_dim %w, dims = [2]
        : (tensor<8xbf16>) -> tensor<1x1x8xbf16>
    %b1 = stablehlo.broadcast_in_dim %one, dims = []
        : (tensor<bf16>) -> tensor<1x1x8xbf16>
    %ws = stablehlo.add %b1, %bw : tensor<1x1x8xbf16>
    %out = stablehlo.multiply %y, %ws : tensor<1x1x8xbf16>
    return %out : tensor<1x1x8xbf16>
  }
}
"""

# keras_hub -- rows 5, 6, 9 and 12 share this one spelling.  Upcasts x FIRST,
# squares with `power(x, 2)` (the exponent itself a converted i32 splat),
# keeps the whole chain and the weight multiply in f32, and rounds back down
# at the very end: the root is that convert, and the weight enters as its
# stored bf16 widened for the arithmetic.
_NORM_KERAS_BF16 = """
module @norm_keras_bf16 {
  func.func public @main(%w: tensor<8xbf16>, %x: tensor<1x1x8xbf16>)
      -> tensor<1x1x8xbf16> {
    %two = stablehlo.constant dense<2> : tensor<i32>
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %xf = stablehlo.convert %x : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %twof = stablehlo.convert %two : (tensor<i32>) -> tensor<f32>
    %btwo = stablehlo.broadcast_in_dim %twof, dims = []
        : (tensor<f32>) -> tensor<1x1x8xf32>
    %sq = stablehlo.power %xf, %btwo : tensor<1x1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1x1xf32>
    %rs = stablehlo.rsqrt %pe : tensor<1x1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %y = stablehlo.multiply %xf, %b10 : tensor<1x1x8xf32>
    %wf = stablehlo.convert %w : (tensor<8xbf16>) -> tensor<8xf32>
    %bw = stablehlo.broadcast_in_dim %wf, dims = [2]
        : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %sc = stablehlo.multiply %y, %bw : tensor<1x1x8xf32>
    %out = stablehlo.convert %sc : (tensor<1x1x8xf32>) -> tensor<1x1x8xbf16>
    return %out : tensor<1x1x8xbf16>
  }
}
"""

# The same layer with `head_dim` set -- Qwen3's per-head q/k norm, rank 4 and
# normed over the head axis.  Its weight arrives through TWO broadcasts, so
# the peel has to compose them and still land on the last axis.
_NORM_KERAS_HEAD_BF16 = """
module @norm_keras_head_bf16 {
  func.func public @main(%w: tensor<4xbf16>, %x: tensor<1x1x2x4xbf16>)
      -> tensor<1x1x2x4xbf16> {
    %two = stablehlo.constant dense<2> : tensor<i32>
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %xf = stablehlo.convert %x
        : (tensor<1x1x2x4xbf16>) -> tensor<1x1x2x4xf32>
    %twof = stablehlo.convert %two : (tensor<i32>) -> tensor<f32>
    %btwo = stablehlo.broadcast_in_dim %twof, dims = []
        : (tensor<f32>) -> tensor<1x1x2x4xf32>
    %sq = stablehlo.power %xf, %btwo : tensor<1x1x2x4xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [3]
        : (tensor<1x1x2x4xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1, 2]
        : (tensor<1x1x2xf32>) -> tensor<1x1x2x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x2x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x2x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1x2x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1x2x1xf32>
    %rs = stablehlo.rsqrt %pe : tensor<1x1x2x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2, 3]
        : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x4xf32>
    %y = stablehlo.multiply %xf, %b10 : tensor<1x1x2x4xf32>
    %wf = stablehlo.convert %w : (tensor<4xbf16>) -> tensor<4xf32>
    %bw1 = stablehlo.broadcast_in_dim %wf, dims = [3]
        : (tensor<4xf32>) -> tensor<1x1x1x4xf32>
    %bw2 = stablehlo.broadcast_in_dim %bw1, dims = [0, 1, 2, 3]
        : (tensor<1x1x1x4xf32>) -> tensor<1x1x2x4xf32>
    %sc = stablehlo.multiply %y, %bw2 : tensor<1x1x2x4xf32>
    %out = stablehlo.convert %sc
        : (tensor<1x1x2x4xf32>) -> tensor<1x1x2x4xbf16>
    return %out : tensor<1x1x2x4xbf16>
  }
}
"""


# keras-hub qwen3.5 (`qwen3_5_layers.Qwen3_5LayerNorm`) -- row 8's input /
# post-attention / q / k norms, and the biggest single decline in the MoE
# band (717 per generate shape against 60 matches).  The `(1 + w)` scale is
# formed at [N] WIDTH and in f32, over a bf16 parameter, so two things the
# matcher used to refuse meet in one chain: the offset add sits above a
# widening convert (a splat folded in a dtype the weight is not stored in),
# and the value the weight chain reaches is f32 while x is bf16.
#
# Both are reproducible exactly -- the emit casts the weight to f32 before
# adding, and MLX promotes the fused op to f32 and rounds once into bf16,
# which is what the chain does -- and measured at 0 ULP, bit-identical to
# the chain at 128/256/2560/4096/5120.
_NORM_KERAS_Q35_OFF32_BF16 = """
module @norm_keras_q35_off32_bf16 {
  func.func public @main(%w: tensor<8xbf16>, %x: tensor<1x1x8xbf16>)
      -> tensor<1x1x8xbf16> {
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %two = stablehlo.constant dense<2> : tensor<i32>
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %xf = stablehlo.convert %x : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %twof = stablehlo.convert %two : (tensor<i32>) -> tensor<f32>
    %btwo = stablehlo.broadcast_in_dim %twof, dims = []
        : (tensor<f32>) -> tensor<1x1x8xf32>
    %sq = stablehlo.power %xf, %btwo : tensor<1x1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1x1xf32>
    %rs = stablehlo.rsqrt %pe : tensor<1x1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %y = stablehlo.multiply %xf, %b10 : tensor<1x1x8xf32>
    %wf = stablehlo.convert %w : (tensor<8xbf16>) -> tensor<8xf32>
    %bone = stablehlo.broadcast_in_dim %one, dims = []
        : (tensor<f32>) -> tensor<8xf32>
    %scale = stablehlo.add %bone, %wf : tensor<8xf32>
    %bw = stablehlo.broadcast_in_dim %scale, dims = [2]
        : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %sc = stablehlo.multiply %y, %bw : tensor<1x1x8xf32>
    %out = stablehlo.convert %sc : (tensor<1x1x8xf32>) -> tensor<1x1x8xbf16>
    return %out : tensor<1x1x8xbf16>
  }
}
"""

# The same layer at RANK 2, which is the shape row 8 actually decodes in
# (one token, [batch, hidden]) -- the reduce loses the feature axis to a
# rank-1 sum and the mean comes back through a [1, 1] broadcast.
_NORM_KERAS_Q35_OFF32_R2 = """
module @norm_keras_q35_off32_r2 {
  func.func public @main(%w: tensor<8xbf16>, %x: tensor<1x8xbf16>)
      -> tensor<1x8xbf16> {
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %two = stablehlo.constant dense<2> : tensor<i32>
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %xf = stablehlo.convert %x : (tensor<1x8xbf16>) -> tensor<1x8xf32>
    %twof = stablehlo.convert %two : (tensor<i32>) -> tensor<f32>
    %btwo = stablehlo.broadcast_in_dim %twof, dims = []
        : (tensor<f32>) -> tensor<1x8xf32>
    %sq = stablehlo.power %xf, %btwo : tensor<1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [1]
        : (tensor<1x8xf32>, tensor<f32>) -> tensor<1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0]
        : (tensor<1xf32>) -> tensor<1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1xf32>
    %rs = stablehlo.rsqrt %pe : tensor<1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x8xf32>
    %y = stablehlo.multiply %xf, %b10 : tensor<1x8xf32>
    %wf = stablehlo.convert %w : (tensor<8xbf16>) -> tensor<8xf32>
    %bone = stablehlo.broadcast_in_dim %one, dims = []
        : (tensor<f32>) -> tensor<8xf32>
    %scale = stablehlo.add %bone, %wf : tensor<8xf32>
    %bw = stablehlo.broadcast_in_dim %scale, dims = [1]
        : (tensor<8xf32>) -> tensor<1x8xf32>
    %sc = stablehlo.multiply %y, %bw : tensor<1x8xf32>
    %out = stablehlo.convert %sc : (tensor<1x8xf32>) -> tensor<1x8xbf16>
    return %out : tensor<1x8xbf16>
  }
}
"""

# keras-hub gemma4 (`gemma4_layers.RMSNormalization`) -- row 3, every norm in
# the model.  Structurally the keras form above, except that the reciprocal
# square root is spelled `ops.power(var + eps, -0.5)`: jax lowers that
# literally, so row 3's modules carry NO `stablehlo.rsqrt` at all and the
# matcher used to walk away silently ("no rsqrt side" is a quiet reject),
# recognizing zero norms in a 30-layer model.  `chlo.square` arrives as a
# plain multiply, which the matcher already read.
_NORM_GEMMA4_POW_BF16 = """
module @norm_gemma4_pow_bf16 {
  func.func public @main(%w: tensor<8xbf16>, %x: tensor<1x1x8xbf16>)
      -> tensor<1x1x8xbf16> {
    %mhalf = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %xf = stablehlo.convert %x : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %wf = stablehlo.convert %w : (tensor<8xbf16>) -> tensor<8xf32>
    %sq = stablehlo.multiply %xf, %xf : tensor<1x1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1x1xf32>
    %bh = stablehlo.broadcast_in_dim %mhalf, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %rs = stablehlo.power %pe, %bh : tensor<1x1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %y = stablehlo.multiply %xf, %b10 : tensor<1x1x8xf32>
    %bw = stablehlo.broadcast_in_dim %wf, dims = [2]
        : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %sc = stablehlo.multiply %y, %bw : tensor<1x1x8xf32>
    %out = stablehlo.convert %sc : (tensor<1x1x8xf32>) -> tensor<1x1x8xbf16>
    return %out : tensor<1x1x8xbf16>
  }
}
"""

# gemma4's `Gemma4VNorm`: the same power spelling with NO learned scale --
# the value norm in every attention block and the router norm in every MoE
# block.  Fuses to `fast::rms_norm(x, nullopt, eps)`, and it is the cell
# that isolates the power-vs-rsqrt difference from the weight's: measured at
# 0 ULP, bit-identical to the chain at every width.
_NORM_GEMMA4_POW_NOSCALE_BF16 = """
module @norm_gemma4_pow_noscale_bf16 {
  func.func public @main(%x: tensor<1x1x8xbf16>) -> tensor<1x1x8xbf16> {
    %mhalf = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %xf = stablehlo.convert %x : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %sq = stablehlo.multiply %xf, %xf : tensor<1x1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1x1xf32>
    %bh = stablehlo.broadcast_in_dim %mhalf, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %rs = stablehlo.power %pe, %bh : tensor<1x1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %y = stablehlo.multiply %xf, %b10 : tensor<1x1x8xf32>
    %out = stablehlo.convert %y : (tensor<1x1x8xf32>) -> tensor<1x1x8xbf16>
    return %out : tensor<1x1x8xbf16>
  }
}
"""

# The promotion rule's OTHER side, and the reason it is a rule rather than a
# removed check: this is the gemma4 chain with its final downcast taken away,
# so a bf16 x and a bf16 scale feed an f32 result.  MLX would type the fused
# op `result_type(bf16, bf16)` = bf16 and compute the whole norm there, then
# widen -- ~256 ULP of the f32 the chain actually produced.  It must NOT
# fuse, and the count in `every_spelling_fires` is what proves it: a rewrite
# that took it would show one match too many.
_NORM_MIXED_NARROW_DECLINE = """
module @norm_mixed_narrow_decline {
  func.func public @main(%w: tensor<8xbf16>, %x: tensor<1x1x8xbf16>)
      -> tensor<1x1x8xf32> {
    %mhalf = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %xf = stablehlo.convert %x : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %wf = stablehlo.convert %w : (tensor<8xbf16>) -> tensor<8xf32>
    %sq = stablehlo.multiply %xf, %xf : tensor<1x1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1x1xf32>
    %bh = stablehlo.broadcast_in_dim %mhalf, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %rs = stablehlo.power %pe, %bh : tensor<1x1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %y = stablehlo.multiply %xf, %b10 : tensor<1x1x8xf32>
    %bw = stablehlo.broadcast_in_dim %wf, dims = [2]
        : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %sc = stablehlo.multiply %y, %bw : tensor<1x1x8xf32>
    return %sc : tensor<1x1x8xf32>
  }
}
"""

# TWO keras norms in one module, sharing the hoisted `broadcast(eps)`,
# `broadcast(N)` and `broadcast(2)` between them -- which is what a real
# 48-layer model looks like after XLA's parse hoists and CSEs the constants,
# every norm at the model width reaching the same three ops.
#
# Absorbing those would collide the second norm out of the plan and leave the
# first with an intermediate escaping into it, so BOTH would decline: one
# fused norm per model instead of the fifty the rewrite is for.  Nothing
# about the answers would change, which is why this is a test and not a
# tolerance.  metal_norm.cc's `IsConstSubtree` keeps them shareable, and DCE
# drops the ones the rewrite made dead.
_NORM_TWO_STACKED = """
module @norm_two_stacked {
  func.func public @main(%w: tensor<8xbf16>, %v: tensor<8xbf16>,
      %x: tensor<1x1x8xbf16>) -> tensor<1x1x8xbf16> {
    %two = stablehlo.constant dense<2> : tensor<i32>
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %twof = stablehlo.convert %two : (tensor<i32>) -> tensor<f32>
    %btwo = stablehlo.broadcast_in_dim %twof, dims = []
        : (tensor<f32>) -> tensor<1x1x8xf32>
    %bn = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>
    %be = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x1x1xf32>

    %xf = stablehlo.convert %x : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %sq = stablehlo.power %xf, %btwo : tensor<1x1x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b4 = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %div = stablehlo.divide %b4, %bn : tensor<1x1x1xf32>
    %pe = stablehlo.add %div, %be : tensor<1x1x1xf32>
    %rs = stablehlo.rsqrt %pe : tensor<1x1x1xf32>
    %b10 = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %y = stablehlo.multiply %xf, %b10 : tensor<1x1x8xf32>
    %wf = stablehlo.convert %w : (tensor<8xbf16>) -> tensor<8xf32>
    %bw = stablehlo.broadcast_in_dim %wf, dims = [2]
        : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %sc = stablehlo.multiply %y, %bw : tensor<1x1x8xf32>
    %o1 = stablehlo.convert %sc : (tensor<1x1x8xf32>) -> tensor<1x1x8xbf16>

    %xf2 = stablehlo.convert %o1 : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %sq2 = stablehlo.power %xf2, %btwo : tensor<1x1x8xf32>
    %red2 = stablehlo.reduce(%sq2 init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %b42 = stablehlo.broadcast_in_dim %red2, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %div2 = stablehlo.divide %b42, %bn : tensor<1x1x1xf32>
    %pe2 = stablehlo.add %div2, %be : tensor<1x1x1xf32>
    %rs2 = stablehlo.rsqrt %pe2 : tensor<1x1x1xf32>
    %b102 = stablehlo.broadcast_in_dim %rs2, dims = [0, 1, 2]
        : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %y2 = stablehlo.multiply %xf2, %b102 : tensor<1x1x8xf32>
    %vf = stablehlo.convert %v : (tensor<8xbf16>) -> tensor<8xf32>
    %bv = stablehlo.broadcast_in_dim %vf, dims = [2]
        : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %sc2 = stablehlo.multiply %y2, %bv : tensor<1x1x8xf32>
    %o2 = stablehlo.convert %sc2 : (tensor<1x1x8xf32>) -> tensor<1x1x8xbf16>
    return %o2 : tensor<1x1x8xbf16>
  }
}
"""

# ---------------------------------------------------------------------------
# flax NNX (`notes/bonsai-coverage-2026-09-01.md`)
#
# flax emits ONE normalization -- LayerNorm -- and spells RMSNorm as that
# with the mean pinned to a literal 0.0.  The coverage probe measured 251
# norms missed in one process against 49 matched (226 in bonsai's Qwen3-0.6B,
# 25 in ViT-base, all of ViT's being genuine LayerNorms), so this is what
# every nnx-based model in the table normalizes with.
#
# The five modules below are what the PLUGIN is handed for
# `nnx.RMSNorm(8, epsilon=1e-6)` and `nnx.LayerNorm(8, epsilon=1e-6)` --
# captured with METALJAX_DUMP_MODULE=1, saved with the script that produced
# them under ~/.cache/metaljax-bench/logs/flax-norm/.  Four spelling details
# are visible here and in none of the modules above: the zero-mean subtract,
# the scale-first `rsqrt * w` product, the mean divided at the REDUCED rank
# and lifted by a reshape, and the weight reshaped `[N] -> [1,1,N]` before
# its broadcast.

# flax `nnx.RMSNorm`, f32.  The root is the outer multiply.
_NORM_FLAX_RMS_F32 = """
module @norm_flax_rms_f32 {
  func.func public @main(%w: tensor<8xf32>, %x: tensor<1x4x8xf32>)
      -> tensor<1x4x8xf32> {
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<1x4x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
    %nb = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %mean = stablehlo.divide %red, %nb : tensor<1x4xf32>
    %z2 = stablehlo.broadcast_in_dim %zero, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %z3 = stablehlo.reshape %z2 : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %mk = stablehlo.reshape %mean : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %zb = stablehlo.broadcast_in_dim %z3, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %cx = stablehlo.subtract %x, %zb : tensor<1x4x8xf32>
    %eb = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x4x1xf32>
    %ve = stablehlo.add %mk, %eb : tensor<1x4x1xf32>
    %rs = stablehlo.rsqrt %ve : tensor<1x4x1xf32>
    %wr = stablehlo.reshape %w : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %rb = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %wb = stablehlo.broadcast_in_dim %wr, dims = [0, 1, 2]
        : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %scale = stablehlo.multiply %rb, %wb : tensor<1x4x8xf32>
    %o = stablehlo.multiply %cx, %scale : tensor<1x4x8xf32>
    return %o : tensor<1x4x8xf32>
  }
}
"""

# The same in bf16.  Two things only this one has: the root is the downcast
# convert (the whole norm runs in f32), and the square is taken over a
# DIFFERENT `convert %x` than the one the zero-mean subtract reads -- XLA did
# not CSE the two, so the matcher has to compare the values under the
# upcasts rather than by SSA identity.
_NORM_FLAX_RMS_BF16 = """
module @norm_flax_rms_bf16 {
  func.func public @main(%w: tensor<8xbf16>, %x: tensor<1x4x8xbf16>)
      -> tensor<1x4x8xbf16> {
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %xf = stablehlo.convert %x : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %sq = stablehlo.multiply %xf, %xf : tensor<1x4x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
    %nb = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %mean = stablehlo.divide %red, %nb : tensor<1x4xf32>
    %z2 = stablehlo.broadcast_in_dim %zero, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %z3 = stablehlo.reshape %z2 : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %mk = stablehlo.reshape %mean : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %xf2 = stablehlo.convert %x : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %zb = stablehlo.broadcast_in_dim %z3, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %cx = stablehlo.subtract %xf2, %zb : tensor<1x4x8xf32>
    %eb = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x4x1xf32>
    %ve = stablehlo.add %mk, %eb : tensor<1x4x1xf32>
    %rs = stablehlo.rsqrt %ve : tensor<1x4x1xf32>
    %wr = stablehlo.reshape %w : (tensor<8xbf16>) -> tensor<1x1x8xbf16>
    %wf = stablehlo.convert %wr : (tensor<1x1x8xbf16>) -> tensor<1x1x8xf32>
    %rb = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %wb = stablehlo.broadcast_in_dim %wf, dims = [0, 1, 2]
        : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %scale = stablehlo.multiply %rb, %wb : tensor<1x4x8xf32>
    %sc = stablehlo.multiply %cx, %scale : tensor<1x4x8xf32>
    %o = stablehlo.convert %sc : (tensor<1x4x8xf32>) -> tensor<1x4x8xbf16>
    return %o : tensor<1x4x8xbf16>
  }
}
"""

# `nnx.RMSNorm(use_scale=False)`: no weight, so the root IS the normalize
# multiply -- the same weightless form gemma's value/router norms take, but
# reached through the zero-mean subtract.
_NORM_FLAX_RMS_NOSCALE_F32 = """
module @norm_flax_rms_noscale_f32 {
  func.func public @main(%x: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<1x4x8xf32>
    %red = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
    %nb = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %mean = stablehlo.divide %red, %nb : tensor<1x4xf32>
    %z2 = stablehlo.broadcast_in_dim %zero, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %z3 = stablehlo.reshape %z2 : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %mk = stablehlo.reshape %mean : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %zb = stablehlo.broadcast_in_dim %z3, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %cx = stablehlo.subtract %x, %zb : tensor<1x4x8xf32>
    %eb = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x4x1xf32>
    %ve = stablehlo.add %mk, %eb : tensor<1x4x1xf32>
    %rs = stablehlo.rsqrt %ve : tensor<1x4x1xf32>
    %rb = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %o = stablehlo.multiply %cx, %rb : tensor<1x4x8xf32>
    return %o : tensor<1x4x8xf32>
  }
}
"""

# `nnx.LayerNorm`, the real thing: the mean subtracted, the variance as
# `max(0, E[x^2] - E[x]^2)`, and an [N] bias added at the root.  Every ViT in
# the model table normalizes with this.  It rewrites into
# `fast::layer_norm`, whose kernel takes a second pass and accumulates
# `sum((x - mean)^2)` -- the more accurate side of the difference.
_NORM_FLAX_LN_F32 = """
module @norm_flax_ln_f32 {
  func.func public @main(%b: tensor<8xf32>, %w: tensor<8xf32>,
      %x: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<1x4x8xf32>
    %rsum = stablehlo.reduce(%x init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
    %nb = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %mu = stablehlo.divide %rsum, %nb : tensor<1x4xf32>
    %rsq = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
    %nb2 = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %m2 = stablehlo.divide %rsq, %nb2 : tensor<1x4xf32>
    %mm = stablehlo.multiply %mu, %mu : tensor<1x4xf32>
    %vr = stablehlo.subtract %m2, %mm : tensor<1x4xf32>
    %zb0 = stablehlo.broadcast_in_dim %zero, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %var = stablehlo.maximum %zb0, %vr : tensor<1x4xf32>
    %muk = stablehlo.reshape %mu : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %vk = stablehlo.reshape %var : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %mub = stablehlo.broadcast_in_dim %muk, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %cx = stablehlo.subtract %x, %mub : tensor<1x4x8xf32>
    %eb = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x4x1xf32>
    %ve = stablehlo.add %vk, %eb : tensor<1x4x1xf32>
    %rs = stablehlo.rsqrt %ve : tensor<1x4x1xf32>
    %wr = stablehlo.reshape %w : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %rb = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %wb = stablehlo.broadcast_in_dim %wr, dims = [0, 1, 2]
        : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %scale = stablehlo.multiply %rb, %wb : tensor<1x4x8xf32>
    %sc = stablehlo.multiply %cx, %scale : tensor<1x4x8xf32>
    %br = stablehlo.reshape %b : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %bb = stablehlo.broadcast_in_dim %br, dims = [0, 1, 2]
        : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %o = stablehlo.add %sc, %bb : tensor<1x4x8xf32>
    return %o : tensor<1x4x8xf32>
  }
}
"""

# `nnx.LayerNorm(use_bias=False)`: the same without the trailing add, so the
# root is the scale multiply.  Both roots match in the module above -- the
# bias one absorbs the other -- and this is the evidence the smaller form
# stands on its own.
_NORM_FLAX_LN_NOBIAS_F32 = """
module @norm_flax_ln_nobias_f32 {
  func.func public @main(%w: tensor<8xf32>, %x: tensor<1x4x8xf32>)
      -> tensor<1x4x8xf32> {
    %eps = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %n = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sq = stablehlo.multiply %x, %x : tensor<1x4x8xf32>
    %rsum = stablehlo.reduce(%x init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
    %nb = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %mu = stablehlo.divide %rsum, %nb : tensor<1x4xf32>
    %rsq = stablehlo.reduce(%sq init: %zero) applies stablehlo.add
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
    %nb2 = stablehlo.broadcast_in_dim %n, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %m2 = stablehlo.divide %rsq, %nb2 : tensor<1x4xf32>
    %mm = stablehlo.multiply %mu, %mu : tensor<1x4xf32>
    %vr = stablehlo.subtract %m2, %mm : tensor<1x4xf32>
    %zb0 = stablehlo.broadcast_in_dim %zero, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %var = stablehlo.maximum %zb0, %vr : tensor<1x4xf32>
    %muk = stablehlo.reshape %mu : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %vk = stablehlo.reshape %var : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %mub = stablehlo.broadcast_in_dim %muk, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %cx = stablehlo.subtract %x, %mub : tensor<1x4x8xf32>
    %eb = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<1x4x1xf32>
    %ve = stablehlo.add %vk, %eb : tensor<1x4x1xf32>
    %rs = stablehlo.rsqrt %ve : tensor<1x4x1xf32>
    %wr = stablehlo.reshape %w : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %rb = stablehlo.broadcast_in_dim %rs, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %wb = stablehlo.broadcast_in_dim %wr, dims = [0, 1, 2]
        : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %scale = stablehlo.multiply %rb, %wb : tensor<1x4x8xf32>
    %o = stablehlo.multiply %cx, %scale : tensor<1x4x8xf32>
    return %o : tensor<1x4x8xf32>
  }
}
"""


def _norm_forms():
    """(label, module, form tag metal_norm.cc must narrate, dtype, matches).

    `matches` is how many norms the module must fuse -- one each, except the
    stacked pair, which is here because BOTH of them have to, and the
    narrowing-promotion cell, which must fuse NONE (tag `None`).
    """
    return [
        ("maxtext dot", _RMS_NORM_FORM, "R3.dot", "f32", 1),
        ("gemma multiply bf16", _NORM_GEMMA_MUL_BF16, "R3.mul", "bf16", 1),
        ("gemma multiply f32", _NORM_GEMMA_MUL_F32, "R3.mul", "f32", 1),
        ("gemma weightless bf16", _NORM_GEMMA_NOSCALE_BF16, "R3.noscale",
         "bf16", 1),
        ("gemma 1+w offset bf16", _NORM_GEMMA_OFFSET_BF16, "R3.mul+off",
         "bf16", 1),
        ("keras power/downcast bf16", _NORM_KERAS_BF16, "R3.cvt", "bf16", 1),
        ("keras head-dim bf16", _NORM_KERAS_HEAD_BF16, "R4.cvt", "bf16", 1),
        ("keras qwen3.5 1+w in f32", _NORM_KERAS_Q35_OFF32_BF16,
         "R3.cvt.mul+off32+up+dn", "bf16", 1),
        ("keras qwen3.5 1+w in f32, rank 2", _NORM_KERAS_Q35_OFF32_R2,
         "R2.cvt.mul+off32+up+dn", "bf16", 1),
        ("gemma4 power(-0.5) bf16", _NORM_GEMMA4_POW_BF16, "R3.cvt.mul",
         "bf16", 1),
        ("gemma4 power(-0.5) weightless", _NORM_GEMMA4_POW_NOSCALE_BF16,
         "R3.cvt.noscale", "bf16", 1),
        ("narrowing promotion declines", _NORM_MIXED_NARROW_DECLINE, None,
         "f32", 0),
        ("two norms, shared constants", _NORM_TWO_STACKED, "R3.cvt", "bf16",
         2),
        ("flax rms f32", _NORM_FLAX_RMS_F32, "R3.scale", "f32", 1),
        ("flax rms bf16", _NORM_FLAX_RMS_BF16, "R3.cvt.scale", "bf16", 1),
        ("flax rms weightless f32", _NORM_FLAX_RMS_NOSCALE_F32, "R3.noscale",
         "f32", 1),
        ("flax layernorm f32", _NORM_FLAX_LN_F32, "R3.scale.ln+b", "f32", 1),
        ("flax layernorm no bias f32", _NORM_FLAX_LN_NOBIAS_F32,
         "R3.scale.ln", "f32", 1),
    ]


def _norm_inputs(text):
    """Deterministic inputs for one norm module, read off @main's signature.

    The [N] operand is the learned scale, so it is drawn around 1 the way a
    trained norm's is -- a weight centred on 0 would hide a dropped weight.
    """
    import re as _re
    import ml_dtypes
    sig = _re.search(r"func\.func public @main\((.*?)\)\s*->", text,
                     _re.S).group(1)
    rng = np.random.default_rng(3011)
    args = []
    for a in _re.findall(r"tensor<([^>]*)>", sig):
        *dims, dt = a.split("x")
        shape = tuple(int(d) for d in dims)
        v = rng.standard_normal(shape)
        if len(shape) == 1:
            v = 1.0 + 0.05 * v
        args.append(v.astype({"f32": np.float32,
                              "bf16": ml_dtypes.bfloat16}[dt]))
    return args


# P30: maxtext's MLA multi-span decode attention, exactly as the captured
# row-10 module spells it (f32, tiny dims), through metal_mla.cc's fused
# rewrite on metal and the literal two-span combine on CPU.
_MLA_TWO_SPAN = """
module @mla_two_span {
  func.func public @main(%q: tensor<1x1x2x4xf32>,
      %kp: tensor<1x4x2x4xf32>, %vp: tensor<1x4x2x4xf32>,
      %sp: tensor<1x4xi32>,
      %ka: tensor<1x2x2x4xf32>, %va: tensor<1x2x2x4xf32>,
      %sa: tensor<1x2xi32>) -> tensor<1x1x2x4xf32> {
    %one = stablehlo.constant dense<1> : tensor<i32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %neg = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %half = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %q5 = stablehlo.reshape %q
        : (tensor<1x1x2x4xf32>) -> tensor<1x1x2x1x4xf32>

    %d1p = stablehlo.dot_general %kp, %q5,
        batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4]
        : (tensor<1x4x2x4xf32>, tensor<1x1x2x1x4xf32>)
        -> tensor<1x2x4x1x1xf32>
    %scp = stablehlo.transpose %d1p, dims = [0, 1, 4, 3, 2]
        : (tensor<1x2x4x1x1xf32>) -> tensor<1x2x1x1x4xf32>
    %sbp = stablehlo.broadcast_in_dim %sp, dims = [0, 4]
        : (tensor<1x4xi32>) -> tensor<1x1x1x1x4xi32>
    %obp = stablehlo.broadcast_in_dim %one, dims = []
        : (tensor<i32>) -> tensor<1x1x1x1x4xi32>
    %eqp = stablehlo.compare EQ, %sbp, %obp, SIGNED
        : (tensor<1x1x1x1x4xi32>, tensor<1x1x1x1x4xi32>)
        -> tensor<1x1x1x1x4xi1>
    %w16p = func.call @mla_w16p(%eqp, %zero, %neg)
        : (tensor<1x1x1x1x4xi1>, tensor<f32>, tensor<f32>)
        -> tensor<1x1x1x1x4xf32>
    %thp = stablehlo.broadcast_in_dim %half, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x4xf32>
    %gep = stablehlo.compare GE, %w16p, %thp, FLOAT
        : (tensor<1x1x1x1x4xf32>, tensor<1x1x1x1x4xf32>)
        -> tensor<1x1x1x1x4xi1>
    %mkp = func.call @mla_w17p(%gep, %scp, %neg)
        : (tensor<1x1x1x1x4xi1>, tensor<1x2x1x1x4xf32>, tensor<f32>)
        -> tensor<1x2x1x1x4xf32>
    %m4p = stablehlo.reshape %mkp
        : (tensor<1x2x1x1x4xf32>) -> tensor<1x2x1x4xf32>
    %rmp = stablehlo.reduce(%m4p init: %ninf) applies stablehlo.maximum
        across dimensions = [3]
        : (tensor<1x2x1x4xf32>, tensor<f32>) -> tensor<1x2x1xf32>
    %bmp = stablehlo.broadcast_in_dim %rmp, dims = [0, 1, 2]
        : (tensor<1x2x1xf32>) -> tensor<1x2x1x1xf32>
    %bm4p = stablehlo.broadcast_in_dim %bmp, dims = [0, 1, 2, 3]
        : (tensor<1x2x1x1xf32>) -> tensor<1x2x1x4xf32>
    %sup = stablehlo.subtract %m4p, %bm4p : tensor<1x2x1x4xf32>
    %exp = stablehlo.exponential %sup : tensor<1x2x1x4xf32>
    %rsp = stablehlo.reduce(%exp init: %zero) applies stablehlo.add
        across dimensions = [3]
        : (tensor<1x2x1x4xf32>, tensor<f32>) -> tensor<1x2x1xf32>
    %bsp = stablehlo.broadcast_in_dim %rsp, dims = [0, 1, 2]
        : (tensor<1x2x1xf32>) -> tensor<1x2x1x1xf32>
    %mtp = stablehlo.transpose %bmp, dims = [0, 2, 1, 3]
        : (tensor<1x2x1x1xf32>) -> tensor<1x1x2x1xf32>
    %ltp = stablehlo.transpose %bsp, dims = [0, 2, 1, 3]
        : (tensor<1x2x1x1xf32>) -> tensor<1x1x2x1xf32>
    %e5p = stablehlo.reshape %exp
        : (tensor<1x2x1x4xf32>) -> tensor<1x2x1x1x4xf32>
    %d2p = stablehlo.dot_general %vp, %e5p,
        batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4]
        : (tensor<1x4x2x4xf32>, tensor<1x2x1x1x4xf32>)
        -> tensor<1x2x4x1x1xf32>
    %otp = stablehlo.transpose %d2p, dims = [0, 4, 1, 3, 2]
        : (tensor<1x2x4x1x1xf32>) -> tensor<1x1x2x1x4xf32>
    %op = stablehlo.reshape %otp
        : (tensor<1x1x2x1x4xf32>) -> tensor<1x1x2x4xf32>

    %d1a = stablehlo.dot_general %ka, %q5,
        batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4]
        : (tensor<1x2x2x4xf32>, tensor<1x1x2x1x4xf32>)
        -> tensor<1x2x2x1x1xf32>
    %sca = stablehlo.transpose %d1a, dims = [0, 1, 4, 3, 2]
        : (tensor<1x2x2x1x1xf32>) -> tensor<1x2x1x1x2xf32>
    %sba = stablehlo.broadcast_in_dim %sa, dims = [0, 4]
        : (tensor<1x2xi32>) -> tensor<1x1x1x1x2xi32>
    %oba = stablehlo.broadcast_in_dim %one, dims = []
        : (tensor<i32>) -> tensor<1x1x1x1x2xi32>
    %eqa = stablehlo.compare EQ, %sba, %oba, SIGNED
        : (tensor<1x1x1x1x2xi32>, tensor<1x1x1x1x2xi32>)
        -> tensor<1x1x1x1x2xi1>
    %w16a = func.call @mla_w16a(%eqa, %zero, %neg)
        : (tensor<1x1x1x1x2xi1>, tensor<f32>, tensor<f32>)
        -> tensor<1x1x1x1x2xf32>
    %tha = stablehlo.broadcast_in_dim %half, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x2xf32>
    %gea = stablehlo.compare GE, %w16a, %tha, FLOAT
        : (tensor<1x1x1x1x2xf32>, tensor<1x1x1x1x2xf32>)
        -> tensor<1x1x1x1x2xi1>
    %mka = func.call @mla_w17a(%gea, %sca, %neg)
        : (tensor<1x1x1x1x2xi1>, tensor<1x2x1x1x2xf32>, tensor<f32>)
        -> tensor<1x2x1x1x2xf32>
    %m4a = stablehlo.reshape %mka
        : (tensor<1x2x1x1x2xf32>) -> tensor<1x2x1x2xf32>
    %rma = stablehlo.reduce(%m4a init: %ninf) applies stablehlo.maximum
        across dimensions = [3]
        : (tensor<1x2x1x2xf32>, tensor<f32>) -> tensor<1x2x1xf32>
    %bma = stablehlo.broadcast_in_dim %rma, dims = [0, 1, 2]
        : (tensor<1x2x1xf32>) -> tensor<1x2x1x1xf32>
    %bm4a = stablehlo.broadcast_in_dim %bma, dims = [0, 1, 2, 3]
        : (tensor<1x2x1x1xf32>) -> tensor<1x2x1x2xf32>
    %sua = stablehlo.subtract %m4a, %bm4a : tensor<1x2x1x2xf32>
    %exa = stablehlo.exponential %sua : tensor<1x2x1x2xf32>
    %rsa = stablehlo.reduce(%exa init: %zero) applies stablehlo.add
        across dimensions = [3]
        : (tensor<1x2x1x2xf32>, tensor<f32>) -> tensor<1x2x1xf32>
    %bsa = stablehlo.broadcast_in_dim %rsa, dims = [0, 1, 2]
        : (tensor<1x2x1xf32>) -> tensor<1x2x1x1xf32>
    %mta = stablehlo.transpose %bma, dims = [0, 2, 1, 3]
        : (tensor<1x2x1x1xf32>) -> tensor<1x1x2x1xf32>
    %lta = stablehlo.transpose %bsa, dims = [0, 2, 1, 3]
        : (tensor<1x2x1x1xf32>) -> tensor<1x1x2x1xf32>
    %e5a = stablehlo.reshape %exa
        : (tensor<1x2x1x2xf32>) -> tensor<1x2x1x1x2xf32>
    %d2a = stablehlo.dot_general %va, %e5a,
        batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4]
        : (tensor<1x2x2x4xf32>, tensor<1x2x1x1x2xf32>)
        -> tensor<1x2x4x1x1xf32>
    %ota = stablehlo.transpose %d2a, dims = [0, 4, 1, 3, 2]
        : (tensor<1x2x4x1x1xf32>) -> tensor<1x1x2x1x4xf32>
    %oa = stablehlo.reshape %ota
        : (tensor<1x1x2x1x4xf32>) -> tensor<1x1x2x4xf32>

    %m = stablehlo.maximum %mtp, %mta : tensor<1x1x2x1xf32>
    %dp = stablehlo.subtract %mtp, %m : tensor<1x1x2x1xf32>
    %e1p = stablehlo.exponential %dp : tensor<1x1x2x1xf32>
    %tp = stablehlo.multiply %e1p, %ltp : tensor<1x1x2x1xf32>
    %zt = stablehlo.broadcast_in_dim %zero, dims = []
        : (tensor<f32>) -> tensor<1x1x2x1xf32>
    %s1 = stablehlo.add %zt, %tp : tensor<1x1x2x1xf32>
    %da = stablehlo.subtract %mta, %m : tensor<1x1x2x1xf32>
    %e1a = stablehlo.exponential %da : tensor<1x1x2x1xf32>
    %ta = stablehlo.multiply %e1a, %lta : tensor<1x1x2x1xf32>
    %l = stablehlo.add %s1, %ta : tensor<1x1x2x1xf32>
    %dp2 = stablehlo.subtract %mtp, %m : tensor<1x1x2x1xf32>
    %e2p = stablehlo.exponential %dp2 : tensor<1x1x2x1xf32>
    %wp = stablehlo.divide %e2p, %l : tensor<1x1x2x1xf32>
    %wbp = stablehlo.broadcast_in_dim %wp, dims = [0, 1, 2, 3]
        : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x4xf32>
    %wsp = stablehlo.multiply %wbp, %op : tensor<1x1x2x4xf32>
    %zo = stablehlo.broadcast_in_dim %zero, dims = []
        : (tensor<f32>) -> tensor<1x1x2x4xf32>
    %acc = stablehlo.add %zo, %wsp : tensor<1x1x2x4xf32>
    %da2 = stablehlo.subtract %mta, %m : tensor<1x1x2x1xf32>
    %e2a = stablehlo.exponential %da2 : tensor<1x1x2x1xf32>
    %wa = stablehlo.divide %e2a, %l : tensor<1x1x2x1xf32>
    %wba = stablehlo.broadcast_in_dim %wa, dims = [0, 1, 2, 3]
        : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x4xf32>
    %wsa = stablehlo.multiply %wba, %oa : tensor<1x1x2x4xf32>
    %root = stablehlo.add %acc, %wsa : tensor<1x1x2x4xf32>
    return %root : tensor<1x1x2x4xf32>
  }
  func.func private @mla_w16p(%p: tensor<1x1x1x1x4xi1>, %t: tensor<f32>,
      %f: tensor<f32>) -> tensor<1x1x1x1x4xf32> {
    %bt = stablehlo.broadcast_in_dim %t, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x4xf32>
    %bf = stablehlo.broadcast_in_dim %f, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x4xf32>
    %s = stablehlo.select %p, %bt, %bf
        : tensor<1x1x1x1x4xi1>, tensor<1x1x1x1x4xf32>
    return %s : tensor<1x1x1x1x4xf32>
  }
  func.func private @mla_w17p(%p: tensor<1x1x1x1x4xi1>,
      %sc: tensor<1x2x1x1x4xf32>, %f: tensor<f32>) -> tensor<1x2x1x1x4xf32> {
    %bp = stablehlo.broadcast_in_dim %p, dims = [0, 1, 2, 3, 4]
        : (tensor<1x1x1x1x4xi1>) -> tensor<1x2x1x1x4xi1>
    %bf = stablehlo.broadcast_in_dim %f, dims = []
        : (tensor<f32>) -> tensor<1x2x1x1x4xf32>
    %s = stablehlo.select %bp, %sc, %bf
        : tensor<1x2x1x1x4xi1>, tensor<1x2x1x1x4xf32>
    return %s : tensor<1x2x1x1x4xf32>
  }
  func.func private @mla_w16a(%p: tensor<1x1x1x1x2xi1>, %t: tensor<f32>,
      %f: tensor<f32>) -> tensor<1x1x1x1x2xf32> {
    %bt = stablehlo.broadcast_in_dim %t, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x2xf32>
    %bf = stablehlo.broadcast_in_dim %f, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x2xf32>
    %s = stablehlo.select %p, %bt, %bf
        : tensor<1x1x1x1x2xi1>, tensor<1x1x1x1x2xf32>
    return %s : tensor<1x1x1x1x2xf32>
  }
  func.func private @mla_w17a(%p: tensor<1x1x1x1x2xi1>,
      %sc: tensor<1x2x1x1x2xf32>, %f: tensor<f32>) -> tensor<1x2x1x1x2xf32> {
    %bp = stablehlo.broadcast_in_dim %p, dims = [0, 1, 2, 3, 4]
        : (tensor<1x1x1x1x2xi1>) -> tensor<1x2x1x1x2xi1>
    %bf = stablehlo.broadcast_in_dim %f, dims = []
        : (tensor<f32>) -> tensor<1x2x1x1x2xf32>
    %s = stablehlo.select %bp, %sc, %bf
        : tensor<1x2x1x1x2xi1>, tensor<1x2x1x1x2xf32>
    return %s : tensor<1x2x1x1x2xf32>
  }
}
"""


def _mla_gqa_two_span(dt, acc, d=4, t=(4, 2)):
    """maxtext's GROUPED-QUERY decode attention, as row 11 spells it.

    Same two-span flash combine as `_MLA_TWO_SPAN`, with the one difference
    that made `AnalyzeMla` decline on Qwen3-0.6B: H query heads over Hkv < H
    KV heads.  q is reshaped [B,1,H,D] -> [B,1,Hkv,G,D] before the scores
    dot, the scores come back [B,Hkv,T,1,G] and transpose to [B,Hkv,G,1,T],
    and a reshape merges dims 1,2 into [B,H,1,T] for the softmax; the values
    dot and the output reshape mirror it.  Shrunk to B=1, Hkv=2, G=2 (so
    H=4), D=Dv=4, spans of 4 and 2; the model narrates
    `B1H16Hkv8D128Dv128T64+72` -- 8 KV heads, 2 groups, D=128, a 64-long
    prefill span and an autoregressive span the size of the decode budget.

    `acc` is the dtype the probability SUM accumulates in.  bf16 attention
    upcasts for it and converts the result back (the captured spelling,
    which the matcher must peel); passing acc == dt leaves identity
    converts, which is the f32 arm.

    `d` is the head dim, and it decides which MLX kernel the emit reaches:
    the fused sdpa VECTOR kernel wants D in {64, 96, 128, 256} (or 192/128),
    so d=4 keeps the module small and lands on MLX's unfused fallback --
    which tests the recognizer and the emit's plumbing -- while d=64 is the
    path row 11 actually takes, GQA head repeat inside the kernel included.

    `t` is the pair of span lengths (prefill, autoregressive).  The B3
    two-span kernel hands key i to simdgroup i % 32, so spans whose SUM
    passes 32 (and is not a multiple of it) are what exercise more than one
    key per simdgroup and a ragged last round.
    """
    T0, T1 = t
    ninf = {"f32": "0xFF800000", "bf16": "0xFF80"}[dt]
    # Two spans, identical but for their key length; the callee names have
    # to differ because their shapes do.
    def span(tag, T, k, v, seg):
        return f"""
    %q5{tag} = stablehlo.reshape %q
        : (tensor<1x1x4x{d}x{dt}>) -> tensor<1x1x2x2x{d}x{dt}>
    %d1{tag} = stablehlo.dot_general {k}, %q5{tag},
        batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4]
        : (tensor<1x{T}x2x{d}x{dt}>, tensor<1x1x2x2x{d}x{dt}>)
        -> tensor<1x2x{T}x1x2x{dt}>
    %sc{tag} = stablehlo.transpose %d1{tag}, dims = [0, 1, 4, 3, 2]
        : (tensor<1x2x{T}x1x2x{dt}>) -> tensor<1x2x2x1x{T}x{dt}>
    %sb{tag} = stablehlo.broadcast_in_dim {seg}, dims = [0, 4]
        : (tensor<1x{T}xi32>) -> tensor<1x1x1x1x{T}xi32>
    %ob{tag} = stablehlo.broadcast_in_dim %one, dims = []
        : (tensor<i32>) -> tensor<1x1x1x1x{T}xi32>
    %eq{tag} = stablehlo.compare EQ, %sb{tag}, %ob{tag}, SIGNED
        : (tensor<1x1x1x1x{T}xi32>, tensor<1x1x1x1x{T}xi32>)
        -> tensor<1x1x1x1x{T}xi1>
    %w16{tag} = func.call @gqa_w16{tag}(%eq{tag}, %zero, %neg)
        : (tensor<1x1x1x1x{T}xi1>, tensor<f32>, tensor<f32>)
        -> tensor<1x1x1x1x{T}xf32>
    %th{tag} = stablehlo.broadcast_in_dim %half, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x{T}xf32>
    %ge{tag} = stablehlo.compare GE, %w16{tag}, %th{tag}, FLOAT
        : (tensor<1x1x1x1x{T}xf32>, tensor<1x1x1x1x{T}xf32>)
        -> tensor<1x1x1x1x{T}xi1>
    %mk{tag} = func.call @gqa_w17{tag}(%ge{tag}, %sc{tag}, %neg)
        : (tensor<1x1x1x1x{T}xi1>, tensor<1x2x2x1x{T}x{dt}>, tensor<f32>)
        -> tensor<1x2x2x1x{T}x{dt}>
    %m4{tag} = stablehlo.reshape %mk{tag}
        : (tensor<1x2x2x1x{T}x{dt}>) -> tensor<1x4x1x{T}x{dt}>
    %rm{tag} = stablehlo.reduce(%m4{tag} init: %ninf)
        applies stablehlo.maximum across dimensions = [3]
        : (tensor<1x4x1x{T}x{dt}>, tensor<{dt}>) -> tensor<1x4x1x{dt}>
    %bm{tag} = stablehlo.broadcast_in_dim %rm{tag}, dims = [0, 1, 2]
        : (tensor<1x4x1x{dt}>) -> tensor<1x4x1x1x{dt}>
    %bm4{tag} = stablehlo.broadcast_in_dim %bm{tag}, dims = [0, 1, 2, 3]
        : (tensor<1x4x1x1x{dt}>) -> tensor<1x4x1x{T}x{dt}>
    %su{tag} = stablehlo.subtract %m4{tag}, %bm4{tag}
        : tensor<1x4x1x{T}x{dt}>
    %ex{tag} = stablehlo.exponential %su{tag} : tensor<1x4x1x{T}x{dt}>
    %ea{tag} = stablehlo.convert %ex{tag}
        : (tensor<1x4x1x{T}x{dt}>) -> tensor<1x4x1x{T}x{acc}>
    %rs{tag} = stablehlo.reduce(%ea{tag} init: %zacc)
        applies stablehlo.add across dimensions = [3]
        : (tensor<1x4x1x{T}x{acc}>, tensor<{acc}>) -> tensor<1x4x1x{acc}>
    %bs{tag} = stablehlo.broadcast_in_dim %rs{tag}, dims = [0, 1, 2]
        : (tensor<1x4x1x{acc}>) -> tensor<1x4x1x1x{acc}>
    %bd{tag} = stablehlo.convert %bs{tag}
        : (tensor<1x4x1x1x{acc}>) -> tensor<1x4x1x1x{dt}>
    %mt{tag} = stablehlo.transpose %bm{tag}, dims = [0, 2, 1, 3]
        : (tensor<1x4x1x1x{dt}>) -> tensor<1x1x4x1x{dt}>
    %lt{tag} = stablehlo.transpose %bd{tag}, dims = [0, 2, 1, 3]
        : (tensor<1x4x1x1x{dt}>) -> tensor<1x1x4x1x{dt}>
    %e5{tag} = stablehlo.reshape %ex{tag}
        : (tensor<1x4x1x{T}x{dt}>) -> tensor<1x2x2x1x{T}x{dt}>
    %d2{tag} = stablehlo.dot_general {v}, %e5{tag},
        batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4]
        : (tensor<1x{T}x2x{d}x{dt}>, tensor<1x2x2x1x{T}x{dt}>)
        -> tensor<1x2x{d}x2x1x{dt}>
    %ot{tag} = stablehlo.transpose %d2{tag}, dims = [0, 4, 1, 3, 2]
        : (tensor<1x2x{d}x2x1x{dt}>) -> tensor<1x1x2x2x{d}x{dt}>
    %o{tag} = stablehlo.reshape %ot{tag}
        : (tensor<1x1x2x2x{d}x{dt}>) -> tensor<1x1x4x{d}x{dt}>
"""

    def where_callees(tag, T):
        return f"""
  func.func private @gqa_w16{tag}(%p: tensor<1x1x1x1x{T}xi1>,
      %t: tensor<f32>, %f: tensor<f32>) -> tensor<1x1x1x1x{T}xf32> {{
    %bt = stablehlo.broadcast_in_dim %t, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x{T}xf32>
    %bf = stablehlo.broadcast_in_dim %f, dims = []
        : (tensor<f32>) -> tensor<1x1x1x1x{T}xf32>
    %s = stablehlo.select %p, %bt, %bf
        : tensor<1x1x1x1x{T}xi1>, tensor<1x1x1x1x{T}xf32>
    return %s : tensor<1x1x1x1x{T}xf32>
  }}
  func.func private @gqa_w17{tag}(%p: tensor<1x1x1x1x{T}xi1>,
      %sc: tensor<1x2x2x1x{T}x{dt}>, %f: tensor<f32>)
      -> tensor<1x2x2x1x{T}x{dt}> {{
    %fc = stablehlo.convert %f : (tensor<f32>) -> tensor<{dt}>
    %bp = stablehlo.broadcast_in_dim %p, dims = [0, 1, 2, 3, 4]
        : (tensor<1x1x1x1x{T}xi1>) -> tensor<1x2x2x1x{T}xi1>
    %bf = stablehlo.broadcast_in_dim %fc, dims = []
        : (tensor<{dt}>) -> tensor<1x2x2x1x{T}x{dt}>
    %s = stablehlo.select %bp, %sc, %bf
        : tensor<1x2x2x1x{T}xi1>, tensor<1x2x2x1x{T}x{dt}>
    return %s : tensor<1x2x2x1x{T}x{dt}>
  }}
"""

    return f"""
module @mla_gqa_two_span {{
  func.func public @main(%q: tensor<1x1x4x{d}x{dt}>,
      %kp: tensor<1x{T0}x2x{d}x{dt}>, %vp: tensor<1x{T0}x2x{d}x{dt}>,
      %sp: tensor<1x{T0}xi32>,
      %ka: tensor<1x{T1}x2x{d}x{dt}>, %va: tensor<1x{T1}x2x{d}x{dt}>,
      %sa: tensor<1x{T1}xi32>) -> tensor<1x1x4x{d}x{dt}> {{
    %one = stablehlo.constant dense<1> : tensor<i32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zacc = stablehlo.constant dense<0.000000e+00> : tensor<{acc}>
    %zdt = stablehlo.constant dense<0.000000e+00> : tensor<{dt}>
    %neg = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %half = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %ninf = stablehlo.constant dense<{ninf}> : tensor<{dt}>
{span("p", T0, "%kp", "%vp", "%sp")}
{span("a", T1, "%ka", "%va", "%sa")}
    %m = stablehlo.maximum %mtp, %mta : tensor<1x1x4x1x{dt}>
    %dp = stablehlo.subtract %mtp, %m : tensor<1x1x4x1x{dt}>
    %e1p = stablehlo.exponential %dp : tensor<1x1x4x1x{dt}>
    %tp = stablehlo.multiply %e1p, %ltp : tensor<1x1x4x1x{dt}>
    %zt = stablehlo.broadcast_in_dim %zdt, dims = []
        : (tensor<{dt}>) -> tensor<1x1x4x1x{dt}>
    %s1 = stablehlo.add %zt, %tp : tensor<1x1x4x1x{dt}>
    %da = stablehlo.subtract %mta, %m : tensor<1x1x4x1x{dt}>
    %e1a = stablehlo.exponential %da : tensor<1x1x4x1x{dt}>
    %ta = stablehlo.multiply %e1a, %lta : tensor<1x1x4x1x{dt}>
    %l = stablehlo.add %s1, %ta : tensor<1x1x4x1x{dt}>
    %dp2 = stablehlo.subtract %mtp, %m : tensor<1x1x4x1x{dt}>
    %e2p = stablehlo.exponential %dp2 : tensor<1x1x4x1x{dt}>
    %wp = stablehlo.divide %e2p, %l : tensor<1x1x4x1x{dt}>
    %wbp = stablehlo.broadcast_in_dim %wp, dims = [0, 1, 2, 3]
        : (tensor<1x1x4x1x{dt}>) -> tensor<1x1x4x{d}x{dt}>
    %wsp = stablehlo.multiply %wbp, %op : tensor<1x1x4x{d}x{dt}>
    %zo = stablehlo.broadcast_in_dim %zdt, dims = []
        : (tensor<{dt}>) -> tensor<1x1x4x{d}x{dt}>
    %acc = stablehlo.add %zo, %wsp : tensor<1x1x4x{d}x{dt}>
    %da2 = stablehlo.subtract %mta, %m : tensor<1x1x4x1x{dt}>
    %e2a = stablehlo.exponential %da2 : tensor<1x1x4x1x{dt}>
    %wa = stablehlo.divide %e2a, %l : tensor<1x1x4x1x{dt}>
    %wba = stablehlo.broadcast_in_dim %wa, dims = [0, 1, 2, 3]
        : (tensor<1x1x4x1x{dt}>) -> tensor<1x1x4x{d}x{dt}>
    %wsa = stablehlo.multiply %wba, %oa : tensor<1x1x4x{d}x{dt}>
    %root = stablehlo.add %acc, %wsa : tensor<1x1x4x{d}x{dt}>
    return %root : tensor<1x1x4x{d}x{dt}>
  }}
{where_callees("p", T0)}{where_callees("a", T1)}}}
"""


_MLA_GQA_TWO_SPAN_F32 = _mla_gqa_two_span("f32", "f32")
_MLA_GQA_TWO_SPAN_BF16 = _mla_gqa_two_span("bf16", "f32")
# D = 64: MLX's fused sdpa VECTOR kernel takes this one, so the GQA head
# repeat happens inside the kernel rather than in its fallback -- the path
# row 11 runs (D = 128 there).
_MLA_GQA_TWO_SPAN_BF16_D64 = _mla_gqa_two_span("bf16", "f32", d=64)
# B3: the two-span KERNEL's own forms (runtime/mla.cc).  Spans past 32 keys
# put several keys on one simdgroup with a ragged last round, at both dtypes
# the kernel names; D = 32 is the smallest head dim the lane split takes.
_MLA_KERNEL_BF16_D64_T40_27 = _mla_gqa_two_span("bf16", "f32", d=64,
                                                t=(40, 27))
_MLA_KERNEL_F32_D64_T33_5 = _mla_gqa_two_span("f32", "f32", d=64, t=(33, 5))


# --------------------------------------------------------------------------
# the gated-delta-net decode step (metal_gdn.cc)
# --------------------------------------------------------------------------
#
# CAPTURED, not hand-written: these are keras-hub 0.30.0's own
# `Qwen3_5GatedDeltaNet` traced under the jax backend at `seq_len == 1` with
# a cache present -- the branch row 8 (Qwen3.6-35B-A3B) and row 21
# (Qwen3.8-27B) decode through -- lowered on CPU at TINY extents (1 key head
# lifted to 2 value heads by keras' `ops.repeat`, head dims 8, hidden 16,
# conv kernel 4).  The spelling is the row-8 module's: only the extents
# differ, which is what makes these a fixture for the recognizer rather than
# a restatement of it.  The capture script is
# `~/.cache/metaljax-bench/logs/gdn-fuse/capture_gdn.py`.
#
# The f32 form is the plain one; the bf16 form is what the real checkpoints
# run and carries the two things the f32 spelling hides -- the `_l2norm`
# whose rsqrt and product round to bf16, and keras' autocast of the f32
# recurrent cache, a `convert(f32->bf16) -> convert(bf16->f32)` round trip
# on 2.1 MB per layer that computes nothing and still ROUNDS.  The kernel
# replays both.
_GDN_TINY_F32 = r"""module @jit_fn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<32x4xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<16x32xf32>, %arg4: tensor<16x16xf32>, %arg5: tensor<16x2xf32>, %arg6: tensor<16x2xf32>, %arg7: tensor<8xf32>, %arg8: tensor<16x16xf32>, %arg9: tensor<1x1x16xf32>, %arg10: tensor<1x32x3xf32>, %arg11: tensor<1x2x8x8xf32>) -> (tensor<1x1x16xf32> {jax.result_info = "result[0]"}, tensor<1x32x3xf32> {jax.result_info = "result[1]"}, tensor<1x2x8x8xf32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.dot_general %arg9, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xf32>, tensor<16x32xf32>) -> tensor<1x1x32xf32>
    %1 = stablehlo.dot_general %arg9, %arg4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xf32>, tensor<16x16xf32>) -> tensor<1x1x16xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x1x16xf32>) -> tensor<1x1x2x8xf32>
    %3 = stablehlo.dot_general %arg9, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xf32>, tensor<16x2xf32>) -> tensor<1x1x2xf32>
    %4 = stablehlo.dot_general %arg9, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xf32>, tensor<16x2xf32>) -> tensor<1x1x2xf32>
    %5 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<1x1x32xf32>) -> tensor<1x32x1xf32>
    %6 = stablehlo.concatenate %arg10, %5, dim = 2 : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x4xf32>
    %7 = stablehlo.slice %6 [0:1, 0:32, 1:4] : (tensor<1x32x4xf32>) -> tensor<1x32x3xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<32x4xf32>) -> tensor<1x32x4xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x32x4xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x32x4xf32>, tensor<f32>) -> tensor<1x32xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %12 = stablehlo.negate %11 : tensor<1x32x1xf32>
    %13 = stablehlo.exponential %12 : tensor<1x32x1xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x1xf32>
    %15 = stablehlo.add %14, %13 : tensor<1x32x1xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x32x1xf32>
    %17 = stablehlo.divide %16, %15 : tensor<1x32x1xf32>
    %18 = stablehlo.multiply %11, %17 : tensor<1x32x1xf32>
    %19 = stablehlo.transpose %18, dims = [0, 2, 1] : (tensor<1x32x1xf32>) -> tensor<1x1x32xf32>
    %20 = stablehlo.slice %19 [0:1, 0:1, 0:8] : (tensor<1x1x32xf32>) -> tensor<1x1x8xf32>
    %21 = stablehlo.slice %19 [0:1, 0:1, 8:16] : (tensor<1x1x32xf32>) -> tensor<1x1x8xf32>
    %22 = stablehlo.slice %19 [0:1, 0:1, 16:32] : (tensor<1x1x32xf32>) -> tensor<1x1x16xf32>
    %23 = stablehlo.reshape %20 : (tensor<1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %24 = stablehlo.reshape %21 : (tensor<1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %25 = stablehlo.reshape %22 : (tensor<1x1x16xf32>) -> tensor<1x1x2x8xf32>
    %26 = stablehlo.negate %3 : tensor<1x1x2xf32>
    %27 = stablehlo.exponential %26 : tensor<1x1x2xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %29 = stablehlo.add %28, %27 : tensor<1x1x2xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %31 = stablehlo.divide %30, %29 : tensor<1x1x2xf32>
    %32 = stablehlo.exponential %arg2 : tensor<2xf32>
    %33 = stablehlo.negate %32 : tensor<2xf32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2xf32>) -> tensor<1x1x2xf32>
    %35 = stablehlo.add %4, %34 : tensor<1x1x2xf32>
    %36 = call @softplus(%35) : (tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    %37 = stablehlo.broadcast_in_dim %33, dims = [2] : (tensor<2xf32>) -> tensor<1x1x2xf32>
    %38 = stablehlo.multiply %37, %36 : tensor<1x1x2xf32>
    %39 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 4] : (tensor<1x1x1x8xf32>) -> tensor<1x1x1x2x8xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
    %41 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2, 4] : (tensor<1x1x1x8xf32>) -> tensor<1x1x1x2x8xf32>
    %42 = stablehlo.reshape %41 : (tensor<1x1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
    %43 = stablehlo.multiply %40, %40 : tensor<1x1x2x8xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.reduce(%43 init: %cst_4) applies stablehlo.add across dimensions = [3] : (tensor<1x1x2x8xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2] : (tensor<1x1x2xf32>) -> tensor<1x1x2x1xf32>
    %cst_5 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1x2x1xf32>
    %47 = stablehlo.add %45, %46 : tensor<1x1x2x1xf32>
    %48 = stablehlo.rsqrt %47 : tensor<1x1x2x1xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x8xf32>
    %50 = stablehlo.multiply %40, %49 : tensor<1x1x2x8xf32>
    %51 = stablehlo.multiply %42, %42 : tensor<1x1x2x8xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %52 = stablehlo.reduce(%51 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x1x2x8xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x1x2xf32>) -> tensor<1x1x2x1xf32>
    %cst_7 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x2x1xf32>
    %55 = stablehlo.add %53, %54 : tensor<1x1x2x1xf32>
    %56 = stablehlo.rsqrt %55 : tensor<1x1x2x1xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2, 3] : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x8xf32>
    %58 = stablehlo.multiply %42, %57 : tensor<1x1x2x8xf32>
    %59 = stablehlo.transpose %50, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %60 = stablehlo.transpose %58, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %61 = stablehlo.transpose %25, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %62 = stablehlo.transpose %31, dims = [0, 2, 1] : (tensor<1x1x2xf32>) -> tensor<1x2x1xf32>
    %63 = stablehlo.transpose %38, dims = [0, 2, 1] : (tensor<1x1x2xf32>) -> tensor<1x2x1xf32>
    %cst_8 = stablehlo.constant dense<0.353553385> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x2x1x8xf32>
    %65 = stablehlo.multiply %59, %64 : tensor<1x2x1x8xf32>
    %66 = stablehlo.reshape %65 : (tensor<1x2x1x8xf32>) -> tensor<1x2x8xf32>
    %67 = stablehlo.reshape %60 : (tensor<1x2x1x8xf32>) -> tensor<1x2x8xf32>
    %68 = stablehlo.reshape %61 : (tensor<1x2x1x8xf32>) -> tensor<1x2x8xf32>
    %69 = stablehlo.reshape %63 : (tensor<1x2x1xf32>) -> tensor<1x2xf32>
    %70 = stablehlo.reshape %62 : (tensor<1x2x1xf32>) -> tensor<1x2xf32>
    %71 = stablehlo.exponential %69 : tensor<1x2xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x1xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2] : (tensor<1x2x1xf32>) -> tensor<1x2x1x1xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x2x1x1xf32>) -> tensor<1x2x8x8xf32>
    %75 = stablehlo.multiply %arg11, %74 : tensor<1x2x8x8xf32>
    %76 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2] : (tensor<1x2x8xf32>) -> tensor<1x2x8x1xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x2x8x1xf32>) -> tensor<1x2x8x8xf32>
    %78 = stablehlo.multiply %75, %77 : tensor<1x2x8x8xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %79 = stablehlo.reduce(%78 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<1x2x8x8xf32>, tensor<f32>) -> tensor<1x2x8xf32>
    %80 = stablehlo.subtract %68, %79 : tensor<1x2x8xf32>
    %81 = stablehlo.broadcast_in_dim %70, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x1xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2] : (tensor<1x2x1xf32>) -> tensor<1x2x8xf32>
    %83 = stablehlo.multiply %80, %82 : tensor<1x2x8xf32>
    %84 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2] : (tensor<1x2x8xf32>) -> tensor<1x2x8x1xf32>
    %85 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 3] : (tensor<1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %86 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x2x8x1xf32>) -> tensor<1x2x8x8xf32>
    %87 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2, 3] : (tensor<1x2x1x8xf32>) -> tensor<1x2x8x8xf32>
    %88 = stablehlo.multiply %86, %87 : tensor<1x2x8x8xf32>
    %89 = stablehlo.add %75, %88 : tensor<1x2x8x8xf32>
    %90 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2] : (tensor<1x2x8xf32>) -> tensor<1x2x8x1xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2, 3] : (tensor<1x2x8x1xf32>) -> tensor<1x2x8x8xf32>
    %92 = stablehlo.multiply %89, %91 : tensor<1x2x8x8xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %93 = stablehlo.reduce(%92 init: %cst_10) applies stablehlo.add across dimensions = [2] : (tensor<1x2x8x8xf32>, tensor<f32>) -> tensor<1x2x8xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [1, 2, 3] : (tensor<1x2x8xf32>) -> tensor<1x1x2x8xf32>
    %95 = stablehlo.transpose %94, dims = [1, 2, 0, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %96 = stablehlo.transpose %95, dims = [0, 2, 1, 3] : (tensor<1x2x1x8xf32>) -> tensor<1x1x2x8xf32>
    %97 = stablehlo.reshape %96 : (tensor<1x1x2x8xf32>) -> tensor<2x8xf32>
    %98 = stablehlo.reshape %2 : (tensor<1x1x2x8xf32>) -> tensor<2x8xf32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %99 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<2x8xf32>
    %101 = stablehlo.power %97, %100 : tensor<2x8xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %102 = stablehlo.reduce(%101 init: %cst_11) applies stablehlo.add across dimensions = [1] : (tensor<2x8xf32>, tensor<f32>) -> tensor<2xf32>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %cst_12 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %104 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %105 = stablehlo.divide %103, %104 : tensor<2x1xf32>
    %cst_13 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %106 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %107 = stablehlo.add %105, %106 : tensor<2x1xf32>
    %108 = stablehlo.rsqrt %107 : tensor<2x1xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x8xf32>
    %110 = stablehlo.multiply %97, %109 : tensor<2x8xf32>
    %111 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<2x8xf32>
    %113 = stablehlo.multiply %112, %110 : tensor<2x8xf32>
    %114 = call @silu(%98) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %115 = stablehlo.multiply %113, %114 : tensor<2x8xf32>
    %116 = stablehlo.reshape %115 : (tensor<2x8xf32>) -> tensor<1x1x16xf32>
    %117 = stablehlo.dot_general %116, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xf32>, tensor<16x16xf32>) -> tensor<1x1x16xf32>
    return %117, %7, %89 : tensor<1x1x16xf32>, tensor<1x32x3xf32>, tensor<1x2x8x8xf32>
  }
  func.func private @softplus(%arg0: tensor<1x1x2xf32>) -> tensor<1x1x2xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x1x2xf32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<1x1x2xf32>
    %4 = stablehlo.compare NE, %3, %3, FLOAT : (tensor<1x1x2xf32>, tensor<1x1x2xf32>) -> tensor<1x1x2xi1>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<1x1x2xf32>
    %7 = stablehlo.abs %3 : tensor<1x1x2xf32>
    %8 = stablehlo.negate %7 : tensor<1x1x2xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x2xf32>
    %10 = stablehlo.log_plus_one %9 : tensor<1x1x2xf32>
    %11 = stablehlo.add %1, %10 : tensor<1x1x2xf32>
    %12 = stablehlo.select %4, %6, %11 : tensor<1x1x2xi1>, tensor<1x1x2xf32>
    return %12 : tensor<1x1x2xf32>
  }
  func.func private @silu(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = stablehlo.negate %arg0 : tensor<2x8xf32>
    %1 = stablehlo.exponential %0 : tensor<2x8xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8xf32>
    %3 = stablehlo.add %2, %1 : tensor<2x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x8xf32>
    %5 = stablehlo.divide %4, %3 : tensor<2x8xf32>
    %6 = stablehlo.multiply %arg0, %5 : tensor<2x8xf32>
    return %6 : tensor<2x8xf32>
  }
}
"""

_GDN_TINY_BF16 = r"""module @jit_fn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<32x4xbf16>, %arg1: tensor<2xbf16>, %arg2: tensor<2xbf16>, %arg3: tensor<16x32xbf16>, %arg4: tensor<16x16xbf16>, %arg5: tensor<16x2xbf16>, %arg6: tensor<16x2xbf16>, %arg7: tensor<8xbf16>, %arg8: tensor<16x16xbf16>, %arg9: tensor<1x1x16xbf16>, %arg10: tensor<1x32x3xbf16>, %arg11: tensor<1x2x8x8xf32>) -> (tensor<1x1x16xbf16> {jax.result_info = "result[0]"}, tensor<1x32x3xbf16> {jax.result_info = "result[1]"}, tensor<1x2x8x8xf32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.convert %arg11 : (tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xbf16>
    %1 = stablehlo.dot_general %arg9, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xbf16>, tensor<16x32xbf16>) -> tensor<1x1x32xbf16>
    %2 = stablehlo.dot_general %arg9, %arg4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xbf16>, tensor<16x16xbf16>) -> tensor<1x1x16xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1x1x16xbf16>) -> tensor<1x1x2x8xbf16>
    %4 = stablehlo.dot_general %arg9, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xbf16>, tensor<16x2xbf16>) -> tensor<1x1x2xbf16>
    %5 = stablehlo.dot_general %arg9, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xbf16>, tensor<16x2xbf16>) -> tensor<1x1x2xbf16>
    %6 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<1x1x32xbf16>) -> tensor<1x32x1xbf16>
    %7 = stablehlo.concatenate %arg10, %6, dim = 2 : (tensor<1x32x3xbf16>, tensor<1x32x1xbf16>) -> tensor<1x32x4xbf16>
    %8 = stablehlo.slice %7 [0:1, 0:32, 1:4] : (tensor<1x32x4xbf16>) -> tensor<1x32x3xbf16>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<32x4xbf16>) -> tensor<1x32x4xbf16>
    %10 = stablehlo.multiply %7, %9 : tensor<1x32x4xbf16>
    %11 = stablehlo.convert %10 : (tensor<1x32x4xbf16>) -> tensor<1x32x4xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x32x4xf32>, tensor<f32>) -> tensor<1x32xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %14 = stablehlo.convert %13 : (tensor<1x32x1xf32>) -> tensor<1x32x1xbf16>
    %15 = stablehlo.negate %14 : tensor<1x32x1xbf16>
    %16 = stablehlo.exponential %15 : tensor<1x32x1xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %17 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<1x32x1xbf16>
    %18 = stablehlo.add %17, %16 : tensor<1x32x1xbf16>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %19 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<bf16>) -> tensor<1x32x1xbf16>
    %20 = stablehlo.divide %19, %18 : tensor<1x32x1xbf16>
    %21 = stablehlo.multiply %14, %20 : tensor<1x32x1xbf16>
    %22 = stablehlo.transpose %21, dims = [0, 2, 1] : (tensor<1x32x1xbf16>) -> tensor<1x1x32xbf16>
    %23 = stablehlo.slice %22 [0:1, 0:1, 0:8] : (tensor<1x1x32xbf16>) -> tensor<1x1x8xbf16>
    %24 = stablehlo.slice %22 [0:1, 0:1, 8:16] : (tensor<1x1x32xbf16>) -> tensor<1x1x8xbf16>
    %25 = stablehlo.slice %22 [0:1, 0:1, 16:32] : (tensor<1x1x32xbf16>) -> tensor<1x1x16xbf16>
    %26 = stablehlo.reshape %23 : (tensor<1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
    %27 = stablehlo.reshape %24 : (tensor<1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
    %28 = stablehlo.reshape %25 : (tensor<1x1x16xbf16>) -> tensor<1x1x2x8xbf16>
    %29 = stablehlo.negate %4 : tensor<1x1x2xbf16>
    %30 = stablehlo.exponential %29 : tensor<1x1x2xbf16>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %31 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<bf16>) -> tensor<1x1x2xbf16>
    %32 = stablehlo.add %31, %30 : tensor<1x1x2xbf16>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %33 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<bf16>) -> tensor<1x1x2xbf16>
    %34 = stablehlo.divide %33, %32 : tensor<1x1x2xbf16>
    %35 = stablehlo.convert %arg2 : (tensor<2xbf16>) -> tensor<2xf32>
    %36 = stablehlo.exponential %35 : tensor<2xf32>
    %37 = stablehlo.negate %36 : tensor<2xf32>
    %38 = stablehlo.convert %5 : (tensor<1x1x2xbf16>) -> tensor<1x1x2xf32>
    %39 = stablehlo.convert %arg1 : (tensor<2xbf16>) -> tensor<2xf32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [2] : (tensor<2xf32>) -> tensor<1x1x2xf32>
    %41 = stablehlo.add %38, %40 : tensor<1x1x2xf32>
    %42 = call @softplus(%41) : (tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    %43 = stablehlo.broadcast_in_dim %37, dims = [2] : (tensor<2xf32>) -> tensor<1x1x2xf32>
    %44 = stablehlo.multiply %43, %42 : tensor<1x1x2xf32>
    %45 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 4] : (tensor<1x1x1x8xbf16>) -> tensor<1x1x1x2x8xbf16>
    %46 = stablehlo.reshape %45 : (tensor<1x1x1x2x8xbf16>) -> tensor<1x1x2x8xbf16>
    %47 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 4] : (tensor<1x1x1x8xbf16>) -> tensor<1x1x1x2x8xbf16>
    %48 = stablehlo.reshape %47 : (tensor<1x1x1x2x8xbf16>) -> tensor<1x1x2x8xbf16>
    %49 = stablehlo.multiply %46, %46 : tensor<1x1x2x8xbf16>
    %50 = stablehlo.convert %49 : (tensor<1x1x2x8xbf16>) -> tensor<1x1x2x8xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51 = stablehlo.reduce(%50 init: %cst_4) applies stablehlo.add across dimensions = [3] : (tensor<1x1x2x8xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2] : (tensor<1x1x2xf32>) -> tensor<1x1x2x1xf32>
    %53 = stablehlo.convert %52 : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x1xbf16>
    %cst_5 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %54 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<1x1x2x1xbf16>
    %55 = stablehlo.add %53, %54 : tensor<1x1x2x1xbf16>
    %56 = stablehlo.rsqrt %55 : tensor<1x1x2x1xbf16>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2, 3] : (tensor<1x1x2x1xbf16>) -> tensor<1x1x2x8xbf16>
    %58 = stablehlo.multiply %46, %57 : tensor<1x1x2x8xbf16>
    %59 = stablehlo.multiply %48, %48 : tensor<1x1x2x8xbf16>
    %60 = stablehlo.convert %59 : (tensor<1x1x2x8xbf16>) -> tensor<1x1x2x8xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %61 = stablehlo.reduce(%60 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x1x2x8xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2] : (tensor<1x1x2xf32>) -> tensor<1x1x2x1xf32>
    %63 = stablehlo.convert %62 : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x1xbf16>
    %cst_7 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %64 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<1x1x2x1xbf16>
    %65 = stablehlo.add %63, %64 : tensor<1x1x2x1xbf16>
    %66 = stablehlo.rsqrt %65 : tensor<1x1x2x1xbf16>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2, 3] : (tensor<1x1x2x1xbf16>) -> tensor<1x1x2x8xbf16>
    %68 = stablehlo.multiply %48, %67 : tensor<1x1x2x8xbf16>
    %69 = stablehlo.transpose %58, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xbf16>) -> tensor<1x2x1x8xbf16>
    %70 = stablehlo.transpose %68, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xbf16>) -> tensor<1x2x1x8xbf16>
    %71 = stablehlo.transpose %28, dims = [0, 2, 1, 3] : (tensor<1x1x2x8xbf16>) -> tensor<1x2x1x8xbf16>
    %72 = stablehlo.transpose %34, dims = [0, 2, 1] : (tensor<1x1x2xbf16>) -> tensor<1x2x1xbf16>
    %73 = stablehlo.transpose %44, dims = [0, 2, 1] : (tensor<1x1x2xf32>) -> tensor<1x2x1xf32>
    %74 = stablehlo.convert %69 : (tensor<1x2x1x8xbf16>) -> tensor<1x2x1x8xf32>
    %75 = stablehlo.convert %70 : (tensor<1x2x1x8xbf16>) -> tensor<1x2x1x8xf32>
    %76 = stablehlo.convert %71 : (tensor<1x2x1x8xbf16>) -> tensor<1x2x1x8xf32>
    %77 = stablehlo.convert %72 : (tensor<1x2x1xbf16>) -> tensor<1x2x1xf32>
    %cst_8 = stablehlo.constant dense<0.353553385> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x2x1x8xf32>
    %79 = stablehlo.multiply %74, %78 : tensor<1x2x1x8xf32>
    %80 = stablehlo.convert %0 : (tensor<1x2x8x8xbf16>) -> tensor<1x2x8x8xf32>
    %81 = stablehlo.reshape %79 : (tensor<1x2x1x8xf32>) -> tensor<1x2x8xf32>
    %82 = stablehlo.reshape %75 : (tensor<1x2x1x8xf32>) -> tensor<1x2x8xf32>
    %83 = stablehlo.reshape %76 : (tensor<1x2x1x8xf32>) -> tensor<1x2x8xf32>
    %84 = stablehlo.reshape %73 : (tensor<1x2x1xf32>) -> tensor<1x2xf32>
    %85 = stablehlo.reshape %77 : (tensor<1x2x1xf32>) -> tensor<1x2xf32>
    %86 = stablehlo.exponential %84 : tensor<1x2xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x1xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2] : (tensor<1x2x1xf32>) -> tensor<1x2x1x1xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x2x1x1xf32>) -> tensor<1x2x8x8xf32>
    %90 = stablehlo.multiply %80, %89 : tensor<1x2x8x8xf32>
    %91 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2] : (tensor<1x2x8xf32>) -> tensor<1x2x8x1xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<1x2x8x1xf32>) -> tensor<1x2x8x8xf32>
    %93 = stablehlo.multiply %90, %92 : tensor<1x2x8x8xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %94 = stablehlo.reduce(%93 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<1x2x8x8xf32>, tensor<f32>) -> tensor<1x2x8xf32>
    %95 = stablehlo.subtract %83, %94 : tensor<1x2x8xf32>
    %96 = stablehlo.broadcast_in_dim %85, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x1xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2] : (tensor<1x2x1xf32>) -> tensor<1x2x8xf32>
    %98 = stablehlo.multiply %95, %97 : tensor<1x2x8xf32>
    %99 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2] : (tensor<1x2x8xf32>) -> tensor<1x2x8x1xf32>
    %100 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 3] : (tensor<1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %101 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2, 3] : (tensor<1x2x8x1xf32>) -> tensor<1x2x8x8xf32>
    %102 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<1x2x1x8xf32>) -> tensor<1x2x8x8xf32>
    %103 = stablehlo.multiply %101, %102 : tensor<1x2x8x8xf32>
    %104 = stablehlo.add %90, %103 : tensor<1x2x8x8xf32>
    %105 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2] : (tensor<1x2x8xf32>) -> tensor<1x2x8x1xf32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2, 3] : (tensor<1x2x8x1xf32>) -> tensor<1x2x8x8xf32>
    %107 = stablehlo.multiply %104, %106 : tensor<1x2x8x8xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %108 = stablehlo.reduce(%107 init: %cst_10) applies stablehlo.add across dimensions = [2] : (tensor<1x2x8x8xf32>, tensor<f32>) -> tensor<1x2x8xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [1, 2, 3] : (tensor<1x2x8xf32>) -> tensor<1x1x2x8xf32>
    %110 = stablehlo.transpose %109, dims = [1, 2, 0, 3] : (tensor<1x1x2x8xf32>) -> tensor<1x2x1x8xf32>
    %111 = stablehlo.transpose %110, dims = [0, 2, 1, 3] : (tensor<1x2x1x8xf32>) -> tensor<1x1x2x8xf32>
    %112 = stablehlo.convert %111 : (tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xbf16>
    %113 = stablehlo.reshape %112 : (tensor<1x1x2x8xbf16>) -> tensor<2x8xbf16>
    %114 = stablehlo.reshape %3 : (tensor<1x1x2x8xbf16>) -> tensor<2x8xbf16>
    %115 = stablehlo.convert %113 : (tensor<2x8xbf16>) -> tensor<2x8xf32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %116 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [] : (tensor<f32>) -> tensor<2x8xf32>
    %118 = stablehlo.power %115, %117 : tensor<2x8xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %119 = stablehlo.reduce(%118 init: %cst_11) applies stablehlo.add across dimensions = [1] : (tensor<2x8xf32>, tensor<f32>) -> tensor<2xf32>
    %120 = stablehlo.broadcast_in_dim %119, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %cst_12 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %121 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %122 = stablehlo.divide %120, %121 : tensor<2x1xf32>
    %cst_13 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %123 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %124 = stablehlo.add %122, %123 : tensor<2x1xf32>
    %125 = stablehlo.rsqrt %124 : tensor<2x1xf32>
    %126 = stablehlo.broadcast_in_dim %125, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x8xf32>
    %127 = stablehlo.multiply %115, %126 : tensor<2x8xf32>
    %128 = stablehlo.convert %arg7 : (tensor<8xbf16>) -> tensor<8xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<2x8xf32>
    %131 = stablehlo.multiply %130, %127 : tensor<2x8xf32>
    %132 = stablehlo.convert %131 : (tensor<2x8xf32>) -> tensor<2x8xbf16>
    %133 = call @silu(%114) : (tensor<2x8xbf16>) -> tensor<2x8xbf16>
    %134 = stablehlo.multiply %132, %133 : tensor<2x8xbf16>
    %135 = stablehlo.reshape %134 : (tensor<2x8xbf16>) -> tensor<1x1x16xbf16>
    %136 = stablehlo.dot_general %135, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x16xbf16>, tensor<16x16xbf16>) -> tensor<1x1x16xbf16>
    return %136, %8, %104 : tensor<1x1x16xbf16>, tensor<1x32x3xbf16>, tensor<1x2x8x8xf32>
  }
  func.func private @softplus(%arg0: tensor<1x1x2xf32>) -> tensor<1x1x2xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x1x2xf32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<1x1x2xf32>
    %4 = stablehlo.compare NE, %3, %3, FLOAT : (tensor<1x1x2xf32>, tensor<1x1x2xf32>) -> tensor<1x1x2xi1>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<1x1x2xf32>
    %7 = stablehlo.abs %3 : tensor<1x1x2xf32>
    %8 = stablehlo.negate %7 : tensor<1x1x2xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x2xf32>
    %10 = stablehlo.log_plus_one %9 : tensor<1x1x2xf32>
    %11 = stablehlo.add %1, %10 : tensor<1x1x2xf32>
    %12 = stablehlo.select %4, %6, %11 : tensor<1x1x2xi1>, tensor<1x1x2xf32>
    return %12 : tensor<1x1x2xf32>
  }
  func.func private @silu(%arg0: tensor<2x8xbf16>) -> tensor<2x8xbf16> {
    %0 = stablehlo.negate %arg0 : tensor<2x8xbf16>
    %1 = stablehlo.exponential %0 : tensor<2x8xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x8xbf16>
    %3 = stablehlo.add %2, %1 : tensor<2x8xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<2x8xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<2x8xbf16>
    %6 = stablehlo.multiply %arg0, %5 : tensor<2x8xbf16>
    return %6 : tensor<2x8xbf16>
  }
}
"""


# The multi-chunk geometry, and the reason it is a fixture at all: the
# kernel splits Dk across `ty` threadgroup rows only when one row's registers
# cannot hold it, and the two dk sums then reduce through threadgroup memory
# and barriers instead of a single serial loop.  The two fixtures above have
# Dk 8 and take ty=1, so they never execute that path -- and row 8's real
# geometry (Dk 128, ty 4, 32 registers a thread) is exactly the path Apple's
# shader compiler miscompiled for msl_scan's time loops in 0.3.0.  Dk 128 /
# Dv 16 resolves to the SAME ty=4, 32-register shape at a sixteenth of the
# width, so the suite exercises it at a size that fits in a test file.
_GDN_CHUNKED_BF16 = r"""module @jit_fn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<288x4xbf16>, %arg1: tensor<2xbf16>, %arg2: tensor<2xbf16>, %arg3: tensor<32x288xbf16>, %arg4: tensor<32x32xbf16>, %arg5: tensor<32x2xbf16>, %arg6: tensor<32x2xbf16>, %arg7: tensor<16xbf16>, %arg8: tensor<32x32xbf16>, %arg9: tensor<1x1x32xbf16>, %arg10: tensor<1x288x3xbf16>, %arg11: tensor<1x2x128x16xf32>) -> (tensor<1x1x32xbf16> {jax.result_info = "result[0]"}, tensor<1x288x3xbf16> {jax.result_info = "result[1]"}, tensor<1x2x128x16xf32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.convert %arg11 : (tensor<1x2x128x16xf32>) -> tensor<1x2x128x16xbf16>
    %1 = stablehlo.dot_general %arg9, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32xbf16>, tensor<32x288xbf16>) -> tensor<1x1x288xbf16>
    %2 = stablehlo.dot_general %arg9, %arg4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32xbf16>, tensor<32x32xbf16>) -> tensor<1x1x32xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1x1x32xbf16>) -> tensor<1x1x2x16xbf16>
    %4 = stablehlo.dot_general %arg9, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32xbf16>, tensor<32x2xbf16>) -> tensor<1x1x2xbf16>
    %5 = stablehlo.dot_general %arg9, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32xbf16>, tensor<32x2xbf16>) -> tensor<1x1x2xbf16>
    %6 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<1x1x288xbf16>) -> tensor<1x288x1xbf16>
    %7 = stablehlo.concatenate %arg10, %6, dim = 2 : (tensor<1x288x3xbf16>, tensor<1x288x1xbf16>) -> tensor<1x288x4xbf16>
    %8 = stablehlo.slice %7 [0:1, 0:288, 1:4] : (tensor<1x288x4xbf16>) -> tensor<1x288x3xbf16>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<288x4xbf16>) -> tensor<1x288x4xbf16>
    %10 = stablehlo.multiply %7, %9 : tensor<1x288x4xbf16>
    %11 = stablehlo.convert %10 : (tensor<1x288x4xbf16>) -> tensor<1x288x4xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x288x4xf32>, tensor<f32>) -> tensor<1x288xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x288xf32>) -> tensor<1x288x1xf32>
    %14 = stablehlo.convert %13 : (tensor<1x288x1xf32>) -> tensor<1x288x1xbf16>
    %15 = stablehlo.negate %14 : tensor<1x288x1xbf16>
    %16 = stablehlo.exponential %15 : tensor<1x288x1xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %17 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<1x288x1xbf16>
    %18 = stablehlo.add %17, %16 : tensor<1x288x1xbf16>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %19 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<bf16>) -> tensor<1x288x1xbf16>
    %20 = stablehlo.divide %19, %18 : tensor<1x288x1xbf16>
    %21 = stablehlo.multiply %14, %20 : tensor<1x288x1xbf16>
    %22 = stablehlo.transpose %21, dims = [0, 2, 1] : (tensor<1x288x1xbf16>) -> tensor<1x1x288xbf16>
    %23 = stablehlo.slice %22 [0:1, 0:1, 0:128] : (tensor<1x1x288xbf16>) -> tensor<1x1x128xbf16>
    %24 = stablehlo.slice %22 [0:1, 0:1, 128:256] : (tensor<1x1x288xbf16>) -> tensor<1x1x128xbf16>
    %25 = stablehlo.slice %22 [0:1, 0:1, 256:288] : (tensor<1x1x288xbf16>) -> tensor<1x1x32xbf16>
    %26 = stablehlo.reshape %23 : (tensor<1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    %27 = stablehlo.reshape %24 : (tensor<1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    %28 = stablehlo.reshape %25 : (tensor<1x1x32xbf16>) -> tensor<1x1x2x16xbf16>
    %29 = stablehlo.negate %4 : tensor<1x1x2xbf16>
    %30 = stablehlo.exponential %29 : tensor<1x1x2xbf16>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %31 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<bf16>) -> tensor<1x1x2xbf16>
    %32 = stablehlo.add %31, %30 : tensor<1x1x2xbf16>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %33 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<bf16>) -> tensor<1x1x2xbf16>
    %34 = stablehlo.divide %33, %32 : tensor<1x1x2xbf16>
    %35 = stablehlo.convert %arg2 : (tensor<2xbf16>) -> tensor<2xf32>
    %36 = stablehlo.exponential %35 : tensor<2xf32>
    %37 = stablehlo.negate %36 : tensor<2xf32>
    %38 = stablehlo.convert %5 : (tensor<1x1x2xbf16>) -> tensor<1x1x2xf32>
    %39 = stablehlo.convert %arg1 : (tensor<2xbf16>) -> tensor<2xf32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [2] : (tensor<2xf32>) -> tensor<1x1x2xf32>
    %41 = stablehlo.add %38, %40 : tensor<1x1x2xf32>
    %42 = call @softplus(%41) : (tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    %43 = stablehlo.broadcast_in_dim %37, dims = [2] : (tensor<2xf32>) -> tensor<1x1x2xf32>
    %44 = stablehlo.multiply %43, %42 : tensor<1x1x2xf32>
    %45 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 4] : (tensor<1x1x1x128xbf16>) -> tensor<1x1x1x2x128xbf16>
    %46 = stablehlo.reshape %45 : (tensor<1x1x1x2x128xbf16>) -> tensor<1x1x2x128xbf16>
    %47 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 4] : (tensor<1x1x1x128xbf16>) -> tensor<1x1x1x2x128xbf16>
    %48 = stablehlo.reshape %47 : (tensor<1x1x1x2x128xbf16>) -> tensor<1x1x2x128xbf16>
    %49 = stablehlo.multiply %46, %46 : tensor<1x1x2x128xbf16>
    %50 = stablehlo.convert %49 : (tensor<1x1x2x128xbf16>) -> tensor<1x1x2x128xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51 = stablehlo.reduce(%50 init: %cst_4) applies stablehlo.add across dimensions = [3] : (tensor<1x1x2x128xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2] : (tensor<1x1x2xf32>) -> tensor<1x1x2x1xf32>
    %53 = stablehlo.convert %52 : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x1xbf16>
    %cst_5 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %54 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<1x1x2x1xbf16>
    %55 = stablehlo.add %53, %54 : tensor<1x1x2x1xbf16>
    %56 = stablehlo.rsqrt %55 : tensor<1x1x2x1xbf16>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2, 3] : (tensor<1x1x2x1xbf16>) -> tensor<1x1x2x128xbf16>
    %58 = stablehlo.multiply %46, %57 : tensor<1x1x2x128xbf16>
    %59 = stablehlo.multiply %48, %48 : tensor<1x1x2x128xbf16>
    %60 = stablehlo.convert %59 : (tensor<1x1x2x128xbf16>) -> tensor<1x1x2x128xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %61 = stablehlo.reduce(%60 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x1x2x128xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2] : (tensor<1x1x2xf32>) -> tensor<1x1x2x1xf32>
    %63 = stablehlo.convert %62 : (tensor<1x1x2x1xf32>) -> tensor<1x1x2x1xbf16>
    %cst_7 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %64 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<1x1x2x1xbf16>
    %65 = stablehlo.add %63, %64 : tensor<1x1x2x1xbf16>
    %66 = stablehlo.rsqrt %65 : tensor<1x1x2x1xbf16>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2, 3] : (tensor<1x1x2x1xbf16>) -> tensor<1x1x2x128xbf16>
    %68 = stablehlo.multiply %48, %67 : tensor<1x1x2x128xbf16>
    %69 = stablehlo.transpose %58, dims = [0, 2, 1, 3] : (tensor<1x1x2x128xbf16>) -> tensor<1x2x1x128xbf16>
    %70 = stablehlo.transpose %68, dims = [0, 2, 1, 3] : (tensor<1x1x2x128xbf16>) -> tensor<1x2x1x128xbf16>
    %71 = stablehlo.transpose %28, dims = [0, 2, 1, 3] : (tensor<1x1x2x16xbf16>) -> tensor<1x2x1x16xbf16>
    %72 = stablehlo.transpose %34, dims = [0, 2, 1] : (tensor<1x1x2xbf16>) -> tensor<1x2x1xbf16>
    %73 = stablehlo.transpose %44, dims = [0, 2, 1] : (tensor<1x1x2xf32>) -> tensor<1x2x1xf32>
    %74 = stablehlo.convert %69 : (tensor<1x2x1x128xbf16>) -> tensor<1x2x1x128xf32>
    %75 = stablehlo.convert %70 : (tensor<1x2x1x128xbf16>) -> tensor<1x2x1x128xf32>
    %76 = stablehlo.convert %71 : (tensor<1x2x1x16xbf16>) -> tensor<1x2x1x16xf32>
    %77 = stablehlo.convert %72 : (tensor<1x2x1xbf16>) -> tensor<1x2x1xf32>
    %cst_8 = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x2x1x128xf32>
    %79 = stablehlo.multiply %74, %78 : tensor<1x2x1x128xf32>
    %80 = stablehlo.convert %0 : (tensor<1x2x128x16xbf16>) -> tensor<1x2x128x16xf32>
    %81 = stablehlo.reshape %79 : (tensor<1x2x1x128xf32>) -> tensor<1x2x128xf32>
    %82 = stablehlo.reshape %75 : (tensor<1x2x1x128xf32>) -> tensor<1x2x128xf32>
    %83 = stablehlo.reshape %76 : (tensor<1x2x1x16xf32>) -> tensor<1x2x16xf32>
    %84 = stablehlo.reshape %73 : (tensor<1x2x1xf32>) -> tensor<1x2xf32>
    %85 = stablehlo.reshape %77 : (tensor<1x2x1xf32>) -> tensor<1x2xf32>
    %86 = stablehlo.exponential %84 : tensor<1x2xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x1xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2] : (tensor<1x2x1xf32>) -> tensor<1x2x1x1xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2, 3] : (tensor<1x2x1x1xf32>) -> tensor<1x2x128x16xf32>
    %90 = stablehlo.multiply %80, %89 : tensor<1x2x128x16xf32>
    %91 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x1xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<1x2x128x1xf32>) -> tensor<1x2x128x16xf32>
    %93 = stablehlo.multiply %90, %92 : tensor<1x2x128x16xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %94 = stablehlo.reduce(%93 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<1x2x128x16xf32>, tensor<f32>) -> tensor<1x2x16xf32>
    %95 = stablehlo.subtract %83, %94 : tensor<1x2x16xf32>
    %96 = stablehlo.broadcast_in_dim %85, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x1xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2] : (tensor<1x2x1xf32>) -> tensor<1x2x16xf32>
    %98 = stablehlo.multiply %95, %97 : tensor<1x2x16xf32>
    %99 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x1xf32>
    %100 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 3] : (tensor<1x2x16xf32>) -> tensor<1x2x1x16xf32>
    %101 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2, 3] : (tensor<1x2x128x1xf32>) -> tensor<1x2x128x16xf32>
    %102 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<1x2x1x16xf32>) -> tensor<1x2x128x16xf32>
    %103 = stablehlo.multiply %101, %102 : tensor<1x2x128x16xf32>
    %104 = stablehlo.add %90, %103 : tensor<1x2x128x16xf32>
    %105 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x1xf32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2, 3] : (tensor<1x2x128x1xf32>) -> tensor<1x2x128x16xf32>
    %107 = stablehlo.multiply %104, %106 : tensor<1x2x128x16xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %108 = stablehlo.reduce(%107 init: %cst_10) applies stablehlo.add across dimensions = [2] : (tensor<1x2x128x16xf32>, tensor<f32>) -> tensor<1x2x16xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [1, 2, 3] : (tensor<1x2x16xf32>) -> tensor<1x1x2x16xf32>
    %110 = stablehlo.transpose %109, dims = [1, 2, 0, 3] : (tensor<1x1x2x16xf32>) -> tensor<1x2x1x16xf32>
    %111 = stablehlo.transpose %110, dims = [0, 2, 1, 3] : (tensor<1x2x1x16xf32>) -> tensor<1x1x2x16xf32>
    %112 = stablehlo.convert %111 : (tensor<1x1x2x16xf32>) -> tensor<1x1x2x16xbf16>
    %113 = stablehlo.reshape %112 : (tensor<1x1x2x16xbf16>) -> tensor<2x16xbf16>
    %114 = stablehlo.reshape %3 : (tensor<1x1x2x16xbf16>) -> tensor<2x16xbf16>
    %115 = stablehlo.convert %113 : (tensor<2x16xbf16>) -> tensor<2x16xf32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %116 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [] : (tensor<f32>) -> tensor<2x16xf32>
    %118 = stablehlo.power %115, %117 : tensor<2x16xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %119 = stablehlo.reduce(%118 init: %cst_11) applies stablehlo.add across dimensions = [1] : (tensor<2x16xf32>, tensor<f32>) -> tensor<2xf32>
    %120 = stablehlo.broadcast_in_dim %119, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %cst_12 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %121 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %122 = stablehlo.divide %120, %121 : tensor<2x1xf32>
    %cst_13 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %123 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %124 = stablehlo.add %122, %123 : tensor<2x1xf32>
    %125 = stablehlo.rsqrt %124 : tensor<2x1xf32>
    %126 = stablehlo.broadcast_in_dim %125, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x16xf32>
    %127 = stablehlo.multiply %115, %126 : tensor<2x16xf32>
    %128 = stablehlo.convert %arg7 : (tensor<16xbf16>) -> tensor<16xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<2x16xf32>
    %131 = stablehlo.multiply %130, %127 : tensor<2x16xf32>
    %132 = stablehlo.convert %131 : (tensor<2x16xf32>) -> tensor<2x16xbf16>
    %133 = call @silu(%114) : (tensor<2x16xbf16>) -> tensor<2x16xbf16>
    %134 = stablehlo.multiply %132, %133 : tensor<2x16xbf16>
    %135 = stablehlo.reshape %134 : (tensor<2x16xbf16>) -> tensor<1x1x32xbf16>
    %136 = stablehlo.dot_general %135, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32xbf16>, tensor<32x32xbf16>) -> tensor<1x1x32xbf16>
    return %136, %8, %104 : tensor<1x1x32xbf16>, tensor<1x288x3xbf16>, tensor<1x2x128x16xf32>
  }
  func.func private @softplus(%arg0: tensor<1x1x2xf32>) -> tensor<1x1x2xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x1x2xf32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<1x1x2xf32>
    %4 = stablehlo.compare NE, %3, %3, FLOAT : (tensor<1x1x2xf32>, tensor<1x1x2xf32>) -> tensor<1x1x2xi1>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<1x1x2xf32>
    %7 = stablehlo.abs %3 : tensor<1x1x2xf32>
    %8 = stablehlo.negate %7 : tensor<1x1x2xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x2xf32>
    %10 = stablehlo.log_plus_one %9 : tensor<1x1x2xf32>
    %11 = stablehlo.add %1, %10 : tensor<1x1x2xf32>
    %12 = stablehlo.select %4, %6, %11 : tensor<1x1x2xi1>, tensor<1x1x2xf32>
    return %12 : tensor<1x1x2xf32>
  }
  func.func private @silu(%arg0: tensor<2x16xbf16>) -> tensor<2x16xbf16> {
    %0 = stablehlo.negate %arg0 : tensor<2x16xbf16>
    %1 = stablehlo.exponential %0 : tensor<2x16xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x16xbf16>
    %3 = stablehlo.add %2, %1 : tensor<2x16xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<2x16xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<2x16xbf16>
    %6 = stablehlo.multiply %arg0, %5 : tensor<2x16xbf16>
    return %6 : tensor<2x16xbf16>
  }
}
"""


def _gdn_forms():
    """(label, module, the form tag metal_gdn.cc must narrate, dtype).

    The tag is `B<batch>H<Hv>/<Hk>D<Dk>x<Dv>`, plus `+l2` when both
    `_l2norm`s were absorbed and `+rt` when the state's dead narrowing round
    trip was.  Pinning it is what says the recognizer claimed the whole
    block and not merely the recurrence: the row-8 diagnosis found the
    `_l2norm` declined 120 times per program shape by `AnalyzeNorm`, and
    a match that quietly stopped short of them would still be correct.
    """
    return [
        ("gdn f32 (Hk=1, Hv=2, D=8)", _GDN_TINY_F32, "B1H2/1D8x8+l2", "f32"),
        ("gdn bf16 (row-8 spelling)", _GDN_TINY_BF16, "B1H2/1D8x8+l2+rt",
         "bf16"),
        ("gdn bf16 chunked (Dk=128, ty=4)", _GDN_CHUNKED_BF16,
         "B1H2/1D128x16+l2+rt", "bf16"),
    ]


def _gdn_inputs(text, seed=4711):
    """Deterministic operands for one captured GDN module, read off @main.

    Every operand is a float tensor (weights, the hidden state, the conv
    cache, the recurrent cache), so one rule covers them: standard normals at
    half scale, in the dtype the signature declares.  `A_log` and `dt_bias`
    enter as `-exp(A_log) * softplus(a + dt_bias)`, which is negative for any
    draw, so `exp(g)` stays in (0, 1) and the recurrence is stable whatever
    the seed.
    """
    import re as _re
    import ml_dtypes
    sig = _re.search(r"func\.func public @main\((.*?)\)\s*->", text,
                     _re.S).group(1)
    rng = np.random.default_rng(seed)
    dts = {"f32": np.float32, "bf16": ml_dtypes.bfloat16, "f16": np.float16}
    args = []
    for a in _re.findall(r"tensor<([^>]*)>", sig):
        *dims, dt = a.split("x")
        shape = tuple(int(d) for d in dims)
        v = (rng.standard_normal(shape) * 0.5).astype(np.float32)
        args.append(v.astype(dts[dt]))
    return args


def _gdn_chain(text, steps=8, seed=99):
    """Run one captured step `steps` times, feeding both caches forward.

    A single step cannot see drift: the recurrent state is an INPUT there.
    Chaining is what puts the fused kernel's own output back under its own
    reduction, which is how a decode of 128 tokens actually runs, so this is
    the arm that would catch a kernel that is right once and wrong in a loop.
    """
    args = _gdn_inputs(text, seed)
    outs = []
    rng = np.random.default_rng(seed + 1)
    hs_dtype = args[-3].dtype
    hs_shape = args[-3].shape
    for t in range(steps):
        args[-3] = (rng.standard_normal(hs_shape) * 0.5).astype(
            np.float32).astype(hs_dtype)
        o, conv, rec = _run_module(text, args)
        outs.append(np.asarray(o).astype(np.float64).ravel())
        args[-2] = conv
        args[-1] = rec
    outs.append(np.asarray(args[-1]).astype(np.float64).ravel())
    return np.concatenate(outs)


def _mla_forms():
    """(label, module, geometry tag metal_mla.cc must narrate, dtype).

    Row 10 built the recognizer on MLA -- one query head per KV head -- and
    row 11 (Qwen3-0.6B, 16 query heads over 8 KV heads) declined on the head
    geometry alone.  What is pinned here is that BOTH geometries fire, with
    the Hkv the matcher claims to have read, and that the GQA arm's answer
    is the literal chain's.
    """
    return [
        ("mla f32 (Hkv == H)", _MLA_TWO_SPAN, "H2Hkv2D4", "f32"),
        ("gqa f32 (H=4, Hkv=2)", _MLA_GQA_TWO_SPAN_F32, "H4Hkv2D4", "f32"),
        ("gqa bf16 (row-11 spelling)", _MLA_GQA_TWO_SPAN_BF16, "H4Hkv2D4",
         "bf16"),
        ("gqa bf16 D=64 (fused kernel)", _MLA_GQA_TWO_SPAN_BF16_D64,
         "H4Hkv2D64", "bf16"),
    ]


def _keras_gqa_module(dt, H, Hkv, D, T, spelling="insert"):
    """keras-hub's grouped-query decode attention, verbatim from the lowered
    Qwen3-MoE step (`qwen3_moe_attention.py`, 2-layer tiny model dumped on
    CPU): `ops.repeat(k, G, axis=2)` is a rank-raising `broadcast_in_dim`
    that INSERTS the group axis right after the kv-head axis, then a reshape
    merging the pair into H (kv head major); the two dots batch over (b, h)
    with the query carrying a unit axis, the scores land in f32, the causal
    `where` goes through a shared callee with keras' -2.38e38 sentinel, the
    softmax subtracts its own max, and the probabilities convert back to the
    model dtype before the values dot.

    Three spellings of the repeat: `insert` (the one above), `unit` (jax's
    other lowering of `jnp.repeat`: a reshape to a unit axis, then a same-rank
    broadcast expanding it), and `tile` -- the group axis inserted BEFORE the
    kv-head axis, which is `ops.tile`, a different function (query head h
    reads kv head h % Hkv); the recognizer must keep that repeat
    materialized, and the plain fusion must still compute it.
    """
    G = H // Hkv
    acc = "f32"
    algo = ""
    tag = f"{dt}_h{H}_kv{Hkv}_d{D}_t{T}_{spelling}"

    def rep(name, src):
        if spelling == "insert":
            return (f"    %{name}b = stablehlo.broadcast_in_dim %{src}, "
                    f"dims = [0, 1, 2, 4] : (tensor<1x{T}x{Hkv}x{D}x{dt}>) "
                    f"-> tensor<1x{T}x{Hkv}x{G}x{D}x{dt}>\n"
                    f"    %{name} = stablehlo.reshape %{name}b : "
                    f"(tensor<1x{T}x{Hkv}x{G}x{D}x{dt}>) "
                    f"-> tensor<1x{T}x{H}x{D}x{dt}>\n")
        if spelling == "unit":
            return (f"    %{name}u = stablehlo.reshape %{src} : "
                    f"(tensor<1x{T}x{Hkv}x{D}x{dt}>) "
                    f"-> tensor<1x{T}x{Hkv}x1x{D}x{dt}>\n"
                    f"    %{name}b = stablehlo.broadcast_in_dim %{name}u, "
                    f"dims = [0, 1, 2, 3, 4] : "
                    f"(tensor<1x{T}x{Hkv}x1x{D}x{dt}>) "
                    f"-> tensor<1x{T}x{Hkv}x{G}x{D}x{dt}>\n"
                    f"    %{name} = stablehlo.reshape %{name}b : "
                    f"(tensor<1x{T}x{Hkv}x{G}x{D}x{dt}>) "
                    f"-> tensor<1x{T}x{H}x{D}x{dt}>\n")
        assert spelling == "tile", spelling
        return (f"    %{name}b = stablehlo.broadcast_in_dim %{src}, "
                f"dims = [0, 1, 3, 4] : (tensor<1x{T}x{Hkv}x{D}x{dt}>) "
                f"-> tensor<1x{T}x{G}x{Hkv}x{D}x{dt}>\n"
                f"    %{name} = stablehlo.reshape %{name}b : "
                f"(tensor<1x{T}x{G}x{Hkv}x{D}x{dt}>) "
                f"-> tensor<1x{T}x{H}x{D}x{dt}>\n")

    # The probabilities convert back to the model dtype before the values
    # dot; in f32 there is nothing to convert and the dot reads them as is.
    probs = ("    %pc = stablehlo.convert %p : "
             f"(tensor<1x1x{H}x1x{T}x{acc}>) -> tensor<1x1x{H}x1x{T}x{dt}>\n"
             if dt != acc else "")
    pv = "pc" if dt != acc else "p"
    return f"""
module @keras_gqa_{tag} {{
  func.func private @_where_{tag}(%p: tensor<1x1x{H}x1x{T}xi1>,
      %x: tensor<1x{H}x1x1x{T}x{acc}>, %c: tensor<{acc}>)
      -> tensor<1x1x{H}x1x{T}x{acc}> {{
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<{acc}>) -> tensor<1x{H}x1x{T}x{acc}>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2, 3, 4] : (tensor<1x{H}x1x{T}x{acc}>) -> tensor<1x1x{H}x1x{T}x{acc}>
    %2 = stablehlo.transpose %x, dims = [3, 0, 1, 2, 4] : (tensor<1x{H}x1x1x{T}x{acc}>) -> tensor<1x1x{H}x1x{T}x{acc}>
    %3 = stablehlo.select %p, %2, %1 : tensor<1x1x{H}x1x{T}xi1>, tensor<1x1x{H}x1x{T}x{acc}>
    return %3 : tensor<1x1x{H}x1x{T}x{acc}>
  }}
  func.func public @main(%q: tensor<1x1x{H}x{D}x{dt}>,
      %k: tensor<1x{T}x{Hkv}x{D}x{dt}>, %v: tensor<1x{T}x{Hkv}x{D}x{dt}>,
      %mask: tensor<1x{T}xi1>) -> tensor<1x1x{H}x{D}x{dt}> {{
{rep("kr", "k")}{rep("vr", "v")}
    %q5 = stablehlo.reshape %q : (tensor<1x1x{H}x{D}x{dt}>) -> tensor<1x1x{H}x1x{D}x{dt}>
    %m3 = stablehlo.broadcast_in_dim %mask, dims = [0, 2] : (tensor<1x{T}xi1>) -> tensor<1x1x{T}xi1>
    %m4 = stablehlo.broadcast_in_dim %m3, dims = [0, 2, 3] : (tensor<1x1x{T}xi1>) -> tensor<1x1x1x{T}xi1>
    %m5 = stablehlo.broadcast_in_dim %m4, dims = [0, 1, 3, 4] : (tensor<1x1x1x{T}xi1>) -> tensor<1x1x1x1x{T}xi1>
    %s = stablehlo.dot_general %q5, %kr, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT]{algo} : (tensor<1x1x{H}x1x{D}x{dt}>, tensor<1x{T}x{H}x{D}x{dt}>) -> tensor<1x{H}x1x1x{T}x{acc}>
    %sc = stablehlo.constant dense<2.500000e-01> : tensor<{acc}>
    %scb = stablehlo.broadcast_in_dim %sc, dims = [] : (tensor<{acc}>) -> tensor<1x{H}x1x1x{T}x{acc}>
    %l = stablehlo.multiply %s, %scb : tensor<1x{H}x1x1x{T}x{acc}>
    %t = stablehlo.constant dense<true> : tensor<i1>
    %tb = stablehlo.broadcast_in_dim %t, dims = [] : (tensor<i1>) -> tensor<1x{H}x1x{T}xi1>
    %tb5 = stablehlo.broadcast_in_dim %tb, dims = [1, 2, 3, 4] : (tensor<1x{H}x1x{T}xi1>) -> tensor<1x1x{H}x1x{T}xi1>
    %mt = stablehlo.transpose %m5, dims = [2, 0, 1, 3, 4] : (tensor<1x1x1x1x{T}xi1>) -> tensor<1x1x1x1x{T}xi1>
    %mb = stablehlo.broadcast_in_dim %mt, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x{T}xi1>) -> tensor<1x1x{H}x1x{T}xi1>
    %pred = stablehlo.and %tb5, %mb : tensor<1x1x{H}x1x{T}xi1>
    %neg = stablehlo.constant dense<-2.38197633E+38> : tensor<{acc}>
    %w = call @_where_{tag}(%pred, %l, %neg) : (tensor<1x1x{H}x1x{T}xi1>, tensor<1x{H}x1x1x{T}x{acc}>, tensor<{acc}>) -> tensor<1x1x{H}x1x{T}x{acc}>
    %ninf = stablehlo.constant dense<0xFF800000> : tensor<{acc}>
    %mx = stablehlo.reduce(%w init: %ninf) applies stablehlo.maximum across dimensions = [4] : (tensor<1x1x{H}x1x{T}x{acc}>, tensor<{acc}>) -> tensor<1x1x{H}x1x{acc}>
    %ninf2 = stablehlo.constant dense<0xFF800000> : tensor<{acc}>
    %nb = stablehlo.broadcast_in_dim %ninf2, dims = [] : (tensor<{acc}>) -> tensor<1x1x{H}x1x{acc}>
    %mx2 = stablehlo.maximum %nb, %mx : tensor<1x1x{H}x1x{acc}>
    %mx3 = stablehlo.broadcast_in_dim %mx2, dims = [0, 1, 2, 3] : (tensor<1x1x{H}x1x{acc}>) -> tensor<1x1x{H}x1x1x{acc}>
    %mx4 = stablehlo.broadcast_in_dim %mx3, dims = [0, 1, 2, 3, 4] : (tensor<1x1x{H}x1x1x{acc}>) -> tensor<1x1x{H}x1x{T}x{acc}>
    %sub = stablehlo.subtract %w, %mx4 : tensor<1x1x{H}x1x{T}x{acc}>
    %e = stablehlo.exponential %sub : tensor<1x1x{H}x1x{T}x{acc}>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<{acc}>
    %sum = stablehlo.reduce(%e init: %zero) applies stablehlo.add across dimensions = [4] : (tensor<1x1x{H}x1x{T}x{acc}>, tensor<{acc}>) -> tensor<1x1x{H}x1x{acc}>
    %sb = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2, 3] : (tensor<1x1x{H}x1x{acc}>) -> tensor<1x1x{H}x1x1x{acc}>
    %sb2 = stablehlo.broadcast_in_dim %sb, dims = [0, 1, 2, 3, 4] : (tensor<1x1x{H}x1x1x{acc}>) -> tensor<1x1x{H}x1x{T}x{acc}>
    %p = stablehlo.divide %e, %sb2 : tensor<1x1x{H}x1x{T}x{acc}>
{probs}    %o = stablehlo.dot_general %vr, %{pv}, batching_dims = [0, 2] x [1, 2], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x{T}x{H}x{D}x{dt}>, tensor<1x1x{H}x1x{T}x{dt}>) -> tensor<1x{H}x{D}x1x1x{dt}>
    %o2 = stablehlo.transpose %o, dims = [3, 0, 4, 1, 2] : (tensor<1x{H}x{D}x1x1x{dt}>) -> tensor<1x1x1x{H}x{D}x{dt}>
    %o3 = stablehlo.transpose %o2, dims = [1, 2, 3, 0, 4] : (tensor<1x1x1x{H}x{D}x{dt}>) -> tensor<1x1x{H}x1x{D}x{dt}>
    %o4 = stablehlo.reshape %o3 : (tensor<1x1x{H}x1x{D}x{dt}>) -> tensor<1x1x{H}x{D}x{dt}>
    return %o4 : tensor<1x1x{H}x{D}x{dt}>
  }}
}}
"""


def _gqa_forms():
    """(label, module, absorbed (H, Hkv) or None, dtype, kept-reason) for the
    keras GQA repeat (gap-rows item 6).

    What is pinned: that the repeat is peeled at the row-7 and row-20 head
    geometries and in both dtypes, through MLX's composite path (D = 16),
    its vector kernel (D = 64 / 128) and at Hkv = 1 (multi-query, no kv-head
    atom left); that a group-MAJOR repeat (`tile`) is refused and still
    fuses plainly; and that the two geometries where MLX would change
    kernels (Tk >= 4096 routes a grouped decode to the 2-pass variant) keep
    the repeat rather than move the numerics.  The Tq * G > 32 rule is
    exercised by `_p37_gqa`'s jitted prefill case.
    """
    return [
        ("keras f32 H=8 Hkv=2 D=16", _keras_gqa_module("f32", 8, 2, 16, 8),
         (8, 2), "f32", None),
        ("keras bf16 H=8 Hkv=2 D=64", _keras_gqa_module("bf16", 8, 2, 64, 8),
         (8, 2), "bf16", None),
        ("keras bf16 H=64 Hkv=4 D=128 (row 20)",
         _keras_gqa_module("bf16", 64, 4, 128, 8), (64, 4), "bf16", None),
        ("keras f32 MQA H=4 Hkv=1 D=64 (unit spelling)",
         _keras_gqa_module("f32", 4, 1, 64, 16, "unit"), (4, 1), "f32", None),
        # Group-major repeats decline on two different checks, by geometry:
        # with G != Hkv the atom stream cannot even form the group axis;
        # with G == Hkv it can, and the pairing check is what refuses it.
        ("keras bf16 H=8 Hkv=2 D=64 tile (group major)",
         _keras_gqa_module("bf16", 8, 2, 64, 8, "tile"), None, "bf16",
         "reshape splits an axis"),
        ("keras bf16 H=4 Hkv=2 D=64 tile (group major, G == Hkv)",
         _keras_gqa_module("bf16", 4, 2, 64, 8, "tile"), None, "bf16",
         "the repeat is not kv-head major"),
        ("keras bf16 H=4 Hkv=2 D=64 Tk=4096",
         _keras_gqa_module("bf16", 4, 2, 64, 4096), None, "bf16",
         "Tk >= 4096 would route MLX to its 2-pass vector kernel"),
    ]


def _gqa_jit_forms():
    """(label, fn, args, absorbed (H, Hkv) or None, dtype, kept-reason): the
    jitted PREFILL-shaped GQA attentions (`jnp.repeat` spelled, Tq > 1) that
    the raw decode modules cannot reach -- MLX's full-attention kernel (Tq >
    8, D = 64) and its composite path (D = 16) with a grouping, both of which
    take the un-repeated heads; and the one geometry its vector kernel
    refuses grouped (Tq = 4 over G = 16: Tq * G > 32), which must keep its
    repeat."""
    import jax
    import jax.numpy as jnp

    def prefill(q, kk, vv, mask, groups, scale):
        k = jnp.repeat(kk, groups, axis=2)
        v = jnp.repeat(vv, groups, axis=2)
        logits = jnp.einsum("bquh,bkuh->buqk", q, k) * scale
        logits = jnp.where(mask[None, None, :, :], logits,
                           jnp.asarray(-1e4, logits.dtype))
        return jnp.einsum("buqk,bkuh->bquh",
                          jax.nn.softmax(logits, -1).astype(v.dtype), v)

    def case(label, dt, Tq, H, Hkv, D, T, seed, absorbed, why):
        pm = np.tril(np.ones((Tq, T), bool), k=T - Tq)
        args = [_rand((1, Tq, H, D), seed).astype(dt) * 0.5,
                _rand((1, T, Hkv, D), seed + 1).astype(dt) * 0.5,
                _rand((1, T, Hkv, D), seed + 2).astype(dt) * 0.5,
                jnp.asarray(pm)]
        fn = (lambda q, k, v, m, g=H // Hkv, s=D ** -0.5:
              prefill(q, k, v, m, g, s))
        return (label, fn, args, absorbed,
                "f32" if dt == "float32" else "bf16", why)

    return [
        case("prefill Tq=4 H=64 Hkv=4 D=64", "float32", 4, 64, 4, 64, 8, 300,
             None, "Tq*G = 64 > 32 would leave MLX's vector kernel"),
        case("prefill Tq=16 H=8 Hkv=2 D=64 f32 (full kernel)", "float32",
             16, 8, 2, 64, 32, 310, (8, 2), None),
        case("prefill Tq=16 H=8 Hkv=2 D=64 bf16 (full kernel)", "bfloat16",
             16, 8, 2, 64, 32, 320, (8, 2), None),
        case("prefill Tq=16 H=8 Hkv=2 D=16 bf16 (composite)", "bfloat16",
             16, 8, 2, 16, 32, 330, (8, 2), None),
    ]


def _gqa_inputs(text):
    """Deterministic inputs for one keras GQA module, read off @main: random
    floats, and a padding mask that keeps the first three quarters of the
    cache (so every row has an unmasked key and the additive rewrite of the
    sentinel select is exact)."""
    import re as _re
    import ml_dtypes
    sig = _re.search(r"func\.func public @main\((.*?)\)\s*->", text,
                     _re.S).group(1)
    rng = np.random.default_rng(2006)
    args = []
    for a in _re.findall(r"tensor<([^>]*)>", sig):
        *dims, dt = a.split("x")
        shape = tuple(int(d) for d in dims)
        if dt == "i1":
            keep = np.zeros(shape, bool)
            keep[..., :max(1, (shape[-1] * 3) // 4)] = True
            args.append(keep)
            continue
        v = (rng.standard_normal(shape) * 0.5).astype(np.float32)
        args.append(v.astype({"f32": np.float32,
                              "bf16": ml_dtypes.bfloat16}[dt]))
    return args


def _mla_kernel_forms():
    """(label, module, kernel key tag runtime/mla.cc must narrate, dtype).

    The geometries the B3 two-span kernel takes -- MLX's own `sdpa_vector`
    head-dim set, so the concat path it is compared against runs the fused
    kernel it copies -- and `_p40_mla_kernel` pins that it BUILDS for each
    and that its answer is the concat path's to the bit.
    """
    return [
        ("kernel bf16 D=64 T4+2", _MLA_GQA_TWO_SPAN_BF16_D64,
         "B1H4Hkv2D64Dv64T4+2_bfloat16_t", "bf16"),
        ("kernel bf16 D=64 T40+27", _MLA_KERNEL_BF16_D64_T40_27,
         "B1H4Hkv2D64Dv64T40+27_bfloat16_t", "bf16"),
        ("kernel f32 D=64 T33+5", _MLA_KERNEL_F32_D64_T33_5,
         "B1H4Hkv2D64Dv64T33+5_float", "f32"),
    ]


def _mla_inputs(text):
    """Deterministic inputs for one attention module, read off @main.

    The i32 operands are segment ids, and the mask keeps position i when
    `seg[i] == 1`.  A span whose every position is masked makes the literal
    chain's running max -inf and its renormalization NaN -- true of the
    reference too, so it would test nothing -- hence the pattern below,
    which always keeps position 0.
    """
    import re as _re
    import ml_dtypes
    sig = _re.search(r"func\.func public @main\((.*?)\)\s*->", text,
                     _re.S).group(1)
    rng = np.random.default_rng(1109)
    args = []
    for a in _re.findall(r"tensor<([^>]*)>", sig):
        *dims, dt = a.split("x")
        shape = tuple(int(d) for d in dims)
        if dt == "i32":
            keep = np.ones(shape, np.int32)
            keep.reshape(-1)[1::3] = 0
            keep.reshape(-1)[0] = 1
            args.append(keep)
            continue
        v = (rng.standard_normal(shape) * 0.5).astype(np.float32)
        args.append(v.astype({"f32": np.float32,
                              "bf16": ml_dtypes.bfloat16}[dt]))
    return args


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

        The second half is that `earn` really is what the live set says: it
        must equal `min(mult * (hi - lo), hi)` -- BOTH terms, the multiplier
        over the swing and the high-water clamp -- over that PROGRAM's own
        `live=` readings, the values the rule sampled, which is why the meter
        prints them separately from `active=` (the two differ by whatever the
        step dropped between the sample and the print; on this probe's
        phase-change flush they read 399 MB and 95 MB).  Per program, because
        that is where the water marks live: a replay pooling every program's
        readings would be checking a rule nothing implements.

        The clamp is not decoration here: this probe drops to 57 MB from a
        419 MB high-water, so `2 * swing` is 724 MB and the high-water is what
        the rule actually hands it.  Checking `mult * swing` alone passed on
        the pre-clamp binary and FAILS on the shipped one -- which is how this
        contract came to be re-run in the first place.
        """
        rows = state.get("swing", ([], None))[0]
        if not rows:
            return False, "the swinging arm did not run"
        worst, worst_row, checked = 0, None, 0
        drift, drift_row = 0, None
        bound_by = {"swing": 0, "high-water": 0}
        for prog in by_program(rows):
            hi = lo = None
            for _active, _cache, _was, bound, _foot, _cap, n, live, earn in prog:
                if earn == -1 or live == -1:
                    return False, "the meter reported no earn with the rule on"
                hi = live if hi is None else max(hi, live)
                lo = live if lo is None else min(lo, live)
                # The shipped rule is both terms (runtime.cc `flush_bound`,
                # notes/cpp-p28-benefit-gate.md §2), and which of them binds
                # is narrated below so a failure names the term.
                want_earn = min(MULT * (hi - lo), hi)
                bound_by["swing" if MULT * (hi - lo) <= hi else "high-water"] += 1
                if abs(earn - want_earn) > drift:
                    drift = abs(earn - want_earn)
                    drift_row = (earn, want_earn, hi, lo)
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
                           f"says min({MULT}*({drift_row[2]}-{drift_row[3]}), "
                           f"{drift_row[2]}) = {drift_row[1]} MB")
        # The identity is vacuous unless the rule is what actually chose the
        # bound on a fair share of the flushes rather than a clamp.
        inside = [r for r in rows if r[6] >= GATE and FLOOR < r[3] < CAP]
        if len(inside) < checked // 4:
            return False, (f"only {len(inside)} of {checked} bounds left "
                           f"their clamps")
        span = (min(r[3] for r in inside), max(r[3] for r in inside))
        return True, (f"ok (bound {span[0]}-{span[1]} MB on {len(inside)} of "
                      f"{checked} flushes past the gate, worst {worst} MB off; "
                      f"over all {len(rows)} narrated flushes earn was bound by "
                      f"the high-water on {bound_by['high-water']} and by the "
                      f"swing on {bound_by['swing']})")

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


_P36_TAPE_GATE = r"""
import os, json, numpy as np, jax, jax.numpy as jnp

# A chain of INTEGER rounds, each holding a full reduce.  Integers because
# the contract compares the compiled and the eager answers for BIT identity,
# and an int32 sum is exact in any order and any fusion; a chain because CSE
# must find nothing to merge (each round depends on the last), so the gap
# between the two counts is entirely the reduce bodies and the terminators
# BlockCost charges and the tape does not carry.
N = int(os.environ.get("P36_ROUNDS", "300"))

def f(x):
    for _ in range(N):
        x = (x * 3 + jnp.sum(x)) % 1000003
    return x

y = np.asarray(jax.jit(f)(jnp.arange(64, dtype=jnp.int32)))
print("[probe] answer " + json.dumps([int(v) for v in y]))
"""


_KERAS_ATTN = r'''
import os, numpy as np, jax, jax.numpy as jnp

WHICH = os.environ["KERAS_ATTN_WHICH"]

def gptoss(q, kk, vv, sinks, mslide, mcausal, groups):
    k = jnp.repeat(kk, groups, axis=2)
    v = jnp.repeat(vv, groups, axis=2)
    logits = jnp.einsum("bquh,bkuh->buqk", q, k)
    logits = logits * jnp.asarray(0.25, logits.dtype)
    adder = jnp.asarray(-1e4, logits.dtype)
    if mslide is not None:
        logits = jnp.where(mslide[None, None, :, :], logits, adder)
    logits = jnp.where(mcausal[None, None, :, :], logits, adder)
    s = jnp.broadcast_to(sinks.reshape(1, -1, 1, 1), logits.shape[:3] + (1,))
    c = jnp.concatenate([logits, s], axis=-1)
    c = c - jnp.max(c, axis=-1, keepdims=True)
    p = jax.nn.softmax(c, axis=-1)[..., :-1]
    return jnp.einsum("buqk,bkuh->bquh", p.astype(v.dtype), v)

def gemma4(q, k, v, mask):
    b, t, n, h = q.shape
    kvh = k.shape[2]
    g = n // kvh
    qt = jnp.transpose(q.reshape(b, t, kvh, g, h), (0, 2, 3, 1, 4))
    ke = jnp.expand_dims(jnp.transpose(k, (0, 2, 3, 1)), 2)
    logits = jnp.matmul(qt, ke).astype(jnp.float32)
    m = mask[:, None, None, :, :]
    logits = jnp.where(m, logits, jnp.asarray(-1e9, jnp.float32))
    p = jax.nn.softmax(logits, axis=-1)
    p = jnp.where(m, p, jnp.asarray(0.0, jnp.float32)).astype(v.dtype)
    ve = jnp.expand_dims(jnp.transpose(v, (0, 2, 1, 3)), 2)
    r = jnp.transpose(jnp.matmul(p, ve), (0, 3, 1, 2, 4))
    return r.reshape(b, t, n, h)

def small(q, k, v, mask):
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * 0.25
    logits = jnp.where(mask, logits, jnp.asarray(-8.0, logits.dtype))
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, -1), v)

rng = np.random.RandomState(7)
KL, H, KVH, HD = 8, 4, 2, 16
# D = 64 in the bf16 arm so MLX's fast `sdpa_vector` kernel takes it (it
# wants a head dim in {64, 96, 128, 256}); D = 16 lands on fast.cpp's
# fallback.  Both implement `sinks`, differently, so both need covering.
DT = np.dtype(os.environ.get("KERAS_ATTN_DTYPE", "float32"))
if DT.name == "bfloat16":
    import ml_dtypes
    DT = np.dtype(ml_dtypes.bfloat16)
    HD = 64
causal = np.zeros((1, KL), bool); causal[0, :5] = True
slide = np.zeros((1, KL), bool); slide[0, 1:5] = True
q = (rng.rand(2, 1, H, HD) * 0.5).astype(DT)
kk = (rng.rand(2, KL, KVH, HD) * 0.5).astype(DT)
vv = (rng.rand(2, KL, KVH, HD) * 0.5).astype(DT)
sinks = rng.rand(H).astype(DT)

if WHICH == "gptoss":
    out = jax.jit(lambda a, b, c, s: gptoss(
        a, b, c, s, None, jnp.asarray(causal), H // KVH))(q, kk, vv, sinks)
elif WHICH == "gptoss_sliding":
    out = jax.jit(lambda a, b, c, s: gptoss(
        a, b, c, s, jnp.asarray(slide), jnp.asarray(causal),
        H // KVH))(q, kk, vv, sinks)
elif WHICH == "gemma4":
    mask = np.repeat(causal[None], 2, axis=0)
    out = jax.jit(lambda a, b, c: gemma4(a, b, c, jnp.asarray(mask)))(
        q, kk, vv)
else:
    sq = (rng.rand(2, H, 1, HD) * 0.5).astype(np.float32)
    sk = (rng.rand(2, H, KL, HD) * 0.5).astype(np.float32)
    sm = np.zeros((2, H, 1, KL), bool); sm[..., :5] = True
    out = jax.jit(lambda a, b, c: small(a, b, c, jnp.asarray(sm)))(sq, sk, sk)
print(f"[probe] checksum {float(np.asarray(out).sum()):.9e}")
'''


def _keras_attn_tags(subprocess, tempfile, pathlib, re):
    """(label, check) pairs for the two keras-hub decode attention spellings.

    Rows 7 (`GptOssCausalLM`) and 3 (`Gemma4CausalLM`) recognized ZERO fused
    attentions before this: the numeric rows in `_recognizer_cases` pass
    whether or not they fuse, because an unfused attention computes the same
    thing.  What has to be proven is that each one fuses, that it fuses for
    the RIGHT reason (the `+sink` / `+mask2` tags in the recognizer's own
    narration), that the widened mask-sentinel rule still refuses a `select`
    whose constant is merely small, and that `METALJAX_SDPA=0` turns all of
    it off.
    """

    def run(which, extra_env=(), dtype="float32"):
        env = dict(os.environ)
        env["METALJAX_DEBUG"] = "1"
        env["KERAS_ATTN_WHICH"] = which
        env["KERAS_ATTN_DTYPE"] = dtype
        env.update(dict(extra_env))
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(_KERAS_ATTN)
            script = fh.name
        try:
            return subprocess.run([sys.executable, script], env=env,
                                  capture_output=True, text=True)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass

    _FUSED = re.compile(
        r"sdpa: (\d+) fused attention\(s\) recognized: (.*)")

    def census(which, extra_env=(), dtype="float32"):
        proc = run(which, extra_env, dtype)
        err = proc.stderr or ""
        n, names = 0, ""
        for m in _FUSED.finditer(err):
            n += int(m.group(1))
            names += " " + m.group(2)
        sums = [float(ln.split()[2]) for ln in (proc.stdout or "").splitlines()
                if ln.startswith("[probe] checksum ")]
        return n, names, (sums[0] if sums else None), proc

    def fuses(which, want_tag, dtype="float32"):
        n, names, ssum, proc = census(which, dtype=dtype)
        if ssum is None:
            return False, f"{which} did not run: {(proc.stderr or '')[-300:]}"
        off_n, _, off_sum, _ = census(which, {"METALJAX_SDPA": "0"}, dtype)
        # In f32 the fused kernel reproduces the literal chain exactly; in
        # bf16 it does NOT and must not be asked to -- it accumulates the
        # probabilities in f32, which is the documented sdpa class (and, as
        # `hd64.py` measures against a float64 reference, the more accurate
        # side).  So the bf16 arm checks the TAG, not the checksum.
        tol = 1e-5 if dtype == "float32" else 2e-2
        ok = (n == 1 and want_tag in names and off_n == 0
              and off_sum is not None
              and abs(ssum - off_sum) <= tol * max(abs(ssum), 1.0))
        return ok, (f"{n} fused{names} (want 1 with {want_tag}); "
                    f"SDPA=0 gives {off_n}; "
                    f"checksum {ssum:.9g} vs {off_sum:.9g} unfused")

    return [
        ("gpt-oss decode attention fuses with its sink",
         lambda: fuses("gptoss", "+sink")),
        ("a gpt-oss sliding layer fuses both of its masks",
         lambda: fuses("gptoss_sliding", "+sink+mask2")),
        ("gemma4 grouped decode attention fuses",
         lambda: fuses("gemma4", "B2N")),
        # ...and in bf16 at D = 64, which is the dtype and the head dim the
        # rows actually run and the only way into MLX's fast vector kernel.
        ("gpt-oss fuses in bf16 through the vector kernel",
         lambda: fuses("gptoss_sliding", "D64+sink+mask2", "bfloat16")),
        ("gemma4 fuses in bf16 through the vector kernel",
         lambda: fuses("gemma4", "D64", "bfloat16")),
        ("a small select constant is still not a mask",
         lambda: (lambda c: (c[0] == 0,
                             f"{c[0]} fused{c[1]} (want none)"))(
             census("small"))),
    ]


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
    # T2: the gate compares the budget with the SMALLER of the MLIR walk
    # (`cost=`) and the body's post-pass tape (`tape=`), so the budget this
    # contract derives is taken from that gate cost -- the fused tape is
    # one entry where the chain was ~30, so the discount is still there.
    _GATE = re.compile(r"while gate: cost=(\d+)(?: tape=(\d+))? .*? "
                       r"body_compile=(\d+)")

    def arm(extra_env):
        proc = run(extra_env)
        err = proc.stderr or ""
        gates = [(min(int(m.group(1)),
                      int(m.group(2) or m.group(1))), int(m.group(3)))
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
        return ok, (f"gate cost {on['cost']} fused / {off['cost']} unfused; at "
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
                r"main: pure=\d+ cost=\d+ (?:tape=\d+ )?bytes=([\d.]+)MB",
                proc.stderr)]
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

    def bf16_takes_a_kernel():
        """The topconfs16k cliff (2026-08-22): Stage 1's dtype table has no
        bf16 row, so every bf16 scan fell to the cascade -- 25x slower and a
        62 us/timestep plateau.  The native table maps bf16 -> bfloat16_t;
        the bf16 msl cases must plan in ALL THREE modes, and the old
        `not eligible (dtype bf16)` must never come back."""
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--msl-bf16"], env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).splitlines()[-1][:80]
        dtype_declines = len(re.findall(r"not eligible \(dtype bf16\)",
                                        proc.stderr))
        if dtype_declines:
            return False, f"{dtype_declines} 'dtype bf16' decline(s) are back"
        modes = re.findall(r"msl_scan: compiled plan .*?mode=(\w+)",
                           proc.stderr)
        missing = {"scalar", "vector", "coop"} - set(modes)
        if missing:
            return False, (f"no bf16 plan in {sorted(missing)} mode "
                           f"(saw {sorted(set(modes))})")
        return True, f"{len(modes)} bf16 kernels: {', '.join(sorted(set(modes)))}"

    def the_f4_flip_fires():
        """The coop flip at F=4 (2026-08-26): square F=4 cells ran vector
        mode -- batch-only lanes, ONE simdgroup, weights re-read per timestep
        -- because 0.4.3's bound (F >= 8) left the F<=4 pocket unmeasured.
        Measured on the topconfs16k regressions (tc038 rnn.4 2.0x, tc044
        mgru.4 2.4x, tc046 1.8x, the latter two now BEAT jax-CPU), the bound
        admits F=4.  Pinned here: the flip fires by default, the knob
        restores the old policy, and the two modes agree on the answer."""
        def run(env_extra):
            child = dict(os.environ)
            child["METALJAX_DEBUG"] = "1"
            child.update(env_extra)
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()),
                 "--msl-f4-coop"], env=child, capture_output=True, text=True)
            if proc.returncode != 0:
                return None, (proc.stderr or proc.stdout).splitlines()[-1][:80]
            modes = re.findall(r"msl_scan: compiled plan .*?mode=(\w+)",
                               proc.stderr)
            out = re.search(r"F4 COOP CHECKSUM ([-\d.e+]+)", proc.stdout)
            return (modes, out.group(1) if out else None), None

        flip, err = run({})
        if err: return False, err
        old, err = run({"METALJAX_MSL_COOP_MIN_F": "8"})
        if err: return False, err
        if "coop" not in flip[0] or "vector" in flip[0]:
            return False, f"default planned {flip[0]}, wanted coop only"
        if "vector" not in old[0] or "coop" in old[0]:
            return False, (f"COOP_MIN_F=8 planned {old[0]}, wanted vector "
                           "only")
        d = abs(float(flip[1]) - float(old[1]))
        rel = d / max(abs(float(old[1])), 1e-30)
        # The two modes contract the recurrent dot in different orders; the
        # bar catches a wrong kernel, not an accumulation order.
        if rel > 1e-4:
            return False, f"coop vs vector answers differ by {rel:.2e}"
        return True, (f"F=4 flips to coop (vector under COOP_MIN_F=8), "
                      f"answers agree to {rel:.1e}")

    def only_the_induction_variable_is_a_counter():
        """The 2026-09-02 non-counter-carry bug, pinned at the CLASSIFICATION
        rather than at the answer.

        `MslAnalyzer::Analyze` used to seed a SymCounter for the counter
        `_analyze_counted` found AND for every i32/i64 scalar carry.  A
        SymCounter is read out of the iteration index, so those carries came
        back as 0, 1, 2, ... -- and were not kernel inputs at all.  The five
        carry cases compare EXACTLY against the CPU, but a decline would make
        them pass by falling back, so what is checked here is the census the
        plugin narrates: each of those loops must plan, and each plan must
        hold EXACTLY ONE counter, the induction variable.

        Under the bug the counts were 2, 2, 1, 2 and 2 (the float-carry case
        was always right, which is why it is in the list).
        """
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--msl-carries"], env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).splitlines()[-1][:80]
        plans = re.findall(
            r"msl_scan: compiled plan .*?counters=(\d+) passthrough=(\d+)",
            proc.stderr)
        if len(plans) != 5:
            declines = re.findall(r"msl_scan: not eligible \((.*?)\)",
                                  proc.stderr)
            return False, (f"{len(plans)} of 5 carry loops planned"
                           + (f" (declines: {declines[:3]})" if declines
                              else ""))
        extra = [c for c, _ in plans if c != "1"]
        if extra:
            return False, (f"a plan classified {extra[0]} carries as "
                           "counters; only the induction variable is one")
        # The two invariant cases must ALSO hand their carry back untouched
        # rather than recompute it, which is the pass-through rule.
        if sum(int(p) for _, p in plans) < 2:
            return False, "no plan carries a pass-through"
        return True, ("5 loops planned, 1 counter each, "
                      f"{sum(int(p) for _, p in plans)} pass-throughs")

    def one_scalar_per_lane_takes_a_kernel():
        """The one-scalar-per-lane cases (2026-09-02), pinned at the plan.

        Their EXACT comparison against the CPU is satisfied by a decline too,
        and declining was 2125c96's answer for the 1-D stacked shape.  What
        is checked here is that each of the five loops PLANS, in vector mode,
        on the lane space its shapes imply -- `lane=` in the narration -- so
        the value shaped like that space was read as one scalar per lane and
        not as `lane[-1]` registers.
        """
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--msl-lane-scalars"], env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).splitlines()[-1][:80]
        plans = re.findall(
            r"msl_scan: compiled plan .*?mode=(\w+) .*?lane=([\d,]+)",
            proc.stderr)
        want = ["4", "3,4", "4", "4,4", "3,4"]
        if len(plans) != len(want):
            declines = re.findall(r"msl_scan: not eligible \((.*?)\)",
                                  proc.stderr)
            return False, (f"{len(plans)} of {len(want)} lane-scalar loops "
                           "planned"
                           + (f" (declines: {declines[:3]})" if declines
                              else ""))
        modes = [m for m, _ in plans]
        if any(m != "vector" for m in modes):
            return False, f"modes {modes}, all must be vector"
        lanes = [l for _, l in plans]
        if lanes != want:
            return False, f"lane spaces {lanes}, expected {want}"
        return True, f"5 loops planned in vector mode on lanes {lanes}"

    return [("msl covers its three modes", modes_are_covered),
            ("METALJAX_MSL=0 builds no kernel", the_kill_switch_kills),
            ("msl coop flip at F=4", the_f4_flip_fires),
            ("msl coop width cap (F>=1024)", the_width_cap_holds),
            ("msl loop charged as one kernel",
             a_planned_loop_is_charged_as_one_kernel),
            ("bf16 msl plans build", bf16_takes_a_kernel),
            ("only the induction variable is an msl counter",
             only_the_induction_variable_is_a_counter),
            ("one scalar per lane takes a kernel",
             one_scalar_per_lane_takes_a_kernel)]


def _p31_norm(subprocess, pathlib, re):
    """The RMS-norm recognizer's coverage of the model table's spellings.

    P30 built `metal_norm.cc` on maxtext's norm and, measured on the captured
    lowerings, matched NONE of the dense band: gemma squares in bf16 and
    takes its rsqrt there, keras squares with `power` and applies the weight
    in f32 under a downcast, and both apply the weight as a broadcast
    multiply rather than maxtext's batching dot.  No correctness test could
    see that -- every answer stayed right, just built out of ~13 ops instead
    of one -- so what is pinned here is that each spelling FIRES, and with
    the form the matcher claims to have recognized it by.

    The answers are pinned twice over: against the SAME binary with
    METALJAX_NORM=0 (which runs the literal chain, so the two arms differ by
    the fusion and nothing else) and against jax-CPU in its own process.
    """
    here = str(pathlib.Path(__file__).resolve())

    def arm(env_extra, platform="metal"):
        import json
        child = dict(os.environ)
        if platform == "cpu":
            child.pop("METALJAX_PLUGIN_PATH", None)
            child["JAX_PLATFORMS"] = "cpu"
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--norm-forms"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if not line.startswith("NORM "):
                continue
            label, payload = line[5:].split("\t", 1)
            answers[label] = np.array(json.loads(payload))
        tags = re.findall(r"norm: matched an? (?:rms|layer) norm \(([^,]*),",
                          proc.stderr)
        return answers, tags

    def worst(a, b):
        """Max error against the tensor's own scale."""
        scale = max(float(np.max(np.abs(b))), 1e-30)
        return float(np.max(np.abs(a - b))) / scale

    # bf16 keeps 8 mantissa bits, so one ULP is 2^-8..2^-7 relative; the bar
    # is two of the wide ones.  Measured worst across these seven: 7.5e-3
    # (gemma, whose chain rounds the MEAN to bf16 and rsqrts it there -- the
    # fused kernel is the more accurate side of that difference).  f32 sits
    # at 1.5e-7, MLX's `precise::rsqrt` against XLA's.
    BAR = {"bf16": 2 * 2.0**-7, "f32": 1e-6}

    def every_spelling_fires():
        _fused, tags = arm({})
        forms = _norm_forms()
        want = [tag for _l, _t, tag, _d, _n in forms if tag is not None]
        missing = [t for t in want if not any(t in got for got in tags)]
        if missing:
            return False, f"no match tagged {missing} (saw {sorted(set(tags))})"
        # The count matters as much as the tags: the stacked pair shares its
        # hoisted constants, and a rewrite that claimed them would fuse ONE
        # of the two with every tag still present -- and the narrowing-
        # promotion cell (tag None) must fuse nothing, which only the count
        # can see, since a wrong fusion there is still numerically close.
        n = sum(k for _l, _t, _g, _d, k in forms)
        if len(tags) != n:
            return False, (f"{len(tags)} matches over {len(forms)} modules, "
                           f"wanted {n}")
        return True, (f"{len(forms)} modules, {n} norms fused: "
                      f"{', '.join(sorted(set(tags)))}")

    def the_kill_switch_kills():
        _answers, tags = arm({"METALJAX_NORM": "0"})
        return (not tags), (f"{len(tags)} matches with METALJAX_NORM=0"
                            if tags else "no match with METALJAX_NORM=0")

    def fused_agrees_with_the_literal_chain():
        fused, _ = arm({})
        literal, tags = arm({"METALJAX_NORM": "0"})
        if tags:
            return False, "the METALJAX_NORM=0 arm still fused"
        bad = []
        for label, _text, _tag, dt, _n in _norm_forms():
            e = worst(fused[label], literal[label])
            if e > BAR[dt]:
                bad.append(f"{label} {e:.2e} > {BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        e = max(worst(fused[l], literal[l])
                for l, _t, _g, _d, _n in _norm_forms())
        return True, (f"{len(_norm_forms())} modules agree with the chain, "
                      f"worst {e:.1e}")

    def fused_agrees_with_cpu():
        fused, _ = arm({})
        cpu, tags = arm({}, platform="cpu")
        if tags:
            return False, "the CPU arm loaded the plugin"
        bad = []
        for label, _text, _tag, dt, _n in _norm_forms():
            e = worst(fused[label], cpu[label])
            if e > BAR[dt]:
                bad.append(f"{label} {e:.2e} > {BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        e = max(worst(fused[l], cpu[l]) for l, _t, _g, _d, _n in _norm_forms())
        return True, (f"{len(_norm_forms())} modules agree with jax-CPU, "
                      f"worst {e:.1e}")

    return [("norm covers the table's spellings", every_spelling_fires),
            ("METALJAX_NORM=0 fuses nothing", the_kill_switch_kills),
            ("norm fused == the literal chain",
             fused_agrees_with_the_literal_chain),
            ("norm fused == jax-CPU", fused_agrees_with_cpu)]


def _p40_mla_kernel(subprocess, pathlib, re):
    """B3: the two-span decode attention KERNEL (runtime/mla.cc).

    The concat path joins the spans for MLX's fused sdpa -- concat(K),
    concat(V), a mask per span and their concat -- about eight copy kernels
    per layer around the one that attends, 26 layers a token on row 10.  The
    kernel reads both spans, the values and the segment ids in place, and it
    is MLX's own `sdpa_vector` with two key pointers, so its answer is meant
    to be the concat path's TO THE BIT (the contract: bit-identical, or
    within the fused-attention ULP class the recognizer discloses -- the
    detail line says which held).

    Pinned: the kernel BUILDS for every eligible geometry (narration, with
    the Spec key), the kill switch keeps the recognizer but runs the concat
    path, the kernel's answer against the kill switch, and against jax-CPU.
    """
    here = str(pathlib.Path(__file__).resolve())

    def arm(env_extra, platform="metal"):
        import json
        child = dict(os.environ)
        if platform == "cpu":
            child.pop("METALJAX_PLUGIN_PATH", None)
            child["JAX_PLATFORMS"] = "cpu"
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--mla-kernel-forms"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if not line.startswith("MLK "):
                continue
            label, payload = line[4:].split("\t", 1)
            answers[label] = np.array(json.loads(payload))
        built = re.findall(r"mla: kernel built mjn_mla2_\d+ for (\S+)",
                           proc.stderr)
        matched = re.findall(
            r"mla: matched a multi-span attention \(([^,]*),", proc.stderr)
        disabled = "kernel disabled (METALJAX_MLA_KERNEL=0)" in proc.stderr
        return answers, built, matched, disabled

    def worst(a, b):
        scale = max(float(np.max(np.abs(b))), 1e-30)
        return float(np.max(np.abs(a - b))) / scale

    # The disclosed class (fused attention, f32 softmax inside the kernel):
    # a couple of ULP of the model dtype -- `_p32_mla`'s bar.
    BAR = {"bf16": 2 * 2.0**-7, "f32": 1e-6}

    def the_kernel_builds():
        _a, built, matched, _d = arm({})
        want = [tag for _l, _t, tag, _d in _mla_kernel_forms()]
        missing = [t for t in want if not any(b.startswith(t) for b in built)]
        if missing:
            return False, f"no kernel built for {missing} (built {built})"
        if len(matched) != len(_mla_kernel_forms()):
            return False, (f"{len(matched)} matches over "
                           f"{len(_mla_kernel_forms())} modules")
        return True, f"{len(built)} kernels built: {', '.join(built)}"

    def the_kill_switch_keeps_the_match():
        _a, built, matched, disabled = arm({"METALJAX_MLA_KERNEL": "0"})
        if built:
            return False, f"{len(built)} kernels built with the switch off"
        if not disabled:
            return False, "no `kernel disabled` narration"
        if len(matched) != len(_mla_kernel_forms()):
            return False, f"the recognizer stopped firing ({len(matched)})"
        return True, "no kernel, the recognizer still fires, concat path runs"

    def the_kernel_is_the_concat_paths_answer():
        on, built, _m, _d = arm({})
        off, built_off, _m2, _d2 = arm({"METALJAX_MLA_KERNEL": "0"})
        if not built or built_off:
            return False, "the arms did not split on the kernel"
        exact, close, bad = [], [], []
        for label, _text, _tag, dt in _mla_kernel_forms():
            if on[label].shape == off[label].shape and \
                    np.array_equal(on[label], off[label]):
                exact.append(label)
                continue
            e = worst(on[label], off[label])
            (close if e <= BAR[dt] else bad).append(f"{label} {e:.2e}")
        if bad:
            return False, "; ".join(bad)
        if close:
            return True, (f"{len(exact)} bit-identical; within the disclosed "
                          f"ULP class: {'; '.join(close)}")
        return True, f"{len(exact)} forms bit-identical with the concat path"

    def the_kernel_agrees_with_cpu():
        fused, built, _m, _d = arm({})
        cpu, cbuilt, _m2, _d2 = arm({}, platform="cpu")
        if not built:
            return False, "the kernel did not build"
        if cbuilt:
            return False, "the CPU arm loaded the plugin"
        bad = []
        for label, _text, _tag, dt in _mla_kernel_forms():
            e = worst(fused[label], cpu[label])
            if e > BAR[dt]:
                bad.append(f"{label} {e:.2e} > {BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        e = max(worst(fused[l], cpu[l])
                for l, _t, _g, _d in _mla_kernel_forms())
        return True, (f"{len(_mla_kernel_forms())} forms agree with jax-CPU, "
                      f"worst {e:.1e}")

    return [("mla kernel builds", the_kernel_builds),
            ("mla kernel kill switch", the_kill_switch_keeps_the_match),
            ("mla kernel == concat path", the_kernel_is_the_concat_paths_answer),
            ("mla kernel == jax-CPU", the_kernel_agrees_with_cpu)]


def _p41_stacked_relayout(subprocess, pathlib, re):
    """B3: the stacked dot's RELAYOUT form (metal_stacked.cc).

    maxtext's attention out-projection stack is [heads, L, head_dim, model]:
    the layer axis sits between the two contracted axes, no [L, K, N] view of
    the buffer exists, and the recognizer declined -- so row 10 copied 8.4 MB
    of weights per layer per token (218 MB/token, ~0.93 ms) through a dynamic
    slice.  The relayout materializes the CARRIED (transposed) layout once
    per executable as a pack, keyed by the argument's identity like a
    quantized pack, and the emit gathers from it in place.

    Pinned: the match FIRES with `relayout planned` and the pack wave
    narrates `relaid` for both spellings; METALJAX_STACKED_RELAYOUT=0 keeps
    the decline (the slice chain runs, no pack); and the answers against the
    kill switch and against METALJAX_STACKED_DOT=0 -- the relayout is the
    same gather_mm over the same matrix at another address, so against its
    own kill switch it is bit-exact; against the slice chain the detail line
    says whether the bits held (a different MLX matmul kernel may accumulate
    K in another order at M > 1).
    """
    here = str(pathlib.Path(__file__).resolve())

    def arm(env_extra, platform="metal"):
        import json
        child = dict(os.environ)
        if platform == "cpu":
            child.pop("METALJAX_PLUGIN_PATH", None)
            child["JAX_PLATFORMS"] = "cpu"
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here,
                               "--stacked-relayout-forms"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if not line.startswith("SRL "):
                continue
            label, payload = line[4:].split("\t", 1)
            answers[label] = np.array(json.loads(payload))
        planned = re.findall(
            r"stacked: matched a stacked dot \(([^,]*), \d+ ops absorbed, "
            r"relayout planned\)", proc.stderr)
        relaid = re.findall(r"stacked: relaid (\S+) \(", proc.stderr)
        matched = re.findall(
            r"stacked: matched a stacked dot \(([^,]*),", proc.stderr)
        return answers, planned, relaid, matched

    def worst(a, b):
        scale = max(float(np.max(np.abs(b))), 1e-30)
        return float(np.max(np.abs(a - b))) / scale

    def the_relayout_fires():
        _a, planned, relaid, _m = arm({})
        n = len(_stacked_relayout_forms())
        if len(planned) < n:
            return False, f"{len(planned)} relayouts planned over {n} forms"
        if len(relaid) < n:
            return False, f"{len(relaid)} stacks relaid over {n} forms"
        return True, f"{len(relaid)} relaid: {', '.join(relaid)}"

    def the_kill_switch_declines():
        _a, planned, relaid, matched = arm({"METALJAX_STACKED_RELAYOUT": "0"})
        if planned or relaid:
            return False, f"{len(planned)}/{len(relaid)} with the switch off"
        if matched:
            return False, f"{len(matched)} matches: these stacks must decline"
        return True, "declined, as before B3"

    def multi_row_reads_stay_sliced():
        """The form is decode-only: at M > 1 (a prefill program) the stack
        keeps its slice chain, with the narration saying why."""
        _a, planned, relaid, matched = arm({"B3_RELAYOUT_M": "3"})
        if planned or relaid:
            return False, f"{len(planned)}/{len(relaid)} relayouts at M=3"
        if matched:
            return False, f"{len(matched)} matches at M=3"
        return True, "M=3 forms decline (decode-only form)"

    def bit_exact_against_the_switch():
        on, planned, _r, _m = arm({})
        off, p2, _r2, _m2 = arm({"METALJAX_STACKED_RELAYOUT": "0"})
        if not planned or p2:
            return False, "the arms did not split on the relayout"
        bad = [k for k in on if on[k].shape != off[k].shape
               or not np.array_equal(on[k], off[k])]
        if bad:
            e = max(worst(on[k], off[k]) for k in bad)
            return False, f"{len(bad)} form(s) differ, worst {e:.2e}: {bad}"
        return True, f"{len(on)} forms bit-identical with the relayout off"

    def against_the_slice_chain_and_cpu():
        on, planned, _r, _m = arm({})
        chain, _p, _r2, matched = arm({"METALJAX_STACKED_DOT": "0"})
        cpu, _p3, _r3, _m3 = arm({}, platform="cpu")
        if matched:
            return False, "STACKED_DOT=0 still matched"
        exact, close, bad = [], [], []
        for k in on:
            if np.array_equal(on[k], chain[k]):
                exact.append(k)
            else:
                e = worst(on[k], chain[k])
                (close if e <= 1e-5 else bad).append(f"{k} {e:.2e}")
        ecpu = max(worst(on[k], cpu[k]) for k in on)
        if bad or ecpu > 1e-5:
            return False, f"chain: {'; '.join(bad)}; cpu worst {ecpu:.2e}"
        how = (f"{len(exact)} bit-identical with the slice chain"
               if not close else
               f"{len(exact)} bit-identical, reordered: {'; '.join(close)}")
        return True, f"{how}; vs jax-CPU worst {ecpu:.1e}"

    return [("stacked relayout fires", the_relayout_fires),
            ("stacked relayout kill switch", the_kill_switch_declines),
            ("stacked relayout is decode-only", multi_row_reads_stay_sliced),
            ("stacked relayout is bit-exact", bit_exact_against_the_switch),
            ("stacked relayout vs chain/CPU", against_the_slice_chain_and_cpu)]


def _p33_gdn(subprocess, pathlib, re):
    """The gated-delta-net decode step: it FIRES, it is the whole block, and
    it is on the literal chain's own values.

    Row 8's decode body is 8,756 tape entries per token, and 6,669 of them
    (76 %) are the 30 GDN layers' straight-line delta rule -- no scan, no
    chunked recurrence, nothing an existing recognizer matched
    (`~/.cache/metaljax-bench/logs/moe-diag/diagnosis.md` §2b).  What the
    fusion removes is not only entries: the [B,Hv,Dk,Dv] f32 state is
    materialised FIVE times per layer per token by the broadcast spelling,
    and the kernel reads it once and writes it once.

    So what is pinned is the MATCH, not only the answer -- a recognizer that
    silently stopped at the recurrence and left the two `_l2norm`s running
    would give identical numbers and none of the win.  Four arms plus the
    chained one: each geometry fires with its form tag; the kill switch
    kills; the fused answer IS the literal chain's on the same binary; and
    eight chained steps, with both caches fed forward, still agree with
    jax-CPU.
    """
    here = str(pathlib.Path(__file__).resolve())

    def arm(flag, env_extra, platform="metal"):
        import json
        child = dict(os.environ)
        if platform == "cpu":
            child.pop("METALJAX_PLUGIN_PATH", None)
            child["JAX_PLATFORMS"] = "cpu"
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, flag],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if not line.startswith("GDN "):
                continue
            label, payload = line[4:].split("\t", 1)
            answers[label] = np.array(json.loads(payload))
        tags = re.findall(
            r"gdn: matched a gated delta step \(([^,]*),", proc.stderr)
        return answers, tags

    def worst(a, b):
        scale = max(float(np.max(np.abs(b))), 1e-30)
        return float(np.max(np.abs(a - b))) / scale

    # Against the CPU: the same band the fused attentions use -- the model
    # dtype's couple of ULP.  Against the LITERAL CHAIN on this binary the
    # bar is far tighter, because the kernel replays every narrowing the
    # chain does and only the order of the two dk reductions is left; at
    # these extents it measures 0.0 for the outputs and 2.5e-08 for the
    # state, and the bar below is one bf16 ULP of headroom over that.
    CPU_BAR = {"bf16": 2 * 2.0**-7, "f32": 1e-6}
    CHAIN_BAR = {"bf16": 1e-3, "f32": 1e-6}

    def both_forms_fire():
        _fused, tags = arm("--gdn-forms", {})
        want = [tag for _l, _t, tag, _d in _gdn_forms()]
        missing = [t for t in want if t not in tags]
        if missing:
            return False, f"no match tagged {missing} (saw {sorted(set(tags))})"
        if len(tags) != len(_gdn_forms()):
            return False, (f"{len(tags)} matches over {len(_gdn_forms())} "
                           "modules, wanted one each")
        return True, f"{len(tags)} fused: {', '.join(tags)}"

    def the_kill_switch_kills():
        _answers, tags = arm("--gdn-forms", {"METALJAX_GDN": "0"})
        if tags:
            return False, f"{len(tags)} matches with METALJAX_GDN=0"
        return True, "no match under METALJAX_GDN=0"

    def fused_agrees_with_the_literal_chain():
        fused, _ = arm("--gdn-forms", {})
        literal, tags = arm("--gdn-forms", {"METALJAX_GDN": "0"})
        if tags:
            return False, "the METALJAX_GDN=0 arm still fused"
        bad = []
        for label, _text, _tag, dt in _gdn_forms():
            for key in (label, label + " state"):
                e = worst(fused[key], literal[key])
                if e > CHAIN_BAR[dt]:
                    bad.append(f"{key} {e:.2e} > {CHAIN_BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        e = max(worst(fused[k], literal[k]) for k in fused)
        return True, (f"{len(fused)} values on the chain's own numbers, "
                      f"worst {e:.1e}")

    def fused_agrees_with_cpu():
        fused, _ = arm("--gdn-forms", {})
        cpu, tags = arm("--gdn-forms", {}, platform="cpu")
        if tags:
            return False, "the CPU arm loaded the plugin"
        bad = []
        for label, _text, _tag, dt in _gdn_forms():
            for key in (label, label + " state"):
                e = worst(fused[key], cpu[key])
                if e > CPU_BAR[dt]:
                    bad.append(f"{key} {e:.2e} > {CPU_BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        e = max(worst(fused[k], cpu[k]) for k in fused)
        return True, f"{len(fused)} values agree with jax-CPU, worst {e:.1e}"

    def chained_steps_agree_with_cpu():
        """Eight decode steps, both caches fed forward: the arm that sees
        drift."""
        fused, tags = arm("--gdn-chain", {})
        if not tags:
            return False, "no match in the chained arm"
        cpu, _ = arm("--gdn-chain", {}, platform="cpu")
        bad = []
        for label, _text, _tag, dt in _gdn_forms():
            e = worst(fused[label], cpu[label])
            if e > CPU_BAR[dt]:
                bad.append(f"{label} {e:.2e} > {CPU_BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        e = max(worst(fused[l], cpu[l]) for l, _t, _g, _d in _gdn_forms())
        return True, f"8 chained steps agree with jax-CPU, worst {e:.1e}"

    return [("gdn covers the f32 and bf16 spellings", both_forms_fire),
            ("METALJAX_GDN=0 fuses nothing", the_kill_switch_kills),
            ("gdn fused == the literal chain",
             fused_agrees_with_the_literal_chain),
            ("gdn fused == jax-CPU", fused_agrees_with_cpu),
            ("gdn chained 8 steps == jax-CPU", chained_steps_agree_with_cpu)]


def _p34_start_plan(subprocess, pathlib, re):
    """The dynamic-slice START PLAN: it fires, and it changes no answer.

    `AppendStartPlan` (metal_lowering.cc) keeps only the axes that need a
    runtime start and resolves the rest at lowering, so the handler stops
    building a `stack` of rank-0 zeros -- an `ExpandDims` per axis plus a
    `Concatenate`, none of them in MLX's fusable set, all of them walked
    again at every replay of a compiled loop body.

    Nothing about the tape's SHAPE changes -- same entries, same slots --
    so unlike a recognizer this rewrite is invisible to a tape census.  What
    is pinned here is therefore the narration (it fires, on the axis counts
    the modules declare) and, against the kill switch, that the answers are
    identical to the LAST BIT: this is pure index bookkeeping and there is
    no tolerance to spend.
    """
    here = str(pathlib.Path(__file__).resolve())

    def arm(env_extra):
        import json
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--start-plan-forms"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if line.startswith("DSP "):
                label, payload = line[4:].split("\t", 1)
                answers[label] = np.array(json.loads(payload))
        tally = [(int(a), int(b)) for a, b in re.findall(
            r"ds plan: (\d+) dynamic slice/update op\(s\), "
            r"(\d+) start axes resolved away", proc.stderr)]
        return answers, tally

    def the_plan_fires_and_counts():
        _a, tally = arm({})
        if not tally:
            return False, "no `ds plan:` narration at all"
        ops = sum(t[0] for t in tally)
        dropped = sum(t[1] for t in tally)
        # Eight forms, one slice or update each. What each sheds: the two
        # rank-4 forms drop their three constant-zero axes (3 + 3), the two
        # all-zero forms collapse to the plan's single kept axis (1 + 1),
        # and the mixed / all-constant forms drop nothing -- both of THEIR
        # axes have a non-zero clamped start, which the plan must carry.
        # 8 ops, 8 axes.
        if ops < 8:
            return False, f"only {ops} ops carried a plan, wanted >= 8"
        if dropped < 8:
            return False, f"only {dropped} axes resolved away, wanted >= 8"
        return True, f"{ops} ops, {dropped} start axes resolved away"

    def the_kill_switch_is_bit_exact():
        on, _ = arm({})
        off, tally = arm({"METALJAX_DS_PLAN": "0"})
        if not tally:
            return False, "the off arm lost the narration too (lowering knob?)"
        bad = [k for k in on
               if on[k].shape != off[k].shape
               or not np.array_equal(on[k], off[k])]
        if bad:
            return False, f"{len(bad)} form(s) differ: {bad[:3]}"
        return True, f"{len(on)} forms bit-identical with the plan off"

    return [("ds start plan fires", the_plan_fires_and_counts),
            ("ds start plan is bit-exact", the_kill_switch_is_bit_exact)]


def _p32_mla(subprocess, pathlib, re):
    """The multi-span decode attention's coverage of BOTH head geometries.

    The recognizer shipped in P30 against maxtext's MLA decode, where every
    query head has its own KV head.  On row 11 (Qwen3-0.6B: 16 query heads
    over 8 KV heads) it declined -- `the keys have the wrong shape`, because
    the keys carry Hkv heads and the matcher compared them against H -- and
    the whole attention ran literally: 81 of the layer body's 216 tape
    entries, 37 %, replayed 28 times per decoded token.  No correctness
    test could see that -- the answers were right, just built out of 82 ops
    instead of one -- which is why what follows pins the MATCH, not only
    the answer.

    So what is pinned is that each geometry FIRES, tagged with the Hkv the
    matcher read; that the kill switches still kill (BOTH of them --
    METALJAX_MLA and the shared METALJAX_SDPA); and that the fused answer is
    the literal chain's and jax-CPU's, at each dtype's own bar.
    """
    here = str(pathlib.Path(__file__).resolve())

    def arm(env_extra, platform="metal"):
        import json
        child = dict(os.environ)
        if platform == "cpu":
            child.pop("METALJAX_PLUGIN_PATH", None)
            child["JAX_PLATFORMS"] = "cpu"
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--mla-forms"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if not line.startswith("MLA "):
                continue
            label, payload = line[4:].split("\t", 1)
            answers[label] = np.array(json.loads(payload))
        tags = re.findall(
            r"mla: matched a multi-span attention \(([^,]*),", proc.stderr)
        return answers, tags

    def worst(a, b):
        scale = max(float(np.max(np.abs(b))), 1e-30)
        return float(np.max(np.abs(a - b))) / scale

    # The fused kernel does the whole softmax in f32; the literal chain
    # subtracts, exps and renormalizes in the model dtype.  bf16 keeps 8
    # mantissa bits, so the bar is a couple of ULP of the wide values --
    # the same band `_p31_norm` uses, and the same "documented fused
    # attention class" the recognizer's header claims.
    BAR = {"bf16": 2 * 2.0**-7, "f32": 1e-6}

    def both_geometries_fire():
        _fused, tags = arm({})
        want = [tag for _l, _t, tag, _d in _mla_forms()]
        missing = [t for t in want if not any(t in got for got in tags)]
        if missing:
            return False, f"no match tagged {missing} (saw {sorted(set(tags))})"
        if len(tags) != len(_mla_forms()):
            return False, (f"{len(tags)} matches over {len(_mla_forms())} "
                           "modules, wanted one each")
        return True, f"{len(tags)} fused: {', '.join(tags)}"

    def the_kill_switches_kill():
        """Two switches, and BOTH have to work: METALJAX_MLA is this
        recognizer's own, METALJAX_SDPA is the shared attention one that
        `AnalyzeMla` also honours."""
        for knob in ("METALJAX_MLA", "METALJAX_SDPA"):
            _answers, tags = arm({knob: "0"})
            if tags:
                return False, f"{len(tags)} matches with {knob}=0"
        return True, "no match under METALJAX_MLA=0 or METALJAX_SDPA=0"

    def fused_agrees_with_the_literal_chain():
        fused, _ = arm({})
        literal, tags = arm({"METALJAX_MLA": "0"})
        if tags:
            return False, "the METALJAX_MLA=0 arm still fused"
        bad = []
        for label, _text, _tag, dt in _mla_forms():
            e = worst(fused[label], literal[label])
            if e > BAR[dt]:
                bad.append(f"{label} {e:.2e} > {BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        e = max(worst(fused[l], literal[l]) for l, _t, _g, _d in _mla_forms())
        return True, (f"{len(_mla_forms())} modules agree with the chain, "
                      f"worst {e:.1e}")

    def fused_agrees_with_cpu():
        fused, _ = arm({})
        cpu, tags = arm({}, platform="cpu")
        if tags:
            return False, "the CPU arm loaded the plugin"
        bad = []
        for label, _text, _tag, dt in _mla_forms():
            e = worst(fused[label], cpu[label])
            if e > BAR[dt]:
                bad.append(f"{label} {e:.2e} > {BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        e = max(worst(fused[l], cpu[l]) for l, _t, _g, _d in _mla_forms())
        return True, (f"{len(_mla_forms())} modules agree with jax-CPU, "
                      f"worst {e:.1e}")

    return [("mla covers MLA and GQA", both_geometries_fire),
            ("mla kill switches fuse nothing", the_kill_switches_kill),
            ("mla fused == the literal chain",
             fused_agrees_with_the_literal_chain),
            ("mla fused == jax-CPU", fused_agrees_with_cpu)]


def _p37_gqa(subprocess, pathlib, re):
    """The keras GQA head repeat, absorbed into the fused attention
    (gap-rows item 6).

    keras-hub lifts the Hkv cached heads to the H query heads with an
    explicit `ops.repeat` BEFORE the attention the recognizer matches, so the
    fused kernel used to read a materialized copy of K and V -- ~1.07 GB per
    token on row 20, 48 copies on row 7.  MLX's kernel takes Hkv < H
    directly, and `repeat` lays the copies out in exactly its `h // G`
    order, so peeling the pair off is data movement only.  No numeric row
    can see any of that (the repeated fusion computes the same thing), so
    what is pinned is the MATCH: that the repeat is absorbed at the row-7 /
    row-20 geometries in both dtypes and both jax spellings, tagged with the
    head counts the matcher read; that a group-major `tile` is refused and
    the two kernel-moving geometries keep their repeat; that
    METALJAX_SDPA_GQA=0 restores the plain fusion; that the absorbed answer
    equals the plain fusion's TO THE LAST BIT; and that it is jax-CPU's at
    the documented fused-attention bar.
    """
    here = str(pathlib.Path(__file__).resolve())

    def arm(env_extra, platform="metal"):
        import json
        child = dict(os.environ)
        if platform == "cpu":
            child.pop("METALJAX_PLUGIN_PATH", None)
            child["JAX_PLATFORMS"] = "cpu"
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--gqa-forms"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if not line.startswith("GQA "):
                continue
            label, payload = line[4:].split("\t", 1)
            answers[label] = np.array(json.loads(payload))
        absorbed = re.findall(
            r"sdpa: gqa absorbed \(H=(\d+) Hkv=(\d+)\) (\S+)", proc.stderr)
        kept = re.findall(r"sdpa: gqa repeat kept x(\d+): (.*)", proc.stderr)
        fused = sum(int(n) for n in re.findall(
            r"sdpa: (\d+) fused attention\(s\) recognized", proc.stderr))
        return answers, absorbed, kept, fused

    def worst(a, b):
        scale = max(float(np.max(np.abs(b))), 1e-30)
        return float(np.max(np.abs(a - b))) / scale

    BAR = {"bf16": 2 * 2.0**-7, "f32": 1e-6}
    forms = _gqa_forms()
    # (label, absorbed, dtype, kept-reason) over both tables.
    expect = ([(l, ab, dt, w) for l, _t, ab, dt, w in forms]
              + [(l, ab, dt, w) for l, _f, _a, ab, dt, w in _gqa_jit_forms()])
    n_programs = len(expect)

    def the_repeat_is_absorbed():
        _answers, absorbed, kept, fused = arm({})
        if fused != n_programs:
            return False, f"{fused} fused attentions over {n_programs} programs"
        want = [(str(h), str(hkv)) for _l, ab, _d, _w in expect
                if ab is not None for h, hkv in [ab]]
        got = [(h, hkv) for h, hkv, _name in absorbed]
        if sorted(got) != sorted(want):
            return False, f"absorbed {got}, wanted {want}"
        if any("+gqa" not in name for _h, _hkv, name in absorbed):
            return False, f"an absorbed match is not tagged +gqa: {absorbed}"
        # The ones that must KEEP their repeat, each for its own reason:
        # the two group-major tiles, the 2-pass geometry, and the Tq * G >
        # 32 prefill.
        reasons = [w for _l, _ab, _d, w in expect if w is not None]
        missing = [r for r in reasons
                   if not any(r in why for _n, why in kept)]
        if missing:
            return False, f"no 'repeat kept' for {missing} (saw {kept})"
        return True, (f"{len(absorbed)} absorbed "
                      f"{' '.join(f'H{h}/Hkv{k}' for h, k in got)}; "
                      f"{len(kept)} kept as spelled")

    def the_kill_switch_keeps_the_plain_fusion():
        _answers, absorbed, _kept, fused = arm({"METALJAX_SDPA_GQA": "0"})
        if absorbed:
            return False, f"{len(absorbed)} absorbed under METALJAX_SDPA_GQA=0"
        if fused != n_programs:
            return False, (f"{fused} fused attentions over {n_programs} "
                           "programs with the absorb off")
        return True, f"{fused} plain fusions, none absorbed"

    def absorbed_equals_plain_bit_for_bit():
        with_, absorbed, _k, _f = arm({})
        without, _a, _k2, _f2 = arm({"METALJAX_SDPA_GQA": "0"})
        if not absorbed:
            return False, "nothing was absorbed"
        bad = [label for label in with_
               if with_[label].tobytes() != without[label].tobytes()]
        if bad:
            return False, "not bit-identical: " + "; ".join(bad)
        return True, (f"{len(with_)} answers identical to the last bit "
                      "across the switch")

    def fused_agrees_with_cpu():
        fused, _a, _k, _f = arm({})
        cpu, absorbed, _k2, _f2 = arm({}, platform="cpu")
        if absorbed:
            return False, "the CPU arm loaded the plugin"
        bad = []
        for label, _ab, dt, _w in expect:
            e = worst(fused[label], cpu[label])
            if e > BAR[dt]:
                bad.append(f"{label} {e:.2e} > {BAR[dt]:.0e}")
        if bad:
            return False, "; ".join(bad)
        return True, f"{len(fused)} programs agree with jax-CPU"

    return [("gqa repeat absorbed into sdpa", the_repeat_is_absorbed),
            ("gqa kill switch keeps the plain fusion",
             the_kill_switch_keeps_the_plain_fusion),
            ("gqa absorbed == plain, bit for bit",
             absorbed_equals_plain_bit_for_bit),
            ("gqa fused == jax-CPU", fused_agrees_with_cpu)]


def _p35_while_pipeline(subprocess, pathlib, re):
    """Which dynamic whiles pipeline, and that the layout cannot change an
    answer.

    A dynamic (data-dependent) `while` can hide its two host round trips per
    iteration by building iteration t+1 and its condition BEFORE reading
    iteration t's condition back.  Whether that pays is one structural fact:
    a COMPILED body speculates for one graph call, an EAGER one for
    `num_ops()` interpreted entries.  `control.cc` used to gate both on the
    entry count, which cost row 11's keras arm (a 1,844-entry whole-model
    decode body, compiled, replayed once per token) 0.8 ms/token in blocking
    round trips it was not saving a build with.

    So the gate is pinned from both sides -- a big body pipelines when it
    compiles and goes serial when it does not -- and, because a speculative
    build that got COMMITTED would run one iteration too many, every arm's
    answer is compared against the serial one.  The carry reports the trip
    count, so "one iteration too many" is visible as a number.
    """
    here = str(pathlib.Path(__file__).resolve())
    BIG = "while_loop (dynamic trip, big compiled body)"
    SMALL = "while_loop (dynamic trip)"

    def arm(env_extra):
        import json
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--dynamic-while"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if line.startswith("WHILE "):
                label, payload = line[6:].split("\t", 1)
                answers[label] = json.loads(payload)
        # One executable per case, each narrating its own census line.
        census = [(int(a), int(b)) for a, b in re.findall(
            r"serial_loops=(\d+) pipelined_loops=(\d+)", proc.stderr)]
        return answers, census

    def totals(census):
        return (sum(s for s, _p in census), sum(p for _s, p in census))

    def a_compiled_body_pipelines():
        answers, census = arm({})
        serial, piped = totals(census)
        if BIG not in answers or SMALL not in answers:
            return False, f"ran {sorted(answers)}"
        # Two dynamic loops in the run, both eligible: the small one by the
        # entry count, the big one only because its body compiles.
        if (serial, piped) != (0, 2):
            return False, (f"serial_loops={serial} pipelined_loops={piped}, "
                           "wanted 0/2")
        return True, "both dynamic loops pipelined"

    def an_eager_big_body_stays_serial():
        """METALJAX_BODY_COMPILE=0 leaves the body interpreted, and then the
        entry-count rule is the whole gate again -- the protection the row-5
        calibration bought (65.1 vs 60.6 ms/step with pipelining forced on an
        op-by-op decode body)."""
        _answers, census = arm({"METALJAX_BODY_COMPILE": "0"})
        serial, piped = totals(census)
        if (serial, piped) != (1, 1):
            return False, (f"serial_loops={serial} pipelined_loops={piped}, "
                           "wanted 1/1 (the big body serial, the small one "
                           "still pipelined)")
        return True, "the eager 400-entry body went serial"

    def the_knob_still_disables():
        _answers, census = arm({"METALJAX_WHILE_PIPELINE": "0"})
        serial, piped = totals(census)
        if (serial, piped) != (2, 0):
            return False, (f"serial_loops={serial} pipelined_loops={piped}, "
                           "wanted 2/0")
        return True, "METALJAX_WHILE_PIPELINE=0 serialized both"

    def ahead_lines(text):
        """(steps, ahead_steps, ahead_declined) per `while(pipelined...)`
        narration line, in order."""
        return [(int(a), int(b), int(c)) for a, b, c in re.findall(
            r"while\(pipelined(?:\+ahead)?\): steps=(\d+) ahead_steps=(\d+) "
            r"ahead_declined=(\d+)", text)]

    def arm_text(env_extra):
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--dynamic-while"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        return proc.stdout + proc.stderr

    def a_pipelined_loop_submits_ahead():
        """B4: from the third iteration on, iteration t+1 is SUBMITTED
        before t's condition is read (`ahead_steps`), and the step count
        is what the serial arm says it is -- the speculation that runs
        past the stop is dropped, never committed."""
        loops = ahead_lines(arm_text({}))
        if not loops:
            return False, "no pipelined-loop narration with ahead counters"
        # Both dynamic loops run many steps; each must have submitted
        # every step but its first two ahead (the first is serial, the
        # second measures) -- or declined it for headroom, which the
        # governor may do on a loaded machine and which still reads first.
        bad = [f"steps={st} ahead={ah} declined={de}"
               for st, ah, de in loops if st >= 3 and ah + de != st - 2]
        if bad:
            return False, "; ".join(bad)
        if not any(ah > 0 for _st, ah, _de in loops):
            return False, f"nothing submitted ahead: {loops}"
        return True, ", ".join(f"steps={st} ahead={ah} declined={de}"
                               for st, ah, de in loops)

    def the_ahead_knob_reads_first():
        loops = ahead_lines(arm_text({"METALJAX_WHILE_SUBMIT_AHEAD": "0"}))
        if loops:
            return False, f"METALJAX_WHILE_SUBMIT_AHEAD=0 still narrates ahead counters: {loops}"
        _answers, census = arm({"METALJAX_WHILE_SUBMIT_AHEAD": "0"})
        if totals(census) != (0, 2):
            return False, f"knob off: census {totals(census)}, wanted 0/2 (still pipelined)"
        return True, "knob off: both loops still pipeline, read-first"

    def the_layout_cannot_move_an_answer():
        """Same ops, same order, only the sync points move -- so the answers
        must be BIT-identical, trip count included.  This is the check that a
        mis-speculated final iteration would fail."""
        base, _ = arm({})
        bad = []
        for label, extra in (("serial", {"METALJAX_WHILE_PIPELINE": "0"}),
                             ("eager body", {"METALJAX_BODY_COMPILE": "0"}),
                             ("read-first", {"METALJAX_WHILE_SUBMIT_AHEAD": "0"})):
            other, _ = arm(extra)
            for case in (BIG, SMALL):
                if base.get(case) != other.get(case):
                    bad.append(f"{case} differs under the {label} layout: "
                               f"{base.get(case)} vs {other.get(case)}")
        if bad:
            return False, "; ".join(bad)
        return True, f"4 layouts agree exactly, {BIG} = {base[BIG]}"

    return [("a compiled while body pipelines", a_compiled_body_pipelines),
            ("an eager big body stays serial", an_eager_big_body_stays_serial),
            ("while-pipeline knob still disables", the_knob_still_disables),
            ("a pipelined loop submits ahead", a_pipelined_loop_submits_ahead),
            ("submit-ahead knob reads first", the_ahead_knob_reads_first),
            ("loop layout cannot move an answer",
             the_layout_cannot_move_an_answer)]


def _p39_kv_inplace(subprocess, pathlib, re):
    """The KV in-place rewrite fires on the stacked-cache decode shape,
    writes the cache in place (the vendored MLX's donation through the
    stream's pins), declines the one shape it must, and cannot move an
    answer: every case is BIT-identical with METALJAX_KV_INPLACE=0."""
    here = str(pathlib.Path(__file__).resolve())

    def arm(env_extra):
        import json
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--kv-inplace"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if line.startswith("KV "):
                label, payload = line[3:].split("\t", 1)
                answers[label] = json.loads(payload)
        text = proc.stdout + proc.stderr
        return answers, text

    def the_rewrite_fires():
        answers, text = arm({})
        rewrote = re.findall(r"kv inplace: rewrote carry (\d+): updates=(\d+)",
                             text)
        declined = re.findall(r"kv inplace declined carry \d+: (.*)", text)
        # Every program is lowered twice (the plain and the fused lowering),
        # so each case narrates twice.  The two dynamic f32 cases (wrapping,
        # and B4's no-wrap tail-rows case) rewrite (2 updated layers x k,v =
        # 4 updates each), the bf16 case rewrites all 3 layers (6), the
        # read-after case DECLINES with the reason, and the fori_loop case
        # narrates nothing: its rebuild sits inside jax's `closed_call`
        # wrapper, which the analysis does not see through (documented).
        updates = sorted(int(u) for _c, u in rewrote)
        if updates != [4, 4, 4, 4, 6, 6]:
            return False, f"rewrote updates {updates}, wanted [4, 4, 4, 4, 6, 6]"
        if len(declined) != 2 or not all(
                "read after an overlapping update" in d for d in declined):
            return False, f"declines {declined}, wanted 2 x read-after"
        return True, (f"rewrote {len(rewrote)} carries (updates {updates}), "
                      f"declined {len(declined)} (read-after)")

    def loop_counts(text):
        """(steps, ahead_steps, donated, copied) of the longest dynamic loop."""
        loops = re.findall(
            r"while\(pipelined(?:\+ahead)?\): steps=(\d+)(?: ahead_steps=(\d+))?"
            r".*?slice_update_donated=(\d+) slice_update_copied=(\d+)", text)
        if not loops:
            return None
        return max(((int(a), int(b or 0), int(c), int(d))
                    for a, b, c, d in loops), key=lambda t: t[0])

    def the_cache_is_written_in_place():
        """On the dynamic loop (12 steps, 4 updates each), read-first
        (METALJAX_WHILE_SUBMIT_AHEAD=0), the chain donates every time but
        the first iteration, whose carry the loop still holds: the
        vendored MLX counts the slice updates that wrote into their
        operand vs copied it first."""
        _answers, text = arm({"METALJAX_WHILE_SUBMIT_AHEAD": "0"})
        got = loop_counts(text)
        if got is None:
            return False, "no pipelined-loop narration"
        steps, _ahead, donated, copied = got
        if donated < 4 * (steps - 1) or copied > 4 + 2:
            return False, (f"steps={steps} donated={donated} copied={copied}: "
                           "the chain did not write in place")
        return True, f"steps={steps} donated={donated} copied={copied}"

    def a_speculation_copies_the_root_and_nothing_else():
        """B4 (submit-ahead): a step submitted before its predecessor's
        condition is read must NOT write into the carry the loop still
        holds -- so its chain copies the cache ROOT into a fresh buffer,
        exactly one copy per ahead step, and donates the rest of the
        chain.  Pinned from both sides: fewer copies would mean a
        speculation wrote in place into a carry the loop might return;
        more would mean the chain stopped donating."""
        _answers, text = arm({})
        got = loop_counts(text)
        if got is None:
            return False, "no pipelined-loop narration"
        steps, ahead, donated, copied = got
        if ahead == 0:
            return False, f"steps={steps} ahead_steps=0: not submitting ahead"
        # Exactly: the first iteration copies once (the loop's own hold),
        # every ahead step copies once (the held carry), and so does the
        # speculation built past the stop -- which RUNS, into buffers
        # nothing reads, and is dropped (steps+1 iterations of 4 updates
        # reach the device).  Every other update in every chain donates.
        want_copied = 2 + ahead
        if copied != want_copied:
            return False, (f"steps={steps} ahead={ahead} copied={copied}, "
                           f"wanted {want_copied} (one root copy per "
                           "speculative iteration, one for the first)")
        if donated != 4 * (steps + 1) - copied:
            return False, (f"steps={steps} donated={donated} copied={copied}: "
                           f"wanted {4 * (steps + 1) - copied} donations "
                           "(the rest of every chain, the dropped one included)")
        return True, (f"steps={steps} ahead={ahead} donated={donated} "
                      f"copied={copied} (= steps+1 iterations x 4 updates)")

    def the_knob_restores_the_literal_tape():
        _answers, text = arm({"METALJAX_KV_INPLACE": "0"})
        if "kv inplace" in text:
            return False, "METALJAX_KV_INPLACE=0 still narrates the rewrite"
        return True, "no rewrite under the knob"

    def the_rewrite_cannot_move_an_answer():
        base, _ = arm({})
        bad = []
        for label, extra in (("the rewrite off", {"METALJAX_KV_INPLACE": "0"}),
                             ("read-first", {"METALJAX_WHILE_SUBMIT_AHEAD": "0"}),
                             ("serial", {"METALJAX_WHILE_PIPELINE": "0"})):
            other, _ = arm(extra)
            for case in sorted(base):
                if base[case] != other.get(case):
                    bad.append(f"{case} under {label}")
        if bad:
            return False, "differs: " + "; ".join(bad)
        return True, (f"{len(base)} cases bit-identical with the rewrite off, "
                      "read-first and serial")

    return [("kv in-place rewrite fires", the_rewrite_fires),
            ("kv cache is written in place", the_cache_is_written_in_place),
            ("a speculation copies the root and nothing else",
             a_speculation_copies_the_root_and_nothing_else),
            ("kv knob restores the literal tape",
             the_knob_restores_the_literal_tape),
            ("kv rewrite cannot move an answer",
             the_rewrite_cannot_move_an_answer)]


def _p42_rope_view(subprocess, pathlib, re):
    """The rotate-half rope apply lowers as a view (metal_rope.cc): the
    rewrite fires on both spellings and at both shapes, removes dispatches,
    is switched off by METALJAX_ROPE_VIEW=0, and cannot move an answer --
    every case is BIT-identical with the knob off."""
    here = str(pathlib.Path(__file__).resolve())

    def arm(env_extra):
        import json
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--rope-view"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if line.startswith("ROPE "):
                label, payload = line[5:].split("\t", 1)
                answers[label] = json.loads(payload)
        return answers, proc.stdout + proc.stderr

    def dispatches(text):
        return {m.group(1): int(m.group(2)) for m in re.finditer(
            r"\[metaljax-native\] (jit_rope_\w+): flushes=.*?dispatches=(\d+)",
            text)}

    def the_rewrite_fires():
        _answers, text = arm({})
        matched = re.findall(
            r"rope: matched (\d+) rotate-half apply\(ies\) as views "
            r"\((\d+) pair-stacked, (\d+) last-axis\)", text)
        views = re.findall(r"(\d+) rope view\(s\)", text)
        # Four programs, two applies each (q and k): three pair-stacked
        # (keras decode, keras prefill, the scan body) and one last-axis
        # concatenate.
        counts = sorted(int(n) for n, _s, _c in matched)
        forms = (sum(int(s) for _n, s, _c in matched),
                 sum(int(c) for _n, _s, c in matched))
        if counts != [2, 2, 2, 2] or forms != (6, 2):
            return False, f"matched {matched}"
        if sorted(int(v) for v in views) != [2, 2, 2, 2]:
            return False, f"rope view(s) narrated {views}"
        return True, f"4 programs, {forms[0]} pair-stacked + {forms[1]} last-axis"

    def the_rewrite_removes_dispatches():
        _a, on = arm({})
        _b, off = arm({"METALJAX_ROPE_VIEW": "0"})
        d_on, d_off = dispatches(on), dispatches(off)
        if not d_on or set(d_on) != set(d_off):
            return False, f"dispatch narration on={d_on} off={d_off}"
        worse = [k for k in d_on if d_on[k] >= d_off[k]]
        if worse:
            return False, f"no fewer dispatches on {worse}: on={d_on} off={d_off}"
        return True, ", ".join(f"{k[4:]} {d_off[k]}->{d_on[k]}" for k in sorted(d_on))

    def the_knob_restores_the_literal_tape():
        _answers, text = arm({"METALJAX_ROPE_VIEW": "0"})
        if "rope: matched" in text:
            return False, "METALJAX_ROPE_VIEW=0 still narrates the rewrite"
        if re.search(r"[1-9]\d* rope view\(s\)", text):
            return False, "METALJAX_ROPE_VIEW=0 still counts rope views"
        return True, "no rewrite under the knob"

    def the_rewrite_cannot_move_an_answer():
        base, _ = arm({})
        other, _ = arm({"METALJAX_ROPE_VIEW": "0"})
        bad = [c for c in sorted(base) if base[c] != other.get(c)]
        if bad:
            return False, "differs under the knob: " + "; ".join(bad)
        return True, f"{len(base)} cases bit-identical with the rewrite off"

    return [("rope view rewrite fires", the_rewrite_fires),
            ("rope view removes dispatches", the_rewrite_removes_dispatches),
            ("rope view knob restores the literal tape",
             the_knob_restores_the_literal_tape),
            ("rope view cannot move an answer",
             the_rewrite_cannot_move_an_answer)]


def _p43_proj_pack(subprocess, pathlib, re):
    """B6: the projection PACK (metal_proj.cc).

    Row 7's K and V projections are two latency-bound [2880 -> 512] bf16
    gemvs; the rope view shortened q's chain and the pair stopped landing in
    one concurrent dispatch group (+41 us per layer).  The pack makes them
    ONE dot over the concatenated weights, materialized once per executable
    (governor-admitted, budgeted, keyed by the weights' identity), with
    per-consumer slice views.

    Pinned, on the forms of `_proj_pack_forms`: the match FIRES and PACKS
    (narration and the executable's count); both kill switches restore the
    literal tape; fewer dispatches; every answer whose members share the
    pack's gemv band (`_gemv_band`) is BIT-identical with
    METALJAX_PROJ_PACK=0 -- the same kernel, the same K order -- and a
    band-crossing one (F2, F4, F9) holds a ULP with its bits REPORTED, not
    failed (design section 4: a PSO change is compiler-dependent); prefill
    declines; the consumers (bias, rope, kv_update, norms) match jax-CPU;
    the `kv`/`all` policies; the qmm recognizer keeps its dots; the byte
    cap and a donated weight decline; the pack's layout follows the
    weights' storage ([N, K] weights pack [n_total, K] and stay
    bit-identical); a tuple barrier is looked through; a weight the
    program updates declines; fresh weights repack a bounded number of
    times and then the executable re-lowers WITHOUT the projection packs,
    keeping its other recognizers.
    """
    here = str(pathlib.Path(__file__).resolve())
    memo = {}

    def arm(env_extra, platform="metal"):
        import json
        key = (platform, tuple(sorted(env_extra.items())))
        if key in memo:
            return memo[key]
        child = dict(os.environ)
        if platform == "cpu":
            child.pop("METALJAX_PLUGIN_PATH", None)
            child["JAX_PLATFORMS"] = "cpu"
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--proj-pack"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if line.startswith("PPK "):
                label, payload = line[4:].split("\t", 1)
                answers[label] = [np.array(a) for a in json.loads(payload)]
        text = proc.stdout + proc.stderr
        memo[key] = (answers, text)
        return memo[key]

    def proj_lines(text):
        return [ln for ln in text.splitlines() if "[metaljax-native] proj:" in ln]

    def matched(text):
        return re.findall(r"proj: matched a projection pack \((\S+), (\d+) dots, ([\d.]+) MB", text)

    def packed(text):
        return re.findall(r"proj: packed (\S+) \(([\d.]+) MB once, (\d+) weights", text)

    def exe_pack_counts(text):
        """Every fused-tape narration per program, in order: {jit name:
        [pack count, ...]} -- one entry per (re)lowering."""
        out = {}
        for m in re.finditer(r"\[metaljax-native\] (jit_proj_\w+): .*?(\d+) projection pack\(s\)", text):
            out.setdefault(m.group(1), []).append(int(m.group(2)))
        return out

    def exe_packs(text):
        """First fused-tape narration per program: {jit name: pack count}."""
        return {k: v[0] for k, v in exe_pack_counts(text).items()}

    def dispatches(text):
        out = {}
        for m in re.finditer(r"\[metaljax-native\] (jit_proj_\w+): flushes=.*?dispatches=(\d+)", text):
            out.setdefault(m.group(1), int(m.group(2)))
        return out

    def same(a, b):
        return len(a) == len(b) and all(
            x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))

    def worst(a, b):
        e = 0.0
        for x, y in zip(a, b):
            scale = max(float(np.max(np.abs(y))), 1e-30)
            e = max(e, float(np.max(np.abs(x - y))) / scale)
        return e

    def form_of(label):
        return label.split()[0]

    # A band-crossing pack accumulates in f32 and rounds once: at most one
    # ULP of the output dtype (bf16 forms), or an f32 reorder's few ULP.
    F32_FORMS = {"F5", "F6", "F10"}

    def ulp_band(form):
        return 1e-5 if form in F32_FORMS else 2 ** -7

    def compare_arms(on, off, bands=_PROJ_FORM_BANDS, skip=()):
        """Every label of `on` against `off`: same-band forms must be
        bit-identical, band-crossing ones within a ULP.  Returns
        (failures, exact labels, reported labels, worst error)."""
        bad, exact, reported, e_max = [], [], [], 0.0
        for k in sorted(on):
            f = form_of(k)
            if f in skip or k not in off:
                continue
            if same(on[k], off[k]):
                exact.append(k)
                continue
            e = worst(on[k], off[k])
            if f in bands and _proj_same_band(f, bands):
                bad.append(f"{k} differs by {e:.2e} (same gemv band: must be bit-identical)")
            elif e > ulp_band(f):
                bad.append(f"{k} differs by {e:.2e} (> 1 ULP)")
            else:
                reported.append(k)
                e_max = max(e_max, e)
        return bad, exact, reported, e_max

    # The pack each form builds under the DEFAULT policy (`auto`: F1 K+V,
    # F2 Q+K+V by the tile rule).
    KV_NAMES = {"F1": "m1k2880n512+512", "F2": "m1k1024n2048+1024+1024",
                "F4": "m1k2048n2048+512", "F5": "m1k2048n32+32",
                "F6": "m1k256n128+128", "F9": "m1k8192n1536+1024",
                "F10": "m1k2560n384+384", "F11": "m1k640n192+192"}

    def the_pack_fires():
        _a, text = arm({})
        names = {n for n, _d, _mb in matched(text)}
        built = {n for n, _mb, _w in packed(text)}
        missing = [f"{f} {n}" for f, n in KV_NAMES.items()
                   if n not in names or n not in built]
        if missing:
            return False, f"not matched+packed: {missing}; lines: {proj_lines(text)[:8]}"
        exe = exe_packs(text)
        want = {"jit_proj_f1": 1, "jit_proj_f2": 1, "jit_proj_f4": 1,
                "jit_proj_f5": 1, "jit_proj_f6": 1, "jit_proj_f9": 1,
                "jit_proj_f10": 1, "jit_proj_f11": 1}
        bad = {k: exe.get(k) for k in want if exe.get(k) != want[k]}
        if bad:
            return False, f"executable pack counts {bad} (want 1 each); have {exe}"
        return True, (f"{len(KV_NAMES)} forms matched and packed: "
                      + ", ".join(sorted(built)))

    def the_kill_switches_decline():
        for env in ({"METALJAX_PROJ_PACK": "0"}, {"METALJAX_PROJ_PACK_MB": "0"}):
            _a, text = arm(env)
            if proj_lines(text):
                return False, f"{env} still narrates: {proj_lines(text)[:3]}"
            if re.search(r"[1-9]\d* projection pack\(s\)", text):
                return False, f"{env} still counts packs"
        return True, "PROJ_PACK=0 and PROJ_PACK_MB=0 narrate nothing, count zero"

    def the_pack_removes_dispatches():
        _a, on = arm({})
        _b, off = arm({"METALJAX_PROJ_PACK": "0"})
        d_on, d_off = dispatches(on), dispatches(off)
        progs = ["jit_proj_f1", "jit_proj_f2", "jit_proj_f4", "jit_proj_f5",
                 "jit_proj_f6", "jit_proj_f10", "jit_proj_f11"]
        if any(p not in d_on or p not in d_off for p in progs):
            return False, f"dispatch narration on={d_on} off={d_off}"
        worse = [p for p in progs if d_on[p] >= d_off[p]]
        if worse:
            return False, f"no fewer dispatches on {worse}: on={d_on} off={d_off}"
        return True, ", ".join(f"{p[9:]} {d_off[p]}->{d_on[p]}" for p in progs)

    def the_pack_cannot_move_an_answer():
        on, _t = arm({})
        off, _u = arm({"METALJAX_PROJ_PACK": "0"})
        if set(on) != set(off):
            return False, f"labels differ: {sorted(set(on) ^ set(off))}"
        strict = sorted(f for f in _PROJ_FORM_BANDS if _proj_same_band(f))
        crossing = sorted(f for f in _PROJ_FORM_BANDS if not _proj_same_band(f))
        # The forms the contract was designed around must fall where the
        # design says: K/V-shaped packs in one band, F2/F4 (bn4 -> bn16)
        # and F9 (K = 8192, N crossing 2048) across one.
        if not {"F1", "F5", "F6", "F10", "F11"} <= set(strict):
            return False, f"band table: strict={strict}"
        if not {"F2", "F4", "F9"} <= set(crossing):
            return False, f"band table: crossing={crossing}"
        bad, exact, reported, e = compare_arms(on, off)
        if bad:
            return False, "differs under the knob: " + "; ".join(bad)
        held = [k for k in exact if form_of(k) in crossing]
        n_cross = sum(1 for k in on if form_of(k) in crossing)
        return True, (f"{len(exact)} cases bit-identical with the pack off "
                      f"(same-band forms {strict} all held); band-crossing "
                      f"{crossing}: bits held on {len(held)} of {n_cross}, "
                      f"worst {e:.1e}")

    def prefill_declines():
        _a, text = arm({})
        decl = [ln for ln in proj_lines(text) if "multi-row (M=59)" in ln]
        if not decl:
            return False, f"no multi-row decline: {proj_lines(text)[:6]}"
        if any(n.startswith("m59") for n, _d, _mb in matched(text)):
            return False, "an M=59 group matched"
        if exe_packs(text).get("jit_proj_f3", 0) != 0:
            return False, "the prefill executable counts packs"
        return True, f"{len(decl)} M=59 dots declined (decode-only), no pack"

    def consumers_match_cpu():
        on, _t = arm({})
        cpu, _u = arm({}, platform="cpu")
        report, bad = [], []
        # bf16 forms: the plugin rounds each op where XLA:CPU keeps f32
        # across fusions, and F1 loops three steps; the band is the bf16
        # differential's (HALF) widened for the loop.
        bands = {"F1 keras decode loop": 2.5e-2, "F2 row11 norms call1": 8e-3,
                 "F2 row11 norms call2": 8e-3, "F2 row11 norms call3": 8e-3,
                 "F2 row11 norms call4": 8e-3, "F4 mismatched heads": 8e-3,
                 "F5 tiny f32 pair": 1e-5, "F6 constant pair": 1e-5,
                 "F7 quantized pair": 1e-5, "F9 K8192 crossing 2048": 8e-3,
                 "F10 NK-stored f32 pair": 1e-5, "F11 tuple barrier pair": 8e-3,
                 "F12 updated pair": 8e-3}
        for k, band in bands.items():
            if k not in on or k not in cpu:
                return False, f"missing {k}: metal={sorted(on)} cpu={sorted(cpu)}"
            e = worst(on[k], cpu[k])
            report.append(f"{k.split()[0]} {e:.1e}")
            if e > band:
                bad.append(f"{k} {e:.2e} > {band:.0e}")
        if bad:
            return False, "; ".join(bad)
        return True, "vs jax-CPU worst: " + ", ".join(report)

    def the_policy():
        # `auto` (the default): F1's K+V (512 + 512 = 1024) stays in the
        # members' gemv_t bn4 tile, so q is left alone; F2's K+V (1024 +
        # 1024 = 2048) would cross into bn16 for nothing, so q is packed too
        # (Q+K+V = 4096 shares q's own tile).  `kv` leaves q alone on both,
        # `all` packs everything.
        _a, au = arm({})
        _b, kv = arm({"METALJAX_PROJ_PACK": "kv"})
        _c, al = arm({"METALJAX_PROJ_PACK": "all"})
        au_left = [ln for ln in proj_lines(au) if "policy auto leaves the widest" in ln]
        au_took = [ln for ln in proj_lines(au) if "policy auto packs the widest" in ln]
        if len(au_left) < 1 or len(au_took) < 1:
            return False, (f"auto narrated leave x{len(au_left)}, pack x"
                           f"{len(au_took)} (want F1 left, F2 packed)")
        kv_left = [ln for ln in proj_lines(kv) if "policy kv leaves the widest" in ln]
        if len(kv_left) < 2:
            return False, f"kv policy narrated {len(kv_left)} times (want F1, F2)"
        au_names = {n for n, _mb, _w in packed(au)}
        kv_names = {n for n, _mb, _w in packed(kv)}
        al_names = {n for n, _mb, _w in packed(al)}
        want_au = {"m1k2880n512+512", "m1k1024n2048+1024+1024", "m1k2048n32+32"}
        want_kv = {"m1k2880n512+512", "m1k1024n1024+1024", "m1k2048n32+32"}
        want_all = {"m1k2880n4096+512+512", "m1k1024n2048+1024+1024",
                    "m1k2048n32+32"}
        if not want_au <= au_names:
            return False, f"auto packed {sorted(au_names)}"
        if not want_kv <= kv_names:
            return False, f"kv packed {sorted(kv_names)}"
        if not want_all <= al_names:
            return False, f"all packed {sorted(al_names)}"
        if "policy kv" in al or "policy auto" in al:
            return False, "the all policy still narrates a policy"
        on, _t = arm({"METALJAX_PROJ_PACK": "all"})
        off, _u = arm({"METALJAX_PROJ_PACK": "0"})
        # Under `all` k/v cross into q's bn16 band (F1, F2): reported.
        bad, exact, reported, e = compare_arms(on, off, _PROJ_FORM_BANDS_ALL)
        if bad:
            return False, "all differs from the literal dots: " + "; ".join(bad)
        return True, (f"auto packs K+V on F1 and Q+K+V on F2 (the tile rule), "
                      f"kv packs K+V on both, all packs Q+K+V; F5 under all "
                      f"three; all vs off: bits held on {len(exact)} of "
                      f"{len(on)}, worst {e:.1e}")

    def quantized_dots_stay_qmm():
        _a, text = arm({})
        q = re.findall(r"qmm: packed (\S+)", text)
        if len(q) < 2:
            return False, f"qmm packed {q}"
        stray = [ln for ln in proj_lines(text)
                 if "k512n64" in ln or "[1,512] x [512,64]" in ln]
        if stray:
            return False, f"proj touched the quantized dots: {stray}"
        return True, f"{len(q)} qmm packs, no proj line on their dots"

    def the_byte_cap_declines():
        cap, text = arm({"METALJAX_PROJ_PACK_MB": "1"})
        decl = [ln for ln in proj_lines(text)
                if "declined over budget" in ln and "m1k2880n512+512" in ln]
        if not decl:
            return False, f"no over-budget decline: {proj_lines(text)[-6:]}"
        if "m1k2880n512+512" in {n for n, _mb, _w in packed(text)}:
            return False, "F1 packed over the cap"
        off, _u = arm({"METALJAX_PROJ_PACK": "0"})
        k = "F1 keras decode loop"
        if not same(cap[k], off[k]):
            return False, f"F1 under the cap differs from the literal dots by {worst(cap[k], off[k]):.2e}"
        small = {n for n, _mb, _w in packed(text)}
        return True, (f"F1 declined over a 1 MB cap and lowered literally "
                      f"(bit-identical); under the cap: {sorted(small)}")

    def a_donated_weight_declines():
        on, text = arm({})
        decl = [ln for ln in proj_lines(text) if "the weight is donated" in ln]
        if not decl:
            return False, f"no donation decline: {proj_lines(text)[:6]}"
        if "m1k512n256+256" in {n for n, _mb, _w in packed(text)}:
            return False, "the donated pair packed"
        if exe_packs(text).get("jit_proj_f8", 0) != 0:
            return False, "the donated executable counts packs"
        off, _u = arm({"METALJAX_PROJ_PACK": "0"})
        calls = [k for k in on if k.startswith("F8")]
        if len(calls) != 3 or any(not same(on[k], off[k]) for k in calls):
            return False, f"F8 calls {calls} differ from the literal dots"
        return True, f"declined ({len(decl)} narrations), 3 calls literal and identical"

    def fresh_weights_repack():
        # F2 runs four times with fresh weights: the pack is keyed by their
        # identity, so calls 2 and 3 repack (narrated, 1 of 2 / 2 of 2) and
        # call 4 re-lowers WITHOUT the projection packs -- the executable's
        # fourth fused narration counts 0 packs but exists (its head norms
        # keep the fused tape), where the old rule would have retired every
        # recognizer after eight misses or re-lowered forever.
        on, text = arm({})
        n = sum(1 for nm, _mb, _w in packed(text)
                if nm == "m1k1024n2048+1024+1024")
        if n != 3:
            return False, f"F2 packed {n} times over 4 calls with fresh weights (want 3)"
        rep = re.findall(r"jit_proj_f2: the projection packs' weights changed \(argument (\d+)\): repacking \((\d) of 2\)", text)
        if [r[1] for r in rep] != ["1", "2"]:
            return False, f"repack narration {rep} (want 1 of 2, 2 of 2)"
        off_line = re.search(r"jit_proj_f2: the projection packs' weights changed 3 times \(argument \d+\): re-lowering without them", text)
        if off_line is None:
            return False, "no 're-lowering without them' narration for F2"
        if "proj: left out of this tape" not in text:
            return False, "the fourth lowering did not narrate the packs left out"
        counts = exe_pack_counts(text).get("jit_proj_f2")
        if counts != [1, 1, 1, 0]:
            return False, f"F2's fused narrations count {counts} packs (want [1, 1, 1, 0]: the fourth tape keeps the norms, drops the pack)"
        if "jit_proj_f2: the packed weights changed" in text:
            return False, "F2 counted against the whole-tape repack bound"
        off, _u = arm({"METALJAX_PROJ_PACK": "0"})
        calls = {k: on[k] for k in on if k.startswith("F2 ")}
        bad, exact, reported, e = compare_arms(calls, off)
        if bad or len(calls) != 4:
            return False, f"F2 calls {sorted(calls)}: {bad}"
        return True, (f"calls 2-3 repacked, call 4 re-lowered with 0 packs and "
                      f"the norms kept; 4 calls vs the pack off: bits held on "
                      f"{len(exact)} of 4 (band-crossing form), worst {e:.1e}")

    def the_layout_follows_storage():
        on, text = arm({})
        by = re.findall(r"proj: matched a projection pack \((\S+), \d+ dots, [\d.]+ MB, layout (\w+) by storage\)", text)
        lay = dict(by)
        if lay.get("m1k2560n384+384") != "nk" or lay.get("m1k2880n512+512") != "kn":
            return False, f"layouts by storage: {lay}"
        built = {n: l for n, l in re.findall(r"proj: packed (\S+) \([\d.]+ MB once, \d+ weights, layout (\w+)\)", text)}
        if built.get("m1k2560n384+384") != "nk":
            return False, f"F10 built as {built.get('m1k2560n384+384')}"
        off, _u = arm({"METALJAX_PROJ_PACK": "0"})
        k = "F10 NK-stored f32 pair"
        if not same(on[k], off[k]):
            return False, f"F10 differs from its literal dots by {worst(on[k], off[k]):.2e} under the storage layout"
        # The forced layout is the A/B arm: F10 then runs the OTHER kernel
        # (gemv_t on an [N, K] weight), not bit-identical, within a few ULP.
        forced, ftext = arm({"METALJAX_PROJ_PACK_LAYOUT": "kn"})
        fl = re.findall(r"proj: matched a projection pack \((\S+), \d+ dots, [\d.]+ MB, layout (\w+) forced\)", ftext)
        if dict(fl).get("m1k2560n384+384") != "kn":
            return False, f"forced layouts: {fl}"
        e = worst(forced[k], off[k])
        if e > 1e-5:
            return False, f"F10 under the forced kn layout differs by {e:.2e}"
        held = same(forced[k], off[k])
        return True, (f"F10 ([N,K] f32) packs nk by storage, bit-identical; F1 "
                      f"packs kn; forced kn on F10 within {e:.1e} (bits "
                      f"{'held' if held else 'not held'}, the A/B arm)")

    def a_tuple_barrier_is_looked_through():
        on, text = arm({})
        if "m1k640n192+192" not in {n for n, _mb, _w in packed(text)}:
            return False, f"F11 not packed: {[ln for ln in proj_lines(text) if 'barrier' in ln or '640' in ln][:4]}"
        if "computed by stablehlo.optimization_barrier" in text:
            return False, "a barrier result was declined as computed"
        if exe_packs(text).get("jit_proj_f11") != 1:
            return False, "F11's executable does not count its pack"
        off, _u = arm({"METALJAX_PROJ_PACK": "0"})
        k = "F11 tuple barrier pair"
        if not same(on[k], off[k]):
            return False, f"F11 differs from the literal dots by {worst(on[k], off[k]):.2e}"
        return True, "F11 packed through the tuple barrier (w1 returned as is: a pass-through, not an update), bit-identical"

    def an_updated_weight_declines():
        on, text = arm({})
        decl = [ln for ln in proj_lines(text) if "the weight is updated by the program" in ln]
        if len(decl) < 2:
            return False, f"updated-weight declines: {decl}; lines: {[ln for ln in proj_lines(text) if '512' in ln][:4]}"
        if "m1k512n320+320" in {n for n, _d, _mb in matched(text)}:
            return False, "the updated pair matched"
        if exe_packs(text).get("jit_proj_f12", 0) != 0:
            return False, "the updated executable counts packs"
        if "jit_proj_f12: the projection packs' weights changed" in text:
            return False, "F12 repacked"
        off, _u = arm({"METALJAX_PROJ_PACK": "0"})
        k = "F12 updated pair"
        if not same(on[k], off[k]):
            return False, "F12 differs from the literal dots"
        return True, f"declined ({len(decl)} narrations: a result of the weight's type derives from it), F1/F2/F11 unaffected, answer literal"

    def the_pack_is_read_inside_the_loop():
        _a, text = arm({})
        if "has no pack in scope" in text:
            return False, "the emit fell back: the pack was not in scope"
        if exe_packs(text).get("jit_proj_f1") != 1:
            return False, "F1's executable does not narrate the fused tape"
        m = re.search(r"\[metaljax-native\] jit_proj_f1: flushes=.*?unrolls=(\d+)", text)
        if m is None or int(m.group(1)) != 0:
            return False, f"F1's loop unrolled ({m and m.group(1)}): not a region"
        return True, "F1's while body reads the pack as a region capture"

    return [("proj pack fires", the_pack_fires),
            ("proj pack kill switches", the_kill_switches_decline),
            ("proj pack removes dispatches", the_pack_removes_dispatches),
            ("proj pack cannot move an answer", the_pack_cannot_move_an_answer),
            ("proj pack is decode-only", prefill_declines),
            ("proj pack consumers vs CPU", consumers_match_cpu),
            ("proj pack policy kv/all", the_policy),
            ("proj pack leaves qmm dots", quantized_dots_stay_qmm),
            ("proj pack byte cap declines", the_byte_cap_declines),
            ("proj pack donation declines", a_donated_weight_declines),
            ("proj pack repacks fresh weights", fresh_weights_repack),
            ("proj pack in the loop body", the_pack_is_read_inside_the_loop),
            ("proj pack layout follows storage", the_layout_follows_storage),
            ("proj pack tuple barrier", a_tuple_barrier_is_looked_through),
            ("proj pack updated weight declines", an_updated_weight_declines)]


def _p44_ragged_decode(subprocess, pathlib, re):
    """Row 10 rewrite 1: the ragged DECODE form (metal_ragged.cc).

    At one-token decode the base emit spends ~8 non-gemv kernels per expert
    dot recovering the group index from `ends` and padding the result to
    the tiling its only readers slice straight back.  The decode form
    proves the ids' range (a top-k over ONE row) and reads the group index
    off the row permutation the graph already computes (`take(ids, perm)`,
    one kRaggedIdx shared by the layer's dots), reads the one replicated
    activation instead of the gathered rows, and skips the pad when every
    reader is `slice[0:m]`.

    Pinned, on the forms of `_ragged_decode_forms`: the match FIRES
    (narration per dot and the executable's `(N decode)` count) on the
    decode forms; prefill, unproven ids and a descending sort DECLINE with
    the stated reasons; the kill switch `METALJAX_RAGGED_DECODE=0` restores
    the base form; fewer dispatches; every answer is BIT-identical with the
    knob-off arm (the same gather_mm kernel over the same rows and
    matrices -- no tolerance anywhere); the unchanged consumers (combine,
    shared experts) agree across arms; and the first layer -- identical
    inputs on both backends -- matches jax-CPU's dense chain within the dot
    tolerances (the bf16 residual chain is pinned by bit-identity, not by
    a CPU tolerance it amplifies past).
    """
    here = str(pathlib.Path(__file__).resolve())
    memo = {}

    def arm(env_extra, platform="metal"):
        import json
        key = (platform, tuple(sorted(env_extra.items())))
        if key in memo:
            return memo[key]
        child = dict(os.environ)
        if platform == "cpu":
            child.pop("METALJAX_PLUGIN_PATH", None)
            child["JAX_PLATFORMS"] = "cpu"
        child["METALJAX_DEBUG"] = "1"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--ragged-decode"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if line.startswith("RGD ") or line.startswith("RGS "):
                label, payload = line[4:].split("\t", 1)
                answers[label] = [np.array(a) for a in json.loads(payload)]
        text = proc.stdout + proc.stderr
        memo[key] = (answers, text)
        return memo[key]

    def per_program(text):
        """The ragged narration lines that precede each program's fused-tape
        line, keyed by jit name (the forms run one after another)."""
        out, pending = {}, []
        for ln in text.splitlines():
            if "[metaljax-native] ragged:" in ln:
                pending.append(ln)
            m = re.search(r"\[metaljax-native\] (jit_rgd_\w+): .*?(\d+) ragged dispatch\(es\) \((\d+) decode\)", ln)
            if m:
                d = out.setdefault(m.group(1), {"ragged": 0, "decode": 0, "lines": []})
                d["ragged"] = int(m.group(2))
                d["decode"] = int(m.group(3))
                d["lines"].extend(pending)
                pending = []
        return out

    def dispatches(text):
        out = {}
        for m in re.finditer(r"\[metaljax-native\] (jit_rgd_\w+): flushes=.*?dispatches=(\d+)", text):
            out.setdefault(m.group(1), int(m.group(2)))
        return out

    def same(a, b):
        return len(a) == len(b) and all(
            x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))

    def worst(a, b):
        e = 0.0
        for x, y in zip(a, b):
            scale = max(float(np.max(np.abs(y))), 1e-30)
            e = max(e, float(np.max(np.abs(x - y))) / scale)
        return e

    FIRING = {"jit_rgd_d1": 3, "jit_rgd_d1f": 3, "jit_rgd_d3": 3,
              "jit_rgd_d5": 3, "jit_rgd_d7": 3, "jit_rgd_d8": 3}
    DECLINING = {"jit_rgd_d2": "rows are not one replicated activation (4 tokens)",
                 "jit_rgd_d4": "ids of unproven range",
                 "jit_rgd_d6": "the sort is not ascending",
                 "jit_rgd_d9": "the sort's payload is not the iota"}
    JAX_FORMS = 10  # D1 .. D9 with D1f; SHARED is the raw module on top.

    def the_form_fires():
        _a, text = arm({})
        progs = per_program(text)
        bad = []
        for prog, want in FIRING.items():
            d = progs.get(prog)
            if d is None:
                bad.append(f"{prog}: no fused-tape line")
                continue
            if d["ragged"] != 3 or d["decode"] != want:
                bad.append(f"{prog}: {d['ragged']} ragged, {d['decode']} decode (want 3/{want})")
            matched = [ln for ln in d["lines"] if "matched a ragged dispatch" in ln]
            x0 = [ln for ln in matched if " decode nopad" in ln]
            rows = [ln for ln in matched if " decode-rows" in ln]
            if prog == "jit_rgd_d5":
                # The gate root has a second reader: decode WITHOUT nopad,
                # and the narration names the reader that kept the pad.
                kept = [ln for ln in matched if " decode," in ln]
                if len(kept) != 1 or len(x0) != 1 or len(rows) != 1:
                    bad.append(f"{prog}: expected one padded decode, one nopad, one rows; got {matched}")
                # (One line: the up and down roots' readers are slices.)
                pad = [ln for ln in d["lines"] if "pad kept (" in ln]
                if pad != ["[metaljax-native] ragged: pad kept (stablehlo.convert reads the root)"]:
                    bad.append(f"{prog}: expected one 'pad kept (stablehlo.convert reads the root)', got {pad}")
            elif prog == "jit_rgd_d8":
                # The [4, 48] reshape: the x0 substitution declines on the
                # reshape width for the gate and up dots (the down dot's
                # rows are the swiglu output, no gather at all) and every
                # dot runs over the REAL rows.
                why = sorted(ln.split("stay real (", 1)[1] for ln in d["lines"]
                             if "decode rows stay real" in ln)
                want = sorted(["rows are not one replicated activation (the reshape width))"] * 2
                              + ["rows are not one replicated activation (no row gather))"])
                if (len(rows) != 3 or x0 or why != want
                        or not all(" nopad" in ln for ln in rows)):
                    bad.append(f"{prog}: expected 3 decode-rows nopad, 2 on the reshape width + 1 no gather; got {matched} / {why}")
            elif len(x0) != 2 or len(rows) != 1 or " nopad" not in rows[0]:
                bad.append(f"{prog}: expected 2 x0-decode + 1 rows-decode, all nopad; got {matched}")
        if bad:
            return False, "; ".join(bad)[:600]
        names = sorted({re.search(r"\((\S+ decode[^,]*)", ln).group(1)
                        for p in FIRING for ln in progs[p]["lines"]
                        if "matched a ragged dispatch" in ln and " decode" in ln})
        return True, f"{len(FIRING)} programs, 3 decode dispatches each: {names}"

    def the_declines():
        _a, text = arm({})
        progs = per_program(text)
        bad = []
        for prog, why in DECLINING.items():
            d = progs.get(prog)
            if d is None:
                bad.append(f"{prog}: no fused-tape line")
                continue
            if d["ragged"] != 3 or d["decode"] != 0:
                bad.append(f"{prog}: {d['ragged']} ragged, {d['decode']} decode (want 3/0)")
            reasons = [ln for ln in d["lines"] if "decode form declined" in ln]
            if not reasons or not all(why in ln for ln in reasons):
                bad.append(f"{prog}: reasons {reasons[:3]} (want '{why}')")
            if any(" decode" in ln for ln in d["lines"] if "matched" in ln):
                bad.append(f"{prog}: a decode match narrated")
        if bad:
            return False, "; ".join(bad)[:600]
        return True, "; ".join(f"{p[8:]}: {w}" for p, w in DECLINING.items())

    def the_kill_switch():
        _a, text = arm({"METALJAX_RAGGED_DECODE": "0"})
        progs = per_program(text)
        if any("decode form declined" in ln or (" decode" in ln and "matched" in ln)
               for d in progs.values() for ln in d["lines"]):
            return False, "RAGGED_DECODE=0 still narrates a decode form"
        counts = {p: (d["ragged"], d["decode"]) for p, d in progs.items()}
        want = {p: (3, 0) for p in counts if p != "jit_rgd_shared"}
        want["jit_rgd_shared"] = (2, 0)
        if counts != want or len(counts) != JAX_FORMS + 1:
            return False, f"counts {counts} (want (3, 0) x{JAX_FORMS} + shared (2, 0))"
        return True, (f"RAGGED_DECODE=0: {JAX_FORMS} programs, 3 base ragged "
                      "dispatches each + the shared module's 2, 0 decode")

    def fewer_dispatches():
        _a, on = arm({})
        _b, off = arm({"METALJAX_RAGGED_DECODE": "0"})
        d_on, d_off = dispatches(on), dispatches(off)
        progs = sorted(FIRING)
        if any(p not in d_on or p not in d_off for p in progs):
            return False, f"dispatch narration on={d_on} off={d_off}"
        worse = [p for p in progs if d_on[p] >= d_off[p]]
        if worse:
            return False, f"no fewer dispatches on {worse}: on={d_on} off={d_off}"
        return True, ", ".join(f"{p[8:]} {d_off[p]}->{d_on[p]}" for p in progs)

    def bit_identical():
        on, _t = arm({})
        off, _u = arm({"METALJAX_RAGGED_DECODE": "0"})
        if set(on) != set(off) or len(on) != JAX_FORMS + 1:
            return False, f"labels differ: on={sorted(on)} off={sorted(off)}"
        bad = [f"{k} differs by {worst(on[k], off[k]):.2e}" for k in sorted(on)
               if not same(on[k], off[k])]
        if bad:
            return False, "; ".join(bad)
        return True, (f"{JAX_FORMS} forms x 3 arrays + the shared module's 2 "
                      "bit-identical (final, expert rows, shared)")

    def consumers_unchanged():
        on, _t = arm({})
        off, _u = arm({"METALJAX_RAGGED_DECODE": "0"})
        bad = [k for k in sorted(on) if k in off and k != "SHARED" and not (
            np.array_equal(on[k][0], off[k][0]) and np.array_equal(on[k][2], off[k][2]))]
        if bad:
            return False, f"final/shared differ on {bad}"
        return True, "final activation and shared-expert output identical across arms on every form"

    def matches_cpu():
        # The suite's own rule (`_compare`): |got - want| <= rtol |want| +
        # atol, at BF16DOT for the bf16 forms and DOT for the f32 one --
        # on the FIRST layer's expert rows and shared output, which both
        # backends compute from identical inputs (the emit against the CPU
        # dense chain), and on the whole 3-layer chain of the f32 form.
        # The bf16 chain is NOT gated end to end: the residual stream
        # amplifies the router's bf16 rounding differences layer by layer
        # (D4, the BASE form with input ids, reaches 2.4x the tolerance at
        # layer 3); the whole chain is pinned by `bit_identical` instead,
        # and reported here.
        on, _t = arm({})
        cpu, _c = arm({}, platform="cpu")
        if set(on) != set(cpu):
            return False, f"labels differ: {sorted(set(on) ^ set(cpu))}"

        def frac(a, c, rtol, atol):
            return float(np.max(np.abs(a - c) / (rtol * np.abs(c) + atol)))

        first, whole, bad = {}, {}, []
        for k in sorted(on):
            rtol, atol = (1e-5, 1e-5) if k == "D1F" else (2e-2, 2e-2)
            if k == "SHARED":
                # One layer, two dots straight off identical inputs: the
                # whole answer is gated.
                f = max(frac(a, c, rtol, atol) for a, c in zip(on[k], cpu[k]))
                first[k] = whole[k] = f
                if f > 1.0:
                    bad.append(f"{k} at {f:.2f}x")
                continue
            layers = 1 if k == "D3" else 3
            f = 0.0
            for a, c in zip(on[k][1:], cpu[k][1:]):
                per = a.size // layers
                f = max(f, frac(a[:per], c[:per], rtol, atol))
            first[k] = f
            whole[k] = max(frac(a, c, rtol, atol) for a, c in zip(on[k], cpu[k]))
            if f > 1.0 or (k == "D1F" and whole[k] > 1.0):
                bad.append(f"{k} at {f:.2f}x (first layer) / {whole[k]:.2f}x (chain)")
        if bad:
            return False, "; ".join(bad)
        return True, ("tolerance used vs CPU, first layer: "
                      + ", ".join(f"{k} {e:.2f}" for k, e in first.items())
                      + "; whole bf16 chain (reported): "
                      + ", ".join(f"{k} {e:.2f}" for k, e in whole.items()))

    def shared_ends():
        # Two roots over ONE `ends`, one of them base: the decode root's
        # absorption must not take the cumsum the base root's emit reads.
        # Before the fix the executable narrated `no fused tape (a value
        # defined outside the entry block)` and lost every fusion.
        on, text = arm({})
        d = per_program(text).get("jit_rgd_shared")
        if d is None:
            return False, "no fused-tape line for jit_rgd_shared"
        if (d["ragged"], d["decode"]) != (2, 1):
            return False, f"{d['ragged']} ragged, {d['decode']} decode (want 2/1)"
        if "jit_rgd_shared: no fused tape" in text:
            return False, "the executable fell back to the plain tape"
        why = [ln for ln in d["lines"] if "decode form declined" in ln]
        if len(why) != 1 or "does not count every row once" not in why[0]:
            return False, f"decline narration {why}"
        if "SHARED" not in on or len(on["SHARED"]) != 2:
            return False, "no SHARED answer"
        return True, ("one ends, 2 ragged dispatches (1 decode), fused tape kept; "
                      "the base root declined on its row count")

    return [("ragged decode fires", the_form_fires),
            ("ragged decode declines", the_declines),
            ("ragged decode kill switch", the_kill_switch),
            ("ragged decode removes dispatches", fewer_dispatches),
            ("ragged decode bit-identical", bit_identical),
            ("ragged decode consumers unchanged", consumers_unchanged),
            ("ragged decode vs CPU", matches_cpu),
            ("ragged decode shared ends", shared_ends)]


def _p36_tape_gate(subprocess, tempfile, pathlib, re):
    """The trace budget is asked of the post-pass TAPE, not only the MLIR.

    `BlockCost` charges every op of every region body -- a reduce's two-op
    combiner, a scatter's apply body, every callee behind a call, the
    terminators -- while the tape that runs carries a reduce as ONE entry and
    has been through CSE and DCE.  The LoRA E2B train step read 27,308 on the
    IR against 16,094 entries, and at the 20,000 default it ran op by op
    through 22 blocking flushes (gap-rows/row18, -79 ms/step under
    METALJAX_TRACE_BUDGET=30000).  Since T2 the main gate, the while-body
    gate and `WhileTraceable` compare the budget with the SMALLER of the two
    counts (`GateCost`), so a program the MLIR count admits stays admitted
    and one it over-charged compiles when its tape fits.

    Nothing here hard-codes a count.  The probe narrates both (`cost=<mlir>
    tape=<entries>`), the budget the contract tests with is derived from
    them, and the answer is INTEGER so the compiled and the eager arms must
    agree bit for bit -- the layout of the decision cannot move a number.
    """
    _MAIN = re.compile(r"main: pure=\d+ cost=(\d+) tape=(\d+) "
                       r"bytes=[\d.]+MB compile=(\d)")

    def run(extra_env):
        env = dict(os.environ)
        env["METALJAX_DEBUG"] = "1"
        env.update(extra_env)
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(_P36_TAPE_GATE)
            script = fh.name
        try:
            proc = subprocess.run([sys.executable, script], env=env,
                                  capture_output=True, text=True)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass
        err = proc.stderr or ""
        # The probe's own program is the BIG main; the harness's tiny
        # transfers narrate too, so take the largest cost.
        gates = sorted((int(m.group(1)), int(m.group(2)), int(m.group(3)))
                       for m in _MAIN.finditer(err))
        answer = next((ln[len("[probe] answer "):]
                       for ln in proc.stdout.splitlines()
                       if ln.startswith("[probe] answer ")), None)
        return proc, (gates[-1] if gates else None), answer

    state = {}

    def both_counts_are_narrated():
        proc, gate, answer = run({})
        if proc.returncode != 0:
            return False, (proc.stderr or "").strip()[-140:]
        if gate is None:
            return False, "no `main: pure= cost= tape=` line narrated"
        cost, tape, compiled = gate
        if not (0 < tape < cost):
            return False, (f"cost={cost} tape={tape}: the reduce chain "
                           "should cost MORE on the IR than on the tape")
        if not compiled:
            return False, f"cost={cost} tape={tape} yet compile=0 at default"
        state.update(cost=cost, tape=tape, answer=answer)
        return True, f"cost={cost} (MLIR) tape={tape}, compiled at default"

    def an_overcharged_main_compiles_on_its_tape():
        if "cost" not in state:
            return False, "the run above did not complete"
        cost, tape = state["cost"], state["tape"]
        # Strictly between the two counts: the tape fits, the IR does not.
        budget = (cost + tape) // 2
        proc, gate, answer = run({"METALJAX_TRACE_BUDGET": str(budget)})
        if proc.returncode != 0:
            return False, (proc.stderr or "").strip()[-140:]
        if gate is None or gate[2] != 1:
            return False, (f"budget {budget} between tape {tape} and cost "
                           f"{cost}: compile={gate[2] if gate else '?'} "
                           "(want 1)")
        state["between"] = answer
        return True, f"budget {budget}: cost={cost} > budget >= tape={tape}, compile=1"

    def a_main_over_both_counts_still_runs_eager():
        if "cost" not in state:
            return False, "the run above did not complete"
        tape = state["tape"]
        budget = max(1, tape // 2)
        proc, gate, answer = run({"METALJAX_TRACE_BUDGET": str(budget)})
        if proc.returncode != 0:
            return False, (proc.stderr or "").strip()[-140:]
        if gate is None or gate[2] != 0:
            return False, (f"budget {budget} under tape {tape}: "
                           f"compile={gate[2] if gate else '?'} (want 0)")
        state["under"] = answer
        return True, f"budget {budget} < tape {tape}: compile=0"

    def the_gate_cannot_move_an_answer():
        want = state.get("answer")
        if want is None:
            return False, "no answer from the default run"
        arms = {"between": state.get("between"),
                "under (eager)": state.get("under")}
        proc, _gate, off = run({"METALJAX_COMPILE": "0"})
        if proc.returncode != 0:
            return False, (proc.stderr or "").strip()[-140:]
        arms["METALJAX_COMPILE=0"] = off
        bad = [k for k, v in arms.items() if v != want]
        if bad:
            return False, "differs from the compiled answer: " + ", ".join(bad)
        return True, "4 layouts agree bit for bit (int32 chain)"

    return [("both gate counts are narrated", both_counts_are_narrated),
            ("an over-charged main compiles on its tape",
             an_overcharged_main_compiles_on_its_tape),
            ("a main over both counts stays eager",
             a_main_over_both_counts_still_runs_eager),
            ("the tape gate cannot move an answer",
             the_gate_cannot_move_an_answer)]


def _p38_chunk_plan(subprocess, pathlib, re):
    """The chunked replay's schedule and its byte bound (T4, findings2).

    A counted compiled loop replays K iterations per compiled graph
    (`run_chunked`): trip/K submitted K-chunks, then the trip%K single-step
    replays left LAZY for the final flush.  The schedule is narrated once
    per (trip, K) under METALJAX_DEBUG and that line is what this pins --
    the plan, the submissions, the blocking flushes, the in-flight window
    (METALJAX_CHUNK_INFLIGHT, 4: how many submitted chunks the host may run
    ahead of the device), against `run_chunked`'s own arithmetic from the
    `cost` the lowering narrates.

    K itself comes from the lowering's gate: METALJAX_CHUNK_MAX (16), the
    cost and byte budgets, and since T4 a BYTE BOUND per chunk
    (METALJAX_CHUNK_BYTES_MB, 2048) on the body's bytes net of its write-only
    accumulators -- the stacked outputs a scan writes one slab per iteration
    by dynamic_update_slice, charged their whole stack by BlockBytes --
    withheld when those accumulators exceed METALJAX_CHUNK_ACC_MB (64),
    because every chunk boundary copies them (the submission pins the
    carries).  The gate narrates `real=` and `acc=` so both halves are
    checkable here: the stacked-output row shows accumulator bytes and a
    zero net body, the plain rows show none.  Only the sync points move
    between layouts, so the carries are BIT-identical across every K.
    """
    here = str(pathlib.Path(__file__).resolve())
    REM = "chunked replay with a remainder (90 x matmul body)"
    FULL = "chunked replay (512 x matmul body)"
    STK = "chunked replay with a stacked output (90 x matmul body)"
    PLAN = re.compile(r"chunked loop: trip=(\d+) K=(\d+) plan=(\d+)x(\d+)\+(\d+)x1 "
                      r"submits=(\d+) flushes=(\d+) inflight=(\d+)")
    GATE = re.compile(r"while gate: cost=(\d+)(?: tape=\d+)? bytes=(\d+)MB .*?kmax=(\d+) "
                      r"period=\d+ real=(\d+)MB acc=(\d+)MB( \(accumulators "
                      r"veto the chunk byte bound\))?")

    def arm(env_extra):
        import json
        child = dict(os.environ)
        child["METALJAX_DEBUG"] = "1"
        # The 512-step row is a coop cell msl_scan replaces with a generated
        # kernel; this contract is about the CHUNKED replay, so the kernel
        # path is off in every arm (the other rows decline it anyway).
        child["METALJAX_MSL"] = "0"
        child.update(env_extra)
        proc = subprocess.run([sys.executable, here, "--chunk-plan"],
                              env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout
                                ).splitlines()[-1][:110])
        answers = {}
        for line in proc.stdout.splitlines():
            if line.startswith("CHUNK "):
                label, payload = line[6:].split("\t", 1)
                answers[label] = json.loads(payload)
        # The runtime narrates its schedule on stdout, the lowering its gate
        # on stderr.  Plans are keyed by (trip, K); the stacked row and the
        # remainder row share trip=90, so both are kept as a list.
        plans = {}
        inflight = set()
        for t, k, n, kk, r, sub, f, w in PLAN.findall(proc.stdout):
            plans.setdefault(int(t), []).append(
                (int(k), int(n), int(kk), int(r), int(sub), int(f)))
            inflight.add(int(w))
        arm.inflight = inflight
        gates = [(int(c), int(b), int(k), int(real), int(acc), bool(veto))
                 for c, b, k, real, acc, veto in GATE.findall(proc.stderr)]
        return answers, plans, gates

    def expected(trip, K, cost):
        """`run_chunked`'s arithmetic: the chunks first, submitted, a blocking
        flush every `sync_every` of them, the singles lazy, one final flush."""
        sync_every = max(1, 75000 // max(K * cost, 1))
        nchunks, rem = divmod(trip, K)
        blocking = nchunks // sync_every
        return (K, nchunks, K, rem, nchunks - blocking, blocking + 1)

    def gate_for(gates, pred):
        for g in gates:
            if pred(g):
                return g
        return None

    def the_chunks_lead_and_the_singles_trail():
        answers, plans, gates = arm({})
        for case in (REM, FULL, STK):
            if case not in answers:
                return False, f"ran {sorted(answers)}"
        if not gates:
            return False, "no `while gate:` narration"
        if 90 not in plans or 512 not in plans:
            return False, f"schedules narrated for trips {sorted(plans)}"
        # Every gate of these rows is cost ~10-20; take each plan's own
        # cost from the gate that produced the same K.
        for trip in (90, 512):
            for plan in plans[trip]:
                K = plan[0]
                g = gate_for(gates, lambda g: g[2] == K)
                cost = g[0] if g else gates[0][0]
                want = expected(trip, K, cost)
                if plan != want:
                    return False, (f"trip={trip}: narrated K={plan[0]} plan="
                                   f"{plan[1]}x{plan[2]}+{plan[3]}x1 submits="
                                   f"{plan[4]} flushes={plan[5]}, wanted "
                                   f"K={want[0]} plan={want[1]}x{want[2]}+"
                                   f"{want[3]}x1 submits={want[4]} "
                                   f"flushes={want[5]} (cost {cost})")
        if any(p[0] != 16 for t in plans for p in plans[t]):
            return False, f"default K is not 16: {plans}"
        if arm.inflight != {4}:
            return False, f"default in-flight window narrated as {arm.inflight}"
        _answers, _plans, _gates = arm({"METALJAX_CHUNK_INFLIGHT": "1"})
        if arm.inflight != {1}:
            return False, f"CHUNK_INFLIGHT=1 narrated as {arm.inflight}"
        k, n, _kk, rem, sub, f = plans[90][0]
        return True, (f"trip=90: {n} chunks of {k} submitted, {rem} singles "
                      f"lazy, {sub} submits + {f} flush, 4 in flight; "
                      f"trip=512: {plans[512][0][4]} submits + "
                      f"{plans[512][0][5]} flush")

    def the_gate_sees_the_accumulator():
        _answers, _plans, gates = arm({})
        stacked = gate_for(gates, lambda g: g[4] >= 5)
        if stacked is None:
            return False, f"no gate with acc>=5MB: {gates}"
        cost, bytes_mb, kmax, real, acc, veto = stacked
        if bytes_mb < 5 or real > 1 or veto:
            return False, (f"stacked row: bytes={bytes_mb}MB real={real}MB "
                           f"acc={acc}MB veto={veto}")
        plain = [g for g in gates if g[4] == 0]
        if not plain or any(g[3] != g[1] for g in plain):
            return False, f"plain rows: {plain}"
        return True, (f"stacked row bytes={bytes_mb}MB real={real}MB "
                      f"acc={acc}MB (kmax={kmax}); {len(plain)} plain rows "
                      f"acc=0 real=bytes")

    def the_knob_still_sets_k():
        bad = []
        for kmax, want in ((2, (2, 45, 2, 0)), (4, (4, 22, 4, 2)),
                           (30, (30, 3, 30, 0))):
            _answers, plans, gates = arm({"METALJAX_CHUNK_MAX": str(kmax)})
            if 90 not in plans:
                bad.append(f"CHUNK_MAX={kmax}: no schedule for trip=90")
                continue
            for got in plans[90]:
                if got[:4] != want:
                    bad.append(f"CHUNK_MAX={kmax}: K={got[0]} plan={got[1]}x"
                               f"{got[2]}+{got[3]}x1, wanted K={want[0]} "
                               f"plan={want[1]}x{want[2]}+{want[3]}x1")
                    break
                g = gate_for(gates, lambda g: g[2] == kmax)
                cost = g[0] if g else 0
                if got != expected(90, kmax, cost):
                    bad.append(f"CHUNK_MAX={kmax}: submits={got[4]} "
                               f"flushes={got[5]}, wanted "
                               f"{expected(90, kmax, cost)[4:]}")
                    break
        if bad:
            return False, "; ".join(bad)
        return True, "K=2/4/30 narrate 45x2+0x1 / 22x4+2x1 / 3x30+0x1"

    def the_byte_bound_caps_k():
        """METALJAX_CHUNK_BYTES_MB bounds a chunk's traffic net of the
        accumulators: these bodies are kilobytes (0 MB after the shift), so
        the budget in MB is the iteration count itself -- 7 MB is K=7 for
        all three rows, the stacked one included (its 5 MB of accumulator
        is under the 64 MB veto and excluded from the net); 1 MB is the
        floor K=2."""
        bad = []
        for mb, want in (("7", (7, 12, 7, 6)), ("1", (2, 45, 2, 0))):
            _answers, plans, gates = arm({"METALJAX_CHUNK_BYTES_MB": mb})
            got = plans.get(90, [])
            if len(got) != 2:
                bad.append(f"CHUNK_BYTES_MB={mb}: {len(got)} trip=90 plans")
                continue
            for p in got:
                if p[:4] != want:
                    bad.append(f"CHUNK_BYTES_MB={mb}: K={p[0]} plan={p[1]}x"
                               f"{p[2]}+{p[3]}x1, wanted K={want[0]} plan="
                               f"{want[1]}x{want[2]}+{want[3]}x1")
                    break
        if bad:
            return False, "; ".join(bad)
        return True, "7 MB -> K=7 (12x7+6x1) on both trip-90 rows; 1 MB -> K=2"

    def big_accumulators_veto_the_byte_bound():
        """With the veto cap under the stacked row's 5 MB, its K stays at
        CHUNK_MAX while the plain remainder row still shrinks to 7."""
        _answers, plans, gates = arm({"METALJAX_CHUNK_BYTES_MB": "7",
                                      "METALJAX_CHUNK_ACC_MB": "4"})
        got = sorted(plans.get(90, []))
        if [p[0] for p in got] != [7, 16]:
            return False, f"trip=90 plans: {got}"
        vetoed = gate_for(gates, lambda g: g[5])
        if vetoed is None or vetoed[2] != 16 or vetoed[4] < 5:
            return False, f"veto gate: {vetoed}"
        return True, (f"stacked row acc={vetoed[4]}MB vetoed -> K=16 "
                      f"(5x16+10x1); plain row K=7")

    def a_single_step_loop_narrates_no_chunks():
        """K=1 is not a chunked replay at all (run_while's BodyRunner arm),
        and a schedule narrated for it would be a schedule nothing ran."""
        _answers, plans, _gates = arm({"METALJAX_CHUNK_MAX": "1"})
        if plans:
            return False, f"CHUNK_MAX=1 narrated a chunk schedule: {plans}"
        return True, "CHUNK_MAX=1 replays single steps, no schedule"

    def the_schedule_cannot_move_an_answer():
        """Same iterations in the same order; only the sync points and the
        graph boundaries move -- so every K, the single-step replay included,
        must hand back the same bits."""
        base, _, _ = arm({})
        bad = []
        for label, extra in (("CHUNK_MAX=1", {"METALJAX_CHUNK_MAX": "1"}),
                             ("CHUNK_MAX=2", {"METALJAX_CHUNK_MAX": "2"}),
                             ("CHUNK_MAX=4", {"METALJAX_CHUNK_MAX": "4"}),
                             ("CHUNK_MAX=30", {"METALJAX_CHUNK_MAX": "30"}),
                             ("CHUNK_BYTES_MB=7",
                              {"METALJAX_CHUNK_BYTES_MB": "7"}),
                             ("CHUNK_BYTES_MB=7,ACC_MB=4",
                              {"METALJAX_CHUNK_BYTES_MB": "7",
                               "METALJAX_CHUNK_ACC_MB": "4"}),
                             ("CHUNK_INFLIGHT=1",
                              {"METALJAX_CHUNK_INFLIGHT": "1"}),
                             ("CHUNK_INFLIGHT=2,CHUNK_MAX=2",
                              {"METALJAX_CHUNK_INFLIGHT": "2",
                               "METALJAX_CHUNK_MAX": "2"})):
            other, _, _ = arm(extra)
            for case in (REM, FULL, STK):
                if base.get(case) != other.get(case):
                    bad.append(f"{case} differs at {label}")
        if bad:
            return False, "; ".join(bad)
        return True, ("K=16 (default), 1, 2, 4, 30, 7, 7+veto, and in-flight "
                      "windows 1 and 2 agree bit for bit")

    return [("the chunks lead, the singles trail lazily",
             the_chunks_lead_and_the_singles_trail),
            ("the gate sees the accumulator", the_gate_sees_the_accumulator),
            ("chunk-max knob still sets K", the_knob_still_sets_k),
            ("chunk byte bound caps K", the_byte_bound_caps_k),
            ("big accumulators veto the byte bound",
             big_accumulators_veto_the_byte_bound),
            ("single-step loop narrates no chunks",
             a_single_step_loop_narrates_no_chunks),
            ("chunk schedule cannot move an answer",
             the_schedule_cannot_move_an_answer)]


def _arm_section(title, env_extra, tag, ref_path, compiled_arm, failures):
    """Re-run every case through the SAME dylib under `env_extra` and compare.

    The child writes its answers with `--eager-arm` (the entry point is the
    plugin arm generally: what changes between the two callers is the
    environment the dylib reads at load).  Bit-identity is reported, not
    demanded: a compile decision or a generated kernel changes which MLX
    kernels run, and the bar is each case's own CPU tolerance.

    The child runs under a TIMEOUT, and that is a contract, not politeness.
    An engine that submits work MLX has not proved it can build hands a
    failure to `mx::async_eval`, which abandons the events it has attached
    and leaves the next blocking eval waiting on one of them forever -- the
    0.11.6 wedge, whose live shape is "msl kernel in a chunked replay" above
    and whose arm is METALJAX_MSL_FORCE_BUILD_FAIL.  A wedge must be a
    reported FAILURE here; without the timeout it would be a test run that
    never ends, which is how it stayed latent the first time.  The budget is
    minutes against a child that takes seconds: only a hang can reach it.
    """
    print(f"\n{title}")
    print("-" * 62)
    path = ref_path.with_name(ref_path.name.replace("reference",
                                                    "arm-" + tag))
    child = dict(os.environ)
    child.update(env_extra)
    try:
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--eager-arm", str(path)],
            env=child, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"{'the arm':<32} {'-':>12}  FAIL: the child WEDGED "
              f"(no exit in 900s -- Event::wait?)")
        failures.append(f"{tag} wedged")
        return
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
    if len(sys.argv) > 2 and sys.argv[1] == "--wedge-probe":
        # The wedge shape, computed once (see the contract that drives it).
        # The platform is whatever the caller put in the environment: this
        # same entry point produces the CPU answer and the forced-failure
        # metal answer, which is what makes comparing them meaningful.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        run_wedge_probe(sys.argv[2])
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
    if len(sys.argv) > 1 and sys.argv[1] == "--msl-bf16":
        # The bf16 rows of the census, alone: these must PLAN (the dtype
        # table maps bf16 since the topconfs16k cliff), and the parent reads
        # the plugin's narration to tell a planned loop from a silently
        # interpreted one.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        for name, fn, args, _rtol, _atol in _cases():
            if name.startswith("msl bf16"):
                jax.jit(fn)(*args)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--msl-carries":
        # The five carry-classification cases, alone, so the parent can read
        # one `counters=` census per loop out of the narration.  They must
        # PLAN: an EXACT comparison against the CPU is satisfied by a decline
        # too, and a decline is exactly what a careless fix would produce.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        for name, fn, args, _rtol, _atol in _cases():
            if name.startswith("msl ") and "carry" in name:
                jax.jit(fn)(*args)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--msl-lane-scalars":
        # The five one-scalar-per-lane cases, alone, so the parent can read
        # one `lane=` census per loop out of the narration (they must PLAN;
        # a decline satisfies the EXACT comparison too).
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        for name, fn, args, _rtol, _atol in _cases():
            if name.startswith("msl lane scalar"):
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
    if len(sys.argv) > 1 and sys.argv[1] == "--msl-f4-coop":
        # The F=4 coop flip (2026-08-26): the same square cell fwd + weight
        # grad the case list carries, alone, so the parent can read the modes
        # out of the narration and compare the checksum across knob settings.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        h0 = _rand((4, 4), 68)
        xs = _rand((16, 4, 4), 69)
        w = _rand((4, 4), 70) * np.float32(0.3)
        _, hs = jax.jit(_msl_rnn)(h0, xs, w)
        g = jax.jit(_msl_rnn_grad)(h0, xs, w)
        s = float(np.asarray(hs).sum()) + float(np.asarray(g).sum())
        print(f"F4 COOP CHECKSUM {s:.9e}")
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
    if len(sys.argv) > 1 and sys.argv[1] == "--kv-inplace":
        # The KV in-place rows alone: answers to stdout, the plugin's
        # narration (what the rewrite did, what donated) to stderr/stdout.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        import json as _json
        for name, fn, args, _rtol, _atol in _cases():
            if not name.startswith("kv cache in place"):
                continue
            out = _flatten(jax.jit(fn)(*args))
            print(f"KV {name}\t"
                  f"{_json.dumps([_canonical(v)[1].ravel().tolist() for v in out])}",
                  flush=True)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--rope-view":
        # The rope-view rows alone: answers to stdout, the plugin's
        # narration (what matched, the dispatch counts) to stderr.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        import json as _json
        for name, fn, args, _rtol, _atol in _cases():
            if not name.startswith("rope view"):
                continue
            out = _flatten(jax.jit(fn)(*args))
            print(f"ROPE {name}\t"
                  f"{_json.dumps([_canonical(v)[1].ravel().tolist() for v in out])}",
                  flush=True)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--dynamic-while":
        # The two dynamic-trip while rows alone, so the parent can read one
        # serial/pipelined census per arm out of the narration, and compare
        # the ANSWERS across loop layouts (the trip count is in the carry).
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        import json as _json
        for name, fn, args, _rtol, _atol in _cases():
            if not name.startswith("while_loop (dynamic trip"):
                continue
            out = _flatten(jax.jit(fn)(*args))
            print(f"WHILE {name}\t"
                  f"{_json.dumps([v.astype(np.float64).ravel().tolist() for v in out])}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--chunk-plan":
        # The two chunked-replay rows alone (one with a remainder, one
        # without), so the parent can read one `chunked loop:` schedule per
        # loop out of the narration and compare the ANSWERS across K.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import jax
        import json as _json
        for name, fn, args, _rtol, _atol in _cases():
            if not name.startswith("chunked replay"):
                continue
            out = _flatten(jax.jit(fn)(*args))
            print(f"CHUNK {name}\t"
                  f"{_json.dumps([v.astype(np.float64).ravel().tolist() for v in out])}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--norm-forms":
        # Every RMS-norm spelling the model table runs, through whichever
        # backend the caller put in the environment.  The answers go to
        # stdout so the parent can compare arms; the plugin's own narration
        # goes to stderr, which is where the parent reads what FIRED.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        for label, text, _tag, _dt, _n in _norm_forms():
            out = _run_module(text, _norm_inputs(text))[0]
            print(f"NORM {label}\t"
                  f"{_json.dumps(out.astype(np.float64).ravel().tolist())}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] in ("--gdn-forms", "--gdn-chain"):
        # The captured gated-delta-net decode steps, through whichever
        # backend the caller put in the environment.  Answers to stdout, the
        # plugin's narration to stderr -- the parent reads what FIRED there.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        chain = sys.argv[1] == "--gdn-chain"
        for label, text, _tag, _dt in _gdn_forms():
            if chain:
                v = _gdn_chain(text)
                print(f"GDN {label}\t"
                      f"{_json.dumps(v.tolist())}")
                continue
            out, _conv, rec = _run_module(text, _gdn_inputs(text))
            print(f"GDN {label}\t"
                  f"{_json.dumps(np.asarray(out).astype(np.float64).ravel().tolist())}")
            print(f"GDN {label} state\t"
                  f"{_json.dumps(np.asarray(rec).astype(np.float64).ravel().tolist())}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--mla-forms":
        # Both decode-attention head geometries, through whichever backend
        # the caller put in the environment.  Answers to stdout, the
        # plugin's narration to stderr -- the parent reads what FIRED there.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        for label, text, _tag, _dt in _mla_forms():
            out = _run_module(text, _mla_inputs(text))[0]
            print(f"MLA {label}\t"
                  f"{_json.dumps(out.astype(np.float64).ravel().tolist())}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--gqa-forms":
        # The keras GQA repeat forms, plus one jitted PREFILL at the row-20
        # head geometry (Tq = 4 over G = 16, which MLX's vector kernel would
        # refuse grouped: the absorb must keep that repeat).  Answers to
        # stdout, the plugin's narration to stderr -- the parent reads what
        # was ABSORBED there, and compares the answers bit for bit against
        # the METALJAX_SDPA_GQA=0 arm.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        for label, text, _absorbed, _dt, _why in _gqa_forms():
            out = _run_module(text, _gqa_inputs(text))[0]
            print(f"GQA {label}\t"
                  f"{_json.dumps(out.astype(np.float64).ravel().tolist())}")
        import jax
        for label, fn, args, _ab, _dt, _why in _gqa_jit_forms():
            out = np.asarray(jax.jit(fn)(*args))
            print(f"GQA {label}\t"
                  f"{_json.dumps(out.astype(np.float64).ravel().tolist())}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--mla-kernel-forms":
        # B3: the geometries the two-span kernel takes, through the plugin
        # (or the CPU when the caller says so): answers to stdout, the
        # kernel's build narration to stderr.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        for label, text, _tag, _dt in _mla_kernel_forms():
            out = _run_module(text, _mla_inputs(text))[0]
            print(f"MLK {label}\t"
                  f"{_json.dumps(out.astype(np.float64).ravel().tolist())}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--ragged-decode":
        # Row 10 rewrite 1: the ragged decode forms, through jax on the
        # plugin -- or on the CPU when the caller says so.  Answers to
        # stdout, the recognizer's and the executable's narration to stderr.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        import jax
        for label, fn, args in _ragged_decode_forms():
            out = jax.jit(fn)(*args)
            outs = [out[0], out[1][0], out[1][1]]
            print(f"RGD {label}\t"
                  f"{_json.dumps([np.asarray(o).astype(np.float64).ravel().tolist() for o in outs])}",
                  flush=True)
        text, args = _ragged_shared_ends_module()
        outs = _run_module(text, args)
        print(f"RGS SHARED\t"
              f"{_json.dumps([np.asarray(o).astype(np.float64).ravel().tolist() for o in outs])}",
              flush=True)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--proj-pack":
        # B6: the projection-pack forms, through jax on the plugin -- or on
        # the CPU when the caller says so.  Answers to stdout (one line per
        # call, fresh inputs per call), the recognizer's, the pack wave's
        # and the executable's narration to stderr.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        for label, fn, make_args, calls in _proj_pack_forms():
            for c in range(calls):
                outs = _flatten(fn(*make_args(1000 + 10 * c)))
                tag = label if calls == 1 else f"{label} call{c + 1}"
                print(f"PPK {tag}\t"
                      f"{_json.dumps([np.asarray(o).astype(np.float64).ravel().tolist() for o in outs])}",
                      flush=True)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--stacked-relayout-forms":
        # B3: the stacked dots whose contracted axes straddle the layer axis
        # (maxtext's attention out-projection), through jax on the plugin --
        # or on the CPU when the caller says so.  Answers to stdout, the
        # recognizer's and the pack wave's narration to stderr.
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
            os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        import jax
        for label, fn, args in _stacked_relayout_forms():
            outs = _flatten(jax.jit(fn)(*args))
            print(f"SRL {label}\t"
                  f"{_json.dumps(np.concatenate([np.asarray(o).astype(np.float64).ravel() for o in outs]).tolist())}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--start-plan-forms":
        # The START PLAN's modules, run through the plugin: answers to
        # stdout, the `ds plan:` tally to stderr.  The plan is a lowering
        # decision, so one execute of each module is all it takes.
        os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
        os.environ["JAX_PLATFORMS"] = "metal"
        import json as _json
        for label, text, ins in _start_plan_forms():
            out = _run_module(text, ins)[0]
            print(f"DSP {label}\t"
                  f"{_json.dumps(out.astype(np.float64).ravel().tolist())}")
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
                         + _p31_norm(subprocess, pathlib, __import__("re"))
                         + _p32_mla(subprocess, pathlib, __import__("re"))
                         + _p37_gqa(subprocess, pathlib, __import__("re"))
                         + _p33_gdn(subprocess, pathlib, __import__("re"))
                         + _p40_mla_kernel(subprocess, pathlib,
                                           __import__("re"))
                         + _p41_stacked_relayout(subprocess, pathlib,
                                                 __import__("re"))
                         + _p34_start_plan(subprocess, pathlib,
                                           __import__("re"))
                         + _p35_while_pipeline(subprocess, pathlib,
                                               __import__("re"))
                         + _p36_tape_gate(subprocess, tempfile, pathlib,
                                          __import__("re"))
                         + _p38_chunk_plan(subprocess, pathlib,
                                           __import__("re"))
                         + _p39_kv_inplace(subprocess, pathlib,
                                           __import__("re"))
                         + _p42_rope_view(subprocess, pathlib,
                                          __import__("re"))
                         + _p43_proj_pack(subprocess, pathlib,
                                          __import__("re"))
                         + _p44_ragged_decode(subprocess, pathlib,
                                              __import__("re"))
                         + _p25_cache_limit(subprocess, tempfile, pathlib,
                                            __import__("re"))
                         + _p27_flush_pressure(subprocess, tempfile, pathlib,
                                               __import__("re"))
                         + _p28_benefit_gate(subprocess, tempfile, pathlib,
                                             __import__("re"))
                         + _p26_callee_sdpa(subprocess, tempfile, pathlib,
                                            __import__("re"))
                         + _keras_attn_tags(subprocess, tempfile, pathlib,
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

    # ------------------------------------------------------------------
    # A forced kernel-build failure must never WEDGE the process (0.11.6).
    #
    # The shape is `_msl_chunked`: a generated kernel traced into a CHUNKED
    # replay, which was the one compiled call nothing settled before handing
    # it to `mx::async_eval`.  MLX attaches a per-stream event to every array
    # such a walk visits and signals them only after it finishes, so a build
    # failure raised mid-walk left those events attached and unsignaled, and
    # the next blocking eval waited on one forever (`Event::wait` ->
    # `waitUntilSignaledValue(..., -1)`).  Observed 2026-08-22; the stack is
    # in ~/.cache/metaljax-bench/logs/event-wedge/.
    #
    # Two things are asserted, and the first is the point: the child EXITS.
    # A wedge cannot be caught in-process -- it is an unkillable wait on a
    # shared event -- so the test is a child under a timeout, and a
    # regression shows up as "WEDGED" rather than as a suite that never
    # finishes.  Second, the answer survives the recovery: with every
    # generated kernel rejected, the interpreted loop each entry carries
    # alongside must produce what the CPU produces.
    print("\nthe no-wedge contract")
    print("-" * 62)
    label = "a rejected kernel never wedges"
    probe = ref_path.with_name(ref_path.name.replace("reference", "wedge"))
    cpu_probe = probe.with_suffix(".cpu.npz")
    try:
        cpu_child = dict(os.environ)
        cpu_child["JAX_PLATFORMS"] = "cpu"
        cpu_child.pop("METALJAX_PLUGIN_PATH", None)
        subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--wedge-probe", str(cpu_probe)],
            env=cpu_child, capture_output=True, text=True, check=True,
            timeout=900)
        forced = dict(os.environ)
        forced["METALJAX_MSL_FORCE_BUILD_FAIL"] = "1"
        subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--wedge-probe", str(probe)],
            env=forced, capture_output=True, text=True, check=True,
            timeout=900)
        got, want = np.load(probe), np.load(cpu_probe)
        worst = max(
            float(np.abs(got[k].astype(np.float64)
                         - want[k].astype(np.float64)).max())
            for k in want.files)
        ok = worst < 1e-5
        detail = (f"ok (worst {worst:.1e})" if ok else
                  f"FAIL: recovered answer is wrong ({worst:.1e})")
    except subprocess.TimeoutExpired:
        ok, detail = False, "FAIL: WEDGED (no exit in 900s -- Event::wait?)"
    except BaseException as exc:  # noqa: BLE001
        ok, detail = False, f"FAIL: {str(exc).splitlines()[0][:90]}"
    print(f"{label:<32} {'':>12}  {detail}")
    if not ok:
        failures.append("no-wedge contract")
    for p in (probe, cpu_probe):
        try:
            p.unlink()
        except OSError:
            pass

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
