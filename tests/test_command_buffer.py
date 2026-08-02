"""Metal command-buffer sizing.

MLX 0.32 returns WRONG values -- silently -- when a single eval is chopped
into many Metal command buffers: work in one command buffer can read a value
an earlier one had not finished writing. MLX starts a new command buffer once
the current one holds `MLX_MAX_OPS_PER_BUFFER` kernels or
`MLX_MAX_MB_PER_BUFFER` megabytes of work, so either budget can trigger it,
and which alignments bite is not predictable from the budget alone.

Both paths through the engine have been hit by real programs:

* compiled graphs, by MLX's 40 MB byte default -- an LLM's per-layer tensors
  are tens of megabytes, so a whole-model compiled main commits constantly.
  maxtext's qwen3-0.6B decode came out as garbage tokens, different on every
  run, while the same program interpreted op-by-op matched the CPU backend.
* op-by-op interpretation, by the 400-kernel budget we had chosen for speed:
  maxtext's qwen3 parameter init (a 28-layer scan, flushed every 5
  iterations by the eager while-loop) computed one layer's RNG key wrong, so
  training started from different weights -- first-step loss 208.78 where
  jax-CPU and the compiled path both say 247.78.

metaljax therefore sets both budgets in `metaljax/__init__.py` before mlx
loads, and the two modules below -- that decode program, and that init scan
-- are what pin the values.
"""

import os

import mlx.core as mx
import numpy as np

from metaljax import dtypes as mdt
from metaljax.interpreter import REGISTRY, Interpreter
from metaljax.ops import control

MODULE = os.path.join(os.path.dirname(__file__), "data",
                      "qwen3_prefill_shrunk.mlir")
INIT_MODULE = os.path.join(os.path.dirname(__file__), "data",
                           "qwen3_init_scan.mlir")


def _inputs(interp):
    """Deterministic pseudo-random arguments for the module."""
    rng = np.random.default_rng(0)
    args = []
    with interp.context:
        avals = interp.in_avals
    for shape, dt in avals:
        if np.issubdtype(dt, np.integer):
            x = rng.integers(0, 4, size=shape).astype(dt)
        elif dt == np.bool_:
            x = rng.integers(0, 2, size=shape).astype(bool)
        else:
            x = (rng.standard_normal(size=shape) * 0.1).astype(np.float32)
            x = x.astype(dt)
        args.append(mdt.to_mx(x))
    mx.eval(*args)
    return args


def _compiled(interp):
    with interp.context:
        underived = control._underived_outputs(interp._main_block(), [])

    def traced(*a):
        prev = interp._in_trace
        interp._in_trace = True
        try:
            return control._anchor_outputs(tuple(interp(*a)), a, underived)
        finally:
            interp._in_trace = prev

    return mx.compile(traced)


def test_command_buffer_budgets_are_bounded():
    # Read by MLX once, at load; metaljax/__init__ sets them before mlx.core
    # is imported. The kernel budget trades launch overhead against the
    # corrupting alignments below (400 is one; the scan test catches it).
    # The byte budget is BOUNDED BOTH WAYS: >=160 MB or MLX 0.32 corrupts
    # split compiled graphs (this file's other tests); <=2048 MB or one
    # command buffer can accumulate tens of GB of unpageable transient
    # intermediates and panic the machine (SD3.5 MMDiT at 1024^2 did,
    # twice) -- every intermediate lives until its command buffer completes.
    assert int(os.environ["MLX_MAX_OPS_PER_BUFFER"]) >= 64
    assert 160 <= int(os.environ["MLX_MAX_MB_PER_BUFFER"]) <= 2048


def test_compiled_llm_step_is_correct_and_deterministic():
    interp = Interpreter(open(MODULE).read().encode())
    args = _inputs(interp)

    want = [mdt.to_np(o).astype(np.float64) for o in interp(*args)]

    fn = _compiled(interp)
    runs = []
    for _ in range(3):
        outs = list(fn(*args))
        mx.eval(*outs)
        runs.append([mdt.to_np(o).astype(np.float64) for o in outs])

    for j, (a, b) in enumerate(zip(runs[0], runs[1])):
        # Nothing here is order-nondeterministic (no scatter-add, no atomic
        # reduction), so replays of one graph must be bit-identical.
        np.testing.assert_array_equal(
            a, b, err_msg=f"output {j} differs between two compiled replays")
        np.testing.assert_array_equal(a, runs[2][j])

    for j, (got, exp) in enumerate(zip(runs[0], want)):
        assert got.shape == exp.shape
        scale = float(np.nanmax(np.abs(exp))) if exp.size else 0.0
        # Loose: fused kernels may round differently from op-by-op
        # evaluation (metal::pow in a fused chain, say). Corruption is not
        # subtle -- it moved these outputs by ~50% of their range.
        np.testing.assert_allclose(
            got, exp, rtol=2e-2, atol=1e-2 * scale + 1e-6,
            err_msg=f"compiled output {j} differs from op-by-op evaluation")


# --- op-by-op path: a scan's result must not depend on the flush cadence ---

# Iterations of the init scan to run. The corruption needs one whole flush
# window plus most of another: 8 iterations are clean at every cadence, 10
# are not. Each iteration interprets ~4.9k ops at the model's real parameter
# shapes -- shrinking the tensors 4x stops it reproducing, so the asset keeps
# them (~2 GB of device memory for the two runs, a few seconds).
_SCAN_ITERS = 10


def _run_to_while(interp):
    """Interpret main up to its stablehlo.while; return (op, carry, env)."""
    env = {}
    for op in interp._main_block().operations:
        o = op.operation
        if o.name == "stablehlo.while":
            return o, [env[v] for v in o.operands], env
        ins = [env[v] for v in o.operands]
        out = REGISTRY[o.name](interp, o, ins, env)
        if isinstance(out, mx.array):
            out = [out]
        for r, v in zip(o.results, out):
            env[r] = v
    raise AssertionError("test module has no while loop")


def _run_scan(interp, body, ins, env, flush_every):
    """Iterate the loop body eagerly, evaluating every `flush_every` steps.

    This is ops.control._while's uncompiled path (METALJAX_COMPILE=0, and
    every fallback from a failed compile), whose cadence for this body is 5.
    """
    fn, free, _ = control._body_fn(interp, body, compile_body=False)
    captures = [env[v] for v in free]
    vals = list(ins)
    for i in range(1, _SCAN_ITERS + 1):
        vals = list(fn(*vals, *captures))
        if i % flush_every == 0:
            mx.eval(*vals)
    mx.eval(*vals)
    return vals


def test_eager_scan_is_independent_of_flush_cadence():
    interp = Interpreter(open(INIT_MODULE).read().encode())
    with interp.context:
        wop, ins, env = _run_to_while(interp)
        body = wop.regions[1].blocks[0]
        # The cadence ops.control._while picks for this body.
        cost = control._block_cost(interp, body)
        period = max(1, min(64, 25_000 // max(cost, 1)))
        assert period > 1, "flush cadence must batch iterations to be a test"

        want = _run_scan(interp, body, ins, env, flush_every=1)
        got = _run_scan(interp, body, ins, env, flush_every=period)

        # Same ops in the same order, only evaluated at different points:
        # nothing here is order-nondeterministic, so this is bit-exact.
        for j, (a, b) in enumerate(zip(want, got)):
            if not bool(mx.all(a == b).item()):
                diff = float(mx.sum(mx.abs(a.astype(mx.float32)
                                           - b.astype(mx.float32))).item())
                raise AssertionError(
                    f"loop carry {j} {tuple(a.shape)} depends on when the "
                    f"graph is evaluated (flush every {period} iterations vs "
                    f"every one): total abs diff {diff}")
