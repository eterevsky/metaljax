"""Metal command-buffer sizing.

MLX 0.32 returns WRONG values -- silently, and differently on every call --
when a single eval of an mx.compile'd graph is chopped into many Metal
command buffers. MLX starts a new command buffer once the current one holds
`MLX_MAX_OPS_PER_BUFFER` kernels or `MLX_MAX_MB_PER_BUFFER` megabytes of
work; both thresholds trigger the corruption when they fire every few
kernels (measured: broken at <=4 kernels apart, clean from ~8 on).

MLX's byte default (40 MB) is the one that bites in practice: an LLM's
per-layer tensors are tens of megabytes, so a whole-model compiled main
commits constantly. maxtext's qwen3-0.6B decode came out as garbage tokens,
different on every run, while the same program interpreted op-by-op (or run
with a larger byte budget) matched the CPU backend exactly.

metaljax therefore raises both budgets in `metaljax/__init__.py` before mlx
loads. The module below is that decode program (a maxtext qwen3 prefill step,
shrunk to 8 layers), which reproduces the corruption in about a second.
"""

import os

import mlx.core as mx
import numpy as np

from metaljax import dtypes as mdt
from metaljax.interpreter import Interpreter
from metaljax.ops import control

MODULE = os.path.join(os.path.dirname(__file__), "data",
                      "qwen3_prefill_shrunk.mlir")


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


def test_command_buffer_budgets_are_raised():
    # Read by MLX once, at load; metaljax/__init__ sets them before mlx.core
    # is imported. The kernel budget is a speed choice, the byte budget a
    # correctness one -- see the module docstring.
    assert int(os.environ["MLX_MAX_OPS_PER_BUFFER"]) >= 64
    assert int(os.environ["MLX_MAX_MB_PER_BUFFER"]) >= 4096


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
