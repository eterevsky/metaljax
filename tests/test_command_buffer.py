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

Every finite budget is a draw in the same lottery, and OUR OWN COMMITS
RESHUFFLE IT: any change that moves a lowering moves where the command
buffers get cut. So the two correctness tests (shipped budgets, must be
clean) are paired with two CANARIES that sweep known-corrupting budgets in
subprocesses and fail if none of them corrupts any more -- a passing
correctness test proves nothing unless the same detector can still see the
bug. Sweeps, not single pinned values: a single value goes stale the moment
a lowering shifts (400 -> 200 did, after fdc7cde's shift peephole).
"""

import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np

import metaljax
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


def _compiled_mismatches():
    """Compiled-graph detector: complaints about the decode program, if any.

    Three replays of one compiled graph must agree bit-for-bit with each
    other and (loosely) with op-by-op interpretation of the same module.
    """
    interp = Interpreter(open(MODULE).read().encode())
    args = _inputs(interp)

    want = [mdt.to_np(o).astype(np.float64) for o in interp(*args)]

    fn = _compiled(interp)
    runs = []
    for _ in range(3):
        outs = list(fn(*args))
        mx.eval(*outs)
        runs.append([mdt.to_np(o).astype(np.float64) for o in outs])

    bad = []
    for j, a in enumerate(runs[0]):
        # Nothing here is order-nondeterministic (no scatter-add, no atomic
        # reduction), so replays of one graph must be bit-identical.
        for k in (1, 2):
            b = runs[k][j]
            if not np.array_equal(a, b, equal_nan=True):
                n = int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))))
                bad.append(f"output {j} {tuple(a.shape)}: compiled replays 0 "
                           f"and {k} differ in {n} of {a.size} elements")
                break

    for j, (got, exp) in enumerate(zip(runs[0], want)):
        if got.shape != exp.shape:
            bad.append(f"output {j}: shape {got.shape} vs {exp.shape}")
            continue
        finite = exp[~np.isnan(exp)] if exp.size else exp
        scale = float(np.max(np.abs(finite))) if finite.size else 0.0
        # Loose: fused kernels may round differently from op-by-op
        # evaluation (metal::pow in a fused chain, say). Corruption is not
        # subtle -- it moved these outputs by ~50% of their range.
        tol = 2e-2 * np.abs(exp) + 1e-2 * scale + 1e-6
        off = (np.abs(got - exp) > tol) | (np.isnan(got) != np.isnan(exp))
        n = int(np.sum(off))
        if n:
            d = np.abs(got - exp)[off]
            d = d[~np.isnan(d)]
            worst = float(d.max()) if d.size else float("nan")
            bad.append(f"compiled output {j} {tuple(exp.shape)} differs from "
                       f"op-by-op evaluation in {n} of {exp.size} elements "
                       f"(worst abs {worst})")
    return bad


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


def _scan_mismatches():
    """Op-by-op detector: complaints about the init scan, if any.

    Same ops in the same order, only evaluated at different points, so the
    carries must be bit-exact whatever the flush cadence.
    """
    interp = Interpreter(open(INIT_MODULE).read().encode())
    bad = []
    with interp.context:
        wop, ins, env = _run_to_while(interp)
        body = wop.regions[1].blocks[0]
        # The cadence ops.control._while picks for this body.
        cost = control._block_cost(interp, body)
        period = max(1, min(64, 25_000 // max(cost, 1)))
        assert period > 1, "flush cadence must batch iterations to be a test"

        want = _run_scan(interp, body, ins, env, flush_every=1)
        got = _run_scan(interp, body, ins, env, flush_every=period)

        for j, (a, b) in enumerate(zip(want, got)):
            if not bool(mx.all(a == b).item()):
                diff = float(mx.sum(mx.abs(a.astype(mx.float32)
                                           - b.astype(mx.float32))).item())
                bad.append(
                    f"loop carry {j} {tuple(a.shape)} depends on when the "
                    f"graph is evaluated (flush every {period} iterations vs "
                    f"every one): total abs diff {diff}")
    return bad


_DETECTORS = {"compiled": _compiled_mismatches, "scan": _scan_mismatches}


# --- the shipped budgets must be clean -------------------------------------

def test_command_buffer_budgets_are_bounded():
    # Read by MLX once, at load; metaljax/__init__ sets them before mlx.core
    # is imported. The bounds are the MEASURED clean bands of the two assets
    # in this file, swept on 114b4d4 (see _CANARY_OPS below for the full map
    # and the log paths). The kernel budget's clean band is 450..1300 with
    # the shipped byte budget: below it the kernel cuts land on bad
    # boundaries (400/200/100/50 all corrupt the scan), above it the BYTE
    # budget starts cutting instead and corrupts (1350 up, all the way to
    # unlimited). The byte budget is bounded both ways: >=416 measured, so
    # ship no lower than 512, or MLX corrupts the split scan (400 MB and
    # below do, 5/5); <=2048 MB or one command buffer can accumulate tens of
    # GB of unpageable transient intermediates and panic the machine (SD3.5
    # MMDiT at 1024^2 did, twice) -- every intermediate lives until its
    # command buffer completes.
    # Moving either value outside these bands means re-running the sweep
    # (scripts in notes/mlx-command-buffer-split.md), not widening the test.
    assert 450 <= int(os.environ["MLX_MAX_OPS_PER_BUFFER"]) <= 1300
    assert 512 <= int(os.environ["MLX_MAX_MB_PER_BUFFER"]) <= 2048


def test_compiled_llm_step_is_correct_and_deterministic():
    bad = _compiled_mismatches()
    assert not bad, "\n".join(bad)


def test_eager_scan_is_independent_of_flush_cadence():
    bad = _scan_mismatches()
    assert not bad, "\n".join(bad)


# --- canaries: the detectors above must still be able to SEE the bug -------

# Budgets that corrupted on 114b4d4 (2026-08-04), each in its own process
# because MLX reads the budgets once, at load. Swept with the detectors
# above; logs in ~/.cache/metaljax-bench/logs/canary-repin/.
#
# Kernel budget, scan detector, byte budget held at the shipped 512
# (bracketed = clean, everything else corrupts; swept in steps of 50):
#   50 100 [150] 200 [250 300 350] 400 [450 ... 1300] 1350 1400 ... 10^9
#   400, 200, 100 and 1600 each corrupt 5/5 fresh processes; the clean
#   band 450..1300 is 0/5 at both edges. Note the whole high tail
#   corrupts: with no kernel cut, the 512 MB byte budget cuts badly by
#   itself (clean again only at >=2048).
# Byte budget, compiled detector, kernel budget held at the shipped 800:
#   8 12 16 24 32 40 48 corrupt, 56 and up clean (the 2026-08-03 note said
#   80 corrupted -- that edge moved, which is exactly why these are sweeps).
#
# The sweeps run in the listed order and stop at the first value that
# corrupts, so the usual cost is one subprocess (~3 s scan, ~2 s compiled);
# the full fallback is ~25 s and only happens when the canary is dying.
_CANARY_OPS = (400, 200, 100, 2000, 50, 1600)
_CANARY_MB = (40, 20, 8, 48)

_CANARY_HELP = (
    "Either MLX fixed the command-buffer split bug (then the shipped budgets "
    "in metaljax/__init__.py are no longer pinned by anything and the whole "
    "workaround should be re-measured), or our lowering moved far enough "
    "that these assets no longer straddle a bad boundary (then the canary "
    "needs a bigger graph -- notes/mlx-command-buffer-split.md, and "
    "notes/data/qwen3_8b_prefill_36layer.mlir is the strongest asset we "
    "have). Do NOT just delete the canary: it is the only evidence that "
    "this file's other two tests can still fail.")


def _probe(mode, **budgets):
    """Run one detector in a fresh process at the given command-buffer budgets.

    Returns (corrupt, output). MLX latches both budgets at load, so this
    cannot be done in-process.
    """
    env = dict(os.environ)
    env.update({k: str(v) for k, v in budgets.items()})
    # Make the child import the metaljax under test, not whatever is
    # installed (worktrees, editable installs, PYTHONPATH runs).
    src = os.path.dirname(os.path.dirname(os.path.abspath(metaljax.__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [src] + [p for p in [env.get("PYTHONPATH")] if p])
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), mode],
                          env=env, capture_output=True, text=True, timeout=900)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        raise AssertionError(
            f"canary probe {mode} {budgets} failed to run "
            f"(rc={proc.returncode})\n{proc.stdout}\n{proc.stderr}")
    rec = json.loads(lines[-1])
    return rec["corrupt"], rec


def test_kernel_budget_canary_still_corrupts():
    """Some kernel budget must still break the eager scan detector."""
    tried = []
    for ops in _CANARY_OPS:
        corrupt, _ = _probe("scan", MLX_MAX_OPS_PER_BUFFER=ops)
        tried.append((ops, corrupt))
        if corrupt:
            return
    raise AssertionError(
        f"no kernel budget in {_CANARY_OPS} corrupts the init scan any more "
        f"(tried {tried}). " + _CANARY_HELP)


def test_byte_budget_canary_still_corrupts():
    """Some byte budget must still break the compiled-graph detector."""
    tried = []
    for mb in _CANARY_MB:
        corrupt, _ = _probe("compiled", MLX_MAX_MB_PER_BUFFER=mb)
        tried.append((mb, corrupt))
        if corrupt:
            return
    raise AssertionError(
        f"no byte budget in {_CANARY_MB} corrupts the compiled decode "
        f"program any more (tried {tried}). " + _CANARY_HELP)


if __name__ == "__main__":
    # Child of _probe: run one detector under this process's budgets and
    # report on stdout as a single JSON line.
    _mode = sys.argv[1]
    _bad = _DETECTORS[_mode]()
    print(json.dumps({
        "mode": _mode,
        "ops": int(os.environ["MLX_MAX_OPS_PER_BUFFER"]),
        "mb": int(os.environ["MLX_MAX_MB_PER_BUFFER"]),
        "corrupt": bool(_bad),
        "detail": _bad,
    }))
