"""Metal command-buffer sizing.

MLX 0.32 returned WRONG values -- silently -- when a single eval was chopped
into many Metal command buffers: work in one command buffer could read a
value an earlier one had not finished writing.  MLX starts a new command
buffer once the current one holds `MLX_MAX_OPS_PER_BUFFER` kernels or
`MLX_MAX_MB_PER_BUFFER` megabytes of work, so either budget can trigger it,
and which alignments bite is not predictable from the budget alone.  The
plugin pins both budgets (plugin-native/metal/metal_client.cc) before MLX
builds its Metal device.

The dropped cross-command-buffer fence is FIXED in the vendored
`libmlx_metaljax` the release links (notes/mlx-patch-diagnosis.md); the
budget-sweeping CANARIES that pinned the corruption on stock MLX were
removed 2026-08-18 (Oleg's call -- on the vendored build no swept budget
corrupts, so every canary failed by design; history at 29bb8eb).

What stays, ported to the PJRT route at the Stage-1 retirement, are the
CORRECTNESS detectors over the two real assets that caught the corruption
classes: the qwen3 decode step (compiled-graph face) and the qwen3
parameter-init scan (loop sync-point face).  Each layout pair runs the same
ops with sync points in different places, so disagreement is exactly the
corruption class -- whatever its cause.  Each layout runs in a fresh
subprocess: MLX latches its budgets at device build, and the METALJAX_*
layout knobs are safest read at a clean process start.

Two axes the Stage-1 file had are gone with it: the in-process flush-cadence
patch (`control._PERIOD_MAX` has no native env knob) and the
engagement-counter asserts (`engine.NATIVE.stats()`); the eager-flush axis
is covered through METALJAX_EAGER_FLUSH_MB instead.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

MODULE = os.path.join(os.path.dirname(__file__), "data",
                      "qwen3_prefill_shrunk.mlir")
INIT_MODULE = os.path.join(os.path.dirname(__file__), "data",
                           "qwen3_init_scan.mlir")

# Iterations of the init scan to run.  The corruption class needs one whole
# flush window plus most of another: 8 iterations were clean at every
# cadence on stock MLX, 10 were not.  The tensors stay at the model's real
# parameter shapes -- shrinking them 4x stopped it reproducing.
_SCAN_ITERS = 10


def _scan_source():
    text = open(INIT_MODULE).read()
    old = "stablehlo.constant dense<28> : tensor<i32>"
    new = f"stablehlo.constant dense<{_SCAN_ITERS}> : tensor<i32>"
    assert old in text, "the init scan's loop bound is no longer a constant 28"
    return text.replace(old, new)


# --- the shipped budgets must be bounded ------------------------------------


def test_command_buffer_budgets_are_bounded():
    # The plugin pins both with setenv() when it loads -- which Python's
    # os.environ (a startup snapshot) cannot see, so read the live process
    # environment through libc after forcing the backend up.
    import ctypes
    import jax

    jax.devices("metal")
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    ops = int(libc.getenv(b"MLX_MAX_OPS_PER_BUFFER"))
    mb = int(libc.getenv(b"MLX_MAX_MB_PER_BUFFER"))
    # The bounds are the MEASURED clean bands of the two assets in this
    # file, swept on 114b4d4 (stock MLX; logs in
    # notes/mlx-command-buffer-split.md). The kernel budget's clean band is
    # 450..1300 with the shipped byte budget. The byte budget is bounded
    # both ways: >=512 or stock MLX corrupts the split scan; <=2048 or one
    # command buffer can accumulate tens of GB of unpageable transient
    # intermediates and panic the machine (SD3.5 MMDiT at 1024^2 did,
    # twice). Moving either value outside these bands means re-running the
    # sweep, not widening the test.
    assert 450 <= ops <= 1300
    assert 512 <= mb <= 2048


# --- subprocess driver ------------------------------------------------------


def _child(asset, out_path, env_overrides, runs=1):
    """Run `asset` through the plugin in a fresh process; save outputs.

    The child executes the module `runs` times through ONE loaded
    executable, asserts the replays bit-identical (nothing in either asset
    is order-nondeterministic), and saves run 0 to `out_path` as npz.
    """
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_overrides.items()})
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(root, "tests")]
        + [p for p in [env.get("PYTHONPATH")] if p])
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), asset, out_path,
         str(runs)],
        env=env, capture_output=True, text=True, timeout=900)
    ok = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not ok:
        raise AssertionError(
            f"probe {env_overrides} failed to run (rc={proc.returncode})\n"
            f"{proc.stdout}\n{proc.stderr}")
    return json.loads(ok[-1])


def _load(path):
    with np.load(path) as z:
        return [z[f"o{i}"] for i in range(int(z["n"]))]


# --- the compiled decode step ----------------------------------------------


def test_compiled_llm_step_is_correct_and_deterministic():
    """Three replays of the compiled decode program must agree bit-for-bit,
    and the compiled path must agree (loosely) with op-by-op execution of
    the same module (METALJAX_COMPILE=0) -- fused kernels may round
    differently, but corruption is not subtle: it moved these outputs by
    ~50% of their range."""
    with tempfile.TemporaryDirectory(prefix="mj-cbuf-") as tmp:
        a = os.path.join(tmp, "compiled.npz")
        b = os.path.join(tmp, "eager.npz")
        _child(MODULE, a, {}, runs=3)          # determinism asserted in-child
        _child(MODULE, b, {"METALJAX_COMPILE": "0"})
        got, want = _load(a), _load(b)

    assert len(got) == len(want)
    bad = []
    for j, (g, exp) in enumerate(zip(got, want)):
        g, exp = g.astype(np.float64), exp.astype(np.float64)
        if g.shape != exp.shape:
            bad.append(f"output {j}: shape {g.shape} vs {exp.shape}")
            continue
        finite = exp[~np.isnan(exp)] if exp.size else exp
        scale = float(np.max(np.abs(finite))) if finite.size else 0.0
        tol = 2e-2 * np.abs(exp) + 1e-2 * scale + 1e-6
        off = (np.abs(g - exp) > tol) | (np.isnan(g) != np.isnan(exp))
        n = int(np.sum(off))
        if n:
            bad.append(f"compiled output {j} {tuple(exp.shape)} differs from "
                       f"op-by-op execution in {n} of {exp.size} elements")
    assert not bad, "\n".join(bad)


# --- the init scan: loop sync-point layouts must agree ----------------------
#
# Same ops in the same order, only evaluated at different points, so the
# results must be BIT-exact across layouts within one execution mode.  The
# layout knobs are the native engine's own: METALJAX_WHILE_PIPELINE moves a
# dynamic loop between the pipelined and serial sync layouts,
# METALJAX_BODY_COMPILE=0 runs loop bodies op by op, METALJAX_COMPILE=0 runs
# the whole program eagerly and METALJAX_EAGER_FLUSH_MB moves the eager
# path's flush cadence.


def _scan_checksums(env_overrides):
    with tempfile.TemporaryDirectory(prefix="mj-scan-") as tmp:
        out = os.path.join(tmp, "out.npz")
        rec = _child("SCAN", out, env_overrides)
    return rec["checksums"]


def test_native_scan_loop_layouts_agree():
    base = {"METALJAX_BODY_COMPILE": "0"}
    shipped = _scan_checksums(base)
    pipelined = _scan_checksums(dict(base, METALJAX_WHILE_PIPELINE=1 << 20))
    serial = _scan_checksums(dict(base, METALJAX_WHILE_PIPELINE=0))
    bad = []
    for name, other in (("pipelined", pipelined), ("serial", serial)):
        for j, (a, b) in enumerate(zip(shipped, other)):
            if a != b:
                bad.append(f"output {j}: checksum {a} at the shipped layout, "
                           f"{b} under the {name} loop layout")
    assert not bad, "\n".join(bad)


def test_eager_scan_is_independent_of_flush_cadence():
    base = {"METALJAX_COMPILE": "0"}
    shipped = _scan_checksums(base)
    frequent = _scan_checksums(dict(base, METALJAX_EAGER_FLUSH_MB=1))
    bad = []
    for j, (a, b) in enumerate(zip(shipped, frequent)):
        if a != b:
            bad.append(f"loop output {j} depends on when the graph is "
                       f"evaluated: checksum {a} at the shipped flush "
                       f"cadence, {b} flushing every MB")
    assert not bad, "\n".join(bad)


# --- child ------------------------------------------------------------------


def _dev_checksum(o):
    """A bitwise checksum, computed ON the device: every element's bit
    pattern summed in 64 bits.  The scan's outputs are whole-model
    parameter buffers, and pulling gigabytes to the host per layout would
    make these the most expensive tests in the suite."""
    import jax
    import jax.numpy as jnp

    v = o
    if v.dtype.itemsize != 4:
        v = v.astype(jnp.float32)
    bits = jax.lax.bitcast_convert_type(v, jnp.uint32).astype(jnp.uint64)
    return int(jnp.sum(bits.reshape(-1)))


def _child_main(asset, out_path, runs):
    import jax
    from jax._src.lib import xla_client as xc

    if asset == "SCAN":
        text = _scan_source()
        args = []
    else:
        text = open(asset).read()
        # Deterministic pseudo-random arguments (the decode asset's inputs).
        import test_sdpa
        args = test_sdpa._asset_inputs(test_sdpa._asset_arg_specs(text))

    dev = jax.devices("metal")[0]
    exe = dev.client.compile_and_load(text, [dev], xc.CompileOptions())
    dargs = [jax.device_put(a, dev) for a in args]

    first = sums = None
    for r in range(runs):
        outs = exe.execute(dargs)
        if asset == "SCAN":
            run_sums = [_dev_checksum(o) for o in outs]
            host = None
        else:
            # f32 on the way out: np.savez cannot serialize ml_dtypes
            # extension dtypes, and the parent's comparison is a loose
            # numeric one anyway.
            host = [np.asarray(o).astype(np.float32) for o in outs]
            run_sums = None
        if first is None and host is not None:
            first = host
        elif host is not None:
            for j, (a, b) in enumerate(zip(first, host)):
                if not np.array_equal(a, b, equal_nan=True):
                    print(json.dumps({"error": f"replay {r} output {j} "
                                      f"differs from replay 0"}))
                    raise SystemExit(3)
        if sums is None:
            sums = run_sums
        elif run_sums is not None and run_sums != sums:
            print(json.dumps({"error": f"replay {r} checksums differ"}))
            raise SystemExit(3)
        del outs

    if first is not None:
        np.savez(out_path, n=len(first),
                 **{f"o{i}": x for i, x in enumerate(first)})
    else:
        np.savez(out_path, n=0)
    print(json.dumps({"asset": os.path.basename(str(asset)),
                      "checksums": sums or []}))


if __name__ == "__main__":
    _child_main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
