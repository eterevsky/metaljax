#!/usr/bin/env python
"""Full-stack proof for a *wheel* carrying the native PJRT plugin.

Unlike `smoke_test.py`, which points METALJAX_PLUGIN_PATH at a bazel-bin
dylib, this runs the way a user would: everything must come from an
installed wheel. Run it from a fresh venv that has the native wheel
installed and nothing else metaljax-shaped on sys.path:

    uv venv --python 3.13 /tmp/mj-native
    uv pip install -p /tmp/mj-native/bin/python dist/metaljax-*.whl
    JAX_PLATFORMS=metal /tmp/mj-native/bin/python plugin-native/wheel_poc_test.py

Checkpoints, in order:

  a. jax.devices() is served by *this* plugin (platform_version carries the
     "metaljax-native-p0" sentinel) out of the wheel's own lib directory
  b. metaljax.engine never entered sys.modules -- no Python trampoline
  c. f32 device_put / np.asarray round-trips bit-exactly (host transfer
     through MetalClient -> mlx::core::array -> back)
  d. jax.jit COMPUTES: CompileAndLoad lowers the parsed StableHLO into the
     executor's tape and Execute replays it on the GPU

Exit status is nonzero if any checkpoint does not behave as described.
"""

import importlib.util
import os
import pathlib
import sys
import traceback

if os.environ.get("METALJAX_PLUGIN_PATH"):
    sys.exit("METALJAX_PLUGIN_PATH is set: this test must resolve the plugin "
             "from the installed wheel (unset it, or use smoke_test.py)")
os.environ.setdefault("JAX_PLATFORMS", "metal")

_spec = importlib.util.find_spec("metaljax")
if _spec is None or not _spec.submodule_search_locations:
    sys.exit("metaljax is not installed in this interpreter")
_LIB = pathlib.Path(next(iter(_spec.submodule_search_locations))) / "lib"
_DYLIB = _LIB / "libmetal_pjrt_native.dylib"
if not _DYLIB.exists():
    sys.exit(f"{_DYLIB} is missing: this is not a native wheel "
             f"(lib/ holds {[p.name for p in _LIB.glob('*')]})")
print(f"python : {sys.version.split()[0]} ({sys.executable})")
print(f"plugin : {_DYLIB} ({_DYLIB.stat().st_size / 1e6:.1f} MB)")

import numpy as np  # noqa: E402
import jax  # noqa: E402

FAILED = []


def step(name):
    def deco(fn):
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"PASS: {name}")
        except BaseException:  # noqa: BLE001 - report and keep going
            traceback.print_exc()
            print(f"FAIL: {name}")
            FAILED.append(name)
        return fn

    return deco


@step("a: device discovery from the installed wheel")
def _devices():
    devices = jax.devices()
    print("jax.devices() ->", devices)
    dev = devices[0]
    print("  platform          :", dev.platform)
    print("  device_kind       :", dev.device_kind)
    print("  client.platform_version:", dev.client.platform_version)
    print("  memory spaces     :", [m.kind for m in dev.addressable_memories()])
    # jax prefixes "PJRT C API\n" to whatever the plugin reports.
    assert "metaljax-native-p0" in dev.client.platform_version, (
        "a different plugin answered: " + dev.client.platform_version)
    assert "site-packages" in str(_DYLIB), f"not an installed wheel: {_DYLIB}"


@step("b: the Python engine is absent")
def _no_engine():
    for mod in ("metaljax.engine", "metaljax.interpreter"):
        assert mod not in sys.modules, f"{mod} was imported"
    print("  metaljax.engine imported: False (fully native)")


@step("c: host round-trip")
def _roundtrip():
    src = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.5 - 1.0
    buf = jax.device_put(src)
    print("  device_put ->", buf.shape, buf.dtype, buf.device)
    back = np.asarray(buf)
    assert back.dtype == src.dtype, (back.dtype, src.dtype)
    assert back.shape == src.shape, (back.shape, src.shape)
    assert np.array_equal(back, src), (back, src)
    print("  round-trip exact")


@step("d: jit compiles and executes")
def _compile():
    # The whole stack, from an installed wheel: StableHLO parsed by the C-API
    # wrapper, lowered to the executor's tape by CompileAndLoad, replayed on
    # the GPU by Execute, read back through the transfer path.
    import jax.numpy as jnp

    x = jax.device_put(np.ones((2, 3), dtype=np.float32))
    out = np.asarray(jax.jit(lambda a: a * 2)(x))
    print("  jit(a * 2) ->", out.tolist())
    assert np.array_equal(out, np.full((2, 3), 2.0, np.float32)), out
    # Milestone zero, in the words CLAUDE.md uses for it.
    zero = np.asarray(2 * jnp.array([1, 2, 3]))
    print("  2 * jnp.array([1, 2, 3]) ->", zero.tolist())
    assert np.array_equal(zero, np.array([2, 4, 6], np.int32)), zero
    # ...and a sort now computes here too (P6), which is the family this
    # checkpoint used to watch decline.
    srt = np.asarray(jax.jit(jnp.sort)(
        jax.device_put(np.array([3.0, 1.0, 2.0], np.float32))))
    print("  jit(jnp.sort) ->", srt.tolist())
    assert np.array_equal(srt, np.array([1.0, 2.0, 3.0], np.float32)), srt
    # ...and a convolution computes here too (P7), which is the family this
    # checkpoint watched decline after sort.
    cv = np.asarray(jax.jit(lambda a, k: jax.lax.conv_general_dilated(
        a, k, (1,), "VALID", dimension_numbers=("NCH", "OIH", "NCH")))(
            jax.device_put(np.ones((1, 2, 4), np.float32)),
            jax.device_put(np.ones((1, 2, 3), np.float32))))
    print("  jit(conv_general_dilated) ->", cv.tolist())
    assert np.array_equal(cv, np.full((1, 1, 2), 6.0, np.float32)), cv
    # ...and so does a cholesky (P9), whose LAPACK the wheel reaches through
    # Accelerate -- which is what says the framework really is linked into the
    # dylib this wheel carries.
    ch = np.asarray(jax.jit(lambda a: jax.numpy.linalg.cholesky(
        a @ a.T + 4 * jax.numpy.eye(3)))(
            jax.device_put(np.eye(3, dtype=np.float32))))
    print("  jit(jnp.linalg.cholesky) ->", ch.diagonal().tolist())
    assert np.allclose(ch, np.sqrt(5.0) * np.eye(3), atol=1e-6), ch
    # An op outside the native set still declines, and says which one.
    # `stablehlo.rng` is the stand-in now (phase 2's P11 gave reduce_precision
    # an executor); neither engine implements XLA's non-deterministic RNG.
    try:
        jax.jit(lambda a, b: jax.lax.rng_uniform(a, b, (4,)))(
            jax.device_put(np.float32(0.0)), jax.device_put(np.float32(1.0)))
    except Exception as e:  # noqa: BLE001 - the expected decline
        msg = str(e)
        print("  rng_uniform declined:", msg.splitlines()[0][:160])
        assert "stablehlo.rng" in msg, "the decline did not name the op"
        return
    raise AssertionError("rng_uniform unexpectedly compiled")


if FAILED:
    sys.exit(f"{len(FAILED)} checkpoint(s) failed: {', '.join(FAILED)}")
print("\nall checkpoints passed")
