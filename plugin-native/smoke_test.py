#!/usr/bin/env python
"""P0 smoke test for the native PJRT plugin.

Run with the repo venv:

    plugin-native/../.venv/bin/python plugin-native/smoke_test.py

It points METALJAX_PLUGIN_PATH at the bazel-built dylib (overridable) and
walks the three checkpoints in order, reporting how far the plugin gets:

  2. jax.devices() lists a device served by *this* dylib
     (platform_version == "metaljax-native-p0")
  3. jax.device_put / np.asarray round-trips an f32 array bit-exactly
  4. jax.jit compiles and executes: the parsed StableHLO becomes the
     executor's tape and runs on the GPU (execute_test.py is the
     differential suite; this is only the "it works at all" checkpoint)
"""

import os
import pathlib
import sys
import traceback

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_DYLIB = _HERE / "bazel-bin" / "metal" / "libmetal_pjrt_native.dylib"

os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
os.environ.setdefault("JAX_PLATFORMS", "metal")

dylib = pathlib.Path(os.environ["METALJAX_PLUGIN_PATH"])
if not dylib.exists():
    sys.exit(f"plugin dylib not found: {dylib}")
print(f"plugin: {dylib} ({dylib.stat().st_size / 1e6:.1f} MB)")

import numpy as np  # noqa: E402
import jax  # noqa: E402


FAILED = []


def step(name):
    def deco(fn):
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"PASS: {name}")
        except BaseException:  # noqa: BLE001 - report and continue
            traceback.print_exc()
            print(f"FAIL: {name}")
            FAILED.append(name)
        return fn

    return deco


@step("checkpoint 2: device discovery")
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
    # Nothing about this path may re-enter Python: the Stage 1 engine must
    # never have been imported.
    assert "metaljax.engine" not in sys.modules, "the Python engine loaded"
    print("  metaljax.engine imported: False (fully native)")


@step("checkpoint 3: host round-trip")
def _roundtrip():
    src = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.5 - 1.0
    buf = jax.device_put(src)
    print("  device_put ->", buf.shape, buf.dtype, buf.device)
    back = np.asarray(buf)
    assert back.dtype == src.dtype, (back.dtype, src.dtype)
    assert back.shape == src.shape, (back.shape, src.shape)
    assert np.array_equal(back, src), (back, src)
    print("  round-trip exact")


@step("checkpoint 4: compile and execute")
def _compile():
    import jax.numpy as jnp

    x = jax.device_put(np.ones((2, 3), dtype=np.float32))
    out = np.asarray(jax.jit(lambda a: a * 2)(x))
    print("  jit(a * 2) ->", out.tolist())
    assert np.array_equal(out, np.full((2, 3), 2.0, np.float32)), out
    # Milestone zero, in the words CLAUDE.md uses for it.
    zero = np.asarray(2 * jnp.array([1, 2, 3]))
    print("  2 * jnp.array([1, 2, 3]) ->", zero.tolist())
    assert np.array_equal(zero, np.array([2, 4, 6], np.int32)), zero
    # ...and so does a convolution, which is the family this checkpoint
    # watched decline until P7 gave the executor a `kConv`.
    cv = np.asarray(jax.jit(lambda a, k: jax.lax.conv_general_dilated(
        a, k, (1,), "VALID", dimension_numbers=("NCH", "OIH", "NCH")))(
            jax.device_put(np.ones((1, 2, 4), np.float32)),
            jax.device_put(np.ones((1, 2, 3), np.float32))))
    print("  jit(conv_general_dilated) ->", cv.tolist())
    assert np.array_equal(cv, np.full((1, 1, 2), 6.0, np.float32)), cv
    # An op outside the native set declines, naming itself.  The host-computed
    # LAPACK targets are what is left (phase 2's next milestone); every other
    # family the census used to name now lowers.
    try:
        jax.jit(lambda a: jnp.linalg.cholesky(a @ a.T + 4 * jnp.eye(3)))(
            jax.device_put(np.eye(3, dtype=np.float32)))
    except Exception as e:  # noqa: BLE001 - the expected decline
        msg = str(e)
        print("  cholesky declined:", msg.splitlines()[0][:160])
        assert "stablehlo.cholesky" in msg, "the decline did not name the op"
        return
    raise AssertionError("cholesky unexpectedly compiled")


if FAILED:
    sys.exit(f"{len(FAILED)} checkpoint(s) failed: {', '.join(FAILED)}")
print("\nall checkpoints passed")
