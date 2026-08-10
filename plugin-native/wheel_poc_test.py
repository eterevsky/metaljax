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
  d. jax.jit fails *from the right layer*: the plugin's Unimplemented,
     raised by CompileAndLoad after it parsed and logged the StableHLO

(d) failing is the expected P0 outcome -- the native plugin has no executor
yet. Exit status is nonzero if any checkpoint does not behave as described.
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


@step("d: jit reaches CompileAndLoad with parsed StableHLO")
def _compile():
    x = jax.device_put(np.ones((2, 3), dtype=np.float32))
    try:
        jax.jit(lambda a: a * 2)(x)
    except Exception as e:  # noqa: BLE001 - the expected P0 outcome
        msg = str(e)
        print("  compile raised:", msg.splitlines()[0][:200])
        assert "metaljax-native P0" in msg, "unexpected failure, see above"
        return
    raise AssertionError("compile unexpectedly succeeded")


if FAILED:
    sys.exit(f"{len(FAILED)} checkpoint(s) failed: {', '.join(FAILED)}")
print("\nall checkpoints passed")
