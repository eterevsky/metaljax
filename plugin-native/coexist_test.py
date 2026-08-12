#!/usr/bin/env python3
"""Does the native plugin survive a process that also holds TensorFlow?

The dylib statically links protobuf, LLVM/MLIR and absl.  dyld coalesces WEAK
definitions across images by name, so an unrestricted export table lets the
plugin's `google::protobuf::internal::AddDescriptors` bind to TensorFlow's
protobuf (SIGSEGV during the plugin's own descriptor registration), and with
the plugin loaded first, TF's LLVM aborts on `Option 'info-output-file'
registered more than once`.  Symmetric, load-order dependent, fatal at dlopen,
and it reached every keras / gemma-lib / maxtext benchmark row.

`metal/exported_symbols.exp` is the fix.  This test is its contract, in both
orders, in a fresh subprocess each way -- a crash here is a SIGSEGV or an
abort(), so nothing but process isolation can report it.

Run it with a python that has TensorFlow AND the metaljax plugin:

    METALJAX_PLUGIN_PATH=.../libmetal_pjrt_native.dylib \\
      ~/.cache/metaljax-bench/venvs/bench/bin/python plugin-native/coexist_test.py
"""

import os
import subprocess
import sys
import textwrap

# Each case is (name, carrier import, order).  "carrier" is the other image
# with a static protobuf/LLVM in it; array_record is what maxtext pulls in.
CARRIERS = [
    ("tensorflow", "import tensorflow"),
    ("array_record", "from array_record.python import array_record_module"),
]

WORK = """
import jax, jax.numpy as jnp, numpy as np
devs = jax.devices()
assert devs and devs[0].platform == 'metal', devs
out = np.asarray(jax.jit(lambda x: x * 2 + 1)(jnp.arange(6, dtype=jnp.float32)))
assert np.array_equal(out, np.arange(6) * 2 + 1), out
print('   metal ok:', devs[0], out.tolist())
"""


def run(name, carrier, plugin_first):
    body = (
        (WORK + "\n" + carrier + "\nprint('   carrier ok (after)')\n")
        if plugin_first
        else (carrier + "\nprint('   carrier ok (before)')\n" + WORK)
    )
    src = textwrap.dedent(
        f"""
        import os
        os.environ.setdefault('JAX_PLATFORMS', 'metal')
        os.environ.setdefault('KERAS_BACKEND', 'jax')
        {textwrap.indent(body, '        ').lstrip()}
        print('OK')
        """
    )
    order = "plugin then " + name if plugin_first else name + " then plugin"
    print(f"--- {order}")
    p = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, timeout=900
    )
    ok = p.returncode == 0 and p.stdout.strip().endswith("OK")
    for line in p.stdout.splitlines():
        if line.startswith("   ") or line == "OK":
            print(line)
    if not ok:
        print(f"  returncode={p.returncode}")
        print(textwrap.indent("\n".join(p.stderr.splitlines()[-25:]), "  | "))
    print(f"{'PASS' if ok else 'FAIL'}: {order}\n")
    return ok


def main():
    plugin = os.environ.get("METALJAX_PLUGIN_PATH", "(default: Stage 1)")
    print(f"plugin: {plugin}")
    print(f"python: {sys.executable}\n")

    ok = True
    for name, carrier in CARRIERS:
        probe = subprocess.run(
            [sys.executable, "-c", carrier], capture_output=True, text=True
        )
        if probe.returncode != 0:
            print(f"--- {name}: not installed in this interpreter, skipped\n")
            continue
        ok &= run(name, carrier, plugin_first=False)
        ok &= run(name, carrier, plugin_first=True)

    print("all coexistence checks passed" if ok else "COEXISTENCE FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
