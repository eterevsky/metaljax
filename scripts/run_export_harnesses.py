"""Multi-platform export harnesses for cpu+metal.

A fork of jax's export_harnesses_multi_platform_test idea with the
platform pair we actually care about: every primitive harness in
jax._src.internal_test_util.test_harnesses is exported once with
platforms=("cpu", "metal") and the exported artifact is executed on
BOTH devices, each compared against the natively-jitted function on
that device. (The upstream file hard-codes ("cpu","cuda","tpu") and
cannot exercise a plugin platform.)

    JAX_PLATFORMS=metal,cpu .venv/bin/python scripts/run_export_harnesses.py [outdir]

Writes <outdir>/results.csv and prints a category summary. Sequential:
do not run alongside other GPU work.
"""

import csv
import sys
import traceback
from pathlib import Path

import numpy as np

import jax
from jax import export

from jax._src.internal_test_util import test_harnesses

SKIP_DTYPES = ("float64", "complex128")  # intentional platform exclusions


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "export_harnesses_out")
    outdir.mkdir(parents=True, exist_ok=True)
    cpu = jax.devices("cpu")[0]
    metal = jax.devices("metal")[0]
    rng = np.random.RandomState(42)

    rows = []
    counts = {}

    def note(h, status, detail=""):
        counts[status] = counts.get(status, 0) + 1
        rows.append({"harness": h.fullname, "status": status,
                     "detail": detail[:300].replace("\n", " ")})

    harnesses = test_harnesses.all_harnesses
    print(f"{len(harnesses)} harnesses", flush=True)
    for i, h in enumerate(harnesses):
        if i % 250 == 0:
            print(f"[{i}/{len(harnesses)}] "
                  + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                  flush=True)
        dt = np.dtype(h.dtype).name if h.dtype is not None else ""
        if dt in SKIP_DTYPES:
            note(h, "skip-dtype", dt)
            continue
        try:
            args = h.dyn_args_maker(rng)
        except Exception as e:
            note(h, "skip-args", repr(e))
            continue
        try:
            exp = export.export(jax.jit(h.dyn_fun),
                                platforms=("cpu", "metal"))(*args)
        except Exception as e:
            note(h, "export-fail", repr(e))
            continue
        status, detail = "pass", ""
        for dev, tag in ((cpu, "cpu"), (metal, "metal")):
            try:
                dargs = jax.tree.map(lambda x: jax.device_put(x, dev), args)
                native = jax.tree.map(np.asarray, jax.jit(h.dyn_fun)(*dargs))
                got = jax.tree.map(np.asarray, exp.call(*dargs))
            except Exception as e:
                status, detail = f"{tag}-run-fail", repr(e)
                break
            for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(native)):
                g = np.asarray(g)
                w = np.asarray(w)
                if g.shape != w.shape:
                    status, detail = f"{tag}-mismatch", f"shape {g.shape} vs {w.shape}"
                    break
                if g.dtype != w.dtype:
                    status, detail = f"{tag}-mismatch", f"dtype {g.dtype} vs {w.dtype}"
                    break
                inexact = (np.issubdtype(g.dtype, np.inexact)
                           or g.dtype.kind == "V")  # ml_dtypes (bf16 etc.)
                if inexact:
                    gg = g.astype(np.float64) if g.dtype.kind == "V" else g
                    ww = w.astype(np.float64) if w.dtype.kind == "V" else w
                    if not np.allclose(gg, ww, rtol=1e-4, atol=1e-4,
                                       equal_nan=True):
                        finite = np.isfinite(ww) & np.isfinite(gg)
                        md = (np.abs(gg[finite] - ww[finite]).max()
                              if finite.any() else float("inf"))
                        status, detail = f"{tag}-mismatch", f"maxdiff {md:.3e}"
                        break
                elif not np.array_equal(g, w):
                    status, detail = f"{tag}-mismatch", "int mismatch"
                    break
            if status != "pass":
                break
        note(h, status, detail)

    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["harness", "status", "detail"])
        w.writeheader()
        w.writerows(rows)
    print("\nSUMMARY:")
    for k in sorted(counts):
        print(f"  {k:16s} {counts[k]}")
    total_run = sum(v for k, v in counts.items() if not k.startswith("skip"))
    print(f"  pass rate over run harnesses: "
          f"{100.0 * counts.get('pass', 0) / max(total_run, 1):.2f}%")


if __name__ == "__main__":
    main()
