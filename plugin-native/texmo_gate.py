#!/usr/bin/env python
"""The phase-2 gate: every texmo suite configuration through the native plugin.

For each configuration in `benchmarks/texmo-suite.csv` this builds the real
ManagerJax model, lowers ONE jitted training chunk (forward + backward +
optimizer, scanned over a few steps) to StableHLO, and then runs that module
BOTH ways: on the jax CPU backend, and through this plugin's
`compile_and_load` + `execute`.  Every output leaf -- updated weights,
optimizer state, per-step losses -- is compared with the retired
`scripts/texmo_check.py`'s tolerance discipline (inlined here when Stage 1
was retired), which scales the bar with the model's own measured 1-ULP
sensitivity (an ill-conditioned training step amplifies cross-backend rounding
identically on both sides, so a fixed tolerance would be meaningless).

    .venv/bin/python plugin-native/texmo_gate.py [suite.csv] [steps_per_chunk]

Options: `--limit N` (the first N configurations), `--only SUBSTR`, `--keep`
(leave the dumped artifacts in place), `--steps N`.

Three outcomes per configuration, and only one of them fails the gate:

* `ok` / `ok~` -- it computed, and matched CPU (`ok~` passed only through the
  sensitivity scaling);
* `decline` -- the plugin refused the program, naming the op it cannot lower.
  Coverage grows phase by phase, so a decline is a TODO, not a regression;
* `FAIL` / `ERROR` -- it computed a different answer, or the harness could not
  measure it.  Either is a nonzero exit.

Why it is shaped as two processes: the CPU reference must come from a process
that never loaded this plugin (jax picks its backend once, at first use).  A
child process per configuration builds the model and the CPU reference under
JAX_PLATFORMS=cpu, writes the module and the arrays to a scratch directory,
and this process -- the one holding the plugin -- executes and compares.  The
artifacts are deleted as they are consumed: a whole suite's weights, optimizer
state and references at once would be tens of gigabytes.
"""

import argparse
import csv
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
_DEFAULT_DYLIB = _HERE / "bazel-bin" / "metal" / "libmetal_pjrt_native.dylib"
_DEFAULT_CSV = _REPO / "benchmarks" / "texmo-suite.csv"

# "UNIMPLEMENTED: metaljax-native: op stablehlo.gather" -> what follows the
# prefix.  A message this does not recognize is reported whole: an unexpected
# failure mode is worth seeing, not truncating.
_REASON = re.compile(r"metaljax-native: ([^\n]*)")


def configs_from_csv(path):
    """The suite-CSV reader (inherited from the retired texmo_check.py)."""
    seen = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["mode"] != "scan" or row.get("error"):
                continue
            # texmo renamed the raw_fold tokenset to fold (2026-08); older
            # suite CSVs still carry the old spec strings.
            spec = row["spec"].replace(".raw_fold", ".fold")
            key = (spec, "fp32", int(row["batch"]), int(row["length"]))
            seen.setdefault(key, row["name"])
    return [(name, *key) for key, name in seen.items()]


def configs_from_jsonl(path):
    """A top-configurations JSONL (`spec, precision, batch, length, ...`),
    named exactly as `scripts/bench_texmo_pjrt.py` names them, so a gate row
    and a bench row with the same name are the same configuration."""
    import json

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            spec = row["spec"].replace(".raw_fold", ".fold")
            name = row.get("name") or f"tc{len(rows):03d}-w{row.get('weights', 0)}"
            rows.append((name, spec, row.get("precision", "fp32"),
                         int(row["batch"]), int(row["length"])))
    return rows


# --------------------------------------------------------------------------
# the child: build one configuration on the CPU and write it out
# --------------------------------------------------------------------------


def dump_one(out_dir, spec, batch, length, nsteps, precision="fp32"):
    """Module text, inputs, CPU reference and 1-ULP sensitivity, to disk."""
    # The suite dataset, exactly as the retired scripts/texmo_check.py built
    # it (inlined at the Stage-1 retirement): texmo itself on sys.path, the
    # shared tokens dir, pride.txt as the training set.  This child runs
    # under JAX_PLATFORMS=cpu (the parent sets it), which is what the old
    # texmo_check import-time platform pin used to achieve.
    texmo_dir = pathlib.Path(os.environ.get(
        "TEXMO_DIR", pathlib.Path.home() / "texmo"))
    sys.path.insert(0, str(texmo_dir))

    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from texmo.configuration import Configuration
    from texmo.dataset import DataSet
    from texmo.manager_jax import ManagerJax
    from texmo.precision import Precision
    from texmo.spec_parser import parse_model2
    from texmo.tokens import set_tokens_dir

    jax.config.update("jax_enable_x64", False)  # matches texmo.py
    set_tokens_dir(str(texmo_dir / "tokens"))
    train_set = DataSet(path=str(texmo_dir / "data" / "pride.txt"))

    conf = Configuration(
        parse_model2(spec, precision=Precision(precision)),
        lr=0.01, length=length, batch=batch, steps=nsteps,
        decay=1.0, cosine=False,
    )
    manager = ManagerJax(conf=conf, system="metaljax-gate",
                         dataset=train_set, test_sample_len=64,
                         test_batch=8)
    tokens_name = manager.model_def.input.tokens_name
    data = manager.dataset.sample_tokens(
        ntokens=length, batch=nsteps * batch, tokenset_name=tokens_name)
    batches = jnp.asarray(data).reshape(nsteps, batch, length)
    args = (manager.weights, manager._opt_state, batches)

    text = manager._train_chunk.lower(*args).as_text()
    flat_in = jax.tree_util.tree_leaves(args)
    ref = manager._train_chunk(*args)

    # Model sensitivity, exactly texmo_check's: +1 ULP on every weight
    # element, measured on the CPU alone.  Configurations whose training step
    # amplifies that to far beyond f32 noise are ill-conditioned, and a
    # cross-backend rounding difference blows up identically -- so the pass
    # tolerance scales with this measured amplification.
    leaves, treedef = jax.tree_util.tree_flatten(args[0])
    pert = []
    for x in leaves:
        a = np.asarray(x).copy()
        if a.dtype == ml_dtypes.bfloat16 and a.size:
            # +1 ULP in the array's OWN precision.  A bf16 numpy array has
            # kind 'V' and np.nextafter refuses it, so bump the bit pattern:
            # toward +inf is +1 on nonnegative patterns, -1 on negative ones.
            f32 = a.astype(np.float32).reshape(-1)
            u = a.view(np.uint16).reshape(-1)
            fin = np.isfinite(f32)
            u[fin & (f32 >= 0)] += 1
            u[fin & (f32 < 0)] -= 1
        elif a.dtype.kind == "f" and a.size:
            f = a.reshape(-1)
            f[:] = np.nextafter(f, np.float32(1e9))
        pert.append(jnp.asarray(a))
    ref_p = manager._train_chunk(
        jax.tree_util.tree_unflatten(treedef, pert), *args[1:])
    sens = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(ref),
                    jax.tree_util.tree_leaves(ref_p)):
        a = np.asarray(np.asarray(a).astype(np.float64)
                       if np.asarray(a).dtype == ml_dtypes.bfloat16
                       else a, dtype=np.float64)
        b = np.asarray(np.asarray(b).astype(np.float64)
                       if np.asarray(b).dtype == ml_dtypes.bfloat16
                       else b, dtype=np.float64)
        if a.dtype.kind != "f" or not a.size:
            continue
        d = float(np.abs(a - b).max())
        sc = max(float(np.abs(a).max()), 1e-3)
        sens = max(sens, d / sc)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mod.mlir").write_text(text)
    saved = {"sens": np.asarray(sens),
             "n_in": np.asarray(len(flat_in))}
    bf16_in = []
    for i, x in enumerate(flat_in):
        # NOT np.ascontiguousarray: it promotes a rank-0 array to rank 1, and
        # a training chunk's arguments include scalars (the step counter, a
        # scalar weight), which would then not match the module's tensor<f32>.
        a = np.asarray(x)
        if a.dtype == ml_dtypes.bfloat16:
            # np.savez cannot serialize the extension dtype; ship the bit
            # pattern and the parent views it back.
            bf16_in.append(i)
            a = a.view(np.uint16)
        saved[f"i{i}"] = a
    saved["bf16_in"] = np.asarray(bf16_in, dtype=np.int64)
    flat_ref = []
    for x in jax.tree_util.tree_leaves(ref):
        a = np.asarray(x)
        # References in f32 always: bf16 outputs are compared upconverted,
        # and the tolerance (not the storage) is what carries the precision.
        if a.dtype == ml_dtypes.bfloat16 or a.dtype.kind == "f":
            a = a.astype(np.float32)
        flat_ref.append(a)
    saved["n_ref"] = np.asarray(len(flat_ref))
    for i, x in enumerate(flat_ref):
        saved[f"r{i}"] = x
    np.savez(out_dir / "data.npz", **saved)
    return 0


# --------------------------------------------------------------------------
# the parent: execute through the plugin and compare
# --------------------------------------------------------------------------


def compare(outs, refs, tol, sens):
    """texmo_check's comparison: worst scale-normalized error over outputs."""
    worst, worst_i, n_bad = 0.0, -1, 0
    if len(outs) != len(refs):
        return None, f"{len(outs)} outputs, the CPU backend gave {len(refs)}"
    for i, (o, r) in enumerate(zip(outs, refs)):
        h = np.asarray(o)
        if h.dtype != r.dtype:
            h = h.astype(r.dtype)
        if h.shape != r.shape:
            return None, f"shape mismatch out[{i}]: {h.shape} vs {r.shape}"
        if r.dtype.kind == "f":
            if (np.isfinite(r) != np.isfinite(h)).any():
                n_bad += 1
                worst, worst_i = float("inf"), i
                continue
            fin = np.isfinite(r)
            d = np.abs(h - r)[fin]
            if not d.size:
                continue
            scale = max(float(np.abs(r[fin]).max()), 1e-3)
            norm = float(d.max()) / scale
            if norm > worst:
                worst, worst_i = norm, i
        else:
            if not np.array_equal(h, r):
                n_bad += 1
                worst, worst_i = float("inf"), i
    eff = max(tol, sens * 500)
    ok = worst <= eff and n_bad == 0
    return (ok, worst, worst_i, len(refs), n_bad, eff), None


def run_one(work, tol):
    """Compile and execute one dumped configuration; returns (tag, detail)."""
    import jax
    import ml_dtypes
    from jax._src.lib import xla_client as xc

    data = np.load(work / "data.npz")
    text = (work / "mod.mlir").read_text()
    bf16_in = set(data["bf16_in"].tolist()) if "bf16_in" in data else set()
    flat_in = [data[f"i{i}"].view(ml_dtypes.bfloat16) if i in bf16_in
               else data[f"i{i}"]
               for i in range(int(data["n_in"]))]
    refs = [data[f"r{i}"] for i in range(int(data["n_ref"]))]
    sens = float(data["sens"])

    dev = jax.devices()[0]
    try:
        exe = dev.client.compile_and_load(text, [dev], xc.CompileOptions())
    except BaseException as exc:                       # noqa: BLE001
        msg = str(exc)
        hit = _REASON.search(msg)
        if "UNIMPLEMENTED" in msg and hit:
            return "decline", hit.group(1).strip().rstrip(".")
        return "ERROR", "compile: " + " ".join(msg.split())[:110]
    outs = exe.execute([jax.device_put(a, dev) for a in flat_in])
    got = [np.asarray(o) for o in outs]
    res, err = compare(got, refs, tol, sens)
    if err is not None:
        return "FAIL", err
    ok, worst, wi, nout, nbad, eff = res
    detail = (f"worst={worst:.2e} sens={sens:.1e} tol={eff:.1e} "
              f"out[{wi}]/{nout} bad={nbad}")
    if not ok:
        return "FAIL", detail
    return ("ok~" if worst > tol else "ok"), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(_DEFAULT_CSV))
    ap.add_argument("steps", nargs="?", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--keep", action="store_true")
    # The child half, one configuration at a time.
    ap.add_argument("--dump", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--length", type=int, default=0)
    ap.add_argument("--precision", default="fp32")
    args = ap.parse_args()

    if args.dump:
        return dump_one(args.dump, args.spec, args.batch, args.length,
                        args.steps, args.precision)

    os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
    dylib = pathlib.Path(os.environ["METALJAX_PLUGIN_PATH"])
    if not dylib.exists():
        sys.exit(f"plugin dylib not found: {dylib}")
    os.environ["JAX_PLATFORMS"] = "metal"
    # Base tolerance by precision (CHECK_TOL overrides both): fp32 keeps the
    # suite's 2e-3; bf16 carries ~3 significant digits, so cross-backend
    # rounding legitimately lands around 1e-2 (texmo_check's bf16 bar).
    tol_env = os.environ.get("CHECK_TOL", "")
    tol_of = lambda precision: (float(tol_env) if tol_env
                                else (2e-2 if precision == "bf16" else 2e-3))
    print(f"plugin: {dylib} ({dylib.stat().st_size / 1e6:.1f} MB)")

    if args.csv.endswith(".jsonl"):
        configs = configs_from_jsonl(args.csv)
    else:
        configs = sorted(configs_from_csv(args.csv))
    if args.only:
        configs = [c for c in configs if args.only in c[0] or args.only in c[1]]
    if args.limit:
        configs = configs[:args.limit]
    print(f"{len(configs)} configurations, {args.steps} steps per chunk\n")

    # Imported here rather than in `run_one`, and after JAX_PLATFORMS is
    # fixed: a plugin that cannot load should say so before the first child
    # spends half a minute building a model for it.
    import jax  # noqa: F401

    root = pathlib.Path(tempfile.mkdtemp(prefix="texmo-gate-"))
    counts = {"ok": 0, "ok~": 0, "decline": 0, "FAIL": 0, "ERROR": 0}
    declines, failures = {}, []
    print(f"{'':<6} {'config':<18} {'spec':<46} detail")
    for name, spec, precision, batch, length in configs:
        t0 = time.perf_counter()
        work = root / name
        child = dict(os.environ)
        child["JAX_PLATFORMS"] = "cpu"
        child.pop("METALJAX_PLUGIN_PATH", None)
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--dump", str(work), "--spec", spec, "--batch", str(batch),
             "--length", str(length), "--precision", precision,
             args.csv, str(args.steps)],
            env=child, capture_output=True, text=True)
        if proc.returncode != 0:
            tag, detail = "ERROR", ("build: " + " ".join(
                (proc.stderr or proc.stdout).split())[-110:])
        else:
            try:
                tag, detail = run_one(work, tol_of(precision))
            except BaseException as exc:                # noqa: BLE001
                tag, detail = "ERROR", (f"{type(exc).__name__}: "
                                        f"{str(exc).splitlines()[0][:100]}")
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        counts[tag] = counts.get(tag, 0) + 1
        if tag == "decline":
            declines.setdefault(detail, []).append(name)
        elif tag in ("FAIL", "ERROR"):
            failures.append(name)
        dt = time.perf_counter() - t0
        print(f"{tag:<6} {name:<18} {spec[:44]:<46} {detail} {dt:.0f}s",
              flush=True)
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)

    total = sum(counts.values())
    print(f"\nSUMMARY: {counts['ok'] + counts['ok~']} ok "
          f"({counts['ok~']} via sensitivity scaling), "
          f"{counts['decline']} decline, {counts['FAIL']} FAIL, "
          f"{counts['ERROR']} error, of {total}")
    if declines:
        print("\ndeclines by reason")
        print("-" * 70)
        for why, names in sorted(declines.items(), key=lambda kv: -len(kv[1])):
            print(f"{len(names):>4}  {why}")
            print(f"      {', '.join(names[:6])}"
                  f"{' ...' if len(names) > 6 else ''}")
    if failures:
        print(f"\n{len(failures)} did not compute correctly: "
              f"{', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
