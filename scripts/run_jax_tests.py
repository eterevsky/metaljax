"""Run the upstream jax test suite against metaljax, one process per file.

    .venv/bin/python scripts/run_jax_tests.py <outdir> [--jobs N] [--filter SUBSTR]

Writes <outdir>/<file>.log per test file, <outdir>/summary.csv with
per-file counts, and <outdir>/failures.txt with every failing test id
(the inventory used to drive gap-closure work).

Each file runs in its own process with JAX_PLATFORMS=metal,cpu (metal
default, CPU available for comparisons), matching how the suite was run
for the 0.4.x campaigns.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "jax" / "tests"
PY = ROOT / ".venv" / "bin" / "python"

COUNT_RE = re.compile(
    r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)")


def run_file(path: Path, outdir: Path, timeout: int):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "metal,cpu"
    env.setdefault("JAX_ENABLE_X64", "0")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(PY), "-m", "pytest", str(path), "-q", "--tb=no", "-rf",
             "-p", "no:cacheprovider"],
            capture_output=True, text=True, env=env, timeout=timeout,
            cwd=str(ROOT))
        out = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        # TimeoutExpired carries bytes even under text=True
        def _txt(v):
            if v is None:
                return ""
            return v.decode(errors="replace") if isinstance(v, bytes) else v
        out = _txt(e.stdout) + _txt(e.stderr) + f"\n\nTIMEOUT {timeout}s\n"
        rc = -9
    dt = time.perf_counter() - t0
    (outdir / (path.stem + ".log")).write_text(out)

    counts = {}
    for line in out.splitlines()[::-1]:
        if " passed" in line or " failed" in line or " error" in line:
            found = COUNT_RE.findall(line)
            if found:
                for n, kind in found:
                    kind = "errors" if kind == "error" else kind
                    counts[kind] = int(n)
                break
    fails = [ln.split(" - ")[0].removeprefix("FAILED ").removeprefix("ERROR ").strip()
             for ln in out.splitlines()
             if ln.startswith("FAILED ") or ln.startswith("ERROR ")]
    return {
        "file": path.name,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "errors": counts.get("errors", 0),
        "rc": rc,
        "seconds": round(dt, 1),
    }, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--filter", default=None,
                    help="only files whose name contains this")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in TESTS.glob("*_test.py"))
    if args.filter:
        files = [p for p in files if args.filter in p.name]
    print(f"{len(files)} test files, {args.jobs} jobs -> {outdir}", flush=True)

    rows, all_fails = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_file, p, outdir, args.timeout): p
                for p in files}
        for fut in futs:
            pass
        for fut, path in list(futs.items()):
            row, fails = fut.result()
            rows.append(row)
            all_fails.extend(fails)
            done += 1
            print(f"[{done}/{len(files)}] {row['file']:44s} "
                  f"pass={row['passed']:5d} fail={row['failed']:4d} "
                  f"skip={row['skipped']:4d} err={row['errors']:3d} "
                  f"{row['seconds']:6.1f}s", flush=True)

    rows.sort(key=lambda r: -r["failed"])
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (outdir / "failures.txt").write_text("\n".join(sorted(all_fails)) + "\n")

    tot = {k: sum(r[k] for r in rows)
           for k in ("passed", "failed", "skipped", "errors")}
    total = tot["passed"] + tot["failed"]
    print(f"\nTOTAL passed={tot['passed']} failed={tot['failed']} "
          f"skipped={tot['skipped']} errors={tot['errors']} "
          f"-> {100.0 * tot['passed'] / max(total, 1):.2f}% pass")
    print(f"failing ids: {len(all_fails)} -> {outdir}/failures.txt")


if __name__ == "__main__":
    main()
