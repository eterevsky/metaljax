"""Release gate step 2 — diff a pinned-suite run against the approved whitelist.

    .venv/bin/python scripts/release/jax_suite_diff.py <run-outdir> \
        [--whitelist notes/data/pinned-0.11.0-failures.txt] \
        [--driver-log LOG] [--driver-rc N] [--seconds S] \
        [--md OUT.md] [--json OUT.json] [--smoke]

<run-outdir> is what scripts/run_jax_tests.py wrote: one <file>.log per test
file plus summary.csv.  Failing test ids are re-derived from the per-file
logs ("^FAILED " lines) rather than from the driver's failures.txt, because
failures.txt also folds in collection/setup "^ERROR " nodes while the
whitelist was built from FAILED lines only (the 22 environment-import files
— Pallas/Mosaic CUDA+TPU deps, optional `hypothesis` — are out of scope and
error identically on CPU).

Gate: NEW failures (in the run, not in the whitelist) fail the gate.  FIXED
tests (in the whitelist, not in the run) are informational — they are what
gets removed from the whitelist when Oleg approves the release.

Only files that actually ran are compared: whitelist entries belonging to
files absent from the run (e.g. a --filter'ed smoke run) are reported as
"not run", never as fixed.

Exit: 0 clean, 1 gate fail, 2 harness problem (missing/empty run).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WHITELIST = ROOT / "notes" / "data" / "pinned-0.11.0-failures.txt"
TESTS_PREFIX = "jax-v0.11.0/tests"

COUNT_RE = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)")


def norm_id(tid):
    """Normalize a pytest node id to the whitelist's relative form."""
    tid = tid.strip()
    i = tid.find(TESTS_PREFIX)
    if i > 0:
        tid = tid[i:]
    return tid


def load_whitelist(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        # Trailing annotations are legal ("<id>  # TRACKED-OPEN"): strip
        # them, or an annotated entry reads as fixed AND its failure as new.
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        ids.append(norm_id(line))
    return set(ids)


def parse_run(outdir):
    """-> (failures, errors, per_file rows, files_run)."""
    failures, errors, files_run = set(), set(), set()
    logs = sorted(Path(outdir).glob("*.log"))
    for log in logs:
        files_run.add(f"{TESTS_PREFIX}/{log.stem}.py")
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("FAILED "):
                failures.add(norm_id(line[7:].split(" - ")[0]))
            elif line.startswith("ERROR "):
                errors.add(norm_id(line[6:].split(" - ")[0]))
    rows = []
    csv_path = Path(outdir) / "summary.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                for k in ("passed", "failed", "skipped", "errors", "rc"):
                    r[k] = int(r[k])
                r["seconds"] = float(r["seconds"])
                rows.append(r)
    return failures, errors, rows, files_run


def by_file(ids):
    out = {}
    for i in sorted(ids):
        out.setdefault(i.split("::")[0], []).append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--whitelist", default=str(DEFAULT_WHITELIST))
    ap.add_argument("--driver-log", default=None)
    ap.add_argument("--driver-rc", type=int, default=0)
    ap.add_argument("--seconds", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoke", action="store_true")
    ns = ap.parse_args()

    outdir = Path(ns.outdir)
    if not outdir.is_dir():
        print(f"HARNESS ERROR: no run directory {outdir}")
        return 2

    whitelist = load_whitelist(ns.whitelist)
    failures, errors, rows, files_run = parse_run(outdir)
    if not files_run:
        print(f"HARNESS ERROR: no per-file logs under {outdir}")
        return 2

    # A file that timed out or died on a signal did not report its failures:
    # that is exactly the under-reporting the sequential run exists to avoid.
    crashed = [r for r in rows if r["rc"] < 0]
    crashed_files = {f"{TESTS_PREFIX}/{r['file']}" for r in crashed}

    scoped_wl = {i for i in whitelist if i.split("::")[0] in files_run}
    not_run = whitelist - scoped_wl
    new_fail = sorted(failures - scoped_wl)
    # never call a test "fixed" because the file it lives in never finished
    fixed = sorted(i for i in scoped_wl - failures
                   if i.split("::")[0] not in crashed_files)
    still = failures & scoped_wl

    tot = {k: sum(r[k] for r in rows)
           for k in ("passed", "failed", "skipped", "errors")} if rows else {}
    executed = tot.get("passed", 0) + tot.get("failed", 0)
    pct = 100.0 * tot.get("passed", 0) / max(executed, 1)

    status = "PASS"
    if new_fail or crashed:
        status = "FAIL"

    L = []
    L.append(f"### Step 2 — JAX pinned suite (`{TESTS_PREFIX}`): **{status}**")
    L.append("")
    if ns.smoke:
        L.append("> SMOKE RUN — a filtered subset, not a release-valid gate.")
        L.append("")
    files_line = f"{len(files_run)} test files"
    if ns.seconds:
        files_line += f", {float(ns.seconds) / 60:.1f} min"
    L.append(f"- run: `--jobs 1 --tests {TESTS_PREFIX}` ({files_line}), "
             f"driver rc={ns.driver_rc}")
    if rows:
        L.append(f"- totals: **{tot['passed']:,} passed / {tot['failed']:,} failed** "
                 f"/ {tot['skipped']:,} skipped / {tot['errors']} collection-errors "
                 f"→ **{pct:.2f}%**")
    L.append(f"- whitelist: {len(whitelist)} known failures "
             f"({len(scoped_wl)} in scope, {len(not_run)} in files not run)")
    L.append(f"- **NEW failures: {len(new_fail)}** · fixed since whitelist: "
             f"{len(fixed)} · still failing: {len(still)}")
    L.append(f"- collection/setup ERROR nodes: {len(errors)} "
             f"(environment imports — Pallas/Mosaic CUDA+TPU, optional "
             f"`hypothesis`; identical on CPU-only, out of scope)")
    L.append("")

    if crashed:
        L.append(f"**{len(crashed)} file(s) timed out / died — results incomplete:**")
        L.append("")
        for r in crashed:
            L.append(f"- `{r['file']}` rc={r['rc']} after {r['seconds']:.0f}s")
        L.append("")

    if new_fail:
        L.append("**NEW failures (gate-fail — discuss and whitelist case by case):**")
        L.append("")
        for f, ids in by_file(new_fail).items():
            L.append(f"- `{f}` ({len(ids)})")
            for i in ids[:20]:
                L.append(f"  - `{i.split('::', 1)[1]}`")
            if len(ids) > 20:
                L.append(f"  - … {len(ids) - 20} more")
        L.append("")
    if fixed:
        L.append(f"<details><summary>Fixed since the whitelist ({len(fixed)}) — "
                 f"remove from notes/data/pinned-0.11.0-failures.txt on approval"
                 f"</summary>")
        L.append("")
        for f, ids in by_file(fixed).items():
            L.append(f"- `{f}` ({len(ids)}): "
                     + ", ".join(i.split("::", 1)[1] for i in ids[:8])
                     + (" …" if len(ids) > 8 else ""))
        L.append("")
        L.append("</details>")
        L.append("")

    if rows:
        worst = sorted(rows, key=lambda r: -r["failed"])[:8]
        worst = [r for r in worst if r["failed"]]
        if worst:
            L.append("| file | pass | fail | skip | s |")
            L.append("|---|---:|---:|---:|---:|")
            for r in worst:
                L.append(f"| `{r['file']}` | {r['passed']} | {r['failed']} | "
                         f"{r['skipped']} | {r['seconds']:.0f} |")
            L.append("")

    L.append(f"Raw: `{outdir}` (per-file logs, summary.csv, failures.txt)"
             + (f", driver log `{ns.driver_log}`" if ns.driver_log else ""))
    md = "\n".join(L)
    print(md)

    if ns.md:
        Path(ns.md).write_text(md + "\n")
    if ns.json:
        Path(ns.json).write_text(json.dumps({
            "step": "jax_suite",
            "title": "Step 2 — JAX pinned suite",
            "status": status,
            "smoke": ns.smoke,
            "seconds": float(ns.seconds) if ns.seconds else None,
            "headline": (f"{tot.get('passed', 0)} passed / {tot.get('failed', 0)} "
                         f"failed ({pct:.2f}%), {len(new_fail)} new, "
                         f"{len(fixed)} fixed"),
            "totals": tot,
            "new_failures": new_fail,
            "fixed": fixed,
            "still_failing": len(still),
            "collection_errors": len(errors),
            "crashed_files": [r["file"] for r in crashed],
            "files_run": len(files_run),
            "whitelist": ns.whitelist,
            "markdown": md,
            "logs": {"run_dir": str(outdir), "driver_log": ns.driver_log},
        }, indent=1) + "\n")

    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
