"""Release gate step 5 — consolidate every gate step into one report.

    .venv/bin/python scripts/release/summary.py \
        [--gate-dir ~/.cache/metaljax-bench/logs/release-gate/<date>] \
        [--out summary.md] [--smoke]

Reads the per-step records the wrappers wrote (jax_suite.json,
texmo_gate.json, model_gate.json — with a `-smoke` suffix in smoke mode) plus
steps.tsv (per-step wall time / exit status written by run_gates.sh) and
emits ONE markdown report: overall verdict, per-step PASS/FAIL with wall
time, new-failure lists, perf tables, links to every raw log.

Steps whose record is missing are reported as NOT RUN rather than silently
dropped.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = Path(os.environ.get(
    "RELEASE_GATE_DIR",
    Path.home() / ".cache" / "metaljax-bench" / "logs" / "release-gate"
    / os.environ.get("GATE_DATE", datetime.now().strftime("%Y-%m-%d"))))

STEPS = [("jax", "jax_suite", "Step 2 — JAX pinned suite"),
         ("texmo", "texmo_gate", "Step 3 — texmo correctness + perf"),
         ("models", "model_gate", "Step 4 — model suite")]

BADGE = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL",
         "ERROR": "💥 ERROR", "SKIP": "⏭ SKIP", "NOT RUN": "⚪ NOT RUN"}


def load_steps_tsv(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            out[parts[0]] = {"status": parts[1], "seconds": float(parts[2]),
                             "rc": int(parts[3])}
    return out


def hms(seconds):
    if seconds is None:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ns = ap.parse_args()

    gd = Path(ns.gate_dir).expanduser()
    suffix = "-smoke" if ns.smoke else ""
    tsv = load_steps_tsv(gd / "steps.tsv")

    records, total = [], 0.0
    for key, stem, title in STEPS:
        p = gd / f"{stem}{suffix}.json"
        rec = None
        if p.exists():
            try:
                rec = json.loads(p.read_text())
            except json.JSONDecodeError:
                rec = None
        t = tsv.get(key, {})
        tstat = t.get("status")
        # the step's own record is the evidence; steps.tsv only overrides it
        # when the wrapper died before writing one, or when the step was
        # skipped in this batch (then the record is from an earlier run).
        note = ""
        if rec is None:
            status = tstat or "NOT RUN"
        elif tstat == "ERROR":
            status = "ERROR"
        elif tstat == "SKIP":
            status = rec.get("status", "NOT RUN")
            note = " (not re-run in this batch — record from an earlier run)"
        else:
            status = rec.get("status") or tstat or "NOT RUN"
        secs = t.get("seconds") if tstat not in (None, "SKIP") \
            else (rec or {}).get("seconds")
        if secs:
            total += secs
        records.append({"key": key, "title": title, "record": rec,
                        "status": status, "seconds": secs, "note": note,
                        "rc": t.get("rc"), "path": p})

    worst = "PASS"
    for r in records:
        s = r["status"]
        if s in ("FAIL", "ERROR", "NOT RUN"):
            worst = "FAIL"
        elif s == "WARN" and worst == "PASS":
            worst = "WARN"

    try:
        import subprocess
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True
                              ).stdout.strip()
        dirty = len(subprocess.run(["git", "-C", str(ROOT), "status",
                                    "--porcelain"], capture_output=True,
                                   text=True).stdout.strip().splitlines())
    except Exception:
        head, dirty = "?", "?"
    version = "?"
    pj = (ROOT / "pyproject.toml").read_text().splitlines()
    for line in pj:
        if line.strip().startswith("version"):
            version = line.split("=", 1)[1].strip().strip('"')
            break

    L = []
    L.append(f"# metaljax release gates — {gd.name}")
    L.append("")
    if ns.smoke:
        L.append("> **SMOKE RUN** — filtered/skipped workloads. "
                 "NOT a release-valid gate.")
        L.append("")
    L.append(f"**Overall: {BADGE.get(worst, worst)}** · version `{version}` · "
             f"tree `{head}`{'' if dirty == 0 else f' (+{dirty} dirty files)'} · "
             f"total gate wall time **{hms(total)}**")
    L.append("")
    L.append("| step | status | wall | headline |")
    L.append("|---|---|---:|---|")
    for r in records:
        rec = r["record"] or {}
        L.append(f"| {r['title']} | {BADGE.get(r['status'], r['status'])} | "
                 f"{hms(r['seconds'])} | {rec.get('headline', '—')}"
                 f"{r['note']} |")
    L.append("")

    # actionable roll-up
    todo = []
    for r in records:
        rec = r["record"] or {}
        if r["status"] == "NOT RUN":
            todo.append(f"{r['title']}: no record at `{r['path']}` — did the "
                        f"step run?")
        for p in rec.get("problems", []):
            todo.append(f"{r['title']}: {p}")
        if rec.get("new_failures"):
            todo.append(f"{r['title']}: {len(rec['new_failures'])} NEW test "
                        f"failures vs the whitelist")
        if rec.get("crashed_files"):
            todo.append(f"{r['title']}: {len(rec['crashed_files'])} file(s) "
                        f"timed out — results incomplete")
    warns = []
    for r in records:
        for w in (r["record"] or {}).get("warnings", []):
            warns.append(f"{r['title']}: {w}")
    if todo:
        L.append("## Blocking")
        L.append("")
        for t in todo:
            L.append(f"- ❌ {t}")
        L.append("")
    if warns:
        L.append("## Non-blocking")
        L.append("")
        for w in warns:
            L.append(f"- ⚠️ {w}")
        L.append("")
    if not todo:
        L.append("No blocking findings — ready for step 5 sign-off "
                 "(then 5.5 wheel smoke, then TestPyPI on Oleg's approval).")
        L.append("")

    L.append("---")
    L.append("")
    for r in records:
        rec = r["record"]
        if not rec:
            L.append(f"### {r['title']}: {BADGE.get(r['status'], r['status'])}")
            L.append("")
            L.append(f"No record at `{r['path']}`.")
            L.append("")
            continue
        L.append(rec.get("markdown", "").rstrip())
        L.append("")
        L.append("")

    L.append("---")
    L.append("")
    L.append("## Raw artifacts")
    L.append("")
    L.append(f"- gate directory: `{gd}`")
    for r in records:
        for name, path in ((r["record"] or {}).get("logs") or {}).items():
            if path:
                L.append(f"- {r['key']} / {name}: `{path}`")
    L.append("")
    L.append("Per-step wall time (from `steps.tsv`): "
             + ", ".join(f"{r['key']} {hms(r['seconds'])}" for r in records))
    md = "\n".join(L)
    print(md)
    if ns.out:
        Path(ns.out).write_text(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
