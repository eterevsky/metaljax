"""Release gate step 3 — classify the texmo correctness + perf run.

Consumes the three logs produced by scripts/release/texmo_gate.sh and the
two topconfs JSONLs, applies the gate policy, writes a markdown block and a
machine-readable record.

Policy
------
correctness (scripts/texmo_check.py):
  * any FAIL line                                   -> gate fail
  * ERROR on a known-environmental config           -> warning
    (spec matches TEXMO_KNOWN_ENV_SPEC *and* the message looks like a
     missing tokenset.  This covered the 8 tokens.32.raw_fold.* configs
     whose tokenset was absent from ~/texmo/tokens; commit 9ef5f58 taught
     texmo_check.py to remap the renamed tokenset (.raw_fold -> .fold), so
     the expected baseline is now 104 ok / 0 errors.  The classifier stays
     as a safety net — it matches either spelling.)
  * any other ERROR                                 -> gate fail
performance (scripts/texmo_topconfs*.py):
  * any check FAIL / error in the sweep             -> gate fail
  * geomean(anchor/new metal ms/step) < 1 - TEXMO_GEOMEAN_TOL (default .03)
                                                    -> gate fail
  * any single config slower than TEXMO_CONFIG_TOL x (default 1.3)
                                                    -> gate fail

Exit: 0 pass, 1 gate fail, 2 harness error.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

KNOWN_ENV_SPEC = os.environ.get("TEXMO_KNOWN_ENV_SPEC",
                                r"tokens\.\d+\.(raw_)?fold")
KNOWN_ENV_MSG = os.environ.get(
    "TEXMO_KNOWN_ENV_MSG",
    r"(FileNotFoundError|No such file|not found|Unknown token|tokenset|"
    r"tokens32|KeyError|Errno 2)")
GEOMEAN_TOL = float(os.environ.get("TEXMO_GEOMEAN_TOL", "0.03"))
CONFIG_TOL = float(os.environ.get("TEXMO_CONFIG_TOL", "1.3"))

LINE_RE = re.compile(r"^(ok~?|FAIL|ERROR)\s+(\S+)\s+(.*)$")
CHECK_SUM_RE = re.compile(r"SUMMARY:\s+(\d+) ok,\s+(\d+) FAIL,\s+(\d+) error")


def _load_comparator():
    """Reuse the tracked comparator's own load()/geomean() (interface parity)."""
    path = ROOT / "scripts" / "texmo_topconfs_compare.py"
    spec = importlib.util.spec_from_file_location("texmo_topconfs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_check(path):
    rows = {"ok": [], "ok~": [], "FAIL": [], "ERROR": []}
    env_re = re.compile(KNOWN_ENV_SPEC)
    msg_re = re.compile(KNOWN_ENV_MSG)
    summary = None
    for line in Path(path).read_text(errors="replace").splitlines():
        m = CHECK_SUM_RE.search(line)
        if m:
            summary = tuple(int(x) for x in m.groups())
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        tag, name, rest = m.group(1), m.group(2), m.group(3)
        spec, msg = rest[:46].strip(), rest[46:].strip()
        rec = {"name": name, "spec": spec, "detail": msg}
        if tag == "ERROR":
            rec["known_env"] = bool(env_re.search(spec) and msg_re.search(msg))
        rows[tag].append(rec)
    return rows, summary


def parse_topconfs(path):
    txt = Path(path).read_text(errors="replace")
    m = re.search(r"SUMMARY:\s+(\d+) ok,\s+(\d+) FAIL,\s+(\d+) error", txt)
    fails = [l for l in txt.splitlines() if l.startswith("FAIL")]
    errs = [l for l in txt.splitlines() if l.startswith("ERROR")]
    return (tuple(int(x) for x in m.groups()) if m else None), fails, errs


def perf_gate(anchor, new):
    cmp_mod = _load_comparator()
    old, cur = cmp_mod.load(anchor), cmp_mod.load(new)
    keys = sorted(set(old) & set(cur), key=lambda k: old[k]["weights"])
    ratios = [(k, old[k]["metal_ms_step"] / cur[k]["metal_ms_step"])
              for k in keys
              if old[k].get("metal_ms_step") and cur[k].get("metal_ms_step")]
    cpu = [old[k]["cpu_ms_step"] / cur[k]["cpu_ms_step"] for k in keys
           if old[k].get("cpu_ms_step") and cur[k].get("cpu_ms_step")]
    gm = cmp_mod.geomean([r for _, r in ratios])
    gm_cpu = cmp_mod.geomean(cpu)
    worst = sorted(ratios, key=lambda kr: kr[1])[:8]
    best = sorted(ratios, key=lambda kr: -kr[1])[:5]
    over = [(k, r) for k, r in ratios if r < 1.0 / CONFIG_TOL]
    return {
        "matched": len(ratios), "only_anchor": len(old) - len(keys),
        "only_new": len(cur) - len(keys),
        "geomean": gm, "geomean_cpu": gm_cpu,
        "worst": [(list(k), r, old[k]["metal_ms_step"], cur[k]["metal_ms_step"])
                  for k, r in worst],
        "best": [(list(k), r, old[k]["metal_ms_step"], cur[k]["metal_ms_step"])
                 for k, r in best],
        "over_threshold": [(list(k), r, old[k]["metal_ms_step"],
                            cur[k]["metal_ms_step"]) for k, r in over],
        "improved_5pct": sum(1 for _, r in ratios if r > 1.05),
        "regressed_5pct": sum(1 for _, r in ratios if r < 0.95),
    }


def fmt_cfg(k):
    # spec strings contain '|' — escape it or the markdown table breaks
    spec = k[0][:46].replace("|", "\\|")
    return f"`{spec}` {k[1]} b{k[2]} l{k[3]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-log", required=True)
    ap.add_argument("--check-rc", type=int, default=0)
    ap.add_argument("--topconfs-log", required=True)
    ap.add_argument("--topconfs-rc", type=int, default=0)
    ap.add_argument("--compare-log", required=True)
    ap.add_argument("--compare-rc", type=int, default=0)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--seconds", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoke", action="store_true")
    ns = ap.parse_args()

    problems, warnings = [], []

    if not Path(ns.check_log).exists():
        print(f"HARNESS ERROR: missing {ns.check_log}")
        return 2
    check, check_sum = parse_check(ns.check_log)
    n_ok = len(check["ok"]) + len(check["ok~"])
    env_errs = [r for r in check["ERROR"] if r.get("known_env")]
    bad_errs = [r for r in check["ERROR"] if not r.get("known_env")]
    if check["FAIL"]:
        problems.append(f"texmo_check: {len(check['FAIL'])} FAIL")
    if bad_errs:
        problems.append(f"texmo_check: {len(bad_errs)} unclassified ERROR")
    if env_errs:
        warnings.append(f"texmo_check: {len(env_errs)} known-environmental "
                        f"ERROR (raw_fold tokensets missing)")
    if ns.check_rc != 0 and not (check["FAIL"] or check["ERROR"] or n_ok):
        problems.append(f"texmo_check driver rc={ns.check_rc}")

    top_sum, top_fails, top_errs = parse_topconfs(ns.topconfs_log)
    if top_fails:
        problems.append(f"texmo_topconfs: {len(top_fails)} check FAIL")
    if top_errs:
        problems.append(f"texmo_topconfs: {len(top_errs)} error")

    perf, perf_err = None, None
    try:
        perf = perf_gate(ns.anchor, ns.new)
    except Exception as e:  # missing/short jsonl, etc.
        perf_err = f"{type(e).__name__}: {e}"
        problems.append(f"perf comparison unavailable ({perf_err})")

    if perf and perf["matched"]:
        if perf["geomean"] < 1.0 - GEOMEAN_TOL:
            problems.append(
                f"geomean {perf['geomean']:.4f}x < {1 - GEOMEAN_TOL:.2f} "
                f"(>{GEOMEAN_TOL:.0%} regression vs anchor)")
        if perf["over_threshold"]:
            problems.append(f"{len(perf['over_threshold'])} config(s) more than "
                            f"{CONFIG_TOL}x slower than anchor")
    elif perf is not None and not perf["matched"]:
        problems.append("perf comparison matched 0 configs")

    status = "FAIL" if problems else ("WARN" if warnings else "PASS")

    L = []
    L.append(f"### Step 3 — texmo (correctness + perf): **{status}**")
    L.append("")
    if ns.smoke:
        L.append("> SMOKE RUN — a 2-3 config slice, not a release-valid gate.")
        L.append("")
    if ns.seconds:
        L.append(f"- wall: {float(ns.seconds) / 60:.1f} min")
    L.append(f"- **correctness** (`texmo_check.py`): {n_ok} ok "
             f"({len(check['ok~'])} via sensitivity scaling), "
             f"{len(check['FAIL'])} FAIL, {len(bad_errs)} unexpected error, "
             f"{len(env_errs)} known-environmental error"
             + (f" · script summary line: {check_sum[0]} ok / {check_sum[1]} "
                f"FAIL / {check_sum[2]} error" if check_sum else ""))
    if top_sum:
        L.append(f"- **perf sweep** (`texmo_topconfs.py`): {top_sum[0]} ok, "
                 f"{top_sum[1]} FAIL, {top_sum[2]} error → `{ns.new}`")
    if perf and perf["matched"]:
        L.append(f"- **geomean vs anchor** (`{Path(ns.anchor).name}`, old/new, "
                 f">1 = faster): **{perf['geomean']:.4f}x** over "
                 f"{perf['matched']} matched configs "
                 f"(CPU control {perf['geomean_cpu']:.4f}x); "
                 f"improved >5%: {perf['improved_5pct']}, "
                 f"regressed >5%: {perf['regressed_5pct']}")
        L.append(f"- thresholds: geomean ≥ {1 - GEOMEAN_TOL:.2f}x, "
                 f"no config > {CONFIG_TOL}x slower "
                 f"(`TEXMO_GEOMEAN_TOL` / `TEXMO_CONFIG_TOL`)")
    L.append("")

    if check["FAIL"] or bad_errs:
        L.append("**Correctness problems (gate-fail):**")
        L.append("")
        for r in check["FAIL"]:
            L.append(f"- FAIL `{r['name']}` `{r['spec']}` — {r['detail']}")
        for r in bad_errs:
            L.append(f"- ERROR `{r['name']}` `{r['spec']}` — {r['detail']}")
        L.append("")
    if env_errs:
        L.append(f"<details><summary>Known-environmental errors "
                 f"({len(env_errs)}) — missing tokensets in ~/texmo/tokens, "
                 f"not a metaljax defect</summary>")
        L.append("")
        for r in env_errs:
            L.append(f"- `{r['name']}` `{r['spec']}` — {r['detail']}")
        L.append("")
        L.append("</details>")
        L.append("")
    if top_fails or top_errs:
        L.append("**Perf-sweep correctness problems:**")
        L.append("")
        for l in (top_fails + top_errs)[:20]:
            L.append(f"- `{l.strip()[:160]}`")
        L.append("")

    if perf and perf["matched"]:
        if perf["over_threshold"]:
            L.append(f"**Configs beyond the {CONFIG_TOL}x single-config "
                     f"threshold:**")
            L.append("")
            for k, r, o, n in perf["over_threshold"]:
                L.append(f"- {1 / r:.2f}x slower — {fmt_cfg(k)} "
                         f"({o} → {n} ms/step)")
            L.append("")
        L.append("| | config | anchor ms | new ms | ratio |")
        L.append("|---|---|---:|---:|---:|")
        for k, r, o, n in perf["worst"]:
            L.append(f"| worst | {fmt_cfg(k)} | {o} | {n} | {r:.3f}x |")
        for k, r, o, n in perf["best"][:3]:
            L.append(f"| best | {fmt_cfg(k)} | {o} | {n} | {r:.3f}x |")
        L.append("")

    L.append(f"Raw: `{ns.check_log}`, `{ns.topconfs_log}`, `{ns.compare_log}`; "
             f"data `{ns.new}` (commit under notes/data/)")
    md = "\n".join(L)
    print(md)

    if ns.md:
        Path(ns.md).write_text(md + "\n")
    if ns.json:
        Path(ns.json).write_text(json.dumps({
            "step": "texmo",
            "title": "Step 3 — texmo correctness + perf",
            "status": status,
            "smoke": ns.smoke,
            "seconds": float(ns.seconds) if ns.seconds else None,
            "headline": (f"{n_ok} ok / {len(check['FAIL'])} FAIL / "
                         f"{len(bad_errs)} unexpected err "
                         f"({len(env_errs)} env), geomean "
                         + (f"{perf['geomean']:.4f}x" if perf and perf["matched"]
                            else "n/a")),
            "problems": problems,
            "warnings": warnings,
            "check": {"ok": n_ok, "fail": len(check["FAIL"]),
                      "env_errors": [r["name"] for r in env_errs],
                      "other_errors": [r["name"] for r in bad_errs],
                      "summary_line": check_sum},
            "topconfs": {"summary_line": top_sum,
                         "fails": top_fails[:20], "errors": top_errs[:20]},
            "perf": perf, "perf_error": perf_err,
            "anchor": ns.anchor, "new": ns.new,
            "markdown": md,
            "logs": {"check": ns.check_log, "topconfs": ns.topconfs_log,
                     "compare": ns.compare_log},
        }, indent=1) + "\n")

    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
