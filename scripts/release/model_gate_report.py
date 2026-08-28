"""Release gate step 4 — audit a model-suite run.

    .venv/bin/python scripts/release/model_gate_report.py \
        --raw ~/.cache/metaljax-bench/logs/final_run.jsonl \
        --merged OUT.jsonl --tokens-log OUT.log \
        --ledger models.md [--md OUT.md] [--json OUT.json]

Three jobs:

1. MERGE.  scripts/model_bench/final_run.sh writes the ordinary rows to
   final_run.jsonl but appends the maxtext rows to a separate
   final_run.jsonl.maxtext as raw adapter stdout: a tag line
   `[maxtext] <id> / <backend> -> '<text>'` (or `[maxtext-train] <id> /
   <backend> step_ms=... loss ...`) followed by `RESULT {json}` — the JSON
   carries neither id nor backend.  We pair each RESULT with the preceding
   tag and emit records shaped like final_run.jsonl (id/backend/ok/...).

2. TOKEN AUDIT.  scripts/model_bench/compare_tokens.py over the merged
   file: bf16 LLM rows must match CPU token-for-token; rows in its
   QUANT_ROWS policy set (int4/int8) are informational.

3. TIMING TABLE.  Compares each benchmark against the newest column of
   models.md (transposed ledger: benchmark rows, run columns).

Exit: 0 pass, 1 gate fail, 2 harness error.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REGRESS_TOL = float(os.environ.get("MODEL_REGRESS_TOL", "0.10"))
REGRESS_FAIL = os.environ.get("MODEL_REGRESS_FAIL", "1") == "1"

# Benchmark ids whose greedy stream is ALLOWED to diverge from CPU:
# certified-benign tie-flips, competing logits within 1-2 bf16 ULPs
# (STATUS.md footnote 21 / notes/int8-divergence-verdict.md).  A divergence
# on any OTHER bf16 row fails the gate.  compare_tokens.py itself only
# exempts its QUANT_ROWS, so its exit code is advisory here; the
# authoritative classification is this list.
TOKEN_KNOWN = set(filter(None, os.environ.get(
    "MODEL_TOKEN_KNOWN", "gemma4-e2b-bf16,llama31-8b-bf16").split(",")))

# models.md row number -> (bench id in final_run.sh, metric field).
# Rows absent from this map are not produced by final_run.sh (blocked cells:
# 8, 9, 10, 12, 15, 17, 20) and are reported as "not in this run".
# UPDATE THIS MAP whenever final_run.sh's row list or models.md changes.
ROW_BENCH = {
    1:  ("gemma4-31b-bf16", "decode_ms_tok"),
    2:  ("gemma4-12b-bf16", "decode_ms_tok"),
    3:  ("gemma4-26b-a4b", "decode_ms_tok"),
    4:  ("gemma4-e2b-bf16", "decode_ms_tok"),
    5:  ("qwen3-8b-bf16", "decode_ms_tok"),
    6:  ("llama31-8b-bf16", "decode_ms_tok"),
    7:  ("gpt-oss-20b", "decode_ms_tok"),
    11: ("qwen3-06b-maxtext", "decode_ms_tok"),
    13: ("gemma4-e2b-int4", "decode_ms_tok"),
    14: ("maxtext-qwix-int8", "decode_ms_tok"),
    16: ("siglip2-so400m", "step_ms"),
    18: ("lora-gemma4-e2b", "step_ms"),
    19: ("maxtext-train-06b", "step_ms"),
}

TAG_RE = re.compile(r"^\[maxtext(?:-train)?\]\s+(\S+)\s+/\s+(\S+)")
SUPERSCRIPTS = "¹²³⁴⁵⁶⁷⁸⁹⁰˟*"


def merge_maxtext(raw, merged):
    """final_run.jsonl + normalized <raw>.maxtext RESULT lines -> merged."""
    recs, n_base, n_mx = [], 0, 0
    if Path(raw).exists():
        for line in Path(raw).read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
                n_base += 1
            except json.JSONDecodeError:
                pass
    side = Path(str(raw) + ".maxtext")
    if side.exists():
        tag = None
        mx_by_key = {}
        for line in side.read_text(errors="replace").splitlines():
            m = TAG_RE.match(line.strip())
            if m:
                tag = (m.group(1), m.group(2))
                continue
            if line.startswith("RESULT ") and tag:
                try:
                    r = json.loads(line[len("RESULT "):])
                except json.JSONDecodeError:
                    continue
                r.update(id=tag[0], backend=tag[1], ok=True, source="maxtext")
                # keep the merged file readable: the echoed sample text is
                # long and is already in the raw side file
                r.pop("text", None)
                # the side file is APPENDED to across runs — last wins
                mx_by_key[(tag[0], tag[1])] = r
                n_mx += 1
                tag = None
        recs.extend(mx_by_key.values())
        n_mx = len(mx_by_key)
    Path(merged).write_text("".join(json.dumps(r) + "\n" for r in recs))
    return recs, n_base, n_mx


def parse_ledger(path):
    """-> (column label, {row number: (label, value or None)})."""
    lines = Path(path).read_text().splitlines()
    tbl = [l for l in lines if l.strip().startswith("|")]
    if len(tbl) < 3:
        return None, {}
    header = [c.strip() for c in tbl[0].strip().strip("|").split("|")]
    col = len(header) - 1
    label = header[col]
    rows = {}
    for l in tbl[2:]:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) <= col:
            continue
        try:
            num = int(cells[0])
        except ValueError:
            continue
        rows[num] = (cells[1], clean_cell(cells[col]))
    return label, rows


def clean_cell(cell):
    s = cell.replace("**", "").strip()
    s = s.strip(SUPERSCRIPTS).strip()
    if not s or s.startswith("✗") or s in {"—", "-", "n/a"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--tokens-log", required=True)
    ap.add_argument("--run-log", default=None)
    ap.add_argument("--run-rc", type=int, default=0)
    ap.add_argument("--ledger", default=str(ROOT / "benchmarks" / "models.md"))
    ap.add_argument("--seconds", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoke", action="store_true")
    ns = ap.parse_args()

    if not Path(ns.raw).exists():
        print(f"HARNESS ERROR: no {ns.raw} — did final_run.sh run?")
        return 2

    recs, n_base, n_mx = merge_maxtext(ns.raw, ns.merged)
    by_key = {}
    for r in recs:
        if "id" in r and "backend" in r:
            by_key[(r["id"], r["backend"])] = r
    failed_cells = [r for r in recs if r.get("ok") is False]

    # ---- token agreement
    tok = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "model_bench" / "compare_tokens.py"),
         ns.merged], capture_output=True, text=True)
    tok_out = tok.stdout + tok.stderr
    Path(ns.tokens_log).write_text(tok_out)
    tok_div = re.findall(r"^\s*(\S+)\s+FAIL: diverges at token (\d+)/(\d+)",
                         tok_out, re.M)
    tok_new = [(i, a, b) for i, a, b in tok_div if i not in TOKEN_KNOWN]
    tok_known = [(i, a, b) for i, a, b in tok_div if i in TOKEN_KNOWN]
    tok_fail = bool(tok_new)

    # ---- timing vs ledger
    col_label, ledger = parse_ledger(ns.ledger)
    table, regressed, newly_failing, newly_passing = [], [], [], []
    for num in sorted(ledger):
        name, prev = ledger[num]
        bench = ROW_BENCH.get(num)
        if not bench:
            table.append((num, name, prev, None, None, "not in final_run.sh"))
            continue
        bid, field = bench
        rec = by_key.get((bid, "metaljax"))
        val = rec.get(field) if rec and rec.get("ok") else None
        if val is None:
            note = "MISSING" if rec is None else (
                f"FAILED: {str(rec.get('error'))[:70]}" if rec.get("ok") is False
                else f"no {field}")
            if prev is not None:
                newly_failing.append((num, name, note))
            table.append((num, name, prev, None, None, note))
            continue
        val = float(val)
        if prev is None:
            newly_passing.append((num, name, val))
            table.append((num, name, None, val, None, "NEW (was blocked)"))
            continue
        delta = (val - prev) / prev
        note = ""
        if delta > REGRESS_TOL:
            note = "REGRESSED"
            regressed.append((num, name, prev, val, delta))
        elif delta < -REGRESS_TOL:
            note = "improved"
        table.append((num, name, prev, val, delta, note))

    problems, warns = [], []
    if tok_new:
        problems.append("token agreement: unexpected bf16 divergence on "
                        + ", ".join(i for i, _, _ in tok_new))
    if tok_known:
        warns.append("token agreement: certified-benign divergence on "
                     + ", ".join(f"{i} (token {a}/{b})" for i, a, b in tok_known))
    if newly_failing:
        problems.append(f"{len(newly_failing)} row(s) newly failing/missing")
    if regressed:
        (problems if REGRESS_FAIL else warns).append(
            f"{len(regressed)} row(s) regressed > {REGRESS_TOL:.0%}")
    if failed_cells:
        warns.append(f"{len(failed_cells)} cell(s) reported ok=false")
    if ns.run_rc != 0:
        warns.append(f"final_run.sh rc={ns.run_rc}")
    status = "FAIL" if problems else ("WARN" if warns else "PASS")

    L = []
    L.append(f"### Step 4 — model suite: **{status}**")
    L.append("")
    if ns.smoke:
        L.append("> SMOKE RUN — `final_run.sh` was NOT executed; this audits "
                 "the previously recorded raw data.")
        L.append("")
    if ns.seconds:
        L.append(f"- wall: {float(ns.seconds) / 60:.1f} min")
    L.append(f"- merged records: {n_base} from `{Path(ns.raw).name}` + "
             f"{n_mx} maxtext RESULT rows → `{ns.merged}`")
    L.append(f"- token agreement (`compare_tokens.py`): "
             f"**{'FAIL' if tok_fail else 'PASS'}** — "
             f"{len(tok_div)} divergent row(s), {len(tok_known)} certified-benign "
             f"(`MODEL_TOKEN_KNOWN`), {len(tok_new)} unexpected"
             + (f"; script verdict: "
                f"{tok_out.strip().splitlines()[-1].strip()}"
                if tok_out.strip() else ""))
    L.append(f"- timing vs ledger column **{col_label.replace('*', '')}** "
             f"(`{Path(ns.ledger).name}`); regression threshold "
             f"{REGRESS_TOL:.0%} (`MODEL_REGRESS_TOL`)")
    L.append("")

    L.append(f"| # | benchmark | {col_label.replace('*', '')} | this run | Δ | |")
    L.append("|---|---|---:|---:|---:|---|")
    for num, name, prev, val, delta, note in table:
        p = "—" if prev is None else f"{prev:g}"
        v = "—" if val is None else f"{val:.4g}"
        d = "—" if delta is None else f"{delta:+.1%}"
        L.append(f"| {num} | {name} | {p} | {v} | {d} | {note} |")
    L.append("")

    if regressed:
        L.append("**Regressions:**")
        L.append("")
        for num, name, prev, val, delta in regressed:
            L.append(f"- row {num} {name}: {prev:g} → {val:.4g} ({delta:+.1%})")
        L.append("")
    if newly_failing:
        L.append("**Newly failing / missing rows:**")
        L.append("")
        for num, name, note in newly_failing:
            L.append(f"- row {num} {name}: {note}")
        L.append("")
    if newly_passing:
        L.append("**Newly measured (previously blocked):**")
        L.append("")
        for num, name, val in newly_passing:
            L.append(f"- row {num} {name}: {val:.4g}")
        L.append("")
    if failed_cells:
        L.append(f"<details><summary>Cells reporting ok=false "
                 f"({len(failed_cells)})</summary>")
        L.append("")
        for r in failed_cells:
            L.append(f"- `{r.get('id')}` / `{r.get('backend')}`: "
                     f"{str(r.get('error'))[:160]}")
        L.append("")
        L.append("</details>")
        L.append("")

    L.append("<details><summary>compare_tokens.py output</summary>")
    L.append("")
    L.append("```")
    L.append(tok_out.strip())
    L.append("```")
    L.append("")
    L.append("</details>")
    L.append("")
    L.append(f"Raw: `{ns.run_log}`, `{ns.raw}`, `{ns.raw}.maxtext`, "
             f"merged `{ns.merged}`, tokens `{ns.tokens_log}`")
    md = "\n".join(L)
    print(md)

    if ns.md:
        Path(ns.md).write_text(md + "\n")
    if ns.json:
        Path(ns.json).write_text(json.dumps({
            "step": "models",
            "title": "Step 4 — model suite",
            "status": status,
            "smoke": ns.smoke,
            "seconds": float(ns.seconds) if ns.seconds else None,
            "headline": (f"tokens {'FAIL' if tok_fail else 'PASS'} "
                         f"({len(tok_new)} unexpected / {len(tok_known)} known "
                         f"divergences), {len(regressed)} regressed, "
                         f"{len(newly_failing)} newly failing"),
            "problems": problems,
            "warnings": warns,
            "ledger_column": col_label,
            "rows": [{"row": n, "name": nm, "prev": p, "value": v,
                      "delta": d, "note": no} for n, nm, p, v, d, no in table],
            "regressed": [{"row": n, "name": nm, "prev": p, "value": v,
                           "delta": d} for n, nm, p, v, d in regressed],
            "newly_failing": [{"row": n, "name": nm, "note": no}
                              for n, nm, no in newly_failing],
            "newly_passing": [{"row": n, "name": nm, "value": v}
                              for n, nm, v in newly_passing],
            "token_audit_pass": not tok_fail,
            "token_divergences_unexpected": tok_new,
            "token_divergences_known": tok_known,
            "compare_tokens_rc": tok.returncode,
            "merged": ns.merged,
            "markdown": md,
            "logs": {"run": ns.run_log, "raw": ns.raw,
                     "tokens": ns.tokens_log, "merged": ns.merged},
        }, indent=1) + "\n")

    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
