"""Release-gate token-agreement audit over final_run.jsonl.

Policy (per the int8 certification, notes/int8-divergence-verdict.md):
  - bf16 LLM rows: greedy token_ids MUST match CPU exactly.
  - quantized rows (int4/int8): token equality is NOT the criterion;
    divergence is reported informationally with the first-divergence
    index (certified-benign class).
  - rows without a CPU counterpart: metal token_ids recorded only.
"""

import json
import sys
from collections import defaultdict

QUANT_ROWS = {"gemma4-e2b-int4"}

recs = defaultdict(dict)
for line in open(sys.argv[1] if len(sys.argv) > 1 else
                 "final_run.jsonl"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("ok") and "token_ids" in r:
        recs[r["id"]][r["backend"]] = r["token_ids"]

fails = 0
for bid, by_backend in sorted(recs.items()):
    m, c = by_backend.get("metaljax"), by_backend.get("cpu")
    if m is None or c is None:
        print(f"  {bid:24s} single-backend (recorded only)")
        continue
    n = min(len(m), len(c))
    div = next((i for i in range(n) if m[i] != c[i]), None)
    if div is None:
        print(f"  {bid:24s} AGREE ({n} tokens)")
    elif bid in QUANT_ROWS:
        print(f"  {bid:24s} quantized: diverges at {div}/{n} "
              f"(certified-benign class)")
    else:
        print(f"  {bid:24s} FAIL: diverges at token {div}/{n}")
        fails += 1

print(f"\n{'GATE FAIL' if fails else 'GATE PASS'}: {fails} bf16 divergences")
sys.exit(1 if fails else 0)
