"""Compare two texmo_topconfs result files.

The tracked headline is the GEOMETRIC MEAN of per-config relative
changes (old/new, so >1.0 = the new run is faster) — each configuration
counts equally regardless of its absolute step time.

    .venv/bin/python scripts/texmo_topconfs_compare.py OLD.jsonl NEW.jsonl

Configs are matched by (spec, precision, batch, length) so the
comparison survives re-exports that reorder or extend the list.
"""

import json
import math
import sys


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "metal_ms_step" in r:
            out[(r["spec"], r["precision"], r["batch"], r["length"])] = r
    return out


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(old) & set(new), key=lambda k: old[k]["weights"])
    only_old, only_new = len(old) - len(keys), len(new) - len(keys)
    print(f"{len(keys)} matched configs"
          + (f" ({only_old} only in old, {only_new} only in new)"
             if only_old or only_new else ""))

    pairs = [(k, old[k], new[k]) for k in keys]
    metal = [(k, o["metal_ms_step"] / n["metal_ms_step"]) for k, o, n in pairs]
    cpu = [o["cpu_ms_step"] / n["cpu_ms_step"] for _, o, n in pairs]
    warm = [o["metal_warmup_s"] / n["metal_warmup_s"] for _, o, n in pairs]

    print(f"\nHEADLINE geomean speedup (old/new, >1 = faster):")
    print(f"  metal ms/step : {geomean([r for _, r in metal]):.4f}x")
    print(f"  metal warmup  : {geomean(warm):.4f}x")
    print(f"  cpu ms/step   : {geomean(cpu):.4f}x   (control; ~1.0 unless "
          f"the machine or jax changed)")

    buckets = [(0, 100), (100, 500), (500, 1500), (1500, 3001)]
    print("\nper weight class (metal geomean):")
    for lo, hi in buckets:
        rs = [r for (k, r) in metal if lo <= old[k]["weights"] < hi]
        if rs:
            print(f"  {lo:>5}-{hi:<5} n={len(rs):<4} {geomean(rs):.4f}x")

    imp = sum(1 for _, r in metal if r > 1.05)
    reg = sum(1 for _, r in metal if r < 0.95)
    print(f"\nimproved >5%: {imp}   regressed >5%: {reg}   "
          f"within noise: {len(metal) - imp - reg}")
    movers = sorted(metal, key=lambda kr: kr[1])
    print("\nworst movers:")
    for k, r in movers[:5]:
        print(f"  {r:.3f}x  {k[0][:52]} {k[1]} b{k[2]} l{k[3]} "
              f"({old[k]['metal_ms_step']} -> {new[k]['metal_ms_step']} ms)")
    print("best movers:")
    for k, r in movers[-5:][::-1]:
        print(f"  {r:.3f}x  {k[0][:52]} {k[1]} b{k[2]} l{k[3]} "
              f"({old[k]['metal_ms_step']} -> {new[k]['metal_ms_step']} ms)")


if __name__ == "__main__":
    main()
