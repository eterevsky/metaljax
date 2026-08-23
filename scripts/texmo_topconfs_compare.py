"""Compare two texmo benchmark result files.

The tracked headline is the GEOMETRIC MEAN of per-config relative
changes (old/new, so >1.0 = the new run is faster) — each configuration
counts equally regardless of its absolute step time.

    .venv/bin/python scripts/texmo_topconfs_compare.py OLD.jsonl NEW.jsonl

Configs are matched by (spec, precision, batch, length) so the
comparison survives re-exports that reorder or extend the list.

Reads `scripts/bench_texmo_pjrt.py`'s schema -- one record per config per
PLATFORM, with `ms_step` / `warmup_s` / `platform` -- which is the only
route since the Stage-1 retirement (0.11.6); the retired
`scripts/texmo_topconfs.py` wrote both legs into one record
(`metal_ms_step` / `cpu_ms_step`), and those files are still read so the
pre-0.11.6 anchors remain comparable.  Files may carry several platforms;
`--platform` picks which one is compared (default metal) and a cpu leg,
where both files have one, is reported as the control.
"""

import argparse
import json
import math


def load(path, platform):
    """{(spec, precision, batch, length): {'ms': .., 'warm': .., 'weights': ..}}
    for `platform`, from either schema."""
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = (r["spec"], r["precision"], r["batch"], r["length"])
        if "ms_step" in r:                       # bench_texmo_pjrt schema
            if r.get("platform", "metal") != platform:
                continue
            ms, warm = r.get("ms_step"), r.get("warmup_s")
        elif f"{platform}_ms_step" in r:         # retired topconfs schema
            ms = r.get(f"{platform}_ms_step")
            warm = r.get(f"{platform}_warmup_s")
        else:
            continue
        if ms:
            out[key] = {"ms": ms, "warm": warm, "weights": r.get("weights", 0)}
    return out


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--platform", default="metal")
    args = ap.parse_args()

    old = load(args.old, args.platform)
    new = load(args.new, args.platform)
    keys = sorted(set(old) & set(new), key=lambda k: old[k]["weights"])
    only_old, only_new = len(old) - len(keys), len(new) - len(keys)
    print(f"{len(keys)} matched configs"
          + (f" ({only_old} only in old, {only_new} only in new)"
             if only_old or only_new else ""))
    if not keys:
        raise SystemExit(
            f"no configs matched on platform {args.platform!r} -- are the "
            f"two files from the same config set?")

    ratios = [(k, old[k]["ms"] / new[k]["ms"]) for k in keys]
    warm = [old[k]["warm"] / new[k]["warm"] for k in keys
            if old[k]["warm"] and new[k]["warm"]]

    print(f"\nHEADLINE geomean speedup (old/new, >1 = faster):")
    print(f"  {args.platform} ms/step : "
          f"{geomean([r for _, r in ratios]):.4f}x")
    if warm:
        print(f"  {args.platform} warmup  : {geomean(warm):.4f}x")

    # The cpu leg, when both files carry one, is the machine control.
    if args.platform != "cpu":
        cold, cnew = load(args.old, "cpu"), load(args.new, "cpu")
        ck = sorted(set(cold) & set(cnew))
        if ck:
            print(f"  cpu ms/step   : "
                  f"{geomean([cold[k]['ms'] / cnew[k]['ms'] for k in ck]):.4f}x"
                  f"   (control; ~1.0 unless the machine or jax changed)")

    buckets = [(0, 100), (100, 500), (500, 1500), (1500, 3001)]
    if any(old[k]["weights"] for k in keys):
        print(f"\nper weight class ({args.platform} geomean):")
        for lo, hi in buckets:
            rs = [r for (k, r) in ratios if lo <= old[k]["weights"] < hi]
            if rs:
                print(f"  {lo:>5}-{hi:<5} n={len(rs):<4} {geomean(rs):.4f}x")

    imp = sum(1 for _, r in ratios if r > 1.05)
    reg = sum(1 for _, r in ratios if r < 0.95)
    print(f"\nimproved >5%: {imp}   regressed >5%: {reg}   "
          f"within noise: {len(ratios) - imp - reg}")
    movers = sorted(ratios, key=lambda kr: kr[1])
    print("\nworst movers:")
    for k, r in movers[:5]:
        print(f"  {r:.3f}x  {k[0][:52]} {k[1]} b{k[2]} l{k[3]} "
              f"({old[k]['ms']} -> {new[k]['ms']} ms)")
    print("best movers:")
    for k, r in movers[-5:][::-1]:
        print(f"  {r:.3f}x  {k[0][:52]} {k[1]} b{k[2]} l{k[3]} "
              f"({old[k]['ms']} -> {new[k]['ms']} ms)")


if __name__ == "__main__":
    main()
