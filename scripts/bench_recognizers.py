#!/usr/bin/env python
"""Synthetic micro-benchmarks for the three recognizer emits (P17).

One decode/inference-shaped program per family, timed through jax/PJRT so the
environment's plugin choice is what is being measured:

    qmm    a gpt-oss-shaped MXFP4 projection and a keras-shaped int4 one
    moe    a dense expert dispatch at decode (T=1) and at prefill
    sdpa   a softmax attention at a diffusion-ish and at a decode shape

Each is run under three configurations of the SAME native plugin --
recognizers on, recognizers off (`METALJAX_RECOGNIZE=0`), and (optionally) the
Stage 1 trampoline -- which isolates exactly what the emit is worth: the
programs, the shapes and the arithmetic are identical, only the rewrite
differs.

`jax.block_until_ready` is a no-op on this backend (CLAUDE.md item 9), so
every timing loop is closed by `np.asarray` on one output.

    scripts/bench_recognizers.py --out notes/data/p17-emits-micro.jsonl
"""

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent.parent
_DYLIB = _HERE / "plugin-native" / "bazel-bin" / "metal" / \
    "libmetal_pjrt_native.dylib"


# --------------------------------------------------------------------------
# the programs
# --------------------------------------------------------------------------


def _rand(shape, seed, dtype=np.float32):
    return np.asarray(np.random.RandomState(seed).standard_normal(shape),
                      dtype=dtype)


def _mxfp4(shape_nk, seed):
    rng = np.random.RandomState(seed)
    n, k = shape_nk[-2], shape_nk[-1]
    lead = tuple(shape_nk[:-2])
    codes = rng.randint(0, 16, size=lead + (n, k)).astype(np.uint8)
    blocks = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    sb = rng.randint(118, 133, size=lead + (n, k // 32)).astype(np.uint8)
    return blocks, sb


def _int4(rows, cols, block, seed, dtype):
    rng = np.random.RandomState(seed)
    q = rng.randint(-8, 8, size=(rows, cols)).astype(np.int8)
    ng = rows // block
    scale = ((rng.rand(ng, cols).astype(np.float32) + 0.5) * 0.05).astype(dtype)
    zero = rng.randint(-3, 4, size=(ng, cols)).astype(np.int8)
    g_idx = (np.arange(rows) // block).astype(np.float32)
    packed = ((q[:, 0::2] & 0x0F) | (q[:, 1::2] << 4)).astype(np.int8)
    return packed, scale, zero, g_idx


def programs():
    """(name, fn, args, note) for every benchmark, built lazily."""
    import jax
    import jax.numpy as jnp

    E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    np.float32)
    tab = np.ldexp(np.ones(256), np.arange(256) - 127)
    tab[255] = np.nan
    STAB = tab.astype(np.float32)

    def mxfp4_weight(blocks, sb, k, dtype):
        vt = jnp.asarray(E2M1, dtype=dtype)
        st = jnp.asarray(STAB)
        lead = tuple(blocks.shape[:-1])
        lo = jnp.bitwise_and(blocks, jnp.uint8(0x0F))
        hi = jnp.right_shift(blocks, jnp.uint8(4))
        nib = jnp.reshape(jnp.stack([lo, hi], axis=-1), lead + (k,))
        vals = jnp.take(vt, nib.astype(jnp.int32), axis=0)
        scale = jnp.take(st, sb.astype(jnp.int32), axis=0)
        w = (jnp.reshape(vals, lead + (k // 32, 32))
             * scale[..., None].astype(dtype))
        return jnp.reshape(w, lead + (k,))

    def unpack(packed, columns):
        lo = jnp.bitwise_and(packed, jnp.int8(0x0F))
        lo = jnp.where(lo > 7, lo - 16, lo)
        hi = jnp.right_shift(packed, jnp.int8(4))
        return jnp.reshape(jnp.stack([lo, hi], axis=-1),
                           packed.shape[:-1] + (columns,))

    out = []

    # --- qmm: one gpt-oss-class projection, at decode and at prefill -----
    for T, tag in ((1, "decode"), (64, "prefill")):
        blocks, sb = _mxfp4((5760, 2880), 1)
        x = _rand((T, 2880), 2, np.float32) * 0.4
        out.append((f"qmm-mxfp4-{tag}",
                    lambda b, s, a: jnp.einsum(
                        "th,nh->tn", a, mxfp4_weight(b, s, 2880, a.dtype)),
                    [blocks, sb, x],
                    "gpt-oss gate_up shape: [T,2880] x mxfp4[5760,2880]"))
    packed, scale, zero, g_idx = _int4(4096, 4096, 128, 3, np.float32)
    x = _rand((1, 4096), 4) * 0.4
    out.append(("qmm-int4-decode",
                lambda p, s, z, g, a: a @ (
                    (unpack(p, 4096).astype(a.dtype)
                     - jnp.take(z, g.astype(jnp.int32), axis=0).astype(a.dtype))
                    * jnp.take(s, g.astype(jnp.int32), axis=0)),
                [packed, scale, zero, g_idx, x],
                "keras int4 sub-channel: [1,4096] x int4[4096,4096]"))

    # --- moe: the dense dispatch, at decode and at prefill ---------------
    def moe_block(x, wg, wd, k):
        logits = x @ wg
        vals, idx = jax.lax.top_k(logits, k)
        w = jax.nn.softmax(vals, axis=-1)
        onehot = (idx[..., None] == jnp.arange(wg.shape[1])).astype(w.dtype)
        scores = jnp.sum(onehot * w[..., None], axis=1)
        y = jnp.einsum("th,ehd->etd", x, wd)
        return jnp.sum(y * scores.T[..., None], axis=0)

    E, D, H = 32, 1024, 1024
    wg = _rand((D, E), 10) * 0.2
    wd = _rand((E, D, H), 11) * 0.1
    for T, tag in ((1, "decode"), (32, "prefill")):
        out.append((f"moe-e{E}k4-{tag}",
                    lambda a, g, w: moe_block(a, g, w, 4),
                    [_rand((T, D), 12), wg, wd],
                    f"dense dispatch E={E} K=4 T={T} d={D} h={H}"))

    # --- sdpa: a diffusion-ish forward and a decode step -----------------
    for (B, Hn, Tq, Tk, Dh), tag in (((2, 16, 1024, 1024, 64), "1024x1024"),
                                     ((1, 32, 1, 2048, 128), "decode")):
        q = _rand((B, Hn, Tq, Dh), 20) * 0.4
        k = _rand((B, Hn, Tk, Dh), 21) * 0.4
        v = _rand((B, Hn, Tk, Dh), 22) * 0.4
        out.append((f"sdpa-{tag}",
                    lambda a, b, c: jnp.einsum(
                        "bhqk,bhkd->bhqd",
                        jax.nn.softmax(
                            jnp.einsum("bhqd,bhkd->bhqk", a, b) * 0.125, -1),
                        c),
                    [q, k, v],
                    f"B={B} H={Hn} Tq={Tq} Tk={Tk} D={Dh}"))
    return out


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------


def run_one(name, reps):
    import jax
    import jax.numpy as jnp

    for pname, fn, args, note in programs():
        if pname != name:
            continue
        dev = jax.devices("metal")[0]
        with jax.default_device(dev):
            moved = [jax.device_put(jnp.asarray(a), dev) for a in args]
            f = jax.jit(fn)
            np.asarray(f(*moved))          # compile + any pack prologue
            np.asarray(f(*moved))          # ...and settle
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                np.asarray(f(*moved))
                times.append((time.perf_counter() - t0) * 1e3)
        return {"name": pname, "note": note, "ms": min(times),
                "median_ms": statistics.median(times), "reps": reps}
    raise SystemExit(f"no such benchmark: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--only", default="")
    ap.add_argument("--child", default="")
    ap.add_argument("--stage1", action="store_true",
                    help="also measure the frozen Stage 1 trampoline")
    args = ap.parse_args()

    if args.child:
        print(json.dumps(run_one(args.child, args.reps)), flush=True)
        return 0

    os.environ.setdefault("JAX_PLATFORMS", "metal")
    names = [p[0] for p in _names()]
    if args.only:
        names = [n for n in names if args.only in n]
    configs = [("native+emits", {"METALJAX_PLUGIN_PATH": str(_DYLIB)}),
               ("native-emits", {"METALJAX_PLUGIN_PATH": str(_DYLIB),
                                 "METALJAX_RECOGNIZE": "0"})]
    if args.stage1:
        configs.append(("stage1", {}))

    rows = []
    for name in names:
        row = {"benchmark": name}
        for cfg, env in configs:
            e = dict(os.environ)
            e.pop("METALJAX_PLUGIN_PATH", None)
            e.pop("METALJAX_RECOGNIZE", None)
            e.update(env)
            e["JAX_PLATFORMS"] = "metal"
            proc = subprocess.run(
                [sys.executable, __file__, "--child", name,
                 "--reps", str(args.reps)],
                capture_output=True, text=True, env=e)
            line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
            if not line:
                row[cfg] = None
                print(f"{name:<24} {cfg:<14} FAILED "
                      f"{proc.stderr.strip().splitlines()[-1][:80]}"
                      if proc.stderr.strip() else "FAILED")
                continue
            got = json.loads(line[-1])
            row[cfg] = got["ms"]
            row["note"] = got["note"]
            print(f"{name:<24} {cfg:<14} {got['ms']:9.3f} ms "
                  f"(median {got['median_ms']:.3f})", flush=True)
        if row.get("native+emits") and row.get("native-emits"):
            row["speedup_vs_no_emits"] = row["native-emits"] / row["native+emits"]
            print(f"{name:<24} {'speedup':<14} "
                  f"{row['speedup_vs_no_emits']:9.2f}x", flush=True)
        if row.get("stage1") and row.get("native+emits"):
            row["native_over_stage1"] = row["native+emits"] / row["stage1"]
        rows.append(row)

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"\nwrote {path}")
    return 0


def _names():
    """The benchmark list without importing jax in the parent process."""
    return [("qmm-mxfp4-decode",), ("qmm-mxfp4-prefill",),
            ("qmm-int4-decode",), ("moe-e32k4-decode",),
            ("moe-e32k4-prefill",), ("sdpa-1024x1024",), ("sdpa-decode",)]


if __name__ == "__main__":
    sys.exit(main())
