#!/usr/bin/env python
"""Row-15 forensics: WHERE the collapse enters, per decoder layer.

Row 15 (`qwix-int8-qwen3-8b`) decodes token id 0 (`!`) forever: the logits have
gone flat, which on this backend is what a SINGLE non-finite value anywhere
upstream produces (`notes/row15-wrong-output-2026-08-17.md` §4b — argmax is
NaN-wins/lowest-index, and `h @ W` turns one bad element into 151936 NaNs).
The text therefore carries no information about the mechanism.  The KV cache
does.

MaxEngine's `prefill` returns the whole prefix, and with `scan_layers=true` the
per-layer KV cache is **stacked on axis 0** — one array of shape
`[layers, batch, seq, kv_heads, head_dim]` per cache variable.  Layer i's K/V
is computed from the residual stream *entering* layer i, so the first layer
index whose K/V is non-finite (or whose signal has collapsed to zeros, which is
what per-tensor absmax quantization does when it sees an `inf`) is the entry
point of the fault, read straight off one array with no model surgery.

This is deliberately NOT a metal-vs-CPU comparison: jax-CPU already runs this
row coherently (`34f627c`), and holding a second 16 GB copy of an 8B parameter
set is exactly the memory profile the panic ledger forbids.  The criterion is
intrinsic — non-finite counts and per-layer signal statistics — so one guarded
run on one backend answers it.

    row15_forensics.py --bench qwix-int8-qwen3-8b     # the failing row
    row15_forensics.py --bench maxtext-qwix-int8      # row 14, the control
    row15_forensics.py --bench qwix-int8-qwen3-8b --decode 2 --scan-params

Stack under test is whatever `METALJAX_PLUGIN_PATH` selects and whatever
`PYTHONPATH` puts in front of the installed `metaljax` — the two knobs rung D
of the ladder uses to ask the provenance question (native today / Stage 1 today
/ 0.11.2's `src/metaljax` on Stage 1's dylib).

Every finding is a `RESULT:`-prefixed JSON line on stdout; the readable table
goes to stderr.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def emit(rec):
    print("RESULT:" + json.dumps(rec), flush=True)


def _stats(a):
    """Intrinsic health of one tensor, computed on the host in f32.

    `nonfinite` and `zero_frac` are the two faces the amplifier has: an `inf`
    that survives is non-finite, an `inf` that has been through one round of
    absmax quantization has turned every OTHER element into a zero.
    """
    a = np.asarray(a, dtype=np.float32)
    finite = np.isfinite(a)
    fin = a[finite]
    return {
        "n": int(a.size),
        "nonfinite": int(a.size - finite.sum()),
        "nan": int(np.isnan(a).sum()),
        "posinf": int(np.isposinf(a).sum()),
        "neginf": int(np.isneginf(a).sum()),
        "zero_frac": float(np.count_nonzero(a == 0) / a.size),
        "absmax": float(np.abs(fin).max()) if fin.size else 0.0,
        "std": float(fin.std()) if fin.size else 0.0,
    }


def _walk(tree, prefix=""):
    """Flatten any pytree of arrays into (path, array) pairs.

    `jax.tree_util` rather than a hand-rolled dict walk: MaxEngine hands back
    nnx `State` objects and `ResultTokens` pytrees, neither of which is a
    `dict`, and a walk that silently skipped them would report "clean" for
    tensors it never looked at.
    """
    import jax
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            yield prefix + jax.tree_util.keystr(path), leaf


# The cache leaves that actually carry the residual stream's signature.  After
# a prefill the AUTOREGRESSIVE half of the cache (`cached_ar_key/value`,
# `cache_ar_index`, ...) is legitimately all zeros — nothing has been decoded
# into it yet — so "absmax == 0" is health there, not collapse.  Verified on
# row 14, which is coherent and shows exactly that pattern.
SIGNAL_LEAVES = ("cached_prefill_key", "cached_prefill_value")


def probe_cache(cache, layers, tag, signal=SIGNAL_LEAVES):
    """Per-layer health of the stacked KV cache — the localization.

    Reports every leaf whose leading dimension is the layer count, sliced layer
    by layer.  A leaf that is not layer-stacked (`cache_ar_index` scalars and
    friends) is reported whole, once.
    """
    first_bad = None
    rows = []
    for path, arr in _walk(cache, tag):
        if arr.size == 0:
            continue
        host = np.asarray(arr, dtype=np.float32)
        if host.ndim >= 1 and host.shape[0] == layers and layers > 1:
            for i in range(layers):
                st = _stats(host[i])
                st.update(path=path, layer=i, shape=list(host.shape[1:]))
                rows.append(st)
        else:
            st = _stats(host)
            st.update(path=path, layer=-1, shape=list(host.shape))
            rows.append(st)
        del host

    # A layer is "bad" when it holds values that are not numbers, or when its
    # signal is gone (a collapsed residual quantizes to all-zeros, and the KV
    # projection of zeros is zeros).  Both are absolute readings: no reference.
    for r in sorted(rows, key=lambda r: (r["layer"], r["path"])):
        carries_signal = any(s in r["path"] for s in signal)
        r["signal"] = carries_signal
        r["bad"] = bool(r["nonfinite"] or
                        (carries_signal and r["n"] > 16 and r["absmax"] == 0.0))
        emit({"probe": "cache", **r})
        if r["bad"] and first_bad is None:
            first_bad = r

    for r in sorted(rows, key=lambda r: (r["layer"], r["path"])):
        if not r["signal"] and not r["bad"]:
            continue
        log(f"  L{r['layer']:>3} {r['path'][-46:]:46s} "
            f"nonfin={r['nonfinite']:<8d} absmax={r['absmax']:<12.5g} "
            f"std={r['std']:<12.5g} zero={r['zero_frac']:.4f}"
            f"{'   <-- BAD' if r['bad'] else ''}")
    return first_bad


def probe_logits(logits, tag):
    st = _stats(logits)
    a = np.asarray(logits, dtype=np.float32).reshape(-1)
    order = np.argsort(np.nan_to_num(a, nan=-np.inf))[::-1][:5]
    rec = {"probe": "logits", "tag": tag, **st,
           "argmax_nanwins": int(np.argmax(np.isnan(a))) if st["nan"] else int(a.argmax()),
           "top5_idx": [int(i) for i in order],
           "top5_val": [float(a[i]) for i in order],
           "flat": bool(st["std"] == 0.0 or st["nonfinite"] > 0)}
    emit(rec)
    log(f"  logits[{tag}] n={st['n']} nonfinite={st['nonfinite']} "
        f"std={st['std']:.6g} absmax={st['absmax']:.6g} top5={rec['top5_idx']}")
    return rec


def probe_params(params):
    """Are the weights themselves already broken before any math runs?

    qwix quantizes on the fly inside the jit, but the checkpoint restore, the
    bf16 cast and the transfer path all run on device before that, and a
    non-finite weight would explain everything downstream with no runtime bug at
    all.  Reduced leaf by leaf so the transients stay tiny.
    """
    import jax.numpy as jnp
    import jax

    @jax.jit
    def health(x):
        f = x.astype(jnp.float32)
        ok = jnp.isfinite(f)
        return (jnp.sum(~ok).astype(jnp.int32),
                jnp.max(jnp.where(ok, jnp.abs(f), 0.0)),
                jnp.sum(jnp.where(ok, f * f, 0.0)))

    # CHUNKED along the leading axis.  The 8B's stacked decoder-layer leaf is
    # 36 x 192.9 M = 6.9 G elements; `astype(f32)` on the whole thing is a
    # 27.8 GB transient on a machine whose panic ledger is written in exactly
    # that kind of allocation.  One slice of it is 0.77 GB.
    CHUNK_ELEMS = 256 << 20
    worst = []
    for path, arr in _walk(params, "params"):
        if arr.size == 0:
            continue
        if arr.size > CHUNK_ELEMS and arr.ndim >= 1 and arr.shape[0] > 1:
            step = max(1, int(CHUNK_ELEMS // max(1, arr.size // arr.shape[0])))
            parts = [health(arr[i:i + step]) for i in range(0, arr.shape[0], step)]
            nf = sum(int(np.asarray(p[0])) for p in parts)
            amax = max(float(np.asarray(p[1])) for p in parts)
            sq = sum(float(np.asarray(p[2])) for p in parts)
        else:
            a, b, c = health(arr)
            nf, amax, sq = int(np.asarray(a)), float(np.asarray(b)), float(np.asarray(c))
        rec = {"probe": "param", "path": path, "shape": list(arr.shape),
               "dtype": str(arr.dtype), "n": int(arr.size),
               "nonfinite": nf, "absmax": amax,
               "rms": float(np.sqrt(sq / arr.size))}
        emit(rec)
        log(f"  param {path[-56:]:56s} {str(arr.dtype):9s} "
            f"nonfin={rec['nonfinite']:<6d} absmax={rec['absmax']:<11.5g} "
            f"rms={rec['rms']:.5g}")
        if rec["nonfinite"] or rec["absmax"] == 0.0:
            worst.append(rec)
    log(f"  params: {len(worst)} bad leaves of {len(list(_walk(params, '')))}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="qwix-int8-qwen3-8b")
    ap.add_argument("--decode", type=int, default=0,
                    help="also run N autoregressive steps and report their tokens")
    ap.add_argument("--scan-params", action="store_true")
    ap.add_argument("--prefill-reps", type=int, default=1,
                    help="repeat the prefill N times on the loaded params — "
                         "the cheap way to measure a rate rather than a verdict")
    ap.add_argument("--probe-decode", action="store_true",
                    help="also localize the AR cache after every decode step")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    import adapter_maxtext as A
    A._ensure_importable()
    import jax

    log(f"row15_forensics bench={args.bench} "
        f"plugin={os.environ.get('METALJAX_PLUGIN_PATH', '(stage 1 default)')}")
    import metaljax
    log(f"metaljax {getattr(metaljax, '__version__', '?')} from {metaljax.__file__}")

    # The bench dict comes from the suite's own manifest, not a literal here:
    # `model` selects the tokenizer, and a hand-written one that drifted from
    # the manifest would decode the right ids against the wrong vocabulary.
    bench = None
    def _find(o):
        nonlocal bench
        if isinstance(o, dict):
            if o.get("id") == args.bench:
                bench = o
            for v in o.values():
                _find(v)
        elif isinstance(o, list):
            for v in o:
                _find(v)
    _find(json.loads((Path(__file__).parent / "manifest.json").read_text()))
    if bench is None:
        raise SystemExit(f"no manifest entry with id {args.bench!r}")
    spec = A._spec(bench)
    ckpt = A.CKPT_ROOT / spec["ckpt"] / "0" / "items"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    tok = A._tokenizer(bench)
    prompt_ids = tok.encode(args.prompt)
    prefill_len = int(os.environ.get("MAXTEXT_PREFILL_LEN", 0)) or \
        1 << max(6, (len(prompt_ids) + 1).bit_length())
    prompt_ids = prompt_ids[: prefill_len - 1]
    target_len = prefill_len + max(args.decode, 1) + 8

    t0 = time.monotonic()
    config = A._config(bench, [
        f"load_parameters_path={ckpt}",
        "per_device_batch_size=1",
        "weight_dtype=bfloat16",
        f"max_prefill_predict_length={prefill_len}",
        f"max_target_length={target_len}",
    ])
    layers = int(config.base_num_decoder_layers)
    from maxtext.inference.maxengine import maxengine
    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load = jax.random.split(rng)
    params = engine.load_params(rng_load)
    load_s = time.monotonic() - t0
    log(f"loaded in {load_s:.1f}s; layers={layers} prefill_len={prefill_len}")
    emit({"probe": "run", "bench": args.bench, "layers": layers,
          "prefill_len": prefill_len, "load_s": load_s,
          "plugin": os.environ.get("METALJAX_PLUGIN_PATH", "(stage1 default)"),
          "metaljax_file": metaljax.__file__,
          "metaljax_version": getattr(metaljax, "__version__", "?")})

    if args.scan_params:
        log("-- params --")
        probe_params(params)

    padded = np.zeros(prefill_len, dtype=np.int32)
    padded[: len(prompt_ids)] = prompt_ids

    # REPEATED DRAWS IN ONE PROCESS.  The load costs 80 s and a prefill costs
    # about one; the failure this is chasing turned out to differ between two
    # runs of the SAME configuration, so the quantity that matters is a rate,
    # not a verdict.  Reusing the loaded params makes a rate affordable.
    verdicts = []
    for rep in range(args.prefill_reps):
        rng, sub = jax.random.split(rng)
        t0 = time.monotonic()
        prefix, first_tok = engine.prefill(params=params, padded_tokens=padded,
                                           true_length=len(prompt_ids), rng=sub,
                                           slot=0)
        tid = int(np.asarray(first_tok.data)[0, 0])
        log(f"-- prefill rep {rep} in {time.monotonic() - t0:.1f}s; "
            f"first token {tid} ({tok.decode([tid])!r}) --")

        lg = probe_logits(prefix["logits"], f"prefill{rep}")
        first_bad = probe_cache(prefix["cache"], layers, "cache")
        v = {"probe": "verdict", "stage": "prefill", "rep": rep,
             "first_token": tid, "token_text": tok.decode([tid]),
             "logits_flat": lg["flat"], "logits_std": lg["std"],
             "first_bad_layer": (first_bad or {}).get("layer"),
             "first_bad_path": (first_bad or {}).get("path")}
        emit(v)
        verdicts.append(v)
        log(f"VERDICT prefill rep={rep}: first_token={tid} "
            f"logits_flat={lg['flat']} "
            f"first_bad_layer={(first_bad or {}).get('layer')}")
        if rep + 1 < args.prefill_reps:
            del prefix, first_tok

    emit({"probe": "draws", "reps": args.prefill_reps,
          "tokens": [v["first_token"] for v in verdicts],
          "distinct_tokens": len({v["first_token"] for v in verdicts}),
          "collapsed_reps": sum(1 for v in verdicts if v["logits_flat"]),
          "first_bad_layers": [v["first_bad_layer"] for v in verdicts]})

    if args.decode:
        ids = [verdicts[-1]["first_token"]]
        rng, sub = jax.random.split(rng)
        state = engine.init_decode_state(sub)
        state = engine.insert(prefix, state, slot=0)
        for step in range(args.decode):
            rng, sub = jax.random.split(rng)
            state, sampled = engine.generate(params, state, rng=sub)
            ids.append(int(np.asarray(sampled.data)[0, 0]))
            if args.probe_decode:
                # The AR half of the cache is the decode-side localization:
                # after `step+1` generated tokens it should carry signal, and
                # a layer that has gone non-finite there names the entry point
                # exactly as the prefill half does.
                log(f"-- decode step {step} cache --")
                fb = probe_cache(state["cache"], layers, f"ar{step}",
                                 signal=("cached_ar_key", "cached_ar_value"))
                emit({"probe": "verdict", "stage": "decode", "step": step,
                      "token": ids[-1],
                      "first_bad_layer": (fb or {}).get("layer"),
                      "first_bad_path": (fb or {}).get("path")})
                log(f"VERDICT decode step={step} token={ids[-1]} "
                    f"first_bad_layer={(fb or {}).get('layer')}")
        log(f"-- decode tokens {ids} -> {tok.decode(ids)!r} --")
        emit({"probe": "decode", "token_ids": ids, "text": tok.decode(ids)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
