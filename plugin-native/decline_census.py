#!/usr/bin/env python
"""What the native plugin still declines, and on which op (phase 2, P3).

Every program the plugin cannot lower is refused WHOLE, naming the op that
stopped it -- so the set of distinct messages over a representative workload
is the priority order for the phases that follow.  This script produces that
census two ways:

* a built-in probe set: one `jax.jit` per shape a real workload is made of
  (embeddings, RNG, losses, optimizer steps, attention, a decode loop, ...);
* `--module a.mlir b.mlir`: StableHLO modules compiled straight through
  `client.compile_and_load`, which is how a texmo training chunk (or any
  module dumped from another process) gets here without its own dependencies.

    JAX_PLATFORMS=metal METALJAX_PLUGIN_PATH=<dylib> \
        .venv/bin/python plugin-native/decline_census.py [--module ...]

Compile only: nothing below is executed, so the census costs no GPU time and
says nothing about whether a program that lowers is CORRECT (that is
execute_test.py's job).
"""

import argparse
import collections
import os
import pathlib
import re
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_DYLIB = _HERE / "bazel-bin" / "metal" / "libmetal_pjrt_native.dylib"

# "UNIMPLEMENTED: metaljax-native: op stablehlo.gather" -> the part after the
# prefix.  Anything that does not match is reported in full: a message this
# does not recognize is a failure mode worth seeing, not one worth truncating.
_REASON = re.compile(r"metaljax-native: ([^\n]*)")


def _rand(shape, seed, dtype=np.float32):
    return np.asarray(np.random.RandomState(seed).standard_normal(shape),
                      dtype=dtype)


def probes():
    """(group, name, fn, args) for each probe.

    The groups are the workloads the roadmap cares about: a texmo training
    step, an LLM's decode, and the jax surface both are written in.
    """
    import jax
    import jax.numpy as jnp

    f = np.float32
    out = []

    def add(group, name, fn, args):
        out.append((group, name, fn, args))

    # --- the pieces a texmo training step is made of ---------------------
    emb = _rand((32, 8), 1)
    toks = np.arange(12, dtype=np.int32) % 32
    add("texmo", "embedding lookup", lambda e, t: e[t], [emb, toks])
    add("texmo", "one-hot matmul",
        lambda t, w: jax.nn.one_hot(t, 32) @ w, [toks, _rand((32, 8), 2)])
    add("texmo", "cross-entropy loss",
        lambda logits, t: -jnp.take_along_axis(
            jax.nn.log_softmax(logits, -1), t[:, None], -1).mean(),
        [_rand((12, 32), 3), toks])
    add("texmo", "softmax + argmax",
        lambda x: (jax.nn.softmax(x, -1), jnp.argmax(x, -1)),
        [_rand((4, 9), 4)])
    add("texmo", "layernorm",
        lambda x, g, b: g * (x - x.mean(-1, keepdims=True)) / jnp.sqrt(
            x.var(-1, keepdims=True) + 1e-5) + b,
        [_rand((4, 8), 5), _rand((8,), 6), _rand((8,), 7)])
    add("texmo", "GRU-shaped scan",
        lambda h0, w, xs: jax.lax.scan(
            lambda h, x: (jnp.tanh(h @ w + x), h), h0, xs),
        [_rand((4, 8), 8), _rand((8, 8), 9), _rand((6, 4, 8), 10)])
    add("texmo", "grad of an MLP",
        lambda w1, w2, x, y: jax.grad(
            lambda a, b: ((jnp.tanh(x @ a) @ b - y) ** 2).mean(),
            argnums=(0, 1))(w1, w2),
        [_rand((8, 16), 11), _rand((16, 4), 12), _rand((6, 8), 13),
         _rand((6, 4), 14)])
    add("texmo", "SGD update",
        lambda w, g: w - 0.01 * g, [_rand((8, 8), 15), _rand((8, 8), 16)])
    add("texmo", "Adam update",
        lambda w, m, v, g, t: (
            lambda mh, vh: (w - 0.01 * mh / (jnp.sqrt(vh) + 1e-8), mh, vh))(
                (0.9 * m + 0.1 * g) / (1 - 0.9 ** t),
                (0.999 * v + 0.001 * g * g) / (1 - 0.999 ** t)),
        [_rand((8, 8), 17), _rand((8, 8), 18), np.abs(_rand((8, 8), 19)),
         _rand((8, 8), 20), np.float32(3.0)])
    add("texmo", "scanned train chunk",
        lambda w, xs: jax.lax.scan(
            lambda p, x: (p - 0.1 * jax.grad(
                lambda q: (jnp.tanh(x @ q) ** 2).sum())(p), None), w, xs)[0],
        [_rand((8, 8), 21), _rand((4, 6, 8), 22)])
    add("texmo", "threefry random normal",
        lambda k: jax.random.normal(jax.random.wrap_key_data(k), (4, 4)),
        [np.array([0, 7], np.uint32)])
    add("texmo", "random split",
        lambda k: jax.random.split(jax.random.wrap_key_data(k), 4),
        [np.array([0, 7], np.uint32)])
    add("texmo", "token sampling (top-1 of a sorted row)",
        lambda x: jax.lax.top_k(x, 3), [_rand((2, 16), 23)])

    # --- an LLM's shapes -------------------------------------------------
    add("llm", "attention (dot + softmax)",
        lambda q, k, v: jax.nn.softmax(
            q @ k.swapaxes(-1, -2) / np.float32(2.83), -1) @ v,
        [_rand((2, 4, 8), 24), _rand((2, 6, 8), 25), _rand((2, 6, 8), 26)])
    add("llm", "KV cache update (dus at a carried index)",
        lambda cache, kv, i: jax.lax.dynamic_update_slice(
            cache, kv, (0, i, 0)),
        [_rand((2, 16, 8), 27), _rand((2, 1, 8), 28), np.int32(3)])
    add("llm", "decode loop (while over a cache)",
        lambda cache, w, i0: jax.lax.while_loop(
            lambda s: s[1] < 8,
            lambda s: (jax.lax.dynamic_update_slice(
                s[0], jnp.tanh(s[0][:, :1, :] @ w), (0, s[1], 0)), s[1] + 1),
            (cache, i0))[0],
        [_rand((2, 16, 8), 29), _rand((8, 8), 30), np.int32(0)])
    add("llm", "rope-style rotate + concat",
        lambda x: jnp.concatenate([-x[..., 1::2], x[..., ::2]], -1),
        [_rand((2, 4, 8), 31)])
    add("llm", "quantized-style dequant matmul",
        lambda q, s, x: x @ (q.astype(jnp.float32) * s),
        [np.arange(64, dtype=np.int8).reshape(8, 8), _rand((1, 8), 32),
         _rand((4, 8), 33)])

    # --- the wider jax surface ------------------------------------------
    add("jax", "cumsum", lambda x: jnp.cumsum(x, 0), [_rand((8, 3), 34)])
    add("jax", "sort", lambda x: jnp.sort(x, -1), [_rand((3, 5), 35)])
    add("jax", "argsort", lambda x: jnp.argsort(x, -1), [_rand((3, 5), 36)])
    add("jax", "where + clip",
        lambda x: jnp.where(x > 0, jnp.clip(x, -1, 1), -x), [_rand((9,), 37)])
    add("jax", "scatter-add (segment_sum)",
        lambda x, s: jax.ops.segment_sum(x, s, 4),
        [_rand((8,), 38), np.arange(8, dtype=np.int32) % 4])
    add("jax", "index update (.at[].set)",
        lambda x, u: x.at[2:5].set(u), [_rand((8,), 39), _rand((3,), 40)])
    add("jax", "convolution",
        lambda x, k: jax.lax.conv_general_dilated(
            x, k, (1,), "SAME", dimension_numbers=("NCH", "OIH", "NCH")),
        [_rand((1, 2, 8), 41), _rand((3, 2, 3), 42)])
    add("jax", "max pooling (reduce_window)",
        lambda x: jax.lax.reduce_window(
            x, -np.inf, jax.lax.max, (1, 2), (1, 2), "VALID"),
        [_rand((2, 8), 43)])
    add("jax", "fft", lambda x: jnp.fft.fft(x), [_rand((8,), 44)])
    add("jax", "cholesky",
        lambda a: jnp.linalg.cholesky(a @ a.T + 4 * jnp.eye(4)),
        [_rand((4, 4), 45)])
    add("jax", "qr", lambda a: jnp.linalg.qr(a), [_rand((5, 4), 46)])
    add("jax", "bitcast", lambda x: jax.lax.bitcast_convert_type(x, jnp.int32),
        [_rand((4,), 47)])
    add("jax", "reverse", lambda x: x[::-1], [_rand((5,), 48)])
    add("jax", "roll", lambda x: jnp.roll(x, 2), [_rand((5,), 49)])
    add("jax", "shift/bitwise", lambda x: (x << 2) ^ (x >> 1),
        [np.arange(8, dtype=np.uint32)])
    add("jax", "vmap of a scan",
        lambda xs: jax.vmap(lambda r: jax.lax.scan(
            lambda c, v: (c + v, None), np.float32(0), r)[0])(xs),
        [_rand((3, 5), 50)])
    add("jax", "debug callback", lambda x: jax.debug.print("{}", x) or x * 2,
        [np.float32(1.0)])
    return out


def reason_of(exc):
    msg = str(exc)
    hit = _REASON.search(msg)
    if hit:
        return hit.group(1).strip().rstrip(".")
    return " ".join(msg.split())[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", nargs="*", default=[],
                    help="StableHLO text modules to compile as well")
    ap.add_argument("--ops", action="store_true",
                    help="also histogram the op names those modules contain")
    args = ap.parse_args()

    os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
    dylib = pathlib.Path(os.environ["METALJAX_PLUGIN_PATH"])
    if not dylib.exists():
        sys.exit(f"plugin dylib not found: {dylib}")
    os.environ.setdefault("JAX_PLATFORMS", "metal")

    import jax
    from jax._src.lib import xla_client as xc

    rows = []
    for group, name, fn, fn_args in probes():
        try:
            jax.jit(fn).lower(*fn_args).compile()
            rows.append((group, name, None))
        except BaseException as exc:      # noqa: BLE001 - the census IS this
            rows.append((group, name, reason_of(exc)))

    if args.module:
        dev = jax.devices("metal")[0]
        for path in args.module:
            text = pathlib.Path(path).read_text()
            try:
                dev.client.compile_and_load(text, [dev], xc.CompileOptions())
                rows.append(("module", pathlib.Path(path).stem, None))
            except BaseException as exc:  # noqa: BLE001
                rows.append(("module", pathlib.Path(path).stem,
                             reason_of(exc)))

    width = max(len(n) for _, n, _ in rows) + 2
    group = None
    for g, name, why in rows:
        if g != group:
            group = g
            print(f"\n{g}")
            print("-" * (width + 46))
        print(f"{name:<{width}} {'lowers' if why is None else 'DECLINES: ' + why}")

    if args.ops and args.module:
        # A decline names the FIRST op that stopped the program, so a module
        # can hide a dozen more gaps behind one.  This is the whole surface:
        # what P4 would have to lower for these modules to run, whatever
        # order it hits them in.
        ops = collections.Counter()
        for path in args.module:
            text = pathlib.Path(path).read_text()
            ops.update(re.findall(r"\b((?:stablehlo|chlo|sdy)\.[a-z_0-9]+)",
                                  text))
        print(f"\nops present in the {len(args.module)} module(s)")
        print("-" * 62)
        for op, n in sorted(ops.items()):
            print(f"{n:>6}  {op}")

    counts = collections.Counter(why for _, _, why in rows if why is not None)
    lowered = sum(1 for _, _, why in rows if why is None)
    print(f"\n{lowered} of {len(rows)} programs lower.  Declines by reason:")
    print("-" * 62)
    for why, n in counts.most_common():
        print(f"{n:>4}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
