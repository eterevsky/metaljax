#!/usr/bin/env python
"""Streaming-ingest test for the native PJRT plugin's transfer path.

A model load is not an execution workload: it is thousands of host->device
transfers, tens of GB of them, with almost no programs in between.  None of
the executor's reclamation points is reached that way, which is why
`runtime.cc`'s `ingest_account` exists -- a cadence denominated in ingested
BYTES, defaulting to the 8 GB the bench harness has been giving Stage 1's
loads (`scripts/model_bench/adapter_keras_extra.py::_stream_clear`).

This test drives that path with synthetic weights and asserts the three
things a flight log has to be able to say afterwards:

  * the cadence ENGAGED, once per `METALJAX_INGEST_CLEAR_MB` of transfer,
    and can be turned off;
  * a load that KEEPS its buffers costs its bytes and no more -- resident
    memory tracks the weights, not the traffic;
  * a load that DROPS them (the churn a converter does when it stages,
    casts and discards) is bounded whatever the volume: MLX's buffer cache
    and the process's resident set both stay flat while tens of GB flow
    through.

Run it from the repo venv (a few GB of transfers, ~30 s):

    plugin-native/../.venv/bin/python plugin-native/ingest_test.py

Exit status is the number of failed checks.  `--child <json>` is how the
parent runs one measurement; nothing else needs it.

`--checkpoint <dir>` is the same path at model scale: every tensor of a real
safetensors checkpoint, mapped and transferred and kept, reporting the load's
peak.  Run that one under the machine lock and `mem_guard.sh`, like any big
load (`notes/ingest-cadence-2026-08-12.md` has the protocol and the numbers).
"""

import glob
import json
import os
import pathlib
import struct
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_DYLIB = _HERE / "bazel-bin" / "metal" / "libmetal_pjrt_native.dylib"

# One transfer's size, and how many of them each phase makes.  100 MB is a
# real weight tensor (a 32B model's embedding table is 1.4 GB, its per-layer
# projections 50-500 MB), big enough that MLX takes the alien-buffer path the
# plugin's staging block relies on.
_MB = 100
_HELD = 24        # 2.4 GB kept, as a load keeps its weights
_CHURNED = 100    # 10 GB moved and dropped, as a converter stages and casts
_CLEAR_MB = 512   # the cadence under test (the shipped default is 8192)


# --------------------------------------------------------------------------
# the child: one measurement, printed as JSON on the last line of stdout
# --------------------------------------------------------------------------


def _rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1048576.0


def child(spec):
    """Transfer `spec['n']` buffers, keeping them or not, and report."""
    import numpy as np
    import jax
    import mlx.core as mx

    gb = float(1 << 30)
    elems = spec["mb"] * (1 << 20) // 2
    host = np.zeros(elems, dtype=jax.numpy.bfloat16.dtype)
    base_rss = _rss_gb()
    held = []
    samples = []
    for i in range(spec["n"]):
        buf = jax.device_put(host)
        # The transfer is synchronous in this plugin, but ask anyway: a lazy
        # handle would make every number below a measurement of nothing.
        buf.block_until_ready()
        if spec["keep"]:
            held.append(buf)
        else:
            del buf
        samples.append((mx.get_active_memory() / gb,
                        mx.get_cache_memory() / gb, _rss_gb()))
    result = {
        "moved_gb": spec["n"] * spec["mb"] / 1024.0,
        "base_rss_gb": base_rss,
        "peak_active_gb": max(s[0] for s in samples),
        "peak_cache_gb": max(s[1] for s in samples),
        "peak_rss_gb": max(s[2] for s in samples),
        "end_active_gb": samples[-1][0],
        "end_cache_gb": samples[-1][1],
        "end_rss_gb": samples[-1][2],
        "held": len(held),
    }
    print(json.dumps(result))
    return 0


# --------------------------------------------------------------------------
# a real checkpoint: the same path, at model scale
# --------------------------------------------------------------------------
#
# `--checkpoint <dir>` streams every tensor of a real safetensors checkpoint
# through the transfer path and KEEPS it, which is a model load minus the
# model: mapped shards on one side, the plugin's staging and the device
# arrays on the other. The header is parsed here rather than through
# `safetensors.numpy` because a bf16 checkpoint has no numpy dtype -- and
# because mmap is the point: the file-backed pages a real load leaves in the
# page cache are half of the VM pressure the ingest cadence exists for.


def _shards(path):
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise SystemExit(f"no .safetensors under {path}")
    return files


def _read_shard(path):
    """Yield (name, numpy view) per tensor, over an mmap of the file."""
    import numpy as np
    import ml_dtypes

    wire = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32,
            "F64": np.float64, "I8": np.int8, "I16": np.int16,
            "I32": np.int32, "I64": np.int64, "U8": np.uint8,
            "BOOL": np.bool_}
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    base = 8 + header_len
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dt = wire.get(spec["dtype"])
        if dt is None:
            continue
        start, stop = spec["data_offsets"]
        raw = mm[base + start:base + stop]
        yield name, raw.view(dt).reshape(spec["shape"])


def checkpoint(path):
    """Load every tensor onto the device, reporting the load's memory.

    MJ_INGEST_CAST=1 puts a dtype round trip on the device between the
    transfer and the keep, which is what a real converter does (keras'
    `assign` casts to the variable's policy dtype).  That is the load shape
    where the cadence has something to reclaim: the transfer's own staging
    never enters MLX's buffer cache, but a cast's intermediate does.
    """
    import jax
    import jax.numpy as jnp
    import mlx.core as mx

    gb = float(1 << 30)
    cast = os.environ.get("MJ_INGEST_CAST") == "1"
    # MJ_INGEST_THROTTLE_GBPS caps the CUMULATIVE fill rate (0 = off): the
    # panic-#4/#7/#8 wedge class is rate-sensitive, and the row-9 ladder
    # holds the load below the rate the surviving Stage 1 run filled at.
    # Sleeps only when AHEAD of the budget line -- a slower load pays nothing.
    throttle = float(os.environ.get("MJ_INGEST_THROTTLE_GBPS", "0"))
    t0 = time.monotonic()
    throttled_s = 0.0
    base_rss = _rss_gb()
    held = []
    moved = 0
    biggest = 0
    peak_rss = base_rss
    peak_cache = 0.0
    for shard in _shards(path):
        for name, array in _read_shard(shard):
            if throttle:
                ahead = moved / gb / throttle - (time.monotonic() - t0)
                if ahead > 0:
                    time.sleep(min(ahead, 1.0))
                    throttled_s += min(ahead, 1.0)
            buf = jax.device_put(array)
            if cast:
                buf = jnp.asarray(buf, jnp.float32).astype(array.dtype)
            buf.block_until_ready()
            held.append(buf)
            moved += array.nbytes
            biggest = max(biggest, array.nbytes)
            peak_rss = max(peak_rss, _rss_gb())
            peak_cache = max(peak_cache, mx.get_cache_memory() / gb)
    result = {
        "checkpoint": path,
        "cast": cast,
        "tensors": len(held),
        "moved_gb": moved / gb,
        "largest_gb": biggest / gb,
        "base_rss_gb": base_rss,
        "peak_rss_gb": peak_rss,
        "peak_cache_gb": peak_cache,
        "throttle_gbps": throttle,
        "throttled_s": throttled_s,
        "wall_s": time.monotonic() - t0,
        "end_active_gb": mx.get_active_memory() / gb,
        "end_cache_gb": mx.get_cache_memory() / gb,
    }
    print(json.dumps(result), flush=True)
    return 0


# --------------------------------------------------------------------------
# the parent
# --------------------------------------------------------------------------


def run(name, n, keep, clear_mb, dylib):
    env = dict(os.environ)
    env.update(JAX_PLATFORMS="metal", METALJAX_PLUGIN_PATH=str(dylib),
               METALJAX_INGEST_CLEAR_MB=str(clear_mb),
               # The cadence narrates itself here; the parent counts the
               # lines, which is exactly what a load's flight log does.
               METALJAX_MEMDBG="1")
    spec = {"n": n, "mb": _MB, "keep": keep}
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()),
         "--child", json.dumps(spec)],
        capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"FAIL {name}: child exited {proc.returncode}\n{proc.stderr}")
        return None, 0
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    clears = sum(1 for ln in lines if "ingest clear #" in ln)
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        print(f"FAIL {name}: child printed no result\n{proc.stdout[-2000:]}")
        return None, 0
    result["clears"] = clears
    result["name"] = name
    return result, clears


def _expected_clears(n):
    """How often the cadence can fire when transfers arrive `_MB` at a time.

    A transfer is charged whole, so the budget is crossed by the first one to
    reach it and the effective period is `_CLEAR_MB` rounded UP to a whole
    transfer -- 600 MB here, not 512.  Weights are far from uniform in a real
    checkpoint, so this exactness is the synthetic test's alone; what the
    cadence promises a load is a clear at least every `_CLEAR_MB` plus one
    tensor.
    """
    period = -(-_CLEAR_MB // _MB) * _MB
    return (n * _MB) // period


def main():
    dylib = pathlib.Path(os.environ.get("METALJAX_PLUGIN_PATH",
                                        _DEFAULT_DYLIB))
    if not dylib.exists():
        print(f"no plugin at {dylib} (bazel build //metal:...)")
        return 1
    failures = 0
    log = []

    def check(ok, what, detail=""):
        nonlocal failures
        print(f"{'ok  ' if ok else 'FAIL'} {what}{'  ' + detail if detail else ''}")
        if not ok:
            failures += 1

    # ---- 1. a load that keeps its weights ---------------------------------
    held, clears = run("held", _HELD, True, _CLEAR_MB, dylib)
    if held is None:
        return 1
    log.append(held)
    want_clears = _expected_clears(_HELD)
    check(clears == want_clears,
          f"held: cadence fired once per {_CLEAR_MB} MB of transfer",
          f"{clears} clears, expected {want_clears}")
    # The bytes are the model's; the transfer path may not add a second copy
    # of them.  One in-flight staging block (100 MB) plus the interpreter's
    # own base is all the headroom this allows.
    predicted = held["moved_gb"] + held["base_rss_gb"] + 2 * _MB / 1024.0
    check(held["peak_rss_gb"] <= predicted,
          "held: resident set is the weights plus one transfer",
          f"peak {held['peak_rss_gb']:.2f} GB <= predicted {predicted:.2f}")
    check(held["peak_cache_gb"] < 0.25,
          "held: MLX's buffer cache stays empty",
          f"peak {held['peak_cache_gb'] * 1024:.0f} MB")

    # ---- 2. a load that drops them ----------------------------------------
    churn, clears = run("churn", _CHURNED, False, _CLEAR_MB, dylib)
    if churn is None:
        return 1
    log.append(churn)
    want_clears = _expected_clears(_CHURNED)
    check(clears == want_clears,
          f"churn: cadence fired once per {_CLEAR_MB} MB of transfer",
          f"{clears} clears over {churn['moved_gb']:.1f} GB moved, "
          f"expected {want_clears}")
    # THE bound this test exists for: 10 GB through the transfer path with
    # nothing kept must cost what one transfer costs.  A cache that grew with
    # the traffic (the failure this cadence guards against) would show here
    # as a peak proportional to `moved_gb`.
    bound = churn["base_rss_gb"] + 4 * _MB / 1024.0
    check(churn["peak_rss_gb"] <= bound,
          "churn: resident set is flat under 10 GB of traffic",
          f"peak {churn['peak_rss_gb']:.2f} GB <= bound {bound:.2f}")
    check(churn["peak_cache_gb"] < 0.25,
          "churn: MLX's buffer cache stays empty",
          f"peak {churn['peak_cache_gb'] * 1024:.0f} MB")
    check(churn["end_active_gb"] < 0.25,
          "churn: nothing is still held at the end",
          f"{churn['end_active_gb'] * 1024:.0f} MB active")

    # ---- 3. the knob ------------------------------------------------------
    off, clears = run("off", _HELD, False, 0, dylib)
    if off is None:
        return 1
    log.append(off)
    check(clears == 0, "METALJAX_INGEST_CLEAR_MB=0 disables the cadence",
          f"{clears} clears")

    out = os.environ.get("MJ_INGEST_LOG")
    if out:
        pathlib.Path(out).write_text(
            "".join(json.dumps(r) + "\n" for r in log))
        print(f"wrote {out}")
    print(f"\n{failures} failed")
    return failures


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        sys.exit(child(json.loads(sys.argv[2])))
    if len(sys.argv) > 2 and sys.argv[1] == "--checkpoint":
        sys.exit(checkpoint(sys.argv[2]))
    sys.exit(main())
