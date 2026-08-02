"""llama.cpp comparison column for the model benchmark suite.

    python scripts/model_bench/adapter_llamacpp.py <row-id> [options]
    python scripts/model_bench/adapter_llamacpp.py --list
    python scripts/model_bench/adapter_llamacpp.py --all

Runs `llama-bench` (Metal, GPU-offloaded) on a GGUF quant of a suite model
and emits the same JSONL record shape as run_bench.py, with
`backend="llamacpp"`.  One row = one model file = one process.

Metric mapping (llama-bench reports throughput, the suite reports latency):

    prefill_ms     = 1000 * n_prompt / pp_tok_s      (whole 51-token prefill)
    decode_ms_tok  = 1000 / tg_tok_s                 (warm per-token decode)
    load_s         = llama-cli's `load time`, warm page cache
    mem_gb         = sum of the Metal buffers llama.cpp reports (weights + KV
                     + compute), i.e. device-active memory, comparable to the
                     metaljax column's mx.get_active_memory()

llama-bench does its own internal warmup run, so `warmup_s` is null: there is
no compile step to amortise the way the JAX rows have.

Cross-backend token agreement is NOT checked here: these rows run quantized
weights through a different tokenizer/chat template, so greedy ids cannot
match the bf16 JAX rows by construction.  Instead each row keeps the first
part of a greedy (`--temp 0`) llama-cli generation in `sanity_text` for a
human coherence check.

Requires: a llama.cpp build with Metal (>= b9493 for Gemma 4).  Point at it
with --llamacpp-bin or $LLAMACPP_BIN; otherwise llama-bench/llama-cli are
taken from $PATH.  GGUFs resolve through the shared ~/.cache/huggingface via
huggingface_hub (downloaded on demand), or pass --gguf explicitly.

Machine discipline: every timed run takes /tmp/metaljax-bench.lock (the
suite-wide "one GPU job at a time" lock) and releases it on exit.  Rows whose
weights exceed --max-weight-gb (default 65) refuse to run, and the run waits
if swap usage is above --max-swap-gb (default 20).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = json.load(open(HERE / "manifest.json"))
LOCK = "/tmp/metaljax-bench.lock"

# --------------------------------------------------------------------- rows
# bench      : manifest benchmark id this row compares against (STATUS row)
# repo/file  : the GGUF actually measured
# revision   : pinned commit of the GGUF repo (recorded at measurement time)
ROWS = {
    "gemma4-12b-bf16": dict(
        bench="gemma4-12b-bf16", quant="BF16",
        repo="ggml-org/gemma-4-12B-it-GGUF",
        file="gemma-4-12B-it-BF16.gguf",
        revision="7e0fbb8205d1f4857f4606a38a65023aaeb5f544"),
    "gemma4-12b-q4": dict(
        bench="gemma4-12b-bf16", quant="Q4_0 (QAT)",
        repo="google/gemma-4-12B-it-qat-q4_0-gguf",
        file="gemma-4-12b-it-qat-q4_0.gguf",
        revision="29d097773436b69ff9feafd636ab4cf873786537"),
    "gemma4-31b-bf16": dict(
        bench="gemma4-31b-bf16", quant="BF16",
        repo="ggml-org/gemma-4-31B-it-GGUF",
        file="gemma-4-31B-it-BF16.gguf",
        revision="4fa4fdf38bee237b5c9e8a5b4e72cf39404c9dcc"),
    "gemma4-31b-q8": dict(
        bench="gemma4-31b-bf16", quant="Q8_0",
        repo="ggml-org/gemma-4-31B-it-GGUF",
        file="gemma-4-31B-it-Q8_0.gguf",
        revision="4fa4fdf38bee237b5c9e8a5b4e72cf39404c9dcc"),
    "gemma4-31b-q4": dict(
        bench="gemma4-31b-bf16", quant="Q4_0 (QAT)",
        repo="google/gemma-4-31B-it-qat-q4_0-gguf",
        file="gemma-4-31B_q4_0-it.gguf",
        revision="59dde24573e7e61570dba08b18a2e1fe246955ed"),
    "gemma4-26b-a4b-q4": dict(
        bench="gemma4-26b-a4b", quant="Q4_0 (QAT)",
        repo="google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
        file="gemma-4-26B_q4_0-it.gguf",
        revision="d1c082be9cf3c8a514acf63b8761f4b41935842e"),
    "qwen3-8b-q8": dict(
        bench="qwen3-8b-bf16", quant="Q8_0",
        repo="Qwen/Qwen3-8B-GGUF", file="Qwen3-8B-Q8_0.gguf",
        revision="7c41481f57cb95916b40956ab2f0b139b296d974"),
    "qwen3-8b-q4": dict(
        bench="qwen3-8b-bf16", quant="Q4_K_M",
        repo="Qwen/Qwen3-8B-GGUF", file="Qwen3-8B-Q4_K_M.gguf",
        revision="7c41481f57cb95916b40956ab2f0b139b296d974"),
    "llama31-8b-q8": dict(
        bench="llama31-8b-bf16", quant="Q8_0",
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        file="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        revision="bf5b95e96dac0462e2a09145ec66cae9a3f12067"),
    "llama31-8b-q4": dict(
        bench="llama31-8b-bf16", quant="Q4_K_M",
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        file="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        revision="bf5b95e96dac0462e2a09145ec66cae9a3f12067"),
    "gpt-oss-20b-mxfp4": dict(
        bench="gpt-oss-20b", quant="MXFP4 (native)",
        repo="ggml-org/gpt-oss-20b-GGUF", file="gpt-oss-20b-MXFP4.gguf",
        revision="ef9b12f2ff56c69cf32153a02784e7a3c88bf524"),
}

ORDER = ["gemma4-12b-bf16", "gemma4-12b-q4", "gemma4-31b-bf16",
         "gemma4-31b-q8", "gemma4-31b-q4", "gemma4-26b-a4b-q4",
         "qwen3-8b-q8", "qwen3-8b-q4", "llama31-8b-q8", "llama31-8b-q4",
         "gpt-oss-20b-mxfp4"]


# ----------------------------------------------------------------- machine

class MachineLock:
    """The suite-wide 'one GPU job at a time' lock (mkdir is atomic)."""

    def __init__(self, enabled=True, path=LOCK, poll=10):
        self.enabled, self.path, self.poll, self.held = enabled, path, poll, False

    def __enter__(self):
        if not self.enabled:
            return self
        t0 = time.monotonic()
        while True:
            try:
                os.mkdir(self.path)
                break
            except FileExistsError:
                time.sleep(self.poll)
        self.held = True
        print(f"[lock] acquired after {time.monotonic() - t0:.0f}s",
              file=sys.stderr, flush=True)
        return self

    def __exit__(self, *exc):
        if self.held:
            try:
                os.rmdir(self.path)
            except OSError:
                pass
            self.held = False
        return False


def swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", out)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v / 1024 if m.group(2) == "M" else v


BENCH_PROC = re.compile(
    r"metaljax-bench|model_bench/adapter_|model_bench/run_bench|llama-bench")


def busy_bench_procs(min_rss_gb=2.0):
    """Other benchmark processes currently holding real memory.

    The machine lock serialises *scheduled* jobs; this is the direct check
    that matters for the failure mode the lock exists to prevent (two big
    models resident at once).  Our own process tree is excluded.
    """
    out = subprocess.run(["ps", "-eo", "pid,ppid,rss,command"],
                         capture_output=True, text=True).stdout
    mine = {os.getpid(), os.getppid()}
    busy = []
    for line in out.splitlines()[1:]:
        f = line.split(None, 3)
        if len(f) < 4:
            continue
        pid, ppid, rss, cmd = int(f[0]), int(f[1]), int(f[2]), f[3]
        if pid in mine or ppid in mine or "adapter_llamacpp" in cmd:
            continue
        gb = rss / 2**20
        if gb >= min_rss_gb and BENCH_PROC.search(cmd):
            busy.append((pid, round(gb, 1), cmd[:70]))
    return busy


def wait_for_idle(min_rss_gb=2.0, poll=30, tries=60):
    """Block until no other benchmark process is holding real memory."""
    for _ in range(tries):
        busy = busy_bench_procs(min_rss_gb)
        if not busy:
            return
        print(f"[wait] other bench processes resident: {busy}",
              file=sys.stderr, flush=True)
        time.sleep(poll)
    raise RuntimeError("other benchmark processes stayed resident")


def wait_for_swap(limit_gb, poll=30, tries=40):
    for _ in range(tries):
        s = swap_used_gb()
        if s <= limit_gb:
            return s
        print(f"[wait] swap {s:.1f} GB > {limit_gb} GB; sleeping",
              file=sys.stderr, flush=True)
        time.sleep(poll)
    raise RuntimeError(f"swap stayed above {limit_gb} GB")


# ------------------------------------------------------------------- model

def resolve_gguf(row, offline=False):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(row["repo"], row["file"], revision=row["revision"],
                           local_files_only=offline)


def bin_path(name, root=None):
    if root:
        p = Path(root) / name
        if p.exists():
            return str(p)
        p = Path(root) / "bin" / name
        if p.exists():
            return str(p)
        raise SystemExit(f"{name} not found under {root}")
    p = shutil.which(name)
    if not p:
        raise SystemExit(f"{name} not on PATH; use --llamacpp-bin")
    return p


def build_id(exe):
    """llama.cpp build string.

    `llama-bench` has no --version; ask the completion/cli binary, and read
    both streams (the version banner goes to stderr, behind Metal init
    chatter).
    """
    p = subprocess.run([exe, "--version"], capture_output=True, text=True)
    out = (p.stderr or "") + "\n" + (p.stdout or "")
    m = re.search(r"version:\s*(\S+)\s*\(([0-9a-f]+)\)", out)
    # NB the number is git rev-list --count, which a shallow clone truncates;
    # the commit hash is the reliable identifier.
    return f"{m.group(1)} ({m.group(2)})" if m else "unknown"


# ----------------------------------------------------------------- parsing

# The Metal device is named MTL0 (mmap'd weights land in "MTL0_Mapped").
# Only lines with "=" are real allocations; the "buffer size is ... matches
# expectation" lines are the context destructor's accounting.
METAL_BUF = re.compile(
    r"MTL0\S*\s+(model|KV|compute|output)\s+buffer size\s*=\s*([\d.]+)\s*MiB")


def metal_mem_gb(stderr):
    """Device-active memory: the Metal buffers llama.cpp allocates.

    llama-bench builds one context per test (pp and tg), so each buffer
    class is reported more than once; take the largest of each rather than
    summing, which would double-count.
    """
    peak = {}
    for line in stderr.splitlines():
        m = METAL_BUF.search(line)
        if m:
            kind, mib = m.group(1), float(m.group(2))
            peak[kind] = max(peak.get(kind, 0.0), mib)
    return sum(peak.values()) * 2**20 / 1e9 if peak else None


def parse_bench_json(stdout):
    """llama-bench -o json -> {'pp': (tok_s, stddev), 'tg': (tok_s, stddev)}."""
    start = stdout.find("[")
    if start < 0:
        raise RuntimeError("no JSON in llama-bench output")
    rows = json.loads(stdout[start:])
    out = {}
    for r in rows:
        n_prompt, n_gen = int(r["n_prompt"]), int(r["n_gen"])
        kind = "pp" if n_prompt and not n_gen else "tg"
        out[kind] = (float(r["avg_ts"]), float(r.get("stddev_ts") or 0.0),
                     n_prompt or n_gen)
    return out


LOAD_T = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")


def parse_load_s(stderr):
    m = LOAD_T.search(stderr)
    return float(m.group(1)) / 1000 if m else None


# -------------------------------------------------------------------- run

def run_row(row_id, args):
    row = ROWS[row_id]
    bench = next((b for b in MANIFEST["benchmarks"]
                  if b["id"] == row["bench"]), {})
    prompt = MANIFEST["prompt"]
    n_decode = args.decode_tokens

    gguf = args.gguf or resolve_gguf(row, args.offline)
    weight_gb = os.path.getsize(gguf) / 1e9
    if weight_gb > args.max_weight_gb:
        raise RuntimeError(f"{row_id}: weights {weight_gb:.1f} GB exceed the "
                           f"{args.max_weight_gb} GB machine cap")

    bench_bin = bin_path("llama-bench", args.llamacpp_bin)
    # llama-cli in recent builds is a full-screen interactive app: it renders
    # a TUI, never prints the perf block, and blocks on stdin.  The scriptable
    # completion path is a separate binary.
    cli_bin = bin_path("llama-completion", args.llamacpp_bin)

    rec = {"id": row["bench"], "backend": "llamacpp", "row": row_id,
           "model": bench.get("model", row["repo"]),
           "size_gb": bench.get("size_gb"),
           "date": time.strftime("%Y-%m-%d"),
           "gguf_repo": row["repo"], "gguf_file": row["file"],
           "gguf_revision": row["revision"], "quant": row["quant"],
           "gguf_gb": round(weight_gb, 2),
           "build": build_id(cli_bin),
           "n_prompt": args.n_prompt, "n_decode": n_decode,
           "reps": args.reps}

    t_wall = time.monotonic()
    with MachineLock(enabled=not args.no_lock):
        if args.idle_guard:
            wait_for_idle(args.idle_rss_gb)
        rec["swap_gb_at_start"] = round(wait_for_swap(args.max_swap_gb), 2)

        # -v only adds log lines (the Metal buffer report we read for
        # mem_gb); it does not change what is timed.
        cmd = [bench_bin, "-m", gguf, "-p", str(args.n_prompt),
               "-n", str(n_decode), "-r", str(args.reps), "-o", "json", "-v"]
        if args.depth:
            # decode starting from a pre-filled context instead of an empty
            # one, matching the suite's "prompt then decode" shape
            cmd += ["-d", str(args.depth)]
            rec["depth"] = args.depth
        rec["cmd"] = " ".join(cmd)
        print("[run ] " + rec["cmd"], file=sys.stderr, flush=True)
        t0 = time.monotonic()
        p = subprocess.run(cmd, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=args.timeout_s)
        rec["bench_wall_s"] = round(time.monotonic() - t0, 1)
        if p.returncode != 0:
            raise RuntimeError(
                f"llama-bench exit {p.returncode}: "
                + "\n".join(p.stderr.strip().splitlines()[-8:]))

        res = parse_bench_json(p.stdout)
        pp_ts, pp_sd, pp_n = res["pp"]
        tg_ts, tg_sd, _ = res["tg"]
        rec.update(
            pp_tok_s=round(pp_ts, 2), pp_stddev_tok_s=round(pp_sd, 2),
            tg_tok_s=round(tg_ts, 2), tg_stddev_tok_s=round(tg_sd, 2),
            prefill_ms=round(1000 * pp_n / pp_ts, 1),
            decode_ms_tok=round(1000 / tg_ts, 2),
            warmup_s=None, out_tokens=n_decode)
        mem = metal_mem_gb(p.stderr)
        if mem:
            rec["mem_gb"] = round(mem, 1)

        if not args.no_sanity:
            # --jinja: Gemma 4's chat template is not one of the built-in
            # formats ("this custom template is not supported, try using
            # --jinja"), and without it llama-completion aborts before
            # generating a single token.
            scmd = [cli_bin, "-m", gguf, "-st", "-p", prompt,
                    "-n", str(args.sanity_tokens), "--temp", "0",
                    "--no-warmup", "--no-display-prompt", "--jinja",
                    "-no-cnv" if args.raw else "-cnv"]
            rec["sanity_cmd"] = " ".join(scmd)
            t0 = time.monotonic()
            try:
                s = subprocess.run(scmd, capture_output=True, text=True,
                                   stdin=subprocess.DEVNULL,
                                   timeout=args.sanity_timeout_s)
            except subprocess.TimeoutExpired:
                rec["sanity_ok"] = False
                rec["sanity_text"] = "TIMEOUT"
                s = None
            rec["sanity_wall_s"] = round(time.monotonic() - t0, 1)
            if s is None:
                rec["wall_s"] = round(time.monotonic() - t_wall, 1)
                rec["ok"] = True
                return rec
            rec["load_s"] = parse_load_s(s.stderr)
            text = (s.stdout or "").strip()
            rec["sanity_text"] = text[:args.sanity_chars]
            rec["sanity_ok"] = bool(len(text) > 40)
    rec["wall_s"] = round(time.monotonic() - t_wall, 1)
    rec["ok"] = True
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="*", help="row ids (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    ap.add_argument("--decode-tokens", type=int,
                    default=MANIFEST["decode_tokens"])
    ap.add_argument("--n-prompt", type=int, default=51,
                    help="prefill length; the manifest prompt is 50-51 "
                         "tokens depending on tokenizer")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--depth", type=int, default=0,
                    help="llama-bench -d: decode from a context this deep "
                         "(0 = llama-bench default, empty context)")
    ap.add_argument("--gguf", help="explicit .gguf path (single row only)")
    ap.add_argument("--llamacpp-bin", default=os.environ.get("LLAMACPP_BIN"),
                    help="directory holding llama-bench/llama-cli")
    ap.add_argument("--offline", action="store_true",
                    help="require the GGUF to be in the HF cache already")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--lock-once", action="store_true",
                    help="hold the machine lock across the whole row list "
                         "instead of re-taking it per row; use for bulk runs "
                         "when other agents hold the lock in long blocks")
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="sanity generation without the chat template")
    ap.add_argument("--sanity-tokens", type=int, default=128)
    ap.add_argument("--sanity-chars", type=int, default=600)
    ap.add_argument("--timeout-s", type=float, default=5400.0,
                    help="llama-bench wall limit")
    ap.add_argument("--sanity-timeout-s", type=float, default=900.0,
                    help="llama-cli coherence-check wall limit")
    ap.add_argument("--max-weight-gb", type=float, default=65.0)
    ap.add_argument("--max-swap-gb", type=float, default=20.0)
    ap.add_argument("--idle-guard", action="store_true",
                    help="before each row, wait until no other benchmark "
                         "process is holding >--idle-rss-gb of memory")
    ap.add_argument("--idle-rss-gb", type=float, default=2.0)
    ns = ap.parse_args()

    if ns.list:
        for r in ORDER:
            row = ROWS[r]
            print(f"{r:22s} {row['quant']:16s} {row['repo']}/{row['file']}")
        return
    rows = ORDER if ns.all else ns.rows
    if not rows:
        ap.error("give at least one row id, or --all / --list")
    if ns.gguf and len(rows) != 1:
        ap.error("--gguf applies to a single row")

    for row_id in rows:
        if row_id not in ROWS:
            sys.exit(f"unknown row {row_id!r}; --list to see them")

    outer = MachineLock(enabled=ns.lock_once and not ns.no_lock)
    if ns.lock_once:
        ns.no_lock = True          # run_row must not re-take it
    with outer:
        sys.exit(1 if run_rows(rows, ns) else 0)


def run_rows(rows, ns):
    """Measure each row, append its record, return the failure count."""
    fail = 0
    for row_id in rows:
        rec = {"id": ROWS[row_id]["bench"], "backend": "llamacpp",
               "row": row_id, "date": time.strftime("%Y-%m-%d")}
        try:
            rec = run_row(row_id, ns)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"[:600]
            fail += 1
        with open(ns.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("RESULT " + json.dumps({k: v for k, v in rec.items()
                                      if k != "sanity_text"}), flush=True)
    return fail


if __name__ == "__main__":
    main()
