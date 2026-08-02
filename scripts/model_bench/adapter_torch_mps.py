"""torch-MPS comparison leg of the model benchmark suite.

PyTorch + transformers on the *same* HF checkpoints the JAX/metaljax and
mlx-lm legs use, running on Apple's Metal Performance Shaders backend.  This
is the "what people actually use on a Mac today" reference point.

Runs in its own interpreter (``torch-venv``) — torch and jax/mlx must never
share a process on this box.  See README_comparisons.md for the venv recipe
and versions.lock.md for the pinned stack.

Metrics returned by :func:`run_torch_mps` match the other adapters in
``run_bench.py``: ``load_s``, ``warmup_s``, ``prefill_ms`` (warm, 1 new
token), ``decode_ms_tok`` ((full generate - prefill) / new tokens),
``token_ids`` (first 64 greedy ids, for cross-stack agreement) and
``mem_gb_torch`` (``torch.mps.driver_allocated_memory``, the honest
device-side peak — RSS on a unified-memory box double-counts).

Standalone use (the runner owns sequencing; this is for smoke tests)::

    torch-venv/bin/python scripts/model_bench/adapter_torch_mps.py \
        --model Qwen/Qwen3-8B --device mps --decode-tokens 8 --out ids.json

TRAPS — all three cost real measurement validity, do not "simplify" them away:

1. ``PYTORCH_ENABLE_MPS_FALLBACK`` MUST be unset.  With it set, any op the
   MPS backend lacks is silently relocated to the CPU: the run still
   produces tokens, but the number it produces is a CPU/GPU hybrid and is
   not a Metal measurement.  We hard-fail instead, so an unimplemented op
   shows up as a named exception we can report.
2. ``device_map`` takes the plain string ``"mps"``.  ``torch.device("mps",0)``
   (or ``"mps:0"``) trips accelerate's device-map bookkeeping in transformers
   and fails to dispatch.
3. MPS SDPA has had correctness bugs on non-contiguous q/k/v.  Every model
   row must be gated against the same checkpoint on torch-CPU (greedy token
   ids, see ``--compare``) before its numbers are published.  ``--attn eager``
   swaps the fused kernel out if a row disagrees.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# Trap 1: refuse to run with the silent-CPU-relocation escape hatch enabled.
# Checked at import so a caller cannot set it after we look.
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
    raise RuntimeError(
        "PYTORCH_ENABLE_MPS_FALLBACK is set (%r).  It silently relocates "
        "unimplemented ops to the CPU, which corrupts every timing in this "
        "suite.  Unset it; if a model needs it, that op belongs in the "
        "report as an MPS gap, not in a benchmark number."
        % os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])

# Never redownload: the shared cache is populated by the other legs.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _repo(bench_or_model):
    """Manifest rows carry `hf://org/name`; mlx-lm strips it, so do we."""
    model = (bench_or_model["model"] if isinstance(bench_or_model, dict)
             else bench_or_model)
    return model.removeprefix("hf://")


def _mem_gb():
    import torch
    try:
        return torch.mps.driver_allocated_memory() / 1e9
    except Exception:
        return float("nan")


def render_prompt(prompt, tokenizer, template_source=None):
    """Chat-template rendering, shared across every stack.

    The comparison is only meaningful if all stacks see the *same* rendered
    prompt.  Community MLX conversions periodically ship a chat template
    that has drifted from the upstream `google/`, `Qwen/`, ... repo, so we
    render with one designated tokenizer and hand token ids to everyone.

    `template_source` (repo id, or the env var
    METALJAX_BENCH_CHAT_TEMPLATE) overrides where the template comes from
    while the tokenizer/vocab still comes from the benchmarked repo.
    Returns (text, ids) — ids are what the harness should pin.
    """
    src = template_source or os.environ.get("METALJAX_BENCH_CHAT_TEMPLATE")
    template = None
    if src:
        from transformers import AutoTokenizer
        ref = AutoTokenizer.from_pretrained(src)
        template = ref.chat_template
        if template is None:
            raise RuntimeError(f"{src} has no chat_template to borrow")

    msgs = [{"role": "user", "content": prompt}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    if template is not None:
        kw["chat_template"] = template
    if getattr(tokenizer, "chat_template", None) is None and template is None:
        return prompt, tokenizer(prompt)["input_ids"]
    text = tokenizer.apply_chat_template(msgs, **kw)
    return text, tokenizer(text, add_special_tokens=False)["input_ids"]


def load_model(repo, device="mps", attn="sdpa", dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tdtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(repo)
    kw = dict(device_map=device, attn_implementation=attn)
    # transformers >=5 renamed torch_dtype -> dtype; keep both paths working.
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=tdtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=tdtype, **kw)
    model.eval()
    return model, tok


def _pad_id(model, tok):
    """Llama-3.1 & co. carry a *list* of eos ids; generate wants a scalar."""
    for cand in (tok.pad_token_id, tok.eos_token_id,
                 model.config.eos_token_id):
        if isinstance(cand, (list, tuple)):
            cand = cand[0] if cand else None
        if isinstance(cand, int):
            return cand
    return None


def _generate(model, tok, ids, n_new, device):
    import torch
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=n_new, do_sample=False,
                             use_cache=True, pad_token_id=_pad_id(model, tok))
    return out[0].tolist()[len(ids):]


def _sync(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()


def run_torch_mps(bench, prompt, n_decode, device="mps", attn="sdpa",
                  template_source=None):
    """Adapter entry point — signature mirrors run_bench.py's adapters."""
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("torch MPS backend unavailable")

    repo = _repo(bench)
    t0 = time.monotonic()
    model, tok = load_model(repo, device=device, attn=attn)
    _sync(device)
    load_s = time.monotonic() - t0

    text, ids = render_prompt(prompt, tok, template_source)

    t0 = time.monotonic()
    _generate(model, tok, ids, 8, device)
    _sync(device)
    warmup_s = time.monotonic() - t0

    t0 = time.monotonic()
    _generate(model, tok, ids, 1, device)
    _sync(device)
    prefill_ms = 1000 * (time.monotonic() - t0)

    t0 = time.monotonic()
    new = _generate(model, tok, ids, n_decode, device)
    _sync(device)
    dt = 1000 * (time.monotonic() - t0)
    decode_ms = (dt - prefill_ms) / max(len(new) - 1, 1)

    return dict(load_s=load_s, warmup_s=warmup_s, prefill_ms=prefill_ms,
                decode_ms_tok=decode_ms, out_tokens=len(new),
                token_ids=new[:64], prompt_tokens=len(ids),
                # pin what was actually fed in, so a mirror re-syncing its
                # chat template shows up as a hash change, not as drift.
                prompt_sha=hashlib.sha256(text.encode()).hexdigest()[:16],
                mem_gb_torch=_mem_gb(), attn=attn, device=device,
                text=tok.decode(new))


def smoke(repo, prompt, n_new, device, attn):
    """Short functional check: load, generate n_new greedy tokens, no timing."""
    model, tok = load_model(repo, device=device, attn=attn)
    text, ids = render_prompt(prompt, tok)
    new = _generate(model, tok, ids, n_new, device)
    rec = dict(repo=repo, device=device, attn=attn, prompt_tokens=len(ids),
               prompt_sha=hashlib.sha256(text.encode()).hexdigest()[:16],
               token_ids=new, text=tok.decode(new),
               mem_gb_torch=_mem_gb() if device == "mps" else None)
    del model
    gc.collect()
    return rec


def compare(a, b):
    """Greedy-token agreement between two smoke records."""
    x, y = a["token_ids"], b["token_ids"]
    n = min(len(x), len(y))
    for i in range(n):
        if x[i] != y[i]:
            return {"agree": False, "first_divergence": i,
                    "a": x[:i + 1], "b": y[:i + 1]}
    return {"agree": True, "compared": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-id", help="row id in manifest.json")
    ap.add_argument("--model", help="HF repo id (overrides --bench-id)")
    ap.add_argument("--prompt")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--smoke", type=int, default=0,
                    help="functional check only: N tokens, no timings")
    ap.add_argument("--compare", metavar="JSON",
                    help="smoke record to compare token ids against")
    ap.add_argument("--out")
    ns = ap.parse_args()

    manifest = {}
    if (HERE / "manifest.json").exists():
        manifest = json.load(open(HERE / "manifest.json"))
    bench = {"model": ns.model} if ns.model else next(
        b for b in manifest["benchmarks"] if b["id"] == ns.bench_id)
    prompt = ns.prompt or manifest.get("prompt") or "Hello!"

    if ns.smoke:
        rec = smoke(_repo(bench), prompt, ns.smoke, ns.device, ns.attn)
        if ns.compare:
            rec["compare"] = compare(json.load(open(ns.compare)), rec)
        print("SMOKE " + json.dumps(rec))
    else:
        rec = {"id": ns.bench_id, "backend": f"torch-{ns.device}",
               "model": bench["model"], "date": time.strftime("%Y-%m-%d")}
        try:
            rec.update(run_torch_mps(bench, prompt, ns.decode_tokens,
                                     device=ns.device, attn=ns.attn))
            rec["mem_gb"] = round(rec["mem_gb_torch"], 1)
            rec["ok"] = True
        except Exception as e:
            import traceback
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            traceback.print_exc()
        for k in ("load_s", "warmup_s"):
            if k in rec:
                rec[k] = round(rec[k], 2)
        for k in ("prefill_ms", "decode_ms_tok"):
            if k in rec:
                rec[k] = round(rec[k], 1)
        print("RESULT " + json.dumps({k: v for k, v in rec.items()
                                      if k not in ("token_ids", "text")}))
    if ns.out:
        with open(ns.out, "w") as f:
            json.dump(rec, f)


if __name__ == "__main__":
    sys.exit(main())
