"""Real-model benchmark runner (public suite).

    bench-venv/bin/python scripts/model_bench/run_bench.py <bench-id> <backend>
        [--out results.jsonl] [--decode-tokens N]

One benchmark, one backend, one PROCESS — the parent script (run_all.py)
handles sequencing, process isolation and the memory watchdog. Backends:

  metaljax  JAX_PLATFORMS=metal through the metaljax plugin
  cpu       JAX_PLATFORMS=cpu (jax's own CPU backend)
  mlx       mlx-lm on the same HF checkpoint (comparison stack)

Metrics (LLM rows): load_s; warmup_s (first 8-token generate — compile);
prefill_ms (warm 1-token generate on the standard prompt); decode_ms_tok
((full generate - prefill)/N); mem_gb (device-active for metal, peak RSS
otherwise); token_ids of the first 64 greedy tokens for cross-backend
agreement checks.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = json.load(open(HERE / "manifest.json"))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def mem_gb(backend):
    if backend == "metaljax":
        import mlx.core as mx
        return mx.get_active_memory() / 1e9
    return rss_gb()


# ------------------------------------------------------------------ adapters

def run_gemma_lib(bench, backend, prompt, n_decode):
    """DeepMind gemma library + our HF-safetensors converter."""
    sys.path.insert(0, str(HERE))
    # MUST run in a TF-free venv (the scratchpad gemma-venv): importing
    # tensorflow poisons pip sentencepiece, and gemma's tokenizer enters
    # through LoadFromSerializedProto, which the keras-path shim does not
    # cover. bench-venv's kauldron also imports TF unconditionally, so
    # there is no in-process fix — the venv split is the fix.
    if "tensorflow" in sys.modules:
        raise RuntimeError("run_gemma_lib requires a tensorflow-free venv "
                           "(use the gemma-venv; see comment)")
    from gemma_loader import load_gemma4  # shared with the 0.11.0 work
    import jax

    t0 = time.monotonic()
    model, params, tok = load_gemma4(bench["model"])
    load_s = time.monotonic() - t0

    from gemma import gm
    sampler = gm.text.ChatSampler(model=model, params=params, tokenizer=tok,
                                  cache_length=1024, max_out_length=512)
    t0 = time.monotonic()
    sampler.chat(prompt, max_new_tokens=8)
    warmup_s = time.monotonic() - t0

    t0 = time.monotonic()
    sampler.chat(prompt, max_new_tokens=1)
    prefill_ms = 1000 * (time.monotonic() - t0)

    t0 = time.monotonic()
    reply = sampler.chat(prompt, max_new_tokens=n_decode)
    dt = time.monotonic() - t0
    ntok = len(tok.encode(reply))
    decode_ms = (1000 * dt - prefill_ms) / max(ntok - 1, 1)
    ids = tok.encode(reply)[:64]
    return dict(load_s=load_s, warmup_s=warmup_s, prefill_ms=prefill_ms,
                decode_ms_tok=decode_ms, out_tokens=ntok, token_ids=ids)


def run_keras_lm(bench, backend, prompt, n_decode, quant=None):
    import keras
    keras.config.set_dtype_policy("bfloat16")
    # `import tensorflow` (pulled in by keras-hub tokenizer metadata paths)
    # poisons pip sentencepiece: SentencePieceProcessor.Init SIGABRTs
    # ("mutex lock failed") — that killed the gemma4-e2b rows. The shim
    # routes the native call through the tf-text tokenizer instead.
    import adapter_keras_extra as extra
    extra.patch_sentencepiece_native()
    import keras_hub

    # Stream the checkpoint into the model instead of building a full set of
    # random weights first (see install_streaming_load): that build phase is
    # what put the >=50 GB rows over the machine — the peak is not
    # "weights + weights", it is "weights + ~30x the largest variable",
    # because our interpreter holds every intermediate of the initializer's
    # threefry chain live at once. Weights land bit-identical either way.
    stream = os.environ.get("BENCH_STREAM_LOAD", "1") == "1"
    stream_info = extra.install_streaming_load() if stream else None

    # Checkpoints whose weights are natively sub-byte: load them PACKED and
    # let the model dequantize in the graph, where metaljax's recognizer
    # folds the chain into one quantized matmul. Without this the keras
    # loader expands them to bfloat16 (gpt-oss-20b: 13.8 GB -> 41.8 GB).
    native = bench.get("native_quant")
    if native == "mxfp4":
        import mxfp4_gpt_oss
        print("[bench] native MXFP4 loading: "
              f"{'on' if mxfp4_gpt_oss.install() else 'OFF (env override)'}",
              flush=True)
    elif native == "mlx3bit":
        # mlx-community's affine 3-bit: uint32 codes packed as a bit STREAM
        # (elements straddle word boundaries), bf16 scales AND biases. The
        # engine repacks it to bytes identical to the file and then aliases
        # the argument, so the 96 GB checkpoint is held once, not twice.
        import qwen3_moe_mlx3bit
        print("[bench] native MLX 3-bit loading: "
              f"{'on' if qwen3_moe_mlx3bit.install() else 'OFF (env override)'}",
              flush=True)
    elif native:
        raise ValueError(f"unknown native_quant {native!r}")

    cls = getattr(keras_hub.models, bench["arch"])
    # keras-hub wants an "hf://" preset; the mlx arm of the same row wants the
    # bare repo id for mlx-lm. Rows that need both carry "preset".
    preset = bench.get("preset") or bench["model"]
    t0 = time.monotonic()
    retry_tokenizer = False
    try:
        lm = cls.from_preset(preset)
    except ValueError as e:
        # Checkpoints that renamed keras-hub's hardcoded special tokens
        # (DeepSeek R1 distills drop Qwen's <|endoftext|>): rebuild the
        # tokenizer around the checkpoint's own EOS, then load the
        # backbone separately. The retry must run OUTSIDE this block:
        # from_preset raises AFTER assigning the full checkpoint (61 GB
        # on R1-32B), and the live traceback pins that dead model while
        # a second copy loads -- the two stacked to a 93 GB guard kill.
        # Same trap as the engine's execute retries (CLAUDE.md item 17).
        if "Cannot find special token" not in str(e):
            raise
        retry_tokenizer = True
    if retry_tokenizer:
        # keras models are refcycles; without a collect the failed
        # attempt's buffers survive the except block. The collect drops
        # them into MLX's cache, which the streaming clears (or the next
        # allocations) then reuse -- measured: the retry load runs FLAT
        # at ~70 GB instead of stacking to 130+.
        import gc
        gc.collect()
        from adapter_keras_extra import _bpe_tokenizer_with_eos_alias
        pre_cls = cls.preprocessor_cls
        tok, _ = _bpe_tokenizer_with_eos_alias(pre_cls.tokenizer_cls,
                                               preset)
        backbone = cls.backbone_cls.from_preset(preset)
        lm = cls(backbone=backbone, preprocessor=pre_cls(tokenizer=tok))
    if stream:
        # Any variable the checkpoint did not cover gets its real random
        # init here, loudly; then keras goes back to stock for quantize().
        stream_info = extra.finalize_streaming_load(lm)
        extra.uninstall_streaming_load()
    if quant:
        try:
            lm.quantize(quant)
        except ValueError as e:
            if "isn't yet built" not in str(e):
                raise
            # Multimodal wrappers carry never-built vision layers in
            # text-only use; the language backbone is the decode path.
            lm.backbone.quantize(quant)
    # Pin the documented contract: token_ids are GREEDY tokens. keras-hub's
    # from_preset auto-compiles with sampler="top_k" (k=5, seeded per
    # process from Python's `random` — task.py:59, causal_lm.py:69), so
    # stock generate() was stochastic until 2026-08-26; token comparisons
    # recorded before then sampled top-5 draws, not argmax chains, and the
    # captured 64 ids start with the ~50-token prompt.
    lm.compile(sampler="greedy")
    load_s = time.monotonic() - t0

    # token count of the prompt, for prefill bookkeeping
    plen = len(lm.preprocessor.tokenizer(prompt))

    t0 = time.monotonic()
    lm.generate(prompt, max_length=plen + 8)
    warmup_s = time.monotonic() - t0

    # Absorb executable builds for BOTH timed shapes before timing them:
    # keras-hub compiles a separate generate function per max_length, and a
    # shape's first call pays jit + engine compile + (on quantized rows) the
    # whole weight-pack wave inside the timed window -- on gpt-oss-20b that
    # read as 96 s of "prefill" and +24 ms/tok of "decode" for one-time
    # work the mlx-lm and torch columns never pay in theirs. The metric is
    # steady-state latency; the one-time cost stays visible as build_s.
    t0 = time.monotonic()
    lm.generate(prompt, max_length=plen + 1)
    lm.generate(prompt, max_length=plen + n_decode)
    build_s = time.monotonic() - t0

    t0 = time.monotonic()
    lm.generate(prompt, max_length=plen + 1)
    prefill_ms = 1000 * (time.monotonic() - t0)

    t0 = time.monotonic()
    out = lm.generate(prompt, max_length=plen + n_decode)
    dt = time.monotonic() - t0
    ids = [int(t) for t in lm.preprocessor.tokenizer(out)][:64]
    n_new = len(lm.preprocessor.tokenizer(out)) - plen
    decode_ms = (1000 * dt - prefill_ms) / max(n_new - 1, 1)
    return dict(load_s=load_s, warmup_s=warmup_s, build_s=build_s,
                prefill_ms=prefill_ms, decode_ms_tok=decode_ms,
                out_tokens=n_new, token_ids=ids,
                prompt_tokens=plen, stream_load=stream_info)


def run_mlx(bench, backend, prompt, n_decode):
    """mlx-lm on the same checkpoint: the same-Metal-library reference."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model_ref = bench["model"].removeprefix("hf://")
    t0 = time.monotonic()
    model, tok = load(model_ref)
    load_s = time.monotonic() - t0

    sampler = make_sampler(temp=0.0)  # greedy, like the JAX rows
    t0 = time.monotonic()
    generate(model, tok, prompt=prompt, max_tokens=8, sampler=sampler)
    warmup_s = time.monotonic() - t0

    t0 = time.monotonic()
    generate(model, tok, prompt=prompt, max_tokens=1, sampler=sampler)
    prefill_ms = 1000 * (time.monotonic() - t0)

    t0 = time.monotonic()
    out = generate(model, tok, prompt=prompt, max_tokens=n_decode,
                   sampler=sampler)
    dt = time.monotonic() - t0
    ids = tok.encode(out)[:64]
    n_new = len(tok.encode(out))
    decode_ms = (1000 * dt - prefill_ms) / max(n_new - 1, 1)
    return dict(load_s=load_s, warmup_s=warmup_s, prefill_ms=prefill_ms,
                decode_ms_tok=decode_ms, out_tokens=n_new, token_ids=ids,
                mem_gb_mlx=mx.get_active_memory() / 1e9)


def _extra(name):
    def call(b, bk, p, n):
        sys.path.insert(0, str(HERE))
        import adapter_keras_extra as x
        return getattr(x, name)(b, bk, p, n)
    return call


ADAPTERS = {
    "gemma_lib": run_gemma_lib,
    "keras_lm": run_keras_lm,
    "keras_lm_quant": lambda b, bk, p, n: run_keras_lm(b, bk, p, n,
                                                       quant=b["quant"]),
    "keras_vision": _extra("run_keras_vision"),
    "keras_diffusion": _extra("run_keras_diffusion"),
    "keras_lora_train": _extra("run_keras_lora_train"),
    "mlx_only": run_mlx,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_id")
    ap.add_argument("backend", choices=["metaljax", "cpu", "mlx"])
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    ap.add_argument("--decode-tokens", type=int,
                    default=MANIFEST["decode_tokens"])
    ns = ap.parse_args()

    bench = next(b for b in MANIFEST["benchmarks"] if b["id"] == ns.bench_id)
    if bench.get("status", "").startswith(("needs", "blocked")) \
            and ns.backend != "mlx":
        sys.exit(f"{ns.bench_id}: {bench['status']}")

    if ns.backend == "mlx":
        fn = run_mlx
    else:
        os.environ.setdefault(
            "JAX_PLATFORMS", "metal" if ns.backend == "metaljax" else "cpu")
        os.environ.setdefault("KERAS_BACKEND", "jax")
        if bench["path"] in ("maxtext", "maxtext_train"):
            # Lazy: the maxtext rows run in their own venv
            # (~/.cache/metaljax-bench/maxtext/venv, see README_maxtext.md
            # and setup_maxtext.sh) whose deps the bench venv does not
            # carry -- importing at module scope would break every other
            # row there.
            from adapter_maxtext import run_maxtext, run_maxtext_train
            fn = (run_maxtext_train if bench["path"] == "maxtext_train"
                  else run_maxtext)
        else:
            fn = ADAPTERS[bench["path"]]

    t0 = time.monotonic()
    rec = {"id": ns.bench_id, "backend": ns.backend,
           "model": bench["model"], "size_gb": bench.get("size_gb"),
           "date": time.strftime("%Y-%m-%d")}
    try:
        rec.update(fn(bench, ns.backend, MANIFEST["prompt"],
                      ns.decode_tokens))
        rec["mem_gb"] = round(mem_gb(ns.backend), 1)
        rec["ok"] = True
    except Exception as e:
        import traceback
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        traceback.print_exc()
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    for k in ("load_s", "warmup_s"):
        if k in rec:
            rec[k] = round(rec[k], 2)
    for k in ("prefill_ms", "decode_ms_tok"):
        if k in rec:
            rec[k] = round(rec[k], 1)
    with open(ns.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT " + json.dumps({k: v for k, v in rec.items()
                                  if k != "token_ids"}), flush=True)


if __name__ == "__main__":
    main()
