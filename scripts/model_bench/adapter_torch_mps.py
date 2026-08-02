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

NON-DECODE ROWS.  Three STATUS rows are not causal-LM decode; each is a
subcommand whose metrics mirror the metaljax/keras cell it is compared
against (`adapter_keras_extra.py`), so the columns are like-for-like::

    ... adapter_torch_mps.py vision      # row 16, SigLIP 2 so400m fwd
    ... adapter_torch_mps.py lora        # row 18, Gemma 4 E2B LoRA train
    ... adapter_torch_mps.py diffusion   # row 17, SD 3.5 Large

The legacy flat CLI above is still accepted verbatim (it is the implicit
`decode` subcommand), so existing callers do not change.  Each non-decode
subcommand appends one JSONL record to
``~/.cache/metaljax-bench/logs/results_new.jsonl``, takes the suite-wide
`/tmp/metaljax-bench.lock` (one GPU job at a time, as `adapter_llamacpp.py`
does) and refuses to start while swap is above --max-swap-gb.

TIMING.  Unlike the metaljax backend -- where `jax.block_until_ready` is a
no-op and only a host materialisation is a real barrier -- `torch.mps.
synchronize()` IS a true barrier, so it is what closes every timed region
here.  Training steps additionally pull `loss.item()`, which syncs anyway.

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
import atexit
import gc
import hashlib
import json
import os
import signal
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


# =====================================================================
# Non-decode rows.  Everything below serves the three STATUS cells that
# are not causal-LM decode; the metric names deliberately mirror
# `adapter_keras_extra.py` so the STATUS columns compare like for like.
# =====================================================================

LOCK = "/tmp/metaljax-bench.lock"
LOGDIR = Path(os.path.expanduser("~/.cache/metaljax-bench/logs"))

# Pinned so a re-measurement is reproducible even if a repo moves.
SIGLIP_REPO = "google/siglip2-so400m-patch14-384"
SIGLIP_REV = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
LORA_REPO = "google/gemma-4-E2B-it"
# stabilityai/stable-diffusion-3.5-large is GATED (HTTP 401 without an
# accepted licence + token).  This is an ungated mirror of the same
# diffusers tree, uploaded a week after the model's release.
SD35_REPO = "adamo1139/stable-diffusion-3.5-large-ungated"
SD35_REV = "5d868ffde1c2396697cc1ab7555dc5f64e056a63"
SD35_UPSTREAM = "stabilityai/stable-diffusion-3.5-large"


class MachineLock:
    """The suite-wide 'one GPU job at a time' lock (mkdir is atomic).

    Same path and protocol as `adapter_llamacpp.py` -- two kernel panics
    this week came from concurrent GPU jobs, and a second job also makes
    every timing here meaningless.
    """

    def __init__(self, enabled=True, path=LOCK, poll=10):
        self.enabled, self.path, self.poll = enabled, path, poll
        self.held = False

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
        # A `with` block does not survive SIGTERM/SIGINT, and a leaked lock
        # dir wedges every later job in the suite (they poll forever), so
        # release on the way out of both.
        atexit.register(self.release)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except ValueError:      # not on the main thread
                pass
        print(f"[lock] acquired after {time.monotonic() - t0:.0f}s",
              file=sys.stderr, flush=True)
        return self

    def _on_signal(self, signum, frame):
        self.release()
        sys.exit(128 + signum)

    def release(self):
        if self.held:
            try:
                os.rmdir(self.path)
            except OSError:
                pass
            self.held = False

    def __exit__(self, *exc):
        self.release()
        return False


def swap_used_gb():
    import re
    import subprocess

    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", out)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v / 1024 if m.group(2) == "M" else v


def wait_for_swap(limit_gb, poll=30, tries=40):
    for _ in range(tries):
        s = swap_used_gb()
        if s <= limit_gb:
            return s
        print(f"[wait] swap {s:.1f} GB > {limit_gb} GB; sleeping",
              file=sys.stderr, flush=True)
        time.sleep(poll)
    raise RuntimeError(f"swap stayed above {limit_gb} GB")


def _timed(fn, device, *a, **kw):
    """(result, seconds) with a real device barrier inside the timer."""
    t0 = time.monotonic()
    out = fn(*a, **kw)
    _sync(device)
    return out, time.monotonic() - t0


def _median_ms(fn, n, device, *a, **kw):
    """Median wall-ms of `n` synced calls, plus the raw samples."""
    import statistics

    samples = []
    for _ in range(n):
        _, dt = _timed(fn, device, *a, **kw)
        samples.append(1000 * dt)
    return statistics.median(samples), [round(s, 2) for s in samples]


def _free(*objs):
    import torch

    for o in objs:
        del o
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass


def _cos(a, b):
    """Cosine similarity of two arrays, flattened, in float64."""
    import numpy as np

    a, b = np.asarray(a, "float64").ravel(), np.asarray(b, "float64").ravel()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d else float("nan")


# --------------------------------------------------------- 1. SigLIP 2

# The keras cell (`_siglip_preprocess`) feeds uniform-random pixels in
# 0..255 through SigLIPImageConverter and three fixed captions through
# SentencePiece at seq 64.  We reproduce both: the HF image processor is
# rescale=1/255 + normalise(mean .5, std .5), i.e. the same
# `x / 127.5 - 1` the keras converter applies, so building pixel_values
# arithmetically is equivalent AND keeps host preprocessing (which is not
# part of the measurement) out of the timed region.
SIGLIP_PROMPTS = ["a photo of a cat", "a diagram of a transformer",
                  "an aerial view of a harbour at sunrise"]


def _siglip_inputs(processor, batch, image_size, seq_len=64,
                   prompts=SIGLIP_PROMPTS):
    import numpy as np

    rng = np.random.default_rng(0)          # same seed as the keras cell
    images = rng.uniform(0.0, 255.0, (batch, image_size, image_size, 3))
    pixel = (images / 127.5 - 1.0).transpose(0, 3, 1, 2)   # NHWC -> NCHW

    ip = processor.image_processor
    assert tuple(ip.image_mean) == (0.5, 0.5, 0.5), ip.image_mean
    assert tuple(ip.image_std) == (0.5, 0.5, 0.5), ip.image_std
    assert abs(ip.rescale_factor - 1 / 255) < 1e-12, ip.rescale_factor

    texts = [prompts[i % len(prompts)] for i in range(batch)]
    tok = processor(text=texts, padding="max_length", max_length=seq_len,
                    truncation=True, return_tensors="np")
    return {"pixel_values": pixel.astype("float32"),
            "input_ids": np.asarray(tok["input_ids"], dtype="int64")}


def _to_device(arrays, device, dtype):
    import torch

    out = {}
    for k, v in arrays.items():
        t = torch.from_numpy(v)
        out[k] = t.to(device) if k == "input_ids" else t.to(device, dtype)
    return out


def run_vision(repo=SIGLIP_REPO, revision=SIGLIP_REV, device="mps",
               dtype="bfloat16", batches=(1, 32), warm_steps=5, seq_len=64,
               compare_batch=4, compare=True):
    """SigLIP 2 so400m/14 @384 image+text forward -- STATUS row 16.

    Metrics mirror `run_keras_vision`: load_s, warmup_s, step_ms (warm
    median at the first batch size) and step_ms_b<N> for the rest, plus
    images_per_s and a logit fingerprint.  Correctness is the same
    checkpoint on torch-CPU in float32: cosine of the logit matrix and of
    the two embedding tables, plus argmax agreement.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    tdtype = getattr(torch, dtype)
    processor = AutoProcessor.from_pretrained(repo, revision=revision)

    t0 = time.monotonic()
    model = AutoModel.from_pretrained(repo, revision=revision, dtype=tdtype)
    model = model.to(device).eval()
    _sync(device)
    load_s = time.monotonic() - t0

    image_size = model.config.vision_config.image_size

    def forward(x):
        with torch.no_grad():
            return model(**x)

    out = dict(load_s=load_s, image_size=image_size, dtype=dtype,
               params=int(sum(p.numel() for p in model.parameters())),
               seq_len=seq_len, warm_steps=warm_steps,
               preprocess_source="arithmetic_rescale+hf_tokenizer",
               # .detach(): these are trainable parameters, and torch warns
               # about scalar-converting anything with requires_grad set.
               logit_scale=round(float(model.logit_scale.detach().float()), 4),
               logit_bias=round(float(model.logit_bias.detach().float()), 4))

    cmp_inputs = None
    for b in batches:
        arrays = _siglip_inputs(processor, b, image_size, seq_len)
        if b == compare_batch:
            cmp_inputs = arrays
        x = _to_device(arrays, device, tdtype)
        _, warm = _timed(forward, device, x)
        step_ms, samples = _median_ms(forward, warm_steps, device, x)
        y = forward(x)
        vl = y.logits_per_image.float().cpu().numpy()

        suffix = "" if b == batches[0] else f"_b{b}"
        out[f"warmup_s{suffix}"] = warm
        out[f"step_ms{suffix}"] = step_ms
        out[f"step_ms_samples{suffix}"] = samples
        out[f"images_per_s{suffix}"] = 1000.0 * b / step_ms
        out[f"logit_diag{suffix}"] = [round(float(v), 4)
                                      for v in np.diag(vl)[:4]]
        out[f"finite{suffix}"] = bool(np.isfinite(vl).all())
        # sample device memory while the weights and this batch are live;
        # `_emit` runs after everything is freed and would read ~0.
        out["mem_gb_torch"] = max(out.get("mem_gb_torch", 0.0), _mem_gb())
        _free(x, y)
    out["batch_sizes"] = list(batches)

    if compare:
        if cmp_inputs is None:
            cmp_inputs = _siglip_inputs(processor, compare_batch,
                                        image_size, seq_len)
        x = _to_device(cmp_inputs, device, tdtype)
        y = forward(x)
        got = {k: getattr(y, k).float().cpu().numpy()
               for k in ("logits_per_image", "image_embeds", "text_embeds")}
        _free(x, y)
        # `_free` can only drop its own argument names, so the caller's
        # binding (and `forward`'s closure cell) must be cleared here or
        # the bf16 model stays resident under the float32 reference load.
        model = forward = None
        _free()

        # float32 on the CPU is the reference, exactly as every other row
        # in this suite gates against jax-CPU.
        ref_model = AutoModel.from_pretrained(
            repo, revision=revision, dtype=torch.float32).eval()
        with torch.no_grad():
            ry = ref_model(**_to_device(cmp_inputs, "cpu", torch.float32))
        ref = {k: getattr(ry, k).float().numpy()
               for k in ("logits_per_image", "image_embeds", "text_embeds")}
        ref_model = ry = None
        _free()

        agree = int((got["logits_per_image"].argmax(-1)
                     == ref["logits_per_image"].argmax(-1)).sum())
        out["compare"] = dict(
            batch=compare_batch, ref="torch-cpu-float32",
            cos_logits=round(_cos(got["logits_per_image"],
                                  ref["logits_per_image"]), 6),
            cos_image_embeds=round(_cos(got["image_embeds"],
                                        ref["image_embeds"]), 6),
            cos_text_embeds=round(_cos(got["text_embeds"],
                                       ref["text_embeds"]), 6),
            max_abs_logit_diff=round(
                float(np.abs(got["logits_per_image"]
                             - ref["logits_per_image"]).max()), 4),
            argmax_agree=f"{agree}/{compare_batch}",
            logit_diag_mps=[round(float(v), 4)
                            for v in np.diag(got["logits_per_image"])],
            logit_diag_cpu=[round(float(v), 4)
                            for v in np.diag(ref["logits_per_image"])])
    return out


# ------------------------------------------------- 2. Gemma 4 E2B + LoRA

def probe_sdpa_backward(device="mps", dtype="bfloat16"):
    """Name the backward implementation autograd will actually call.

    STATUS footnote 10 says MPS SDPA has no backward kernel and falls back
    to math attention, which any torch training comparison must disclose.

    TRAP -- the obvious probe does not work.  Pinning a backend with
    `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])` and
    seeing the backward succeed proves nothing: those flags are
    CUDA-scoped and do not constrain dispatch on MPS, so all three
    "succeed" on this box whatever the kernel underneath is.

    What is real evidence is the autograd node recorded by the forward.
    A fused path names itself (`ScaledDotProductEfficientAttention...`,
    `...Flash...`, `...Cudnn...`); the math decomposition names itself
    too.  We record the node for MPS and, as a control, for CPU.
    """
    import torch

    def node_for(dev, dt):
        try:
            qkv = [torch.randn(1, 2, 8, 16, device=dev, dtype=dt,
                               requires_grad=True) for _ in range(3)]
            o = torch.nn.functional.scaled_dot_product_attention(*qkv)
            name = type(o.grad_fn).__name__ if o.grad_fn else "no grad_fn"
            o.sum().backward()
            _sync(dev)
            return {"grad_fn": name, "backward": "ok"}
        except Exception as e:
            return {"grad_fn": None,
                    "backward": f"{type(e).__name__}: {str(e)[:90]}"}

    dt = getattr(torch, dtype)
    return {"note": ("sdpa_kernel() backend flags are CUDA-scoped and do "
                     "not constrain MPS dispatch; the autograd node name "
                     "is the evidence"),
            device: node_for(device, dt),
            "cpu_control": node_for("cpu", torch.float32)}


def _lora_batch(tokenizer, n, seq_len, device):
    """The keras cell's synthetic batch, tokenised for torch.

    `run_keras_lora_train` builds word-salad strings from a fixed vocab
    with `default_rng(0)`, splits each into a prompt half and a response
    half and lets the Gemma preprocessor mask the prompt out of the loss
    via sample_weight.  We build the same strings and reproduce the
    masking with `labels = -100` over the first half of the window, so
    both stacks backprop the same amount of the sequence.
    """
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    words = ["metal", "kernel", "tensor", "gradient", "apple", "silicon",
             "unified", "memory", "compile", "shader"]
    texts = [" ".join(rng.choice(words, size=4 * seq_len)) for _ in range(n)]
    enc = tokenizer(texts, max_length=seq_len, truncation=True,
                    padding="max_length", return_tensors="pt")
    ids = enc["input_ids"]
    labels = ids.clone()
    labels[:, : seq_len // 2] = -100                     # prompt half
    labels[enc["attention_mask"] == 0] = -100            # padding
    return (ids.to(device), enc["attention_mask"].to(device),
            labels.to(device))


# Gemma 4 E2B is multimodal, and `AutoModelForCausalLM` gives the whole
# Gemma4ForConditionalGeneration (5.1 B params: text + vision + audio).
# Two reasons the LoRA targets are the *language model's* projections
# only, rather than a bare ["q_proj", "v_proj"]:
#   1. it is what the keras cell adapts (`lm.backbone.enable_lora`), and
#      the train step is text-only, so vision/audio adapters would be
#      trainable parameters that never receive a gradient;
#   2. the towers' projections are `Gemma4ClippableLinear` wrappers, which
#      peft 0.20 refuses ("only torch.nn.Linear ... supported").  The text
#      decoder's are plain nn.Linear.
LORA_TARGETS = r".*language_model.*\.(q_proj|v_proj)$"


def run_lora_train(repo=LORA_REPO, device="mps", dtype="bfloat16", rank=4,
                   seq_len=256, batch_size=1, steps=8, lr=1e-4,
                   attn="sdpa", targets=LORA_TARGETS):
    """Gemma 4 E2B + LoRA rank 4, one train step per batch -- STATUS row 18.

    Metrics mirror `run_keras_lora_train`: load_s, compile_s (there is no
    compile in torch eager -- it is the first, warm-up step, named to line
    the columns up), step_ms (median of the 8 warm steps), the loss
    series, and the trainable-parameter accounting.

    DISCLOSURE (STATUS footnote 10): MPS has no backward kernel for fused
    SDPA, so the backward pass runs the math decomposition.  The record
    carries attn_backward="math" and the measured `sdpa_backward_probe`
    that establishes it.
    """
    import statistics

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tdtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(repo)

    t0 = time.monotonic()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=tdtype, attn_implementation=attn)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=tdtype, attn_implementation=attn)
    model = model.to(device)
    _sync(device)
    load_s = time.monotonic() - t0

    total = sum(p.numel() for p in model.parameters())
    # keras `enable_lora(rank)` adapts the attention query/value
    # projections and applies no alpha scaling; lora_alpha == r gives peft
    # the same unit scaling.
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none",
        target_modules=targets, task_type="CAUSAL_LM"))
    model.train()
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    adapted = sum(1 for n, _ in model.named_modules()
                  if n.endswith("lora_A.default"))
    if not adapted:
        raise RuntimeError(f"LoRA matched no modules: {targets!r}")

    ids, mask, labels = _lora_batch(tokenizer, batch_size * (steps + 1),
                                    seq_len, device)
    opt = torch.optim.Adam([p for p in model.parameters()
                            if p.requires_grad], lr=lr)

    def step(i):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids[sl], attention_mask=mask[sl],
                     labels=labels[sl]).loss
        loss.backward()
        opt.step()
        return float(loss.item())        # .item() is itself a barrier

    ms, losses = [], []
    for i in range(steps + 1):
        t0 = time.monotonic()
        losses.append(step(i))
        _sync(device)
        ms.append(1000 * (time.monotonic() - t0))

    warm = ms[1:] or ms
    mem_gb = _mem_gb()          # weights + grads + Adam state still live
    return dict(mem_gb_torch=mem_gb,
                load_s=load_s, compile_s=ms[0] / 1000.0,
                first_step_ms=round(ms[0], 1),
                step_ms=statistics.median(warm),
                step_ms_samples=[round(v, 1) for v in ms],
                losses=[round(v, 4) for v in losses],
                lora_rank=rank, lora_targets=targets,
                lora_adapted_modules=adapted,
                seq_len=seq_len, batch_size=batch_size,
                steps=steps, lr=lr, dtype=dtype, optimizer="Adam",
                trainable_params=int(trainable), total_params=int(total),
                trainable_pct=round(100.0 * trainable / float(total), 5),
                attn_implementation=attn, attn_backward="math",
                attn_backward_note=(
                    "MPS has no fused-SDPA backward kernel; the backward "
                    "pass runs the math decomposition (STATUS fn.10)"),
                sdpa_backward_probe=probe_sdpa_backward(device, dtype))


# -------------------------------------------------------- 3. SD 3.5 Large

def run_diffusion(repo=SD35_REPO, revision=SD35_REV, device="mps",
                  dtype="bfloat16", num_steps=20, image_size=512,
                  prompt=None, out_png=None, marginal=True, seed=0):
    """Stable Diffusion 3.5 Large, one image -- STATUS row 17.

    Metrics mirror `run_keras_diffusion`: load_s, warmup_s (a short
    generate), ms_per_diffusion_step = warm total / num_steps and
    ms_per_diffusion_step_marginal = (t_long - t_short) / (n_long -
    n_short), which nets out the fixed text-encode + VAE-decode cost.
    The pixel statistics are the non-black gate the metaljax cell failed.
    """
    import numpy as np
    import torch
    from diffusers import StableDiffusion3Pipeline

    text = prompt or ("a photograph of an astronaut riding a horse on the "
                      "surface of Mars, golden hour, 50mm")
    tdtype = getattr(torch, dtype)

    t0 = time.monotonic()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        repo, revision=revision, torch_dtype=tdtype)
    pipe = pipe.to(device)
    _sync(device)
    load_s = time.monotonic() - t0

    def gen(n):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return pipe(text, num_inference_steps=n, height=image_size,
                    width=image_size, generator=g).images[0]

    short = max(2, num_steps // 5)
    _, warmup_s = _timed(gen, device, short)
    img, dt_long = _timed(gen, device, num_steps)

    a = np.asarray(img)
    out = dict(mem_gb_torch=_mem_gb(),      # pipeline still resident
               load_s=load_s, warmup_s=warmup_s, dtype=dtype,
               num_steps=num_steps, image_size=image_size,
               repo=repo, upstream_repo=SD35_UPSTREAM,
               generate_ms=1000 * dt_long,
               ms_per_diffusion_step=1000 * dt_long / num_steps,
               out_shape=list(a.shape), out_dtype=str(a.dtype),
               pixel_mean=round(float(a.mean()), 3),
               pixel_std=round(float(a.std()), 3),
               pixel_min=int(a.min()), pixel_max=int(a.max()),
               # the metaljax cell's failure mode was an all-black image,
               # so state the verdict rather than leaving it to the reader
               non_black=bool(a.std() > 5 and a.max() > 32))
    if out_png:
        img.save(out_png)
        out["image_path"] = str(out_png)
    if marginal:
        _, dt_short = _timed(gen, device, short)
        out["short_steps"] = short
        out["ms_per_diffusion_step_marginal"] = (
            1000 * (dt_long - dt_short) / max(num_steps - short, 1))
    return out


# ----------------------------------------------------------- record I/O

def _emit(rec, out_path, t0, verbose_keys=()):
    """Round, stamp, append one JSONL record; print the short form."""
    import torch

    # rows sample this while their weights are still resident; only fall
    # back to "now" (post-teardown, so ~0) when a row did not.
    rec.setdefault("mem_gb_torch", _mem_gb())
    rec["mem_gb"] = round(rec["mem_gb_torch"], 1)
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    rec["torch_version"] = torch.__version__
    for k, v in list(rec.items()):
        if isinstance(v, float):
            # significant digits, NOT decimal places: round(1e-4, 3) is
            # 0.0, which silently recorded the LoRA learning rate as zero.
            rec[k] = float(f"{v:.6g}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT " + json.dumps({k: v for k, v in rec.items()
                                  if k not in verbose_keys}), flush=True)
    return rec


def _run_row(ns, bench_id, fn, kwargs, verbose_keys=()):
    """Lock, swap-guard, run, record -- shared by the three subcommands."""
    rec = {"id": bench_id, "backend": f"torch-{ns.device}",
           "model": kwargs.get("repo"), "date": time.strftime("%Y-%m-%d")}
    t0 = time.monotonic()
    with MachineLock(enabled=not ns.no_lock):
        rec["swap_gb_at_start"] = round(wait_for_swap(ns.max_swap_gb), 2)
        try:
            rec.update(fn(**kwargs))
            rec["ok"] = True
        except Exception as e:
            import traceback

            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            traceback.print_exc()
    _emit(rec, ns.out, t0, verbose_keys)
    return 0 if rec.get("ok") else 1


def _common(ap):
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=str(LOGDIR / "results_new.jsonl"))
    ap.add_argument("--no-lock", action="store_true",
                    help="skip /tmp/metaljax-bench.lock (NOT for timings)")
    ap.add_argument("--max-swap-gb", type=float, default=20.0)
    return ap


def cmd_vision(argv):
    ap = _common(argparse.ArgumentParser(prog="adapter_torch_mps vision"))
    ap.add_argument("--bench-id", default="siglip2-so400m")
    ap.add_argument("--model", default=SIGLIP_REPO)
    ap.add_argument("--revision", default=SIGLIP_REV)
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--warm-steps", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--compare-batch", type=int, default=4)
    ap.add_argument("--no-compare", action="store_true")
    ns = ap.parse_args(argv)
    return _run_row(
        ns, ns.bench_id, run_vision,
        dict(repo=ns.model, revision=ns.revision, device=ns.device,
             dtype=ns.dtype,
             batches=tuple(int(b) for b in ns.batches.split(",")),
             warm_steps=ns.warm_steps, seq_len=ns.seq_len,
             compare_batch=ns.compare_batch, compare=not ns.no_compare),
        verbose_keys=("step_ms_samples", "step_ms_samples_b32"))


def cmd_lora(argv):
    ap = _common(argparse.ArgumentParser(prog="adapter_torch_mps lora"))
    ap.add_argument("--bench-id", default="lora-gemma4-e2b")
    ap.add_argument("--model", default=LORA_REPO)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--targets", default=LORA_TARGETS,
                    help="peft target_modules (regex or comma list)")
    ns = ap.parse_args(argv)
    targets = (ns.targets.split(",") if "," in ns.targets else ns.targets)
    return _run_row(
        ns, ns.bench_id, run_lora_train,
        dict(repo=ns.model, device=ns.device, dtype=ns.dtype, rank=ns.rank,
             seq_len=ns.seq_len, batch_size=ns.batch_size, steps=ns.steps,
             lr=ns.lr, attn=ns.attn, targets=targets))


def cmd_diffusion(argv):
    ap = _common(argparse.ArgumentParser(prog="adapter_torch_mps diffusion"))
    ap.add_argument("--bench-id", default="sd35-large")
    ap.add_argument("--model", default=SD35_REPO)
    ap.add_argument("--revision", default=SD35_REV)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--prompt")
    ap.add_argument("--png", help="where to write the generated image")
    ns = ap.parse_args(argv)
    png = ns.png or str(LOGDIR / f"sd35_torch_mps_{ns.image_size}.png")
    return _run_row(
        ns, ns.bench_id, run_diffusion,
        dict(repo=ns.model, revision=ns.revision, device=ns.device,
             dtype=ns.dtype, num_steps=ns.steps, image_size=ns.image_size,
             prompt=ns.prompt, out_png=png))


def main():
    """Dispatch; a sub-command-less argv is the legacy decode CLI."""
    cmds = {"vision": cmd_vision, "lora": cmd_lora,
            "diffusion": cmd_diffusion}
    argv = sys.argv[1:]
    if argv and argv[0] in cmds:
        return cmds[argv[0]](argv[1:])
    if argv and argv[0] == "decode":
        argv = argv[1:]
    return main_decode(argv)


def main_decode(argv=None):
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
    ns = ap.parse_args(argv)

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
