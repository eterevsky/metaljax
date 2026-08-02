"""Extra keras-hub adapters for the real-model benchmark suite.

Benchmark shapes that `run_bench.py` does not cover (it only knows LLM
decode rows).  Each function has the same signature as the adapters in
`run_bench.py` -- ``fn(bench, backend, prompt, n_decode)`` -- and returns a
plain metric dict, so wiring them in later is a one-line ``ADAPTERS``
update there::

    ADAPTERS.update(adapter_keras_extra.ADAPTERS)

  keras_vision      SigLIP 2 so400m/14 @384: image+text embedding forward,
                    batch 1 and batch 32.  load_s, warmup_s, step_ms.
  keras_diffusion   Stable Diffusion 3.5 Large: one 1024x1024 image, 20
                    steps.  load_s, warmup_s, ms_per_diffusion_step.
  keras_lora_train  Gemma 4 E2B + LoRA rank 4: one .fit() step, seq 256.
                    load_s, compile_s, step_ms.
  keras_lm_smoke    not a row -- "does this checkpoint load and generate
                    at all", for archs no keras preset covers.

Until then this file is also runnable on its own -- same CLI and same
jsonl record shape as run_bench.py::

    bench-venv/bin/python scripts/model_bench/adapter_keras_extra.py \
        siglip2-so400m cpu [--out results.jsonl]

TIMING DISCIPLINE (CLAUDE.md): `jax.block_until_ready` is a no-op on the
metaljax backend -- events are born ready and MLX evaluates async.  Every
timed region below therefore ends in a host materialisation: keras'
`predict_on_batch` finishes with `np.array(...)`, `generate()` returns a
uint8 numpy image, and the fit callback pulls `float(logs["loss"])`.
Never trust a number from this file that did not pass through `_sync`.

HOSTING (see README_keras_extra.md): every Kaggle-hosted preset used here
is anonymously downloadable -- no Kaggle account, no token, no consent
click.  keras-hub 0.30.0 has NO huggingface converter for `siglip`/
`siglip2` or for Stable Diffusion 3, so `hf://` is not an alternative
route for those two; Gemma 4 and Qwen 2 use `hf://` as usual.

ENVIRONMENT BUG: `import tensorflow` poisons the pip `sentencepiece`
extension in this venv, and keras-hub then SIGABRTs while loading any
SentencePiece preset -- see `patch_sentencepiece_native` below.  That is
what silently drops the gemma4 rows from the main suite.
"""

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


# ----------------------------------------------------------------- helpers

def _sync(tree):
    """Force every leaf to the host.  Returns the tree of numpy arrays.

    The only reliable barrier on this backend -- see the module docstring.
    """
    import numpy as np

    def leaf(x):
        if hasattr(x, "__array__") or hasattr(x, "shape"):
            return np.asarray(x)
        return x

    if isinstance(tree, dict):
        return {k: leaf(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(leaf(v) for v in tree)
    return leaf(tree)


def _timed(fn, *a, **kw):
    """(result, elapsed_seconds) with a host barrier inside the timer."""
    t0 = time.monotonic()
    out = _sync(fn(*a, **kw))
    return out, time.monotonic() - t0


def _median_ms(fn, n, *a, **kw):
    """Median wall-ms of `n` synced calls (plus the raw samples)."""
    samples = []
    for _ in range(n):
        _, dt = _timed(fn, *a, **kw)
        samples.append(1000 * dt)
    return statistics.median(samples), [round(s, 2) for s in samples]


def patch_sentencepiece_native():
    """Route keras-hub's SentencePiece metadata through tf-text.

    THE BUG (this venv: tensorflow 2.20.0 / tensorflow-text 2.20.1 /
    sentencepiece 0.2.1, macOS arm64).  `import tensorflow` alone is
    enough to poison the pip `sentencepiece` extension -- the two ship
    colliding copies of the sentencepiece/protobuf C++ symbols and the
    flat dyld namespace merges them.  Afterwards

        sentencepiece.SentencePieceProcessor().Init(model_proto=...)

    does not raise, it ABORTS the interpreter:

        libc++abi: terminating due to uncaught exception of type
        std::__1::system_error: mutex lock failed: Invalid argument

    Doing the `Init` first and importing tensorflow afterwards deadlocks
    instead (observed: a process wedged for 30+ min inside the import).
    Blocking `tensorflow_text` does NOT help -- plain `tensorflow` is
    already fatal, and keras_hub imports it unconditionally.

    THE BLAST RADIUS.  keras-hub 0.30.0's
    `SentencePieceTokenizer.set_proto` always calls `_set_proto_spm`,
    even when the tf-text tokenizer built fine, because it wants the
    vocabulary metadata from it.  So EVERY SentencePiece preset -- Gemma,
    SigLIP, Mixtral, T5Gemma -- kills the process during `from_preset`.
    This is what makes the `gemma4-e2b-bf16` rows disappear from the main
    suite with no error line: the abort is a SIGABRT, not an exception,
    so `run_bench.py`'s `except` never runs.

    THE FIX.  tf-text's own `SentencepieceTokenizer` (already built by
    `_set_proto_tf`) exposes `vocab_size()` / `id_to_string()`, which is
    all `set_proto` actually needs.  We swap `_set_proto_spm` for a shim
    over it and force `_allow_python_workflow = False` so the native
    processor is never reached.  Everything else -- tokenize,
    detokenize, tf.data preprocessing -- keeps working.

    Returns True if the patch was installed.  BENCH_SPM_PATCH=0 opts out.
    """
    if os.environ.get("BENCH_SPM_PATCH", "1") != "1":
        return False
    from keras_hub.src.tokenizers import sentence_piece_tokenizer as spt

    if getattr(spt.SentencePieceTokenizer, "_metaljax_patched", False):
        return True

    class _TfTextSpmShim:
        """The three methods `set_proto` calls on the native processor."""

        def __init__(self, tf_tokenizer):
            import numpy as np

            self._t = tf_tokenizer
            self._n = int(tf_tokenizer.vocab_size().numpy())
            pieces = tf_tokenizer.id_to_string(
                np.arange(self._n, dtype="int32")).numpy()
            self._pieces = [p.decode("utf-8", "replace") for p in pieces]

        def vocab_size(self):
            return self._n

        def IdToPiece(self, ids):
            if isinstance(ids, (list, tuple)):
                return [self._pieces[i] for i in ids]
            return self._pieces[ids]

        def unk_id(self):
            for tok in ("<unk>", "[UNK]", "<UNK>"):
                if tok in self._pieces:
                    return self._pieces.index(tok)
            return 0

    def _set_proto_spm(self, proto):
        if getattr(self, "_sentence_piece", None) is None:
            self._set_proto_tf(proto)
        self._sentence_piece_spm = _TfTextSpmShim(self._sentence_piece)
        # never take the native-sentencepiece tokenize/detokenize paths
        self._allow_python_workflow = False

    spt.SentencePieceTokenizer._set_proto_spm = _set_proto_spm
    spt.SentencePieceTokenizer._metaljax_patched = True
    return True


def _keras_bf16():
    import keras

    keras.config.set_dtype_policy("bfloat16")
    return keras


def _mem_gb(backend):
    if backend == "metaljax":
        import mlx.core as mx

        return mx.get_active_memory() / 1e9
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ------------------------------------------------------------ 1. SigLIP 2

# SentencePiece cannot be initialised in a process that has already loaded
# `tensorflow_text` (which keras_hub imports eagerly): the second copy of
# the sentencepiece C++ symbols wins and `SentencePieceProcessor.Init`
# aborts the interpreter with
#   libc++abi: ... std::__1::system_error: mutex lock failed: Invalid argument
# So `SigLIPTokenizer.from_preset` is unusable here and we tokenise in a
# clean child process that imports sentencepiece and nothing else.  This is
# outside every timed region -- the benchmark measures the forward pass.
_SPM_CHILD = r'''
import json, re, sys
import sentencepiece
proto, seq, texts = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
sp = sentencepiece.SentencePieceProcessor()
sp.Init(model_proto=open(proto, "rb").read())
end = sp.piece_to_id("</s>")
PUNCT = r"""[!"#$%&\\'()*+,\-./:;<=>?@\[\]^_`{|}~]"""
rows = []
for t in texts:
    t = re.sub(r"\s+", " ", re.sub(PUNCT, "", t.lower())).strip()
    ids = sp.encode(t)[: seq - 1] + [end]
    rows.append(ids + [end] * (seq - len(ids)))
print("SPMJSON" + json.dumps({"ids": rows, "end": end,
                              "vocab": sp.vocab_size()}))
'''


def _spm_tokenize(proto_path, texts, seq_len):
    """SigLIP text canonicalisation + SentencePiece, in a child process.

    Mirrors `SigLIPPreprocessor` exactly: lowercase, strip punctuation,
    collapse whitespace, truncate to `seq_len - 1`, append `</s>`, pad
    with `</s>` (SigLIP uses the end token as its pad token).
    """
    import subprocess

    out = subprocess.run(
        [sys.executable, "-c", _SPM_CHILD, str(proto_path), str(seq_len),
         json.dumps(texts)],
        capture_output=True, text=True, check=True)
    line = next(ln for ln in out.stdout.splitlines()
                if ln.startswith("SPMJSON"))
    return json.loads(line[len("SPMJSON"):])


def _siglip_preprocess(preset, prompts, batch, image_size, seq_len=64):
    """token_ids + images for the backbone.

    The SigLIP 2 Kaggle preset ships `tokenizer.json` and
    `image_converter.json` but NO `preprocessor.json`, so
    `SigLIPPreprocessor.from_preset` cannot be used either; the image half
    is `SigLIPImageConverter` (pure keras ops, safe in-process) and the
    text half goes through `_spm_tokenize`.
    """
    import numpy as np

    import keras_hub
    from keras_hub.src.utils.preset_utils import get_file

    rng = np.random.default_rng(0)
    images = rng.uniform(0.0, 255.0, (batch, image_size, image_size, 3))
    texts = [prompts[i % len(prompts)] for i in range(batch)]

    proto = get_file(preset, "assets/tokenizer/vocabulary.spm")
    tok = _spm_tokenize(proto, texts, seq_len)
    ids = np.asarray(tok["ids"], dtype="int32")

    try:
        # exported under keras_hub.layers, not keras_hub.models
        converter = keras_hub.layers.SigLIPImageConverter.from_preset(preset)
        imgs = np.asarray(converter(images))
        src = "image_converter+spm_subprocess"
    except Exception as e:  # pragma: no cover - reported, not swallowed
        print(f"  [siglip] image_converter unavailable ({type(e).__name__}: "
              f"{str(e)[:120]}); using the -1..1 rescale it applies",
              flush=True)
        imgs = (images / 127.5 - 1.0).astype("float32")
        src = "manual_rescale+spm_subprocess"
    return {"images": imgs, "token_ids": ids}, src


def run_keras_vision(bench, backend, prompt=None, n_decode=None,
                     batches=(1, 32), warm_steps=None):
    """SigLIP 2 image+text embedding forward at batch 1 and batch 32.

    Metrics: load_s, warmup_s (first, compiling call at batch 1), step_ms
    (warm batch-1 forward) and step_ms_b<N> / warmup_s_b<N> for every
    other batch size.  `images_per_s_b<N>` is the throughput view.
    """
    keras = _keras_bf16()
    import keras_hub

    preset = bench["model"]
    warm_steps = warm_steps or _int_env("BENCH_WARM_STEPS", 5)

    t0 = time.monotonic()
    backbone = keras_hub.models.SigLIPBackbone.from_preset(preset)
    load_s = time.monotonic() - t0

    image_size = backbone.vision_encoder.image_shape[0]
    prompts = [prompt or "a photo of a cat", "a diagram of a transformer",
               "an aerial view of a harbour at sunrise"]

    out = dict(load_s=load_s, image_size=image_size,
               params=backbone.count_params())
    # Evidence the checkpoint actually landed: a randomly initialised
    # SigLIPHead starts at scale=1 / bias=0, the trained so400m head sits
    # near scale~4.7 (log space) and bias~-13.
    try:
        import numpy as np

        head = backbone.siglip_head
        out["head_logit_scale"] = round(
            float(np.asarray(head.logit_scale)), 4)
        out["head_logit_bias"] = round(float(np.asarray(head.logit_bias)), 4)
    except Exception as e:  # pragma: no cover
        out["head_probe_error"] = f"{type(e).__name__}: {e}"[:120]
    for b in batches:
        x, src = _siglip_preprocess(preset, prompts, b, image_size)
        out.setdefault("preprocess_source", src)
        _, warm = _timed(backbone.predict_on_batch, x)
        step_ms, samples = _median_ms(backbone.predict_on_batch,
                                      warm_steps, x)
        y = _sync(backbone.predict_on_batch(x))
        suffix = "" if b == batches[0] else f"_b{b}"
        out[f"warmup_s{suffix}"] = warm
        out[f"step_ms{suffix}"] = step_ms
        out[f"step_ms_samples{suffix}"] = samples
        out[f"images_per_s{suffix}"] = 1000.0 * b / step_ms
        out[f"out_shapes{suffix}"] = {k: list(v.shape)
                                      for k, v in y.items()}
        # cheap correctness fingerprint: the diagonal of the image/text
        # logit matrix must be finite and batch-consistent.
        import numpy as np

        vl = y["vision_logits"]
        out[f"logit_diag{suffix}"] = [round(float(v), 4)
                                      for v in np.diag(vl)[:4]]
        out[f"finite{suffix}"] = bool(np.isfinite(vl).all())
        del x, y
        gc.collect()
    # canonical names for the primary (first) batch size
    out["batch_sizes"] = list(batches)
    return out


# -------------------------------------------------------- 2. SD 3.5 Large

def run_keras_diffusion(bench, backend, prompt=None, n_decode=None,
                        num_steps=None, image_size=None, marginal=True):
    """Stable Diffusion 3.5 Large: one image, `num_steps` diffusion steps.

    Metrics: load_s; warmup_s (first generate -- compiles the sampler);
    ms_per_diffusion_step = warm total / num_steps.  When `marginal` is
    set we also time a short generate and report
    `ms_per_diffusion_step_marginal` = (t_long - t_short) / (n_long -
    n_short), which removes the fixed text-encode + VAE-decode cost.
    `num_steps` is a traced argument of the jitted sampler, so the second
    call does NOT recompile.
    """
    keras = _keras_bf16()
    import keras_hub

    preset = bench["model"]
    num_steps = num_steps or _int_env("BENCH_DIFFUSION_STEPS", 20)
    image_size = image_size or _int_env("BENCH_IMAGE_SIZE", 1024)
    text = prompt or ("a photograph of an astronaut riding a horse on the "
                      "surface of Mars, golden hour, 50mm")

    t0 = time.monotonic()
    t2i = keras_hub.models.StableDiffusion3TextToImage.from_preset(
        preset, image_shape=(image_size, image_size, 3))
    load_s = time.monotonic() - t0

    short = max(2, num_steps // 5)
    img, warmup_s = _timed(t2i.generate, text, num_steps=short)
    total, dt_long = _timed(t2i.generate, text, num_steps=num_steps)

    import numpy as np

    img = np.asarray(total)
    out = dict(load_s=load_s, warmup_s=warmup_s,
               num_steps=num_steps, image_size=image_size,
               generate_ms=1000 * dt_long,
               ms_per_diffusion_step=1000 * dt_long / num_steps,
               out_shape=list(img.shape), out_dtype=str(img.dtype),
               pixel_mean=round(float(img.mean()), 3),
               pixel_std=round(float(img.std()), 3),
               pixel_min=int(img.min()), pixel_max=int(img.max()),
               params=t2i.backbone.count_params())
    dump = os.environ.get("BENCH_IMAGE_OUT")
    if dump:
        try:
            from PIL import Image

            Image.fromarray(img).save(dump)
            out["image_path"] = dump
        except Exception as e:  # pragma: no cover
            out["image_dump_error"] = f"{type(e).__name__}: {e}"[:120]
    if marginal:
        _, dt_short = _timed(t2i.generate, text, num_steps=short)
        out["short_steps"] = short
        out["ms_per_diffusion_step_marginal"] = (
            1000 * (dt_long - dt_short) / max(num_steps - short, 1))
    return out


# ------------------------------------------------- 3. Gemma 4 E2B + LoRA

class _StepTimer:
    """keras callback that times train batches WITH a host barrier.

    `on_train_batch_end` receives jax arrays; touching `logs["loss"]` as a
    python float is what actually waits for the step to land.
    """

    def __init__(self):
        self.ms = []
        self.losses = []
        self._t0 = None

    def make(self, keras):
        outer = self

        class Cb(keras.callbacks.Callback):
            def on_train_batch_begin(self, batch, logs=None):
                outer._t0 = time.monotonic()

            def on_train_batch_end(self, batch, logs=None):
                loss = float(logs["loss"]) if logs and "loss" in logs else 0.0
                outer.ms.append(1000 * (time.monotonic() - outer._t0))
                outer.losses.append(loss)

        return Cb()


def run_keras_lora_train(bench, backend, prompt=None, n_decode=None,
                         rank=4, seq_len=256, steps=None, batch_size=1):
    """One LoRA fine-tune step on Gemma 4 E2B, synthetic seq-256 batch.

    Metrics: load_s; compile_s (the first `.fit()` step -- trace + XLA/MLX
    compile); step_ms (median of the warm steps).  Only the LoRA A/B
    factors on the attention query/value projections are trainable.
    """
    keras = _keras_bf16()
    import numpy as np

    import keras_hub

    patched = patch_sentencepiece_native()
    steps = steps or _int_env("BENCH_LORA_STEPS", 8)
    preset = bench["model"]
    cls = getattr(keras_hub.models, bench.get("arch", "Gemma4CausalLM"))

    t0 = time.monotonic()
    lm = cls.from_preset(preset)
    load_s = time.monotonic() - t0

    # Gemma 4 is multimodal: the backbone wants eleven inputs (vision and
    # audio placeholders included), so build the batch with the model's own
    # preprocessor EAGERLY, then detach it -- that keeps host tokenisation
    # out of the timed step and makes it a pure device train step.
    lm.preprocessor.sequence_length = seq_len
    n = batch_size * (steps + 1)
    rng = np.random.default_rng(0)
    words = ["metal", "kernel", "tensor", "gradient", "apple", "silicon",
             "unified", "memory", "compile", "shader"]
    texts = [" ".join(rng.choice(words, size=4 * seq_len)) for _ in range(n)]
    try:
        # instruction-tuned preprocessors (Gemma 4) want prompt/response
        half = 2 * seq_len
        raw = {"prompts": [t[:half] for t in texts],
               "responses": [t[half:] for t in texts]}
        out = lm.preprocessor(raw)
    except (TypeError, KeyError, ValueError):
        out = lm.preprocessor(texts)
    x, y, sw = keras.utils.unpack_x_y_sample_weight(out)
    x = {k: np.asarray(v) for k, v in x.items()}
    y, sw = np.asarray(y), np.asarray(sw)
    lm.preprocessor = None

    lm.backbone.enable_lora(rank)
    trainable = int(sum(np.prod(w.shape) for w in lm.backbone.trainable_weights))
    total = lm.backbone.count_params()

    lm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[],
        jit_compile=True,
    )

    timer = _StepTimer()
    lm.fit(x=x, y=y, sample_weight=sw, batch_size=batch_size, epochs=1,
           shuffle=False, verbose=0, callbacks=[timer.make(keras)])

    if not timer.ms:  # pragma: no cover
        raise RuntimeError("no train batches ran")
    compile_s = timer.ms[0] / 1000.0
    warm = timer.ms[1:] or timer.ms
    return dict(load_s=load_s, compile_s=compile_s,
                step_ms=statistics.median(warm),
                step_ms_samples=[round(v, 1) for v in timer.ms],
                losses=[round(v, 4) for v in timer.losses],
                lora_rank=rank, seq_len=seq_len, batch_size=batch_size,
                spm_patched=patched,
                input_keys=sorted(x.keys()),
                trainable_params=trainable, total_params=int(total),
                trainable_pct=round(100.0 * trainable / float(total), 5))


# --------------------------------------------------- 4. DeepSeek-R1 smoke

def _bpe_tokenizer_with_eos_alias(cls, preset):
    """Rebuild a BPE tokenizer whose checkpoint renamed the EOS token.

    keras-hub 0.30.0's `QwenTokenizer.__init__` hardcodes

        eos_token = "<|endoftext|>"
        self._add_special_token(eos_token, "end_token")

    and `_update_special_token_ids` raises if that exact string is absent
    from the vocabulary.  The DeepSeek R1 distills are Qwen 2.5
    architecturally (`model_type: qwen2`, so the converter and the
    backbone are right) but ship DeepSeek's own tokenizer, where the
    Qwen special tokens were replaced -- `<|endoftext|>`, `<|im_start|>`
    and `<|im_end|>` are all gone and id 151643 is
    `<|end of sentence|>` (with full-width bars).  Loading the task then
    dies AFTER the 65 GB of weights are already in memory:

        ValueError: Cannot find special token `'<|endoftext|>'` in the
        provided vocabulary for `QwenTokenizer`.

    We repeat `convert_qwen.convert_tokenizer`'s vocabulary assembly and
    build a subclass whose `__init__` registers the checkpoint's REAL
    `eos_token` (from `tokenizer_config.json`) instead of the hardcoded
    one, bypassing `cls.__init__` and calling `BytePairTokenizer` di-
    rectly.  Aliasing inside the vocabulary does NOT work: keras-hub's
    id->token lookup is a tf `HashTable` and rejects two names for one
    id --

        FailedPreconditionError: HashTable has different value for same
        key. Key 151643 has <|end of sentence|> and trying to add value
        <|endoftext|>
    """
    from keras_hub.src.tokenizers.byte_pair_tokenizer import (
        BytePairTokenizer,
    )
    from keras_hub.src.utils.preset_utils import load_json

    cfg = load_json(preset, "tokenizer.json")
    vocab = cfg["model"]["vocab"]
    merges = cfg["model"]["merges"]
    if merges and isinstance(merges[0], list) and len(merges[0]) == 2:
        merges = [" ".join(m) for m in merges]
    special = set()
    for tok in cfg.get("added_tokens", []):
        if not tok["content"].startswith("<|reserved_special_token_"):
            vocab[tok["content"]] = tok["id"]
            special.add(tok["content"])

    tcfg = load_json(preset, "tokenizer_config.json")
    eos = tcfg.get("eos_token")
    if isinstance(eos, dict):
        eos = eos.get("content")
    if eos is None or eos not in vocab:
        raise ValueError(f"eos_token={eos!r} is not in the vocabulary")

    class _RealEosTokenizer(cls):
        def __init__(self, **kw):
            self._add_special_token(eos, "end_token")
            self.start_token_id = None
            self.start_token = None
            self.pad_token_id = 0  # as in QwenTokenizer
            BytePairTokenizer.__init__(self, **kw)

    return _RealEosTokenizer(vocabulary=vocab, merges=merges,
                             unsplittable_tokens=list(special)), eos


def run_keras_lm_smoke(bench, backend, prompt=None, n_decode=5):
    """Cheap converter smoke: does this HF checkpoint load and generate?

    Not a benchmark row -- it answers "is `arch` the right keras-hub class
    for this `model_type`" for checkpoints that no keras preset covers.
    Set BENCH_SMOKE_WEIGHTS=0 to check the config/class mapping only
    (no weight load), which costs seconds instead of minutes.
    """
    keras = _keras_bf16()
    import keras_hub

    patch_sentencepiece_native()
    preset = bench["model"]
    arch = bench.get("arch")
    load_weights = _int_env("BENCH_SMOKE_WEIGHTS", 1) == 1
    text = prompt or "The capital of France is"

    cls = getattr(keras_hub.models, arch)
    pre_cls = cls.preprocessor_cls
    note, alias = "from_preset", None
    t0 = time.monotonic()

    # Build the preprocessor FIRST -- it is cheap, and it is where
    # renamed special tokens blow up.  `cls.from_preset` loads the
    # backbone before it touches the tokenizer, so letting it fail there
    # would mean paying for the weights twice.
    try:
        pre = pre_cls.from_preset(preset)
    except ValueError as e:
        if "Cannot find special token" not in str(e):
            raise
        print(f"  [smoke] {e}\n  [smoke] rebuilding the tokenizer with the "
              f"checkpoint's own eos_token", flush=True)
        tok, alias = _bpe_tokenizer_with_eos_alias(
            pre_cls.tokenizer_cls, preset)
        pre = pre_cls(tokenizer=tok)
        note = "backbone + tokenizer rebuilt with the real EOS"
    backbone = cls.backbone_cls.from_preset(preset,
                                            load_weights=load_weights)
    lm = cls(backbone=backbone, preprocessor=pre)
    load_s = time.monotonic() - t0

    plen = len(lm.preprocessor.tokenizer(text))
    t0 = time.monotonic()
    out = lm.generate(text, max_length=plen + n_decode)
    gen_s = time.monotonic() - t0
    ids = [int(t) for t in lm.preprocessor.tokenizer(out)]
    return dict(load_s=load_s, warmup_s=gen_s, load_weights=load_weights,
                backbone_cls=type(lm.backbone).__name__, load_path=note,
                eos_alias=alias,
                end_token_id=getattr(lm.preprocessor.tokenizer,
                                     "end_token_id", None),
                generated=out, token_ids=ids[:64],
                out_tokens=len(ids) - plen, prompt_tokens=plen,
                params=int(lm.backbone.count_params()))


ADAPTERS = {
    "keras_vision": run_keras_vision,
    "keras_diffusion": run_keras_diffusion,
    "keras_lora_train": run_keras_lora_train,
    "keras_lm_smoke": run_keras_lm_smoke,
}


# --------------------------------------------------------------- runner

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bench_id")
    ap.add_argument("backend", choices=["metaljax", "cpu"])
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    ap.add_argument("--manifest", default=str(HERE / "manifest.json"),
                    help="manifest.json to read the bench entry from")
    ap.add_argument("--path", help="override the manifest adapter key "
                                   "(e.g. keras_lm_smoke)")
    ap.add_argument("--kw", action="append", default=[],
                    help="extra adapter kwarg, k=v (ints/floats parsed)")
    ns = ap.parse_args()

    manifest = json.load(open(ns.manifest))
    bench = next(b for b in manifest["benchmarks"] if b["id"] == ns.bench_id)
    path = ns.path or bench["path"]

    os.environ.setdefault(
        "JAX_PLATFORMS", "metal" if ns.backend == "metaljax" else "cpu")
    os.environ.setdefault("KERAS_BACKEND", "jax")

    kw = {}
    for item in ns.kw:
        k, _, v = item.partition("=")
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        kw[k] = v

    rec = {"id": ns.bench_id, "backend": ns.backend, "model": bench["model"],
           "size_gb": bench.get("size_gb"), "path": path,
           "date": time.strftime("%Y-%m-%d")}
    t0 = time.monotonic()
    try:
        rec.update(ADAPTERS[path](bench, ns.backend, manifest["prompt"], 5,
                                  **kw))
        rec["mem_gb"] = round(_mem_gb(ns.backend), 1)
        rec["ok"] = True
    except Exception as e:
        import traceback

        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        traceback.print_exc()
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    for k, v in list(rec.items()):
        if isinstance(v, float):
            rec[k] = round(v, 3)
    with open(ns.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT " + json.dumps({k: v for k, v in rec.items()
                                  if k not in ("token_ids",)}), flush=True)


if __name__ == "__main__":
    main()
