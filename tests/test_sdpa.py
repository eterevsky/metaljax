"""Fused-attention recognizer: coverage, numerics, and the fallbacks.

The recognizer lives in the native plugin (plugin-native/metal/metal_sdpa.cc)
and is controlled by METALJAX_SDPA, read at compile time.  Three layers of
testing survive the Stage-1 retirement (the Python-side match-count
introspection did not -- what is asserted now is values, which is what a
recognizer can break):

* **Structure/coverage.** Every spelling of attention jax emits is executed
  and checked against jax-CPU: the `[B,H,T,D]` and `[B,T,H,D]` conventions,
  `jax.nn.dot_product_attention`, causal `select` masks, additive bias
  masks, grouped-query attention in both forms, multi-query attention, and
  the vmapped three-batching-axes family.

* **Numerics.** The fused kernel accumulates the softmax in float32 whatever
  the input dtype, so it is compared against an exact float64 reference
  ALONGSIDE the literal chain it replaces (METALJAX_SDPA=0): at f16/bf16 it
  has to be at least as accurate (measured: 3-5x better), at f32 within
  1.5x.  The bf16 >2x-better assertion doubles as the proof that the fused
  kernel actually engages -- it cannot pass on the literal chain.

* **Fallbacks.** Anything the recognizer cannot prove has to run literally
  and correctly: probabilities consumed twice, a non-splat scale, a `select`
  whose constant is not a mask sentinel, softmax without max subtraction,
  training graphs (the softmax residual is consumed twice), and the whole
  thing switched off.
"""

import functools
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import helpers

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

needs_sdpa = pytest.mark.skipif(
    os.environ.get("METALJAX_SDPA", "1") == "0", reason="METALJAX_SDPA=0")

B, H, T, D = 2, 4, 8, 16


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
#
# METALJAX_SDPA is latched by the plugin in a function-local static
# (metal_sdpa.cc::SdpaEnabled), so flipping it in this process does
# NOTHING once the dylib has been loaded.  Every on/off comparison
# therefore runs in a FRESH PROCESS -- which is also what makes it an
# honest A/B: the two runs share no cached executable, kernel or pack.


def _run_child(mod_path, in_path, out_path, enabled):
    env = dict(os.environ, METALJAX_SDPA="1" if enabled else "0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(root, "tests")]
        + [p for p in [env.get("PYTHONPATH")] if p])
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), mod_path, in_path,
         out_path],
        env=env, capture_output=True, text=True, timeout=900)
    ok = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not ok:
        raise AssertionError(
            f"sdpa probe (enabled={enabled}) failed to run "
            f"(rc={proc.returncode})\n{proc.stdout}\n{proc.stderr}")
    return _loadz_any(out_path)


def _run(f, args, enabled):
    """Metal outputs of `f` with the recognizer on/off, in a fresh process."""
    with tempfile.TemporaryDirectory(prefix="mj-sdpa-") as tmp:
        mod = os.path.join(tmp, "mod.mlir")
        with open(mod, "wb") as fh:
            fh.write(helpers.lower_bytes(f, *args))
        ins = os.path.join(tmp, "in.npz")
        leaves = [np.asarray(a) for a in jax.tree.leaves(args)]
        _savez_any(ins, leaves)
        out = os.path.join(tmp, "out.npz")
        return _run_child(mod, ins, out, enabled)


def _savez_any(path, arrays):
    """np.savez, with ml_dtypes (bf16 &c) stored as their bit patterns."""
    saved = {"n": np.asarray(len(arrays))}
    narrow = []
    for i, a in enumerate(arrays):
        a = np.asarray(a)
        if a.dtype.kind == "V":          # ml_dtypes extension dtype
            narrow.append((i, str(a.dtype)))
            a = a.view({2: np.uint16, 1: np.uint8}[a.dtype.itemsize])
        saved[f"o{i}"] = a
    saved["narrow"] = np.asarray(json.dumps(narrow))
    np.savez(path, **saved)


def _loadz_any(path):
    import ml_dtypes
    with np.load(path, allow_pickle=False) as z:
        arrays = [z[f"o{i}"] for i in range(int(z["n"]))]
        for i, name in json.loads(str(z["narrow"])):
            arrays[i] = arrays[i].view(getattr(ml_dtypes, name))
    return arrays


def _arrays(shapes, dtype, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in shapes:
        x = rng.standard_normal(s).astype(np.float32)
        out.append(jnp.asarray(x, dtype))
    return out


# --------------------------------------------------------------------------
# the attention spellings
# --------------------------------------------------------------------------


def attn_bhtd(q, k, v):
    """[B, H, T, D] -- the torch/HF convention."""
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)


def attn_bthd(q, k, v):
    """[B, T, H, D] -- the flax convention, and `/ sqrt(D)` not `* D**-0.5`."""
    lg = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(D)
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(lg, -1), v)


def attn_causal(q, k, v):
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
    mask = jnp.tril(jnp.ones((T, T), bool))
    lg = jnp.where(mask, lg, jnp.finfo(lg.dtype).min)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)


def attn_bias(q, k, v, m):
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5) + m
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)


def attn_gqa(q, k, v):
    """K/V broadcast up to the query head count before the dot."""
    g = q.shape[1] // k.shape[1]
    kk, vv = jnp.repeat(k, g, 1), jnp.repeat(v, g, 1)
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, kk) * (D ** -0.5)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), vv)


def attn_gqa_reshaped(q, k, v):
    """maxtext's form: Q reshaped to [B, H_kv, g, T, D], K/V left alone."""
    b, hq, t, d = q.shape
    hkv = k.shape[1]
    qr = q.reshape(b, hkv, hq // hkv, t, d)
    lg = jnp.einsum("bhgqd,bhkd->bhgqk", qr, k) * (D ** -0.5)
    o = jnp.einsum("bhgqk,bhkd->bhgqd", jax.nn.softmax(lg, -1), v)
    return o.reshape(b, hq, t, d)


def attn_deferred(q, k, v):
    """Normalization AFTER the values dot, as real LLM lowerings emit it."""
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
    e = jnp.exp(lg - jnp.max(lg, -1, keepdims=True))
    o = jnp.einsum("bhqk,bhkd->bhqd", e, v)
    return o / jnp.sum(e, -1, keepdims=True)


def nn_dpa(q, k, v):
    return jax.nn.dot_product_attention(q, k, v)


def nn_dpa_causal(q, k, v):
    return jax.nn.dot_product_attention(q, k, v, is_causal=True)


S4 = [(B, H, T, D)] * 3
GQA4 = [(B, 8, T, D), (B, 2, T, D), (B, 2, T, D)]
GQA2 = [(B, 8, T, D), (B, 4, T, D), (B, 4, T, D)]

SPELLINGS = [
    ("bhtd", attn_bhtd, S4),
    ("bthd", attn_bthd, [(B, T, H, D)] * 3),
    ("causal", attn_causal, S4),
    ("bias", attn_bias, S4 + [(B, 1, T, T)]),
    ("gqa4:1", attn_gqa, GQA4),
    ("gqa2:1", attn_gqa, GQA2),
    ("gqa-reshaped", attn_gqa_reshaped, GQA4),
    ("deferred-norm", attn_deferred, S4),
    ("jax.nn.dpa", nn_dpa, [(B, T, H, D)] * 3),
    ("jax.nn.dpa-causal", nn_dpa_causal, [(B, T, H, D)] * 3),
    ("jax.nn.dpa-gqa", nn_dpa, [(B, T, 8, D), (B, T, 2, D), (B, T, 2, D)]),
]


@pytest.mark.parametrize("name,fn,shapes",
                         SPELLINGS, ids=[s[0] for s in SPELLINGS])
def test_spelling_matches_cpu(name, fn, shapes):
    args = _arrays(shapes, jnp.float32)
    helpers.check(fn, *args, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype,rtol,atol",
                         [(jnp.float32, 2e-6, 2e-6),
                          (jnp.float16, 4e-3, 4e-3),
                          (jnp.bfloat16, 3e-2, 3e-2)],
                         ids=["f32", "f16", "bf16"])
def test_dtypes_match_cpu(dtype, rtol, atol):
    args = _arrays(S4, dtype)
    helpers.check(attn_bhtd, *args, rtol=rtol, atol=atol)


@pytest.mark.parametrize("fn,shapes", [(attn_bhtd, S4)], ids=["mha"])
def test_training_graph_matches_cpu(fn, shapes):
    """Autodiff keeps the softmax output as a residual, so the probabilities
    are consumed twice and the chain must run literally -- and right."""
    args = _arrays(shapes, jnp.float32)
    grad = jax.grad(lambda *a: jnp.sum(fn(*a) ** 2))
    helpers.check(grad, *args, rtol=4e-6, atol=4e-6)


# --------------------------------------------------------------------------
# numerics: the fused kernel must not be less accurate than what it replaces
# --------------------------------------------------------------------------


def _f64_attention(q, k, v):
    """Exact reference, float64 throughout, [B, N, T, D] with GQA broadcast."""
    q = np.asarray(q, np.float64)
    k = np.asarray(k, np.float64)
    v = np.asarray(v, np.float64)
    g = q.shape[1] // k.shape[1]
    kk, vv = np.repeat(k, g, 1), np.repeat(v, g, 1)
    s = np.matmul(q, np.swapaxes(kk, -1, -2)) * np.float64(D ** -0.5)
    e = np.exp(s - s.max(-1, keepdims=True))
    return np.matmul(e / e.sum(-1, keepdims=True), vv)


def _relerr(got, ref):
    got = np.asarray(got, np.float32).astype(np.float64)
    return float(np.abs(got - ref).max()) / max(float(np.abs(ref).max()), 1e-30)


# f32 is a TIE: the fused kernel is allowed to be slightly worse (measured
# 1.0-1.2x) because both sit at the f32 reduction's own error. The half
# precisions must be at least as good -- that is the whole reason the
# recognizer is safe to enable by default.
_ACCURACY_BUDGET = {jnp.float32: 1.5, jnp.float16: 1.0, jnp.bfloat16: 1.0}


@needs_sdpa
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16],
                         ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize("fn,shapes,label",
                         [(attn_bhtd, S4, "mha"),
                          (attn_gqa, GQA4, "gqa4:1"),
                          (attn_gqa, GQA2, "gqa2:1")],
                         ids=["mha", "gqa4", "gqa2"])
def test_fused_is_no_less_accurate_than_the_literal_chain(dtype, fn, shapes,
                                                          label):
    args = _arrays(shapes, dtype)
    # `_f64_attention` broadcasts K/V itself, so it is the reference for the
    # grouped spellings too.
    ref = _f64_attention(*[np.asarray(a, np.float32) for a in args])

    fused = _run(fn, args, True)
    literal = _run(fn, args, False)

    e_fused = _relerr(fused[0], ref)
    e_literal = _relerr(literal[0], ref)
    budget = _ACCURACY_BUDGET[dtype]
    assert e_fused <= budget * e_literal + 1e-9, (
        f"{label}/{dtype.__name__}: fused {e_fused:.3e} vs literal "
        f"{e_literal:.3e} (budget {budget}x)")


@needs_sdpa
def test_fused_is_much_more_accurate_in_bfloat16():
    """The measured headline: f32 softmax accumulation inside the kernel.

    Also the de-facto engagement test: the literal bf16 chain cannot beat
    itself by 2x, so this failing means the fused kernel no longer fires.
    """
    d = 64
    args = _arrays([(1, 8, 128, d)] * 3, jnp.bfloat16)

    def attn(q, k, v):
        lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (d ** -0.5)
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)

    qq, kk, vv = [np.asarray(a, np.float32) for a in args]
    s = np.matmul(np.float64(qq), np.swapaxes(np.float64(kk), -1, -2)) * d ** -0.5
    e = np.exp(s - s.max(-1, keepdims=True))
    ref = np.matmul(e / e.sum(-1, keepdims=True), np.float64(vv))

    fused = _run(attn, args, True)
    literal = _run(attn, args, False)
    e_fused, e_literal = _relerr(fused[0], ref), _relerr(literal[0], ref)
    assert e_fused < e_literal / 2, (
        f"expected the fused kernel to beat the bf16 chain by >2x, got "
        f"{e_literal:.3e} -> {e_fused:.3e}")


@needs_sdpa
def test_fully_masked_row_keeps_the_literal_semantics():
    """An all-masked row must behave as the literal chain does.

    This is why masks are handed to MLX additively: its BOOLEAN mask gives
    such a row a finite value, while `select(pred, L, finfo.min)` -- what jax
    emits -- gives a uniform row, and an `-inf` bias gives NaN.
    """
    def attn(q, k, v, m):
        lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg + m, -1), v)

    m = np.zeros((1, 1, T, T), np.float32)
    m[0, 0, 0, :] = -np.inf                    # row 0 attends to nothing
    args = _arrays(S4, jnp.float32) + [jnp.asarray(m)]
    fused = _run(attn, args, True)
    literal = _run(attn, args, False)
    a, b = np.asarray(fused[0]), np.asarray(literal[0])
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)],
                               rtol=2e-6, atol=2e-6)


# --------------------------------------------------------------------------
# fallbacks: anything unproven runs literally, and correctly
# --------------------------------------------------------------------------


def attn_probs_escape(q, k, v):
    """The probabilities are also returned: they must be materialized."""
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
    p = jax.nn.softmax(lg, -1)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v), p


def attn_nonsplat_scale(q, k, v, s):
    """A per-head scale is not a scalar, so it cannot become MLX's `scale`."""
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * s[None, :, None, None]
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)


def attn_small_select(q, k, v):
    """`select` with a small constant is NOT the same function as `add`."""
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
    mask = jnp.tril(jnp.ones((T, T), bool))
    lg = jnp.where(mask, lg, -1.0)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)


def attn_softmax_no_max(q, k, v):
    """No max subtraction: MLX always subtracts one, which differs on
    overflow, so this must not be silently "improved"."""
    lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (D ** -0.5)
    e = jnp.exp(lg)
    p = e / jnp.sum(e, -1, keepdims=True)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v)


FALLBACKS = [
    ("probs-escape", attn_probs_escape, S4),
    ("non-splat-scale", attn_nonsplat_scale, S4 + [(H,)]),
    ("small-select-constant", attn_small_select, S4),
    ("softmax-without-max", attn_softmax_no_max, S4),
]


def attn_texmo_mqa(inputs, wq, wk, wv):
    """texmo's `attn.512.8.64`: ONE shared K/V head (multi-query), a windowed
    causal mask, and an f32 softmax island. The head axis is a free axis of
    Q alone -- there is no head batching dimension at all."""
    b, t, _ = inputs.shape
    q = (inputs @ wq.T).reshape(b, t, H, D)
    k = inputs @ wk.T
    v = inputs @ wv.T
    scores = jnp.einsum("bthd,bsd->bhts", q, k) * (float(D) ** -0.5)
    i = jnp.arange(t)[:, None]
    j = jnp.arange(t)[None, :]
    allowed = (j <= i) & (j > i - 512)
    scores = jnp.where(allowed[None, None], scores.astype(jnp.float32),
                       jnp.float32(-jnp.inf))
    a = jax.nn.softmax(scores, -1).astype(inputs.dtype)
    return jnp.einsum("bhts,bsd->bthd", a, v).reshape(b, t, H * D)


_MQA_SHAPES = [(B, T, H * D), (H * D, H * D), (D, H * D), (D, H * D)]


def test_multi_query_attention_matches_cpu():
    args = _arrays(_MQA_SHAPES, jnp.float32)
    helpers.check(attn_texmo_mqa, *args, rtol=2e-6, atol=2e-6)


# --------------------------------------------------------------------------
# the batch/head split (vmapped attention: three batching axes)
# --------------------------------------------------------------------------

_A, _B2 = 2, 3


def attn_5d(q, k, v):
    lg = jnp.einsum("abhqd,abhkd->abhqk", q, k) * (D ** -0.5)
    return jnp.einsum("abhqk,abhkd->abhqd", jax.nn.softmax(lg, -1), v)


def attn_5d_bias(q, k, v, m):
    lg = jnp.einsum("abhqd,abhkd->abhqk", q, k) * (D ** -0.5) + m
    return jnp.einsum("abhqk,abhkd->abhqd", jax.nn.softmax(lg, -1), v)


_S5 = [(_A, _B2, H, T, D)] * 3

# Which batching axes the bias varies along.  The recognizer must express
# each as some division into MLX's [B, N] or run it literally; either way
# the values must match CPU.  The middle-only case is the one the
# recognizer refuses (a partial slot under every division).
_BIAS_SPLITS = [
    ("innermost", (1, 1, H, T, T)),
    ("outermost", (_A, 1, 1, T, T)),
    ("all", (_A, _B2, H, T, T)),
    ("none", (1, 1, 1, T, T)),
    ("middle-only", (1, _B2, 1, T, T)),
]


def test_three_batching_axes_match_cpu():
    args = _arrays(_S5, jnp.float32)
    helpers.check(attn_5d, *args, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("name,mshape",
                         _BIAS_SPLITS, ids=[s[0] for s in _BIAS_SPLITS])
def test_batch_head_split_matches_cpu(name, mshape):
    args = _arrays(_S5 + [mshape], jnp.float32)
    helpers.check(attn_5d_bias, *args, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("name,fn,shapes",
                         FALLBACKS, ids=[s[0] for s in FALLBACKS])
def test_fallback_still_matches_cpu(name, fn, shapes):
    args = _arrays(shapes, jnp.float32)
    helpers.check(fn, *args, rtol=2e-6, atol=2e-6)


def test_disabled_recognizer_runs_the_literal_chain(monkeypatch):
    monkeypatch.setenv("METALJAX_SDPA", "0")
    args = _arrays(S4, jnp.float32)
    helpers.check(attn_bhtd, *args, rtol=2e-6, atol=2e-6)


# --------------------------------------------------------------------------
# through jax.jit (the route a user program takes)
# --------------------------------------------------------------------------


@needs_sdpa
def test_end_to_end_through_jit():
    metal = jax.devices("metal")[0]
    cpu = jax.devices("cpu")[0]
    args = _arrays(S4, jnp.float32)
    fn = jax.jit(attn_causal)
    got = np.asarray(fn(*[jax.device_put(a, metal) for a in args]))
    with jax.default_device(cpu):
        want = np.asarray(jax.jit(attn_causal)(*args))
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-6)


# --------------------------------------------------------------------------
# the real model asset
# --------------------------------------------------------------------------


ASSET = os.path.join(os.path.dirname(__file__), "data",
                     "qwen3_prefill_shrunk.mlir")


def _asset_arg_specs(text):
    """(shape, np dtype) of the asset's entry arguments, via jaxlib's MLIR
    bindings alone (the same registration scripts/run_stablehlo_bench.py
    uses -- captured jax modules carry sdy/chlo custom assembly)."""
    import ml_dtypes
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo
    from jaxlib.mlir._mlir_libs import _jax_mlir_ext

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _jax_mlir_ext.register_dialects(reg)
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()
    stablehlo.register_dialect(ctx)
    for _name in ("chlo", "sdy", "mpmd"):
        try:
            __import__(f"jaxlib.mlir.dialects.{_name}",
                       fromlist=[_name]).register_dialect(ctx)
        except Exception:
            pass
    dt = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16,
          "i64": np.int64, "i32": np.int32, "i8": np.int8, "i1": np.bool_,
          "ui32": np.uint32, "ui8": np.uint8}
    with ctx:
        module = ir.Module.parse(text)
        entry = None
        for op in module.body.operations:
            if op.operation.name == "func.func":
                name = ir.StringAttr(op.attributes["sym_name"]).value
                if name == "main" or entry is None:
                    entry = op
        specs = []
        for a in entry.regions[0].blocks[0].arguments:
            t = ir.RankedTensorType(a.type)
            specs.append((tuple(t.shape), np.dtype(dt[str(t.element_type)])))
    return specs


def _asset_inputs(specs):
    """Deterministic host arguments for the asset, one per input aval."""
    rng = np.random.default_rng(0)
    args = []
    for shape, dt in specs:
        if np.issubdtype(dt, np.integer):
            x = rng.integers(0, 4, size=shape).astype(dt)
        elif dt == np.bool_:
            x = rng.integers(0, 2, size=shape).astype(bool)
        else:
            x = (rng.standard_normal(size=shape) * 0.1).astype(np.float32)
            x = x.astype(dt)
        args.append(x)
    return args


@functools.lru_cache(maxsize=2)
def _run_asset(enabled):
    """Cached: the two tests below need the same pair of runs, and this is
    the most expensive thing in the file (a real 8-layer decode step)."""
    text = open(ASSET).read()
    args = _asset_inputs(_asset_arg_specs(text))
    with tempfile.TemporaryDirectory(prefix="mj-sdpa-asset-") as tmp:
        ins = os.path.join(tmp, "in.npz")
        _savez_any(ins, args)
        out = os.path.join(tmp, "out.npz")
        outs = _run_child(ASSET, ins, out, enabled)
    return [np.asarray(o, np.float32).astype(np.float64) for o in outs]


@needs_sdpa
def test_real_llm_asset_fuses_and_stays_correct():
    """maxtext's qwen3 prefill: GQA by reshaped Q, a `@_where` call mask, an
    f32 softmax island, and the deferred normalization -- all at once.
    Only the attention may change between the two runs, so everything may
    differ by the softmax accumulation only. The asset's own shipped test
    uses rtol 2e-2."""
    got = _run_asset(True)
    ref = _run_asset(False)

    worst = 0.0
    for a, b in zip(got, ref):
        assert a.shape == b.shape
        scale = float(np.nanmax(np.abs(b))) if b.size else 0.0
        worst = max(worst, float(np.nanmax(np.abs(a - b)))
                    / max(scale, 1e-30))
    assert worst < 2e-2, f"asset outputs drifted by {worst:.3e}"


@needs_sdpa
def test_real_llm_asset_first_layer_is_bit_identical():
    """Layer 0's K/V cache is computed before any attention, so the rewrite
    must not touch it -- this is what separates "accumulated rounding" from
    "the rewrite moved something it should not have"."""
    got = _run_asset(True)
    ref = _run_asset(False)
    # outputs 6 and 7 are cached_prefill_key / cached_prefill_value, layer
    # major.
    for j in (6, 7):
        np.testing.assert_array_equal(
            got[j][0], ref[j][0],
            err_msg=f"out[{j}] layer 0 changed; it precedes any attention")


# --------------------------------------------------------------------------
# child: run one module under this process's METALJAX_SDPA
# --------------------------------------------------------------------------


if __name__ == "__main__":
    _mod, _in, _out = sys.argv[1], sys.argv[2], sys.argv[3]
    _text = open(_mod, "rb").read()
    try:                       # a .mlir asset is text; a lowering is bytecode
        _text = _text.decode()
    except UnicodeDecodeError:
        pass
    _args = _loadz_any(_in)
    _res = helpers.run_module(_text, _args)
    _savez_any(_out, _res)
    print(json.dumps({"n": len(_res),
                      "sdpa": os.environ.get("METALJAX_SDPA", "1")}))
