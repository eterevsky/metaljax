"""Fused-attention recognizer: coverage, numerics, and the fallbacks.

Three layers of testing:

* **Structure.** Every spelling of attention jax emits has to be recognized:
  the `[B,H,T,D]` and `[B,T,H,D]` conventions, `jax.nn.dot_product_attention`
  (which contracts through a rank-5 unit axis and puts the probabilities on
  the dot's right), causal `select` masks, additive bias masks, and grouped
  query attention both as a pre-broadcast K/V and as maxtext's reshaped-Q
  form. Each is also checked against jax-CPU.

* **Numerics.** The fused kernel accumulates the softmax in float32 whatever
  the input dtype, so it is compared against an exact float64 reference
  ALONGSIDE the literal chain it replaces: at f16/bf16 it has to be at least
  as accurate (measured: 3-5x better), at f32 within 1.5x (both sit at the
  f32 reduction's own error). This is what makes the recognizer safe to
  default on, so it is asserted, not just documented.

* **Fallbacks.** Anything the recognizer cannot prove has to run literally
  and correctly: probabilities consumed twice, a non-splat scale, a `select`
  whose constant is not a mask sentinel (where `select` and `add` are
  genuinely different functions), and the whole thing switched off.
"""

import functools
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import helpers
from metaljax import sdpa

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

needs_sdpa = pytest.mark.skipif(not sdpa.ENABLED, reason="METALJAX_SDPA=0")

B, H, T, D = 2, 4, 8, 16


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def _matches(f, *args):
    """How many attentions the recognizer finds in `f`'s lowering."""
    from metaljax import engine

    interp = engine.compile_program(helpers.lower_bytes(f, *args),
                                    "mlir").interpreter
    with interp.context:
        return len(sdpa.analyze(interp).matches)


def _run(f, args, enabled):
    """Compile and execute through the engine -- the route the native tape
    hooks, and the one that runs the recognizer prologues."""
    old = sdpa.ENABLED
    sdpa.ENABLED = enabled
    try:
        ex, outs = helpers.execute_module(helpers.lower_bytes(f, *args),
                                          jax.tree.leaves(args))
        got = [helpers.from_buffer(o) for o in outs]
        with ex.interpreter.context:
            n = len(sdpa.analyze(ex.interpreter).matches)
        return got, n
    finally:
        sdpa.ENABLED = old


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


@needs_sdpa
@pytest.mark.parametrize("name,fn,shapes",
                         SPELLINGS, ids=[s[0] for s in SPELLINGS])
def test_spelling_is_recognized(name, fn, shapes):
    args = _arrays(shapes, jnp.float32)
    assert _matches(fn, *args) == 1, f"{name} was not recognized"


@pytest.mark.parametrize("name,fn,shapes",
                         SPELLINGS, ids=[s[0] for s in SPELLINGS])
def test_spelling_matches_cpu(name, fn, shapes):
    args = _arrays(shapes, jnp.float32)
    helpers.check(fn, *args, rtol=2e-6, atol=2e-6)


@needs_sdpa
@pytest.mark.parametrize("dtype,rtol,atol",
                         [(jnp.float32, 2e-6, 2e-6),
                          (jnp.float16, 4e-3, 4e-3),
                          (jnp.bfloat16, 3e-2, 3e-2)],
                         ids=["f32", "f16", "bf16"])
def test_dtypes_recognized_and_match_cpu(dtype, rtol, atol):
    args = _arrays(S4, dtype)
    assert _matches(attn_bhtd, *args) == 1
    helpers.check(attn_bhtd, *args, rtol=rtol, atol=atol)


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

    fused, n = _run(fn, args, True)
    literal, n0 = _run(fn, args, False)
    assert n == 1 and n0 == 0

    e_fused = _relerr(fused[0], ref)
    e_literal = _relerr(literal[0], ref)
    budget = _ACCURACY_BUDGET[dtype]
    assert e_fused <= budget * e_literal + 1e-9, (
        f"{label}/{dtype.__name__}: fused {e_fused:.3e} vs literal "
        f"{e_literal:.3e} (budget {budget}x)")


@needs_sdpa
def test_fused_is_much_more_accurate_in_bfloat16():
    """The measured headline: f32 softmax accumulation inside the kernel."""
    d = 64
    args = _arrays([(1, 8, 128, d)] * 3, jnp.bfloat16)

    def attn(q, k, v):
        lg = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (d ** -0.5)
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(lg, -1), v)

    qq, kk, vv = [np.asarray(a, np.float32) for a in args]
    s = np.matmul(np.float64(qq), np.swapaxes(np.float64(kk), -1, -2)) * d ** -0.5
    e = np.exp(s - s.max(-1, keepdims=True))
    ref = np.matmul(e / e.sum(-1, keepdims=True), np.float64(vv))

    fused, _ = _run(attn, args, True)
    literal, _ = _run(attn, args, False)
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
    fused, n = _run(attn, args, True)
    literal, _ = _run(attn, args, False)
    assert n == 1
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


@needs_sdpa
def test_multi_query_attention_is_recognized():
    args = _arrays(_MQA_SHAPES, jnp.float32)
    assert _matches(attn_texmo_mqa, *args) == 1


def test_multi_query_attention_matches_cpu():
    args = _arrays(_MQA_SHAPES, jnp.float32)
    helpers.check(attn_texmo_mqa, *args, rtol=2e-6, atol=2e-6)


@needs_sdpa
@pytest.mark.parametrize("fn,shapes",
                         [(attn_bhtd, S4), (attn_texmo_mqa, _MQA_SHAPES)],
                         ids=["mha", "mqa"])
def test_training_graphs_do_not_fuse(fn, shapes):
    """Autodiff keeps the softmax output as a residual for the backward pass,
    so the probabilities are consumed twice and the chain must run literally.

    Fusing would mean recognizing and replacing the whole forward AND
    backward, which needs a fused attention VJP MLX does not expose at this
    level. This is why texmo (a training workload) is unaffected either way,
    and it is asserted so that a future change to the use-count discipline
    cannot start fusing away a residual something else still needs.
    """
    args = _arrays(shapes, jnp.float32)
    assert _matches(fn, *args) == 1, "the forward pass alone should fuse"
    grad = jax.grad(lambda *a: jnp.sum(fn(*a) ** 2))
    assert _matches(grad, *args) == 0, "a training graph must not fuse"


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

# Which batching axes the bias varies along, and whether some division of
# them into MLX's [B, N] can express it. Everything but the MIDDLE axis
# works: that one leaves a partial slot under every division, so it is
# (correctly) refused rather than materialized to the full slot.
_BIAS_SPLITS = [
    ("innermost", (1, 1, H, T, T), 1),
    ("outermost", (_A, 1, 1, T, T), 1),
    ("all", (_A, _B2, H, T, T), 1),
    ("none", (1, 1, 1, T, T), 1),
    ("middle-only", (1, _B2, 1, T, T), 0),
]


@needs_sdpa
def test_three_batching_axes_fuse_without_a_mask():
    assert _matches(attn_5d, *_arrays(_S5, jnp.float32)) == 1


@needs_sdpa
@pytest.mark.parametrize("name,mshape,want",
                         _BIAS_SPLITS, ids=[s[0] for s in _BIAS_SPLITS])
def test_batch_head_split_follows_the_mask(name, mshape, want):
    args = _arrays(_S5 + [mshape], jnp.float32)
    assert _matches(attn_5d_bias, *args) == want


@pytest.mark.parametrize("name,mshape,want",
                         _BIAS_SPLITS, ids=[s[0] for s in _BIAS_SPLITS])
def test_batch_head_split_matches_cpu(name, mshape, want):
    args = _arrays(_S5 + [mshape], jnp.float32)
    helpers.check(attn_5d_bias, *args, rtol=2e-6, atol=2e-6)


@needs_sdpa
@pytest.mark.parametrize("name,fn,shapes",
                         FALLBACKS, ids=[s[0] for s in FALLBACKS])
def test_fallback_is_not_fused(name, fn, shapes):
    args = _arrays(shapes, jnp.float32)
    assert _matches(fn, *args) == 0, f"{name} should not have been fused"


@pytest.mark.parametrize("name,fn,shapes",
                         FALLBACKS, ids=[s[0] for s in FALLBACKS])
def test_fallback_still_matches_cpu(name, fn, shapes):
    args = _arrays(shapes, jnp.float32)
    helpers.check(fn, *args, rtol=2e-6, atol=2e-6)


def test_disabled_recognizer_runs_the_literal_chain(monkeypatch):
    monkeypatch.setattr(sdpa, "ENABLED", False)
    args = _arrays(S4, jnp.float32)
    assert _matches(attn_bhtd, *args) == 0
    helpers.check(attn_bhtd, *args, rtol=2e-6, atol=2e-6)


# --------------------------------------------------------------------------
# through the real backend
# --------------------------------------------------------------------------


@needs_sdpa
def test_end_to_end_through_the_plugin():
    """The rewrite has to survive `mx.compile` tracing in engine.execute."""
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


def _asset_inputs(interp):
    """Deterministic host arguments for the asset, one per input aval."""
    rng = np.random.default_rng(0)
    args = []
    with interp.context:
        avals = interp.in_avals
    for shape, dt in avals:
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
    from metaljax import engine
    from metaljax.ops import control
    old = sdpa.ENABLED
    sdpa.ENABLED = enabled
    sdpa.reset_stats()
    try:
        ex = engine.compile_program(open(ASSET).read().encode(), "mlir")
        interp = ex.interpreter
        args = _asset_inputs(interp)
        with interp.context:
            n = len(sdpa.analyze(interp).matches)
            cost = control._block_cost(interp, interp._main_block())
        outs = engine.execute(ex, [helpers.to_buffer(a) for a in args])
        native = bool(ex._native_prog)
        return ([np.asarray(helpers.from_buffer(o), np.float32).astype(np.float64)
                 for o in outs], n, cost, sdpa.stats()["fused"], native)
    finally:
        sdpa.ENABLED = old


@needs_sdpa
def test_real_llm_asset_fuses_and_stays_correct():
    """maxtext's qwen3 prefill: GQA by reshaped Q, a `@_where` call mask, an
    f32 softmax island, and the deferred normalization -- all at once."""
    got, n, cost_on, fused, native = _run_asset(True)
    ref, n0, cost_off, _, _ = _run_asset(False)

    assert n == 1 and n0 == 0, f"recognized {n} attentions (off: {n0})"
    # The layer body is a while body called 8 times, and the two engines
    # count `fused` at different moments: the Python engine ticks it inside
    # `emit`, once per iteration, while the native tape emits ONCE at
    # lowering and replays the entry (tape._count, Stage 2 M4). Same eight
    # fused attentions either way; one site is what the tape must show.
    want_fused = 1 if native else 8
    assert fused == want_fused, (
        f"expected {want_fused} fused emit(s), got {fused}")
    assert cost_on < cost_off, "the fused chain should shrink the trace cost"

    # Only the attention changed, so the pre-attention outputs must be
    # BIT-identical and everything downstream may differ by the softmax
    # accumulation only. The asset's own shipped test uses rtol 2e-2.
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
    got, *_ = _run_asset(True)
    ref, *_ = _run_asset(False)
    # outputs 6 and 7 are cached_prefill_key / cached_prefill_value, layer
    # major.
    for j in (6, 7):
        np.testing.assert_array_equal(
            got[j][0], ref[j][0],
            err_msg=f"out[{j}] layer 0 changed; it precedes any attention")
