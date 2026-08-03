"""Recognizer that maps dequantize-then-matmul chains onto mx.quantized_matmul.

Weight-only quantized models (keras `quantize("int4")`, `jnp.int4` weights,
anything that stores integer codes plus a scale/zero-point map) lower to
StableHLO that *materializes the whole dequantized weight* and then runs a
float dot:

    w_f = (convert(codes) - convert(zeros)) * scales     # full weight size
    y   = dot_general(x, w_f)

XLA:CPU fuses that reconstruction into the dot's operand loop and pays
almost nothing; interpreting it literally costs ~20 full-weight-sized MLX
kernels per matmul per token, which is why LLM decode on packed-int4 weights
was 5x slower here than on the CPU backend.

This module recognizes the pattern structurally at compile time, repacks the
codes ONCE into MLX's affine layout at the first execute, and replaces the
whole chain with a single `mx.quantized_matmul`.

Three forms are recognized:

  sub-channel / asymmetric   dot(x, [reshape/transpose](mul(sub(cvt(codes),
                                                    cvt(zeros)), scales)))
  per-channel / symmetric    divide(dot(x, cvt(codes)), broadcast(scale))
  MXFP4 (OCP micro-scaling)  dot(x, [reshape](mul(decode(codes),
                                                  broadcast(2**(e8m0-127)))))

The first two are what keras 3.15 emits; the third is what an MXFP4
checkpoint (gpt-oss, and the whole OCP-MX family) looks like once the
nibble unpack and the E2M1 value decode are done IN the graph instead of at
load time. MXFP4's 16 codes are a NON-UNIFORM grid (0, +-0.5, +-1, +-1.5,
+-2, +-3, +-4, +-6 times a per-32 power-of-two scale), so no affine
`scale * (q - zp)` can represent it -- MLX has a separate `mode="mxfp4"`
kernel, and this module packs for it separately.

The chain between the integer codes and the dot is NOT pattern-matched op by
op (keras' nibble unpack is a dozen ops of `and`/`shift`/`xor`/`concatenate`
whose exact shape is not contractual). Instead the recognizer verifies the
top-level affine form, records the three operand subtrees, and defers every
VALUE-level question to the first execute, where the subtrees are evaluated
once on concrete buffers with the interpreter's own op handlers:

  * codes must be integral and fit in 4 or 8 bits;
  * the scale and zero maps must be constant inside contiguous groups along
    the contraction axis (this subsumes checking that keras' `g_idx` gather
    is the identity ramp -- any group-preserving permutation is fine because
    the group scales are read off the MATERIALIZED map, not the source
    argument);
  * zero points must be integers.

A dot whose groups are INTERLEAVED along the canonical contraction axis is
still fused, by permuting that axis (see `_regroup`). Permuting the
contraction axis identically on both operands leaves the dot unchanged, so
whenever the group members can be brought together by a permutation the
weight packs exactly and the activations are gathered through the same
permutation at run time. keras' EinsumDense produces exactly this: an
attention projection `btnh,nhd->btd` lowers with `contracting_dims =
[3, 2] x [1, 0]`, so the canonical K axis runs h-major while keras' groups
run along its own n-major flattening of the same axis.

Any check failing permanently disables that one dot (it falls back to
executing the chain literally, which is what happened before this module
existed).

A dot with BATCHING dimensions (an einsum over a stack of per-expert
weights, `etm,ehm->eth`) is fused too: `mx.quantized_matmul` broadcasts over
leading dimensions, so the pack keeps them and the operands are reshaped to
`[B, M, K]` / `[B, N, K]`. METALJAX_QMM_BATCH=0 restores the old
batching-dims-are-a-reject behaviour.

Packing is exact: `w = scale * (q - zp)` maps onto MLX's affine dequant
`w = scales * q_hat + biases` with an unsigned `q_hat = q + offset` and
`biases = -scale * (offset + zp)`; MLX's kernel evaluates that as a single
FMA, so whenever the bias is exactly representable the reconstructed weight
is bit-identical to a float32 dequantization. `offset` is `2^(bits-1)` (the
signed-code convention) whenever the codes fit it, so a zero-point-free
quantization keeps a power-of-two bias. Scales/biases are kept in the source
dtype when that downcast is lossless and widened to f32 otherwise -- see
METALJAX_QMM_SCALES.

MXFP4 packing is exact by construction and needs none of that care: the
codes are read back off the e2m1 grid by exact equality and the scale is
recovered as an exact power of two (its f32 exponent field IS the e8m0
byte), so `values * scale` is reproduced bit for bit -- in f32 and, because
an e2m1 value carries a single mantissa bit, in bf16/f16 as well. Anything
not exactly on the grid rejects the dot.

Packing needs CONCRETE buffers and must never run inside an `mx.compile`
trace, so it happens in an eager prologue (`prologue()`, called by
engine.execute) and the packed arrays are threaded into traced code as
explicit inputs -- never as captured constants, which mx.compile bakes by
value and would silently freeze at their first value. Decode loops need one
more thing: jax lowers a while_loop's closed-over weights as loop-carried
state rather than region captures, so the operand subtrees are followed out
through invariant carries (`_hoist`) to the values the loop was handed.

METALJAX_QMM=0 disables the whole recognizer.
"""

from __future__ import annotations

import gc
import math
import os
import weakref
from collections import deque

import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes

# The quantized-matmul recognizer itself.
QMM_ENABLED = os.environ.get("METALJAX_QMM", "1") != "0"
# engine.execute gates the whole eager recognizer prologue on `ENABLED`, and
# metaljax.moe's expert gather shares that prologue (its verification has to
# run outside any trace), so either recognizer being on has to turn it on.
# The flag is read here rather than imported to keep moe -> qmm the only
# direction of the dependency.
ENABLED = QMM_ENABLED or os.environ.get("METALJAX_MOE", "1") != "0"
_DEBUG = os.environ.get("METALJAX_DEBUG", "") == "1"

# mx.quantize supports these only, and requires K % group_size == 0.
# Largest first: gs=32 measured 1.8x slower than 64/128 at decode.
_GROUP_SIZES = (128, 64, 32)
# Packs retained per recognized dot (keyed by source-buffer identity).
_MAX_PACKS = int(os.environ.get("METALJAX_QMM_CACHE", "2"))
# Give up on a dot whose operands keep changing: repacking every execute
# (weights being trained, or a "scale" that is really per-call data) costs
# far more than the literal chain it replaces.
_MAX_REPACKS = int(os.environ.get("METALJAX_QMM_REPACKS", "8"))
# Width of the scale/bias tables. "auto" keeps the source (bf16/f16) width
# when the folded bias is exactly representable in it -- always true for a
# zero-point-free quantization -- and widens to f32 otherwise, which keeps
# the reconstruction exact at the cost of 3-12% of the matmul (measured, M=1,
# 4-bit: scale traffic is +12.5% of the weight at group 128 instead of
# +6.25%). "source" always narrows (faster, bias rounded to <=0.5 ULP);
# "f32" never narrows.
_SCALE_WIDTH = os.environ.get("METALJAX_QMM_SCALES", "auto")
# Fuse dots that carry batching dimensions (a stack of per-expert weights).
_BATCH = os.environ.get("METALJAX_QMM_BATCH", "1") != "0"

# OCP MXFP4: one shared power-of-two scale per 32 elements of a row, and a
# 4-bit E2M1 element (sign | 2-bit exponent, bias 1 | 1-bit mantissa). The
# magnitudes below are indexed by the low three bits of the code; the sign
# bit is bit 3. MLX's `mode="mxfp4"` kernel uses exactly this encoding, and
# so does the HF checkpoint format -- see `mxfp4_codes`.
_E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MXFP4_GROUP = 32

# Diagnostics (tests assert on these; nothing else reads them).
STATS = {"recognized": 0, "packs": 0, "fallbacks": 0, "perms": 0,
         "mxfp4": 0, "batched": 0, "shared": 0}


def stats() -> dict:
    return dict(STATS)


def reset_stats():
    for k in STATS:
        STATS[k] = 0


_INT_ELS = {"i4", "ui4", "i8", "i16", "i32", "i64",
            "ui8", "ui16", "ui32", "ui64"}
_FLOAT_ELS = {"f16", "f32", "bf16"}
_FLOAT_MX = (mx.float32, mx.float16, mx.bfloat16)
_SHAPE_OPS = ("stablehlo.reshape", "stablehlo.transpose")


class _Reject(Exception):
    """A candidate does not match (or cannot be trusted): run it literally."""


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


def pack_codes(codes: mx.array, bits: int) -> mx.array:
    """Unsigned codes `[..., K]` -> uint32 words `[..., K*bits//32]`.

    MLX packs each row of the last axis as one contiguous little-endian bit
    stream, LSB first: element i occupies bits [i*bits, (i+1)*bits). Only
    bits that divide 32 (2/4/8/16) are handled here -- the ones we emit --
    so no value ever straddles a word boundary.
    """
    if 32 % bits:
        raise ValueError(f"pack_codes: bits={bits} does not divide 32")
    per = 32 // bits
    k = codes.shape[-1]
    if k % per:
        raise ValueError(f"pack_codes: K={k} not a multiple of {per}")
    lead = list(codes.shape[:-1])
    c = mx.reshape(codes.astype(mx.uint32), lead + [k // per, per])
    mask = mx.array((1 << bits) - 1, mx.uint32)
    out = None
    for i in range(per):
        # Defensive mask: an out-of-range code would otherwise spill into the
        # NEXT element's bits and corrupt an unrelated weight. Callers are
        # expected to have range-checked (see _build_pack).
        v = mx.bitwise_and(c[..., i], mask)
        if i:
            v = mx.left_shift(v, mx.array(i * bits, mx.uint32))
        out = v if out is None else mx.bitwise_or(out, v)
    return out


def pack_exact(codes: mx.array, scales: mx.array, zeros, bits: int,
               scale_dtype=None, offset=None):
    """Repack already-quantized weights into MLX's affine format.

    `codes` are signed codes `[N, K]`; `scales` and `zeros` are the per-group
    values `[N, K/group_size]` of the source quantization
    `w = scale * (code - zero)`. Returns `(packed_u32, scales, biases)` for
    `mx.quantized_matmul(..., bits=bits)`.

    The algebra: MLX dequantizes `scales*q_hat + biases` with an UNSIGNED
    `q_hat`, so shifting the codes by any integer `offset` that makes them
    non-negative works, as long as the shift is undone in the bias:

        q_hat  = code + offset          (must fit in `bits`)
        biases = -scale * (offset + zero)

    `offset` defaults to `2^(bits-1)`, which is exactly the signed-code
    convention. Whenever `scale * (offset + zero)` is representable the
    reconstruction is bit-identical to a float32 dequantization, because
    MLX's kernel evaluates the dequant as a single FMA.
    """
    off = (1 << (bits - 1)) if offset is None else int(offset)
    packed = pack_codes(mx.contiguous(codes).astype(mx.int32) + off, bits)
    s32 = mx.contiguous(scales).astype(mx.float32)
    z32 = (mx.contiguous(zeros).astype(mx.float32)
           if zeros is not None else mx.array(0.0, mx.float32))
    b32 = -(s32 * (z32 + off))
    if (scale_dtype is not None and scale_dtype != mx.float32
            and _SCALE_WIDTH != "f32"):
        # Keep the source (bf16/f16) width when nothing is lost by it: it
        # halves scale traffic and keeps the output in the compute dtype.
        # zp == 0 always qualifies (the bias is a power-of-two multiple of
        # the scale); a general zero point usually does not.
        if _SCALE_WIDTH == "source" or (_lossless(s32, scale_dtype)
                                        and _lossless(b32, scale_dtype)):
            return packed, s32.astype(scale_dtype), b32.astype(scale_dtype)
    return packed, s32, b32


def _lossless(x: mx.array, dtype) -> bool:
    return bool(mx.all(x.astype(dtype).astype(mx.float32) == x).item())


# --------------------------------------------------------------------------
# MXFP4
# --------------------------------------------------------------------------


def _uint_view(dtype):
    """The unsigned integer type `dtype`'s bit pattern fits, or None."""
    return {mx.float32: mx.uint32, mx.bfloat16: mx.uint16,
            mx.float16: mx.uint16}.get(dtype)


def mxfp4_codes(values: mx.array) -> mx.array:
    """4-bit MXFP4 codes for values that lie EXACTLY on the E2M1 grid.

    Raises `_Reject` on the first value that does not. Works on the bit
    pattern rather than on the numbers so that the sign of a zero survives
    (code 8 is -0.0, code 0 is +0.0) and so that "exactly" means exactly:
    the grid magnitudes are converted into `values`' own dtype and compared
    as integers, which no float comparison can round into agreement.
    """
    uint = _uint_view(values.dtype)
    if uint is None:
        raise _Reject(f"MXFP4 values in {values.dtype}")
    bits = mx.contiguous(values).view(uint)
    width = 32 if uint == mx.uint32 else 16
    sign = mx.right_shift(bits, mx.array(width - 1, uint))
    mag = mx.bitwise_and(bits, mx.array((1 << (width - 1)) - 1, uint))
    grid = np.array(mx.array(_E2M1_MAGS, dtype=values.dtype).view(uint))
    if len(set(grid.tolist())) != len(_E2M1_MAGS):
        raise _Reject(f"the E2M1 grid is not distinct in {values.dtype}")
    codes = mx.zeros(mag.shape, mx.uint8)
    ok = mag == mx.array(int(grid[0]), uint)
    for i in range(1, len(_E2M1_MAGS)):
        hit = mag == mx.array(int(grid[i]), uint)
        codes = mx.where(hit, mx.array(i, mx.uint8), codes)
        ok = mx.bitwise_or(ok, hit)
    codes = mx.bitwise_or(codes, mx.left_shift(sign.astype(mx.uint8),
                                               mx.array(3, mx.uint8)))
    # One reduction, one sync: the per-element masks die with this call.
    good = mx.all(ok)
    mx.eval(codes, good)
    if not bool(good.item()):
        raise _Reject("weight values are not on the MXFP4 (E2M1) grid")
    return codes


def mxfp4_scale_bytes(scales: mx.array) -> mx.array:
    """E8M0 bytes for per-group scales that are EXACT powers of two.

    An E8M0 scale is `2**(byte - 127)`, i.e. an f32 with a zero mantissa --
    so the byte is just the f32 exponent field, and requiring the mantissa
    (and the sign) to be zero is the whole verification.

    Exponent fields 0 and 255 are rejected rather than encoded: field 0
    means zero or a subnormal (never an exact power of two we could name)
    and field 255 means an infinity or a NaN. Byte 255 is also NaN in E8M0
    itself, so there is nothing to lose.
    """
    bits = mx.contiguous(scales.astype(mx.float32)).view(mx.uint32)
    exp = mx.bitwise_and(mx.right_shift(bits, _u32(23)), _u32(0xFF))
    bad = mx.any(mx.bitwise_or(
        mx.bitwise_and(bits, _u32(0x807FFFFF)) != _u32(0),   # sign or mantissa
        mx.bitwise_or(exp == _u32(0), exp == _u32(0xFF))))
    out = exp.astype(mx.uint8)
    mx.eval(out, bad)
    if bool(bad.item()):
        raise _Reject("MXFP4 group scales are not exact positive powers of "
                      "two")
    return out


# --------------------------------------------------------------------------
# regrouping an interleaved contraction axis
# --------------------------------------------------------------------------

# Rows hashed per pass. The scale/zero maps are full weight size, so the
# digest must not materialize a second copy of one.
_KEY_CHUNK = 1 << 22

_MIX = 2246822519


def _u32(v):
    return mx.array(v, mx.uint32)


def _mix(u):
    """A cheap 32-bit avalanche (multiply-xorshift), elementwise."""
    u = u * _u32(_MIX)
    return u ^ mx.right_shift(u, _u32(15))


def _bits_u32(x):
    """`x` widened to uint32 injectively (equal values -> equal words)."""
    if x.dtype in _FLOAT_MX:
        # f32 is the widest float MLX has, and bf16/f16 -> f32 is exact, so
        # this distinguishes every distinct value (including -0.0 vs 0.0).
        return x.astype(mx.float32).view(mx.uint32)
    return x.astype(mx.uint32)


def _column_keys(x) -> np.ndarray:
    """Per-column digest of `x` ([..., K]) as a numpy uint32 `[K, 2]`.

    Equal columns always digest equally; distinct columns collide with
    probability ~2^-64, and a collision can only MERGE two groups -- which
    the exact group-constancy check downstream rejects if the merge was not
    legitimate. The digest is a sum of per-row terms in wrapping uint32, so
    it does not depend on the order MLX reduces in.
    """
    k = x.shape[-1]
    x = mx.reshape(x, (-1, k))
    n = x.shape[0]
    step = max(1, min(n, _KEY_CHUNK // max(k, 1)))
    h1 = mx.zeros((k,), mx.uint32)
    h2 = mx.zeros((k,), mx.uint32)
    for lo in range(0, n, step):
        u = _bits_u32(mx.contiguous(x[lo:lo + step]))
        rows = mx.reshape(mx.arange(lo, lo + u.shape[0]).astype(mx.uint32),
                          (-1, 1))
        r = _mix(rows * _u32(0x9E3779B9) + _u32(1))
        v = _mix(u)
        h1 = h1 + mx.sum(v * mx.bitwise_or(r, _u32(1)), axis=0)
        h2 = h2 + mx.sum(_mix(v ^ r), axis=0)
        # Settle each pass so the chunk's intermediates die with it.
        mx.eval(h1, h2)
    return np.stack([np.array(h1), np.array(h2)], axis=1)


def _regroup(k: int, maps) -> np.ndarray | None:
    """A permutation of the contraction axis that un-interleaves the groups.

    Quantization groups are, by definition, runs of columns sharing one
    (scale, zero) pair. When they are interleaved along the canonical K axis
    the pack cannot read them off contiguous slices -- but permuting K on
    BOTH dot operands is exact, so clustering the columns by their (scale,
    zero) column pair and sorting by cluster recovers a layout that packs.

    Returns None when no permutation can help (run lengths admit no legal
    group size) or when the columns are already grouped, in which case the
    caller's own search has already had its chance and the identity
    permutation must stay the zero-overhead path.
    """
    keys = np.concatenate([_column_keys(x) for x in maps if x is not None],
                          axis=1)
    # Stable first-occurrence ids: `np.unique` returns them in sorted-key
    # order, which would reshuffle whole groups for no reason.
    _, first, inv = np.unique(keys, axis=0, return_index=True,
                              return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    order = np.argsort(first)
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    ids = rank[inv]
    # Every cluster becomes a contiguous run of its own length; a legal group
    # size must divide all of them (then it also divides every run's offset,
    # since the runs are laid out end to end). Duplicate (scale, zero)
    # columns merge two groups into one double-length run -- harmless, the
    # values in it are identical by construction.
    runs = np.bincount(ids)
    g = 0
    for length in runs:
        g = math.gcd(g, int(length))
    if not any(g and g % s == 0 for s in _GROUP_SIZES):
        return None
    perm = np.argsort(ids, kind="stable").astype(np.int32)
    if np.array_equal(perm, np.arange(k, dtype=np.int32)):
        return None
    return perm


def _take_k(x, perm):
    """`x[..., perm]`, materialized (these are full-weight-size maps)."""
    out = mx.contiguous(mx.take(x, perm, axis=-1))
    mx.eval(out)
    return out


# --------------------------------------------------------------------------
# structural matching
# --------------------------------------------------------------------------


def _owner(v):
    return v.owner.operation if isinstance(v, ir.OpResult) else None


def _okey(op):
    """Identity key for an operation: its first result.

    ir.Value hashes by the underlying MLIR value; the python wrappers around
    operations are transient, and this project has twice been bitten by
    keying sets on id() of such wrappers.
    """
    res = op.results
    return res[0] if len(res) else None


def _el(v) -> str:
    return str(_ir.tensor_type(v).element_type)


def _is_int(v) -> bool:
    try:
        return _el(v) in _INT_ELS
    except Exception:
        return False


def _shape(v) -> list[int]:
    return list(_ir.tensor_type(v).shape)


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _strip(v, names):
    """Walk down through `names`-kind ops. Returns (base, ops-outermost-last)."""
    chain = []
    while True:
        o = _owner(v)
        if o is None or o.name not in names:
            return v, list(reversed(chain))
        chain.append(o)
        v = o.operands[0]


def _strip_shape(v):
    return _strip(v, _SHAPE_OPS + ("stablehlo.convert",))


def _strip_converts(v):
    return _strip(v, ("stablehlo.convert",))


_SIGNED_WIDTH = {"i4": 4, "i8": 8, "i16": 16, "i32": 32, "i64": 64}


def _parse_codes(v):
    """`v` as an integer-code operand: (codes, zero_point, subtract_range).

    `subtract_range` is the (lo, hi) the graph's `codes - zero` is computed
    in when that subtraction happens in INTEGER arithmetic -- the rewrite
    evaluates it exactly, so a graph that would wrap must not be rewritten.
    It is None when the subtraction is done in floating point (what keras
    emits: both operands are converted first).
    """
    base, _ = _strip_converts(v)
    o = _owner(base)
    if o is not None and o.name == "stablehlo.subtract":
        c, _ = _strip_converts(o.operands[0])
        if not _is_int(c):
            return None
        rng = None
        if _is_int(base):
            w = _SIGNED_WIDTH.get(_el(base))
            if w is None:
                return None  # unsigned wrap semantics: not worth modelling
            rng = (-(1 << (w - 1)), (1 << (w - 1)) - 1)
        return c, o.operands[1], rng
    if _is_int(base):
        return base, None, None
    return None


def _dot_dims(op):
    from metaljax.ops.linalg import _dot_dims as impl
    return impl(op)


def _bcast_chain(v):
    """Peel a chain of broadcast_in_dim (jax emits one per rank step).

    Returns (base, dims) where `dims[i]` is the dimension of `v` that base's
    dimension i lands in, or (v, None) when the head is not a broadcast.
    """
    dims = list(range(len(_shape(v))))
    base = None
    while True:
        o = _owner(v)
        if o is None:
            break
        if o.name == "stablehlo.broadcast_in_dim":
            d = _ir.i64_list(o, "broadcast_dimensions")
            dims = [dims[j] for j in d]
            v = o.operands[0]
            base = v
            continue
        if o.name == "stablehlo.convert":
            v = o.operands[0]
            continue
        break
    return (v, dims) if base is not None else (v, None)


class Match:
    """One recognized quantized matmul."""

    __slots__ = ("root", "key", "lhs", "codes", "zero", "scale", "recip",
                 "post", "bcast_dims", "ops", "arg_indices", "required",
                 "lperm", "rperm", "rshape", "bshape", "mshape", "nshape",
                 "B", "M", "K", "N", "out_dtype", "disabled", "slot", "gs",
                 "bits", "packs", "repacks", "name", "sub_range", "has_perm",
                 "mode", "nvals", "swapped", "absorbed")

    def __init__(self):
        self.recip = False
        # Set by metaljax.moe when an expert-gather rewrite takes over this
        # dot: the weight is still packed here, but the dense
        # quantized_matmul is never emitted (gather_qmm replaces it).
        self.absorbed = False
        self.mode = "affine"
        self.zero = None
        self.sub_range = None
        self.post = []
        self.bcast_dims = None
        self.ops = {}
        self.required = []
        self.arg_indices = ()
        self.disabled = False
        self.slot = -1
        self.nvals = 0
        self.gs = 0
        self.bits = 0
        self.has_perm = False
        self.packs = []
        self.repacks = 0
        self.name = "?"
        self.bshape = []
        self.B = 1
        self.swapped = False


class _Pack:
    __slots__ = ("refs", "w", "scales", "biases", "perm", "gs", "bits", "mode")

    def __init__(self, leaves, w, scales, biases, perm, gs, bits,
                 mode="affine"):
        # Weak references: identity is what we care about, and pinning a
        # multi-GB weight buffer alive because we packed a copy of it would
        # be its own bug. A dead referent simply misses and repacks.
        self.refs = [weakref.ref(a) for a in leaves]
        self.w, self.scales, self.biases = w, scales, biases
        self.perm = perm
        self.gs, self.bits = gs, bits
        self.mode = mode

    def arrays(self):
        """The packed arrays, in the order `emit` reads them back.

        Variable length: MXFP4 has no bias table, and a permutation is only
        carried when the groups were interleaved. `Match.nvals` records how
        many slots this pack occupies in the traced-input list.
        """
        out = [self.w, self.scales]
        if self.biases is not None:
            out.append(self.biases)
        if self.perm is not None:
            out.append(self.perm)
        return out

    def matches(self, leaves) -> bool:
        if len(leaves) != len(self.refs):
            return False
        return all(r() is a for r, a in zip(self.refs, leaves))


class State:
    """Per-program recognizer state (quantized matmuls AND MoE gathers).

    `moe` holds metaljax.moe.Match objects. They share this state because
    they share the machinery: the same skip set, the same root dispatch in
    the interpreter, and -- when an expert dot was quantized -- the same
    packed weights, which a gathered dispatch reads instead of the dense
    `quantized_matmul` it replaces.
    """

    __slots__ = ("matches", "skip", "roots", "values", "active", "moe")

    def __init__(self):
        self.matches = []
        self.moe = []
        self.skip = frozenset()
        self.roots = {}
        self.values = []
        self.active = False

    def rebuild(self):
        live = [m for m in self.matches if not m.disabled]
        self.roots = {m.key: m for m in live if not m.absorbed}
        skip = set()
        for m in live:
            skip.update(m.ops)
        for g in self.moe:
            if g.disabled:
                continue
            self.roots[g.key] = g
            skip.update(g.skip_keys)
        self.skip = frozenset(skip)
        self.active = bool(self.roots)


def _walk_blocks(block):
    yield block
    for op in block.operations:
        for region in op.operation.regions:
            for b in region.blocks:
                yield from _walk_blocks(b)


_OPAQUE = ("stablehlo.while", "stablehlo.if", "stablehlo.case",
           "stablehlo.custom_call", "stablehlo.optimization_barrier")


def _hoist(v):
    """Follow a value out of the loops that merely carry it around.

    jax does NOT lower a while_loop's closed-over constants as region
    captures: they become loop-carried state that the body returns
    unchanged (`%iterArg_0 = %arg0` ... `stablehlo.return %iterArg_0`).
    A decode loop's weights therefore arrive as body block arguments, and
    the value that is actually constant for the whole loop -- the one the
    packing prologue can evaluate -- is the while's initial operand.
    """
    for _ in range(8):  # depth guard; real programs nest 1-2 loops
        if not isinstance(v, ir.BlockArgument):
            return v
        blk = v.owner
        owner = blk.owner
        if owner is None:
            return v
        op = owner.operation if hasattr(owner, "operation") else owner
        if op.name != "stablehlo.while":
            return v
        i = v.arg_number
        body = op.regions[1].blocks[0]
        term = list(body.operations)[-1].operation
        if term.operands[i] != list(body.arguments)[i]:
            return v  # the carry changes: not loop-invariant
        v = op.operands[i]
    return v


def _closure(values, main_args, ops, arg_indices, register=()):
    """Backward closure of `values`, stopping at block arguments/constants.

    Every block argument reached must be one of @main's -- otherwise the
    subtree depends on a loop carry and cannot be packed once. `register`
    ops join the absorbed set without having their own operands walked (the
    caller passes the operands it wants absorbed in `values`; the per-channel
    form absorbs a dot's weight operand but not its activations).
    """
    for o in register:
        ops[_okey(o)] = o
    stack = list(values)
    seen = set()
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        if isinstance(v, ir.BlockArgument):
            idx = main_args.get(v)
            if idx is not None:
                arg_indices.add(idx)
                continue
            outer = _hoist(v)
            if outer is v:
                raise _Reject("operand depends on an inner block argument")
            stack.append(outer)
            continue
        o = _owner(v)
        if o is None:
            raise _Reject("operand is neither a block argument nor a result")
        if o.name in _OPAQUE:
            raise _Reject(f"operand subtree contains {o.name}")
        ops.setdefault(_okey(o), o)
        stack.extend(o.operands)


def _finish(ctx, m, root, absorb, dot, qside=1, required=(), register=()):
    """Fill in `m` for a dot whose operand `qside` is the quantized one."""
    lb, rb, lc, rc = _dot_dims(dot)
    (qb, qc), (ab, ac) = (((rb, rc), (lb, lc)) if qside == 1
                          else ((lb, lc), (rb, rc)))
    if (qb or ab) and not _BATCH:
        raise _Reject("batching dimensions")
    quant, act = dot.operands[qside], dot.operands[1 - qside]
    if _el(act) not in _FLOAT_ELS:
        raise _Reject(f"activation element type {_el(act)}")
    out_el = _el(root.results[0])
    if out_el not in _FLOAT_ELS:
        raise _Reject(f"result element type {out_el}")
    qshape, ashape = _shape(quant), _shape(act)
    if [ashape[d] for d in ab] != [qshape[d] for d in qb]:
        raise _Reject("batching dimensions disagree")
    afree = [d for d in range(len(ashape)) if d not in ac and d not in ab]
    qfree = [d for d in range(len(qshape)) if d not in qc and d not in qb]
    B = _prod(ashape[d] for d in ab)
    M, K, N = (_prod(ashape[d] for d in afree), _prod(ashape[d] for d in ac),
               _prod(qshape[d] for d in qfree))
    if B == 0 or M == 0 or K == 0 or N == 0:
        raise _Reject("empty matmul")
    if K % _GROUP_SIZES[-1]:
        raise _Reject(f"K={K} is not a multiple of {_GROUP_SIZES[-1]}")
    if K != _prod(qshape[d] for d in qc):
        raise _Reject("contracting dimensions disagree")

    m.root = root
    m.key = _okey(root)
    m.lhs = act
    m.swapped = qside == 0
    # dot_general's result is laid out batch dims, then LHS free, then RHS
    # free. `quantized_matmul` on [B, M, K] x [B, N, K] returns [B, M, N],
    # which is that layout when the quantized operand is the rhs and its
    # transpose when it is the lhs (jax lowers `th,emh->etm` that way).
    m.lperm = ab + afree + ac
    m.rperm = qb + qfree + qc
    m.rshape = qshape
    m.bshape = [ashape[d] for d in ab]
    m.mshape = [ashape[d] for d in afree]
    m.nshape = [qshape[d] for d in qfree]
    m.B, m.M, m.K, m.N = B, M, K, N
    m.out_dtype = dtypes.mx_result_dtype(root.results[0])
    m.name = (f"{m.M}x{m.K}x{m.N}" if not m.bshape
              else f"{'x'.join(map(str, m.bshape))}|{m.M}x{m.K}x{m.N}")
    if m.swapped:
        m.name += "'"

    main_args, donated = ctx
    arg_indices = set()
    _closure(absorb, main_args, m.ops, arg_indices, register)
    if arg_indices & donated:
        raise _Reject("quantized operand is donated (buffer may be reused)")
    m.arg_indices = tuple(sorted(arg_indices))
    # Ops that MUST end up skipped for the rewrite to be a win: everything
    # between the affine reconstruction and the dot. If some other consumer
    # forces the dequantized weight to be materialized anyway, running a
    # quantized matmul on top of it would only add work.
    m.required = [_okey(o) for o in required]
    m.required += [_okey(o) for o in m.post]
    return m


def _has_int_leaf(v, limit=64):
    """True if `v`'s subtree bottoms out on an integer tensor somewhere.

    A weight-reconstruction chain always does -- the codes are integers,
    however elaborately they are unpacked. Nothing else about the chain is
    inspected: this is only a cheap guard that keeps the MXFP4 branch from
    adopting an ordinary `x @ (w * mask)` and paying a full-weight-size
    evaluation to find out.
    """
    stack, seen = [v], set()
    while stack and len(seen) < limit:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if _is_int(cur):
            return True
        o = _owner(cur)
        if o is None or o.name in _OPAQUE:
            continue
        stack.extend(o.operands)
    return False


def _split_scaled(mul):
    """(values, scale) for a `values * broadcast(per-group scale)` product.

    Both operands are floats in the MXFP4 form -- the E2M1 decode has
    already turned the codes into numbers -- so the split is made
    structurally: the scale is the operand BROADCAST from a tensor holding
    exactly one value per 32, which is the only group size MXFP4 has.

    That exact 32 is what keeps this from adopting an RMS norm. `x *
    broadcast(rsqrt(...))` has the same shape as an MXFP4 reconstruction and
    reaches an integer leaf too (the token ids under the embedding gather),
    but its scale is broadcast one-per-ROW, so the ratio is the row length
    rather than 32.
    """
    cands = []
    total = _prod(_shape(mul.results[0]))
    for i in (0, 1):
        v = mul.operands[i]
        if _el(v) not in _FLOAT_ELS:
            raise _Reject(f"multiply operand element type {_el(v)}")
        base, dims = _bcast_chain(v)
        if dims is None:
            continue
        n = _prod(_shape(base))
        if n and n * _MXFP4_GROUP == total:
            cands.append((n, i))
    if not cands:
        raise _Reject("neither multiply operand broadcasts one scale per "
                      f"{_MXFP4_GROUP} values")
    if len(cands) == 2:
        raise _Reject("cannot tell the scale operand from the values")
    return mul.operands[1 - cands[0][1]], mul.operands[cands[0][1]]


def _try_affine(ctx, dot):
    """The weight-times-scale forms, on whichever operand carries them.

    jax puts the quantized weight on the RHS for a plain projection but on
    the LHS for an einsum like `th,emh->etm` (the expert gate/up
    projection), so both sides are tried. The RHS goes first: it is the
    common case, and trying it first keeps every already-recognized graph on
    exactly the path it was on before.
    """
    reject = None
    for qside in (1, 0):
        try:
            return _try_affine_side(ctx, dot, qside)
        except _Reject as e:
            reject = reject or e
    raise reject


def _try_affine_side(ctx, dot, qside):
    """dot(x, [shape ops](mul(sub(cvt(codes), cvt(zeros)), scales))), or the
    MXFP4 form dot(x, [shape ops](mul(decode(codes), broadcast(scale))))."""
    qop = dot.operands[qside]
    base, post = _strip_shape(qop)
    mul = _owner(base)
    if mul is None or mul.name != "stablehlo.multiply":
        raise _Reject("the operand is not a multiply")
    parsed = scale = None
    for i in (0, 1):
        parsed = _parse_codes(mul.operands[i])
        if parsed is not None:
            scale = mul.operands[1 - i]
            break
    m = Match()
    if parsed is None:
        # No integer operand: MXFP4's grid is non-uniform, so its codes have
        # become floats before the scale is applied.
        values, scale = _split_scaled(mul)
        if not _has_int_leaf(values):
            raise _Reject("the scaled operand holds no integer codes")
        m.mode = "mxfp4"
        m.codes = values
    else:
        m.codes, m.zero, m.sub_range = parsed
    if _el(scale) not in _FLOAT_ELS:
        raise _Reject(f"scale element type {_el(scale)}")
    m.scale = scale
    m.post = [o for o in post if o.name in _SHAPE_OPS]
    _finish(ctx, m, dot, [qop], dot, qside=qside, required=(mul,))
    return m


def _try_perchannel(ctx, div):
    """divide(dot(x, [shape ops](cvt(codes))), broadcast(scale))."""
    num, den = div.operands
    dot = _owner(num)
    if dot is None or dot.name != "stablehlo.dot_general":
        raise _Reject("divide numerator is not a dot")
    codes, post = _strip_shape(dot.operands[1])
    if not _is_int(codes):
        raise _Reject("dot rhs is not an integer code tensor")
    scale, dims = _bcast_chain(den)
    if dims is None:
        raise _Reject("divisor is not a broadcast")
    if _el(scale) not in _FLOAT_ELS:
        raise _Reject("divisor is not float")
    lb, rb, lc, _rc = _dot_dims(dot)
    if lb or rb:
        raise _Reject("batching dimensions")
    nm = len(_shape(dot.operands[0])) - len(lc)  # leading output (M) dims
    if any(d < nm for d in dims):
        # A divisor that varies along the M axis is not a weight scale.
        raise _Reject("divisor depends on the batch axis")
    m = Match()
    m.recip = True
    m.codes = codes
    m.scale = scale
    m.bcast_dims = [d - nm for d in dims]
    m.post = [o for o in post if o.name in _SHAPE_OPS]
    _finish(ctx, m, div, [dot.operands[1], den], dot,
            required=(dot,), register=(dot,))
    return m


def _users(op):
    for r in op.results:
        for use in r.uses:
            yield use.owner.operation


def _prune(cands):
    """Decide which absorbed ops may actually be skipped.

    An op can only be skipped if every consumer of every result is itself
    skipped (or is a rewritten root) -- otherwise the value is still needed.
    Constants and their broadcasts are routinely shared between layers by
    CSE; those simply stay, at a couple of op-units each. A candidate whose
    reconstruction ops cannot be skipped is dropped entirely.
    """
    live = list(cands)
    while True:
        ops = {}
        for m in live:
            ops.update(m.ops)
        roots = {m.key for m in live}
        work = deque(ops)
        while work:
            key = work.popleft()
            op = ops.get(key)
            if op is None:
                continue
            keep = True
            for u in _users(op):
                ukey = _okey(u)
                if ukey is None or (ukey not in ops and ukey not in roots):
                    keep = False
                    break
            if keep:
                continue
            del ops[key]
            for opd in op.operands:
                if isinstance(opd, ir.OpResult):
                    k = _okey(_owner(opd))
                    if k in ops:
                        work.append(k)
        kept = []
        for m in live:
            missing = [k for k in m.required if k not in ops]
            if missing:
                if _DEBUG:
                    print(f"[metaljax] qmm: dropped {m.name} "
                          f"(reconstruction is used elsewhere)", flush=True)
                continue
            kept.append(m)
        if len(kept) == len(live):
            for m in live:
                m.ops = {k: v for k, v in m.ops.items() if k in ops}
            return live
        live = kept


def analyze(interp) -> State:
    """Structural analysis of a program, once. Never touches values."""
    st = interp._qmm
    if st is not None:
        return st
    st = State()
    interp._qmm = st
    try:
        with interp.context:
            if not QMM_ENABLED:
                raise _Reject("METALJAX_QMM=0")
            ctx = ({a: i for i, a in
                    enumerate(interp._main_block().arguments)},
                   set(interp.donated_args))
            cands = []
            for block in _walk_blocks(interp._main_block()):
                for op in block.operations:
                    o = op.operation
                    try:
                        if o.name == "stablehlo.dot_general":
                            cands.append(_try_affine(ctx, o))
                        elif o.name == "stablehlo.divide":
                            cands.append(_try_perchannel(ctx, o))
                    except _Reject:
                        pass
            # A candidate whose result feeds another candidate's operand
            # subtree is evaluated during that one's packing prologue; it
            # cannot also be rewritten at run time.
            absorbed = set()
            for m in cands:
                absorbed.update(m.ops)
            cands = [m for m in cands if m.key not in absorbed]
            st.matches = _prune(cands) if cands else []
            STATS["recognized"] += len(st.matches)
            if _DEBUG and st.matches:
                print(f"[metaljax] qmm: {len(st.matches)} quantized "
                      f"matmul(s) recognized", flush=True)
    except _Reject:
        st.matches = []                      # METALJAX_QMM=0
    except Exception as e:  # analysis must never break a program
        if _DEBUG:
            print(f"[metaljax] qmm: analysis failed ({e})", flush=True)
        st.matches = []
    # Expert-gather rewrites are found on top of the quantized ones: a MoE
    # dispatch whose dots were packed here reuses those packs and marks them
    # absorbed (see metaljax.moe).
    from metaljax import moe as _moe
    _moe.analyze(interp, st)
    st.rebuild()
    return st


# --------------------------------------------------------------------------
# first-execute verification + packing
# --------------------------------------------------------------------------


def _eval(interp, value, env):
    """Evaluate an operand subtree eagerly on concrete arguments."""
    from metaljax.interpreter import REGISTRY

    cached = env.get(value)
    if cached is not None:
        return cached
    if isinstance(value, ir.BlockArgument):
        # A loop-invariant carry: evaluate what the loop was handed.
        outer = _hoist(value)
        if outer is value:
            raise _Reject("unbound value in operand subtree")
        env[value] = _eval(interp, outer, env)
        return env[value]
    o = _owner(value)
    if o is None:
        raise _Reject("unbound value in operand subtree")
    ins = [_eval(interp, x, env) for x in o.operands]
    handler = REGISTRY.get(o.name)
    if handler is None:
        raise _Reject(f"no handler for {o.name}")
    out = handler(interp, o, ins, env)
    if isinstance(out, mx.array):
        out = [out]
    for r, v in zip(o.results, out):
        env[r] = v
    return env[value]


def _replay(x, post):
    for o in post:
        if o.name == "stablehlo.reshape":
            x = mx.reshape(x, list(_ir.tensor_type(o.results[0]).shape))
        else:
            x = mx.transpose(x, _ir.i64_list(o, "permutation"))
    return x


def _to_nk(x, m):
    """A rhs-shaped tensor as the [(B,) N, K] matrix `quantized_matmul` wants.

    Materialized on the spot: these are full-weight-sized reconstructions
    (a 262k-vocab head is ~800 MB per map), and leaving them lazy would keep
    every intermediate of the unpack chain alive until the pack is finished.
    """
    x = _replay(x, m.post)
    if list(x.shape) != m.rshape:
        raise _Reject(f"expected rhs shape {m.rshape}, got {list(x.shape)}")
    shape = (m.bshape + [m.N, m.K]) if m.bshape else [m.N, m.K]
    out = mx.contiguous(mx.reshape(mx.transpose(x, m.rperm), shape))
    mx.eval(out)
    return out


def _group_const(x, gs) -> bool:
    k = x.shape[-1]
    v = mx.reshape(x, (-1, k // gs, gs))
    return bool(mx.all(v == v[:, :, :1]).item())


def _group_heads(x, gs):
    """The first element of each group: `[..., K] -> [..., K/gs]`."""
    k = x.shape[-1]
    lead = list(x.shape[:-1])
    return mx.contiguous(
        mx.reshape(x, lead + [k // gs, gs])[..., 0])


def _pick_group(k, scale_map, zero_map):
    """The largest legal group size these maps are constant within."""
    for g in _GROUP_SIZES:
        if k % g:
            continue
        if _group_const(scale_map, g) and (
                zero_map is None or _group_const(zero_map, g)):
            return g
    return None


# Packs that are still alive, grouped by a digest of their contents. jax
# lowers a decode loop's PREFILL and its while body as two separate dots
# over one set of weights, so a model's whole packed weight set would
# otherwise be built and held twice (measured on gpt-oss-20b: 2 x 10.2 GB).
# Entries are weak: a pack lives exactly as long as some Match holds it.
_SHARED = {}
_MAX_SHARED = int(os.environ.get("METALJAX_QMM_SHARE", "512"))


def _digest(a) -> int:
    """A cheap content digest of a packed array (order-independent sum)."""
    u = _bits_u32(a) if a.dtype in _FLOAT_MX else a.astype(mx.uint32)
    u = mx.reshape(u, (-1,))
    idx = mx.arange(u.size).astype(mx.uint32)
    h = mx.sum(_mix(u ^ _mix(idx)))
    mx.eval(h)
    return int(h.item()) ^ (a.size << 8) ^ hash(str(a.dtype))


def _share(pk):
    """Alias `pk`'s arrays onto an identical earlier pack's, if there is one.

    Content-addressed on purpose: the digest only narrows the search, and
    nothing is aliased until the arrays compare EQUAL element for element.
    Two chains that happen to reconstruct the same weight therefore share
    one copy, and two that do not can never be confused -- no structural
    reasoning about the graphs is involved.
    """
    arrays = pk.arrays()
    key = (pk.mode, pk.gs, pk.bits, len(arrays),
           tuple(tuple(a.shape) for a in arrays), _digest(arrays[0]))
    bucket = _SHARED.get(key)
    if bucket is not None:
        for i, refs in enumerate(bucket):
            other = [r() for r in refs]
            if any(o is None for o in other):
                continue
            if all(a.dtype == b.dtype and bool(mx.all(a == b).item())
                   for a, b in zip(arrays, other)):
                pk.w, pk.scales = other[0], other[1]
                rest = list(other[2:])
                if pk.biases is not None:
                    pk.biases = rest.pop(0)
                if pk.perm is not None:
                    pk.perm = rest.pop(0)
                bucket.insert(0, bucket.pop(i))
                STATS["shared"] += 1
                return pk
    if len(_SHARED) > _MAX_SHARED:
        _SHARED.clear()          # coarse, and only ever costs a repack
    bucket = _SHARED.setdefault(key, [])
    # Drop entries whose packs have died with the executable that held them,
    # so a long-running worker that keeps requantizing fresh weights does
    # not accumulate dead references.
    bucket[:] = [refs for refs in bucket if all(r() is not None
                                                for r in refs)]
    bucket.insert(0, [weakref.ref(a) for a in arrays])
    return pk


def _record_pack(m, pk):
    pk = _share(pk)
    STATS["packs"] += 1
    if pk.perm is not None:
        STATS["perms"] += 1
    if pk.mode == "mxfp4":
        STATS["mxfp4"] += 1
    if m.bshape:
        STATS["batched"] += 1
    if _DEBUG:
        print(f"[metaljax] qmm: packed {m.name} mode={pk.mode} "
              f"bits={pk.bits} group={pk.gs} scales={pk.scales.dtype}"
              f"{' regrouped' if pk.perm is not None else ''}", flush=True)
    return pk


def _build_mxfp4_pack(m, values, scale_map, leaves):
    """Verify and repack an MXFP4 weight from its two evaluated factors.

    Both arrive full weight size, so each is dropped the moment the small
    per-group form has been derived from it.
    """
    gs = _MXFP4_GROUP
    if m.K % gs:
        raise _Reject(f"K={m.K} is not a multiple of {gs}")
    perm = None
    if not _group_const(scale_map, gs):
        # The same interleaving story as the affine path: permuting the
        # contraction axis on BOTH operands leaves the dot unchanged.
        perm = _regroup(m.K, (scale_map,))
        if perm is not None:
            perm = mx.array(perm)
            scale_map = _take_k(scale_map, perm)
            values = _take_k(values, perm)
        if not _group_const(scale_map, gs):
            raise _Reject("MXFP4 scales are not constant within a group "
                          f"of {gs}")
    sbytes = mxfp4_scale_bytes(_group_heads(scale_map, gs))
    scale_map = None
    codes = mxfp4_codes(values)
    values = None
    # The E2M1 nibble order is MLX's: element i of a row occupies bits
    # [4i, 4i+4) of the little-endian uint32 stream, the same layout the
    # affine 4-bit packer emits.
    w = pack_codes(codes, 4)
    codes = None
    mx.eval(w, sbytes)
    return _Pack(leaves, w, sbytes, None, perm, gs, 4, mode="mxfp4")


def _build_pack(interp, m, args, leaves):
    # One environment per operand subtree: they share only cheap prefixes
    # (the group-index cast), while each one's intermediates are full weight
    # size -- keeping all three alive at once would triple the peak.
    def evaluate(value):
        env = dict(zip(interp._main_block().arguments, args))
        return _to_nk(_eval(interp, value, env), m)

    if m.mode == "mxfp4":
        return _record_pack(
            m, _build_mxfp4_pack(m, evaluate(m.codes), evaluate(m.scale),
                                 leaves))

    codes = evaluate(m.codes)
    lo = int(mx.min(codes).item())
    hi = int(mx.max(codes).item())
    # The codes only have to FIT: any integer shift that makes them
    # non-negative is undone in the bias, so the code range decides the
    # width (a symmetric [-8, 7] weight lands on 4 bits either way).
    if hi - lo < 16:
        bits = 4
    elif hi - lo < 256:
        bits = 8
    else:
        raise _Reject(f"codes span [{lo}, {hi}]: more than 8 bits")
    # Prefer the signed-code convention (offset = 2^(bits-1)): its bias is a
    # power-of-two multiple of the scale, which is what keeps a zero-point-
    # free quantization exactly representable in bf16.
    offset = 1 << (bits - 1)
    if lo + offset < 0 or hi + offset >= (1 << bits):
        offset = -lo

    scale_dtype = None
    perm = None
    if m.recip:
        s = _eval(interp, m.scale,
                  dict(zip(interp._main_block().arguments, args)))
        scale_dtype = s.dtype  # a reciprocal is rarely exact in bf16, so
        # "auto" will widen to f32 here; METALJAX_QMM_SCALES=source forces
        # the narrow form for the traffic saving.
        interim = [1] * len(m.nshape)
        dims = m.bcast_dims
        if dims != sorted(dims):
            s = mx.transpose(s, sorted(range(len(dims)), key=lambda i: dims[i]))
            dims = sorted(dims)
        for i, d in enumerate(dims):
            interim[d] = s.shape[i]
        s = mx.reshape(mx.broadcast_to(mx.reshape(s, interim), m.nshape),
                       (m.N, 1))
        # The graph divides the OUTPUT by a per-output-channel scale; fold
        # the reciprocal into the weight scale. (A rounding change: 1/s is
        # computed once in f32 instead of dividing every output element.)
        scales = mx.divide(mx.array(1.0, mx.float32), s.astype(mx.float32))
        gs = next((g for g in _GROUP_SIZES if m.K % g == 0), None)
        if gs is None:
            raise _Reject(f"K={m.K} has no legal group size")
        scales = mx.broadcast_to(scales, (m.N, m.K // gs))
        zeros = None
    else:
        scale_map = evaluate(m.scale)
        zero_map = evaluate(m.zero) if m.zero is not None else None
        gs = _pick_group(m.K, scale_map, zero_map)
        note = ""
        if gs is None:
            # The groups may still be there, interleaved: recover the
            # permutation that makes them contiguous and re-verify EXACTLY
            # (the clustering is only a proposal -- `_pick_group` on the
            # permuted maps is what the pack's exactness rests on).
            perm = _regroup(m.K, (scale_map, zero_map))
            if perm is not None:
                perm = mx.array(perm)
                # Rebind as each permuted copy lands: these are full weight
                # size, and holding the originals as well would raise the
                # peak by one whole map each. Nothing below reads the
                # unpermuted ones -- a failed verification rejects the dot.
                scale_map = _take_k(scale_map, perm)
                if zero_map is not None:
                    zero_map = _take_k(zero_map, perm)
                codes = _take_k(codes, perm)
                gs = _pick_group(m.K, scale_map, zero_map)
                if gs is None:
                    perm = None
                    note = " (even regrouped)"
        if gs is None:
            raise _Reject(
                f"scales/zeros are not constant within any group{note}")
        scale_dtype = scale_map.dtype
        scales = _group_heads(scale_map, gs)
        zeros = None
        if zero_map is not None:
            # In f32 throughout: the zero map may be an integer tensor, and
            # MLX refuses to compare one against out-of-range literals.
            zeros = _group_heads(zero_map, gs).astype(mx.float32)
            if not bool(mx.all(zeros == mx.round(zeros)).item()):
                raise _Reject("zero points are not integers")
            if bool(mx.any(mx.abs(zeros) > 32768).item()):
                raise _Reject("zero points out of range")
            if m.sub_range is not None:
                # The graph subtracts the zero point in INTEGER arithmetic,
                # which wraps; the rewrite computes it exactly. Only fuse
                # when nothing can wrap.
                zlo = int(mx.min(zeros).item())
                zhi = int(mx.max(zeros).item())
                if (lo - zhi) < m.sub_range[0] or (hi - zlo) > m.sub_range[1]:
                    raise _Reject("integer zero-point subtraction can wrap")
        # Free the full-size maps before packing: only the per-group values
        # are needed from here on.
        mx.eval(*[a for a in (scales, zeros) if a is not None])
        scale_map = zero_map = None

    w, scales, biases = pack_exact(codes, scales, zeros, bits,
                                   scale_dtype=scale_dtype, offset=offset)
    # Materialize: a lazy packed weight would pin the whole reconstruction
    # graph (and its full-size intermediates) for the life of the cache.
    mx.eval(w, scales, biases)
    return _record_pack(m, _Pack(leaves, w, scales, biases, perm, gs, bits))


def _resolve(interp, m, args):
    """(pack, freshly_built) for these argument buffers."""
    leaves = [args[i] for i in m.arg_indices]
    for i, pk in enumerate(m.packs):
        if pk.matches(leaves):
            if i:
                m.packs.insert(0, m.packs.pop(i))
            return pk, False
    m.repacks += 1
    if m.repacks > _MAX_REPACKS:
        raise _Reject(f"operands changed {m.repacks} times; repacking costs "
                      f"more than the chain it replaces")
    pk = _build_pack(interp, m, args, leaves)
    m.packs.insert(0, pk)
    del m.packs[_MAX_PACKS:]
    return pk, True


def prologue(interp, args) -> bool:
    """Pack every recognized weight against these arguments (eagerly).

    Returns True when the set of rewritten dots changed, i.e. any cached
    trace built around the previous structure has to be dropped.
    """
    st = interp._qmm
    if st is None:
        st = analyze(interp)
    if not st.matches and not st.moe:
        return False
    changed = False
    values = []
    packed_any = False
    with interp.context:
        for m in st.matches:
            if m.disabled:
                continue
            try:
                pk, fresh = _resolve(interp, m, args)
                packed_any = packed_any or fresh
                if fresh:
                    # Packing one weight peaks ~10 GB active and leaves ~8 GB
                    # in MLX's cache at gpt-oss scale; clearing only at the
                    # END of the prologue let 24 layers of that accumulate
                    # (the +14 GB/sample ramp that guard-killed the row-7
                    # re-measure). Reclaim per pack: gc first, dead refcycles
                    # pin buffers clear_cache cannot free (CLAUDE.md item 19).
                    gc.collect()
                    mx.clear_cache()
            except Exception as e:
                m.disabled = True
                changed = True
                STATS["fallbacks"] += 1
                if _DEBUG:
                    print(f"[metaljax] qmm: {m.name} falls back to the "
                          f"literal chain ({e})", flush=True)
                continue
            if (m.mode, m.gs, m.bits, m.has_perm) != (
                    pk.mode, pk.gs, pk.bits, pk.perm is not None):
                # mode/gs/bits are baked into any trace built earlier, and
                # the presence of a permutation (or of a bias table) changes
                # the traced arity.
                changed = changed or m.gs != 0
                m.mode, m.gs, m.bits = pk.mode, pk.gs, pk.bits
                m.has_perm = pk.perm is not None
            m.slot = len(values)
            # The packed arrays travel as inputs like the weights do, not
            # baked: a repack against different weights may well produce a
            # different permutation, and mx.compile would keep the first
            # one forever.
            arrays = pk.arrays()
            m.nvals = len(arrays)
            values.extend(arrays)
    # Expert-gather matches are verified here too (eagerly, before any
    # trace): the check syncs with the host, which a traced emit cannot.
    from metaljax import moe as _moe
    changed = _moe.prologue(interp, st) or changed
    with interp.context:
        if changed:
            st.rebuild()
    st.values = values
    if packed_any:
        # The reconstruction ran once at full weight size; its intermediates
        # are dead now and Metal counts live buffers, not bytes.
        mx.clear_cache()
    return changed


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------


def values(interp):
    """Packed arrays for the current context (traced ones inside a trace)."""
    st = getattr(interp, "_qmm", None)
    return st.values if st is not None else ()


def push(interp, vals):
    """Rebind the packed arrays to a traced function's own inputs."""
    st = getattr(interp, "_qmm", None)
    if st is None or not st.values:
        return None
    prev = st.values
    st.values = list(vals)
    return (prev,)


def pop(interp, token):
    if token is not None:
        interp._qmm.values = token[0]


def pack_arrays(interp, m):
    """This match's packed arrays, as `emit` sees them (traced inside a
    trace). Shared with metaljax.moe, which reads the pack of a dot it has
    taken over rather than emitting the dense quantized_matmul."""
    st = interp._qmm
    return st.values[m.slot:m.slot + m.nvals]


def emit_reads(m):
    """The `env` values `emit` (below) reads, for liveness pruning.

    Exactly one: the activation operand. Everything else the fused dot needs
    -- the packed weight, the scales, the bias table and the group
    permutation -- travels in `State.values` (see `pack_arrays`), which the
    packing prologue owns and `env` never holds. The dequantized weight the
    literal dot would have read is produced by ops this rewrite ABSORBS, so
    it is never in `env` at all.

    Keep this in step with `emit`: a missing entry lets the interpreter drop
    a value the fused op still reads. METALJAX_PRUNE_VERIFY=1 checks it at
    run time (metaljax.interpreter._CheckedEnv); tests/test_eager_prune.py
    runs this recognizer under it, with a negative control.
    """
    if not isinstance(m, Match):
        from metaljax import moe as _moe
        return _moe.emit_reads(m)
    return (m.lhs,)


def emit_bytes(m) -> int:
    """Device bytes `emit` materializes BEYOND the root's own result.

    ops.control._block_bytes charges every absorbed op nothing -- right, they
    never run -- and the root its declared result, which for a quantized
    matmul is the product itself. What that misses is the activation copy:
    `emit` may transpose it, cast it to a float type and take a group
    permutation of it before handing it to `quantized_matmul`. One copy of
    the LHS is the bound, and it is charged whether or not this particular
    match needs one (a few MB on decode shapes, and the direction to err in).

    The packed weight and its scales are NOT counted: the packing prologue
    owns them, they exist before the trace begins and outlive it, so they are
    not part of what tracing this block materializes.

    Keep in step with `emit`, like `emit_reads`.
    """
    if not isinstance(m, Match):
        from metaljax import moe as _moe
        return _moe.emit_bytes(m)
    from metaljax.interpreter import value_bytes
    return value_bytes(m.lhs)


def emit(interp, m, env):
    """One `mx.quantized_matmul` in place of the whole dequant-and-dot."""
    if not isinstance(m, Match):
        # An expert-gather match (metaljax.moe.Match): same root dispatch,
        # different rewrite. Kept here so the interpreter has one hook.
        from metaljax import moe as _moe
        return _moe.emit(interp, m, env)
    st = interp._qmm
    vals = st.values[m.slot:m.slot + m.nvals]
    w, scales = vals[0], vals[1]
    biases = vals[2] if m.mode == "affine" else None
    x = env[m.lhs]
    if m.lperm != list(range(len(x.shape))):
        x = mx.transpose(x, m.lperm)
    x = mx.reshape(x, ([m.B, m.M, m.K] if m.bshape else [m.M, m.K]))
    if x.dtype not in _FLOAT_MX:
        x = x.astype(m.out_dtype)
    if m.has_perm:
        # The weight was packed with its contraction axis permuted (its
        # groups were interleaved); permuting BOTH operands identically
        # leaves the dot itself unchanged.
        x = mx.take(x, vals[-1], axis=-1)
    y = mx.quantized_matmul(x, w, scales, biases, transpose=True,
                            group_size=m.gs, bits=m.bits, mode=m.mode)
    if m.swapped:
        # The dot had the weight on its LHS, so its result is [..., N, M].
        y = mx.swapaxes(y, -1, -2)
        y = mx.reshape(y, m.bshape + m.nshape + m.mshape)
    else:
        y = mx.reshape(y, m.bshape + m.mshape + m.nshape)
    if y.dtype != m.out_dtype:
        y = y.astype(m.out_dtype)
    return [y]
