"""stablehlo.rng_bit_generator — bit-exact XLA Philox4x32.

Semantics reverse-engineered from xla/hlo/builder/lib/prng.cc and the
rng_bit_generator expander, verified word-for-word against the CPU
backend: state = [key_u64, counter_u64]; the 128-bit philox counter is
(low = counter + block_index, high = key + carry); ten rounds; the four
output words of consecutive blocks are interleaved round-robin; the new
state keeps the key and stores counter + num_blocks. XLA's DEFAULT
algorithm on CPU is Philox.
"""

import mlx.core as mx

from metaljax.interpreter import register, UnsupportedOpError

_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85
_LO32 = 0xFFFFFFFF


def _philox_blocks(key_lo, key_hi, c0, c1, c2, c3):
    """Ten philox rounds over vectors of u32 counter words."""
    k0 = key_lo
    k1 = key_hi
    x0, x1, x2, x3 = c0, c1, c2, c3
    for _ in range(10):
        p0 = x0.astype(mx.uint64) * _M0
        p1 = x2.astype(mx.uint64) * _M1
        hi0 = (p0 >> 32).astype(mx.uint32)
        lo0 = (p0 & _LO32).astype(mx.uint32)
        hi1 = (p1 >> 32).astype(mx.uint32)
        lo1 = (p1 & _LO32).astype(mx.uint32)
        x0, x1, x2, x3 = hi1 ^ x1 ^ k0, lo1, hi0 ^ x3 ^ k1, lo0
        k0 = k0 + mx.array(_W0, dtype=mx.uint32)
        k1 = k1 + mx.array(_W1, dtype=mx.uint32)
    return x0, x1, x2, x3


@register("stablehlo.rng_bit_generator")
def _rng_bit_generator(interp, op, ins, env):
    from jaxlib.mlir import ir
    from metaljax import dtypes

    try:
        algo = str(op.attributes["rng_algorithm"])
    except Exception:
        algo = "DEFAULT"
    (state,) = ins
    if state.dtype == mx.uint32 and state.shape == (4,):
        state = mx.view(state, mx.uint64)
    if state.dtype != mx.uint64 or state.shape != (2,):
        raise UnsupportedOpError(
            f"rng_bit_generator state {state.dtype}{state.shape}")

    out_t = ir.RankedTensorType(op.results[1].type)
    out_dtype = dtypes.mx_dtype_for(out_t.element_type)
    out_shape = list(out_t.shape)
    if "THREE_FRY" in algo:
        if state.dtype == mx.uint32 and state.shape == (4,):
            state = mx.view(state, mx.uint64)
        if out_dtype.size * 8 > 32:
            raise UnsupportedOpError("rng_bit_generator THREE_FRY 64-bit")
        new_state, bits = _threefry_bits(state, out_shape, out_dtype)
        if ins[0].dtype == mx.uint32:
            new_state = mx.view(new_state, mx.uint32)
        return [new_state, bits]
    n = 1
    for s in out_shape:
        n *= s
    width = out_dtype.size * 8

    # number of u32 words to generate (narrow types truncate one u32 each)
    num_u32 = n * 2 if width == 64 else n
    nv4 = max(1, -(-num_u32 // 4))

    key = state[0]
    ctr = state[1]
    iota = mx.arange(nv4, dtype=mx.uint64)
    low = ctr + iota
    carry = (low < ctr).astype(mx.uint64)
    high = key + carry
    c0 = (low & _LO32).astype(mx.uint32)
    c1 = (low >> 32).astype(mx.uint32)
    c2 = (high & _LO32).astype(mx.uint32)
    c3 = (high >> 32).astype(mx.uint32)
    key_lo = (key & _LO32).astype(mx.uint32)
    key_hi = (key >> 32).astype(mx.uint32)

    x0, x1, x2, x3 = _philox_blocks(key_lo, key_hi, c0, c1, c2, c3)

    unsigned = {8: mx.uint8, 16: mx.uint16, 32: mx.uint32,
                64: mx.uint64}[width]
    if width == 64:
        b0 = x0.astype(mx.uint64) | (x1.astype(mx.uint64) << 32)
        b1 = x2.astype(mx.uint64) | (x3.astype(mx.uint64) << 32)
        bits = mx.reshape(mx.stack([b0, b1], axis=1), (2 * nv4,))[:n]
    else:
        bits = mx.reshape(mx.stack([x0, x1, x2, x3], axis=1),
                          (4 * nv4,))[:num_u32]
        if width < 32:
            bits = bits.astype(unsigned)  # narrow: truncate per element
    if out_dtype != unsigned:
        bits = mx.view(bits, out_dtype)  # signed variants: same bits
    new_state = mx.stack([key, ctr + mx.array(nv4, dtype=mx.uint64)])
    if ins[0].dtype == mx.uint32:
        new_state = mx.view(new_state, mx.uint32)
    return [new_state, mx.reshape(bits, out_shape)]


_TF_ROT = (13, 15, 26, 6, 17, 29, 16, 24)


def _threefry2x32(x0, x1, k0, k1):
    """XLA's ThreeFry2x32: 20 rounds, key injection every 4."""
    parity = mx.array(0x1BD11BDA, dtype=mx.uint32)
    ks = [k0, k1, parity ^ k0 ^ k1]
    x0 = x0 + ks[0]
    x1 = x1 + ks[1]
    for g in range(5):
        for r in range(4):
            rot = _TF_ROT[(g % 2) * 4 + r]
            x0 = x0 + x1
            x1 = (x1 << rot) | (x1 >> (32 - rot))
            x1 = x0 ^ x1
        j = g + 1
        x0 = x0 + ks[j % 3]
        x1 = x1 + ks[(j + 1) % 3] + mx.array(j, dtype=mx.uint32)
    return x0, x1


def _threefry_bits(state, out_shape, out_dtype):
    """XLA ThreeFryBitGenerator (32-bit-and-narrower path): split the
    output shape in half, one threefry block per half-element pair."""
    dims = list(out_shape)
    if not dims:
        dims = [1]
        scalar = True
    else:
        scalar = False
    split_dim = next((i for i, d_ in enumerate(dims) if d_ % 2 == 0), None)
    if split_dim is None:
        split_dim = max(range(len(dims)), key=lambda i: dims[i])
    half = dims.copy()
    half[split_dim] = -(-dims[split_dim] // 2)
    n_half = 1
    for d_ in half:
        n_half *= d_
    key = state[0]
    ctr = state[1]
    # per-half-element u64 counter = state + row-major linear index
    lin = mx.arange(n_half, dtype=mx.uint64)
    u64 = ctr + lin
    x0 = (u64 & _LO32).astype(mx.uint32)
    x1 = (u64 >> 32).astype(mx.uint32)
    k0 = (key & _LO32).astype(mx.uint32)
    k1 = (key >> 32).astype(mx.uint32)
    o0, o1 = _threefry2x32(x0, x1, k0, k1)
    # combine: concat halves on a fresh axis after split_dim, reshape to
    # the rounded-up shape, slice back down
    h = tuple(half[:split_dim + 1]) + (1,) + tuple(half[split_dim + 1:])
    both = mx.concatenate([mx.reshape(o0, h), mx.reshape(o1, h)],
                          axis=split_dim + 1)
    rounded = dims.copy()
    rounded[split_dim] = half[split_dim] * 2
    both = mx.reshape(both, rounded)
    if rounded[split_dim] != dims[split_dim]:
        sl = [slice(None)] * len(dims)
        sl[split_dim] = slice(0, dims[split_dim])
        both = both[tuple(sl)]
    if scalar:
        both = mx.reshape(both, ())
    new_state = mx.stack([key, ctr + mx.array(n_half, dtype=mx.uint64)])
    width = out_dtype.size * 8
    if width < 32:
        both = both.astype({8: mx.uint8, 16: mx.uint16}[width])
    return new_state, both
