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
    if "THREE_FRY" in algo:
        raise UnsupportedOpError("rng_bit_generator THREE_FRY")

    (state,) = ins
    if state.dtype == mx.uint32 and state.shape == (4,):
        state = mx.view(state, mx.uint64)
    if state.dtype != mx.uint64 or state.shape != (2,):
        raise UnsupportedOpError(
            f"rng_bit_generator state {state.dtype}{state.shape}")

    out_t = ir.RankedTensorType(op.results[1].type)
    out_dtype = dtypes.mx_dtype_for(out_t.element_type)
    out_shape = list(out_t.shape)
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
