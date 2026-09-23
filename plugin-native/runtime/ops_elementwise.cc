// metaljax native engine — elementwise ops (ported from Stage 1's
// src/metaljax/ops/elementwise.py, deleted 0.11.6, ef5774d).
//
// Unary and binary maps, comparison, select/clamp, the dtype convert, the
// complex64 accessors and the FFT, and the SWAR bit counts. Each handler is
// a transliteration of the Python one: where that table carried a dtype
// branch, so does the switch below, because the differential test compares
// output BYTES and both engines call the same MLX kernels.

#include "program.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace metaljax {

namespace {

// ops/elementwise._int_trunc_div: C-style (truncated) integer division.
// The Python handler takes the sign off the magnitudes and puts it back,
// on the assumption that floor_divide floors -- which MLX's does NOT for
// integers (it forwards to `divide`, and integer division truncates
// toward zero already, measured: mx.floor_divide(-7, 2) == -3). The two
// spellings therefore agree everywhere except at INT_MIN, where abs()
// wraps to itself and the sign flip is real: metaljax answers
// int8(-128)/2 == 64 where XLA says -64. That is a pre-existing property
// of the Python engine, and this is a transliteration of it -- the
// differential test pins the two engines to each other, INT_MIN included.
mx::array int_trunc_div(const mx::array& a, const mx::array& b) {
  if (is_unsigned(a.dtype())) return mx::floor_divide(a, b);
  mx::array q = mx::floor_divide(mx::abs(a), mx::abs(b));
  mx::array neg = mx::not_equal(mx::less(a, weak_int(0, a)),
                                mx::less(b, weak_int(0, b)));
  return mx::astype(mx::where(neg, mx::negative(q), q), a.dtype());
}

// ops/elementwise._popcount: SWAR, in u64 for 64-bit operands and in u32
// for everything narrower (which is where the Python handler's astype
// goes). Every literal adopts the working dtype, as a python int does.
mx::array popcount_swar(mx::array u, bool wide) {
  mx::Dtype dt = mx::uint32;
  int64_t c1 = 0x55555555, c2 = 0x33333333, c4 = 0x0F0F0F0F, m = 0x01010101;
  int shift = 24;
  if (wide) {
    dt = mx::uint64;
    c1 = 0x5555555555555555LL;
    c2 = 0x3333333333333333LL;
    c4 = 0x0F0F0F0F0F0F0F0FLL;
    m = 0x0101010101010101LL;
    shift = 56;
  } else {
    u = mx::astype(u, mx::uint32);
  }
  auto k = [&](int64_t v) { return mx::array(v, dt); };
  u = mx::subtract(u, mx::bitwise_and(mx::right_shift(u, k(1)), k(c1)));
  u = mx::add(mx::bitwise_and(u, k(c2)),
              mx::bitwise_and(mx::right_shift(u, k(2)), k(c2)));
  u = mx::bitwise_and(mx::add(u, mx::right_shift(u, k(4))), k(c4));
  return mx::right_shift(mx::multiply(u, k(m)), k(shift));
}

// ops/elementwise._as_unsigned, resolved by tape.py: 0 casts (bool), 1
// views (signed), 2 leaves the operand alone (already unsigned).
mx::array as_unsigned(const mx::array& x, int64_t how, mx::Dtype u) {
  if (how == 0) return mx::astype(x, u);
  if (how == 1) return mx::view(x, u);
  return x;
}

// XLA shifts the operand's LOGICAL bits -- `bits` of them: the storage's
// width for a real type, 2 or 4 for the emulated sub-byte integers, whose
// int8/uint8 storage holds the VALUE (sign- or zero-extended).  These are
// the two views of those bits every shift is spelled with.  Same-width
// signedness changes are astype, which is modular (the Python engine's
// logical shift already round-tripped through it).
//
// The low `bits` bits, ZERO-extended, in the storage's unsigned twin: what a
// logical shift moves, and what the shift AMOUNT is compared by.
mx::array zext_bits(const mx::array& a, int64_t bits) {
  mx::array u = mx::astype(a, unsigned_of(a.dtype()));
  if (bits < 8 * static_cast<int64_t>(a.itemsize()))
    u = mx::bitwise_and(u, mx::array((int64_t{1} << bits) - 1, u.dtype()));
  return u;
}

// ...and SIGN-extended from bit bits-1, in the storage's signed twin: what an
// arithmetic shift moves, whatever the type's signedness.  XLA's AShr is an
// LLVM op on signless integers, so an unsigned operand's top bit fills too
// (uint8 0x80 >> 1 is 0xC0, not 0x40).
mx::array sext_bits(const mx::array& a, int64_t bits) {
  const mx::Dtype s = signed_of(a.dtype());
  if (bits >= 8 * static_cast<int64_t>(a.itemsize())) return mx::astype(a, s);
  // (z ^ half) - half on the zero-extended bits.
  mx::array z = mx::astype(zext_bits(a, bits), s);
  mx::array half = mx::array(int64_t{1} << (bits - 1), s);
  return mx::subtract(mx::bitwise_xor(z, half), half);
}

// kind: 0 shift_left, 1 shift_right_logical, 2 shift_right_arithmetic, by an
// amount already known to be below `bits`.  A sub-byte result may leave the
// type's range (a left shift, the logical shift of a negative i4, the
// arithmetic shift of a ui4 with its top bit set): the entry's regrid wraps
// it (metal_lowering.cc `IsIntGridWrapOp`).
mx::array shift_apply(int kind, const mx::array& a, const mx::array& b,
                      int64_t bits) {
  if (kind == 0) return mx::left_shift(a, b);
  if (kind == 1)
    return mx::astype(mx::right_shift(zext_bits(a, bits), zext_bits(b, bits)),
                      a.dtype());
  mx::array s = sext_bits(a, bits);
  return mx::astype(mx::right_shift(s, mx::astype(b, s.dtype())), a.dtype());
}

// What XLA yields for a shift by at least the operand's bit width: zero,
// except that an arithmetic shift keeps filling with the top bit.
mx::array shift_fill(int kind, const mx::array& a, int64_t bits) {
  if (kind == 2) {
    mx::array s = sext_bits(a, bits);
    const int w = static_cast<int>(s.itemsize()) * 8;
    return mx::astype(mx::right_shift(s, mx::array(w - 1, s.dtype())),
                      a.dtype());
  }
  return mx::zeros_like(a);
}

// ops/elementwise._shift_guard. Metal's shifts are mod-width (x86-style),
// XLA's saturate, and XLA compares the amount UNSIGNED against the width
// (elemental_ir_emitter's ICmpULT): a negative amount is a huge one, and
// saturates like it.  `at` is [static?, amount, bits]: whether the lowering
// found the amount to be a compile-time splat -- in which case only one arm
// is emitted, and `amount` is already `bits` when it is out of range -- and
// the operand's logical width.
mx::array shift_guard(int kind, const mx::array& a, const mx::array& b,
                      const std::vector<int64_t>& at) {
  const int64_t bits = at[2];
  if (at[0]) {
    return at[1] >= bits ? shift_fill(kind, a, bits)
                         : shift_apply(kind, a, b, bits);
  }
  mx::array bu = zext_bits(b, bits);
  mx::array over = mx::greater_equal(bu, mx::array(bits, bu.dtype()));
  return mx::where(over, shift_fill(kind, a, bits),
                   shift_apply(kind, a, b, bits));
}

// stablehlo.remainder on floats: C's fmod, which is EXACT (the remainder of
// two floats is always representable in their format, so there is one right
// answer, and np.fmod / jax-CPU give it).
//
// Neither MLX spelling can be trusted with it.  a - trunc(a / b) * b (the
// Python engine's, until 0.11.2's lax_test testOpAgainstNumpy594 caught it)
// rounds the quotient and the product: ~55 % of random elements wrong, 1.2e-6
// in f32, 2.2 in bf16, inf in f16 where a / b overflows.  mx::remainder is
// Metal's `fmod` plus a python-sign fix-up -- exact in MLX's PRECOMPILED
// kernels (the metallib is built -fno-fast-math, where `metal::fmod` is the
// precise one) but NOT once mx::compile fuses it: MLX JIT-builds fused
// kernels with the library defaults, and there `metal::fmod` is fast::fmod,
// x - y * trunc(x / y) again (measured: 72 of 20k f32 pairs and 82 bf16 ones
// wrong in a jit, sign flips and |r| >= |b| among them; 0 eager).
//
// So the remainder is its own small kernel, spelled with
// `metal::precise::fmod` so no compile option can take it back: fmod on the
// MAGNITUDES, then the dividend's sign -- a nonzero r takes a's, and a zero
// takes it too (a * 0 is +-0 with a's sign for every finite a; an infinite
// or NaN a has a NaN r and never reaches that arm).  Computed in float for
// every float dtype: fmod's result is exact in the operands' own format, so
// the store rounds nothing.  Bit-exact against np.fmod -- value and sign of
// zero, +-inf, NaN, tiny and huge divisors -- in f32, f16 and bf16, eager
// and fused (execute_test `_p50`).  One dispatch that does not fuse; float
// remainder is rare, and a wrong answer is not a speed.
const char kFmodSource[] = R"(
    uint i = thread_position_in_grid.x;
    float a = static_cast<float>(x[i]);
    float b = static_cast<float>(y[i]);
    float r = metal::precise::fmod(metal::abs(a), metal::abs(b));
    out[i] = static_cast<T>((r == 0.0f) ? a * 0.0f : ((a < 0.0f) ? -r : r));
)";

struct FmodState {
  std::mutex mu;
  std::optional<mx::fast::CustomKernelFunction> fn;
  // Per dtype: absent = never proven, false = would not build or answered
  // wrong (the fused-op fallback runs), true = the kernel runs.
  std::map<int, bool> ok;
};

FmodState& Fmod() {
  static FmodState* s = new FmodState();
  return *s;
}

int FmodKey(mx::Dtype dt) {
  if (dt == mx::float32) return 0;
  if (dt == mx::float16) return 1;
  if (dt == mx::bfloat16) return 2;
  return -1;
}

// Always on FLAT operands: the kernel indexes one dimension, and a rank-0
// operand would change its signature (MLX passes a 0-dim input by value) and
// with it the library -- one the lowering never proved.  Flat, every call of
// a dtype is the very kernel `prove_float_remainder` built.
mx::array fmod_launch(const mx::fast::CustomKernelFunction& fn,
                      const mx::array& a, const mx::array& b) {
  const int64_t n = static_cast<int64_t>(a.size());
  const int tg = static_cast<int>(std::min<int64_t>(n, 256));
  const mx::Shape flat{static_cast<mx::ShapeElem>(n)};
  mx::array r = fn({mx::reshape(a, flat), mx::reshape(b, flat)}, {flat},
                   {a.dtype()}, {static_cast<int>(n), 1, 1}, {tg, 1, 1},
                   {{"T", a.dtype()}}, std::nullopt, false, {})[0];
  return mx::reshape(r, a.shape());
}

// The fused-op arm, for a dtype the kernel could not be proven on (and a
// zero-size or >2^31-element operand): mx::remainder is exact wherever it
// is not fused.
mx::array fmod_ops(const mx::array& a, const mx::array& b) {
  mx::array r = mx::remainder(mx::abs(a), mx::abs(b));
  return mx::where(mx::equal(r, weak(0.0, r)), mx::multiply(a, weak(0.0, a)),
                   mx::where(mx::less(a, weak(0.0, a)), mx::negative(r), r));
}

mx::array float_remainder(const mx::array& a0, const mx::array& b0) {
  mx::array a = a0, b = b0;
  if (a.shape() != b.shape()) {
    std::vector<mx::array> ab = mx::broadcast_arrays({a, b});
    a = ab[0];
    b = ab[1];
  }
  const int key = FmodKey(a.dtype());
  if (key < 0 || b.dtype() != a.dtype() || a.size() == 0 ||
      a.size() >= (size_t{1} << 31))
    return fmod_ops(a, b);
  FmodState& st = Fmod();
  std::optional<mx::fast::CustomKernelFunction> fn;
  {
    std::lock_guard<std::mutex> lock(st.mu);
    auto it = st.ok.find(key);
    if (it == st.ok.end() || !it->second) return fmod_ops(a, b);
    fn = st.fn;
  }
  return fmod_launch(*fn, a, b);
}

}  // namespace

bool prove_float_remainder(mx::Dtype dt) {
  const int key = FmodKey(dt);
  if (key < 0) return false;
  FmodState& st = Fmod();
  std::lock_guard<std::mutex> lock(st.mu);
  auto it = st.ok.find(key);
  if (it != st.ok.end()) return it->second;
  bool ok = false;
  try {
    if (!st.fn.has_value())
      st.fn = mx::fast::metal_kernel("mjn_fmod", {"x", "y"}, {"out"},
                                     kFmodSource);
    // Built at its first eval, so prove it here, synchronously, on the
    // cases that tell a precise fmod from a fast one and the sign rules
    // from a careless spelling: fmod(5.5, 2) = 1.5, fmod(-4, 2) = -0,
    // fmod(-7, -2.5) = -2, fmod(1, 0) = NaN, and 0.2578125 = 16 * 0.01611328125
    // exactly, where x - y * trunc(x / y) answers y instead of 0.  TWICE:
    // MLX binds an operand of fewer than 8 elements in the `constant`
    // address space and a larger one in `device`, which is a different
    // kernel (and library) -- so both are the ones that later run.
    const std::vector<float> av{5.5f, -4.0f, -7.0f, 1.0f, 0.2578125f};
    const std::vector<float> bv{2.0f, 2.0f, -2.5f, 0.0f, 0.01611328125f};
    ok = true;
    for (int reps : {1, 4}) {
      std::vector<float> ar, br;
      for (int k = 0; k < reps; k++) {
        ar.insert(ar.end(), av.begin(), av.end());
        br.insert(br.end(), bv.begin(), bv.end());
      }
      const int n = static_cast<int>(ar.size());
      mx::array a = mx::astype(mx::array(ar.data(), {n}, mx::float32), dt);
      mx::array b = mx::astype(mx::array(br.data(), {n}, mx::float32), dt);
      mx::array r = mx::astype(fmod_launch(*st.fn, a, b), mx::float32);
      mx::eval(r);
      const float* v = r.data<float>();
      for (int k = 0; k < reps; k++, v += 5)
        ok = ok && v[0] == 1.5f && v[1] == 0.0f && std::signbit(v[1]) &&
             v[2] == -2.0f && std::isnan(v[3]) && v[4] == 0.0f &&
             !std::signbit(v[4]);
    }
    static const char* const kNames[] = {"f32", "f16", "bf16"};
    debug_print(ok ? std::string("float remainder kernel proven for ") +
                         kNames[key]
                   : std::string("float remainder kernel answered wrong for ") +
                         kNames[key] + "; the fused ops run instead");
  } catch (const std::exception& e) {
    if (is_oom(e)) return false;   // the machine, not the kernel: unproven
    debug_print(std::string("float remainder kernel did not build (") +
                e.what() + "); the fused ops run instead");
    ok = false;
  }
  st.ok[key] = ok;
  return ok;
}

bool Program::step_elementwise(const Entry& e,
                               std::vector<std::optional<mx::array>>& env,
                               bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;

  switch (e.op) {
    // --- unary (ops/elementwise.py _UNARY) ---
    //
    // The functions with a complex arm in the Python table have one here
    // too, dispatched on the same question (`x.dtype == mx.complex64`);
    // the rest call the same MLX function whatever the dtype is, which is
    // what the Python table does.
    case kAbs:
      env[e.outs[0]] = is_complex(in(0).dtype()) ? cabs(in(0))
                                                 : mx::abs(in(0));
      break;
    case kCeil: env[e.outs[0]] = mx::ceil(in(0)); break;
    case kCos: env[e.outs[0]] = mx::cos(in(0)); break;
    case kErf: env[e.outs[0]] = mx::erf(in(0)); break;
    case kErfInv: env[e.outs[0]] = mx::erfinv(in(0)); break;
    case kExp: {
      const mx::array& x = in(0);
      if (!is_complex(x.dtype())) {
        env[e.outs[0]] = mx::exp(x);
        break;
      }
      // _exp: e^a * (cos b, sin b), with sin's zero kept exact so that
      // inf * 0 does not become NaN.
      mx::array a = mx::real(x), b = mx::imag(x);
      mx::array ex = mx::exp(a);
      env[e.outs[0]] = make_complex(
          mx::multiply(ex, mx::cos(b)),
          mx::where(mx::equal(b, weak(0.0, b)), b,
                    mx::multiply(ex, mx::sin(b))));
      break;
    }
    case kFloor: env[e.outs[0]] = mx::floor(in(0)); break;
    case kIsFinite: env[e.outs[0]] = mx::isfinite(in(0)); break;
    case kLog: env[e.outs[0]] = mx::log(in(0)); break;
    case kLog1p: env[e.outs[0]] = mx::log1p(in(0)); break;
    case kLogistic: env[e.outs[0]] = mx::sigmoid(in(0)); break;
    case kNegate: env[e.outs[0]] = mx::negative(in(0)); break;
    case kRsqrt: {
      const mx::array& x = in(0);
      if (!is_complex(x.dtype())) {
        env[e.outs[0]] = mx::rsqrt(x);
        break;
      }
      // _rsqrt: conj(sqrt(z))/|z| -- both factors cancellation-free.
      mx::array s = csqrt(x);
      mx::array m = cabs(x);
      mx::array zero = mx::array(0.0f, mx::float32);
      mx::array msafe = mx::where(mx::equal(m, zero), mx::ones_like(m), m);
      mx::array out = make_complex(mx::divide(mx::real(s), msafe),
                                   mx::negative(mx::divide(mx::imag(s),
                                                           msafe)));
      mx::array ok = mx::logical_and(
          mx::logical_and(mx::isfinite(mx::real(x)),
                          mx::isfinite(mx::imag(x))),
          mx::not_equal(m, zero));
      env[e.outs[0]] = mx::where(ok, out, mx::rsqrt(x));
      break;
    }
    case kSin: env[e.outs[0]] = mx::sin(in(0)); break;
    case kSqrt:
      env[e.outs[0]] = is_complex(in(0).dtype()) ? csqrt(in(0))
                                                 : mx::sqrt(in(0));
      break;
    case kTan: {
      const mx::array& x = in(0);
      if (!is_complex(x.dtype())) {
        env[e.outs[0]] = mx::tan(x);
        break;
      }
      // _tan: C99 says tan(x +- i*inf) = +-i whatever the real part.
      mx::array im = mx::imag(x);
      mx::array pole = mx::isinf(im);
      mx::array safe =
          mx::where(pole, make_complex(mx::zeros_like(im), im), x);
      env[e.outs[0]] = mx::where(
          pole, make_complex(mx::zeros_like(im), mx::sign(im)),
          mx::tan(safe));
      break;
    }
    case kTanh: env[e.outs[0]] = mx::tanh(in(0)); break;
    case kSquare: env[e.outs[0]] = mx::square(in(0)); break;

    case kCbrt: {
      // _cbrt: sign(x) * |x|**(1/3)
      const mx::array& x = in(0);
      env[e.outs[0]] =
          mx::multiply(mx::sign(x),
                       mx::power(mx::abs(x), weak(1.0 / 3.0, x)));
      break;
    }
    case kSign: {
      // _sign: mx.sign returns 0 for NaN, stablehlo.sign propagates it;
      // on complex the sign is z/|z|, with zero mapping to itself (which
      // keeps a signed zero's own bits).
      const mx::array& x = in(0);
      if (is_complex(x.dtype())) {
        mx::array re = mx::real(x), im = mx::imag(x);
        mx::array m = cabs(x);
        mx::array zero = mx::array(0.0f, mx::float32);
        env[e.outs[0]] = mx::where(
            mx::logical_and(mx::equal(re, zero), mx::equal(im, zero)), x,
            make_complex(mx::divide(re, m), mx::divide(im, m)));
        break;
      }
      env[e.outs[0]] = is_float(x.dtype())
                           ? mx::where(mx::isnan(x), x, mx::sign(x))
                           : mx::sign(x);
      break;
    }
    case kRoundAfz: {
      // _round_afz: sign(x) * floor(|x| + 0.5)
      const mx::array& x = in(0);
      env[e.outs[0]] = mx::multiply(
          mx::sign(x), mx::floor(mx::add(mx::abs(x), weak(0.5, x))));
      break;
    }
    case kExpm1: {
      // _expm1 / _expm1_f32: MLX's Metal expm1 kernel is fast-math (worst
      // relative error 2.0e-5), and exp(x)-1 is ~1 ULP except near zero
      // where it cancels -- so use expm1 only there. Halves keep their
      // own expm1, which is already accurate.
      const mx::array& x = in(0);
      if (is_complex(x.dtype())) {
        // exp(z)-1 cancels catastrophically; the C99 reconstruction.
        mx::array a = mx::real(x), b = mx::imag(x);
        mx::array hs = mx::sin(mx::divide(b, weak(2.0, b)));
        env[e.outs[0]] = make_complex(
            // `2 * hs * hs` associates left in Python, and the order is
            // visible once hs * hs is subnormal.
            mx::subtract(mx::multiply(expm1_f32(a), mx::cos(b)),
                         mx::multiply(mx::multiply(weak(2.0, hs), hs), hs)),
            mx::where(mx::equal(b, weak(0.0, b)), b,
                      mx::multiply(mx::exp(a), mx::sin(b))));
        break;
      }
      env[e.outs[0]] =
          x.dtype() == mx::float32
              ? expm1_f32(x)
              : mx::expm1(x);
      break;
    }
    case kNot: {
      // _not: bool -> logical, integer -> bitwise (mlx 0.32 has
      // bitwise_invert, so the xor-with-minus-one fallback is dead).
      const mx::array& x = in(0);
      env[e.outs[0]] = is_bool(x.dtype()) ? mx::logical_not(x)
                                          : mx::bitwise_invert(x);
      break;
    }
    case kRoundEven: {
      // _round_even, verbatim: the tie goes to the even neighbour.
      const mx::array& x = in(0);
      mx::array f = mx::floor(x);
      mx::array d = mx::subtract(x, f);
      mx::array f_is_even =
          mx::equal(mx::remainder(f, weak(2.0, f)), weak(0.0, f));
      mx::array up = mx::add(f, weak(1.0, f));
      env[e.outs[0]] = mx::where(
          mx::greater(d, weak(0.5, d)), up,
          mx::where(mx::less(d, weak(0.5, d)), f,
                    mx::where(f_is_even, f, up)));
      break;
    }

    // --- binary (ops/elementwise.py _BINARY) ---
    case kAdd: {
      const mx::array& a = in(0);
      env[e.outs[0]] = is_bool(a.dtype()) ? mx::logical_or(a, in(1))
                                          : mx::add(a, in(1));
      break;
    }
    case kMultiply: {
      const mx::array& a = in(0);
      env[e.outs[0]] = is_bool(a.dtype()) ? mx::logical_and(a, in(1))
                                          : mx::multiply(a, in(1));
      break;
    }
    case kSubtract: env[e.outs[0]] = mx::subtract(in(0), in(1)); break;
    case kMaximum: env[e.outs[0]] = mx::maximum(in(0), in(1)); break;
    case kMinimum: env[e.outs[0]] = mx::minimum(in(0), in(1)); break;
    case kAnd: {
      // _logical_or_bitwise: bool -> logical, integer -> bitwise.
      const mx::array& a = in(0);
      env[e.outs[0]] = is_bool(a.dtype()) ? mx::logical_and(a, in(1))
                                          : mx::bitwise_and(a, in(1));
      break;
    }
    case kOr: {
      const mx::array& a = in(0);
      env[e.outs[0]] = is_bool(a.dtype()) ? mx::logical_or(a, in(1))
                                          : mx::bitwise_or(a, in(1));
      break;
    }
    case kXor: {
      // bool xor lowers as not_equal (mx has no logical_xor in the
      // Python handler's table either).
      const mx::array& a = in(0);
      env[e.outs[0]] = is_bool(a.dtype()) ? mx::not_equal(a, in(1))
                                          : mx::bitwise_xor(a, in(1));
      break;
    }

    case kDivide: {
      // _divide: integers truncate toward zero, floats divide.
      const mx::array& a = in(0);
      env[e.outs[0]] = is_int(a.dtype()) ? int_trunc_div(a, in(1))
                                         : mx::divide(a, in(1));
      break;
    }
    case kRemainder: {
      // _remainder: StableHLO's remainder takes the sign of the
      // DIVIDEND (C's fmod, truncated division); the integer arm rides on
      // int_trunc_div, the float arm is `float_remainder` (its note says
      // why it is a kernel of its own).
      const mx::array& a = in(0);
      const mx::array& b = in(1);
      if (is_int(a.dtype())) {
        env[e.outs[0]] = mx::astype(
            mx::subtract(a, mx::multiply(int_trunc_div(a, b), b)),
            a.dtype());
      } else {
        env[e.outs[0]] = float_remainder(a, b);
      }
      break;
    }
    case kPower: env[e.outs[0]] = mx::power(in(0), in(1)); break;
    case kAtan2: env[e.outs[0]] = mx::arctan2(in(0), in(1)); break;
    case kShiftLeft:
      env[e.outs[0]] = shift_guard(0, in(0), in(1), at);
      break;
    case kShiftRightLogical:
      env[e.outs[0]] = shift_guard(1, in(0), in(1), at);
      break;
    case kShiftRightArithmetic:
      env[e.outs[0]] = shift_guard(2, in(0), in(1), at);
      break;

    // --- selection ---
    case kCompare: {
      mx::array a = in(0);
      mx::array b = in(1);
      if (at[1]) {
        // IEEE totalOrder: compare the order-preserving integer keys
        // instead of the raw floats (_compare's TOTALORDER arm).
        a = total_order_key(a);
        b = total_order_key(b);
      }
      switch (at[0]) {
        case 0: env[e.outs[0]] = mx::equal(a, b); break;
        case 1: env[e.outs[0]] = mx::not_equal(a, b); break;
        case 2: env[e.outs[0]] = mx::less(a, b); break;
        case 3: env[e.outs[0]] = mx::less_equal(a, b); break;
        case 4: env[e.outs[0]] = mx::greater(a, b); break;
        case 5: env[e.outs[0]] = mx::greater_equal(a, b); break;
        default: throw std::invalid_argument("tape: bad compare direction");
      }
      break;
    }
    case kSelect:
      env[e.outs[0]] = mx::where(in(0), in(1), in(2));
      break;
    case kClamp:
      // _clamp: minimum(maximum(x, lo), hi) -- operand order (lo, x, hi).
      env[e.outs[0]] = mx::minimum(mx::maximum(in(1), in(0)), in(2));
      break;

    case kConvert: {
      // _convert. XLA's complex -> real convert keeps the REAL part,
      // which mx::astype would not do on its own; whether that arm runs
      // is a question about two element types, so tape.py answered it.
      const mx::array x = at[1] ? mx::real(in(0)) : in(0);
      // A convert onto an EMULATED grid is the entry's `regrid` and nothing
      // else: `quantize_emulated` reads the operand's own value and ends in
      // the storage dtype itself. Casting to the storage first would round
      // TWICE -- f32 -> f16 -> f8E4M3FN is not f32 -> f8E4M3FN -- and would
      // put a saturating float->int cast in front of the 4-bit wrap.
      env[e.outs[0]] = is_emulated(at[0]) ? x : mx::astype(x, dtype_of(at[0]));
      break;
    }

    case kReducePrecision: {
      // _reduce_precision. Which arm runs is a question about the operand's
      // dtype and the two attributes, all static, so the lowering answered
      // it; what is left is the arithmetic.
      const mx::array& x = in(0);
      if (at[0] == 0) {           // e >= 8 and m >= 23: nothing to lose
        env[e.outs[0]] = x;
        break;
      }
      if (at[0] == 1 || at[0] == 2) {   // exactly bf16's or f16's grid
        env[e.outs[0]] = mx::astype(
            mx::astype(x, at[0] == 1 ? mx::bfloat16 : mx::float16), x.dtype());
        break;
      }
      const mx::Dtype orig = x.dtype();
      mx::array f = orig == mx::float32 ? x : mx::astype(x, mx::float32);
      const int64_t exp = at[1], man = at[2];
      auto u32 = [](int64_t v) { return mx::array(v, mx::uint32); };
      mx::array isnan = mx::isnan(f);
      mx::array u = mx::view(f, mx::uint32);
      if (man < 23) {
        // Round the f32 mantissa to `man` bits, to nearest-even.
        const int64_t shift = 23 - man;
        mx::array half = u32((int64_t{1} << (shift - 1)) - 1);
        mx::array lsb = mx::bitwise_and(mx::right_shift(u, u32(shift)), u32(1));
        u = mx::bitwise_and(mx::add(mx::add(u, half), lsb),
                            u32(~((int64_t{1} << shift) - 1) & 0xFFFFFFFF));
      }
      mx::array r = mx::view(u, mx::float32);
      if (exp < 8) {
        // ...then clamp to an `exp`-bit exponent range: overflow to an
        // infinity, underflow to a zero. XLA's reduce_precision has no
        // subnormals, so there is nothing between the two. (exp == 1 is
        // degenerate but well defined -- bias 0 makes every finite value
        // either overflow or underflow, which this already produces.)
        mx::array biased = mx::astype(
            mx::bitwise_and(mx::right_shift(u, u32(23)), u32(0xFF)),
            mx::int32);
        const int64_t max_e = (int64_t{1} << (exp - 1)) - 1;
        const int64_t min_e = 2 - (int64_t{1} << (exp - 1));
        mx::array sign =
            mx::where(mx::less(r, mx::array(0.0f, mx::float32)),
                      mx::array(-1.0f, mx::float32),
                      mx::array(1.0f, mx::float32));
        mx::array over = mx::greater(biased, mx::array(127 + max_e, mx::int32));
        mx::array under = mx::less(biased, mx::array(127 + min_e, mx::int32));
        r = mx::where(over,
                      mx::multiply(sign,
                                   mx::array(std::numeric_limits<float>::
                                                 infinity(),
                                             mx::float32)),
                      r);
        r = mx::where(under,
                      mx::multiply(sign, mx::array(0.0f, mx::float32)), r);
      }
      r = mx::where(isnan, f, r);
      env[e.outs[0]] = mx::astype(r, orig);
      break;
    }

    // --- complex64 (ops/elementwise.py) ---
    case kReal: env[e.outs[0]] = mx::real(in(0)); break;
    case kImag:
      // _imag on a real operand is zeros, not an error.
      env[e.outs[0]] = is_complex(in(0).dtype()) ? mx::imag(in(0))
                                                 : mx::zeros_like(in(0));
      break;
    case kMakeComplex:
      env[e.outs[0]] = make_complex(in(0), in(1));
      break;
    case kFft: {
      // _fft. Which transform runs, over which axes and lengths, and
      // whether the empty or unit-length rewrite applies are all static;
      // the two MLX workarounds behind those rewrites are documented in
      // the Python handler.
      Cursor c(at);
      int64_t form = c.next();
      if (form == 0) {
        // MLX rejects zero-size transforms; XLA returns the typed empty
        // result, and a transform of nothing is a sum over nothing.
        mx::Dtype dt = dtype_of(c.next());
        env[e.outs[0]] = mx::zeros(c.shp(), dt);
        break;
      }
      const mx::array& x = in(0);
      // MLX's FFT kernels can read an input buffer whose producing copy
      // is still in flight. Inside a trace the whole program is one
      // graph MLX orders itself, so only the eager path needs this.
      if (!in_trace) {
        std::vector<mx::array> one{x};
        mx::eval(one);
      }
      if (form == 1) {
        int64_t kind = c.next();
        std::vector<int> n = c.vec();
        std::vector<int> axes = c.vec();
        mx::Shape s(n.begin(), n.end());
        switch (kind) {
          case 0: env[e.outs[0]] = mx::fft::fftn(x, s, axes); break;
          case 1: env[e.outs[0]] = mx::fft::ifftn(x, s, axes); break;
          case 2: env[e.outs[0]] = mx::fft::rfftn(x, s, axes); break;
          default: env[e.outs[0]] = mx::fft::irfftn(x, s, axes); break;
        }
        break;
      }
      // The unit-length rewrite: a length-1 real transform is the
      // identity on the single DC bin, so the real axis just drops its
      // imaginary part (irfft) or gains one (rfft), and the leading axes
      // take an ordinary complex transform.
      bool has_lead = c.flag();
      std::vector<int> n = c.vec();
      std::vector<int> axes = c.vec();
      mx::Shape s(n.begin(), n.end());
      if (form == 2) {  // IRFFT
        mx::array y = has_lead ? mx::fft::ifftn(x, s, axes) : x;
        env[e.outs[0]] = mx::real(y);
      } else {          // RFFT
        mx::array y = mx::astype(x, mx::complex64);
        env[e.outs[0]] = has_lead ? mx::fft::fftn(y, s, axes) : y;
      }
      break;
    }

    case kPopcnt:
    case kClz: {
      // ops/elementwise._popcnt / _clz. `_as_unsigned` then SWAR; clz
      // first smears the highest set bit down (log2(width) rounds) so
      // the population count of the smear is width - leading zeros.
      // `bits` is the LOGICAL width: an i2/i4 storage is sign-extended, so
      // its unsigned view is cut to the type's own bits first.
      const mx::array& x = in(0);
      mx::Dtype u_dt = dtype_of(at[0]);
      bool wide = at[2] != 0;
      int64_t bits = at[3];
      mx::array u = as_unsigned(x, at[1], u_dt);
      if (bits < 8 * static_cast<int64_t>(u.itemsize()))
        u = mx::bitwise_and(u, mx::array((int64_t{1} << bits) - 1,
                                         u.dtype()));
      if (e.op == kClz) {
        for (int64_t s = 1; s < bits; s *= 2)
          u = mx::bitwise_or(u, mx::right_shift(u, mx::array(s, u.dtype())));
        mx::array pc = popcount_swar(u, wide);
        env[e.outs[0]] = mx::astype(
            mx::subtract(mx::array(bits, pc.dtype()), pc), dtype_of(at[4]));
      } else {
        env[e.outs[0]] = mx::astype(popcount_swar(u, wide), dtype_of(at[4]));
      }
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
