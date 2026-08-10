// metaljax native engine — elementwise ops (src/metaljax/ops/elementwise.py).
//
// Unary and binary maps, comparison, select/clamp, the dtype convert, the
// complex64 accessors and the FFT, and the SWAR bit counts. Each handler is
// a transliteration of the Python one: where that table carries a dtype
// branch, so does the switch below, because the differential test compares
// output BYTES and both engines call the same MLX kernels.

#include "program.h"

#include <optional>
#include <stdexcept>
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

// ops/elementwise._shift_right_logical: a signed operand shifts as its
// unsigned twin so the sign bit does not fill.
mx::array shift_right_logical(const mx::array& a, const mx::array& b) {
  if (is_unsigned(a.dtype()) || is_bool(a.dtype()))
    return mx::right_shift(a, b);
  mx::Dtype u = unsigned_of(a.dtype());
  return mx::astype(mx::right_shift(mx::astype(a, u), mx::astype(b, u)),
                    a.dtype());
}

// kind: 0 shift_left, 1 shift_right_logical, 2 shift_right_arithmetic.
mx::array shift_apply(int kind, const mx::array& a, const mx::array& b) {
  if (kind == 0) return mx::left_shift(a, b);
  if (kind == 1) return shift_right_logical(a, b);
  return mx::right_shift(a, b);
}

// What XLA yields for a shift by at least the operand's bit width: zero,
// except that an arithmetic shift keeps filling with the sign bit.
mx::array shift_fill(int kind, const mx::array& a, const mx::array& b) {
  if (kind == 2) {
    int w = static_cast<int>(a.itemsize()) * 8;
    return mx::right_shift(a, mx::array(w - 1, b.dtype()));
  }
  return mx::zeros_like(a);
}

// ops/elementwise._shift_guard. Metal's shifts are mod-width (x86-style),
// XLA's saturate; `at` says whether tape.py found the amount to be a
// compile-time splat, in which case only one arm is emitted.
mx::array shift_guard(int kind, const mx::array& a, const mx::array& b,
                      const std::vector<int64_t>& at) {
  int w = static_cast<int>(a.itemsize()) * 8;
  if (at[0]) {
    return at[1] >= w ? shift_fill(kind, a, b) : shift_apply(kind, a, b);
  }
  mx::array over = mx::greater_equal(mx::astype(b, mx::int32),
                                     mx::array(w, mx::int32));
  return mx::where(over, shift_fill(kind, a, b), shift_apply(kind, a, b));
}

}  // namespace

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
      // DIVIDEND (truncated division). mx.remainder takes the sign of
      // the divisor, like python's %, so the float arm spells the
      // truncation out; the integer arm rides on int_trunc_div.
      const mx::array& a = in(0);
      const mx::array& b = in(1);
      if (is_int(a.dtype())) {
        env[e.outs[0]] = mx::astype(
            mx::subtract(a, mx::multiply(int_trunc_div(a, b), b)),
            a.dtype());
      } else {
        mx::array q = mx::divide(a, b);
        // _trunc
        mx::array t = mx::where(mx::less(q, weak(0.0, q)), mx::ceil(q),
                                mx::floor(q));
        env[e.outs[0]] = mx::subtract(a, mx::multiply(t, b));
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

    case kConvert:
      // _convert. XLA's complex -> real convert keeps the REAL part,
      // which mx::astype would not do on its own; whether that arm runs
      // is a question about two element types, so tape.py answered it.
      env[e.outs[0]] = mx::astype(at[1] ? mx::real(in(0)) : in(0),
                                  dtype_of(at[0]));
      break;

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
      const mx::array& x = in(0);
      mx::Dtype u_dt = dtype_of(at[0]);
      bool wide = at[2] != 0;
      int64_t bits = at[3];
      mx::array u = as_unsigned(x, at[1], u_dt);
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
