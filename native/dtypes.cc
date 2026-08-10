// metaljax native engine — element types (src/metaljax/dtypes.py).
//
// The table Python gates on, the predicates every handler branches on, the
// weak-literal rule that keeps a ported expression's dtype (and therefore
// its BITS) identical to the Python engine's, and the complex64
// rearrangements: MLX's complex kernels compute the naive formulas, which
// overflow or go NaN exactly where C99 Annex G and XLA are exact.

#include "program.h"

#include <cmath>
#include <limits>
#include <vector>

namespace metaljax {

namespace {

// Keyed by the MLIR element type as printed, so tape.py can gate straight
// off the IR: an element type absent from this table (complex, f64, and the
// whole emulated i4/f8/f6/f4 family, whose device storage is a WIDER dtype
// holding values rather than bits) declines the program.

struct NamedDtype { const char* name; mx::Dtype dtype; };

const NamedDtype kDtypes[] = {
    {"i1", mx::bool_},     {"i8", mx::int8},     {"i16", mx::int16},
    {"i32", mx::int32},    {"i64", mx::int64},   {"ui8", mx::uint8},
    {"ui16", mx::uint16},  {"ui32", mx::uint32}, {"ui64", mx::uint64},
    {"f16", mx::float16},  {"f32", mx::float32}, {"bf16", mx::bfloat16},
    // complex64 IS its own bits on the device (two f32 lanes), so it
    // belongs here; what it does NOT share with the real types is the
    // arithmetic, and every handler whose Python counterpart branches on
    // complex64 branches on it below.
    {"complex<f32>", mx::complex64},
};
constexpr int kNumDtypes = sizeof(kDtypes) / sizeof(kDtypes[0]);

}  // namespace

mx::Dtype dtype_of(int64_t code) {
  if (code < 0 || code >= kNumDtypes)
    throw std::invalid_argument("tape: bad dtype code");
  return kDtypes[code].dtype;
}

bool is_bool(const mx::Dtype& d) { return d == mx::bool_; }

// dtypes.is_float: complex is NOT float here, exactly as in Python — the
// handlers that ask this question (sign's NaN rule, argmax's NaN rule)
// have a separate complex arm or none at all.
bool is_float(const mx::Dtype& d) {
  return d == mx::float32 || d == mx::float16 || d == mx::bfloat16;
}

bool is_complex(const mx::Dtype& d) { return d == mx::complex64; }

// dtypes.is_unsigned / dtypes.is_int: bool is NEITHER, which several
// handlers below depend on (a bool `divide` takes the float arm).
bool is_unsigned(const mx::Dtype& d) {
  return d == mx::uint8 || d == mx::uint16 || d == mx::uint32 ||
         d == mx::uint64;
}

bool is_int(const mx::Dtype& d) {
  return is_unsigned(d) || d == mx::int8 || d == mx::int16 ||
         d == mx::int32 || d == mx::int64;
}

// dtypes.unsigned_of: the same-width unsigned type, or the type itself.
mx::Dtype unsigned_of(const mx::Dtype& d) {
  if (d == mx::int8) return mx::uint8;
  if (d == mx::int16) return mx::uint16;
  if (d == mx::int32) return mx::uint32;
  if (d == mx::int64) return mx::uint64;
  return d;
}

// MLX's Python bindings promote a python scalar to the ARRAY's dtype (a
// "weak" type): `mx.abs(x) + 0.5` on float16 stays float16, while a bare
// C++ `array(0.5)` is float32 and would promote the whole expression to
// f32. Every literal in the ported handlers below therefore carries the
// operand's dtype explicitly. Not cosmetic — it changes the result bits.
mx::array weak(double v, const mx::array& a) {
  return mx::array(v, is_float(a.dtype()) || is_complex(a.dtype())
                          ? a.dtype()
                          : mx::float32);
}

// The same rule for a python INT literal, which adopts the array's dtype
// whatever it is (`int8_array < 0` compares in int8). Only ever called on
// arrays whose dtype can hold the literal, which is every use below.
mx::array weak_int(int64_t v, const mx::array& a) {
  return mx::array(v, a.dtype());
}

// An array with the same bits in a buffer of its own. MLX has no such op:
// `copy`, `contiguous` and `astype` to the same dtype all short-circuit to
// a shared buffer (measured — the pointers come back equal), which is the
// exact opposite of what a de-aliasing copy needs. A select between two
// copies of the same value writes a new buffer and moves bits rather than
// computing on them, so -0 and NaN payloads survive; it is the same trick
// ops/control._anchor_outputs uses on compiled graphs.
mx::array fresh_copy(const mx::array& a) {
  return mx::where(mx::array(true), a, a);
}

// dtypes.total_order_key: the unsigned key whose ascending order is IEEE
// totalOrder (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN). Only f16/
// f32/bf16 reach it — f64 and the emulated types are declined — so the
// top bit is always 1<<15 or 1<<31 and fits an int64 literal.
mx::array total_order_key(const mx::array& x) {
  mx::Dtype ut = x.dtype() == mx::float32 ? mx::uint32 : mx::uint16;
  mx::Dtype it = x.dtype() == mx::float32 ? mx::int32 : mx::int16;
  mx::array u = mx::view(x, ut);
  mx::array neg = mx::less(mx::view(x, it), mx::array(0, it));
  mx::array top =
      mx::array(int64_t{1} << (static_cast<int>(ut.size()) * 8 - 1), ut);
  return mx::where(neg, mx::bitwise_invert(u), mx::bitwise_or(u, top));
}

// --------------------------------------------------------------------------
// complex64 (src/metaljax/dtypes.py, ops/elementwise.py)
// --------------------------------------------------------------------------
//
// MLX's complex kernels compute the naive formulas, which overflow or go
// NaN exactly where C99 Annex G (and XLA) are exact. The Python handlers
// carry a rearrangement per function; these are those rearrangements, and
// like them they exist only on the complex arm — a real program emits the
// same ops it always did.

const double kInf = std::numeric_limits<double>::infinity();

// dtypes.make_complex: build the value by writing the halves into place.
// Doing it arithmetically (re + im*1j) runs a complex multiply that
// destroys the very values this exists to preserve — (1, inf) becomes
// (nan, inf) because inf*0 is nan, and -0 + 0 collapses to +0.
mx::array make_complex(mx::array re, mx::array im) {
  re = mx::astype(re, mx::float32);
  im = mx::astype(im, mx::float32);
  if (re.shape() != im.shape()) {
    std::vector<mx::array> both = mx::broadcast_arrays({re, im});
    re = both[0];
    im = both[1];
  }
  return mx::reshape(mx::view(mx::stack({re, im}, -1), mx::complex64),
                     re.shape());
}

// _cabs: |z| by scaled hypot — the naive sqrt(re^2+im^2) overflows for
// large parts and underflows for tiny ones.
mx::array cabs(const mx::array& z) {
  mx::array a = mx::abs(mx::real(z)), b = mx::abs(mx::imag(z));
  mx::array big = mx::maximum(a, b), small = mx::minimum(a, b);
  mx::array zero = mx::array(0.0f, mx::float32);
  mx::array is0 = mx::equal(big, zero);
  mx::array safe = mx::where(is0, mx::ones_like(big), big);
  mx::array r = mx::where(is0, mx::zeros_like(big), mx::divide(small, safe));
  mx::array out = mx::multiply(
      big, mx::sqrt(mx::add(mx::array(1.0f, mx::float32),
                            mx::multiply(r, r))));
  return mx::where(mx::logical_or(mx::isinf(a), mx::isinf(b)),
                   mx::full(a.shape(), kInf, mx::float32), out);
}

// _expm1_f32: MLX's Metal expm1 kernel is fast-math; exp(x)-1 is ~1 ULP
// except near zero where it cancels, so expm1 is used only there.
mx::array expm1_f32(const mx::array& x) {
  return mx::where(mx::less(mx::abs(x), weak(0.25, x)), mx::expm1(x),
                   mx::subtract(mx::exp(x), weak(1.0, x)));
}

// _csqrt: Kahan's rearrangement of the C99 formula. The textbook
// expression underflows the real part to a spurious 0 near the negative
// real axis and overflows once 2|z| leaves f32; taking the small component
// from `y` directly cancels in neither branch. Non-finite inputs keep
// MLX's own answers, as the Python handler leaves them.
mx::array csqrt(const mx::array& z) {
  mx::array x = mx::real(z), y = mx::imag(z);
  mx::array huge = mx::greater(mx::maximum(mx::abs(x), mx::abs(y)),
                               mx::array(1e18f, mx::float32));
  mx::array scale = mx::where(huge, mx::array(std::pow(2.0, -60), mx::float32),
                              mx::array(1.0f, mx::float32));
  mx::array xs = mx::multiply(x, scale), ys = mx::multiply(y, scale);
  mx::array t = mx::sqrt(mx::multiply(
      mx::array(2.0f, mx::float32),
      mx::add(cabs(make_complex(xs, ys)), mx::abs(xs))));
  mx::array tsafe =
      mx::where(mx::equal(t, mx::array(0.0f, mx::float32)), mx::ones_like(t), t);
  mx::array half = mx::divide(t, mx::array(2.0f, mx::float32));
  // copysign(half, ys): MLX has no signbit, so read the bit (-0.0 counts).
  mx::array neg = mx::less(mx::view(ys, mx::int32), mx::array(0, mx::int32));
  mx::array pos_re = mx::greater_equal(xs, mx::array(0.0f, mx::float32));
  mx::array re = mx::where(pos_re, half, mx::divide(mx::abs(ys), tsafe));
  mx::array im = mx::where(pos_re, mx::divide(ys, tsafe),
                           mx::where(neg, mx::negative(half), half));
  mx::array unscale = mx::where(huge, mx::array(std::pow(2.0, 30), mx::float32),
                                mx::array(1.0f, mx::float32));
  mx::array out =
      make_complex(mx::multiply(re, unscale), mx::multiply(im, unscale));
  return mx::where(mx::logical_and(mx::isfinite(x), mx::isfinite(y)), out,
                   mx::sqrt(z));
}

nb::dict dtype_codes() {
  nb::dict d;
  for (int i = 0; i < kNumDtypes; i++) d[kDtypes[i].name] = i;
  return d;
}

}  // namespace metaljax
