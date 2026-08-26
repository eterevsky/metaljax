// metaljax native engine — element types (ported from Stage 1's
// src/metaljax/dtypes.py, deleted 0.11.6, ef5774d).
//
// The table Python gated on, the predicates every handler branches on, the
// weak-literal rule that keeps a ported expression's dtype (and therefore
// its BITS) identical to the Python engine's, and the complex64
// rearrangements: MLX's complex kernels compute the naive formulas, which
// overflow or go NaN exactly where C99 Annex G and XLA are exact.

#include "program.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace metaljax {

namespace {

// Keyed by the MLIR element type as printed, so a tape builder can gate
// straight off the IR: an element type absent from this table (f64 above
// all) declines the program.
//
// The EMULATED entries at the end are the one place where the code's dtype
// is not the type's own bits: dtypes.py stored an i4 in an int8, an f8 in a
// float16 and E8M0's exponent range in a float32, holding the VALUE rather
// than the encoding. `dtype_of` therefore answers with the storage, which is
// what every handler wants, and the one question that needs the logical
// type ask `is_emulated` / `quantize_emulated` instead.  (The LOGICAL bit
// width a bitcast reads, and the wire encoding a host transfer needs, are the
// tape BUILDER's business and live there -- a replay needs neither.)

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
    // The emulated grids, in dtypes.py `EMULATED` order. Appended, never
    // interleaved: the codes are handed to a tape builder by name at run
    // time (`dtype_codes`), so their VALUES bind nothing, but keeping the
    // real types' codes stable keeps a dumped tape comparable across builds.
    {"f8E4M3FN", mx::float16},      {"f8E5M2", mx::float16},
    {"f8E4M3", mx::float16},        {"f8E3M4", mx::float16},
    {"f8E8M0FNU", mx::float32},     {"f8E4M3B11FNUZ", mx::float16},
    {"f8E5M2FNUZ", mx::float16},    {"f8E4M3FNUZ", mx::float16},
    {"f6E2M3FN", mx::float16},      {"f6E3M2FN", mx::float16},
    {"f4E2M1FN", mx::float16},      {"i4", mx::int8},
    {"ui4", mx::uint8},
};
constexpr int kNumDtypes = sizeof(kDtypes) / sizeof(kDtypes[0]);

// The first emulated code. Everything below it is a real MLX dtype.
constexpr int kFirstEmulated = 13;

// What rounding a value onto one of those grids needs (dtypes.py
// `quantize_emulated`, `NO_NAN_EMULATED`, `_HAS_INF`, and ml_dtypes' finfo,
// whose numbers are all exact powers of two or exact halves and are written
// out here rather than recomputed).
enum Kind { kFloatGrid = 0, kInt4, kUint4, kE8M0 };
// What a magnitude past the grid's largest finite value becomes.
enum Over { kOverNaN = 0, kOverInf, kOverSaturate };

struct Emulated {
  Kind kind;
  int nmant;      // mantissa bits of the grid
  double tiny;    // smallest normal
  double maxval;  // largest finite magnitude
  Over over;
  double minexp, maxexp;   // E8M0 only
};

const Emulated kEmulated[] = {
    // f8E4M3FN: only the all-ones pattern is NaN, so overflow has one to go
    // to; the FNUZ formats likewise.
    {kFloatGrid, 3, 0x1p-6, 448.0, kOverNaN, 0, 0},        // f8E4M3FN
    {kFloatGrid, 2, 0x1p-14, 57344.0, kOverInf, 0, 0},     // f8E5M2
    {kFloatGrid, 3, 0x1p-6, 240.0, kOverInf, 0, 0},        // f8E4M3
    {kFloatGrid, 4, 0x1p-2, 15.5, kOverInf, 0, 0},         // f8E3M4
    {kE8M0, 0, 0, 0, kOverNaN, -127, 128},                 // f8E8M0FNU
    {kFloatGrid, 3, 0x1p-10, 30.0, kOverNaN, 0, 0},        // f8E4M3B11FNUZ
    {kFloatGrid, 2, 0x1p-15, 57344.0, kOverNaN, 0, 0},     // f8E5M2FNUZ
    {kFloatGrid, 3, 0x1p-7, 240.0, kOverNaN, 0, 0},        // f8E4M3FNUZ
    // The OCP microscaling formats have neither inf nor NaN: every bit
    // pattern is a finite value, so XLA's convert saturates.
    {kFloatGrid, 3, 0x1p0, 7.5, kOverSaturate, 0, 0},      // f6E2M3FN
    {kFloatGrid, 2, 0x1p-2, 28.0, kOverSaturate, 0, 0},    // f6E3M2FN
    {kFloatGrid, 1, 0x1p0, 6.0, kOverSaturate, 0, 0},      // f4E2M1FN
    {kInt4, 0, 0, 0, kOverNaN, 0, 0},                      // i4
    {kUint4, 0, 0, 0, kOverNaN, 0, 0},                     // ui4
};
static_assert(sizeof(kEmulated) / sizeof(kEmulated[0]) ==
                  kNumDtypes - kFirstEmulated,
              "one grid per emulated dtype");

const Emulated& grid(int64_t code) {
  if (code < kFirstEmulated || code >= kNumDtypes)
    throw std::invalid_argument("tape: not an emulated dtype code");
  return kEmulated[code - kFirstEmulated];
}

}  // namespace

mx::Dtype dtype_of(int64_t code) {
  if (code < 0 || code >= kNumDtypes)
    throw std::invalid_argument("tape: bad dtype code");
  return kDtypes[code].dtype;
}

bool is_emulated(int64_t code) {
  return code >= kFirstEmulated && code < kNumDtypes;
}

// dtypes.quantize_emulated: round values onto an emulated dtype's grid, in
// its wide storage. A transliteration, MLX call for MLX call, because this
// is the arithmetic the values themselves are made of -- the Python engine's
// answers are what the jax suite's expectations were measured against.
mx::array quantize_emulated(const mx::array& x, int64_t code) {
  const Emulated& g = grid(code);
  const mx::Dtype storage = dtype_of(code);
  if (g.kind == kInt4) {
    // 4-bit wrap, sign-extended: ((v + 8) mod 16) - 8. `mx::remainder` is
    // Python's `%` (the sign follows the divisor), which is what makes this
    // right for negatives.
    mx::array v = mx::astype(x, mx::int32);
    v = mx::subtract(
        mx::remainder(mx::add(v, mx::array(8, mx::int32)),
                      mx::array(16, mx::int32)),
        mx::array(8, mx::int32));
    return mx::astype(v, storage);
  }
  if (g.kind == kUint4) {
    return mx::astype(mx::remainder(mx::astype(x, mx::int32),
                                    mx::array(16, mx::int32)),
                      storage);
  }
  if (g.kind == kE8M0) {
    // Exponent-only log-scale format: the nearest power of two. The floor at
    // f32's smallest subnormal keeps log2 off -inf for a zero input.
    mx::array f = mx::maximum(mx::astype(x, mx::float32),
                              mx::array(1e-45f, mx::float32));
    mx::array e = mx::round(mx::log2(f));
    e = mx::clip(e, mx::array(static_cast<float>(g.minexp), mx::float32),
                 mx::array(static_cast<float>(g.maxexp), mx::float32));
    return mx::power(mx::array(2.0f, mx::float32), e);
  }

  const int man = g.nmant;
  mx::array f = mx::astype(x, mx::float32);
  mx::array isnan = mx::isnan(f);
  // RNE mantissa rounding in f32 bit-space.
  auto u32 = [](int64_t v) { return mx::array(v, mx::uint32); };
  mx::array u = mx::view(f, mx::uint32);
  const int shift = 23 - man;
  mx::array half = u32((int64_t{1} << (shift - 1)) - 1);
  mx::array lsb = mx::bitwise_and(mx::right_shift(u, u32(shift)), u32(1));
  u = mx::bitwise_and(mx::add(mx::add(u, half), lsb),
                      u32(~((int64_t{1} << shift) - 1) & 0xFFFFFFFF));
  mx::array rounded = mx::view(u, mx::float32);
  // Subnormal range: a uniform grid of spacing tiny * 2^-man. Round onto it
  // by borrowing f32's own rounding -- adding 1.5 * 2^23 * step puts the
  // value where f32's ulp IS step, so the addition rounds it onto the grid
  // and the subtraction is exact. Done on the MAGNITUDE so that -0 (and
  // anything rounding to zero from below) keeps its sign.
  const double step = g.tiny / static_cast<double>(int64_t{1} << man);
  mx::array shifter = mx::array(
      static_cast<float>(step * static_cast<double>(int64_t{1} << 23) * 1.5),
      mx::float32);
  mx::array neg = mx::not_equal(mx::right_shift(mx::view(f, mx::uint32),
                                                u32(31)),
                                u32(0));
  mx::array sub_mag = mx::subtract(mx::add(mx::abs(f), shifter), shifter);
  mx::array sub = mx::where(neg, mx::negative(sub_mag), sub_mag);
  mx::array q = mx::where(
      mx::less(mx::abs(f), mx::array(static_cast<float>(g.tiny), mx::float32)),
      sub, rounded);
  // Overflow, by what the format can encode: an infinity where it has one,
  // NaN where it keeps a NaN but no infinity (the FN/FNUZ float8s, matching
  // XLA's convert -- ml_dtypes saturates, the CPU backend does not), and the
  // largest finite value for the OCP FP4/FP6 formats, which have neither.
  mx::array mx_max = mx::array(static_cast<float>(g.maxval), mx::float32);
  mx::array over = mx::greater(mx::abs(q), mx_max);
  mx::array big =
      g.over == kOverInf
          ? mx::array(std::numeric_limits<float>::infinity(), mx::float32)
          : (g.over == kOverSaturate
                 ? mx_max
                 : mx::array(std::numeric_limits<float>::quiet_NaN(),
                             mx::float32));
  q = mx::where(over, mx::where(neg, mx::negative(big), big), q);
  // NaN passes through the storage dtype. For the no-NaN formats that is
  // still what XLA produces once the value reaches the host: their cast maps
  // NaN to a zero, which is what the host transfer does. (Do NOT substitute
  // a literal -0 here -- MLX bakes scalar constants into its fused kernels
  // and the sign of a zero does not survive that.)
  q = mx::where(isnan, f, q);
  return mx::astype(q, storage);
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

// Rationale at the declaration (program.h): 16-bit float scatter-add runs
// through MLX's emulated atomics, ~33x slower than f32 under contention
// (an embedding backward); accumulate wide, round once.
mx::array scatter_add_wide(const mx::array& a,
                           const std::vector<mx::array>& idx,
                           const mx::array& u, const std::vector<int>& axes) {
  static const bool wide = [] {
    const char* v = std::getenv("METALJAX_SCATTER_ADD_F32");
    return v == nullptr || std::strcmp(v, "0") != 0;
  }();
  const mx::Dtype dt = a.dtype();
  if (!wide || (dt != mx::bfloat16 && dt != mx::float16))
    return mx::scatter_add(a, idx, u, axes);
  return mx::astype(mx::scatter_add(mx::astype(a, mx::float32), idx,
                                    mx::astype(u, mx::float32), axes),
                    dt);
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

std::vector<std::pair<std::string, int>> dtype_codes() {
  std::vector<std::pair<std::string, int>> v;
  v.reserve(static_cast<size_t>(kNumDtypes));
  for (int i = 0; i < kNumDtypes; i++) v.emplace_back(kDtypes[i].name, i);
  return v;
}

}  // namespace metaljax
