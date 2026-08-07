// metaljax native engine — M2: the prepared-program tape.
//
// src/metaljax/tape.py lowers an analyzed executable's main block into one
// of these Programs: a flat op tape with SSA slots resolved to indices,
// attributes decoded to integers, and constants already sitting on the
// device. `run` then walks it with no MLIR, no Python and no GIL — the
// point of the milestone, since decode is Python-dispatch-bound (~120 ms
// per token at gemma-31B scale, notes/cpp-migration-plan.md).
//
// The op set is small and grows monotonically. Anything tape.py cannot
// lower declines the WHOLE program, which then runs on the Python engine
// exactly as before, so a missing op is a performance question and never a
// correctness one. Every handler here is a transliteration of the Python
// handler in src/metaljax/ops/: where those carry a dtype branch or a
// zero-size guard, so does this file, because the differential test
// compares output BYTES and both engines call the same MLX kernels.
//
// C++ owns the opcode enum. Python asks for it via `opcodes()` and an op
// name that is not in the dict declines — so adding an op here is what
// makes it reachable, and there is no second table to forget.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mlx/mlx.h>

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

// --------------------------------------------------------------------------
// opcodes
// --------------------------------------------------------------------------

enum Op : int {
  // unary
  kAbs = 0, kCbrt, kCeil, kCos, kErf, kErfInv, kExp, kExpm1, kFloor,
  kIsFinite, kLog, kLog1p, kLogistic, kNegate, kNot, kRoundAfz, kRoundEven,
  kRsqrt, kSign, kSin, kSqrt, kTan, kTanh, kSquare,
  // binary
  kAdd, kMultiply, kSubtract, kMaximum, kMinimum, kAnd, kOr, kXor,
  kDivide, kRemainder, kPower, kAtan2,
  kShiftLeft, kShiftRightLogical, kShiftRightArithmetic,
  // selection
  kCompare, kSelect, kClamp,
  // dtype
  kConvert,
  // shape
  kReshape, kTranspose, kBroadcastInDim, kSlice, kConcatenate, kIota,
  // data / structured
  kConstant, kReduce, kArgReduce, kDotGeneral,
};

// StableHLO name -> opcode. Several names may share an opcode (chlo.erf is
// stablehlo.erf's handler verbatim); the emulated-dtype regrid the Python
// unary/binary wrappers apply is deliberately absent, because every element
// type that could trigger it is declined in tape.py.
struct NamedOp { const char* name; int op; };

const NamedOp kOpNames[] = {
    {"stablehlo.abs", kAbs},
    {"stablehlo.cbrt", kCbrt},
    {"stablehlo.ceil", kCeil},
    {"stablehlo.cosine", kCos},
    {"stablehlo.erf", kErf},
    {"chlo.erf", kErf},
    {"stablehlo.erf_inv", kErfInv},
    {"chlo.erf_inv", kErfInv},
    {"stablehlo.exponential", kExp},
    {"stablehlo.exponential_minus_one", kExpm1},
    {"stablehlo.floor", kFloor},
    {"stablehlo.is_finite", kIsFinite},
    {"stablehlo.log", kLog},
    {"stablehlo.log_plus_one", kLog1p},
    {"stablehlo.logistic", kLogistic},
    {"stablehlo.negate", kNegate},
    {"stablehlo.not", kNot},
    {"stablehlo.round_nearest_afz", kRoundAfz},
    {"stablehlo.round_nearest_even", kRoundEven},
    {"stablehlo.rsqrt", kRsqrt},
    {"stablehlo.sign", kSign},
    {"stablehlo.sine", kSin},
    {"stablehlo.sqrt", kSqrt},
    {"stablehlo.tan", kTan},
    {"stablehlo.tanh", kTanh},
    {"chlo.square", kSquare},
    {"stablehlo.add", kAdd},
    {"stablehlo.multiply", kMultiply},
    {"stablehlo.subtract", kSubtract},
    {"stablehlo.maximum", kMaximum},
    {"stablehlo.minimum", kMinimum},
    {"stablehlo.and", kAnd},
    {"stablehlo.or", kOr},
    {"stablehlo.xor", kXor},
    {"stablehlo.divide", kDivide},
    {"stablehlo.remainder", kRemainder},
    {"stablehlo.power", kPower},
    {"stablehlo.atan2", kAtan2},
    {"stablehlo.shift_left", kShiftLeft},
    {"stablehlo.shift_right_logical", kShiftRightLogical},
    {"stablehlo.shift_right_arithmetic", kShiftRightArithmetic},
    {"stablehlo.compare", kCompare},
    {"stablehlo.select", kSelect},
    {"stablehlo.clamp", kClamp},
    {"stablehlo.convert", kConvert},
    {"stablehlo.reshape", kReshape},
    {"stablehlo.transpose", kTranspose},
    {"stablehlo.broadcast_in_dim", kBroadcastInDim},
    {"stablehlo.slice", kSlice},
    {"stablehlo.concatenate", kConcatenate},
    {"stablehlo.iota", kIota},
    {"stablehlo.constant", kConstant},
    {"stablehlo.reduce", kReduce},
    // ops/reduction.py reads ONE stablehlo.reduce two ways depending on the
    // body — the single-operand monoid and the (values, indices) pair jax
    // lowers argmax/argmin to. tape.py decides which, then asks for the
    // opcode by this pseudo-name; C++ still owns both enum values.
    {"stablehlo.reduce.arg_pair", kArgReduce},
    {"stablehlo.dot_general", kDotGeneral},
};

// --------------------------------------------------------------------------
// dtypes
// --------------------------------------------------------------------------
//
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
};
constexpr int kNumDtypes = sizeof(kDtypes) / sizeof(kDtypes[0]);

mx::Dtype dtype_of(int64_t code) {
  if (code < 0 || code >= kNumDtypes)
    throw std::invalid_argument("tape: bad dtype code");
  return kDtypes[code].dtype;
}

bool is_bool(const mx::Dtype& d) { return d == mx::bool_; }

bool is_float(const mx::Dtype& d) {
  return d == mx::float32 || d == mx::float16 || d == mx::bfloat16;
}

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
  return mx::array(v, is_float(a.dtype()) ? a.dtype() : mx::float32);
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

// --------------------------------------------------------------------------
// the tape
// --------------------------------------------------------------------------
//
// Attribute layouts, by opcode (all int64; shapes/perms carry their own
// rank so a reader never needs the IR again):
//
//   unary / binary / select / clamp   (none)
//   kCompare            [direction]            0=EQ 1=NE 2=LT 3=LE 4=GT 5=GE
//   kConvert            [dtype]
//   kReshape            [rank, shape...]
//   kTranspose          [rank, perm...]
//   kBroadcastInDim     [transpose?, in_rank, perm...,
//                        out_rank, interim..., out_shape...]
//   kSlice              [rank, start..., stop..., strides...]
//   kConcatenate        [axis]
//   kIota               [dim, ramp_dtype, dtype, rank, shape...]
//   kConstant           (none — the value rides in `payload`)
//   kReduce             [kind, ndims, dims...]  kind: 0 sum 1 prod 2 max
//                                               3 min 4 any 5 all
//   kArgReduce          [is_max, dim]           two results: (value, index)
//   kShift*             [static?, amount]       see shift_guard
//   kDotGeneral         [lrank, lperm..., rrank, rperm..., B, M, K, N,
//                        out_dtype, out_rank, out_shape..., kind, chunk]
//                       kind: 0 float matmul, 1 exact-f32 K-chunks,
//                             2 int64 outer product, 3 the same in bool

struct Entry {
  int op;
  std::vector<int> ins;
  std::vector<int> outs;
  std::vector<int64_t> attrs;
  std::optional<mx::array> payload;  // kConstant only
  std::vector<int> drops;            // slots whose last use is this op
};

// Reduce monoids, resolved at lowering time from the body op and the input
// element type (ops/reduction.py picks _BOOL_REDUCERS vs _REDUCERS on the
// dtype, which is static in the IR).
mx::array reduce_apply(int64_t kind, const mx::array& x,
                       const std::vector<int>& axes) {
  switch (kind) {
    case 0: return mx::sum(x, axes);
    case 1: return mx::prod(x, axes);
    case 2: return mx::max(x, axes);
    case 3: return mx::min(x, axes);
    case 4: return mx::any(x, axes);
    case 5: return mx::all(x, axes);
    default: throw std::invalid_argument("tape: bad reduce kind");
  }
}

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

mx::array reduce_combine(int64_t kind, const mx::array& a,
                         const mx::array& b) {
  switch (kind) {
    case 0: return mx::add(a, b);
    case 1: return mx::multiply(a, b);
    case 2: return mx::maximum(a, b);
    case 3: return mx::minimum(a, b);
    case 4: return mx::logical_or(a, b);
    case 5: return mx::logical_and(a, b);
    default: throw std::invalid_argument("tape: bad reduce kind");
  }
}

class Program {
 public:
  explicit Program(int num_slots, int num_args)
      : nslots_(num_slots), nargs_(num_args) {
    if (num_slots < 0 || num_args < 0 || num_args > num_slots)
      throw std::invalid_argument("tape: bad slot counts");
  }

  void add(int op, std::vector<int> ins, std::vector<int> outs,
           std::vector<int64_t> attrs, std::optional<mx::array> payload,
           std::vector<int> drops) {
    for (int s : ins) check_slot(s);
    for (int s : outs) check_slot(s);
    for (int s : drops) check_slot(s);
    ops_.push_back(Entry{op, std::move(ins), std::move(outs),
                         std::move(attrs), std::move(payload),
                         std::move(drops)});
  }

  void set_outputs(std::vector<int> outs) {
    for (int s : outs) check_slot(s);
    outputs_ = std::move(outs);
  }

  std::vector<mx::array> run(std::vector<mx::array> inputs) {
    if (static_cast<int>(inputs.size()) != nargs_)
      throw std::invalid_argument("tape: wrong number of inputs");
    std::vector<std::optional<mx::array>> env(nslots_);
    for (size_t i = 0; i < inputs.size(); i++) env[i] = std::move(inputs[i]);
    std::vector<mx::array> outs;
    {
      // Nothing below touches Python: MLX builds a lazy graph and the
      // arrays are already C++ objects. The GIL comes back for the cast
      // of the results, in nanobind's own code.
      nb::gil_scoped_release nogil;
      for (const Entry& e : ops_) step(e, env);
      outs.reserve(outputs_.size());
      for (int s : outputs_) {
        if (!env[s]) throw std::runtime_error("tape: output slot is empty");
        outs.push_back(*env[s]);
      }
      // XLA's no-alias contract, the half object identity cannot express
      // across the language boundary: two outputs reading the SAME slot
      // are one array, and nanobind hands each of them a fresh Python
      // wrapper, so engine.execute's `seen_out` pass cannot see the
      // duplicate. (Input aliasing is a static property of the program and
      // is handled where it belongs — tape.py declines the shapes of
      // forwarding that would reach here.)
      for (size_t i = 1; i < outs.size(); i++) {
        for (size_t j = 0; j < i; j++) {
          if (outs[i].id() == outs[j].id()) {
            outs[i] = fresh_copy(outs[i]);
            break;
          }
        }
      }
    }
    return outs;
  }

  size_t num_ops() const { return ops_.size(); }
  int num_slots() const { return nslots_; }

  // Peak number of slots the environment holds at once, from the drop
  // lists alone. The tests assert on it: it is what says liveness pruning
  // is really running, and a chain whose intermediates are not dropped
  // shows up here as a count that grows with the chain.
  int max_live() const {
    std::vector<char> live(nslots_, 0);
    int cur = nargs_, peak = nargs_;
    for (int i = 0; i < nargs_; i++) live[i] = 1;
    for (const Entry& e : ops_) {
      for (int s : e.outs) {
        if (!live[s]) { live[s] = 1; cur++; }
      }
      if (cur > peak) peak = cur;
      for (int s : e.drops) {
        if (live[s]) { live[s] = 0; cur--; }
      }
    }
    return peak;
  }

 private:
  void check_slot(int s) const {
    if (s < 0 || s >= nslots_)
      throw std::invalid_argument("tape: slot index out of range");
  }

  void step(const Entry& e, std::vector<std::optional<mx::array>>& env) const {
    auto in = [&](size_t i) -> const mx::array& {
      const auto& v = env[e.ins[i]];
      if (!v) throw std::runtime_error("tape: read of a dropped slot");
      return *v;
    };
    const std::vector<int64_t>& at = e.attrs;

    switch (e.op) {
      // --- unary (ops/elementwise.py _UNARY, real-dtype branches) ---
      case kAbs: env[e.outs[0]] = mx::abs(in(0)); break;
      case kCeil: env[e.outs[0]] = mx::ceil(in(0)); break;
      case kCos: env[e.outs[0]] = mx::cos(in(0)); break;
      case kErf: env[e.outs[0]] = mx::erf(in(0)); break;
      case kErfInv: env[e.outs[0]] = mx::erfinv(in(0)); break;
      case kExp: env[e.outs[0]] = mx::exp(in(0)); break;
      case kFloor: env[e.outs[0]] = mx::floor(in(0)); break;
      case kIsFinite: env[e.outs[0]] = mx::isfinite(in(0)); break;
      case kLog: env[e.outs[0]] = mx::log(in(0)); break;
      case kLog1p: env[e.outs[0]] = mx::log1p(in(0)); break;
      case kLogistic: env[e.outs[0]] = mx::sigmoid(in(0)); break;
      case kNegate: env[e.outs[0]] = mx::negative(in(0)); break;
      case kRsqrt: env[e.outs[0]] = mx::rsqrt(in(0)); break;
      case kSin: env[e.outs[0]] = mx::sin(in(0)); break;
      case kSqrt: env[e.outs[0]] = mx::sqrt(in(0)); break;
      case kTan: env[e.outs[0]] = mx::tan(in(0)); break;
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
        // _sign: mx.sign returns 0 for NaN, stablehlo.sign propagates it.
        const mx::array& x = in(0);
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
        env[e.outs[0]] =
            x.dtype() == mx::float32
                ? mx::where(mx::less(mx::abs(x), weak(0.25, x)), mx::expm1(x),
                            mx::subtract(mx::exp(x), weak(1.0, x)))
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
        const mx::array& a = in(0);
        const mx::array& b = in(1);
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
        env[e.outs[0]] = mx::astype(in(0), dtype_of(at[0]));
        break;

      // --- shape (ops/shape.py) ---
      case kReshape:
        env[e.outs[0]] = mx::reshape(in(0), shape(at, 1, at[0]));
        break;
      case kTranspose:
        env[e.outs[0]] = mx::transpose(in(0), axes(at, 1, at[0]));
        break;
      case kBroadcastInDim: {
        // _broadcast_in_dim: unsorted broadcast_dimensions become a
        // transpose, then the operand reshapes to an interim shape with a
        // 1 in every dim it does not name and broadcasts out. The perm and
        // the interim shape are static, so tape.py resolved both.
        mx::array x = in(0);
        size_t p = 0;
        bool do_transpose = at[p++] != 0;
        int64_t in_rank = at[p++];
        if (do_transpose) x = mx::transpose(x, axes(at, p, in_rank));
        p += static_cast<size_t>(in_rank);
        int64_t out_rank = at[p++];
        mx::Shape interim = shape(at, p, out_rank);
        p += static_cast<size_t>(out_rank);
        mx::Shape out = shape(at, p, out_rank);
        env[e.outs[0]] = mx::broadcast_to(mx::reshape(x, interim), out);
        break;
      }
      case kSlice: {
        int64_t rank = at[0];
        env[e.outs[0]] = mx::slice(
            in(0), shape(at, 1, rank), shape(at, 1 + rank, rank),
            shape(at, 1 + 2 * rank, rank));
        break;
      }
      case kConcatenate: {
        std::vector<mx::array> parts;
        parts.reserve(e.ins.size());
        for (size_t i = 0; i < e.ins.size(); i++) parts.push_back(in(i));
        env[e.outs[0]] =
            mx::concatenate(std::move(parts), static_cast<int>(at[0]));
        break;
      }
      case kIota: {
        // _iota: ramp along `dim`, broadcast, cast. MLX has no bool arange,
        // so the ramp runs in int32 for a bool result (the Python handler's
        // complex arm is unreachable: complex declines).
        int dim = static_cast<int>(at[0]);
        mx::Dtype ramp_dt = dtype_of(at[1]);
        mx::Dtype dt = dtype_of(at[2]);
        int64_t rank = at[3];
        mx::Shape out = shape(at, 4, rank);
        mx::array ramp = mx::arange(static_cast<double>(out[dim]), ramp_dt);
        mx::Shape view(static_cast<size_t>(rank));
        for (int64_t i = 0; i < rank; i++) view[i] = 1;
        view[dim] = out[dim];
        env[e.outs[0]] =
            mx::astype(mx::broadcast_to(mx::reshape(ramp, view), out), dt);
        break;
      }

      case kConstant:
        // Decoded once, in Python, by the same battle-tested path the
        // eager engine uses (splat broadcast, bf16 text/hex forms, the
        // rank-0 literal rule); it crosses at lowering and never again.
        env[e.outs[0]] = *e.payload;
        break;

      case kReduce: {
        // ops/reduction.py _reduce, single-operand monoid form.
        const mx::array& x = in(0);
        const mx::array& init = in(1);
        int64_t kind = at[0];
        int64_t ndims = at[1];
        std::vector<int> dims = axes(at, 2, ndims);
        bool empty = false;
        for (auto s : x.shape()) if (s == 0) empty = true;
        if (empty) {
          // MLX reducers crash on zero-size inputs (mx.max raises; a
          // zero-size uint32 sum aborts in a missing Metal kernel). An
          // empty fold is well defined: the init value.
          mx::Shape out;
          for (size_t i = 0; i < x.shape().size(); i++) {
            bool reduced = false;
            for (int d : dims) if (static_cast<size_t>(d) == i) reduced = true;
            if (!reduced) out.push_back(x.shape()[i]);
          }
          bool reduced_empty = false;
          for (int d : dims) if (x.shape()[d] == 0) reduced_empty = true;
          env[e.outs[0]] =
              reduced_empty
                  ? mx::broadcast_to(mx::astype(init, x.dtype()), out)
                  : mx::zeros(out, x.dtype());
          break;
        }
        mx::array out = dims.empty() ? x : reduce_apply(kind, x, dims);
        env[e.outs[0]] = reduce_combine(kind, out, init);
        break;
      }

      case kArgReduce: {
        // ops/reduction.py _reduce, the (values, indices) form jax lowers
        // argmax/argmin to. Ties resolve to the lowest index, which is
        // MLX's first-occurrence answer; the NaN rules are XLA's, not
        // MLX's, and are the reason this is not just an argmax call.
        const mx::array& x = in(0);
        const mx::array& ids = in(1);
        bool is_max = at[0] != 0;
        int d = static_cast<int>(at[1]);
        bool empty = false;
        for (auto s : x.shape()) if (s == 0) empty = true;
        if (empty) {
          // Only BATCH dims can be zero here (jax forbids argmax over an
          // empty reduced axis) and MLX's reducers crash on empties.
          mx::Shape out;
          for (size_t i = 0; i < x.shape().size(); i++)
            if (static_cast<int>(i) != d) out.push_back(x.shape()[i]);
          env[e.outs[0]] = mx::zeros(out, x.dtype());
          env[e.outs[1]] = mx::zeros(out, ids.dtype());
          break;
        }
        mx::array val = is_max ? mx::max(x, d) : mx::min(x, d);
        mx::array arg = is_max ? mx::argmax(x, d) : mx::argmin(x, d);
        if (is_float(x.dtype())) {
          // XLA/numpy: a NaN wins argmax AND argmin, and the FIRST one's
          // index is the answer. MLX skips NaNs entirely.
          mx::array nans = mx::isnan(x);
          mx::array has_nan = mx::any(nans, std::vector<int>{d});
          mx::array first_nan = mx::argmax(nans, d);
          arg = mx::where(has_nan, first_nan, arg);
          val = mx::where(
              has_nan,
              mx::array(std::numeric_limits<double>::quiet_NaN(), val.dtype()),
              val);
        }
        mx::array idx = mx::take_along_axis(ids, mx::expand_dims(arg, d), d);
        env[e.outs[0]] = val;
        env[e.outs[1]] = mx::squeeze(idx, d);
        break;
      }

      case kDotGeneral: {
        // ops/linalg.py _dot_general. Which of the three arms runs is a
        // static property of the operand dtypes, so tape.py resolved it;
        // all three are here because MLX has no integer matmul.
        size_t p = 0;
        int64_t lrank = at[p++];
        std::vector<int> lperm = axes(at, p, lrank);
        p += static_cast<size_t>(lrank);
        int64_t rrank = at[p++];
        std::vector<int> rperm = axes(at, p, rrank);
        p += static_cast<size_t>(rrank);
        int64_t b = at[p++], m = at[p++], k = at[p++], n = at[p++];
        mx::Dtype out_dt = dtype_of(at[p++]);
        int64_t out_rank = at[p++];
        mx::Shape out_shape = shape(at, p, out_rank);
        p += static_cast<size_t>(out_rank);
        int64_t kind = at[p++];
        int64_t chunk = at[p++];

        mx::array l = mx::transpose(in(0), lperm);
        mx::array r = mx::transpose(in(1), rperm);
        mx::array l3 = mx::reshape(
            l, mx::Shape{static_cast<int>(b), static_cast<int>(m),
                         static_cast<int>(k)});
        mx::array r3 = mx::reshape(
            r, mx::Shape{static_cast<int>(b), static_cast<int>(k),
                         static_cast<int>(n)});
        if (b * m * n == 0) {
          // mx.matmul with an empty M/N output yields an array whose host
          // conversion segfaults (null data pointer, MLX 0.32).
          env[e.outs[0]] = mx::zeros(out_shape, out_dt);
          break;
        }
        mx::array o3 = l3;
        if (kind == 1) {
          // _int_dot_via_f32: f32 holds every integer up to 2**24 exactly,
          // so an 8-bit integer dot over short enough K-slices is exact in
          // a real matmul; the per-chunk results accumulate in integer
          // arithmetic, which wraps exactly like XLA's integer dot.
          if (chunk <= 0)
            throw std::invalid_argument("tape: bad dot chunk");
          mx::Dtype acc_dt = out_dt.size() == 8 ? mx::int64 : mx::int32;
          std::optional<mx::array> acc;
          for (int64_t s = 0; s < k; s += chunk) {
            mx::array lp = l3, rp = r3;
            if (k > chunk) {
              int64_t hi = std::min(s + chunk, k);
              lp = mx::slice(l3, mx::Shape{0, 0, static_cast<int>(s)},
                             mx::Shape{static_cast<int>(b),
                                       static_cast<int>(m),
                                       static_cast<int>(hi)});
              rp = mx::slice(r3, mx::Shape{0, static_cast<int>(s), 0},
                             mx::Shape{static_cast<int>(b),
                                       static_cast<int>(hi),
                                       static_cast<int>(n)});
            }
            mx::array part = mx::astype(
                mx::matmul(mx::astype(lp, mx::float32),
                           mx::astype(rp, mx::float32)),
                acc_dt);
            acc = acc ? mx::add(*acc, part) : part;
          }
          o3 = mx::astype(*acc, out_dt);
        } else if (kind == 2 || kind == 3) {
          // MLX matmul is float-only: an explicit multiply-accumulate over
          // a materialized [B, M, K, N] product. kind 3 is the bool arm,
          // whose accumulator stays bool until mx.sum promotes it.
          mx::array acc = kind == 3 ? l3 : mx::astype(l3, mx::int64);
          mx::array prod = mx::multiply(mx::expand_dims(acc, 3),
                                        mx::expand_dims(
                                            mx::astype(r3, acc.dtype()), 1));
          o3 = mx::astype(mx::sum(prod, std::vector<int>{2}), out_dt);
        } else {
          if (l3.dtype() != out_dt) l3 = mx::astype(l3, out_dt);
          if (r3.dtype() != out_dt) r3 = mx::astype(r3, out_dt);
          o3 = mx::matmul(l3, r3);
        }
        env[e.outs[0]] = mx::reshape(o3, out_shape);
        break;
      }

      default:
        throw std::invalid_argument("tape: unknown opcode");
    }

    for (int s : e.drops) env[s].reset();
  }

  static mx::Shape shape(const std::vector<int64_t>& at, size_t off,
                         int64_t n) {
    mx::Shape s(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++)
      s[i] = static_cast<mx::ShapeElem>(at[off + static_cast<size_t>(i)]);
    return s;
  }

  static std::vector<int> axes(const std::vector<int64_t>& at, size_t off,
                               int64_t n) {
    std::vector<int> a(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++)
      a[i] = static_cast<int>(at[off + static_cast<size_t>(i)]);
    return a;
  }

  int nslots_;
  int nargs_;
  std::vector<Entry> ops_;
  std::vector<int> outputs_;
};

nb::dict opcodes() {
  nb::dict d;
  for (const NamedOp& n : kOpNames) d[n.name] = n.op;
  return d;
}

nb::dict dtype_codes() {
  nb::dict d;
  for (int i = 0; i < kNumDtypes; i++) d[kDtypes[i].name] = i;
  return d;
}

}  // namespace

void register_tape(nb::module_& m) {
  m.def("opcodes", &opcodes,
        "StableHLO op name -> opcode; a name absent here declines");
  m.def("dtype_codes", &dtype_codes,
        "MLIR element type -> dtype code; absent means unsupported");
  nb::class_<Program>(m, "Program")
      .def(nb::init<int, int>(), nb::arg("num_slots"), nb::arg("num_args"))
      .def("add", &Program::add, nb::arg("opcode"), nb::arg("operands"),
           nb::arg("results"), nb::arg("attrs"),
           nb::arg("payload").none(), nb::arg("drops"))
      .def("set_outputs", &Program::set_outputs, nb::arg("slots"))
      .def("run", &Program::run, nb::arg("inputs"))
      .def_prop_ro("num_ops", &Program::num_ops)
      .def_prop_ro("num_slots", &Program::num_slots)
      .def_prop_ro("max_live", &Program::max_live);
}
