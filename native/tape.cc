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
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mlx/mlx.h>
#include <mlx/compile_impl.h>

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
  kBitcastConvert, kDynamicSlice, kDynamicUpdateSlice, kSort, kTopK,
  // control flow (M3): each carries its regions as sub-Programs
  kWhile, kIf, kCase,
  // recognizer emits (M4): a REWRITTEN program's roots. These are not
  // StableHLO ops — src/metaljax/tape.py resolves the recognizer's plan
  // (metaljax.interpreter._rewrite_plan) at lowering time and asks for
  // them by the pseudo-names below, exactly as it does for the argmax
  // reduce. The absorbed ops never reach the tape at all.
  kQmm, kSdpa, kSdpaMask,
  kMoeEIdx, kMoeTIdx, kMoeGather, kMoeConcat, kMoeView, kMoeDot, kMoeTail,
  // M5b: a counted loop msl_scan planned into one generated Metal kernel,
  // and a site where the handler computes on the HOST. Both are lowered by
  // src/metaljax/tape.py; the pseudo-names below are how it asks for them.
  kMslScan, kHostCall,
  // Ordered-effect tokens: no data, only order (dtypes.token_value).
  kToken,
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
    {"stablehlo.bitcast_convert", kBitcastConvert},
    {"stablehlo.dynamic_slice", kDynamicSlice},
    {"stablehlo.dynamic_update_slice", kDynamicUpdateSlice},
    {"stablehlo.sort", kSort},
    // chlo.top_k survives a direct jax lowering; through a portable
    // artifact it arrives already decomposed into the sort above. Both
    // reach the plugin, so both are here.
    {"chlo.top_k", kTopK},
    {"stablehlo.while", kWhile},
    {"stablehlo.if", kIf},
    {"stablehlo.case", kCase},
    // M4 recognizer emits. Pseudo-names: a build that predates a given
    // emit simply does not offer it, and tape.py declines the program —
    // which is why there is no version negotiation anywhere else.
    {"metaljax.qmm", kQmm},
    {"metaljax.sdpa", kSdpa},
    {"metaljax.sdpa.mask", kSdpaMask},
    {"metaljax.moe.eidx", kMoeEIdx},
    {"metaljax.moe.tidx", kMoeTIdx},
    {"metaljax.moe.gather", kMoeGather},
    {"metaljax.moe.concat", kMoeConcat},
    {"metaljax.moe.view", kMoeView},
    {"metaljax.moe.dot", kMoeDot},
    {"metaljax.moe.tail", kMoeTail},
    // M5b pseudo-names.
    {"metaljax.msl_scan", kMslScan},
    {"metaljax.host_call", kHostCall},
    {"stablehlo.create_token", kToken},
    {"stablehlo.after_all", kToken},
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
// runtime disciplines (M3)
// --------------------------------------------------------------------------
//
// Every cadence below is a MEASURED value with a crash or a corruption story
// behind it (see src/metaljax/interpreter.py and ops/control.py, which own
// the comments). None of them is re-derived here: `configure` copies them in
// from the Python modules that parse the environment, so the two engines can
// never drift apart on a number the command-buffer lottery is pinned to.

struct Config {
  int64_t eager_flush_bytes = 1024LL << 20;   // METALJAX_EAGER_FLUSH_MB
  int64_t flush_sync_every = 1;               // METALJAX_EAGER_FLUSH_SYNC
  int64_t flush_clear_bytes = 2048LL << 20;   // METALJAX_FLUSH_CLEAR_MB
  int64_t loop_clear_cost = 500000;           // METALJAX_LOOP_CLEAR_COST
  int64_t while_pipeline = 1;                 // METALJAX_WHILE_PIPELINE
  bool debug = false;                         // METALJAX_DEBUG
  bool memdbg = false;                        // METALJAX_MEMDBG
};

Config g_cfg;

struct Stats {
  int64_t flushes = 0;         // eager byte-denominated sync points
  int64_t cache_clears = 0;    // ...that returned cache memory to the OS
  int64_t loop_flushes = 0;    // loop sync points
  int64_t loop_clears = 0;     // ...that cleared on the op-unit cadence
  int64_t limit_retries = 0;   // Metal buffer-limit recoveries
  int64_t compiled_calls = 0;  // replays of a compiled tape (main or body)
  int64_t compiles = 0;        // mx::compile traces built
  int64_t compile_drops = 0;   // compiled paths abandoned after a failure
  int64_t chunk_drops = 0;     // chunked loop replays abandoned
  int64_t unrolls = 0;         // counted loops unrolled into a trace
  int64_t pipelined_loops = 0;  // dynamic whiles that ran pipelined
  int64_t pipelined_steps = 0;  // ...iterations they retired that way
  int64_t serial_loops = 0;     // dynamic whiles that could not pipeline
  int64_t msl_launches = 0;     // generated persistent kernels launched
  int64_t msl_failures = 0;     // ...plans retired to their loops
  int64_t host_calls = 0;       // entries that reacquired the GIL
};

Stats g_stats;

// A distinctive high tag: mx::detail::compile keys its cache by an integer
// the CALLER owns, and MLX's own Python bindings use the address of a Python
// function object. Ids from this counter can collide with neither those nor
// any real pointer (user-space addresses on arm64 macOS live far below).
std::atomic<std::uintptr_t> g_next_compile_id{0x6D6A5F0000000001ULL};

std::uintptr_t new_compile_id() { return g_next_compile_id.fetch_add(1); }

bool is_resource_limit(const std::exception& e) {
  return std::string(e.what()).find("Resource limit") != std::string::npos;
}

// Python's cycle collector barely triggers under array workloads, and dead
// refcycles pin buffers mx::clear_cache cannot free (CLAUDE.md item 19).
// Needs the GIL, which the hot path has released.
void gc_collect() {
  nb::gil_scoped_acquire gil;
  try {
    nb::module_::import_("gc").attr("collect")();
  } catch (...) {
    // A collection that cannot run must not take the recovery down with it.
  }
}

// Narration. Deliberately not routed through Python: these fire from paths
// that have released the GIL, some of them while recovering from an
// allocation failure.
void debug_line(const std::string& line) {
  fputs((line + "\n").c_str(), stdout);
  fflush(stdout);
}

// METALJAX_DEBUG.
void debug_print(const std::string& msg) {
  if (g_cfg.debug) debug_line("[metaljax] " + msg);
}

// interpreter.flush_eval: settle `arrays`, recovering once from Metal buffer
// exhaustion. Programs are pure, so clearing and retrying is safe.
void flush_eval(const std::vector<mx::array>& arrays, bool hard) {
  try {
    if (hard) {
      mx::eval(arrays);
    } else {
      mx::async_eval(arrays);
    }
    return;
  } catch (const std::exception& e) {
    if (!is_resource_limit(e)) throw;
  }
  debug_print("Metal buffer limit hit at eager flush; clearing cache and "
              "retrying");
  g_stats.limit_retries++;
  gc_collect();
  mx::clear_cache();
  mx::eval(arrays);
}

// ops.control._loop_flush: a sync point inside a loop. Evaluates pending
// work, keeps the Metal buffer COUNT bounded (MLX's cache is bounded by
// bytes only, and long loops over small models accumulate tiny buffers until
// metal::malloc dies), and recovers once if the limit is hit anyway.
int64_t g_flushed_cost = 0;

// The blocking half: settle `arrays`, recovering once from exhaustion.
void loop_eval(const std::vector<mx::array>& arrays) {
  bool retry = false;
  try {
    mx::eval(arrays);
  } catch (const std::exception& e) {
    if (!is_resource_limit(e)) throw;
    retry = true;
  }
  if (retry) {
    debug_print("Metal buffer limit hit at loop flush; clearing cache and "
                "retrying");
    g_stats.limit_retries++;
    gc_collect();  // dead refcycles pin buffers clear_cache cannot free
    mx::clear_cache();
    mx::eval(arrays);
  }
}

// The bookkeeping half: charge a sync point's work against the op-unit
// budget and return cached buffers to the OS when it is spent. Split out
// because a pipelined loop's blocking point is a HOST READ of the condition
// rather than an eval of the carry -- a different array to wait on, the same
// iteration of work to charge, and the same cadence either way.
void loop_account(int64_t cost_units) {
  g_stats.loop_flushes++;
  g_flushed_cost += cost_units;
  if (g_cfg.loop_clear_cost > 0 && g_flushed_cost >= g_cfg.loop_clear_cost) {
    g_flushed_cost = 0;
    g_stats.loop_clears++;
    mx::clear_cache();
    if (g_cfg.memdbg)
      debug_line("[metaljax-mem] loop clear: active=" +
                 std::to_string(mx::get_active_memory()) + "B cache=" +
                 std::to_string(mx::get_cache_memory()) + "B");
  }
}

void loop_flush(const std::vector<mx::array>& arrays, int64_t cost_units) {
  loop_eval(arrays);
  loop_account(cost_units);
}

// ops.control._anchor_outputs: give constant outputs a bitwise-exact data
// dependency on an input. where(x == x, out, out) == out for every bit
// pattern (both branches are `out`), but the result is a computed node, not
// a constant MLX's compiler can bake into a table KEYED BY VALUE -- where
// two equal-valued constant outputs collide and the compiled call dies with
// unordered_map::at.
void anchor_outputs(std::vector<mx::array>& outs,
                    const std::vector<mx::array>& args,
                    const std::vector<int>& underived) {
  if (underived.empty() || args.empty()) return;
  std::optional<mx::array> anchor;
  for (const mx::array& a : args) {
    if (a.size() == 0) continue;
    mx::array head = mx::slice(mx::reshape(a, mx::Shape{-1}), mx::Shape{0},
                               mx::Shape{1});
    anchor = mx::reshape(mx::equal(head, head), mx::Shape{});
    break;
  }
  if (!anchor) return;
  for (int i : underived) {
    if (i >= 0 && static_cast<size_t>(i) < outs.size())
      outs[i] = mx::where(*anchor, outs[i], outs[i]);
  }
}

// The loop counter / branch index of a control-flow op, on the host. Reading
// one is a sync point, exactly as the Python handlers' `.item()` is.
int64_t item_int(const mx::array& a) {
  mx::array v = mx::astype(a, mx::int64);
  v.eval();
  return v.item<int64_t>();
}

bool item_bool(const mx::array& a) {
  mx::array v = mx::astype(a, mx::bool_);
  v.eval();
  return v.item<bool>();
}

// A loop condition's host read. Same value as item_bool, but the wait goes
// through the loop's recovery path: on a pipelined loop this IS the sync
// point, so it is where Metal's buffer limit would surface.
bool loop_item_bool(const mx::array& a) {
  mx::array v = mx::astype(a, mx::bool_);
  loop_eval({v});
  return v.item<bool>();
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
//   kBitcastConvert     [dtype, kind]           kind: 0 same width,
//                                               1 narrowing, 2 widening
//   kDynamicSlice       [rank, clamp bounds..., sizes...]
//   kDynamicUpdateSlice [rank, clamp bounds...]  (sizes = update's shape)
//   kWhile              [ncarry, ncond_caps, nbody_caps, counted, k,
//                        bound_kind, bound, cost, period, chunkable, kmax,
//                        body_compile_max]
//                       regions [cond, body]; ins [carry..., cond caps...,
//                       body caps...]; bound_kind 0 static N, 1 carry index,
//                       2 index into the cond's captures
//   kIf / kCase         [ncaps_0, ncaps_1, ...] one per region
//                       regions = the branches; ins [pred/index, caps...]
//   kQmm/kSdpa/kMoe*    documented beside their handlers — the M4 emits'
//                       layouts are long enough to want reading in place,
//                       and they are read with a Cursor, not by index
//
// A region is a Program of its own, whose arguments are the region block's
// arguments followed by its CAPTURES -- the values it reads from enclosing
// scopes, resolved to parent slots at lowering. That is the whole of the
// nesting: no environment is shared, so a sub-program is compiled, replayed
// and reasoned about exactly like a top-level one.

class Program;
class MslPlan;

struct Entry {
  int op;
  std::vector<int> ins;
  std::vector<int> outs;
  std::vector<int64_t> attrs;
  // Float attributes. Only the recognizer emits have any: an attention's
  // scale and a mask sentinel are real numbers, and squeezing them through
  // the integer vector would be a bit-pattern encoding nobody could read.
  std::vector<double> fattrs;
  std::optional<mx::array> payload;  // kConstant only
  std::vector<int> drops;            // slots whose last use is this op
  std::vector<std::shared_ptr<Program>> regions;  // control flow only
  int64_t bytes = 0;  // estimated result bytes, for the eager flush cadence
  // M5b. `msl`: a generated persistent kernel for this loop (the entry
  // still carries the interpreted loop in `regions`, as the fallback).
  // `host`: the Python handler of an op that computes on the host, the one
  // place a native run reacquires the GIL.
  std::shared_ptr<MslPlan> msl;
  nb::object host;
};

// Sequential reader for the attribute vectors the M4 emits carry: they are
// long enough (three shape recipes for one attention) that positional
// indices would be unreadable on both sides. The layouts are documented
// beside each handler; tape.py writes them in the same order.
class Cursor {
 public:
  explicit Cursor(const std::vector<int64_t>& at) : at_(at) {}
  int64_t next() {
    if (p_ >= at_.size()) throw std::invalid_argument("tape: attrs underrun");
    return at_[p_++];
  }
  bool flag() { return next() != 0; }
  std::vector<int> vec() {
    int64_t n = next();
    std::vector<int> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) v[i] = static_cast<int>(next());
    return v;
  }
  std::vector<int64_t> vec64() {   // strides, which are int64 in MLX
    int64_t n = next();
    std::vector<int64_t> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) v[i] = next();
    return v;
  }
  bool done() const { return p_ >= at_.size(); }
  mx::Shape shp() {
    int64_t n = next();
    mx::Shape s(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++)
      s[i] = static_cast<mx::ShapeElem>(next());
    return s;
  }

 private:
  const std::vector<int64_t>& at_;
  size_t p_ = 0;
};

// metaljax.sdpa._apply: an optional transpose then an optional reshape.
struct Rec {
  bool has_perm = false;
  std::vector<int> perm;
  bool has_shape = false;
  mx::Shape shape;
};

Rec read_rec(Cursor& c) {
  Rec r;
  r.has_perm = c.flag();
  r.perm = c.vec();
  r.has_shape = c.flag();
  r.shape = c.shp();
  return r;
}

mx::array apply_rec(const Rec& r, mx::array x) {
  if (r.has_perm) x = mx::transpose(x, r.perm);
  if (r.has_shape) x = mx::reshape(x, r.shape);
  return x;
}

// MLX names its quantization modes with strings; the tape carries the code
// the Python match recorded.
const char* qmode(int64_t mode) { return mode == 0 ? "affine" : "mxfp4"; }

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

// moe._pair_lead: `x` with a leading pair axis (size P or 1).
mx::array pair_lead(const mx::array& x, bool paired) {
  if (paired) return x;
  mx::Shape s{1};
  for (auto d : x.shape()) s.push_back(d);
  return mx::reshape(x, s);
}

// moe._split_experts: a qmm pack's leading `E * N` rows as [E, N, ...].
mx::array split_experts(const mx::array& a, int64_t E) {
  if (a.ndim() >= 3) return a;
  return mx::reshape(a, mx::Shape{static_cast<mx::ShapeElem>(E),
                                  a.shape()[0] / static_cast<int>(E),
                                  a.shape()[1]});
}

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

// --------------------------------------------------------------------------
// M5b: generated persistent kernels (src/metaljax/msl_scan.py)
// --------------------------------------------------------------------------
//
// A counted loop whose body msl_scan can express becomes ONE Metal kernel:
// a thread per lane, the carries in registers, the whole time loop inside
// the shader. The planning — pattern match, mode choice, MSL generation —
// is compile-time work and stays in Python. What crosses is the LAUNCH: the
// generated source, the binding order, the geometry, and the little recipe
// `Plan.run` applies afterwards to turn the kernel's outputs back into the
// loop's carries. Everything below is that recipe, transliterated; the
// layouts are documented at each parse, and metaljax.tape._lower_msl writes
// them in this order.
//
// A plan that fails at run time is DEAD for the rest of the process
// (Metal's shader compiler rejecting a generated source is not a transient
// condition), and the entry falls back to the interpreted loop it carries
// alongside — the C++ half of engine.MetalExecutable.disable_msl.

// One node of an accumulator's stacking recipe (`Plan.run`'s `stacked`).
struct AccNode {
  int kind = 0;             // 0 hidden, 1 buffer, 2 red, 3 dot
  int idx = 0;              // hidden: output index; buffer: source id
  int64_t a = 0, b = 0;     // buffer: the read's affine index a*t + b
  mx::Shape shape;          // buffer/red: the per-step shape
  std::vector<int> dims, perm;
  std::vector<int> lb, rb, lc, rc;   // dot: batch/contracting dims
  mx::Shape lshape, rshape;
  std::vector<AccNode> kids;
};

bool is_identity_perm(const std::vector<int>& p) {
  for (size_t i = 0; i < p.size(); i++)
    if (p[i] != static_cast<int>(i)) return false;
  return true;
}

int64_t shape_prod(const mx::Shape& s, const std::vector<int>& idx) {
  int64_t p = 1;
  for (int i : idx) p *= s[static_cast<size_t>(i)];
  return p;
}

class MslPlan {
 public:
  MslPlan(std::string name, std::string source, std::string header,
          std::vector<std::string> input_names,
          std::vector<std::string> output_names,
          std::vector<std::vector<int>> out_shapes,
          std::vector<int> out_dtypes, std::vector<int64_t> layout)
      : name_(std::move(name)), source_(std::move(source)),
        header_(std::move(header)), in_names_(std::move(input_names)),
        out_names_(std::move(output_names)) {
    for (const auto& s : out_shapes)
      out_shapes_.push_back(mx::Shape(s.begin(), s.end()));
    for (int c : out_dtypes) out_dtypes_.push_back(dtype_of(c));
    if (out_shapes_.size() != out_dtypes_.size() ||
        out_shapes_.size() != out_names_.size())
      throw std::invalid_argument("msl: output spec mismatch");
    parse(layout);
  }

  // How many arrays the entry hands us after the loop's own inputs.
  size_t num_sources() const { return norms_.size(); }
  bool dead() const { return dead_; }
  void kill() { dead_ = true; }
  bool validated() const { return validated_; }
  void validate() { validated_ = true; }
  const std::string& name() const { return name_; }

  // One launch: the whole loop. `carries` are the loop's initial carries,
  // `srcs` the arrays its sources resolved to, in plan order.
  std::vector<mx::array> run(const std::vector<mx::array>& carries,
                             const std::vector<mx::array>& srcs,
                             bool in_trace,
                             std::vector<MslPlan*>& pending) {
    if (carries.size() != static_cast<size_t>(ncarry_))
      throw std::runtime_error("msl: wrong carry count");
    if (srcs.size() != norms_.size())
      throw std::runtime_error("msl: wrong source count");
    build();

    // `Plan.run`'s `bufs`: a weight whose layout was canonicalized at plan
    // time is materialized here, per call, exactly as it is there (the
    // backward pass reads W^T, which is uncoalesced without this).
    std::vector<mx::array> bufs;
    bufs.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); i++) {
      const Norm& n = norms_[i];
      if (!n.on) {
        bufs.push_back(srcs[i]);
        continue;
      }
      mx::array v = mx::as_strided(srcs[i], n.shape, n.strides, n.offset);
      bufs.push_back(mx::contiguous(mx::transpose(v, n.perm)));
    }

    // ...and its `feed`: the unpacked sources, then one pooled buffer per
    // dtype (the 0.4.3 input packing, whose element offsets are already
    // baked into the generated source), then the state initializers.
    std::vector<mx::array> feed;
    feed.reserve(unpacked_.size() + packs_.size() + state_pos_.size());
    for (int sid : unpacked_) feed.push_back(bufs[static_cast<size_t>(sid)]);
    for (const Pack& p : packs_) {
      std::vector<mx::array> parts;
      parts.reserve(p.sids.size());
      for (int sid : p.sids)
        parts.push_back(mx::astype(
            mx::reshape(bufs[static_cast<size_t>(sid)], mx::Shape{-1}),
            p.dtype));
      feed.push_back(mx::concatenate(parts, 0));
    }
    for (int pos : state_pos_)
      feed.push_back(carries[static_cast<size_t>(pos)]);
    if (feed.size() != in_names_.size())
      throw std::runtime_error("msl: input count does not match the kernel");

    // Once per plan (the first launch), like msl_scan's own narration: the
    // binding order is what a mis-encoded recipe gets wrong, and it does
    // not change from call to call.
    if (g_cfg.debug && !narrated_) {
      narrated_ = true;
      std::string msg = "[metaljax] msl " + name_ +
                        ": grid=" + std::to_string(N_) +
                        " tg=" + std::to_string(tg_) + " in";
      for (size_t i = 0; i < feed.size(); i++) {
        msg += " " + in_names_[i] + "(";
        for (auto d : feed[i].shape()) msg += std::to_string(d) + ",";
        msg += std::to_string(feed[i].itemsize()) + "B)";
      }
      debug_line(msg);
    }
    std::vector<mx::array> outs = kernel_(
        feed, out_shapes_, out_dtypes_,
        std::tuple<int, int, int>{static_cast<int>(N_), 1, 1},
        std::tuple<int, int, int>{static_cast<int>(tg_), 1, 1}, {},
        std::nullopt, false, {});
    g_stats.msl_launches++;

    if (!validated_) {
      if (!in_trace) {
        // The first call proves the kernel: MLX generates the Metal library
        // at EVAL, and a build error raised on an async worker aborts the
        // process. Synchronous here, so the caller's catch can retire the
        // plan to the interpreted loop instead.
        mx::eval(outs);
        validated_ = true;
      } else {
        // Traced into an enclosing mx::compile graph: there is nothing to
        // evaluate here (these are tracers). The plan is unproven until the
        // whole call is settled, which Program::run does synchronously
        // while this list is non-empty -- msl_scan's own contract, and
        // engine.execute's `_msl_pending`.
        bool known = false;
        for (MslPlan* p : pending) known = known || p == this;
        if (!known) pending.push_back(this);
      }
    }

    if (outs.size() != out_shapes_.size())
      throw std::runtime_error("msl: kernel returned the wrong output count");
    std::vector<mx::array> vals(carries);
    const size_t ns = stacked_pos_.size();
    for (size_t q = 0; q < ns; q++)
      vals[static_cast<size_t>(stacked_pos_[q])] = outs[q];
    for (size_t j = 0; j < state_pos_.size(); j++)
      vals[static_cast<size_t>(state_pos_[j])] =
          outs[ns + static_cast<size_t>(nhidden_) + j];
    for (const auto& c : counters_) {
      const mx::array& x = carries[static_cast<size_t>(c.first)];
      vals[static_cast<size_t>(c.first)] =
          mx::add(x, weak_int(c.second * trip_, x));
    }
    for (const auto& a : acc_) {
      const mx::array& x = carries[static_cast<size_t>(a.first)];
      std::optional<mx::array> total;
      for (const AccNode& spec : a.second) {
        mx::array t = mx::sum(stacked_of(spec, outs, ns, srcs),
                              std::vector<int>{0});
        t = mx::reshape(t, x.shape());
        total = total ? mx::add(*total, t) : t;
      }
      if (!total) throw std::runtime_error("msl: accumulator with no terms");
      vals[static_cast<size_t>(a.first)] = mx::add(x, *total);
    }
    return vals;
  }

 private:
  struct Norm {
    bool on = false;
    mx::Shape shape;
    mx::Strides strides;
    size_t offset = 0;
    std::vector<int> perm;
  };
  struct Pack {
    mx::Dtype dtype = mx::float32;
    std::vector<int> sids;
  };

  // Layout, in the order metaljax.tape._lower_msl writes it:
  //   N, threadgroup, trip, start, nsources, nhidden, ncarry
  //   per source: on?, [shape], [strides], offset, [perm]   (weight norm)
  //   [unpacked sids]
  //   npacks, then per pack: dtype, [sids]
  //   [state carry positions], [stacked carry positions], [pass-throughs]
  //   ncounters, then per counter: position, per-iteration delta
  //   naccumulators, then per one: position, nterms, term trees
  void parse(const std::vector<int64_t>& layout) {
    Cursor c(layout);
    N_ = c.next();
    tg_ = c.next();
    trip_ = c.next();
    start_ = c.next();
    const int64_t nsrc = c.next();
    nhidden_ = static_cast<int>(c.next());
    ncarry_ = static_cast<int>(c.next());
    for (int64_t i = 0; i < nsrc; i++) {
      Norm n;
      n.on = c.flag();
      if (n.on) {
        n.shape = c.shp();
        std::vector<int64_t> st = c.vec64();
        n.strides = mx::Strides(st.begin(), st.end());
        n.offset = static_cast<size_t>(c.next());
        n.perm = c.vec();
      }
      norms_.push_back(std::move(n));
    }
    unpacked_ = c.vec();
    int64_t npacks = c.next();
    for (int64_t i = 0; i < npacks; i++) {
      Pack p;
      p.dtype = dtype_of(c.next());
      p.sids = c.vec();
      packs_.push_back(std::move(p));
    }
    state_pos_ = c.vec();
    stacked_pos_ = c.vec();
    passthrough_ = c.vec();
    int64_t ncnt = c.next();
    for (int64_t i = 0; i < ncnt; i++) {
      int pos = static_cast<int>(c.next());
      counters_.emplace_back(pos, c.next());
    }
    int64_t nacc = c.next();
    for (int64_t i = 0; i < nacc; i++) {
      int pos = static_cast<int>(c.next());
      int64_t nterms = c.next();
      std::vector<AccNode> terms;
      for (int64_t t = 0; t < nterms; t++) terms.push_back(parse_node(c));
      acc_.emplace_back(pos, std::move(terms));
    }
    if (!c.done()) throw std::invalid_argument("msl: layout has trailing data");
    if (out_shapes_.size() !=
        stacked_pos_.size() + static_cast<size_t>(nhidden_) +
            state_pos_.size())
      throw std::invalid_argument("msl: output count does not match the plan");
    // Every index the recipe will subscript with, checked once, here.
    auto carry = [&](int p) {
      if (p < 0 || p >= ncarry_)
        throw std::invalid_argument("msl: carry index out of range");
    };
    auto source = [&](int s) {
      if (s < 0 || static_cast<size_t>(s) >= norms_.size())
        throw std::invalid_argument("msl: source index out of range");
    };
    for (int p : state_pos_) carry(p);
    for (int p : stacked_pos_) carry(p);
    for (int p : passthrough_) carry(p);
    for (const auto& c2 : counters_) carry(c2.first);
    for (const auto& a : acc_) carry(a.first);
    for (int s : unpacked_) source(s);
    for (const Pack& p : packs_)
      for (int s : p.sids) source(s);
  }

  static AccNode parse_node(Cursor& c) {
    AccNode n;
    n.kind = static_cast<int>(c.next());
    switch (n.kind) {
      case 0:
        n.idx = static_cast<int>(c.next());
        break;
      case 1:
        n.idx = static_cast<int>(c.next());
        n.a = c.next();
        n.b = c.next();
        n.shape = c.shp();
        break;
      case 2:
        n.kids.push_back(parse_node(c));
        n.dims = c.vec();
        n.perm = c.vec();
        n.shape = c.shp();
        break;
      case 3:
        n.kids.push_back(parse_node(c));
        n.kids.push_back(parse_node(c));
        n.lb = c.vec();
        n.rb = c.vec();
        n.lc = c.vec();
        n.rc = c.vec();
        n.perm = c.vec();
        n.lshape = c.shp();
        n.rshape = c.shp();
        break;
      default:
        throw std::invalid_argument("msl: bad accumulator node");
    }
    return n;
  }

  void build() {
    if (built_) return;
    kernel_ = mx::fast::metal_kernel(name_, in_names_, out_names_, source_,
                                     header_);
    built_ = true;
  }

  // `Plan.run`'s `stacked`: the (L, *per-step) array an accumulator term
  // contributes, either straight out of the kernel, straight out of a
  // device buffer, or reduced/contracted after it (loop fission -- a
  // cross-lane dot cannot run per lane, so the kernel stacks its operands
  // and one batched matmul finishes the job here).
  mx::array stacked_of(const AccNode& s, const std::vector<mx::array>& outs,
                       size_t ns, const std::vector<mx::array>& srcs) const {
    // Bounds are checked rather than trusted: an index the lowering got
    // wrong reads whatever is next in memory, which is a NaN or a crash
    // depending on the day (it was both, before the source count and the
    // accumulator recipes were made to agree).
    switch (s.kind) {
      case 0: {
        const size_t i = ns + static_cast<size_t>(s.idx);
        if (s.idx < 0 || i >= outs.size())
          throw std::runtime_error("msl: hidden stack out of range");
        return outs[i];
      }
      case 1: {
        if (s.idx < 0 || static_cast<size_t>(s.idx) >= srcs.size())
          throw std::runtime_error("msl: accumulator source out of range");
        const mx::array& src = srcs[static_cast<size_t>(s.idx)];
        const int64_t b2 = s.a * start_ + s.b;
        mx::Shape start(src.ndim(), 0), stop(src.shape());
        mx::array sl = src;
        if (s.a == 1) {
          start[0] = static_cast<mx::ShapeElem>(b2);
          stop[0] = static_cast<mx::ShapeElem>(b2 + trip_);
          sl = mx::slice(src, start, stop);
        } else {
          start[0] = static_cast<mx::ShapeElem>(b2 - trip_ + 1);
          stop[0] = static_cast<mx::ShapeElem>(b2 + 1);
          sl = mx::flip(mx::slice(src, start, stop), 0);
        }
        mx::Shape want{static_cast<mx::ShapeElem>(trip_)};
        for (auto d : s.shape) want.push_back(d);
        return mx::reshape(sl, want);
      }
      case 2: {
        mx::array a = stacked_of(s.kids[0], outs, ns, srcs);
        std::vector<int> axes;
        for (int d : s.dims) axes.push_back(d + 1);
        a = mx::sum(a, axes);
        if (!is_identity_perm(s.perm)) {
          std::vector<int> p{0};
          for (int d : s.perm) p.push_back(d + 1);
          a = mx::transpose(a, p);
        }
        mx::Shape want{static_cast<mx::ShapeElem>(trip_)};
        for (auto d : s.shape) want.push_back(d);
        return mx::reshape(a, want);
      }
      case 3: {
        mx::array A = stacked_of(s.kids[0], outs, ns, srcs);
        mx::array B = stacked_of(s.kids[1], outs, ns, srcs);
        // Hidden stacks may have been unit-squeezed at plan time; restore
        // the operand's exact rank (free -- all the dropped dims are 1).
        A = with_lead(A, s.lshape);
        B = with_lead(B, s.rshape);
        const int lrank = static_cast<int>(s.lshape.size());
        const int rrank = static_cast<int>(s.rshape.size());
        std::vector<int> zb_l{0}, zb_r{0}, c_l, c_r, f_l, f_r;
        for (int d : s.lb) zb_l.push_back(d + 1);
        for (int d : s.rb) zb_r.push_back(d + 1);
        for (int d : s.lc) c_l.push_back(d + 1);
        for (int d : s.rc) c_r.push_back(d + 1);
        auto has = [](const std::vector<int>& v, int x) {
          return std::find(v.begin(), v.end(), x) != v.end();
        };
        for (int i = 0; i <= lrank; i++)
          if (!has(zb_l, i) && !has(c_l, i)) f_l.push_back(i);
        for (int i = 0; i <= rrank; i++)
          if (!has(zb_r, i) && !has(c_r, i)) f_r.push_back(i);
        std::vector<int> pl(zb_l), pr(zb_r);
        pl.insert(pl.end(), f_l.begin(), f_l.end());
        pl.insert(pl.end(), c_l.begin(), c_l.end());
        pr.insert(pr.end(), c_r.begin(), c_r.end());
        pr.insert(pr.end(), f_r.begin(), f_r.end());
        mx::array At = mx::transpose(A, pl);
        mx::array Bt = mx::transpose(B, pr);
        const int64_t bsz = shape_prod(A.shape(), zb_l);
        const int64_t m = shape_prod(A.shape(), f_l);
        const int64_t kk = shape_prod(A.shape(), c_l);
        const int64_t n2 = shape_prod(B.shape(), f_r);
        mx::array out = mx::matmul(
            mx::reshape(At, mx::Shape{static_cast<mx::ShapeElem>(bsz),
                                      static_cast<mx::ShapeElem>(m),
                                      static_cast<mx::ShapeElem>(kk)}),
            mx::reshape(Bt, mx::Shape{static_cast<mx::ShapeElem>(bsz),
                                      static_cast<mx::ShapeElem>(kk),
                                      static_cast<mx::ShapeElem>(n2)}));
        mx::Shape want;
        for (int i : zb_l) want.push_back(A.shape()[static_cast<size_t>(i)]);
        for (int i : f_l) want.push_back(A.shape()[static_cast<size_t>(i)]);
        for (int i : f_r) want.push_back(B.shape()[static_cast<size_t>(i)]);
        mx::array arr = mx::reshape(out, want);
        if (!is_identity_perm(s.perm)) {
          std::vector<int> p{0};
          for (int d : s.perm) p.push_back(d + 1);
          arr = mx::transpose(arr, p);
        }
        return arr;
      }
      default:
        throw std::runtime_error("msl: bad accumulator node");
    }
  }

  // `x` reshaped to (L, *want) when its trailing dims are not already that.
  static mx::array with_lead(const mx::array& x, const mx::Shape& want) {
    if (x.ndim() == want.size() + 1) {
      bool same = true;
      for (size_t i = 0; i < want.size(); i++)
        same = same && x.shape()[i + 1] == want[i];
      if (same) return x;
    }
    mx::Shape s{x.shape()[0]};
    for (auto d : want) s.push_back(d);
    return mx::reshape(x, s);
  }

  std::string name_, source_, header_;
  std::vector<std::string> in_names_, out_names_;
  std::vector<mx::Shape> out_shapes_;
  std::vector<mx::Dtype> out_dtypes_;
  int64_t N_ = 0, tg_ = 1, trip_ = 0, start_ = 0;
  int nhidden_ = 0, ncarry_ = 0;
  std::vector<Norm> norms_;
  std::vector<int> unpacked_, state_pos_, stacked_pos_, passthrough_;
  std::vector<Pack> packs_;
  std::vector<std::pair<int, int64_t>> counters_;
  std::vector<std::pair<int, std::vector<AccNode>>> acc_;
  mx::fast::CustomKernelFunction kernel_;
  bool built_ = false;
  bool validated_ = false;
  bool dead_ = false;
  bool narrated_ = false;
};

// Plans this thread has traced into a compiled graph and never proven. The
// list lives across the whole of one top-level Program::run, which settles
// the call synchronously while it is non-empty; thread-local because a
// trace happens on the calling thread and two threads may run different
// executables at once.
thread_local std::vector<MslPlan*> t_msl_pending;

class Program {
 public:
  explicit Program(int num_slots, int num_args)
      : nslots_(num_slots), nargs_(num_args) {
    if (num_slots < 0 || num_args < 0 || num_args > num_slots)
      throw std::invalid_argument("tape: bad slot counts");
  }

  ~Program() {
    // MLX keeps a compiled graph per id for as long as nobody erases it.
    for (const auto& kv : compiled_) mx::detail::compile_erase(kv.second.id);
  }

  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  void add(int op, std::vector<int> ins, std::vector<int> outs,
           std::vector<int64_t> attrs, std::optional<mx::array> payload,
           std::vector<int> drops,
           std::vector<std::shared_ptr<Program>> regions, int64_t bytes,
           std::vector<double> fattrs, std::shared_ptr<MslPlan> msl,
           nb::object host) {
    for (int s : ins) check_slot(s);
    for (int s : outs) check_slot(s);
    for (int s : drops) check_slot(s);
    if ((op == kMslScan) != (msl != nullptr))
      throw std::invalid_argument("tape: msl plan without an msl entry");
    if ((op == kHostCall) != (host.is_valid() && !host.is_none()))
      throw std::invalid_argument("tape: host callable without a host entry");
    ops_.push_back(Entry{op, std::move(ins), std::move(outs),
                         std::move(attrs), std::move(fattrs),
                         std::move(payload), std::move(drops),
                         std::move(regions), bytes, std::move(msl),
                         std::move(host)});
  }

  // `copies` are output POSITIONS whose array may not be handed out as it
  // is: it could be one of the program's own constants (which the Program
  // holds for the life of the executable, so two calls would share a
  // buffer) or an input's array reaching an output through no-ops. Which
  // ones those are is a static property tape.py works out; making the copy
  // here is XLA's no-alias contract, the half object identity cannot
  // express across the language boundary.
  void set_outputs(std::vector<int> outs, std::vector<int> copies) {
    for (int s : outs) check_slot(s);
    outputs_ = std::move(outs);
    copies_ = std::move(copies);
  }

  // Whether this program's tape is traced through mx::compile, and which of
  // its outputs need anchoring when it is. BOTH decided in Python: the cost
  // and byte estimators that gate compilation live there (ops/control.py),
  // and re-deriving them here would be a second opinion nothing keeps in
  // step with the first.
  void set_compile(bool on, std::vector<int> anchors, int64_t max_repeat) {
    compile_ = on;
    anchors_ = std::move(anchors);
    max_repeat_ = max_repeat;
  }

  std::vector<mx::array> run(std::vector<mx::array> inputs) {
    if (static_cast<int>(inputs.size()) != nargs_)
      throw std::invalid_argument("tape: wrong number of inputs");
    std::vector<mx::array> outs;
    {
      // Nothing below touches Python: MLX builds a lazy graph and the
      // arrays are already C++ objects. The GIL comes back for the cast
      // of the results, in nanobind's own code (and briefly, deliberately,
      // in the recovery paths, which call the cycle collector).
      nb::gil_scoped_release nogil;
      // Everything below builds on the CALLING thread's default MLX
      // stream, which engine.execute has already pointed at a
      // cross-thread-evaluable stream of this thread's own
      // (engine.bind_thread) — it is the one entry point a native run can
      // be reached through, so a threadbare caller is bound before it gets
      // here.
      std::lock_guard<std::mutex> guard(lock_);
      t_msl_pending.clear();
      outs = run_recovering(inputs);
      settle_msl(inputs, outs);
      for (int i : copies_) {
        if (i >= 0 && static_cast<size_t>(i) < outs.size())
          outs[i] = fresh_copy(outs[i]);
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

  // One call of this program's tape: the compiled graph when it has one,
  // the op-by-op walk otherwise. `in_trace` says an enclosing mx::compile
  // is already tracing us, which forbids both a nested compile and any
  // sync point (there is nothing to evaluate: the values are tracers).
  std::vector<mx::array> call(const std::vector<mx::array>& inputs,
                              bool in_trace) {
    if (in_trace) return interpret(inputs, true);
    if (compile_ && !compile_disabled_) {
      g_stats.compiled_calls++;
      return compiled(1)(inputs);
    }
    return interpret(inputs, false);
  }

  // `repeat` applications of this program's tape as one compiled graph.
  // Bodies are compiled per repeat count (one chunked replay of K
  // iterations is a different graph from a single step), and each variant
  // gets a cache id of its own.
  const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
  compiled(int repeat) {
    auto it = compiled_.find(repeat);
    if (it != compiled_.end()) return it->second.fn;
    Compiled c;
    c.id = new_compile_id();
    Program* self = this;
    std::vector<int> anchors = anchors_;
    auto traced = [self, repeat, anchors](const std::vector<mx::array>& flat)
        -> std::vector<mx::array> {
      std::vector<mx::array> vals = self->interpret(flat, true);
      if (repeat > 1) {
        // A body's outputs are its next carries, and its captures ride
        // along unchanged: feed the outputs back in, keep the tail.
        for (int r = 1; r < repeat; r++) {
          std::vector<mx::array> next(vals);
          next.insert(next.end(), flat.begin() + vals.size(), flat.end());
          vals = self->interpret(next, true);
        }
      }
      anchor_outputs(vals, flat, anchors);
      return vals;
    };
    g_stats.compiles++;
    c.fn = mx::detail::compile(traced, c.id, false, {});
    auto ins = compiled_.emplace(repeat, std::move(c));
    return ins.first->second.fn;
  }

  bool may_compile(int repeat) const {
    return compile_ && !compile_disabled_ && repeat <= max_repeat_;
  }

  // A compiled path failed at CALL time. Every such failure is permanent
  // for this program (MLX will reject the same graph again), so drop the
  // compiled variants and never build another.
  void drop_compiled() {
    compile_disabled_ = true;
    for (const auto& kv : compiled_) mx::detail::compile_erase(kv.second.id);
    compiled_.clear();
    g_stats.compile_drops++;
  }

  bool compiled_dropped() const { return compile_disabled_; }
  bool no_chunk() const { return no_chunk_; }
  void set_no_chunk() { no_chunk_ = true; }

  size_t num_ops() const { return ops_.size(); }
  int num_slots() const { return nslots_; }
  int num_args() const { return nargs_; }

  // opcode -> how many entries carry it, THIS program and its regions.
  // For the tests: it is what says a recognizer's root really became its
  // fused op on the tape rather than the literal chain it replaces —
  // including inside a decode loop's body, which is its own Program.
  void tally(std::map<int, int64_t>& counts) const {
    for (const Entry& e : ops_) {
      counts[e.op]++;
      for (const auto& r : e.regions) r->tally(counts);
    }
  }

  nb::dict op_histogram() const {
    std::map<int, int64_t> counts;
    tally(counts);
    nb::dict d;
    for (const auto& kv : counts) d[nb::int_(kv.first)] = kv.second;
    return d;
  }

  // Does running this program read anything back to the HOST? Every
  // control-flow op does: a while reads its trip count or its condition, an
  // if/case its predicate. Structure, not policy -- read off the tape itself
  // so it cannot drift from what the tape actually holds.
  //
  // It is what says whether a loop body may be BUILT ahead of the condition
  // that decides whether it runs (run_while's pipelined path). Building a
  // graph is free and pure; a nested host read is neither -- it would
  // evaluate real work at a carry the loop may be about to abandon, which
  // for a nested dynamic while is not even guaranteed to terminate.
  bool reads_host() const {
    if (reads_host_ < 0) {
      reads_host_ = 0;
      for (const Entry& e : ops_) {
        if (e.op == kWhile || e.op == kIf || e.op == kCase) {
          reads_host_ = 1;
          break;
        }
        for (const auto& r : e.regions) {
          if (r->reads_host()) { reads_host_ = 1; break; }
        }
        if (reads_host_) break;
      }
    }
    return reads_host_ != 0;
  }

  // The op-by-op walk. `in_trace` is threaded rather than stored: the same
  // Program is walked both ways (a counted loop unrolls its body into an
  // enclosing trace and replays it eagerly on the next call), and a stored
  // flag would make that a race with itself.
  std::vector<mx::array> interpret(const std::vector<mx::array>& inputs,
                                   bool in_trace) {
    if (static_cast<int>(inputs.size()) != nargs_)
      throw std::invalid_argument("tape: wrong number of inputs");
    std::vector<std::optional<mx::array>> env(nslots_);
    for (size_t i = 0; i < inputs.size(); i++) env[i] = inputs[i];
    // interpreter.run_block's byte-denominated safety net: once this much
    // estimated result data has been produced with no sync point, settle
    // what is still live so the pending graph -- and the Metal buffers it
    // pins -- stay bounded. Never inside a trace: there is nothing to
    // evaluate there, and MLX manages the traced tape itself.
    const bool flushing = !in_trace && g_cfg.eager_flush_bytes > 0;
    int64_t acc = 0;
    for (const Entry& e : ops_) {
      step(e, env, in_trace);
      if (flushing) {
        acc += e.bytes;
        if (acc >= g_cfg.eager_flush_bytes) {
          acc = 0;
          eager_flush(env);
        }
      }
    }
    std::vector<mx::array> outs;
    outs.reserve(outputs_.size());
    for (int s : outputs_) {
      if (!env[s]) throw std::runtime_error("tape: output slot is empty");
      outs.push_back(*env[s]);
    }
    return outs;
  }

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

  void step(const Entry& e, std::vector<std::optional<mx::array>>& env,
            bool in_trace) const {
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

      case kBitcastConvert: {
        // ops/shape.py _bitcast_convert, byte-multiple arm only: MLX's
        // storage IS the XLA layout there, so the whole op is a view. The
        // 4-bit forms cannot reach this handler -- i4/ui4 are not in the
        // dtype table, and every other emulated type is declined for the
        // same reason (their bit patterns do not exist on the device).
        mx::Dtype dt = dtype_of(at[0]);
        if (at[1] == 0) {
          env[e.outs[0]] = mx::view(in(0), dt);
        } else if (at[1] == 1) {
          // Narrowing: the result gains a trailing dim of the size ratio,
          // and mx::view rescales the LAST axis -- so split a fresh unit
          // axis (which also makes a rank-0 input legal).
          env[e.outs[0]] = mx::view(mx::expand_dims(in(0), -1), dt);
        } else {
          // Widening: the input's trailing ratio-sized dim collapses.
          env[e.outs[0]] = mx::squeeze(mx::view(in(0), dt), -1);
        }
        break;
      }

      case kDynamicSlice:
      case kDynamicUpdateSlice: {
        // ops/shape.py _dynamic_slice / _dynamic_update_slice. XLA clamps
        // the start indices so the window stays inside the operand; the
        // clamp bounds are shape arithmetic, resolved at lowering.
        const bool update = e.op == kDynamicUpdateSlice;
        const size_t first = update ? 2 : 1;
        int64_t rank = at[0];
        std::vector<mx::array> parts;
        parts.reserve(static_cast<size_t>(rank));
        for (int64_t i = 0; i < rank; i++)
          parts.push_back(
              mx::reshape(mx::astype(in(first + static_cast<size_t>(i)),
                                     mx::int32),
                          mx::Shape{}));
        mx::Shape bounds = shape(at, 1, rank);
        std::vector<int> ax(static_cast<size_t>(rank));
        for (int64_t i = 0; i < rank; i++) ax[i] = static_cast<int>(i);
        mx::array starts =
            mx::clip(mx::stack(parts), mx::array(0, mx::int32),
                     mx::array(bounds.begin(),
                               mx::Shape{static_cast<int>(rank)}, mx::int32));
        if (update) {
          env[e.outs[0]] = mx::slice_update(in(0), in(1), starts, ax);
        } else {
          env[e.outs[0]] =
              mx::slice(in(0), starts, ax, shape(at, 1 + rank, rank));
        }
        break;
      }

      case kSort: {
        // ops/sort.py _sort -> _gather_sorted, the arm whose comparator
        // compares an operand pair DIRECTLY (no key chain to evaluate).
        // That is the shape every top_k lowers to — jax's chlo.top_k
        // decomposition is `sort(values, iota)` under a strict TOTALORDER
        // GT — and it is what a plain jnp.sort/argsort emits too. A
        // comparator that computes a key first is declined in tape.py,
        // where the structural recognizer lives.
        // attrs [dim, descending?, key operand, key kind]
        //   key kind: 0 integer (already ordered), 1 float totalOrder,
        //             2 bool (widened to uint8, as the Python handler does)
        int dim = static_cast<int>(at[0]);
        bool descending = at[1] != 0;
        mx::array key = in(static_cast<size_t>(at[2]));
        if (at[3] == 1) {
          key = total_order_key(key);
        } else if (at[3] == 2) {
          key = mx::astype(key, mx::uint8);
        }
        // Bitwise NOT reverses the order for signed and unsigned alike.
        if (descending) key = mx::bitwise_invert(key);
        // mx::argsort reads the WRONG elements from a non-contiguous input
        // (MLX 0.32), and a non-last-axis sort arrives as a strided view.
        mx::array idx = mx::argsort(mx::contiguous(key), dim);
        for (size_t i = 0; i < e.outs.size(); i++)
          env[e.outs[i]] = mx::take_along_axis(in(i), idx, dim);
        break;
      }

      case kTopK: {
        // ops/sort.py _top_k: a stable DESCENDING argsort of the last axis,
        // cut to k. attrs [k, key kind] (kinds as kSort's).
        const mx::array& x = in(0);
        mx::array key = x;
        if (at[1] == 1) {
          key = total_order_key(key);
        } else if (at[1] == 2) {
          key = mx::astype(key, mx::uint8);
        }
        mx::array idx =
            mx::argsort(mx::contiguous(mx::bitwise_invert(key)), -1);
        mx::Shape lo(idx.shape().size(), 0), hi = idx.shape();
        hi.back() = static_cast<mx::ShapeElem>(at[0]);
        idx = mx::slice(idx, lo, hi);
        env[e.outs[0]] = mx::take_along_axis(x, idx, -1);
        env[e.outs[1]] = mx::astype(idx, mx::int32);
        break;
      }

      // --- recognizer emits (M4) -------------------------------------
      //
      // Each is a transliteration of the `emit` in the Python module that
      // owns the recognizer; the analysis, the packing and every static
      // decision stay there, and what arrives here is the plan. The
      // differential tests compare output BYTES against those very
      // functions, so a drift is a failure and not a tolerance question.

      case kQmm: {
        // metaljax.qmm.emit. attrs:
        //   [transpose?, lrank, lperm..., batched?, B, M, K,
        //    group_size, bits, mode, perm?, swapped?, out_dtype,
        //    bshape, mshape, nshape]     (each shape: rank, dims...)
        // ins: [activation, w, scales, (biases if affine), (perm if any)]
        Cursor c(at);
        bool do_transpose = c.flag();
        std::vector<int> lperm = c.vec();
        bool batched = c.flag();
        int64_t B = c.next(), M = c.next(), K = c.next();
        int64_t gs = c.next(), bits = c.next(), mode = c.next();
        bool has_perm = c.flag();
        bool swapped = c.flag();
        mx::Dtype out_dt = dtype_of(c.next());
        mx::Shape bshape = c.shp(), mshape = c.shp(), nshape = c.shp();

        mx::array x = in(0);
        if (do_transpose) x = mx::transpose(x, lperm);
        x = mx::reshape(x, batched
                               ? mx::Shape{static_cast<mx::ShapeElem>(B),
                                           static_cast<mx::ShapeElem>(M),
                                           static_cast<mx::ShapeElem>(K)}
                               : mx::Shape{static_cast<mx::ShapeElem>(M),
                                           static_cast<mx::ShapeElem>(K)});
        if (!is_float(x.dtype())) x = mx::astype(x, out_dt);
        if (has_perm)
          x = mx::take(x, in(e.ins.size() - 1),
                       static_cast<int>(x.ndim()) - 1);
        std::optional<mx::array> biases;
        if (mode == 0) biases = in(3);
        mx::array y = mx::quantized_matmul(
            x, in(1), in(2), biases, /*transpose=*/true,
            static_cast<int>(gs), static_cast<int>(bits), qmode(mode));
        mx::Shape out;
        for (auto d : bshape) out.push_back(d);
        if (swapped) {
          // The dot had the weight on its LHS, so the result is [..., N, M].
          y = mx::swapaxes(y, static_cast<int>(y.ndim()) - 1,
                           static_cast<int>(y.ndim()) - 2);
          for (auto d : nshape) out.push_back(d);
          for (auto d : mshape) out.push_back(d);
        } else {
          for (auto d : mshape) out.push_back(d);
          for (auto d : nshape) out.push_back(d);
        }
        y = mx::reshape(y, out);
        if (y.dtype() != out_dt) y = mx::astype(y, out_dt);
        env[e.outs[0]] = y;
        break;
      }

      case kSdpaMask: {
        // metaljax.sdpa._mask_array, the cache MISS body. The cache itself
        // is this entry: tape.py emits one per distinct mask key, at the
        // first attention that reads it, and every later root reads the
        // slot — which is what `env`'s tuple-keyed entry does in Python.
        // attrs [kind (0 select, 1 additive), dtype, scaled?];
        // fattrs [const * mul, mul].
        int64_t kind = at[0];
        mx::Dtype dt = dtype_of(at[1]);
        mx::array mask = in(0);
        if (kind == 0) {
          // Additive, never boolean: MLX's boolean mask gives a
          // fully-masked row a finite value where the literal chain gives
          // NaN. `L + C == C` for a sentinel C, so this is exact.
          mask = mx::where(mask, mx::array(0.0, dt),
                           mx::array(e.fattrs[0], dt));
        } else {
          if (mask.dtype() != dt) mask = mx::astype(mask, dt);
          if (at[2]) mask = mx::multiply(mask, mx::array(e.fattrs[1],
                                                         mask.dtype()));
        }
        env[e.outs[0]] = mask;
        break;
      }

      case kSdpa: {
        // metaljax.sdpa.emit. attrs [q_rec, k_rec, v_rec, dtype, out_dtype,
        //   mask?, mask_rec, out_pre, out_perm, out_post]; fattrs [scale].
        // ins: [q, k, v, (mask)].
        Cursor c(at);
        Rec q_rec = read_rec(c), k_rec = read_rec(c), v_rec = read_rec(c);
        mx::Dtype dt = dtype_of(c.next());
        mx::Dtype out_dt = dtype_of(c.next());
        bool has_mask = c.flag();
        Rec mask_rec = read_rec(c);
        bool has_pre = c.flag();
        mx::Shape pre = c.shp();
        bool has_perm = c.flag();
        std::vector<int> perm = c.vec();
        bool has_post = c.flag();
        mx::Shape post = c.shp();

        mx::array q = apply_rec(q_rec, in(0));
        mx::array k = apply_rec(k_rec, in(1));
        mx::array v = apply_rec(v_rec, in(2));
        if (q.dtype() != dt) q = mx::astype(q, dt);
        if (k.dtype() != dt) k = mx::astype(k, dt);
        if (v.dtype() != dt) v = mx::astype(v, dt);
        std::optional<mx::array> mask;
        if (has_mask) mask = apply_rec(mask_rec, in(3));
        mx::array out = mx::fast::scaled_dot_product_attention(
            q, k, v, static_cast<float>(e.fattrs[0]), /*mask_mode=*/"",
            mask);
        if (has_pre) out = mx::reshape(out, pre);
        if (has_perm) out = mx::transpose(out, perm);
        if (has_post) out = mx::reshape(out, post);
        if (out.dtype() != out_dt) out = mx::astype(out, out_dt);
        env[e.outs[0]] = out;
        break;
      }

      // --- the MoE expert gather (metaljax.moe.emit) -------------------
      //
      // `emit` is a small interpreter over a plan of pair-space nodes, so
      // the lowering spreads that plan over the tape: one entry per node,
      // in the plan's own dependency order, with the root's result written
      // by the tail. Liveness then prunes the pair-space intermediates
      // exactly as the tape prunes anything else, and only the tail is
      // charged the root's bytes — so the eager flush cadence sees the one
      // sync-point-moving number the Python engine sees.

      case kMoeEIdx:
        // ctx.eidx = reshape(indices, (P,)).astype(uint32)
        env[e.outs[0]] = mx::astype(
            mx::reshape(in(0), mx::Shape{static_cast<mx::ShapeElem>(at[0])}),
            mx::uint32);
        break;

      case kMoeTIdx:
        // ctx.tidx = repeat(arange(T, uint32), K)
        env[e.outs[0]] = mx::repeat(
            mx::arange(static_cast<double>(at[0]), mx::uint32),
            static_cast<int>(at[1]));
        break;

      case kMoeGather: {
        // moe._emit_ext, the `env` arm. attrs [ea?, ea, ta?, ta, P];
        // ins [value, eidx, tidx].
        bool has_ea = at[0] != 0;
        int ea = static_cast<int>(at[1]);
        bool has_ta = at[2] != 0;
        int ta = static_cast<int>(at[3]);
        const mx::array& x0 = in(0);
        if (!has_ea && !has_ta) {
          env[e.outs[0]] = x0;
          break;
        }
        if (has_ea && has_ta) {
          mx::array x = mx::moveaxis(x0, ea, 0);
          x = mx::moveaxis(x, ta < ea ? ta + 1 : ta, 1);
          // x[eidx, tidx]: one element of the (expert, token) grid per
          // pair, the whole trailing slice.
          mx::Shape sizes = x.shape();
          sizes[0] = 1;
          sizes[1] = 1;
          mx::array g = mx::gather(x, {in(1), in(2)}, {0, 1}, sizes);
          mx::Shape out{static_cast<mx::ShapeElem>(at[4])};
          for (size_t i = 2; i < x.shape().size(); i++)
            out.push_back(x.shape()[i]);
          env[e.outs[0]] = mx::reshape(g, out);
          break;
        }
        int axis = has_ea ? ea : ta;
        const mx::array& idx = has_ea ? in(1) : in(2);
        env[e.outs[0]] = mx::moveaxis(mx::take(x0, idx, axis), axis, 0);
        break;
      }

      case kMoeConcat: {
        // moe._emit_elem's concatenate arm: pair leads may be P or 1.
        int axis = static_cast<int>(at[0]);
        int lead = 0;
        for (size_t i = 0; i < e.ins.size(); i++)
          lead = std::max(lead, in(i).shape()[0]);
        std::vector<mx::array> parts;
        parts.reserve(e.ins.size());
        for (size_t i = 0; i < e.ins.size(); i++) {
          mx::array x = in(i);
          if (x.shape()[0] != lead) {
            mx::Shape s = x.shape();
            s[0] = lead;
            x = mx::broadcast_to(x, s);
          }
          parts.push_back(x);
        }
        env[e.outs[0]] = mx::concatenate(std::move(parts), axis);
        break;
      }

      case kMoeView: {
        // moe._emit_view. attrs [src paired?, kind, ...]:
        //   0 bcast    [nkeep, (kept?, dest)..., trailing]
        //   1 perm     [order]
        //   2 reshape  [trailing]
        //   3 slice    [n, (start, stop, stride)...]
        //   4 dotperm  [transpose?, perm, trailing]
        Cursor c(at);
        mx::array x = pair_lead(in(0), c.flag());
        int64_t kind = c.next();
        if (kind == 0) {
          int64_t nkeep = c.next();
          std::vector<int64_t> kept(static_cast<size_t>(nkeep));
          std::vector<int64_t> dest(static_cast<size_t>(nkeep));
          for (int64_t j = 0; j < nkeep; j++) {
            kept[j] = c.next();
            dest[j] = c.next();
          }
          mx::Shape trailing = c.shp();
          mx::Shape view(trailing.size(), 1);
          mx::Shape lo(x.shape().size(), 0), hi = x.shape();
          std::vector<int> squeeze_axes;
          for (int64_t j = 0; j < nkeep; j++) {
            if (!kept[j]) {
              hi[static_cast<size_t>(1 + j)] = 1;
              squeeze_axes.push_back(static_cast<int>(1 + j));
            } else {
              view[static_cast<size_t>(dest[j])] = x.shape()[1 + j];
            }
          }
          if (!squeeze_axes.empty())
            x = mx::squeeze(mx::slice(x, lo, hi), squeeze_axes);
          mx::Shape mid{x.shape()[0]};
          for (auto d : view) mid.push_back(d);
          mx::Shape out{x.shape()[0]};
          for (auto d : trailing) out.push_back(d);
          env[e.outs[0]] = mx::broadcast_to(mx::reshape(x, mid), out);
        } else if (kind == 1) {
          std::vector<int> order = c.vec();
          std::vector<int> perm{0};
          for (int i : order) perm.push_back(1 + i);
          env[e.outs[0]] = mx::transpose(x, perm);
        } else if (kind == 2) {
          mx::Shape trailing = c.shp();
          mx::Shape out{x.shape()[0]};
          for (auto d : trailing) out.push_back(d);
          env[e.outs[0]] = mx::reshape(x, out);
        } else if (kind == 3) {
          int64_t n = c.next();
          mx::Shape lo{0}, hi{x.shape()[0]}, st{1};
          for (int64_t j = 0; j < n; j++) {
            lo.push_back(static_cast<mx::ShapeElem>(c.next()));
            hi.push_back(static_cast<mx::ShapeElem>(c.next()));
            st.push_back(static_cast<mx::ShapeElem>(c.next()));
          }
          env[e.outs[0]] = mx::slice(x, lo, hi, st);
        } else if (kind == 4) {
          bool do_transpose = c.flag();
          std::vector<int> order = c.vec();
          mx::Shape trailing = c.shp();
          if (do_transpose) {
            std::vector<int> perm{0};
            for (int i : order) perm.push_back(1 + i);
            x = mx::transpose(x, perm);
          }
          mx::Shape out{x.shape()[0]};
          for (auto d : trailing) out.push_back(d);
          env[e.outs[0]] = mx::reshape(x, out);
        } else {
          throw std::invalid_argument("tape: unknown moe view");
        }
        break;
      }

      case kMoeDot: {
        // moe._emit_dot. attrs [shortcut?, src paired?, ta, packed?, E,
        //   n_first?, group_size, bits, mode, perm?, P, M, K, N, out_dtype,
        //   mshape, nshape]; ins [data, eidx, (tidx if shortcut),
        //   weight | w, scales, (biases if affine), (perm if any)].
        Cursor c(at);
        bool shortcut = c.flag();
        bool src_paired = c.flag();
        int ta = static_cast<int>(c.next());
        bool packed = c.flag();
        int64_t E = c.next();
        bool n_first = c.flag();
        int64_t gs = c.next(), bits = c.next(), mode = c.next();
        bool has_perm = c.flag();
        int64_t P = c.next(), M = c.next(), K = c.next(), N = c.next();
        mx::Dtype out_dt = dtype_of(c.next());
        mx::Shape mshape = c.shp(), nshape = c.shp();

        mx::array x = in(0);
        std::optional<mx::array> lhs_idx;
        if (shortcut) {
          // Token-indexed and nothing else: hand gather_* the ungathered
          // rows and let it do the indexing, instead of materializing K
          // copies.
          std::vector<int> perm{ta};
          for (size_t i = 0; i < x.shape().size(); i++)
            if (static_cast<int>(i) != ta) perm.push_back(static_cast<int>(i));
          int rows = x.shape()[ta];
          x = mx::reshape(mx::transpose(x, perm),
                          mx::Shape{rows, static_cast<mx::ShapeElem>(M),
                                    static_cast<mx::ShapeElem>(K)});
          lhs_idx = in(2);
        } else {
          x = pair_lead(x, src_paired);
          if (x.shape()[0] != static_cast<int>(P)) {
            mx::Shape s = x.shape();
            s[0] = static_cast<mx::ShapeElem>(P);
            x = mx::broadcast_to(x, s);
          }
          x = mx::reshape(x, mx::Shape{static_cast<mx::ShapeElem>(P),
                                       static_cast<mx::ShapeElem>(M),
                                       static_cast<mx::ShapeElem>(K)});
          lhs_idx = mx::arange(static_cast<double>(P), mx::uint32);
        }
        std::optional<mx::array> rhs_idx = in(1);
        const size_t wbase = shortcut ? 3 : 2;

        mx::array y = x;
        if (packed) {
          mx::array w = split_experts(in(wbase), E);
          mx::array scales = split_experts(in(wbase + 1), E);
          std::optional<mx::array> biases;
          if (mode == 0) biases = split_experts(in(wbase + 2), E);
          if (has_perm)
            x = mx::take(x, in(e.ins.size() - 1),
                         static_cast<int>(x.ndim()) - 1);
          if (!is_float(x.dtype())) x = mx::astype(x, out_dt);
          y = mx::gather_qmm(x, w, scales, biases, lhs_idx, rhs_idx,
                             /*transpose=*/true, static_cast<int>(gs),
                             static_cast<int>(bits), qmode(mode));
        } else {
          const mx::array& wv = in(wbase);
          mx::array w3 =
              n_first
                  // gather_mm computes `a @ b`, so an [E, N, K] weight has
                  // to be transposed — as a LAZY view: materializing it
                  // would copy the whole expert set.
                  ? mx::swapaxes(
                        mx::reshape(wv,
                                    mx::Shape{static_cast<mx::ShapeElem>(E),
                                              static_cast<mx::ShapeElem>(N),
                                              static_cast<mx::ShapeElem>(K)}),
                        -1, -2)
                  : mx::reshape(wv,
                                mx::Shape{static_cast<mx::ShapeElem>(E),
                                          static_cast<mx::ShapeElem>(K),
                                          static_cast<mx::ShapeElem>(N)});
          y = mx::gather_mm(x, w3, lhs_idx, rhs_idx);
        }
        y = mx::reshape(y, mx::Shape{static_cast<mx::ShapeElem>(P),
                                     static_cast<mx::ShapeElem>(M),
                                     static_cast<mx::ShapeElem>(N)});
        if (y.dtype() != out_dt) y = mx::astype(y, out_dt);
        mx::Shape out{static_cast<mx::ShapeElem>(P)};
        for (auto d : mshape) out.push_back(d);
        for (auto d : nshape) out.push_back(d);
        env[e.outs[0]] = mx::reshape(y, out);
        break;
      }

      case kMoeTail: {
        // moe.emit's tail: weight the pairs, sum over K, put the token axis
        // back. attrs [out paired?, P, T, K, sum_dtype, out_dtype,
        // out_axis, out_shape]; ins [plan output, router weights].
        Cursor c(at);
        mx::array y = pair_lead(in(0), c.flag());
        int64_t P = c.next(), T = c.next(), Kk = c.next();
        mx::Dtype sum_dt = dtype_of(c.next());
        mx::Dtype out_dt = dtype_of(c.next());
        int out_axis = static_cast<int>(c.next());
        mx::Shape want = c.shp();

        mx::Shape trailing(y.shape().begin() + 1, y.shape().end());
        if (y.shape()[0] != static_cast<int>(P)) {
          mx::Shape s{static_cast<mx::ShapeElem>(P)};
          for (auto d : trailing) s.push_back(d);
          y = mx::broadcast_to(y, s);
        }
        mx::array w = mx::astype(
            mx::reshape(in(1), mx::Shape{static_cast<mx::ShapeElem>(P)}),
            y.dtype());
        mx::Shape wshape{static_cast<mx::ShapeElem>(P)};
        for (size_t i = 0; i < trailing.size(); i++) wshape.push_back(1);
        y = mx::multiply(y, mx::reshape(w, wshape));
        // The dense graph converts the products before summing them.
        if (y.dtype() != sum_dt) y = mx::astype(y, sum_dt);
        mx::Shape split{static_cast<mx::ShapeElem>(T),
                        static_cast<mx::ShapeElem>(Kk)};
        for (auto d : trailing) split.push_back(d);
        y = mx::sum(mx::reshape(y, split), std::vector<int>{1});
        if (out_axis) y = mx::moveaxis(y, 0, out_axis);
        if (y.shape() != want)
          throw std::runtime_error("tape: moe produced the wrong shape");
        if (y.dtype() != out_dt) y = mx::astype(y, out_dt);
        env[e.outs[0]] = y;
        break;
      }

      // --- control flow (ops/control.py) -----------------------------
      case kWhile:
        run_while(e, env, in_trace);
        break;

      case kMslScan:
        run_msl(e, env, in_trace);
        break;

      // ops/callbacks.py `_token`: an ordered-effect token carries no data
      // and exists only to be somewhere in the data flow. create_token
      // makes one; after_all joins several into one, which is the same
      // empty array.
      case kToken:
        env[e.outs[0]] = mx::zeros(mx::Shape{0}, mx::bool_);
        break;

      // --- host ops (M5b) --------------------------------------------
      case kHostCall: {
        // The one place a native run reacquires the GIL. Never inside a
        // trace: a block holding one of these is impure in the Python
        // analysis (interpreter._IMPURE_OPS, custom_call_host_hook), so
        // neither engine ever compiles it -- and a host handler reading
        // tracers would compute on nothing at all.
        if (in_trace)
          throw std::runtime_error("tape: a host call cannot run in a trace");
        std::vector<mx::array> args;
        args.reserve(e.ins.size());
        for (size_t i = 0; i < e.ins.size(); i++) args.push_back(in(i));
        std::vector<mx::array> res;
        {
          nb::gil_scoped_acquire gil;
          try {
            nb::object out = e.host(nb::cast(args));
            res = nb::cast<std::vector<mx::array>>(out);
          } catch (nb::python_error& err) {
            // A Python exception must not travel up a stack that has
            // released the GIL: the recovery paths above call what() on
            // whatever they catch, and nanobind formats a python_error's
            // message through the Python C API. Say it here, where the GIL
            // is held, and carry a plain string the rest of the way.
            throw std::runtime_error(std::string("tape: host call raised: ") +
                                     err.what());
          }
        }
        g_stats.host_calls++;
        if (res.size() != e.outs.size())
          throw std::runtime_error("tape: host call result count mismatch");
        for (size_t i = 0; i < res.size(); i++) env[e.outs[i]] = res[i];
        break;
      }

      case kIf:
      case kCase: {
        // _if / _case: the branch is chosen on the HOST, so both make a
        // block impure in the Python analysis and no program containing
        // one is ever compiled -- which is why reading the predicate here
        // cannot be a sync point inside a trace.
        int64_t which;
        if (e.op == kIf) {
          which = item_bool(in(0)) ? 0 : 1;
        } else {
          which = item_int(in(0));
          which = std::min<int64_t>(
              std::max<int64_t>(which, 0),
              static_cast<int64_t>(e.regions.size()) - 1);
        }
        size_t base = 1;  // ins[0] is the predicate/index
        for (int64_t r = 0; r < which; r++)
          base += static_cast<size_t>(at[static_cast<size_t>(r)]);
        std::vector<mx::array> args;
        int64_t ncaps = at[static_cast<size_t>(which)];
        args.reserve(static_cast<size_t>(ncaps));
        for (int64_t i = 0; i < ncaps; i++) args.push_back(in(base + i));
        std::vector<mx::array> outs =
            e.regions[static_cast<size_t>(which)]->call(args, in_trace);
        if (outs.size() != e.outs.size())
          throw std::runtime_error("tape: branch result count mismatch");
        for (size_t i = 0; i < outs.size(); i++) env[e.outs[i]] = outs[i];
        break;
      }

      default:
        throw std::invalid_argument("tape: unknown opcode");
    }

    for (int s : e.drops) env[s].reset();
  }

  // A counted loop msl_scan planned into one generated kernel. The entry is
  // a kWhile in every other respect -- same attrs, same carries, the same
  // cond/body sub-programs when they lowered -- with the kernel's launch
  // recipe and the arrays it reads appended. So a plan that dies falls back
  // to the loop it was standing in for, in this very call: the carries are
  // untouched (the kernel writes only its own outputs), and the Python
  // engine does exactly this when `try_run` raises.
  void run_msl(const Entry& e, std::vector<std::optional<mx::array>>& env,
               bool in_trace) const {
    auto in = [&](size_t i) -> const mx::array& {
      const auto& v = env[e.ins[i]];
      if (!v) throw std::runtime_error("tape: read of a dropped slot");
      return *v;
    };
    MslPlan* plan = e.msl.get();
    if (plan != nullptr && !plan->dead()) {
      const int64_t ncarry = e.attrs[0];
      const size_t base = e.ins.size() - plan->num_sources();
      std::vector<mx::array> carries, srcs;
      for (int64_t i = 0; i < ncarry; i++) carries.push_back(in(i));
      for (size_t i = base; i < e.ins.size(); i++) srcs.push_back(in(i));
      std::exception_ptr err;
      try {
        std::vector<mx::array> vals =
            plan->run(carries, srcs, in_trace, t_msl_pending);
        write_results(e, env, vals);
        return;
      } catch (const std::exception&) {
        err = std::current_exception();
      }
      // Outside the handler, holding none of the failed launch's arrays.
      // msl_scan.try_run's policy: whatever went wrong, this kernel is not
      // going to work, so the loop stops asking for it.
      plan->kill();
      g_stats.msl_failures++;
      if (e.regions.size() < 2) {
        // Nothing to fall back to (the body is outside the native op set):
        // the caller retires the tape and the Python engine runs the
        // program, kernel recovery and all.
        std::rethrow_exception(err);
      }
      debug_print("msl_scan kernel failed; running the loop instead");
    }
    run_while(e, env, in_trace);
  }

  // ops/control.py _while, transliterated. Every branch here has a comment
  // in that file explaining what it is for; the policy numbers (cost,
  // cadence, chunk size, which bodies may be compiled) are computed by the
  // same Python estimators and arrive in `attrs`.
  void run_while(const Entry& e, std::vector<std::optional<mx::array>>& env,
                 bool in_trace) const {
    auto in = [&](size_t i) -> const mx::array& {
      const auto& v = env[e.ins[i]];
      if (!v) throw std::runtime_error("tape: read of a dropped slot");
      return *v;
    };
    const std::vector<int64_t>& at = e.attrs;
    const int64_t ncarry = at[0], ncond_caps = at[1], nbody_caps = at[2];
    const bool counted = at[3] != 0;
    const int64_t k = at[4], bound_kind = at[5], bound = at[6];
    const int64_t cost = std::max<int64_t>(at[7], 1);
    const int64_t period = std::max<int64_t>(at[8], 1);
    const bool chunkable = at[9] != 0;
    const int64_t kmax = at[10];

    Program* cond = e.regions[0].get();
    Program* body = e.regions[1].get();

    std::vector<mx::array> ins, cond_caps, body_caps;
    ins.reserve(static_cast<size_t>(ncarry));
    for (int64_t i = 0; i < ncarry; i++) ins.push_back(in(i));
    for (int64_t i = 0; i < ncond_caps; i++)
      cond_caps.push_back(in(static_cast<size_t>(ncarry + i)));
    for (int64_t i = 0; i < nbody_caps; i++)
      body_caps.push_back(in(static_cast<size_t>(ncarry + ncond_caps + i)));

    std::vector<mx::array> vals;
    if (counted) {
      int64_t n;
      if (bound_kind == 0) {
        n = bound;
      } else if (bound_kind == 1) {
        n = item_int(ins[static_cast<size_t>(bound)]);
      } else {
        n = item_int(cond_caps[static_cast<size_t>(bound)]);
      }
      const int64_t start = item_int(ins[static_cast<size_t>(k)]);
      const int64_t trip = std::max<int64_t>(n - start, 0);
      if (in_trace) {
        // An enclosing mx::compile is tracing us: inline the iterations
        // into that graph. Past 64 of them the trace holds more
        // intermediates than Metal's buffer budget allows, and the answer
        // is the same as the Python engine's -- abort, and let the caller
        // fall back to the eager path (run_recovering does that here).
        if (trip > 64)
          throw std::runtime_error(
              "metaljax: refusing to unroll trip=" + std::to_string(trip) +
              " inside a trace");
        g_stats.unrolls++;
        vals = ins;
        for (int64_t i = 0; i < trip; i++)
          vals = run_body(body, vals, body_caps, true);
        write_results(e, env, vals);
        return;
      }
      // Eager loop. Chained replays are expensive (a compiled call
      // evaluates its inputs), so unroll as many iterations as the trace
      // budget allows into each compiled chunk and replay trip/K chunks
      // instead of trip single steps -- while flushing often enough that
      // the buffers a pending replay pins stay bounded.
      int64_t K = 1;
      if (chunkable && !body->no_chunk())
        K = std::max<int64_t>(1, std::min<int64_t>(trip, kmax));
      if (K > 1) {
        bool failed = false;
        try {
          vals = run_chunked(body, ins, body_caps, trip, K, cost);
        } catch (const std::exception& ex) {
          // MLX's compiler can reject big fused traces ("Too many
          // inputs/outputs fused..."). Fall back to single-step replays,
          // from the ORIGINAL carries -- a failed chunk changed nothing.
          debug_print(std::string("chunked loop failed (") + ex.what() +
                      "); falling back to single-step");
          failed = true;
        }
        if (!failed) {
          write_results(e, env, vals);
          return;
        }
        body->set_no_chunk();
        g_stats.chunk_drops++;
      }
      BodyRunner runner(body, body_caps, 1);
      vals = ins;
      for (int64_t i = 1; i <= trip; i++) {
        vals = runner.run_one(vals);
        if (i % period == 0) loop_flush(vals, period * cost);
      }
      write_results(e, env, vals);
      return;
    }

    // Dynamic (non-counted) loop: evaluate the condition each iteration.
    // The BODY still gets compiled -- a data-dependent trip count says
    // nothing about the body, and interpreting it op by op is what made
    // LLM decode Python-dispatch-bound. The cond stays eager: it ends in
    // a host read.
    if (in_trace)
      throw std::runtime_error(
          "metaljax: a dynamic while cannot run inside a trace");
    BodyRunner runner(body, body_caps, 1);
    vals = ins;
    // The condition of ONE carry, as a lazy array. `cargs` is scoped to die
    // here on purpose: it is a second handle on every carry, and a handle
    // still alive when the next iteration's update EVALUATES is the
    // difference between MLX writing a KV cache in place and copying the
    // whole thing (mx::array::is_donatable is a use_count test). Holding it
    // across the body cost 4.6 us per megabyte of cache per token.
    auto cond_of = [&](const std::vector<mx::array>& v) {
      std::vector<mx::array> cargs(v);
      cargs.insert(cargs.end(), cond_caps.begin(), cond_caps.end());
      std::vector<mx::array> pred = cond->interpret(cargs, false);
      if (pred.size() != 1)
        throw std::runtime_error("tape: while cond must return one value");
      return pred[0];
    };

    // Can the body be BUILT before its condition is known? Building an MLX
    // graph is pure and lazy, so a body built for an iteration that turns
    // out not to run is simply dropped -- unless the body reads something
    // back to the host, which would make "building" it mean RUNNING it.
    //
    // AND is it worth it? Building costs ~per-op Python-free but real
    // work; the saved host round trip is ~150 us. On a 4-matmul synthetic
    // body the build is noise and pipelining won 1.9x; on a whole-model
    // decode body (~2000 entries) the speculative build costs ~4.5 ms/tok
    // and LOSES (row 5 measured 65.1 vs 60.6 with pipeline off). Gate on
    // tape size: above ~256 entries the round trip is the cheaper side.
    // g_cfg.while_pipeline doubles as the threshold when > 1.
    const int64_t max_entries =
        g_cfg.while_pipeline > 1 ? g_cfg.while_pipeline : 256;
    const bool pipeline =
        g_cfg.while_pipeline > 0 &&
        static_cast<int64_t>(body->num_ops()) <= max_entries &&
        !body->reads_host() && !cond->reads_host();
    if (!pipeline) {
      g_stats.serial_loops++;
      for (;;) {
        mx::array pred = cond_of(vals);
        if (!item_bool(pred)) break;
        vals = runner.run_one(vals);
        // Flush the carry, not nothing: the cond only forces the values it
        // reads, so anything else in the carry (a KV cache) would pile up as
        // unevaluated graph across iterations.
        loop_flush(vals, cost);
      }
      write_results(e, env, vals);
      return;
    }

    // Pipelined dynamic loop. Two host round trips per iteration is what
    // made LLM decode stall (M4's verdict): the old shape submitted the
    // condition and waited, then submitted the body and waited, so the GPU
    // was idle for both decisions. Here the body and the NEXT condition are
    // built and submitted before the current condition is read back, so by
    // the time the host wakes up the device is already a token ahead.
    //
    // The order of what happens once the condition says "keep going" is
    // load-bearing:
    //   1. drop this iteration's carry, so the update ops in the body it
    //      feeds can donate their buffers (see cond_of);
    //   2. THEN submit -- the WHOLE carry, since the condition only forces
    //      what it reads and a KV cache would otherwise pile up as
    //      unevaluated graph;
    //   3. charge the iteration against the op-unit budget, exactly as the
    //      serial path's loop_flush does.
    // Speculation never touches the carry a loop may return: the body of
    // iteration t is only ever SUBMITTED once t's condition has said true.
    // The one place it is EVALUATED earlier is BodyRunner's probe after a
    // compiled body has been rebound mid-loop -- and that is still safe,
    // because `vals` is held across the build, which is precisely what
    // stops MLX donating (and therefore mutating) anything in it.
    g_stats.pipelined_loops++;
    // The first iteration stays unpipelined. BodyRunner probes a freshly
    // bound compiled body with a SYNCHRONOUS eval (a Metal build error
    // raised on an async worker aborts the process), and that probe should
    // land on an iteration the loop is known to want -- a zero-trip loop
    // must cost exactly one condition here, as it does on the serial path.
    {
      mx::array first = cond_of(vals);
      if (!loop_item_bool(first)) {
        write_results(e, env, vals);
        return;
      }
    }
    vals = runner.run_one(vals);
    loop_flush(vals, cost);
    mx::array pred = cond_of(vals);
    for (;;) {
      // Built, not run: `next` is a lazy graph until something asks for its
      // values, and nothing does until `pred` says this iteration happens.
      // Pure host work, and it overlaps whatever the device is still doing
      // for the carry and condition submitted last time round.
      std::vector<mx::array> next = runner.run_one(vals);
      mx::array npred = cond_of(next);
      const bool go = loop_item_bool(pred);    // the one blocking point
      // An evaluated condition is detached from the carry it was computed
      // from -- but only once MLX has actually walked it, and only if the
      // host read did not have to build a converted copy first. Dropping it
      // outright is one move and needs neither to be true.
      pred = npred;
      if (!go) break;                          // `vals` is intact: see above
      vals = std::move(next);                  // (1) release the old carry
      std::vector<mx::array> pending(vals);
      pending.push_back(pred);
      mx::async_eval(pending);                 // (2) submit, do not wait
      g_stats.pipelined_steps++;
      loop_account(cost);                      // (3) same cadence, same clears
    }
    write_results(e, env, vals);
  }

  static void write_results(const Entry& e,
                            std::vector<std::optional<mx::array>>& env,
                            const std::vector<mx::array>& vals) {
    if (vals.size() != e.outs.size())
      throw std::runtime_error("tape: loop result count mismatch");
    for (size_t i = 0; i < vals.size(); i++) env[e.outs[i]] = vals[i];
  }

  // One uncompiled application of a loop body.
  static std::vector<mx::array> run_body(Program* body,
                                         const std::vector<mx::array>& vals,
                                         const std::vector<mx::array>& caps,
                                         bool in_trace) {
    std::vector<mx::array> flat(vals);
    flat.insert(flat.end(), caps.begin(), caps.end());
    return body->interpret(flat, in_trace);
  }

  // ops/control.py _run_chunked.
  static std::vector<mx::array> run_chunked(
      Program* body, const std::vector<mx::array>& ins,
      const std::vector<mx::array>& caps, int64_t trip, int64_t K,
      int64_t cost) {
    std::vector<mx::array> vals = ins;
    const int64_t sync_every =
        std::max<int64_t>(1, 75000 / std::max<int64_t>(K * cost, 1));
    auto chunk = [&](int64_t repeat, const std::vector<mx::array>& v) {
      std::vector<mx::array> flat(v);
      flat.insert(flat.end(), caps.begin(), caps.end());
      if (body->may_compile(static_cast<int>(repeat))) {
        g_stats.compiled_calls++;
        return body->compiled(static_cast<int>(repeat))(flat);
      }
      std::vector<mx::array> out = v;
      for (int64_t r = 0; r < repeat; r++) out = run_body(body, out, caps, false);
      return out;
    };
    for (int64_t i = 0; i < trip / K; i++) {
      vals = chunk(K, vals);
      // Async-flush each chunk (a blocking sync per chunk serializes CPU
      // and GPU); block only often enough to bound pending buffers.
      if ((i + 1) % sync_every == 0) {
        loop_flush(vals, sync_every * K * cost);
      } else {
        mx::async_eval(vals);
      }
    }
    const int64_t rem = trip % K;
    for (int64_t i = 0; i < rem; i++) vals = chunk(1, vals);
    loop_flush(vals, (trip % std::max<int64_t>(sync_every * K, 1)) * cost);
    return vals;
  }

  // ops/control.py _BodyRunner: runs a while body, recovering from the ways
  // MLX's compiled path can fail at CALL time. Every recovery simply redoes
  // the iteration -- bodies are pure, and a failed call leaves the caller's
  // carry untouched.
  class BodyRunner {
   public:
    BodyRunner(Program* body, const std::vector<mx::array>& caps, int repeat)
        : body_(body), caps_(caps), repeat_(repeat) {
      bind();
    }

    std::vector<mx::array> run_one(const std::vector<mx::array>& vals) {
      for (;;) {
        std::optional<std::vector<mx::array>> out = step(vals);
        if (out) return std::move(*out);
      }
    }

   private:
    void bind() {
      compiled_ = body_->may_compile(repeat_);
      // Probe the freshly bound compiled body once: a compiled call only
      // BUILDS the graph, and MLX generates the fused Metal kernels at
      // eval, so failures like "Too many inputs/outputs fused" land at the
      // next sync point -- by which time the carry has advanced past the
      // iteration a redo could repair. One sync per loop ENTRY: the shapes
      // are fixed for the life of the loop, so one buildable call proves
      // every later one.
      probe_ = compiled_;
    }

    std::optional<std::vector<mx::array>> step(
        const std::vector<mx::array>& vals) {
      bool resource_limit = false;
      std::exception_ptr err;
      try {
        std::vector<mx::array> flat(vals);
        flat.insert(flat.end(), caps_.begin(), caps_.end());
        std::vector<mx::array> out;
        if (compiled_) {
          g_stats.compiled_calls++;
          out = body_->compiled(repeat_)(flat);
        } else {
          out = body_->interpret(flat, false);
        }
        if (probe_) {
          // Inside the try on purpose: this is where a body MLX cannot
          // generate kernels for reports itself. Synchronous (not
          // async_eval) — a Metal build error raised on a worker thread
          // aborts the process.
          mx::eval(out);
          probe_ = false;
        }
        limit_retries_ = 0;
        return out;
      } catch (const std::exception& ex) {
        resource_limit = is_resource_limit(ex);
        err = std::current_exception();
      }
      // Recovery OUTSIDE the handler: the failed attempt's arrays must be
      // gone before anything tries to allocate again (the C++ analogue of
      // the traceback that pins a failed trace's buffers in Python).
      if (resource_limit) {
        debug_print("Metal buffer limit hit in while body; clearing cache "
                    "and retrying");
        g_stats.limit_retries++;
        gc_collect();
        mx::clear_cache();
        limit_retries_++;
        // BOUNDED: retrying an oversized compiled trace forever once
        // livelocked a worker for hours.
        if (limit_retries_ == 2 && compiled_) {
          body_->drop_compiled();
          bind();
        } else if (limit_retries_ > 3) {
          std::rethrow_exception(err);
        }
        return std::nullopt;
      }
      if (!compiled_) std::rethrow_exception(err);
      debug_print("compiled while body failed; retrying eagerly");
      body_->drop_compiled();
      bind();
      return std::nullopt;
    }

    Program* body_;
    const std::vector<mx::array>& caps_;
    int repeat_;
    bool compiled_ = false;
    bool probe_ = false;
    int limit_retries_ = 0;
  };

  // engine.execute's msl recovery, on the native side of the boundary: a
  // generated kernel that was TRACED into a compiled graph has never been
  // built, so the call is settled here, synchronously, while any such plan
  // is unproven -- a Metal build error surfacing later, on an async worker,
  // aborts the process. On a failure the plans that could be at fault are
  // retired (they fall back to their interpreted loops), the compiled
  // graphs that embedded their kernels go with them, and the program is run
  // again. Once, and only for programs with no host call in them: a rerun
  // repeats whatever the first attempt already did to the world.
  void settle_msl(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outs) {
    if (t_msl_pending.empty()) return;
    std::exception_ptr err;
    try {
      mx::eval(outs);
      for (MslPlan* p : t_msl_pending) p->validate();
      t_msl_pending.clear();
      return;
    } catch (const std::exception&) {
      err = std::current_exception();
    }
    for (MslPlan* p : t_msl_pending) p->kill();
    g_stats.msl_failures++;
    t_msl_pending.clear();
    if (has_host()) std::rethrow_exception(err);
    debug_print("a traced msl_scan kernel failed to build; dropping it and "
                "rerunning the program");
    drop_compiled_deep();
    gc_collect();
    mx::clear_cache();
    outs = run_recovering(inputs);
    // A second failure is not this milestone's business: the caller retires
    // the tape and the Python engine takes the program back.
    if (!t_msl_pending.empty()) {
      mx::eval(outs);
      for (MslPlan* p : t_msl_pending) p->validate();
      t_msl_pending.clear();
    }
  }

  // Whether anything in this program (or its regions) calls back into
  // Python. Structure, not policy -- read off the tape, like reads_host.
  bool has_host() const {
    for (const Entry& e : ops_) {
      if (e.op == kHostCall) return true;
      for (const auto& r : e.regions)
        if (r->has_host()) return true;
    }
    return false;
  }

  // Drop every compiled graph in this program and its regions: a compiled
  // trace that embedded a now-dead kernel would keep calling it. The
  // compile DECISION stands (disable_msl clears `_compiled` and no more) --
  // the next trace simply builds the loop where the kernel was.
  void drop_compiled_deep() {
    for (const auto& kv : compiled_) mx::detail::compile_erase(kv.second.id);
    compiled_.clear();
    for (const Entry& e : ops_)
      for (const auto& r : e.regions) r->drop_compiled_deep();
  }

  // engine.execute's recovery policy, on the native side of the boundary:
  // a compiled path that fails hands the program to the eager one (which
  // is always correct), and buffer exhaustion clears and retries once.
  // Programs are pure, so a rerun is safe.
  std::vector<mx::array> run_recovering(const std::vector<mx::array>& inputs) {
    for (int attempt = 0;; attempt++) {
      bool used_compiled = compile_ && !compile_disabled_;
      bool resource_limit = false;
      std::exception_ptr err;
      try {
        return call(inputs, false);
      } catch (const std::exception& ex) {
        resource_limit = is_resource_limit(ex);
        err = std::current_exception();
      }
      // Outside the handler, holding none of the failed attempt's arrays.
      if (used_compiled) {
        // Whatever went wrong, the compiled graph is what was running:
        // MLX's compiler rejects some traces outright, and buffer
        // exhaustion during a trace means the trace is too big to hold.
        debug_print("compiled native tape failed; running eagerly");
        drop_compiled();
        if (resource_limit) {
          gc_collect();
          mx::clear_cache();
        }
        continue;
      }
      if (resource_limit && attempt < 2) {
        debug_print("Metal buffer limit hit in the native tape; clearing "
                    "cache and retrying");
        g_stats.limit_retries++;
        gc_collect();
        mx::clear_cache();
        continue;
      }
      std::rethrow_exception(err);
    }
  }

  // interpreter._eager_flush: settle everything the environment still
  // holds. Only that needs to survive -- pruned intermediates are already
  // unreferenced, so MLX frees them as the evaluation walks the graph
  // instead of materializing the whole chain at once.
  static void eager_flush(const std::vector<std::optional<mx::array>>& env) {
    std::vector<mx::array> live;
    for (const auto& v : env)
      if (v) live.push_back(*v);
    if (live.empty()) return;
    g_stats.flushes++;
    const bool hard = g_cfg.flush_sync_every > 0 &&
                      g_stats.flushes % g_cfg.flush_sync_every == 0;
    flush_eval(live, hard);
    // "Frees" into MLX's CACHE, whose byte bound is the memory limit -- so
    // an eager phase whose traffic is many times its live set claims the
    // traffic. Return it to the OS past the configured watermark.
    if (hard && g_cfg.flush_clear_bytes >= 0 &&
        static_cast<int64_t>(mx::get_cache_memory()) > g_cfg.flush_clear_bytes) {
      g_stats.cache_clears++;
      mx::clear_cache();
    }
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

  struct Compiled {
    std::uintptr_t id = 0;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> fn;
  };

  int nslots_;
  int nargs_;
  std::vector<Entry> ops_;
  std::vector<int> outputs_;
  std::vector<int> copies_;
  // M3: the compiled path. `compile_` and `anchors_` are Python's decision
  // (see set_compile); everything else is run-time state that only ever
  // moves one way -- toward the eager path, which is always correct.
  bool compile_ = false;
  bool compile_disabled_ = false;
  bool no_chunk_ = false;
  mutable int reads_host_ = -1;   // lazily derived, never un-derived
  int64_t max_repeat_ = 1;
  std::vector<int> anchors_;
  std::map<int, Compiled> compiled_;
  // `run` mutates all of the run-time state above -- the compiled-graph
  // cache above all -- with the GIL released, so two threads calling the
  // SAME executable would race on it (jax lets one jitted function be
  // called from any number of threads). One lock per program serializes
  // those and nothing else; distinct executables still run concurrently.
  // Taken INSIDE the GIL release, never outside: a waiter must not hold the
  // GIL, or the holder's recovery paths (which reacquire it to collect
  // cycles) would deadlock against it.
  std::mutex lock_;
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

// The runtime cadences, copied in from the Python modules that parse the
// environment (metaljax.interpreter and metaljax.ops.control). One source of
// truth per number: a second env-var reader here would be a second opinion,
// and these are values the command-buffer lottery is pinned to.
void configure(int64_t eager_flush_bytes, int64_t flush_sync_every,
               int64_t flush_clear_bytes, int64_t loop_clear_cost,
               int64_t while_pipeline, bool debug, bool memdbg) {
  g_cfg.eager_flush_bytes = eager_flush_bytes;
  g_cfg.flush_sync_every = flush_sync_every;
  g_cfg.flush_clear_bytes = flush_clear_bytes;
  g_cfg.loop_clear_cost = loop_clear_cost;
  g_cfg.while_pipeline = while_pipeline;
  g_cfg.debug = debug;
  g_cfg.memdbg = memdbg;
}

nb::dict stats() {
  nb::dict d;
  d["flushes"] = g_stats.flushes;
  d["cache_clears"] = g_stats.cache_clears;
  d["loop_flushes"] = g_stats.loop_flushes;
  d["loop_clears"] = g_stats.loop_clears;
  d["limit_retries"] = g_stats.limit_retries;
  d["compiled_calls"] = g_stats.compiled_calls;
  d["compiles"] = g_stats.compiles;
  d["compile_drops"] = g_stats.compile_drops;
  d["chunk_drops"] = g_stats.chunk_drops;
  d["unrolls"] = g_stats.unrolls;
  d["pipelined_loops"] = g_stats.pipelined_loops;
  d["pipelined_steps"] = g_stats.pipelined_steps;
  d["serial_loops"] = g_stats.serial_loops;
  d["msl_launches"] = g_stats.msl_launches;
  d["msl_failures"] = g_stats.msl_failures;
  d["host_calls"] = g_stats.host_calls;
  return d;
}

}  // namespace

void register_tape(nb::module_& m) {
  m.def("opcodes", &opcodes,
        "StableHLO op name -> opcode; a name absent here declines");
  m.def("dtype_codes", &dtype_codes,
        "MLIR element type -> dtype code; absent means unsupported");
  m.def("configure", &configure, nb::arg("eager_flush_bytes"),
        nb::arg("flush_sync_every"), nb::arg("flush_clear_bytes"),
        nb::arg("loop_clear_cost"), nb::arg("while_pipeline"),
        nb::arg("debug"), nb::arg("memdbg"),
        "Copy the Python-side runtime cadences into the native engine");
  m.def("stats", &stats, "Native-engine counters (sync points, compiles)");
  // M5b: one generated persistent kernel's launch recipe. Built by
  // metaljax.tape._lower_msl from an msl_scan Plan; `layout` is the
  // Cursor-encoded rest of it (documented at MslPlan::parse).
  nb::class_<MslPlan>(m, "MslPlan")
      .def(nb::init<std::string, std::string, std::string,
                    std::vector<std::string>, std::vector<std::string>,
                    std::vector<std::vector<int>>, std::vector<int>,
                    std::vector<int64_t>>(),
           nb::arg("name"), nb::arg("source"), nb::arg("header"),
           nb::arg("input_names"), nb::arg("output_names"),
           nb::arg("out_shapes"), nb::arg("out_dtypes"), nb::arg("layout"))
      .def_prop_ro("dead", &MslPlan::dead)
      .def_prop_ro("validated", &MslPlan::validated)
      .def_prop_ro("name", &MslPlan::name);
  nb::class_<Program>(m, "Program")
      .def(nb::init<int, int>(), nb::arg("num_slots"), nb::arg("num_args"))
      .def("add", &Program::add, nb::arg("opcode"), nb::arg("operands"),
           nb::arg("results"), nb::arg("attrs"),
           nb::arg("payload").none(), nb::arg("drops"),
           nb::arg("regions") = std::vector<std::shared_ptr<Program>>(),
           nb::arg("bytes") = 0,
           nb::arg("fattrs") = std::vector<double>(),
           nb::arg("msl").none() = nb::none(),
           nb::arg("host").none() = nb::none())
      .def("set_outputs", &Program::set_outputs, nb::arg("slots"),
           nb::arg("copies") = std::vector<int>())
      .def("set_compile", &Program::set_compile, nb::arg("compile"),
           nb::arg("anchors") = std::vector<int>(),
           nb::arg("max_repeat") = 1)
      .def("run", &Program::run, nb::arg("inputs"))
      .def_prop_ro("compiled_dropped", &Program::compiled_dropped)
      .def_prop_ro("num_ops", &Program::num_ops)
      .def_prop_ro("num_slots", &Program::num_slots)
      .def_prop_ro("num_args", &Program::num_args)
      .def("op_histogram", &Program::op_histogram,
           "opcode -> entry count, this program and its regions")
      .def_prop_ro("max_live", &Program::max_live);
}
