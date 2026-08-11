/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_lowering.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "metal/metal_dtypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlx/mlx.h"
#include "program.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/shape_util.h"

namespace metaljax {

namespace {

namespace mx = mlx::core;

absl::Status Decline(absl::string_view what) {
  return absl::UnimplementedError(absl::StrCat("metaljax-native: ", what));
}

absl::string_view View(llvm::StringRef s) {
  return absl::string_view(s.data(), s.size());
}

const absl::flat_hash_map<std::string, int>& OpcodeTable() {
  static const auto* table = [] {
    auto* m = new absl::flat_hash_map<std::string, int>();
    for (const std::pair<std::string, int>& kv : opcodes()) m->emplace(kv);
    return m;
  }();
  return *table;
}

// Ops whose whole lowering is "look the opcode up and bind the results": no
// attributes to resolve, and a C++ handler that branches on nothing but the
// operand dtypes (which is what its Python counterpart in
// src/metaljax/ops/elementwise.py does too).  Spelled out rather than derived
// from the registry, because the registry also holds names whose handler READS
// an attribute vector -- reaching one of those with an empty vector would be
// an out-of-bounds read, not a decline.
const absl::flat_hash_set<std::string>& SimpleOps() {
  static const auto* set = new absl::flat_hash_set<std::string>{
      // unary (ops/elementwise.py _UNARY)
      "stablehlo.abs", "stablehlo.cbrt", "stablehlo.ceil", "stablehlo.cosine",
      "stablehlo.erf", "chlo.erf", "stablehlo.erf_inv", "chlo.erf_inv",
      "stablehlo.exponential", "stablehlo.exponential_minus_one",
      "stablehlo.floor", "stablehlo.is_finite", "stablehlo.log",
      "stablehlo.log_plus_one", "stablehlo.logistic", "stablehlo.negate",
      "stablehlo.not", "stablehlo.round_nearest_afz",
      "stablehlo.round_nearest_even", "stablehlo.rsqrt", "stablehlo.sign",
      "stablehlo.sine", "stablehlo.sqrt", "stablehlo.tan", "stablehlo.tanh",
      "chlo.square",
      // binary (ops/elementwise.py _BINARY)
      "stablehlo.add", "stablehlo.multiply", "stablehlo.subtract",
      "stablehlo.maximum", "stablehlo.minimum", "stablehlo.and",
      "stablehlo.or", "stablehlo.xor", "stablehlo.divide",
      "stablehlo.remainder", "stablehlo.power", "stablehlo.atan2",
      // selection
      "stablehlo.select", "stablehlo.clamp",
      // complex64 constructors and projections
      "stablehlo.real", "stablehlo.imag", "stablehlo.complex",
  };
  return *set;
}

// Ops whose result may be a VIEW of an operand's storage rather than new
// storage (tape.py `_VIEW_OPS`, minus the ones this phase does not lower).
// Used only to keep a constant's buffer out of an output position.
bool IsViewOp(absl::string_view name) {
  return name == "stablehlo.reshape" || name == "stablehlo.transpose" ||
         name == "stablehlo.convert" || name == "stablehlo.slice" ||
         name == "stablehlo.broadcast_in_dim" ||
         name == "stablehlo.concatenate" || name == "stablehlo.dynamic_slice" ||
         name == "stablehlo.bitcast_convert";
}

// tape.py `_scatter_noop`: whether ops/gather.py `_scatter` hands the operand
// straight back.  Empty updates apply nothing, and a zero-size operand drops
// every update as out of bounds.  The result IS the operand array, so the
// tape aliases the slot (and the taints ride along) rather than emitting an
// entry that happens to compute nothing.
bool IsScatterNoop(mlir::Operation* op) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1) return false;
  auto empty = [](mlir::Value v) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t) return false;
    for (int64_t d : t.getShape())
      if (d == 0) return true;
    return false;
  };
  return empty(op->getOperand(2)) || empty(op->getOperand(0));
}

// The three ops that carry regions and become sub-Programs (tape.py
// `_REGION_OPS`).
bool IsControlOp(absl::string_view name) {
  return name == "stablehlo.while" || name == "stablehlo.if" ||
         name == "stablehlo.case";
}

// Reduce monoids: body op -> kind, mirroring tape.py's _REDUCE_KINDS and
// _BOOL_REDUCE_KINDS.  Which table applies is decided by the input element
// type, which is static in the IR.
std::optional<int64_t> ReduceKind(absl::string_view body_op, bool is_bool) {
  if (is_bool) {
    if (body_op == "stablehlo.or") return 4;
    if (body_op == "stablehlo.and") return 5;
    if (body_op == "stablehlo.add") return 4;  // or on i1 sometimes adds
    return std::nullopt;
  }
  if (body_op == "stablehlo.add") return 0;
  if (body_op == "stablehlo.multiply") return 1;
  if (body_op == "stablehlo.maximum") return 2;
  if (body_op == "stablehlo.minimum") return 3;
  return std::nullopt;
}

int64_t Product(const std::vector<int64_t>& xs) {
  int64_t p = 1;
  for (int64_t x : xs) p *= x;
  return p;
}

// Python's `//`, which the window arithmetic below depends on: a window that
// does not fit its padded axis gives a NEGATIVE numerator, and C++'s
// truncation would round it towards zero and invent an output element where
// Python's floor gives none.  (`(-1) // 2` is -1 in Python and 0 in C++;
// after the `+ 1` and the `max(0, ...)` that is one window versus none.)
int64_t FloorDiv(int64_t a, int64_t b) {
  int64_t q = a / b, r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) q--;
  return q;
}

// Python's `-(-a // b)` on non-negative operands.
int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// tape.py `_opt_i64_list`: an i64 array-ish attribute, or the default when
// the op does not carry it.  StableHLO spells the window attributes as
// DenseI64ArrayAttr and `padding` as an elements attribute, and
// `metaljax._ir.i64_list` reads either -- so this does too.
std::vector<int64_t> OptI64List(mlir::Operation* op, llvm::StringRef name,
                                std::vector<int64_t> fallback) {
  if (auto arr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name))
    return std::vector<int64_t>(arr.asArrayRef().begin(),
                                arr.asArrayRef().end());
  if (auto den = op->getAttrOfType<mlir::DenseIntElementsAttr>(name)) {
    std::vector<int64_t> out;
    for (const llvm::APInt& v : den.getValues<llvm::APInt>())
      out.push_back(v.getSExtValue());
    return out;
  }
  return fallback;
}

// The element-type name of a value, as src/metaljax/tape.py's `_element`
// spells it ("i1", "ui32", "bf16", "complex<f32>").
std::optional<std::string> ElementName(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return std::nullopt;
  return TapeElementName(t.getElementType());
}

// reduce_window's own materialization cap (tape.py `_WINDOW_MAX`); above it
// the Python handler raises, so the tape declines instead of asking MLX for a
// view with more elements than the device can hold.
constexpr int64_t kWindowMax = 200000000;

// The static half of ops/reduction.py `_extract_windows`, transliterated from
// tape.py `_window_plan`: base dilation, then padding, then the strided window
// view -- every shape of which follows from the attributes, so none of it
// needs an array.  The attribute order is `read_window_plan`'s in
// native/ops_reduce.cc, field for field.
struct WindowPlanOut {
  std::vector<int64_t> attrs;
  std::vector<int64_t> out_sizes;
  int64_t wflat = 0;
};

absl::StatusOr<WindowPlanOut> BuildWindowPlan(
    int64_t rank, const std::vector<int64_t>& src,
    const std::vector<int64_t>& wd, const std::vector<int64_t>& strides,
    const std::vector<int64_t>& bdil, const std::vector<int64_t>& wdil,
    const std::vector<std::pair<int64_t, int64_t>>& pad) {
  WindowPlanOut out;
  std::vector<int64_t>& attrs = out.attrs;
  std::vector<int64_t> shape = src;

  std::vector<std::pair<int64_t, int64_t>> dil;   // (axis, base dilation)
  for (int64_t ax = 0; ax < rank; ax++)
    if (bdil[ax] != 1) dil.push_back({ax, bdil[ax]});
  attrs.push_back(static_cast<int64_t>(dil.size()));
  for (const auto& [ax, b] : dil) {
    if (b < 1) return Decline("reduce_window base dilation");
    shape[ax] = shape[ax] * b;
    attrs.push_back(ax);
    attrs.push_back(b);
    attrs.push_back(static_cast<int64_t>(shape.size()));
    attrs.insert(attrs.end(), shape.begin(), shape.end());
    shape[ax] = shape[ax] - (b - 1);
    attrs.push_back(shape[ax]);
  }

  bool any_pad = false, any_neg = false;
  for (const auto& [lo, hi] : pad) {
    any_pad = any_pad || lo != 0 || hi != 0;
    any_neg = any_neg || lo < 0 || hi < 0;
  }
  if (any_pad) {
    // ops/reduction.py raises here: mx::pad has no negative widths, and a
    // crop-then-window rewrite is not what the Python engine computes.
    if (any_neg) return Decline("reduce_window negative padding");
    attrs.push_back(1);
    attrs.push_back(rank);
    for (const auto& p : pad) attrs.push_back(p.first);
    attrs.push_back(rank);
    for (const auto& p : pad) attrs.push_back(p.second);
    for (int64_t i = 0; i < rank; i++)
      shape[i] += pad[i].first + pad[i].second;
  } else {
    attrs.push_back(0);
    attrs.push_back(0);
    attrs.push_back(0);
  }

  for (int64_t i = 0; i < rank; i++) {
    const int64_t span = (wd[i] - 1) * wdil[i] + 1;
    out.out_sizes.push_back(
        std::max<int64_t>(0, FloorDiv(shape[i] - span, strides[i]) + 1));
  }
  out.wflat = Product(wd);
  if (out.wflat * Product(out.out_sizes) > kWindowMax)
    return Decline("reduce_window materialization too large");

  std::vector<int64_t> elem(rank, 1);
  for (int64_t i = rank - 2; i >= 0; i--) elem[i] = elem[i + 1] * shape[i + 1];
  std::vector<int64_t> view_shape = out.out_sizes;
  view_shape.insert(view_shape.end(), wd.begin(), wd.end());
  std::vector<int64_t> view_strides;
  for (int64_t i = 0; i < rank; i++) view_strides.push_back(elem[i] * strides[i]);
  for (int64_t i = 0; i < rank; i++) view_strides.push_back(elem[i] * wdil[i]);
  std::vector<int64_t> flat = out.out_sizes;
  flat.push_back(out.wflat);
  attrs.push_back(static_cast<int64_t>(view_shape.size()));
  attrs.insert(attrs.end(), view_shape.begin(), view_shape.end());
  attrs.push_back(static_cast<int64_t>(view_strides.size()));
  attrs.insert(attrs.end(), view_strides.begin(), view_strides.end());
  attrs.push_back(static_cast<int64_t>(flat.size()));
  attrs.insert(attrs.end(), flat.begin(), flat.end());

  bool empty = out.wflat == 0;
  for (int64_t s : out.out_sizes) empty = empty || s == 0;
  attrs.push_back(empty ? 1 : 0);
  attrs.push_back(static_cast<int64_t>(out.out_sizes.size()));
  attrs.insert(attrs.end(), out.out_sizes.begin(), out.out_sizes.end());
  return out;
}

// --------------------------------------------------------------------------
// sort comparators (src/metaljax/ops/sort.py)
// --------------------------------------------------------------------------
//
// tape.py lowers only the comparator that IS a bare compare, because the
// Python ENGINE evaluates the other shape -- a comparator that computes a KEY
// from its argument pair -- by running the block's code on whole arrays, and
// tape.py has no way to hand a block to that opcode.  This plugin has no
// Python engine behind it, so the recognizer is ported instead: the key chain
// is elementwise scalar code, and elementwise scalar code lowered into the
// ENCLOSING frame computes the same key on the whole operand.  What the
// Python does with an interpreter at run time, this does with tape entries at
// compile time; `_gather_sorted` is then the kSort entry, keyed on the chain's
// output rather than on the operand.

// ops/sort.py `_arg_deps`: the transitive comparator-block-argument
// dependencies of a value.  Values defined outside the comparator block
// (constants XLA's parse hoisted out) are opaque leaves.
std::vector<unsigned> ArgDeps(mlir::Value root, mlir::Block& block) {
  llvm::DenseSet<mlir::Value> seen;
  llvm::DenseSet<unsigned> deps;
  std::vector<mlir::Value> stack{root};
  while (!stack.empty()) {
    mlir::Value v = stack.back();
    stack.pop_back();
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      if (arg.getOwner() == &block) deps.insert(arg.getArgNumber());
      continue;
    }
    mlir::Operation* def = v.getDefiningOp();
    if (def == nullptr || def->getBlock() != &block) continue;
    if (!seen.insert(v).second) continue;
    for (mlir::Value w : def->getOperands()) stack.push_back(w);
  }
  std::vector<unsigned> out(deps.begin(), deps.end());
  std::sort(out.begin(), out.end());
  return out;
}

// ops/sort.py `_serialize`: the canonical form of a value's def DAG with the
// comparator's own argument renamed, so the two sides of the compare can be
// checked for computing the SAME function of their respective arguments.  An
// asymmetric comparator is not a sort by a key and must not be treated as one.
void SerializeKey(mlir::Value v, mlir::BlockArgument key, mlir::Block& block,
                  std::string* out) {
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    absl::StrAppend(out, arg == key ? "KEY" : "OTHER_ARG");
    return;
  }
  mlir::Operation* def = v.getDefiningOp();
  if (def == nullptr || def->getBlock() != &block) {
    // An external leaf is identified by the value itself: the two sides
    // referring to the SAME hoisted constant is what makes them symmetric.
    std::string text;
    llvm::raw_string_ostream os(text);
    v.printAsOperand(os, mlir::OpPrintingFlags());
    absl::StrAppend(out, "(EXT ", os.str(), ")");
    return;
  }
  absl::StrAppend(out, "(", View(def->getName().getStringRef()));
  std::vector<std::pair<std::string, std::string>> attrs;
  for (const mlir::NamedAttribute& na : def->getAttrs()) {
    std::string text;
    llvm::raw_string_ostream os(text);
    na.getValue().print(os);
    attrs.push_back({na.getName().str(), os.str()});
  }
  std::sort(attrs.begin(), attrs.end());
  for (const auto& [n, a] : attrs) absl::StrAppend(out, " ", n, "=", a);
  for (mlir::Value w : def->getOperands()) {
    absl::StrAppend(out, " ");
    SerializeKey(w, key, block, out);
  }
  absl::StrAppend(out, ")");
}

// Which ops a key chain may be made of.  The Python recognizer accepts any op
// with a handler and relies on the chain being elementwise; here the rule is
// spelled out, because an op that is NOT elementwise would be lowered against
// its rank-0 IR type and then run on a whole array -- a wrong answer rather
// than a decline.  Every operand and result being rank-0 is checked as well:
// together the two make "scalar block code, run on arrays" a fact rather than
// an assumption.
bool IsChainOp(absl::string_view name) {
  return SimpleOps().contains(name) || name == "stablehlo.compare" ||
         name == "stablehlo.convert" || name == "stablehlo.constant" ||
         name == "stablehlo.shift_left" ||
         name == "stablehlo.shift_right_logical" ||
         name == "stablehlo.shift_right_arithmetic";
}

bool IsRank0(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return t && t.getRank() == 0;
}

// tape.py `_CUM_KINDS`: the monoid of a cumulative reduce_window, which jax
// lowers as a full-width window with prefix (or suffix) padding.
std::optional<int64_t> CumKind(absl::string_view body_op) {
  if (body_op == "stablehlo.add") return 0;
  if (body_op == "stablehlo.maximum") return 1;
  if (body_op == "stablehlo.minimum") return 2;
  if (body_op == "stablehlo.multiply") return 3;
  return std::nullopt;
}

// ops/linalg.py `_exact_f32_chunk`: the contraction-dim chunk that keeps an
// integer dot exact under an f32 matmul, or 0 when these operands cannot take
// that path.  f32 holds every integer below 2**24 exactly, so a sum of
// products stays exact while the products' magnitudes do.
int64_t ExactF32Chunk(mx::Dtype l, mx::Dtype r) {
  auto max_abs = [](mx::Dtype d) -> int64_t {
    if (d == mx::int8) return 128;
    if (d == mx::uint8) return 255;
    return 0;
  };
  const int64_t ml = max_abs(l), mr = max_abs(r);
  if (ml == 0 || mr == 0) return 0;
  int64_t bound = (int64_t{1} << 24) / (ml * mr);
  int64_t bits = 0;
  for (int64_t v = bound; v > 0; v >>= 1) bits++;   // python's bit_length
  return int64_t{1} << (bits - 1);
}

// Estimated device bytes of a value (interpreter.value_bytes): element count
// times dtype size, straight off the IR.  It meters the eager flush cadence
// and nothing else, so an over-estimate only makes the safety net fire early.
int64_t ValueBytes(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return 0;
  std::optional<mx::Dtype> dt = MxDtypeOf(t.getElementType());
  if (!dt.has_value()) return 0;
  int64_t n = 1;
  for (int64_t d : t.getShape()) {
    if (d < 0) return 0;
    n *= d;
  }
  return n * static_cast<int64_t>(dt->size());
}

// METALJAX_DUMP_TAPE=1 prints the finished tape, one entry per line, in the
// form `opcode-name ins -> outs attrs`.  It exists to be diffed against the
// Stage 1 lowering's tape for the same jitted function
// (notes/cpp-p2-lowering.md): the two builders need not agree entry for
// entry, but for a given opcode the attribute LAYOUT must, because one
// executor decodes both.
const bool kDumpTape = [] {
  const char* v = std::getenv("METALJAX_DUMP_TAPE");
  return v != nullptr && std::string(v) == "1";
}();

// METALJAX_DUMP_MODULE=1 prints the module this plugin was HANDED, which is
// not the module jax printed: XLA's own parse runs ahead of `CompileAndLoad`
// and legalizes chlo, CSEs and hoists constants out of regions.  A tape diff
// against the Stage 1 lowering is meaningless without it -- the two builders
// would be walking different programs -- so the text is dumpable on its own.
const bool kDumpModule = [] {
  const char* v = std::getenv("METALJAX_DUMP_MODULE");
  return v != nullptr && std::string(v) == "1";
}();

const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();

// One integer knob, on metal_client.cc's contract: Python's `int(...)` raises
// on garbage, a lowering cannot usefully raise out of a compile, so garbage
// keeps the documented default and says so under METALJAX_DEBUG.
int64_t EnvInt(const char* name, int64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const long long parsed = std::strtoll(v, &end, 10);
  if (end == v || *end != '\0') {
    if (kDebug)
      std::fprintf(stderr, "[metaljax-native] ignoring %s=%s (not an integer)\n",
                   name, v);
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

// The `X != "0"` form of every metaljax on/off switch.
bool EnvOn(const char* name) {
  const char* v = std::getenv(name);
  return v == nullptr || std::string(v) != "0";
}

// --------------------------------------------------------------------------
// the compile decisions' budgets (src/metaljax/ops/control.py)
// --------------------------------------------------------------------------
//
// Each default and the measurement behind it belong to the Python module that
// owns the variable; nothing here is a preference.  In one line each:
//
//  * `kCompileEnabled` -- METALJAX_COMPILE=0 is the global off switch, and it
//    must reproduce the all-eager plugin exactly (interpreter.COMPILE_ENABLED).
//  * `kTraceBudget` -- ops one mx::compile trace may hold, counted loops
//    unrolled.  MLX retains every intermediate of a trace, so an oversized one
//    exhausts Metal's ~500k live-buffer limit.
//  * `kBodyCompile` -- METALJAX_BODY_COMPILE=0 keeps everything compiled
//    EXCEPT while bodies (the targeted mitigation for the command-buffer
//    corruption, which bites REPLAYED bodies).
//  * `kChunkMax` / `kChunkMaxCost` -- how many iterations one chunked replay
//    may unroll, and the body size past which chunking stops paying.
//  * `kCompileBytes` -- bytes one trace may materialize.  The op budget bounds
//    the live-buffer COUNT and says nothing about their SIZE; a jitted
//    parameter init is 365 ops and 15 GB of traffic.  0 disables the gate.
const bool kCompileEnabled = EnvOn("METALJAX_COMPILE");
const int64_t kTraceBudget = EnvInt("METALJAX_TRACE_BUDGET", 20000);
const bool kBodyCompile = EnvOn("METALJAX_BODY_COMPILE");
const int64_t kChunkMax = EnvInt("METALJAX_CHUNK_MAX", 16);
const int64_t kChunkMaxCost = EnvInt("METALJAX_CHUNK_MAX_COST", 1500);
const int64_t kCompileBytesMb = EnvInt("METALJAX_COMPILE_BYTES_MB", 65536);
const int64_t kCompileBytes =
    std::max<int64_t>(kCompileBytesMb, 0) * (int64_t{1} << 20);

std::string OpcodeName(int code) {
  for (const std::pair<std::string, int>& kv : opcodes())
    if (kv.second == code) return kv.first;
  return absl::StrCat("op", code);
}

// An mx::array owning a fresh copy of `bytes` host bytes.  MLX frees it.
mx::array OwnedArray(const char* src, size_t nbytes, const mx::Shape& shape,
                     mx::Dtype dtype) {
  void* buf = std::malloc(nbytes == 0 ? 1 : nbytes);
  if (buf == nullptr) throw std::bad_alloc();
  if (nbytes != 0) std::memcpy(buf, src, nbytes);
  return mx::array(buf, shape, dtype, [](void* p) { std::free(p); });
}

// --------------------------------------------------------------------------
// the analyses a control-flow entry carries (src/metaljax/ops/control.py)
// --------------------------------------------------------------------------
//
// A while entry records the answers to three questions the executor is not
// allowed to ask for itself: is this the counted loop jax emits for
// scan/fori_loop, what does one iteration cost, and how often must the loop
// settle its carries.  Each is transliterated from the Python function named
// beside it, cache and all -- the caches are on the Interpreter there and on
// the `LowerContext` here, because the same block is walked once per enclosing
// analysis and the whole-model bodies are large.

// `_analyze_counted`'s answer: the carry index of the counter, and where the
// bound comes from.
struct Counted {
  enum Kind { kStatic, kCarry, kValue };
  int64_t k = 0;
  Kind kind = kStatic;
  int64_t n = 0;        // kStatic: the bound N; kCarry: the carry index
  mlir::Value value;    // kValue: the captured value holding N
};

// Everything shared by the frames of ONE lowering: the module (symbol lookup
// for inlined callees) and the analysis caches.  Each one is the Interpreter
// cache named beside it, and each is keyed by BLOCK for the same reason: the
// same block is walked once per enclosing analysis, and whole-model bodies are
// large.
struct LowerContext {
  mlir::ModuleOp module;
  absl::flat_hash_map<mlir::Block*, int64_t> cost;   // interp._cost_cache
  // interp._counted_cache, keyed by the COND block exactly as the Python one
  // is (no two while ops can share one).
  absl::flat_hash_map<mlir::Block*, std::optional<Counted>> counted;
  absl::flat_hash_map<mlir::Block*, int64_t> bytes;    // interp._bytes_cache
  absl::flat_hash_map<mlir::Block*, bool> pure;        // interp._pure_cache
  // interp._traceable_cache, keyed by the loop's BODY block as the Python is.
  absl::flat_hash_map<mlir::Block*, bool> traceable;
  // interp._bytes_gated: blocks the byte gate has already reported. A RECORD,
  // not an authority -- the decision depends on how many copies are asked for.
  absl::flat_hash_set<mlir::Block*> gated;
};

// interpreter.free_values: the SSA values used inside `block` but defined
// outside it, in FIRST-USE order.  The order is part of the encoding, not an
// implementation detail: captures become the region Program's trailing
// arguments in exactly this order, and a counted loop whose bound is captured
// records the bound's index into this list.
std::vector<mlir::Value> FreeValues(mlir::Block& block) {
  llvm::DenseSet<mlir::Value> defined;
  llvm::DenseSet<mlir::Value> seen;
  std::vector<mlir::Value> free;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& blk) {
    for (mlir::BlockArgument a : blk.getArguments()) defined.insert(a);
    for (mlir::Operation& op : blk) {
      for (mlir::Value v : op.getOperands()) {
        if (!defined.contains(v) && seen.insert(v).second) free.push_back(v);
      }
      for (mlir::Region& r : op.getRegions())
        for (mlir::Block& b : r.getBlocks()) walk(b);
      for (mlir::Value v : op.getResults()) defined.insert(v);
    }
  };
  walk(block);
  return free;
}

// ops/control.py `_splat_int`: the value of a rank-0 constant as an integer,
// or nothing.  Python reaches it through `int(...)` on the decoded numpy
// scalar, so a float constant truncates rather than declining -- and a NaN or
// an infinity raises there, which is this function's `nullopt`.
std::optional<int64_t> SplatInt(mlir::Operation* op) {
  if (op == nullptr) return std::nullopt;
  auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
  if (!cst) return std::nullopt;
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!t || t.getRank() != 0) return std::nullopt;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || dense.getNumElements() != 1) return std::nullopt;
  mlir::Type el = t.getElementType();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(el)) {
    if (it.getWidth() > 64) return std::nullopt;
    const llvm::APInt v = *dense.getValues<llvm::APInt>().begin();
    if (it.isUnsigned() || it.getWidth() == 1)
      return static_cast<int64_t>(v.getZExtValue());
    return v.getSExtValue();
  }
  if (mlir::isa<mlir::FloatType>(el)) {
    llvm::APFloat v = *dense.getValues<llvm::APFloat>().begin();
    if (!v.isFinite()) return std::nullopt;
    bool lost = false;
    v.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
              &lost);
    return static_cast<int64_t>(v.convertToDouble());
  }
  return std::nullopt;
}

// ops/control.py `_static_start`: the static initial value of carry `k`.
std::optional<int64_t> StaticStart(mlir::Operation* op, int64_t k) {
  if (k < 0 || k >= static_cast<int64_t>(op->getNumOperands()))
    return std::nullopt;
  return SplatInt(op->getOperand(static_cast<unsigned>(k)).getDefiningOp());
}

// ops/control.py `_cond_has_effects`: a cond region with host-visible side
// effects must take the DYNAMIC path, because the counted fast path never
// executes the cond at all.
bool CondHasEffects(LowerContext& ctx, mlir::Block& block) {
  for (mlir::Operation& o : block) {
    const llvm::StringRef n = o.getName().getStringRef();
    if (n == "stablehlo.custom_call") {
      auto attr = o.getAttrOfType<mlir::BoolAttr>("has_side_effect");
      if (attr && attr.getValue()) return true;
    }
    if (n == "func.call" || n == "stablehlo.composite") {
      auto sym = o.getAttrOfType<mlir::FlatSymbolRefAttr>(
          n == "func.call" ? "callee" : "decomposition");
      if (sym) {
        auto fn = ctx.module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
        if (fn && !fn.getBody().empty() &&
            CondHasEffects(ctx, fn.getBody().front()))
          return true;
      }
    }
    for (mlir::Region& r : o.getRegions())
      for (mlir::Block& b : r.getBlocks())
        if (CondHasEffects(ctx, b)) return true;
  }
  return false;
}

// ops/control.py `_analyze_counted`, structure for structure.  The shape jax
// emits for scan/fori_loop:
//
//   cond:  %n = constant N            (or N captured from an enclosing scope,
//                                      or carried unchanged in the state)
//          %p = compare LT, %arg_k, %n
//          return %p
//   body:  ... return ..., add(%arg_k, 1), ...
//
// Getting this WRONG in the permissive direction is not a cadence question:
// the counted path never evaluates the condition, so a loop wrongly called
// counted runs the wrong number of iterations.  Every test below is therefore
// the Python one, including the ones that look redundant.
std::optional<Counted> AnalyzeCounted(LowerContext& ctx, mlir::Operation* op) {
  if (op->getNumRegions() != 2) return std::nullopt;
  if (op->getRegion(0).getBlocks().size() != 1 ||
      op->getRegion(1).getBlocks().size() != 1)
    return std::nullopt;
  mlir::Block& cond = op->getRegion(0).front();
  auto cached = ctx.counted.find(&cond);
  if (cached != ctx.counted.end()) return cached->second;

  std::optional<Counted> result;
  do {
    if (CondHasEffects(ctx, cond)) break;
    mlir::Operation* ret = cond.getTerminator();
    if (ret == nullptr || ret->getNumOperands() < 1) break;
    auto cmp = mlir::dyn_cast_or_null<mlir::stablehlo::CompareOp>(
        ret->getOperand(0).getDefiningOp());
    if (!cmp ||
        cmp.getComparisonDirection() != mlir::stablehlo::ComparisonDirection::LT)
      break;
    auto lhs = mlir::dyn_cast<mlir::BlockArgument>(cmp->getOperand(0));
    if (!lhs || lhs.getOwner() != &cond) break;
    const int64_t k = lhs.getArgNumber();
    mlir::Value rhs = cmp->getOperand(1);
    mlir::Block& body = op->getRegion(1).front();
    mlir::Operation* body_ret = body.getTerminator();
    if (body_ret == nullptr) break;

    Counted found;
    found.k = k;
    bool have_bound = false;
    if (mlir::Operation* def = rhs.getDefiningOp()) {
      // Defined by an op: only a rank-0 integer constant is a bound.
      std::optional<int64_t> n = SplatInt(def);
      if (n.has_value()) {
        found.kind = Counted::kStatic;
        found.n = *n;
        have_bound = true;
      }
    } else if (auto ra = mlir::dyn_cast<mlir::BlockArgument>(rhs)) {
      if (ra.getOwner() == &cond) {
        // Carried in the loop state (lbfgs carries maxiter): counted only if
        // the body forwards it unchanged.
        const unsigned j = ra.getArgNumber();
        if (j < body_ret->getNumOperands() && j < body.getNumArguments() &&
            body_ret->getOperand(j) == body.getArgument(j)) {
          found.kind = Counted::kCarry;
          found.n = j;
          have_bound = true;
        }
      } else {
        // A block argument of an enclosing scope: a capture of the cond.
        found.kind = Counted::kValue;
        found.value = rhs;
        have_bound = true;
      }
    }
    if (!have_bound) break;

    // ...and the body must return arg_k + 1 at position k.
    if (k >= static_cast<int64_t>(body.getNumArguments()) ||
        k >= static_cast<int64_t>(body_ret->getNumOperands()))
      break;
    mlir::Operation* add =
        body_ret->getOperand(static_cast<unsigned>(k)).getDefiningOp();
    if (add == nullptr || add->getName().getStringRef() != "stablehlo.add" ||
        add->getNumOperands() != 2)
      break;
    const mlir::Value barg = body.getArgument(static_cast<unsigned>(k));
    bool ok = false;
    for (int swap = 0; swap < 2; swap++) {
      const mlir::Value x = add->getOperand(swap);
      const mlir::Value y = add->getOperand(1 - swap);
      if (x == barg && SplatInt(y.getDefiningOp()) == std::optional<int64_t>(1))
        ok = true;
    }
    if (ok) result = found;
  } while (false);

  ctx.counted[&cond] = result;
  return result;
}

// ops/gather.py `_scatter_combiner`: which `.at[]` method the update
// computation is, read structurally out of the region.  `kApply` is a body
// that computes something else entirely (jax's scatter_apply); it is not in
// this opcode's plan, but it IS what decides the cost of the op, so it is a
// value here rather than an error.
enum class Combiner { kSet, kAdd, kMultiply, kMaximum, kMinimum, kSubtract,
                      kApply, kNonUpdate, kBad };

// The tape's method code, in the order native/ops_index.cc switches on
// (tape.py `_SCATTER_METHODS`).  Nothing else is a method.
std::optional<int64_t> MethodCode(Combiner c) {
  switch (c) {
    case Combiner::kSet: return 0;
    case Combiner::kAdd: return 1;
    case Combiner::kMultiply: return 2;
    case Combiner::kMaximum: return 3;
    case Combiner::kMinimum: return 4;
    case Combiner::kSubtract: return 5;
    default: return std::nullopt;
  }
}

Combiner ScatterCombiner(mlir::Operation* op) {
  if (op->getNumRegions() != 1 || op->getRegion(0).getBlocks().size() != 1)
    return Combiner::kBad;
  mlir::Block& block = op->getRegion(0).front();
  std::vector<mlir::Operation*> body;
  for (mlir::Operation& o : block) body.push_back(&o);
  const std::vector<mlir::Value> args(block.getArguments().begin(),
                                      block.getArguments().end());
  auto is_return = [](mlir::Operation* o) {
    return o->getName().getStringRef() == "stablehlo.return";
  };
  if (body.size() == 1 && is_return(body[0])) {
    // `return %update`: an assignment.  Returning anything else (the OPERAND,
    // say) is a body this opcode cannot express.
    if (body[0]->getNumOperands() == 1 && args.size() == 2 &&
        body[0]->getOperand(0) == args[1])
      return Combiner::kSet;
    return Combiner::kNonUpdate;
  }
  if (body.size() == 2 && is_return(body[1])) {
    const absl::string_view n = View(body[0]->getName().getStringRef());
    Combiner c = Combiner::kBad;
    if (n == "stablehlo.add") c = Combiner::kAdd;
    else if (n == "stablehlo.multiply") c = Combiner::kMultiply;
    else if (n == "stablehlo.maximum") c = Combiner::kMaximum;
    else if (n == "stablehlo.minimum") c = Combiner::kMinimum;
    else if (n == "stablehlo.subtract") c = Combiner::kSubtract;
    // A genuine combiner takes BOTH block arguments: an f(operand)-style body
    // (scatter_apply's) must not be mistaken for one, and subtract is
    // order-sensitive where the rest are commutative.
    if (c != Combiner::kBad && args.size() == 2 &&
        body[0]->getNumOperands() == 2) {
      const mlir::Value x = body[0]->getOperand(0), y = body[0]->getOperand(1);
      if (c == Combiner::kSubtract) {
        if (x == args[0] && y == args[1]) return c;
      } else if ((x == args[0] && y == args[1]) ||
                 (x == args[1] && y == args[0])) {
        return c;
      }
    }
  }
  // Anything else built out of stablehlo/chlo ops is jax's scatter_apply.
  for (mlir::Operation* o : body) {
    const llvm::StringRef n = o->getName().getStringRef();
    if (!n.starts_with("stablehlo.") && !n.starts_with("chlo."))
      return Combiner::kBad;
  }
  return Combiner::kApply;
}

int64_t BlockCost(LowerContext& ctx, mlir::Block& block);

// ops/control.py `_scatter_cost`: a computed body with possibly-duplicate
// indices is replayed once per update (the sequential apply path in
// ops/gather.py), so charge the whole expansion rather than a single body.
// Never fails -- every uncertain answer falls back to one body, which is what
// the Python function's bare `except` does.
int64_t ScatterCost(LowerContext& ctx, mlir::Operation* op) {
  int64_t body = 0;
  for (mlir::Region& r : op->getRegions())
    for (mlir::Block& b : r.getBlocks()) body += BlockCost(ctx, b);
  auto unique = op->getAttrOfType<mlir::BoolAttr>("unique_indices");
  if (unique && unique.getValue()) return body;
  if (ScatterCombiner(op) != Combiner::kApply) return body;
  if (op->getNumOperands() < 2) return body;
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto dn = op->getAttrOfType<mlir::stablehlo::ScatterDimensionNumbersAttr>(
      "scatter_dimension_numbers");
  if (!t || !dn) return body;
  const int64_t ivd = dn.getIndexVectorDim();
  int64_t n = 1;
  for (int64_t i = 0; i < t.getRank(); i++) {
    if (i == ivd) continue;
    if (t.getShape()[i] < 0) return body;
    n *= t.getShape()[i];
  }
  return std::max<int64_t>(n, 1) * body;
}

// ops/control.py `_block_cost`: the approximate op count of this block when
// it is TRACED, loops unrolled.  It sizes the loop's flush period, so it is
// a cadence number and a wrong one is a memory-pressure question rather than
// a wrong answer -- but an UNDER-count is what once compiled a whole 256-step
// chunk into one trace and exhausted the buffer pool, so callees are recursed
// into and an unknown trip count is charged pessimistically.
int64_t BlockCost(LowerContext& ctx, mlir::Block& block) {
  auto cached = ctx.cost.find(&block);
  if (cached != ctx.cost.end()) return cached->second;
  ctx.cost[&block] = 1;   // break cycles defensively, as the Python does
  int64_t cost = 0;
  for (mlir::Operation& o : block) {
    cost += 1;
    const llvm::StringRef n = o.getName().getStringRef();
    if (n == "stablehlo.while") {
      std::optional<Counted> c = AnalyzeCounted(ctx, &o);
      // Unknown trip counts must be PESSIMISTIC: a dynamic-bound 2048-step
      // loop once cost-counted as one body, so the enclosing block passed the
      // trace budget and the whole thing was traced.
      int64_t trip = 1024;
      if (c.has_value() && c->kind == Counted::kStatic) {
        std::optional<int64_t> start = StaticStart(&o, c->k);
        if (start.has_value()) trip = std::max<int64_t>(c->n - *start, 1);
      }
      if (o.getNumRegions() >= 2 && !o.getRegion(1).empty())
        cost += trip * BlockCost(ctx, o.getRegion(1).front());
    } else if (n == "stablehlo.scatter") {
      cost += ScatterCost(ctx, &o);
    } else if (n == "func.call" || n == "stablehlo.composite") {
      auto sym = o.getAttrOfType<mlir::FlatSymbolRefAttr>(
          n == "func.call" ? "callee" : "decomposition");
      if (sym) {
        auto fn = ctx.module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
        if (fn && !fn.getBody().empty())
          cost += BlockCost(ctx, fn.getBody().front());
      }
    } else {
      for (mlir::Region& r : o.getRegions())
        for (mlir::Block& b : r.getBlocks()) cost += BlockCost(ctx, b);
    }
  }
  ctx.cost[&block] = cost;
  return cost;
}

// ops/control.py `_flush_period`: how many iterations of a body this cheap
// may run between the loop's sync points.
int64_t FlushPeriod(int64_t cost) {
  return std::max<int64_t>(
      1, std::min<int64_t>(64, 25000 / std::max<int64_t>(cost, 1)));
}

// --------------------------------------------------------------------------
// the compile decisions (P5): the byte estimate, purity, the anchors
// --------------------------------------------------------------------------
//
// `cost` above bounds how many OPS a trace may hold; these bound how much
// MEMORY it holds and whether it may be traced at all.  Every one of them is
// the Python function named beside it, walked in the same order, because a
// disagreement is not a wrong answer but something worse: the two engines
// would put a loop's sync points in different places, and where those fall is
// a ticket in MLX 0.32's command-buffer lottery
// (notes/mlx-command-buffer-split.md).  That is also why the decisions are
// worth having at all -- a 1024-step EAGER loop is exposed to that bug where
// the compiled one is not (notes/cpp-p4-gather-scatter.md).

int64_t BlockBytes(LowerContext& ctx, mlir::Block& block);
bool BlockIsPure(LowerContext& ctx, mlir::Block& block);
bool WhileTraceable(LowerContext& ctx, mlir::Operation* op);

// interpreter.op_bytes: the device bytes one operation materializes, which is
// its declared result size with one correction -- a SPLAT constant carries ONE
// value however big its type says it is (`dense<1.0> : tensor<151936x1024xf32>`
// is four bytes of IR), and `LowerConstant` broadcasts it from a one-element
// buffer rather than materializing the shape.  Charging such a constant its
// nominal size is not conservatism, it is an invented gigabyte: jax lowers a
// single `random.normal` with 23 full-shape splat coefficients.
//
// Everything else is charged its declared result size.  The views MLX produces
// without copying and the chains mx::compile fuses are deliberately NOT
// modelled -- see ops/control.py `_block_bytes` on why an over-estimate is the
// right direction for a memory gate.
int64_t OpBytes(mlir::Operation* op) {
  int64_t total = 0;
  for (mlir::Value r : op->getResults()) total += ValueBytes(r);
  if (total == 0) return 0;
  auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
  if (!cst) return total;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!dense || !t || !dense.isSplat()) return total;
  std::optional<mx::Dtype> dt = MxDtypeOf(t.getElementType());
  if (!dt.has_value()) return total;
  return static_cast<int64_t>(dt->size());
}

// ops/control.py `_block_bytes`: the device bytes this block materializes when
// it is TRACED.  Deliberately the same walk as `BlockCost` -- loops unrolled
// (pessimistic trip 1024 when the bound is not static), callees recursed into,
// every branch of an if/case charged because which one runs is data.
//
// A program's own pass-through outputs are NOT here; they belong to
// `ProgramBytes`, which is what a whole-program decision uses.
int64_t BlockBytes(LowerContext& ctx, mlir::Block& block) {
  auto cached = ctx.bytes.find(&block);
  if (cached != ctx.bytes.end()) return cached->second;
  ctx.bytes[&block] = 0;   // break cycles defensively, as the Python does
  int64_t total = 0;
  for (mlir::Operation& o : block) {
    total += OpBytes(&o);
    const llvm::StringRef n = o.getName().getStringRef();
    if (n == "stablehlo.while") {
      std::optional<Counted> c = AnalyzeCounted(ctx, &o);
      int64_t trip = 1024;
      if (c.has_value() && c->kind == Counted::kStatic) {
        std::optional<int64_t> start = StaticStart(&o, c->k);
        if (start.has_value()) trip = std::max<int64_t>(c->n - *start, 1);
      }
      if (o.getNumRegions() >= 2 && !o.getRegion(1).empty())
        total += trip * BlockBytes(ctx, o.getRegion(1).front());
    } else if (n == "func.call" || n == "stablehlo.composite") {
      auto sym = o.getAttrOfType<mlir::FlatSymbolRefAttr>(
          n == "func.call" ? "callee" : "decomposition");
      if (sym) {
        auto fn = ctx.module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
        if (fn && !fn.getBody().empty())
          total += BlockBytes(ctx, fn.getBody().front());
      }
    } else {
      for (mlir::Region& r : o.getRegions())
        for (mlir::Block& b : r.getBlocks()) total += BlockBytes(ctx, b);
    }
  }
  ctx.bytes[&block] = total;
  return total;
}

// ops/control.py `_passthrough_bytes`: the bytes of results no operation in the
// block produces.  A program whose output IS one of its inputs still
// materializes that output -- XLA's no-alias contract makes it a copy -- and
// the op walk cannot see it, because there is no op.  Counted once per RETURNED
// POSITION, not per distinct value: two outputs may not share a buffer either.
//
// Whole programs only, never inside `BlockBytes`' recursion: a loop body
// returns its carries, most of them arguments passed through untouched, and
// charging those per iteration would move the chunk sizing.
int64_t PassthroughBytes(mlir::Block& block) {
  if (block.getNumArguments() == 0) return 0;
  llvm::DenseSet<mlir::Value> args;
  for (mlir::BlockArgument a : block.getArguments()) args.insert(a);
  int64_t total = 0;
  for (mlir::Operation& o : block) {
    if (!o.getName().getStringRef().ends_with(".return") || o.getNumResults())
      continue;
    for (mlir::Value v : o.getOperands())
      if (args.contains(v)) total += ValueBytes(v);
  }
  return total;
}

int64_t ProgramBytes(LowerContext& ctx, mlir::Block& block) {
  return BlockBytes(ctx, block) + PassthroughBytes(block);
}

// ops/control.py `_bytes_ok`: whether tracing `mult` copies of this block fits
// the byte budget.  Every compile decision asks it -- the whole-main compile
// (mult 1, whole), a compiled while body (mult repeat) and unrolling a counted
// loop into an enclosing trace (mult trip) -- and all of them multiply the SAME
// per-block estimate, which is what keeps them coherent.
bool BytesOk(LowerContext& ctx, mlir::Block& block, int64_t mult,
             absl::string_view what, bool whole = false) {
  if (kCompileBytes <= 0) return true;
  const int64_t per = whole ? ProgramBytes(ctx, block) : BlockBytes(ctx, block);
  const int64_t nb = mult * per;
  if (nb <= kCompileBytes) return true;
  if (kDebug && ctx.gated.insert(&block).second) {
    std::fprintf(stderr,
                 "[metaljax-native] compile bytes gate: %s would trace %.0f MB "
                 "(x%lld) > %lld MB (METALJAX_COMPILE_BYTES_MB); running "
                 "eagerly\n",
                 std::string(what).c_str(),
                 static_cast<double>(nb) / static_cast<double>(1 << 20),
                 static_cast<long long>(mult),
                 static_cast<long long>(kCompileBytesMb));
  }
  return false;
}

// ops/control.py `_bytes_chunks`: how many copies of this block one trace may
// hold.  Never less than 1 -- the single-step case is gated by `BytesOk` where
// the body compile is decided, which says NO when one iteration alone is over
// budget.
int64_t BytesChunks(LowerContext& ctx, mlir::Block& block) {
  if (kCompileBytes <= 0) return int64_t{1} << 30;
  return std::max<int64_t>(
      1, kCompileBytes / std::max<int64_t>(BlockBytes(ctx, block), 1));
}

// interpreter.block_is_pure: true when executing the block never synchronizes
// with the HOST.  It is the gate on every tracing path, because a trace cannot
// contain a host read at all.
//
// Two of the Python's arms cannot fire here: a `custom_call` with a host
// handler and a token-carrying value both decline this plugin's lowering
// outright (the op set and `CheckValue`), long before anything asks about
// purity.  The token tests are written anyway -- a reader checks a
// transliteration line by line, and they cost a type comparison -- while the
// custom_call arm has nothing to call and is left out.
bool BlockIsPure(LowerContext& ctx, mlir::Block& block) {
  auto cached = ctx.pure.find(&block);
  if (cached != ctx.pure.end()) return cached->second;
  for (mlir::BlockArgument a : block.getArguments()) {
    // Token-carrying code sequences host-visible effects: keep it off every
    // tracing path. Such programs sync with the host anyway.
    if (mlir::isa<mlir::stablehlo::TokenType>(a.getType())) {
      ctx.pure[&block] = false;
      return false;
    }
  }
  ctx.pure[&block] = true;   // optimistic; no recursion in jax IR
  bool pure = true;
  for (mlir::Operation& o : block) {
    const llvm::StringRef n = o.getName().getStringRef();
    bool token_result = false;
    for (mlir::Value r : o.getResults())
      token_result = token_result ||
                     mlir::isa<mlir::stablehlo::TokenType>(r.getType());
    if (token_result) { pure = false; break; }
    // interpreter._IMPURE_OPS. The two host-computed linalg ops are in the
    // list because the Python handler runs them on the CPU through numpy;
    // they decline here, and are listed so the two readings agree on paper.
    if (n == "stablehlo.while" || n == "stablehlo.if" || n == "stablehlo.case" ||
        n == "stablehlo.triangular_solve" || n == "stablehlo.cholesky") {
      if (n == "stablehlo.while" && WhileTraceable(ctx, &o))
        continue;      // statically-counted small loop: unrollable
      pure = false;
      break;
    }
    if (n == "func.call" || n == "stablehlo.composite") {
      auto sym = o.getAttrOfType<mlir::FlatSymbolRefAttr>(
          n == "func.call" ? "callee" : "decomposition");
      if (sym) {
        auto fn = ctx.module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
        if (fn && !fn.getBody().empty() &&
            !BlockIsPure(ctx, fn.getBody().front())) {
          pure = false;
          break;
        }
      }
    }
    bool stop = false;
    for (mlir::Region& r : o.getRegions()) {
      for (mlir::Block& b : r.getBlocks()) {
        if (!BlockIsPure(ctx, b)) { pure = false; stop = true; break; }
      }
      if (stop) break;
    }
    if (stop) break;
  }
  ctx.pure[&block] = pure;
  return pure;
}

// ops/control.py `_while_traceable`: whether this loop can be unrolled INSIDE
// an enclosing mx::compile trace -- statically counted, pure body, and small
// enough for both budgets.  It is what lets the purity analysis see through a
// small counted loop, so a main holding one is still compiled whole.
//
// (`_msl_plan_for`'s early yes has no analogue: this plugin generates no
// kernels, which is the neutral answer the Python gives with METALJAX_MSL=0.)
bool WhileTraceable(LowerContext& ctx, mlir::Operation* op) {
  if (op->getNumRegions() < 2 || op->getRegion(1).empty()) return false;
  mlir::Block& body = op->getRegion(1).front();
  auto cached = ctx.traceable.find(&body);
  if (cached != ctx.traceable.end()) return cached->second;
  ctx.traceable[&body] = false;   // break recursion
  bool ok = false;
  std::optional<Counted> c = AnalyzeCounted(ctx, op);
  if (c.has_value() && c->kind == Counted::kStatic) {
    std::optional<int64_t> start = StaticStart(op, c->k);
    if (start.has_value()) {
      const int64_t trip = std::max<int64_t>(c->n - *start, 0);
      // The byte gate is asked LAST, so a gate that fires means "everything
      // else said compile" -- which is what makes its debug line mean
      // something.
      ok = trip * BlockCost(ctx, body) <= kTraceBudget &&
           BlockIsPure(ctx, body) &&
           BytesOk(ctx, body, trip, "unroll-in-trace");
    }
  }
  ctx.traceable[&body] = ok;
  return ok;
}

// ops/control.py `_underived_outputs`: the terminator operands with NO data
// path from any block argument or capture.  When traced, such outputs are baked
// by MLX's compiler into a constants table KEYED BY VALUE -- two equal-valued
// constant outputs collide and the compiled call dies with unordered_map::at
// (repro: mx.compile(lambda x: (x+1, mx.array(.9), mx.array(.9)))).  The
// executor anchors exactly these positions (native/compile.cc).
std::vector<int> UnderivedOutputs(mlir::Block& block,
                                  const std::vector<mlir::Value>& free) {
  std::vector<mlir::Operation*> ops;
  for (mlir::Operation& o : block) ops.push_back(&o);
  if (ops.empty()) return {};
  llvm::DenseSet<mlir::Value> derived;
  for (mlir::BlockArgument a : block.getArguments()) derived.insert(a);
  for (mlir::Value v : free) derived.insert(v);
  for (size_t i = 0; i + 1 < ops.size(); i++) {
    mlir::Operation* o = ops[i];
    bool dep = false;
    for (mlir::Value v : o->getOperands()) dep = dep || derived.contains(v);
    if (!dep) {
      for (mlir::Region& r : o->getRegions()) {
        for (mlir::Block& b : r.getBlocks()) {
          for (mlir::Value v : FreeValues(b)) {
            if (derived.contains(v)) { dep = true; break; }
          }
          if (dep) break;
        }
        if (dep) break;
      }
    }
    if (dep)
      for (mlir::Value r : o->getResults()) derived.insert(r);
  }
  std::vector<int> out;
  mlir::Operation* term = ops.back();
  for (unsigned i = 0; i < term->getNumOperands(); i++) {
    if (!derived.contains(term->getOperand(i)))
      out.push_back(static_cast<int>(i));
  }
  return out;
}

// --------------------------------------------------------------------------
// the lowering
// --------------------------------------------------------------------------

// The two aliasing taints of one slot, in the frame that owns it: which of
// the frame's ARGUMENTS its array may be, and whether it may view a constant
// the Program holds for the life of the executable.  tape.py keeps the first
// as a set rather than a flag because a region maps its outputs' taints back
// through the parent's operands -- a loop that forwards a carry untouched has
// to be recognized as its own input on the other side of the frame.
struct Taint {
  absl::flat_hash_set<int> args;
  bool cv = false;
};

// One program's tape, as text, for METALJAX_DUMP_TAPE.  Built as a tree
// rather than printed on the fly because a region's entries are lowered
// before the entry that carries them, and the dump is meant to be diffed
// against the Stage 1 lowering's, which is recorded the same way.
struct DumpNode {
  struct Entry {
    int op = 0;
    std::vector<int> ins;
    std::vector<int> outs;
    std::vector<int64_t> attrs;
    bool payload = false;
    std::vector<DumpNode> regions;
  };
  std::vector<Entry> entries;
  std::vector<int> outputs;
  std::vector<int> copies;
  int slots = 0;
  // The compile decision, when there is one to record: a `set_compile` call on
  // this Program. Printed on a line of its own so a tape that compiles nothing
  // still diffs byte for byte against P3/P4's recorded dumps.
  bool compile = false;
  std::vector<int> anchors;
  int64_t max_repeat = 0;
};

void RenderDump(const DumpNode& node, const std::string& indent,
                std::string* out) {
  auto join = [](const auto& xs) {
    std::string s;
    for (size_t i = 0; i < xs.size(); i++)
      absl::StrAppend(&s, i ? "," : "", xs[i]);
    return s;
  };
  for (const DumpNode::Entry& e : node.entries) {
    absl::StrAppend(out, indent, "[tape] ", OpcodeName(e.op), " ", join(e.ins),
                    " -> ", join(e.outs), " [", join(e.attrs), "]",
                    e.payload ? " const" : "", "\n");
    for (size_t r = 0; r < e.regions.size(); r++) {
      absl::StrAppend(out, indent, "[tape] region ", r, " {\n");
      RenderDump(e.regions[r], indent + "  ", out);
      absl::StrAppend(out, indent, "[tape] }\n");
    }
  }
  absl::StrAppend(out, indent, "[tape] outputs ", join(node.outputs),
                  " copies ", join(node.copies), " slots ", node.slots, "\n");
  if (node.compile) {
    absl::StrAppend(out, indent, "[tape] compile anchors ", join(node.anchors),
                    " max_repeat ", node.max_repeat, "\n");
  }
}

class Lowering {
 public:
  explicit Lowering(LowerContext* ctx) : ctx_(ctx) {}
  absl::StatusOr<LoweredProgram> Run(mlir::func::FuncOp fn);

 private:
  struct Pending {
    int op;
    std::vector<int> ins;
    std::vector<int> outs;
    std::vector<int64_t> attrs;
    std::optional<mx::array> payload;
    int64_t bytes = 0;
    std::vector<std::shared_ptr<Program>> regions;
    std::vector<DumpNode> region_dumps;
  };

  // One block lowered into THIS frame: the block's own arguments first, then
  // `captures` (already resolved to parent slots by the caller), then the
  // walk and the finished Program.  tape.py's `lower_block`.
  struct Built {
    std::shared_ptr<Program> program;
    std::vector<int> outputs;
    std::vector<mlir::Value> returned;
    DumpNode dump;
  };
  absl::StatusOr<Built> LowerBlock(mlir::Block& block,
                                   const std::vector<mlir::Value>& captures);
  absl::StatusOr<Built> Finish(int nargs, const std::vector<int>& outputs);

  // One region block as a sub-Program, lowered in a CHILD frame (tape.py's
  // `_region`).  `caps` are the capture slots in THIS frame, in the order
  // `free` names them; `taints` are the child's per-output taints, still in
  // the CHILD's argument numbering (the caller maps them: `MapTaint`).
  struct Region {
    std::shared_ptr<Program> program;
    std::vector<mlir::Value> free;
    std::vector<int> caps;
    std::vector<int> outputs;
    std::vector<Taint> taints;
    DumpNode dump;
  };
  absl::StatusOr<Region> LowerRegion(mlir::Block& block);

  absl::Status LowerOp(mlir::Operation* op);
  absl::Status LowerControl(mlir::Operation* op);
  absl::Status LowerWhile(mlir::Operation* op);
  absl::Status LowerBranch(mlir::Operation* op);

  // Splice a single-block callee's ops into this tape (tape.py `_inline`).
  absl::Status Inline(mlir::Operation* op, llvm::StringRef callee_attr);

  // Per-op lowerings, each returning the attribute vector the matching C++
  // handler decodes.  Named after tape.py's `_lower_*`, and in its order.
  absl::StatusOr<std::vector<int64_t>> LowerCompare(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerConvert(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerReshape(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerTranspose(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerBroadcastInDim(
      mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerSlice(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerConcatenate(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerIota(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerPad(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerDotGeneral(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerDynamicSlice(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerDynamicUpdateSlice(
      mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerShift(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerBitcastConvert(
      mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerReverse(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerPopcnt(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerFft(mlir::Operation* op);
  absl::StatusOr<std::vector<int64_t>> LowerGather(mlir::Operation* op);
  // ...and the one with no `_lower_*` to be named after: Stage 1's tape
  // declines convolution, so this layout is phase 2's own (P7).
  absl::StatusOr<std::vector<int64_t>> LowerConv(mlir::Operation* op);
  absl::Status LowerConstant(mlir::Operation* op);
  absl::Status LowerReduce(mlir::Operation* op);
  // The families that bind and emit themselves: each may hand back more than
  // one array (so the single-result path cannot serve them), and
  // reduce_window's third arm carries a sub-Program besides.
  absl::Status LowerReduceWindow(mlir::Operation* op);
  absl::Status LowerSort(mlir::Operation* op);
  absl::Status LowerTopK(mlir::Operation* op);
  absl::Status LowerRng(mlir::Operation* op);

  // tape.py `_generic_reduce_attrs`: a reduce body neither table recognizes,
  // lowered into a sub-Program the handler calls once per halving round.
  // Shared by stablehlo.reduce and reduce_window, whose window axis goes to
  // the same routine (`rank` is the WINDOW rank there, not the operand's).
  struct GenericBody {
    std::vector<int64_t> attrs;
    std::vector<int> caps;
    std::shared_ptr<Program> program;
    DumpNode dump;
  };
  absl::StatusOr<GenericBody> LowerGenericBody(size_t n,
                                               const std::vector<int64_t>& dims,
                                               mlir::Block& body, int64_t rank);
  // Scatter emits itself: its drop strategy may carry a neutral VALUE, which
  // is a payload rather than an attribute (tape.py returns one too).
  absl::Status LowerScatter(mlir::Operation* op);

  // Whether MLX may hand this op's operand array back as its result (an
  // exact no-op).  tape.py's `_is_identity`, on the ops this phase lowers.
  bool IsIdentity(absl::string_view name, mlir::Operation* op);

  absl::Status CheckValue(mlir::Value v);
  absl::StatusOr<std::vector<int64_t>> Dims(mlir::Value v);
  absl::StatusOr<int> DtypeCode(mlir::Value v);
  bool IsBoolElement(mlir::Value v);
  bool IsComplexElement(mlir::Value v);
  bool IsFloatElement(mlir::Value v);

  absl::StatusOr<int> Slot(mlir::Value v);
  int Bind(mlir::Value v);
  void Alias(mlir::Value v, int slot) { slots_[v] = slot; }
  absl::StatusOr<int> Opcode(absl::string_view name);

  void Emit(int op, std::vector<int> ins, std::vector<int> outs,
            std::vector<int64_t> attrs, std::optional<mx::array> payload,
            int64_t bytes,
            std::vector<std::shared_ptr<Program>> regions = {},
            std::vector<DumpNode> region_dumps = {}) {
    entries_.push_back(Pending{op, std::move(ins), std::move(outs),
                               std::move(attrs), std::move(payload), bytes,
                               std::move(regions), std::move(region_dumps)});
  }

  int64_t ResultBytes(mlir::Operation* op) {
    int64_t n = 0;
    for (mlir::Value r : op->getResults()) n += ValueBytes(r);
    return n;
  }

  // Slots this op's results inherit taints from, given its operands.
  void TaintResults(mlir::Operation* op, absl::string_view name,
                    const std::vector<int>& ins, const std::vector<int>& outs);

  // tape.py `_tainted`: what this frame knows about one of its own slots.
  Taint TaintOf(int slot) const {
    Taint t;
    auto it = arg_alias_.find(slot);
    if (it != arg_alias_.end()) t.args = it->second;
    t.cv = const_view_.count(slot) > 0;
    return t;
  }

  // tape.py `_region_taints`: a child's taint, restated in THIS frame.  A
  // region output that may be one of the region's own arguments may, here, be
  // whatever that argument was -- a carry init, a capture, and through those
  // an argument of main or a constant the Program holds forever.
  Taint MapTaint(const Taint& child,
                 const std::vector<int>& parent_slots) const {
    Taint out;
    out.cv = child.cv;
    for (int i : child.args) {
      if (i < 0 || i >= static_cast<int>(parent_slots.size())) continue;
      const int p = parent_slots[i];
      auto it = arg_alias_.find(p);
      if (it != arg_alias_.end())
        out.args.insert(it->second.begin(), it->second.end());
      out.cv = out.cv || const_view_.count(p) > 0;
    }
    return out;
  }

  void ApplyTaint(int slot, const Taint& t) {
    if (t.cv) const_view_.insert(slot);
    if (!t.args.empty()) arg_alias_[slot] = t.args;
  }

  // tape.py's taint rule for an op with a REGION (or one of `_TAINTING_OPS`):
  // every result inherits every operand's taints.  A body that returns one of
  // its own arguments hands an operand's array straight back -- a degenerate
  // combiner does exactly that -- and rng_bit_generator returns its state
  // operand unchanged when it consumes no blocks.  Conservative on purpose:
  // being wrong here means an output aliasing an argument across calls.
  void TaintFromAll(const std::vector<int>& ins,
                    const std::vector<int>& outs) {
    Taint t;
    for (int s : ins) {
      auto it = arg_alias_.find(s);
      if (it != arg_alias_.end()) t.args.insert(it->second.begin(),
                                                it->second.end());
      t.cv = t.cv || const_view_.count(s) > 0;
    }
    for (int s : outs) ApplyTaint(s, t);
  }

  LowerContext* ctx_;
  llvm::DenseMap<mlir::Value, int> slots_;
  int nslots_ = 0;
  std::vector<Pending> entries_;
  std::vector<std::string> calls_;   // callees currently being inlined
  // The two aliasing taints, consumed by the output-copy rule in `Run` and
  // carried across frames by `MapTaint`.  `arg_alias_` maps a slot to the
  // ARGUMENT slots of this frame whose very array object it may be.
  absl::flat_hash_map<int, absl::flat_hash_set<int>> arg_alias_;
  absl::flat_hash_set<int> const_view_;
};

absl::Status Lowering::CheckValue(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return Decline("a value that is not a ranked tensor");
  for (int64_t d : t.getShape()) {
    if (d < 0) return Decline("a dynamic dimension");
  }
  if (!TapeDtypeCode(t.getElementType()).has_value()) {
    std::optional<std::string> name = TapeElementName(t.getElementType());
    return Decline(absl::StrCat("element type ",
                                name.has_value() ? *name : "<unknown>"));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int64_t>> Lowering::Dims(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return Decline("a value that is not a ranked tensor");
  std::vector<int64_t> dims;
  for (int64_t d : t.getShape()) {
    if (d < 0) return Decline("a dynamic dimension");
    dims.push_back(d);
  }
  return dims;
}

absl::StatusOr<int> Lowering::DtypeCode(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return Decline("a value that is not a ranked tensor");
  std::optional<int> code = TapeDtypeCode(t.getElementType());
  if (!code.has_value()) {
    std::optional<std::string> name = TapeElementName(t.getElementType());
    return Decline(absl::StrCat("element type ",
                                name.has_value() ? *name : "<unknown>"));
  }
  return *code;
}

bool Lowering::IsBoolElement(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return false;
  std::optional<std::string> n = TapeElementName(t.getElementType());
  return n.has_value() && *n == "i1";
}

bool Lowering::IsComplexElement(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return false;
  std::optional<std::string> n = TapeElementName(t.getElementType());
  return n.has_value() && *n == "complex<f32>";
}

bool Lowering::IsFloatElement(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return false;
  std::optional<std::string> n = TapeElementName(t.getElementType());
  return n.has_value() && (*n == "f16" || *n == "f32" || *n == "bf16");
}

absl::StatusOr<int> Lowering::Slot(mlir::Value v) {
  auto it = slots_.find(v);
  if (it == slots_.end())
    return Decline("a value defined outside the entry block");
  return it->second;
}

int Lowering::Bind(mlir::Value v) {
  const int s = nslots_++;
  slots_[v] = s;
  return s;
}

absl::StatusOr<int> Lowering::Opcode(absl::string_view name) {
  const auto& table = OpcodeTable();
  auto it = table.find(name);
  if (it == table.end()) return Decline(absl::StrCat("op ", name));
  return it->second;
}

// --------------------------------------------------------------------------
// per-op attribute lowering (src/metaljax/tape.py `_lower_*`)
// --------------------------------------------------------------------------

absl::StatusOr<std::vector<int64_t>> Lowering::LowerCompare(
    mlir::Operation* op) {
  auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op);
  if (!cmp) return Decline("stablehlo.compare in an unexpected form");
  int64_t code;
  switch (cmp.getComparisonDirection()) {
    case mlir::stablehlo::ComparisonDirection::EQ: code = 0; break;
    case mlir::stablehlo::ComparisonDirection::NE: code = 1; break;
    case mlir::stablehlo::ComparisonDirection::LT: code = 2; break;
    case mlir::stablehlo::ComparisonDirection::LE: code = 3; break;
    case mlir::stablehlo::ComparisonDirection::GT: code = 4; break;
    case mlir::stablehlo::ComparisonDirection::GE: code = 5; break;
    default: return Decline("compare direction");
  }
  // IEEE totalOrder compares order-preserving integer keys instead of the
  // raw floats; the Python handler asks the operand's dtype, and the element
  // type answers the same question statically.
  const bool total =
      cmp.getCompareType().has_value() &&
      *cmp.getCompareType() == mlir::stablehlo::ComparisonType::TOTALORDER &&
      IsFloatElement(op->getOperand(0));
  return std::vector<int64_t>{code, total ? 1 : 0};
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerConvert(
    mlir::Operation* op) {
  ASSIGN_OR_RETURN(int code, DtypeCode(op->getResult(0)));
  // `_convert`'s complex arm: XLA's complex -> real convert keeps the real
  // part, which mx::astype alone would not do.
  const bool real_part = IsComplexElement(op->getOperand(0)) &&
                         !IsComplexElement(op->getResult(0));
  return std::vector<int64_t>{code, real_part ? 1 : 0};
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerReshape(
    mlir::Operation* op) {
  ASSIGN_OR_RETURN(std::vector<int64_t> shape, Dims(op->getResult(0)));
  std::vector<int64_t> attrs{static_cast<int64_t>(shape.size())};
  attrs.insert(attrs.end(), shape.begin(), shape.end());
  return attrs;
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerTranspose(
    mlir::Operation* op) {
  auto tr = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
  if (!tr) return Decline("stablehlo.transpose in an unexpected form");
  std::vector<int64_t> perm(tr.getPermutation().begin(),
                            tr.getPermutation().end());
  std::vector<int64_t> attrs{static_cast<int64_t>(perm.size())};
  attrs.insert(attrs.end(), perm.begin(), perm.end());
  return attrs;
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerBroadcastInDim(
    mlir::Operation* op) {
  auto bc = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op);
  if (!bc) return Decline("stablehlo.broadcast_in_dim in an unexpected form");
  std::vector<int64_t> dims(bc.getBroadcastDimensions().begin(),
                            bc.getBroadcastDimensions().end());
  ASSIGN_OR_RETURN(std::vector<int64_t> out_shape,
                          Dims(op->getResult(0)));
  ASSIGN_OR_RETURN(std::vector<int64_t> in_shape,
                          Dims(op->getOperand(0)));
  if (dims.size() != in_shape.size())
    return Decline("broadcast_dimensions rank");
  for (int64_t d : dims) {
    if (d < 0 || d >= static_cast<int64_t>(out_shape.size()))
      return Decline("broadcast dimension out of range");
  }
  // Unsorted broadcast_dimensions become a transpose first; then the operand
  // reshapes to an interim shape holding a 1 in every dim it does not name.
  std::vector<int64_t> perm(dims.size());
  for (size_t i = 0; i < perm.size(); i++) perm[i] = static_cast<int64_t>(i);
  int64_t do_transpose = 0;
  std::vector<int64_t> src = in_shape;
  std::vector<int64_t> sorted = dims;
  bool ascending = true;
  for (size_t i = 1; i < dims.size(); i++)
    if (dims[i] < dims[i - 1]) ascending = false;
  if (!ascending) {
    // Stable, because tape.py's `sorted(range(n), key=...)` is: the two
    // builders must produce the same permutation, and StableHLO's uniqueness
    // rule for broadcast_dimensions is a verifier's promise, not this file's.
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
      return dims[a] < dims[b];
    });
    for (size_t i = 0; i < perm.size(); i++) src[i] = in_shape[perm[i]];
    std::sort(sorted.begin(), sorted.end());
    do_transpose = 1;
  }
  std::vector<int64_t> interim(out_shape.size(), 1);
  for (size_t i = 0; i < sorted.size(); i++)
    interim[static_cast<size_t>(sorted[i])] = src[i];

  std::vector<int64_t> attrs{do_transpose,
                             static_cast<int64_t>(perm.size())};
  attrs.insert(attrs.end(), perm.begin(), perm.end());
  attrs.push_back(static_cast<int64_t>(out_shape.size()));
  attrs.insert(attrs.end(), interim.begin(), interim.end());
  attrs.insert(attrs.end(), out_shape.begin(), out_shape.end());
  return attrs;
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerSlice(
    mlir::Operation* op) {
  auto sl = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op);
  if (!sl) return Decline("stablehlo.slice in an unexpected form");
  std::vector<int64_t> starts(sl.getStartIndices().begin(),
                              sl.getStartIndices().end());
  std::vector<int64_t> limits(sl.getLimitIndices().begin(),
                              sl.getLimitIndices().end());
  std::vector<int64_t> strides(sl.getStrides().begin(),
                               sl.getStrides().end());
  if (starts.size() != limits.size() || starts.size() != strides.size())
    return Decline("slice attribute ranks disagree");
  std::vector<int64_t> attrs{static_cast<int64_t>(starts.size())};
  attrs.insert(attrs.end(), starts.begin(), starts.end());
  attrs.insert(attrs.end(), limits.begin(), limits.end());
  attrs.insert(attrs.end(), strides.begin(), strides.end());
  return attrs;
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerConcatenate(
    mlir::Operation* op) {
  auto cat = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op);
  if (!cat) return Decline("stablehlo.concatenate in an unexpected form");
  return std::vector<int64_t>{static_cast<int64_t>(cat.getDimension())};
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerIota(mlir::Operation* op) {
  auto io = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op);
  if (!io) return Decline("stablehlo.iota in an unexpected form");
  ASSIGN_OR_RETURN(std::vector<int64_t> shape, Dims(op->getResult(0)));
  const int64_t dim = static_cast<int64_t>(io.getIotaDimension());
  if (dim < 0 || dim >= static_cast<int64_t>(shape.size()))
    return Decline("iota dimension out of range");
  ASSIGN_OR_RETURN(int code, DtypeCode(op->getResult(0)));
  // MLX has no bool or complex arange: ramp in i32 and cast, which is what
  // the Python handler's `ramp_dt` picks for both.
  int ramp = code;
  if (IsBoolElement(op->getResult(0)) || IsComplexElement(op->getResult(0))) {
    static const int i32 = [] {
      for (const std::pair<std::string, int>& kv : dtype_codes())
        if (kv.first == "i32") return kv.second;
      return -1;
    }();
    if (i32 < 0) return Decline("iota ramp dtype");
    ramp = i32;
  }
  std::vector<int64_t> attrs{dim, ramp, code,
                             static_cast<int64_t>(shape.size())};
  attrs.insert(attrs.end(), shape.begin(), shape.end());
  return attrs;
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerPad(mlir::Operation* op) {
  auto pd = mlir::dyn_cast<mlir::stablehlo::PadOp>(op);
  if (!pd) return Decline("stablehlo.pad in an unexpected form");
  std::vector<int64_t> low(pd.getEdgePaddingLow().begin(),
                           pd.getEdgePaddingLow().end());
  std::vector<int64_t> high(pd.getEdgePaddingHigh().begin(),
                            pd.getEdgePaddingHigh().end());
  std::vector<int64_t> interior(pd.getInteriorPadding().begin(),
                                pd.getInteriorPadding().end());
  ASSIGN_OR_RETURN(std::vector<int64_t> src, Dims(op->getOperand(0)));
  const size_t rank = src.size();
  if (low.size() != rank || high.size() != rank || interior.size() != rank)
    return Decline("pad attributes do not match the operand rank");
  for (int64_t i : interior)
    if (i < 0) return Decline("negative interior padding");

  // Three (flag, vectors) groups, read in this order by the C++ handler
  // whether or not their flag is set: interior dilation, edge pads, crop.
  std::vector<int64_t> attrs;
  std::vector<int64_t> shape = src;
  bool any_interior = false;
  for (int64_t i : interior) any_interior = any_interior || i > 0;
  if (any_interior) {
    for (size_t i = 0; i < rank; i++)
      shape[i] = src[i] > 0 ? (src[i] - 1) * (interior[i] + 1) + 1 : 0;
    attrs.push_back(1);
    attrs.push_back(static_cast<int64_t>(rank));
    attrs.insert(attrs.end(), shape.begin(), shape.end());
    attrs.push_back(static_cast<int64_t>(rank));
    for (int64_t i : interior) attrs.push_back(i + 1);
  } else {
    attrs.insert(attrs.end(), {0, 0, 0});
  }

  std::vector<int64_t> plo(rank), phi(rank);
  bool any_edge = false;
  for (size_t i = 0; i < rank; i++) {
    plo[i] = low[i] > 0 ? low[i] : 0;
    phi[i] = high[i] > 0 ? high[i] : 0;
    any_edge = any_edge || plo[i] != 0 || phi[i] != 0;
  }
  if (any_edge) {
    attrs.push_back(1);
    attrs.push_back(static_cast<int64_t>(rank));
    attrs.insert(attrs.end(), plo.begin(), plo.end());
    attrs.push_back(static_cast<int64_t>(rank));
    attrs.insert(attrs.end(), phi.begin(), phi.end());
    for (size_t i = 0; i < rank; i++) shape[i] += plo[i] + phi[i];
  } else {
    attrs.insert(attrs.end(), {0, 0, 0});
  }

  bool any_negative = false;
  for (size_t i = 0; i < rank; i++)
    any_negative = any_negative || low[i] < 0 || high[i] < 0;
  if (any_negative) {
    // Negative pads CROP: the handler slices the padded array.
    attrs.push_back(1);
    attrs.push_back(static_cast<int64_t>(rank));
    for (size_t i = 0; i < rank; i++) attrs.push_back(low[i] < 0 ? -low[i] : 0);
    attrs.push_back(static_cast<int64_t>(rank));
    for (size_t i = 0; i < rank; i++)
      attrs.push_back(high[i] < 0 ? shape[i] + high[i] : shape[i]);
  } else {
    attrs.insert(attrs.end(), {0, 0, 0});
  }
  return attrs;
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerDotGeneral(
    mlir::Operation* op) {
  auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
  if (!dot) return Decline("stablehlo.dot_general in an unexpected form");
  mlir::stablehlo::DotDimensionNumbersAttr dn = dot.getDotDimensionNumbers();
  std::vector<int64_t> lb(dn.getLhsBatchingDimensions().begin(),
                          dn.getLhsBatchingDimensions().end());
  std::vector<int64_t> rb(dn.getRhsBatchingDimensions().begin(),
                          dn.getRhsBatchingDimensions().end());
  std::vector<int64_t> lc(dn.getLhsContractingDimensions().begin(),
                          dn.getLhsContractingDimensions().end());
  std::vector<int64_t> rc(dn.getRhsContractingDimensions().begin(),
                          dn.getRhsContractingDimensions().end());
  ASSIGN_OR_RETURN(std::vector<int64_t> lhs, Dims(op->getOperand(0)));
  ASSIGN_OR_RETURN(std::vector<int64_t> rhs, Dims(op->getOperand(1)));

  auto holds = [](const std::vector<int64_t>& xs, int64_t v) {
    for (int64_t x : xs) if (x == v) return true;
    return false;
  };
  std::vector<int64_t> lfree, rfree;
  for (int64_t d = 0; d < static_cast<int64_t>(lhs.size()); d++)
    if (!holds(lb, d) && !holds(lc, d)) lfree.push_back(d);
  for (int64_t d = 0; d < static_cast<int64_t>(rhs.size()); d++)
    if (!holds(rb, d) && !holds(rc, d)) rfree.push_back(d);

  std::vector<int64_t> lperm = lb;
  lperm.insert(lperm.end(), lfree.begin(), lfree.end());
  lperm.insert(lperm.end(), lc.begin(), lc.end());
  std::vector<int64_t> rperm = rb;
  rperm.insert(rperm.end(), rc.begin(), rc.end());
  rperm.insert(rperm.end(), rfree.begin(), rfree.end());
  if (lperm.size() != lhs.size() || rperm.size() != rhs.size())
    return Decline("dot_general dimension numbers do not permute the operands");

  auto pick = [](const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& dims) {
    std::vector<int64_t> out;
    for (int64_t d : dims) out.push_back(shape[d]);
    return out;
  };
  std::vector<int64_t> batch = pick(lhs, lb);
  std::vector<int64_t> m = pick(lhs, lfree);
  std::vector<int64_t> k = pick(lhs, lc);
  std::vector<int64_t> n = pick(rhs, rfree);
  std::vector<int64_t> out_shape = batch;
  out_shape.insert(out_shape.end(), m.begin(), m.end());
  out_shape.insert(out_shape.end(), n.begin(), n.end());
  ASSIGN_OR_RETURN(std::vector<int64_t> declared,
                          Dims(op->getResult(0)));
  if (declared != out_shape)
    return Decline("dot_general result shape does not follow from its dims");

  ASSIGN_OR_RETURN(int out_code, DtypeCode(op->getResult(0)));
  const mx::Dtype out_dt = dtype_of(out_code);
  ASSIGN_OR_RETURN(int l_code, DtypeCode(op->getOperand(0)));
  ASSIGN_OR_RETURN(int r_code, DtypeCode(op->getOperand(1)));

  // Which of ops/linalg._dot_general's arms runs is a static property of the
  // dtypes: 0 float matmul, 1 exact-f32 K-chunks, 2 int64 outer product, 3
  // the same in bool.
  int64_t chunk = 0, kind;
  if (is_int(out_dt) && Product(k) != 0)
    chunk = ExactF32Chunk(dtype_of(l_code), dtype_of(r_code));
  if (chunk != 0) {
    kind = 1;
  } else if (is_int(out_dt)) {
    kind = 2;
  } else if (is_bool(out_dt)) {
    kind = 3;
  } else if (is_float(out_dt) || is_complex(out_dt)) {
    kind = 0;
  } else {
    return Decline("dot_general result dtype");
  }

  std::vector<int64_t> attrs{static_cast<int64_t>(lperm.size())};
  attrs.insert(attrs.end(), lperm.begin(), lperm.end());
  attrs.push_back(static_cast<int64_t>(rperm.size()));
  attrs.insert(attrs.end(), rperm.begin(), rperm.end());
  attrs.push_back(Product(batch));
  attrs.push_back(Product(m));
  attrs.push_back(Product(k));
  attrs.push_back(Product(n));
  attrs.push_back(out_code);
  attrs.push_back(static_cast<int64_t>(out_shape.size()));
  attrs.insert(attrs.end(), out_shape.begin(), out_shape.end());
  attrs.push_back(kind);
  attrs.push_back(chunk);
  return attrs;
}

// tape.py `_lower_dynamic_slice`.  XLA CLAMPS the start indices so the window
// stays inside the operand; MLX's own slice clamps nothing, so the bounds are
// shape arithmetic resolved here and the handler builds the clip from them.
// Getting this wrong is silent wrongness rather than a crash, which is why
// the differential suite tests out-of-range indices in both directions.
absl::StatusOr<std::vector<int64_t>> Lowering::LowerDynamicSlice(
    mlir::Operation* op) {
  auto ds = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(op);
  if (!ds) return Decline("stablehlo.dynamic_slice in an unexpected form");
  if (op->getNumOperands() < 1)
    return Decline("stablehlo.dynamic_slice without an operand");
  std::vector<int64_t> sizes(ds.getSliceSizes().begin(),
                             ds.getSliceSizes().end());
  ASSIGN_OR_RETURN(std::vector<int64_t> src, Dims(op->getOperand(0)));
  if (op->getNumOperands() - 1 != sizes.size())
    return Decline("dynamic_slice index arity mismatch");
  if (src.size() != sizes.size())
    return Decline("dynamic_slice slice_sizes rank");
  std::vector<int64_t> attrs{static_cast<int64_t>(sizes.size())};
  for (size_t i = 0; i < sizes.size(); i++) attrs.push_back(src[i] - sizes[i]);
  attrs.insert(attrs.end(), sizes.begin(), sizes.end());
  return attrs;
}

absl::StatusOr<std::vector<int64_t>> Lowering::LowerDynamicUpdateSlice(
    mlir::Operation* op) {
  if (op->getNumOperands() < 2)
    return Decline("stablehlo.dynamic_update_slice without an update");
  ASSIGN_OR_RETURN(std::vector<int64_t> sizes, Dims(op->getOperand(1)));
  ASSIGN_OR_RETURN(std::vector<int64_t> src, Dims(op->getOperand(0)));
  if (op->getNumOperands() - 2 != sizes.size())
    return Decline("dynamic_update_slice index arity mismatch");
  if (src.size() != sizes.size())
    return Decline("dynamic_update_slice operand ranks disagree");
  std::vector<int64_t> attrs{static_cast<int64_t>(sizes.size())};
  for (size_t i = 0; i < sizes.size(); i++) attrs.push_back(src[i] - sizes[i]);
  return attrs;
}

// The tape's dtype code for an element-type NAME.  The dtype table is the
// runtime's, keyed by the names src/metaljax/tape.py gates on, so a handler
// that needs the code of a type no VALUE has (popcnt's unsigned twin) asks
// for it the same way tape.py's `_dtype_code` does.
std::optional<int> CodeForName(absl::string_view name) {
  static const auto* codes = [] {
    auto* m = new absl::flat_hash_map<std::string, int>();
    for (const std::pair<std::string, int>& kv : dtype_codes()) m->emplace(kv);
    return m;
  }();
  auto it = codes->find(std::string(name));
  if (it == codes->end()) return std::nullopt;
  return it->second;
}

// ops/elementwise.py `_static_splat_int`: the value of an integer SSA value
// that is statically a splat constant, possibly broadcast or reshaped to the
// operand's shape.  A shift amount almost always is one, and knowing it lets
// the guard emit one arm instead of a compare and a select.
std::optional<int64_t> StaticSplatInt(mlir::Value v) {
  while (mlir::Operation* def = v.getDefiningOp()) {
    const llvm::StringRef n = def->getName().getStringRef();
    if (n == "stablehlo.broadcast_in_dim" || n == "stablehlo.reshape" ||
        n == "stablehlo.broadcast") {
      if (def->getNumOperands() < 1) return std::nullopt;
      v = def->getOperand(0);
      continue;
    }
    auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def);
    if (!cst) return std::nullopt;
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(def->getResult(0).getType());
    if (!t) return std::nullopt;
    std::optional<mx::Dtype> dt = MxDtypeOf(t.getElementType());
    if (!dt.has_value() || !is_int(*dt)) return std::nullopt;
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    // A splat's value, or a rank-0 constant's: a real multi-valued constant
    // has no single amount, which is the Python's `return None` there.
    if (!dense || dense.getNumElements() < 1) return std::nullopt;
    if (!dense.isSplat() && t.getRank() != 0) return std::nullopt;
    auto it = mlir::dyn_cast<mlir::IntegerType>(t.getElementType());
    if (!it || it.getWidth() > 64) return std::nullopt;
    const llvm::APInt a = *dense.getValues<llvm::APInt>().begin();
    if (it.isUnsigned()) return static_cast<int64_t>(a.getZExtValue());
    return a.getSExtValue();
  }
  return std::nullopt;
}

// tape.py `_lower_shift`.  XLA defines a shift by >= the operand's bit width
// as 0 (logical/left) or the sign fill (arithmetic); Metal's shifts are
// mod-width.  Whether the amount is a compile-time splat is a pure IR
// question, answered once here; which side of the width it falls on stays in
// C++, where the operand's byte width is known.
absl::StatusOr<std::vector<int64_t>> Lowering::LowerShift(
    mlir::Operation* op) {
  if (op->getNumOperands() != 2) return Decline("a shift without an amount");
  std::optional<int64_t> c = StaticSplatInt(op->getOperand(1));
  if (c.has_value() && *c >= 0) return std::vector<int64_t>{1, *c};
  return std::vector<int64_t>{0, 0};
}

// tape.py `_lower_bitcast_convert`, the byte-multiple arm.  Which arm runs is
// a static property of the two element widths, and the other arm -- the
// 4-bit one -- cannot be reached: i4/ui4 have no dtype code, so a program
// holding one declines on its element type long before this.
absl::StatusOr<std::vector<int64_t>> Lowering::LowerBitcastConvert(
    mlir::Operation* op) {
  ASSIGN_OR_RETURN(int code, DtypeCode(op->getResult(0)));
  auto src_t =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto dst_t =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!src_t || !dst_t) return Decline("bitcast_convert on a non-tensor");
  std::optional<mx::Dtype> src = MxDtypeOf(src_t.getElementType());
  std::optional<mx::Dtype> dst = MxDtypeOf(dst_t.getElementType());
  if (!src.has_value() || !dst.has_value())
    return Decline("bitcast_convert element type");
  int64_t kind = 0;
  if (dst->size() < src->size()) kind = 1;
  else if (dst->size() > src->size()) kind = 2;
  return std::vector<int64_t>{code, kind};
}

// tape.py `_lower_reverse`: one descending take per reversed dim.  A dim of
// extent 0 or 1 is skipped (mx::take chokes on empties), and both the dims
// and their extents are static.
absl::StatusOr<std::vector<int64_t>> Lowering::LowerReverse(
    mlir::Operation* op) {
  auto rev = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(op);
  if (!rev) return Decline("stablehlo.reverse in an unexpected form");
  ASSIGN_OR_RETURN(std::vector<int64_t> shape, Dims(op->getOperand(0)));
  std::vector<int64_t> attrs{0};
  int64_t n = 0;
  for (int64_t d : rev.getDimensions()) {
    if (d < 0 || d >= static_cast<int64_t>(shape.size()))
      return Decline("reverse dimension out of range");
    if (shape[static_cast<size_t>(d)] <= 1) continue;
    attrs.push_back(d);
    attrs.push_back(shape[static_cast<size_t>(d)]);
    n++;
  }
  attrs[0] = n;
  return attrs;
}

// tape.py `_lower_popcnt`: ops/elementwise's `_as_unsigned` then SWAR, whose
// width is the operand's.  Both choices are dtype questions the element type
// answers statically, so the tape carries the answers and the C++ handler
// carries the arithmetic.
absl::StatusOr<std::vector<int64_t>> Lowering::LowerPopcnt(
    mlir::Operation* op) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!t) return Decline("popcnt on a non-tensor");
  std::optional<std::string> el = TapeElementName(t.getElementType());
  if (!el.has_value()) return Decline("popcnt element type");
  ASSIGN_OR_RETURN(int code, DtypeCode(op->getOperand(0)));
  std::string unsigned_name = *el;
  if (*el == "i8") unsigned_name = "ui8";
  else if (*el == "i16") unsigned_name = "ui16";
  else if (*el == "i32") unsigned_name = "ui32";
  else if (*el == "i64") unsigned_name = "ui64";
  else if (*el == "i1") unsigned_name = "ui8";
  if (unsigned_name.rfind("ui", 0) != 0)
    return Decline(absl::StrCat("popcnt/clz on ", *el));
  std::optional<int> ucode = CodeForName(unsigned_name);
  if (!ucode.has_value()) return Decline("popcnt unsigned dtype");
  // `_as_unsigned` VIEWS a signed operand (same bits) but CASTS a bool.
  const int64_t view = *el == "i1" ? 0 : (*el != unsigned_name ? 1 : 2);
  const bool wide = unsigned_name == "ui64";
  int64_t bits = 32;
  if (unsigned_name == "ui8") bits = 8;
  else if (unsigned_name == "ui16") bits = 16;
  else if (unsigned_name == "ui64") bits = 64;
  // A bool widens to uint8, and the handler counts over mx::bool_'s byte.
  if (*el == "i1") bits = 8;
  return std::vector<int64_t>{*ucode, view, wide ? 1 : 0, bits, code};
}

// tape.py `_lower_fft`: ops/elementwise.py `_fft`, with its two workarounds
// intact.  Both are MLX bugs the Python handler documents -- a transform of an
// empty input is the typed empty result rather than an MLX error, and a unit
// LAST length on a real transform silently DROPS the transforms over the
// remaining axes, so that case is spelled out as the identity on the DC bin
// plus an ordinary complex transform over the leading axes.  The third
// workaround, the barrier before an EAGER transform (MLX's FFT kernels can
// read an input whose producing copy is still in flight), lives in the
// handler, where the `in_trace` flag it keys on is known.
absl::StatusOr<std::vector<int64_t>> Lowering::LowerFft(mlir::Operation* op) {
  auto fft = mlir::dyn_cast<mlir::stablehlo::FftOp>(op);
  if (!fft) return Decline("stablehlo.fft in an unexpected form");
  ASSIGN_OR_RETURN(std::vector<int64_t> in_shape, Dims(op->getOperand(0)));
  ASSIGN_OR_RETURN(std::vector<int64_t> out_shape, Dims(op->getResult(0)));
  ASSIGN_OR_RETURN(int out_code, DtypeCode(op->getResult(0)));
  const int64_t rank = static_cast<int64_t>(in_shape.size());
  std::vector<int64_t> s(fft.getFftLength().begin(), fft.getFftLength().end());
  if (s.empty() || static_cast<int64_t>(s.size()) > rank)
    return Decline("fft length rank");
  std::vector<int64_t> axes;
  for (int64_t i = rank - static_cast<int64_t>(s.size()); i < rank; i++)
    axes.push_back(i);

  bool empty = Product(in_shape) == 0;
  for (int64_t d : s) empty = empty || d == 0;
  if (empty) {
    std::vector<int64_t> attrs{0, out_code,
                               static_cast<int64_t>(out_shape.size())};
    attrs.insert(attrs.end(), out_shape.begin(), out_shape.end());
    return attrs;
  }

  int64_t code = 0;
  switch (fft.getFftType()) {
    case mlir::stablehlo::FftType::FFT: code = 0; break;
    case mlir::stablehlo::FftType::IFFT: code = 1; break;
    case mlir::stablehlo::FftType::RFFT: code = 2; break;
    case mlir::stablehlo::FftType::IRFFT: code = 3; break;
  }

  if (s.back() == 1 && (code == 2 || code == 3)) {
    // The unit-length rewrite: `1` for the real axis, the leading axes
    // transformed (or not, when there are none).
    std::vector<int64_t> lead_s(s.begin(), s.end() - 1);
    std::vector<int64_t> lead_axes(axes.begin(), axes.end() - 1);
    std::vector<int64_t> attrs{code == 3 ? 2 : 3, lead_axes.empty() ? 0 : 1,
                               static_cast<int64_t>(lead_s.size())};
    attrs.insert(attrs.end(), lead_s.begin(), lead_s.end());
    attrs.push_back(static_cast<int64_t>(lead_axes.size()));
    attrs.insert(attrs.end(), lead_axes.begin(), lead_axes.end());
    return attrs;
  }

  std::vector<int64_t> attrs{1, code, static_cast<int64_t>(s.size())};
  attrs.insert(attrs.end(), s.begin(), s.end());
  attrs.push_back(static_cast<int64_t>(axes.size()));
  attrs.insert(attrs.end(), axes.begin(), axes.end());
  return attrs;
}

// --------------------------------------------------------------------------
// gather / scatter (src/metaljax/tape.py `_index_plan` / `_lower_gather` /
// `_lower_scatter`)
// --------------------------------------------------------------------------
//
// Both StableHLO ops carry the same two index sources: the COMPONENTS of the
// start-index vector (one per mapped operand dim, clamped so the whole window
// fits -- XLA's rule, and MLX's own primitives clamp nothing: gather wraps a
// negative index like `take` and reads past the end otherwise, and scatter
// does no bounds checking at all, which for a write is memory corruption
// rather than a wrong number) and the implicit BATCHING coordinates, an iota
// along the index dim paired with the operand dim.

struct IndexEntry {
  int64_t kind;   // 0 index-vector component, 1 batching iota, 2 constant zero
  int64_t a;      // component number, or the iota's length
  int64_t b;      // clamp bound, or the iota's batch position
  int64_t axis;   // the operand dim this index array is keyed to
};

struct IndexPlan {
  std::vector<IndexEntry> entries;
  std::vector<int64_t> batch_shape;
  bool split = false;
  int64_t ivd = 0;
};

absl::StatusOr<IndexPlan> BuildIndexPlan(
    int64_t ivd, const std::vector<int64_t>& op_batching,
    const std::vector<int64_t>& idx_batching,
    const std::vector<int64_t>& idx_shape,
    const std::vector<int64_t>& op_shape, const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& smap, absl::string_view name) {
  const int64_t idx_rank = static_cast<int64_t>(idx_shape.size());
  // Python reaches a negative index_vector_dim through numpy-style negative
  // indexing rather than declining; StableHLO's verifier forbids one, and
  // guessing which end it meant is exactly the kind of thing that becomes a
  // wrong answer, so it is refused here.
  if (ivd < 0 || ivd > idx_rank)
    return Decline(absl::StrCat(name, " index_vector_dim out of range"));

  IndexPlan plan;
  plan.ivd = ivd;
  std::vector<int64_t> batch_dims_idx;
  for (int64_t i = 0; i < idx_rank; i++) {
    if (i == ivd) continue;
    batch_dims_idx.push_back(i);
    plan.batch_shape.push_back(idx_shape[static_cast<size_t>(i)]);
  }
  // index_vector_dim == rank means the whole array IS one component, which is
  // the shape ops/gather.py's `index_arrays` builds too.
  plan.split = ivd != idx_rank;
  const int64_t k = plan.split ? idx_shape[static_cast<size_t>(ivd)] : 1;
  if (static_cast<int64_t>(smap.size()) != k)
    return Decline(absl::StrCat(name, " index map does not match the index "
                                      "vector"));

  const int64_t op_rank = static_cast<int64_t>(op_shape.size());
  for (size_t j = 0; j < smap.size(); j++) {
    const int64_t dim = smap[j];
    if (dim < 0 || dim >= op_rank)
      return Decline(absl::StrCat(name, " index map dim out of range"));
    const int64_t bound = std::max<int64_t>(
        op_shape[static_cast<size_t>(dim)] - sizes[static_cast<size_t>(dim)],
        0);
    plan.entries.push_back(
        IndexEntry{0, static_cast<int64_t>(j), bound, dim});
  }
  // `zip` stops at the shorter list, and so does this.
  const size_t nb = std::min(op_batching.size(), idx_batching.size());
  for (size_t i = 0; i < nb; i++) {
    const int64_t op_dim = op_batching[i], i_dim = idx_batching[i];
    int64_t pos = -1;
    for (size_t p = 0; p < batch_dims_idx.size(); p++)
      if (batch_dims_idx[p] == i_dim) pos = static_cast<int64_t>(p);
    if (pos < 0 || op_dim < 0 || op_dim >= op_rank)
      return Decline(absl::StrCat(name, " batching dims out of range"));
    plan.entries.push_back(
        IndexEntry{1, idx_shape[static_cast<size_t>(i_dim)], pos, op_dim});
  }
  absl::flat_hash_set<int64_t> axes;
  for (const IndexEntry& e : plan.entries) {
    if (!axes.insert(e.axis).second)
      return Decline(absl::StrCat(name, " indexes an operand dim twice"));
  }
  if (plan.entries.empty()) {
    // An op that maps NO index component and carries no batching dim: every
    // start index is 0, so the whole thing is a static read (or write) at the
    // origin.  Both primitives want at least one index array, so synthesize
    // the constant zero -- its clamp bound is 0, so there is nothing it could
    // ever be out of.
    if (op_shape.empty())
      return Decline(absl::StrCat(name, " on a rank-0 operand with no "
                                        "indices"));
    plan.entries.push_back(IndexEntry{2, 0, 0, 0});
  }
  return plan;
}

// tape.py `_index_attrs`: n, then n (kind, a, b, axis) quads.
void AppendIndexAttrs(const IndexPlan& plan, std::vector<int64_t>* out) {
  out->push_back(static_cast<int64_t>(plan.entries.size()));
  for (const IndexEntry& e : plan.entries) {
    out->push_back(e.kind);
    out->push_back(e.a);
    out->push_back(e.b);
    out->push_back(e.axis);
  }
}

// tape.py `_lower_gather`, as ONE mx::gather.  StableHLO's gather IS MLX's,
// modulo three static rearrangements: the coordinate vector is split into one
// index array per mapped operand dim (all of the index batch shape, which
// makes MLX's broadcast rule a no-op); `slice_sizes` crosses verbatim, since
// MLX starts an unindexed axis at 0 and an indexed one at its index --
// exactly XLA's rule, windows on indexed dims included; and the result is
// reshaped past the collapsed (extent-1) dims, then transposed into the
// offset_dims interleaving.
absl::StatusOr<std::vector<int64_t>> Lowering::LowerGather(
    mlir::Operation* op) {
  auto ga = mlir::dyn_cast<mlir::stablehlo::GatherOp>(op);
  if (!ga) return Decline("stablehlo.gather in an unexpected form");
  mlir::stablehlo::GatherDimensionNumbersAttr d = ga.getDimensionNumbers();
  std::vector<int64_t> slice_sizes(ga.getSliceSizes().begin(),
                                   ga.getSliceSizes().end());
  ASSIGN_OR_RETURN(std::vector<int64_t> op_shape, Dims(op->getOperand(0)));
  ASSIGN_OR_RETURN(std::vector<int64_t> idx_shape, Dims(op->getOperand(1)));
  ASSIGN_OR_RETURN(std::vector<int64_t> out_shape, Dims(op->getResult(0)));
  ASSIGN_OR_RETURN(int out_dt, DtypeCode(op->getResult(0)));
  if (slice_sizes.size() != op_shape.size())
    return Decline("gather slice_sizes rank mismatch");

  const int64_t out_rank = static_cast<int64_t>(out_shape.size());
  for (int64_t s : out_shape) {
    if (s == 0) {
      // The Python handler's short-circuit: nothing to gather, and the
      // decomposition mis-shapes empty batches.  The attrs stop here.
      std::vector<int64_t> attrs{1, out_dt, out_rank};
      attrs.insert(attrs.end(), out_shape.begin(), out_shape.end());
      return attrs;
    }
  }
  for (size_t i = 0; i < slice_sizes.size(); i++) {
    if (slice_sizes[i] < 0 || slice_sizes[i] > op_shape[i])
      return Decline("gather slice does not fit the operand");
  }

  std::vector<int64_t> op_batching(d.getOperandBatchingDims().begin(),
                                   d.getOperandBatchingDims().end());
  std::vector<int64_t> idx_batching(d.getStartIndicesBatchingDims().begin(),
                                    d.getStartIndicesBatchingDims().end());
  std::vector<int64_t> smap(d.getStartIndexMap().begin(),
                            d.getStartIndexMap().end());
  ASSIGN_OR_RETURN(
      IndexPlan plan,
      BuildIndexPlan(d.getIndexVectorDim(), op_batching, idx_batching,
                     idx_shape, op_shape, slice_sizes, smap, "gather"));

  absl::flat_hash_set<int64_t> collapsed(d.getCollapsedSliceDims().begin(),
                                         d.getCollapsedSliceDims().end());
  collapsed.insert(op_batching.begin(), op_batching.end());
  for (int64_t i : collapsed) {
    if (i < 0 || i >= static_cast<int64_t>(op_shape.size()) ||
        slice_sizes[static_cast<size_t>(i)] != 1)
      return Decline("gather collapses a dim whose slice is not 1");
  }
  std::vector<int64_t> mid = plan.batch_shape;
  int64_t nuncollapsed = 0;
  for (size_t i = 0; i < op_shape.size(); i++) {
    if (collapsed.contains(static_cast<int64_t>(i))) continue;
    mid.push_back(slice_sizes[i]);
    nuncollapsed++;
  }

  std::vector<int64_t> offset_dims(d.getOffsetDims().begin(),
                                   d.getOffsetDims().end());
  absl::flat_hash_set<int64_t> offset_set(offset_dims.begin(),
                                          offset_dims.end());
  std::vector<int64_t> out_batch;
  for (int64_t i = 0; i < out_rank; i++)
    if (!offset_set.contains(i)) out_batch.push_back(i);
  if (out_batch.size() != plan.batch_shape.size() ||
      static_cast<int64_t>(offset_dims.size()) != nuncollapsed)
    return Decline("gather output rank does not match its dimension numbers");
  for (int64_t i : offset_dims) {
    if (i < 0 || i >= out_rank)
      return Decline("gather output rank does not match its dimension "
                     "numbers");
  }
  std::vector<int64_t> perm(static_cast<size_t>(out_rank), 0);
  for (size_t cur = 0; cur < out_batch.size(); cur++)
    perm[static_cast<size_t>(out_batch[cur])] = static_cast<int64_t>(cur);
  for (size_t cur = 0; cur < offset_dims.size(); cur++)
    perm[static_cast<size_t>(offset_dims[cur])] =
        static_cast<int64_t>(plan.batch_shape.size() + cur);
  // What the rearrangement lands on must be what the IR declares.  Free here,
  // and it turns any future disagreement about a dimension-numbers corner
  // into a decline rather than a wrong answer.
  for (int64_t i = 0; i < out_rank; i++) {
    const int64_t p = perm[static_cast<size_t>(i)];
    // The size checks above make an out-of-range `p` unreachable; Python
    // would raise IndexError here and decline, and a C++ read past the end
    // would be undefined, so it is spelled out.
    if (p < 0 || p >= static_cast<int64_t>(mid.size()) ||
        mid[static_cast<size_t>(p)] != out_shape[static_cast<size_t>(i)])
      return Decline("gather result shape does not follow from its dims");
  }

  std::vector<int64_t> attrs{0, out_dt, out_rank};
  attrs.insert(attrs.end(), out_shape.begin(), out_shape.end());
  attrs.push_back(static_cast<int64_t>(plan.batch_shape.size()));
  attrs.insert(attrs.end(), plan.batch_shape.begin(), plan.batch_shape.end());
  attrs.push_back(plan.split ? 1 : 0);
  attrs.push_back(plan.ivd);
  attrs.push_back(static_cast<int64_t>(slice_sizes.size()));
  attrs.insert(attrs.end(), slice_sizes.begin(), slice_sizes.end());
  AppendIndexAttrs(plan, &attrs);
  attrs.push_back(static_cast<int64_t>(mid.size()));
  attrs.insert(attrs.end(), mid.begin(), mid.end());
  attrs.push_back(static_cast<int64_t>(perm.size()));
  attrs.insert(attrs.end(), perm.begin(), perm.end());
  return attrs;
}

// ops/conv.py `_convolution` -- the one lowering in this file with no tape.py
// anchor, because Stage 1 declines convolution and runs it on the Python
// ENGINE instead.  So the `kConv` attribute layout is phase 2's own; it is
// documented at the handler that reads it (native/ops_conv.cc), and the
// Python handler is the specification for the arithmetic.
//
// Everything the Python handler decides from the IR is decided here: the
// three layout permutations (input -> (N, *spatial, C_in), weight ->
// (C_out, *spatial, C_in), and the transpose back), the window attributes,
// the negative-padding rewrite, and which of the four arms runs.  What is
// left for the handler is the arithmetic and the two shape guards.
absl::StatusOr<std::vector<int64_t>> Lowering::LowerConv(mlir::Operation* op) {
  auto cv = mlir::dyn_cast<mlir::stablehlo::ConvolutionOp>(op);
  if (!cv) return Decline("stablehlo.convolution in an unexpected form");
  ASSIGN_OR_RETURN(std::vector<int64_t> lhs_shape, Dims(op->getOperand(0)));
  ASSIGN_OR_RETURN(std::vector<int64_t> rhs_shape, Dims(op->getOperand(1)));
  ASSIGN_OR_RETURN(std::vector<int64_t> out_shape, Dims(op->getResult(0)));
  ASSIGN_OR_RETURN(int out_code, DtypeCode(op->getResult(0)));

  // The result is all zeros: every output element sums an EMPTY set of
  // products.  Also where a crop lands below -- both stop the attributes
  // here, and neither lets MLX see an operand it would size differently
  // from XLA (the conv overread, native/ops_conv.cc).
  auto zeros_attrs = [&]() {
    std::vector<int64_t> attrs{1, out_code,
                               static_cast<int64_t>(out_shape.size())};
    attrs.insert(attrs.end(), out_shape.begin(), out_shape.end());
    return attrs;
  };
  bool empty = Product(lhs_shape) == 0 || Product(rhs_shape) == 0;
  for (int64_t s : out_shape) empty = empty || s == 0;
  if (empty) return zeros_attrs();

  mlir::stablehlo::ConvDimensionNumbersAttr dn = cv.getDimensionNumbers();
  std::vector<int64_t> ispatial(dn.getInputSpatialDimensions().begin(),
                                dn.getInputSpatialDimensions().end());
  std::vector<int64_t> kspatial(dn.getKernelSpatialDimensions().begin(),
                                dn.getKernelSpatialDimensions().end());
  std::vector<int64_t> ospatial(dn.getOutputSpatialDimensions().begin(),
                                dn.getOutputSpatialDimensions().end());
  const int64_t rank = static_cast<int64_t>(ispatial.size());
  if (static_cast<int64_t>(kspatial.size()) != rank ||
      static_cast<int64_t>(ospatial.size()) != rank)
    return Decline("conv dimension numbers of different ranks");

  std::vector<int64_t> lperm{dn.getInputBatchDimension()};
  lperm.insert(lperm.end(), ispatial.begin(), ispatial.end());
  lperm.push_back(dn.getInputFeatureDimension());
  std::vector<int64_t> rperm{dn.getKernelOutputFeatureDimension()};
  rperm.insert(rperm.end(), kspatial.begin(), kspatial.end());
  rperm.push_back(dn.getKernelInputFeatureDimension());
  // The transpose back: MLX hands back (N, *spatial, C_out), and the output
  // layout says where each of those belongs.
  std::vector<int64_t> operm(static_cast<size_t>(rank) + 2, 0);
  auto place = [&](int64_t at, int64_t from) {
    if (at < 0 || at >= static_cast<int64_t>(operm.size())) return false;
    operm[static_cast<size_t>(at)] = from;
    return true;
  };
  bool placed = place(dn.getOutputBatchDimension(), 0) &&
                place(dn.getOutputFeatureDimension(), rank + 1);
  for (int64_t k = 0; k < rank; k++)
    placed = placed && place(ospatial[static_cast<size_t>(k)], 1 + k);
  // A permutation the handler could not apply is a crash inside MLX (or, at
  // a repeated dim, a silently wrong layout), so it is checked here where it
  // costs nothing.
  auto is_perm = [](const std::vector<int64_t>& p, size_t n) {
    if (p.size() != n) return false;
    std::vector<bool> seen(n, false);
    for (int64_t v : p) {
      if (v < 0 || v >= static_cast<int64_t>(n) ||
          seen[static_cast<size_t>(v)])
        return false;
      seen[static_cast<size_t>(v)] = true;
    }
    return true;
  };
  if (!placed || !is_perm(lperm, lhs_shape.size()) ||
      !is_perm(rperm, rhs_shape.size()) || !is_perm(operm, out_shape.size()))
    return Decline("conv dimension numbers do not permute the operands");

  auto opt_int = [&](llvm::StringRef n, int64_t dflt) -> int64_t {
    if (auto a = op->getAttrOfType<mlir::IntegerAttr>(n)) return a.getInt();
    return dflt;
  };
  const int64_t fgc = opt_int("feature_group_count", 1);
  const int64_t bgc = opt_int("batch_group_count", 1);
  if (fgc < 1 || bgc < 1) return Decline("conv group count");

  const mx::Dtype out_dt = dtype_of(out_code);
  std::vector<int64_t> attrs{0, out_code, rank};
  auto push = [&](const std::vector<int64_t>& v) {
    attrs.push_back(static_cast<int64_t>(v.size()));
    attrs.insert(attrs.end(), v.begin(), v.end());
  };
  push(lperm);
  push(rperm);
  push(operm);
  attrs.push_back(fgc);
  attrs.push_back(bgc);

  if (rank == 0) {
    // No spatial dims: a (grouped) matmul over features.  The Python handler
    // runs it in f32 whatever the operands are, which drops the imaginary
    // part of a complex one -- so that combination declines here rather than
    // transliterating an answer neither engine could defend.
    if (is_complex(out_dt))
      return Decline("conv: complex with no spatial dimensions");
    const int64_t split = bgc > 1 ? bgc : fgc;
    const int64_t nbatch = lhs_shape[static_cast<size_t>(lperm[0])];
    const int64_t nin = lhs_shape[static_cast<size_t>(lperm[1])];
    const int64_t nout = rhs_shape[static_cast<size_t>(rperm[0])];
    if (split > 1 && (nout % split != 0 ||
                      (bgc > 1 ? nbatch : nin) % split != 0))
      return Decline("conv group count does not divide the operands");
    return attrs;
  }

  std::vector<int64_t> strides =
      OptI64List(op, "window_strides", std::vector<int64_t>(rank, 1));
  std::vector<int64_t> ldil =
      OptI64List(op, "lhs_dilation", std::vector<int64_t>(rank, 1));
  std::vector<int64_t> rdil =
      OptI64List(op, "rhs_dilation", std::vector<int64_t>(rank, 1));
  std::vector<int64_t> pad =
      OptI64List(op, "padding", std::vector<int64_t>(2 * rank, 0));
  if (static_cast<int64_t>(strides.size()) != rank ||
      static_cast<int64_t>(ldil.size()) != rank ||
      static_cast<int64_t>(rdil.size()) != rank ||
      static_cast<int64_t>(pad.size()) != 2 * rank)
    return Decline("conv window attributes do not match the spatial rank");
  for (int64_t v : strides) if (v < 1) return Decline("conv window stride");
  for (int64_t v : ldil) if (v < 1) return Decline("conv lhs dilation");
  for (int64_t v : rdil) if (v < 1) return Decline("conv rhs dilation");

  // window_reversal flips the kernel (a correlation becomes a convolution).
  // MLX's `flip` is all-or-nothing, and so is the Python handler.
  bool flip = false;
  if (mlir::Attribute rev = op->getAttr("window_reversal")) {
    std::vector<bool> bits;
    if (auto arr = mlir::dyn_cast<mlir::DenseBoolArrayAttr>(rev)) {
      for (bool b : arr.asArrayRef()) bits.push_back(b);
    } else if (auto den = mlir::dyn_cast<mlir::DenseIntElementsAttr>(rev)) {
      for (const llvm::APInt& v : den.getValues<llvm::APInt>())
        bits.push_back(!v.isZero());
    } else {
      return Decline("conv window_reversal in an unexpected form");
    }
    bool any = false, all = true;
    for (bool b : bits) { any = any || b; all = all && b; }
    if (any && !all) return Decline("conv: mixed window_reversal");
    flip = any;
  }

  // The negative-padding rewrite.  XLA pads AFTER lhs dilation, so a negative
  // pad crops the DILATED array; MLX crops the undilated operand instead (its
  // output comes out a whole dilation step short per cropped element), and
  // mx::pad has no negative widths on the integer path at all.  Each crop of
  // k elements becomes: drop q = ceil(k / dilation) OPERAND elements, which
  // removes q*dilation entries from the dilated array, and pad back the
  // excess -- which is exactly that many interior holes, i.e. zeros, on that
  // side.
  std::vector<int64_t> xs(static_cast<size_t>(rank) + 2);
  for (size_t i = 0; i < lperm.size(); i++)
    xs[i] = lhs_shape[static_cast<size_t>(lperm[i])];
  std::vector<int64_t> cstart(xs.size(), 0), cstop = xs;
  bool crop = false;
  for (int64_t a = 0; a < 2 * rank; a++) crop = crop || pad[a] < 0;
  if (crop) {
    for (int64_t a = 0; a < rank; a++) {
      const size_t u = static_cast<size_t>(a);
      const int64_t dl = ldil[u];
      const int64_t k0 = std::max<int64_t>(0, -pad[2 * u]);
      const int64_t k1 = std::max<int64_t>(0, -pad[2 * u + 1]);
      if (!k0 && !k1) continue;
      const int64_t q0 = CeilDiv(k0, dl), q1 = CeilDiv(k1, dl);
      const int64_t n = xs[u + 1];
      const int64_t start = std::min(q0, n);
      const int64_t stop = std::max<int64_t>(n - q1, 0);
      cstart[u + 1] = start;
      cstop[u + 1] = std::max(stop, start);
      if (k0) pad[2 * u] = q0 * dl - k0;
      if (k1) pad[2 * u + 1] = q1 * dl - k1;
    }
    for (size_t i = 0; i < xs.size(); i++)
      if (cstop[i] <= cstart[i]) return zeros_attrs();
  }
  std::vector<int64_t> lo(rank), hi(rank);
  for (int64_t a = 0; a < rank; a++) {
    lo[static_cast<size_t>(a)] = pad[2 * a];
    hi[static_cast<size_t>(a)] = pad[2 * a + 1];
    if (lo[static_cast<size_t>(a)] < 0 || hi[static_cast<size_t>(a)] < 0)
      return Decline("conv padding");
  }

  // Integers take the exact im2col path (MLX's convolution is float-only and
  // an f32 emulation would round); complex is four real convolutions, so it
  // uses mx::conv_general like the floats.  ...and MLX implements feature
  // groups for 1-D and 2-D only.
  const bool mx_conv = is_float(out_dt) || is_complex(out_dt);
  const int64_t mode = is_complex(out_dt) ? 2 : (mx_conv ? 0 : 1);
  const bool native_groups = mx_conv && rank <= 2;
  if (fgc > 1) {
    const int64_t cin = xs.back(), cout = rhs_shape[
        static_cast<size_t>(rperm[0])];
    if (cin % fgc != 0 || cout % fgc != 0)
      return Decline("conv feature group count does not divide the operands");
  }
  if (bgc > 1) {
    const int64_t nbatch = xs[0], cout = rhs_shape[
        static_cast<size_t>(rperm[0])];
    if (nbatch % bgc != 0 || cout % bgc != 0)
      return Decline("conv batch group count does not divide the operands");
  }

  push(strides);
  push(ldil);
  push(rdil);
  push(lo);
  push(hi);
  attrs.push_back(crop ? 1 : 0);
  push(cstart);
  push(cstop);
  attrs.push_back(flip ? 1 : 0);
  attrs.push_back(mode);
  attrs.push_back(native_groups ? 1 : 0);
  // The result shape in MLX's layout: the guard the handler measures what MLX
  // produced against, which is what keeps a window MLX sizes differently from
  // XLA a loud failure instead of a read past the end of a short buffer.
  std::vector<int64_t> want{out_shape[
      static_cast<size_t>(dn.getOutputBatchDimension())]};
  for (int64_t k = 0; k < rank; k++)
    want.push_back(out_shape[static_cast<size_t>(ospatial[
        static_cast<size_t>(k)])]);
  want.push_back(out_shape[
      static_cast<size_t>(dn.getOutputFeatureDimension())]);
  push(want);
  return attrs;
}

// ops/gather.py `_combiner_neutral`: the update value that makes the combiner
// a no-op, for the drop strategy that neutralizes rather than redirects.  The
// payload has to be the same BITS the Python engine's is -- adding a neutral
// 0 and not adding it disagree at -0.0 -- so the integer extremes are built
// from the C++ types rather than through a double.
std::optional<mx::array> CombinerNeutral(int64_t method, mx::Dtype dt) {
  if (method == 1 || method == 5) return mx::array(0, dt);   // add / subtract
  if (method == 2) return mx::array(1, dt);                  // multiply
  if (method != 3 && method != 4) return std::nullopt;
  const bool least = method == 3;              // maximum's identity is lowest
  if (dt == mx::bool_) return mx::array(!least);
  if (is_float(dt)) {
    const float inf = std::numeric_limits<float>::infinity();
    return mx::array(least ? -inf : inf, dt);
  }
  auto pick = [least](auto lo, auto hi, mx::Dtype d) {
    return least ? mx::array(lo, d) : mx::array(hi, d);
  };
  if (dt == mx::int8)
    return pick(std::numeric_limits<int8_t>::min(),
                std::numeric_limits<int8_t>::max(), dt);
  if (dt == mx::int16)
    return pick(std::numeric_limits<int16_t>::min(),
                std::numeric_limits<int16_t>::max(), dt);
  if (dt == mx::int32)
    return pick(std::numeric_limits<int32_t>::min(),
                std::numeric_limits<int32_t>::max(), dt);
  if (dt == mx::int64)
    return pick(std::numeric_limits<int64_t>::min(),
                std::numeric_limits<int64_t>::max(), dt);
  if (dt == mx::uint8)
    return pick(std::numeric_limits<uint8_t>::min(),
                std::numeric_limits<uint8_t>::max(), dt);
  if (dt == mx::uint16)
    return pick(std::numeric_limits<uint16_t>::min(),
                std::numeric_limits<uint16_t>::max(), dt);
  if (dt == mx::uint32)
    return pick(std::numeric_limits<uint32_t>::min(),
                std::numeric_limits<uint32_t>::max(), dt);
  if (dt == mx::uint64)
    return pick(std::numeric_limits<uint64_t>::min(),
                std::numeric_limits<uint64_t>::max(), dt);
  return std::nullopt;
}

// tape.py `_lower_scatter`, as one mx::scatter/_add/_prod/_max/_min.  The
// combiner picks the primitive; everything else is index and update
// preprocessing, resolved here:
//
//  * the start vector splits into one clamped index array per mapped operand
//    dim, plus an iota per batching dim (`BuildIndexPlan`);
//  * the updates are transposed into [index batch dims, window dims in
//    OPERAND order] and reshaped to insert an extent-1 axis for every
//    inserted window dim -- MLX wants `updates.ndim == indices.ndim +
//    operand.ndim`, and its update slice starts at the index on an indexed
//    axis and at 0 elsewhere, which is XLA's window rule for the windowed
//    dims and for the partial windows on free dims alike;
//  * XLA's OOB-DROP semantics, of which MLX has none, through the same two
//    strategies ops/gather.py picks between, chosen from the same static
//    sizes so the two engines cannot pick differently.
absl::Status Lowering::LowerScatter(mlir::Operation* op) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return Decline("variadic scatter");
  if (IsComplexElement(op->getOperand(0))) {
    // MLX has no complex GPU scatter kernels at all: the Python handler
    // scatters the two parts separately and recombines them, which is a
    // different composition from the single primitive this entry calls.
    return Decline("scatter on complex");
  }
  ASSIGN_OR_RETURN(std::vector<int64_t> op_shape, Dims(op->getOperand(0)));
  ASSIGN_OR_RETURN(std::vector<int64_t> idx_shape, Dims(op->getOperand(1)));
  ASSIGN_OR_RETURN(std::vector<int64_t> upd_shape, Dims(op->getOperand(2)));
  auto sc = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(op);
  if (!sc) return Decline("stablehlo.scatter in an unexpected form");
  mlir::stablehlo::ScatterDimensionNumbersAttr d =
      sc.getScatterDimensionNumbers();

  const Combiner combiner = ScatterCombiner(op);
  std::optional<int64_t> method = MethodCode(combiner);
  if (!method.has_value()) {
    if (combiner == Combiner::kApply) return Decline("scatter combiner apply");
    if (combiner == Combiner::kNonUpdate)
      return Decline("scatter body: scatter body returns non-update value");
    return Decline("scatter body: not implemented");
  }

  absl::flat_hash_set<int64_t> inserted(d.getInsertedWindowDims().begin(),
                                        d.getInsertedWindowDims().end());
  std::vector<int64_t> op_batching(d.getInputBatchingDims().begin(),
                                   d.getInputBatchingDims().end());
  inserted.insert(op_batching.begin(), op_batching.end());
  std::vector<int64_t> uwd(d.getUpdateWindowDims().begin(),
                           d.getUpdateWindowDims().end());
  std::vector<int64_t> window_dims;
  for (size_t i = 0; i < op_shape.size(); i++)
    if (!inserted.contains(static_cast<int64_t>(i)))
      window_dims.push_back(static_cast<int64_t>(i));
  if (uwd.size() != window_dims.size())
    return Decline("scatter window rank mismatch");
  for (int64_t w : uwd) {
    if (w < 0 || w >= static_cast<int64_t>(upd_shape.size()))
      return Decline("scatter update_window_dims out of range");
  }
  absl::flat_hash_map<int64_t, int64_t> uwd_of;   // operand dim -> update axis
  for (size_t i = 0; i < window_dims.size(); i++) uwd_of[window_dims[i]] = uwd[i];

  // The update slice, per operand dim: an inserted window dim has an implicit
  // extent of 1, every other dim takes its update axis.
  std::vector<int64_t> sshape;
  for (size_t i = 0; i < op_shape.size(); i++) {
    const int64_t dim = static_cast<int64_t>(i);
    sshape.push_back(inserted.contains(dim)
                         ? 1
                         : upd_shape[static_cast<size_t>(uwd_of[dim])]);
    if (sshape.back() <= 0 || sshape.back() > op_shape[i])
      return Decline("scatter window does not fit the operand");
  }

  std::vector<int64_t> idx_batching(d.getScatterIndicesBatchingDims().begin(),
                                    d.getScatterIndicesBatchingDims().end());
  std::vector<int64_t> smap(d.getScatterDimsToOperandDims().begin(),
                            d.getScatterDimsToOperandDims().end());
  ASSIGN_OR_RETURN(
      IndexPlan plan,
      BuildIndexPlan(d.getIndexVectorDim(), op_batching, idx_batching,
                     idx_shape, op_shape, sshape, smap, "scatter"));

  absl::flat_hash_set<int64_t> uwd_set(uwd.begin(), uwd.end());
  std::vector<int64_t> uperm;
  for (size_t i = 0; i < upd_shape.size(); i++)
    if (!uwd_set.contains(static_cast<int64_t>(i)))
      uperm.push_back(static_cast<int64_t>(i));
  for (int64_t dim : window_dims) uperm.push_back(uwd_of[dim]);
  {
    std::vector<int64_t> got, want = plan.batch_shape;
    for (int64_t p : uperm) got.push_back(upd_shape[static_cast<size_t>(p)]);
    for (int64_t dim : window_dims)
      want.push_back(sshape[static_cast<size_t>(dim)]);
    if (got != want)
      return Decline("scatter updates do not match its dimension numbers");
  }
  std::vector<int64_t> ushape = plan.batch_shape;
  ushape.insert(ushape.end(), sshape.begin(), sshape.end());

  // XLA drops an update whose start is out of bounds in ANY component, and
  // only the mapped components can be (a batching iota never is).
  std::optional<mx::array> payload;
  int64_t strategy = 0;
  std::vector<int64_t> extra;
  if (!smap.empty()) {
    int64_t op_numel = 1, upd_numel = 1;
    for (int64_t s : op_shape) op_numel *= s;
    for (int64_t s : upd_shape) upd_numel *= s;
    if (combiner == Combiner::kSet || op_numel < upd_numel) {
      strategy = 2;
      // Pad the LOWEST indexed axis, which is the one the Python handler pads
      // (it transposes that dim to the front and concatenates there).  Any of
      // them would do; agreeing keeps the two engines' allocation shapes
      // comparable.
      size_t pos = 0;
      for (size_t i = 1; i < plan.entries.size(); i++)
        if (plan.entries[i].axis < plan.entries[pos].axis) pos = i;
      const int64_t axis = plan.entries[pos].axis;
      extra = {static_cast<int64_t>(pos), sshape[static_cast<size_t>(axis)],
               op_shape[static_cast<size_t>(axis)]};
    } else {
      strategy = 1;
      auto t =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
      std::optional<mx::Dtype> dt =
          t ? MxDtypeOf(t.getElementType()) : std::nullopt;
      if (!dt.has_value()) return Decline("scatter neutral: element type");
      payload = CombinerNeutral(*method, *dt);
      if (!payload.has_value())
        return Decline("scatter neutral: no neutral for this combiner");
      // The mask broadcasts against the updates: one axis per batch dim, then
      // an extent-1 axis per operand dim.
      extra.push_back(static_cast<int64_t>(plan.batch_shape.size() +
                                           op_shape.size()));
      extra.insert(extra.end(), plan.batch_shape.begin(),
                   plan.batch_shape.end());
      extra.insert(extra.end(), op_shape.size(), 1);
    }
  }

  // `strategy` rides second, ahead of everything variable-length, so a reader
  // (the C++ handler, a test) can see which drop rule this scatter took
  // without walking the whole vector.
  std::vector<int64_t> attrs{*method, strategy,
                             static_cast<int64_t>(plan.batch_shape.size())};
  attrs.insert(attrs.end(), plan.batch_shape.begin(), plan.batch_shape.end());
  attrs.push_back(plan.split ? 1 : 0);
  attrs.push_back(plan.ivd);
  AppendIndexAttrs(plan, &attrs);
  attrs.push_back(static_cast<int64_t>(sshape.size()));
  attrs.insert(attrs.end(), sshape.begin(), sshape.end());
  attrs.push_back(static_cast<int64_t>(uperm.size()));
  attrs.insert(attrs.end(), uperm.begin(), uperm.end());
  attrs.push_back(static_cast<int64_t>(ushape.size()));
  attrs.insert(attrs.end(), ushape.begin(), ushape.end());
  attrs.insert(attrs.end(), extra.begin(), extra.end());

  ASSIGN_OR_RETURN(int opcode, Opcode("stablehlo.scatter"));
  std::vector<int> ins;
  for (mlir::Value v : op->getOperands()) {
    ASSIGN_OR_RETURN(int s, Slot(v));
    ins.push_back(s);
  }
  std::vector<int> outs{Bind(op->getResult(0))};
  Emit(opcode, std::move(ins), std::move(outs), std::move(attrs),
       std::move(payload), ResultBytes(op));
  return absl::OkStatus();
}

absl::Status Lowering::LowerConstant(mlir::Operation* op) {
  auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
  if (!cst) return Decline("stablehlo.constant in an unexpected form");
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!type) return Decline("a constant that is not a ranked tensor");
  RETURN_IF_ERROR(CheckValue(op->getResult(0)));
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense)
    return Decline("stablehlo.constant whose value is not a dense attribute");

  const mx::Dtype dt = *MxDtypeOf(type.getElementType());
  mx::Shape shape;
  int64_t numel = 1;
  for (int64_t d : type.getShape()) {
    if (d < 0) return Decline("a constant with a dynamic dimension");
    shape.push_back(static_cast<mx::ShapeElem>(d));
    numel *= d;
  }
  const size_t item = dt.size();

  mx::array value = mx::zeros(shape, dt);
  if (dt == mx::bool_) {
    // i1 elements are BIT-packed in the raw data, splat or not; the typed
    // iterator is the only honest way to read them.
    std::vector<char> bytes(static_cast<size_t>(numel), 0);
    int64_t i = 0;
    for (bool b : dense.getValues<bool>()) {
      if (i >= numel) break;
      bytes[static_cast<size_t>(i++)] = b ? 1 : 0;
    }
    if (i != numel) return Decline("a bool constant of the wrong length");
    value = OwnedArray(bytes.data(), bytes.size(), shape, dt);
  } else if (numel > 1 && dense.isSplat()) {
    // One value, broadcast from a ONE-ELEMENT buffer -- never materialized,
    // which is what keeps a splat's device cost independent of its shape
    // (ops/elementwise.py `_constant`, and the note there on the 127 GB of
    // retained splat coefficients that made it necessary).
    llvm::ArrayRef<char> raw = dense.getRawData();
    if (raw.size() != item)
      return Decline("a splat constant whose raw element is the wrong size");
    const mx::Shape unit(shape.size(), 1);
    value = mx::broadcast_to(
        mx::reshape(OwnedArray(raw.data(), item, mx::Shape{1}, dt), unit),
        shape);
  } else {
    // Whatever is left holds at most one element (a splat of one, or a rank-0
    // value) or is a genuine dense blob, and in both cases the raw data is
    // the elements themselves -- byte for byte what the device wants, which
    // is why bf16 needs no decoding here at all.
    llvm::ArrayRef<char> raw = dense.getRawData();
    if (raw.size() != item * static_cast<size_t>(numel))
      return Decline("a constant whose raw data is the wrong size");
    value = OwnedArray(raw.data(), raw.size(), shape, dt);
  }

  const int op_code = OpcodeTable().at("stablehlo.constant");
  const int out = Bind(op->getResult(0));
  const_view_.insert(out);
  Emit(op_code, {}, {out}, {}, std::move(value), ResultBytes(op));
  return absl::OkStatus();
}

absl::Status Lowering::LowerReduce(mlir::Operation* op) {
  auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
  if (!red) return Decline("stablehlo.reduce in an unexpected form");
  const size_t n = op->getNumOperands() / 2;
  if (n * 2 != op->getNumOperands()) return Decline("reduce operand arity");
  std::vector<int64_t> dims(red.getDimensions().begin(),
                            red.getDimensions().end());
  if (red.getBody().getBlocks().size() != 1)
    return Decline("a reduce with a multi-block body");
  mlir::Block& body = red.getBody().front();
  std::vector<mlir::Operation*> body_ops;
  for (mlir::Operation& o : body) body_ops.push_back(&o);

  std::vector<int> ins;
  for (mlir::Value v : op->getOperands()) {
    ASSIGN_OR_RETURN(int s, Slot(v));
    ins.push_back(s);
  }

  // The order of the tests is the Python handler's: the single-operand
  // monoid first, then the (values, indices) pair jax lowers argmax/argmin
  // to.  Anything else is `_generic_reduce`, which runs the body block on
  // whole arrays -- a sub-Program this phase does not build, so it declines.
  if (n == 1 && body_ops.size() == 2) {
    std::optional<int64_t> kind =
        ReduceKind(View(body_ops[0]->getName().getStringRef()),
                   IsBoolElement(op->getOperand(0)));
    if (kind.has_value()) {
      if (op->getNumResults() != 1)
        return Decline("a monoid reduce with several results");
      std::vector<int64_t> attrs{*kind, static_cast<int64_t>(dims.size())};
      attrs.insert(attrs.end(), dims.begin(), dims.end());
      const int out = Bind(op->getResult(0));
      Emit(OpcodeTable().at("stablehlo.reduce"), std::move(ins), {out},
           std::move(attrs), std::nullopt, ResultBytes(op));
      return absl::OkStatus();
    }
  }

  if (n == 2 && dims.size() == 1 && !IsBoolElement(op->getOperand(0))) {
    std::optional<mlir::stablehlo::ComparisonDirection> first;
    for (mlir::Operation* o : body_ops) {
      if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(o)) {
        first = cmp.getComparisonDirection();
        break;
      }
    }
    if (first.has_value()) {
      const bool is_max =
          *first == mlir::stablehlo::ComparisonDirection::GT ||
          *first == mlir::stablehlo::ComparisonDirection::GE;
      const bool is_min =
          *first == mlir::stablehlo::ComparisonDirection::LT ||
          *first == mlir::stablehlo::ComparisonDirection::LE;
      if (is_max || is_min) {
        if (op->getNumResults() != 2)
          return Decline("an argmax-pair reduce with the wrong result count");
        std::vector<int> outs{Bind(op->getResult(0)), Bind(op->getResult(1))};
        Emit(OpcodeTable().at("stablehlo.reduce.arg_pair"), std::move(ins),
             std::move(outs), {is_max ? 1 : 0, dims[0]}, std::nullopt,
             ResultBytes(op));
        return absl::OkStatus();
      }
    }
  }
  // _generic_reduce: any associative body, any arity.  The reduced dims move
  // to one trailing axis and the body combines the two halves of it until one
  // element is left, then folds the init in.  Everything the Python version
  // derives from the arrays is static here; the BODY is the one thing that is
  // not, so it becomes a sub-Program.
  if (op->getNumResults() != n) return Decline("generic reduce result count");
  ASSIGN_OR_RETURN(std::vector<int64_t> src, Dims(op->getOperand(0)));
  ASSIGN_OR_RETURN(GenericBody gb,
                   LowerGenericBody(n, dims, body,
                                    static_cast<int64_t>(src.size())));
  ins.insert(ins.end(), gb.caps.begin(), gb.caps.end());
  std::vector<int> outs;
  for (mlir::Value r : op->getResults()) outs.push_back(Bind(r));
  TaintFromAll(ins, outs);
  std::vector<DumpNode> dumps;
  if (kDumpTape) dumps.push_back(std::move(gb.dump));
  ASSIGN_OR_RETURN(int opcode, Opcode("stablehlo.reduce.generic"));
  Emit(opcode, std::move(ins), std::move(outs), std::move(gb.attrs),
       std::nullopt, ResultBytes(op), {std::move(gb.program)},
       std::move(dumps));
  return absl::OkStatus();
}

// tape.py `_generic_reduce_attrs`.  The body's own arguments are the 2n
// (accumulator, element) pairs; the values it reads from enclosing scopes ride
// as extra operands after the op's, which is where the handler looks for them.
absl::StatusOr<Lowering::GenericBody> Lowering::LowerGenericBody(
    size_t n, const std::vector<int64_t>& dims, mlir::Block& body,
    int64_t rank) {
  for (int64_t d : dims)
    if (d < 0 || d >= rank) return Decline("reduce dimension out of range");
  std::vector<int64_t> keep;
  for (int64_t i = 0; i < rank; i++)
    if (std::find(dims.begin(), dims.end(), i) == dims.end()) keep.push_back(i);
  if (body.getNumArguments() != 2 * n) return Decline("reduce body arity");
  ASSIGN_OR_RETURN(Region region, LowerRegion(body));
  if (region.outputs.size() != n) return Decline("reduce body result count");

  GenericBody out;
  out.attrs.push_back(static_cast<int64_t>(n));
  out.attrs.push_back(static_cast<int64_t>(keep.size()));
  out.attrs.insert(out.attrs.end(), keep.begin(), keep.end());
  out.attrs.push_back(static_cast<int64_t>(dims.size()));
  out.attrs.insert(out.attrs.end(), dims.begin(), dims.end());
  out.attrs.push_back(static_cast<int64_t>(region.caps.size()));
  out.caps = std::move(region.caps);
  out.program = std::move(region.program);
  out.dump = std::move(region.dump);
  return out;
}

// tape.py `_lower_reduce_window`: ops/reduction.py `_reduce_window` with its
// three arms resolved.  The cum-op peephole (jax lowers cumsum and friends as
// a full-width window with prefix padding), the windowed reduction over an
// as_strided view, and -- where the body is neither a monoid nor the single
// compare of select_and_gather_add -- the generic pairwise reduce over the
// window axis, whose body becomes a sub-Program.
absl::Status Lowering::LowerReduceWindow(mlir::Operation* op) {
  auto rw = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(op);
  if (!rw) return Decline("stablehlo.reduce_window in an unexpected form");
  const size_t n = op->getNumOperands() / 2;
  if (n * 2 != op->getNumOperands())
    return Decline("reduce_window operand arity");
  if (op->getNumResults() != n) return Decline("reduce_window result count");
  ASSIGN_OR_RETURN(std::vector<int64_t> src, Dims(op->getOperand(0)));
  const int64_t rank = static_cast<int64_t>(src.size());
  const std::vector<int64_t> ones(static_cast<size_t>(rank), 1);
  std::vector<int64_t> wd = OptI64List(op, "window_dimensions", ones);
  std::vector<int64_t> strides = OptI64List(op, "window_strides", ones);
  std::vector<int64_t> bdil = OptI64List(op, "base_dilations", ones);
  std::vector<int64_t> wdil = OptI64List(op, "window_dilations", ones);
  std::vector<std::pair<int64_t, int64_t>> pad(static_cast<size_t>(rank),
                                               {0, 0});
  if (auto pa = op->getAttrOfType<mlir::DenseIntElementsAttr>("padding")) {
    std::vector<int64_t> flat;
    for (const llvm::APInt& v : pa.getValues<llvm::APInt>())
      flat.push_back(v.getSExtValue());
    if (static_cast<int64_t>(flat.size()) != 2 * rank)
      return Decline("reduce_window padding rank");
    for (int64_t i = 0; i < rank; i++)
      pad[static_cast<size_t>(i)] = {flat[2 * i], flat[2 * i + 1]};
  }
  if (static_cast<int64_t>(wd.size()) != rank ||
      static_cast<int64_t>(strides.size()) != rank ||
      static_cast<int64_t>(bdil.size()) != rank ||
      static_cast<int64_t>(wdil.size()) != rank)
    return Decline("reduce_window attribute rank");
  for (int64_t s : strides)
    if (s < 1) return Decline("reduce_window stride or dilation");
  for (int64_t d : wdil)
    if (d < 1) return Decline("reduce_window stride or dilation");
  if (rw.getBody().getBlocks().size() != 1)
    return Decline("a reduce_window with a multi-block body");
  mlir::Block& body = rw.getBody().front();
  std::vector<mlir::Operation*> body_ops;
  for (mlir::Operation& o : body) body_ops.push_back(&o);

  std::vector<int> ins;
  for (mlir::Value v : op->getOperands()) {
    ASSIGN_OR_RETURN(int s, Slot(v));
    ins.push_back(s);
  }
  auto emit = [&](std::vector<int64_t> attrs,
                  std::vector<std::shared_ptr<Program>> regions,
                  std::vector<DumpNode> dumps,
                  bool taint_all) -> absl::Status {
    std::vector<int> outs;
    for (mlir::Value r : op->getResults()) outs.push_back(Bind(r));
    if (taint_all) TaintFromAll(ins, outs);
    ASSIGN_OR_RETURN(int opcode, Opcode("stablehlo.reduce_window"));
    Emit(opcode, ins, std::move(outs), std::move(attrs), std::nullopt,
         ResultBytes(op), std::move(regions), std::move(dumps));
    return absl::OkStatus();
  };

  // The cumulative pattern, recognized exactly as the handler does: one axis
  // of full width, padded on one side by width-1, everything else trivial.
  if (n == 1 && body_ops.size() == 2) {
    std::optional<int64_t> cum =
        CumKind(View(body_ops[0]->getName().getStringRef()));
    bool trivial = true;
    for (int64_t s : strides) trivial = trivial && s == 1;
    for (int64_t d : bdil) trivial = trivial && d == 1;
    for (int64_t d : wdil) trivial = trivial && d == 1;
    if (cum.has_value() && trivial) {
      std::vector<int64_t> big;
      for (int64_t i = 0; i < rank; i++)
        if (wd[static_cast<size_t>(i)] > 1) big.push_back(i);
      if (big.size() == 1) {
        const size_t ax = static_cast<size_t>(big[0]);
        const int64_t size = src[ax];
        bool others = true;
        for (int64_t i = 0; i < rank; i++) {
          if (static_cast<size_t>(i) == ax) continue;
          others = others && wd[static_cast<size_t>(i)] == 1 &&
                   pad[static_cast<size_t>(i)].first == 0 &&
                   pad[static_cast<size_t>(i)].second == 0;
        }
        if (others && wd[ax] == size) {
          if (pad[ax].first == size - 1 && pad[ax].second == 0)
            return emit({0, *cum, big[0], 0}, {}, {}, false);
          if (pad[ax].first == 0 && pad[ax].second == size - 1)
            return emit({0, *cum, big[0], 1}, {}, {}, false);
        }
      }
    }
  }

  for (size_t i = 1; i < n; i++) {
    ASSIGN_OR_RETURN(std::vector<int64_t> other, Dims(op->getOperand(i)));
    if (other != src) return Decline("reduce_window inputs of different shapes");
  }
  ASSIGN_OR_RETURN(WindowPlanOut plan,
                   BuildWindowPlan(rank, src, wd, strides, bdil, wdil, pad));
  std::vector<int64_t> attrs{1, static_cast<int64_t>(n)};
  attrs.insert(attrs.end(), plan.attrs.begin(), plan.attrs.end());

  if (n >= 2) {
    // select_and_gather_add: one compare on the first pair plus selects picks
    // a single window element for every output.
    std::vector<mlir::stablehlo::CompareOp> cmps;
    size_t nsel = 0;
    for (mlir::Operation* o : body_ops) {
      if (auto c = mlir::dyn_cast<mlir::stablehlo::CompareOp>(o))
        cmps.push_back(c);
      if (o->getName().getStringRef() == "stablehlo.select") nsel++;
    }
    if (cmps.size() == 1 && nsel >= n) {
      const auto d = cmps[0].getComparisonDirection();
      const bool is_max = d == mlir::stablehlo::ComparisonDirection::GE ||
                          d == mlir::stablehlo::ComparisonDirection::GT;
      attrs.push_back(1);
      attrs.push_back(is_max ? 1 : 0);
      return emit(std::move(attrs), {}, {}, false);
    }
  } else if (body_ops.size() == 2) {
    std::optional<int64_t> kind =
        ReduceKind(View(body_ops[0]->getName().getStringRef()),
                   IsBoolElement(op->getOperand(0)));
    if (kind.has_value()) {
      attrs.push_back(0);
      attrs.push_back(*kind);
      return emit(std::move(attrs), {}, {}, false);
    }
  }

  // Everything else folds the window axis with the body itself.  What that
  // reduce sees is the extracted window VIEW, so its rank is the output rank
  // plus the one flattened window axis, and the reduced dim is that axis.
  const int64_t wrank = static_cast<int64_t>(plan.out_sizes.size()) + 1;
  ASSIGN_OR_RETURN(
      GenericBody gb,
      LowerGenericBody(n, {static_cast<int64_t>(plan.out_sizes.size())}, body,
                       wrank));
  attrs.push_back(2);
  attrs.insert(attrs.end(), gb.attrs.begin(), gb.attrs.end());
  ins.insert(ins.end(), gb.caps.begin(), gb.caps.end());
  std::vector<DumpNode> dumps;
  if (kDumpTape) dumps.push_back(std::move(gb.dump));
  return emit(std::move(attrs), {std::move(gb.program)}, std::move(dumps),
              true);
}

// ops/sort.py `_sort`, the arm whose comparator ends in a compare -- which is
// every sort jax emits except the two lexicographic select trees.  Two shapes
// reach here and both are handled: the comparator's two sides ARE the (lhs,
// rhs) block-argument pair (an integer sort, and the `sort(values, iota)` a
// top_k decomposes to), in which case the sort key is an operand and the entry
// is exactly tape.py's; or the sides compute the same KEY function of their
// argument (jax's float canonicalization: -0 -> +0, NaN -> canonical qNaN,
// then a TOTALORDER compare), in which case that chain is lowered into THIS
// frame -- it is scalar elementwise code, so it computes the key of the whole
// operand -- and the entry keys on its output.
absl::Status Lowering::LowerSort(mlir::Operation* op) {
  auto sort = mlir::dyn_cast<mlir::stablehlo::SortOp>(op);
  if (!sort) return Decline("stablehlo.sort in an unexpected form");
  const int64_t dim = sort.getDimension();
  if (sort.getComparator().getBlocks().size() != 1)
    return Decline("a sort with a multi-block comparator");
  mlir::Block& block = sort.getComparator().front();
  std::vector<mlir::Operation*> body;
  for (mlir::Operation& o : block) body.push_back(&o);
  if (body.empty()) return Decline("a sort with an empty comparator");
  mlir::Operation* ret = body.back();
  if (ret->getName().getStringRef() != "stablehlo.return" ||
      ret->getNumOperands() != 1)
    return Decline("sort: comparator must return one value");
  mlir::Value cmp_val = ret->getOperand(0);
  mlir::Operation* cmp = cmp_val.getDefiningOp();
  if (cmp == nullptr || cmp->getBlock() != &block)
    return Decline("sort: comparator returns an argument");
  auto cmp_op = mlir::dyn_cast<mlir::stablehlo::CompareOp>(cmp);
  if (!cmp_op) {
    // The two lexicographic select trees.  Both mean a DIFFERENT execution
    // shape -- successive stable argsorts threaded through a permutation, and
    // for complex a (re, im) key packed into one u64 -- which the sort entry
    // cannot express: it computes one argsort and gathers with it, and there
    // is no opcode that gathers by a permutation the tape computed.  Named
    // rather than approximated (see notes/cpp-p6-tail.md).
    const std::vector<unsigned> deps = ArgDeps(cmp_val, block);
    std::vector<unsigned> keys;
    for (unsigned d : deps)
      if (keys.empty() || keys.back() != d / 2) keys.push_back(d / 2);
    if (keys.size() == 1 && keys[0] < op->getNumOperands() &&
        IsComplexElement(op->getOperand(keys[0])))
      return Decline("sort: complex lexicographic comparator");
    return Decline(
        absl::StrCat("sort: comparator ends in ",
                     View(cmp->getName().getStringRef()), ", not compare"));
  }
  const auto d = cmp_op.getComparisonDirection();
  const bool gt = d == mlir::stablehlo::ComparisonDirection::GT;
  if (!gt && d != mlir::stablehlo::ComparisonDirection::LT)
    return Decline("sort: non-strict compare");
  if (block.getNumArguments() != 2 * op->getNumOperands() ||
      op->getNumResults() != op->getNumOperands())
    return Decline("sort arity");

  mlir::Value lhs = cmp->getOperand(0), rhs = cmp->getOperand(1);
  const std::vector<unsigned> ldeps = ArgDeps(lhs, block);
  const std::vector<unsigned> rdeps = ArgDeps(rhs, block);
  if (ldeps.size() != 1 || rdeps.size() != 1)
    return Decline("sort: comparator mixes operands");
  const unsigned li = ldeps[0], ri = rdeps[0];
  if (ri != li + 1 || li % 2 != 0)
    return Decline("sort comparator args are not an (lhs, rhs) pair");
  const size_t k = static_cast<size_t>(li / 2);
  // Both sides must compute the same key function of their own argument.
  std::string lkey, rkey;
  SerializeKey(lhs, block.getArgument(li), block, &lkey);
  SerializeKey(rhs, block.getArgument(ri), block, &rkey);
  if (lkey != rkey) return Decline("sort: asymmetric comparator");

  std::vector<int> ins;
  for (mlir::Value v : op->getOperands()) {
    ASSIGN_OR_RETURN(int s, Slot(v));
    ins.push_back(s);
  }

  // The key: an operand when the comparator compares the pair directly, and
  // otherwise the chain's output, lowered here as ordinary entries.
  int64_t key_index = static_cast<int64_t>(k);
  mlir::Value key_value = op->getOperand(k);
  if (!mlir::isa<mlir::BlockArgument>(lhs)) {
    // Every argument pair stands for its operand array, exactly as the Python
    // seeds its evaluation environment.
    for (unsigned j = 0; j * 2 < block.getNumArguments(); j++) {
      if (j >= op->getNumOperands()) break;
      ASSIGN_OR_RETURN(int s, Slot(op->getOperand(j)));
      Alias(block.getArgument(2 * j), s);
      if (2 * j + 1 < block.getNumArguments())
        Alias(block.getArgument(2 * j + 1), s);
    }
    // The cone of ops the key depends on, in block order.
    llvm::DenseSet<mlir::Operation*> cone;
    std::vector<mlir::Value> stack{lhs};
    while (!stack.empty()) {
      mlir::Value v = stack.back();
      stack.pop_back();
      mlir::Operation* def = v.getDefiningOp();
      if (def == nullptr || def->getBlock() != &block) continue;
      if (!cone.insert(def).second) continue;
      for (mlir::Value w : def->getOperands()) stack.push_back(w);
    }
    const int64_t chain_bytes = ValueBytes(op->getOperand(k));
    for (mlir::Operation* o : body) {
      if (!cone.contains(o)) continue;
      const absl::string_view name = View(o->getName().getStringRef());
      if (!IsChainOp(name))
        return Decline(absl::StrCat("sort: comparator op ", name));
      for (mlir::Value v : o->getOperands())
        if (!IsRank0(v)) return Decline("sort: comparator op is not scalar");
      for (mlir::Value v : o->getResults())
        if (!IsRank0(v)) return Decline("sort: comparator op is not scalar");
      const size_t before = entries_.size();
      RETURN_IF_ERROR(LowerOp(o));
      // The IR says rank 0; what the entry materializes is a whole operand.
      // The flush cadence meters device bytes, so it is told the truth.
      if (name != "stablehlo.constant")
        for (size_t i = before; i < entries_.size(); i++)
          entries_[i].bytes = chain_bytes;
    }
    ASSIGN_OR_RETURN(int key_slot, Slot(lhs));
    key_index = static_cast<int64_t>(ins.size());
    key_value = lhs;
    ins.push_back(key_slot);
  }

  // `_sort_key`'s complex arm packs canonicalized (re, im) order keys into one
  // u64.  Not a `kind` this opcode carries -- and taking the integer arm would
  // sort by raw complex values -- so it declines.
  if (IsComplexElement(key_value)) return Decline("sort on complex");
  int64_t kind = 0;
  if (IsFloatElement(key_value)) kind = 1;
  else if (IsBoolElement(key_value)) kind = 2;

  std::vector<int> outs;
  for (mlir::Value r : op->getResults()) outs.push_back(Bind(r));
  ASSIGN_OR_RETURN(int opcode, Opcode("stablehlo.sort"));
  Emit(opcode, std::move(ins), std::move(outs),
       {dim, gt ? 1 : 0, key_index, kind}, std::nullopt, ResultBytes(op));
  return absl::OkStatus();
}

// tape.py `_lower_top_k`: ops/sort.py `_top_k`.  chlo.top_k survives a direct
// jax lowering; through a portable artifact it arrives already decomposed into
// the sort above, so both forms are lowered.
absl::Status Lowering::LowerTopK(mlir::Operation* op) {
  auto k_attr = op->getAttrOfType<mlir::IntegerAttr>("k");
  if (!k_attr) return Decline("chlo.top_k without a k");
  if (op->getNumOperands() != 1)
    return Decline("chlo.top_k operand arity");
  if (op->getNumResults() != 2)
    return Decline("top_k does not return (values, indices)");
  if (IsComplexElement(op->getOperand(0))) return Decline("top_k on complex");
  int64_t kind = 0;
  if (IsFloatElement(op->getOperand(0))) kind = 1;
  else if (IsBoolElement(op->getOperand(0))) kind = 2;
  ASSIGN_OR_RETURN(int slot, Slot(op->getOperand(0)));
  std::vector<int> outs{Bind(op->getResult(0)), Bind(op->getResult(1))};
  ASSIGN_OR_RETURN(int opcode, Opcode("chlo.top_k"));
  Emit(opcode, {slot}, std::move(outs), {k_attr.getInt(), kind}, std::nullopt,
       ResultBytes(op));
  return absl::OkStatus();
}

// tape.py `_lower_rng`: ops/rng.py `_rng_bit_generator`, whose whole schedule
// is static.  How many philox blocks are consumed, where a threefry output
// shape splits in half, which halves are sliced back down, whether the state
// arrives as four u32 words or two u64 ones -- all of it follows from the
// result type and the state's type, so it is resolved here and the C++ handler
// is the arithmetic only.  Bit-exactness against the Python engine (and so
// against XLA) is the whole point of the family, so nothing here rounds a
// shape differently: every expression below is copied from the handler.
absl::Status Lowering::LowerRng(mlir::Operation* op) {
  auto rng = mlir::dyn_cast<mlir::stablehlo::RngBitGeneratorOp>(op);
  if (!rng) return Decline("stablehlo.rng_bit_generator in an unexpected form");
  if (op->getNumResults() != 2)
    return Decline("rng_bit_generator result count");
  const bool threefry =
      rng.getRngAlgorithm() == mlir::stablehlo::RngAlgorithm::THREE_FRY;

  std::optional<std::string> st_el = ElementName(op->getOperand(0));
  ASSIGN_OR_RETURN(std::vector<int64_t> st_shape, Dims(op->getOperand(0)));
  int64_t state_u32;
  if (st_el == "ui32" && st_shape == std::vector<int64_t>{4}) {
    state_u32 = 1;
  } else if (st_el == "ui64" && st_shape == std::vector<int64_t>{2}) {
    state_u32 = 0;
  } else {
    return Decline(absl::StrCat("rng_bit_generator state ",
                                st_el.has_value() ? *st_el : "<unknown>"));
  }

  std::optional<std::string> out_el = ElementName(op->getResult(1));
  ASSIGN_OR_RETURN(std::vector<int64_t> out_shape, Dims(op->getResult(1)));
  ASSIGN_OR_RETURN(int out_code, DtypeCode(op->getResult(1)));
  int64_t width = 0;
  if (out_el == "ui8" || out_el == "i8") width = 8;
  else if (out_el == "ui16" || out_el == "i16") width = 16;
  else if (out_el == "ui32" || out_el == "i32") width = 32;
  else if (out_el == "ui64" || out_el == "i64") width = 64;
  else
    return Decline(absl::StrCat("rng_bit_generator output ",
                                out_el.has_value() ? *out_el : "<unknown>"));
  std::optional<int> unsigned_code =
      CodeForName(width == 8 ? "ui8"
                             : (width == 16 ? "ui16"
                                            : (width == 32 ? "ui32" : "ui64")));
  if (!unsigned_code.has_value())
    return Decline("rng_bit_generator unsigned dtype");

  const int64_t n = Product(out_shape);
  std::vector<int64_t> attrs{threefry ? 1 : 0, state_u32, out_code,
                             *unsigned_code,
                             static_cast<int64_t>(out_shape.size())};
  attrs.insert(attrs.end(), out_shape.begin(), out_shape.end());

  if (threefry) {
    if (width > 32) return Decline("rng_bit_generator THREE_FRY 64-bit");
    // `_threefry_bits`: one block per half-element pair, the output shape
    // split at the first even dim (else the largest).
    std::vector<int64_t> dims = out_shape;
    const int64_t scalar = out_shape.empty() ? 1 : 0;
    if (dims.empty()) dims.push_back(1);
    int64_t split = -1;
    for (size_t i = 0; i < dims.size(); i++) {
      if (dims[i] % 2 == 0) { split = static_cast<int64_t>(i); break; }
    }
    if (split < 0) {
      split = 0;   // python's `max(range(n), key=dims.__getitem__)`: FIRST max
      for (size_t i = 1; i < dims.size(); i++)
        if (dims[i] > dims[static_cast<size_t>(split)])
          split = static_cast<int64_t>(i);
    }
    std::vector<int64_t> half = dims;
    half[static_cast<size_t>(split)] = CeilDiv(dims[static_cast<size_t>(split)],
                                               2);
    const int64_t n_half = Product(half);
    std::vector<int64_t> h(half.begin(), half.begin() + split + 1);
    h.push_back(1);
    h.insert(h.end(), half.begin() + split + 1, half.end());
    std::vector<int64_t> rounded = dims;
    rounded[static_cast<size_t>(split)] = half[static_cast<size_t>(split)] * 2;
    const int64_t needs_slice =
        rounded[static_cast<size_t>(split)] != dims[static_cast<size_t>(split)]
            ? 1
            : 0;
    attrs.push_back(n_half);
    attrs.push_back(split);
    attrs.push_back(scalar);
    attrs.push_back(needs_slice);
    attrs.push_back(static_cast<int64_t>(h.size()));
    attrs.insert(attrs.end(), h.begin(), h.end());
    attrs.push_back(static_cast<int64_t>(rounded.size()));
    attrs.insert(attrs.end(), rounded.begin(), rounded.end());
    attrs.push_back(static_cast<int64_t>(dims.size()));
    attrs.insert(attrs.end(), dims.begin(), dims.end());
  } else {
    const int64_t num_u32 = width == 64 ? n * 2 : n;
    attrs.push_back(n);
    attrs.push_back(width);
    attrs.push_back(num_u32);
    attrs.push_back(CeilDiv(num_u32, 4));
  }

  std::vector<int> ins;
  for (mlir::Value v : op->getOperands()) {
    ASSIGN_OR_RETURN(int s, Slot(v));
    ins.push_back(s);
  }
  std::vector<int> outs{Bind(op->getResult(0)), Bind(op->getResult(1))};
  // tape.py `_TAINTING_OPS`: with an empty output XLA consumes no blocks, and
  // the handler hands the STATE operand's own array back.
  TaintFromAll(ins, outs);
  ASSIGN_OR_RETURN(int opcode, Opcode("stablehlo.rng_bit_generator"));
  Emit(opcode, std::move(ins), std::move(outs), std::move(attrs), std::nullopt,
       ResultBytes(op));
  return absl::OkStatus();
}

bool Lowering::IsIdentity(absl::string_view name, mlir::Operation* op) {
  auto dims_of = [&](mlir::Value v) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    return t ? std::vector<int64_t>(t.getShape().begin(), t.getShape().end())
             : std::vector<int64_t>{};
  };
  if (name == "stablehlo.reshape")
    return dims_of(op->getResult(0)) == dims_of(op->getOperand(0));
  if (name == "stablehlo.transpose") {
    auto tr = mlir::cast<mlir::stablehlo::TransposeOp>(op);
    int64_t i = 0;
    for (int64_t p : tr.getPermutation())
      if (p != i++) return false;
    return true;
  }
  if (name == "stablehlo.convert") {
    auto a = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto b = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    return a && b && a.getElementType() == b.getElementType();
  }
  if (name == "stablehlo.slice") {
    auto sl = mlir::cast<mlir::stablehlo::SliceOp>(op);
    const std::vector<int64_t> src = dims_of(op->getOperand(0));
    if (sl.getStartIndices().size() != src.size()) return true;
    for (size_t i = 0; i < src.size(); i++) {
      if (sl.getStartIndices()[i] != 0 || sl.getLimitIndices()[i] != src[i] ||
          sl.getStrides()[i] != 1)
        return false;
    }
    return true;
  }
  if (name == "stablehlo.broadcast_in_dim") {
    auto bc = mlir::cast<mlir::stablehlo::BroadcastInDimOp>(op);
    int64_t i = 0;
    for (int64_t d : bc.getBroadcastDimensions())
      if (d != i++) return false;
    return dims_of(op->getResult(0)) == dims_of(op->getOperand(0));
  }
  if (name == "stablehlo.concatenate") return op->getNumOperands() == 1;
  return false;
}

void Lowering::TaintResults(mlir::Operation* op, absl::string_view name,
                            const std::vector<int>& ins,
                            const std::vector<int>& outs) {
  if (outs.size() != 1) return;
  bool from_const = false;
  for (int s : ins) from_const = from_const || const_view_.count(s) > 0;
  if (from_const && IsViewOp(name)) const_view_.insert(outs[0]);
  if (IsIdentity(name, op)) {
    absl::flat_hash_set<int> src;
    for (int s : ins) {
      auto it = arg_alias_.find(s);
      if (it != arg_alias_.end()) src.insert(it->second.begin(),
                                             it->second.end());
    }
    if (!src.empty()) arg_alias_[outs[0]] = std::move(src);
  }
}

// --------------------------------------------------------------------------
// control flow (src/metaljax/tape.py `_control` / `_while` / `_branch`)
// --------------------------------------------------------------------------
//
// Each region becomes a Program of its own, whose arguments are the region
// block's own followed by its CAPTURES -- the values it reads from enclosing
// scopes, resolved to slots in THIS frame and passed in as ordinary inputs.
// That is the whole of the nesting: no environment is shared, so the executor
// enters a loop body exactly as it enters a top-level program.

absl::StatusOr<Lowering::Region> Lowering::LowerRegion(mlir::Block& block) {
  if (block.empty()) return Decline("an empty region");
  Region out;
  out.free = FreeValues(block);
  for (mlir::Value v : out.free) {
    auto it = slots_.find(v);
    if (it == slots_.end())
      return Decline("a region capture defined outside the block");
    out.caps.push_back(it->second);
  }
  Lowering child(ctx_);
  ASSIGN_OR_RETURN(Built built, child.LowerBlock(block, out.free));
  out.program = std::move(built.program);
  out.outputs = built.outputs;
  out.dump = std::move(built.dump);
  for (int s : out.outputs) out.taints.push_back(child.TaintOf(s));
  return out;
}

absl::Status Lowering::LowerControl(mlir::Operation* op) {
  const absl::string_view name = View(op->getName().getStringRef());
  for (mlir::Value v : op->getOperands()) RETURN_IF_ERROR(CheckValue(v));
  for (mlir::Value v : op->getResults()) RETURN_IF_ERROR(CheckValue(v));
  for (mlir::Region& r : op->getRegions()) {
    if (r.getBlocks().size() != 1)
      return Decline(absl::StrCat("op ", name, " has a multi-block region"));
  }
  if (name == "stablehlo.while") return LowerWhile(op);
  return LowerBranch(op);
}

absl::Status Lowering::LowerWhile(mlir::Operation* op) {
  if (op->getNumRegions() != 2)
    return Decline("stablehlo.while without a cond and a body");
  mlir::Block& cond_block = op->getRegion(0).front();
  mlir::Block& body_block = op->getRegion(1).front();
  const int64_t ncarry = static_cast<int64_t>(op->getNumOperands());
  if (static_cast<int64_t>(body_block.getNumArguments()) != ncarry)
    return Decline("while body arity mismatch");
  if (static_cast<int64_t>(cond_block.getNumArguments()) != ncarry)
    return Decline("while cond arity mismatch");
  if (static_cast<int64_t>(op->getNumResults()) != ncarry)
    return Decline("while result count mismatch");

  ASSIGN_OR_RETURN(Region cond, LowerRegion(cond_block));
  if (cond.outputs.size() != 1)
    return Decline("while cond does not return one value");
  ASSIGN_OR_RETURN(Region body, LowerRegion(body_block));
  if (static_cast<int64_t>(body.outputs.size()) != ncarry)
    return Decline("while body result count mismatch");

  // Where the trip count comes from, if this is the counted loop jax emits:
  // 0 a static N, 1 the carry at index `bound`, 2 the cond capture at index
  // `bound`.  A bound that is out of reach is not an error -- the Python
  // engine treats it as a dynamic loop, and so does this.
  int64_t is_counted = 0, k = 0, bound_kind = 0, bound = 0;
  std::optional<Counted> counted = AnalyzeCounted(*ctx_, op);
  if (counted.has_value()) {
    k = counted->k;
    if (counted->kind == Counted::kStatic) {
      is_counted = 1;
      bound_kind = 0;
      bound = counted->n;
    } else if (counted->kind == Counted::kCarry) {
      is_counted = 1;
      bound_kind = 1;
      bound = counted->n;
    } else {
      for (size_t i = 0; i < cond.free.size(); i++) {
        if (cond.free[i] == counted->value) {
          is_counted = 1;
          bound_kind = 2;
          bound = static_cast<int64_t>(i);
          break;
        }
      }
    }
  }

  const int64_t cost = BlockCost(*ctx_, body_block);
  const int64_t period = FlushPeriod(cost);

  // The compile decisions (tape.py `_while`, P5).  Two budgets, solved for the
  // two things the executor is allowed to do with a body: replay K iterations
  // as ONE compiled chunk (`chunkable` / `kmax`), and compile the single-step
  // body at all (`body_compile_max`, which is `_body_fn`'s gates -- purity, the
  // op budget, the byte budget -- solved for `repeat`).
  const bool pure = BlockIsPure(*ctx_, body_block);
  const int64_t by_cost = kTraceBudget / std::max<int64_t>(cost, 1);
  // `BytesChunks` never returns less than 1: its callers ask "how many
  // iterations may one trace hold", and the single-step case is gated
  // separately below, which says NO when one iteration alone is over budget.
  // Solving that gate for `repeat` is this division, and it must NOT be
  // rounded up to 1 -- a compiled body holds every intermediate of an
  // iteration instead of flushing inside it (measured on the byte-gated
  // random.normal init: 1.19 GB peak eager, 2.38 GB compiled).
  const int64_t by_bytes =
      kCompileBytes <= 0
          ? by_cost
          : kCompileBytes /
                std::max<int64_t>(BlockBytes(*ctx_, body_block), 1);
  // `interp._no_chunk` / `interp._no_body_compile` have no term here: they are
  // what the PYTHON engine remembers about a body whose chunk or compiled call
  // failed at run time, and the executor keeps the same memory itself
  // (`Program::set_no_chunk`, `Program::drop_compiled`).  Empty at lowering on
  // both engines, which is what a tape diff compares.
  const int64_t chunkable =
      (kCompileEnabled && cost <= kChunkMaxCost && pure) ? 1 : 0;
  const int64_t kmax = std::max<int64_t>(
      1, std::min<int64_t>({by_cost, kChunkMax,
                            BytesChunks(*ctx_, body_block)}));
  int64_t body_compile_max = 0;
  if (kCompileEnabled && kBodyCompile && pure)
    body_compile_max = std::max<int64_t>(0, std::min(by_cost, by_bytes));
  if (body_compile_max > 0) {
    std::vector<int> anchors = UnderivedOutputs(body_block, body.free);
    if (kDumpTape) {
      body.dump.compile = true;
      body.dump.anchors = anchors;
      body.dump.max_repeat = body_compile_max;
    }
    body.program->set_compile(true, std::move(anchors), body_compile_max);
  }

  std::vector<int64_t> attrs{ncarry,
                             static_cast<int64_t>(cond.caps.size()),
                             static_cast<int64_t>(body.caps.size()),
                             is_counted,
                             k,
                             bound_kind,
                             bound,
                             cost,
                             period,
                             chunkable,
                             kmax,
                             body_compile_max};

  std::vector<int> ins;
  for (mlir::Value v : op->getOperands()) {
    ASSIGN_OR_RETURN(int s, Slot(v));
    ins.push_back(s);
  }
  std::vector<int> body_parents = ins;
  body_parents.insert(body_parents.end(), body.caps.begin(), body.caps.end());
  ins.insert(ins.end(), cond.caps.begin(), cond.caps.end());
  ins.insert(ins.end(), body.caps.begin(), body.caps.end());
  std::vector<int> outs;
  for (mlir::Value r : op->getResults()) outs.push_back(Bind(r));

  // A loop's result j is its carry j: the body's output when the loop ran,
  // the INITIAL value when the trip count was zero -- and an initial value is
  // very often main's own argument, so which of the two it is decides whether
  // the result may alias one.  A statically counted loop answers that here;
  // anything else is charged both.
  std::optional<int64_t> static_trip;
  if (is_counted && bound_kind == 0) {
    std::optional<int64_t> start = StaticStart(op, k);
    if (start.has_value()) static_trip = std::max<int64_t>(bound - *start, 0);
  }
  for (int64_t j = 0; j < ncarry; j++) {
    Taint t = MapTaint(body.taints[static_cast<size_t>(j)], body_parents);
    const int init = body_parents[static_cast<size_t>(j)];
    if (static_trip.has_value() && *static_trip == 0) {
      t = TaintOf(init);            // the body never runs
    } else if (!static_trip.has_value()) {
      const Taint from_init = TaintOf(init);
      t.args.insert(from_init.args.begin(), from_init.args.end());
      t.cv = t.cv || from_init.cv;
    }
    ApplyTaint(outs[static_cast<size_t>(j)], t);
  }

  ASSIGN_OR_RETURN(int opcode, Opcode("stablehlo.while"));
  std::vector<std::shared_ptr<Program>> regions{cond.program, body.program};
  std::vector<DumpNode> dumps;
  if (kDumpTape) dumps = {cond.dump, body.dump};
  Emit(opcode, std::move(ins), std::move(outs), std::move(attrs), std::nullopt,
       ResultBytes(op), std::move(regions), std::move(dumps));
  return absl::OkStatus();
}

// stablehlo.if / stablehlo.case.  The branch blocks take no arguments (every
// value they read is a capture) and the branch is chosen on the HOST, which
// is what makes a block holding one impure -- so no program containing one is
// compiled, on either engine.
absl::Status Lowering::LowerBranch(mlir::Operation* op) {
  if (op->getNumOperands() != 1)
    return Decline("a branch op with more than a predicate");
  ASSIGN_OR_RETURN(int pred, Slot(op->getOperand(0)));
  std::vector<int64_t> attrs;
  std::vector<int> ins{pred};
  std::vector<int> outs;
  for (mlir::Value r : op->getResults()) outs.push_back(Bind(r));

  std::vector<std::shared_ptr<Program>> regions;
  std::vector<DumpNode> dumps;
  std::vector<std::vector<Taint>> per_branch;
  for (mlir::Region& region : op->getRegions()) {
    mlir::Block& blk = region.front();
    if (blk.getNumArguments() != 0)
      return Decline("a branch region that takes arguments");
    ASSIGN_OR_RETURN(Region br, LowerRegion(blk));
    if (br.outputs.size() != outs.size())
      return Decline("branch result count mismatch");
    attrs.push_back(static_cast<int64_t>(br.caps.size()));
    ins.insert(ins.end(), br.caps.begin(), br.caps.end());
    std::vector<Taint> mapped;
    for (const Taint& t : br.taints) mapped.push_back(MapTaint(t, br.caps));
    per_branch.push_back(std::move(mapped));
    regions.push_back(std::move(br.program));
    if (kDumpTape) dumps.push_back(std::move(br.dump));
  }
  if (regions.empty()) return Decline("a branch op with no regions");

  for (size_t j = 0; j < outs.size(); j++) {
    Taint t;
    for (const std::vector<Taint>& branch : per_branch) {
      t.args.insert(branch[j].args.begin(), branch[j].args.end());
      t.cv = t.cv || branch[j].cv;
    }
    ApplyTaint(outs[j], t);
  }

  ASSIGN_OR_RETURN(int opcode, Opcode(View(op->getName().getStringRef())));
  Emit(opcode, std::move(ins), std::move(outs), std::move(attrs), std::nullopt,
       ResultBytes(op), std::move(regions), std::move(dumps));
  return absl::OkStatus();
}

absl::Status Lowering::Inline(mlir::Operation* op, llvm::StringRef attr) {
  // Purely a lowering-time move: the tape never sees a call.  The callee's
  // block arguments alias the call's operand slots and the call's results
  // alias whatever the callee returned, so the inlined ops read and write
  // exactly the arrays a real call would have handed them -- which is also
  // what carries the aliasing taints across the boundary by construction.
  auto sym = op->getAttrOfType<mlir::FlatSymbolRefAttr>(attr);
  if (!sym) return Decline(absl::StrCat("a call with no ", View(attr)));
  const std::string name = sym.getValue().str();
  for (const std::string& active : calls_) {
    if (active == name)
      return Decline(absl::StrCat("a recursive call to @", name));
  }
  auto fn = ctx_->module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
  if (!fn) return Decline(absl::StrCat("a call to unknown symbol @", name));
  if (fn.getBody().getBlocks().size() != 1)
    return Decline(absl::StrCat("callee @", name, " is not single-block"));
  mlir::Block& block = fn.getBody().front();
  if (block.getNumArguments() != op->getNumOperands())
    return Decline(absl::StrCat("callee @", name, " arity mismatch"));
  for (unsigned i = 0; i < op->getNumOperands(); i++) {
    ASSIGN_OR_RETURN(int s, Slot(op->getOperand(i)));
    Alias(block.getArgument(i), s);
  }

  calls_.push_back(name);
  std::vector<mlir::Value> returned;
  bool terminated = false;
  for (mlir::Operation& inner : block) {
    if (mlir::isa<mlir::func::ReturnOp>(inner) ||
        mlir::isa<mlir::stablehlo::ReturnOp>(inner)) {
      returned.assign(inner.getOperands().begin(), inner.getOperands().end());
      terminated = true;
      break;
    }
    RETURN_IF_ERROR(LowerOp(&inner));
  }
  calls_.pop_back();
  if (!terminated)
    return Decline(absl::StrCat("callee @", name, " has no terminator"));
  if (returned.size() != op->getNumResults())
    return Decline(absl::StrCat("callee @", name, " result count mismatch"));
  for (unsigned i = 0; i < op->getNumResults(); i++) {
    ASSIGN_OR_RETURN(int s, Slot(returned[i]));
    Alias(op->getResult(i), s);
  }
  return absl::OkStatus();
}

absl::Status Lowering::LowerOp(mlir::Operation* op) {
  const llvm::StringRef ref = op->getName().getStringRef();
  const absl::string_view name = View(ref);

  // Symbol-carrying calls are spliced in rather than lowered: both run the
  // callee's block on the caller's arrays, so inlining is a transliteration
  // of the handler and not an optimization.
  if (name == "func.call") return Inline(op, "callee");
  if (name == "stablehlo.composite") return Inline(op, "decomposition");

  // Ops whose handler is `return list(ins)`: the result IS the operand array.
  // Lowered by aliasing slots, so no entry reaches the tape and the aliasing
  // taints ride along by construction.
  if (name == "stablehlo.optimization_barrier" ||
      name == "sdy.sharding_constraint" || name == "sdy.reshard") {
    if (op->getNumResults() != op->getNumOperands())
      return Decline(absl::StrCat("op ", name, " is not an arity-preserving "
                                              "alias"));
    for (unsigned i = 0; i < op->getNumResults(); i++) {
      ASSIGN_OR_RETURN(int s, Slot(op->getOperand(i)));
      Alias(op->getResult(i), s);
    }
    return absl::OkStatus();
  }

  // A rank-0 dynamic slice or update has no index operands and nothing to
  // slice: ops/shape.py hands the operand array straight back, so this is an
  // alias like the ones above (tape.py `_rank0_passthrough`).
  if (name == "stablehlo.dynamic_slice" && op->getNumOperands() == 1) {
    ASSIGN_OR_RETURN(int s, Slot(op->getOperand(0)));
    Alias(op->getResult(0), s);
    return absl::OkStatus();
  }
  if (name == "stablehlo.dynamic_update_slice" && op->getNumOperands() == 2) {
    ASSIGN_OR_RETURN(int s, Slot(op->getOperand(1)));
    Alias(op->getResult(0), s);   // the update replaces all of the operand
    return absl::OkStatus();
  }
  // Empty updates or a zero-size operand: the same kind of alias, and it sits
  // here -- ahead of the dtype checks -- exactly where tape.py's does.
  if (name == "stablehlo.scatter" && IsScatterNoop(op)) {
    ASSIGN_OR_RETURN(int s, Slot(op->getOperand(0)));
    Alias(op->getResult(0), s);
    return absl::OkStatus();
  }

  if (IsControlOp(name)) return LowerControl(op);

  // tape.py `_REGION_BODY_OPS`: the four ops whose region is a BODY the
  // lowering reads (structurally, or into a sub-Program) rather than a branch
  // of control flow.
  if (!op->getRegions().empty() && name != "stablehlo.reduce" &&
      name != "stablehlo.scatter" && name != "stablehlo.sort" &&
      name != "stablehlo.reduce_window")
    return Decline(absl::StrCat("op ", name, " (it carries a region)"));

  for (mlir::Value v : op->getOperands()) RETURN_IF_ERROR(CheckValue(v));
  for (mlir::Value v : op->getResults()) RETURN_IF_ERROR(CheckValue(v));

  if (name == "stablehlo.constant") return LowerConstant(op);
  if (name == "stablehlo.reduce") return LowerReduce(op);
  // Scatter binds and emits itself: a drop strategy may carry a neutral VALUE
  // rather than an attribute, and a variadic scatter must decline as one
  // instead of as "an op with two results".
  if (name == "stablehlo.scatter") return LowerScatter(op);
  // tape.py `_MULTI_RESULT_OPS`: the rest of the handlers that decide for
  // themselves how many arrays they hand back.
  if (name == "stablehlo.reduce_window") return LowerReduceWindow(op);
  if (name == "stablehlo.sort") return LowerSort(op);
  if (name == "chlo.top_k") return LowerTopK(op);
  if (name == "stablehlo.rng_bit_generator") return LowerRng(op);

  if (op->getNumResults() != 1)
    return Decline(absl::StrCat("op ", name, " with ", op->getNumResults(),
                                " results"));

  std::vector<int64_t> attrs;
  if (name == "stablehlo.compare") {
    ASSIGN_OR_RETURN(attrs, LowerCompare(op));
  } else if (name == "stablehlo.convert") {
    ASSIGN_OR_RETURN(attrs, LowerConvert(op));
  } else if (name == "stablehlo.reshape") {
    ASSIGN_OR_RETURN(attrs, LowerReshape(op));
  } else if (name == "stablehlo.transpose") {
    ASSIGN_OR_RETURN(attrs, LowerTranspose(op));
  } else if (name == "stablehlo.broadcast_in_dim") {
    ASSIGN_OR_RETURN(attrs, LowerBroadcastInDim(op));
  } else if (name == "stablehlo.slice") {
    ASSIGN_OR_RETURN(attrs, LowerSlice(op));
  } else if (name == "stablehlo.concatenate") {
    ASSIGN_OR_RETURN(attrs, LowerConcatenate(op));
  } else if (name == "stablehlo.iota") {
    ASSIGN_OR_RETURN(attrs, LowerIota(op));
  } else if (name == "stablehlo.pad") {
    ASSIGN_OR_RETURN(attrs, LowerPad(op));
  } else if (name == "stablehlo.dot_general") {
    ASSIGN_OR_RETURN(attrs, LowerDotGeneral(op));
  } else if (name == "stablehlo.dynamic_slice") {
    ASSIGN_OR_RETURN(attrs, LowerDynamicSlice(op));
  } else if (name == "stablehlo.dynamic_update_slice") {
    ASSIGN_OR_RETURN(attrs, LowerDynamicUpdateSlice(op));
  } else if (name == "stablehlo.shift_left" ||
             name == "stablehlo.shift_right_logical" ||
             name == "stablehlo.shift_right_arithmetic") {
    ASSIGN_OR_RETURN(attrs, LowerShift(op));
  } else if (name == "stablehlo.bitcast_convert") {
    ASSIGN_OR_RETURN(attrs, LowerBitcastConvert(op));
  } else if (name == "stablehlo.reverse") {
    ASSIGN_OR_RETURN(attrs, LowerReverse(op));
  } else if (name == "stablehlo.popcnt" ||
             name == "stablehlo.count_leading_zeros") {
    ASSIGN_OR_RETURN(attrs, LowerPopcnt(op));
  } else if (name == "stablehlo.fft") {
    ASSIGN_OR_RETURN(attrs, LowerFft(op));
  } else if (name == "stablehlo.gather") {
    ASSIGN_OR_RETURN(attrs, LowerGather(op));
  } else if (name == "stablehlo.convolution") {
    ASSIGN_OR_RETURN(attrs, LowerConv(op));
  } else if (SimpleOps().contains(name)) {
    // no attributes
  } else {
    return Decline(absl::StrCat("op ", name));
  }

  // Convolution's opcode is registered under a pseudo-name, so that a tape
  // builder without a convolution lowering (Stage 1's) cannot reach the
  // handler with an empty attribute vector -- see native/config.cc.
  ASSIGN_OR_RETURN(int opcode,
                   Opcode(name == "stablehlo.convolution" ? "metaljax.conv"
                                                          : name));
  std::vector<int> ins;
  for (mlir::Value v : op->getOperands()) {
    ASSIGN_OR_RETURN(int s, Slot(v));
    ins.push_back(s);
  }
  std::vector<int> outs{Bind(op->getResult(0))};
  TaintResults(op, name, ins, outs);
  Emit(opcode, std::move(ins), outs, std::move(attrs), std::nullopt,
       ResultBytes(op));
  return absl::OkStatus();
}

// tape.py `_build`: the entries, their drop lists, and the Program.
absl::StatusOr<Lowering::Built> Lowering::Finish(
    int nargs, const std::vector<int>& outputs) {
  // Per-op drop lists: the slots whose last use is that op (tape.py
  // `_liveness`).  Straight-line, so this is "highest index that reads it";
  // a result nothing reads is let go at the op that produced it, and an
  // output is never dropped.
  absl::flat_hash_map<int, size_t> last;
  for (size_t i = 0; i < entries_.size(); i++)
    for (int s : entries_[i].ins) last[s] = i;
  for (size_t i = 0; i < entries_.size(); i++)
    for (int s : entries_[i].outs) last.emplace(s, i);
  for (int s : outputs) last.erase(s);
  std::vector<std::vector<int>> drops(entries_.size());
  for (const auto& kv : last) drops[kv.second].push_back(kv.first);

  Built built;
  built.program = std::make_shared<Program>(nslots_, nargs);
  for (size_t i = 0; i < entries_.size(); i++) {
    Pending& e = entries_[i];
    if (kDumpTape) {
      built.dump.entries.push_back(DumpNode::Entry{
          e.op, e.ins, e.outs, e.attrs, e.payload.has_value(),
          e.region_dumps});
    }
    built.program->add(e.op, e.ins, e.outs, e.attrs, e.payload, drops[i],
                       e.regions, e.bytes, {}, nullptr, {});
  }
  // A region Program never needs output copies: its results are a loop's
  // carries or a branch's values, which stay inside this engine.  Only a
  // whole program's outputs cross to a caller, and `Run` sets those.
  built.program->set_outputs(outputs, {});
  built.outputs = outputs;
  built.dump.outputs = outputs;
  built.dump.slots = nslots_;
  return built;
}

absl::StatusOr<Lowering::Built> Lowering::LowerBlock(
    mlir::Block& block, const std::vector<mlir::Value>& captures) {
  // The block's own arguments first, then the captures: that order is what
  // the executor feeds a region (native/control.cc builds `carry... caps...`),
  // so it is part of the encoding.
  for (mlir::BlockArgument arg : block.getArguments()) {
    RETURN_IF_ERROR(CheckValue(arg));
    const int s = Bind(arg);
    arg_alias_[s] = {s};
  }
  for (mlir::Value cap : captures) {
    RETURN_IF_ERROR(CheckValue(cap));
    const int s = Bind(cap);
    arg_alias_[s] = {s};
  }

  std::vector<mlir::Value> returned;
  bool terminated = false;
  for (mlir::Operation& op : block) {
    if (mlir::isa<mlir::func::ReturnOp>(op) ||
        mlir::isa<mlir::stablehlo::ReturnOp>(op)) {
      returned.assign(op.getOperands().begin(), op.getOperands().end());
      terminated = true;
      break;
    }
    RETURN_IF_ERROR(LowerOp(&op));
  }
  if (!terminated) return Decline("a block without a terminator");

  std::vector<int> outputs;
  for (mlir::Value v : returned) {
    RETURN_IF_ERROR(CheckValue(v));
    ASSIGN_OR_RETURN(int s, Slot(v));
    outputs.push_back(s);
  }
  ASSIGN_OR_RETURN(Built built,
                   Finish(static_cast<int>(block.getNumArguments() +
                                           captures.size()),
                          outputs));
  built.returned = std::move(returned);
  return built;
}

absl::StatusOr<LoweredProgram> Lowering::Run(mlir::func::FuncOp fn) {
  if (fn.getBody().getBlocks().size() != 1)
    return Decline("a main function with several blocks");
  mlir::Block& block = fn.getBody().front();

  LoweredProgram lowered;
  for (mlir::BlockArgument arg : block.getArguments()) {
    RETURN_IF_ERROR(CheckValue(arg));
    ASSIGN_OR_RETURN(std::vector<int64_t> dims, Dims(arg));
    auto t = mlir::cast<mlir::RankedTensorType>(arg.getType());
    lowered.parameters.push_back(
        ValueSpec{*PrimitiveTypeOf(t.getElementType()), std::move(dims),
                  *MxDtypeOf(t.getElementType())});
  }

  ASSIGN_OR_RETURN(Built built, LowerBlock(block, {}));
  for (mlir::Value v : built.returned) {
    ASSIGN_OR_RETURN(std::vector<int64_t> dims, Dims(v));
    auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
    lowered.results.push_back(
        ValueSpec{*PrimitiveTypeOf(t.getElementType()), std::move(dims),
                  *MxDtypeOf(t.getElementType())});
  }

  // Which outputs may not be handed out as they stand: one that may BE an
  // argument's array, or one that reads a constant the Program holds for the
  // life of the executable.  Either would alias across calls, and XLA's
  // contract wants a fresh buffer (jax asserts it through
  // unsafe_buffer_pointer, and a consumer that DONATES such an output would
  // clobber a constant for every later call).
  //
  // Stricter than src/metaljax/tape.py's rule in one place, deliberately: it
  // exempts an output that syntactically names a block argument, because
  // engine.execute copies those on the way out whatever engine ran.  There is
  // no engine.execute here, so this plugin copies them itself.
  std::vector<int> copies;
  for (size_t j = 0; j < built.outputs.size(); j++) {
    if (const_view_.count(built.outputs[j]) ||
        arg_alias_.count(built.outputs[j]))
      copies.push_back(static_cast<int>(j));
  }
  lowered.num_copies = static_cast<int64_t>(copies.size());
  built.program->set_outputs(built.outputs, copies);

  // Whether the WHOLE tape traces through mx::compile (engine.py
  // `MetalExecutable.runner`, P5).  Two independent budgets: `cost` bounds how
  // many ops one trace may hold, and so the Metal live-buffer count; the byte
  // gate bounds how much those buffers HOLD, which op count says nothing about
  // -- a jitted parameter init is 365 ops and 15 GB of traffic.  Over either,
  // the program runs op by op, where last-use pruning and the byte-denominated
  // eager flush bound it and the compiled path does not.
  const int64_t cost = BlockCost(*ctx_, block);
  const bool pure = BlockIsPure(*ctx_, block);
  const bool compile_main =
      kCompileEnabled && pure && cost <= kTraceBudget &&
      BytesOk(*ctx_, block, 1, "main", /*whole=*/true);
  lowered.compiled = compile_main;
  if (compile_main) {
    // A main block has no captures, so `free` is empty: an output with no data
    // path from an ARGUMENT is what MLX would bake by value.
    std::vector<int> anchors = UnderivedOutputs(block, {});
    if (kDumpTape) {
      built.dump.compile = true;
      built.dump.anchors = anchors;
      built.dump.max_repeat = 1;
    }
    built.program->set_compile(true, std::move(anchors), /*max_repeat=*/1);
  }
  if (kDebug) {
    std::fprintf(stderr,
                 "[metaljax-native] main: pure=%d cost=%lld bytes=%.1fMB "
                 "compile=%d\n",
                 static_cast<int>(pure), static_cast<long long>(cost),
                 static_cast<double>(ProgramBytes(*ctx_, block)) /
                     static_cast<double>(1 << 20),
                 static_cast<int>(compile_main));
  }
  if (kDumpTape) {
    built.dump.copies = copies;
    std::string text;
    RenderDump(built.dump, "", &text);
    std::fputs(text.c_str(), stderr);
    std::fflush(stderr);
  }
  lowered.num_entries = static_cast<int64_t>(entries_.size());
  lowered.program = std::move(built.program);
  return lowered;
}

}  // namespace

xla::Shape ValueSpec::shape() const {
  return xla::ShapeUtil::MakeShape(type, dims);
}

int64_t ValueSpec::element_count() const {
  int64_t n = 1;
  for (int64_t d : dims) n *= d;
  return n;
}

absl::StatusOr<LoweredProgram> LowerModule(mlir::ModuleOp module) {
  if (!module) return Decline("an empty module");
  if (kDumpModule) {
    std::string text;
    llvm::raw_string_ostream os(text);
    module.print(os);
    std::fputs("[module]\n", stderr);
    std::fputs(text.c_str(), stderr);
    std::fputs("\n[/module]\n", stderr);
    std::fflush(stderr);
  }
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    // jax always names it @main; anything else is a program shape this
    // plugin has never seen, so say so rather than guessing which to run.
    return Decline("a module with no @main function");
  }
  LowerContext ctx{module, {}, {}};
  Lowering lowering(&ctx);
  try {
    return lowering.Run(main);
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("metaljax-native: lowering failed: ", e.what()));
  }
}

}  // namespace metaljax
