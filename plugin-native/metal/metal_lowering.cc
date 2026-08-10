/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_lowering.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "metal/metal_dtypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
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
         name == "stablehlo.concatenate";
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
// the lowering
// --------------------------------------------------------------------------

class Lowering {
 public:
  explicit Lowering(mlir::ModuleOp module) : module_(module) {}
  absl::StatusOr<LoweredProgram> Run(mlir::func::FuncOp fn);

 private:
  struct Pending {
    int op;
    std::vector<int> ins;
    std::vector<int> outs;
    std::vector<int64_t> attrs;
    std::optional<mx::array> payload;
    int64_t bytes = 0;
  };

  absl::Status LowerOp(mlir::Operation* op);

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
  absl::Status LowerConstant(mlir::Operation* op);
  absl::Status LowerReduce(mlir::Operation* op);

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
            int64_t bytes) {
    entries_.push_back(Pending{op, std::move(ins), std::move(outs),
                               std::move(attrs), std::move(payload), bytes});
  }

  int64_t ResultBytes(mlir::Operation* op) {
    int64_t n = 0;
    for (mlir::Value r : op->getResults()) n += ValueBytes(r);
    return n;
  }

  // Slots this op's results inherit taints from, given its operands.
  void Taint(mlir::Operation* op, absl::string_view name,
             const std::vector<int>& ins, const std::vector<int>& outs);

  mlir::ModuleOp module_;
  llvm::DenseMap<mlir::Value, int> slots_;
  int nslots_ = 0;
  std::vector<Pending> entries_;
  std::vector<std::string> calls_;   // callees currently being inlined
  // The two aliasing taints, consumed by the output-copy rule in `Run`.
  // Which argument a slot may be does not matter here (this phase lowers no
  // regions, so nothing maps taints across a frame): what an output position
  // asks is only whether it may be an argument's array at all.
  absl::flat_hash_set<int> arg_alias_;
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
  return Decline("stablehlo.reduce with a general body");
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

void Lowering::Taint(mlir::Operation* op, absl::string_view name,
                     const std::vector<int>& ins,
                     const std::vector<int>& outs) {
  if (outs.size() != 1) return;
  bool from_const = false;
  for (int s : ins) from_const = from_const || const_view_.count(s) > 0;
  if (from_const && IsViewOp(name)) const_view_.insert(outs[0]);
  if (IsIdentity(name, op)) {
    for (int s : ins) {
      if (arg_alias_.count(s)) {
        arg_alias_.insert(outs[0]);
        break;
      }
    }
  }
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
  auto fn = module_.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
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

  if (!op->getRegions().empty() && name != "stablehlo.reduce")
    return Decline(absl::StrCat("op ", name, " (it carries a region)"));

  for (mlir::Value v : op->getOperands()) RETURN_IF_ERROR(CheckValue(v));
  for (mlir::Value v : op->getResults()) RETURN_IF_ERROR(CheckValue(v));

  if (name == "stablehlo.constant") return LowerConstant(op);
  if (name == "stablehlo.reduce") return LowerReduce(op);

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
  } else if (SimpleOps().contains(name)) {
    // no attributes
  } else {
    return Decline(absl::StrCat("op ", name));
  }

  ASSIGN_OR_RETURN(int opcode, Opcode(name));
  std::vector<int> ins;
  for (mlir::Value v : op->getOperands()) {
    ASSIGN_OR_RETURN(int s, Slot(v));
    ins.push_back(s);
  }
  std::vector<int> outs{Bind(op->getResult(0))};
  Taint(op, name, ins, outs);
  Emit(opcode, std::move(ins), outs, std::move(attrs), std::nullopt,
       ResultBytes(op));
  return absl::OkStatus();
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
    const int s = Bind(arg);
    arg_alias_.insert(s);
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
    ASSIGN_OR_RETURN(std::vector<int64_t> dims, Dims(v));
    auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
    lowered.results.push_back(
        ValueSpec{*PrimitiveTypeOf(t.getElementType()), std::move(dims),
                  *MxDtypeOf(t.getElementType())});
  }

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

  auto program = std::make_shared<Program>(
      nslots_, static_cast<int>(lowered.parameters.size()));
  for (size_t i = 0; i < entries_.size(); i++) {
    Pending& e = entries_[i];
    if (kDumpTape) {
      std::string line = absl::StrCat("[tape] ", OpcodeName(e.op), " ");
      for (size_t k = 0; k < e.ins.size(); k++)
        absl::StrAppend(&line, k ? "," : "", e.ins[k]);
      absl::StrAppend(&line, " -> ");
      for (size_t k = 0; k < e.outs.size(); k++)
        absl::StrAppend(&line, k ? "," : "", e.outs[k]);
      absl::StrAppend(&line, " [");
      for (size_t k = 0; k < e.attrs.size(); k++)
        absl::StrAppend(&line, k ? "," : "", e.attrs[k]);
      absl::StrAppend(&line, "]", e.payload.has_value() ? " const" : "");
      std::fprintf(stderr, "%s\n", line.c_str());
    }
    program->add(e.op, e.ins, e.outs, e.attrs, e.payload, drops[i], {},
                 e.bytes, {}, nullptr, {});
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
  for (size_t j = 0; j < outputs.size(); j++) {
    if (const_view_.count(outputs[j]) || arg_alias_.count(outputs[j]))
      copies.push_back(static_cast<int>(j));
  }
  lowered.num_copies = static_cast<int64_t>(copies.size());
  if (kDumpTape) {
    std::string line = "[tape] outputs ";
    for (size_t j = 0; j < outputs.size(); j++)
      absl::StrAppend(&line, j ? "," : "", outputs[j]);
    absl::StrAppend(&line, " copies ");
    for (size_t j = 0; j < copies.size(); j++)
      absl::StrAppend(&line, j ? "," : "", copies[j]);
    absl::StrAppend(&line, " slots ", nslots_);
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
  }
  program->set_outputs(outputs, copies);
  lowered.num_entries = static_cast<int64_t>(entries_.size());
  lowered.program = std::move(program);
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
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    // jax always names it @main; anything else is a program shape this
    // plugin has never seen, so say so rather than guessing which to run.
    return Decline("a module with no @main function");
  }
  Lowering lowering(module);
  try {
    return lowering.Run(main);
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("metaljax-native: lowering failed: ", e.what()));
  }
}

}  // namespace metaljax
