/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

`src/metaljax/moe.py`, as a pass over the parsed StableHLO.

The dense mixture-of-experts dispatch every jax MoE lowers to -- run all `E`
experts, then null the `E - K` contributions the router never selected --
becomes a gathered one: the `(expert, token)` grid collapses onto the
`P = T * K` selected pairs and each per-expert dot becomes one `gather_mm`
(float weights) or `gather_qmm` (weights the quantized-matmul recognizer
already packed).

This file is the analysis: the router proof, the pair-space planner and the
use-count discipline, transliterated.  The emission is
`Lowering::LowerMoe` -- one tape entry per plan node, in the plan's own order,
which is what `runtime/emits.cc` runs.

`VerifyMoe` is the Python's `_verify`, and it runs where the Python's runs --
in the eager prologue of the first execute, before any trace, because it syncs
with the host.  Its logits are SYNTHETIC, which is not a shortcut: the router
tail below the top-k depends on nothing else, and the real ones are a loop
carry no prologue could evaluate for a dispatch inside a decode loop.  The one
thing this cannot reach is a top-k bound inside a callee, whose input a cone
has no way to pin; such a match stays unverified and runs dense.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/StringRef.h"
#include "metal/metal_dtypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlx/mlx.h"
#include "program.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace metaljax {
namespace {

const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();

// moe.py `_VERIFY_DRAWS` (METALJAX_MOE_VERIFY_DRAWS).
const int kVerifyDraws = [] {
  const char* v = std::getenv("METALJAX_MOE_VERIFY_DRAWS");
  const int n = v == nullptr ? 3 : std::atoi(v);
  return n < 1 ? 1 : n;
}();

bool EnvOff(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "0";
}

// This is not a gatherable MoE dispatch: run the dense region.
struct Reject {
  std::string why;
};
[[noreturn]] void Bail(const std::string& why) { throw Reject{why}; }

void Debug(const std::string& line) {
  if (!kDebug) return;
  std::fprintf(stderr, "[metaljax-native] moe: %s\n", line.c_str());
  std::fflush(stderr);
}

// --------------------------------------------------------------------------
// structural helpers
// --------------------------------------------------------------------------

mlir::Operation* Owner(mlir::Value v) {
  return v == nullptr ? nullptr : v.getDefiningOp();
}

std::string OpName(mlir::Operation* op) {
  return op == nullptr ? std::string() : op->getName().getStringRef().str();
}

std::vector<int64_t> ShapeOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) Bail("a value that is not a ranked tensor");
  return std::vector<int64_t>(t.getShape().begin(), t.getShape().end());
}

int64_t Prod(const std::vector<int64_t>& xs) {
  int64_t p = 1;
  for (int64_t x : xs) p *= x;
  return p;
}

std::vector<int64_t> I64List(mlir::Operation* op, llvm::StringRef name) {
  if (auto arr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name))
    return std::vector<int64_t>(arr.asArrayRef().begin(),
                                arr.asArrayRef().end());
  if (auto den = op->getAttrOfType<mlir::DenseIntElementsAttr>(name)) {
    std::vector<int64_t> out;
    for (const llvm::APInt& v : den.getValues<llvm::APInt>())
      out.push_back(v.getSExtValue());
    return out;
  }
  return {};
}

int DtypeCodeOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) Bail("a value that is not a ranked tensor");
  std::optional<int> code = TapeDtypeCode(t.getElementType());
  if (!code.has_value()) Bail("an element type this tape cannot spell");
  return *code;
}

// moe.py `_splat`: the scalar value of a rank-0 / splat constant, else none.
std::optional<double> Splat(mlir::Operation* op) {
  if (op == nullptr || OpName(op) != "stablehlo.constant") return std::nullopt;
  auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
  if (!cst) return std::nullopt;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense) return std::nullopt;
  if (dense.getNumElements() != 1 && !dense.isSplat()) return std::nullopt;
  mlir::Type el =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
          .getElementType();
  if (mlir::isa<mlir::FloatType>(el))
    return dense.getSplatValue<llvm::APFloat>().convertToDouble();
  if (mlir::isa<mlir::IntegerType>(el))
    return static_cast<double>(
        dense.getSplatValue<llvm::APInt>().getSExtValue());
  return std::nullopt;
}

// moe.py `_reduce_info` + `_zero_sum`: the reduced axis of a
// `reduce(add, init=0)` over one operand.
int64_t ZeroSum(mlir::Operation* op) {
  if (op == nullptr || OpName(op) != "stablehlo.reduce") Bail("not a reduce");
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    Bail("variadic reduce");
  if (op->getNumRegions() != 1 || op->getRegion(0).getBlocks().size() != 1)
    Bail("reduce body is not a single block");
  mlir::Block& body = op->getRegion(0).front();
  int n = 0;
  std::string first;
  for (mlir::Operation& inner : body) {
    if (n == 0) first = OpName(&inner);
    n++;
  }
  if (n != 2 || first != "stablehlo.add") Bail("reduce body is not one add");
  std::vector<int64_t> dims = I64List(op, "dimensions");
  if (dims.size() != 1) Bail("the sum spans more than one dimension");
  std::optional<double> init = Splat(Owner(op->getOperand(1)));
  if (!init.has_value() || *init != 0.0) Bail("the sum does not start at zero");
  return dims[0];
}

// moe.py `_is_topk` / `_topk_k`.  A top-k arrives as `chlo.top_k` from a
// direct lowering but as a `stablehlo.composite` wrapping it once the program
// has been through a portable artifact -- which is how every program reaches
// this plugin.
bool IsTopk(mlir::Operation* op) {
  if (op == nullptr) return false;
  const std::string name = OpName(op);
  if (name == "chlo.top_k") return true;
  if (name != "stablehlo.composite") return false;
  auto n = op->getAttrOfType<mlir::StringAttr>("name");
  return n && n.getValue() == "chlo.top_k";
}

int64_t TopkK(mlir::Operation* op) {
  if (OpName(op) == "chlo.top_k") {
    if (auto k = op->getAttrOfType<mlir::IntegerAttr>("k"))
      return k.getValue().getSExtValue();
    Bail("chlo.top_k with no k");
  }
  auto attrs = op->getAttrOfType<mlir::DictionaryAttr>("composite_attributes");
  if (!attrs) Bail("a top-k composite with no attributes");
  auto k = attrs.getAs<mlir::IntegerAttr>("k");
  if (!k) Bail("a top-k composite with no k");
  return k.getValue().getSExtValue();
}

bool IsCall(const std::string& name) {
  return name == "func.call" || name == "stablehlo.composite";
}

// moe.py `_ELEMENTWISE`.
const absl::flat_hash_set<std::string>& Elementwise() {
  static const auto* set = new absl::flat_hash_set<std::string>{
      "stablehlo.abs", "stablehlo.add", "stablehlo.and", "stablehlo.atan2",
      "stablehlo.cbrt", "stablehlo.ceil", "stablehlo.clamp",
      "stablehlo.compare", "stablehlo.convert", "stablehlo.cosine",
      "stablehlo.count_leading_zeros", "stablehlo.divide",
      "stablehlo.exponential", "stablehlo.exponential_minus_one",
      "stablehlo.floor", "stablehlo.is_finite", "stablehlo.log",
      "stablehlo.log_plus_one", "stablehlo.logistic", "stablehlo.maximum",
      "stablehlo.minimum", "stablehlo.multiply", "stablehlo.negate",
      "stablehlo.not", "stablehlo.or", "stablehlo.popcnt", "stablehlo.power",
      "stablehlo.remainder", "stablehlo.round_nearest_afz",
      "stablehlo.round_nearest_even", "stablehlo.rsqrt", "stablehlo.select",
      "stablehlo.shift_left", "stablehlo.shift_right_arithmetic",
      "stablehlo.shift_right_logical", "stablehlo.sign", "stablehlo.sine",
      "stablehlo.sqrt", "stablehlo.subtract", "stablehlo.tan",
      "stablehlo.tanh", "stablehlo.xor"};
  return *set;
}

const absl::flat_hash_set<std::string>& NeverSweep() {
  static const auto* set = new absl::flat_hash_set<std::string>{
      "stablehlo.custom_call", "stablehlo.while", "stablehlo.if",
      "stablehlo.case", "stablehlo.optimization_barrier", "stablehlo.outfeed",
      "stablehlo.send", "stablehlo.recv", "stablehlo.infeed"};
  return *set;
}

// moe.py `_trailing` / `_tpos`.
std::vector<int64_t> Trailing(const std::vector<int64_t>& shape, int ea,
                              int ta) {
  std::vector<int64_t> out;
  for (int i = 0; i < static_cast<int>(shape.size()); i++)
    if (i != ea && i != ta) out.push_back(shape[i]);
  return out;
}

int64_t TPos(const std::vector<int64_t>& shape, int ea, int ta, int64_t axis) {
  int64_t n = 0;
  for (int64_t i = 0; i < axis; i++)
    if (i != ea && i != ta) n++;
  return n;
}

// --------------------------------------------------------------------------
// call frames: everything below sees through func.call / composite
// --------------------------------------------------------------------------

// A dereferenced value: the value itself, the frame it lives in (0 = the root
// frame, Python's None) and its defining op (null for a block argument
// nothing in scope defines).
struct Deref {
  mlir::Value value;
  int frame = 0;
  mlir::Operation* op = nullptr;
};

class Scope {
 public:
  explicit Scope(mlir::ModuleOp module) : module_(module) { frames_.resize(1); }

  // The root-frame call ops entered, in the order they were entered: they are
  // region work, and the region is what gets skipped.
  const std::vector<mlir::Operation*>& calls() const { return calls_; }

  Deref deref(mlir::Value v, int frame) {
    for (int guard = 0; guard < 64; guard++) {
      while (mlir::isa<mlir::BlockArgument>(v) && frame != 0) {
        auto it = frames_[frame].bind.find(v);
        if (it == frames_[frame].bind.end()) return Deref{v, frame, nullptr};
        v = it->second.first;
        frame = it->second.second;
      }
      if (mlir::isa<mlir::BlockArgument>(v)) return Deref{v, 0, nullptr};
      mlir::Operation* o = Owner(v);
      const std::string name = OpName(o);
      if (o == nullptr || !IsCall(name) || IsTopk(o)) return Deref{v, frame, o};
      auto sym = o->getAttrOfType<mlir::FlatSymbolRefAttr>(
          name == "func.call" ? "callee" : "decomposition");
      if (!sym) return Deref{v, frame, o};
      auto fn = module_.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
      if (!fn || fn.getBody().getBlocks().size() != 1)
        return Deref{v, frame, o};
      mlir::Block& blk = fn.getBody().front();
      if (blk.empty()) return Deref{v, frame, o};
      mlir::Operation* term = &blk.back();
      int i = -1;
      for (unsigned r = 0; r < o->getNumResults(); r++)
        if (o->getResult(r) == v) i = static_cast<int>(r);
      if (i < 0 || i >= static_cast<int>(term->getNumOperands()))
        return Deref{v, frame, o};
      if (blk.getNumArguments() != o->getNumOperands())
        return Deref{v, frame, o};
      if (frame == 0) calls_.push_back(o);
      Frame f;
      for (unsigned a = 0; a < blk.getNumArguments(); a++)
        f.bind[blk.getArgument(a)] = {o->getOperand(a), frame};
      frames_.push_back(std::move(f));
      frame = static_cast<int>(frames_.size()) - 1;
      v = term->getOperand(i);
    }
    Bail("call nesting too deep");
  }

 private:
  struct Frame {
    llvm::DenseMap<mlir::Value, std::pair<mlir::Value, int>> bind;
  };
  mlir::ModuleOp module_;
  std::vector<Frame> frames_;
  std::vector<mlir::Operation*> calls_;
};

// moe.py `_peel`: peel broadcast / transpose / unit-reshape / convert,
// tracking axes.  `dims[j]` is the axis of the ORIGINAL value that base axis
// `j` lands in, or -1 for a unit axis the chain dropped.  `stop` halts the
// walk at the first value of that shape -- a caller that will READ the value
// at run time must pass it, or a half-precision router's convert would be
// peeled away and the rewrite would multiply by the unrounded tensor.
struct Peeled {
  mlir::Value value;
  int frame = 0;
  std::vector<int64_t> dims;
};

Peeled Peel(Scope* scope, mlir::Value v, int frame,
            const std::vector<int64_t>* stop = nullptr) {
  std::vector<int64_t> dims(ShapeOf(v).size());
  for (size_t i = 0; i < dims.size(); i++) dims[i] = static_cast<int64_t>(i);
  while (true) {
    Deref d = scope->deref(v, frame);
    v = d.value;
    frame = d.frame;
    if (stop != nullptr && ShapeOf(v) == *stop) return Peeled{v, frame, dims};
    if (d.op == nullptr) return Peeled{v, frame, dims};
    const std::string name = OpName(d.op);
    if (name == "stablehlo.convert") {
      v = d.op->getOperand(0);
      continue;
    }
    if (name == "stablehlo.broadcast_in_dim") {
      std::vector<int64_t> bd = I64List(d.op, "broadcast_dimensions");
      std::vector<int64_t> next;
      for (int64_t x : bd) {
        if (x < 0 || x >= static_cast<int64_t>(dims.size()))
          Bail("a broadcast with out-of-range dimensions");
        next.push_back(dims[x]);
      }
      dims = std::move(next);
      v = d.op->getOperand(0);
      continue;
    }
    if (name == "stablehlo.transpose") {
      std::vector<int64_t> perm = I64List(d.op, "permutation");
      std::vector<int64_t> inv(perm.size());
      for (size_t i = 0; i < perm.size(); i++) {
        if (perm[i] < 0 || perm[i] >= static_cast<int64_t>(perm.size()))
          Bail("a transpose with an out-of-range permutation");
        inv[perm[i]] = static_cast<int64_t>(i);
      }
      std::vector<int64_t> next(perm.size());
      for (size_t j = 0; j < perm.size(); j++) next[j] = dims[inv[j]];
      dims = std::move(next);
      v = d.op->getOperand(0);
      continue;
    }
    if (name == "stablehlo.reshape") {
      std::vector<int64_t> src = ShapeOf(d.op->getOperand(0));
      std::vector<int64_t> dst = ShapeOf(d.op->getResult(0));
      std::vector<int64_t> sn, dn;
      for (int64_t x : src) if (x != 1) sn.push_back(x);
      for (int64_t x : dst) if (x != 1) dn.push_back(x);
      if (sn != dn) return Peeled{v, frame, dims};
      std::vector<int64_t> keep;
      for (size_t i = 0; i < dst.size(); i++)
        if (dst[i] != 1) keep.push_back(static_cast<int64_t>(i));
      std::vector<int64_t> next;
      size_t n = 0;
      for (int64_t x : src) {
        if (x == 1) {
          next.push_back(-1);
        } else {
          next.push_back(dims[keep[n]]);
          n++;
        }
      }
      dims = std::move(next);
      v = d.op->getOperand(0);
      continue;
    }
    return Peeled{v, frame, dims};
  }
}

// moe.py `_is_iota`.
bool IsIota(Scope* scope, mlir::Value v, int frame, int64_t axis,
            int64_t length) {
  Peeled p = Peel(scope, v, frame);
  Deref d = scope->deref(p.value, p.frame);
  if (d.op == nullptr || OpName(d.op) != "stablehlo.iota") return false;
  auto attr = d.op->getAttrOfType<mlir::IntegerAttr>("iota_dimension");
  if (!attr) return false;
  const int64_t dim = attr.getValue().getSExtValue();
  std::vector<int64_t> shp = ShapeOf(d.op->getResult(0));
  return dim >= 0 && dim < static_cast<int64_t>(p.dims.size()) &&
         p.dims[dim] == axis && shp[dim] == length && Prod(shp) == length;
}

// --------------------------------------------------------------------------
// the router: proving S is zero off a top-k selection
// --------------------------------------------------------------------------

struct Router {
  mlir::Value indices, weights;
  mlir::Operation* topk = nullptr;
  int tframe = 0;
  mlir::Value score;
  int sframe = 0;
  int64_t T = 0, K = 0, E = 0;
  int64_t e_in_score = 0, t_in_score = 0;
};

// moe.py `_match_router`: prove `score` is `w` scattered at a top-k's
// indices.  `se` / `st` are the axes of `score`'s OWN shape that carry the
// expert and the token dimension.
Router MatchRouter(Scope* scope, mlir::Value score, int sframe, int64_t se,
                   int64_t st) {
  Router r;
  r.score = score;
  r.sframe = sframe;
  r.e_in_score = se;
  r.t_in_score = st;
  std::vector<int64_t> sshape = ShapeOf(score);
  if (se >= static_cast<int64_t>(sshape.size()) ||
      st >= static_cast<int64_t>(sshape.size()))
    Bail("the score axes are out of range");
  r.E = sshape[se];
  r.T = sshape[st];

  Deref red = scope->deref(score, sframe);
  const int64_t kd = ZeroSum(red.op);

  Deref mul = scope->deref(red.op->getOperand(0), red.frame);
  while (mul.op != nullptr && OpName(mul.op) == "stablehlo.convert") {
    // A half-precision router sums the one-hot product in f32.  Nothing here
    // depends on the dtype: only `indices` and `weights` are read.
    mul = scope->deref(mul.op->getOperand(0), mul.frame);
  }
  if (mul.op == nullptr || OpName(mul.op) != "stablehlo.multiply")
    Bail("router scores do not reduce a product");
  std::vector<int64_t> mshape = ShapeOf(mul.op->getResult(0));
  if (mshape.size() != 3) Bail("the one-hot product is not rank 3");
  // The reduce drops `kd`; the surviving axes keep their relative order.
  std::vector<int64_t> surv;
  for (int64_t i = 0; i < 3; i++)
    if (i != kd) surv.push_back(i);
  const int64_t e_ax = surv[se], t_ax = surv[st];
  r.K = mshape[kd];
  if (mshape[e_ax] != r.E || mshape[t_ax] != r.T)
    Bail("the one-hot product does not match the score shape");

  const std::vector<int64_t> tk_shape{r.T, r.K};
  int hot = -1;
  for (int i = 0; i < 2; i++) {
    Peeled b = Peel(scope, mul.op->getOperand(i), mul.frame, &tk_shape);
    if (b.dims == std::vector<int64_t>{t_ax, kd} && ShapeOf(b.value) ==
                                                        tk_shape) {
      if (b.frame != 0) Bail("the routing weights live inside a callee");
      r.weights = b.value;
      hot = 1 - i;
      break;
    }
  }
  if (hot < 0) Bail("neither product operand is a [tokens, k] weight");

  Deref cmp = scope->deref(mul.op->getOperand(hot), mul.frame);
  // keras' one_hot returns f32 and the caller converts it: in bf16 there are
  // two converts here (one inside the callee, one outside).
  while (cmp.op != nullptr && OpName(cmp.op) == "stablehlo.convert")
    cmp = scope->deref(cmp.op->getOperand(0), cmp.frame);
  if (cmp.op == nullptr || OpName(cmp.op) != "stablehlo.compare")
    Bail("the one-hot is not a comparison");
  auto dir = cmp.op->getAttrOfType<mlir::stablehlo::ComparisonDirectionAttr>(
      "comparison_direction");
  if (!dir || dir.getValue() != mlir::stablehlo::ComparisonDirection::EQ)
    Bail("the one-hot comparison is not EQ");
  if (ShapeOf(cmp.op->getResult(0)) != mshape)
    Bail("the one-hot is not the shape of the product");
  int idx = -1;
  for (int i = 0; i < 2; i++) {
    if (IsIota(scope, cmp.op->getOperand(i), cmp.frame, e_ax, r.E)) {
      idx = 1 - i;
      break;
    }
  }
  if (idx < 0) Bail("the one-hot compares against no expert iota");
  Peeled b = Peel(scope, cmp.op->getOperand(idx), cmp.frame, &tk_shape);
  if (b.dims != std::vector<int64_t>{t_ax, kd} || ShapeOf(b.value) != tk_shape)
    Bail("the one-hot indices are not laid out [tokens, k]");
  if (b.frame != 0) Bail("the routing indices live inside a callee");
  r.indices = b.value;

  // The indices must come from a top-k.  Downstream nothing needs this (a
  // scatter of ANY index set is just as gatherable), but a different index
  // source means the pattern is something else -- a capacity-factor dispatch,
  // say -- and this recognizer has not established what.
  Deref tk = scope->deref(b.value, b.frame);
  if (!IsTopk(tk.op))
    Bail(absl::StrCat("routing indices do not come from a top-k (",
                      tk.op == nullptr ? "block argument" : OpName(tk.op),
                      ")"));
  if (TopkK(tk.op) != r.K) Bail("top-k width disagrees with the one-hot axis");
  if (ShapeOf(tk.op->getOperand(0)) != std::vector<int64_t>{r.T, r.E})
    Bail("top-k input is not [tokens, experts]");
  r.topk = tk.op;
  r.tframe = tk.frame;
  return r;
}

// --------------------------------------------------------------------------
// the planner
// --------------------------------------------------------------------------

class Planner {
 public:
  Planner(Scope* scope, RewritePlan* plan, int64_t E, int64_t T)
      : scope_(scope), plan_(plan), E_(E), T_(T) {}

  std::vector<MoeNode>& order() { return order_; }
  const std::vector<mlir::Value>& reads() const { return reads_; }
  llvm::DenseSet<mlir::Operation*>& region() { return region_; }

  int Plan(mlir::Value v, int frame, int ea, int ta) {
    Deref d = scope_->deref(v, frame);
    const Key key{d.frame, d.value, ea, ta};
    auto hit = nodes_.find(key);
    if (hit != nodes_.end()) return hit->second;
    const int at = Build(d.value, d.frame, d.op, ea, ta);
    nodes_[key] = at;
    return at;
  }

 private:
  struct Key {
    int frame;
    mlir::Value value;
    int ea, ta;
    bool operator<(const Key& o) const {
      if (frame != o.frame) return frame < o.frame;
      if (value.getAsOpaquePointer() != o.value.getAsOpaquePointer())
        return value.getAsOpaquePointer() < o.value.getAsOpaquePointer();
      if (ea != o.ea) return ea < o.ea;
      return ta < o.ta;
    }
  };

  int Push(MoeNode node) {
    order_.push_back(std::move(node));
    return static_cast<int>(order_.size()) - 1;
  }

  // A value computed OUTSIDE the region and gathered on the way in.
  int External(mlir::Value v, int frame, int ea, int ta) {
    if (frame != 0) {
      // Nothing in the environment holds a callee's internal values, so there
      // is no dense tensor to gather from.
      Bail("region value is bound inside a callee");
    }
    reads_.push_back(v);
    MoeNode n;
    n.kind = MoeNode::kExt;
    n.value = v;
    n.ea = ea;
    n.ta = ta;
    n.shape = ShapeOf(v);
    return Push(std::move(n));
  }

  int Build(mlir::Value v, int frame, mlir::Operation* o, int ea, int ta) {
    std::vector<int64_t> shape = ShapeOf(v);
    if (ea >= 0 && shape[ea] != E_) Bail("expert axis size disagrees");
    if (ta >= 0 && shape[ta] != T_) Bail("token axis size disagrees");
    // THE BOUNDARY: in the enclosing scope, only values that carry the expert
    // axis are region work.  Anything else is computed once, densely, and
    // gathered here -- traversing it would duplicate token-only work K times.
    // Inside a callee there is no such choice: its values are not in the
    // environment, so they are all region work (and a value with neither axis
    // stays unduplicated anyway).
    if (o == nullptr || (frame == 0 && ea < 0))
      return External(v, frame, ea, ta);
    const std::string name = OpName(o);
    if (name == "stablehlo.constant" || name == "stablehlo.iota") {
      if (ea >= 0 || ta >= 0)
        Bail(absl::StrCat("a per-expert ", name, " inside the region"));
      if (frame == 0) return External(v, frame, ea, ta);
      MoeNode n;                        // re-run the handler at emit
      n.kind = MoeNode::kExt;
      n.value = v;
      n.op = o;
      n.ea = ea;
      n.ta = ta;
      n.shape = shape;
      return Push(std::move(n));
    }
    if (frame == 0) region_.insert(o);
    if (Elementwise().contains(name)) return Elem(o, frame, ea, ta, shape);
    if (name == "stablehlo.concatenate")
      return Concat(o, frame, ea, ta, shape);
    if (name == "stablehlo.broadcast_in_dim")
      return Bcast(o, frame, ea, ta, shape);
    if (name == "stablehlo.transpose")
      return Transpose(o, frame, ea, ta, shape);
    if (name == "stablehlo.reshape") return Reshape(o, frame, ea, ta, shape);
    if (name == "stablehlo.slice") return Slice(o, frame, ea, ta, shape);
    if (name == "stablehlo.dot_general") return Dot(o, frame, ea, ta, shape);
    Bail(absl::StrCat(name, " is not modelled in pair space"));
  }

  // --- shape-preserving ---

  int Elem(mlir::Operation* o, int frame, int ea, int ta,
           const std::vector<int64_t>& shape) {
    MoeNode n;
    n.kind = MoeNode::kElem;
    n.op = o;
    n.ea = ea;
    n.ta = ta;
    n.shape = shape;
    for (mlir::Value x : o->getOperands()) {
      std::vector<int64_t> xs = ShapeOf(x);
      if (xs == shape) {
        n.srcs.push_back(Plan(x, frame, ea, ta));
      } else if (xs.empty()) {
        n.srcs.push_back(Plan(x, frame, -1, -1));
      } else {
        Bail(absl::StrCat(OpName(o), " operand shape does not match"));
      }
    }
    return Push(std::move(n));
  }

  int Concat(mlir::Operation* o, int frame, int ea, int ta,
             const std::vector<int64_t>& shape) {
    auto attr = o->getAttrOfType<mlir::IntegerAttr>("dimension");
    if (!attr) Bail("a concatenate with no dimension");
    const int64_t d = attr.getValue().getSExtValue();
    if (d == ea || d == ta)
      Bail("concatenate along the expert or token axis");
    MoeNode n;
    n.kind = MoeNode::kElem;
    n.op = o;
    n.ea = ea;
    n.ta = ta;
    n.shape = shape;
    for (mlir::Value x : o->getOperands()) n.srcs.push_back(Plan(x, frame, ea,
                                                                ta));
    return Push(std::move(n));
  }

  // --- shape ops ---

  int Bcast(mlir::Operation* o, int frame, int ea, int ta,
            const std::vector<int64_t>& shape) {
    std::vector<int64_t> bd = I64List(o, "broadcast_dimensions");
    std::vector<int64_t> sorted = bd;
    std::sort(sorted.begin(), sorted.end());
    if (bd != sorted) Bail("broadcast_dimensions are not increasing");
    mlir::Value src = o->getOperand(0);
    std::vector<int64_t> sshape = ShapeOf(src);
    int bea = -1, bta = -1;
    absl::flat_hash_set<int> drop;
    for (size_t j = 0; j < bd.size(); j++) {
      const int64_t d = bd[j];
      if (d != ea && d != ta) continue;
      const int64_t full = d == ea ? E_ : T_;
      // A UNIT source axis is a BROADCAST, never the axis itself.  The two
      // tests overlap when the axis is degenerate -- T = 1, which is every
      // decode step -- and reading the unit axis as "the token axis" is what
      // kept gemma4-26B-A4B's decode dense: it demands a token axis from
      // everything upstream, and a per-expert scale of shape [E] has none.
      if (sshape[j] == 1) {
        drop.insert(static_cast<int>(j));
      } else if (sshape[j] == full) {
        if (d == ea) bea = static_cast<int>(j);
        else bta = static_cast<int>(j);
      } else {
        Bail("expert/token axis broadcast from a part");
      }
    }
    const int node = Plan(src, frame, bea, bta);
    std::vector<int64_t> keep;
    for (size_t j = 0; j < sshape.size(); j++) {
      if (static_cast<int>(j) == bea || static_cast<int>(j) == bta) continue;
      keep.push_back(drop.contains(static_cast<int>(j))
                         ? -1
                         : TPos(shape, ea, ta, bd[j]));
    }
    std::vector<int64_t> dest;
    for (int64_t k : keep)
      if (k >= 0) dest.push_back(k);
    std::vector<int64_t> ordered = dest;
    std::sort(ordered.begin(), ordered.end());
    if (dest != ordered) Bail("broadcast reorders the trailing axes");
    MoeNode n;
    n.kind = MoeNode::kView;
    n.src = node;
    n.view = 0;
    n.keep = std::move(keep);
    n.trailing = Trailing(shape, ea, ta);
    n.ea = ea;
    n.ta = ta;
    n.shape = shape;
    return Push(std::move(n));
  }

  int Transpose(mlir::Operation* o, int frame, int ea, int ta,
                const std::vector<int64_t>& shape) {
    std::vector<int64_t> perm = I64List(o, "permutation");
    if (perm.size() != shape.size()) Bail("a transpose with a short perm");
    mlir::Value src = o->getOperand(0);
    const int bea = ea < 0 ? -1 : static_cast<int>(perm[ea]);
    const int bta = ta < 0 ? -1 : static_cast<int>(perm[ta]);
    const int node = Plan(src, frame, bea, bta);
    std::vector<int64_t> sshape = ShapeOf(src);
    std::vector<int64_t> order;
    for (size_t i = 0; i < shape.size(); i++) {
      if (static_cast<int>(i) == ea || static_cast<int>(i) == ta) continue;
      order.push_back(TPos(sshape, bea, bta, perm[i]));
    }
    MoeNode n;
    n.kind = MoeNode::kView;
    n.src = node;
    n.view = 1;
    n.order = std::move(order);
    n.ea = ea;
    n.ta = ta;
    n.shape = shape;
    return Push(std::move(n));
  }

  int Reshape(mlir::Operation* o, int frame, int ea, int ta,
              const std::vector<int64_t>& shape) {
    mlir::Value src = o->getOperand(0);
    std::vector<int64_t> sshape = ShapeOf(src);
    const int m = std::max(ea, ta) + 1;
    if (static_cast<int>(sshape.size()) < m ||
        !std::equal(sshape.begin(), sshape.begin() + m, shape.begin()))
      Bail("reshape crosses the expert/token axes");
    const int node = Plan(src, frame, ea, ta);
    MoeNode n;
    n.kind = MoeNode::kView;
    n.src = node;
    n.view = 2;
    n.trailing = Trailing(shape, ea, ta);
    n.ea = ea;
    n.ta = ta;
    n.shape = shape;
    return Push(std::move(n));
  }

  int Slice(mlir::Operation* o, int frame, int ea, int ta,
            const std::vector<int64_t>& shape) {
    std::vector<int64_t> start = I64List(o, "start_indices");
    std::vector<int64_t> limit = I64List(o, "limit_indices");
    std::vector<int64_t> stride = I64List(o, "strides");
    mlir::Value src = o->getOperand(0);
    std::vector<int64_t> sshape = ShapeOf(src);
    if (start.size() != sshape.size() || limit.size() != sshape.size() ||
        stride.size() != sshape.size())
      Bail("a slice whose attributes do not match its operand");
    for (int a : {ea, ta}) {
      if (a < 0) continue;
      if (start[a] != 0 || limit[a] != sshape[a] || stride[a] != 1)
        Bail("slice cuts the expert or token axis");
    }
    const int node = Plan(src, frame, ea, ta);
    MoeNode n;
    n.kind = MoeNode::kView;
    n.src = node;
    n.view = 3;
    for (size_t i = 0; i < sshape.size(); i++) {
      if (static_cast<int>(i) == ea || static_cast<int>(i) == ta) continue;
      n.slices.push_back(start[i]);
      n.slices.push_back(limit[i]);
      n.slices.push_back(stride[i]);
    }
    n.ea = ea;
    n.ta = ta;
    n.shape = shape;
    return Push(std::move(n));
  }

  // --- the per-expert dots ---

  int Dot(mlir::Operation* o, int frame, int ea, int ta,
          const std::vector<int64_t>& shape) {
    auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(o);
    if (!dot) Bail("a dot_general in an unexpected form");
    mlir::stablehlo::DotDimensionNumbersAttr dn = dot.getDotDimensionNumbers();
    std::vector<int64_t> lb(dn.getLhsBatchingDimensions().begin(),
                            dn.getLhsBatchingDimensions().end());
    std::vector<int64_t> rb(dn.getRhsBatchingDimensions().begin(),
                            dn.getRhsBatchingDimensions().end());
    std::vector<int64_t> lc(dn.getLhsContractingDimensions().begin(),
                            dn.getLhsContractingDimensions().end());
    std::vector<int64_t> rc(dn.getRhsContractingDimensions().begin(),
                            dn.getRhsContractingDimensions().end());
    mlir::Value lhs = o->getOperand(0), rhs = o->getOperand(1);
    std::vector<int64_t> lsh = ShapeOf(lhs), rsh = ShapeOf(rhs);
    auto holds = [](const std::vector<int64_t>& xs, int64_t v) {
      return std::find(xs.begin(), xs.end(), v) != xs.end();
    };
    std::vector<int64_t> lfree, rfree;
    for (int64_t d = 0; d < static_cast<int64_t>(lsh.size()); d++)
      if (!holds(lb, d) && !holds(lc, d)) lfree.push_back(d);
    for (int64_t d = 0; d < static_cast<int64_t>(rsh.size()); d++)
      if (!holds(rb, d) && !holds(rc, d)) rfree.push_back(d);
    const int64_t nb = static_cast<int64_t>(lb.size());
    if (ta < 0) Bail("per-expert dot without a token axis");

    // `(side, dim)`; side 2 means a batching dim (both sides).
    auto where = [&](int64_t axis, int* side, int64_t* dim) {
      if (axis < nb) {
        *side = 2;
        *dim = axis;
      } else if (axis < nb + static_cast<int64_t>(lfree.size())) {
        *side = 0;
        *dim = lfree[axis - nb];
      } else {
        const int64_t i = axis - nb - static_cast<int64_t>(lfree.size());
        if (i >= static_cast<int64_t>(rfree.size()))
          Bail("a dot axis outside the result");
        *side = 1;
        *dim = rfree[i];
      }
    };
    int eside = 0, tside = 0;
    int64_t edim = 0, tdim = 0;
    where(ea, &eside, &edim);
    where(ta, &tside, &tdim);
    if (tside == 2) Bail("the token axis is a dot batching dimension");
    const int dside = tside, wside = 1 - tside;
    int64_t wdim = 0, ddim = -1;
    if (eside == 2) {
      wdim = (wside == 0 ? lb : rb)[edim];
      ddim = (dside == 0 ? lb : rb)[edim];
    } else if (eside == wside) {
      wdim = edim;
      ddim = -1;
    } else {
      Bail("the expert axis belongs to the data operand only");
    }

    mlir::Value wv = wside == 0 ? lhs : rhs;
    mlir::Value dv = dside == 0 ? lhs : rhs;
    std::vector<int64_t> wsh = ShapeOf(wv), dsh = ShapeOf(dv);
    const std::vector<int64_t>& wb = wside == 0 ? lb : rb;
    const std::vector<int64_t>& wc = wside == 0 ? lc : rc;
    const std::vector<int64_t>& db = dside == 0 ? lb : rb;
    const std::vector<int64_t>& dc = dside == 0 ? lc : rc;
    std::vector<int64_t> wfree, dfree;
    for (int64_t d = 0; d < static_cast<int64_t>(wsh.size()); d++)
      if (!holds(wb, d) && !holds(wc, d)) wfree.push_back(d);
    for (int64_t d = 0; d < static_cast<int64_t>(dsh.size()); d++)
      if (!holds(db, d) && !holds(dc, d)) dfree.push_back(d);
    if (wb.size() > 1 || (!wb.empty() && wb != std::vector<int64_t>{wdim}))
      Bail("the dot batches over more than the experts");
    if (db.size() > 1) Bail("the data operand has extra batching dims");
    std::vector<int64_t> wn;
    for (int64_t d : wfree)
      if (d != wdim) wn.push_back(d);
    const std::vector<int64_t> wk = wc;
    // gather_* wants the weight as [E, ., .]: the expert axis outermost and
    // the two matrix axes contiguous, so the 3-D view stays free.
    if (wdim != 0) Bail("the expert axis is not the weight's leading axis");
    std::vector<int64_t> ramp;
    for (size_t i = 1; i < wsh.size(); i++) ramp.push_back(
        static_cast<int64_t>(i));
    std::vector<int64_t> nk = wn, kn = wk;
    nk.insert(nk.end(), wk.begin(), wk.end());
    kn.insert(kn.end(), wn.begin(), wn.end());
    bool n_first;
    if (nk == ramp) {
      n_first = true;
    } else if (kn == ramp) {
      n_first = false;
    } else {
      Bail("the weight's matrix dims are interleaved");
    }

    MoeNode node;
    node.kind = MoeNode::kDot;
    node.ea = ea;
    node.ta = ta;
    node.shape = shape;
    node.n_first = n_first;
    for (int64_t d : dfree)
      if (d != tdim) node.mshape.push_back(dsh[d]);
    for (int64_t d : wn) node.nshape.push_back(wsh[d]);
    node.M = Prod(node.mshape);
    node.N = Prod(node.nshape);
    std::vector<int64_t> kdims;
    for (int64_t d : wk) kdims.push_back(wsh[d]);
    node.K = Prod(kdims);
    std::vector<int64_t> dk;
    for (int64_t d : dc) dk.push_back(dsh[d]);
    if (node.K != Prod(dk)) Bail("contraction sizes disagree");
    if (node.M == 0 || node.N == 0 || node.K == 0) Bail("empty per-expert dot");
    node.out_dtype = DtypeCodeOf(o->getResult(0));

    // The dense result lists batching, then lhs free, then rhs free;
    // gather_* returns [P] + M + N, so record the reordering.
    std::vector<int64_t> mdims;
    for (int64_t d : dfree)
      if (d != tdim) mdims.push_back(d);
    std::vector<int64_t> perm;
    for (size_t i = 0; i < shape.size(); i++) {
      if (static_cast<int>(i) == ea || static_cast<int>(i) == ta) continue;
      int side = 0;
      int64_t dim = 0;
      where(static_cast<int64_t>(i), &side, &dim);
      if (side == 2) Bail("unmodelled batching dim in a per-expert dot");
      if (side == dside) {
        auto it = std::find(mdims.begin(), mdims.end(), dim);
        if (it == mdims.end()) Bail("a dot result axis with no data dim");
        perm.push_back(it - mdims.begin());
      } else {
        auto it = std::find(wn.begin(), wn.end(), dim);
        if (it == wn.end()) Bail("a dot result axis with no weight dim");
        perm.push_back(static_cast<int64_t>(mdims.size()) +
                       (it - wn.begin()));
      }
    }

    QmmMatch* pack = Packed(o, wside);
    if (pack != nullptr) {
      node.pack = pack;
    } else {
      // The weight stays outside the region: read dense from the environment
      // and let gather_mm do the indexing (no [P, N, K] copy is ever made).
      Deref w = scope_->deref(wv, frame);
      if (w.frame != 0) Bail("per-expert weight is bound inside a callee");
      if (ShapeOf(w.value) != wsh)
        Bail("per-expert weight changed shape under deref");
      node.weight = w.value;
      reads_.push_back(w.value);
    }
    node.data = Plan(dv, frame, static_cast<int>(ddim),
                     static_cast<int>(tdim));
    const int at = Push(std::move(node));

    MoeNode view;
    view.kind = MoeNode::kView;
    view.src = at;
    view.view = 4;
    view.order = perm;
    view.trailing = Trailing(shape, ea, ta);
    view.ea = ea;
    view.ta = ta;
    view.shape = shape;
    return Push(std::move(view));
  }

  // The qmm match that already fused this dot, if there is one.
  QmmMatch* Packed(mlir::Operation* dot, int wside) {
    for (const auto& m : plan_->qmm) {
      if (m->root != dot || m->disabled) continue;
      if (m->swapped != (wside == 0)) Bail("qmm packed the other dot operand");
      if (m->bshape.size() > 1)
        Bail("packed weight has several batching dims");
      if (!m->bshape.empty() && m->bshape[0] != E_)
        Bail("packed batching dim is not the expert axis");
      if (m->bshape.empty() && m->N % E_)
        Bail("packed N is not a multiple of the experts");
      return m.get();
    }
    return nullptr;
  }

  Scope* scope_;
  RewritePlan* plan_;
  int64_t E_, T_;
  std::map<Key, int> nodes_;
  std::vector<MoeNode> order_;
  llvm::DenseSet<mlir::Operation*> region_;
  std::vector<mlir::Value> reads_;
};

// --------------------------------------------------------------------------
// matches
// --------------------------------------------------------------------------

// moe.py `_protect_closure`: `values` plus everything their computation
// depends on.
llvm::DenseSet<mlir::Value> ProtectClosure(
    const std::vector<mlir::Value>& values) {
  llvm::DenseSet<mlir::Value> out;
  std::vector<mlir::Value> stack = values;
  while (!stack.empty()) {
    mlir::Value v = stack.back();
    stack.pop_back();
    if (!out.insert(v).second) continue;
    mlir::Operation* o = Owner(v);
    if (o == nullptr) continue;
    for (mlir::Value x : o->getOperands()) stack.push_back(x);
    for (mlir::Region& r : o->getRegions())
      for (mlir::Block& b : r.getBlocks())
        for (mlir::Operation& inner : b)
          for (mlir::Value res : inner.getResults()) stack.push_back(res);
  }
  return out;
}

// moe.py `_dead_sweep`: add ops whose every use is inside the rewritten
// region.  The dense router scores, the one-hot and their broadcasts are only
// ever read by the sum being replaced; leaving them behind would keep
// computing a [tokens, experts] tensor nobody looks at.
void DeadSweep(mlir::Block& block, llvm::DenseSet<mlir::Operation*>* skip,
               const llvm::DenseSet<mlir::Value>& protect,
               mlir::Operation* root) {
  llvm::DenseSet<mlir::Operation*> inblock;
  for (mlir::Operation& op : block)
    if (op.getNumResults() > 0) inblock.insert(&op);

  std::deque<mlir::Operation*> work;
  llvm::DenseSet<mlir::Operation*> queued;
  auto enqueue = [&](mlir::Operation* o) {
    for (mlir::Value v : o->getOperands()) {
      mlir::Operation* d = Owner(v);
      if (d == nullptr || skip->contains(d) || d == root ||
          !inblock.contains(d) || queued.contains(d))
        continue;
      queued.insert(d);
      work.push_back(d);
    }
  };
  // Seeds: everything already inside the region, plus the root itself -- a
  // candidate qualifies only if all its users are in one of those, so it
  // necessarily defines an operand of one of them.
  std::vector<mlir::Operation*> seeds(skip->begin(), skip->end());
  seeds.push_back(root);
  for (mlir::Operation* o : seeds) enqueue(o);

  while (!work.empty()) {
    mlir::Operation* o = work.front();
    work.pop_front();
    queued.erase(o);
    if (skip->contains(o) || NeverSweep().contains(OpName(o))) continue;
    bool protected_result = false;
    for (mlir::Value r : o->getResults())
      protected_result = protected_result || protect.contains(r);
    if (protected_result) continue;
    bool any = false, all_inside = true;
    for (mlir::Value r : o->getResults()) {
      for (mlir::Operation* u : r.getUsers()) {
        any = true;
        if (!skip->contains(u) && u != root) all_inside = false;
      }
    }
    if (!any || !all_inside) continue;
    skip->insert(o);
    enqueue(o);
  }
}

std::unique_ptr<MoeMatch> AnalyzeRoot(mlir::ModuleOp module, RewritePlan* plan,
                                      mlir::Block& block,
                                      mlir::Operation* root) {
  Scope scope(module);
  const int64_t e_axis = ZeroSum(root);
  Deref mul = scope.deref(root->getOperand(0), 0);
  // A bf16/f16 model sums in f32, so jax puts a convert between the product
  // and the sum.  It stays in the region: the gathered products are converted
  // the same way before being summed (see `sum_dtype`).
  std::vector<mlir::Operation*> casts;
  while (mul.op != nullptr && OpName(mul.op) == "stablehlo.convert") {
    casts.push_back(mul.op);
    mul = scope.deref(mul.op->getOperand(0), mul.frame);
  }
  if (mul.op == nullptr || OpName(mul.op) != "stablehlo.multiply" ||
      mul.frame != 0)
    Bail("the expert sum does not reduce a product");
  std::vector<int64_t> mshape = ShapeOf(mul.op->getResult(0));
  if (mshape.size() < 2) Bail("the expert product is rank < 2");

  std::optional<Router> router;
  int64_t t_axis = -1;
  mlir::Value yv;
  for (int i = 0; i < 2; i++) {
    Peeled b = Peel(&scope, mul.op->getOperand(i), mul.frame);
    if (b.dims.size() != 2) continue;
    auto it = std::find(b.dims.begin(), b.dims.end(), e_axis);
    if (it == b.dims.end()) continue;
    const int64_t se = it - b.dims.begin();
    const int64_t st = 1 - se;
    if (b.dims[st] < 0 || b.dims[st] == e_axis) continue;
    router = MatchRouter(&scope, b.value, b.frame, se, st);
    t_axis = b.dims[st];
    yv = mul.op->getOperand(1 - i);
    break;
  }
  if (!router.has_value())
    Bail("neither product operand is a routing-score tensor");
  if (mshape[e_axis] != router->E || mshape[t_axis] != router->T)
    Bail("routing scores do not match the expert-output shape");

  auto m = std::make_unique<MoeMatch>();
  m->root = root;
  m->indices = router->indices;
  m->weights = router->weights;
  m->E = router->E;
  m->T = router->T;
  m->K = router->K;
  m->P = router->T * router->K;
  for (size_t i = 0; i < mshape.size(); i++)
    if (static_cast<int64_t>(i) != e_axis) m->out_shape.push_back(mshape[i]);
  m->out_axis = TPos(mshape, static_cast<int>(e_axis), -1, t_axis);
  m->out_dtype = DtypeCodeOf(root->getResult(0));
  m->sum_dtype = DtypeCodeOf(root->getOperand(0));
  // What the first-execute check reads (see `VerifyMoe`): the routing-score
  // operand as the PRODUCT saw it -- broadcast to the expert output's shape,
  // which is where the two axes the match read are `e_axis` and `t_axis`.
  m->score3 = mul.op->getOperand(yv == mul.op->getOperand(0) ? 1 : 0);
  m->e_ax = e_axis;
  m->t_ax = t_axis;
  {
    // The top-k's input is what the check substitutes; a callee's value
    // cannot be pinned by a cone, so such a match stays unverified.
    Deref lv = scope.deref(router->topk->getOperand(0), router->tframe);
    if (lv.frame == 0) {
      m->logits = lv.value;
      m->logits_dtype = DtypeCodeOf(lv.value);
    }
  }

  Planner planner(&scope, plan, router->E, router->T);
  planner.region().insert(mul.op);
  for (mlir::Operation* c : casts) planner.region().insert(c);
  // Only the calls entered from HERE on belong to the region; the router
  // match above may have walked through calls of its own (jax emits `one_hot`
  // as one), and those stay in the dense graph.
  const size_t mark = scope.calls().size();
  const int out = planner.Plan(yv, mul.frame, static_cast<int>(e_axis),
                               static_cast<int>(t_axis));
  std::vector<MoeNode>& order = planner.order();
  if (order[out].kind == MoeNode::kExt)
    Bail("the expert output is not computed in the region");
  bool has_dot = false;
  for (const MoeNode& n : order) has_dot = has_dot || n.kind == MoeNode::kDot;
  if (!has_dot) Bail("the region contains no per-expert dot");
  if (order[out].shape != mshape)
    Bail("the expert output is not the product's shape");

  for (size_t i = mark; i < scope.calls().size(); i++)
    planner.region().insert(scope.calls()[i]);
  // USE-COUNT DISCIPLINE: a region op may only be skipped when nothing
  // outside reads it.  Anything shared with an aux-loss head, a residual, or
  // a second consumer of the expert outputs fails the match closed.
  for (mlir::Operation* o : planner.region()) {
    for (mlir::Value r : o->getResults()) {
      for (mlir::Operation* u : r.getUsers()) {
        if (!planner.region().contains(u) && u != root)
          Bail(absl::StrCat(OpName(o), " feeds ", OpName(u),
                            " outside the region"));
      }
    }
  }
  // Invariant: everything the emit reads out of the environment must still be
  // computed.  (A value reached both as region work and as an outside read
  // would be skipped and then looked up.)
  for (mlir::Value v : planner.reads()) {
    mlir::Operation* o = Owner(v);
    if (o != nullptr && planner.region().contains(o))
      Bail(absl::StrCat(OpName(o), " is both region work and a run-time read"));
  }

  m->order = std::move(order);
  m->out = out;
  llvm::DenseSet<mlir::Operation*> skip = planner.region();
  std::vector<mlir::Value> protect = planner.reads();
  protect.push_back(router->indices);
  protect.push_back(router->weights);
  DeadSweep(block, &skip, ProtectClosure(protect), root);
  m->ops.assign(skip.begin(), skip.end());
  m->name = absl::StrFormat("E%d/K%d/T%d->%s", m->E, m->K, m->T,
                            absl::StrJoin(m->out_shape, "x"));
  return m;
}

// moe.py `_verify`'s draws: `[T, E]` standard normals, coarse from the second
// one on (exact ties are where a top-k and a hand-rolled scatter would most
// easily disagree).  Deterministic, on the host, so a failure is reproducible.
mx::array SyntheticLogits(int64_t T, int64_t E, int draw, mx::Dtype dt) {
  const size_t n = static_cast<size_t>(T * E);
  std::vector<float> host(n);
  uint64_t state = 20260803u + static_cast<uint64_t>(draw) * 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i < n; i++) {
    // A Box-Muller pair off a 64-bit LCG: the distribution only has to be
    // spread out, not principled.
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const double u1 = ((state >> 11) + 1.0) / 9007199254740994.0;
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const double u2 = (state >> 11) / 9007199254740992.0;
    double v = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    if (draw) v = std::round(v * 2.0) / 2.0;
    host[i] = static_cast<float>(v);
  }
  void* buf = std::malloc(std::max<size_t>(1, n * sizeof(float)));
  if (buf == nullptr) throw std::bad_alloc();
  std::memcpy(buf, host.data(), n * sizeof(float));
  mx::array raw(buf, mx::Shape{static_cast<mx::ShapeElem>(T),
                               static_cast<mx::ShapeElem>(E)},
                mx::float32, [](void* p) { std::free(p); });
  return dt == mx::float32 ? raw : mx::astype(raw, dt);
}

// The check itself: the scores must BE the top-k weights scattered at the
// matched indices, and no token may have more than K of them.
void CheckRouter(const MoeMatch& m, const mx::array& score3,
                 const mx::array& idx, const mx::array& wgt) {
  const auto T = static_cast<mx::ShapeElem>(m.T);
  const auto E = static_cast<mx::ShapeElem>(m.E);
  const auto P = static_cast<mx::ShapeElem>(m.P);
  mx::array rows = mx::repeat(mx::arange(static_cast<double>(m.T), mx::uint32),
                              static_cast<int>(m.K));
  mx::array flat_idx = mx::astype(mx::reshape(idx, mx::Shape{P}), mx::uint32);
  mx::array ref = mx::zeros(mx::Shape{T, E}, score3.dtype());
  // One element of the [T, E] grid per pair: 1-D index vectors, so the
  // updates carry one trailing unit axis per scattered axis.
  ref = mx::scatter_add(
      ref, {rows, flat_idx},
      mx::reshape(mx::astype(wgt, score3.dtype()), mx::Shape{P, 1, 1}),
      std::vector<int>{0, 1});
  // ...laid out the way the product's operand is: the two axes the match read
  // carry it and every other axis of the product is a broadcast.
  mx::Shape want(score3.ndim(), 1);
  if (m.e_ax >= static_cast<int64_t>(want.size()) ||
      m.t_ax >= static_cast<int64_t>(want.size()))
    Bail("the score operand lost an axis the match read");
  want[m.e_ax] = E;
  want[m.t_ax] = T;
  mx::array ref3 = mx::broadcast_to(
      mx::reshape(m.t_ax < m.e_ax ? ref : mx::transpose(ref, {1, 0}), want),
      score3.shape());
  mx::array ok = mx::all(mx::equal(score3, ref3));
  mx::array nz = mx::max(
      mx::sum(mx::astype(
                  mx::not_equal(ref, mx::zeros(mx::Shape{}, ref.dtype())),
                  mx::int32),
              std::vector<int>{1}),
      false);
  mx::eval(ok, nz);
  if (!ok.item<bool>())
    Bail("router scores are not the top-k weights scattered at the matched "
         "indices");
  if (nz.item<int>() > static_cast<int>(m.K))
    Bail("a token has more than K nonzero routing weights");
}

}  // namespace

// --------------------------------------------------------------------------
// the public surface
// --------------------------------------------------------------------------

void AnalyzeMoe(mlir::func::FuncOp fn, RewritePlan* plan) {
  if (EnvOff("METALJAX_MOE")) return;
  if (fn.getBody().getBlocks().size() != 1) return;
  auto module = fn->getParentOfType<mlir::ModuleOp>();
  if (!module) return;

  // Every block reachable from @main is searched, which includes the bodies
  // of while / if regions -- a sampler's decode step is a MoE dispatch like
  // any other and must fuse like one.
  std::vector<std::unique_ptr<MoeMatch>> found;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      if (OpName(&op) == "stablehlo.reduce") {
        try {
          found.push_back(AnalyzeRoot(module, plan, block, &op));
        } catch (const Reject& e) {
          if (kDebug && op.getNumOperands() > 0 &&
              mlir::isa<mlir::RankedTensorType>(op.getOperand(0).getType()) &&
              mlir::cast<mlir::RankedTensorType>(op.getOperand(0).getType())
                      .getRank() >= 3) {
            Debug(absl::StrCat("rejected a candidate (", e.why, ")"));
          }
        } catch (const std::exception& e) {
          if (kDebug) Debug(absl::StrCat("analysis error (", e.what(), ")"));
        }
      }
      for (mlir::Region& r : op.getRegions())
        for (mlir::Block& b : r.getBlocks()) walk(b);
    }
  };
  walk(fn.getBody().front());

  llvm::DenseSet<mlir::Operation*> seen;
  for (auto& m : found) {
    if (seen.contains(m->root)) continue;
    bool overlaps = false;
    for (mlir::Operation* o : m->ops) overlaps = overlaps || seen.contains(o);
    if (overlaps) continue;
    for (mlir::Operation* o : m->ops) seen.insert(o);
    seen.insert(m->root);
    // A quantized dot this dispatch takes over is still PACKED, but the dense
    // quantized_matmul is never emitted: gather_qmm replaces it.
    for (const MoeNode& n : m->order)
      if (n.kind == MoeNode::kDot && n.pack != nullptr) n.pack->absorbed = true;
    plan->moe.push_back(std::move(m));
  }
  if (!plan->moe.empty()) {
    std::vector<std::string> names;
    for (const auto& m : plan->moe) names.push_back(m->name);
    Debug(absl::StrCat(plan->moe.size(), " expert dispatch(es) gathered (",
                       absl::StrJoin(names, ", "), ")"));
  }
  plan->rebuild();
}

absl::Status VerifyMoe(RewritePlan* plan, const SubtreeEval& eval) {
  bool changed = false;
  for (auto& m : plan->moe) {
    if (m->disabled) continue;
    // A dispatch whose quantized weight did not pack falls back with it.
    for (const MoeNode& n : m->order) {
      if (n.kind == MoeNode::kDot && n.pack != nullptr && n.pack->disabled) {
        m->disabled = true;
        changed = true;
        Debug(absl::StrCat(m->name,
                           " falls back (its quantized weight did not pack)"));
      }
    }
    if (m->disabled || EnvOff("METALJAX_MOE_VERIFY")) continue;
    try {
      // moe.py `_verify`: the router tail below the top-k depends on nothing
      // but the logits, so a random draw is a complete functional check of
      // the structural reading -- which axis is which, and that the weights
      // really land at the matched indices -- and it needs no real
      // activations at all.  Substituting them is also the only way to reach
      // a dispatch inside a DECODE LOOP, whose real logits are a loop carry
      // no prologue could evaluate.
      if (m->logits == nullptr)
        Bail("the top-k's input is bound inside a callee");
      for (int draw = 0; draw < kVerifyDraws; draw++) {
        absl::StatusOr<std::vector<mx::array>> got =
            eval({m->score3, m->indices, m->weights},
                 {{m->logits, SyntheticLogits(m->T, m->E, draw,
                                              dtype_of(m->logits_dtype))}});
        if (!got.ok()) Bail(std::string(got.status().message()));
        CheckRouter(*m, (*got)[0], (*got)[1], (*got)[2]);
      }
      Debug(absl::StrCat(m->name, " verified"));
    } catch (const Reject& e) {
      m->disabled = true;
      changed = true;
      Debug(absl::StrCat(m->name, " falls back to the dense dispatch (", e.why,
                         ")"));
    } catch (const std::exception& e) {
      m->disabled = true;
      changed = true;
      Debug(absl::StrCat(m->name, " falls back to the dense dispatch (",
                         e.what(), ")"));
    }
  }
  if (changed) {
    // A disabled dispatch hands its packed dots back to the dense rewrite.
    for (auto& m : plan->moe) {
      if (!m->disabled) continue;
      for (const MoeNode& n : m->order)
        if (n.kind == MoeNode::kDot && n.pack != nullptr)
          n.pack->absorbed = false;
    }
    std::vector<std::unique_ptr<MoeMatch>> live;
    for (auto& m : plan->moe)
      if (!m->disabled) live.push_back(std::move(m));
    plan->moe = std::move(live);
    plan->rebuild();
  }
  return absl::OkStatus();
}

}  // namespace metaljax
