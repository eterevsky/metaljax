/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The RMS-norm recognizer: jax's spelled-out root-mean-square norm, rewritten
into MLX's fused `fast::rms_norm`.

maxtext (and every flax/keras transformer in the model table) spells one
norm as ~13 StableHLO ops: upcast to f32, square, sum over the feature
axis, divide by N, add eps, rsqrt, scale, downcast, then apply the learned
[N] weight as a batching-only dot (`x * w` — the K=1 dot the runtime
already turns into a broadcast multiply) plus its transpose.  Three of
those per decode layer put ~39 tape entries per layer on a path where
every entry costs ~2 us of build/schedule/launch (the row-10 op-count
campaign, notes/row10-decode-floor-2026-08-29.md).

The rewrite emits ONE kRmsNorm (runtime/emits.cc): `fast::rms_norm(x, w,
eps)`, whose kernel accumulates in f32 exactly as the literal chain does.
One numeric difference, the usual fused-kernel class: the chain rounds
`x * rsqrt(mean)` to bf16 BEFORE multiplying by the weight, the kernel
multiplies in f32 and rounds once — a 1-ULP-class difference on bf16,
tolerance-level against the CPU.

A half-matched pattern lowers as ORDINARY ops (`Bail`), and a match whose
intermediates escape is declined whole — the fused op computes everything
internally, so a partial absorption cannot stand.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace metaljax {
namespace {

const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();

void Debug(const std::string& line) {
  if (!kDebug) return;
  std::fprintf(stderr, "[metaljax-native] norm: %s\n", line.c_str());
  std::fflush(stderr);
}

bool EnvOff(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "0";
}

struct Reject {
  std::string why;
};
[[noreturn]] void Bail(const std::string& why) { throw Reject{why}; }

std::string OpName(mlir::Operation* op) {
  return op == nullptr ? std::string() : op->getName().getStringRef().str();
}

std::vector<int64_t> ShapeOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t || !t.hasStaticShape()) Bail("a value without a static shape");
  return std::vector<int64_t>(t.getShape().begin(), t.getShape().end());
}

mlir::Type ElemOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) Bail("a value that is not a ranked tensor");
  return t.getElementType();
}

mlir::Operation* DefOf(mlir::Value v, const char* what) {
  mlir::Operation* op = v.getDefiningOp();
  if (op == nullptr) Bail(absl::StrCat(what, " is a block argument"));
  return op;
}

std::optional<double> SplatFloatOf(mlir::Value v) {
  auto cst =
      mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return std::nullopt;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return std::nullopt;
  if (!mlir::isa<mlir::FloatType>(dense.getElementType())) return std::nullopt;
  return dense.getSplatValue<mlir::APFloat>().convertToDouble();
}

// A splat, or a broadcast of one, absorbed on success.
std::optional<double> SplatOrBroadcastFloat(
    mlir::Value v, std::vector<mlir::Operation*>* ops) {
  if (auto d = SplatFloatOf(v)) return d;
  auto b = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      v.getDefiningOp());
  if (!b) return std::nullopt;
  auto d = SplatFloatOf(b.getOperand());
  if (d.has_value() && ops != nullptr) ops->push_back(b);
  return d;
}

bool IsZeroInit(mlir::Value v) {
  auto cst =
      mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return false;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return false;
  if (mlir::isa<mlir::FloatType>(dense.getElementType()))
    return dense.getSplatValue<mlir::APFloat>().isZero();
  return false;
}

std::unique_ptr<RmsNormMatch> MatchRoot(mlir::Operation* op) {
  auto tr = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
  if (!tr) Bail("not a transpose");
  std::vector<int64_t> out_shape = ShapeOf(tr.getResult());
  const size_t R = out_shape.size();
  if (R < 1) Bail("a rank-0 norm");
  // perm must be [1, 2, .., R-1, 0]: the [N, leading...] dot back to x's
  // layout.
  auto perm = tr.getPermutation();
  if (perm.size() != R) Bail("rank mismatch");
  for (size_t i = 0; i + 1 < R; i++)
    if (perm[i] != static_cast<int64_t>(i) + 1) Bail("not the weight-apply");
  if (perm[R - 1] != 0) Bail("not the weight-apply");
  const int64_t N = out_shape[R - 1];

  auto m = std::make_unique<RmsNormMatch>();
  m->root = op;
  std::vector<mlir::Operation*>& ops = m->ops;

  // dot_general(w', y): batching [0] x [R-1], no contraction.
  auto dot = mlir::dyn_cast_or_null<mlir::stablehlo::DotGeneralOp>(
      DefOf(tr.getOperand(), "the weight apply"));
  if (!dot) Bail("the weight apply is not a dot");
  ops.push_back(dot);
  auto dn = dot.getDotDimensionNumbers();
  auto eq_dims = [](llvm::ArrayRef<int64_t> a, const std::vector<int64_t>& b) {
    return std::vector<int64_t>(a.begin(), a.end()) == b;
  };
  if (!eq_dims(dn.getLhsBatchingDimensions(), {0}) ||
      !eq_dims(dn.getRhsBatchingDimensions(),
               {static_cast<int64_t>(R) - 1}) ||
      !dn.getLhsContractingDimensions().empty() ||
      !dn.getRhsContractingDimensions().empty())
    Bail("the weight apply is not the batching multiply");

  // w' = w, or w + broadcast(0).
  mlir::Value w = dot.getLhs();
  if (auto add = mlir::dyn_cast_or_null<mlir::stablehlo::AddOp>(
          w.getDefiningOp())) {
    for (bool swap : {false, true}) {
      mlir::Value a = swap ? add.getRhs() : add.getLhs();
      mlir::Value b = swap ? add.getLhs() : add.getRhs();
      std::vector<mlir::Operation*> zops;
      auto z = SplatOrBroadcastFloat(b, &zops);
      if (z.has_value() && *z == 0.0) {
        ops.push_back(add);
        ops.insert(ops.end(), zops.begin(), zops.end());
        w = a;
        break;
      }
    }
  }
  if (ShapeOf(w) != std::vector<int64_t>{N}) Bail("the weight is not [N]");

  // y = convert?(xn * broadcast(rsqrt)).
  mlir::Value y = dot.getRhs();
  if (auto cv = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(
          y.getDefiningOp())) {
    ops.push_back(cv);
    y = cv.getOperand();
  }
  auto mul2 = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(
      DefOf(y, "the normalized value"));
  if (!mul2) Bail("the normalized value is not a multiply");
  ops.push_back(mul2);
  mlir::Value xn, b10;
  {
    auto is_b = [](mlir::Value v) {
      return mlir::isa_and_nonnull<mlir::stablehlo::BroadcastInDimOp>(
          v.getDefiningOp());
    };
    // The rsqrt side is the broadcast; try both orders, preferring the one
    // whose broadcast leads to an rsqrt.
    for (bool swap : {false, true}) {
      mlir::Value a = swap ? mul2.getRhs() : mul2.getLhs();
      mlir::Value b = swap ? mul2.getLhs() : mul2.getRhs();
      if (!is_b(b)) continue;
      auto bc = mlir::cast<mlir::stablehlo::BroadcastInDimOp>(
          b.getDefiningOp());
      if (mlir::isa_and_nonnull<mlir::stablehlo::RsqrtOp>(
              bc.getOperand().getDefiningOp())) {
        xn = a;
        b10 = b;
        break;
      }
    }
  }
  if (!xn) Bail("the normalized value has no rsqrt side");
  auto b10op =
      mlir::cast<mlir::stablehlo::BroadcastInDimOp>(b10.getDefiningOp());
  ops.push_back(b10op);
  auto rs = mlir::cast<mlir::stablehlo::RsqrtOp>(
      b10op.getOperand().getDefiningOp());
  ops.push_back(rs);

  // rsqrt(divide(broadcast(sum(x^2)), N) + eps).
  auto add6 = mlir::dyn_cast_or_null<mlir::stablehlo::AddOp>(
      rs.getOperand().getDefiningOp());
  if (!add6) Bail("no eps add");
  ops.push_back(add6);
  mlir::Value div_v;
  std::optional<double> eps;
  for (bool swap : {false, true}) {
    mlir::Value a = swap ? add6.getRhs() : add6.getLhs();
    mlir::Value b = swap ? add6.getLhs() : add6.getRhs();
    std::vector<mlir::Operation*> eops;
    auto e = SplatOrBroadcastFloat(b, &eops);
    if (!e.has_value()) continue;
    if (!mlir::isa_and_nonnull<mlir::stablehlo::DivOp>(a.getDefiningOp()))
      continue;
    eps = e;
    div_v = a;
    ops.insert(ops.end(), eops.begin(), eops.end());
    break;
  }
  if (!eps.has_value()) Bail("no eps splat");
  auto div6 = mlir::cast<mlir::stablehlo::DivOp>(div_v.getDefiningOp());
  ops.push_back(div6);
  std::vector<mlir::Operation*> nops;
  auto divisor = SplatOrBroadcastFloat(div6.getRhs(), &nops);
  if (!divisor.has_value() || *divisor != static_cast<double>(N))
    Bail("the divisor is not the feature size");
  ops.insert(ops.end(), nops.begin(), nops.end());
  std::vector<int64_t> lead_dims;
  for (size_t i = 0; i + 1 < R; i++)
    lead_dims.push_back(static_cast<int64_t>(i));
  auto b4 = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      div6.getLhs().getDefiningOp());
  if (!b4) Bail("the sum is not broadcast back");
  {
    auto bd = b4.getBroadcastDimensions();
    if (std::vector<int64_t>(bd.begin(), bd.end()) != lead_dims)
      Bail("the sum broadcast has the wrong dims");
  }
  ops.push_back(b4);
  auto red = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(
      b4.getOperand().getDefiningOp());
  if (!red || red.getNumOperands() != 2 || red->getNumResults() != 1)
    Bail("the sum is not a single-input reduce");
  {
    auto dims = red.getDimensions();
    if (dims.size() != 1 || dims[0] != static_cast<int64_t>(R) - 1)
      Bail("the sum reduces the wrong axis");
    mlir::Block& body = red.getBody().front();
    if (body.getOperations().size() != 2 ||
        OpName(&body.front()) != "stablehlo.add")
      Bail("the sum body is not an add");
    if (!IsZeroInit(red.getInitValues()[0])) Bail("the sum init is not zero");
  }
  ops.push_back(red);
  auto sq = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(
      red.getInputs()[0].getDefiningOp());
  if (!sq || sq.getLhs() != sq.getRhs() || sq.getLhs() != xn)
    Bail("the sum is not over the square of the normed value");
  ops.push_back(sq);

  // xn = convert?(x); the emit hands the ORIGINAL x to the fused kernel.
  mlir::Value x = xn;
  if (auto cv = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(
          x.getDefiningOp())) {
    if (!mlir::isa<mlir::Float32Type>(ElemOf(cv.getResult())))
      Bail("the upcast is not to f32");
    ops.push_back(cv);
    x = cv.getOperand();
  }
  if (ShapeOf(x) != out_shape) Bail("the input shape disagrees");
  mlir::Type elem = ElemOf(x);
  if (!mlir::isa<mlir::FloatType>(elem) || mlir::isa<mlir::Float64Type>(elem))
    Bail("not a float norm");
  if (ElemOf(w) != elem || ElemOf(tr.getResult()) != elem)
    Bail("mixed dtypes");

  m->x = x;
  m->w = w;
  m->eps = *eps;
  m->name = absl::StrCat("N", N, "R", R);
  return m;
}

}  // namespace

void AnalyzeNorm(mlir::func::FuncOp fn, RewritePlan* plan) {
  if (EnvOff("METALJAX_NORM")) return;
  if (fn.getBody().getBlocks().size() != 1) return;
  auto module = fn->getParentOfType<mlir::ModuleOp>();
  if (!module) return;

  llvm::DenseSet<mlir::Operation*> taken = plan->skip;
  auto take = [&](mlir::Operation* root,
                  const std::vector<mlir::Operation*>& ops) {
    taken.insert(root);
    for (mlir::Operation* o : ops) taken.insert(o);
  };
  for (const auto& m : plan->qmm) take(m->root, m->ops);
  for (const auto& m : plan->sdpa) take(m->root, m->ops);
  for (const auto& m : plan->moe) take(m->root, m->ops);
  for (const auto& m : plan->ragged) take(m->root, m->ops);
  for (const auto& m : plan->stacked) take(m->root, m->ops);
  for (const auto& m : plan->mla) take(m->root, m->ops);

  std::vector<std::unique_ptr<RmsNormMatch>> found;
  llvm::DenseSet<mlir::Operation*> visited_fns;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      const std::string name = OpName(&op);
      if (name == "stablehlo.transpose" && !taken.contains(&op)) {
        try {
          found.push_back(MatchRoot(&op));
        } catch (const Reject& e) {
          if (kDebug && e.why != std::string("not the weight-apply") &&
              e.why != std::string("the weight apply is not a dot") &&
              e.why !=
                  std::string("the weight apply is not the batching multiply"))
            Debug(absl::StrCat("rejected a candidate (", e.why, ")"));
        } catch (const std::exception& e) {
          if (kDebug) Debug(absl::StrCat("analysis error (", e.what(), ")"));
        }
      }
      if (name == "func.call" || name == "stablehlo.composite") {
        auto sym = op.getAttrOfType<mlir::FlatSymbolRefAttr>(
            name == "func.call" ? "callee" : "decomposition");
        if (sym) {
          auto callee = module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
          if (callee && callee.getBody().getBlocks().size() == 1 &&
              visited_fns.insert(callee.getOperation()).second)
            walk(callee.getBody().front());
        }
      }
      for (mlir::Region& r : op.getRegions())
        for (mlir::Block& bb : r.getBlocks()) walk(bb);
    }
  };
  walk(fn.getBody().front());
  if (found.empty()) return;

  llvm::DenseSet<mlir::Operation*> roots;
  std::vector<std::unique_ptr<RmsNormMatch>> kept;
  for (auto& m : found) {
    llvm::DenseSet<mlir::Operation*> seen;
    std::vector<mlir::Operation*> uniq;
    for (mlir::Operation* o : m->ops)
      if (seen.insert(o).second) uniq.push_back(o);
    m->ops = std::move(uniq);
    bool overlaps = roots.contains(m->root) || taken.contains(m->root);
    for (mlir::Operation* o : m->ops)
      overlaps = overlaps || taken.contains(o) || roots.contains(o);
    if (overlaps) continue;
    roots.insert(m->root);
    kept.push_back(std::move(m));
  }
  if (kept.empty()) return;

  llvm::DenseSet<mlir::Operation*> cand;
  for (const auto& m : kept)
    for (mlir::Operation* o : m->ops)
      if (!mlir::isa<mlir::stablehlo::ConstantOp>(o)) cand.insert(o);
  auto inside_users = [&](llvm::DenseSet<mlir::Operation*>& cs,
                          mlir::Operation* o) {
    for (mlir::Value r : o->getResults()) {
      for (mlir::OpOperand& use : r.getUses()) {
        mlir::Operation* u = use.getOwner();
        if (roots.contains(u) || cs.contains(u)) continue;
        if (OpName(u) == "func.call") {
          auto sym = u->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
          auto callee =
              sym ? module.lookupSymbol<mlir::func::FuncOp>(sym.getValue())
                  : nullptr;
          if (callee && callee.getBody().getBlocks().size() == 1 &&
              callee.getBody().front().getNumArguments() ==
                  u->getNumOperands()) {
            mlir::BlockArgument arg = callee.getBody().front().getArgument(
                use.getOperandNumber());
            bool arg_inside = true;
            for (mlir::Operation* au : arg.getUsers())
              arg_inside =
                  arg_inside && (roots.contains(au) || cs.contains(au));
            if (arg_inside) continue;
          }
        }
        return false;
      }
    }
    return true;
  };
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<mlir::Operation*> drop;
    for (mlir::Operation* o : cand)
      if (!inside_users(cand, o)) drop.push_back(o);
    for (mlir::Operation* o : drop) {
      cand.erase(o);
      changed = true;
    }
  }

  for (auto& m : kept) {
    bool complete = true;
    for (mlir::Operation* o : m->ops) {
      if (mlir::isa<mlir::stablehlo::ConstantOp>(o)) continue;
      complete = complete && cand.contains(o);
    }
    if (!complete) {
      Debug(absl::StrCat("declined ", m->name,
                         ": an intermediate escapes the norm"));
      continue;
    }
    std::vector<mlir::Operation*> absorbed;
    for (mlir::Operation* o : m->ops)
      if (!mlir::isa<mlir::stablehlo::ConstantOp>(o)) absorbed.push_back(o);
    m->ops = std::move(absorbed);
    Debug(absl::StrCat("matched an rms norm (", m->name, ", ", m->ops.size(),
                       " ops absorbed)"));
    plan->norm.push_back(std::move(m));
  }
}

}  // namespace metaljax
