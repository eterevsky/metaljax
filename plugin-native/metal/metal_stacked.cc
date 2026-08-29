/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The stacked-weight dot recognizer: jax's scanned-layer weight read,
rewritten into a matmul that reads the stack in place.

Every scanned-layer jax model (maxtext with scan_layers, flax nn.scan)
carries its weights as one stack per parameter with the layer axis inside —
maxtext's param_scan_axis=1 gives [k, L, n] — and reads layer `i` in the
loop body as `dynamic_index_in_dim(stack, i)`: a transpose (hoisted
loop-invariant, so it rides a pass-through carry), then a helper call whose
body is reshape(dynamic_slice(arg0, i, 0...)).  MLX's dynamic slice is a
COPY because the offset is data, so a decode step re-materialized every
layer's weights every token — ~31 MB x 28 layers on qwen3-0.6b, measured
~3.8 ms of a 16.9 ms token (row10-opt2 phase B/C, the static-slice A/B).

The rewrite emits `gather_mm(x, stack_view, [0], [layer])`: MLX's gather
kernels take the batch stride and the leading (ld) stride from the array
(`ensure_batch_contiguous` / `check_transpose` accept any row stride with a
unit column stride), so an [L, K, N] `as_strided` view of the ORIGINAL
contiguous stack is read in place — no copy on any of the four dispatch
paths (rhs, rhs_nax, mv, generic).  The geometry — which stack axes the dot
contracts, whether they collapse to one K and one N stride, and that the N
stride is 1 — is proven here on static shapes; anything else Bails and the
ordinary slice chain runs (the recognizer file rule, metal_recognize.h).

The dense chain's dynamic_slice clamps the layer index to [0, L-1]; the
emit clamps identically, so out-of-range indices agree bit for bit.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
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
  std::fprintf(stderr, "[metaljax-native] stacked: %s\n", line.c_str());
  std::fflush(stderr);
}

bool EnvOff(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "0";
}

// Not a stacked-weight read: run the chain as written.
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

bool IsZeroSplat(mlir::Operation* op) {
  auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(op);
  if (!cst) return false;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return false;
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(cst.getType());
  if (!t) return false;
  if (mlir::isa<mlir::FloatType>(t.getElementType()))
    return dense.getSplatValue<mlir::APFloat>().isZero();
  if (mlir::isa<mlir::IntegerType>(t.getElementType()))
    return dense.getSplatValue<mlir::APInt>().isZero();
  return false;
}

// The dtype code the tape uses, resolved through the same table the lowering
// reads (metal_dtypes.cc TapeDtypeCode is not exported here; dtype_codes()
// maps names).  The recognizer only needs "same float, not f64", so the code
// itself is filled by the lowering at emit time — this returns nothing.

std::unique_ptr<StackedDotMatch> MatchRoot(
    mlir::Operation* op, mlir::ModuleOp module, mlir::Block* main_block,
    const llvm::DenseMap<mlir::StringRef, std::vector<mlir::Operation*>>&
        call_sites) {
  auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
  if (!dot) Bail("not a dot_general");
  mlir::stablehlo::DotDimensionNumbersAttr dn = dot.getDotDimensionNumbers();
  if (!dn.getLhsBatchingDimensions().empty() ||
      !dn.getRhsBatchingDimensions().empty())
    Bail("the dot has batching dims");
  auto lc = dn.getLhsContractingDimensions();
  auto rc = dn.getRhsContractingDimensions();
  const size_t nc = lc.size();
  if (nc == 0 || nc != rc.size()) Bail("no contraction");

  std::vector<int64_t> lshape = ShapeOf(dot.getLhs());
  std::vector<int64_t> rshape = ShapeOf(dot.getRhs());
  if (rshape.size() < 2) Bail("the weight is not a matrix");
  if (nc >= rshape.size()) Bail("the dot leaves no free weight axis");
  // lhs contracts its TRAILING dims and rhs its LEADING dims, both in
  // ascending order with the pairs aligned — that is what makes flattening
  // x to [M, K] and the stack view to [L, K, N] element-order exact.
  for (size_t i = 0; i < nc; i++) {
    if (lc[i] != static_cast<int64_t>(lshape.size() - nc + i))
      Bail("the lhs does not contract its trailing dims in order");
    if (rc[i] != static_cast<int64_t>(i))
      Bail("the rhs does not contract its leading dims in order");
  }

  // One float dtype throughout (a preferred_element_type dot accumulates
  // wider than the gather computes).
  mlir::Type elem = ElemOf(dot.getLhs());
  if (!mlir::isa<mlir::FloatType>(elem) || mlir::isa<mlir::Float64Type>(elem))
    Bail("not a matmul-able float dtype");
  if (ElemOf(dot.getRhs()) != elem || ElemOf(dot.getResult()) != elem)
    Bail("mixed dtypes");

  auto match = std::make_unique<StackedDotMatch>();
  std::vector<mlir::Operation*>& absorb = match->ops;
  mlir::Value v = dot.getRhs();

  // Peel the sharding aliases (arity-preserving no-ops on one device).
  for (int depth = 0; depth < 4; depth++) {
    mlir::Operation* d = v.getDefiningOp();
    if (d == nullptr) break;
    const std::string n = OpName(d);
    if (n != "sdy.sharding_constraint" && n != "sdy.reshard") break;
    if (d->getNumOperands() != 1 || d->getNumResults() != 1)
      Bail("a sharding alias with unexpected arity");
    absorb.push_back(d);
    v = d->getOperand(0);
  }

  // Cross ONE call boundary upward: the dot usually sits in an outlined
  // layer body, and the weights arrive as its argument (metal_ragged.cc's
  // MatchStacked, verbatim).
  mlir::Block* frame = op->getBlock();
  if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    auto fn = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
        ba.getOwner()->getParentOp());
    if (!fn || &fn.getBody().front() != ba.getOwner())
      Bail("the weights are an argument of something that is not a callee");
    auto it = call_sites.find(fn.getName());
    if (it == call_sites.end() || it->second.size() != 1)
      Bail("the callee has no unique call site");
    mlir::Operation* site = it->second[0];
    if (site->getNumOperands() != ba.getOwner()->getNumArguments())
      Bail("the call site arity disagrees with the callee");
    v = site->getOperand(ba.getArgNumber());
    frame = site->getBlock();
  }

  // Two spellings of the same read.  Scan lowering outlines it as
  // func.call @dynamic_index_in_dim(stack, idx), whose body is exactly
  // reshape(dynamic_slice(arg0, arg1, zeros...)) on axis 0; a traced
  // `lax.dynamic_index_in_dim` inlines the same reshape(dynamic_slice(...))
  // with the index at an arbitrary axis.
  mlir::Value stack, layer;
  int64_t a = 0;                // the sliced axis, in `stack` coords
  std::vector<int64_t> wdims;   // the stack's (slice operand's) dims
  mlir::Operation* must_absorb = nullptr;  // the slice; if it survives the
                                           // fixpoint the match is dropped
  mlir::Operation* def = v.getDefiningOp();
  if (def != nullptr && OpName(def) == "func.call") {
    mlir::Operation* call = def;
    if (call->getNumOperands() != 2 || call->getNumResults() != 1 ||
        call->getResult(0) != v)
      Bail("the weights are not a two-argument helper call");
    if (call->getBlock() != frame)
      Bail("the helper call sits in a different block than the layer call");
    auto sym = call->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
    if (!sym) Bail("a call with no callee");
    auto helper = module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
    if (!helper || helper.getBody().getBlocks().size() != 1)
      Bail("the helper is not a single-block function");
    mlir::Block& hb = helper.getBody().front();
    if (hb.getNumArguments() != 2) Bail("the helper arity");

    mlir::stablehlo::DynamicSliceOp ds;
    mlir::stablehlo::ReshapeOp rs;
    for (mlir::Operation& ho : hb) {
      const std::string n = OpName(&ho);
      if (n == "stablehlo.constant") {
        if (!IsZeroSplat(&ho)) Bail("a non-zero helper constant");
      } else if (n == "stablehlo.dynamic_slice") {
        if (ds) Bail("two helper slices");
        ds = mlir::cast<mlir::stablehlo::DynamicSliceOp>(&ho);
      } else if (n == "stablehlo.reshape") {
        if (rs) Bail("two helper reshapes");
        rs = mlir::cast<mlir::stablehlo::ReshapeOp>(&ho);
      } else if (n == "func.return") {
        if (ho.getNumOperands() != 1 || !rs ||
            ho.getOperand(0) != rs.getResult())
          Bail("the helper does not return its reshape");
      } else {
        Bail(absl::StrCat("an unexpected helper op ", n));
      }
    }
    if (!ds || !rs) Bail("the helper has no slice");
    if (ds.getOperand() != hb.getArgument(0))
      Bail("the helper does not slice its first argument");
    wdims = ShapeOf(hb.getArgument(0));
    auto starts = ds.getStartIndices();
    if (starts.size() != wdims.size() || starts.empty() ||
        starts[0] != hb.getArgument(1))
      Bail("the helper does not index axis 0 by its second argument");
    for (size_t i = 1; i < starts.size(); i++) {
      mlir::Operation* z = starts[i].getDefiningOp();
      if (z == nullptr || !IsZeroSplat(z))
        Bail("a helper start that is not zero");
    }
    auto sizes = ds.getSliceSizes();
    if (sizes.size() != wdims.size() || sizes[0] != 1)
      Bail("the slice is not one whole layer");
    for (size_t i = 1; i < sizes.size(); i++)
      if (sizes[i] != wdims[i]) Bail("the slice is not one whole layer");
    std::vector<int64_t> unit(wdims);
    unit[0] = 1;
    if (ShapeOf(rs.getOperand()) != unit || ShapeOf(rs.getResult()) != rshape)
      Bail("the helper reshape is not the unit-axis drop");
    stack = call->getOperand(0);
    layer = call->getOperand(1);
    a = 0;
    must_absorb = call;
    absorb.push_back(call);
  } else if (auto rs = mlir::dyn_cast_or_null<mlir::stablehlo::ReshapeOp>(
                 def)) {
    auto ds = mlir::dyn_cast_or_null<mlir::stablehlo::DynamicSliceOp>(
        rs.getOperand().getDefiningOp());
    if (!ds) Bail("the weights are not a sliced stack");
    wdims = ShapeOf(ds.getOperand());
    auto starts = ds.getStartIndices();
    auto sizes = ds.getSliceSizes();
    if (starts.size() != wdims.size() || sizes.size() != wdims.size())
      Bail("the slice rank disagrees with the stack");
    int64_t ai = -1;
    for (size_t i = 0; i < starts.size(); i++) {
      mlir::Operation* z = starts[i].getDefiningOp();
      if (z != nullptr && IsZeroSplat(z)) continue;
      if (ai >= 0) Bail("the slice indexes two axes");
      ai = static_cast<int64_t>(i);
    }
    if (ai < 0) Bail("the slice indexes no axis");
    if (sizes[ai] != 1) Bail("the slice is not one whole layer");
    for (size_t i = 0; i < sizes.size(); i++)
      if (static_cast<int64_t>(i) != ai && sizes[i] != wdims[i])
        Bail("the slice is not one whole layer");
    std::vector<int64_t> unit(wdims);
    unit[ai] = 1;
    if (ShapeOf(rs.getOperand()) != unit || ShapeOf(rs.getResult()) != rshape)
      Bail("the reshape is not the unit-axis drop");
    stack = ds.getOperand();
    layer = starts[ai];
    a = ai;
    must_absorb = ds;
    absorb.push_back(rs);
    absorb.push_back(ds);
  } else {
    Bail("the weights are not a sliced stack");
  }
  if (!mlir::isa<mlir::IntegerType>(ElemOf(layer)) || !ShapeOf(layer).empty())
    Bail("the layer index is not a scalar integer");

  // The layout proof.  The stack the helper reads is either the transpose
  // of a contiguous @main argument (hoisted loop-invariant, riding a
  // pass-through carry) or such an argument directly.  Either way the emit
  // can reconstruct a row-contiguous view (transposing the carried view
  // back), and every stride below is that of the ORIGINAL buffer.
  mlir::Value hoisted = HoistInvariant(stack);
  std::vector<int64_t> perm;   // w_carry axis i = original axis perm[i]
  std::vector<int64_t> sdims;  // the ORIGINAL (contiguous) dims
  mlir::Value origin;
  if (auto tr = mlir::dyn_cast_or_null<mlir::stablehlo::TransposeOp>(
          hoisted.getDefiningOp())) {
    auto p = tr.getPermutation();
    perm.assign(p.begin(), p.end());
    origin = tr.getOperand();
  } else {
    perm.resize(wdims.size());
    std::iota(perm.begin(), perm.end(), 0);
    origin = hoisted;
  }
  {
    auto ba = mlir::dyn_cast<mlir::BlockArgument>(origin);
    if (!ba || ba.getOwner() != main_block)
      Bail("the stack does not hoist to a @main argument");
  }
  sdims = ShapeOf(origin);
  if (perm.size() != sdims.size() || perm.size() != wdims.size())
    Bail("the stack rank disagrees with the slice");

  // Row-major strides of the original buffer, in elements.
  std::vector<int64_t> st(sdims.size());
  {
    int64_t acc = 1;
    for (int i = static_cast<int>(sdims.size()) - 1; i >= 0; i--) {
      st[i] = acc;
      acc *= std::max<int64_t>(sdims[i], 1);
    }
  }
  // The rhs's dims, as ORIGINAL stack axes: rhs dim j is stack axis
  // (j < a ? j : j + 1) — the slice axis dropped — mapped through the
  // transpose.  The contracted axes (K) are the first nc of them, the free
  // axes (N) the rest; each group must collapse to a single stride in that
  // order for the [L, K, N] view to exist.
  std::vector<int64_t> in_order;
  for (size_t j = 0; j + 1 < wdims.size(); j++) {
    const int64_t wax = static_cast<int64_t>(j) < a
                            ? static_cast<int64_t>(j)
                            : static_cast<int64_t>(j) + 1;
    in_order.push_back(perm[wax]);
  }
  auto collapse = [&](size_t lo, size_t hi,  // in_order indices [lo, hi)
                      const char* what) -> std::pair<int64_t, int64_t> {
    int64_t extent = 1;
    for (size_t i = lo; i < hi; i++) {
      const int64_t ax = in_order[i];
      extent *= sdims[ax];
      if (i + 1 < hi) {
        const int64_t nx = in_order[i + 1];
        if (st[ax] != sdims[nx] * st[nx])
          Bail(absl::StrCat(what, " axes do not collapse in the stack"));
      }
    }
    return {extent, st[in_order[hi - 1]]};
  };
  auto [K, sk] = collapse(0, nc, "the contracted");
  auto [N, sn] = collapse(nc, in_order.size(), "the free");
  if (sn != 1) Bail("the free stride is not 1");
  const int64_t L = wdims[a];
  const int64_t sl = st[perm[a]];
  if (L < 1 || K < 1 || N < 1) Bail("degenerate sizes");
  if (K * N < 16384) Bail("the weight is too small to matter");

  // The inverse permutation: transpose(w_carry, back_perm) is the original.
  match->back_perm.resize(perm.size());
  for (size_t i = 0; i < perm.size(); i++) match->back_perm[perm[i]] = i;

  int64_t M = 1;
  for (size_t i = 0; i + nc < lshape.size(); i++) M *= lshape[i];
  if (M < 1) Bail("degenerate rows");

  // Frame guard: the fused root lowers inside `frame`'s splice, so both
  // values must be resolvable there.
  for (mlir::Value fv : {stack, layer}) {
    if (auto fba = mlir::dyn_cast<mlir::BlockArgument>(fv)) {
      if (fba.getOwner() != frame) Bail("a stack input from another frame");
    } else if (fv.getDefiningOp()->getBlock() != frame) {
      Bail("a stack input defined in another frame");
    }
  }

  match->root = op;
  match->x = dot.getLhs();
  match->w_carry = stack;
  match->layer = layer;
  match->helper_call = must_absorb;
  match->L = L;
  match->K = K;
  match->N = N;
  match->sl = sl;
  match->sk = sk;
  match->sn = sn;
  match->M = M;
  match->out_shape = ShapeOf(dot.getResult());
  match->name = absl::StrCat("L", L, "m", M, "k", K, "n", N);
  return match;
}

}  // namespace

void AnalyzeStackedDot(mlir::func::FuncOp fn, RewritePlan* plan) {
  if (EnvOff("METALJAX_STACKED_DOT") || !RecognizeEnabled()) return;
  if (fn.getBody().getBlocks().size() != 1) return;
  auto module = fn->getParentOfType<mlir::ModuleOp>();
  if (!module) return;
  mlir::Block* main_block = &fn.getBody().front();

  // Ops another recognizer already owns.
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

  llvm::DenseMap<mlir::StringRef, std::vector<mlir::Operation*>> call_sites;
  module.walk([&](mlir::Operation* op) {
    if (OpName(op) != "func.call") return;
    if (auto sym = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      call_sites[sym.getValue()].push_back(op);
  });

  // Every block reachable from @main, callees included (metal_ragged.cc).
  std::vector<std::unique_ptr<StackedDotMatch>> found;
  llvm::DenseSet<mlir::Operation*> visited_fns;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      const std::string name = OpName(&op);
      if (name == "stablehlo.dot_general" && !taken.contains(&op)) {
        try {
          found.push_back(MatchRoot(&op, module, main_block, call_sites));
        } catch (const Reject& e) {
          // Almost every dot_general is not a stacked read; narrate only
          // near-misses (ones that found a helper call).
          if (kDebug && e.why.find("helper") == std::string::npos &&
              e.why != std::string("the dot has batching dims") &&
              e.why != std::string(
                           "the weights are not a two-argument helper call") &&
              e.why != std::string("the lhs does not contract its trailing "
                                   "dims in order") &&
              e.why != std::string("the rhs does not contract its leading "
                                   "dims in order"))
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

  // Drop overlapping candidates, then the joint use-count fixpoint over all
  // kept matches (metal_ragged.cc: sibling matches may share ops, and a
  // func.call user is looked through to the callee argument's users).
  llvm::DenseSet<mlir::Operation*> roots;
  std::vector<std::unique_ptr<StackedDotMatch>> kept;
  for (auto& m : found) {
    bool overlaps = roots.contains(m->root) || taken.contains(m->root);
    for (mlir::Operation* o : m->ops)
      overlaps = overlaps || taken.contains(o);
    if (overlaps) continue;
    roots.insert(m->root);
    kept.push_back(std::move(m));
  }
  if (kept.empty()) return;

  // The joint use-count fixpoint (metal_ragged.cc), iterated to a stable
  // KEPT set: a match whose helper call could not be absorbed is dropped
  // entirely — its plain lowering then reads every op of its chain — and
  // dropping one match shrinks the root set, which can strand another
  // match's candidates, so the whole thing reruns until nothing drops.
  for (;;) {
    llvm::DenseSet<mlir::Operation*> live_roots;
    for (const auto& m : kept) live_roots.insert(m->root);
    llvm::DenseSet<mlir::Operation*> cand;
    for (const auto& m : kept)
      for (mlir::Operation* o : m->ops)
        if (!mlir::isa<mlir::stablehlo::ConstantOp>(o)) cand.insert(o);
    auto inside_users = [&](llvm::DenseSet<mlir::Operation*>& cs,
                            mlir::Operation* o) {
      for (mlir::Value r : o->getResults()) {
        for (mlir::OpOperand& use : r.getUses()) {
          mlir::Operation* u = use.getOwner();
          if (live_roots.contains(u) || cs.contains(u)) continue;
          if (OpName(u) == "func.call") {
            auto sym = u->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
            auto callee =
                sym ? module.lookupSymbol<mlir::func::FuncOp>(sym.getValue())
                    : nullptr;
            if (callee && callee.getBody().getBlocks().size() == 1 &&
                callee.getBody().front().getNumArguments() ==
                    u->getNumOperands()) {
              mlir::BlockArgument arg =
                  callee.getBody().front().getArgument(use.getOperandNumber());
              bool arg_inside = true;
              for (mlir::Operation* au : arg.getUsers())
                arg_inside =
                    arg_inside && (live_roots.contains(au) || cs.contains(au));
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

    std::vector<std::unique_ptr<StackedDotMatch>> still;
    bool dropped = false;
    for (auto& m : kept) {
      if (m->helper_call == nullptr || !cand.contains(m->helper_call)) {
        // The slice runs anyway (someone else reads it): a gather would
        // read the same weights a second time.  Keep the chain as written.
        Debug(absl::StrCat("weights stay sliced (", m->name,
                           ": the helper call has other readers)"));
        dropped = true;
        continue;
      }
      still.push_back(std::move(m));
    }
    kept = std::move(still);
    if (!dropped) {
      for (auto& m : kept) {
        std::vector<mlir::Operation*> absorbed;
        for (mlir::Operation* o : m->ops)
          if (cand.contains(o)) absorbed.push_back(o);
        m->ops = std::move(absorbed);
        Debug(absl::StrCat("matched a stacked dot (", m->name, ", ",
                           m->ops.size(), " ops absorbed)"));
        plan->stacked.push_back(std::move(m));
      }
      return;
    }
    if (kept.empty()) return;
  }
}

}  // namespace metaljax
