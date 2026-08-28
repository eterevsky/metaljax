/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The ragged-dot recognizer: jax's `lax.ragged_dot` DENSE FALLBACK, rewritten
into the gathered dispatch it stands for.

Every backend without a native ragged_dot lowering gets jax's
`_ragged_dot_general_impl` ("ragged_to_dense", jax/_src/lax/lax.py): the
[m, k] rows are broadcast to [g, m, k], masked down to the half-open row
intervals a cumsum of `group_sizes` defines, and contracted against the
whole [g, k, n] weight stack over BOTH g and k.  Semantically each row is
multiplied by ITS group's [k, n] matrix; the dense form merely reaches that
by computing every row against every group and zeroing the g - 1
non-members.  maxtext's sparse MoE path (`sparse_matmul=true`, the only
non-Pallas option) runs three of these per expert layer, padded to the
ragged tiling — for DeepSeek-V2-Lite decode that is a [64, 512, 2048] x
[64, 2048, 1408] GEMM per dot for 6 real rows and 6 live experts, a ~910x
FLOP inflation that made row 10 measure 1948 ms/token (20.8 ms per dot of
which 18 ms is the GEMM itself; see
~/.cache/metaljax-bench/logs/row10-opt/).

The rewrite emits the row-vs-own-group form literally: one `gather_mm`
(kRaggedDot, runtime/emits.cc) over the real rows, group index per row
recovered from the same cumsum the mask was built from.  Rows the dense
mask zeroes everywhere — the tiling pad, and anything at or past
cumsum[-1] — come back as exact zero rows, which is what the dense form
computes for them.

No runtime verification is needed; the equivalence is structural.  Two
documented deviations, both defensible as implementing `ragged_dot`'s own
contract rather than the fallback's accidents:

  * A non-finite value in a NEVER-SELECTED group's weights: the dense form
    multiplies it by zero (0 * NaN = NaN) and pollutes the sum; the gather
    never reads it.  `ragged_dot` semantics never reads it either.
  * `group_sizes` whose cumsum is not a partition of the rows (negative
    sizes — nothing a bincount produces): the dense form SUMS every group
    whose interval covers a row; the gather takes the first.  jax documents
    group_sizes as group sizes, and the TPU lowering (chlo.ragged_dot)
    assumes the same partition.

A half-matched pattern lowers as ORDINARY ops: every rejection below is a
`Bail`, and the consequence is the correct slow program — never a wrong
fused one (the recognizer file rule, metal_recognize.h).

Unlike AnalyzeMoe's, this walk FOLLOWS func.call/composite symbols: maxtext
wraps each scan-layer body in a private callee, and a region-only walk never
sees inside one.  The lowering splices callees, so a root or an absorbed op
inside one is dispatched exactly like a top-level op.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
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
  std::fprintf(stderr, "[metaljax-native] ragged: %s\n", line.c_str());
  std::fflush(stderr);
}

bool EnvOff(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "0";
}

// This is not a ragged-dot dispatch: run the dense chain as written.
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

// The op defining `v`, which must exist (a block argument is not a pattern
// op — the pattern never needs one).
mlir::Operation* DefOf(mlir::Value v, const char* what) {
  mlir::Operation* op = v.getDefiningOp();
  if (op == nullptr) Bail(absl::StrCat(what, " is a block argument"));
  return op;
}

// A splat constant equal to zero (int or float).
bool IsZeroSplat(mlir::Operation* op) {
  auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(op);
  if (!cst) return false;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return false;
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(cst.getType());
  if (!t) return false;
  if (mlir::isa<mlir::FloatType>(t.getElementType())) {
    auto v = dense.getSplatValue<mlir::APFloat>();
    return v.isZero();
  }
  if (mlir::isa<mlir::IntegerType>(t.getElementType())) {
    auto v = dense.getSplatValue<mlir::APInt>();
    return v.isZero();
  }
  return false;
}

// A value that is provably an all-zero tensor: a zero splat, or a
// broadcast / slice / reshape chain over one.  `ops` collects the chain
// (constants excluded — they cost nothing and are usually shared).
bool IsZeroTensor(mlir::Value v, std::vector<mlir::Operation*>* ops) {
  mlir::Operation* op = v.getDefiningOp();
  for (int depth = 0; op != nullptr && depth < 4; depth++) {
    if (IsZeroSplat(op)) return true;
    const std::string name = OpName(op);
    if (name != "stablehlo.broadcast_in_dim" && name != "stablehlo.slice" &&
        name != "stablehlo.reshape")
      return false;
    if (ops != nullptr) ops->push_back(op);
    op = op->getOperand(0).getDefiningOp();
  }
  return false;
}

// broadcast_in_dim of a rank-1 value along dim 0 of a rank-3 shape.
mlir::Value LeadBroadcastOf(mlir::Value v, std::vector<mlir::Operation*>* ops,
                            const char* what) {
  auto b = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      DefOf(v, what));
  if (!b) Bail(absl::StrCat(what, " is not a broadcast"));
  auto dims = b.getBroadcastDimensions();
  if (dims.size() != 1 || dims[0] != 0)
    Bail(absl::StrCat(what, " does not broadcast a leading vector"));
  if (ShapeOf(b.getOperand()).size() != 1)
    Bail(absl::StrCat(what, " does not broadcast a rank-1 value"));
  ops->push_back(b);
  return b.getOperand();
}

struct Compares {
  mlir::Value iota;    // the [g, M, k] row iota, shared by both compares
  mlir::Value starts;  // [g]
  mlir::Value ends;    // [g]
};

// One side of the AND: `starts <= iota` (LE/GE either way around) or
// `iota < ends` (LT/GT).  Returns (bound, is_lower) with `iota` checked.
struct OneCompare {
  mlir::Value bound;
  mlir::Value iota;
  bool is_lower = false;  // true: starts <= iota; false: iota < ends
};

OneCompare MatchCompare(mlir::Value v, std::vector<mlir::Operation*>* ops) {
  auto cmp =
      mlir::dyn_cast_or_null<mlir::stablehlo::CompareOp>(DefOf(v, "mask"));
  if (!cmp) Bail("mask operand is not a compare");
  auto dir = cmp.getComparisonDirection();
  mlir::Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
  using D = mlir::stablehlo::ComparisonDirection;
  // Normalize to `a <= b` / `a < b`.
  bool strict;
  if (dir == D::LE) {
    strict = false;
  } else if (dir == D::LT) {
    strict = true;
  } else if (dir == D::GE) {
    std::swap(lhs, rhs);
    strict = false;
  } else if (dir == D::GT) {
    std::swap(lhs, rhs);
    strict = true;
  } else {
    Bail("mask compare direction");
  }
  ops->push_back(cmp);
  OneCompare out;
  auto is_iota = [](mlir::Value x) {
    auto i = mlir::dyn_cast_or_null<mlir::stablehlo::IotaOp>(x.getDefiningOp());
    return i && i.getIotaDimension() == 1;
  };
  if (!strict && is_iota(rhs)) {
    // starts <= iota
    out.is_lower = true;
    out.iota = rhs;
    out.bound = LeadBroadcastOf(lhs, ops, "the interval start");
  } else if (strict && is_iota(lhs)) {
    // iota < ends
    out.is_lower = false;
    out.iota = lhs;
    out.bound = LeadBroadcastOf(rhs, ops, "the interval end");
  } else {
    Bail("mask compare does not bound the row iota");
  }
  return out;
}

// starts = concatenate([0], ends[:-1]): the shifted cumsum.  Verifies the
// zero head and that the tail is a [0 : g-1] slice of the SAME `ends`.
void MatchStarts(mlir::Value starts, mlir::Value ends, int64_t g,
                 std::vector<mlir::Operation*>* ops) {
  auto cat = mlir::dyn_cast_or_null<mlir::stablehlo::ConcatenateOp>(
      DefOf(starts, "the interval starts"));
  if (!cat || cat.getDimension() != 0 || cat.getNumOperands() != 2)
    Bail("the interval starts are not a shifted cumsum");
  mlir::Value head = cat.getOperand(0), tail = cat.getOperand(1);
  if (ShapeOf(head) != std::vector<int64_t>{1} ||
      ShapeOf(tail) != std::vector<int64_t>{g - 1})
    Bail("the shifted cumsum has the wrong split");
  std::vector<mlir::Operation*> zeros;
  if (!IsZeroTensor(head, &zeros))
    Bail("the shifted cumsum does not start at zero");
  auto sl = mlir::dyn_cast_or_null<mlir::stablehlo::SliceOp>(
      tail.getDefiningOp());
  if (!sl || sl.getOperand() != ends)
    Bail("the shifted cumsum tail is not a slice of the ends");
  auto starts_idx = sl.getStartIndices();
  auto strides = sl.getStrides();
  if (starts_idx.size() != 1 || starts_idx[0] != 0 || strides[0] != 1)
    Bail("the shifted cumsum tail is not ends[:-1]");
  ops->push_back(cat);
  ops->push_back(sl);
  ops->insert(ops->end(), zeros.begin(), zeros.end());
}

// The stacked-weights extension: prove `m->w` is a dynamic-index-in-dim out
// of a pass-through carry over a transposed [g, L, k, n] stack, and absorb
// the whole slice chain — MLX's dynamic slice is a COPY (the offset is
// data), and three of them per layer re-materialized ~1.1 GB of expert
// weights per decode token.  On any structural surprise the base 3-input
// form stands (Bail is caught by the caller).
//
// The chain, as jax 0.11 spells a scanned layer stack:
//
//   root.rhs <- sdy.sharding_constraint* <- [callee block arg <- the callee's
//   UNIQUE call site's operand] <- func.call @dynamic_index_in_dim(stack,
//   idx) whose body is reshape(dynamic_slice(arg0, arg1, 0...)) <- `stack` a
//   while carry the body returns unchanged, whose init is
//   stablehlo.transpose [1, 0, 2, 3] of a [g, L, k, n] value.
//
// The transpose proof is what licenses the emit: transposing the carried
// view BACK and flattening [g, L] is then a zero-copy view of the original
// buffer, so `gather_mm` reads matrix `e * L + l` straight out of it.
void MatchStacked(
    RaggedMatch* m, mlir::ModuleOp module,
    const llvm::DenseMap<mlir::StringRef, std::vector<mlir::Operation*>>&
        call_sites) {
  std::vector<mlir::Operation*> absorb;
  mlir::Value v = m->w;

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
  // layer body, and the weights arrive as its argument.
  mlir::Block* frame = m->root->getBlock();
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

  // The helper: func.call @dynamic_index_in_dim(stack, idx), whose body is
  // exactly reshape(dynamic_slice(arg0, arg1, zeros...)).
  mlir::Operation* call = v.getDefiningOp();
  if (call == nullptr || OpName(call) != "func.call" ||
      call->getNumOperands() != 2 || call->getNumResults() != 1 ||
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
      if (ho.getNumOperands() != 1 || !rs || ho.getOperand(0) != rs.getResult())
        Bail("the helper does not return its reshape");
    } else {
      Bail(absl::StrCat("an unexpected helper op ", n));
    }
  }
  if (!ds || !rs) Bail("the helper has no slice");
  if (ds.getOperand() != hb.getArgument(0))
    Bail("the helper does not slice its first argument");
  auto starts = ds.getStartIndices();
  if (starts.size() != 4 || starts[0] != hb.getArgument(1))
    Bail("the helper does not index axis 0 by its second argument");
  for (size_t i = 1; i < starts.size(); i++)
    if (!IsZeroTensor(starts[i], nullptr))
      Bail("a helper start that is not zero");
  std::vector<int64_t> stack_shape = ShapeOf(hb.getArgument(0));
  if (stack_shape.size() != 4 || stack_shape[1] != m->g ||
      stack_shape[2] != m->k || stack_shape[3] != m->n)
    Bail("the stack shape disagrees with the dot");
  const int64_t L = stack_shape[0];
  auto sizes = ds.getSliceSizes();
  if (sizes.size() != 4 || sizes[0] != 1 || sizes[1] != m->g ||
      sizes[2] != m->k || sizes[3] != m->n)
    Bail("the slice is not one whole layer");
  if (ShapeOf(rs.getOperand()) !=
          std::vector<int64_t>{1, m->g, m->k, m->n} ||
      ShapeOf(rs.getResult()) != std::vector<int64_t>{m->g, m->k, m->n})
    Bail("the helper reshape is not the unit-axis drop");

  mlir::Value stack = call->getOperand(0);
  mlir::Value layer = call->getOperand(1);
  if (!mlir::isa<mlir::IntegerType>(ElemOf(layer)) ||
      !ShapeOf(layer).empty())
    Bail("the layer index is not a scalar integer");

  // The layout proof: the carry's init is the [1, 0, 2, 3] transpose of a
  // [g, L, k, n] value, so the emit's transpose-back + flatten is a view.
  auto tr = mlir::dyn_cast_or_null<mlir::stablehlo::TransposeOp>(
      HoistInvariant(stack).getDefiningOp());
  if (!tr) Bail("the stack does not hoist to a transpose");
  auto perm = tr.getPermutation();
  if (perm.size() != 4 || perm[0] != 1 || perm[1] != 0 || perm[2] != 2 ||
      perm[3] != 3)
    Bail("the stack transpose is not the layer swap");

  // Frame guard: the fused root lowers inside `frame`'s splice, so both
  // values must be resolvable there — the block's own arguments or defs.
  for (mlir::Value fv : {stack, layer}) {
    if (auto fba = mlir::dyn_cast<mlir::BlockArgument>(fv)) {
      if (fba.getOwner() != frame) Bail("a stack input from another frame");
    } else if (fv.getDefiningOp()->getBlock() != frame) {
      Bail("a stack input defined in another frame");
    }
  }

  absorb.push_back(call);
  m->helper_call = call;
  m->stacked = true;
  m->w_stack = stack;
  m->layer = layer;
  m->L = L;
  m->ops.insert(m->ops.end(), absorb.begin(), absorb.end());
  m->name = absl::StrCat(m->name, "xL", L);
}

std::unique_ptr<RaggedMatch> MatchRoot(mlir::Operation* op) {
  auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
  if (!dot) Bail("not a dot_general");
  mlir::stablehlo::DotDimensionNumbersAttr dn = dot.getDotDimensionNumbers();
  if (!dn.getLhsBatchingDimensions().empty() ||
      !dn.getRhsBatchingDimensions().empty())
    Bail("the dot has batching dims");
  auto lc = dn.getLhsContractingDimensions();
  auto rc = dn.getRhsContractingDimensions();
  // The exact contraction _ragged_dot_general_impl builds for the basic
  // `lax.ragged_dot`: lhs [g, M, k] over (k, g), rhs [g, k, n] over (k, g).
  if (lc.size() != 2 || rc.size() != 2 || lc[0] != 2 || lc[1] != 0 ||
      rc[0] != 1 || rc[1] != 0)
    Bail("not the ragged contraction");

  std::vector<int64_t> lshape = ShapeOf(dot.getLhs());
  std::vector<int64_t> rshape = ShapeOf(dot.getRhs());
  if (lshape.size() != 3 || rshape.size() != 3) Bail("operand ranks");
  const int64_t g = lshape[0], M = lshape[1], k = lshape[2];
  if (rshape[0] != g || rshape[1] != k) Bail("operand shapes disagree");
  const int64_t n = rshape[2];
  if (g < 1 || M < 1 || k < 1 || n < 1) Bail("degenerate sizes");

  // One dtype throughout: the fused gather computes in the input dtype, so
  // an f32-accumulating dot over bf16 inputs (preferred_element_type) must
  // keep the dense chain and its output precision.
  mlir::Type elem = ElemOf(dot.getLhs());
  if (!mlir::isa<mlir::FloatType>(elem) ||
      mlir::isa<mlir::Float64Type>(elem))
    Bail("not a matmul-able float dtype");
  if (ElemOf(dot.getRhs()) != elem || ElemOf(dot.getResult()) != elem)
    Bail("mixed dtypes");

  auto match = std::make_unique<RaggedMatch>();
  std::vector<mlir::Operation*>& ops = match->ops;

  // lhs = select(mask, broadcast(x_padded), zeros)
  auto sel = mlir::dyn_cast_or_null<mlir::stablehlo::SelectOp>(
      DefOf(dot.getLhs(), "the dot lhs"));
  if (!sel) Bail("the dot lhs is not a select");
  ops.push_back(sel);
  std::vector<mlir::Operation*> zeros;
  if (!IsZeroTensor(sel.getOnFalse(), &zeros))
    Bail("the mask does not zero the non-members");
  ops.insert(ops.end(), zeros.begin(), zeros.end());

  auto xb = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      DefOf(sel.getOnTrue(), "the masked rows"));
  if (!xb) Bail("the masked rows are not a broadcast");
  auto xdims = xb.getBroadcastDimensions();
  if (xdims.size() != 2 || xdims[0] != 1 || xdims[1] != 2)
    Bail("the rows are not broadcast over the groups");
  ops.push_back(xb);
  mlir::Value xp = xb.getOperand();  // [M, k]

  // mask = and(starts <= iota, iota < ends), either order.
  auto mask = mlir::dyn_cast_or_null<mlir::stablehlo::AndOp>(
      DefOf(sel.getPred(), "the mask"));
  if (!mask) Bail("the mask is not an and");
  ops.push_back(mask);
  OneCompare a = MatchCompare(mask.getLhs(), &ops);
  OneCompare b = MatchCompare(mask.getRhs(), &ops);
  if (a.is_lower == b.is_lower) Bail("the mask is not an interval");
  const OneCompare& lower = a.is_lower ? a : b;
  const OneCompare& upper = a.is_lower ? b : a;
  if (lower.iota != upper.iota) Bail("the two bounds index different iotas");
  if (ShapeOf(lower.iota) != lshape) Bail("the iota has the wrong shape");
  if (!mlir::isa<mlir::IntegerType>(ElemOf(lower.iota)))
    Bail("the iota is not integral");
  ops.push_back(lower.iota.getDefiningOp());

  mlir::Value ends = upper.bound;
  if (ShapeOf(ends) != std::vector<int64_t>{g})
    Bail("the interval ends have the wrong shape");
  MatchStarts(lower.bound, ends, g, &ops);

  // Optional tiling pad: x_padded = pad(x, 0, high=[M - m, 0]).
  mlir::Value x = xp;
  int64_t m = M;
  if (auto pad = mlir::dyn_cast_or_null<mlir::stablehlo::PadOp>(
          xp.getDefiningOp())) {
    auto low = pad.getEdgePaddingLow();
    auto high = pad.getEdgePaddingHigh();
    auto interior = pad.getInteriorPadding();
    std::vector<mlir::Operation*> pad_zero;
    if (low.size() == 2 && low[0] == 0 && low[1] == 0 && high[1] == 0 &&
        high[0] >= 0 && interior[0] == 0 && interior[1] == 0 &&
        IsZeroTensor(pad.getPaddingValue(), &pad_zero)) {
      x = pad.getOperand();
      m = M - high[0];
      ops.push_back(pad);
      ops.insert(ops.end(), pad_zero.begin(), pad_zero.end());
    }
  }

  match->root = op;
  match->x = x;
  match->w = dot.getRhs();
  match->ends = ends;
  match->g = g;
  match->m = m;
  match->M = M;
  match->k = k;
  match->n = n;
  match->name = absl::StrCat("g", g, "m", m, "M", M, "k", k, "n", n);
  return match;
}

}  // namespace

void AnalyzeRagged(mlir::func::FuncOp fn, RewritePlan* plan) {
  if (EnvOff("METALJAX_RAGGED") || EnvOff("METALJAX_MOE")) return;
  if (fn.getBody().getBlocks().size() != 1) return;
  auto module = fn->getParentOfType<mlir::ModuleOp>();
  if (!module) return;

  // Ops another recognizer already owns: a candidate overlapping one is
  // dropped, so no op ever has two owners.  Read from the match lists, not
  // `plan->skip` — rebuild() has not run yet when the analyses do.
  llvm::DenseSet<mlir::Operation*> taken = plan->skip;
  for (const auto& m : plan->qmm) {
    taken.insert(m->root);
    for (mlir::Operation* o : m->ops) taken.insert(o);
  }
  for (const auto& m : plan->sdpa) {
    taken.insert(m->root);
    for (mlir::Operation* o : m->ops) taken.insert(o);
  }
  for (const auto& m : plan->moe) {
    taken.insert(m->root);
    for (mlir::Operation* o : m->ops) taken.insert(o);
  }

  // Every call site in the module, by callee name: the stacked-weights
  // extension needs to know a callee's UNIQUE caller to map its block
  // arguments to values.
  llvm::DenseMap<mlir::StringRef, std::vector<mlir::Operation*>> call_sites;
  module.walk([&](mlir::Operation* op) {
    if (OpName(op) != "func.call") return;
    if (auto sym = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      call_sites[sym.getValue()].push_back(op);
  });

  // Every block reachable from @main — including CALLEES: maxtext wraps a
  // scan layer's body in a private function, and a region-only walk never
  // sees inside one.  The lowering splices callees, so a root there is
  // dispatched like any other op.
  std::vector<std::unique_ptr<RaggedMatch>> found;
  llvm::DenseSet<mlir::Operation*> visited_fns;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      const std::string name = OpName(&op);
      if (name == "stablehlo.dot_general" && !taken.contains(&op)) {
        try {
          std::unique_ptr<RaggedMatch> m = MatchRoot(&op);
          try {
            MatchStacked(m.get(), module, call_sites);
          } catch (const Reject& e) {
            // The base form stands; the copy chain just keeps running.
            if (kDebug)
              Debug(absl::StrCat("weights stay sliced (", e.why, ")"));
          }
          found.push_back(std::move(m));
        } catch (const Reject& e) {
          // Almost every dot_general is not a ragged dispatch; only narrate
          // the ones that got past the contraction fingerprint.
          if (kDebug && e.why != std::string("not the ragged contraction") &&
              e.why != std::string("the dot has batching dims"))
            Debug(absl::StrCat("rejected a candidate (", e.why, ")"));
        } catch (const std::exception& e) {
          if (kDebug) Debug(absl::StrCat("analysis error (", e.what(), ")"));
        }
      }
      if (name == "func.call" || name == "stablehlo.composite") {
        auto sym = op.getAttrOfType<mlir::FlatSymbolRefAttr>(
            name == "func.call" ? "callee" : "decomposition");
        if (sym) {
          auto callee =
              module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
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

  // Drop overlapping candidates (first match wins), then run the use-count
  // discipline over ALL kept matches at once: sibling dots share their mask
  // subtrees (two dots read one masked-x tree), so an op is absorbable only
  // if every user is a kept root or itself absorbed — seeded jointly, or
  // the shared trees would keep each other alive.
  llvm::DenseSet<mlir::Operation*> roots;
  std::vector<std::unique_ptr<RaggedMatch>> kept;
  for (auto& m : found) {
    bool overlaps = roots.contains(m->root) || taken.contains(m->root);
    for (mlir::Operation* o : m->ops)
      overlaps = overlaps || taken.contains(o);
    if (overlaps) continue;
    roots.insert(m->root);
    kept.push_back(std::move(m));
  }
  if (kept.empty()) return;

  llvm::DenseSet<mlir::Operation*> cand;
  for (const auto& m : kept)
    for (mlir::Operation* o : m->ops)
      if (!mlir::isa<mlir::stablehlo::ConstantOp>(o)) cand.insert(o);
  // Greatest fixpoint: repeatedly drop any candidate with a user outside
  // the candidate set and the roots.  A func.call user is looked THROUGH:
  // the value only reaches the callee's block argument, so the use is
  // inside iff that argument's own users all are (the absorbed helper call
  // feeds an outlined layer body whose weight argument is read only by the
  // absorbed sharding alias; `Inline` skips the binding of an absorbed
  // operand on the same proof).
  auto inside_users = [&](llvm::DenseSet<mlir::Operation*>& cs,
                          mlir::Operation* o) {
    for (mlir::Value r : o->getResults()) {
      for (mlir::OpOperand& use : r.getUses()) {
        mlir::Operation* u = use.getOwner();
        if (roots.contains(u) || cs.contains(u)) continue;
        if (OpName(u) == "func.call") {
          auto sym = u->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
          auto callee = sym ? module.lookupSymbol<mlir::func::FuncOp>(
                                  sym.getValue())
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
    std::vector<mlir::Operation*> absorbed;
    for (mlir::Operation* o : m->ops)
      if (cand.contains(o)) absorbed.push_back(o);
    m->ops = std::move(absorbed);
    if (m->stacked && (m->helper_call == nullptr ||
                       !cand.contains(m->helper_call))) {
      // The fixpoint could not absorb the slice: it runs anyway, so the
      // fused op must read its result rather than gather a second copy.
      m->stacked = false;
    }
    Debug(absl::StrCat("matched a ragged dispatch (", m->name, ", ",
                       m->ops.size(), " ops absorbed)"));
    plan->ragged.push_back(std::move(m));
  }
}

}  // namespace metaljax
