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

THE DECODE FORM (row 10 rewrite 1, `MatchDecode`).  At one-token decode the
base emit spends ~8 non-gemv kernels per dot recovering the group index
from `ends` (arange, compare, sum, minimum, where) and padding the [m, n]
result to the tiling its only readers slice straight back; the row
permutation the graph already computes holds the same information.  The
proof, all structural:

  * `ends` is the inclusive cumsum (reduce_window, window g, pad [g-1, 0],
    add) of a bincount (scatter-add of ones onto zeros[g]) of an id vector
    V[m], whose provenance — the index result of a top-k over ONE row, or
    a sort by an iota / an in-range constant — bounds every id to [0, g).
    Then every row is in some group, `#(ends <= i)` is the i-th smallest
    id, and `take(V, perm)` for `perm` = the stable ascending argsort of V
    (the permutation that sorted the rows) IS that vector: the base emit's
    `eid`, entry for entry.  The base `valid` mask is then all-true.
    `perm` must be `sort(V, iota[m]).result1` exactly — the payload the
    IOTA (or the [0..m) constant the ingest folds it into), never merely
    an in-range constant: `sort(V, [1, 0])` permutes its payload, not the
    row order.
  * The rows: either `x` is jax's `repeat(x0, k)[perm]` at T = 1 — a
    clamping gather over a broadcast whose every non-unit source axis is
    the feature axis, FLATTENED to [rows, k] — so each row is `x0`
    whatever the index (`decode` 1); or they are real rows read as they
    are (`decode` 2, the down projection over the swiglu output).  A
    reshape to any other width cuts rotated rows and is real rows.
  * `nopad`: every reader of the root is `slice[0:m, 0:n]`; the slices are
    absorbed and aliased to the un-padded output.
  * Two roots over ONE `ends` of which only one takes the form: the base
    root's emit reads `ends`, so the joint fixpoint seeds every value a
    kept root's emit reads as live and the cumsum stays lowered.

Same `gather_mm` kernel, same rows, same matrices in the same (sorted)
order: bit-identical to the base form, whose `where` was an identity and
whose pad rows were never read.  Callees are looked through where jax
outlines them (`@argsort`, `@clip`, `@cumsum`; `chlo.top_k` as an op or a
composite) — only the CALL ops are absorbed, never a callee's body.
`METALJAX_RAGGED_DECODE=0` keeps the base form.

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

// --------------------------------------------------------------------------
// the decode form
// --------------------------------------------------------------------------

// A splat INTEGER constant, possibly behind a broadcast / reshape / convert
// chain (`ops` collects the chain; the constant itself is never absorbed).
bool IsSplatInt(mlir::Value v, int64_t* value,
                std::vector<mlir::Operation*>* ops) {
  mlir::Operation* op = v.getDefiningOp();
  std::vector<mlir::Operation*> chain;
  for (int depth = 0; op != nullptr && depth < 4; depth++) {
    if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
      auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.isSplat()) return false;
      if (!mlir::isa<mlir::IntegerType>(dense.getElementType())) return false;
      *value = dense.getSplatValue<mlir::APInt>().getSExtValue();
      if (ops != nullptr) ops->insert(ops->end(), chain.begin(), chain.end());
      return true;
    }
    const std::string name = OpName(op);
    if (name != "stablehlo.broadcast_in_dim" && name != "stablehlo.reshape" &&
        name != "stablehlo.convert")
      return false;
    chain.push_back(op);
    op = op->getOperand(0).getDefiningOp();
  }
  return false;
}

// The single-block callee of a `func.call`, or null.
mlir::func::FuncOp CalleeOf(mlir::Operation* call, mlir::ModuleOp module) {
  if (call == nullptr || OpName(call) != "func.call") return nullptr;
  auto sym = call->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
  if (!sym) return nullptr;
  auto fn = module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
  if (!fn || fn.getBody().getBlocks().size() != 1) return nullptr;
  if (fn.getBody().front().getNumArguments() != call->getNumOperands())
    return nullptr;
  return fn;
}

// A block whose ops are exactly `body...` then a return of `ret`: the
// small callees jax outlines (argsort, clip, cumsum) are recognized by
// listing them, so an extra op anywhere declines.
std::vector<mlir::Operation*> BodyOps(mlir::Block& block) {
  std::vector<mlir::Operation*> ops;
  for (mlir::Operation& o : block) ops.push_back(&o);
  return ops;
}

bool ReturnsExactly(mlir::Block& block, mlir::Value v) {
  mlir::Operation* term = block.getTerminator();
  return term != nullptr && term->getNumOperands() == 1 &&
         term->getOperand(0) == v;
}

// `ends = reduce_window(counts, 0)` with window [g], padding [g - 1, 0],
// body add -- the inclusive cumsum -- inline or as jax's `@cumsum` callee.
// Returns `counts`.
mlir::Value MatchCumsum(mlir::Value ends, int64_t g, mlir::ModuleOp module,
                        std::vector<mlir::Operation*>* ops) {
  mlir::Operation* def = DefOf(ends, "the interval ends");
  mlir::Value counts;
  mlir::stablehlo::ReduceWindowOp rw;
  std::vector<mlir::Operation*> absorb;
  if (auto fn = CalleeOf(def, module)) {
    mlir::Block& body = fn.getBody().front();
    if (body.getNumArguments() != 1 || def->getNumResults() != 1)
      Bail("the cumsum callee arity");
    for (mlir::Operation* o : BodyOps(body)) {
      const std::string n = OpName(o);
      if (n == "stablehlo.reduce_window") {
        if (rw) Bail("two cumsum windows");
        rw = mlir::cast<mlir::stablehlo::ReduceWindowOp>(o);
      } else if (n != "stablehlo.constant" && n != "stablehlo.broadcast_in_dim" &&
                 n != "func.return") {
        Bail(absl::StrCat("an unexpected cumsum callee op ", n));
      }
    }
    if (!rw || !ReturnsExactly(body, rw.getResult(0)))
      Bail("the cumsum callee does not return its window");
    if (rw.getInputs().size() != 1 || rw.getInputs()[0] != body.getArgument(0))
      Bail("the cumsum callee does not scan its argument");
    counts = def->getOperand(0);
    absorb.push_back(def);
  } else {
    rw = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(def);
    if (!rw) Bail("the interval ends are not a cumsum");
    if (rw.getInputs().size() != 1) Bail("a variadic cumsum");
    counts = rw.getInputs()[0];
    absorb.push_back(def);
  }
  // The window: [g] over [g] with low padding g - 1, no strides/dilations.
  auto wd = rw.getWindowDimensions();
  if (wd.size() != 1 || wd[0] != g) Bail("the cumsum window is not the length");
  auto all_ones = [](std::optional<llvm::ArrayRef<int64_t>> a) {
    if (!a) return true;
    for (int64_t x : *a) if (x != 1) return false;
    return true;
  };
  if (!all_ones(rw.getWindowStrides()) || !all_ones(rw.getBaseDilations()) ||
      !all_ones(rw.getWindowDilations()))
    Bail("the cumsum window is strided or dilated");
  auto pad = rw.getPadding();
  if (!pad) Bail("the cumsum window has no padding");
  auto pv = pad->getValues<int64_t>();
  std::vector<int64_t> pads(pv.begin(), pv.end());
  if (pads.size() != 2 || pads[0] != g - 1 || pads[1] != 0)
    Bail("the cumsum window padding is not [g - 1, 0]");
  if (rw.getInitValues().size() != 1 ||
      !IsZeroTensor(rw.getInitValues()[0], nullptr))
    Bail("the cumsum does not start at zero");
  mlir::Block& body = rw.getBody().front();
  std::vector<mlir::Operation*> bo = BodyOps(body);
  if (body.getNumArguments() != 2 || bo.size() != 2 ||
      !mlir::isa<mlir::stablehlo::AddOp>(bo[0]) ||
      bo[0]->getOperand(0) != body.getArgument(0) ||
      bo[0]->getOperand(1) != body.getArgument(1) ||
      !ReturnsExactly(body, bo[0]->getResult(0)))
    Bail("the cumsum body is not an add");
  if (ShapeOf(counts) != std::vector<int64_t>{g}) Bail("the counts shape");
  ops->insert(ops->end(), absorb.begin(), absorb.end());
  return counts;
}

// jnp's negative-index wrap, `select(x < 0, x + n, x)`, peeled to `x`
// (the ops absorbed); returns `v` itself when it is not one.
mlir::Value PeelWrap(mlir::Value v, std::vector<mlir::Operation*>* ops) {
  auto sel = mlir::dyn_cast_or_null<mlir::stablehlo::SelectOp>(v.getDefiningOp());
  if (!sel) return v;
  auto cmp = mlir::dyn_cast_or_null<mlir::stablehlo::CompareOp>(
      sel.getPred().getDefiningOp());
  auto add = mlir::dyn_cast_or_null<mlir::stablehlo::AddOp>(
      sel.getOnTrue().getDefiningOp());
  if (!cmp || !add) return v;
  mlir::Value x = sel.getOnFalse();
  int64_t zero = -1, n = -1;
  std::vector<mlir::Operation*> chain;
  if (cmp.getComparisonDirection() != mlir::stablehlo::ComparisonDirection::LT ||
      cmp.getLhs() != x || !IsSplatInt(cmp.getRhs(), &zero, &chain) || zero != 0)
    return v;
  if (add.getLhs() != x || !IsSplatInt(add.getRhs(), &n, &chain) || n < 0)
    return v;
  ops->push_back(sel);
  ops->push_back(cmp);
  ops->push_back(add);
  ops->insert(ops->end(), chain.begin(), chain.end());
  return x;
}

// jnp.bincount's `clip(x, 0)`: `maximum(0, x)` inline, or the `@clip`
// callee (`maximum(broadcast(convert(arg1)), arg0)` with a non-negative
// bound).  Returns `x`, or `v` when it is not one.
mlir::Value PeelClip(mlir::Value v, mlir::ModuleOp module,
                     std::vector<mlir::Operation*>* ops) {
  mlir::Operation* def = v.getDefiningOp();
  if (def == nullptr) return v;
  if (auto mxo = mlir::dyn_cast<mlir::stablehlo::MaxOp>(def)) {
    int64_t lo = 0;
    std::vector<mlir::Operation*> chain;
    if (IsSplatInt(mxo.getLhs(), &lo, &chain) && lo >= 0) {
      ops->push_back(def);
      ops->insert(ops->end(), chain.begin(), chain.end());
      return mxo.getRhs();
    }
    chain.clear();
    if (IsSplatInt(mxo.getRhs(), &lo, &chain) && lo >= 0) {
      ops->push_back(def);
      ops->insert(ops->end(), chain.begin(), chain.end());
      return mxo.getLhs();
    }
    return v;
  }
  auto fn = CalleeOf(def, module);
  if (!fn || def->getNumResults() != 1) return v;
  mlir::Block& body = fn.getBody().front();
  if (body.getNumArguments() < 1 || body.getNumArguments() > 2) return v;
  mlir::stablehlo::MaxOp mxo;
  for (mlir::Operation* o : BodyOps(body)) {
    const std::string n = OpName(o);
    if (n == "stablehlo.maximum") {
      if (mxo) return v;
      mxo = mlir::cast<mlir::stablehlo::MaxOp>(o);
    } else if (n != "stablehlo.constant" && n != "stablehlo.broadcast_in_dim" &&
               n != "stablehlo.convert" && n != "func.return") {
      return v;
    }
  }
  if (!mxo || !ReturnsExactly(body, mxo.getResult())) return v;
  // One side is the argument, the other a bound that is a non-negative
  // constant or the call's second operand (a non-negative constant).
  mlir::Value x, bound;
  if (mxo.getRhs() == body.getArgument(0)) {
    x = mxo.getRhs(); bound = mxo.getLhs();
  } else if (mxo.getLhs() == body.getArgument(0)) {
    x = mxo.getLhs(); bound = mxo.getRhs();
  } else {
    return v;
  }
  // Peel the bound to a block argument or a constant.
  for (int depth = 0; depth < 4; depth++) {
    mlir::Operation* bd = bound.getDefiningOp();
    if (bd == nullptr) break;
    const std::string n = OpName(bd);
    if (n != "stablehlo.broadcast_in_dim" && n != "stablehlo.convert") break;
    bound = bd->getOperand(0);
  }
  int64_t lo = -1;
  if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(bound)) {
    if (ba.getOwner() != &body || ba.getArgNumber() != 1) return v;
    if (!IsSplatInt(def->getOperand(1), &lo, nullptr) || lo < 0) return v;
  } else if (!IsSplatInt(bound, &lo, nullptr) || lo < 0) {
    return v;
  }
  ops->push_back(def);
  return def->getOperand(0);
}

// `counts = scatter-add(zeros[g], ids', ones)` -- jnp.bincount -- whose
// index chain peels (broadcast / reshape, the wrap, the clip) to an
// integer V of shape [m].  Returns V.
mlir::Value MatchBincount(mlir::Value counts, int64_t g, int64_t m,
                          mlir::ModuleOp module,
                          std::vector<mlir::Operation*>* ops) {
  auto sc = mlir::dyn_cast_or_null<mlir::stablehlo::ScatterOp>(
      DefOf(counts, "the group counts"));
  if (!sc) Bail("the group counts are not a bincount");
  if (sc.getInputs().size() != 1 || sc.getUpdates().size() != 1)
    Bail("a variadic bincount");
  std::vector<mlir::Operation*> absorb;
  if (!IsZeroTensor(sc.getInputs()[0], &absorb))
    Bail("the bincount does not start at zero");
  int64_t one = 0;
  if (!IsSplatInt(sc.getUpdates()[0], &one, &absorb) || one != 1)
    Bail("the bincount does not count ones");
  auto dn = sc.getScatterDimensionNumbers();
  if (!dn.getUpdateWindowDims().empty() ||
      dn.getInsertedWindowDims().size() != 1 || dn.getInsertedWindowDims()[0] != 0 ||
      dn.getScatterDimsToOperandDims().size() != 1 ||
      dn.getScatterDimsToOperandDims()[0] != 0 || dn.getIndexVectorDim() != 1 ||
      !dn.getInputBatchingDims().empty())
    Bail("the bincount scatter geometry");
  mlir::Block& body = sc.getUpdateComputation().front();
  std::vector<mlir::Operation*> bo = BodyOps(body);
  if (body.getNumArguments() != 2 || bo.size() != 2 ||
      !mlir::isa<mlir::stablehlo::AddOp>(bo[0]) ||
      !ReturnsExactly(body, bo[0]->getResult(0)))
    Bail("the bincount body is not an add");
  mlir::Value idx = sc.getScatterIndices();
  if (ShapeOf(idx) != std::vector<int64_t>{m, 1})
    Bail("the bincount does not count every row once");
  absorb.push_back(sc);
  // The index chain, down to the id vector: shape ops only until the
  // value is [m] (the vector the argsort reads), then the wrap and the
  // clip, which keep the shape.
  mlir::Value v = idx;
  const std::vector<int64_t> want{m};
  for (int depth = 0; depth < 8; depth++) {
    mlir::Operation* d = v.getDefiningOp();
    if (d == nullptr) break;
    if (ShapeOf(v) != want) {
      const std::string n = OpName(d);
      if (n != "stablehlo.broadcast_in_dim" && n != "stablehlo.reshape") break;
      std::vector<int64_t> a = ShapeOf(d->getOperand(0)), b = ShapeOf(v);
      int64_t na = 1, nb = 1;
      for (int64_t x : a) na *= x;
      for (int64_t x : b) nb *= x;
      if (na != nb) Bail("the bincount index chain changes the row count");
      absorb.push_back(d);
      v = d->getOperand(0);
      continue;
    }
    mlir::Value w = PeelWrap(v, &absorb);
    if (w != v) { v = w; continue; }
    w = PeelClip(v, module, &absorb);
    if (w != v) { v = w; continue; }
    break;
  }
  if (ShapeOf(v) != std::vector<int64_t>{m}) {
    std::string shape;
    for (int64_t d : ShapeOf(v)) absl::StrAppend(&shape, d, "x");
    Bail(absl::StrCat("the ids are not one per row (", shape, " from ",
                      OpName(v.getDefiningOp()), ")"));
  }
  if (!mlir::isa<mlir::IntegerType>(ElemOf(v))) Bail("the ids are not integral");
  ops->insert(ops->end(), absorb.begin(), absorb.end());
  return v;
}

// The comparator of `sort` is the plain ascending compare on operand 0:
// `compare LT arg0, arg1` (or `GT arg1, arg0`) returned as is.
bool IsAscendingSort(mlir::stablehlo::SortOp sort) {
  if (sort.getComparator().getBlocks().size() != 1) return false;
  mlir::Block& block = sort.getComparator().front();
  if (block.getNumArguments() < 2) return false;
  std::vector<mlir::Operation*> bo = BodyOps(block);
  if (bo.size() != 2) return false;
  auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(bo[0]);
  if (!cmp || !ReturnsExactly(block, cmp.getResult())) return false;
  using D = mlir::stablehlo::ComparisonDirection;
  mlir::Value a0 = block.getArgument(0), a1 = block.getArgument(1);
  if (cmp.getComparisonDirection() == D::LT)
    return cmp.getLhs() == a0 && cmp.getRhs() == a1;
  if (cmp.getComparisonDirection() == D::GT)
    return cmp.getLhs() == a1 && cmp.getRhs() == a0;
  return false;
}

// Every element of a dense integer constant lies in [0, g).
bool ConstantInRange(mlir::Value v, int64_t g) {
  auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return false;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !mlir::isa<mlir::IntegerType>(dense.getElementType()))
    return false;
  for (const mlir::APInt& x : dense.getValues<mlir::APInt>()) {
    const int64_t i = x.getSExtValue();
    if (i < 0 || i >= g) return false;
  }
  return true;
}

// The ids' provenance: V peels (reshape / convert / slice) to the index
// result of a top-k over ONE row (`chlo.top_k`, as an op or the composite
// a portable artifact makes of it), or to result r of a sort whose
// operand r is an iota along the sorted axis of extent <= g or a dense
// constant with every element in [0, g).  Either bounds every id to
// [0, g); the one-row requirement is what makes this a DECODE form
// (`tokens` = the product of the leading dims).
void ProveIdRange(mlir::Value v, int64_t g, int64_t* tokens) {
  for (int depth = 0; depth < 6; depth++) {
    mlir::Operation* d = v.getDefiningOp();
    if (d == nullptr) Bail("ids of unproven range (an argument)");
    const std::string n = OpName(d);
    if (n == "stablehlo.reshape" || n == "stablehlo.convert" ||
        n == "stablehlo.slice") {
      v = d->getOperand(0);
      continue;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(v);
    const unsigned r = result ? result.getResultNumber() : 0u;
    if (n == "chlo.top_k" ||
        (n == "stablehlo.composite" &&
         d->getAttrOfType<mlir::StringAttr>("name") &&
         d->getAttrOfType<mlir::StringAttr>("name").getValue() == "chlo.top_k")) {
      if (r != 1) Bail("ids of unproven range (top-k values, not indices)");
      std::vector<int64_t> s = ShapeOf(d->getOperand(0));
      if (s.empty() || s.back() > g) Bail("ids of unproven range (the top-k width)");
      *tokens = 1;
      for (size_t i = 0; i + 1 < s.size(); i++) *tokens *= s[i];
      return;
    }
    if (auto sort = mlir::dyn_cast<mlir::stablehlo::SortOp>(d)) {
      if (r >= sort.getNumOperands()) Bail("ids of unproven range (sort arity)");
      mlir::Value src = sort.getOperand(r);
      const int64_t dim = sort.getDimension();
      std::vector<int64_t> s = ShapeOf(src);
      auto iota = mlir::dyn_cast_or_null<mlir::stablehlo::IotaOp>(src.getDefiningOp());
      bool ok = false;
      if (iota && static_cast<int64_t>(iota.getIotaDimension()) == dim &&
          dim < static_cast<int64_t>(s.size()) && s[static_cast<size_t>(dim)] <= g)
        ok = true;
      if (!ok && ConstantInRange(src, g)) ok = true;
      if (!ok) Bail("ids of unproven range (the sort's index operand)");
      *tokens = 1;
      for (size_t i = 0; i < s.size(); i++)
        if (static_cast<int64_t>(i) != dim) *tokens *= s[i];
      return;
    }
    Bail(absl::StrCat("ids of unproven range (", n, ")"));
  }
  Bail("ids of unproven range (chain too deep)");
}

// The sort payload that makes `sort(V, payload).result1` the ARGSORT of V:
// the [m] iota along axis 0 -- as `stablehlo.iota`, or as the dense
// constant [0, 1, ..., m - 1] the ingest folds an iota into.  An in-range
// constant is NOT enough: `[1, 0]` sorts to a permutation of itself, not
// of the row order, and `take(V, perm)` would name the wrong expert.
bool IsIotaPayload(mlir::Value v, int64_t m) {
  if (ShapeOf(v) != std::vector<int64_t>{m}) return false;
  if (auto iota = mlir::dyn_cast_or_null<mlir::stablehlo::IotaOp>(
          v.getDefiningOp()))
    return iota.getIotaDimension() == 0;
  auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return false;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !mlir::isa<mlir::IntegerType>(dense.getElementType()))
    return false;
  int64_t i = 0;
  for (const mlir::APInt& x : dense.getValues<mlir::APInt>())
    if (x.getSExtValue() != i++) return false;
  return i == m;
}

// `perm` = the stable ascending argsort of V: a `func.call @argsort(V)`
// whose body is `iota; sort(arg0, iota) dim 0; return result 1`, or an
// inline `sort(V, iota)` -- the second result of either.  Every sort of V
// that is not that is named: the comparator, or the payload.
mlir::Value FindPerm(mlir::Value v, int64_t m, mlir::ModuleOp module) {
  const char* why = nullptr;
  for (mlir::Operation* u : v.getUsers()) {
    if (auto sort = mlir::dyn_cast<mlir::stablehlo::SortOp>(u)) {
      if (sort.getNumOperands() != 2 || sort.getOperand(0) != v ||
          sort.getDimension() != 0)
        continue;
      if (!IsAscendingSort(sort)) {
        why = "the sort is not ascending";
        continue;
      }
      if (!IsIotaPayload(sort.getOperand(1), m)) {
        why = "the sort's payload is not the iota";
        continue;
      }
      return sort.getResult(1);
    }
    auto fn = CalleeOf(u, module);
    if (!fn || u->getNumOperands() != 1 || u->getNumResults() != 1) continue;
    mlir::Block& body = fn.getBody().front();
    mlir::stablehlo::SortOp sort;
    bool plain = true;
    for (mlir::Operation* o : BodyOps(body)) {
      const std::string n = OpName(o);
      if (n == "stablehlo.sort") {
        if (sort) plain = false;
        sort = mlir::cast<mlir::stablehlo::SortOp>(o);
      } else if (n != "stablehlo.iota" && n != "stablehlo.constant" &&
                 n != "func.return") {
        plain = false;
      }
    }
    if (!sort) continue;
    if (!plain || sort.getNumOperands() != 2 ||
        sort.getOperand(0) != body.getArgument(0) || sort.getDimension() != 0 ||
        !ReturnsExactly(body, sort.getResult(1)) || !IsAscendingSort(sort)) {
      why = "the sort is not ascending";
      continue;
    }
    if (!IsIotaPayload(sort.getOperand(1), m)) {
      why = "the sort's payload is not the iota";
      continue;
    }
    return u->getResult(0);
  }
  if (why) Bail(why);
  Bail("the ids are not argsorted in this block");
}

// The rows are ONE replicated activation: `x = gather(reshape(
// broadcast_in_dim(x0)), idx)` where every non-unit axis of x0 maps to the
// LAST axis of the broadcast, the reshape flattens to [m, k], and the
// gather takes whole rows (clamped, so the index value is immaterial).
// Records `x0`, absorbs the gather / reshape / broadcast and the index
// chain that feeds only the gather.
void MatchReplicated(RaggedMatch* m, std::vector<mlir::Operation*>* ops) {
  auto gather = mlir::dyn_cast_or_null<mlir::stablehlo::GatherOp>(
      m->x.getDefiningOp());
  if (!gather) Bail("rows are not one replicated activation (no row gather)");
  auto dn = gather.getDimensionNumbers();
  if (dn.getOffsetDims().size() != 1 || dn.getOffsetDims()[0] != 1 ||
      dn.getCollapsedSliceDims().size() != 1 || dn.getCollapsedSliceDims()[0] != 0 ||
      dn.getStartIndexMap().size() != 1 || dn.getStartIndexMap()[0] != 0 ||
      dn.getIndexVectorDim() != 1 || !dn.getOperandBatchingDims().empty())
    Bail("rows are not one replicated activation (the gather geometry)");
  auto ss = gather.getSliceSizes();
  if (ss.size() != 2 || ss[0] != 1 || ss[1] != m->k)
    Bail("rows are not one replicated activation (the gather slice)");
  if (ShapeOf(gather.getResult()) != std::vector<int64_t>{m->m, m->k})
    Bail("rows are not one replicated activation (the gather result)");
  auto rs = mlir::dyn_cast_or_null<mlir::stablehlo::ReshapeOp>(
      gather.getOperand().getDefiningOp());
  if (!rs) Bail("rows are not one replicated activation (no reshape)");
  auto bc = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      rs.getOperand().getDefiningOp());
  if (!bc) Bail("rows are not one replicated activation (no broadcast)");
  std::vector<int64_t> bshape = ShapeOf(bc.getResult());
  std::vector<int64_t> x0shape = ShapeOf(bc.getOperand());
  if (bshape.empty() || bshape.back() != m->k)
    Bail("rows are not one replicated activation (the broadcast shape)");
  auto dims = bc.getBroadcastDimensions();
  int64_t x0_numel = 1;
  for (size_t i = 0; i < x0shape.size(); i++) {
    x0_numel *= x0shape[i];
    if (x0shape[i] != 1 && dims[i] != static_cast<int64_t>(bshape.size()) - 1)
      Bail(absl::StrCat("rows are not one replicated activation (",
                        ShapeOf(bc.getOperand())[i], " rows)"));
  }
  if (x0_numel != m->k)
    Bail("rows are not one replicated activation (the activation width)");
  // The reshape must FLATTEN the broadcast to [numel / k, k]: a k-wide row
  // of it is then one copy of x0.  Any other width ([1, 6, 32] -> [4, 48])
  // cuts rows that are x0 rotated, and the gather reads those.
  int64_t b_numel = 1;
  for (int64_t d : bshape) b_numel *= d;
  if (ShapeOf(rs.getResult()) != std::vector<int64_t>{b_numel / m->k, m->k})
    Bail("rows are not one replicated activation (the reshape width)");
  std::vector<mlir::Operation*> absorb{gather, rs, bc};
  // The index chain: pure shape / elementwise ops feeding only the gather
  // are absorbable (the fixpoint checks the "only"); anything else stays.
  std::vector<mlir::Value> stack{gather.getStartIndices()};
  llvm::DenseSet<mlir::Operation*> seen;
  for (int steps = 0; !stack.empty() && steps < 16; steps++) {
    mlir::Value v = stack.back();
    stack.pop_back();
    mlir::Operation* d = v.getDefiningOp();
    if (d == nullptr || !seen.insert(d).second) continue;
    const std::string n = OpName(d);
    if (n == "stablehlo.constant") continue;
    if (n != "stablehlo.broadcast_in_dim" && n != "stablehlo.reshape" &&
        n != "stablehlo.select" && n != "stablehlo.compare" &&
        n != "stablehlo.add" && n != "stablehlo.convert" &&
        n != "stablehlo.maximum" && n != "stablehlo.minimum")
      continue;
    if (d->getNumResults() != 1) continue;
    absorb.push_back(d);
    for (mlir::Value o : d->getOperands()) stack.push_back(o);
  }
  m->x0 = bc.getOperand();
  m->decode = 1;
  ops->insert(ops->end(), absorb.begin(), absorb.end());
}

// Every reader of the root is `slice[0:m, 0:n]`: the slices are absorbed
// (aliased to the un-padded [m, n] output by the lowering) and the pad is
// never made.  A root with any other reader keeps the pad, and says which
// reader (only the debug narration tells the two forms apart).
void MatchNoPad(RaggedMatch* m) {
  if (m->M <= m->m) return;
  std::vector<mlir::Operation*> slices;
  for (mlir::Operation* u : m->root->getUsers()) {
    auto sl = mlir::dyn_cast<mlir::stablehlo::SliceOp>(u);
    bool rows = false;
    if (sl) {
      auto st = sl.getStartIndices();
      auto li = sl.getLimitIndices();
      auto sd = sl.getStrides();
      rows = st.size() == 2 && st[0] == 0 && st[1] == 0 && li[0] == m->m &&
             li[1] == m->n && sd[0] == 1 && sd[1] == 1;
    }
    if (!rows) {
      if (kDebug)
        Debug(absl::StrCat("pad kept (", OpName(u), " reads the root)"));
      return;
    }
    slices.push_back(u);
  }
  if (slices.empty()) return;
  m->nopad = true;
  m->row_slices = std::move(slices);
}

const bool kDecodeOff = EnvOff("METALJAX_RAGGED_DECODE");

// The decode form (file comment): prove the ids and their sort, then the
// replicated rows (optional: the down projection reads real rows), then
// the pad-only readers.  Bail leaves the base form standing.
void MatchDecode(RaggedMatch* m, mlir::ModuleOp module) {
  if (kDecodeOff) return;
  std::vector<mlir::Operation*> absorb;
  mlir::Value counts = MatchCumsum(m->ends, m->g, module, &absorb);
  mlir::Value ids = MatchBincount(counts, m->g, m->m, module, &absorb);
  int64_t tokens = 0;
  ProveIdRange(ids, m->g, &tokens);
  if (tokens != 1)
    Bail(absl::StrCat("rows are not one replicated activation (", tokens,
                      " tokens)"));
  mlir::Value perm = FindPerm(ids, m->m, module);
  // Frame guard (MatchStacked's): the root lowers inside its block's
  // splice, so the emit's inputs must be resolvable there.
  mlir::Block* frame = m->root->getBlock();
  auto in_frame = [&](mlir::Value v) {
    if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v))
      return ba.getOwner() == frame;
    return v.getDefiningOp()->getBlock() == frame;
  };
  if (!in_frame(ids) || !in_frame(perm))
    Bail("the ids or their sort come from another frame");
  m->ids = ids;
  m->perm = perm;
  m->decode = 2;
  try {
    std::vector<mlir::Operation*> rep;
    MatchReplicated(m, &rep);
    if (!in_frame(m->x0)) Bail("the activation comes from another frame");
    absorb.insert(absorb.end(), rep.begin(), rep.end());
  } catch (const Reject& e) {
    // Real rows: the form still drops the prefix and the pad.
    m->decode = 2;
    m->x0 = mlir::Value();
    if (kDebug) Debug(absl::StrCat("decode rows stay real (", e.why, ")"));
  }
  MatchNoPad(m);
  m->ops.insert(m->ops.end(), absorb.begin(), absorb.end());
  m->name = absl::StrCat(m->name, m->decode == 1 ? " decode" : " decode-rows",
                         m->nopad ? " nopad" : "");
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
          try {
            MatchDecode(m.get(), module);
          } catch (const Reject& e) {
            // The base form stands (a partial proof absorbed nothing).
            if (kDebug)
              Debug(absl::StrCat("decode form declined (", e.why, ")"));
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
    // A row slice another recognizer owns is not ours to alias: the match
    // keeps its pad and the slice runs (the same two-owner rule as above,
    // applied to the readers the decode form absorbs).
    if (m->nopad) {
      bool slice_taken = false;
      for (mlir::Operation* o : m->row_slices)
        slice_taken = slice_taken || taken.contains(o);
      if (slice_taken) {
        m->nopad = false;
        m->row_slices.clear();
        const std::string suffix = " nopad";
        if (m->name.size() >= suffix.size() &&
            m->name.compare(m->name.size() - suffix.size(), suffix.size(),
                            suffix) == 0)
          m->name.resize(m->name.size() - suffix.size());
        if (kDebug) Debug("pad kept (another recognizer owns a row slice)");
      }
    }
    roots.insert(m->root);
    kept.push_back(std::move(m));
  }
  if (kept.empty()) return;

  // What each kept root's EMIT reads (LowerRagged): those values must stay
  // lowered whatever another root absorbed.  Two roots over one `ends`
  // where only one takes the decode form is the case: the decode root
  // absorbs the cumsum, the base root reads it, and without this seed the
  // fixpoint saw every user of the cumsum inside the candidate set --
  // then the base root's Slot(ends) failed and the WHOLE fused tape fell
  // back to the plain one.
  llvm::DenseSet<mlir::Value> emit_reads;
  for (const auto& m : kept) {
    emit_reads.insert(m->stacked ? m->w_stack : m->w);
    if (m->stacked) emit_reads.insert(m->layer);
    if (m->decode == 0) {
      emit_reads.insert(m->x);
      emit_reads.insert(m->ends);
    } else {
      emit_reads.insert(m->decode == 1 ? m->x0 : m->x);
      emit_reads.insert(m->ids);
      emit_reads.insert(m->perm);
    }
  }
  llvm::DenseSet<mlir::Operation*> cand;
  for (const auto& m : kept)
    for (mlir::Operation* o : m->ops) {
      if (mlir::isa<mlir::stablehlo::ConstantOp>(o)) continue;
      bool read = false;
      for (mlir::Value r : o->getResults()) read = read || emit_reads.contains(r);
      if (!read) cand.insert(o);
    }
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
    // The decode form's row slices are readers OF the root, outside the
    // fixpoint's producer discipline: they are absorbed unconditionally
    // (the lowering aliases each to the un-padded output).
    if (m->nopad)
      absorbed.insert(absorbed.end(), m->row_slices.begin(),
                      m->row_slices.end());
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
