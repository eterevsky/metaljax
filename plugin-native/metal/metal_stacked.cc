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

Two things are materialized ONCE per executable on top of that read, both in
`BuildStackedPacks`, both governor-admitted and both charged to
METALJAX_STACKED_RELAYOUT_MB:

* the RELAYOUT (B3): a stack whose contracted axes STRADDLE the layer axis
  in the original buffer -- maxtext's attention out-projection,
  [heads, L, head_dim, model] -- has no [L, K, N] view of that buffer, but
  its carried (transposed) layout does, so the pack wave materializes that
  layout and the emit gathers from it;
* the sibling PACK (row 10 rewrite 3, METALJAX_STACKED_PACK): several of
  these dots in one block read ONE activation against DIFFERENT per-layer
  stacks -- row 10's q and kv_a over the attention norm, its router and two
  shared-expert gates over the MLP norm -- and their [L, K, N] views
  concatenated on the free axis make one [L, K, n_total] the emit reads with
  a SINGLE gather_mm, the members taking last-axis slices of the result
  (views at M == 1).  B6 for the dots this file owns (metal_proj.cc, whose
  tile policy this one follows verbatim); worth a dispatch and a
  command-buffer count and NOT the pair overlap B6 found on rows 7/11, which
  `row10-scout/analysis/profile-findings.md` section 3 refutes here.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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

int64_t RelayoutBudgetBytes();

// METALJAX_STACKED_RELAYOUT=0 declines the relayout form (the slice chain
// runs, as before B3); METALJAX_STACKED_RELAYOUT_MB caps what one
// executable's relaid packs may hold in total (default 2048 MB, 0 = none).
bool RelayoutEnabled() {
  static const bool on = !EnvOff("METALJAX_STACKED_RELAYOUT") &&
                         RelayoutBudgetBytes() > 0;
  return on;
}

int64_t RelayoutBudgetBytes() {
  static const int64_t bytes = [] {
    const char* v = std::getenv("METALJAX_STACKED_RELAYOUT_MB");
    if (v == nullptr || *v == '\0') return 2048LL << 20;
    char* end = nullptr;
    const long long mb = std::strtoll(v, &end, 10);
    if (end == v || mb < 0) return 2048LL << 20;
    return static_cast<int64_t>(mb) << 20;
  }();
  return bytes;
}

// Bytes per element of a float type the dot accepts (the match already
// required a non-f64 float).
int ElemBytes(mlir::Type t) {
  if (auto f = mlir::dyn_cast<mlir::FloatType>(t))
    return static_cast<int>((f.getWidth() + 7) / 8);
  return 4;
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
  int origin_arg = -1;
  {
    auto ba = mlir::dyn_cast<mlir::BlockArgument>(origin);
    if (!ba || ba.getOwner() != main_block)
      Bail("the stack does not hoist to a @main argument");
    origin_arg = static_cast<int>(ba.getArgNumber());
  }
  sdims = ShapeOf(origin);
  if (perm.size() != sdims.size() || perm.size() != wdims.size())
    Bail("the stack rank disagrees with the slice");

  // One layout's proof: `dims` row-major with `p` mapping carried axes to
  // its axes.  The rhs's dims, as axes of that layout: rhs dim j is carried
  // axis (j < a ? j : j + 1) -- the slice axis dropped -- mapped through p.
  // The contracted axes (K) are the first nc of them, the free axes (N) the
  // rest; each group must collapse to a single stride in that order for the
  // [L, K, N] view to exist.
  struct Layout {
    int64_t L = 0, K = 0, N = 0, sl = 0, sk = 0, sn = 0;
  };
  auto prove = [&](const std::vector<int64_t>& dims,
                   const std::vector<int64_t>& p,
                   std::string* why) -> std::optional<Layout> {
    // Row-major strides of that layout, in elements.
    std::vector<int64_t> st(dims.size());
    {
      int64_t acc = 1;
      for (int i = static_cast<int>(dims.size()) - 1; i >= 0; i--) {
        st[i] = acc;
        acc *= std::max<int64_t>(dims[i], 1);
      }
    }
    std::vector<int64_t> in_order;
    for (size_t j = 0; j + 1 < wdims.size(); j++) {
      const int64_t wax = static_cast<int64_t>(j) < a
                              ? static_cast<int64_t>(j)
                              : static_cast<int64_t>(j) + 1;
      in_order.push_back(p[wax]);
    }
    auto collapse = [&](size_t lo, size_t hi,  // in_order indices [lo, hi)
                        const char* what,
                        std::pair<int64_t, int64_t>* out) -> bool {
      int64_t extent = 1;
      for (size_t i = lo; i < hi; i++) {
        const int64_t ax = in_order[i];
        extent *= dims[ax];
        if (i + 1 < hi) {
          const int64_t nx = in_order[i + 1];
          if (st[ax] != dims[nx] * st[nx]) {
            *why = absl::StrCat(what, " axes do not collapse in the stack");
            return false;
          }
        }
      }
      *out = {extent, st[in_order[hi - 1]]};
      return true;
    };
    std::pair<int64_t, int64_t> kk, nn;
    if (!collapse(0, nc, "the contracted", &kk)) return std::nullopt;
    if (!collapse(nc, in_order.size(), "the free", &nn)) return std::nullopt;
    Layout l;
    l.K = kk.first;
    l.sk = kk.second;
    l.N = nn.first;
    l.sn = nn.second;
    if (l.sn != 1) {
      *why = "the free stride is not 1";
      return std::nullopt;
    }
    l.L = wdims[a];
    l.sl = st[p[a]];
    if (l.L < 1 || l.K < 1 || l.N < 1) {
      *why = "degenerate sizes";
      return std::nullopt;
    }
    return l;
  };

  std::string why;
  std::optional<Layout> lay = prove(sdims, perm, &why);
  bool relayout = false;
  if (!lay.has_value()) {
    // The RELAYOUT form: the ORIGINAL buffer has no [L, K, N] view (maxtext's
    // attention out-projection: the layer axis sits between the two
    // contracted axes), but the CARRIED layout -- the transpose the graph
    // itself hoisted -- is row-major in `wdims`, and if THAT collapses the
    // pack wave materializes it once per executable (BuildStackedPacks) and
    // the emit reads the pack.  A stack that is not a transposed @main
    // argument has no other layout to try.
    std::vector<int64_t> ident(wdims.size());
    std::iota(ident.begin(), ident.end(), 0);
    const bool transposed = perm != ident;
    std::string why2;
    if (transposed && RelayoutEnabled()) {
      lay = prove(wdims, ident, &why2);
      if (lay.has_value()) relayout = true;
    }
    if (!lay.has_value()) Bail(why);
  }
  const int64_t K = lay->K, N = lay->N, L = lay->L;
  const int64_t sk = lay->sk, sn = lay->sn, sl = lay->sl;
  if (K * N < 16384) Bail("the weight is too small to matter");

  // The inverse permutation: transpose(w_carry, back_perm) is the original
  // -- or the identity when the emit reads a relaid pack, which IS the
  // carried layout, contiguous.
  match->back_perm.resize(perm.size());
  // The @main argument the stack hoists to, for BOTH forms: the relayout
  // pack reads it transposed, and a sibling pack (below) reads the proven
  // [L, K, N] view of it straight.
  match->origin_arg = origin_arg;
  if (relayout) {
    std::iota(match->back_perm.begin(), match->back_perm.end(), 0);
    match->relayout = true;
    match->relayout_perm = perm;
    int64_t numel = 1;
    for (int64_t d : wdims) numel *= std::max<int64_t>(d, 1);
    match->relayout_bytes =
        numel * static_cast<int64_t>(ElemBytes(ElemOf(dot.getRhs())));
  } else {
    for (size_t i = 0; i < perm.size(); i++) match->back_perm[perm[i]] = i;
  }

  int64_t M = 1;
  for (size_t i = 0; i + nc < lshape.size(); i++) M *= lshape[i];
  if (M < 1) Bail("degenerate rows");
  // The relayout pays only on the DECODE geometry.  At M == 1 the pack is
  // read by the same gemv the other stacked reads dispatch, bit-identical
  // to the slice chain and 0.7 ms/token faster on row 10; at M > 1
  // gather_mm takes a different steel path from the slice chain's matmul,
  // and on row 11's PREFILL program (M = 64, 28 layers) that measured
  // +3 ms over a 25 ms prefill (b3-row10/findings.txt section 5).  So a
  // multi-row read keeps its slice chain -- the form it always had.
  if (relayout && M != 1)
    Bail(absl::StrCat("the relayout form is decode-only (M=", M,
                      "): the slice chain runs"));

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

// Fills `plan->stacked`; `GroupStackedPacks` (below) runs on top of it.
void AnalyzeStackedDotImpl(mlir::func::FuncOp fn, RewritePlan* plan) {
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
                           m->ops.size(), " ops absorbed",
                           m->relayout ? ", relayout planned" : "", ")"));
        plan->stacked.push_back(std::move(m));
      }
      return;
    }
    if (kept.empty()) return;
  }
}

// --------------------------------------------------------------------------
// the sibling PACK (row 10 rewrite 3): stacked dots over ONE activation
// --------------------------------------------------------------------------

// METALJAX_STACKED_PACK: 0 = off (every member keeps its own gather_mm);
// unset/1/auto = the `auto` policy, metal_proj.cc's verbatim; all = pack
// every member of every group (the A/B arm for the policy).
enum class PackPolicy { kOff, kAuto, kAll };

PackPolicy StackedPackPolicy() {
  static const PackPolicy p = [] {
    const char* v = std::getenv("METALJAX_STACKED_PACK");
    if (v == nullptr || *v == '\0') return PackPolicy::kAuto;
    const std::string s(v);
    if (s == "0") return PackPolicy::kOff;
    if (s == "all") return PackPolicy::kAll;
    return PackPolicy::kAuto;
  }();
  return p;
}

const char* PackPolicyName(PackPolicy p) {
  switch (p) {
    case PackPolicy::kOff: return "off";
    case PackPolicy::kAuto: return "auto";
    case PackPolicy::kAll: return "all";
  }
  return "?";
}

// A per-layer matrix this large is bandwidth-bound already, so merging it
// with a sibling buys nothing and costs L times its size in resident memory
// (metal_proj.cc `kMaxMemberBytes`, the same constant for the same reason).
constexpr int64_t kMaxPackMemberBytes = 64LL << 20;

// The M = 1 `gemv_t` tile MLX picks for a [K -> N] read (metal_proj.cc
// `TileOf`, its `kn` branch).  Every stacked view is [L, K, N] with the free
// stride 1, which is exactly the `gemv_t_gather` a [K, N] weight runs -- the
// census reads `bn4` for row 10's kv_a (N = 576) and `bn16` for its q
// (N = 3072), which is what this returns.
struct GemvTile {
  int sm = 0, sn = 0, bn = 0;
  bool operator==(const GemvTile& o) const {
    return sm == o.sm && sn == o.sn && bn == o.bn;
  }
  bool operator!=(const GemvTile& o) const { return !(*this == o); }
};

GemvTile TileFor(int64_t K, int64_t N) {
  int sm = 8, sn = 4;
  if (K >= 8192 && N >= 2048) {
    sm = 4;
    sn = 8;
  }
  const int bn = N >= 2048 ? 16 : N >= 512 ? 4 : 2;
  const int tn = N < 4 ? 1 : 4;
  while (sn > 4 && (N + bn * sn * tn - 1) / (bn * sn * tn) < 32) {
    sn /= 2;
    sm *= 2;
  }
  return GemvTile{sm, sn, bn};
}

// The ACTIVATION behind a member's `dot.getLhs()`, for the group key alone.
//
// Two sibling projections over one activation do not necessarily name the
// same MLIR Value: jax spells row 10's q as a dot over the attention norm's
// result and its kv_a as a dot over a `reshape` of it, so the two lhs Values
// differ while the bytes they read are the same.  Peeling those views is
// what lets the pair meet in one group.
//
// Only element-order- AND numel-preserving steps are peeled:
//
//   `stablehlo.reshape`           row-major reinterpretation by definition;
//   `optimization_barrier`,       the lowering's own arity-preserving
//   `sdy.sharding_constraint`,    aliases -- `LowerOp` binds the result to
//   `sdy.reshard`,                the OPERAND'S SLOT, so the two spellings
//   `Sharding` / `annotate_       are already one array at replay time.
//   device_placement` custom calls
//
// so the peeled value holds exactly the elements the lhs holds, in the same
// order.  The recognizer has already proven each member contracts its
// TRAILING dims in order (`MatchRoot`), i.e. its lhs flattens to [M, K]
// element-exactly; with M == 1 and one K across the group, two members whose
// peels land on the same Value therefore read the SAME [K] band, byte for
// byte.  The emit keeps the group ROOT's own lhs, so nothing about the
// peeled value has to be lowerable.
mlir::Value PeelActivationView(mlir::Value v) {
  for (int depth = 0; depth < 8; depth++) {
    mlir::Operation* d = v.getDefiningOp();
    if (d == nullptr) break;
    const std::string n = OpName(d);
    if (n == "stablehlo.reshape") {
      if (d->getNumOperands() != 1) break;
      v = d->getOperand(0);
      continue;
    }
    bool alias = n == "stablehlo.optimization_barrier" ||
                 n == "sdy.sharding_constraint" || n == "sdy.reshard";
    if (n == "stablehlo.custom_call") {
      auto t = d->getAttrOfType<mlir::StringAttr>("call_target_name");
      alias = t && (t.getValue() == "Sharding" ||
                    t.getValue() == "annotate_device_placement" ||
                    t.getValue() == "LayoutConstraint");
    }
    if (!alias || d->getNumResults() != d->getNumOperands()) break;
    // Arity-preserving: result i is operand i.
    auto res = mlir::dyn_cast<mlir::OpResult>(v);
    if (!res) break;
    v = d->getOperand(res.getResultNumber());
  }
  return v;
}

// `x` is the PEELED activation (above), not the lhs any member names: that
// is what makes a reshaped sibling a member instead of a group of one.  The
// lhs contracting dims are NOT part of the key -- they differ exactly when
// the reshape does -- because every member is already known to flatten to
// [M = 1, K] and the key pins K.
struct PackGroupKey {
  mlir::Block* block = nullptr;
  mlir::Value x, layer;
  int64_t L = 0, K = 0;
  mlir::Type elem;
  bool operator==(const PackGroupKey& o) const {
    return block == o.block && x == o.x && layer == o.layer && L == o.L &&
           K == o.K && elem == o.elem;
  }
};

// Group `plan->stacked`'s decode members by the activation they read and
// hand each group to `BuildStackedPacks` as one `StackedPackMatch`.  Pure
// structure: no buffer is touched here, and a group whose pack cannot be
// built gives its members back.
void GroupStackedPacks(RewritePlan* plan) {
  const PackPolicy policy = StackedPackPolicy();
  // The packs' own kill switch, and the budget they share with the
  // relayout; METALJAX_STACKED_RELAYOUT=0 is the RELAYOUT's switch alone,
  // so the two forms can be bisected apart.
  if (policy == PackPolicy::kOff || RelayoutBudgetBytes() <= 0) return;
  if (plan->stacked.size() < 2) return;

  struct Group {
    PackGroupKey key;
    std::vector<size_t> idx;  // positions in plan->stacked, block order
  };
  std::vector<Group> groups;
  llvm::DenseSet<size_t> grouped;

  for (size_t i = 0; i < plan->stacked.size(); i++) {
    StackedDotMatch& m = *plan->stacked[i];
    // DECODE only (the relayout's rule, for the relayout's reason), and
    // never a RELAID member: its own pack is materialized first and packing
    // it again would hold the same stack twice.
    if (m.M != 1 || m.relayout || m.origin_arg < 0 || m.sn != 1) continue;
    if (m.root == nullptr || m.root->getNumResults() != 1) continue;
    auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(m.root);
    if (!dot) continue;
    if (m.root->getBlock() == nullptr) continue;
    PackGroupKey key;
    try {
      key.elem = ElemOf(dot.getResult());
      const int64_t elem = ElemBytes(key.elem);
      if (m.K * m.N * elem > kMaxPackMemberBytes) {
        Debug(absl::StrCat("not a pack member (", m.name, ": the layer's ",
                           (m.K * m.N * elem) >> 20,
                           " MB matrix is bandwidth-bound already)"));
        continue;
      }
      const size_t nc =
          dot.getDotDimensionNumbers().getLhsContractingDimensions().size();
      const std::vector<int64_t> xshape = ShapeOf(m.x);
      if (xshape.size() < nc) continue;
      // The band this member reads IS its whole lhs: `MatchRoot` proved the
      // contraction takes the trailing dims in order, so numel(x) = M * K,
      // and M == 1 here.  Stated rather than assumed, because it is the
      // premise that makes a peeled sibling's band the same bytes.
      int64_t numel = 1;
      for (int64_t d : xshape) numel *= d;
      if (numel != m.K) {
        Debug(absl::StrCat("not a pack member (", m.name, ": its lhs holds ",
                           numel, " elements for a K of ", m.K, ")"));
        continue;
      }
    } catch (const Reject&) {
      // `Reject` is not a std::exception: both arms, or a shape question
      // this file answers by throwing would escape the analysis.
      continue;
    } catch (const std::exception&) {
      continue;
    }
    key.block = m.root->getBlock();
    key.x = PeelActivationView(m.x);
    key.layer = m.layer;
    key.L = m.L;
    key.K = m.K;
    bool placed = false;
    for (Group& g : groups) {
      if (!(g.key == key)) continue;
      // Pairwise distinct stacks: the same argument twice would be a CSE
      // miss upstream, and packing it buys nothing (metal_proj.cc).
      bool dup = false;
      for (size_t j : g.idx)
        dup = dup || plan->stacked[j]->origin_arg == m.origin_arg;
      if (dup) {
        Debug(absl::StrCat("not a pack member (", m.name,
                           ": the same stack twice in one group)"));
      } else {
        g.idx.push_back(i);
      }
      placed = true;
      break;
    }
    if (!placed) {
      Group g;
      g.key = key;
      g.idx.push_back(i);
      groups.push_back(std::move(g));
    }
  }

  for (Group& g : groups) {
    if (g.idx.size() < 2) continue;
    // metal_proj.cc's policy, verbatim: in a group of >= 3 with one member
    // at least twice as wide as the next, `auto` leaves the wide one alone
    // when the rest, packed, stay in their own gemv tile, and packs it too
    // when they would cross into a wider tile for nothing.
    if (policy == PackPolicy::kAuto && g.idx.size() >= 3) {
      size_t widest = 0;
      for (size_t i = 1; i < g.idx.size(); i++)
        if (plan->stacked[g.idx[i]]->N > plan->stacked[g.idx[widest]]->N)
          widest = i;
      int64_t second = 0, rest = 0;
      for (size_t i = 0; i < g.idx.size(); i++)
        if (i != widest) {
          second = std::max(second, plan->stacked[g.idx[i]]->N);
          rest += plan->stacked[g.idx[i]]->N;
        }
      if (plan->stacked[g.idx[widest]]->N >= 2 * second) {
        const GemvTile packed = TileFor(g.key.K, rest);
        bool leave = true;
        for (size_t i = 0; i < g.idx.size(); i++)
          if (i != widest &&
              TileFor(g.key.K, plan->stacked[g.idx[i]]->N) != packed)
            leave = false;
        if (leave) {
          Debug(absl::StrCat(
              "policy auto leaves the widest member alone (n=",
              plan->stacked[g.idx[widest]]->N, " >= 2 x ", second,
              "; the rest pack as n=", rest, " in their own gemv_t bn",
              packed.bn, "; METALJAX_STACKED_PACK=all packs it too)"));
          g.idx.erase(g.idx.begin() + static_cast<std::ptrdiff_t>(widest));
        }
      }
    }
    if (g.idx.size() < 2) continue;

    int64_t n_total = 0;
    for (size_t j : g.idx) n_total += plan->stacked[j]->N;
    auto pack = std::make_unique<StackedPackMatch>();
    std::vector<std::string> ns;
    for (size_t j : g.idx) ns.push_back(absl::StrCat(plan->stacked[j]->N));
    // The pack's lhs is the SURVIVING root's own `dot.getLhs()` -- its
    // spelling, not the peeled one the key carries, so `Slot(m.x)` resolves
    // to a value this block defines before the root; and the root is
    // `g.idx.front()`, which the policy branch above may just have changed.
    // The band takes its free dims from that same lhs; a member whose lhs is
    // spelled with a different rank gets the emit's per-member `reshape`
    // (`LowerStackedPack`), which its own `out_shape` already drives.
    StackedDotMatch& rep = *plan->stacked[g.idx.front()];
    auto rep_dot = mlir::cast<mlir::stablehlo::DotGeneralOp>(rep.root);
    const size_t rep_nc =
        rep_dot.getDotDimensionNumbers().getLhsContractingDimensions().size();
    pack->x = rep.x;
    pack->layer = g.key.layer;
    pack->L = g.key.L;
    pack->K = g.key.K;
    pack->M = 1;
    pack->n_total = n_total;
    pack->elem_bytes = ElemBytes(g.key.elem);
    pack->pack_bytes = g.key.L * g.key.K * n_total * pack->elem_bytes;
    try {
      std::vector<int64_t> lshape = ShapeOf(rep.x);
      pack->out_band.assign(
          lshape.begin(),
          lshape.end() - static_cast<std::ptrdiff_t>(rep_nc));
      pack->out_band.push_back(n_total);
    } catch (const Reject& e) {
      Debug(absl::StrCat("not a pack (", pack->name, ": ", e.why, ")"));
      continue;
    } catch (const std::exception& e) {
      Debug(absl::StrCat("not a pack (", pack->name, ": ", e.what(), ")"));
      continue;
    }
    pack->name = absl::StrCat("L", g.key.L, "m1k", g.key.K, "n",
                              absl::StrJoin(ns, "+"));
    for (size_t j : g.idx) {
      grouped.insert(j);
      // members[1..] are ABSORBED: their roots never dispatch, and their
      // results are bound by the pack's emit (the GDN/B6 precedent).
      if (j != g.idx.front()) pack->ops.push_back(plan->stacked[j]->root);
      for (mlir::Operation* o : plan->stacked[j]->ops)
        pack->ops.push_back(o);
      pack->members.push_back(std::move(plan->stacked[j]));
    }
    // How many members reached the group through a peeled view rather than
    // by naming the pack's own lhs: the one number that says whether the
    // peel is what built this pack.
    int peeled = 0;
    for (const auto& mem : pack->members)
      if (mem->x != pack->x) peeled++;
    Debug(absl::StrCat("matched a stacked pack (", pack->name, ", ",
                       pack->members.size(), " dots, ",
                       pack->pack_bytes >> 20, " MB, gemv_t bn",
                       TileFor(g.key.K, n_total).bn, ", policy ",
                       PackPolicyName(policy), ", ", peeled,
                       " member(s) joined through a peeled view)"));
    plan->stacked_pack.push_back(std::move(pack));
  }

  if (grouped.empty()) return;
  std::vector<std::unique_ptr<StackedDotMatch>> rest;
  for (size_t i = 0; i < plan->stacked.size(); i++)
    if (!grouped.contains(i)) rest.push_back(std::move(plan->stacked[i]));
  plan->stacked = std::move(rest);
}

}  // namespace

void AnalyzeStackedDot(mlir::func::FuncOp fn, RewritePlan* plan) {
  AnalyzeStackedDotImpl(fn, plan);
  // ...and, on top of the matches it kept, the sibling packs: several of
  // those dots read ONE activation against different per-layer stacks, and
  // the group becomes one gather_mm over the stacks concatenated on N.
  try {
    GroupStackedPacks(plan);
  } catch (const Reject& e) {
    Debug(absl::StrCat("pack grouping failed (", e.why,
                       "): the members keep their own dots"));
  } catch (const std::exception& e) {
    Debug(absl::StrCat("pack grouping failed (", e.what(),
                       "): the members keep their own dots"));
  }
}

absl::Status BuildStackedPacks(RewritePlan* plan, const PackContext& ctx) {
  bool any = !plan->stacked_pack.empty();
  for (const auto& m : plan->stacked) any = any || m->relayout;
  if (!any) return absl::OkStatus();
  if (ctx.args == nullptr) return absl::OkStatus();
  const int64_t budget = RelayoutBudgetBytes();
  absl::flat_hash_set<int> args(plan->pack_args.begin(),
                                plan->pack_args.end());
  std::vector<std::unique_ptr<StackedDotMatch>> kept;
  int64_t total = 0;
  int64_t relaid = 0;
  bool dropped = false;
  for (auto& m : plan->stacked) {
    if (!m->relayout) {
      kept.push_back(std::move(m));
      continue;
    }
    std::string why;
    try {
      if (m->origin_arg < 0 ||
          m->origin_arg >= static_cast<int>(ctx.args->size()))
        Bail("the stack's argument is out of range");
      if (total + m->relayout_bytes > budget)
        Bail(absl::StrCat("the relayout budget is spent (",
                          (total + m->relayout_bytes) >> 20, " MB > ",
                          budget >> 20, " MB, METALJAX_STACKED_RELAYOUT_MB)"));
      const mx::array& src = (*ctx.args)[m->origin_arg];
      std::vector<int> perm(m->relayout_perm.begin(), m->relayout_perm.end());
      if (static_cast<size_t>(src.ndim()) != perm.size())
        Bail("the argument's rank disagrees with the stack");
      // The no-panic contract: the pack is device memory held for the
      // executable's life, admitted like a transfer of its size.
      governor_admit(m->relayout_bytes, MemWhere::kExecute);
      mx::array pk = mx::contiguous(mx::transpose(src, perm));
      mx::eval(pk);
      m->pack_slot = static_cast<int>(plan->packs.size());
      plan->packs.push_back(pk);
      args.insert(m->origin_arg);
      total += m->relayout_bytes;
      relaid++;
      Debug(absl::StrCat(
          "relaid ", m->name, " (", m->relayout_bytes >> 20,
          " MB once, in place of a ",
          (m->relayout_bytes / std::max<int64_t>(m->L, 1)) >> 20,
          " MB slice per layer)"));
      kept.push_back(std::move(m));
    } catch (const Reject& e) {
      why = e.why;
    } catch (const std::exception& e) {
      why = e.what();
    }
    if (!why.empty()) {
      Debug(absl::StrCat("weights stay sliced (", m->name,
                         ": relayout declined, ", why, ")"));
      dropped = true;
    }
  }
  plan->stacked = std::move(kept);

  // ...and the sibling PACKS, out of the SAME budget: each group's members'
  // [L, K, N] views, concatenated on the free axis into one contiguous
  // [L, K, n_total] the emit reads with a single gather_mm.  The relayout
  // runs first on purpose -- it is the measured 0.69 ms/token form, and a
  // group is worth 0.35 -- so under budget pressure the packs are what
  // declines.  A declined group gives its members back to `plan->stacked`
  // and they keep their own dots (the recognizer file rule).
  std::vector<std::unique_ptr<StackedPackMatch>> kept_packs;
  int64_t packed = 0;
  for (auto& g : plan->stacked_pack) {
    std::string why;
    try {
      if (g->members.size() < 2) Bail("a group that lost its siblings");
      if (total + g->pack_bytes > budget)
        Bail(absl::StrCat("the relayout budget is spent (",
                          (total + g->pack_bytes) >> 20, " MB > ",
                          budget >> 20, " MB, METALJAX_STACKED_RELAYOUT_MB)"));
      std::vector<mx::array> views;
      views.reserve(g->members.size());
      for (const auto& mem : g->members) {
        if (mem->origin_arg < 0 ||
            mem->origin_arg >= static_cast<int>(ctx.args->size()))
          Bail("a member's stack argument is out of range");
        const mx::array& src = (*ctx.args)[mem->origin_arg];
        // The analysis proved these strides against the ORIGINAL buffer's
        // row-major layout, and @main's arguments are contiguous, so this is
        // the same view the member's own emit would have gathered from.
        views.push_back(mx::as_strided(
            src,
            mx::Shape{static_cast<mx::ShapeElem>(g->L),
                      static_cast<mx::ShapeElem>(g->K),
                      static_cast<mx::ShapeElem>(mem->N)},
            mx::Strides{mem->sl, mem->sk, mem->sn}, /*offset=*/0));
      }
      // The no-panic contract: device memory held for the executable's life,
      // admitted like a transfer of its size.
      governor_admit(g->pack_bytes, MemWhere::kExecute);
      mx::array pk = mx::contiguous(mx::concatenate(views, 2));
      mx::eval(pk);
      g->pack_slot = static_cast<int>(plan->packs.size());
      plan->packs.push_back(pk);
      for (const auto& mem : g->members) args.insert(mem->origin_arg);
      total += g->pack_bytes;
      packed++;
      Debug(absl::StrCat("packed ", g->name, " (", g->pack_bytes >> 20,
                         " MB once, ", g->members.size(),
                         " dots become one)"));
      kept_packs.push_back(std::move(g));
      continue;
    } catch (const Reject& e) {
      why = e.why;
    } catch (const std::exception& e) {
      why = e.what();
    }
    Debug(absl::StrCat("the members keep their own dots (", g->name,
                       ": the pack declined, ", why, ")"));
    for (auto& mem : g->members) plan->stacked.push_back(std::move(mem));
    dropped = true;
  }
  plan->stacked_pack = std::move(kept_packs);

  plan->pack_args.assign(args.begin(), args.end());
  std::sort(plan->pack_args.begin(), plan->pack_args.end());
  if (dropped) plan->rebuild();
  if (relaid > 0)
    Debug(absl::StrCat(relaid, " stack(s) relaid, ", total >> 20,
                       " MB held for the executable"));
  if (packed > 0)
    Debug(absl::StrCat(packed, " sibling pack(s) built, ", total >> 20,
                       " MB held for the executable in all"));
  return absl::OkStatus();
}

}  // namespace metaljax
