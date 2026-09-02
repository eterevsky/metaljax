/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The RMS-norm recognizer: jax's spelled-out root-mean-square norm, rewritten
into MLX's fused `fast::rms_norm`.

Every transformer in the model table computes the same function -- `x *
rsqrt(mean(x^2) + eps) * w` -- and every library spells it differently.  P30
fingerprinted ONE spelling, maxtext/DeepSeek's (rows 10/11/14/15), and the
dense band matched none of it (captured lowerings and the numbers below:
`~/.cache/metaljax-bench/logs/dense-band-norm/`).  What actually varies:

  * the square: `multiply(x, x)`, or keras' `power(x, 2)` -- the exponent
    itself spelled `broadcast(convert(2 : i32))`;
  * where the f32 upcast sits: before the square (maxtext, keras) or on the
    square's RESULT (gemma, which squares in bf16);
  * the dtype of the eps add and the rsqrt: f32 (maxtext, keras) or the
    model dtype, the mean having been rounded back down first (gemma);
  * the weight apply: a batching-only `dot_general` plus its transpose
    (maxtext), or `broadcast_in_dim` and a multiply (gemma, keras -- keras
    spells the head-dim norm with TWO broadcasts);
  * an additive constant on the weight: `w + 0` (maxtext) or `1 + w`
    (gemma 2/3).  Both fold into one [N]-wide add in the emit;
  * whether there is a weight at all: gemma's value/router/qk norms are
    `RMSNorm(with_scale=False)`, and MLX's weight argument is optional;
  * where the result rounds back to the model dtype: keras applies the
    weight in f32 and converts AFTER, so the root op is that convert.

flax's NNX modules (bonsai's Qwen3 and ViT, and any other nnx-based model --
`notes/bonsai-coverage-2026-09-01.md`) add three more, all of them from the
same design decision: flax emits ONE normalization, LayerNorm, and spells
RMSNorm as that with the mean pinned to a literal zero.

  * `x - broadcast(0.0)` stands between x and the normalize multiply.  The
    hop is VALUE-EXACT, not merely close: IEEE-754 subtraction of +0.0 is
    the identity on every finite value, on both infinities and on NaN, and
    on the zeros `(+0.0) - (+0.0) = +0.0` while `(-0.0) - (+0.0) = -0.0`
    under round-to-nearest (the only rounding mode that would disagree is
    round-toward-negative, which neither XLA nor Metal uses).  The sign of
    the subtrahend is the whole argument, so the matcher takes POSITIVE
    zero only: `x - (-0.0)` is `x + 0.0`, which turns -0.0 into +0.0;
  * the scale is associated the other way.  maxtext/gemma/keras compute
    `x * rsqrt` and apply the weight to that; flax multiplies `rsqrt * w`
    FIRST, at full rank, and scales x by the product.  Same function, a
    different tree -- and the inner multiply reads as a bare normalize, so
    what tells the two apart is whether the value being scaled is an
    activation or a broadcast of the [N] weight;
  * the rank comes back differently.  maxtext broadcasts the SUM to
    `[.., 1]` and divides there; flax divides at the reduced rank and lifts
    the mean with a `reshape`.  A `reshape` also sits under the weight's
    broadcast (`[N] -> [1, .., N]`), and under the zero itself.

So the matcher reads BACKWARDS from a root and treats `convert` as a
transparent dtype hop, rather than pinning one op sequence.  Four roots are
tried: a transpose (the maxtext dot form), a multiply (the broadcast weight
apply, the scale-first product, or the bare normalize when there is no
weight), a convert (keras' downcast, or flax's, wrapping either), and an add
(a LayerNorm's bias).

GENUINE LayerNorm -- flax's `nnx.LayerNorm`, which is what every ViT in the
model table normalizes with -- is the same graph with a real mean, and it
rewrites into `fast::layer_norm(x, w, b, eps)`.  It needs three more things
recognized, and all three are what flax emits: the variance as
`max(0, E[x^2] - E[x]^2)`, the subtracted mean proved to BE `E[x]` (the same
divide, reached through the reshape and the broadcast), and an optional [N]
bias added at the root.  Its numeric class is NOT the rms one: MLX's kernel
takes a second pass and accumulates `sum((x - mean)^2)`, where the chain
subtracts two moments that are nearly equal.  The fused side is therefore
the more accurate one, by a margin that grows with `mean/stddev` -- at
`mean = 0` (what a trained LayerNorm's input looks like) the two agree to
f32 rounding, and the fixtures in `execute_test.py` measure it.

Three of these per decode layer put ~39 tape entries per layer on a path
where every entry costs ~2 us of build/schedule/launch (the row-10 op-count
campaign, notes/row10-decode-floor-2026-08-29.md).

The rewrite emits ONE kRmsNorm (runtime/emits.cc): `fast::rms_norm(x, w,
eps)`.  Its kernel (mlx rms_norm.metal `rms_single_row`) accumulates in f32,
rounds `x * rsqrt` to the output dtype, and multiplies the weight in that
dtype.  Measured against each literal chain at the real feature widths
(2560/4096/5120, bf16 activations):

  * keras   -- max 1 bf16 ULP, ~77 % bit-identical.  The chain keeps the
    weight multiply in f32 and rounds once; the kernel rounds before it.
  * gemma   -- max 2 bf16 ULP, ~75 % bit-identical.  The chain rounds the
    MEAN to bf16 and takes its rsqrt there, so the fused kernel is the more
    accurate side; the difference is the chain's own precision loss.

Both are the documented fused-kernel class, and 2 ULP on rows 1/2 is a
token-stream tie-flip risk that belongs in the gate report.

A half-matched pattern lowers as ORDINARY ops (`Bail`), and a match whose
intermediates escape is declined whole -- the fused op computes everything
internally, so a partial absorption cannot stand.  Candidates are kept
LARGEST first: the bare normalize multiply is a sub-match of the weight
apply built on top of it, and taking the small one would strand the weight.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <algorithm>
#include <cmath>
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

// A splat constant, float OR integer: keras spells the square's exponent as
// `convert(constant dense<2> : tensor<i32>)`.
std::optional<double> SplatNumberOf(mlir::Value v) {
  auto cst =
      mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return std::nullopt;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return std::nullopt;
  if (mlir::isa<mlir::FloatType>(dense.getElementType()))
    return dense.getSplatValue<mlir::APFloat>().convertToDouble();
  if (mlir::isa<mlir::IntegerType>(dense.getElementType()))
    return static_cast<double>(
        dense.getSplatValue<mlir::APInt>().getSExtValue());
  return std::nullopt;
}

// The same splat through any chain of broadcasts, reshapes and dtype
// converts.  A reshape of a splat is a splat of the same number, and flax
// reaches its zero mean through broadcast -> reshape -> broadcast.  The
// whole chain hangs off a constant, so absorbing it on success is safe; on
// failure nothing is recorded.
std::optional<double> SplatThroughCasts(mlir::Value v,
                                        std::vector<mlir::Operation*>* ops) {
  std::vector<mlir::Operation*> walked;
  for (int guard = 0; guard < 8; guard++) {
    if (auto d = SplatNumberOf(v)) {
      if (ops != nullptr) ops->insert(ops->end(), walked.begin(), walked.end());
      return d;
    }
    mlir::Operation* def = v.getDefiningOp();
    if (!mlir::isa_and_nonnull<mlir::stablehlo::BroadcastInDimOp>(def) &&
        !mlir::isa_and_nonnull<mlir::stablehlo::ReshapeOp>(def) &&
        !mlir::isa_and_nonnull<mlir::stablehlo::ConvertOp>(def))
      return std::nullopt;
    walked.push_back(def);
    v = def->getOperand(0);
  }
  return std::nullopt;
}

// A POSITIVE-zero splat, reached the same way.  `x - (+0.0)` is x for every
// x -- the identity on finite values, on both infinities and on NaN, and on
// the zeros too under round-to-nearest ((+0.0)-(+0.0) = +0.0,
// (-0.0)-(+0.0) = -0.0).  `x - (-0.0)` is `x + 0.0`, which turns -0.0 into
// +0.0, so the sign is checked, not just the value.
bool IsPositiveZeroSplat(mlir::Value v, std::vector<mlir::Operation*>* ops) {
  std::vector<mlir::Operation*> walked;
  auto d = SplatThroughCasts(v, &walked);
  if (!d.has_value() || *d != 0.0 || std::signbit(*d)) return false;
  if (ops != nullptr) ops->insert(ops->end(), walked.begin(), walked.end());
  return true;
}

// Is `op` part of a subtree that computes a constant -- the constant itself,
// or the broadcasts and converts standing between it and its use?
//
// Such an op is SHAREABLE between matches and must never be absorbed.  XLA
// hoists and CSEs these before the plugin sees the module, so in a 48-layer
// model one `broadcast(eps)` and one `broadcast(N)` serve every norm at that
// width: claiming them for the first match would collide the second out of
// the plan AND leave the first with an intermediate that escapes into it, so
// both would decline.  Left unabsorbed they lower as the two cheap ops they
// are, and the DCE pass drops whichever the rewrite made dead.
bool IsConstSubtree(mlir::Operation* op) {
  for (int guard = 0; guard < 8 && op != nullptr; guard++) {
    if (mlir::isa<mlir::stablehlo::ConstantOp>(op)) return true;
    if (!mlir::isa<mlir::stablehlo::BroadcastInDimOp>(op) &&
        !mlir::isa<mlir::stablehlo::ReshapeOp>(op) &&
        !mlir::isa<mlir::stablehlo::ConvertOp>(op))
      return false;
    op = op->getOperand(0).getDefiningOp();
  }
  return false;
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

// A widening float convert is value-exact, and the fused kernel accumulates
// in f32 itself, so the matcher hands it the value UNDERNEATH the upcast.
// Only an upcast to f32: that is the accumulation dtype every one of these
// chains widens to, and a convert anywhere else in the chain is a real
// rounding step this rewrite must not walk past.
mlir::Value PeelUpcast(mlir::Value v, std::vector<mlir::Operation*>* ops) {
  auto cv =
      mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(v.getDefiningOp());
  if (!cv) return v;
  if (!mlir::isa<mlir::Float32Type>(ElemOf(cv.getResult())))
    Bail("the upcast is not to f32");
  ops->push_back(cv);
  return cv.getOperand();
}

// flax reduces the feature axis away and lifts the rank back with a
// `reshape`; maxtext never loses it (it broadcasts the SUM instead).  This
// is the only reshape worth walking here: `[lead..] -> [lead.., 1]`.
mlir::Value PeelKeepReshape(mlir::Value v,
                            const std::vector<int64_t>& out_shape,
                            std::vector<mlir::Operation*>* ops) {
  std::vector<int64_t> lead(out_shape.begin(), out_shape.end() - 1);
  std::vector<int64_t> keep = lead;
  keep.push_back(1);
  if (ShapeOf(v) != keep) return v;
  auto rs =
      mlir::dyn_cast_or_null<mlir::stablehlo::ReshapeOp>(v.getDefiningOp());
  if (!rs || ShapeOf(rs.getOperand()) != lead) return v;
  ops->push_back(rs);
  return rs.getOperand();
}

// The mean subtraction flax puts in front of every normalize.  A splat
// POSITIVE zero is transparent (the header's IEEE argument) and vanishes
// into the absorbed ops; anything else is a genuine LayerNorm mean, handed
// back through `mean` for the caller to prove.
mlir::Value PeelMeanSubtract(mlir::Value v, std::vector<mlir::Operation*>* ops,
                             mlir::Value* mean) {
  auto sub =
      mlir::dyn_cast_or_null<mlir::stablehlo::SubtractOp>(v.getDefiningOp());
  if (!sub) return v;
  std::vector<mlir::Operation*> zops;
  if (IsPositiveZeroSplat(sub.getRhs(), &zops)) {
    ops->push_back(sub);
    ops->insert(ops->end(), zops.begin(), zops.end());
    return sub.getLhs();
  }
  ops->push_back(sub);
  *mean = sub.getRhs();
  return sub.getLhs();
}

// Does `v` reach an [N] vector through broadcasts, reshapes and converts?
// That is what a learned scale (or a bias) looks like at full rank, and it
// is what tells the scale-first spelling's `rsqrt * w` apart from a bare
// normalize `x * rsqrt`.  A predicate, so it never bails: a norm whose
// normed value really were a broadcast [N] row would simply not fuse.
bool IsFeatureBroadcast(mlir::Value v, int64_t N) {
  if (!mlir::isa_and_nonnull<mlir::stablehlo::BroadcastInDimOp>(
          v.getDefiningOp()))
    return false;
  for (int guard = 0; guard < 8; guard++) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t || !t.hasStaticShape()) return false;
    if (t.getRank() == 1) return t.getShape()[0] == N;
    mlir::Operation* def = v.getDefiningOp();
    if (!mlir::isa_and_nonnull<mlir::stablehlo::BroadcastInDimOp>(def) &&
        !mlir::isa_and_nonnull<mlir::stablehlo::ReshapeOp>(def) &&
        !mlir::isa_and_nonnull<mlir::stablehlo::ConvertOp>(def))
      return false;
    v = def->getOperand(0);
  }
  return false;
}

// The `broadcast_in_dim(rsqrt(..))` operand of a multiply, with the other
// one handed back.  Null when neither side is one.
mlir::Value RsqrtSide(mlir::stablehlo::MulOp mul, mlir::Value* other) {
  for (bool swap : {false, true}) {
    mlir::Value a = swap ? mul.getLhs() : mul.getRhs();
    mlir::Value b = swap ? mul.getRhs() : mul.getLhs();
    auto bc =
        mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
            a.getDefiningOp());
    if (bc && mlir::isa_and_nonnull<mlir::stablehlo::RsqrtOp>(
                  bc.getOperand().getDefiningOp())) {
      if (other != nullptr) *other = b;
      return a;
    }
  }
  return nullptr;
}

// Peel `v` back to the [N] learned scale.  `dim` says which axis of `v`
// currently carries the feature axis (R-1 for a value at the norm's own
// shape, 0 for the dot form's already-[N] operand).  Broadcasts and dtype
// converts are transparent; splat adds accumulate into `offset`, which the
// emit applies once, [N] wide, instead of at full rank.
mlir::Value PeelWeight(mlir::Value v, int64_t N, int64_t dim,
                       std::vector<mlir::Operation*>* ops, double* offset) {
  std::optional<mlir::Type> add_elem;
  for (int guard = 0; guard < 12; guard++) {
    std::vector<int64_t> shape = ShapeOf(v);
    mlir::Operation* def = v.getDefiningOp();
    // A WIDENING dtype hop is transparent at any rank: keras stores the
    // weight in the model dtype and applies it in f32, and the fused kernel
    // wants the stored one.  Only widening -- a narrowing convert is a real
    // rounding step, and the weight it produces is the value the chain
    // multiplies by.
    if (auto c = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(def)) {
      auto from = mlir::dyn_cast<mlir::FloatType>(ElemOf(c.getOperand()));
      auto to = mlir::dyn_cast<mlir::FloatType>(ElemOf(c.getResult()));
      if (!from || !to || from.getWidth() > to.getWidth())
        Bail("the weight convert is not a widening hop");
      ops->push_back(def);
      v = c.getOperand();
      continue;
    }
    if (shape.size() == 1 && dim == 0) {
      if (shape[0] != N) Bail("the weight is not [N]");
      // Folding a NONZERO offset down to the weight's own dtype has to be
      // the same arithmetic the chain did (gemma 2/3 adds its 1 in bf16,
      // alongside a bf16 weight).  An offset formed one dtype up would
      // round differently, so decline rather than approximate it.
      if (offset != nullptr && *offset != 0.0 && add_elem.has_value() &&
          *add_elem != ElemOf(v))
        Bail("the weight offset is formed in a wider dtype");
      return v;
    }
    if (def == nullptr) Bail("the weight apply reaches a block argument");
    if (auto b = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
      auto bd = b.getBroadcastDimensions();
      std::vector<int64_t> os = ShapeOf(b.getOperand());
      if (bd.size() != os.size()) Bail("the weight broadcast is malformed");
      int64_t src = -1;
      for (size_t i = 0; i < bd.size(); i++)
        if (bd[i] == dim) src = static_cast<int64_t>(i);
      if (src < 0) Bail("the weight broadcast does not carry the feature axis");
      if (os[src] != N)
        Bail("the weight broadcast stretches the feature axis");
      for (size_t i = 0; i < os.size(); i++)
        if (static_cast<int64_t>(i) != src && os[i] != 1)
          Bail("the weight broadcast carries more than the feature axis");
      ops->push_back(def);
      v = b.getOperand();
      dim = src;
      continue;
    }
    if (auto rs = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
      // flax stores the scale as [N] and reshapes it to [1, .., N] before
      // the broadcast.  Transparent only when the reshape does nothing but
      // add or drop size-1 axes -- by this point every axis of `v` except
      // the feature one is already 1, so the source must hold exactly one
      // non-unit axis and it must be the feature.
      std::vector<int64_t> os = ShapeOf(rs.getOperand());
      if (shape[dim] != N || N == 1)
        Bail("the weight reshape does not carry the feature axis");
      for (size_t i = 0; i < shape.size(); i++)
        if (static_cast<int64_t>(i) != dim && shape[i] != 1)
          Bail("the weight reshape carries more than the feature axis");
      int64_t src = -1;
      for (size_t i = 0; i < os.size(); i++) {
        if (os[i] == 1) continue;
        if (os[i] != N || src >= 0) Bail("the weight reshape is not a unit-dim reshape");
        src = static_cast<int64_t>(i);
      }
      if (src < 0) Bail("the weight reshape drops the feature axis");
      ops->push_back(def);
      v = rs.getOperand();
      dim = src;
      continue;
    }
    if (auto a = mlir::dyn_cast<mlir::stablehlo::AddOp>(def)) {
      bool took = false;
      for (bool swap : {false, true}) {
        mlir::Value keep = swap ? a.getRhs() : a.getLhs();
        mlir::Value cst = swap ? a.getLhs() : a.getRhs();
        std::vector<mlir::Operation*> cops;
        auto d = SplatThroughCasts(cst, &cops);
        if (!d.has_value()) continue;
        if (offset != nullptr) *offset += *d;
        add_elem = ElemOf(a.getResult());
        ops->push_back(def);
        ops->insert(ops->end(), cops.begin(), cops.end());
        v = keep;
        took = true;
        break;
      }
      if (took) continue;
      Bail("the weight add is not a splat offset");
    }
    Bail(absl::StrCat("the weight apply walks a ", OpName(def)));
  }
  Bail("the weight chain is too deep");
}

// `divide(broadcast?(reduce_add(v)), N)`: the mean over the feature axis, in
// either of the shapes the libraries spell it -- maxtext broadcasts the SUM
// back to `[.., 1]` and divides there, flax divides at the reduced rank and
// lifts the result with a reshape.  Returns the value that was SUMMED.
mlir::Value MatchMean(mlir::Value m, const std::vector<int64_t>& out_shape,
                      std::vector<mlir::Operation*>& ops) {
  const size_t R = out_shape.size();
  const int64_t N = out_shape[R - 1];
  const std::vector<int64_t> lead(out_shape.begin(), out_shape.end() - 1);
  std::vector<int64_t> lead_dims;
  for (size_t i = 0; i + 1 < R; i++)
    lead_dims.push_back(static_cast<int64_t>(i));

  m = PeelKeepReshape(m, out_shape, &ops);
  auto div =
      mlir::dyn_cast_or_null<mlir::stablehlo::DivOp>(m.getDefiningOp());
  if (!div) Bail("the mean is not a divide");
  ops.push_back(div);
  std::vector<mlir::Operation*> nops;
  auto divisor = SplatThroughCasts(div.getRhs(), &nops);
  if (!divisor.has_value() || *divisor != static_cast<double>(N))
    Bail("the divisor is not the feature size");
  ops.insert(ops.end(), nops.begin(), nops.end());

  mlir::Value sum = div.getLhs();
  if (auto b4 = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
          sum.getDefiningOp())) {
    auto bd = b4.getBroadcastDimensions();
    if (std::vector<int64_t>(bd.begin(), bd.end()) != lead_dims)
      Bail("the sum broadcast has the wrong dims");
    ops.push_back(b4);
    sum = b4.getOperand();
  }
  if (ShapeOf(sum) != lead) Bail("the sum has the wrong shape");
  auto red =
      mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(sum.getDefiningOp());
  if (!red || red.getNumOperands() != 2 || red->getNumResults() != 1)
    Bail("the sum is not a single-input reduce");
  auto dims = red.getDimensions();
  if (dims.size() != 1 || dims[0] != static_cast<int64_t>(R) - 1)
    Bail("the sum reduces the wrong axis");
  mlir::Block& body = red.getBody().front();
  if (body.getOperations().size() != 2 ||
      OpName(&body.front()) != "stablehlo.add")
    Bail("the sum body is not an add");
  if (!IsZeroInit(red.getInitValues()[0])) Bail("the sum init is not zero");
  ops.push_back(red);
  return red.getInputs()[0];
}

// `sqv` is the square of `x`: `x * x` (every library but keras) or
// `pow(x, 2)` (keras).  The comparison runs UNDER the widening upcasts, not
// on SSA identity: flax's bf16 norms square one `convert(x)` and subtract
// the mean from a second, textually identical one that XLA did not CSE.
void MatchSquareOf(mlir::Value sqv, mlir::Value x,
                   std::vector<mlir::Operation*>& ops) {
  mlir::Operation* sq = DefOf(sqv, "the summed square");
  if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(sq)) {
    if (mul.getLhs() != mul.getRhs() || PeelUpcast(mul.getLhs(), &ops) != x)
      Bail("the sum is not over the square of the normed value");
  } else if (auto pw = mlir::dyn_cast<mlir::stablehlo::PowOp>(sq)) {
    // keras: `ops.power(x, 2)`.
    if (PeelUpcast(pw.getLhs(), &ops) != x)
      Bail("the sum is not over a power of the normed value");
    std::vector<mlir::Operation*> pops;
    auto e = SplatThroughCasts(pw.getRhs(), &pops);
    if (!e.has_value() || *e != 2.0) Bail("the power is not a square");
    ops.insert(ops.end(), pops.begin(), pops.end());
  } else {
    Bail(absl::StrCat("the summed value is a ", OpName(sq)));
  }
  ops.push_back(sq);
}

// What a matched chain yields.
struct NormChain {
  mlir::Value x;       // the value the fused kernel reads, upcast peeled off
  double eps = 0.0;
  bool layer = false;  // a real mean was subtracted -> fast::layer_norm
};

// The normalize itself, given its two halves: the value being scaled and the
// `broadcast_in_dim(rsqrt(..))` scaling it.  Split out from the multiply
// entry below because flax's spelling has no such multiply -- it scales x by
// `rsqrt * w`, so the two halves come from two different ops.
NormChain MatchNormalizedParts(mlir::Value xn, mlir::Value b10,
                               const std::vector<int64_t>& out_shape,
                               std::vector<mlir::Operation*>& ops) {
  const size_t R = out_shape.size();
  const int64_t N = out_shape[R - 1];
  NormChain ch;

  // In the scale-first spelling `rsqrt * w` is itself a multiply with a
  // broadcast rsqrt on one side, so it reads as a bare normalize too.  It is
  // not one: what it scales is a broadcast of the [N] weight.
  if (IsFeatureBroadcast(xn, N))
    Bail("the normalized value is a weight broadcast");

  auto b10op =
      mlir::cast<mlir::stablehlo::BroadcastInDimOp>(b10.getDefiningOp());
  ops.push_back(b10op);
  auto rs = mlir::cast<mlir::stablehlo::RsqrtOp>(
      b10op.getOperand().getDefiningOp());
  ops.push_back(rs);

  // rsqrt(convert?(variance) + eps).
  auto add6 = mlir::dyn_cast_or_null<mlir::stablehlo::AddOp>(
      rs.getOperand().getDefiningOp());
  if (!add6) Bail("no eps add");
  ops.push_back(add6);
  mlir::Value var_v;
  std::optional<double> eps;
  for (bool swap : {false, true}) {
    mlir::Value a = swap ? add6.getRhs() : add6.getLhs();
    mlir::Value b = swap ? add6.getLhs() : add6.getRhs();
    std::vector<mlir::Operation*> eops;
    auto e = SplatThroughCasts(b, &eops);
    if (!e.has_value()) continue;
    eps = e;
    var_v = a;
    ops.insert(ops.end(), eops.begin(), eops.end());
    break;
  }
  if (!eps.has_value()) Bail("no eps splat");
  // gemma rounds the mean back to the model dtype before the eps add.
  if (auto cv = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(
          var_v.getDefiningOp())) {
    ops.push_back(cv);
    var_v = cv.getOperand();
  }

  // The normed value: the mean subtraction, then the upcast under it.  The
  // emit hands the ORIGINAL x to the fused kernel.
  mlir::Value mean_sub;
  mlir::Value x = PeelUpcast(PeelMeanSubtract(xn, &ops, &mean_sub), &ops);
  if (ShapeOf(x) != out_shape) Bail("the input shape disagrees");

  var_v = PeelKeepReshape(var_v, out_shape, &ops);
  if (!mean_sub) {
    // RMS: the variance IS the mean of the squares.
    MatchSquareOf(PeelUpcast(MatchMean(var_v, out_shape, ops), &ops), x, ops);
  } else {
    // LayerNorm, as flax computes it: `max(0, E[x^2] - E[x]^2)`.
    ch.layer = true;
    auto mx = mlir::dyn_cast_or_null<mlir::stablehlo::MaxOp>(
        var_v.getDefiningOp());
    if (!mx) Bail("the variance is not a clamped moment difference");
    ops.push_back(mx);
    mlir::Value diff;
    for (bool swap : {false, true}) {
      mlir::Value a = swap ? mx.getRhs() : mx.getLhs();
      mlir::Value b = swap ? mx.getLhs() : mx.getRhs();
      std::vector<mlir::Operation*> zops;
      if (!IsPositiveZeroSplat(b, &zops)) continue;
      diff = a;
      ops.insert(ops.end(), zops.begin(), zops.end());
      break;
    }
    if (!diff) Bail("the variance clamp is not against zero");
    auto sub = mlir::dyn_cast_or_null<mlir::stablehlo::SubtractOp>(
        diff.getDefiningOp());
    if (!sub) Bail("the variance is not a moment difference");
    ops.push_back(sub);
    // E[x^2].
    MatchSquareOf(PeelUpcast(MatchMean(sub.getLhs(), out_shape, ops), &ops), x,
                  ops);
    // E[x]^2 -- and the mean squared here has to be the very mean the normed
    // value subtracted, or this is some other function of x.
    auto msq = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(
        sub.getRhs().getDefiningOp());
    if (!msq || msq.getLhs() != msq.getRhs())
      Bail("the second moment is not the square of a mean");
    ops.push_back(msq);
    mlir::Value mu = msq.getLhs();
    if (PeelUpcast(MatchMean(mu, out_shape, ops), &ops) != x)
      Bail("the mean is not over the normed input");
    auto mb = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
        mean_sub.getDefiningOp());
    if (!mb) Bail("the subtracted mean is not broadcast");
    std::vector<int64_t> ident;
    for (size_t i = 0; i < R; i++) ident.push_back(static_cast<int64_t>(i));
    auto bd = mb.getBroadcastDimensions();
    if (std::vector<int64_t>(bd.begin(), bd.end()) != ident)
      Bail("the subtracted mean broadcast has the wrong dims");
    ops.push_back(mb);
    if (PeelKeepReshape(mb.getOperand(), out_shape, &ops) != mu)
      Bail("the subtracted mean is not the mean");
  }

  ch.x = x;
  ch.eps = *eps;
  return ch;
}

// `y = x * rsqrt(var + eps)`, read backwards from `y`.
NormChain MatchNormalized(mlir::Value y,
                          const std::vector<int64_t>& out_shape,
                          std::vector<mlir::Operation*>& ops) {
  // maxtext rounds the normalized value back down before the weight dot.
  if (auto cv =
          mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(y.getDefiningOp())) {
    ops.push_back(cv);
    y = cv.getOperand();
  }
  auto mul2 = mlir::dyn_cast<mlir::stablehlo::MulOp>(
      DefOf(y, "the normalized value"));
  if (!mul2) Bail("the normalized value is not a multiply");
  ops.push_back(mul2);

  mlir::Value xn;
  mlir::Value b10 = RsqrtSide(mul2, &xn);
  if (!b10) Bail("the normalized value has no rsqrt side");
  return MatchNormalizedParts(xn, b10, out_shape, ops);
}

// Is `v` the normalize multiply -- one side a broadcast of an rsqrt?  This
// is what tells the weight apply's two operands apart.
bool IsNormalizeMul(mlir::Value v) {
  mlir::Operation* def = v.getDefiningOp();
  if (auto cv = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(def))
    def = cv.getOperand().getDefiningOp();
  auto mul = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(def);
  return mul && RsqrtSide(mul, nullptr) != mlir::Value();
}

std::unique_ptr<RmsNormMatch> MatchRoot(mlir::Operation* op) {
  auto m = std::make_unique<RmsNormMatch>();
  m->root = op;
  std::vector<mlir::Operation*>& ops = m->ops;

  const std::vector<int64_t> out_shape = ShapeOf(op->getResult(0));
  const size_t R = out_shape.size();
  if (R < 1) Bail("a rank-0 norm");
  const int64_t N = out_shape[R - 1];

  mlir::Value y;                 // the normalized value the weight applies to
  mlir::Value xs, rq;            // the scale-first form's two halves instead
  mlir::Value w;                 // null when the norm has no learned scale
  mlir::Value bias;              // null unless a LayerNorm carries one
  double offset = 0.0;
  std::string form;

  if (auto tr = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
    // maxtext/DeepSeek: the weight as a batching-only dot, transposed back.
    // perm must be [1, 2, .., R-1, 0].
    auto perm = tr.getPermutation();
    if (perm.size() != R) Bail("rank mismatch");
    for (size_t i = 0; i + 1 < R; i++)
      if (perm[i] != static_cast<int64_t>(i) + 1) Bail("not the weight-apply");
    if (perm[R - 1] != 0) Bail("not the weight-apply");

    auto dot = mlir::dyn_cast_or_null<mlir::stablehlo::DotGeneralOp>(
        DefOf(tr.getOperand(), "the weight apply"));
    if (!dot) Bail("the weight apply is not a dot");
    ops.push_back(dot);
    auto dn = dot.getDotDimensionNumbers();
    auto eq_dims = [](llvm::ArrayRef<int64_t> a, const std::vector<int64_t>& b) {
      return std::vector<int64_t>(a.begin(), a.end()) == b;
    };
    if (!eq_dims(dn.getLhsBatchingDimensions(), {0}) ||
        !eq_dims(dn.getRhsBatchingDimensions(), {static_cast<int64_t>(R) - 1}) ||
        !dn.getLhsContractingDimensions().empty() ||
        !dn.getRhsContractingDimensions().empty())
      Bail("the weight apply is not the batching multiply");
    w = PeelWeight(dot.getLhs(), N, /*dim=*/0, &ops, &offset);
    y = dot.getRhs();
    form = "dot";
  } else {
    // The broadcast-multiply families.  keras and flax apply the weight in
    // f32 and round after, so the root can be that convert; a LayerNorm's
    // bias makes it an add over the whole thing.
    mlir::Operation* mul_op = op;
    if (auto ad = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
      mlir::Operation* inner = nullptr;
      for (bool swap : {false, true}) {
        mlir::Value a = swap ? ad.getRhs() : ad.getLhs();
        mlir::Value b = swap ? ad.getLhs() : ad.getRhs();
        if (!mlir::isa_and_nonnull<mlir::stablehlo::MulOp>(a.getDefiningOp()))
          continue;
        if (!IsFeatureBroadcast(b, N)) continue;
        inner = a.getDefiningOp();
        double boff = 0.0;
        bias = PeelWeight(b, N, static_cast<int64_t>(R) - 1, &ops, &boff);
        // The emit forms `w + offset` for the weight; a bias offset would
        // need a second one, and no library spells it.
        if (boff != 0.0) Bail("the bias carries an offset");
        break;
      }
      if (inner == nullptr) Bail("not a norm bias");
      ops.push_back(inner);
      mul_op = inner;
    } else if (auto cv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
      mul_op = cv.getOperand().getDefiningOp();
      if (!mlir::isa_and_nonnull<mlir::stablehlo::MulOp>(mul_op))
        Bail("not a norm downcast");
      ops.push_back(mul_op);
      form = "cvt.";
    }
    auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(mul_op);
    if (!mul) Bail("not a multiply");

    if (IsNormalizeMul(mul.getResult())) {
      // `RMSNorm(with_scale=False)`: no learned scale at all.
      y = mul.getResult();
      form = absl::StrCat(form, "noscale");
    } else {
      mlir::Value wsrc;
      bool took = false;
      for (bool swap : {false, true}) {
        mlir::Value a = swap ? mul.getRhs() : mul.getLhs();
        mlir::Value b = swap ? mul.getLhs() : mul.getRhs();
        if (!IsNormalizeMul(a)) continue;
        // Which spelling?  In maxtext/gemma/keras `a` IS `x * rsqrt` and `b`
        // is the weight.  In flax's, `a` is `rsqrt * w` and `b` is x -- and
        // what separates them is the inner multiply's OTHER operand: an
        // activation at the norm's own shape, or a broadcast of the [N]
        // weight.
        auto inner = mlir::dyn_cast<mlir::stablehlo::MulOp>(a.getDefiningOp());
        mlir::Value other, rsb;
        if (inner) rsb = RsqrtSide(inner, &other);
        if (rsb && IsFeatureBroadcast(other, N)) {
          if (ShapeOf(a) != out_shape || ShapeOf(b) != out_shape)
            Bail("the scale apply has the wrong shape");
          ops.push_back(inner);
          w = PeelWeight(other, N, static_cast<int64_t>(R) - 1, &ops, &offset);
          xs = b;
          rq = rsb;
          form = absl::StrCat(form, "scale");
        } else {
          if (ShapeOf(b) != ShapeOf(mul.getResult()))
            Bail("the weight apply has the wrong shape");
          y = a;
          wsrc = b;
          w = PeelWeight(wsrc, N, static_cast<int64_t>(R) - 1, &ops, &offset);
          form = absl::StrCat(form, "mul");
        }
        took = true;
        break;
      }
      if (!took) Bail("neither side of the multiply is a normalized value");
    }
  }

  NormChain ch = xs ? MatchNormalizedParts(xs, rq, out_shape, ops)
                    : MatchNormalized(y, out_shape, ops);
  mlir::Value x = ch.x;
  double eps = ch.eps;
  if (ch.layer) form = absl::StrCat(form, ".ln");
  // `fast::rms_norm` has no bias, and a mean-free norm with one is some
  // other layer.  The inner multiply is a candidate root of its own, so the
  // norm underneath is still found -- without the add.
  if (bias && !ch.layer) Bail("an rms norm with a bias");

  mlir::Type elem = ElemOf(x);
  if (!mlir::isa<mlir::FloatType>(elem) || mlir::isa<mlir::Float64Type>(elem))
    Bail("not a float norm");
  // MLX types the result from x, w and b together, and the tape binds ONE
  // result: a norm whose weight, bias or result lives in another dtype is
  // not this op.
  if (w && ElemOf(w) != elem) Bail("mixed dtypes");
  if (bias && ElemOf(bias) != elem) Bail("mixed dtypes");
  if (ElemOf(op->getResult(0)) != elem) Bail("mixed dtypes");

  // In the weightless form the root IS the normalize multiply, which
  // MatchNormalized walked as part of the chain.  A root is replaced, not
  // absorbed -- and its result is exactly what escapes to the rest of the
  // model -- so it must never appear in the absorbed list.
  ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());

  m->x = x;
  m->w = w;
  m->b = bias;
  m->layer = ch.layer;
  m->eps = eps;
  m->offset = offset;
  m->name = absl::StrCat("N", N, "R", R, ".", form, bias ? "+b" : "",
                         offset != 0.0 ? "+off" : "");
  return m;
}

// The roots worth trying.  A transpose is maxtext's; a multiply is the
// broadcast weight apply, flax's scale-first product, or the bare normalize;
// a convert is keras' (or flax's) downcast over either; an add is a
// LayerNorm's bias.
bool IsCandidateRoot(const std::string& name) {
  return name == "stablehlo.transpose" || name == "stablehlo.multiply" ||
         name == "stablehlo.convert" || name == "stablehlo.add";
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
      if (IsCandidateRoot(name) && !taken.contains(&op)) {
        try {
          found.push_back(MatchRoot(&op));
        } catch (const Reject& e) {
          // Every multiply and convert in the module reaches here, so the
          // narration keeps only rejects that got past the shape of a norm.
          static const char* const kQuiet[] = {
              "not the weight-apply", "the weight apply is not a dot",
              "the weight apply is not the batching multiply",
              "not a norm downcast", "not a multiply", "a rank-0 norm",
              "not a norm bias",
              "the normalized value is not a multiply",
              "the normalized value has no rsqrt side",
              "the normalized value is a weight broadcast",
              "neither side of the multiply is a normalized value",
              "the weight apply has the wrong shape"};
          bool quiet = false;
          for (const char* q : kQuiet) quiet = quiet || e.why == q;
          if (kDebug && !quiet)
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

  for (auto& m : found) {
    llvm::DenseSet<mlir::Operation*> seen;
    std::vector<mlir::Operation*> uniq;
    for (mlir::Operation* o : m->ops)
      if (seen.insert(o).second) uniq.push_back(o);
    m->ops = std::move(uniq);
  }
  // LARGEST first.  The bare normalize multiply matches on its own as a
  // weightless norm, and it is a sub-match of the weight apply built over
  // it: taking the small one would leave the weight multiply behind.
  std::stable_sort(found.begin(), found.end(),
                   [](const std::unique_ptr<RmsNormMatch>& a,
                      const std::unique_ptr<RmsNormMatch>& b) {
                     return a->ops.size() > b->ops.size();
                   });

  llvm::DenseSet<mlir::Operation*> roots;
  // Roots AND absorbed ops of the matches kept so far.  Constants stay out:
  // XLA hoists and CSEs them, so every norm in a 48-layer model shares one
  // eps and one feature-size constant.
  llvm::DenseSet<mlir::Operation*> used;
  std::vector<std::unique_ptr<RmsNormMatch>> kept;
  for (auto& m : found) {
    bool overlaps = used.contains(m->root) || taken.contains(m->root);
    for (mlir::Operation* o : m->ops) {
      if (IsConstSubtree(o)) continue;
      overlaps = overlaps || taken.contains(o) || used.contains(o);
    }
    if (overlaps) continue;
    used.insert(m->root);
    for (mlir::Operation* o : m->ops)
      if (!IsConstSubtree(o)) used.insert(o);
    roots.insert(m->root);
    kept.push_back(std::move(m));
  }
  if (kept.empty()) return;

  llvm::DenseSet<mlir::Operation*> cand;
  for (const auto& m : kept)
    for (mlir::Operation* o : m->ops)
      if (!IsConstSubtree(o)) cand.insert(o);
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
      if (IsConstSubtree(o)) continue;
      complete = complete && cand.contains(o);
    }
    if (!complete) {
      Debug(absl::StrCat("declined ", m->name,
                         ": an intermediate escapes the norm"));
      continue;
    }
    std::vector<mlir::Operation*> absorbed;
    for (mlir::Operation* o : m->ops)
      if (!IsConstSubtree(o)) absorbed.push_back(o);
    m->ops = std::move(absorbed);
    Debug(absl::StrCat(m->layer ? "matched a layer norm ("
                                : "matched an rms norm (",
                       m->name, ", ", m->ops.size(), " ops absorbed)"));
    plan->norm.push_back(std::move(m));
  }
}

}  // namespace metaljax
