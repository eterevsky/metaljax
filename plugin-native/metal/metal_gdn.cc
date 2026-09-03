/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The gated-delta-net decode-step recognizer: the Qwen3.5 family's linear-
attention layer, at `seq_len == 1`, rewritten into ONE generated Metal
kernel.

keras-hub spells the layer in `qwen3_5_gated_delta_net.py`.  At decode every
`else` branch is taken and `for t in range(seq_len)` unrolls to a single
step, so `_recurrent_gated_delta_rule` becomes straight-line code with no
scan and nothing for a loop recognizer to match -- ~220 tape entries per
layer, 30 of the 40 layers on row 8 (Qwen3.6-35B-A3B) and 48 of the 64 on
row 21 (Qwen3.8-27B), 76 % of a decoded token.

The delta rule proper is five lines:

    S  <- S * exp(g)                        [B, Hv, Dk, Dv]
    kv <- sum_dk S[dk, dv] * k[dk]          [B, Hv, Dv]
    d  <- (v - kv) * beta                   [B, Hv, Dv]
    S  <- S + k[dk] * d[dv]                 the layer's new recurrent cache
    o  <- sum_dk S[dk, dv] * q[dk]          [B, Hv, Dv]

and the graph spells it with the rank-4 state BROADCAST against rank-3
vectors, so the [1,32,128,128] f32 state -- 2.1 MB -- is materialised FIVE
times per layer per token: ~0.63 GB/token of intermediate traffic on row 8
alone (`~/.cache/metaljax-bench/logs/moe-diag/diagnosis.md` §2b).  The fused
kernel holds the state in registers, reads it once and writes it once.

WHAT THE MATCH CLAIMS.  The recurrence, plus everything upstream that is
pure per-element preparation of its operands: the two `_l2norm` chains
(`rsqrt(sum(x*x) + eps) * x` -- which `AnalyzeNorm` declines, the mean not
being a divide, 120 times per row-8 program shape), the `1/sqrt(Dk)` pre-
scale, keras' `ops.repeat` lifting Hk key heads to Hv value heads, and the
transposes / reshapes / converts between them.  It stops at the output
reduce: the gated RMSNorm that follows is a norm, and belongs to
metal_norm.cc.  This is why `AnalyzeGdn` runs BEFORE `AnalyzeNorm` -- the
`_l2norm`s are inside the block and whichever recognizer runs first owns
them.

THE PEEL.  Everything absorbed upstream of the recurrence either preserves
the row-major element ORDER of its operand (reshape, convert, a transpose
that only moves size-1 axes, a broadcast that only inserts them) or is one
of three named forms (the scale multiply, the `_l2norm`, the head repeat).
Order preservation is what lets the emit bind the value the peel reached and
simply `mx::reshape` it to the canonical rank-3 form: the bytes are already
in the order the kernel indexes.  A step is absorbed only when EVERY user of
its result is already absorbed, so it can never strand a consumer -- and not
merely when it has one user, because the key is read twice (the memory read
and the outer product) and a single-use rule would stop its peel at the
first shared reshape.

TWO RESULTS.  The new state escapes -- it is the recurrent cache the layer
writes back -- so this is the only recognizer whose fused entry has two
results: `root` (the output reduce) and `state_root` (the state-update add).
Both are absorbed and both are bound by `LowerGdn`.  The escape fixpoint is
told that `state_root` is a second root, and the matcher requires that
`state_root` precede `root` in the same block with every outside user of its
result AFTER `root` -- which is what makes one entry able to define it.

NUMERICS.  The kernel does the chain's arithmetic element for element, in
the order keras wrote it (decay, then kv_mem, then delta, then the state
update, then the output), and it replays the dtype NARROWINGS the matcher
observed: the `_l2norm`'s bf16-rounded rsqrt and product, and the state's
dead `convert(f32->bf16) -> convert(bf16->f32)` round trip (keras autocasts
the cache operand and converts straight back -- 2 entries and 2 x 2.1 MB of
waste that nonetheless ROUNDS, so the kernel rounds too).  What remains is
the ORDER of the two dk reductions: the graph's `stablehlo.reduce` and the
kernel's chunked threadgroup sum add the same Dk terms differently.  That is
the same class of change as any fused attention -- tolerance-level against
the CPU, greedy near-ties may flip -- and a tighter one, because every other
rounding point is reproduced.

A half-matched pattern lowers as ORDINARY ops: every rejection below is a
`Bail`, and the consequence is the correct slow program -- never a wrong
fused one (the recognizer file rule, metal_recognize.h).  METALJAX_GDN=0
disables the recognizer outright.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseSet.h"
#include "metal/metal_dtypes.h"
#include "metal/metal_recognize.h"
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
  std::fprintf(stderr, "[metaljax-native] gdn: %s\n", line.c_str());
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

int64_t Numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

mlir::Type ElemOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) Bail("a value that is not a ranked tensor");
  return t.getElementType();
}

bool IsF32(mlir::Value v) {
  auto f = mlir::dyn_cast<mlir::FloatType>(ElemOf(v));
  return f && f.getWidth() == 32 && f.isF32();
}

// A float element type narrower than f32 -- the only kind this recognizer
// replays a rounding for.
bool IsNarrowFloat(mlir::Type t) {
  auto f = mlir::dyn_cast<mlir::FloatType>(t);
  return f && f.getWidth() < 32;
}

int NarrowCode(mlir::Type t) {
  std::optional<int> c = TapeDtypeCode(t);
  if (!c.has_value()) Bail("a narrowing this tape has no dtype for");
  return *c;
}

int DtypeCodeOf(mlir::Value v) {
  std::optional<int> c = TapeDtypeCode(ElemOf(v));
  if (!c.has_value()) Bail("an operand this tape has no dtype for");
  return *c;
}

mlir::Operation* DefOf(mlir::Value v, const char* what) {
  mlir::Operation* op = v.getDefiningOp();
  if (op == nullptr) Bail(absl::StrCat(what, " is a block argument"));
  return op;
}

// The single-use rule the peel runs under: absorbing an op with another
// consumer would strand it.
bool SingleUse(mlir::Operation* op) {
  if (op == nullptr || op->getNumResults() != 1) return false;
  return op->getResult(0).hasOneUse();
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

// A rank-0 float splat, or a broadcast of one (the two spellings a scalar
// constant reaches an elementwise op in).  Pushes the broadcast when it
// consumed one.
std::optional<double> SplatOrBroadcastFloat(mlir::Value v,
                                            std::vector<mlir::Operation*>* ops) {
  if (auto d = SplatFloatOf(v)) return d;
  auto b = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      v.getDefiningOp());
  if (!b) return std::nullopt;
  auto d = SplatFloatOf(b.getOperand());
  if (d.has_value() && ops != nullptr) ops->push_back(b);
  return d;
}

bool IsZeroInit(mlir::Value v) {
  auto d = SplatFloatOf(v);
  return d.has_value() && *d == 0.0;
}

// ---------------------------------------------------------------------------
// order-preserving layout ops
// ---------------------------------------------------------------------------

// A transpose that only moves size-1 axes: the non-unit axes keep their
// relative order, so reading the result row-major yields the operand's own
// element sequence.  keras' `(0, 2, 1, 3)` on a [B, 1, H, D] decode value is
// exactly this.
bool UnitTranspose(mlir::stablehlo::TransposeOp t) {
  const std::vector<int64_t> in = ShapeOf(t.getOperand());
  auto p = t.getPermutation();
  int64_t last = -1;
  for (int64_t src : p) {
    if (src < 0 || src >= static_cast<int64_t>(in.size())) return false;
    if (in[src] == 1) continue;
    if (src < last) return false;
    last = src;
  }
  return true;
}

// A broadcast_in_dim that only INSERTS size-1 axes: every operand axis maps
// to an output axis of the same extent, in increasing order, and every axis
// it does not name has extent 1.  Order-preserving, unlike a broadcast that
// duplicates.
bool UnitBroadcast(mlir::stablehlo::BroadcastInDimOp b) {
  const std::vector<int64_t> in = ShapeOf(b.getOperand());
  const std::vector<int64_t> out = ShapeOf(b.getResult());
  auto dims = b.getBroadcastDimensions();
  if (dims.size() != in.size()) return false;
  std::vector<bool> named(out.size(), false);
  int64_t last = -1;
  for (size_t i = 0; i < in.size(); i++) {
    const int64_t d = dims[i];
    if (d <= last || d < 0 || d >= static_cast<int64_t>(out.size()))
      return false;
    if (out[d] != in[i]) return false;
    named[d] = true;
    last = d;
  }
  for (size_t d = 0; d < out.size(); d++)
    if (!named[d] && out[d] != 1) return false;
  return true;
}

// keras' `ops.repeat(x, R, axis=a)`: a broadcast that inserts one axis of
// extent R immediately after axis `a`, then a reshape merging the two.  The
// row-major merge makes value head `hv` read key head `hv / R`, which is the
// indexing the kernel does.  Returns R, or 0 when this is not that form.
int64_t RepeatFactor(mlir::stablehlo::BroadcastInDimOp b, int64_t* head_axis) {
  const std::vector<int64_t> in = ShapeOf(b.getOperand());
  const std::vector<int64_t> out = ShapeOf(b.getResult());
  if (out.size() != in.size() + 1) return 0;
  auto dims = b.getBroadcastDimensions();
  if (dims.size() != in.size()) return 0;
  // The one output axis the broadcast does not name is the repeat axis.
  std::vector<bool> named(out.size(), false);
  int64_t last = -1;
  for (size_t i = 0; i < in.size(); i++) {
    const int64_t d = dims[i];
    if (d <= last || d < 0 || d >= static_cast<int64_t>(out.size())) return 0;
    if (out[d] != in[i]) return 0;
    named[d] = true;
    last = d;
  }
  int64_t rep = -1;
  for (size_t d = 0; d < out.size(); d++)
    if (!named[d]) {
      if (rep >= 0) return 0;
      rep = static_cast<int64_t>(d);
    }
  // It must sit immediately after a real axis: that axis is the head count.
  if (rep <= 0 || out[rep] < 2) return 0;
  *head_axis = rep - 1;
  return out[rep];
}

// ---------------------------------------------------------------------------
// the `_l2norm` chain
// ---------------------------------------------------------------------------

struct L2Norm {
  bool found = false;
  mlir::Value x;
  double eps = 0.0;
  int narrow = -1;   // the dtype the rsqrt and the product were rounded to
};

// `x * rsqrt(sum(x * x, -1, keepdims) + eps)`, as keras spells it: the sum
// accumulates in f32, the rest runs in the model dtype.  Every op must be
// single-use; the two multiplies read the SAME `x` value.
L2Norm MatchL2Norm(mlir::Value v, std::vector<mlir::Operation*>* ops) {
  L2Norm out;
  auto mul = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(v.getDefiningOp());
  if (!mul) return out;
  for (int side = 0; side < 2; side++) {
    std::vector<mlir::Operation*> got;
    mlir::Value x = side == 0 ? mul.getLhs() : mul.getRhs();
    mlir::Value r = side == 0 ? mul.getRhs() : mul.getLhs();
    // r = broadcast(rsqrt(add(broadcast(reduce_add(convert?(x * x))), eps)))
    auto rb = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
        r.getDefiningOp());
    if (!rb || !SingleUse(rb)) continue;
    got.push_back(rb);
    auto rs = mlir::dyn_cast_or_null<mlir::stablehlo::RsqrtOp>(
        rb.getOperand().getDefiningOp());
    if (!rs || !SingleUse(rs)) continue;
    got.push_back(rs);
    auto add = mlir::dyn_cast_or_null<mlir::stablehlo::AddOp>(
        rs.getOperand().getDefiningOp());
    if (!add || !SingleUse(add)) continue;
    got.push_back(add);
    // One side of the add is the eps splat, the other the summed squares.
    mlir::Value sums;
    std::optional<double> eps;
    for (int e = 0; e < 2 && !eps.has_value(); e++) {
      std::vector<mlir::Operation*> epsops;
      mlir::Value cand = e == 0 ? add.getRhs() : add.getLhs();
      eps = SplatOrBroadcastFloat(cand, &epsops);
      if (eps.has_value()) {
        for (mlir::Operation* o : epsops) got.push_back(o);
        sums = e == 0 ? add.getLhs() : add.getRhs();
      }
    }
    if (!eps.has_value()) continue;
    // An optional downcast of the f32 sum to the model dtype, then the
    // keepdims broadcast, then the reduce.
    mlir::Value cur = sums;
    if (auto cv = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(
            cur.getDefiningOp())) {
      if (!SingleUse(cv)) continue;
      got.push_back(cv);
      cur = cv.getOperand();
    }
    auto kb = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
        cur.getDefiningOp());
    if (!kb || !SingleUse(kb) || !UnitBroadcast(kb)) continue;
    got.push_back(kb);
    auto red = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(
        kb.getOperand().getDefiningOp());
    if (!red || red.getNumOperands() != 2 || red->getNumResults() != 1 ||
        !SingleUse(red))
      continue;
    auto rdims = red.getDimensions();
    const std::vector<int64_t> xshape = ShapeOf(x);
    if (rdims.size() != 1 ||
        rdims[0] != static_cast<int64_t>(xshape.size()) - 1)
      continue;
    mlir::Block& body = red.getBody().front();
    if (body.getOperations().size() != 2) continue;
    if (OpName(&body.front()) != "stablehlo.add") continue;
    if (!IsZeroInit(red.getInitValues()[0])) continue;
    // The sum must accumulate in f32: a chain that squares and sums in the
    // model dtype is a different computation, and declining is safe.
    if (!IsF32(red.getResult(0))) continue;
    got.push_back(red);
    mlir::Value sq = red.getInputs()[0];
    if (auto cv = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(
            sq.getDefiningOp())) {
      if (!SingleUse(cv)) continue;
      got.push_back(cv);
      sq = cv.getOperand();
    }
    auto sqm = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(
        sq.getDefiningOp());
    if (!sqm || sqm.getLhs() != x || sqm.getRhs() != x) continue;
    // The square feeds only this chain (`x` itself is used twice more).
    if (!SingleUse(sqm)) continue;
    got.push_back(sqm);
    // The add / rsqrt / product must agree on one element type; when it is
    // narrower than f32 the kernel rounds at those three points.
    mlir::Type t = ElemOf(add.getResult());
    if (ElemOf(rs.getResult()) != t || ElemOf(mul.getResult()) != t) continue;
    out.found = true;
    out.x = x;
    out.eps = *eps;
    out.narrow = IsNarrowFloat(t) ? NarrowCode(t) : -1;
    got.push_back(mul);
    for (mlir::Operation* o : got) ops->push_back(o);
    return out;
  }
  return out;
}

// ---------------------------------------------------------------------------
// the operand peel
// ---------------------------------------------------------------------------

struct Peel {
  mlir::Value value;
  double scale = 1.0;
  bool l2 = false;
  double eps = 0.0;
  int narrow = -1;
  int64_t repeat = 1;
  // The narrowest float type a chain of converts round-tripped through, when
  // it came back to where it started (keras' autocast of an f32 cache
  // operand): the kernel replays that rounding.
  int round_trip = -1;
};

// Walk backwards from an operand of the recurrence to the earliest value
// whose row-major element order is the operand's own, absorbing what it
// passes.  `allow_named` enables the three forms that are not merely layout
// (the scale multiply, the `_l2norm`, the head repeat); `g` and `beta` are
// peeled without them.
//
// The rule for absorbing a step is that EVERY user of its result is already
// absorbed -- not that it has one user.  The key is read twice by the
// recurrence (once for the memory read, once for the outer product) through
// two broadcast chains that the core match already claimed, so a single-use
// rule would stop the key's peel at the first shared reshape and leave its
// whole `_l2norm` and head repeat running.  `taken` is the absorbed set so
// far, and each step joins it.
Peel PeelOperand(mlir::Value v, std::vector<mlir::Operation*>* ops,
                 llvm::DenseSet<mlir::Operation*>* taken, bool allow_named) {
  Peel p;
  p.value = v;
  std::vector<mlir::Type> convert_types;
  auto absorb = [&](mlir::Operation* o) {
    ops->push_back(o);
    taken->insert(o);
  };
  auto free_to_take = [&](mlir::Operation* o) {
    if (o == nullptr || o->getNumResults() != 1) return false;
    for (mlir::Operation* u : o->getResult(0).getUsers())
      if (!taken->contains(u)) return false;
    return true;
  };
  for (int step = 0; step < 24; step++) {
    mlir::Operation* d = p.value.getDefiningOp();
    if (d == nullptr || !free_to_take(d)) break;
    const std::string n = OpName(d);
    if (n == "stablehlo.reshape") {
      absorb(d);
      p.value = d->getOperand(0);
      continue;
    }
    if (n == "stablehlo.convert") {
      convert_types.push_back(ElemOf(d->getResult(0)));
      absorb(d);
      p.value = d->getOperand(0);
      continue;
    }
    if (n == "sdy.sharding_constraint" || n == "sdy.reshard") {
      absorb(d);
      p.value = d->getOperand(0);
      continue;
    }
    if (auto t = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(d)) {
      if (!UnitTranspose(t)) break;
      absorb(d);
      p.value = t.getOperand();
      continue;
    }
    if (auto b = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(d)) {
      if (UnitBroadcast(b)) {
        absorb(d);
        p.value = b.getOperand();
        continue;
      }
      int64_t head = 0;
      const int64_t rep = allow_named && p.repeat == 1
                              ? RepeatFactor(b, &head)
                              : 0;
      if (rep == 0) break;
      p.repeat = rep;
      absorb(d);
      p.value = b.getOperand();
      continue;
    }
    if (allow_named && n == "stablehlo.multiply") {
      // The `1 / sqrt(Dk)` pre-scale.  ORDER MATTERS and the peel runs
      // backwards, so a scale is only absorbable while no `_l2norm` has been
      // absorbed yet: seeing it first here means it applies AFTER the norm in
      // the dataflow, which is where the kernel applies it.  keras spells it
      // that way; a graph that scaled BEFORE normalizing would be a different
      // computation (the norm divides the scale straight back out), so that
      // one stops the peel instead of being folded into `scale`.
      std::vector<mlir::Operation*> sops;
      std::optional<double> c =
          p.l2 ? std::nullopt : SplatOrBroadcastFloat(d->getOperand(1), &sops);
      mlir::Value other = d->getOperand(0);
      if (!c.has_value() && !p.l2) {
        sops.clear();
        c = SplatOrBroadcastFloat(d->getOperand(0), &sops);
        other = d->getOperand(1);
      }
      if (c.has_value()) {
        p.scale *= *c;
        for (mlir::Operation* o : sops) absorb(o);
        absorb(d);
        p.value = other;
        continue;
      }
      // ...or the `_l2norm`, at most one per operand.
      if (!p.l2) {
        std::vector<mlir::Operation*> nops;
        L2Norm l2 = MatchL2Norm(p.value, &nops);
        if (l2.found) {
          p.l2 = true;
          p.eps = l2.eps;
          p.narrow = l2.narrow;
          for (mlir::Operation* o : nops) absorb(o);
          p.value = l2.x;
          continue;
        }
      }
      break;
    }
    break;
  }
  // A convert chain that left f32 and came back rounds; record the narrowest
  // type it passed through so the kernel can round the same way.
  if (IsF32(p.value)) {
    for (mlir::Type t : convert_types)
      if (IsNarrowFloat(t)) p.round_trip = NarrowCode(t);
  }
  return p;
}

// ---------------------------------------------------------------------------
// the recurrence
// ---------------------------------------------------------------------------

// Strip a chain of broadcast_in_dim ops down to the value they lifted to
// `want`, requiring that the result be `want`-shaped and every step
// single-use.  Returns the base value.
mlir::Value StripBroadcasts(mlir::Value v, std::vector<mlir::Operation*>* ops,
                            size_t want_rank, const char* what) {
  for (int step = 0; step < 6; step++) {
    const std::vector<int64_t> s = ShapeOf(v);
    if (s.size() == want_rank) return v;
    auto b = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
        v.getDefiningOp());
    if (!b) Bail(absl::StrCat(what, " is not a broadcast chain"));
    if (!SingleUse(b)) Bail(absl::StrCat(what, " broadcast is shared"));
    ops->push_back(b);
    v = b.getOperand();
  }
  Bail(absl::StrCat(what, " broadcast chain is too deep"));
}

// One rank-3 vector lifted to the rank-4 state shape along `axis`: the
// broadcast chain that turns [B, Hv, N] into [B, Hv, Dk, Dv] by placing N on
// dimension `axis` (2 for a key/query over Dk, 3 for a delta over Dv).
//
// The AXIS is checked, not just the extent: with Dk == Dv -- which is the
// row-8 and row-21 geometry -- the key and the delta lift to the same shape
// from the same rank, and only the broadcast_dims of the rank-3 -> rank-4
// step say which is which.  Reading them the wrong way round would silently
// transpose the outer product.
mlir::Value LiftedVector(mlir::Value v, int64_t axis,
                         const std::vector<int64_t>& state_shape,
                         std::vector<mlir::Operation*>* ops, const char* what) {
  if (ShapeOf(v) != state_shape) Bail(absl::StrCat(what, " is not state-shaped"));
  // Walk down to rank 3, remembering how the step that left rank 3 placed
  // the vector's own axis.
  std::vector<mlir::Operation*> trial;
  mlir::Value cur = v;
  int64_t placed = -1;
  for (int step = 0; step < 6; step++) {
    const std::vector<int64_t> s = ShapeOf(cur);
    if (s.size() == 3) break;
    auto b = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
        cur.getDefiningOp());
    if (!b) Bail(absl::StrCat(what, " is not a broadcast chain"));
    if (!SingleUse(b)) Bail(absl::StrCat(what, " broadcast is shared"));
    auto dims = b.getBroadcastDimensions();
    if (ShapeOf(b.getOperand()).size() == 3) {
      if (dims.size() != 3) Bail(absl::StrCat(what, " lift arity"));
      if (dims[0] != 0 || dims[1] != 1)
        Bail(absl::StrCat(what, " does not keep the batch and head axes"));
      placed = dims[2];
    }
    trial.push_back(b);
    cur = b.getOperand();
  }
  if (ShapeOf(cur).size() != 3)
    Bail(absl::StrCat(what, " does not reduce to a rank-3 vector"));
  if (placed != axis)
    Bail(absl::StrCat(what, " is lifted along the wrong axis"));
  const std::vector<int64_t> s = ShapeOf(cur);
  if (s[0] != state_shape[0] || s[1] != state_shape[1] ||
      s[2] != state_shape[axis])
    Bail(absl::StrCat(what, " has the wrong extents"));
  for (mlir::Operation* o : trial) ops->push_back(o);
  return cur;
}

// A rank-2 [B, Hv] scalar-per-head lifted to the state shape.
mlir::Value LiftedScalar(mlir::Value v, const std::vector<int64_t>& state_shape,
                         std::vector<mlir::Operation*>* ops, const char* what) {
  if (ShapeOf(v) != state_shape) Bail(absl::StrCat(what, " is not state-shaped"));
  mlir::Value base = StripBroadcasts(v, ops, 2, what);
  const std::vector<int64_t> s = ShapeOf(base);
  if (s[0] != state_shape[0] || s[1] != state_shape[1])
    Bail(absl::StrCat(what, " has the wrong extents"));
  return base;
}

// The `sum over dk` the recurrence does twice: a single-input reduce with an
// add body and a zero init over dimension 2 of a rank-4 f32 value.
mlir::Value DkReduce(mlir::Value v, std::vector<mlir::Operation*>* ops,
                     const char* what) {
  auto red =
      mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(DefOf(v, what));
  if (!red || red.getNumOperands() != 2 || red->getNumResults() != 1)
    Bail(absl::StrCat(what, " is not a single-input reduce"));
  auto dims = red.getDimensions();
  if (dims.size() != 1 || dims[0] != 2)
    Bail(absl::StrCat(what, " reduces the wrong dimension"));
  mlir::Block& body = red.getBody().front();
  if (body.getOperations().size() != 2 ||
      OpName(&body.front()) != "stablehlo.add")
    Bail(absl::StrCat(what, " body is not an add"));
  if (!IsZeroInit(red.getInitValues()[0]))
    Bail(absl::StrCat(what, " has a non-zero init"));
  ops->push_back(red);
  return red.getInputs()[0];
}

// Split a multiply into its two operands.
std::pair<mlir::Value, mlir::Value> Mul(mlir::Value v,
                                        std::vector<mlir::Operation*>* ops,
                                        const char* what) {
  auto m = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(DefOf(v, what));
  if (!m) Bail(absl::StrCat(what, " is not a multiply"));
  ops->push_back(m);
  return {m.getLhs(), m.getRhs()};
}

// `S * broadcast(exp(g))`, either operand order.  Reports failure rather
// than bailing, so the caller can try the other half of an add.
bool MatchDecay(mlir::Value v, const std::vector<int64_t>& st,
                std::vector<mlir::Operation*>* ops, mlir::Value* state,
                mlir::Value* g) {
  for (int side = 0; side < 2; side++) {
    std::vector<mlir::Operation*> trial;
    try {
      std::pair<mlir::Value, mlir::Value> ab = Mul(v, &trial, "the decay");
      mlir::Value s = side == 0 ? ab.first : ab.second;
      mlir::Value gl = side == 0 ? ab.second : ab.first;
      if (ShapeOf(s) != st) Bail("the decayed state is not state-shaped");
      mlir::Value gv = LiftedScalar(gl, st, &trial, "the decay gate");
      auto ex =
          mlir::dyn_cast_or_null<mlir::stablehlo::ExpOp>(gv.getDefiningOp());
      if (!ex) Bail("the decay gate is not an exponential");
      if (!SingleUse(ex)) Bail("the decay gate is shared");
      trial.push_back(ex);
      *state = s;
      *g = ex.getOperand();
      for (mlir::Operation* o : trial) ops->push_back(o);
      return true;
    } catch (const Reject&) {
      continue;
    }
  }
  return false;
}

// `broadcast(k)[dk] * broadcast(delta)[dv]`, either operand order.
bool MatchOuter(mlir::Value v, const std::vector<int64_t>& st,
                std::vector<mlir::Operation*>* ops, mlir::Value* k,
                mlir::Value* delta) {
  for (int side = 0; side < 2; side++) {
    std::vector<mlir::Operation*> trial;
    try {
      std::pair<mlir::Value, mlir::Value> ab = Mul(v, &trial, "the outer");
      mlir::Value kl = side == 0 ? ab.first : ab.second;
      mlir::Value dl = side == 0 ? ab.second : ab.first;
      mlir::Value kv = LiftedVector(kl, 2, st, &trial, "the outer key");
      mlir::Value dv = LiftedVector(dl, 3, st, &trial, "the outer delta");
      *k = kv;
      *delta = dv;
      for (mlir::Operation* o : trial) ops->push_back(o);
      return true;
    } catch (const Reject&) {
      continue;
    }
  }
  return false;
}

class Matcher {
 public:
  // `op` is a candidate output reduce.  Throws `Reject` unless the whole
  // decode step hangs off it.
  std::unique_ptr<GdnMatch> MatchRoot(mlir::Operation* op) {
    auto m = std::make_unique<GdnMatch>();
    m->root = op;
    std::vector<mlir::Operation*>& ops = m->ops;

    // o = sum_dk (S_new * broadcast(q))
    auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
    if (!red) Bail("not a reduce");
    if (red.getNumOperands() != 2 || red->getNumResults() != 1)
      Bail("not a single-input reduce");
    auto dims = red.getDimensions();
    if (dims.size() != 1 || dims[0] != 2) Bail("not a reduce over dim 2");
    if (!IsZeroInit(red.getInitValues()[0])) Bail("not a zero init");
    mlir::Block& body = red.getBody().front();
    if (body.getOperations().size() != 2 ||
        OpName(&body.front()) != "stablehlo.add")
      Bail("not an add reduce");
    mlir::Value prod = red.getInputs()[0];
    const std::vector<int64_t> st = ShapeOf(prod);
    if (st.size() != 4) Bail("not a rank-4 reduce");
    if (!IsF32(prod)) Bail("not an f32 recurrence");
    const int64_t B = st[0], Hv = st[1], Dk = st[2], Dv = st[3];
    if (B < 1 || Hv < 1 || Dk < 2 || Dv < 1) Bail("degenerate extents");

    auto [pa, pb] = Mul(prod, &ops, "the output product");
    // One side is the new state, the other the lifted query.
    mlir::Value snew, qlift;
    if (mlir::isa_and_nonnull<mlir::stablehlo::AddOp>(pa.getDefiningOp())) {
      snew = pa;
      qlift = pb;
    } else if (mlir::isa_and_nonnull<mlir::stablehlo::AddOp>(
                   pb.getDefiningOp())) {
      snew = pb;
      qlift = pa;
    } else {
      Bail("the output product has no state-update side");
    }
    mlir::Value q = LiftedVector(qlift, 2, st, &ops, "the query");

    // S_new = S_decay + broadcast(k) * broadcast(delta), either order.
    auto add = mlir::cast<mlir::stablehlo::AddOp>(snew.getDefiningOp());
    m->state_root = add;
    ops.push_back(add);
    mlir::Value k, delta, state, gexp, decay;
    bool ok = false;
    for (int flip = 0; flip < 2 && !ok; flip++) {
      std::vector<mlir::Operation*> trial;
      mlir::Value d = flip == 0 ? add.getLhs() : add.getRhs();
      mlir::Value o = flip == 0 ? add.getRhs() : add.getLhs();
      if (!MatchDecay(d, st, &trial, &state, &gexp)) continue;
      if (!MatchOuter(o, st, &trial, &k, &delta)) continue;
      decay = d;
      for (mlir::Operation* x : trial) ops.push_back(x);
      ok = true;
    }
    if (!ok) Bail("the state update is not decay plus outer product");

    // delta = (v - sum_dk(S_decay * broadcast(k))) * broadcast(beta)
    auto [sa, sb] = Mul(delta, &ops, "the delta product");
    mlir::Value sub, betalift;
    if (mlir::isa_and_nonnull<mlir::stablehlo::SubtractOp>(sa.getDefiningOp())) {
      sub = sa;
      betalift = sb;
    } else if (mlir::isa_and_nonnull<mlir::stablehlo::SubtractOp>(
                   sb.getDefiningOp())) {
      sub = sb;
      betalift = sa;
    } else {
      Bail("the delta product has no subtract side");
    }
    // beta arrives at [B, Hv, Dv]; strip its broadcasts down to [B, Hv].
    mlir::Value beta;
    {
      const std::vector<int64_t> bs = ShapeOf(betalift);
      if (bs.size() != 3 || bs[0] != B || bs[1] != Hv || bs[2] != Dv)
        Bail("the write gate is not [B, Hv, Dv]");
      beta = StripBroadcasts(betalift, &ops, 2, "the write gate");
      const std::vector<int64_t> s2 = ShapeOf(beta);
      if (s2[0] != B || s2[1] != Hv) Bail("the write gate has the wrong extents");
    }
    auto subop = mlir::cast<mlir::stablehlo::SubtractOp>(sub.getDefiningOp());
    ops.push_back(subop);
    mlir::Value v = subop.getLhs();
    mlir::Value kv = subop.getRhs();
    // kv_mem = sum_dk (S_decay * broadcast(k))
    mlir::Value kvprod = DkReduce(kv, &ops, "the memory read");
    auto [ma, mb] = Mul(kvprod, &ops, "the memory product");
    mlir::Value kv_decay, kv_klift;
    if (ma == decay) {
      kv_decay = ma;
      kv_klift = mb;
    } else if (mb == decay) {
      kv_decay = mb;
      kv_klift = ma;
    } else {
      Bail("the memory read does not reuse the decayed state");
    }
    mlir::Value k2 = LiftedVector(kv_klift, 2, st, &ops, "the memory key");
    if (k2 != k) Bail("the memory read uses a different key");

    // The five values the kernel reads, peeled back through their glue.
    llvm::DenseSet<mlir::Operation*> taken(ops.begin(), ops.end());
    taken.insert(op);
    Peel pq = PeelOperand(q, &ops, &taken, /*allow_named=*/true);
    Peel pk = PeelOperand(k, &ops, &taken, /*allow_named=*/true);
    Peel pv = PeelOperand(v, &ops, &taken, /*allow_named=*/false);
    Peel pg = PeelOperand(gexp, &ops, &taken, /*allow_named=*/false);
    Peel pbeta = PeelOperand(beta, &ops, &taken, /*allow_named=*/false);
    Peel pstate = PeelOperand(state, &ops, &taken, /*allow_named=*/false);

    if (pq.repeat != pk.repeat)
      Bail("the query and key repeat differently");
    const int64_t repeat = pq.repeat;
    if (repeat < 1 || Hv % repeat != 0) Bail("the head repeat does not divide");
    const int64_t Hk = Hv / repeat;

    if (Numel(ShapeOf(pq.value)) != B * Hk * Dk) Bail("the query numel");
    if (Numel(ShapeOf(pk.value)) != B * Hk * Dk) Bail("the key numel");
    if (Numel(ShapeOf(pv.value)) != B * Hv * Dv) Bail("the value numel");
    if (Numel(ShapeOf(pg.value)) != B * Hv) Bail("the decay numel");
    if (Numel(ShapeOf(pbeta.value)) != B * Hv) Bail("the write gate numel");
    if (Numel(ShapeOf(pstate.value)) != B * Hv * Dk * Dv) Bail("the state numel");
    if (!IsF32(pstate.value)) Bail("the state is not f32");
    if (pk.scale != 1.0) Bail("the key carries a scale");

    m->q = pq.value;
    m->k = pk.value;
    m->v = pv.value;
    m->g = pg.value;
    m->beta = pbeta.value;
    m->state = pstate.value;
    m->B = B;
    m->Hv = Hv;
    m->Hk = Hk;
    m->Dk = Dk;
    m->Dv = Dv;
    m->scale = pq.scale;
    m->l2q = pq.l2;
    m->l2k = pk.l2;
    m->eps_q = pq.eps;
    m->eps_k = pk.eps;
    m->narrow_q = pq.narrow;
    m->narrow_k = pk.narrow;
    m->narrow_state = pstate.round_trip;
    m->q_dtype = DtypeCodeOf(m->q);
    m->k_dtype = DtypeCodeOf(m->k);
    m->v_dtype = DtypeCodeOf(m->v);
    m->g_dtype = DtypeCodeOf(m->g);
    m->beta_dtype = DtypeCodeOf(m->beta);
    m->state_dtype = DtypeCodeOf(m->state);
    m->out_dtype = DtypeCodeOf(op->getResult(0));
    m->new_state_dtype = DtypeCodeOf(add.getResult());

    // The geometry the kernel can launch: one threadgroup per (b, hv), Dv
    // threads wide, Dk split into chunks the registers hold.  A geometry
    // that does not fit declines, and the literal chain runs.
    if (Dv > 512) Bail("the value head is wider than the kernel supports");
    bool fits = false;
    for (int64_t ty : {1, 2, 4, 8, 16, 32}) {
      if (Dv * ty > 1024) break;
      if ((Dk + ty - 1) / ty <= 32) {
        fits = true;
        break;
      }
    }
    if (!fits) Bail("no threadgroup geometry holds the state in registers");

    m->name = absl::StrCat("B", B, "H", Hv, "/", Hk, "D", Dk, "x", Dv,
                           m->l2q && m->l2k ? "+l2" : "",
                           m->narrow_state >= 0 ? "+rt" : "");
    return m;
  }
};

}  // namespace

void AnalyzeGdn(mlir::func::FuncOp fn, RewritePlan* plan) {
  if (EnvOff("METALJAX_GDN")) return;
  if (fn.getBody().getBlocks().size() != 1) return;
  auto module = fn->getParentOfType<mlir::ModuleOp>();
  if (!module) return;

  // Ops another recognizer already owns (rebuild() has not run yet).
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

  // Every block reachable from @main, callees included.
  std::vector<std::unique_ptr<GdnMatch>> found;
  llvm::DenseSet<mlir::Operation*> visited_fns;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      const std::string name = OpName(&op);
      if (name == "stablehlo.reduce" && !taken.contains(&op)) {
        try {
          Matcher matcher;
          found.push_back(matcher.MatchRoot(&op));
        } catch (const Reject& e) {
          // Almost every reduce is not a delta-rule output; narrate only the
          // ones that got past the shape gate.
          if (kDebug && e.why != std::string("not a reduce over dim 2") &&
              e.why != std::string("not a rank-4 reduce") &&
              e.why != std::string("not an f32 recurrence") &&
              e.why != std::string("not a single-input reduce") &&
              e.why != std::string("not a zero init") &&
              e.why != std::string("not an add reduce") &&
              e.why != std::string("degenerate extents") &&
              e.why != std::string("the output product is not a multiply"))
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

  // Drop overlaps.
  llvm::DenseSet<mlir::Operation*> roots;
  std::vector<std::unique_ptr<GdnMatch>> kept;
  for (auto& m : found) {
    llvm::DenseSet<mlir::Operation*> seen;
    std::vector<mlir::Operation*> uniq;
    for (mlir::Operation* o : m->ops)
      if (seen.insert(o).second) uniq.push_back(o);
    m->ops = std::move(uniq);
    bool overlaps = roots.contains(m->root) || taken.contains(m->root) ||
                    roots.contains(m->state_root) ||
                    taken.contains(m->state_root);
    for (mlir::Operation* o : m->ops)
      overlaps = overlaps || taken.contains(o) || roots.contains(o);
    if (overlaps) continue;
    // The second result must be definable by the one entry the root emits:
    // the state update has to precede the root in the same block, and every
    // user of its result that this match does not absorb has to follow it.
    if (m->state_root->getBlock() != m->root->getBlock() ||
        !m->state_root->isBeforeInBlock(m->root)) {
      Debug(absl::StrCat("declined ", m->name,
                         ": the state update does not precede the output"));
      continue;
    }
    llvm::DenseSet<mlir::Operation*> mine(m->ops.begin(), m->ops.end());
    mine.insert(m->root);
    bool late = true;
    for (mlir::Operation* u : m->state_root->getResult(0).getUsers()) {
      if (mine.contains(u)) continue;
      late = late && u->getBlock() == m->root->getBlock() &&
             m->root->isBeforeInBlock(u);
    }
    if (!late) {
      Debug(absl::StrCat("declined ", m->name,
                         ": the new state is read before the output"));
      continue;
    }
    roots.insert(m->root);
    roots.insert(m->state_root);
    kept.push_back(std::move(m));
  }
  if (kept.empty()) return;

  // The use-count fixpoint, with ONE exception: `state_root` is a second
  // root, so it is allowed -- required -- to escape.  Everything else must
  // be absorbed, or a consumer would read a value with no slot.
  llvm::DenseSet<mlir::Operation*> state_roots;
  for (const auto& m : kept) state_roots.insert(m->state_root);
  llvm::DenseSet<mlir::Operation*> cand;
  for (const auto& m : kept)
    for (mlir::Operation* o : m->ops)
      if (!mlir::isa<mlir::stablehlo::ConstantOp>(o) && !state_roots.contains(o))
        cand.insert(o);
  auto inside_users = [&](llvm::DenseSet<mlir::Operation*>& cs,
                          mlir::Operation* o) {
    for (mlir::Value r : o->getResults()) {
      for (mlir::OpOperand& use : r.getUses()) {
        mlir::Operation* u = use.getOwner();
        if (roots.contains(u) || cs.contains(u)) continue;
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
      if (o == m->state_root) continue;
      complete = complete && cand.contains(o);
    }
    if (!complete) {
      Debug(absl::StrCat("declined ", m->name,
                         ": an intermediate escapes the step"));
      continue;
    }
    // Constants stay shared; drop them from the absorb list.
    std::vector<mlir::Operation*> absorbed;
    for (mlir::Operation* o : m->ops)
      if (!mlir::isa<mlir::stablehlo::ConstantOp>(o)) absorbed.push_back(o);
    m->ops = std::move(absorbed);
    Debug(absl::StrCat("matched a gated delta step (", m->name, ", ",
                       m->ops.size(), " ops absorbed)"));
    plan->gdn.push_back(std::move(m));
  }
}

}  // namespace metaljax
