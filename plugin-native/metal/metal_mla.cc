/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The multi-span decode-attention recognizer: maxtext's MLA decode, rewritten
into one fused `scaled_dot_product_attention` over the concatenated spans.

maxtext's MLA attention (DeepSeek-family models) keeps its KV cache in two
spans — the prefill span and the autoregressive span — and computes decode
attention per span as a masked softmax with a running max and sum, joining
the partials with the flash-attention renormalization:

    m = max(m_p, m_ar)
    l = exp(m_p - m) * l_p + exp(m_ar - m) * l_ar
    out = (exp(m_p - m) / l) * o_p + (exp(m_ar - m) / l) * o_ar

That IS the softmax over the concatenated scores, computed blockwise.  The
graph spells it as ~80 ops per layer (two scores dots, two mask trees
through outlined `_where` callees, two max/sum softmax chains, the combine
algebra) and jax re-emits the whole thing for every scanned layer — on
DeepSeek-V2-Lite decode this chain is the single largest slice of the
~700-node layer body (notes/row10-decode-floor-2026-08-29.md).

The rewrite binds the OUTSIDE values — q, each span's keys/values as the
dots read them, each span's segment-id vector — and emits one kMlaSdpa
(runtime/emits.cc): concat the spans, build the additive mask from the
segment ids (`seg == seg_val ? mask_true : mask_false`, exactly the numbers
the matched `_where` chain selects), and call MLX's fused sdpa, whose
vector kernel supports the MLA 192/128 head geometry.

Numerics: the literal chain computes probabilities in bf16 (max-subtract,
exp) with an f32 sum; the fused kernel computes in f32 throughout.  Same
class of reduction-order change as any fused attention — tolerance-level
against the CPU, greedy near-ties may flip (both cells reported wherever a
stream changes).  Masked positions get `score + (-2.38e38)` instead of
exactly `-2.38e38`; both are the softmax zero.

A half-matched pattern lowers as ORDINARY ops: every rejection below is a
`Bail`, and the consequence is the correct slow program — never a wrong
fused one (the recognizer file rule, metal_recognize.h).  The match also
declines unless the use-count fixpoint absorbs EVERY matched op: a partial
absorption would leave a consumer reading values that no longer exist.

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
  std::fprintf(stderr, "[metaljax-native] mla: %s\n", line.c_str());
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

// A splat float constant's value (bf16/f16/f32 splats included — APFloat
// reads the hex-splat encodings correctly, unlike the Python bindings).
std::optional<double> SplatFloatOf(mlir::Value v) {
  auto cst =
      mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return std::nullopt;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return std::nullopt;
  if (!mlir::isa<mlir::FloatType>(dense.getElementType())) return std::nullopt;
  return dense.getSplatValue<mlir::APFloat>().convertToDouble();
}

std::optional<int64_t> SplatIntOf(mlir::Value v) {
  auto cst =
      mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return std::nullopt;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return std::nullopt;
  if (!mlir::isa<mlir::IntegerType>(dense.getElementType()))
    return std::nullopt;
  return dense.getSplatValue<mlir::APInt>().getSExtValue();
}

bool IsZeroSplat(mlir::Operation* op) {
  auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(op);
  if (!cst) return false;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return false;
  if (mlir::isa<mlir::FloatType>(dense.getElementType()))
    return dense.getSplatValue<mlir::APFloat>().isZero();
  if (mlir::isa<mlir::IntegerType>(dense.getElementType()))
    return dense.getSplatValue<mlir::APInt>().isZero();
  return false;
}

// A value that is provably an all-zero tensor: a zero splat, or a
// broadcast / reshape chain over one (constants excluded from `ops`).
bool IsZeroTensor(mlir::Value v, std::vector<mlir::Operation*>* ops) {
  mlir::Operation* op = v.getDefiningOp();
  size_t mark = ops == nullptr ? 0 : ops->size();
  for (int depth = 0; op != nullptr && depth < 4; depth++) {
    if (IsZeroSplat(op)) return true;
    const std::string name = OpName(op);
    if (name != "stablehlo.broadcast_in_dim" && name != "stablehlo.reshape")
      break;
    if (ops != nullptr) ops->push_back(op);
    op = op->getOperand(0).getDefiningOp();
  }
  if (ops != nullptr) ops->resize(mark);
  return false;
}

// A rank-0 splat, or a broadcast of one: the mask threshold's two spellings.
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

std::optional<int64_t> SplatOrBroadcastInt(
    mlir::Value v, std::vector<mlir::Operation*>* ops) {
  if (auto d = SplatIntOf(v)) return d;
  auto b = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      v.getDefiningOp());
  if (!b) return std::nullopt;
  auto d = SplatIntOf(b.getOperand());
  if (d.has_value() && ops != nullptr) ops->push_back(b);
  return d;
}

// Flatten an add tree, stripping provably-zero terms (jax seeds its
// accumulators with `broadcast(0) + x`).
void FlattenAdd(mlir::Value v, std::vector<mlir::Operation*>* ops,
                std::vector<mlir::Value>* terms, int depth = 0) {
  if (depth > 6) Bail("an add tree too deep");
  if (auto add = mlir::dyn_cast_or_null<mlir::stablehlo::AddOp>(
          v.getDefiningOp())) {
    ops->push_back(add);
    for (mlir::Value side : {add.getLhs(), add.getRhs()}) {
      std::vector<mlir::Operation*> zeros;
      if (IsZeroTensor(side, &zeros)) {
        ops->insert(ops->end(), zeros.begin(), zeros.end());
        continue;
      }
      FlattenAdd(side, ops, terms, depth + 1);
    }
    return;
  }
  terms->push_back(v);
}

// The callee of a 3-operand func.call, verified single-block.
mlir::Block* CalleeBody(mlir::Operation* call, mlir::ModuleOp module) {
  if (OpName(call) != "func.call" || call->getNumOperands() != 3 ||
      call->getNumResults() != 1)
    Bail("not a three-operand helper call");
  auto sym = call->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
  if (!sym) Bail("a call with no callee");
  auto fn = module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
  if (!fn || fn.getBody().getBlocks().size() != 1)
    Bail("the helper is not a single-block function");
  mlir::Block* body = &fn.getBody().front();
  if (body->getNumArguments() != 3) Bail("the helper arity");
  return body;
}

// jax's outlined `_where(pred, t, f)` over two SCALARS:
//   select(arg0, broadcast(arg1), broadcast(arg2))
// Returns the call-site operands (pred tensor, t, f).
struct WhereScalar {
  mlir::Value pred, tval, fval;
};
WhereScalar MatchWhereScalar(mlir::Operation* call, mlir::ModuleOp module) {
  mlir::Block* body = CalleeBody(call, module);
  mlir::stablehlo::SelectOp sel;
  for (mlir::Operation& o : *body) {
    const std::string n = OpName(&o);
    if (n == "stablehlo.broadcast_in_dim") continue;
    if (n == "stablehlo.select") {
      if (sel) Bail("two helper selects");
      sel = mlir::cast<mlir::stablehlo::SelectOp>(&o);
    } else if (n == "func.return") {
      if (o.getNumOperands() != 1 || !sel ||
          o.getOperand(0) != sel.getResult())
        Bail("the helper does not return its select");
    } else {
      Bail(absl::StrCat("an unexpected helper op ", n));
    }
  }
  if (!sel) Bail("the helper has no select");
  auto traces_to = [&](mlir::Value v, int arg) {
    if (v == body->getArgument(arg)) return true;
    auto b = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
        v.getDefiningOp());
    return b && b.getOperand() == body->getArgument(arg);
  };
  if (!traces_to(sel.getPred(), 0) || !traces_to(sel.getOnTrue(), 1) ||
      !traces_to(sel.getOnFalse(), 2))
    Bail("the helper select does not read its arguments");
  return {call->getOperand(0), call->getOperand(1), call->getOperand(2)};
}

// jax's outlined `_where(pred, scores, f)` over a TENSOR true arm:
//   select(broadcast(arg0), arg1, broadcast(convert?(arg2)))
struct WhereTensor {
  mlir::Value pred, scores, fval;
};
WhereTensor MatchWhereTensor(mlir::Operation* call, mlir::ModuleOp module) {
  mlir::Block* body = CalleeBody(call, module);
  mlir::stablehlo::SelectOp sel;
  for (mlir::Operation& o : *body) {
    const std::string n = OpName(&o);
    if (n == "stablehlo.broadcast_in_dim" || n == "stablehlo.convert")
      continue;
    if (n == "stablehlo.select") {
      if (sel) Bail("two helper selects");
      sel = mlir::cast<mlir::stablehlo::SelectOp>(&o);
    } else if (n == "func.return") {
      if (o.getNumOperands() != 1 || !sel ||
          o.getOperand(0) != sel.getResult())
        Bail("the helper does not return its select");
    } else {
      Bail(absl::StrCat("an unexpected helper op ", n));
    }
  }
  if (!sel) Bail("the helper has no select");
  auto peel = [&](mlir::Value v) {
    for (int i = 0; i < 3; i++) {
      mlir::Operation* d = v.getDefiningOp();
      if (d == nullptr) return v;
      const std::string n = OpName(d);
      if (n != "stablehlo.broadcast_in_dim" && n != "stablehlo.convert")
        return v;
      v = d->getOperand(0);
    }
    return v;
  };
  if (peel(sel.getPred()) != body->getArgument(0) ||
      sel.getOnTrue() != body->getArgument(1) ||
      peel(sel.getOnFalse()) != body->getArgument(2))
    Bail("the helper select does not read its arguments");
  return {call->getOperand(0), call->getOperand(1), call->getOperand(2)};
}

// Peel converts and sharding aliases, collecting them for absorption.
mlir::Value Peel(mlir::Value v, std::vector<mlir::Operation*>* ops,
                 int limit = 4) {
  for (int i = 0; i < limit; i++) {
    mlir::Operation* d = v.getDefiningOp();
    if (d == nullptr) return v;
    const std::string n = OpName(d);
    if (n != "stablehlo.convert" && n != "sdy.sharding_constraint" &&
        n != "sdy.reshard")
      return v;
    if (d->getNumOperands() != 1) return v;
    ops->push_back(d);
    v = d->getOperand(0);
  }
  return v;
}

// One reduce over dimension `dim` with the given body op and an init that
// `check_init` approves.  Returns the reduced operand.
mlir::Value MatchReduce(mlir::Value v, absl::string_view body_op, int64_t dim,
                        const std::function<bool(mlir::Value)>& check_init,
                        std::vector<mlir::Operation*>* ops,
                        const char* what) {
  auto red = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(
      DefOf(v, what));
  if (!red || red.getNumOperands() != 2 || red->getNumResults() != 1)
    Bail(absl::StrCat(what, " is not a single-input reduce"));
  auto dims = red.getDimensions();
  if (dims.size() != 1 || dims[0] != dim)
    Bail(absl::StrCat(what, " reduces the wrong dimension"));
  mlir::Block& body = red.getBody().front();
  if (body.getOperations().size() != 2)
    Bail(absl::StrCat(what, " has a compound body"));
  mlir::Operation& first = body.front();
  if (OpName(&first) != absl::StrCat("stablehlo.", body_op))
    Bail(absl::StrCat(what, " body is not ", body_op));
  if (!check_init(red.getInitValues()[0]))
    Bail(absl::StrCat(what, " has the wrong init"));
  ops->push_back(red);
  return red.getInputs()[0];
}

bool IsNegInfSplat(mlir::Value v) {
  auto cst =
      mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return false;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return false;
  if (!mlir::isa<mlir::FloatType>(dense.getElementType())) return false;
  auto f = dense.getSplatValue<mlir::APFloat>();
  return f.isInfinity() && f.isNegative();
}

bool IsZeroInit(mlir::Value v) {
  auto cst =
      mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  return cst != nullptr && IsZeroSplat(cst);
}

// A broadcast_in_dim with exactly these dims.
mlir::Value MatchBroadcast(mlir::Value v, const std::vector<int64_t>& dims,
                           std::vector<mlir::Operation*>* ops,
                           const char* what) {
  auto b = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      DefOf(v, what));
  if (!b) Bail(absl::StrCat(what, " is not a broadcast"));
  auto bd = b.getBroadcastDimensions();
  if (std::vector<int64_t>(bd.begin(), bd.end()) != dims)
    Bail(absl::StrCat(what, " broadcasts the wrong dims"));
  ops->push_back(b);
  return b.getOperand();
}

mlir::Value MatchTranspose(mlir::Value v, const std::vector<int64_t>& perm,
                           std::vector<mlir::Operation*>* ops,
                           const char* what) {
  auto t = mlir::dyn_cast_or_null<mlir::stablehlo::TransposeOp>(
      DefOf(v, what));
  if (!t) Bail(absl::StrCat(what, " is not a transpose"));
  auto p = t.getPermutation();
  if (std::vector<int64_t>(p.begin(), p.end()) != perm)
    Bail(absl::StrCat(what, " has the wrong permutation"));
  ops->push_back(t);
  return t.getOperand();
}

mlir::Value MatchReshape(mlir::Value v, const std::vector<int64_t>& to,
                         std::vector<mlir::Operation*>* ops,
                         const char* what) {
  auto r = mlir::dyn_cast_or_null<mlir::stablehlo::ReshapeOp>(
      DefOf(v, what));
  if (!r) Bail(absl::StrCat(what, " is not a reshape"));
  if (ShapeOf(r.getResult()) != to)
    Bail(absl::StrCat(what, " reshapes to the wrong shape"));
  ops->push_back(r);
  return r.getOperand();
}

// One span, matched from its transposed running max `m_i` and the values
// the combine handed us.  Fills k/v/seg/T and pushes every chain op.
struct SpanChains {
  mlir::Value m_i;   // [B, 1, H, 1], the transposed span max
  mlir::Value l_i;   // [B, 1, H, 1], the transposed span sum
  mlir::Value o_i;   // [B, 1, H, Dv], the unnormalized span output
};

struct MaskInfo {
  mlir::Value seg;
  int64_t seg_val = 0;
  double mask_true = 0.0, mask_false = 0.0;
};

class Matcher {
 public:
  Matcher(mlir::ModuleOp module) : module_(module) {}

  std::unique_ptr<MlaMatch> MatchRoot(mlir::Operation* op) {
    auto root_add = mlir::dyn_cast<mlir::stablehlo::AddOp>(op);
    if (!root_add) Bail("not an add");
    std::vector<int64_t> oshape = ShapeOf(root_add.getResult());
    if (oshape.size() != 4 || oshape[1] != 1) Bail("not the [B,1,H,Dv] shape");
    if (!mlir::isa<mlir::FloatType>(ElemOf(root_add.getResult())) ||
        mlir::isa<mlir::Float64Type>(ElemOf(root_add.getResult())))
      Bail("not a float combine");
    auto m = std::make_unique<MlaMatch>();
    m->root = op;
    m->B = oshape[0];
    m->H = oshape[2];
    m->Dv = oshape[3];

    // The combine: exactly two weighted terms.
    std::vector<mlir::Value> terms;
    // The root op itself is the plan's root, not an absorbed op; flatten
    // its sides directly.
    for (mlir::Value side : {root_add.getLhs(), root_add.getRhs()}) {
      std::vector<mlir::Operation*> zeros;
      if (IsZeroTensor(side, &zeros)) {
        m->ops.insert(m->ops.end(), zeros.begin(), zeros.end());
        continue;
      }
      FlattenAdd(side, &m->ops, &terms);
    }
    if (terms.size() != 2) Bail("the combine does not have two terms");

    // term = multiply(broadcast(w_i), o_i); w_i = divide(exp(m_i - m), l).
    struct Term {
      mlir::Value m_i, o_i, mth, l;
    };
    std::vector<Term> ts;
    for (mlir::Value t : terms) {
      auto mul = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(
          DefOf(t, "a combine term"));
      if (!mul) Bail("a combine term is not a multiply");
      m->ops.push_back(mul);
      Term out;
      bool found = false;
      for (bool swap : {false, true}) {
        mlir::Value a = swap ? mul.getRhs() : mul.getLhs();
        mlir::Value b = swap ? mul.getLhs() : mul.getRhs();
        auto bc = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
            a.getDefiningOp());
        if (!bc) continue;
        auto div = mlir::dyn_cast_or_null<mlir::stablehlo::DivOp>(
            bc.getOperand().getDefiningOp());
        if (!div) continue;
        auto ex = mlir::dyn_cast_or_null<mlir::stablehlo::ExpOp>(
            div.getLhs().getDefiningOp());
        if (!ex) continue;
        auto sub = mlir::dyn_cast_or_null<mlir::stablehlo::SubtractOp>(
            ex.getOperand().getDefiningOp());
        if (!sub) continue;
        m->ops.push_back(bc);
        m->ops.push_back(div);
        m->ops.push_back(ex);
        m->ops.push_back(sub);
        out.m_i = sub.getLhs();
        out.mth = sub.getRhs();
        out.l = div.getRhs();
        out.o_i = b;
        found = true;
        break;
      }
      if (!found) Bail("a combine term is not broadcast(w) * o");
      ts.push_back(out);
    }
    if (ts[0].mth != ts[1].mth || ts[0].l != ts[1].l)
      Bail("the two terms disagree on the joint max or sum");

    // m = maximum(m_0, m_1) over exactly the two per-span maxes.
    auto mx_op = mlir::dyn_cast_or_null<mlir::stablehlo::MaxOp>(
        DefOf(ts[0].mth, "the joint max"));
    if (!mx_op) Bail("the joint max is not a maximum");
    m->ops.push_back(mx_op);
    llvm::DenseSet<mlir::Value> span_maxes{ts[0].m_i, ts[1].m_i};
    if (ts[0].m_i == ts[1].m_i) Bail("the two terms share a span max");
    if (!span_maxes.contains(mx_op.getLhs()) ||
        !span_maxes.contains(mx_op.getRhs()))
      Bail("the joint max is not over the span maxes");

    // l = sum of exp(m_i - m) * l_i, paired to the terms by m_i.
    std::vector<mlir::Value> lterms;
    FlattenAdd(ts[0].l, &m->ops, &lterms);
    if (lterms.size() != 2) Bail("the joint sum does not have two terms");
    llvm::DenseMap<mlir::Value, mlir::Value> l_by_max;
    for (mlir::Value lt : lterms) {
      auto mul = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(
          DefOf(lt, "a joint-sum term"));
      if (!mul) Bail("a joint-sum term is not a multiply");
      m->ops.push_back(mul);
      bool found = false;
      for (bool swap : {false, true}) {
        mlir::Value a = swap ? mul.getRhs() : mul.getLhs();
        mlir::Value b = swap ? mul.getLhs() : mul.getRhs();
        auto ex = mlir::dyn_cast_or_null<mlir::stablehlo::ExpOp>(
            a.getDefiningOp());
        if (!ex) continue;
        auto sub = mlir::dyn_cast_or_null<mlir::stablehlo::SubtractOp>(
            ex.getOperand().getDefiningOp());
        if (!sub || sub.getRhs() != ts[0].mth ||
            !span_maxes.contains(sub.getLhs()))
          continue;
        m->ops.push_back(ex);
        m->ops.push_back(sub);
        if (!l_by_max.try_emplace(sub.getLhs(), b).second)
          Bail("two joint-sum terms share a span max");
        found = true;
        break;
      }
      if (!found) Bail("a joint-sum term is not exp(m_i - m) * l_i");
    }

    // Per span: walk the softmax partials down to the scores dot.
    for (const Term& t : ts) {
      auto it = l_by_max.find(t.m_i);
      if (it == l_by_max.end()) Bail("a span has no matching sum term");
      MatchSpan(m.get(), t.m_i, it->second, t.o_i);
    }
    if (m->spans.size() != 2) Bail("span count");

    // One q, one dtype.
    if (!q_) Bail("no query");
    std::vector<int64_t> qshape = ShapeOf(q_);
    if (qshape != std::vector<int64_t>{m->B, 1, m->H, D_})
      Bail("the query shape disagrees");
    m->q = q_;
    m->D = D_;
    m->seg_val = seg_val_;
    m->mask_true = mask_true_;
    m->mask_false = mask_false_;
    m->name = absl::StrCat("B", m->B, "H", m->H, "D", m->D, "Dv", m->Dv,
                           "T", m->spans[0].T, "+", m->spans[1].T);
    return m;
  }

 private:
  // The per-span chains, entered from the transposed span max `m_i`.
  void MatchSpan(MlaMatch* m, mlir::Value m_i, mlir::Value l_i,
                 mlir::Value o_i) {
    const int64_t B = m->B, H = m->H, Dv = m->Dv;
    std::vector<mlir::Operation*>& ops = m->ops;

    // m_i = transpose([0,2,1,3], broadcast([0,1,2], reduce_max)).
    mlir::Value bcast_m =
        MatchTranspose(m_i, {0, 2, 1, 3}, &ops, "the span max");
    mlir::Value red_max_v =
        MatchBroadcast(bcast_m, {0, 1, 2}, &ops, "the span max broadcast");
    mlir::Value masked4 =
        MatchReduce(red_max_v, "maximum", 3, IsNegInfSplat, &ops,
                    "the span max reduction");
    std::vector<int64_t> mshape = ShapeOf(masked4);  // [B, H, 1, T]
    if (mshape.size() != 4 || mshape[0] != B || mshape[1] != H ||
        mshape[2] != 1)
      Bail("the masked scores have the wrong shape");
    const int64_t T = mshape[3];

    // masked4 = reshape(where_tensor(...)), [B,H,1,1,T] -> [B,H,1,T].
    mlir::Value where_t = MatchReshape(masked4, {B, H, 1, T}, &ops,
                                       "the masked scores");
    mlir::Operation* wt_call = DefOf(where_t, "the mask select");
    WhereTensor wt = MatchWhereTensor(wt_call, module_);
    ops.push_back(wt_call);
    if (!SplatFloatOf(wt.fval).has_value())
      Bail("the mask sentinel is not a splat");

    // pred = compare(GE, where_scalar(eq, true, false), threshold).
    auto ge = mlir::dyn_cast_or_null<mlir::stablehlo::CompareOp>(
        DefOf(wt.pred, "the mask gate"));
    if (!ge ||
        ge.getComparisonDirection() != mlir::stablehlo::ComparisonDirection::GE)
      Bail("the mask gate is not a GE compare");
    ops.push_back(ge);
    std::optional<double> thresh = SplatOrBroadcastFloat(ge.getRhs(), &ops);
    if (!thresh.has_value()) Bail("the mask threshold is not a splat");
    mlir::Operation* ws_call = DefOf(ge.getLhs(), "the mask values");
    WhereScalar ws = MatchWhereScalar(ws_call, module_);
    ops.push_back(ws_call);
    std::optional<double> tval = SplatFloatOf(ws.tval);
    std::optional<double> fval = SplatFloatOf(ws.fval);
    if (!tval.has_value() || !fval.has_value())
      Bail("the mask values are not splats");
    // The gate must reproduce the select: true-arm >= threshold > false-arm.
    if (!(*fval < *thresh && *thresh <= *tval))
      Bail("the mask gate does not separate the mask values");

    // eq = compare(EQ, broadcast(seg, [0,4]), splat), either order.
    auto eq = mlir::dyn_cast_or_null<mlir::stablehlo::CompareOp>(
        DefOf(ws.pred, "the segment compare"));
    if (!eq ||
        eq.getComparisonDirection() != mlir::stablehlo::ComparisonDirection::EQ)
      Bail("the segment compare is not EQ");
    ops.push_back(eq);
    mlir::Value seg;
    std::optional<int64_t> seg_val;
    for (bool swap : {false, true}) {
      mlir::Value a = swap ? eq.getRhs() : eq.getLhs();
      mlir::Value b = swap ? eq.getLhs() : eq.getRhs();
      std::vector<mlir::Operation*> side_ops;
      auto sv = SplatOrBroadcastInt(b, &side_ops);
      if (!sv.has_value()) continue;
      auto bc = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
          a.getDefiningOp());
      if (!bc) continue;
      auto bd = bc.getBroadcastDimensions();
      if (std::vector<int64_t>(bd.begin(), bd.end()) !=
          std::vector<int64_t>{0, 4})
        continue;
      seg = bc.getOperand();
      seg_val = sv;
      ops.push_back(bc);
      ops.insert(ops.end(), side_ops.begin(), side_ops.end());
      break;
    }
    if (!seg_val.has_value()) Bail("the segment compare has no splat side");
    if (ShapeOf(seg) != std::vector<int64_t>{B, T})
      Bail("the segment ids have the wrong shape");
    if (!mlir::isa<mlir::IntegerType>(ElemOf(seg)))
      Bail("the segment ids are not integral");

    // scores5 = transpose([0,1,4,3,2], dot1(k, reshape(q))).
    mlir::Value dot1_v = MatchTranspose(wt.scores, {0, 1, 4, 3, 2}, &ops,
                                        "the span scores");
    auto dot1 = mlir::dyn_cast_or_null<mlir::stablehlo::DotGeneralOp>(
        DefOf(dot1_v, "the scores dot"));
    if (!dot1) Bail("the scores are not a dot_general");
    ops.push_back(dot1);
    auto dn1 = dot1.getDotDimensionNumbers();
    auto eq_dims = [](llvm::ArrayRef<int64_t> a,
                      const std::vector<int64_t>& b) {
      return std::vector<int64_t>(a.begin(), a.end()) == b;
    };
    if (!eq_dims(dn1.getLhsBatchingDimensions(), {0, 2}) ||
        !eq_dims(dn1.getRhsBatchingDimensions(), {0, 2}) ||
        !eq_dims(dn1.getLhsContractingDimensions(), {3}) ||
        !eq_dims(dn1.getRhsContractingDimensions(), {4}))
      Bail("the scores dot has the wrong dims");
    mlir::Value k = dot1.getLhs();
    std::vector<int64_t> kshape = ShapeOf(k);  // [B, T, H, D]
    if (kshape.size() != 4 || kshape[0] != B || kshape[1] != T ||
        kshape[2] != H)
      Bail("the keys have the wrong shape");
    const int64_t D = kshape[3];
    if (D_ == 0) D_ = D;
    if (D != D_) Bail("the spans disagree on the head dim");
    mlir::Value q5 = dot1.getRhs();  // [B, 1, H, 1, D]
    if (ShapeOf(q5) != std::vector<int64_t>{B, 1, H, 1, D})
      Bail("the query operand has the wrong shape");
    mlir::Value q4 = MatchReshape(q5, {B, 1, H, 1, D}, &ops, "the query");
    if (ShapeOf(q4) != std::vector<int64_t>{B, 1, H, D})
      Bail("the query reshape has the wrong source");
    if (!q_) q_ = q4;
    if (q4 != q_) Bail("the spans disagree on the query");

    // l_i = transpose([0,2,1,3], convert?(broadcast([0,1,2],
    //       reduce_add(convert?(exp_i))))).
    std::vector<mlir::Operation*> lops;
    mlir::Value l_in = MatchTranspose(l_i, {0, 2, 1, 3}, &lops,
                                      "the span sum");
    l_in = Peel(l_in, &lops);
    mlir::Value red_sum_v =
        MatchBroadcast(l_in, {0, 1, 2}, &lops, "the span sum broadcast");
    mlir::Value exp_conv =
        MatchReduce(red_sum_v, "add", 3, IsZeroInit, &lops,
                    "the span sum reduction");
    mlir::Value exp_v = Peel(exp_conv, &lops);
    auto exp_op = mlir::dyn_cast_or_null<mlir::stablehlo::ExpOp>(
        DefOf(exp_v, "the span probabilities"));
    if (!exp_op) Bail("the span probabilities are not an exp");
    lops.push_back(exp_op);
    auto sub = mlir::dyn_cast_or_null<mlir::stablehlo::SubtractOp>(
        exp_op.getOperand().getDefiningOp());
    if (!sub || sub.getLhs() != masked4)
      Bail("the probabilities do not subtract from the masked scores");
    lops.push_back(sub);
    mlir::Value max_b = MatchBroadcast(sub.getRhs(), {0, 1, 2, 3}, &lops,
                                       "the max broadcast");
    if (max_b != bcast_m)
      Bail("the probabilities subtract a different max");
    ops.insert(ops.end(), lops.begin(), lops.end());

    // o_i = reshape(transpose([0,4,1,3,2], dot2(v, reshape(exp_i)))).
    mlir::Value tr2 = MatchReshape(o_i, {B, 1, H, Dv}, &ops, "the span out");
    mlir::Value dot2_v = MatchTranspose(tr2, {0, 4, 1, 3, 2}, &ops,
                                        "the span out transpose");
    auto dot2 = mlir::dyn_cast_or_null<mlir::stablehlo::DotGeneralOp>(
        DefOf(dot2_v, "the values dot"));
    if (!dot2) Bail("the span out is not a dot_general");
    ops.push_back(dot2);
    auto dn2 = dot2.getDotDimensionNumbers();
    if (!eq_dims(dn2.getLhsBatchingDimensions(), {0, 2}) ||
        !eq_dims(dn2.getRhsBatchingDimensions(), {0, 1}) ||
        !eq_dims(dn2.getLhsContractingDimensions(), {1}) ||
        !eq_dims(dn2.getRhsContractingDimensions(), {4}))
      Bail("the values dot has the wrong dims");
    mlir::Value v = dot2.getLhs();
    if (ShapeOf(v) != std::vector<int64_t>{B, T, H, Dv})
      Bail("the values have the wrong shape");
    mlir::Value exp5 = dot2.getRhs();
    mlir::Value exp_back = MatchReshape(exp5, {B, H, 1, 1, T}, &ops,
                                        "the probabilities operand");
    if (exp_back != exp_op.getResult())
      Bail("the values dot reads different probabilities");

    // dtypes: one compute dtype across q/k/v and the root.
    mlir::Type elem = ElemOf(q4);
    if (!mlir::isa<mlir::FloatType>(elem) ||
        mlir::isa<mlir::Float64Type>(elem))
      Bail("not a float attention");
    if (ElemOf(k) != elem || ElemOf(v) != elem ||
        ElemOf(m->root->getResult(0)) != elem)
      Bail("mixed dtypes");

    // Mask values: the same numbers on both spans.
    if (m->spans.empty()) {
      seg_val_ = *seg_val;
      mask_true_ = *tval;
      mask_false_ = *fval;
    } else if (seg_val_ != *seg_val || mask_true_ != *tval ||
               mask_false_ != *fval) {
      Bail("the spans disagree on the mask");
    }

    MlaSpan span;
    span.k = k;
    span.v = v;
    span.seg = seg;
    span.T = T;
    m->spans.push_back(span);
  }

  mlir::ModuleOp module_;
  mlir::Value q_;
  int64_t D_ = 0;
  int64_t seg_val_ = 0;
  double mask_true_ = 0.0, mask_false_ = 0.0;
};

}  // namespace

void AnalyzeMla(mlir::func::FuncOp fn, RewritePlan* plan) {
  if (EnvOff("METALJAX_MLA") || EnvOff("METALJAX_SDPA")) return;
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

  // Every block reachable from @main, callees included (maxtext outlines
  // scan-layer bodies; the lowering splices them).
  std::vector<std::unique_ptr<MlaMatch>> found;
  llvm::DenseSet<mlir::Operation*> visited_fns;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      const std::string name = OpName(&op);
      if (name == "stablehlo.add" && !taken.contains(&op)) {
        try {
          Matcher matcher(module);
          found.push_back(matcher.MatchRoot(&op));
        } catch (const Reject& e) {
          // Almost every add is not an attention combine; only narrate the
          // ones that got past the shape gate.
          if (kDebug && e.why != std::string("not the [B,1,H,Dv] shape") &&
              e.why != std::string("the combine does not have two terms") &&
              e.why != std::string("a combine term is not a multiply") &&
              e.why != std::string("not a float combine") &&
              e.why != std::string("not an add"))
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

  // Drop overlaps, then require the use-count fixpoint to absorb EVERY
  // matched op: the fused emit computes the whole chain internally, so a
  // partially-escaping match cannot stand (an unabsorbed consumer would
  // read values with no slots).
  llvm::DenseSet<mlir::Operation*> roots;
  std::vector<std::unique_ptr<MlaMatch>> kept;
  for (auto& m : found) {
    // De-duplicate the op list (shared subtrees are collected per use).
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
                         ": an intermediate escapes the attention"));
      continue;
    }
    // Constants stay shared; drop them from the absorb list.
    std::vector<mlir::Operation*> absorbed;
    for (mlir::Operation* o : m->ops)
      if (!mlir::isa<mlir::stablehlo::ConstantOp>(o)) absorbed.push_back(o);
    m->ops = std::move(absorbed);
    Debug(absl::StrCat("matched a multi-span attention (", m->name, ", ",
                       m->ops.size(), " ops absorbed)"));
    plan->mla.push_back(std::move(m));
  }
}

}  // namespace metaljax
