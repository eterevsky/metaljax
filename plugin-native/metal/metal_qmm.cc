/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

`src/metaljax/qmm.py`, as a pass over the parsed StableHLO.

The two halves are the Python module's two halves, and neither is re-designed
here: `AnalyzeQmm` is its structural matching (`_try_affine`, `_try_perchannel`,
`_finish`, `_prune`) and touches no value; `BuildQmmPacks` is its first-execute
verification and packing (`_build_affine_pack`, `_build_mxfp4_pack`), on the
concrete buffers of this execute.  What is deliberately NOT ported is
`_Source`'s row-blocked evaluation and the cross-executable build cache: both
are memory/latency optimizations over an evaluation the tape already stages
op by op with last-use pruning, and leaving them out costs peak memory on a
20-GB weight set, never an answer.  The two facts that ARE load-bearing are
kept: MLX's buffer cache is off while a pack runs (a pack's dead buffers are
claimed memory a watchdog reads), and every exactness check is exact.

A check that fails DISABLES that one dot: its chain then lowers literally, as
it did before this file existed.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/StringRef.h"
#include "metal/metal_dtypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlx/mlx.h"
#include "program.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace metaljax {
namespace {

// The environment, read once (qmm.py's module-level knobs).
bool EnvOff(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "0";
}
const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();
// qmm.py `_SCALE_WIDTH` (METALJAX_QMM_SCALES): "auto" keeps the source width
// when nothing is lost by it, "source" always narrows, "f32" never does.
const std::string kScaleWidth = [] {
  const char* v = std::getenv("METALJAX_QMM_SCALES");
  return std::string(v == nullptr ? "auto" : v);
}();
// qmm.py `_BATCH` (METALJAX_QMM_BATCH): fuse dots that carry batching dims.
const bool kBatch = !EnvOff("METALJAX_QMM_BATCH");
// qmm.py `_GROUP_SIZES`, largest first (gs=32 measured 1.8x slower at decode).
constexpr int64_t kGroupSizes[3] = {128, 64, 32};
constexpr int64_t kMinGroup = 32;
// OCP MXFP4: one shared power-of-two scale per 32 elements, a 4-bit E2M1
// element.  The magnitudes are indexed by the low three bits of the code.
constexpr float kE2M1Mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constexpr int64_t kMxfp4Group = 32;
constexpr uint32_t kMix = 2246822519u;
// Rows hashed per pass in `_column_keys`: the maps are full weight size, so
// the digest must not materialize a second copy of one.
constexpr int64_t kKeyChunk = 1 << 22;

// A candidate does not match (or cannot be trusted): run it literally.
struct Reject {
  std::string why;
};
[[noreturn]] void Bail(const std::string& why) { throw Reject{why}; }

void Debug(const std::string& line) {
  if (!kDebug) return;
  std::fprintf(stderr, "[metaljax-native] qmm: %s\n", line.c_str());
  std::fflush(stderr);
}

// --------------------------------------------------------------------------
// structural matching (qmm.py, same section)
// --------------------------------------------------------------------------

mlir::Operation* Owner(mlir::Value v) {
  return v == nullptr ? nullptr : v.getDefiningOp();
}

std::string OpName(mlir::Operation* op) {
  return op == nullptr ? std::string()
                       : op->getName().getStringRef().str();
}

std::vector<int64_t> ShapeOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) Bail("a value that is not a ranked tensor");
  return std::vector<int64_t>(t.getShape().begin(), t.getShape().end());
}

std::string ElName(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return std::string();
  std::optional<std::string> n = TapeElementName(t.getElementType());
  return n.has_value() ? *n : std::string();
}

bool IsIntEl(mlir::Value v) {
  const std::string el = ElName(v);
  return el == "i4" || el == "ui4" || el == "i8" || el == "i16" ||
         el == "i32" || el == "i64" || el == "ui8" || el == "ui16" ||
         el == "ui32" || el == "ui64";
}

bool IsFloatEl(mlir::Value v) {
  const std::string el = ElName(v);
  return el == "f16" || el == "f32" || el == "bf16";
}

int64_t Prod(const std::vector<int64_t>& xs) {
  int64_t p = 1;
  for (int64_t x : xs) p *= x;
  return p;
}

bool IsShapeOp(const std::string& name) {
  return name == "stablehlo.reshape" || name == "stablehlo.transpose";
}

// qmm.py `_strip`: walk down through the named ops, outermost last.
mlir::Value Strip(mlir::Value v, bool shape_ops, bool converts,
                  std::vector<mlir::Operation*>* chain) {
  std::vector<mlir::Operation*> got;
  while (true) {
    mlir::Operation* o = Owner(v);
    const std::string name = OpName(o);
    const bool ok = (converts && name == "stablehlo.convert") ||
                    (shape_ops && IsShapeOp(name));
    if (o == nullptr || !ok) break;
    got.push_back(o);
    v = o->getOperand(0);
  }
  if (chain != nullptr) {
    chain->assign(got.rbegin(), got.rend());
  }
  return v;
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

// qmm.py `_parse_codes`: `v` as an integer-code operand.
struct Codes {
  mlir::Value codes;
  mlir::Value zero;             // null when there is no zero point
  bool has_range = false;       // the subtraction is done in INTEGER arithmetic
  int64_t lo = 0, hi = 0;
};

bool ParseCodes(mlir::Value v, Codes* out) {
  mlir::Value base = Strip(v, /*shape_ops=*/false, /*converts=*/true, nullptr);
  mlir::Operation* o = Owner(base);
  if (o != nullptr && OpName(o) == "stablehlo.subtract") {
    mlir::Value c =
        Strip(o->getOperand(0), /*shape_ops=*/false, /*converts=*/true,
              nullptr);
    if (!IsIntEl(c)) return false;
    out->codes = c;
    out->zero = o->getOperand(1);
    out->has_range = false;
    if (IsIntEl(base)) {
      // Unsigned wrap semantics: not worth modelling (qmm.py).
      const std::string el = ElName(base);
      int width = 0;
      if (el == "i4") width = 4;
      else if (el == "i8") width = 8;
      else if (el == "i16") width = 16;
      else if (el == "i32") width = 32;
      else if (el == "i64") width = 64;
      else return false;
      out->has_range = true;
      out->lo = -(int64_t{1} << (width - 1));
      out->hi = (int64_t{1} << (width - 1)) - 1;
    }
    return true;
  }
  if (IsIntEl(base)) {
    out->codes = base;
    out->zero = nullptr;
    out->has_range = false;
    return true;
  }
  return false;
}

// qmm.py `_bcast_chain`: peel a chain of broadcast_in_dim (jax emits one per
// rank step).  `dims[i]` is the dimension of `v` that base's dimension i lands
// in; `found` is false when the head is not a broadcast at all.
mlir::Value BcastChain(mlir::Value v, std::vector<int64_t>* dims, bool* found) {
  std::vector<int64_t> cur(ShapeOf(v).size());
  std::iota(cur.begin(), cur.end(), 0);
  *found = false;
  while (true) {
    mlir::Operation* o = Owner(v);
    if (o == nullptr) break;
    const std::string name = OpName(o);
    if (name == "stablehlo.broadcast_in_dim") {
      std::vector<int64_t> d = I64List(o, "broadcast_dimensions");
      std::vector<int64_t> next;
      next.reserve(d.size());
      for (int64_t j : d) {
        if (j < 0 || j >= static_cast<int64_t>(cur.size()))
          Bail("a broadcast_in_dim with out-of-range dimensions");
        next.push_back(cur[j]);
      }
      cur = std::move(next);
      v = o->getOperand(0);
      *found = true;
      continue;
    }
    if (name == "stablehlo.convert") {
      v = o->getOperand(0);
      continue;
    }
    break;
  }
  *dims = cur;
  return v;
}

// The dot's dimension numbers, as ops/linalg.py reads them.
struct DotDims {
  std::vector<int64_t> lb, rb, lc, rc;
};

DotDims ReadDotDims(mlir::Operation* op) {
  auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
  if (!dot) Bail("a dot_general in an unexpected form");
  mlir::stablehlo::DotDimensionNumbersAttr dn = dot.getDotDimensionNumbers();
  DotDims d;
  d.lb.assign(dn.getLhsBatchingDimensions().begin(),
              dn.getLhsBatchingDimensions().end());
  d.rb.assign(dn.getRhsBatchingDimensions().begin(),
              dn.getRhsBatchingDimensions().end());
  d.lc.assign(dn.getLhsContractingDimensions().begin(),
              dn.getLhsContractingDimensions().end());
  d.rc.assign(dn.getRhsContractingDimensions().begin(),
              dn.getRhsContractingDimensions().end());
  return d;
}

// qmm.py `_hoist`: follow a value out of the loops that merely carry it.
// jax does NOT lower a while_loop's closed-over constants as region captures;
// they become loop-carried state the body returns unchanged, so a decode
// loop's weights arrive as body block arguments and what is constant for the
// whole loop is the while's initial operand.
mlir::Value Hoist(mlir::Value v) {
  for (int guard = 0; guard < 8; guard++) {
    auto ba = mlir::dyn_cast<mlir::BlockArgument>(v);
    if (!ba) return v;
    mlir::Block* blk = ba.getOwner();
    mlir::Operation* op = blk->getParentOp();
    if (op == nullptr || OpName(op) != "stablehlo.while") return v;
    const unsigned i = ba.getArgNumber();
    if (op->getNumRegions() < 2) return v;
    mlir::Block& body = op->getRegion(1).front();
    if (body.empty()) return v;
    mlir::Operation* term = &body.back();
    if (i >= term->getNumOperands() || i >= body.getNumArguments()) return v;
    if (term->getOperand(i) != body.getArgument(i)) return v;  // carry changes
    if (i >= op->getNumOperands()) return v;
    v = op->getOperand(i);
  }
  return v;
}

// qmm.py `_OPAQUE`.
bool IsOpaque(const std::string& name) {
  return name == "stablehlo.while" || name == "stablehlo.if" ||
         name == "stablehlo.case" || name == "stablehlo.custom_call" ||
         name == "stablehlo.optimization_barrier";
}

// qmm.py `_closure`: the backward closure of `values`, stopping at @main's
// block arguments and at constants.  Every block argument reached must be
// one of @main's -- otherwise the subtree depends on a loop carry and cannot
// be packed once.
void Closure(const std::vector<mlir::Value>& values,
             const llvm::DenseMap<mlir::Value, int>& main_args,
             llvm::DenseSet<mlir::Operation*>* ops,
             absl::flat_hash_set<int>* arg_indices,
             const std::vector<mlir::Operation*>& register_ops) {
  for (mlir::Operation* o : register_ops) ops->insert(o);
  std::vector<mlir::Value> stack = values;
  llvm::DenseSet<mlir::Value> seen;
  while (!stack.empty()) {
    mlir::Value v = stack.back();
    stack.pop_back();
    if (!seen.insert(v).second) continue;
    if (mlir::isa<mlir::BlockArgument>(v)) {
      auto it = main_args.find(v);
      if (it != main_args.end()) {
        arg_indices->insert(it->second);
        continue;
      }
      mlir::Value outer = Hoist(v);
      if (outer == v) Bail("operand depends on an inner block argument");
      stack.push_back(outer);
      continue;
    }
    mlir::Operation* o = Owner(v);
    if (o == nullptr) Bail("operand is neither a block argument nor a result");
    const std::string name = OpName(o);
    if (IsOpaque(name)) Bail(absl::StrCat("operand subtree contains ", name));
    ops->insert(o);
    for (mlir::Value x : o->getOperands()) stack.push_back(x);
  }
}

// qmm.py `_finish`: fill in `m` for a dot whose operand `qside` is quantized.
void Finish(const llvm::DenseMap<mlir::Value, int>& main_args,
            const absl::flat_hash_set<int>& donated, QmmMatch* m,
            mlir::Operation* root, const std::vector<mlir::Value>& absorb,
            mlir::Operation* dot, int qside,
            const std::vector<mlir::Operation*>& required,
            const std::vector<mlir::Operation*>& register_ops) {
  const DotDims d = ReadDotDims(dot);
  const std::vector<int64_t>& qb = qside == 1 ? d.rb : d.lb;
  const std::vector<int64_t>& qc = qside == 1 ? d.rc : d.lc;
  const std::vector<int64_t>& ab = qside == 1 ? d.lb : d.rb;
  const std::vector<int64_t>& ac = qside == 1 ? d.lc : d.rc;
  if ((!qb.empty() || !ab.empty()) && !kBatch) Bail("batching dimensions");
  mlir::Value quant = dot->getOperand(qside);
  mlir::Value act = dot->getOperand(1 - qside);
  if (!IsFloatEl(act))
    Bail(absl::StrCat("activation element type ", ElName(act)));
  if (!IsFloatEl(root->getResult(0)))
    Bail(absl::StrCat("result element type ", ElName(root->getResult(0))));
  const std::vector<int64_t> qshape = ShapeOf(quant);
  const std::vector<int64_t> ashape = ShapeOf(act);
  auto pick = [](const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& dims) {
    std::vector<int64_t> out;
    for (int64_t i : dims) {
      if (i < 0 || i >= static_cast<int64_t>(shape.size()))
        Bail("dimension numbers out of range");
      out.push_back(shape[i]);
    }
    return out;
  };
  if (pick(ashape, ab) != pick(qshape, qb)) Bail("batching dimensions disagree");
  auto holds = [](const std::vector<int64_t>& xs, int64_t v) {
    return std::find(xs.begin(), xs.end(), v) != xs.end();
  };
  std::vector<int64_t> afree, qfree;
  for (int64_t i = 0; i < static_cast<int64_t>(ashape.size()); i++)
    if (!holds(ac, i) && !holds(ab, i)) afree.push_back(i);
  for (int64_t i = 0; i < static_cast<int64_t>(qshape.size()); i++)
    if (!holds(qc, i) && !holds(qb, i)) qfree.push_back(i);
  const int64_t B = Prod(pick(ashape, ab));
  const int64_t M = Prod(pick(ashape, afree));
  const int64_t K = Prod(pick(ashape, ac));
  const int64_t N = Prod(pick(qshape, qfree));
  if (B == 0 || M == 0 || K == 0 || N == 0) Bail("empty matmul");
  if (K % kMinGroup)
    Bail(absl::StrFormat("K=%d is not a multiple of %d", K, kMinGroup));
  if (K != Prod(pick(qshape, qc))) Bail("contracting dimensions disagree");

  m->root = root;
  m->lhs = act;
  m->swapped = qside == 0;
  // dot_general's result is laid out batch dims, then LHS free, then RHS free.
  // `quantized_matmul` on [B, M, K] x [B, N, K] returns [B, M, N], which is
  // that layout when the quantized operand is the rhs and its transpose when
  // it is the lhs (jax lowers `th,emh->etm` that way).
  m->lperm = ab;
  m->lperm.insert(m->lperm.end(), afree.begin(), afree.end());
  m->lperm.insert(m->lperm.end(), ac.begin(), ac.end());
  m->rperm = qb;
  m->rperm.insert(m->rperm.end(), qfree.begin(), qfree.end());
  m->rperm.insert(m->rperm.end(), qc.begin(), qc.end());
  m->rshape = qshape;
  m->bshape = pick(ashape, ab);
  m->mshape = pick(ashape, afree);
  m->nshape = pick(qshape, qfree);
  m->B = B;
  m->M = M;
  m->K = K;
  m->N = N;
  std::optional<int> code = TapeDtypeCode(
      mlir::cast<mlir::RankedTensorType>(root->getResult(0).getType())
          .getElementType());
  if (!code.has_value()) Bail("result dtype has no tape code");
  m->out_dtype = *code;
  m->name = m->bshape.empty()
                ? absl::StrFormat("%dx%dx%d", M, K, N)
                : absl::StrFormat("%d|%dx%dx%d", B, M, K, N);
  if (m->swapped) m->name += "'";

  llvm::DenseSet<mlir::Operation*> ops;
  absl::flat_hash_set<int> args;
  Closure(absorb, main_args, &ops, &args, register_ops);
  for (int i : args) {
    if (donated.contains(i))
      Bail("quantized operand is donated (buffer may be reused)");
  }
  m->arg_indices.assign(args.begin(), args.end());
  std::sort(m->arg_indices.begin(), m->arg_indices.end());
  m->ops.assign(ops.begin(), ops.end());
  // Ops that MUST end up skipped for the rewrite to be a win: everything
  // between the affine reconstruction and the dot.  If some other consumer
  // forces the dequantized weight to be materialized anyway, running a
  // quantized matmul on top of it would only add work.
  m->required = required;
  for (mlir::Operation* o : m->post) m->required.push_back(o);
}

// qmm.py `_has_int_leaf`: a cheap guard that keeps the MXFP4 branch from
// adopting an ordinary `x @ (w * mask)`.
bool HasIntLeaf(mlir::Value v, int limit = 64) {
  std::vector<mlir::Value> stack{v};
  llvm::DenseSet<mlir::Value> seen;
  while (!stack.empty() && static_cast<int>(seen.size()) < limit) {
    mlir::Value cur = stack.back();
    stack.pop_back();
    if (!seen.insert(cur).second) continue;
    if (IsIntEl(cur)) return true;
    mlir::Operation* o = Owner(cur);
    if (o == nullptr || IsOpaque(OpName(o))) continue;
    for (mlir::Value x : o->getOperands()) stack.push_back(x);
  }
  return false;
}

// qmm.py `_split_scaled`: (values, scale) for a `values * broadcast(per-group
// scale)` product.  The split is structural -- the scale is the operand
// broadcast from a tensor holding exactly one value per 32, which is the only
// group size MXFP4 has, and that exact 32 is what keeps this from adopting an
// RMS norm (whose scale is broadcast one per ROW).
void SplitScaled(mlir::Operation* mul, mlir::Value* values, mlir::Value* scale) {
  const int64_t total = Prod(ShapeOf(mul->getResult(0)));
  std::vector<int> hits;
  for (int i = 0; i < 2; i++) {
    mlir::Value v = mul->getOperand(i);
    if (!IsFloatEl(v))
      Bail(absl::StrCat("multiply operand element type ", ElName(v)));
    std::vector<int64_t> dims;
    bool found = false;
    mlir::Value base = BcastChain(v, &dims, &found);
    if (!found) continue;
    const int64_t n = Prod(ShapeOf(base));
    if (n != 0 && n * kMxfp4Group == total) hits.push_back(i);
  }
  if (hits.empty())
    Bail(absl::StrFormat(
        "neither multiply operand broadcasts one scale per %d values",
        kMxfp4Group));
  if (hits.size() == 2) Bail("cannot tell the scale operand from the values");
  *values = mul->getOperand(1 - hits[0]);
  *scale = mul->getOperand(hits[0]);
}

// qmm.py `_try_affine_side`.
std::unique_ptr<QmmMatch> TryAffineSide(
    const llvm::DenseMap<mlir::Value, int>& main_args,
    const absl::flat_hash_set<int>& donated, mlir::Operation* dot, int qside) {
  mlir::Value qop = dot->getOperand(qside);
  std::vector<mlir::Operation*> post;
  mlir::Value base = Strip(qop, /*shape_ops=*/true, /*converts=*/true, &post);
  mlir::Operation* mul = Owner(base);
  if (mul == nullptr || OpName(mul) != "stablehlo.multiply")
    Bail("the operand is not a multiply");
  auto m = std::make_unique<QmmMatch>();
  Codes parsed;
  mlir::Value scale;
  bool have = false;
  for (int i = 0; i < 2; i++) {
    if (ParseCodes(mul->getOperand(i), &parsed)) {
      scale = mul->getOperand(1 - i);
      have = true;
      break;
    }
  }
  if (!have) {
    // No integer operand: MXFP4's grid is non-uniform, so its codes have
    // become floats before the scale is applied.
    mlir::Value values;
    SplitScaled(mul, &values, &scale);
    if (!HasIntLeaf(values)) Bail("the scaled operand holds no integer codes");
    m->mode = 1;
    m->codes = values;
  } else {
    m->codes = parsed.codes;
    m->zero = parsed.zero;
    m->has_sub_range = parsed.has_range;
    m->sub_lo = parsed.lo;
    m->sub_hi = parsed.hi;
  }
  if (!IsFloatEl(scale))
    Bail(absl::StrCat("scale element type ", ElName(scale)));
  m->scale = scale;
  for (mlir::Operation* o : post)
    if (IsShapeOp(OpName(o))) m->post.push_back(o);
  Finish(main_args, donated, m.get(), dot, {qop}, dot, qside, {mul}, {});
  return m;
}

// qmm.py `_try_affine`: the weight-times-scale forms, on whichever operand
// carries them.  jax puts the quantized weight on the RHS for a plain
// projection but on the LHS for an einsum like `th,emh->etm`, so both sides
// are tried; the RHS goes first, which is the common case.
std::unique_ptr<QmmMatch> TryAffine(
    const llvm::DenseMap<mlir::Value, int>& main_args,
    const absl::flat_hash_set<int>& donated, mlir::Operation* dot) {
  std::optional<Reject> first;
  for (int qside : {1, 0}) {
    try {
      return TryAffineSide(main_args, donated, dot, qside);
    } catch (const Reject& e) {
      if (!first.has_value()) first = e;
    }
  }
  throw *first;
}

// qmm.py `_try_perchannel`: divide(dot(x, [shape ops](cvt(codes))),
// broadcast(scale)).
std::unique_ptr<QmmMatch> TryPerChannel(
    const llvm::DenseMap<mlir::Value, int>& main_args,
    const absl::flat_hash_set<int>& donated, mlir::Operation* div) {
  mlir::Value num = div->getOperand(0);
  mlir::Value den = div->getOperand(1);
  mlir::Operation* dot = Owner(num);
  if (dot == nullptr || OpName(dot) != "stablehlo.dot_general")
    Bail("divide numerator is not a dot");
  std::vector<mlir::Operation*> post;
  mlir::Value codes =
      Strip(dot->getOperand(1), /*shape_ops=*/true, /*converts=*/true, &post);
  if (!IsIntEl(codes)) Bail("dot rhs is not an integer code tensor");
  std::vector<int64_t> dims;
  bool found = false;
  mlir::Value scale = BcastChain(den, &dims, &found);
  if (!found) Bail("divisor is not a broadcast");
  if (!IsFloatEl(scale)) Bail("divisor is not float");
  const DotDims d = ReadDotDims(dot);
  if (!d.lb.empty() || !d.rb.empty()) Bail("batching dimensions");
  const int64_t nm =
      static_cast<int64_t>(ShapeOf(dot->getOperand(0)).size()) -
      static_cast<int64_t>(d.lc.size());
  for (int64_t x : dims) {
    // A divisor that varies along the M axis is not a weight scale.
    if (x < nm) Bail("divisor depends on the batch axis");
  }
  auto m = std::make_unique<QmmMatch>();
  m->recip = true;
  m->codes = codes;
  m->scale = scale;
  for (int64_t x : dims) m->bcast_dims.push_back(x - nm);
  for (mlir::Operation* o : post)
    if (IsShapeOp(OpName(o))) m->post.push_back(o);
  Finish(main_args, donated, m.get(), div, {dot->getOperand(1), den}, dot,
         /*qside=*/1, {dot}, {dot});
  return m;
}

// qmm.py `_prune`: an op can only be skipped if every consumer of every result
// is itself skipped (or is a rewritten root).  A candidate whose
// reconstruction ops cannot be skipped is dropped entirely.
void Prune(std::vector<std::unique_ptr<QmmMatch>>* cands) {
  std::vector<QmmMatch*> live;
  for (auto& m : *cands) live.push_back(m.get());
  llvm::DenseSet<mlir::Operation*> ops;
  while (true) {
    ops.clear();
    llvm::DenseSet<mlir::Operation*> roots;
    for (QmmMatch* m : live) {
      for (mlir::Operation* o : m->ops) ops.insert(o);
      roots.insert(m->root);
    }
    std::vector<mlir::Operation*> work(ops.begin(), ops.end());
    while (!work.empty()) {
      mlir::Operation* op = work.back();
      work.pop_back();
      if (!ops.contains(op)) continue;
      bool keep = true;
      for (mlir::Value r : op->getResults()) {
        for (mlir::Operation* u : r.getUsers()) {
          if (u->getNumResults() == 0 ||
              (!ops.contains(u) && !roots.contains(u))) {
            keep = false;
            break;
          }
        }
        if (!keep) break;
      }
      if (keep) continue;
      ops.erase(op);
      for (mlir::Value opd : op->getOperands()) {
        mlir::Operation* o = Owner(opd);
        if (o != nullptr && ops.contains(o)) work.push_back(o);
      }
    }
    std::vector<QmmMatch*> kept;
    for (QmmMatch* m : live) {
      bool ok = true;
      for (mlir::Operation* o : m->required)
        if (!ops.contains(o)) { ok = false; break; }
      if (!ok) {
        Debug(absl::StrCat("dropped ", m->name,
                           " (reconstruction is used elsewhere)"));
        continue;
      }
      kept.push_back(m);
    }
    if (kept.size() == live.size()) break;
    live = std::move(kept);
  }
  llvm::DenseSet<QmmMatch*> keep(live.begin(), live.end());
  for (auto& m : *cands) {
    if (!keep.contains(m.get())) {
      m->disabled = true;
      continue;
    }
    std::vector<mlir::Operation*> filtered;
    for (mlir::Operation* o : m->ops)
      if (ops.contains(o)) filtered.push_back(o);
    m->ops = std::move(filtered);
  }
}

void WalkBlocks(mlir::Block& block,
                const std::function<void(mlir::Block&)>& fn) {
  fn(block);
  for (mlir::Operation& op : block)
    for (mlir::Region& r : op.getRegions())
      for (mlir::Block& b : r.getBlocks()) WalkBlocks(b, fn);
}

// --------------------------------------------------------------------------
// packing (qmm.py, same section)
// --------------------------------------------------------------------------

mx::Shape ToMxShape(const std::vector<int64_t>& dims) {
  mx::Shape s;
  s.reserve(dims.size());
  for (int64_t d : dims) s.push_back(static_cast<mx::ShapeElem>(d));
  return s;
}

std::vector<int64_t> FromMxShape(const mx::Shape& s) {
  return std::vector<int64_t>(s.begin(), s.end());
}

mx::array U32(uint32_t v) { return mx::array(v, mx::uint32); }

// MLX's buffer cache, off for the duration of a pack (qmm.py `_NoCache`): a
// pack's dead buffers stay CLAIMED until some other allocation wants them, so
// the figure a memory watchdog reads is the pack's whole traffic rather than
// its live set (measured on gpt-oss-20b: 7.7 GB live, 15.9 GB claimed).
class NoCache {
 public:
  NoCache() : prev_(mx::set_cache_limit(0)) {}
  ~NoCache() {
    mx::clear_cache();
    mx::set_cache_limit(prev_);
  }
  NoCache(const NoCache&) = delete;
  NoCache& operator=(const NoCache&) = delete;

 private:
  size_t prev_;
};

// qmm.py `_replay` + `_to_nk`: a rhs-shaped tensor as the [(B,) N, K] matrix
// `quantized_matmul` wants, then flattened to [rows, K] -- every check and
// every packer below reads groups along the last axis and nothing else.
mx::array ToRows(mx::array x, const QmmMatch& m) {
  for (mlir::Operation* o : m.post) {
    if (OpName(o) == "stablehlo.reshape") {
      x = mx::reshape(x, ToMxShape(ShapeOf(o->getResult(0))));
    } else {
      std::vector<int64_t> perm = I64List(o, "permutation");
      std::vector<int> p(perm.begin(), perm.end());
      x = mx::transpose(x, p);
    }
  }
  if (FromMxShape(x.shape()) != m.rshape)
    Bail(absl::StrCat("the weight subtree evaluated to the wrong shape"));
  std::vector<int> perm(m.rperm.begin(), m.rperm.end());
  mx::array out = mx::contiguous(mx::reshape(
      mx::transpose(x, perm),
      mx::Shape{-1, static_cast<mx::ShapeElem>(m.K)}));
  mx::eval(out);
  return out;
}

// qmm.py `_group_const`.
bool GroupConst(const mx::array& x, int64_t gs) {
  const int64_t k = x.shape().back();
  if (k % gs) return false;
  mx::array v = mx::reshape(x, mx::Shape{-1,
                                         static_cast<mx::ShapeElem>(k / gs),
                                         static_cast<mx::ShapeElem>(gs)});
  mx::Shape lo(3, 0);
  mx::Shape hi = v.shape();
  hi[2] = 1;
  mx::array ok = mx::all(mx::equal(v, mx::slice(v, lo, hi)));
  mx::eval(ok);
  return ok.item<bool>();
}

// qmm.py `_group_heads`: the first element of each group, `[..., K] ->
// [..., K/gs]`.
mx::array GroupHeads(const mx::array& x, int64_t gs) {
  const int64_t k = x.shape().back();
  mx::Shape s(x.shape().begin(), x.shape().end() - 1);
  s.push_back(static_cast<mx::ShapeElem>(k / gs));
  s.push_back(static_cast<mx::ShapeElem>(gs));
  mx::array v = mx::reshape(x, s);
  return mx::contiguous(mx::take(v, 0, static_cast<int>(v.ndim()) - 1));
}

// qmm.py `_pick_group_heads`: the largest legal group size the whole weight
// allows, read off the per-32 heads.
int64_t PickGroupHeads(int64_t k, const mx::array& scale_heads,
                       const mx::array* zero_heads) {
  const int64_t g0 = kMinGroup;
  for (int64_t g : kGroupSizes) {
    if (g < g0 || k % g) continue;
    const int64_t r = g / g0;
    if (r == 1 || (GroupConst(scale_heads, r) &&
                   (zero_heads == nullptr || GroupConst(*zero_heads, r))))
      return g;
  }
  return 0;
}

// qmm.py `pack_codes`: unsigned codes `[..., K]` -> uint32 words.  MLX packs
// each row of the last axis as one contiguous little-endian bit stream, LSB
// first: element i occupies bits [i*bits, (i+1)*bits).
mx::array PackCodes(const mx::array& codes, int64_t bits) {
  if (32 % bits) Bail("pack_codes: bits does not divide 32");
  const int64_t per = 32 / bits;
  const int64_t k = codes.shape().back();
  if (k % per) Bail("pack_codes: K is not a multiple of the packing factor");
  mx::Shape s(codes.shape().begin(), codes.shape().end() - 1);
  s.push_back(static_cast<mx::ShapeElem>(k / per));
  s.push_back(static_cast<mx::ShapeElem>(per));
  mx::array c = mx::reshape(mx::astype(codes, mx::uint32), s);
  const int axis = static_cast<int>(c.ndim()) - 1;
  mx::array mask = U32((1u << bits) - 1);
  std::optional<mx::array> out;
  for (int64_t i = 0; i < per; i++) {
    // Defensive mask: an out-of-range code would otherwise spill into the NEXT
    // element's bits and corrupt an unrelated weight.
    mx::array v =
        mx::bitwise_and(mx::take(c, static_cast<int>(i), axis), mask);
    if (i) v = mx::left_shift(v, U32(static_cast<uint32_t>(i * bits)));
    out = out.has_value() ? mx::bitwise_or(*out, v) : v;
  }
  return *out;
}

bool Lossless(const mx::array& x, mx::Dtype dt) {
  mx::array ok = mx::all(
      mx::equal(mx::astype(mx::astype(x, dt), mx::float32), x));
  mx::eval(ok);
  return ok.item<bool>();
}

// qmm.py `_scale_bias`: MLX dequantizes `scales * q_hat + biases` with an
// UNSIGNED q_hat, so shifting the codes by any integer offset that makes them
// non-negative works as long as the shift is undone in the bias.
void ScaleBias(const mx::array& scales, const mx::array* zeros, int64_t offset,
               std::optional<mx::Dtype> scale_dtype, mx::array* out_scales,
               mx::array* out_biases) {
  mx::array s32 = mx::astype(mx::contiguous(scales), mx::float32);
  mx::array z32 = zeros == nullptr
                      ? mx::array(0.0f, mx::float32)
                      : mx::astype(mx::contiguous(*zeros), mx::float32);
  mx::array b32 = mx::negative(mx::multiply(
      s32, mx::add(z32, mx::array(static_cast<float>(offset), mx::float32))));
  if (scale_dtype.has_value() && *scale_dtype != mx::float32 &&
      kScaleWidth != "f32") {
    // Keep the source (bf16/f16) width when nothing is lost by it: it halves
    // scale traffic and keeps the output in the compute dtype.
    if (kScaleWidth == "source" ||
        (Lossless(s32, *scale_dtype) && Lossless(b32, *scale_dtype))) {
      *out_scales = mx::astype(s32, *scale_dtype);
      *out_biases = mx::astype(b32, *scale_dtype);
      return;
    }
  }
  *out_scales = s32;
  *out_biases = b32;
}

mx::Dtype UintView(mx::Dtype dt) {
  if (dt == mx::float32) return mx::uint32;
  if (dt == mx::bfloat16 || dt == mx::float16) return mx::uint16;
  Bail("MXFP4 values in an unsupported dtype");
}

// qmm.py `mxfp4_codes`: 4-bit codes for values that lie EXACTLY on the E2M1
// grid.  Works on the bit pattern rather than on the numbers so that the sign
// of a zero survives (code 8 is -0.0) and so that "exactly" means exactly.
mx::array Mxfp4Codes(const mx::array& values) {
  const mx::Dtype uint = UintView(values.dtype());
  mx::array bits = mx::view(mx::contiguous(values), uint);
  const int width = uint == mx::uint32 ? 32 : 16;
  mx::array sign = mx::right_shift(bits, mx::array(width - 1, uint));
  mx::array mag = mx::bitwise_and(
      bits, mx::array((int64_t{1} << (width - 1)) - 1, uint));
  mx::array grid = mx::view(
      mx::contiguous(mx::astype(
          mx::array(std::initializer_list<float>{
                        kE2M1Mags[0], kE2M1Mags[1], kE2M1Mags[2], kE2M1Mags[3],
                        kE2M1Mags[4], kE2M1Mags[5], kE2M1Mags[6], kE2M1Mags[7]}),
          values.dtype())),
      uint);
  mx::eval(grid);
  std::vector<int64_t> pattern(8);
  for (int i = 0; i < 8; i++) {
    pattern[i] = uint == mx::uint32
                     ? static_cast<int64_t>(grid.data<uint32_t>()[i])
                     : static_cast<int64_t>(grid.data<uint16_t>()[i]);
    for (int j = 0; j < i; j++)
      if (pattern[j] == pattern[i]) Bail("the E2M1 grid is not distinct");
  }
  mx::array codes = mx::zeros(mag.shape(), mx::uint8);
  mx::array ok = mx::equal(mag, mx::array(pattern[0], uint));
  for (int i = 1; i < 8; i++) {
    mx::array hit = mx::equal(mag, mx::array(pattern[i], uint));
    codes = mx::where(hit, mx::array(static_cast<uint8_t>(i), mx::uint8), codes);
    ok = mx::bitwise_or(ok, hit);
  }
  codes = mx::bitwise_or(
      codes, mx::left_shift(mx::astype(sign, mx::uint8),
                            mx::array(static_cast<uint8_t>(3), mx::uint8)));
  // One reduction, one sync: the per-element masks die with this call.
  mx::array good = mx::all(ok);
  mx::eval(codes, good);
  if (!good.item<bool>()) Bail("weight values are not on the MXFP4 grid");
  return codes;
}

// qmm.py `mxfp4_scale_bytes`: E8M0 bytes for per-group scales that are EXACT
// powers of two -- an f32 with a zero mantissa, so the byte IS the exponent
// field.  Fields 0 and 255 are rejected rather than encoded.
mx::array Mxfp4ScaleBytes(const mx::array& scales) {
  mx::array bits =
      mx::view(mx::contiguous(mx::astype(scales, mx::float32)), mx::uint32);
  mx::array exp = mx::bitwise_and(mx::right_shift(bits, U32(23)), U32(0xFF));
  mx::array bad = mx::any(mx::bitwise_or(
      mx::not_equal(mx::bitwise_and(bits, U32(0x807FFFFF)), U32(0)),
      mx::bitwise_or(mx::equal(exp, U32(0)), mx::equal(exp, U32(0xFF)))));
  mx::array out = mx::astype(exp, mx::uint8);
  mx::eval(out, bad);
  if (bad.item<bool>())
    Bail("MXFP4 group scales are not exact positive powers of two");
  return out;
}

// --------------------------------------------------------------------------
// regrouping an interleaved contraction axis (qmm.py, same section)
// --------------------------------------------------------------------------

mx::array Mix(const mx::array& u) {
  mx::array v = mx::multiply(u, U32(kMix));
  return mx::bitwise_xor(v, mx::right_shift(v, U32(15)));
}

// `x` widened to uint32 injectively (equal values -> equal words).
mx::array BitsU32(const mx::array& x) {
  if (mx::issubdtype(x.dtype(), mx::floating))
    return mx::view(mx::contiguous(mx::astype(x, mx::float32)), mx::uint32);
  return mx::astype(x, mx::uint32);
}

// qmm.py `_column_keys`: a per-column digest of `[rows, K]` as `[K, 2]` on the
// host.  Equal columns always digest equally; a collision can only MERGE two
// groups, which the exact group-constancy check downstream rejects if the
// merge was not legitimate.
void ColumnKeys(const mx::array& x, std::vector<uint32_t>* h1_out,
                std::vector<uint32_t>* h2_out) {
  const int64_t k = x.shape().back();
  const int64_t n = x.size() / std::max<int64_t>(k, 1);
  const int64_t step = std::max<int64_t>(
      1, std::min<int64_t>(n, kKeyChunk / std::max<int64_t>(k, 1)));
  mx::array flat = mx::reshape(x, mx::Shape{static_cast<mx::ShapeElem>(n),
                                            static_cast<mx::ShapeElem>(k)});
  mx::array h1 = mx::zeros(mx::Shape{static_cast<mx::ShapeElem>(k)}, mx::uint32);
  mx::array h2 = h1;
  for (int64_t lo = 0; lo < n; lo += step) {
    const int64_t hi = std::min<int64_t>(lo + step, n);
    mx::array u = BitsU32(mx::contiguous(mx::slice(
        flat, mx::Shape{static_cast<mx::ShapeElem>(lo), 0},
        mx::Shape{static_cast<mx::ShapeElem>(hi),
                  static_cast<mx::ShapeElem>(k)})));
    mx::array rows = mx::reshape(
        mx::astype(mx::arange(static_cast<double>(lo),
                              static_cast<double>(hi), 1.0, mx::float32),
                   mx::uint32),
        mx::Shape{static_cast<mx::ShapeElem>(hi - lo), 1});
    mx::array r = Mix(mx::add(mx::multiply(rows, U32(0x9E3779B9)), U32(1)));
    mx::array v = Mix(u);
    h1 = mx::add(h1, mx::sum(mx::multiply(v, mx::bitwise_or(r, U32(1))),
                             std::vector<int>{0}));
    h2 = mx::add(h2, mx::sum(Mix(mx::bitwise_xor(v, r)), std::vector<int>{0}));
    // Settle each pass so the chunk's intermediates die with it.
    mx::eval(h1, h2);
  }
  mx::eval(h1, h2);
  h1_out->assign(h1.data<uint32_t>(), h1.data<uint32_t>() + k);
  h2_out->assign(h2.data<uint32_t>(), h2.data<uint32_t>() + k);
}

// qmm.py `_regroup`: a permutation of the contraction axis that
// un-interleaves the groups, or none.  Permuting K on BOTH dot operands is
// exact, so clustering the columns by their (scale, zero) column pair and
// sorting by cluster recovers a layout that packs.
bool Regroup(int64_t k, const std::vector<const mx::array*>& maps,
             std::vector<int32_t>* perm) {
  if (k <= 0) return false;
  // One key per COLUMN, holding both digest words of every map: the columns
  // are what is being clustered, so the words of one have to sit together.
  std::vector<std::vector<uint32_t>> keys(static_cast<size_t>(k));
  for (const mx::array* x : maps) {
    if (x == nullptr) continue;
    std::vector<uint32_t> h1, h2;
    ColumnKeys(*x, &h1, &h2);
    if (static_cast<int64_t>(h1.size()) != k) return false;
    for (int64_t i = 0; i < k; i++) {
      keys[i].push_back(h1[i]);
      keys[i].push_back(h2[i]);
    }
  }
  if (keys[0].empty()) return false;
  // Stable first-occurrence ids: sorted-key order would reshuffle whole groups
  // for no reason.
  absl::flat_hash_map<std::string, int64_t> seen;
  std::vector<int64_t> ids(k);
  std::vector<int64_t> counts;
  for (int64_t c = 0; c < k; c++) {
    std::string key(reinterpret_cast<const char*>(keys[c].data()),
                    keys[c].size() * sizeof(uint32_t));
    auto it = seen.find(key);
    if (it == seen.end()) {
      const int64_t id = static_cast<int64_t>(counts.size());
      seen.emplace(std::move(key), id);
      counts.push_back(1);
      ids[c] = id;
    } else {
      counts[it->second]++;
      ids[c] = it->second;
    }
  }
  // Every cluster becomes a contiguous run of its own length; a legal group
  // size must divide all of them (then it also divides every run's offset).
  int64_t g = 0;
  for (int64_t len : counts) g = std::gcd(g, len);
  bool legal = false;
  for (int64_t s : kGroupSizes) legal = legal || (g != 0 && g % s == 0);
  if (!legal) return false;
  std::vector<int32_t> order(k);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int32_t a, int32_t b) { return ids[a] < ids[b]; });
  bool identity = true;
  for (int64_t i = 0; i < k; i++) identity = identity && order[i] == i;
  if (identity) return false;
  *perm = std::move(order);
  return true;
}

// qmm.py `_take_k`: `x[..., perm]`, materialized (these are full-weight-size
// maps).
mx::array TakeK(const mx::array& x, const mx::array& perm) {
  mx::array out =
      mx::contiguous(mx::take(x, perm, static_cast<int>(x.ndim()) - 1));
  mx::eval(out);
  return out;
}

mx::array HostPerm(const std::vector<int32_t>& perm) {
  void* buf = std::malloc(std::max<size_t>(1, perm.size() * sizeof(int32_t)));
  if (buf == nullptr) throw std::bad_alloc();
  std::memcpy(buf, perm.data(), perm.size() * sizeof(int32_t));
  return mx::array(buf, mx::Shape{static_cast<mx::ShapeElem>(perm.size())},
                   mx::int32, [](void* p) { std::free(p); });
}

// The finished pack, in the order `emit` reads the arrays back.
struct Pack {
  mx::array w;
  mx::array scales;
  std::optional<mx::array> biases;
  std::optional<mx::array> perm;
  int64_t gs = 0;
  int64_t bits = 0;
  int mode = 0;
};

// qmm.py `_join`: packed rows back into the shape `emit` reads them in.
mx::array Join(mx::array out, const QmmMatch& m) {
  if (!m.bshape.empty()) {
    mx::Shape s = ToMxShape(m.bshape);
    s.push_back(static_cast<mx::ShapeElem>(m.N));
    s.push_back(out.shape().back());
    out = mx::reshape(out, s);
  }
  mx::eval(out);
  return out;
}

// qmm.py `_build_mxfp4_pack`.
Pack BuildMxfp4(const QmmMatch& m, mx::array scale_map, mx::array values) {
  const int64_t gs = kMxfp4Group;
  if (m.K % gs)
    Bail(absl::StrFormat("K=%d is not a multiple of %d", m.K, gs));
  std::optional<mx::array> perm;
  if (!GroupConst(scale_map, gs)) {
    // The same interleaving story as the affine path: permuting the
    // contraction axis on BOTH operands leaves the dot unchanged.
    std::vector<int32_t> p;
    if (Regroup(m.K, {&scale_map}, &p)) {
      perm = HostPerm(p);
      scale_map = TakeK(scale_map, *perm);
    }
    if (!GroupConst(scale_map, gs))
      Bail(absl::StrFormat("MXFP4 scales are not constant within a group of %d",
                           gs));
  }
  Pack pk{mx::array(0.0f), mx::array(0.0f), std::nullopt, std::nullopt, gs, 4,
          1};
  mx::array sb = Mxfp4ScaleBytes(GroupHeads(scale_map, gs));
  if (perm.has_value()) values = TakeK(values, *perm);
  // The E2M1 nibble order is MLX's: element i of a row occupies bits
  // [4i, 4i+4) of the little-endian uint32 stream.
  mx::array w = PackCodes(Mxfp4Codes(values), 4);
  mx::eval(w, sb);
  pk.w = Join(w, m);
  pk.scales = Join(sb, m);
  pk.perm = perm;
  return pk;
}

// qmm.py `_build_affine_pack`.
Pack BuildAffine(const QmmMatch& m, mx::array codes,
                 std::optional<mx::array> scale_map,
                 std::optional<mx::array> zero_map,
                 std::optional<mx::array> recip_scale) {
  // The code range first: the offset that makes the codes unsigned has to be
  // one number for the whole weight.
  mx::array lo_a = mx::min(codes, false);
  mx::array hi_a = mx::max(codes, false);
  mx::eval(lo_a, hi_a);
  const int64_t lo = mx::astype(lo_a, mx::int32).item<int>();
  const int64_t hi = mx::astype(hi_a, mx::int32).item<int>();
  int64_t bits;
  if (hi - lo < 16) {
    bits = 4;
  } else if (hi - lo < 256) {
    bits = 8;
  } else {
    Bail(absl::StrFormat("codes span [%d, %d]: more than 8 bits", lo, hi));
  }
  // Prefer the signed-code convention (offset = 2^(bits-1)): its bias is a
  // power-of-two multiple of the scale, which is what keeps a zero-point-free
  // quantization exactly representable in bf16.
  int64_t offset = int64_t{1} << (bits - 1);
  if (lo + offset < 0 || hi + offset >= (int64_t{1} << bits)) offset = -lo;

  std::optional<mx::Dtype> scale_dtype;
  std::optional<mx::array> perm;
  mx::array scales = mx::array(0.0f);
  std::optional<mx::array> zeros;
  int64_t gs = 0;
  if (m.recip) {
    mx::array s = *recip_scale;
    // A reciprocal is rarely exact in bf16, so "auto" widens to f32 here.
    scale_dtype = s.dtype();
    std::vector<int64_t> dims = m.bcast_dims;
    std::vector<int64_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());
    if (dims != sorted_dims) {
      std::vector<int> order(dims.size());
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(),
                       [&](int a, int b) { return dims[a] < dims[b]; });
      s = mx::transpose(s, order);
      dims = sorted_dims;
    }
    mx::Shape interim(m.nshape.size(), 1);
    for (size_t i = 0; i < dims.size(); i++) {
      if (dims[i] < 0 || dims[i] >= static_cast<int64_t>(interim.size()))
        Bail("the per-channel scale does not broadcast over the N axis");
      interim[dims[i]] = s.shape()[i];
    }
    s = mx::reshape(
        mx::broadcast_to(mx::reshape(s, interim), ToMxShape(m.nshape)),
        mx::Shape{static_cast<mx::ShapeElem>(m.N), 1});
    // The graph divides the OUTPUT by a per-output-channel scale; fold the
    // reciprocal into the weight scale.
    scales = mx::divide(mx::array(1.0f, mx::float32),
                        mx::astype(s, mx::float32));
    for (int64_t g : kGroupSizes)
      if (m.K % g == 0) { gs = g; break; }
    if (gs == 0) Bail("K has no legal group size");
    scales = mx::broadcast_to(
        scales, mx::Shape{static_cast<mx::ShapeElem>(m.N),
                          static_cast<mx::ShapeElem>(m.K / gs)});
  } else {
    const int64_t g0 = kMinGroup;
    mx::array smap = *scale_map;
    if (!GroupConst(smap, g0) ||
        (zero_map.has_value() && !GroupConst(*zero_map, g0))) {
      // The groups may still be there, interleaved: recover the permutation
      // that makes them contiguous and re-verify EXACTLY (the clustering is
      // only a proposal -- the constancy check on the permuted maps is what
      // the pack's exactness rests on).
      std::vector<const mx::array*> maps{&smap};
      if (zero_map.has_value()) maps.push_back(&*zero_map);
      std::vector<int32_t> p;
      std::string note;
      if (Regroup(m.K, maps, &p)) {
        perm = HostPerm(p);
        // Rebind as each permuted copy lands: these are full weight size.
        smap = TakeK(smap, *perm);
        if (zero_map.has_value()) zero_map = TakeK(*zero_map, *perm);
        note = " (even regrouped)";
      }
      if (!GroupConst(smap, g0) ||
          (zero_map.has_value() && !GroupConst(*zero_map, g0)))
        Bail(absl::StrCat("scales/zeros are not constant within any group",
                          note));
    }
    scale_dtype = smap.dtype();
    scales = GroupHeads(smap, g0);
    if (zero_map.has_value()) zeros = GroupHeads(*zero_map, g0);
    mx::eval(scales);
    if (zeros.has_value()) mx::eval(*zeros);
    scale_map.reset();
    zero_map.reset();
    gs = PickGroupHeads(m.K, scales, zeros.has_value() ? &*zeros : nullptr);
    if (gs == 0) Bail("scales/zeros are not constant within any group");
    if (gs != g0) {
      scales = GroupHeads(scales, gs / g0);
      if (zeros.has_value()) zeros = GroupHeads(*zeros, gs / g0);
    }
    if (zeros.has_value()) {
      // In f32 throughout: the zero map may be an integer tensor, and MLX
      // refuses to compare one against out-of-range literals.
      zeros = mx::astype(*zeros, mx::float32);
      mx::array whole = mx::all(mx::equal(*zeros, mx::round(*zeros, 0)));
      mx::array big = mx::any(mx::greater(mx::abs(*zeros),
                                          mx::array(32768.0f, mx::float32)));
      mx::array zlo = mx::min(*zeros, false);
      mx::array zhi = mx::max(*zeros, false);
      mx::eval(whole, big, zlo, zhi);
      if (!whole.item<bool>()) Bail("zero points are not integers");
      if (big.item<bool>()) Bail("zero points out of range");
      if (m.has_sub_range) {
        // The graph subtracts the zero point in INTEGER arithmetic, which
        // wraps; the rewrite computes it exactly.  Only fuse when nothing can.
        const int64_t zl = static_cast<int64_t>(zlo.item<float>());
        const int64_t zh = static_cast<int64_t>(zhi.item<float>());
        if ((lo - zh) < m.sub_lo || (hi - zl) > m.sub_hi)
          Bail("integer zero-point subtraction can wrap");
      }
    }
    mx::eval(scales);
    if (zeros.has_value()) mx::eval(*zeros);
  }

  if (perm.has_value()) codes = TakeK(codes, *perm);
  mx::array w = PackCodes(
      mx::add(mx::astype(mx::contiguous(codes), mx::int32),
              mx::array(static_cast<int>(offset), mx::int32)),
      bits);
  mx::eval(w);
  mx::array out_scales = mx::array(0.0f);
  mx::array out_biases = mx::array(0.0f);
  ScaleBias(scales, zeros.has_value() ? &*zeros : nullptr, offset, scale_dtype,
            &out_scales, &out_biases);
  // Materialize: a lazy packed weight would pin the whole reconstruction graph
  // (and its full-size intermediates) for the life of the cache.
  mx::eval(out_scales, out_biases);
  Pack pk{Join(w, m), Join(out_scales, m), Join(out_biases, m), perm, gs, bits,
          0};
  return pk;
}

}  // namespace

// --------------------------------------------------------------------------
// the public surface
// --------------------------------------------------------------------------

bool QmmEnabled() {
  static const bool on = !EnvOff("METALJAX_QMM");
  return on;
}

bool RecognizeEnabled() {
  static const bool on = !EnvOff("METALJAX_RECOGNIZE");
  return on;
}

mlir::Value HoistInvariant(mlir::Value v) { return Hoist(v); }

void RewritePlan::rebuild() {
  skip.clear();
  qmm_roots.clear();
  sdpa_roots.clear();
  moe_roots.clear();
  for (const auto& m : qmm) {
    if (m->disabled) continue;
    // An ABSORBED match still owns its absorbed ops -- its weight is packed
    // and read by the expert gather that took it over -- but its dense
    // quantized_matmul is never emitted.
    if (!m->absorbed) qmm_roots[m->root] = m.get();
    for (mlir::Operation* o : m->ops) skip.insert(o);
  }
  for (const auto& m : sdpa) {
    sdpa_roots[m->root] = m.get();
    for (mlir::Operation* o : m->ops) skip.insert(o);
  }
  for (const auto& m : moe) {
    if (m->disabled) continue;
    moe_roots[m->root] = m.get();
    for (mlir::Operation* o : m->ops) skip.insert(o);
  }
}

void AnalyzeQmm(mlir::func::FuncOp fn, const absl::flat_hash_set<int>& donated,
                RewritePlan* plan) {
  if (!QmmEnabled()) return;
  if (fn.getBody().getBlocks().size() != 1) return;
  mlir::Block& main = fn.getBody().front();
  llvm::DenseMap<mlir::Value, int> main_args;
  for (mlir::BlockArgument a : main.getArguments())
    main_args[a] = static_cast<int>(a.getArgNumber());

  std::vector<std::unique_ptr<QmmMatch>> cands;
  WalkBlocks(main, [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      const std::string name = OpName(&op);
      try {
        if (name == "stablehlo.dot_general") {
          cands.push_back(TryAffine(main_args, donated, &op));
        } else if (name == "stablehlo.divide") {
          cands.push_back(TryPerChannel(main_args, donated, &op));
        }
      } catch (const Reject&) {
        // Not a quantized matmul: it lowers as itself.
      }
    }
  });
  if (cands.empty()) return;
  // A candidate whose result feeds another candidate's operand subtree is
  // evaluated during that one's packing prologue; it cannot also be rewritten.
  llvm::DenseSet<mlir::Operation*> absorbed;
  for (const auto& m : cands)
    for (mlir::Operation* o : m->ops) absorbed.insert(o);
  std::vector<std::unique_ptr<QmmMatch>> live;
  for (auto& m : cands) {
    if (absorbed.contains(m->root)) continue;
    live.push_back(std::move(m));
  }
  if (live.empty()) return;
  Prune(&live);
  for (auto& m : live) {
    if (m->disabled) continue;
    plan->qmm.push_back(std::move(m));
  }
  if (!plan->qmm.empty())
    Debug(absl::StrCat(plan->qmm.size(), " quantized matmul(s) recognized"));
  plan->rebuild();
}

absl::Status BuildQmmPacks(RewritePlan* plan, const SubtreeEval& eval) {
  plan->packs.clear();
  absl::flat_hash_set<int> args;
  for (auto& m : plan->qmm) {
    if (m->disabled) continue;
    try {
      NoCache no_cache;
      // The three operand subtrees, evaluated on this execute's buffers.  A
      // recip match's scale is NOT weight-shaped -- the graph divides the
      // OUTPUT by it -- so it is evaluated on its own.
      std::vector<mlir::Value> want{m->codes};
      if (!m->recip) want.push_back(m->scale);
      if (m->zero != nullptr) want.push_back(m->zero);
      if (m->recip) want.push_back(m->scale);
      absl::StatusOr<std::vector<mx::array>> got = eval(want, {});
      if (!got.ok()) Bail(std::string(got.status().message()));
      size_t at = 0;
      mx::array codes = ToRows((*got)[at++], *m);
      std::optional<mx::array> scale_map, zero_map, recip_scale;
      if (!m->recip) scale_map = ToRows((*got)[at++], *m);
      if (m->zero != nullptr) zero_map = ToRows((*got)[at++], *m);
      if (m->recip) recip_scale = (*got)[at++];
      Pack pk = m->mode == 1
                    ? BuildMxfp4(*m, *scale_map, codes)
                    : BuildAffine(*m, codes, scale_map, zero_map, recip_scale);
      m->gs = pk.gs;
      m->bits = pk.bits;
      m->mode = pk.mode;
      m->has_perm = pk.perm.has_value();
      m->slot = static_cast<int>(plan->packs.size());
      plan->packs.push_back(pk.w);
      plan->packs.push_back(pk.scales);
      if (pk.biases.has_value()) plan->packs.push_back(*pk.biases);
      if (pk.perm.has_value()) plan->packs.push_back(*pk.perm);
      m->nvals = static_cast<int>(plan->packs.size()) - m->slot;
      for (int i : m->arg_indices) args.insert(i);
      Debug(absl::StrFormat("packed %s mode=%s bits=%d group=%d%s",
                            m->name, pk.mode == 1 ? "mxfp4" : "affine",
                            pk.bits, pk.gs,
                            pk.perm.has_value() ? " regrouped" : ""));
    } catch (const Reject& e) {
      m->disabled = true;
      Debug(absl::StrCat(m->name, " falls back to the literal chain (",
                         e.why, ")"));
    } catch (const std::exception& e) {
      m->disabled = true;
      Debug(absl::StrCat(m->name, " falls back to the literal chain (",
                         e.what(), ")"));
    }
  }
  plan->pack_args.assign(args.begin(), args.end());
  std::sort(plan->pack_args.begin(), plan->pack_args.end());
  plan->rebuild();
  // The reconstruction ran at full weight size; its intermediates are dead
  // now and Metal counts live buffers, not bytes.
  gc_collect();
  mx::clear_cache();
  return absl::OkStatus();
}

}  // namespace metaljax
