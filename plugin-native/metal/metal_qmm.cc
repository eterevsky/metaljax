/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

`src/metaljax/qmm.py`, as a pass over the parsed StableHLO.

The two halves are the Python module's two halves, and neither is re-designed
here: `AnalyzeQmm` is its structural matching (`_try_affine`, `_try_perchannel`,
`_finish`, `_prune`) and touches no value; `BuildQmmPacks` is its first-execute
verification and packing (`_build_affine_pack`, `_build_mxfp4_pack`), on the
concrete buffers of this execute.  MLX's buffer cache is off while a pack runs
(a pack's dead buffers are claimed memory a watchdog reads), and every
exactness check is exact.  A check that fails DISABLES that one dot: its chain
then lowers literally, as it did before this file existed.

P19 adds the two things the P17 port left out, which is what a 20-GB weight
set costs in PEAK MEMORY rather than in answers:

* **`RowSource`** is qmm.py's `_Source`: the operand subtrees are evaluated one
  BLOCK of the weight's rows at a time and packed as they go, so the
  reconstruction -- several times the size of the weight it packs -- never
  exists at once.  What is derived from a block (a 4-bit code, one scale per
  group) is an eighth of it, and every element is still read exactly once by
  the same exact checks: blocking changes WHEN the verification sees a value,
  never whether.  A subtree not provably row-local raises `NotBlockable` and
  the caller retries whole, where `blocks()` yields one block covering
  everything and the packers above are unchanged.
* **The build cache** keys a finished pack on a canonical serialization of its
  reconstruction plus the identity of the buffers that subtree reads, so the
  prefill and decode executables of one model -- separate programs, separate
  plans, separate empty pack sets -- build each weight once between them.

The rules that decide which values a block may narrow are qmm.py `_Source._op`'s
and they live in `RowLocalOps` + `RowSource::DemandOp`; the LOWERING asserts
their consequence a second time (`Lowering::LowerOp`'s row-block guard), because
a wrong narrowing is a weight that passes every exactness check on the wrong
rows.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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

// qmm.py `_BLOCK_ELEMS` (METALJAX_QMM_BLOCK): weight elements per block.  The
// reconstruction chain is much wider than its result -- jax's `take` wrapper
// alone carries three int32 copies of the index tensor -- so a block costs
// ~16 bytes per element while it runs; 16M elements keeps that under a few
// hundred MB and still gives every kernel millions of lanes of work.
const int64_t kBlockElems = [] {
  const char* v = std::getenv("METALJAX_QMM_BLOCK");
  if (v == nullptr) return int64_t{1} << 24;
  const int64_t n = std::atoll(v);
  return n > 0 ? n : (int64_t{1} << 24);
}();
// qmm.py `_WHOLE_MAX`: a ceiling on a value the blocking decides is
// block-INDEPENDENT (a decode table, a splat constant).  Those are
// re-evaluated for every block, so a big one means the blocking misread the
// graph and evaluating whole is honest.
constexpr int64_t kWholeMax = int64_t{1} << 22;
// qmm.py `_MAX_BUILT` (METALJAX_QMM_BUILD_CACHE): packs retained by the
// cross-executable build cache; 0 turns the reuse off entirely (every
// executable rebuilds, which is what P17 did).
const int64_t kMaxBuilt = [] {
  const char* v = std::getenv("METALJAX_QMM_BUILD_CACHE");
  if (v == nullptr) return int64_t{512};
  const int64_t n = std::atoll(v);
  return n >= 0 ? n : int64_t{512};
}();
// qmm.py `_FP_DENSE_ELEMS` / `_FP_ATTR_CHARS`: what a fingerprint may carry.
// A weight subtree's constants are decode tables and thresholds; thousands of
// elements means the weight itself is baked into the graph, and hashing that
// costs more than the build it would save.
constexpr int64_t kFpDenseElems = 1024;
constexpr size_t kFpAttrChars = 1u << 16;

// A candidate does not match (or cannot be trusted): run it literally.
struct Reject {
  std::string why;
};
[[noreturn]] void Bail(const std::string& why) { throw Reject{why}; }

// qmm.py `_NotBlockable`: this subtree cannot be evaluated one row block at a
// time.  NOT a `Reject` -- the weight still packs, from a whole evaluation.
struct NotBlockable {
  std::string why;
};
[[noreturn]] void NoBlock(const std::string& why) { throw NotBlockable{why}; }

// qmm.py `_NoFingerprint`: this reconstruction cannot be serialized exactly,
// so it is not build-cached.  Declining is free, and it is the only thing
// standing between "we can prove it" and a wrong weight.
struct NoFingerprint {
  std::string why;
};
[[noreturn]] void NoFp(const std::string& why) { throw NoFingerprint{why}; }

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

// --------------------------------------------------------------------------
// row-blocked evaluation of the operand subtrees (qmm.py `_Source`)
// --------------------------------------------------------------------------

// qmm.py `_replay_rows`: `_replay` on ONE block -- the shape ops between the
// reconstruction and the dot must leave the blocked axis be.
mx::array ReplayRows(mx::array x, const std::vector<mlir::Operation*>& post,
                     int64_t lead, int64_t c) {
  for (mlir::Operation* o : post) {
    if (OpName(o) == "stablehlo.reshape") {
      std::vector<int64_t> src = ShapeOf(o->getOperand(0));
      std::vector<int64_t> out = ShapeOf(o->getResult(0));
      if (src.empty() || out.empty() || src[0] != lead || out[0] != lead ||
          Prod(std::vector<int64_t>(src.begin() + 1, src.end())) !=
              Prod(std::vector<int64_t>(out.begin() + 1, out.end())))
        NoBlock("a reshape below the weight mixes rows");
      out[0] = c;
      x = mx::reshape(x, ToMxShape(out));
    } else {
      std::vector<int64_t> perm = I64List(o, "permutation");
      if (perm.empty() || perm[0] != 0)
        NoBlock("a transpose below the weight moves rows");
      std::vector<int> p(perm.begin(), perm.end());
      x = mx::transpose(x, p);
    }
  }
  return x;
}

// qmm.py `_ROW_LOCAL`, plus the ops whose row-locality is a rule about their
// dimension attributes rather than a property of the handler.  A `Lowering`
// consults this to assert that nothing else was ever narrowed.
const absl::flat_hash_set<std::string>& RowLocalNames() {
  // Elementwise ops, whose handler is a pure function of the arrays it is
  // handed and never of the shape the IR declares: feeding it a slice of the
  // leading axis produces exactly that slice of its result.
  static const auto* set = new absl::flat_hash_set<std::string>{
      "stablehlo.abs", "stablehlo.add", "stablehlo.and", "stablehlo.cbrt",
      "stablehlo.ceil", "stablehlo.clamp", "stablehlo.compare",
      "stablehlo.convert", "stablehlo.cosine",
      "stablehlo.count_leading_zeros", "stablehlo.divide",
      "stablehlo.exponential", "stablehlo.exponential_minus_one",
      "stablehlo.floor", "stablehlo.is_finite", "stablehlo.log",
      "stablehlo.log_plus_one", "stablehlo.logistic", "stablehlo.maximum",
      "stablehlo.minimum", "stablehlo.multiply", "stablehlo.negate",
      "stablehlo.not", "stablehlo.or", "stablehlo.popcnt", "stablehlo.power",
      "stablehlo.reduce_precision", "stablehlo.remainder",
      "stablehlo.round_nearest_afz", "stablehlo.round_nearest_even",
      "stablehlo.rsqrt", "stablehlo.select", "stablehlo.shift_left",
      "stablehlo.shift_right_arithmetic", "stablehlo.shift_right_logical",
      "stablehlo.sign", "stablehlo.sine", "stablehlo.sqrt",
      "stablehlo.subtract", "stablehlo.tanh", "stablehlo.xor",
      // ...and the seven whose leading axis survives under a condition
      // `RowSource::DemandOp` checks before it narrows anything.
      "stablehlo.reshape", "stablehlo.broadcast_in_dim", "stablehlo.slice",
      "stablehlo.transpose", "stablehlo.concatenate", "stablehlo.gather",
      "stablehlo.reduce",
      // A call emits no entry at all -- the lowering splices the callee's
      // body into the frame and aliases the results -- so narrowing one is
      // narrowing the values it forwards, each of which is guarded on its
      // own.  jax's `take` wrapper is a call, and it is on the critical path
      // of every MXFP4 weight.
      "func.call", "stablehlo.composite"};
  return *set;
}

// qmm.py `_Source`.  The demand analysis (`Demand`/`DemandOp`) decides which
// values of a subtree carry the weight's row axis; `Rows` then asks the
// lowering for a Program narrowed to those values and runs it once per block.
//
// `row` is a DEMAND, passed down from the weight: a chain read elementwise
// wants a block of each operand, a table gathered through wants all of it.
// Stage 1 keys its evaluation on (value, demand) and may read one value both
// ways; a tape has ONE slot per value, so reading a value both ways declines
// the blocking here instead (the weight then packs whole).  Nothing in reach
// does it: the case Stage 1's comment is about -- a 16-entry decode table in
// a model with 16 experts -- is two different values.
class RowSource {
 public:
  RowSource(const QmmMatch& m, const PackContext& ctx) : m_(m), ctx_(ctx) {
    for (mlir::BlockArgument a : ctx.main->getArguments())
      main_args_[a] = static_cast<int>(a.getArgNumber());
    // qmm.py `_blocking`: blocks are slices of the weight's LEADING axis.
    // That axis is the row axis of the [(B,) N, K] matrix quantized_matmul
    // wants only when the dot needs no transpose to get there, and when it is
    // not part of the contraction (`per % K`: whole rows have to live inside
    // one block, since the group scales and the packed nibble stream both run
    // along K).
    lead_ = m.rshape.empty() ? 0 : m.rshape[0];
    std::vector<int64_t> ramp(m.rshape.size());
    std::iota(ramp.begin(), ramp.end(), 0);
    if (m.rperm != ramp || lead_ < 2) return;
    const int64_t per =
        Prod(std::vector<int64_t>(m.rshape.begin() + 1, m.rshape.end()));
    if (per <= 0 || m.K <= 0 || per % m.K) return;
    const int64_t step = std::max<int64_t>(1, kBlockElems / per);
    if (step < lead_) step_ = step;
  }

  bool blocked() const { return step_ > 0; }

  void unblock() {
    step_ = 0;
    whole_.clear();
    plans_.clear();
    // ...and the narrowed Programs, which hold their constants' arrays: the
    // whole path is about to build the widest thing this weight ever needs.
    progs_.clear();
  }

  std::vector<std::pair<int64_t, int64_t>> blocks() const {
    if (step_ <= 0) return {{0, lead_}};
    std::vector<std::pair<int64_t, int64_t>> out;
    for (int64_t lo = 0; lo < lead_; lo += step_)
      out.emplace_back(lo, std::min(lo + step_, lead_));
    return out;
  }

  // Forget a whole-evaluated subtree (it is full weight size).
  void Drop(mlir::Value v) {
    if (v != nullptr) whole_.erase(v.getAsOpaquePointer());
  }

  // `value`'s rows for this block, as a `[rows, K]` matrix.  Always
  // two-dimensional, batching dimensions included in the row count: every
  // check and every packer reads groups along the last axis and nothing else,
  // and the packed arrays are folded back into `[(B,) N, ...]` once, at the
  // end (`Join`).
  mx::array Rows(mlir::Value value, int64_t lo, int64_t hi) {
    if (step_ <= 0) {
      auto it = whole_.find(value.getAsOpaquePointer());
      if (it != whole_.end()) return it->second;
      mx::array got = ToRows(Whole(value), m_);
      whole_.insert({value.getAsOpaquePointer(), got});
      return got;
    }
    const int64_t c = hi - lo;
    const Plan& p = PlanFor(value);
    std::shared_ptr<Program> prog = ProgramFor(value, p, c);
    std::vector<mx::array> ins = *ctx_.args;
    for (int i : p.blocked_args) {
      const mx::array& a = ins[i];
      mx::Shape start(a.ndim(), 0), stop = a.shape();
      start[0] = static_cast<mx::ShapeElem>(lo);
      stop[0] = static_cast<mx::ShapeElem>(hi);
      ins[i] = mx::contiguous(mx::slice(a, start, stop));
    }
    std::vector<mx::array> outs = prog->run(std::move(ins));
    if (outs.size() != 1) NoBlock("a blocked cone with several outputs");
    mx::array x = ReplayRows(outs[0], m_.post, lead_, c);
    std::vector<int64_t> want{c};
    want.insert(want.end(), m_.rshape.begin() + 1, m_.rshape.end());
    if (FromMxShape(x.shape()) != want)
      NoBlock("a block evaluated to the wrong shape");
    mx::array out = mx::contiguous(
        mx::reshape(x, mx::Shape{-1, static_cast<mx::ShapeElem>(m_.K)}));
    mx::eval(out);
    return out;
  }

  // A value that is NOT weight-shaped (the per-channel form's scale divides
  // the OUTPUT), which is only ever read whole.
  mx::array Whole(mlir::Value value) {
    absl::StatusOr<std::vector<mx::array>> got = ctx_.eval({value}, {});
    if (!got.ok()) Bail(std::string(got.status().message()));
    if (got->size() != 1) Bail("a cone with several outputs");
    return (*got)[0];
  }

 private:
  struct Plan {
    llvm::DenseSet<mlir::Value> blocked;
    std::vector<int> blocked_args;   // @main argument positions, sorted
  };

  const Plan& PlanFor(mlir::Value root) {
    auto it = plans_.find(root.getAsOpaquePointer());
    if (it != plans_.end()) return it->second;
    Plan p;
    demand_.clear();
    binds_.clear();
    bodies_.clear();
    Demand(root, /*row=*/true, &p);
    // A callee is INLINED whole by the lowering, dead ops included, so an op
    // this walk never reached can still be handed a block.  Sweeping the
    // bodies here turns that into a fall-back to the whole evaluation rather
    // than a decline of the cone (which would disable the dot entirely).
    for (mlir::Block* b : bodies_) {
      for (mlir::Operation& o : *b) {
        if (o.hasTrait<mlir::OpTrait::IsTerminator>()) continue;
        bool reads = false;
        for (mlir::Value x : o.getOperands())
          reads = reads || p.blocked.contains(x);
        if (!reads) continue;
        for (mlir::Value r : o.getResults())
          if (!p.blocked.contains(r))
            NoBlock(absl::StrCat("a callee op (", OpName(&o),
                                 ") reads the block and does not keep it"));
      }
    }
    for (const auto& [v, i] : main_args_)
      if (p.blocked.contains(v)) p.blocked_args.push_back(i);
    std::sort(p.blocked_args.begin(), p.blocked_args.end());
    return plans_.insert({root.getAsOpaquePointer(), std::move(p)})
        .first->second;
  }

  std::shared_ptr<Program> ProgramFor(mlir::Value root, const Plan& p,
                                      int64_t c) {
    auto key = std::make_pair(root.getAsOpaquePointer(), c);
    auto it = progs_.find(key);
    if (it != progs_.end()) return it->second;
    absl::StatusOr<std::shared_ptr<Program>> prog =
        ctx_.blocked_cone({root}, p.blocked, c);
    // A cone the tape cannot build from a NARROWED graph is exactly what
    // `NotBlockable` is for: the caller retries whole, where the same cone is
    // the one P17 already builds.
    if (!prog.ok()) NoBlock(std::string(prog.status().message()));
    progs_.insert({key, *prog});
    return *prog;
  }

  void Demand(mlir::Value v, bool row, Plan* p) {
    auto it = demand_.find(v);
    if (it != demand_.end()) {
      if (it->second != row)
        NoBlock("a value read both whole and one block at a time");
      return;
    }
    demand_[v] = row;
    if (row) p->blocked.insert(v);
    if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      auto arg = main_args_.find(v);
      if (arg != main_args_.end()) {
        if (row) {
          std::vector<int64_t> s = ShapeOf(v);
          if (s.empty() || s[0] != lead_)
            NoBlock(absl::StrCat("argument ", arg->second,
                                 " does not carry the blocked axis"));
        }
        return;
      }
      auto bound = binds_.find(v);
      if (bound != binds_.end()) {
        // A callee parameter: resolve the demand in the caller.
        Demand(bound->second, row, p);
        return;
      }
      mlir::Value outer = Hoist(v);
      if (outer == v) Bail("unbound value in operand subtree");
      Demand(outer, row, p);
      return;
    }
    mlir::Operation* o = Owner(v);
    if (o == nullptr) Bail("unbound value in operand subtree");
    for (mlir::Value r : o->getResults()) {
      if (r == v) continue;
      auto seen = demand_.find(r);
      if (seen != demand_.end() && seen->second != row)
        NoBlock("an op whose results are read both ways");
      demand_[r] = row;
      if (row) p->blocked.insert(r);
    }
    DemandOp(o, row, p);
  }

  void DemandOp(mlir::Operation* o, bool row, Plan* p) {
    const std::string name = OpName(o);
    if (!row) {
      // Block-independent: the declared shapes are the real ones, so the
      // cone lowers this op exactly as it always did.  What it may NOT be is
      // big -- it is re-evaluated for every block, and a big one means the
      // blocking misread the graph.
      for (mlir::Value r : o->getResults()) {
        if (Prod(ShapeOf(r)) > kWholeMax)
          NoBlock(absl::StrCat(name, " is block-independent but produces ",
                               Prod(ShapeOf(r)), " elements"));
      }
      for (mlir::Value x : o->getOperands()) Demand(x, false, p);
      return;
    }
    for (mlir::Value r : o->getResults()) {
      std::vector<int64_t> s = ShapeOf(r);
      if (s.empty() || s[0] != lead_)
        NoBlock(absl::StrCat(name, " does not keep the blocked axis"));
    }
    if (name == "stablehlo.reshape") {
      std::vector<int64_t> src = ShapeOf(o->getOperand(0));
      std::vector<int64_t> out = ShapeOf(o->getResult(0));
      if (src.empty() || src[0] != lead_ ||
          Prod(std::vector<int64_t>(src.begin() + 1, src.end())) !=
              Prod(std::vector<int64_t>(out.begin() + 1, out.end())))
        NoBlock("reshape mixes the blocked axis");
      Demand(o->getOperand(0), true, p);
    } else if (name == "stablehlo.broadcast_in_dim") {
      std::vector<int64_t> dims = I64List(o, "broadcast_dimensions");
      std::vector<int64_t> src = ShapeOf(o->getOperand(0));
      // The operand is blocked only when its own leading axis IS the
      // result's; a size-1 leading axis expanded to the full extent is one
      // row repeated, which every block can read whole.
      const bool sub =
          !dims.empty() && !src.empty() && dims[0] == 0 && src[0] == lead_;
      // ...and when it is NOT, nothing it contributes to the result's leading
      // axis may be the blocked extent (qmm.py's `interim[0] not in (1, c)`).
      if (!sub) {
        for (size_t i = 0; i < dims.size() && i < src.size(); i++)
          if (dims[i] == 0 && src[i] != 1)
            NoBlock("broadcast expands the blocked axis");
      }
      Demand(o->getOperand(0), sub, p);
    } else if (name == "stablehlo.slice") {
      std::vector<int64_t> starts = I64List(o, "start_indices");
      std::vector<int64_t> limits = I64List(o, "limit_indices");
      std::vector<int64_t> strides = I64List(o, "strides");
      if (starts.empty() || starts[0] != 0 || limits[0] != lead_ ||
          strides[0] != 1)
        NoBlock("slice cuts the blocked axis");
      Demand(o->getOperand(0), true, p);
    } else if (name == "stablehlo.gather") {
      auto ga = mlir::dyn_cast<mlir::stablehlo::GatherOp>(o);
      if (!ga) NoBlock("a gather in an unexpected form");
      mlir::stablehlo::GatherDimensionNumbersAttr d = ga.getDimensionNumbers();
      std::vector<int64_t> offset(d.getOffsetDims().begin(),
                                  d.getOffsetDims().end());
      std::vector<int64_t> idx = ShapeOf(o->getOperand(1));
      // Only the INDICES may carry the block, which is the shape of a table
      // lookup (jax's `take`): the handler reads `slice_sizes` off the op, so
      // a blocked OPERAND -- a per-group scale gathered by a group ramp, say
      // -- would have it reassemble the result with the full leading extent.
      if (std::find(offset.begin(), offset.end(), 0) != offset.end() ||
          d.getIndexVectorDim() == 0 || !d.getOperandBatchingDims().empty() ||
          idx.empty() || idx[0] != lead_)
        NoBlock("gather does not batch over the blocked axis");
      Demand(o->getOperand(0), false, p);
      Demand(o->getOperand(1), true, p);
    } else if (name == "stablehlo.reduce") {
      std::vector<int64_t> dims = I64List(o, "dimensions");
      if (std::find(dims.begin(), dims.end(), 0) != dims.end())
        NoBlock("reduce folds the blocked axis");
      const size_t n = o->getNumOperands() / 2;
      for (size_t i = 0; i < o->getNumOperands(); i++)
        Demand(o->getOperand(i), i < n, p);
    } else if (name == "func.call" || name == "stablehlo.composite") {
      DemandCall(o, name == "func.call" ? "callee" : "decomposition", p);
    } else if (RowLocalNames().contains(name) &&
               name != "stablehlo.transpose" &&
               name != "stablehlo.concatenate") {
      DemandElementwise(o, p);
    } else if (name == "stablehlo.transpose" &&
               !I64List(o, "permutation").empty() &&
               I64List(o, "permutation")[0] == 0) {
      DemandElementwise(o, p);
    } else if (name == "stablehlo.concatenate" &&
               mlir::cast<mlir::stablehlo::ConcatenateOp>(o).getDimension() !=
                   0) {
      DemandElementwise(o, p);
    } else {
      NoBlock(absl::StrCat(name, " is not known to be row-local"));
    }
  }

  // An operand is blocked exactly when it carries the blocked axis;
  // StableHLO's elementwise ops take operands of the result's own shape, with
  // rank-0 scalars (select's predicate, clamp's bounds) the only exception.
  void DemandElementwise(mlir::Operation* o, Plan* p) {
    for (mlir::Value x : o->getOperands()) {
      std::vector<int64_t> s = ShapeOf(x);
      Demand(x, !s.empty() && s[0] == lead_, p);
    }
  }

  // A call, demanded INSIDE the callee.  The lowering splices a callee's body
  // into the frame with its parameters aliased to the call's operands, so the
  // parameters have to carry the demand too -- and a callee invoked TWICE
  // would need one parameter to be two things at once, which is a decline.
  void DemandCall(mlir::Operation* o, const char* attr, Plan* p) {
    auto sym = o->getAttrOfType<mlir::FlatSymbolRefAttr>(attr);
    if (!sym) NoBlock("a call with no callee");
    mlir::ModuleOp mod = ctx_.module;
    auto fn = mod.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
    if (!fn || fn.getBody().getBlocks().size() != 1)
      NoBlock("a callee that is not a single block");
    mlir::Block& body = fn.getBody().front();
    if (body.getNumArguments() != o->getNumOperands())
      NoBlock("a call whose arity does not match its callee");
    if (body.empty()) NoBlock("an empty callee");
    mlir::Operation* term = &body.back();
    if (!term->hasTrait<mlir::OpTrait::IsTerminator>() ||
        term->getNumOperands() != o->getNumResults())
      NoBlock("a callee with no matching terminator");
    for (unsigned i = 0; i < o->getNumOperands(); i++) {
      mlir::Value param = body.getArgument(i);
      auto had = binds_.find(param);
      if (had != binds_.end() && had->second != o->getOperand(i))
        NoBlock("a callee invoked more than once in one subtree");
      binds_[param] = o->getOperand(i);
    }
    bodies_.insert(&body);
    for (unsigned i = 0; i < o->getNumResults(); i++)
      Demand(term->getOperand(i), true, p);
  }

  const QmmMatch& m_;
  const PackContext& ctx_;
  llvm::DenseMap<mlir::Value, int> main_args_;
  int64_t lead_ = 0;
  int64_t step_ = 0;
  // The whole-evaluation memo (qmm.py `_Source.memo`), and the per-root
  // blocking analysis with the Programs it produced.  Keyed by the value's
  // opaque pointer in an ordered map: `PlanFor` hands back a reference, and a
  // DenseMap's would not survive the next insertion.
  std::map<const void*, mx::array> whole_;
  std::map<const void*, Plan> plans_;
  std::map<std::pair<const void*, int64_t>, std::shared_ptr<Program>> progs_;
  // Scratch for one analysis.
  llvm::DenseMap<mlir::Value, bool> demand_;
  llvm::DenseMap<mlir::Value, mlir::Value> binds_;
  llvm::DenseSet<mlir::Block*> bodies_;
};

// qmm.py `_cat`.
mx::array Cat(const std::vector<mx::array>& parts) {
  if (parts.size() == 1) return parts[0];
  return mx::concatenate(parts, 0);
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

// qmm.py `_build_mxfp4_pack`: verify and repack an MXFP4 weight, one row block
// at a time.  Each block's two factors are derived and dropped before the next
// block is evaluated: the reconstruction is full weight size, what is kept
// from it (a nibble per value, a byte per 32) is an eighth of that, and the
// verification -- every value read back off the E2M1 grid by exact integer
// equality, every group scale an exact power of two -- covers every element
// either way.
Pack BuildMxfp4(const QmmMatch& m, RowSource& src) {
  const int64_t gs = kMxfp4Group;
  if (m.K % gs)
    Bail(absl::StrFormat("K=%d is not a multiple of %d", m.K, gs));
  std::optional<mx::array> perm;
  std::vector<mx::array> ws, sbs;
  for (auto [lo, hi] : src.blocks()) {
    {
      mx::array scale_map = src.Rows(m.scale, lo, hi);
      if (!GroupConst(scale_map, gs)) {
        if (src.blocked())
          // The permutation that un-interleaves the groups is a property of
          // the whole contraction axis: hand this weight back to the
          // unblocked path, which can see all of it.
          NoBlock(absl::StrFormat(
              "MXFP4 scales are not constant within a group of %d", gs));
        // The same interleaving story as the affine path: permuting the
        // contraction axis on BOTH operands leaves the dot unchanged.
        std::vector<int32_t> p;
        if (Regroup(m.K, {&scale_map}, &p)) {
          perm = HostPerm(p);
          scale_map = TakeK(scale_map, *perm);
        }
        if (!GroupConst(scale_map, gs))
          Bail(absl::StrFormat(
              "MXFP4 scales are not constant within a group of %d", gs));
      }
      sbs.push_back(Mxfp4ScaleBytes(GroupHeads(scale_map, gs)));
    }
    src.Drop(m.scale);
    {
      mx::array values = src.Rows(m.codes, lo, hi);
      if (perm.has_value()) values = TakeK(values, *perm);
      // The E2M1 nibble order is MLX's: element i of a row occupies bits
      // [4i, 4i+4) of the little-endian uint32 stream.
      ws.push_back(PackCodes(Mxfp4Codes(values), 4));
    }
    src.Drop(m.codes);
    mx::eval(ws.back(), sbs.back());
  }
  Pack pk{Join(Cat(ws), m), Join(Cat(sbs), m), std::nullopt, perm, gs, 4, 1};
  return pk;
}

// qmm.py `_build_affine_pack`: verify and repack a `scale * (code - zero)`
// weight, block at a time.
Pack BuildAffine(const QmmMatch& m, RowSource& src) {
  // The code range first: the offset that makes the codes unsigned has to be
  // one number for the whole weight, so the codes are walked twice -- free
  // unblocked (`RowSource` memoizes the one evaluation), a second pass over
  // the subtree when blocked, which is the price of not holding it.
  int64_t lo = 0, hi = 0;
  bool have_range = false;
  for (auto [blo, bhi] : src.blocks()) {
    mx::array codes = src.Rows(m.codes, blo, bhi);
    mx::array lo_a = mx::min(codes, false);
    mx::array hi_a = mx::max(codes, false);
    mx::eval(lo_a, hi_a);
    const int64_t clo = mx::astype(lo_a, mx::int32).item<int>();
    const int64_t chi = mx::astype(hi_a, mx::int32).item<int>();
    lo = have_range ? std::min(lo, clo) : clo;
    hi = have_range ? std::max(hi, chi) : chi;
    have_range = true;
  }
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
    mx::array s = src.Whole(m.scale);
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
    // Heads at the SMALLEST legal group size, per block; the group size the
    // whole weight allows is then read off the heads, which are 32x smaller
    // than the maps and can be held for every block at once.
    const int64_t g0 = kMinGroup;
    std::vector<mx::array> sheads, zheads;
    for (auto [blo, bhi] : src.blocks()) {
      mx::array smap = src.Rows(m.scale, blo, bhi);
      std::optional<mx::array> zmap;
      if (m.zero != nullptr) zmap = src.Rows(m.zero, blo, bhi);
      if (!GroupConst(smap, g0) ||
          (zmap.has_value() && !GroupConst(*zmap, g0))) {
        if (src.blocked())
          NoBlock(absl::StrFormat(
              "scales/zeros are not constant within a group of %d", g0));
        // The groups may still be there, interleaved: recover the permutation
        // that makes them contiguous and re-verify EXACTLY (the clustering is
        // only a proposal -- the constancy check on the permuted maps is what
        // the pack's exactness rests on).
        std::vector<const mx::array*> maps{&smap};
        if (zmap.has_value()) maps.push_back(&*zmap);
        std::vector<int32_t> p;
        std::string note;
        if (Regroup(m.K, maps, &p)) {
          perm = HostPerm(p);
          // Rebind as each permuted copy lands: these are full weight size,
          // and holding the originals as well would raise the peak by one
          // whole map each.
          smap = TakeK(smap, *perm);
          if (zmap.has_value()) zmap = TakeK(*zmap, *perm);
          note = " (even regrouped)";
        }
        if (!GroupConst(smap, g0) ||
            (zmap.has_value() && !GroupConst(*zmap, g0)))
          Bail(absl::StrCat("scales/zeros are not constant within any group",
                            note));
      }
      scale_dtype = smap.dtype();
      sheads.push_back(GroupHeads(smap, g0));
      if (zmap.has_value()) zheads.push_back(GroupHeads(*zmap, g0));
      mx::eval(sheads.back());
      if (!zheads.empty()) mx::eval(zheads.back());
      src.Drop(m.scale);
      src.Drop(m.zero);
    }
    scales = Cat(sheads);
    if (!zheads.empty()) zeros = Cat(zheads);
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

  std::vector<mx::array> ws;
  for (auto [blo, bhi] : src.blocks()) {
    {
      mx::array codes = src.Rows(m.codes, blo, bhi);
      if (perm.has_value()) codes = TakeK(codes, *perm);
      ws.push_back(PackCodes(
          mx::add(mx::astype(mx::contiguous(codes), mx::int32),
                  mx::array(static_cast<int>(offset), mx::int32)),
          bits));
    }
    src.Drop(m.codes);
    mx::eval(ws.back());
  }
  mx::array out_scales = mx::array(0.0f);
  mx::array out_biases = mx::array(0.0f);
  ScaleBias(scales, zeros.has_value() ? &*zeros : nullptr, offset, scale_dtype,
            &out_scales, &out_biases);
  // Materialize: a lazy packed weight would pin the whole reconstruction graph
  // (and its full-size intermediates) for the life of the cache.
  mx::eval(out_scales, out_biases);
  Pack pk{Join(Cat(ws), m), Join(out_scales, m), Join(out_biases, m), perm, gs,
          bits, 0};
  return pk;
}

// qmm.py `_build_pack`: verify and repack one match's weight, on the concrete
// argument buffers.  A subtree the blocking cannot follow falls back to a
// whole evaluation, where `blocks()` yields one block covering everything and
// the two packers above are unchanged.
Pack BuildPack(const QmmMatch& m, RowSource& src, bool* was_blocked) {
  while (true) {
    try {
      NoCache no_cache;
      Pack pk = m.mode == 1 ? BuildMxfp4(m, src) : BuildAffine(m, src);
      *was_blocked = src.blocked();
      return pk;
    } catch (const NotBlockable& e) {
      if (!src.blocked()) Bail(absl::StrCat("not blockable: ", e.why));
      Debug(absl::StrCat(m.name, " packs from a whole evaluation (", e.why,
                         ")"));
      src.unblock();
    }
  }
}

// --------------------------------------------------------------------------
// the cross-executable build cache (qmm.py, same section)
// --------------------------------------------------------------------------
//
// A pack is a deterministic pure function of exactly two things: the argument
// buffers its reconstruction reads, and the reconstruction itself.  Two
// EXECUTABLES over one model share the first and duplicate the second --
// keras-hub compiles a separate generate program per sequence-length shape,
// and each gets its own plan whose matches hold no packs, so every weight was
// re-evaluated and re-verified from nothing (0.9 s x 94 packs on gpt-oss-20b,
// once per executable shape, and a full pack set live per shape).
//
// Reuse is sound only when both halves are PROVABLY the same, so the key is a
// canonical serialization of the reconstruction plus the identity of the
// buffers it bottoms out on.  Anything the serialization cannot cover exactly
// declines to be cached and is built exactly as before -- declining is free,
// and it is the only thing standing between "we can prove it" and a wrong
// weight.  Verification itself is never weakened: a miss runs the full build,
// every element checked.

// One attribute's COMPLETE text, or a decline (qmm.py `_attr_text`).  Complete
// is the whole point.  A `dense_resource` prints its blob name and two modules
// can bind one name to different bytes, so it can never stand in for its
// contents; a big dense attribute could be printed in full, but reading it is
// the cost the cache exists to avoid.
std::string AttrText(mlir::Operation* o, llvm::StringRef name,
                     mlir::Attribute attr) {
  if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    // A splat carries one element however big its type says it is.
    if (!dense.isSplat()) {
      auto st = mlir::dyn_cast<mlir::ShapedType>(dense.getType());
      const int64_t n = (st && st.hasStaticShape()) ? st.getNumElements() : -1;
      if (n < 0 || n > kFpDenseElems)
        NoFp(absl::StrCat(OpName(o), " carries ", n, " constant elements"));
    }
  } else if (OpName(o) == "stablehlo.constant") {
    // An encoding whose size cannot be read off a DenseElementsAttr -- the
    // size has to come off the result type, before anything asks for text.
    const int64_t n = Prod(ShapeOf(o->getResult(0)));
    if (n > kFpDenseElems)
      NoFp(absl::StrCat(OpName(o), " carries ", n,
                        " constant elements this reader cannot size"));
  }
  std::string s;
  llvm::raw_string_ostream os(s);
  attr.print(os);
  os.flush();
  if (s.size() > kFpAttrChars)
    NoFp(absl::StrCat(OpName(o), "'s ", name.str(), " prints ", s.size(),
                      " characters"));
  if (s.find("dense_resource<") != std::string::npos)
    NoFp(absl::StrCat(OpName(o), "'s ", name.str(),
                      " is a dense_resource (its contents are not in the IR)"));
  return s;
}

// qmm.py `_Fingerprint`: a canonical serialization of what a pack gets built
// from.  Deterministic by construction, and independent of everything that is
// not the computation: values are named by the order THIS walk reaches them
// (never by SSA name), operands in operand order, attributes sorted by name
// and printed in full, callees serialized through their BODY -- jax renumbers
// private helpers per program, so `@_take` and `@_take_0` have to fingerprint
// identically when their bodies agree, and two modules that bind one name to
// different bodies must not.
class Fingerprint {
 public:
  Fingerprint(mlir::ModuleOp module, mlir::Block* main) : module_(module) {
    for (mlir::BlockArgument a : main->getArguments())
      main_args_[a] = static_cast<int>(a.getArgNumber());
  }

  std::string text() const { return parts_; }
  const std::vector<int>& leaves() const { return leaves_; }
  void Append(const std::string& s) { parts_ += s; }

  // `v` and everything under it (the demand-driven half of the walk).
  void Value(mlir::Value v) {
    if (sealed_)
      // A callee body is serialized ONCE and reused for every call of it, so
      // it must not name anything from a caller.  Functions are isolated from
      // above, so this cannot fire -- but a body that somehow did capture
      // would otherwise be memoized with one caller's operands baked in.
      NoFp("a callee body reads a value from its caller");
    auto it = ids_.find(v);
    if (it != ids_.end()) {
      absl::StrAppend(&parts_, "#", it->second, ";");
      return;
    }
    ids_[v] = static_cast<int>(ids_.size());
    if (mlir::isa<mlir::BlockArgument>(v)) {
      auto arg = main_args_.find(v);
      if (arg == main_args_.end()) {
        mlir::Value outer = Hoist(v);
        if (outer == v) NoFp("an unbound block argument");
        parts_ += "carry(";
        Value(outer);
        parts_ += ");";
        return;
      }
      // Leaves are numbered by the walk, not by their argument position: two
      // programs may take one model's weights in different orders, and the
      // reconstruction is what has to match.
      absl::StrAppend(&parts_, "leaf", leaves_.size(), ":", TypeText(v), ";");
      leaves_.push_back(arg->second);
      return;
    }
    mlir::Operation* o = Owner(v);
    if (o == nullptr) NoFp("a value that is neither argument nor result");
    absl::StrAppend(&parts_, "r", ResultNumber(v), "=");
    Head(o);
    parts_ += "(";
    for (mlir::Value x : o->getOperands()) Value(x);
    parts_ += ")";
    llvm::DenseMap<mlir::Value, int> ids;
    Regions(o, ids);
  }

  // `o`'s name, result types and attributes -- everything but operands.
  void Head(mlir::Operation* o) {
    absl::StrAppend(&parts_, OpName(o), "[");
    for (mlir::Value r : o->getResults())
      absl::StrAppend(&parts_, TypeText(r), ",");
    parts_ += ":";
    std::vector<std::pair<std::string, mlir::Attribute>> named;
    for (mlir::NamedAttribute na : o->getAttrs())
      named.emplace_back(na.getName().str(), na.getValue());
    std::sort(named.begin(), named.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    const bool call = OpName(o) == "func.call";
    for (const auto& [name, attr] : named) {
      // The callee's BODY stands in for its name (see `Callee`); printing the
      // symbol as well would make two copies of one helper fingerprint
      // differently for no reason.
      if (call && name == "callee") continue;
      absl::StrAppend(&parts_, name, "=", AttrText(o, name, attr), ",");
    }
    parts_ += "]";
  }

  // `o`'s regions, walked top to bottom with their own numbering.  Reduce
  // bodies and sort comparators live here; the prologue evaluates them, so
  // what they compute is part of what the pack is.
  void Regions(mlir::Operation* o, llvm::DenseMap<mlir::Value, int>& ids) {
    if (OpName(o) == "func.call") parts_ += Callee(o);
    for (mlir::Region& region : o->getRegions()) {
      parts_ += "{";
      for (mlir::Block& blk : region.getBlocks()) {
        parts_ += "|";
        for (mlir::BlockArgument a : blk.getArguments()) {
          ids[a] = static_cast<int>(ids.size());
          absl::StrAppend(&parts_, "p", ids[a], ";");
        }
        for (mlir::Operation& inner : blk) BodyOp(&inner, ids);
      }
      parts_ += "}";
    }
  }

  // One op of a region or callee body, in source order.
  void BodyOp(mlir::Operation* o, llvm::DenseMap<mlir::Value, int>& ids) {
    Head(o);
    parts_ += "(";
    for (mlir::Value x : o->getOperands()) {
      auto got = ids.find(x);
      if (got == ids.end()) {
        // A value the region captures from outside: name it in the walk that
        // owns it, so one capture reads the same either way.
        Value(x);
      } else {
        absl::StrAppend(&parts_, "%", got->second, ";");
      }
    }
    parts_ += ")";
    Regions(o, ids);
    for (mlir::Value r : o->getResults()) ids[r] = static_cast<int>(ids.size());
    parts_ += ";";
  }

  std::string Callee(mlir::Operation* o) {
    auto sym = o->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
    if (!sym) NoFp("a call with no resolvable callee");
    const std::string name = sym.getValue().str();
    auto got = callees_.find(name);
    if (got != callees_.end()) {
      if (got->second == kWalking)
        NoFp(absl::StrCat("callee @", name, " is recursive"));
      return got->second;
    }
    auto fn = module_.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
    if (!fn) NoFp(absl::StrCat("callee @", name, " is not in the module"));
    callees_[name] = kWalking;
    std::string outer = std::move(parts_);
    const bool sealed = sealed_;
    parts_.clear();
    sealed_ = true;
    std::string body;
    try {
      llvm::DenseMap<mlir::Value, int> ids;
      Regions(fn.getOperation(), ids);
      body = parts_;
    } catch (...) {
      parts_ = std::move(outer);
      sealed_ = sealed;
      callees_.erase(name);
      throw;
    }
    parts_ = std::move(outer);
    sealed_ = sealed;
    callees_[name] = body;
    return body;
  }

 private:
  static std::string TypeText(mlir::Value v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    v.getType().print(os);
    os.flush();
    return s;
  }
  static int ResultNumber(mlir::Value v) {
    auto r = mlir::dyn_cast<mlir::OpResult>(v);
    return r ? static_cast<int>(r.getResultNumber()) : 0;
  }

  static constexpr const char* kWalking = "\x01walking";

  mlir::ModuleOp module_;
  llvm::DenseMap<mlir::Value, int> main_args_;
  llvm::DenseMap<mlir::Value, int> ids_;
  std::vector<int> leaves_;
  std::map<std::string, std::string> callees_;
  std::string parts_;
  bool sealed_ = false;
};

// qmm.py `_fingerprint`: (serialization, leaf argument positions) for `m`'s
// pack inputs.  What goes in is exactly what the pack build reads.  The
// ACTIVATION side of the dot does not: `m.M`, `m.mshape` and `m.lperm`
// describe the other operand, they are precisely what makes two executables of
// one model differ, and no packed byte depends on them.
bool FingerprintOf(const QmmMatch& m, const PackContext& ctx, std::string* text,
                   std::vector<int>* leaves, std::string* why) {
  try {
    Fingerprint fp(ctx.module, ctx.main);
    fp.Append("qmm1|");
    absl::StrAppend(text, "");
    std::string head = absl::StrCat(
        m.mode, "|", m.K, "|", m.N, "|", absl::StrJoin(m.bshape, ","), "|",
        absl::StrJoin(m.rshape, ","), "|", absl::StrJoin(m.rperm, ","), "|",
        absl::StrJoin(m.nshape, ","), "|", m.recip ? 1 : 0, "|",
        m.has_sub_range ? 1 : 0, "|", m.sub_lo, "|", m.sub_hi, "|",
        absl::StrJoin(m.bcast_dims, ","), "|");
    fp.Append(head);
    for (mlir::Operation* o : m.post) {
      // The shape ops between the reconstruction and the dot: their result
      // shapes and permutations are read off the IR by `ToRows`.
      fp.Append(absl::StrCat(absl::StrJoin(ShapeOf(o->getOperand(0)), ","),
                             "->"));
      fp.Head(o);
      fp.Append("|");
    }
    fp.Append("codes:");
    fp.Value(m.codes);
    fp.Append("scale:");
    fp.Value(m.scale);
    if (m.zero != nullptr) {
      fp.Append("zero:");
      fp.Value(m.zero);
    }
    *text = fp.text();
    *leaves = fp.leaves();
    return true;
  } catch (const NoFingerprint& e) {
    *why = e.why;
    return false;
  } catch (const std::exception& e) {
    *why = e.what();
    return false;
  }
}

// One built pack, and the identity of the buffers it was built from.
//
// Stage 1's entry is weak throughout -- a Python weakref hands the object
// back, so an entry can keep neither the weights nor the pack alive.  MLX's
// C++ array cannot be rebuilt from a weak handle, so the PACK is held
// strongly here and the LEAVES weakly, through `data_shared_ptr`: an entry
// whose source weight has been freed is swept, which is what happens when the
// model is unloaded, and the bound below is what stops a config-sweeping
// worker from growing this forever.
//
// The leaf identity is checked three ways on every hit.  `mx::array::id()` is
// the address of a refcounted descriptor and CPython-style address recycling
// applies to it exactly as it does in the Python (this project has twice
// shipped a bug from a set keyed on an address alone), so the weak handle
// proves the buffer is still alive, the data pointer proves it is the SAME
// buffer, and the shape and dtype prove it is the same view of it.
struct BuiltEntry {
  struct Leaf {
    std::uintptr_t id = 0;
    std::weak_ptr<mx::array::Data> data;
    const void* raw = nullptr;
    std::vector<int64_t> shape;
    uint32_t dtype = 0;
  };
  std::vector<Leaf> leaves;
  std::optional<Pack> pack;
  uint64_t serial = 0;
};

std::map<std::string, std::vector<BuiltEntry>>& BuiltCache() {
  static auto* cache = new std::map<std::string, std::vector<BuiltEntry>>();
  return *cache;
}
uint64_t g_built_serial = 0;
PackStats g_pack_stats;
// The cache and the counters are process-wide, and a pack wave normally runs
// under the process-wide submission lock (metal_stream.h) -- but
// METALJAX_CONCURRENT_EXECUTE=1 lifts that one, and a torn map would be a
// crash rather than a slow program.  Held only around the map, never around a
// build.
std::mutex g_built_mu;

bool LeafAlive(const BuiltEntry::Leaf& l) { return l.data.lock() != nullptr; }

bool LeafMatches(const BuiltEntry::Leaf& l, const mx::array& a) {
  if (l.id != a.id()) return false;
  std::shared_ptr<mx::array::Data> live = l.data.lock();
  if (live == nullptr || live.get() != l.raw) return false;
  if (a.data_shared_ptr() == nullptr || a.data_shared_ptr().get() != l.raw)
    return false;
  if (l.shape != FromMxShape(a.shape())) return false;
  return l.dtype == static_cast<uint32_t>(a.dtype().val());
}

int64_t CacheSize() {
  int64_t n = 0;
  for (const auto& [k, v] : BuiltCache()) n += static_cast<int64_t>(v.size());
  return n;
}

// The pack this key was built for, or nothing.
std::optional<Pack> CachedBuild(const std::string& key,
                                const std::vector<mx::array>& leaves) {
  if (kMaxBuilt <= 0) return std::nullopt;
  std::lock_guard<std::mutex> lock(g_built_mu);
  auto bucket = BuiltCache().find(key);
  if (bucket == BuiltCache().end()) return std::nullopt;
  for (BuiltEntry& e : bucket->second) {
    if (e.leaves.size() != leaves.size()) continue;
    bool same = true;
    for (size_t i = 0; i < leaves.size() && same; i++)
      same = LeafMatches(e.leaves[i], leaves[i]);
    if (!same) continue;
    e.serial = ++g_built_serial;
    return e.pack;   // an optional<Pack>, which is what the caller wants
  }
  return std::nullopt;
}

void RememberBuild(const std::string& key, const std::vector<mx::array>& leaves,
                   const Pack& pk) {
  if (kMaxBuilt <= 0) return;
  std::lock_guard<std::mutex> lock(g_built_mu);
  BuiltEntry e;
  e.leaves.reserve(leaves.size());
  for (const mx::array& a : leaves) {
    const std::shared_ptr<mx::array::Data>& data = a.data_shared_ptr();
    // A leaf with no storage cannot be proven identical later; an argument of
    // an execute always has some, so this is a decline and not a case.
    if (data == nullptr) return;
    BuiltEntry::Leaf l;
    l.id = a.id();
    l.data = data;
    l.raw = data.get();
    l.shape = FromMxShape(a.shape());
    l.dtype = static_cast<uint32_t>(a.dtype().val());
    e.leaves.push_back(std::move(l));
  }
  e.pack = pk;
  e.serial = ++g_built_serial;
  if (CacheSize() >= kMaxBuilt) {
    // Dead entries first: a pack whose source weight is gone is stale, which
    // is what an unloaded model looks like from here.
    for (auto it = BuiltCache().begin(); it != BuiltCache().end();) {
      auto& v = it->second;
      v.erase(std::remove_if(v.begin(), v.end(),
                             [](const BuiltEntry& x) {
                               for (const auto& l : x.leaves)
                                 if (!LeafAlive(l)) return true;
                               return false;
                             }),
              v.end());
      it = v.empty() ? BuiltCache().erase(it) : std::next(it);
    }
    // ...then the least recently used, one at a time: a bound that never
    // clears the whole cache keeps a steady-state model's packs resident.
    while (CacheSize() >= kMaxBuilt) {
      auto oldest = BuiltCache().end();
      size_t at = 0;
      uint64_t best = UINT64_MAX;
      for (auto it = BuiltCache().begin(); it != BuiltCache().end(); ++it) {
        for (size_t i = 0; i < it->second.size(); i++) {
          if (it->second[i].serial < best) {
            best = it->second[i].serial;
            oldest = it;
            at = i;
          }
        }
      }
      if (oldest == BuiltCache().end()) break;
      oldest->second.erase(oldest->second.begin() + at);
      if (oldest->second.empty()) BuiltCache().erase(oldest);
    }
  }
  BuiltCache()[key].push_back(std::move(e));
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

const absl::flat_hash_set<std::string>& RowLocalOps() { return RowLocalNames(); }

PackStats QmmPackStats() {
  std::lock_guard<std::mutex> lock(g_built_mu);
  PackStats out = g_pack_stats;
  out.entries = CacheSize();
  return out;
}

void ResetQmmPackStats() {
  std::lock_guard<std::mutex> lock(g_built_mu);
  g_pack_stats = PackStats{};
  BuiltCache().clear();
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

absl::Status BuildQmmPacks(RewritePlan* plan, const PackContext& ctx) {
  plan->packs.clear();
  absl::flat_hash_set<int> args;
  // What this wave of packs CLAIMS, which is the figure a memory watchdog
  // reads and the one row-blocking exists to bound.  It is measured from the
  // plugin's own libmlx: `mlx.core` in the host process is a different
  // runtime and its counters read zero here (the P16 campaign's note).
  const size_t peak_before = mx::get_peak_memory();
  mx::reset_peak_memory();
  for (auto& m : plan->qmm) {
    if (m->disabled) continue;
    try {
      // Has this very reconstruction, over these very buffers, been built
      // already in this process?  Two executables of one model -- keras-hub's
      // per-shape generate programs, a prefill and its decode loop -- ask the
      // same question of the same weights, and answering it twice costs both
      // the build and a second copy of the pack.
      std::string key, why;
      std::vector<int> positions;
      std::vector<mx::array> leaves;
      bool cacheable = kMaxBuilt > 0 &&
                       FingerprintOf(*m, ctx, &key, &positions, &why);
      if (cacheable) {
        for (int i : positions) {
          if (i < 0 || i >= static_cast<int>(ctx.args->size())) {
            cacheable = false;
            break;
          }
          leaves.push_back((*ctx.args)[i]);
        }
      } else if (kMaxBuilt > 0) {
        g_pack_stats.build_declines++;
        Debug(absl::StrCat(m->name, " is not build-cached (", why, ")"));
      }
      std::optional<Pack> hit =
          cacheable ? CachedBuild(key, leaves) : std::nullopt;
      const bool reused = hit.has_value();
      int64_t blocks = 1;
      if (reused) {
        g_pack_stats.build_hits++;
      } else {
        // The operand subtrees, evaluated on this execute's buffers one row
        // block at a time (qmm.py `_Source`).
        RowSource src(*m, ctx);
        bool was_blocked = false;
        hit = BuildPack(*m, src, &was_blocked);
        blocks = static_cast<int64_t>(src.blocks().size());
        g_pack_stats.build_misses++;
        (was_blocked ? g_pack_stats.blocked : g_pack_stats.whole)++;
        if (cacheable) RememberBuild(key, leaves, *hit);
      }
      const Pack& pk = *hit;
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
      Debug(absl::StrFormat("%s %s mode=%s bits=%d group=%d%s%s",
                            reused ? "reused" : "packed", m->name,
                            pk.mode == 1 ? "mxfp4" : "affine", pk.bits, pk.gs,
                            pk.perm.has_value() ? " regrouped" : "",
                            reused ? "" : (blocks > 1
                                ? absl::StrFormat(" in %d row blocks", blocks)
                                : std::string(" whole"))));
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
  const size_t peak = mx::get_peak_memory();
  if (peak > g_pack_stats.peak_bytes) g_pack_stats.peak_bytes = peak;
  Debug(absl::StrFormat("pack wave peak %.3f GB", peak / 1e9));
  // The process-wide high-water mark is a shared diagnostic; put back
  // whichever of the two is higher, so measuring the pack wave does not lower
  // what anything else reads.
  if (peak_before > peak) {
    mx::reset_peak_memory();
    Debug(absl::StrFormat("(the process peak was %.3f GB before this wave)",
                          peak_before / 1e9));
  }
  // The reconstruction ran at full weight size; its intermediates are dead
  // now and Metal counts live buffers, not bytes.
  gc_collect();
  mx::clear_cache();
  return absl::OkStatus();
}

}  // namespace metaljax
