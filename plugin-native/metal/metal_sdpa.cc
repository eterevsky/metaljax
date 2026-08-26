/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Stage 1's `src/metaljax/sdpa.py` (deleted 0.11.6, ef5774d), as a pass over
the parsed StableHLO: the analysis that
reads a softmax attention out of the graph and fills a `SdpaMatch` for the
lowering to emit as one `mx::fast::scaled_dot_product_attention`.

Only the ANALYSIS half of the Python module is here -- its `emit`/`_apply`/
`_mask_array` are the lowering's, and the recipes this file computes are
exactly what those read.  Everything else is a transliteration: the atom
algebra (`_roles_*`, `_recipe`), the two chains (`_logits` down to the first
dot, `_probs` down to the `exp`), the two reductions (`_check_reduce`), the
deferred normalization (`_find_norm`), and the survivor rules (`_exclusive`,
`_escapes`).  Where the two could drift, the Python is the specification.

The one rule the file exists to keep: a half-matched pattern lowers as
ORDINARY ops.  Every `_Reject` in the Python is a `Bail` here, and the
consequence is a correct slow program -- never a wrong fused one.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "metal/metal_dtypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace metaljax {
namespace {

// The environment, read once (sdpa.py's module-level knobs).
bool EnvOff(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "0";
}
const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();

// sdpa.py `_MAX_DEPTH` / `_MAX_FANOUT`: chains between stages are a handful of
// ops, so these bounds only stop a pathological graph from making analysis
// quadratic.
constexpr int kMaxDepth = 32;
constexpr int kMaxFanout = 32;
// sdpa.py `_MASK_FRACTION`: a `select` constant counts as a mask sentinel when
// it is at least this fraction of the dtype's largest finite value, negated.
// jax uses `finfo.min` (1.0) and maxtext -0.35 / -0.7 of `finfo.max`.
constexpr double kMaskFraction = 0.1;

// This subgraph is not (provably) attention: run it literally.
struct Reject {
  std::string why;
};
[[noreturn]] void Bail(const std::string& why) { throw Reject{why}; }

void Debug(const std::string& line) {
  if (!kDebug) return;
  std::fprintf(stderr, "[metaljax-native] sdpa: %s\n", line.c_str());
  std::fflush(stderr);
}

// --------------------------------------------------------------------------
// IR helpers (sdpa.py, same section)
// --------------------------------------------------------------------------

mlir::Operation* Owner(mlir::Value v) {
  return v == nullptr ? nullptr : v.getDefiningOp();
}

std::string OpName(mlir::Operation* op) {
  return op == nullptr ? std::string() : op->getName().getStringRef().str();
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

// sdpa.py `_FINITE_MAX` / `_FLOAT_ELS`.
bool IsFloatEl(const std::string& el) {
  return el == "f32" || el == "bf16" || el == "f16";
}

double FiniteMax(const std::string& el) {
  if (el == "f32") return 3.4028234663852886e38;
  if (el == "bf16") return 3.3895313892515355e38;
  if (el == "f16") return 65504.0;
  return 0.0;
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

// The dot's dimension numbers, as ops/linalg.py `_dot_dims` reads them.
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

bool Holds(const std::vector<int64_t>& xs, int64_t v) {
  return std::find(xs.begin(), xs.end(), v) != xs.end();
}

// sdpa.py `_IDENTITY`: ops that pass a tensor through elementwise-identically.
bool IsIdentity(const std::string& n) {
  return n == "stablehlo.convert" || n == "sdy.sharding_constraint";
}
// sdpa.py `_CALL`.
bool IsCall(const std::string& n) {
  return n == "func.call" || n == "stablehlo.composite";
}
// sdpa.py `_FORWARD`: pure reassociation, invertible in both directions.
bool IsForward(const std::string& n) {
  return IsIdentity(n) || n == "stablehlo.reshape" ||
         n == "stablehlo.transpose";
}
// sdpa.py `_TRANSPARENT`: ops a value can be followed DOWN through while its
// axes stay trackable.  Only downwards: a broadcast is not invertible walking
// forward.
bool IsTransparent(const std::string& n) {
  return IsForward(n) || n == "stablehlo.broadcast_in_dim";
}
// sdpa.py `_CLAMP`: `jnp.max(..., initial=-inf)` lowers to
// `maximum(broadcast(-inf), reduce)`.  Dropping a splat identity operand is
// exact for every input, NaN included (both forms propagate it), so these are
// followed through like a convert.
bool ClampIdentity(const std::string& n, double* ident) {
  if (n == "stablehlo.maximum") {
    *ident = -std::numeric_limits<double>::infinity();
    return true;
  }
  if (n == "stablehlo.minimum") {
    *ident = std::numeric_limits<double>::infinity();
    return true;
  }
  return false;
}
// sdpa.py `_SAME_SHAPE`: shape-preserving ops, for the role propagators.
bool IsSameShape(const std::string& n) {
  double ignored = 0.0;
  return IsIdentity(n) || IsCall(n) || ClampIdentity(n, &ignored);
}

void WalkBlocks(mlir::Block& block,
                const std::function<void(mlir::Block&)>& fn) {
  fn(block);
  for (mlir::Operation& op : block)
    for (mlir::Region& r : op.getRegions())
      for (mlir::Block& b : r.getBlocks()) WalkBlocks(b, fn);
}

// --------------------------------------------------------------------------
// axis roles (sdpa.py, same section)
// --------------------------------------------------------------------------
//
// Every axis of the first dot gets an ATOM; a tensor's ROLE VECTOR gives, per
// dimension, the list of atoms that dimension carries -- empty for a unit
// dimension, several once a reshape has merged axes.  Atoms:
//
//   ('b', i)  batching dim i of the first dot   (shared by Q and K)
//   ('d', i)  contracting dim i                 (the head dim; not in logits)
//   ('l', i)  free dim i of the first dot's lhs
//   ('r', i)  free dim i of the first dot's rhs
//   ('v', i)  free dim i of V in the second dot (the output feature dim)
//   ('*', n)  a broadcast axis: the value is REPLICATED along it

using Atom = std::pair<char, int>;
using Role = std::vector<Atom>;
using Roles = std::vector<Role>;

// sdpa.py `_Atoms`.
struct Atoms {
  std::map<Atom, int64_t> size;
  int n = 0;

  Atom bcast(int64_t sz) {
    n += 1;
    Atom a{'*', n};
    size[a] = sz;
    return a;
  }
};

int64_t AtomSize(const Atoms& atoms, const Atom& a) {
  auto it = atoms.size.find(a);
  if (it == atoms.size.end()) Bail("an axis role with no recorded size");
  return it->second;
}

int64_t GroupSize(const Atoms& atoms, const std::vector<Atom>& g) {
  int64_t p = 1;
  for (const Atom& a : g) p *= AtomSize(atoms, a);
  return p;
}

// sdpa.py `_is_bcast`: one dimension's atoms, all of them replicated.
bool IsBcastRole(const Role& r) {
  if (r.empty()) return false;
  for (const Atom& a : r)
    if (a.first != '*') return false;
  return true;
}

Roles RolesTranspose(const Roles& roles, const std::vector<int64_t>& perm) {
  Roles out;
  out.reserve(perm.size());
  for (int64_t p : perm) {
    if (p < 0 || p >= static_cast<int64_t>(roles.size()))
      Bail("transpose permutation out of range");
    out.push_back(roles[p]);
  }
  return out;
}

Roles RolesUntranspose(const Roles& roles, const std::vector<int64_t>& perm) {
  Roles out(perm.size());
  for (size_t i = 0; i < perm.size(); i++) {
    const int64_t p = perm[i];
    if (p < 0 || p >= static_cast<int64_t>(perm.size()) || i >= roles.size())
      Bail("transpose permutation out of range");
    out[p] = roles[i];
  }
  return out;
}

// sdpa.py `_roles_reshape`: carry roles through a reshape by matching the atom
// stream.  A reshape regroups one ordered stream of atoms, so merges
// (`[H, g] -> [H*g]`) and splits both work; splitting an ATOM does not.  This
// is exact in both directions, so the inverse reshape uses it too.
Roles RolesReshape(const std::vector<int64_t>& in_shape, const Roles& roles,
                   const std::vector<int64_t>& out_shape, const Atoms& atoms) {
  std::vector<Atom> stream;
  const size_t n = std::min(in_shape.size(), roles.size());
  for (size_t i = 0; i < n; i++) {
    for (const Atom& a : roles[i]) stream.push_back(a);
    if (roles[i].empty() && in_shape[i] != 1)
      Bail("a role-less axis is not unit-sized");
  }
  Roles out;
  size_t i = 0;
  for (int64_t size : out_shape) {
    if (size == 1) {
      out.push_back(Role());
      continue;
    }
    int64_t got = 1;
    Role take;
    while (got < size) {
      if (i >= stream.size())
        Bail("reshape does not line up with the axis roles");
      got *= AtomSize(atoms, stream[i]);
      take.push_back(stream[i]);
      i++;
    }
    if (got != size) Bail("reshape splits an axis");
    out.push_back(std::move(take));
  }
  while (i < stream.size()) {
    if (AtomSize(atoms, stream[i]) != 1) Bail("reshape drops a non-unit axis");
    i++;
  }
  return out;
}

Roles RolesBroadcast(const std::vector<int64_t>& in_shape, const Roles& roles,
                     const std::vector<int64_t>& out_shape,
                     const std::vector<int64_t>& dims, Atoms& atoms) {
  Roles out(out_shape.size());
  std::vector<bool> filled(out_shape.size(), false);
  for (size_t i = 0; i < dims.size(); i++) {
    const int64_t d = dims[i];
    if (d < 0 || d >= static_cast<int64_t>(out_shape.size()) ||
        i >= in_shape.size() || i >= roles.size())
      Bail("broadcast_in_dim dimensions out of range");
    if (in_shape[i] == out_shape[d]) {
      out[d] = roles[i];
    } else if (in_shape[i] == 1) {
      out[d] = Role{atoms.bcast(out_shape[d])};
    } else {
      Bail("broadcast_in_dim is not a pure expansion");
    }
    filled[d] = true;
  }
  for (size_t d = 0; d < out_shape.size(); d++) {
    if (filled[d]) continue;
    out[d] = out_shape[d] == 1 ? Role() : Role{atoms.bcast(out_shape[d])};
  }
  return out;
}

// sdpa.py `_roles_unbroadcast`: roles of a broadcast's INPUT -- axes it
// expanded carry nothing.
Roles RolesUnbroadcast(const Roles& out_roles,
                       const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& out_shape,
                       const std::vector<int64_t>& dims) {
  Roles inn;
  for (size_t i = 0; i < dims.size(); i++) {
    const int64_t d = dims[i];
    if (d < 0 || d >= static_cast<int64_t>(out_shape.size()) ||
        i >= in_shape.size())
      Bail("broadcast_in_dim dimensions out of range");
    if (in_shape[i] == out_shape[d]) {
      if (d >= static_cast<int64_t>(out_roles.size()))
        Bail("broadcast roles do not match its rank");
      inn.push_back(out_roles[d]);
    } else if (in_shape[i] == 1) {
      inn.push_back(Role());
    } else {
      Bail("broadcast_in_dim is not a pure expansion");
    }
  }
  return inn;
}

// sdpa.py `_recipe`: the `(perm, shape)` reshaping a tensor with `roles` into
// `groups`.  `groups` is a list of atom lists, one per target dimension.
// Every atom of every group must appear, and the target order must be
// reachable by a permutation -- atoms merged into one source dimension have to
// stay adjacent.
Rec Recipe(const Roles& roles, const std::vector<std::vector<Atom>>& groups,
           const Atoms& atoms) {
  std::vector<Atom> flat;
  for (const std::vector<Atom>& g : groups)
    for (const Atom& a : g) flat.push_back(a);
  std::map<Atom, int> pos;
  for (size_t i = 0; i < flat.size(); i++)
    pos.emplace(flat[i], static_cast<int>(i));
  if (pos.size() != flat.size()) Bail("duplicate axis role");
  std::vector<std::pair<int, int>> order;   // (position of r[0], dimension)
  std::vector<int64_t> units;
  for (size_t d = 0; d < roles.size(); d++) {
    if (roles[d].empty()) {
      units.push_back(static_cast<int64_t>(d));
      continue;
    }
    for (const Atom& a : roles[d])
      if (pos.find(a) == pos.end()) Bail("axis is not part of the attention");
    order.emplace_back(pos.at(roles[d][0]), static_cast<int>(d));
  }
  std::sort(order.begin(), order.end());
  std::vector<Atom> seen;
  for (const std::pair<int, int>& od : order)
    for (const Atom& a : roles[od.second]) seen.push_back(a);
  if (seen != flat) Bail("axes cannot be permuted into the fused layout");
  std::vector<int64_t> perm;
  for (const std::pair<int, int>& od : order) perm.push_back(od.second);
  perm.insert(perm.end(), units.begin(), units.end());
  std::vector<int64_t> shape;
  for (const std::vector<Atom>& g : groups)
    shape.push_back(GroupSize(atoms, g));
  // Everything here is static, so decide NOW whether either step is a no-op:
  // the lowering runs per attention per execute, and an identity transpose
  // still builds a graph node.
  std::vector<int64_t> got;
  for (int64_t d : perm) got.push_back(GroupSize(atoms, roles[d]));
  Rec rec;
  bool identity = true;
  for (size_t i = 0; i < perm.size(); i++)
    identity = identity && perm[i] == static_cast<int64_t>(i);
  if (!identity) {
    rec.has_perm = true;
    rec.perm = perm;
  }
  if (got != shape) {
    rec.has_shape = true;
    rec.shape = shape;
  }
  return rec;
}

// sdpa.py `_out_recipe`: the `(pre_shape, perm, post_shape)` turning the fused
// `[B, N, Tq, Dv]` result -- whose atoms, expanded, are `order` -- into the
// root's layout.  Any step that is a no-op comes back absent.
struct OutRec {
  bool has_pre = false;
  std::vector<int64_t> pre;
  bool has_perm = false;
  std::vector<int64_t> perm;
  bool has_post = false;
  std::vector<int64_t> post;
};

OutRec OutRecipe(const Roles& roles, const std::vector<Atom>& order,
                 const Atoms& atoms, const std::vector<int64_t>& out_shape,
                 const std::vector<int64_t>& mlx_shape) {
  std::vector<int64_t> pre;
  for (const Atom& a : order) pre.push_back(AtomSize(atoms, a));
  std::map<Atom, int> idx;
  for (size_t i = 0; i < order.size(); i++)
    idx[order[i]] = static_cast<int>(i);
  std::vector<int64_t> perm;
  for (const Role& r : roles) {
    for (const Atom& a : r) {
      auto it = idx.find(a);
      if (it == idx.end()) Bail("a result axis is not produced by the kernel");
      perm.push_back(it->second);
    }
  }
  std::vector<int64_t> sorted = perm;
  std::sort(sorted.begin(), sorted.end());
  std::vector<int64_t> range(order.size());
  std::iota(range.begin(), range.end(), 0);
  if (sorted != range) Bail("result axes do not cover the fused output");
  std::vector<int64_t> permuted;
  for (int64_t p : perm) permuted.push_back(pre[p]);
  // As in `Recipe`: all static, so the no-op steps are dropped here rather
  // than re-tested on every execute.
  OutRec out;
  if (pre != mlx_shape) {
    out.has_pre = true;
    out.pre = pre;
  }
  bool identity = true;
  for (size_t i = 0; i < perm.size(); i++)
    identity = identity && perm[i] == static_cast<int64_t>(i);
  if (!identity) {
    out.has_perm = true;
    out.perm = perm;
  }
  if (permuted != out_shape) {
    out.has_post = true;
    out.post = out_shape;
  }
  return out;
}

// --------------------------------------------------------------------------
// candidate state (sdpa.py `_Cand`)
// --------------------------------------------------------------------------

// One call frame: what a callee's block arguments resolve to at the call site,
// and the frame that site itself lives in.  `nullptr` is Python's `None` --
// the block being interpreted, the only place the lowering can read a value
// from.
struct Frame {
  llvm::DenseMap<mlir::Value, std::pair<mlir::Value, const Frame*>> bind;
};

// sdpa.py's `m.mask` tuple, as found on the logits path.
struct MaskInfo {
  bool has = false;
  int kind = 0;             // 0 select, 1 add
  mlir::Value value;
  const Frame* frame = nullptr;
  Roles roles;
  double konst = 0.0;
};

struct Cand {
  mlir::ModuleOp module;
  Atoms atoms;
  // The absorbed ops, in the order they were found (the Python's insertion-
  // ordered dict).  `mlir::Operation*` is a stable key in C++, so there is no
  // need for the `_okey` first-result trick the transient python wrappers
  // force.
  std::vector<mlir::Operation*> ops;
  llvm::DenseSet<mlir::Operation*> op_set;
  // Every value on the logits chain, with its roles: the softmax's max is not
  // always reduced from the very value the subtract reads
  // (jax.nn.dot_product_attention transposes in between), so any of them is a
  // legitimate anchor.
  llvm::DenseMap<mlir::Value, Roles> chain;
  std::optional<Atom> k_atom;
  mlir::Operation* dot1 = nullptr;
  Roles lroles, rroles;
  double scale = 1.0;
  MaskInfo mask;
  double mask_mul = 1.0;
  mlir::Value exp_val;
  Roles exp_roles;
  // Frames outlive the walk that made them, and only the candidate owns them.
  std::vector<std::unique_ptr<Frame>> frames;

  void absorb(mlir::Operation* op, const Frame* frame) {
    // Ops inside a callee are never absorbed: the callee is shared with every
    // other call site, and skipping the CALL is what keeps this one's copy
    // from running.
    if (frame != nullptr) return;
    if (op_set.insert(op).second) ops.push_back(op);
  }
};

// sdpa.py `_snapshot` / `_restore`: a failed branch must leave nothing behind,
// because absorbing an op the rewrite then does not replace would delete it.
struct Snapshot {
  std::vector<mlir::Operation*> ops;
  llvm::DenseSet<mlir::Operation*> op_set;
  std::optional<Atom> k_atom;
  mlir::Operation* dot1 = nullptr;
  Roles lroles, rroles;
  double scale = 1.0;
  MaskInfo mask;
  double mask_mul = 1.0;
  mlir::Value exp_val;
  Roles exp_roles;
  int atoms_n = 0;
  std::map<Atom, int64_t> atoms_size;
  llvm::DenseMap<mlir::Value, Roles> chain;
};

Snapshot Save(const Cand& m) {
  Snapshot s;
  s.ops = m.ops;
  s.op_set = m.op_set;
  s.k_atom = m.k_atom;
  s.dot1 = m.dot1;
  s.lroles = m.lroles;
  s.rroles = m.rroles;
  s.scale = m.scale;
  s.mask = m.mask;
  s.mask_mul = m.mask_mul;
  s.exp_val = m.exp_val;
  s.exp_roles = m.exp_roles;
  s.atoms_n = m.atoms.n;
  s.atoms_size = m.atoms.size;
  s.chain = m.chain;
  return s;
}

void Restore(Cand& m, const Snapshot& s) {
  m.ops = s.ops;
  m.op_set = s.op_set;
  m.k_atom = s.k_atom;
  m.dot1 = s.dot1;
  m.lroles = s.lroles;
  m.rroles = s.rroles;
  m.scale = s.scale;
  m.mask = s.mask;
  m.mask_mul = s.mask_mul;
  m.exp_val = s.exp_val;
  m.exp_roles = s.exp_roles;
  m.atoms.n = s.atoms_n;
  m.atoms.size = s.atoms_size;
  m.chain = s.chain;
}

// --------------------------------------------------------------------------
// walking through calls (sdpa.py `_deref` / `_callee` / `_splat_float`)
// --------------------------------------------------------------------------

struct Resolved {
  mlir::Value value;
  const Frame* frame = nullptr;
};

// sdpa.py `_deref`: resolve callee block arguments outward through the call
// frames.  `frame == nullptr` means the value lives in the block being
// interpreted, which is the only place the lowering can read it from.  A block
// argument with no frame left to resolve through is such a value, so it comes
// back as a leaf and callers that need an operation reject it themselves.
Resolved Deref(mlir::Value v, const Frame* frame) {
  for (int i = 0; i < kMaxDepth; i++) {
    if (!mlir::isa<mlir::BlockArgument>(v)) return Resolved{v, frame};
    if (frame == nullptr) return Resolved{v, frame};
    auto it = frame->bind.find(v);
    if (it == frame->bind.end()) Bail("unbound block argument");
    v = it->second.first;
    frame = it->second.second;
  }
  Bail("call frames nested too deep");
}

// sdpa.py `_callee`: `(returned_value, inner_frame)` for a call-like op.
Resolved Callee(Cand& m, mlir::Operation* op, const Frame* frame) {
  const llvm::StringRef attr =
      OpName(op) == "func.call" ? "callee" : "decomposition";
  auto sym = op->getAttrOfType<mlir::FlatSymbolRefAttr>(attr);
  if (!sym) Bail(absl::StrCat(OpName(op), " without a resolvable callee"));
  auto fn = m.module ? m.module.lookupSymbol<mlir::func::FuncOp>(sym.getValue())
                     : mlir::func::FuncOp();
  if (!fn || op->getNumResults() != 1)
    Bail("unresolvable or multi-result call");
  if (fn.getBody().getBlocks().size() != 1) Bail("multi-block callee");
  mlir::Block& body = fn.getBody().front();
  if (body.empty()) Bail("empty callee");
  mlir::Operation* term = &body.back();
  if (term->getNumOperands() != 1) Bail("callee returns several values");
  auto inner = std::make_unique<Frame>();
  const size_t n =
      std::min<size_t>(body.getNumArguments(), op->getNumOperands());
  for (size_t i = 0; i < n; i++)
    inner->bind[body.getArgument(static_cast<unsigned>(i))] = {
        op->getOperand(static_cast<unsigned>(i)), frame};
  const Frame* held = inner.get();
  m.frames.push_back(std::move(inner));
  return Resolved{term->getOperand(0), held};
}

// The scalar of a rank-0 or splat dense constant (`_ir.splat_scalar_np` plus
// its rank-0 `dense_to_np` fallback), as a double.  Complex constants and
// anything the attribute cannot be read as fall out as nothing.
std::optional<double> ConstScalar(mlir::Operation* op) {
  auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
  if (!cst) return std::nullopt;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || dense.getNumElements() == 0 || !dense.isSplat())
    return std::nullopt;
  mlir::Type el = dense.getElementType();
  if (mlir::isa<mlir::FloatType>(el)) {
    llvm::APFloat v = dense.getSplatValue<llvm::APFloat>();
    bool lost = false;
    v.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
              &lost);
    return v.convertToDouble();
  }
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(el)) {
    if (it.getWidth() > 64) return std::nullopt;
    const llvm::APInt v = dense.getSplatValue<llvm::APInt>();
    if (it.isUnsigned() || it.getWidth() == 1)
      return static_cast<double>(v.getZExtValue());
    return static_cast<double>(v.getSExtValue());
  }
  return std::nullopt;
}

// sdpa.py `_splat_float`: the value of a splat float constant reached through
// converts, broadcasts and calls, else nothing.  Every rejection on the way is
// this function's `nullopt`, exactly as the Python swallows `_Reject`.
std::optional<double> SplatFloat(Cand& m, mlir::Value v, const Frame* frame) {
  for (int i = 0; i < kMaxDepth; i++) {
    Resolved r;
    try {
      r = Deref(v, frame);
    } catch (const Reject&) {
      return std::nullopt;
    }
    v = r.value;
    frame = r.frame;
    mlir::Operation* o = Owner(v);
    if (o == nullptr) return std::nullopt;
    const std::string name = OpName(o);
    if (IsIdentity(name) || name == "stablehlo.broadcast_in_dim") {
      if (o->getNumOperands() < 1) return std::nullopt;
      v = o->getOperand(0);
      continue;
    }
    if (IsCall(name)) {
      Resolved c;
      try {
        c = Callee(m, o, frame);
      } catch (const Reject&) {
        return std::nullopt;
      }
      v = c.value;
      frame = c.frame;
      continue;
    }
    if (name != "stablehlo.constant") return std::nullopt;
    return ConstScalar(o);
  }
  return std::nullopt;
}

// sdpa.py `_reduce_kind`: "add" / "maximum" for a single-input reduce with
// that body, else "".
std::string ReduceKind(mlir::Operation* op) {
  if (op == nullptr || OpName(op) != "stablehlo.reduce") return "";
  if (op->getNumOperands() != 2 || op->getNumRegions() < 1) return "";
  mlir::Region& region = op->getRegion(0);
  if (region.getBlocks().size() != 1) return "";
  mlir::Block& body = region.front();
  int count = 0;
  mlir::Operation* first = nullptr;
  for (mlir::Operation& o : body) {
    if (count == 0) first = &o;
    count++;
    if (count > 2) return "";
  }
  if (count != 2) return "";
  const std::string n = OpName(first);
  if (n == "stablehlo.add") return "add";
  if (n == "stablehlo.maximum") return "maximum";
  return "";
}

// --------------------------------------------------------------------------
// walking (sdpa.py, same section)
// --------------------------------------------------------------------------

struct Step {
  mlir::Operation* op;
  const Frame* frame;
};

struct DownResult {
  std::vector<Step> steps;
  mlir::Value value;
  const Frame* frame = nullptr;
};

using Anchors = llvm::DenseMap<mlir::Value, Roles>;

// sdpa.py `_down`: walk down through transparent ops.  `steps` is
// outermost-first.  `stop` halts the walk at a specific value: without it a
// `func.call` wrapping the value (maxtext's `@_where`) would be descended
// straight past, and the caller would compare against the callee's innards.
DownResult Down(Cand& m, mlir::Value v, const Frame* frame,
                const Anchors* stop, bool enter_calls) {
  DownResult out;
  Resolved cur = Deref(v, frame);
  for (int i = 0; i < kMaxDepth; i++) {
    if (stop != nullptr && cur.frame == nullptr &&
        stop->find(cur.value) != stop->end())
      break;
    mlir::Operation* o = Owner(cur.value);
    if (o == nullptr) break;
    const std::string name = OpName(o);
    if (IsTransparent(name)) {
      out.steps.push_back(Step{o, cur.frame});
      cur = Deref(o->getOperand(0), cur.frame);
      continue;
    }
    if (IsCall(name)) {
      if (!enter_calls) break;
      Resolved inner = Callee(m, o, cur.frame);
      out.steps.push_back(Step{o, cur.frame});
      cur = Deref(inner.value, inner.frame);
      continue;
    }
    double ident = 0.0;
    if (ClampIdentity(name, &ident)) {
      if (o->getNumOperands() != 2) break;
      int keep = -1;
      for (int side = 0; side < 2; side++) {
        std::optional<double> s =
            SplatFloat(m, o->getOperand(1 - side), cur.frame);
        if (s.has_value() && *s == ident) {
          keep = side;
          break;
        }
      }
      if (keep < 0) break;
      out.steps.push_back(Step{o, cur.frame});
      cur = Deref(o->getOperand(static_cast<unsigned>(keep)), cur.frame);
      continue;
    }
    break;
  }
  out.value = cur.value;
  out.frame = cur.frame;
  return out;
}

// sdpa.py `_up`: propagate roles UP through `steps` (outermost-first).
Roles Up(Cand& m, Roles roles, std::vector<int64_t> shape,
         const std::vector<Step>& steps) {
  for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
    mlir::Operation* op = it->op;
    const std::string n = OpName(op);
    if (IsSameShape(n)) continue;
    const std::vector<int64_t> out = ShapeOf(op->getResult(0));
    if (n == "stablehlo.transpose") {
      roles = RolesTranspose(roles, I64List(op, "permutation"));
    } else if (n == "stablehlo.reshape") {
      roles = RolesReshape(shape, roles, out, m.atoms);
    } else {
      roles = RolesBroadcast(shape, roles, out,
                             I64List(op, "broadcast_dimensions"), m.atoms);
    }
    shape = out;
  }
  return roles;
}

// sdpa.py `_down_roles`: propagate roles DOWN through `steps`.
Roles DownRoles(Cand& m, Roles roles, std::vector<int64_t> shape,
                const std::vector<Step>& steps) {
  for (const Step& st : steps) {
    const std::string n = OpName(st.op);
    if (IsSameShape(n)) continue;
    const std::vector<int64_t> inn = ShapeOf(st.op->getOperand(0));
    if (n == "stablehlo.transpose") {
      roles = RolesUntranspose(roles, I64List(st.op, "permutation"));
    } else if (n == "stablehlo.reshape") {
      roles = RolesReshape(shape, roles, inn, m.atoms);
    } else {
      roles = RolesUnbroadcast(roles, inn, shape,
                               I64List(st.op, "broadcast_dimensions"));
    }
    shape = inn;
  }
  return roles;
}

void AbsorbSteps(Cand& m, const std::vector<Step>& steps) {
  for (const Step& st : steps) m.absorb(st.op, st.frame);
}

// sdpa.py `_aligned`: `got` may only differ from `want` by replicating
// `allowed_missing`.
void Aligned(const Roles& got, const Roles& want,
             const std::set<Atom>& allowed_missing) {
  if (got.size() != want.size())
    Bail("rank mismatch in the softmax alignment");
  for (size_t i = 0; i < got.size(); i++) {
    if (got[i] == want[i]) continue;
    bool ok = IsBcastRole(got[i]) && !want[i].empty();
    for (const Atom& a : want[i])
      ok = ok && allowed_missing.find(a) != allowed_missing.end();
    if (ok) continue;
    Bail("softmax term is not aligned with the logits");
  }
}

// --------------------------------------------------------------------------
// the logits path (first dot -> softmax input)
// --------------------------------------------------------------------------

Roles Logits(Cand& m, mlir::Value v, const Frame* frame, int depth);

// sdpa.py `_at_dot1`.
Roles AtDot1(Cand& m, mlir::Operation* dot, const Frame* frame) {
  if (frame != nullptr) Bail("the first dot is inside a callee");
  if (m.dot1 != nullptr) Bail("two dots on the logits path");
  const DotDims d = ReadDotDims(dot);
  mlir::Value lhs = dot->getOperand(0);
  mlir::Value rhs = dot->getOperand(1);
  const std::vector<int64_t> lshape = ShapeOf(lhs);
  const std::vector<int64_t> rshape = ShapeOf(rhs);
  if (d.lc.empty() || d.lc.size() != d.rc.size() || d.lb.size() != d.rb.size())
    Bail("degenerate dot dimensions");
  auto at = [](const std::vector<int64_t>& shape, int64_t i) {
    if (i < 0 || i >= static_cast<int64_t>(shape.size()))
      Bail("dimension numbers out of range");
    return shape[i];
  };
  for (size_t i = 0; i < d.lc.size(); i++)
    if (at(lshape, d.lc[i]) != at(rshape, d.rc[i]))
      Bail("contracting dimensions disagree");
  for (size_t i = 0; i < d.lb.size(); i++)
    if (at(lshape, d.lb[i]) != at(rshape, d.rb[i]))
      Bail("batching dimensions disagree");
  if (!IsFloatEl(ElName(lhs)) || !IsFloatEl(ElName(rhs)))
    Bail("non-float attention");
  for (int64_t s : lshape)
    if (s == 0) Bail("empty attention");
  for (int64_t s : rshape)
    if (s == 0) Bail("empty attention");

  Atoms& A = m.atoms;
  std::vector<int64_t> lfree, rfree;
  for (int64_t i = 0; i < static_cast<int64_t>(lshape.size()); i++)
    if (!Holds(d.lb, i) && !Holds(d.lc, i)) lfree.push_back(i);
  for (int64_t i = 0; i < static_cast<int64_t>(rshape.size()); i++)
    if (!Holds(d.rb, i) && !Holds(d.rc, i)) rfree.push_back(i);
  for (size_t i = 0; i < d.lb.size(); i++)
    A.size[Atom{'b', static_cast<int>(i)}] = lshape[d.lb[i]];
  for (size_t i = 0; i < d.lc.size(); i++)
    A.size[Atom{'d', static_cast<int>(i)}] = lshape[d.lc[i]];
  for (size_t i = 0; i < lfree.size(); i++)
    A.size[Atom{'l', static_cast<int>(i)}] = lshape[lfree[i]];
  for (size_t i = 0; i < rfree.size(); i++)
    A.size[Atom{'r', static_cast<int>(i)}] = rshape[rfree[i]];

  Roles lroles(lshape.size()), rroles(rshape.size());
  for (size_t i = 0; i < d.lb.size(); i++)
    lroles[d.lb[i]] = Role{Atom{'b', static_cast<int>(i)}};
  for (size_t i = 0; i < d.rb.size(); i++)
    rroles[d.rb[i]] = Role{Atom{'b', static_cast<int>(i)}};
  for (size_t i = 0; i < d.lc.size(); i++)
    lroles[d.lc[i]] = Role{Atom{'d', static_cast<int>(i)}};
  for (size_t i = 0; i < d.rc.size(); i++)
    rroles[d.rc[i]] = Role{Atom{'d', static_cast<int>(i)}};
  // Free axes of size 1 carry no atom: jax inserts them freely (rank-5
  // `jax.nn.dot_product_attention`) and they must not become heads.
  for (size_t i = 0; i < lfree.size(); i++)
    lroles[lfree[i]] = lshape[lfree[i]] == 1
                           ? Role()
                           : Role{Atom{'l', static_cast<int>(i)}};
  for (size_t i = 0; i < rfree.size(); i++)
    rroles[rfree[i]] = rshape[rfree[i]] == 1
                           ? Role()
                           : Role{Atom{'r', static_cast<int>(i)}};

  m.dot1 = dot;
  m.lroles = lroles;
  m.rroles = rroles;
  m.absorb(dot, frame);
  Roles out;
  for (size_t i = 0; i < d.lb.size(); i++)
    out.push_back(Role{Atom{'b', static_cast<int>(i)}});
  for (int64_t dd : lfree) out.push_back(lroles[dd]);
  for (int64_t dd : rfree) out.push_back(rroles[dd]);
  return out;
}

// sdpa.py `_logits_inner`.
Roles LogitsInner(Cand& m, mlir::Value v, const Frame* frame, int depth) {
  mlir::Operation* o = Owner(v);
  if (o == nullptr) Bail("logits path reaches a block argument");
  const std::string name = OpName(o);

  if (name == "stablehlo.dot_general") return AtDot1(m, o, frame);

  if (IsCall(name)) {
    Resolved inner = Callee(m, o, frame);
    Roles roles = Logits(m, inner.value, inner.frame, depth + 1);
    m.absorb(o, frame);
    return roles;
  }

  if (IsIdentity(name)) {
    Roles roles = Logits(m, o->getOperand(0), frame, depth + 1);
    m.absorb(o, frame);
    return roles;
  }

  if (name == "stablehlo.transpose") {
    Roles roles = Logits(m, o->getOperand(0), frame, depth + 1);
    m.absorb(o, frame);
    return RolesTranspose(roles, I64List(o, "permutation"));
  }

  if (name == "stablehlo.reshape") {
    Roles roles = Logits(m, o->getOperand(0), frame, depth + 1);
    m.absorb(o, frame);
    return RolesReshape(ShapeOf(o->getOperand(0)), roles,
                        ShapeOf(o->getResult(0)), m.atoms);
  }

  if (name == "stablehlo.multiply") {
    if (o->getNumOperands() != 2) Bail("multiply is not binary");
    for (int i = 0; i < 2; i++) {
      std::optional<double> s =
          SplatFloat(m, o->getOperand(static_cast<unsigned>(1 - i)), frame);
      if (!s.has_value()) continue;
      const Snapshot saved = Save(m);
      Roles roles;
      try {
        roles = Logits(m, o->getOperand(static_cast<unsigned>(i)), frame,
                       depth + 1);
      } catch (const Reject&) {
        // A failed branch must leave nothing behind: absorbing an op the
        // rewrite then does not replace would delete it.
        Restore(m, saved);
        continue;
      }
      m.scale *= *s;
      if (m.mask.has) {
        // This factor multiplies an already-applied mask as well:
        // (L + mask) * s == L*s + mask*s.
        m.mask_mul *= *s;
      }
      m.absorb(o, frame);
      return roles;
    }
    Bail("neither multiply operand is a splat scale");
  }

  if (name == "stablehlo.divide") {
    // `logits / sqrt(D)`.  MLX only takes a multiplicative scale, so the
    // reciprocal is formed here -- exact whenever the divisor is a power of
    // two (every D in {4, 16, 64, 256}), and otherwise off by at most one ULP
    // of the logits, far inside the softmax's own error.
    if (o->getNumOperands() != 2) Bail("divide is not binary");
    std::optional<double> s = SplatFloat(m, o->getOperand(1), frame);
    if (!s.has_value() || *s == 0.0)
      Bail("divide by a non-splat on the logits path");
    Roles roles = Logits(m, o->getOperand(0), frame, depth + 1);
    const double f = 1.0 / *s;
    m.scale *= f;
    if (m.mask.has) m.mask_mul *= f;
    m.absorb(o, frame);
    return roles;
  }

  if (name == "stablehlo.add") {
    if (o->getNumOperands() != 2) Bail("add is not binary");
    for (int i = 0; i < 2; i++) {
      const Snapshot saved = Save(m);
      Roles roles;
      try {
        roles = Logits(m, o->getOperand(static_cast<unsigned>(i)), frame,
                       depth + 1);
      } catch (const Reject&) {
        Restore(m, saved);
        continue;
      }
      if (m.mask.has) Bail("two masks on one logits path");
      m.mask.has = true;
      m.mask.kind = 1;                       // "add"
      m.mask.value = o->getOperand(static_cast<unsigned>(1 - i));
      m.mask.frame = frame;
      m.mask.roles = roles;
      m.mask.konst = 0.0;
      m.absorb(o, frame);
      return roles;
    }
    Bail("neither add operand leads to the first dot");
  }

  if (name == "stablehlo.select") {
    if (o->getNumOperands() != 3) Bail("select is not ternary");
    Roles roles;
    try {
      roles = Logits(m, o->getOperand(1), frame, depth + 1);
    } catch (const Reject&) {
      // select(pred, C, L) would need the predicate negated; jax does not
      // emit it and guessing is not worth a silent sign error.
      Bail("select's true branch does not lead to the dot");
    }
    if (m.mask.has) Bail("two masks on one logits path");
    std::optional<double> c = SplatFloat(m, o->getOperand(2), frame);
    if (!c.has_value()) Bail("select's false branch is not a splat");
    const std::string el = ElName(o->getResult(0));
    if (!IsFloatEl(el)) Bail(absl::StrCat("select on ", el));
    if (!(*c < 0 && (std::isinf(*c) || -*c >= kMaskFraction * FiniteMax(el)))) {
      // Not a mask sentinel: `select` and `add` are then genuinely different
      // functions, so this has to run literally.
      Bail(absl::StrFormat("select constant %g is not a mask sentinel", *c));
    }
    m.mask.has = true;
    m.mask.kind = 0;                         // "select"
    m.mask.value = o->getOperand(0);
    m.mask.frame = frame;
    m.mask.roles = roles;
    m.mask.konst = *c;
    m.absorb(o, frame);
    return roles;
  }

  Bail(absl::StrCat(name, " on the logits path"));
}

// sdpa.py `_logits`: the role vector of a value on the first-dot -> softmax
// path.  Records every outer value it passes through, so the softmax's max
// reduction can be anchored to whichever of them it actually reads.
Roles Logits(Cand& m, mlir::Value v, const Frame* frame, int depth) {
  if (depth > kMaxDepth) Bail("logits chain too deep");
  Resolved r = Deref(v, frame);
  Roles roles = LogitsInner(m, r.value, r.frame, depth);
  if (r.frame == nullptr) m.chain[r.value] = roles;
  return roles;
}

// --------------------------------------------------------------------------
// the softmax reductions
// --------------------------------------------------------------------------

// sdpa.py `_check_reduce`: verify `v` is `broadcast(reduce(anchor, kind))`
// reduced over `k`.  Returns the roles of the reduce's result.  `anchors` maps
// the values the reduce is allowed to be computed from to their roles -- the
// logits chain for the max, the exponential alone for the sum.  Keeping the
// two sets apart is load-bearing: a sum reduced over the LOGITS would
// otherwise validate as the softmax denominator and silently change the
// result.  `missing` is the set of atoms the broadcast back up may replicate
// (the key axis for the max and for a pre-dot normalization; V's feature axes
// when the normalization is deferred past the second dot); nothing means "the
// key axis", which the max reduction is what discovers.
Roles CheckReduce(Cand& m, mlir::Value v, const Frame* frame,
                  const std::string& kind, const Anchors& anchors,
                  const Roles& want_roles,
                  const std::optional<std::set<Atom>>& missing) {
  DownResult down = Down(m, v, frame, /*stop=*/nullptr, /*enter_calls=*/true);
  if (down.frame != nullptr)
    Bail("the softmax reduction lives inside a callee");
  mlir::Operation* red = Owner(down.value);
  if (red == nullptr || ReduceKind(red) != kind)
    Bail(absl::StrCat("softmax has no ", kind, " reduction"));
  std::optional<double> init = SplatFloat(m, red->getOperand(1), nullptr);
  if (kind == "add") {
    // A non-zero init would change the denominator outright.
    if (!init.has_value() || *init != 0.0)
      Bail("sum reduction has a non-zero init");
  } else if (!init.has_value() || !(std::isinf(*init) && *init < 0)) {
    Bail("max reduction does not start at -inf");
  }

  DownResult in = Down(m, red->getOperand(0), nullptr, &anchors,
                       /*enter_calls=*/true);
  auto anchor = anchors.find(in.value);
  if (in.frame != nullptr || anchor == anchors.end())
    Bail(absl::StrCat(kind, " reduction is not computed from the softmax "
                            "input"));
  const std::vector<int64_t> in_shape = ShapeOf(red->getOperand(0));
  const Roles in_roles =
      Up(m, anchor->second, ShapeOf(in.value), in.steps);
  if (in_roles.size() != in_shape.size())
    Bail("reduce input roles do not match its rank");

  const std::vector<int64_t> dims = I64List(red, "dimensions");
  std::vector<Atom> flat;
  for (int64_t d : dims) {
    if (d < 0 || d >= static_cast<int64_t>(in_roles.size()))
      Bail("reduce dimensions out of range");
    for (const Atom& a : in_roles[d]) flat.push_back(a);
  }
  if (!m.k_atom.has_value()) {
    if (flat.size() != 1) Bail("softmax reduces more than one axis");
    m.k_atom = flat[0];
  }
  if (flat.size() != 1 || flat[0] != *m.k_atom)
    Bail("softmax does not reduce the key axis");

  Roles out_roles;
  std::vector<int64_t> out_shape;
  for (size_t d = 0; d < in_roles.size(); d++) {
    if (Holds(dims, static_cast<int64_t>(d))) continue;
    out_roles.push_back(in_roles[d]);
    out_shape.push_back(in_shape[d]);
  }
  const Roles roles = Up(m, out_roles, out_shape, down.steps);
  std::set<Atom> allowed;
  if (missing.has_value()) {
    allowed = *missing;
  } else {
    allowed.insert(*m.k_atom);
  }
  Aligned(roles, want_roles, allowed);
  AbsorbSteps(m, down.steps);
  AbsorbSteps(m, in.steps);
  m.absorb(red, nullptr);
  return roles;
}

// --------------------------------------------------------------------------
// the probability path (softmax output -> second dot)
// --------------------------------------------------------------------------

// sdpa.py `_probs`: walk a second-dot operand down to the `exp`.  `normalized`
// says the softmax division already happened, so the second dot's result is
// the final output.
std::pair<Roles, bool> Probs(Cand& m, mlir::Value v, const Frame* frame,
                             int depth) {
  if (depth > kMaxDepth) Bail("probability chain too deep");
  Resolved r = Deref(v, frame);
  mlir::Operation* o = Owner(r.value);
  if (o == nullptr) Bail("probability path reaches a block argument");
  frame = r.frame;
  const std::string name = OpName(o);

  if (name == "stablehlo.exponential") {
    DownResult down =
        Down(m, o->getOperand(0), frame, /*stop=*/nullptr,
             /*enter_calls=*/true);
    mlir::Operation* so = Owner(down.value);
    if (so == nullptr || OpName(so) != "stablehlo.subtract")
      Bail("exp's operand is not a max subtraction");
    if (down.frame != nullptr || frame != nullptr)
      Bail("the softmax lives inside a callee");
    if (so->getNumOperands() != 2) Bail("subtract is not binary");
    Roles roles = Logits(m, so->getOperand(0), down.frame, 0);
    // The max reduction is what discovers the key axis (hence which operand of
    // the first dot is K); everything downstream verifies against it.
    CheckReduce(m, so->getOperand(1), down.frame, "maximum", m.chain, roles,
                std::nullopt);
    AbsorbSteps(m, down.steps);
    m.absorb(so, down.frame);
    m.absorb(o, frame);
    m.exp_val = o->getResult(0);
    m.exp_roles = Up(m, roles, ShapeOf(so->getOperand(0)), down.steps);
    return {m.exp_roles, false};
  }

  if (IsIdentity(name)) {
    std::pair<Roles, bool> got = Probs(m, o->getOperand(0), frame, depth + 1);
    m.absorb(o, frame);
    return got;
  }

  if (IsCall(name)) {
    Resolved inner = Callee(m, o, frame);
    std::pair<Roles, bool> got = Probs(m, inner.value, inner.frame, depth + 1);
    m.absorb(o, frame);
    return got;
  }

  if (name == "stablehlo.transpose") {
    std::pair<Roles, bool> got = Probs(m, o->getOperand(0), frame, depth + 1);
    m.absorb(o, frame);
    return {RolesTranspose(got.first, I64List(o, "permutation")), got.second};
  }

  if (name == "stablehlo.reshape") {
    std::pair<Roles, bool> got = Probs(m, o->getOperand(0), frame, depth + 1);
    m.absorb(o, frame);
    return {RolesReshape(ShapeOf(o->getOperand(0)), got.first,
                         ShapeOf(o->getResult(0)), m.atoms),
            got.second};
  }

  if (name == "stablehlo.divide") {
    if (o->getNumOperands() != 2) Bail("divide is not binary");
    std::pair<Roles, bool> got = Probs(m, o->getOperand(0), frame, depth + 1);
    if (got.second) Bail("two normalizations");
    if (m.exp_val == nullptr) Bail("normalization without an exponential");
    Anchors anchors;
    anchors[m.exp_val] = m.exp_roles;
    CheckReduce(m, o->getOperand(1), frame, "add", anchors, got.first,
                std::nullopt);
    m.absorb(o, frame);
    return {got.first, true};
  }

  Bail(absl::StrCat(name, " on the probability path"));
}

// --------------------------------------------------------------------------
// candidates
// --------------------------------------------------------------------------

// sdpa.py `_pick_query_axis`'s resolved mask: the base value the lowering
// reads, the axes it varies along, and the two constants.
struct ResolvedMask {
  bool has = false;
  int kind = 0;
  mlir::Value base;
  Roles roles;
  double konst = 0.0;
  double mul = 1.0;
};

// sdpa.py `_pick_query_axis`: choose the query-sequence atom, and resolve the
// mask to its base.
//
// Every query-side axis indexes an independent softmax row, so with no mask
// the split between "the query axis" and "extra head axes" is free, and the
// largest axis is taken (the one MLX's kernel wants to tile over).  A mask
// forces the choice: MLX broadcasts it against `[B, N, Tq, Tk]`, so the axis
// the mask varies along has to be `Tq`.
std::optional<Atom> PickQueryAxis(Cand& m, const std::vector<Atom>& q_free,
                                  ResolvedMask* out) {
  auto largest = [&]() -> std::optional<Atom> {
    if (q_free.empty()) return std::nullopt;
    auto it = std::max_element(q_free.begin(), q_free.end(),
                               [&](const Atom& a, const Atom& b) {
                                 return AtomSize(m.atoms, a) <
                                        AtomSize(m.atoms, b);
                               });
    return *it;
  };
  if (!m.mask.has) {
    out->has = false;
    return largest();
  }
  // Never descend into a callee here: the mask only has to be reduced to a
  // value the lowering can read and to the axes it varies along, and jax wraps
  // the mask's own construction in calls (`@tril`) whose insides are not ours
  // to skip.
  DownResult down = Down(m, m.mask.value, m.mask.frame, /*stop=*/nullptr,
                         /*enter_calls=*/false);
  if (down.frame != nullptr) Bail("the mask is computed inside a callee");
  const Roles base_roles =
      DownRoles(m, m.mask.roles, ShapeOf(m.mask.value), down.steps);
  if (base_roles.size() != ShapeOf(down.value).size())
    Bail("mask roles do not match its rank");
  std::set<Atom> have;
  for (const Role& r : base_roles)
    for (const Atom& a : r) have.insert(a);
  std::vector<Atom> varying;
  for (const Atom& a : q_free)
    if (have.find(a) != have.end()) varying.push_back(a);
  if (varying.size() > 1) Bail("the mask varies along several query axes");
  std::optional<Atom> q_atom =
      varying.empty() ? largest() : std::optional<Atom>(varying[0]);
  AbsorbSteps(m, down.steps);
  out->has = true;
  out->kind = m.mask.kind;
  out->base = down.value;
  out->roles = base_roles;
  out->konst = m.mask.konst;
  out->mul = m.mask_mul;
  return q_atom;
}

// sdpa.py `_mask_recipe`: lay the mask out as the rank-4 tensor MLX
// broadcasts.  Each of the four slots must be either the whole slot or absent
// (size 1).  A mask varying along only PART of a slot -- one head axis of a
// GQA pair, say -- could only be expressed by materializing it to the full
// slot, which is the allocation this recognizer exists to avoid.
Rec MaskRecipe(const Cand& m, const ResolvedMask& mask,
               const std::vector<Atom>& batch_atoms,
               const std::vector<Atom>& n_atoms,
               const std::vector<Atom>& qslot, const Atoms& atoms) {
  std::set<Atom> have;
  for (const Role& r : mask.roles)
    for (const Atom& a : r) have.insert(a);
  std::vector<std::vector<Atom>> groups;
  const std::vector<Atom> kslot{*m.k_atom};
  for (const std::vector<Atom>* slot : {&batch_atoms, &n_atoms, &qslot,
                                        &kslot}) {
    std::vector<Atom> present;
    for (const Atom& a : *slot)
      if (have.find(a) != have.end()) present.push_back(a);
    if (!present.empty() && present.size() != slot->size())
      Bail("the mask varies along part of a broadcast axis");
    groups.push_back(std::move(present));
  }
  return Recipe(mask.roles, groups, atoms);
}

// sdpa.py `_find_norm`: forward from the second dot's result to the deferred
// normalization.  Real LLM lowerings divide by the softmax denominator AFTER
// the values dot (`O = (e @ V) / broadcast(sum e)`) -- legal because the
// denominator is constant along V's feature axis, and cheaper when there are
// more keys than features.
std::pair<mlir::Operation*, Roles> FindNorm(Cand& m, mlir::Operation* dot2,
                                            const Roles& roles,
                                            const std::set<Atom>& v_atoms) {
  struct Item {
    mlir::Value value;
    Roles roles;
    std::vector<int64_t> shape;
    std::vector<mlir::Operation*> path;
  };
  std::vector<Item> queue;
  queue.push_back(Item{dot2->getResult(0), roles, ShapeOf(dot2->getResult(0)),
                       {}});
  size_t head = 0;
  int seen = 0;
  while (head < queue.size()) {
    const Item item = queue[head++];
    seen++;
    if (seen > kMaxFanout) Bail("normalization search too wide");
    for (mlir::Operation* u : item.value.getUsers()) {
      const std::string n = OpName(u);
      if (n == "stablehlo.divide" && u->getNumOperands() == 2 &&
          u->getOperand(0) == item.value) {
        const Snapshot saved = Save(m);
        try {
          if (m.exp_val == nullptr)
            Bail("normalization without an exponential");
          Anchors anchors;
          anchors[m.exp_val] = m.exp_roles;
          CheckReduce(m, u->getOperand(1), nullptr, "add", anchors, item.roles,
                      v_atoms);
        } catch (const Reject&) {
          Restore(m, saved);
          continue;
        }
        for (mlir::Operation* op : item.path) m.absorb(op, nullptr);
        return {u, item.roles};
      }
      // Forward through pure reassociation only: a broadcast on this path
      // would replicate the output, which is not a normalization.
      if (!IsForward(n) || u->getNumOperands() < 1 ||
          u->getOperand(0) != item.value || u->getNumResults() != 1)
        continue;
      std::vector<mlir::Operation*> path = item.path;
      path.push_back(u);
      queue.push_back(Item{u->getResult(0),
                           Up(m, item.roles, item.shape,
                              {Step{u, nullptr}}),
                           ShapeOf(u->getResult(0)), std::move(path)});
    }
  }
  Bail("no softmax normalization after the second dot");
}

// sdpa.py `_dot2_roles`: the role vector of the second dot's result.
Roles Dot2Roles(const Roles& p_roles, const Roles& v_roles,
                const std::vector<int64_t>& pb, const std::vector<int64_t>& pc,
                const std::vector<int64_t>& vb, const std::vector<int64_t>& vc,
                int pside, const std::vector<int64_t>& pshape,
                const std::vector<int64_t>& vshape) {
  Roles p_free, v_free;
  for (int64_t d = 0; d < static_cast<int64_t>(pshape.size()); d++)
    if (!Holds(pb, d) && !Holds(pc, d)) p_free.push_back(p_roles[d]);
  for (int64_t d = 0; d < static_cast<int64_t>(vshape.size()); d++)
    if (!Holds(vb, d) && !Holds(vc, d)) v_free.push_back(v_roles[d]);
  const Roles& lhs_free = pside == 0 ? p_free : v_free;
  const Roles& rhs_free = pside == 0 ? v_free : p_free;
  Roles out;
  for (int64_t d : pb) out.push_back(p_roles[d]);
  out.insert(out.end(), lhs_free.begin(), lhs_free.end());
  out.insert(out.end(), rhs_free.begin(), rhs_free.end());
  return out;
}

int TapeCodeOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) Bail("a value that is not a ranked tensor");
  std::optional<int> code = TapeDtypeCode(t.getElementType());
  if (!code.has_value()) Bail("an element type with no tape dtype code");
  return *code;
}

// sdpa.py `_try_side`.
std::unique_ptr<SdpaMatch> TrySide(mlir::ModuleOp module,
                                   mlir::Operation* dot2, int pside) {
  Cand m;
  m.module = module;
  std::pair<Roles, bool> got =
      Probs(m, dot2->getOperand(static_cast<unsigned>(pside)), nullptr, 0);
  const Roles p_roles = got.first;
  const bool normalized = got.second;
  if (!m.k_atom.has_value()) Bail("no key axis");

  const DotDims d = ReadDotDims(dot2);
  const std::vector<int64_t>& pb = pside == 0 ? d.lb : d.rb;
  const std::vector<int64_t>& pc = pside == 0 ? d.lc : d.rc;
  const std::vector<int64_t>& vb = pside == 0 ? d.rb : d.lb;
  const std::vector<int64_t>& vc = pside == 0 ? d.rc : d.lc;
  mlir::Value vval = dot2->getOperand(static_cast<unsigned>(1 - pside));
  const std::vector<int64_t> vshape = ShapeOf(vval);
  const std::vector<int64_t> pshape =
      ShapeOf(dot2->getOperand(static_cast<unsigned>(pside)));
  if (p_roles.size() != pshape.size())
    Bail("probability roles do not match the second dot");
  if (!IsFloatEl(ElName(vval))) Bail("empty or non-float values");
  for (int64_t s : vshape)
    if (s == 0) Bail("empty or non-float values");
  if (pb.size() != vb.size() || pc.size() != vc.size() || pc.empty())
    Bail("degenerate second dot");

  auto at = [](const std::vector<int64_t>& shape, int64_t i) {
    if (i < 0 || i >= static_cast<int64_t>(shape.size()))
      Bail("dimension numbers out of range");
    return shape[i];
  };
  // The probabilities contract with V over the key axis...
  std::vector<Atom> pc_atoms;
  for (int64_t dd : pc) {
    if (dd < 0 || dd >= static_cast<int64_t>(p_roles.size()))
      Bail("dimension numbers out of range");
    for (const Atom& a : p_roles[dd]) pc_atoms.push_back(a);
  }
  if (pc_atoms.size() != 1 || pc_atoms[0] != *m.k_atom)
    Bail("the second dot does not contract the key axis");
  int64_t vc_extent = 1;
  for (int64_t dd : vc) vc_extent *= at(vshape, dd);
  if (vc_extent != AtomSize(m.atoms, *m.k_atom))
    Bail("the second dot contracts the wrong extent");

  Atoms& A = m.atoms;
  std::vector<int64_t> vfree;
  for (int64_t i = 0; i < static_cast<int64_t>(vshape.size()); i++)
    if (!Holds(vb, i) && !Holds(vc, i)) vfree.push_back(i);
  Roles vroles(vshape.size());
  // ...and share every batch/head axis with them.
  std::vector<Atom> shared;
  for (size_t i = 0; i < vb.size(); i++) {
    const int64_t dd = vb[i];
    if (dd < 0 || dd >= static_cast<int64_t>(vroles.size()) ||
        pb[i] < 0 || pb[i] >= static_cast<int64_t>(p_roles.size()))
      Bail("dimension numbers out of range");
    vroles[dd] = p_roles[pb[i]];
    for (const Atom& a : vroles[dd]) shared.push_back(a);
    if (vshape[dd] != GroupSize(A, vroles[dd]))
      Bail("second dot batching sizes disagree");
  }
  for (size_t i = 0; i < vc.size(); i++) {
    const int64_t dd = vc[i];
    if (dd < 0 || dd >= static_cast<int64_t>(vroles.size()))
      Bail("dimension numbers out of range");
    // V's contracting axes carry the key atom: the first takes it and the rest
    // have to be unit, so that one recipe can place it.
    if (i == 0) {
      vroles[dd] = Role{*m.k_atom};
    } else if (vshape[dd] != 1) {
      Bail("V contracts several non-unit axes");
    } else {
      vroles[dd] = Role();
    }
  }
  for (size_t i = 0; i < vfree.size(); i++) {
    const int64_t dd = vfree[i];
    A.size[Atom{'v', static_cast<int>(i)}] = vshape[dd];
    vroles[dd] = vshape[dd] == 1 ? Role()
                                 : Role{Atom{'v', static_cast<int>(i)}};
  }
  std::vector<Atom> v_atoms;
  for (int64_t dd : vfree)
    for (const Atom& a : vroles[dd]) v_atoms.push_back(a);

  const DotDims d1 = ReadDotDims(m.dot1);
  std::vector<Atom> b_atoms, d_atoms;
  for (size_t i = 0; i < d1.lb.size(); i++)
    b_atoms.push_back(Atom{'b', static_cast<int>(i)});
  for (size_t i = 0; i < d1.lc.size(); i++)
    d_atoms.push_back(Atom{'d', static_cast<int>(i)});
  {
    std::vector<Atom> a = shared, b = b_atoms;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) Bail("batch axes are not shared with the values");
  }

  Roles q_roles, k_roles;
  mlir::Value qval, kval;
  if (m.k_atom->first == 'l') {
    q_roles = m.rroles;
    k_roles = m.lroles;
    qval = m.dot1->getOperand(1);
    kval = m.dot1->getOperand(0);
  } else {
    q_roles = m.lroles;
    k_roles = m.rroles;
    qval = m.dot1->getOperand(0);
    kval = m.dot1->getOperand(1);
  }
  auto free_atoms = [](const Roles& roles) {
    std::vector<Atom> out;
    for (const Role& r : roles)
      for (const Atom& a : r)
        if (a.first == 'l' || a.first == 'r') out.push_back(a);
    return out;
  };
  const std::vector<Atom> k_extra = free_atoms(k_roles);
  if (k_extra.size() != 1 || k_extra[0] != *m.k_atom)
    Bail("the key operand has extra free axes");
  const std::vector<Atom> q_free = free_atoms(q_roles);
  std::vector<Atom> p_free;
  for (int64_t dd = 0; dd < static_cast<int64_t>(pshape.size()); dd++) {
    if (Holds(pb, dd) || Holds(pc, dd)) continue;
    for (const Atom& a : p_roles[dd]) p_free.push_back(a);
  }
  {
    std::vector<Atom> a = p_free, b = q_free;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) Bail("query axes are not free in the second dot");
  }
  if (ElName(qval) != ElName(vval)) Bail("mixed operand dtypes");

  ResolvedMask mask;
  const std::optional<Atom> q_atom = PickQueryAxis(m, q_free, &mask);
  std::vector<Atom> extra;
  for (const Atom& a : q_free)
    if (!(q_atom.has_value() && a == *q_atom)) extra.push_back(a);
  std::vector<Atom> qslot;
  if (q_atom.has_value()) qslot.push_back(*q_atom);

  const Roles d2_roles = Dot2Roles(p_roles, vroles, pb, pc, vb, vc, pside,
                                   pshape, vshape);
  mlir::Operation* root = nullptr;
  Roles root_roles;
  if (normalized) {
    root = dot2;
    root_roles = d2_roles;
  } else {
    std::set<Atom> vset(v_atoms.begin(), v_atoms.end());
    std::pair<mlir::Operation*, Roles> found =
        FindNorm(m, dot2, d2_roles, vset);
    root = found.first;
    root_roles = found.second;
  }
  m.absorb(dot2, nullptr);
  const std::vector<int64_t> out_shape = ShapeOf(root->getResult(0));

  auto mm = std::make_unique<SdpaMatch>();
  mm->q = qval;
  mm->k = kval;
  mm->v = vval;
  mm->scale = m.scale;
  mm->root = root;
  mm->dtype = TapeCodeOf(qval);
  mm->out_dtype = TapeCodeOf(root->getResult(0));
  for (mlir::Operation* o : m.ops)
    if (o != root) mm->ops.push_back(o);

  // Splitting the batching axes between MLX's B and N is FREE for correctness
  // -- the kernel is independent per (b, n) pair, and the `h // g` head
  // pairing holds for any split because the kv-head axes stay MAJOR within N.
  // It is NOT free for expressiveness: a mask must cover a whole slot or none
  // of it, so a bias varying along an outer batching axis needs that axis in
  // B.  Try the conventional split (last batching axis = the head) first, then
  // the rest.
  const int natural =
      std::max<int>(static_cast<int>(b_atoms.size()) - 1, 0);
  std::vector<int> splits{natural};
  for (int x = 0; x <= static_cast<int>(b_atoms.size()); x++)
    if (x != natural) splits.push_back(x);
  Reject last{"no batch/head split expresses this attention"};
  for (int s : splits) {
    const std::vector<Atom> batch_atoms(b_atoms.begin(), b_atoms.begin() + s);
    const std::vector<Atom> head_atoms(b_atoms.begin() + s, b_atoms.end());
    // The query heads are (kv head, group) with the kv head MAJOR, which is
    // exactly MLX's `h // g` pairing for grouped-query attention.
    std::vector<Atom> n_atoms = head_atoms;
    n_atoms.insert(n_atoms.end(), extra.begin(), extra.end());
    try {
      mm->q_rec = Recipe(q_roles, {batch_atoms, n_atoms, qslot, d_atoms}, A);
      mm->k_rec = Recipe(k_roles,
                         {batch_atoms, head_atoms, {*m.k_atom}, d_atoms}, A);
      mm->v_rec = Recipe(vroles,
                         {batch_atoms, head_atoms, {*m.k_atom}, v_atoms}, A);
      if (mask.has) {
        mm->mask_rec = MaskRecipe(m, mask, batch_atoms, n_atoms, qslot, A);
      }
      std::vector<Atom> order = batch_atoms;
      order.insert(order.end(), n_atoms.begin(), n_atoms.end());
      order.insert(order.end(), qslot.begin(), qslot.end());
      order.insert(order.end(), v_atoms.begin(), v_atoms.end());
      const std::vector<int64_t> mlx_shape{
          GroupSize(A, batch_atoms), GroupSize(A, n_atoms),
          GroupSize(A, qslot), GroupSize(A, v_atoms)};
      const OutRec out =
          OutRecipe(root_roles, order, A, out_shape, mlx_shape);
      mm->has_pre = out.has_pre;
      mm->pre = out.pre;
      mm->has_out_perm = out.has_perm;
      mm->out_perm = out.perm;
      mm->has_post = out.has_post;
      mm->post = out.post;
      mm->has_mask = mask.has;
      if (mask.has) {
        mm->mask_kind = mask.kind;
        mm->mask_base = mask.base;
        mm->mask_const = mask.konst;
        mm->mask_mul = mask.mul;
      }
      mm->name = absl::StrFormat(
          "B%dN%dQ%dK%dD%d", GroupSize(A, batch_atoms), GroupSize(A, n_atoms),
          q_atom.has_value() ? AtomSize(A, *q_atom) : 1,
          AtomSize(A, *m.k_atom), GroupSize(A, d_atoms));
      return mm;
    } catch (const Reject& e) {
      last = e;
      continue;
    }
  }
  throw last;
}

// sdpa.py `_try`: read `dot2` as attention's probabilities-times-values dot,
// or reject.  Either operand can be the probabilities: jax puts them on the
// right in `jax.nn.dot_product_attention` and maxtext, on the left in
// hand-rolled einsum attention.
std::unique_ptr<SdpaMatch> Try(mlir::ModuleOp module, mlir::Operation* dot2) {
  for (int side = 0; side < 2; side++) {
    try {
      return TrySide(module, dot2, side);
    } catch (const Reject&) {
      continue;
    }
  }
  Bail("not attention");
}

// --------------------------------------------------------------------------
// analysis
// --------------------------------------------------------------------------

// sdpa.py `_escapes`: true if anything `m` absorbs is still read from outside
// the chain.  An op with no results can never be a chain member, so a
// consumer like `func.return` always counts as an escape (which is what the
// Python's `_okey(...) is None` does).
bool Escapes(const SdpaMatch& m,
             const llvm::DenseSet<mlir::Operation*>& keys) {
  for (mlir::Operation* op : m.ops)
    for (mlir::Value r : op->getResults())
      for (mlir::Operation* u : r.getUsers())
        if (u->getNumResults() == 0 || !keys.contains(u)) return true;
  return false;
}

// sdpa.py `_exclusive`: drop candidates whose absorbed ops are needed
// elsewhere.  Every absorbed op must be consumed only by the chain itself:
// attention probabilities that also feed a dropout mask, or are returned as an
// auxiliary output, have to be materialized anyway.
//
// Iterated to a fixpoint, exactly as qmm's `Prune` is.  Dropping one candidate
// turns its root back into an ordinary op, which can be the outside consumer
// that disqualifies another; settling for a single pass would leave that other
// candidate skipping an op the now-literal root still reads, which is a
// missing slot at execute time.
std::vector<SdpaMatch*> Exclusive(std::vector<SdpaMatch*> live) {
  while (true) {
    llvm::DenseSet<mlir::Operation*> keys;
    for (SdpaMatch* m : live) {
      for (mlir::Operation* o : m->ops) keys.insert(o);
      keys.insert(m->root);
    }
    std::vector<SdpaMatch*> kept;
    for (SdpaMatch* m : live) {
      if (Escapes(*m, keys)) {
        Debug(absl::StrCat("dropped ", m->name,
                           " (an intermediate is used outside the "
                           "attention)"));
        continue;
      }
      kept.push_back(m);
    }
    if (kept.size() == live.size()) return kept;
    live = std::move(kept);
  }
}

}  // namespace

// --------------------------------------------------------------------------
// the public surface
// --------------------------------------------------------------------------

bool SdpaEnabled() {
  static const bool on = !EnvOff("METALJAX_SDPA");
  return on;
}

// sdpa.py `_all_blocks`: every block of every function in the module.
//
// Attention does not have to sit in @main.  The gemma-lib sampler and maxtext
// both put a whole decode step inside a private `@closed_call` a `func.call`
// reaches (a `fori_loop` over a non-inlined jit lowers exactly that way), and
// an attention rooted wholly inside such a callee was invisible to the walk
// that only looked at @main and its nested regions -- P26 measured the cost:
// zero of the 31B's 60 decode attentions fused where Stage 1 fused all 60, and
// the 1500 units of BlockCost that discount is worth are what put that body
// 388 units OVER the trace budget, so it dispatched op by op, per token.
//
// Rewriting inside a callee is safe for the same reason Stage 1 gives: the
// rewrite is a pure function of the IR and carries no per-call-site state, and
// `Lowering::Inline` seeds the callee's block arguments with the call's
// operand slots, which is all the emission ever reads.  Two call sites of one
// callee therefore emit two fused attentions over their own operands.
//
// Only sdpa widens.  qmm and moe deliberately keep the @main-rooted walk their
// Stage 1 counterparts had (`qmm.py`/`moe.py` both walked
// `_walk_blocks(interp._main_block())`): what the two stacks had to agree on
// above all is the set of matches, because BlockCost is computed from it and
// the compile decision is computed from that.
void AnalyzeSdpa(mlir::func::FuncOp fn, RewritePlan* plan) {
  if (!SdpaEnabled()) return;
  if (fn.getBody().getBlocks().size() != 1) return;
  mlir::Block& main = fn.getBody().front();
  mlir::ModuleOp module = fn->getParentOfType<mlir::ModuleOp>();

  // sdpa.py `_claimed`: ops the quantized-matmul recognizer has already spoken
  // for.  The two rewrites compose (a dequantized weight feeding attention's
  // projections is fine), but neither may swallow an op the other emits, so an
  // overlap simply drops the sdpa candidate.
  //
  // Deliberately reads every qmm CANDIDATE, including ones a later packing
  // prologue may disable: qmm decides that against concrete buffers, long
  // after this runs.  The conservative cost is that an attention overlapping a
  // quantized matmul that then falls back stays unfused for the life of the
  // executable; the alternative is two rewrites claiming one op.
  llvm::DenseSet<mlir::Operation*> claimed;
  for (const std::unique_ptr<QmmMatch>& q : plan->qmm) {
    if (q->root != nullptr) claimed.insert(q->root);
    for (mlir::Operation* o : q->ops) claimed.insert(o);
  }

  std::vector<std::unique_ptr<SdpaMatch>> cands;
  const std::function<void(mlir::Block&)> scan = [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      if (OpName(&op) != "stablehlo.dot_general") continue;
      std::unique_ptr<SdpaMatch> mm;
      try {
        mm = Try(module, &op);
      } catch (const Reject&) {
        continue;
      } catch (const std::exception& e) {   // never break a program
        Debug(absl::StrCat("candidate failed (", e.what(), ")"));
        continue;
      }
      bool overlap = claimed.contains(mm->root);
      for (mlir::Operation* o : mm->ops)
        overlap = overlap || claimed.contains(o);
      if (overlap) continue;
      cands.push_back(std::move(mm));
    }
  };
  if (module) {
    for (mlir::func::FuncOp f : module.getOps<mlir::func::FuncOp>())
      for (mlir::Region& r : f->getRegions())
        for (mlir::Block& b : r.getBlocks()) WalkBlocks(b, scan);
  } else {
    WalkBlocks(main, scan);   // a detached function: itself and nothing else
  }
  if (cands.empty()) return;

  // A candidate absorbed by another one cannot also be a root.
  llvm::DenseSet<mlir::Operation*> absorbed;
  for (const std::unique_ptr<SdpaMatch>& mm : cands)
    for (mlir::Operation* o : mm->ops) absorbed.insert(o);
  std::vector<std::unique_ptr<SdpaMatch>> uniq;
  llvm::DenseSet<mlir::Operation*> seen;
  for (std::unique_ptr<SdpaMatch>& mm : cands) {
    if (absorbed.contains(mm->root)) continue;
    if (!seen.insert(mm->root).second) continue;
    uniq.push_back(std::move(mm));
  }
  if (uniq.empty()) return;

  std::vector<SdpaMatch*> raw;
  for (const std::unique_ptr<SdpaMatch>& mm : uniq) raw.push_back(mm.get());
  const std::vector<SdpaMatch*> live = Exclusive(std::move(raw));
  llvm::DenseSet<SdpaMatch*> keep(live.begin(), live.end());
  size_t found = 0;
  for (std::unique_ptr<SdpaMatch>& mm : uniq) {
    if (!keep.contains(mm.get())) continue;
    found++;
    plan->sdpa.push_back(std::move(mm));
  }
  if (found > 0) Debug(absl::StrCat(found, " fused attention(s) recognized"));
}

}  // namespace metaljax
