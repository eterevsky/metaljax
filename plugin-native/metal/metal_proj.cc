/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The projection PACK (task B6, row 7's un-overlapped K/V projections).

A decode step's attention projections are N `stablehlo.dot_general`s that
read ONE activation -- the post-norm token, [1, 1, D] -- against N distinct
loop-invariant weights: keras-hub's `bqm,muh->bquh` q, k and v (W[D, H, h]),
a GDN layer's z/b/a inputs, an MLP's gate and up.  At M == 1 each such dot
is a `gemv_t` whose whole K reduction is one loop-carried chain per
threadgroup: a [2880 -> 512] bf16 member costs ~41 us alone and only ~66 us
for the PAIR when MLX happens to dispatch them into one concurrent group
(`~/.cache/metaljax-bench/logs/b5-row7/findings.txt` sections 8-10) -- and
whether it does is an accident of what sits between them in the tape.  The
rope view (B5) shortened q's chain and the K/V pair straddled a barrier:
+41 us per layer, +0.7 ms per token on gpt-oss-20b.

The rewrite makes the pair ONE dot.  The weights, each viewed as
[K, n_i] (their free axes merged -- a view, proven at analysis), are
concatenated along n once per executable into a row-major [K, n_total]
pack, exactly the way the stacked dot's relayout materializes its carried
layout (metal_stacked.cc `BuildStackedPacks`): governor-admitted, budgeted,
keyed by the arguments' identity so a later call with other weights repacks
(`Tape`), dropped -- every member lowering literally -- on any failure.
The emit (`Lowering::LowerProjPack`) is plain opcodes: one `dot_general`
reading the pack, then per member a last-axis `slice` and, where the
declared result is [.., H, h], a `reshape`; at M == 1 both are views
(shared-buffer slices of a row-contiguous [1, 1, n_total]).  Nothing in the
runtime changes.

Numerics.  The pack's LAYOUT follows the weights' storage, per group, so
the packed dot runs the SAME kernel the member dots ran.  A weight stored
[K, N] (keras' W[D, H, h], `x @ W`) is a `gemv_t` in MLX -- row-major
[K, n_total] is consumed in place by `check_transpose` -- and that kernel's
per-output K order depends only on (BM, SM, TM), which for K < 8192 does
not change with N (matmul.cpp 1091-1106, gemv.h 263-420): bit-identical
to the literal dots there.  A weight stored [N, K] (`x @ W.T`, an
Equinox/PyTorch-port `bqm,hm->bqh`) is the non-T `gemv`, whose order
depends on (BN, SN, TN) and changes with N only across `K >= 16 N`
(matmul.cpp 1136-1153, gemv.h 39-247): such a group packs as a row-major
[n_total, K] contracted on its dim 1, which MLX reads as transposed --
again the members' own kernel, bit-identical below that band.  The two
storage classes never share a pack (the storage is part of the group key);
a group whose members straddle a kernel band (k inside a q+k+v pack, bn4
-> bn16) changes only the PSO and the kernels are built `-fno-fast-math`,
so it is expected identical but compiler-dependent; K >= 8192 with the
packed N crossing 2048 changes the summation tree (tolerance-level, f32-
accumulated, one bf16 rounding).  METALJAX_PROJ_PACK_LAYOUT=kn|nk forces
one layout on every group -- the A/B arm for the other kernel (a [K, N]
weight under `nk` runs the non-T gemv: 22 serial K steps instead of 90,
no in-loop barrier), not bit-identical to the literal dots.

Policy.  `METALJAX_PROJ_PACK` unset/1 = `auto`: in a group of >= 3 members
with one at least twice as wide as the next (q beside k and v), the wide
one is left alone when the rest, packed, stay in their own gemv tile
(gpt-oss: 512 + 512 = 1024 keeps `bn4`, K+V -7 %) and packed too when they
would cross into a wider tile for nothing (Qwen3-0.6B: 1024 + 1024 = 2048
is `bn16`, flat; Q+K+V at 4096 shares q's tile, -9 %).  `kv` always leaves
the wide member alone, `all` packs every eligible member, `0` declines
everything.  A member over
`kMaxMemberBytes` (64 MiB, an MLP matrix) is dropped from its group.
`METALJAX_PROJ_PACK_MB` caps the packs one executable may hold (default
2048 MB; 0 = off).  DECODE-ONLY: M > 1 declines with the reason narrated.

Weights that change.  A pack is keyed by its weights' identity (`Tape`):
a call that hands over other buffers repacks, and an executable whose
projection weights keep changing retires ITS PROJECTION PACKS ONLY (the
other recognizers keep their tape; metal_executable.cc `kMaxProjRepacks`).
Two such programs decline at analysis instead: a DONATED weight (freed by
the executable, so fresh every call) and one the program UPDATES -- @main
returns a value of the weight's type that derives from it other than by
passing it through (a training step's `W - lr * g`, `UpdatedByProgram`).

A half-matched group lowers as ORDINARY ops (the recognizer file rule in
metal_recognize.h), and the emit never Declines on a matched shape: a pack
that is not in scope falls back to the literal dots in place.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_recognize.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlx/mlx.h"
#include "program.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace metaljax {

// Would `reshape(transpose(x, perm), target)` hand back a VIEW, for a
// row-contiguous `x` of shape `src`?  When it would not, MLX materializes a
// full copy of the operand -- `reshape_gpu` (mlx/backend/metal/copy.cpp) asks
// `prepare_reshape` (mlx/backend/common/common.cpp) and falls through to
// `copy_gpu_inplace` on a `true`.  This mirrors that decision: collapse the
// input's contiguous runs, then try to factor the target's dims out of them.
//
// It is used to price the dot's own lowering and to admit a projection-pack
// member, so a divergence from MLX can only cost or save an optimization --
// never a result.
bool ReshapeIsView(const std::vector<int64_t>& src,
                   const std::vector<int64_t>& perm,
                   const std::vector<int64_t>& target) {
  const size_t r = src.size();
  if (perm.size() != r) return false;
  std::vector<int64_t> stride(r, 1);
  for (size_t i = r; i-- > 1;) stride[i - 1] = stride[i] * src[i];
  std::vector<int64_t> pshape(r), pstride(r);
  for (size_t i = 0; i < r; i++) {
    pshape[i] = src[perm[i]];
    pstride[i] = stride[perm[i]];
  }

  int64_t numel = 1;
  for (int64_t d : src) numel *= d;
  if (numel == 0) return true;  // prepare_reshape's empty early-out

  // `collapse_contiguous_dims`: drop unit axes, merge a run when the outer
  // stride is exactly the inner stride times the inner extent.  MLX keeps
  // unit axes IN the contiguity test (a unit axis whose stride differs from
  // its neighbour's breaks the run there); dropping them is the permissive
  // side, and permissive here only ever declines an optimization.
  std::vector<int64_t> cshape, cstride;
  for (size_t i = 0; i < r; i++) {
    if (pshape[i] == 1) continue;
    if (!cshape.empty() && pstride[i] * pshape[i] == cstride.back()) {
      cshape.back() *= pshape[i];
      cstride.back() = pstride[i];
    } else {
      cshape.push_back(pshape[i]);
      cstride.push_back(pstride[i]);
    }
  }
  if (cshape.empty()) return true;  // a scalar after collapsing
  // Row-contiguous input is prepare_reshape's other early-out: one run left
  // whose stride is 1.
  if (cshape.size() == 1 && cstride[0] == 1) return true;

  // The factoring loop, verbatim in effect: peel each target extent off the
  // front of the current run; a run that does not divide forces the copy.
  size_t j = 0;
  for (size_t i = 0; i < target.size(); i++) {
    const int64_t want = target[i];
    if (j < cshape.size() && cshape[j] % want == 0) {
      cshape[j] /= want;
      if (cshape[j] == 1) j++;
    } else if (want == 1) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

namespace {

const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();

void Debug(const std::string& line) {
  if (!kDebug) return;
  std::fprintf(stderr, "[metaljax-native] proj: %s\n", line.c_str());
  std::fflush(stderr);
}

// METALJAX_PROJ_PACK: 0 = off; unset/1/auto = the `auto` policy (the
// widest member is left alone only when the rest stay in their own gemv
// tile); kv = always leave the widest alone; all = every eligible member.
enum class Policy { kOff, kAuto, kKv, kAll };

Policy PackPolicy() {
  static const Policy p = [] {
    const char* v = std::getenv("METALJAX_PROJ_PACK");
    if (v == nullptr || *v == '\0') return Policy::kAuto;
    const std::string s(v);
    if (s == "0") return Policy::kOff;
    if (s == "all") return Policy::kAll;
    if (s == "kv") return Policy::kKv;
    return Policy::kAuto;
  }();
  return p;
}

const char* PolicyName(Policy p) {
  switch (p) {
    case Policy::kOff: return "off";
    case Policy::kAuto: return "auto";
    case Policy::kKv: return "kv";
    case Policy::kAll: return "all";
  }
  return "?";
}

// The M = 1 gemv TILE MLX picks for a [K -> N] dot (mlx-src/mlx/backend/
// metal/matmul.cpp 1085-1153), reduced to what sets its speed: for `gemv_t`
// (a [K, N] weight) the (sm, sn) split and `bn`, which steps at N >= 512 and
// N >= 2048 -- past 2048 a threadgroup is 512 threads and every one of its
// ~K/32 loop-carried iterations moves 4x the bytes; for the non-T `gemv`
// (an [N, K] weight) `bm`, the rows per threadgroup, which steps at 4096.
// Two dots in the same tile cost the same per output; a pack that lifts
// its members into a wider tile can cost MORE than the members did (row
// 11's K+V, 1024 + 1024 -> 2048: flat), while a pack that shares a wide
// member's tile costs about that member alone (Q+K+V there: -9 %).
struct Tile {
  int a = 0, b = 0, c = 0;
  bool operator==(const Tile& o) const {
    return a == o.a && b == o.b && c == o.c;
  }
  bool operator!=(const Tile& o) const { return !(*this == o); }
};

Tile TileOf(int64_t K, int64_t N, bool nk) {
  if (!nk) {
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
    return Tile{sm, sn, bn};
  }
  int bn = 1, sm = 1, sn = 32;
  if (K <= 64) {
    sm = 8;
    sn = 4;
  } else if (K >= 16 * N) {
    bn = 8;
  }
  const int bm = N >= 4096 ? 8 : 4;
  return Tile{bn * 100 + sm, sn, bm};
}

std::string TileName(const Tile& t, bool nk) {
  return nk ? absl::StrCat("gemv bm", t.c) : absl::StrCat("gemv_t bn", t.c);
}

// METALJAX_PROJ_PACK_MB: total pack bytes one executable may hold (default
// 2048 MB, the relayout's default for the same reason: governor-admitted,
// unpageable); 0 = off.
int64_t BudgetBytes() {
  static const int64_t bytes = [] {
    const char* v = std::getenv("METALJAX_PROJ_PACK_MB");
    if (v == nullptr || *v == '\0') return 2048LL << 20;
    char* end = nullptr;
    const long long mb = std::strtoll(v, &end, 10);
    if (end == v || mb < 0) return 2048LL << 20;
    return static_cast<int64_t>(mb) << 20;
  }();
  return bytes;
}

// METALJAX_PROJ_PACK_LAYOUT: unset = each group packs in its weights' own
// storage order (the members' kernel, bit-identical); `kn` / `nk` force the
// row-major [K, n_total] / [n_total, K] pack on every group (design 7.3, the
// A/B arm for the other kernel).
enum class Layout { kByStorage, kKn, kNk };

Layout PackLayout() {
  static const Layout l = [] {
    const char* v = std::getenv("METALJAX_PROJ_PACK_LAYOUT");
    if (v == nullptr) return Layout::kByStorage;
    const std::string s(v);
    if (s == "nk") return Layout::kNk;
    if (s == "kn") return Layout::kKn;
    return Layout::kByStorage;
  }();
  return l;
}

const char* LayoutName(bool nk) { return nk ? "nk" : "kn"; }

// A member this large is bandwidth-bound already (N >= 2048 x 512-thread
// threadgroups stream ~200-260 GB/s); packing it can only cost bytes.  It
// keeps the MLP matrices (100-283 MB each) out of the packs.
constexpr int64_t kMaxMemberBytes = 64LL << 20;
// Below this the dots are too small to matter (the stacked dot's rule).
constexpr int64_t kMinWork = 16384;

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

int64_t Product(const std::vector<int64_t>& xs) {
  int64_t p = 1;
  for (int64_t x : xs) p *= x;
  return p;
}

std::string Dims(const std::vector<int64_t>& d) {
  return absl::StrCat("[", absl::StrJoin(d, ","), "]");
}

// The shape for a narration: never throws (a handler prints it).
std::string DimsOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t || !t.hasStaticShape()) return "[?]";
  return Dims(std::vector<int64_t>(t.getShape().begin(), t.getShape().end()));
}

std::string Mb(int64_t bytes) {
  return absl::StrFormat("%.1f", static_cast<double>(bytes) / (1 << 20));
}

// Bytes per element, and the MLX dtype, of the float types the dot accepts.
int ElemBytes(mlir::Type t) {
  if (auto f = mlir::dyn_cast<mlir::FloatType>(t))
    return static_cast<int>((f.getWidth() + 7) / 8);
  return 4;
}

std::optional<mx::Dtype> MxDtype(mlir::Type t) {
  if (mlir::isa<mlir::BFloat16Type>(t)) return mx::bfloat16;
  if (mlir::isa<mlir::Float16Type>(t)) return mx::float16;
  if (mlir::isa<mlir::Float32Type>(t)) return mx::float32;
  return std::nullopt;
}

bool IsPackFloat(mlir::Type t) {
  return mlir::isa<mlir::BFloat16Type>(t) || mlir::isa<mlir::Float16Type>(t) ||
         mlir::isa<mlir::Float32Type>(t);
}

// The lowering's identity ops: result i IS operand i (metal_lowering.cc
// aliases them per result, so jax's tuple `optimization_barrier((w1, w2))`
// is one op with two aliased results).
bool IsAlias(mlir::Operation* op) {
  if (op == nullptr) return false;
  const std::string n = OpName(op);
  return (n == "sdy.sharding_constraint" || n == "sdy.reshard" ||
          n == "stablehlo.optimization_barrier") &&
         op->getNumOperands() == op->getNumResults();
}

// The value behind any chain of aliases.  Nothing is absorbed: an alias
// lowers to a slot alias (no entry), so it costs nothing whether or not the
// pack replaces its reader.
mlir::Value LookThrough(mlir::Value v) {
  for (int guard = 0; guard < 8; guard++) {
    auto res = mlir::dyn_cast<mlir::OpResult>(v);
    if (!res || !IsAlias(res.getOwner())) return v;
    v = res.getOwner()->getOperand(res.getResultNumber());
  }
  return v;
}

using CallSites =
    llvm::DenseMap<mlir::StringRef, std::vector<mlir::Operation*>>;

// The value `v` stands for once every pass-through is peeled: aliases, an
// unchanged loop carry (from inside, `HoistInvariant`; from outside, the
// while result the body yields as its own argument), a callee returning its
// own argument (one unique call site).  What is left is a @main argument, a
// constant, or something computed.
mlir::Value PeelPassThrough(mlir::Value v, mlir::Block* main_block,
                            mlir::ModuleOp module,
                            const CallSites& call_sites) {
  for (int guard = 0; guard < 32; guard++) {
    v = LookThrough(HoistInvariant(v));
    if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      if (ba.getOwner() == main_block) return v;
      auto fn = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
          ba.getOwner()->getParentOp());
      if (!fn || &fn.getBody().front() != ba.getOwner()) return v;
      auto it = call_sites.find(fn.getName());
      if (it == call_sites.end() || it->second.size() != 1) return v;
      mlir::Operation* site = it->second[0];
      if (site->getNumOperands() != ba.getOwner()->getNumArguments())
        return v;
      v = site->getOperand(ba.getArgNumber());
      continue;
    }
    auto res = mlir::dyn_cast<mlir::OpResult>(v);
    if (!res) return v;
    mlir::Operation* def = res.getOwner();
    const unsigned i = res.getResultNumber();
    const std::string name = OpName(def);
    if (name == "stablehlo.while") {
      if (def->getNumRegions() < 2 || def->getRegion(1).empty()) return v;
      mlir::Block& body = def->getRegion(1).front();
      if (body.empty()) return v;
      mlir::Operation* term = &body.back();
      if (i >= term->getNumOperands() || i >= body.getNumArguments() ||
          i >= def->getNumOperands())
        return v;
      if (LookThrough(term->getOperand(i)) != body.getArgument(i)) return v;
      v = def->getOperand(i);
      continue;
    }
    if (name == "func.call" || name == "stablehlo.composite") {
      auto sym = def->getAttrOfType<mlir::FlatSymbolRefAttr>(
          name == "func.call" ? "callee" : "decomposition");
      auto callee = sym ? module.lookupSymbol<mlir::func::FuncOp>(
                              sym.getValue())
                        : mlir::func::FuncOp();
      if (!callee || callee.getBody().getBlocks().size() != 1) return v;
      mlir::Block& body = callee.getBody().front();
      if (body.empty()) return v;
      mlir::Operation* term = &body.back();
      if (i >= term->getNumOperands()) return v;
      auto arg = mlir::dyn_cast<mlir::BlockArgument>(
          LookThrough(term->getOperand(i)));
      if (!arg || arg.getOwner() != &body ||
          arg.getArgNumber() >= def->getNumOperands())
        return v;
      v = def->getOperand(arg.getArgNumber());
      continue;
    }
    return v;
  }
  return v;
}

// Does the program UPDATE the weight -- does @main return a value of the
// weight's type that derives from it other than by passing it through?  A
// training step returns `W - lr * g`; a decode step returns its weights
// untouched (unchanged carries, aliases) or not at all.  Forward taint from
// the argument over every use, into regions and callees; positional where
// an op yields positionally (a while), every result otherwise (the
// over-approximation can only decline).  Returns the reason, or "".
std::string UpdatedByProgram(mlir::BlockArgument arg, mlir::Block* main_block,
                             mlir::ModuleOp module,
                             const CallSites& call_sites) {
  llvm::DenseSet<mlir::Value> tainted;
  std::vector<mlir::Value> work;
  auto taint = [&](mlir::Value v) {
    if (v && tainted.insert(v).second) work.push_back(v);
  };
  auto taint_results = [&](mlir::Operation* op) {
    for (mlir::Value r : op->getResults()) taint(r);
  };
  auto callee_of = [&](mlir::Operation* op,
                       const std::string& name) -> mlir::func::FuncOp {
    auto sym = op->getAttrOfType<mlir::FlatSymbolRefAttr>(
        name == "func.call" ? "callee" : "decomposition");
    if (!sym) return mlir::func::FuncOp();
    auto fn = module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
    if (!fn || fn.getBody().getBlocks().size() != 1) return mlir::func::FuncOp();
    return fn;
  };
  std::vector<std::pair<unsigned, mlir::Value>> returned;
  taint(arg);
  while (!work.empty()) {
    mlir::Value v = work.back();
    work.pop_back();
    for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* op = use.getOwner();
      const unsigned idx = use.getOperandNumber();
      const std::string name = OpName(op);
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::Block* blk = op->getBlock();
        mlir::Operation* parent = blk ? blk->getParentOp() : nullptr;
        if (parent == nullptr) continue;
        if (auto fn = mlir::dyn_cast<mlir::func::FuncOp>(parent)) {
          if (blk == main_block) {
            returned.emplace_back(idx, v);
            continue;
          }
          auto it = call_sites.find(fn.getName());
          if (it == call_sites.end()) continue;
          for (mlir::Operation* site : it->second)
            if (idx < site->getNumResults()) taint(site->getResult(idx));
          continue;
        }
        if (OpName(parent) == "stablehlo.while") {
          // The cond yields a predicate, the body its carries.
          if (parent->getNumRegions() >= 2 &&
              blk->getParent() == &parent->getRegion(0))
            continue;
          if (idx < parent->getNumResults()) taint(parent->getResult(idx));
          for (mlir::Region& r : parent->getRegions())
            if (!r.empty() && idx < r.front().getNumArguments())
              taint(r.front().getArgument(idx));
          continue;
        }
        taint_results(parent);
        continue;
      }
      if (name == "func.call" || name == "stablehlo.composite") {
        mlir::func::FuncOp callee = callee_of(op, name);
        if (callee && idx < callee.getBody().front().getNumArguments())
          taint(callee.getBody().front().getArgument(idx));
        else
          taint_results(op);
        continue;
      }
      if (op->getNumRegions() > 0) {
        for (mlir::Region& r : op->getRegions()) {
          if (r.empty()) continue;
          mlir::Block& b = r.front();
          if (name == "stablehlo.while") {
            if (idx < b.getNumArguments()) taint(b.getArgument(idx));
          } else {
            for (mlir::BlockArgument a : b.getArguments()) taint(a);
          }
        }
      }
      taint_results(op);
    }
  }
  for (const auto& [idx, v] : returned) {
    if (v.getType() != arg.getType()) continue;
    mlir::Value base = PeelPassThrough(v, main_block, module, call_sites);
    if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(base))
      if (ba.getOwner() == main_block) continue;
    if (base.getDefiningOp() != nullptr &&
        mlir::isa<mlir::stablehlo::ConstantOp>(base.getDefiningOp()))
      continue;
    return absl::StrCat("the weight is updated by the program (result ", idx,
                        " has its type and derives from it: a training "
                        "step's W - lr * g; the pack would repack every "
                        "call)");
  }
  return std::string();
}

struct Member {
  mlir::Operation* dot = nullptr;
  mlir::Value rhs;              // behind aliases and loop carries
  int origin_arg = -1;          // @main argument, or -1 for a constant
  std::vector<int64_t> rdims;   // the rhs as the dot read it
  std::vector<int64_t> kdims;   // contracted extents, pair order
  int64_t n = 0;                // product of the rhs free dims
  int64_t bytes = 0;            // K * n * elem
  std::vector<int64_t> out_shape;
  bool nk = false;              // the pack layout its storage asks for
};

// What a group is keyed by: the same activation, the same contraction, the
// same dtype, the same storage order (a [K, N] and an [N, K] weight run
// different kernels and never share a pack).  Batch dims are empty by the
// member rule, so they are not in the key.
struct GroupKey {
  mlir::Value lhs;
  std::vector<int64_t> lc, rc;
  mlir::Type elem;
  bool nk = false;
  bool operator==(const GroupKey& o) const {
    return lhs == o.lhs && lc == o.lc && rc == o.rc && elem == o.elem &&
           nk == o.nk;
  }
};

struct Group {
  GroupKey key;
  std::vector<int64_t> lhs_dims;
  std::vector<int64_t> lb, rb;   // both empty
  int64_t K = 0, M = 1;
  std::vector<Member> members;   // block order
};

// The weight, resolved to what a pack can be built from: a @main argument
// (through any loop carries that pass it unchanged and at most ONE unique
// call site, the stacked dot's rule) or a constant.
void ResolveWeight(mlir::Value v, mlir::Block* main_block,
                   mlir::ModuleOp module,
                   const llvm::DenseMap<mlir::StringRef,
                                        std::vector<mlir::Operation*>>&
                       call_sites,
                   Member* m) {
  bool crossed = false;
  for (int guard = 0; guard < 16; guard++) {
    v = LookThrough(HoistInvariant(v));
    if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      if (ba.getOwner() == main_block) {
        m->rhs = v;
        m->origin_arg = static_cast<int>(ba.getArgNumber());
        return;
      }
      auto fn = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
          ba.getOwner()->getParentOp());
      if (!fn || &fn.getBody().front() != ba.getOwner())
        Bail("the weight is an inner block argument (a loop carry that "
             "changes, or a region's own value)");
      if (crossed) Bail("the weight crosses two call boundaries");
      auto it = call_sites.find(fn.getName());
      if (it == call_sites.end() || it->second.size() != 1)
        Bail("the weight is an argument of a callee with no unique call site");
      mlir::Operation* site = it->second[0];
      if (site->getNumOperands() != ba.getOwner()->getNumArguments())
        Bail("the call site arity disagrees with the callee");
      v = site->getOperand(ba.getArgNumber());
      crossed = true;
      continue;
    }
    mlir::Operation* def = v.getDefiningOp();
    if (def == nullptr) Bail("the weight has no definition");
    if (mlir::isa<mlir::stablehlo::ConstantOp>(def)) {
      m->rhs = v;
      m->origin_arg = -1;
      return;
    }
    Bail(absl::StrCat("the weight is computed by ", OpName(def),
                      " (not an argument or a constant)"));
  }
  Bail("the weight resolution did not converge");
}

// One dot as a candidate member: its own shape rules, and the group it
// belongs to.  Bails with the reason.  `updated` caches `UpdatedByProgram`
// per @main argument for one analysis (a weight is asked once, not once per
// dot that reads it).
Member MatchMember(mlir::Operation* op, mlir::Block* main_block,
                   mlir::ModuleOp module, const CallSites& call_sites,
                   const absl::flat_hash_set<int>& donated,
                   llvm::DenseMap<int, std::string>* updated, Group* g) {
  auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
  if (!dot) Bail("not a dot_general");
  if (op->getNumResults() != 1) Bail("a dot with several results");
  mlir::stablehlo::DotDimensionNumbersAttr dn = dot.getDotDimensionNumbers();
  if (!dn.getLhsBatchingDimensions().empty() ||
      !dn.getRhsBatchingDimensions().empty())
    Bail("the dot has batching dims (v1 packs none)");
  std::vector<int64_t> lc(dn.getLhsContractingDimensions().begin(),
                          dn.getLhsContractingDimensions().end());
  std::vector<int64_t> rc(dn.getRhsContractingDimensions().begin(),
                          dn.getRhsContractingDimensions().end());
  const size_t nc = lc.size();
  if (nc == 0 || nc != rc.size()) Bail("no contraction");

  const std::vector<int64_t> lshape = ShapeOf(dot.getLhs());
  const std::vector<int64_t> rshape = ShapeOf(dot.getRhs());
  const std::vector<int64_t> oshape = ShapeOf(dot.getResult());
  if (rshape.size() <= nc) Bail("the dot leaves no free weight axis");
  mlir::Type elem = ElemOf(dot.getLhs());
  if (!IsPackFloat(elem)) Bail("not a bf16/f16/f32 dot");
  if (ElemOf(dot.getRhs()) != elem || ElemOf(dot.getResult()) != elem)
    Bail("mixed dtypes (a preferred_element_type dot)");
  // The dot's operand-side merges follow the lowering's canonical PAIR
  // order (the bigger operand's contracting dims ascending), so the view
  // proof below asks about the transpose the runtime really applies.
  if (nc > 1) {
    const int64_t lsize = Product(lshape), rsize = Product(rshape);
    const std::vector<int64_t>& key = rsize >= lsize ? rc : lc;
    std::vector<size_t> order(nc);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return key[a] < key[b]; });
    std::vector<int64_t> lc2, rc2;
    for (size_t i : order) {
      lc2.push_back(lc[i]);
      rc2.push_back(rc[i]);
    }
    lc = std::move(lc2);
    rc = std::move(rc2);
  }
  auto holds = [](const std::vector<int64_t>& xs, int64_t v) {
    for (int64_t x : xs) if (x == v) return true;
    return false;
  };
  std::vector<int64_t> lfree, rfree, kdims;
  int64_t M = 1, K = 1, n = 1;
  for (int64_t d = 0; d < static_cast<int64_t>(lshape.size()); d++)
    if (!holds(lc, d)) {
      lfree.push_back(d);
      M *= lshape[d];
    }
  for (int64_t d = 0; d < static_cast<int64_t>(rshape.size()); d++)
    if (!holds(rc, d)) {
      rfree.push_back(d);
      n *= rshape[d];
    }
  for (size_t i = 0; i < nc; i++) {
    if (lc[i] < 0 || lc[i] >= static_cast<int64_t>(lshape.size()) ||
        rc[i] < 0 || rc[i] >= static_cast<int64_t>(rshape.size()))
      Bail("a contracting dim out of range");
    if (lshape[lc[i]] != rshape[rc[i]])
      Bail("the contracted extents disagree");
    K *= lshape[lc[i]];
    kdims.push_back(lshape[lc[i]]);
  }
  if (K < 1 || n < 1 || M < 1) Bail("degenerate sizes");
  if (M != 1)
    Bail(absl::StrCat("multi-row (M=", M,
                      "): the pack is decode-only, the dots run as written"));
  // The weight must read in place as [K, n]: its free axes merge to one
  // stride-1 axis behind its contracted ones.  keras' W[D, H, h] does; a
  // middle-contracted W[H, D, h] (gemma's batched arm) does not, and neither
  // does a weight the lowering would copy -- both decline here, which is
  // also what keeps `batch_side == 0` for the original and the packed dot.
  std::vector<int64_t> rperm = rc;
  rperm.insert(rperm.end(), rfree.begin(), rfree.end());
  if (!ReshapeIsView(rshape, rperm, {K, n}))
    Bail("the weight's free axes do not merge to a [K, n] view (a "
         "middle-contracted or batched weight)");
  // Which axis of that view is stride-1 -- the weight's storage order, and
  // so the kernel its dot runs: the source's innermost non-unit axis is
  // either contracted ([N, K] storage, the non-T `gemv`) or free ([K, N],
  // `gemv_t`).  The pack keeps that order so the packed dot runs the same
  // kernel (the file comment's numerics).
  int64_t innermost = -1;
  for (size_t i = rshape.size(); i-- > 0;)
    if (rshape[i] != 1) {
      innermost = static_cast<int64_t>(i);
      break;
    }
  const bool nk_storage = innermost >= 0 && holds(rc, innermost);

  Member m;
  m.dot = op;
  m.rdims = rshape;
  m.kdims = kdims;
  m.n = n;
  m.out_shape = oshape;
  m.bytes = K * n * ElemBytes(elem);
  switch (PackLayout()) {
    case Layout::kByStorage: m.nk = nk_storage; break;
    case Layout::kKn: m.nk = false; break;
    case Layout::kNk: m.nk = true; break;
  }
  ResolveWeight(dot.getRhs(), main_block, module, call_sites, &m);
  if (m.origin_arg >= 0 && donated.contains(m.origin_arg))
    Bail(absl::StrCat("the weight is donated (argument ", m.origin_arg,
                      ", the pack would repack every call)"));
  if (m.origin_arg >= 0) {
    auto it = updated->find(m.origin_arg);
    if (it == updated->end()) {
      auto arg = mlir::cast<mlir::BlockArgument>(m.rhs);
      it = updated->try_emplace(m.origin_arg,
                                UpdatedByProgram(arg, main_block, module,
                                                 call_sites)).first;
    }
    if (!it->second.empty()) Bail(it->second);
  }
  if (m.bytes > kMaxMemberBytes)
    Bail(absl::StrCat("the weight is ", m.bytes, " bytes (", Mb(m.bytes),
                      " MB), over the ", kMaxMemberBytes,
                      "-byte (64 MiB) member cap (bandwidth-bound "
                      "already)"));

  g->key.lhs = LookThrough(dot.getLhs());
  g->key.lc = lc;
  g->key.rc = rc;
  g->key.elem = elem;
  g->key.nk = m.nk;
  g->lhs_dims = lshape;
  g->K = K;
  g->M = M;
  return m;
}

}  // namespace

void AnalyzeProjPack(mlir::func::FuncOp fn,
                     const absl::flat_hash_set<int>& donated,
                     RewritePlan* plan) {
  if (PackPolicy() == Policy::kOff || BudgetBytes() <= 0) return;
  if (!RecognizeEnabled()) return;
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
  for (const auto& m : plan->stacked) take(m->root, m->ops);

  llvm::DenseMap<mlir::StringRef, std::vector<mlir::Operation*>> call_sites;
  module.walk([&](mlir::Operation* op) {
    if (OpName(op) != "func.call") return;
    if (auto sym = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      call_sites[sym.getValue()].push_back(op);
  });

  const Policy policy = PackPolicy();
  llvm::DenseMap<int, std::string> updated;
  int found = 0;

  // One block: group its dots, filter each group, keep the survivors.
  auto analyze_block = [&](mlir::Block& block) {
    std::vector<Group> groups;
    for (mlir::Operation& op : block) {
      if (OpName(&op) != "stablehlo.dot_general" || taken.contains(&op))
        continue;
      Group g;
      try {
        Member m = MatchMember(&op, main_block, module, call_sites, donated,
                               &updated, &g);
        bool placed = false;
        for (Group& have : groups) {
          if (have.key == g.key && have.lhs_dims == g.lhs_dims) {
            // Pairwise distinct weights: the same buffer twice would be a
            // CSE miss upstream, and packing it buys nothing.
            bool dup = false;
            for (const Member& o : have.members) dup = dup || o.rhs == m.rhs;
            if (dup) {
              Debug(absl::StrCat("declined a duplicate weight ",
                                 Dims(m.rdims), " (the same value twice)"));
            } else {
              have.members.push_back(std::move(m));
            }
            placed = true;
            break;
          }
        }
        if (!placed) {
          g.members.push_back(std::move(m));
          groups.push_back(std::move(g));
        }
      } catch (const Reject& e) {
        Debug(absl::StrCat("declined ", e.why, " (dot ",
                           DimsOf(op.getOperand(0)), " x ",
                           DimsOf(op.getOperand(1)), ")"));
      } catch (const std::exception& e) {
        Debug(absl::StrCat("analysis error (", e.what(), ")"));
      }
    }
    for (Group& g : groups) {
      if (g.members.size() < 2) continue;  // no sibling: not a candidate
      // The policy, for a group of >= 3 with one member at least twice as
      // wide as the next (q beside k and v): `kv` leaves that member alone;
      // `auto` does so only when the rest, packed, stay in their own gemv
      // tile (row 7: 512 + 512 = 1024 keeps bn4) and otherwise packs it too
      // (row 11: 1024 + 1024 would cross into bn16 for nothing, Q+K+V at
      // 4096 shares q's tile); `all` packs everything.
      if ((policy == Policy::kKv || policy == Policy::kAuto) &&
          g.members.size() >= 3) {
        size_t widest = 0;
        for (size_t i = 1; i < g.members.size(); i++)
          if (g.members[i].n > g.members[widest].n) widest = i;
        int64_t second = 0, rest = 0;
        for (size_t i = 0; i < g.members.size(); i++)
          if (i != widest) {
            second = std::max(second, g.members[i].n);
            rest += g.members[i].n;
          }
        if (g.members[widest].n >= 2 * second) {
          bool leave = policy == Policy::kKv;
          const Tile packed = TileOf(g.K, rest, g.key.nk);
          if (policy == Policy::kAuto) {
            leave = true;
            for (size_t i = 0; i < g.members.size(); i++)
              if (i != widest && TileOf(g.K, g.members[i].n, g.key.nk) != packed)
                leave = false;
          }
          if (leave) {
            Debug(absl::StrCat("policy ", PolicyName(policy),
                               " leaves the widest member alone (n=",
                               g.members[widest].n, " >= 2 x ", second,
                               "; the rest pack as n=", rest, " in their own ",
                               TileName(packed, g.key.nk),
                               "; METALJAX_PROJ_PACK=all packs it too)"));
            g.members.erase(g.members.begin() +
                            static_cast<std::ptrdiff_t>(widest));
          } else {
            Debug(absl::StrCat("policy auto packs the widest member too (n=",
                               g.members[widest].n, "; the rest alone, n=",
                               rest, ", would cross into ",
                               TileName(packed, g.key.nk),
                               " for nothing; METALJAX_PROJ_PACK=kv leaves it)"));
          }
        }
      }
      int64_t n_total = 0;
      for (const Member& m : g.members) n_total += m.n;
      if (g.K * n_total < kMinWork) {
        Debug(absl::StrCat("declined a group too small to matter (K=", g.K,
                           ", n_total=", n_total, ")"));
        continue;
      }
      auto match = std::make_unique<ProjPackMatch>();
      std::vector<std::string> ns;
      for (const Member& m : g.members) {
        match->roots.push_back(m.dot);
        match->rhs.push_back(m.rhs);
        match->origin_args.push_back(m.origin_arg);
        match->n.push_back(m.n);
        match->rhs_dims.push_back(m.rdims);
        match->out_shapes.push_back(m.out_shape);
        ns.push_back(absl::StrCat(m.n));
      }
      // The plan root is the earliest dot; every consumer of any member is
      // after its own dot in SSA order, hence after the root.
      match->lhs = g.members[0].dot->getOperand(0);
      match->lc = g.key.lc;
      match->rc = g.key.rc;
      match->lhs_dims = g.lhs_dims;
      match->kdims = g.members[0].kdims;
      match->K = g.K;
      match->M = g.M;
      match->n_total = n_total;
      match->elem_bytes = ElemBytes(g.key.elem);
      match->pack_bytes = g.K * n_total * match->elem_bytes;
      match->nk = g.key.nk;
      for (size_t i = 1; i < match->roots.size(); i++)
        match->ops.push_back(match->roots[i]);
      match->name = absl::StrCat("m", g.M, "k", g.K, "n",
                                 absl::StrJoin(ns, "+"));
      Debug(absl::StrCat("matched a projection pack (", match->name, ", ",
                         match->roots.size(), " dots, ", Mb(match->pack_bytes),
                         " MB, layout ", LayoutName(match->nk),
                         PackLayout() == Layout::kByStorage
                             ? " by storage" : " forced", ")"));
      plan->proj.push_back(std::move(match));
      found++;
    }
  };

  // Every block reachable from @main, callees included.
  llvm::DenseSet<mlir::Operation*> visited_fns;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& block) {
    analyze_block(block);
    for (mlir::Operation& op : block) {
      const std::string name = OpName(&op);
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
  if (found > 0)
    Debug(absl::StrCat(found, " projection pack(s) planned in ",
                       fn.getName().str()));
}

absl::Status BuildProjPacks(RewritePlan* plan, const PackContext& ctx) {
  if (plan->proj.empty()) return absl::OkStatus();
  const int64_t budget = BudgetBytes();
  absl::flat_hash_set<int> args(plan->pack_args.begin(),
                                plan->pack_args.end());
  // The arguments the earlier pack waves (qmm, stacked) already key on: an
  // argument only the projection packs read is one whose change costs the
  // executable its projection packs and nothing else (`Tape`).
  const absl::flat_hash_set<int> before = args;
  std::vector<std::unique_ptr<ProjPackMatch>> kept;
  int64_t total = 0;
  int built = 0;
  bool dropped = false;

  // Smallest first: the budget then favours the latency-bound small packs
  // over the giants.
  std::vector<size_t> order(plan->proj.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return plan->proj[a]->pack_bytes < plan->proj[b]->pack_bytes;
  });

  for (size_t idx : order) {
    std::unique_ptr<ProjPackMatch>& m = plan->proj[idx];
    std::string why;
    try {
      if (ctx.args == nullptr) Bail("no buffers to pack from");
      if (total + m->pack_bytes > budget)
        Bail(absl::StrCat("over budget (", Mb(total + m->pack_bytes),
                          " MB > ", budget >> 20,
                          " MB, METALJAX_PROJ_PACK_MB)"));
      std::optional<mx::Dtype> want;
      std::vector<mx::array> ws;
      for (size_t i = 0; i < m->roots.size(); i++) {
        mx::array src = [&]() -> mx::array {
          if (m->origin_args[i] >= 0) {
            if (m->origin_args[i] >= static_cast<int>(ctx.args->size()))
              Bail("the weight's argument is out of range");
            return (*ctx.args)[static_cast<size_t>(m->origin_args[i])];
          }
          absl::StatusOr<std::vector<mx::array>> got =
              ctx.eval({m->rhs[i]}, {});
          if (!got.ok() || got->size() != 1)
            Bail(absl::StrCat("the constant weight could not be evaluated (",
                              got.ok() ? "arity" :
                              std::string(got.status().message()), ")"));
          return (*got)[0];
        }();
        const std::vector<int64_t>& rd = m->rhs_dims[i];
        if (static_cast<size_t>(src.ndim()) != rd.size())
          Bail("the weight's rank disagrees with the dot");
        for (size_t d = 0; d < rd.size(); d++)
          if (src.shape(static_cast<int>(d)) != rd[d])
            Bail("the weight's shape disagrees with the dot");
        if (!want.has_value()) want = src.dtype();
        if (src.dtype() != *want)
          Bail("the weights' dtypes disagree");
        if (static_cast<int64_t>(src.dtype().size()) != m->elem_bytes)
          Bail("the weight's dtype disagrees with the dot");
        // rperm = rc ++ rfree, then merge to [K, n_i] -- a view (proven at
        // analysis); under `nk` the member is presented as [n_i, K].
        std::vector<int> perm(m->rc.begin(), m->rc.end());
        for (int d = 0; d < static_cast<int>(rd.size()); d++) {
          bool contracted = false;
          for (int64_t c : m->rc) contracted = contracted || c == d;
          if (!contracted) perm.push_back(d);
        }
        mx::array w = mx::reshape(mx::transpose(src, perm),
                                  mx::Shape{static_cast<int>(m->K),
                                            static_cast<int>(m->n[i])});
        ws.push_back(m->nk ? mx::transpose(w) : w);
      }
      // The no-panic contract: device memory held for the executable's
      // life, admitted like a transfer of its size.  A refusal throws and
      // is caught below -- the match is dropped, never RESOURCE_EXHAUSTED.
      governor_admit(m->pack_bytes, MemWhere::kExecute);
      mx::array pk = mx::concatenate(ws, m->nk ? 0 : 1);
      // ...viewed at the rank the emit declares: [k_1, .., k_nc, n_total]
      // (or [n_total, k_1, .., k_nc]), a view of the contiguous pack.
      mx::Shape pshape;
      if (m->nk) pshape.push_back(static_cast<int>(m->n_total));
      for (int64_t k : m->kdims) pshape.push_back(static_cast<int>(k));
      if (!m->nk) pshape.push_back(static_cast<int>(m->n_total));
      pk = mx::reshape(pk, pshape);
      mx::eval(pk);
      m->pack_slot = static_cast<int>(plan->packs.size());
      plan->packs.push_back(pk);
      for (int a : m->origin_args)
        if (a >= 0) args.insert(a);
      total += m->pack_bytes;
      built++;
      Debug(absl::StrCat("packed ", m->name, " (", Mb(m->pack_bytes),
                         " MB once, ", m->roots.size(), " weights, layout ",
                         LayoutName(m->nk), ")"));
      kept.push_back(std::move(m));
    } catch (const Reject& e) {
      why = e.why;
    } catch (const std::exception& e) {
      why = e.what();
    }
    if (!why.empty()) {
      Debug(absl::StrCat("declined ", why, " (", m->name,
                         ": the dots run as written)"));
      dropped = true;
      m.reset();
    }
  }
  plan->proj = std::move(kept);
  plan->pack_args.assign(args.begin(), args.end());
  std::sort(plan->pack_args.begin(), plan->pack_args.end());
  plan->proj_only_args.clear();
  for (int a : plan->pack_args)
    if (!before.contains(a)) plan->proj_only_args.push_back(a);
  if (dropped) plan->rebuild();
  if (built > 0)
    Debug(absl::StrCat(built, " projection pack(s) built, ", Mb(total),
                       " MB held for the executable"));
  return absl::OkStatus();
}

}  // namespace metaljax
