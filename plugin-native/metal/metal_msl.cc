/* metaljax: msl_scan's planning half, natively (src/metaljax/msl_scan.py).

The loop body is evaluated SYMBOLICALLY into the `Sym` graph declared in
metal_msl.h, every carry is then classified (pass-through / counter / state /
stacked output / cross-lane accumulator), a mode is chosen, and one of the
three emitters in metal_msl_emit.cc generates the kernel.

Every check here is msl_scan.py's, in its order, with its message: a plan that
qualifies on one engine and not the other would compute a carry by different
arithmetic, and no byte-level differential could hold.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal_msl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "metal_dtypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace metaljax {

void MslDecline(std::string what) {
  MslUnsupported e;
  e.msg = std::move(what);
  throw e;
}

const char kMslKernelHeader[] = R"(
static inline float mj_expm1(float x) {
    float u = metal::precise::exp(x);
    float d = u - 1.0f;
    return (u == 1.0f) ? x : d * (x / metal::precise::log(u));
}
)";

namespace {

// ------------------------------------------------------------- environment
//
// msl_scan.py's module-level knobs, read once.  The plugin is the only engine
// in its process, so a function-local static is the same thing the Python's
// import-time read is.

struct MslFlags {
  bool enabled = true;
  bool debug = false;
  int64_t reg_limit = 16;
  int64_t coop_cap = 2200000;
  // A phase-2 divergence from msl_scan.py, measured: see the width cap below.
  int64_t coop_max_f = 1024;
  bool coop_pref = true;
  int64_t coop_pref_min_f = 8;
  bool inlane_dots = true;
  int64_t pack_trigger = 30;
  bool wnorm = true;
  // bf16 dot weights of at most this many elements (per source) are fed to
  // the kernel as an f32 copy, materialized once per call instead of the
  // dot loop's per-element (float)W[off].  MEASURED OFF (2026-08-22): the
  // suspected in-loop conversion cost is zero on this compiler -- the
  // dumped tc106 bwd kernel runs 7.86us/launch in BOTH dtypes standalone,
  // and with the real bf16 cliff fixed (contended 16-bit scatter-add,
  // scatter_add_wide) the conversion measures neutral to -3% (tc146,
  // per-call astype overhead).  The knob remains for retesting on other
  // silicon/OS: METALJAX_MSL_BF16_WCONV=<max elems>, 0 = off.
  int64_t bf16_wconv = 0;
  std::string volatile_mode = "t";
  std::string t_decl = "  volatile uint t = t_;";
  // The MJDBG_* probes msl_scan.py carries, so an investigation can bisect a
  // plan the same way on either engine.
  bool no_hoist = false;
  bool no_nested = false;
  bool no_redreg = false;
  bool no_dsize = false;
};

bool EnvFlag(const char* name, bool dflt) {
  const char* v = std::getenv(name);
  if (v == nullptr) return dflt;
  return std::strcmp(v, "0") != 0;
}

bool EnvSet(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0';
}

int64_t EnvInt(const char* name, int64_t dflt) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  return std::strtoll(v, nullptr, 10);
}

const MslFlags& Flags() {
  static const MslFlags* f = [] {
    auto* out = new MslFlags();
    out->enabled = EnvFlag("METALJAX_MSL", true);
    const char* dbg = std::getenv("METALJAX_DEBUG");
    out->debug = dbg != nullptr && std::strcmp(dbg, "1") == 0;
    out->reg_limit = EnvInt("METALJAX_MSL_REG", 16);
    out->coop_cap = EnvInt("METALJAX_MSL_COOP_CAP", 2200000);
    out->coop_max_f = EnvInt("METALJAX_MSL_COOP_MAX_F", 1024);
    out->coop_pref = EnvFlag("METALJAX_MSL_COOP_PREF", true);
    out->coop_pref_min_f = EnvInt("METALJAX_MSL_COOP_MIN_F", 8);
    out->inlane_dots = EnvFlag("METALJAX_MSL_INLANE", true);
    out->pack_trigger = EnvInt("METALJAX_MSL_PACK_TRIGGER", 30);
    out->wnorm = EnvFlag("METALJAX_MSL_WNORM", true);
    out->bf16_wconv = EnvInt("METALJAX_MSL_BF16_WCONV", 0);
    const char* vol = std::getenv("METALJAX_MSL_VOLATILE");
    out->volatile_mode = vol == nullptr ? "t" : std::string(vol);
    if (out->volatile_mode == "1") out->volatile_mode = "t";
    // WORKAROUND for a Metal shader-compiler bug: multi-iteration kernel time
    // loops are miscompiled unless the counter is read through a volatile at
    // EVERY use (CLAUDE.md item 12c).  The weaker variants are kept only so a
    // future macOS can be retested with them.
    if (out->volatile_mode == "t")
      out->t_decl = "  volatile uint t = t_;";
    else if (out->volatile_mode == "tv")
      out->t_decl = "  volatile uint t__ = t_; uint t = t__;";
    else if (out->volatile_mode == "tmap")
      out->t_decl = "  uint t = tmap[t_];";
    else
      out->t_decl = "  uint t = t_;";
    out->no_hoist = EnvSet("MJDBG_NOHOIST");
    out->no_nested = EnvSet("MJDBG_NONESTED");
    out->no_redreg = EnvSet("MJDBG_NOREDREG");
    out->no_dsize = EnvSet("MJDBG_NODSIZE");
    return out;
  }();
  return *f;
}

// Test hook: emit deliberately invalid MSL so every generated kernel fails to
// build, exercising the executor's recovery.  Read per plan (not once), so a
// test can flip it on a live process.
bool ForceBuildFail() {
  const char* v = std::getenv("METALJAX_MSL_FORCE_BUILD_FAIL");
  return v != nullptr && std::strcmp(v, "1") == 0;
}

void Debug(const std::string& line) {
  if (Flags().debug) std::fprintf(stderr, "[metaljax] %s\n", line.c_str());
}

// ------------------------------------------------------------------ dtypes

const std::map<std::string, std::string>& MslDtypes() {
  // bf16 spells MLX's `bfloat16_t` (a typedef of the native MSL 3.1 `bfloat`
  // in the vendored build) rather than `bfloat` itself: mx::fast::metal_kernel
  // prepends MLX's kernel preamble (utils.h -> bf16.h + bf16_math.h), which
  // both defines the typedef and instantiates the metal:: math library for it
  // -- so the name works wherever MLX's own kernels do.  A DIVERGENCE from
  // msl_scan.py, whose table has no bf16 row at all: Stage 1 declines every
  // bf16 loop, and the 25x topconfs16k cliff (notes/topconfs16k-sweep-
  // 2026-08-22.md) is what that costs.
  static const auto* m = new std::map<std::string, std::string>{
      {"f32", "float"}, {"f16", "half"},    {"i32", "int"},
      {"i1", "bool"},   {"i64", "long"},    {"i16", "short"},
      {"i8", "char"},   {"ui64", "ulong"},  {"ui32", "uint"},
      {"ui16", "ushort"}, {"ui8", "uchar"}, {"bf16", "bfloat16_t"}};
  return *m;
}

// msl_scan.py `_dt`: the dtype name of an element type, refused when the
// generated kernel has no type for it (bf16 and complex are exactly that).
std::string Dt(mlir::Type t) {
  std::optional<std::string> name = TapeElementName(t);
  if (!name.has_value()) MslDecline("dtype <unknown>");
  if (MslDtypes().count(*name) == 0)
    MslDecline(absl::StrCat("dtype ", *name));
  return *name;
}

std::string Dt(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) MslDecline("a value that is not a ranked tensor");
  return Dt(t.getElementType());
}

MslShape ShapeOf(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) MslDecline("a value that is not a ranked tensor");
  MslShape out;
  for (int64_t d : t.getShape()) {
    if (d < 0) MslDecline("a dynamic dimension");
    out.push_back(d);
  }
  return out;
}

// ------------------------------------------------------------ small helpers

int64_t Numel(const MslShape& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

MslShape Rowmajor(const MslShape& shape) {
  MslShape strides(shape.size(), 1);
  int64_t acc = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

// msl_scan.py `_remap_strides`: strides after a reshape that only adds or
// drops size-1 dims.
MslShape RemapStrides(const MslShape& old_shape, const MslShape& old_strides,
                      const MslShape& new_shape) {
  std::vector<std::pair<int64_t, int64_t>> core;
  for (size_t i = 0; i < old_shape.size(); i++)
    if (old_shape[i] != 1) core.emplace_back(old_shape[i], old_strides[i]);
  MslShape out;
  size_t it = 0;
  for (int64_t d : new_shape) {
    if (d == 1) {
      out.push_back(0);
    } else {
      if (it >= core.size()) MslDecline("reshape strides");
      if (core[it].first != d) MslDecline("reshape strides");
      out.push_back(core[it].second);
      it++;
    }
  }
  return out;
}

// msl_scan.py `_bshape`.
MslShape Bshape(const MslShape& a, const MslShape& b) {
  MslShape ra(a), rb(b);
  while (ra.size() < rb.size()) ra.insert(ra.begin(), 1);
  while (rb.size() < ra.size()) rb.insert(rb.begin(), 1);
  MslShape out;
  for (size_t i = 0; i < ra.size(); i++) {
    if (ra[i] != rb[i] && ra[i] != 1 && rb[i] != 1)
      MslDecline(absl::StrCat("broadcast ", absl::StrJoin(a, ","), " vs ",
                              absl::StrJoin(b, ",")));
    out.push_back(std::max(ra[i], rb[i]));
  }
  return out;
}

std::string Join(const MslShape& s) { return absl::StrJoin(s, ","); }

bool IsZeroConst(const Sym* s) {
  return s != nullptr && s->kind == SymKind::kConst && s->dval == 0.0;
}

std::string RoleStr(const MslRole& r) {
  switch (r.kind) {
    case RoleKind::kData: return absl::StrCat("d", r.idx);
    case RoleKind::kReg: return "reg";
    case RoleKind::kOne: return "one";
    case RoleKind::kC: return "c";
    default: return "D";
  }
}

// msl_scan.py `_dump`, for the decline messages that carry one.
std::string Dump(const Sym* s) {
  if (s == nullptr) return "None";
  switch (s->kind) {
    case SymKind::kAccRed:
      return absl::StrCat("ACCRED[dims=", absl::StrJoin(s->dims, ","), "](",
                          Dump(s->inner), ")");
    case SymKind::kPerm:
      return absl::StrCat("PERM", absl::StrJoin(s->perm, ","), "(",
                          Dump(s->inner), ")");
    case SymKind::kElem: {
      std::string nm = s->op;
      const size_t dot = nm.rfind('.');
      if (dot != std::string::npos) nm = nm.substr(dot + 1);
      std::vector<std::string> args;
      for (const Sym* a : s->args) args.push_back(Dump(a));
      return absl::StrCat(nm, Join(s->shape), "(", absl::StrJoin(args, ","),
                          ")");
    }
    case SymKind::kAccDot:
      return absl::StrCat("ACC", Join(s->shape), "perm",
                          absl::StrJoin(s->perm, ","));
    case SymKind::kLeaf: {
      const char* k = s->leaf == LeafKind::kRead      ? "read"
                      : s->leaf == LeafKind::kWhole   ? "whole"
                      : s->leaf == LeafKind::kArg     ? "arg"
                                                      : "updated";
      return absl::StrCat(k, Join(s->shape), "st", Join(s->strides), "@",
                          s->source.kind == SrcKind::kCarry
                              ? absl::StrCat("carry", s->source.carry)
                              : (s->source.kind == SrcKind::kFree ? "free"
                                                                  : "hoist"));
    }
    case SymKind::kConst:
      return absl::StrCat("c", s->dval);
    case SymKind::kDot:
      return absl::StrCat("DOT", Join(s->shape));
    case SymKind::kCounter: return "SymCounter";
    case SymKind::kPad: return "SymPad";
    case SymKind::kStack: return "SymStack";
    case SymKind::kIota: return "SymIota";
    case SymKind::kRedReg: return "SymRedReg";
  }
  return "Sym";
}

const char* KindName(const Sym* s) {
  switch (s->kind) {
    case SymKind::kCounter: return "SymCounter";
    case SymKind::kConst: return "SymConst";
    case SymKind::kLeaf: return "SymLeaf";
    case SymKind::kElem: return "SymElem";
    case SymKind::kDot: return "SymDot";
    case SymKind::kPerm: return "SymPerm";
    case SymKind::kPad: return "SymPad";
    case SymKind::kStack: return "SymStack";
    case SymKind::kIota: return "SymIota";
    case SymKind::kRedReg: return "SymRedReg";
    case SymKind::kAccRed: return "SymAccRed";
    case SymKind::kAccDot: return "SymAccDot";
  }
  return "Sym";
}

// ---------------------------------------------------------- attribute reads

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
  MslDecline(absl::StrCat("attribute ", std::string(name)));
}

int64_t IntAttr(mlir::Operation* op, llvm::StringRef name) {
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>(name))
    return a.getValue().getSExtValue();
  MslDecline(absl::StrCat("attribute ", std::string(name)));
}

// msl_scan.py `_const_scalar`: the single value of a splat / rank-0 dense
// constant, never decoding the whole tensor.
struct ConstVal {
  double dval = 0.0;
  int64_t ival = 0;
  bool uns = false;
};

ConstVal ConstScalar(mlir::DenseElementsAttr attr, mlir::Type element) {
  ConstVal out;
  if (attr.getNumElements() == 0) return out;
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(element)) {
    llvm::APInt v = attr.isSplat() ? attr.getSplatValue<llvm::APInt>()
                                   : *attr.getValues<llvm::APInt>().begin();
    if (it.isUnsigned() || it.getWidth() == 1) {
      out.uns = it.getWidth() > 32;
      out.ival = static_cast<int64_t>(v.getZExtValue());
      out.dval = static_cast<double>(v.getZExtValue());
    } else {
      out.ival = v.getSExtValue();
      out.dval = static_cast<double>(out.ival);
    }
    return out;
  }
  if (mlir::isa<mlir::FloatType>(element)) {
    llvm::APFloat v = attr.isSplat() ? attr.getSplatValue<llvm::APFloat>()
                                     : *attr.getValues<llvm::APFloat>().begin();
    bool lost = false;
    v.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
              &lost);
    out.dval = v.convertToDouble();
    out.ival = static_cast<int64_t>(out.dval);
    return out;
  }
  MslDecline("constant element type");
}

// ------------------------------------------------------------- Sym builders

}  // namespace

namespace {

Sym* MakeCounter(SymArena& a, int64_t ca, int64_t cb, int64_t base) {
  Sym* s = a.make(SymKind::kCounter);
  s->a = ca;
  s->b = cb;
  s->base = base;
  s->dtype = "int";
  return s;
}

Sym* MakeConst(SymArena& a, const ConstVal& v, const std::string& dtype,
               const MslShape& shape = {}) {
  Sym* s = a.make(SymKind::kConst);
  s->dval = v.dval;
  s->ival = v.ival;
  s->uns = v.uns;
  s->dtype = dtype;
  s->shape = shape;
  return s;
}

Sym* MakeIntConst(SymArena& a, int64_t v, const std::string& dtype,
                  const MslShape& shape = {}) {
  ConstVal c;
  c.ival = v;
  c.dval = static_cast<double>(v);
  return MakeConst(a, c, dtype, shape);
}

Sym* MakeLeaf(SymArena& a, LeafKind kind, const MslShape& shape,
              const std::string& dtype, const MslSrc& src) {
  Sym* s = a.make(SymKind::kLeaf);
  s->leaf = kind;
  s->shape = shape;
  s->dtype = dtype;
  s->source = src;
  s->strides = Rowmajor(shape);
  return s;
}

Sym* MakeElem(SymArena& a, const std::string& op, std::vector<Sym*> args,
              const std::string& dtype, const MslShape& shape,
              const std::string& extra = "") {
  Sym* s = a.make(SymKind::kElem);
  s->op = op;
  s->args = std::move(args);
  s->dtype = dtype;
  s->shape = shape;
  s->extra = extra;
  return s;
}

Sym* MakeDot(SymArena& a, Sym* data, Sym* weight, std::vector<MslRole> roles,
             std::vector<MslRole> widx, int64_t csize, int64_t dsize,
             const std::string& dtype, const MslShape& shape) {
  Sym* s = a.make(SymKind::kDot);
  s->data = data;
  s->weight = weight;
  s->roles = std::move(roles);
  s->widx = std::move(widx);
  s->csize = csize;
  s->dsize = dsize;
  s->dtype = dtype;
  s->shape = shape;
  return s;
}

Sym* MakePerm(SymArena& a, Sym* inner, const std::vector<int64_t>& perm) {
  Sym* s = a.make(SymKind::kPerm);
  s->inner = inner;
  s->perm = perm;
  for (int64_t p : perm) s->shape.push_back(inner->shape[p]);
  s->dtype = inner->dtype;
  return s;
}

Sym* MakePad(SymArena& a, Sym* inner, int64_t lo, int64_t n,
             const MslShape& shape) {
  Sym* s = a.make(SymKind::kPad);
  s->inner = inner;
  s->lo = lo;
  s->n = n;
  s->shape = shape;
  s->dtype = inner->dtype;
  return s;
}

Sym* MakeStack(SymArena& a, Sym* base, std::map<int64_t, Sym*> parts,
               const MslShape& shape, const std::string& dtype,
               int64_t axis = 0) {
  Sym* s = a.make(SymKind::kStack);
  s->sbase = base;
  s->parts = std::move(parts);
  s->axis = axis;
  s->shape = shape;
  s->dtype = dtype;
  return s;
}

Sym* MakeIota(SymArena& a, int64_t axis, int64_t start, const MslShape& shape,
              const std::string& dtype) {
  Sym* s = a.make(SymKind::kIota);
  s->axis = axis;
  s->start = start;
  s->shape = shape;
  s->dtype = dtype;
  return s;
}

Sym* MakeRedReg(SymArena& a, Sym* inner, const MslShape& shape,
                const std::string& dtype) {
  Sym* s = a.make(SymKind::kRedReg);
  s->inner = inner;
  s->shape = shape;
  s->dtype = dtype;
  return s;
}

Sym* MakeAccRed(SymArena& a, Sym* inner, const std::vector<int64_t>& dims,
                const MslShape& shape, const std::string& dtype,
                const std::vector<int64_t>* perm = nullptr) {
  Sym* s = a.make(SymKind::kAccRed);
  s->inner = inner;
  s->dims = dims;
  s->shape = shape;
  s->dtype = dtype;
  if (perm != nullptr) {
    s->perm = *perm;
  } else {
    for (size_t i = 0; i < shape.size(); i++)
      s->perm.push_back(static_cast<int64_t>(i));
  }
  return s;
}

Sym* MakeAccDot(SymArena& a, Sym* lhs, Sym* rhs,
                const std::vector<std::vector<int64_t>>& dims4,
                const MslShape& shape, const std::string& dtype,
                const std::vector<int64_t>* perm = nullptr) {
  Sym* s = a.make(SymKind::kAccDot);
  s->lhs = lhs;
  s->rhs = rhs;
  s->dims4 = dims4;
  s->shape = shape;
  s->dtype = dtype;
  if (perm != nullptr) {
    s->perm = *perm;
  } else {
    for (size_t i = 0; i < shape.size(); i++)
      s->perm.push_back(static_cast<int64_t>(i));
  }
  return s;
}

// msl_scan.py `_clone_leafish`.
Sym* CloneLeafish(SymArena& a, const Sym* x) {
  if (x->kind == SymKind::kLeaf) {
    Sym* s = a.make(SymKind::kLeaf);
    *s = *x;
    return s;
  }
  if (x->kind == SymKind::kElem) {
    Sym* s = a.make(SymKind::kElem);
    *s = *x;
    return s;
  }
  MslDecline(absl::StrCat("reshape of ", KindName(x)));
}

// msl_scan.py `_transpose_sym`: a transpose commutes with elementwise ops, so
// push it down to the leaves; anything else defers as a SymPerm.
Sym* TransposeSym(SymArena& a, Sym* x, const std::vector<int64_t>& perm) {
  auto permuted = [&](const MslShape& s) {
    MslShape out;
    for (int64_t p : perm) out.push_back(s[p]);
    return out;
  };
  switch (x->kind) {
    case SymKind::kAccRed: {
      std::vector<int64_t> np;
      for (int64_t p : perm) np.push_back(x->perm[p]);
      return MakeAccRed(a, x->inner, x->dims, permuted(x->shape), x->dtype,
                        &np);
    }
    case SymKind::kAccDot: {
      std::vector<int64_t> np;
      for (int64_t p : perm) np.push_back(x->perm[p]);
      return MakeAccDot(a, x->lhs, x->rhs, x->dims4, permuted(x->shape),
                        x->dtype, &np);
    }
    case SymKind::kDot: {
      std::vector<MslRole> roles;
      for (int64_t p : perm) roles.push_back(x->roles[p]);
      return MakeDot(a, x->data, x->weight, roles, x->widx, x->csize, x->dsize,
                     x->dtype, permuted(x->shape));
    }
    case SymKind::kLeaf: {
      Sym* out = CloneLeafish(a, x);
      out->shape = permuted(x->shape);
      MslShape st;
      for (int64_t p : perm) st.push_back(x->strides[p]);
      out->strides = st;
      return out;
    }
    case SymKind::kConst: {
      ConstVal v{x->dval, x->ival, x->uns};
      return MakeConst(a, v, x->dtype, permuted(x->shape));
    }
    case SymKind::kCounter:
      return x;
    case SymKind::kPerm: {
      std::vector<int64_t> np;
      for (int64_t p : perm) np.push_back(x->perm[p]);
      return MakePerm(a, x->inner, np);
    }
    case SymKind::kRedReg:
      return MakeRedReg(a, x->inner, permuted(x->shape), x->dtype);
    case SymKind::kElem: {
      const size_t n = x->shape.size();
      std::vector<Sym*> args;
      for (Sym* arg : x->args) {
        const MslShape& ashape = arg->shape;
        bool all_one = true;
        for (int64_t d : ashape) all_one = all_one && d == 1;
        if (ashape.empty() || all_one) {
          args.push_back(arg);
          continue;
        }
        const size_t k = n - ashape.size();
        Sym* a2 = arg;
        if (k != 0) {
          // Expand right-aligned args to full rank so the perm applies.
          if (arg->kind == SymKind::kLeaf) {
            a2 = CloneLeafish(a, arg);
            MslShape s(k, 1), st(k, 0);
            s.insert(s.end(), ashape.begin(), ashape.end());
            st.insert(st.end(), arg->strides.begin(), arg->strides.end());
            a2->shape = s;
            a2->strides = st;
          } else if (arg->kind == SymKind::kConst) {
            MslShape s(k, 1);
            s.insert(s.end(), ashape.begin(), ashape.end());
            ConstVal v{arg->dval, arg->ival, arg->uns};
            a2 = MakeConst(a, v, arg->dtype, s);
          } else if (arg->kind == SymKind::kIota) {
            MslShape s(k, 1);
            s.insert(s.end(), ashape.begin(), ashape.end());
            a2 = MakeIota(a, arg->axis + static_cast<int64_t>(k), arg->start, s,
                          arg->dtype);
          } else {
            return MakePerm(a, x, perm);
          }
        }
        args.push_back(TransposeSym(a, a2, perm));
      }
      return MakeElem(a, x->op, args, x->dtype, permuted(x->shape), x->extra);
    }
    default:
      return MakePerm(a, x, perm);
  }
}

// msl_scan.py `_contains_accdot`.
bool ContainsAccDot(Sym* root) {
  std::vector<Sym*> stack{root};
  llvm::DenseSet<const Sym*> seen;
  while (!stack.empty()) {
    Sym* s = stack.back();
    stack.pop_back();
    if (s == nullptr || !seen.insert(s).second) continue;
    if (s->kind == SymKind::kAccDot || s->kind == SymKind::kAccRed) return true;
    if (s->kind == SymKind::kElem) {
      for (Sym* a : s->args) stack.push_back(a);
    } else if (s->kind == SymKind::kDot) {
      stack.push_back(s->data);
    } else if (s->kind == SymKind::kPad || s->kind == SymKind::kPerm ||
               s->kind == SymKind::kRedReg) {
      stack.push_back(s->inner);
    }
  }
  return false;
}

// msl_scan.py `_collect_dots`.
std::vector<Sym*> CollectDots(const std::vector<Sym*>& roots) {
  llvm::DenseSet<const Sym*> seen;
  std::vector<Sym*> stack(roots), out;
  while (!stack.empty()) {
    Sym* s = stack.back();
    stack.pop_back();
    if (s == nullptr || !seen.insert(s).second) continue;
    if (s->kind == SymKind::kDot) {
      out.push_back(s);
      stack.push_back(s->data);
    } else if (s->kind == SymKind::kElem) {
      for (Sym* a : s->args) stack.push_back(a);
    } else if (s->kind == SymKind::kPad) {
      stack.push_back(s->inner);
    }
  }
  return out;
}

// msl_scan.py `_needs_registers`: register-resident constructs scalar mode
// cannot emit.  The `parts.values()` half is load-bearing -- iterating the
// dict yields KEYS, and the bug that came of it lost plans silently.
bool NeedsRegisters(const std::vector<Sym*>& roots) {
  llvm::DenseSet<const Sym*> seen;
  std::vector<Sym*> stack(roots);
  while (!stack.empty()) {
    Sym* s = stack.back();
    stack.pop_back();
    if (s == nullptr || !seen.insert(s).second) continue;
    if (s->kind == SymKind::kRedReg || s->kind == SymKind::kPad) return true;
    if (s->kind == SymKind::kElem) {
      for (Sym* a : s->args) stack.push_back(a);
    } else if (s->kind == SymKind::kStack) {
      stack.push_back(s->sbase);
      for (const auto& kv : s->parts) stack.push_back(kv.second);
    }
  }
  return false;
}

// msl_scan.py `_match_accum`: state' = state + a sum of cross-lane
// accumulators.
bool MatchAccum(Sym* s, int64_t pos, std::vector<Sym*>* out) {
  std::vector<Sym*> terms;
  std::function<void(Sym*)> flatten = [&](Sym* x) {
    if (x->kind == SymKind::kElem && x->op == "stablehlo.add") {
      for (Sym* a : x->args) flatten(a);
    } else {
      terms.push_back(x);
    }
  };
  flatten(s);
  std::vector<Sym*> carry, accs, rest;
  for (Sym* t : terms) {
    const bool is_carry = t->kind == SymKind::kLeaf &&
                          t->leaf == LeafKind::kArg &&
                          t->source.kind == SrcKind::kCarry &&
                          t->source.carry == pos &&
                          t->strides == Rowmajor(t->shape);
    const bool is_acc =
        t->kind == SymKind::kAccDot || t->kind == SymKind::kAccRed;
    if (is_carry) {
      carry.push_back(t);
    } else if (is_acc) {
      accs.push_back(t);
    } else if (!IsZeroConst(t)) {
      rest.push_back(t);
    }
  }
  if (carry.size() == 1 && !accs.empty() && rest.empty()) {
    *out = accs;
    return true;
  }
  return false;
}

}  // namespace

// ============================================================== the analyzer
//
// msl_scan.py's `_Analyzer`: symbolically evaluates a loop body block into Sym
// expressions.

namespace {

class MslAnalyzer {
 public:
  MslAnalyzer(const MslEnv& env, SymArena& arena, mlir::Block& body,
              int64_t counter_pos, bool no_inlane_dot,
              MslPlanned::Fired* fired)
      : env_(env), arena_(arena), body_(body), counter_pos_(counter_pos),
        no_inlane_dot_(no_inlane_dot), fired_(fired) {}

  using Env = llvm::DenseMap<mlir::Value, Sym*>;

  std::vector<Sym*> Analyze();

  // Everything Plan.__init__ reads off the analyzer afterwards.
  std::set<int64_t> counter_seeded;
  std::set<int64_t> hoisted_read_srcs;
  llvm::DenseMap<const Sym*, std::pair<Sym*, Sym*>> updates;
  bool inlane_fired = false;

  // Used by Plan.__init__'s `_squeeze_units`.
  Sym* DropDim(Sym* s, int64_t axis);

 private:
  Sym* Lookup(Env& env, mlir::Value v);
  std::vector<Sym*> EvalBlock(mlir::Block& block, Env& env);
  bool TryHoist(mlir::Operation* o, Env& env,
                const llvm::DenseSet<mlir::Value>& block_args,
                std::vector<Sym*>* out);
  std::vector<Sym*> EvalOp(mlir::Operation* o, Env& env);

  bool Varies(const Sym* s, int64_t axis);
  Sym* SqueezeAxis(Sym* s, int64_t axis);
  Sym* ReduceAsDot(Sym* x, const std::vector<int64_t>& dims,
                   const MslShape& out_shape);
  Sym* DotGeneral(mlir::Operation* o, const std::vector<Sym*>& ins);
  Sym* InlaneDot(Sym* lhs, Sym* rhs,
                 const std::vector<std::vector<int64_t>>& dims4,
                 const MslShape& shape, const std::string& rdt);
  Sym* CounterArith(const std::string& name, Sym* a, Sym* b);
  Sym* Sliced(Sym* x, const std::vector<int64_t>& starts,
              const MslShape& sizes);
  Sym* Reshaped(Sym* x, const MslShape& out_shape);
  Sym* Broadcasted(Sym* x, const MslShape& out_shape,
                   const std::vector<int64_t>& dims);
  static std::optional<int64_t> StaticIndex(const Sym* sym);
  Sym* DynamicSlice(mlir::Operation* o, const std::vector<Sym*>& ins);
  Sym* DynamicUpdate(mlir::Operation* o, const std::vector<Sym*>& ins);

  const MslEnv& env_;
  SymArena& arena_;
  mlir::Block& body_;
  int64_t counter_pos_;
  bool no_inlane_dot_;
  MslPlanned::Fired* fired_;
  llvm::DenseSet<mlir::Value> hoist_ok_;
};

std::vector<Sym*> MslAnalyzer::Analyze() {
  Env env;
  int64_t i = 0;
  for (mlir::BlockArgument arg : body_.getArguments()) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
    if (!t) MslDecline("a carry that is not a ranked tensor");
    const MslShape shape = ShapeOf(arg);
    std::optional<std::string> el = TapeElementName(t.getElementType());
    const bool int_scalar =
        shape.empty() && el.has_value() && (*el == "i32" || *el == "i64");
    if (i == counter_pos_ || int_scalar) {
      env[arg] = MakeCounter(arena_, 1, 0, i);
      counter_seeded.insert(i);
    } else {
      MslSrc src;
      src.kind = SrcKind::kCarry;
      src.carry = i;
      env[arg] = MakeLeaf(arena_, LeafKind::kArg, shape, Dt(arg), src);
    }
    i++;
  }
  return EvalBlock(body_, env);
}

Sym* MslAnalyzer::Lookup(Env& env, mlir::Value v) {
  auto it = env.find(v);
  if (it != env.end()) return it->second;
  // Captured from an enclosing scope.  Splat constants fold to SymConst (jax
  // hoists e.g. the loop increment out of the body); anything else is a
  // loop-invariant tensor input.
  Sym* s = nullptr;
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) MslDecline("a value that is not a ranked tensor");
  if (mlir::Operation* o = v.getDefiningOp()) {
    if (o->getName().getStringRef() == "stablehlo.constant") {
      auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(o->getAttr("value"));
      if (attr && (attr.isSplat() || t.getRank() == 0)) {
        s = MakeConst(arena_, ConstScalar(attr, t.getElementType()),
                      Dt(t.getElementType()), ShapeOf(v));
      }
    }
  }
  if (s == nullptr) {
    MslSrc src;
    src.kind = SrcKind::kFree;
    src.value = v;
    s = MakeLeaf(arena_, LeafKind::kWhole, ShapeOf(v), Dt(v), src);
  }
  env[v] = s;
  return s;
}

std::vector<Sym*> MslAnalyzer::EvalBlock(mlir::Block& block, Env& env) {
  llvm::DenseSet<mlir::Value> block_args;
  for (mlir::BlockArgument a : block.getArguments()) block_args.insert(a);
  std::vector<Sym*> result;
  bool have_result = false;
  for (mlir::Operation& op : block) {
    mlir::Operation* o = &op;
    const llvm::StringRef name = o->getName().getStringRef();
    if (name == "func.return" || name == "stablehlo.return") {
      for (mlir::Value v : o->getOperands()) result.push_back(Lookup(env, v));
      have_result = true;
      break;
    }
    std::vector<Sym*> outs;
    try {
      outs = EvalOp(o, env);
    } catch (const MslUnsupported&) {
      // The op cannot be symified: it may still be loop-invariant, in which
      // case the plan reads its value as a whole tensor.  Otherwise the
      // original decline stands.
      if (!TryHoist(o, env, block_args, &outs)) throw;
    }
    if (outs.size() != o->getNumResults())
      MslDecline(absl::StrCat("op ", std::string(name), " result count"));
    for (unsigned r = 0; r < o->getNumResults(); r++)
      env[o->getResult(r)] = outs[r];
  }
  if (!have_result) MslDecline("block without a terminator");
  return result;
}

bool MslAnalyzer::TryHoist(mlir::Operation* o, Env& env,
                           const llvm::DenseSet<mlir::Value>& block_args,
                           std::vector<Sym*>* out) {
  // Represent a loop-invariant op we cannot symify as a 'whole' input: the
  // plan evaluates its IR subgraph once per call and feeds the result as a
  // buffer.
  if (o->getNumRegions() != 0 || Flags().no_hoist) return false;

  std::function<bool(const Sym*)> sym_invariant = [&](const Sym* sym) -> bool {
    if (sym->kind == SymKind::kConst || sym->kind == SymKind::kIota)
      return true;
    if (sym->kind == SymKind::kLeaf) {
      if (sym->leaf == LeafKind::kWhole) return true;
      if (sym->leaf == LeafKind::kRead &&
          (sym->idx == nullptr || sym->idx->a == 0)) {
        // A fixed-index slice is constant across iterations as long as the
        // source carry is never mutated (checked at classification through
        // `hoisted_read_srcs`).
        if (sym->source.kind == SrcKind::kCarry)
          hoisted_read_srcs.insert(sym->source.carry);
        return true;
      }
      return false;
    }
    if (sym->kind == SymKind::kElem) {
      for (const Sym* a : sym->args)
        if (!sym_invariant(a)) return false;
      return true;
    }
    if (sym->kind == SymKind::kPad || sym->kind == SymKind::kPerm ||
        sym->kind == SymKind::kRedReg)
      return sym_invariant(sym->inner);
    if (sym->kind == SymKind::kDot) return sym_invariant(sym->data);
    return false;
  };

  for (mlir::Value v : o->getOperands()) {
    if (block_args.contains(v)) return false;
    auto it = env.find(v);
    if (it != env.end()) {
      if (!hoist_ok_.contains(v) && !sym_invariant(it->second)) return false;
    }
    // A capture from an enclosing scope is invariant.
  }
  if (!env_.op_supported(o->getName().getStringRef())) return false;
  out->clear();
  for (mlir::Value r : o->getResults()) {
    MslSrc src;
    src.kind = SrcKind::kHoist;
    src.value = r;
    out->push_back(MakeLeaf(arena_, LeafKind::kWhole, ShapeOf(r), Dt(r), src));
    hoist_ok_.insert(r);
  }
  return true;
}

std::vector<Sym*> MslAnalyzer::EvalOp(mlir::Operation* o, Env& env) {
  const std::string name(o->getName().getStringRef());
  std::vector<Sym*> ins;
  for (mlir::Value v : o->getOperands()) ins.push_back(Lookup(env, v));

  if (name == "stablehlo.constant") {
    auto t = mlir::cast<mlir::RankedTensorType>(o->getResult(0).getType());
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(o->getAttr("value"));
    if (!attr) MslDecline("constant without a dense value");
    if (!attr.isSplat() && t.getRank() != 0)
      MslDecline("non-splat constant");
    return {MakeConst(arena_, ConstScalar(attr, t.getElementType()),
                      Dt(t.getElementType()), ShapeOf(o->getResult(0)))};
  }

  if (name == "func.call") {
    auto sym = o->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
    if (!sym) MslDecline("func.call without a callee");
    mlir::ModuleOp module = env_.module;   // ModuleOp's lookup is non-const
    auto fn = module.lookupSymbol<mlir::func::FuncOp>(sym.getValue());
    if (!fn || fn.getBody().empty() ||
        fn.getBody().getBlocks().size() != 1)
      MslDecline("func.call to an unknown callee");
    mlir::Block& blk = fn.getBody().front();
    if (blk.getNumArguments() != ins.size())
      MslDecline("func.call arity mismatch");
    Env sub;
    for (unsigned i = 0; i < blk.getNumArguments(); i++)
      sub[blk.getArgument(i)] = ins[i];
    return EvalBlock(blk, sub);
  }

  if (name == "stablehlo.add" || name == "stablehlo.subtract" ||
      name == "stablehlo.multiply") {
    Sym* a = ins[0];
    Sym* b = ins[1];
    auto scalarish = [](const Sym* s) {
      return (s->kind == SymKind::kCounter || s->kind == SymKind::kConst) &&
             s->shape.empty() &&
             (s->dtype == "int" || s->dtype == "i32" || s->dtype == "i64");
    };
    if (scalarish(a) && scalarish(b)) return {CounterArith(name, a, b)};
  }

  if (name == "stablehlo.reshape")
    return {Reshaped(ins[0], ShapeOf(o->getResult(0)))};

  if (name == "stablehlo.broadcast_in_dim")
    return {Broadcasted(ins[0], ShapeOf(o->getResult(0)),
                        I64List(o, "broadcast_dimensions"))};

  if (name == "stablehlo.while") {
    if (Flags().no_nested) MslDecline("op stablehlo.while (disabled)");
    // Symbolically unroll small statically-counted nested loops (lrnn's rank
    // recursion nests a tiny scan inside the sequence scan); the inner counter
    // becomes a constant per iteration.
    std::optional<std::pair<int64_t, int64_t>> counted = env_.counted(o);
    if (!counted.has_value())
      MslDecline("op stablehlo.while (dynamic nested)");
    const int64_t k = counted->first;
    const int64_t bound = counted->second;
    std::optional<int64_t> start = env_.static_start(o, k);
    if (!start.has_value()) {
      if (k < static_cast<int64_t>(ins.size())) {
        Sym* st = ins[k];
        if (st->kind == SymKind::kConst) {
          start = st->ival;
        } else if (st->kind == SymKind::kCounter && st->a == 0) {
          start = st->b;
        }
      }
    }
    if (!start.has_value())
      MslDecline("op stablehlo.while (dynamic nested start)");
    const int64_t trip = bound - *start;
    if (trip <= 0 || trip > 64)
      MslDecline(absl::StrCat("op stablehlo.while (nested trip ", trip, ")"));
    mlir::Block& body = o->getRegion(1).front();
    std::vector<Sym*> vals(ins);
    for (int64_t it = 0; it < trip; it++) {
      Env env2 = env;
      for (unsigned j = 0; j < body.getNumArguments(); j++)
        env2[body.getArgument(j)] = vals[j];
      env2[body.getArgument(static_cast<unsigned>(k))] =
          MakeIntConst(arena_, *start + it, "i32");
      vals = EvalBlock(body, env2);
    }
    return vals;
  }

  if (name == "stablehlo.iota") {
    auto t = mlir::cast<mlir::RankedTensorType>(o->getResult(0).getType());
    return {MakeIota(arena_, IntAttr(o, "iota_dimension"), 0,
                     ShapeOf(o->getResult(0)), Dt(t.getElementType()))};
  }

  if (name == "stablehlo.concatenate") {
    const int64_t dim = IntAttr(o, "dimension");
    const MslShape out_shape = ShapeOf(o->getResult(0));
    if (dim != static_cast<int64_t>(out_shape.size()) - 1)
      MslDecline(absl::StrCat("concat on non-feature dim ", dim, " of ",
                              Join(out_shape)));
    Sym* acc = nullptr;
    int64_t off = 0;
    for (Sym* x : ins) {
      // The span is the PART's own width, not the concat total.
      const int64_t w = x->shape.empty() ? 1 : x->shape.back();
      Sym* part = MakePad(arena_, x, off, w, out_shape);
      off += w;
      acc = acc == nullptr
                ? part
                : MakeElem(arena_, "stablehlo.add", {acc, part}, x->dtype,
                           out_shape);
    }
    return {acc};
  }

  if (name == "stablehlo.slice") {
    const std::vector<int64_t> starts = I64List(o, "start_indices");
    const std::vector<int64_t> limits = I64List(o, "limit_indices");
    const std::vector<int64_t> steps = I64List(o, "strides");
    for (int64_t st : steps)
      if (st != 1) MslDecline("strided slice");
    MslShape sizes;
    for (size_t i = 0; i < starts.size(); i++)
      sizes.push_back(limits[i] - starts[i]);
    return {Sliced(ins[0], starts, sizes)};
  }

  if (name == "stablehlo.pad") {
    Sym* x = ins[0];
    Sym* pv = ins[1];
    if (!IsZeroConst(pv)) MslDecline("non-zero pad");
    const std::vector<int64_t> low = I64List(o, "edge_padding_low");
    const std::vector<int64_t> high = I64List(o, "edge_padding_high");
    const std::vector<int64_t> interior = I64List(o, "interior_padding");
    for (int64_t i : interior)
      if (i != 0) MslDecline("interior pad");
    const MslShape out_shape = ShapeOf(o->getResult(0));
    for (size_t i = 0; i + 1 < low.size(); i++)
      if (low[i] != 0 || high[i] != 0) MslDecline("pad on lane dims");
    if (low.empty() || (low.back() == 0 && high.back() == 0)) return {x};
    return {MakePad(arena_, x, low.back(),
                    x->shape.empty() ? 1 : x->shape.back(), out_shape)};
  }

  if (name == "stablehlo.reduce") {
    if (ins.size() != 2) MslDecline("variadic reduce");
    Sym* x = ins[0];
    Sym* init = ins[1];
    if (!IsZeroConst(init)) MslDecline("reduce with non-zero init");
    std::vector<mlir::Operation*> body_ops;
    for (mlir::Operation& b : o->getRegion(0).front()) body_ops.push_back(&b);
    if (body_ops.size() != 2 ||
        body_ops[0]->getName().getStringRef() != "stablehlo.add")
      MslDecline("non-add reduce");
    const std::vector<int64_t> dims = I64List(o, "dimensions");
    const MslShape out_shape = ShapeOf(o->getResult(0));
    bool all_unit = true;
    for (int64_t d : dims) all_unit = all_unit && x->shape[d] == 1;
    if (all_unit) {
      // Reducing unit dims is a squeeze, not a contraction.
      if (x->kind == SymKind::kAccDot || x->kind == SymKind::kAccRed ||
          x->kind == SymKind::kDot)
        return {Reshaped(x, out_shape)};
      Sym* out = x;
      std::vector<int64_t> sorted(dims);
      std::sort(sorted.begin(), sorted.end(), std::greater<int64_t>());
      for (int64_t d : sorted) out = DropDim(out, d);
      return {out};
    }
    Sym* dot = ReduceAsDot(x, dims, out_shape);
    if (dot != nullptr) return {dot};
    if (dims.size() == 1 &&
        dims[0] == static_cast<int64_t>(x->shape.size()) - 1 &&
        !Flags().no_redreg) {
      // In-lane: reduce over the register dim.
      return {MakeRedReg(arena_, x, out_shape, x->dtype)};
    }
    return {MakeAccRed(arena_, x, dims, out_shape, x->dtype)};
  }

  if (name == "stablehlo.dynamic_slice") return {DynamicSlice(o, ins)};
  if (name == "stablehlo.dynamic_update_slice") return {DynamicUpdate(o, ins)};
  if (name == "stablehlo.dot_general") return {DotGeneral(o, ins)};

  if (name == "stablehlo.transpose")
    return {TransposeSym(arena_, ins[0], I64List(o, "permutation"))};

  if (name == "stablehlo.convert") {
    auto t = mlir::cast<mlir::RankedTensorType>(o->getResult(0).getType());
    const std::string dt = Dt(t.getElementType());
    return {MakeElem(arena_, "convert", {ins[0]}, dt, ins[0]->shape, dt)};
  }

  if (name == "stablehlo.compare") {
    auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(o);
    if (!cmp) MslDecline("compare direction");
    std::string dir;
    switch (cmp.getComparisonDirection()) {
      case mlir::stablehlo::ComparisonDirection::EQ: dir = "EQ"; break;
      case mlir::stablehlo::ComparisonDirection::NE: dir = "NE"; break;
      case mlir::stablehlo::ComparisonDirection::LT: dir = "LT"; break;
      case mlir::stablehlo::ComparisonDirection::LE: dir = "LE"; break;
      case mlir::stablehlo::ComparisonDirection::GT: dir = "GT"; break;
      case mlir::stablehlo::ComparisonDirection::GE: dir = "GE"; break;
      default: MslDecline("compare direction");
    }
    return {MakeElem(arena_, "compare", ins, "i1",
                     Bshape(ins[0]->shape, ins[1]->shape), dir)};
  }

  if (name == "stablehlo.select") {
    return {MakeElem(arena_, "select", ins, ins[1]->dtype,
                     Bshape(Bshape(ins[0]->shape, ins[1]->shape),
                            ins[2]->shape))};
  }

  if (name == "stablehlo.clamp")
    return {MakeElem(arena_, "clamp", ins, ins[1]->dtype, ins[1]->shape)};

  if (MslUnaryOps().count(name) != 0)
    return {MakeElem(arena_, name, ins, ins[0]->dtype, ins[0]->shape)};

  if (MslBinaryOps().count(name) != 0)
    return {MakeElem(arena_, name, ins, ins[0]->dtype,
                     Bshape(ins[0]->shape, ins[1]->shape))};

  MslDecline(absl::StrCat("op ", name));
}

// msl_scan.py `_varies`: does `s` vary along `axis`?  Conservative toward
// True for unknown node kinds.
bool MslAnalyzer::Varies(const Sym* s, int64_t axis) {
  const MslShape& shape = s->shape;
  if (axis >= static_cast<int64_t>(shape.size()) || shape[axis] == 1)
    return false;
  if (s->kind == SymKind::kConst || s->kind == SymKind::kCounter) return false;
  if (s->kind == SymKind::kIota) return axis == s->axis;
  if (s->kind == SymKind::kLeaf) return s->strides[axis] != 0;
  if (s->kind == SymKind::kElem) {
    const int64_t k = static_cast<int64_t>(shape.size());
    for (const Sym* a : s->args) {
      const int64_t ax = axis - (k - static_cast<int64_t>(a->shape.size()));
      if (ax >= 0 && Varies(a, ax)) return true;
    }
    return false;
  }
  return true;
}

// msl_scan.py `_dropdim`: remove a size-1 axis.
Sym* MslAnalyzer::DropDim(Sym* s, int64_t axis) {
  MslShape shape(s->shape);
  shape.erase(shape.begin() + axis);
  switch (s->kind) {
    case SymKind::kConst: {
      ConstVal v{s->dval, s->ival, s->uns};
      return MakeConst(arena_, v, s->dtype, shape);
    }
    case SymKind::kLeaf: {
      Sym* out = CloneLeafish(arena_, s);
      out->shape = shape;
      MslShape st(s->strides);
      st.erase(st.begin() + axis);
      out->strides = st;
      return out;
    }
    case SymKind::kElem: {
      const int64_t k = static_cast<int64_t>(s->shape.size());
      std::vector<Sym*> args;
      for (Sym* a : s->args) {
        const int64_t ax = axis - (k - static_cast<int64_t>(a->shape.size()));
        if (ax < 0) {
          args.push_back(a);
        } else if (a->shape[ax] == 1) {
          args.push_back(DropDim(a, ax));
        } else {
          MslDecline("dropdim of varying arg");
        }
      }
      return MakeElem(arena_, s->op, args, s->dtype, shape, s->extra);
    }
    case SymKind::kRedReg:
      return MakeRedReg(arena_, s->inner, shape, s->dtype);
    case SymKind::kIota: {
      if (axis == s->axis) MslDecline("dropdim of iota axis");
      return MakeIota(arena_, s->axis - (axis < s->axis ? 1 : 0), s->start,
                      shape, s->dtype);
    }
    case SymKind::kStack: {
      if (axis == s->axis) {
        auto it = s->parts.find(0);
        if (s->shape[axis] == 1 && it != s->parts.end())
          return DropDim(it->second, axis);   // a one-slot stack IS its part
        MslDecline("dropdim of stack axis");
      }
      std::map<int64_t, Sym*> parts;
      for (const auto& kv : s->parts)
        parts[kv.first] = DropDim(kv.second, axis);
      return MakeStack(arena_, DropDim(s->sbase, axis), parts, shape, s->dtype,
                       s->axis - (axis < s->axis ? 1 : 0));
    }
    default:
      MslDecline(absl::StrCat("dropdim of ", KindName(s)));
  }
}

// msl_scan.py `_squeeze_axis`: slice a non-varying axis to width 1, drop it.
Sym* MslAnalyzer::SqueezeAxis(Sym* s, int64_t axis) {
  std::vector<int64_t> starts(s->shape.size(), 0);
  MslShape sizes(s->shape);
  sizes[axis] = 1;
  return DropDim(Sliced(s, starts, sizes), axis);
}

// msl_scan.py `_reduce_as_dot`: recognize reduce(add, multiply(a, b), dims) as
// a contraction -- texmo's lrnn lowers its block matvec that way.
Sym* MslAnalyzer::ReduceAsDot(Sym* x, const std::vector<int64_t>& dims,
                              const MslShape& out_shape) {
  if (!(x->kind == SymKind::kElem && x->op == "stablehlo.multiply" &&
        x->args.size() == 2))
    return nullptr;
  const int64_t rank = static_cast<int64_t>(x->shape.size());
  Sym* a = x->args[0];
  Sym* b = x->args[1];

  auto var_axes = [&](Sym* s) {
    std::set<int64_t> out;
    const int64_t k = rank - static_cast<int64_t>(s->shape.size());
    for (int64_t ax = 0; ax < rank; ax++)
      if (ax >= k && Varies(s, ax - k)) out.insert(ax);
    return out;
  };
  const std::set<int64_t> va = var_axes(a), vb = var_axes(b);
  auto invariant_leaf = [](const Sym* s) {
    return s->kind == SymKind::kLeaf &&
           (s->leaf == LeafKind::kWhole || s->leaf == LeafKind::kArg);
  };

  // Case 1: in-lane dot.  Exactly one reduced dim; it and the 'd' output dim
  // must be the two trailing dims.
  if (dims.size() == 1) {
    const int64_t red = dims[0];
    for (int side = 0; side < 2; side++) {
      Sym* w = side == 0 ? a : b;
      Sym* d_expr = side == 0 ? b : a;
      const std::set<int64_t>& vw = side == 0 ? va : vb;
      const std::set<int64_t>& vd = side == 0 ? vb : va;
      if (!(invariant_leaf(w) && vw.count(red) && vd.count(red))) continue;
      std::vector<int64_t> d_axes;
      for (int64_t ax = 0; ax < rank; ax++)
        if (ax != red && vw.count(ax) && !vd.count(ax)) d_axes.push_back(ax);
      if (d_axes.size() != 1) continue;
      const int64_t d_ax = d_axes[0];
      if (!((d_ax == rank - 2 && red == rank - 1) ||
            (d_ax == rank - 1 && red == rank - 2)))
        continue;
      // f32 or bf16: the emitters accumulate every dot in a float register
      // regardless, and the typed result register rounds once at the end --
      // XLA's own bf16-dot semantics (f32 products, f32 accumulate, one
      // rounding).  A DIVERGENCE from msl_scan.py, which is f32-only here.
      if (x->dtype != "f32" && x->dtype != "bf16") continue;
      const int64_t csize = x->shape[red], dsize = x->shape[d_ax];
      if (csize > 4096 || dsize > 4096) continue;
      // Compact weight: drop broadcast (stride-0 or size-1) dims.
      const int64_t k = rank - static_cast<int64_t>(w->shape.size());
      Sym* wleaf = CloneLeafish(arena_, w);
      std::vector<int64_t> keep;
      for (int64_t i = 0; i < static_cast<int64_t>(w->shape.size()); i++)
        if (vw.count(i + k)) keep.push_back(i);
      MslShape wshape, wstrides;
      for (int64_t i : keep) {
        wshape.push_back(w->shape[i]);
        wstrides.push_back(w->strides[i]);
      }
      wleaf->shape = wshape;
      wleaf->strides = wstrides;
      std::map<int64_t, int64_t> lane_map;
      std::vector<int64_t> lane_axes;
      for (int64_t ax = 0; ax < rank; ax++)
        if (ax != red && ax != d_ax) lane_axes.push_back(ax);
      for (size_t li = 0; li < lane_axes.size(); li++)
        lane_map[lane_axes[li]] = static_cast<int64_t>(li);
      std::vector<MslRole> widx;
      for (int64_t i : keep) {
        const int64_t ax = i + k;
        MslRole r;
        if (ax == red) {
          r.kind = RoleKind::kC;
        } else if (ax == d_ax) {
          r.kind = RoleKind::kD;
        } else {
          r.kind = RoleKind::kData;
          r.idx = lane_map[ax];
        }
        widx.push_back(r);
      }
      Sym* full = d_expr;
      if (rank != static_cast<int64_t>(d_expr->shape.size())) {
        std::vector<int64_t> bdims;
        for (int64_t i = rank - static_cast<int64_t>(d_expr->shape.size());
             i < rank; i++)
          bdims.push_back(i);
        full = Broadcasted(d_expr, x->shape, bdims);
      }
      Sym* data = SqueezeAxis(full, d_ax);
      // `data` now has the reduced dim last.
      std::vector<MslRole> roles;
      for (int64_t ax = 0; ax < rank; ax++) {
        if (ax == red) continue;
        MslRole r;
        if (ax == d_ax) {
          r.kind = RoleKind::kReg;
        } else {
          r.kind = RoleKind::kData;
          r.idx = lane_map[ax];
        }
        roles.push_back(r);
      }
      return MakeDot(arena_, data, wleaf, roles, widx, csize, dsize, x->dtype,
                     out_shape);
    }
  }

  // Case 2: cross-lane contraction (weight gradients).
  bool case2 = true;
  for (int64_t ax : dims)
    case2 = case2 && va.count(ax) && vb.count(ax) && ax != rank - 1;
  if (case2) {
    std::map<int64_t, int64_t> ma, mb;
    Sym* ca = nullptr;
    Sym* cb = nullptr;
    auto compact = [&](Sym* s, const std::set<int64_t>& own,
                       std::map<int64_t, int64_t>* mapping) -> Sym* {
      const int64_t k = rank - static_cast<int64_t>(s->shape.size());
      Sym* out = s;
      if (k != 0) {
        std::vector<int64_t> bdims;
        for (int64_t i = k; i < rank; i++) bdims.push_back(i);
        out = Broadcasted(s, x->shape, bdims);
      }
      int64_t removed = 0;
      for (int64_t ax = 0; ax < rank; ax++) {
        if (own.count(ax)) {
          (*mapping)[ax] = ax - removed;
        } else {
          out = SqueezeAxis(out, ax - removed);
          removed++;
        }
      }
      return out;
    };
    try {
      ca = compact(a, va, &ma);
      cb = compact(b, vb, &mb);
    } catch (const MslUnsupported&) {
      return nullptr;
    }
    std::vector<int64_t> kept;
    for (int64_t ax = 0; ax < rank; ax++)
      if (std::find(dims.begin(), dims.end(), ax) == dims.end())
        kept.push_back(ax);
    for (int64_t ax : kept)
      if (!va.count(ax) && !vb.count(ax) && x->shape[ax] != 1) return nullptr;
    std::vector<int64_t> batch, a_only, b_only, ones;
    for (int64_t ax : kept) {
      const bool ia = va.count(ax) != 0, ib = vb.count(ax) != 0;
      if (ia && ib) batch.push_back(ax);
      else if (ia) a_only.push_back(ax);
      else if (ib) b_only.push_back(ax);
      else ones.push_back(ax);
    }
    if (!ones.empty()) return nullptr;   // size-1 leftovers: rare, skip
    std::vector<int64_t> lb, rb, lc, rc;
    for (int64_t ax : batch) {
      lb.push_back(ma[ax]);
      rb.push_back(mb[ax]);
    }
    for (int64_t ax : dims) {
      lc.push_back(ma[ax]);
      rc.push_back(mb[ax]);
    }
    std::vector<int64_t> einsum_order(batch);
    einsum_order.insert(einsum_order.end(), a_only.begin(), a_only.end());
    einsum_order.insert(einsum_order.end(), b_only.begin(), b_only.end());
    std::vector<int64_t> perm;
    for (int64_t ax : kept) {
      const auto it = std::find(einsum_order.begin(), einsum_order.end(), ax);
      perm.push_back(static_cast<int64_t>(it - einsum_order.begin()));
    }
    return MakeAccDot(arena_, ca, cb, {lb, rb, lc, rc}, out_shape, x->dtype,
                      &perm);
  }
  return nullptr;
}

// ops/linalg.py `_dot_dims`.
std::vector<std::vector<int64_t>> DotDims(mlir::Operation* o) {
  auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(o);
  if (!dot) MslDecline("dot_general in an unexpected form");
  const auto dn = dot.getDotDimensionNumbers();
  auto vec = [](llvm::ArrayRef<int64_t> a) {
    return std::vector<int64_t>(a.begin(), a.end());
  };
  return {vec(dn.getLhsBatchingDimensions()), vec(dn.getRhsBatchingDimensions()),
          vec(dn.getLhsContractingDimensions()),
          vec(dn.getRhsContractingDimensions())};
}

Sym* MslAnalyzer::DotGeneral(mlir::Operation* o, const std::vector<Sym*>& ins) {
  std::vector<std::vector<int64_t>> d = DotDims(o);
  std::vector<int64_t> lb = d[0], rb = d[1], lc = d[2], rc = d[3];
  Sym* lhs = ins[0];
  Sym* rhs = ins[1];
  if (lhs->kind == SymKind::kPerm) {
    for (int64_t& x : lb) x = lhs->perm[x];
    for (int64_t& x : lc) x = lhs->perm[x];
    lhs = lhs->inner;
  }
  if (rhs->kind == SymKind::kPerm) {
    for (int64_t& x : rb) x = rhs->perm[x];
    for (int64_t& x : rc) x = rhs->perm[x];
    rhs = rhs->inner;
  }
  auto invariant = [](const Sym* s) {
    // Loop-invariant weights: a free capture, or an untouched carry (verified
    // to be pass-through at classification time).
    return s->kind == SymKind::kLeaf &&
           (s->leaf == LeafKind::kWhole || s->leaf == LeafKind::kArg);
  };
  const MslShape shape = ShapeOf(o->getResult(0));
  // The RESULT dtype is the dot's own (preferred_element_type: bf16 operands
  // may produce an f32 result).  The emitters accumulate in a float register
  // either way; this dtype is the register the sum lands in.
  const std::string rdt = Dt(o->getResult(0));
  auto dot_dtype_ok = [](const std::string& d) {
    return d == "f32" || d == "bf16";
  };

  auto attempt = [&](Sym* w, Sym* data, const std::vector<int64_t>& wb,
                     const std::vector<int64_t>& db,
                     const std::vector<int64_t>& wc,
                     const std::vector<int64_t>& dc, bool w_is_lhs) -> Sym* {
    if (!invariant(w)) return nullptr;
    if (wc.size() != 1 || dc.size() != 1) return nullptr;
    if (dc[0] != static_cast<int64_t>(data->shape.size()) - 1) return nullptr;
    const int64_t csize = data->shape[dc[0]];
    std::vector<int64_t> wfree;
    for (int64_t i = 0; i < static_cast<int64_t>(w->shape.size()); i++)
      if (std::find(wb.begin(), wb.end(), i) == wb.end() && i != wc[0])
        wfree.push_back(i);
    if (wfree.size() != 1) return nullptr;
    const int64_t dsize = w->shape[wfree[0]];
    if (csize > 4096 || dsize > 4096) return nullptr;
    if (!dot_dtype_ok(w->dtype) || !dot_dtype_ok(data->dtype) ||
        w->dtype != data->dtype || !dot_dtype_ok(rdt))
      return nullptr;
    std::vector<MslRole> widx(w->shape.size());
    for (size_t i = 0; i < wb.size(); i++) {
      MslRole r;
      r.kind = RoleKind::kData;
      r.idx = db[i];
      widx[wb[i]] = r;
    }
    widx[wc[0]].kind = RoleKind::kC;
    widx[wfree[0]].kind = RoleKind::kD;
    std::vector<int64_t> dfree;
    for (int64_t i = 0; i < static_cast<int64_t>(data->shape.size()) - 1; i++)
      if (std::find(db.begin(), db.end(), i) == db.end()) dfree.push_back(i);
    std::vector<MslRole> roles;
    for (int64_t dd : db) {
      MslRole r;
      r.kind = RoleKind::kData;
      r.idx = dd;
      roles.push_back(r);
    }
    MslRole reg;
    reg.kind = RoleKind::kReg;
    if (w_is_lhs) {
      roles.push_back(reg);
      for (int64_t dd : dfree) {
        MslRole r;
        r.kind = RoleKind::kData;
        r.idx = dd;
        roles.push_back(r);
      }
    } else {
      for (int64_t dd : dfree) {
        MslRole r;
        r.kind = RoleKind::kData;
        r.idx = dd;
        roles.push_back(r);
      }
      roles.push_back(reg);
    }
    return MakeDot(arena_, data, w, roles, widx, csize, dsize, rdt, shape);
  };

  Sym* out = attempt(rhs, lhs, rb, lb, rc, lc, false);
  if (out == nullptr) out = attempt(lhs, rhs, lb, rb, lc, rc, true);
  if (out == nullptr) {
    out = InlaneDot(lhs, rhs, {lb, rb, lc, rc}, shape, rdt);
    if (out != nullptr) {
      inlane_fired = true;
      if (fired_ != nullptr) fired_->inlane = true;
    }
  }
  if (out == nullptr) {
    // Cross-lane contraction (weight-gradient accumulation): representable
    // only as a hoisted post-kernel einsum.
    out = MakeAccDot(arena_, lhs, rhs, {lb, rb, lc, rc}, shape, rdt);
  }
  return out;
}

// msl_scan.py `_inlane_dot`: small dots between two LOOP-COMPUTED values.
Sym* MslAnalyzer::InlaneDot(Sym* lhs, Sym* rhs,
                            const std::vector<std::vector<int64_t>>& dims4,
                            const MslShape& shape, const std::string& rdt) {
  const std::vector<int64_t>& lb = dims4[0];
  const std::vector<int64_t>& rb = dims4[1];
  const std::vector<int64_t>& lc = dims4[2];
  const std::vector<int64_t>& rc = dims4[3];
  if (no_inlane_dot_ || !Flags().inlane_dots) return nullptr;
  auto ok = [](const std::string& d) { return d == "f32" || d == "bf16"; };
  if (!ok(lhs->dtype) || !ok(rhs->dtype) || lhs->dtype != rhs->dtype ||
      !ok(rdt))
    return nullptr;
  if (std::max(lhs->shape.size(), rhs->shape.size()) < 3 && shape.size() < 3) {
    // Matrix-state signature only: rank-2 cases are fission territory.
    return nullptr;
  }
  try {
    const int64_t nb = static_cast<int64_t>(lb.size());
    std::vector<int64_t> lfree, rfree;
    for (int64_t i = 0; i < static_cast<int64_t>(lhs->shape.size()); i++)
      if (std::find(lb.begin(), lb.end(), i) == lb.end() &&
          std::find(lc.begin(), lc.end(), i) == lc.end())
        lfree.push_back(i);
    for (int64_t i = 0; i < static_cast<int64_t>(rhs->shape.size()); i++)
      if (std::find(rb.begin(), rb.end(), i) == rb.end() &&
          std::find(rc.begin(), rc.end(), i) == rc.end())
        rfree.push_back(i);
    if (lc.empty() && rc.empty()) {
      // Outer product: result = batch + lfree + rfree.
      if (shape.empty() || shape.back() > Flags().reg_limit) return nullptr;
      std::vector<int64_t> ldims(lhs->shape.size(), 0),
          rdims(rhs->shape.size(), 0);
      for (size_t k = 0; k < lb.size(); k++) ldims[lb[k]] = k;
      for (size_t k = 0; k < lfree.size(); k++) ldims[lfree[k]] = nb + k;
      for (size_t k = 0; k < rb.size(); k++) rdims[rb[k]] = k;
      for (size_t k = 0; k < rfree.size(); k++)
        rdims[rfree[k]] = nb + static_cast<int64_t>(lfree.size()) + k;
      Sym* a = Broadcasted(lhs, shape, ldims);
      Sym* b = Broadcasted(rhs, shape, rdims);
      // A widened result dtype (preferred_element_type) converts the
      // operands EXPLICITLY: the products are computed wide, as XLA's dot
      // does, and the emitters never mix dtypes inside one Elem.
      if (a->dtype != rdt) a = MakeElem(arena_, "convert", {a}, rdt, a->shape, rdt);
      if (b->dtype != rdt) b = MakeElem(arena_, "convert", {b}, rdt, b->shape, rdt);
      return MakeElem(arena_, "stablehlo.multiply", {a, b}, rdt, shape);
    }
    if (lc.size() == 1 && rc.size() == 1 && lhs->shape.size() >= 2 &&
        rhs->shape.size() >= 2 &&
        lc[0] == static_cast<int64_t>(lhs->shape.size()) - 1 &&
        rc[0] == static_cast<int64_t>(rhs->shape.size()) - 1 &&
        lhs->shape[lc[0]] <= Flags().reg_limit) {
      // Contraction over the trailing dim of both sides: multiply on
      // (batch + lfree + rfree + c), reduce c in registers.
      MslShape inner(shape);
      inner.push_back(lhs->shape[lc[0]]);
      const int64_t last = static_cast<int64_t>(inner.size()) - 1;
      std::vector<int64_t> ldims(lhs->shape.size(), 0),
          rdims(rhs->shape.size(), 0);
      for (size_t k = 0; k < lb.size(); k++) ldims[lb[k]] = k;
      for (size_t k = 0; k < lfree.size(); k++) ldims[lfree[k]] = nb + k;
      ldims[lc[0]] = last;
      for (size_t k = 0; k < rb.size(); k++) rdims[rb[k]] = k;
      for (size_t k = 0; k < rfree.size(); k++)
        rdims[rfree[k]] = nb + static_cast<int64_t>(lfree.size()) + k;
      rdims[rc[0]] = last;
      Sym* a = Broadcasted(lhs, inner, ldims);
      Sym* b = Broadcasted(rhs, inner, rdims);
      if (a->dtype != rdt) a = MakeElem(arena_, "convert", {a}, rdt, a->shape, rdt);
      if (b->dtype != rdt) b = MakeElem(arena_, "convert", {b}, rdt, b->shape, rdt);
      Sym* mul = MakeElem(arena_, "stablehlo.multiply", {a, b}, rdt, inner);
      return MakeRedReg(arena_, mul, shape, rdt);
    }
  } catch (const MslUnsupported& e) {
    Debug(absl::StrCat("inlane_dot bail: ", e.msg));
    return nullptr;
  }
  return nullptr;
}

Sym* MslAnalyzer::CounterArith(const std::string& name, Sym* a, Sym* b) {
  auto parts = [](const Sym* s, int64_t* pa, int64_t* pb, int64_t* base) {
    if (s->kind == SymKind::kCounter) {
      *pa = s->a;
      *pb = s->b;
      *base = s->base;
    } else {
      *pa = 0;
      *pb = s->ival;
      *base = -1;
    }
  };
  int64_t aa, ab, abase, ba, bb, bbase;
  parts(a, &aa, &ab, &abase);
  parts(b, &ba, &bb, &bbase);
  if (abase >= 0 && bbase >= 0) MslDecline("counter + counter");
  const int64_t base = abase >= 0 ? abase : bbase;
  int64_t r0 = 0, r1 = 0;
  if (name == "stablehlo.add") {
    r0 = aa + ba;
    r1 = ab + bb;
  } else if (name == "stablehlo.subtract") {
    r0 = aa - ba;
    r1 = ab - bb;
  } else {
    if (abase >= 0) {
      r0 = aa * bb;
      r1 = ab * bb;
    } else {
      r0 = ba * ab;
      r1 = bb * ab;
    }
  }
  if (base < 0) return MakeIntConst(arena_, r1, "int");
  return MakeCounter(arena_, r0, r1, base);
}

Sym* MslAnalyzer::Sliced(Sym* x, const std::vector<int64_t>& starts,
                         const MslShape& sizes) {
  switch (x->kind) {
    case SymKind::kConst: {
      ConstVal v{x->dval, x->ival, x->uns};
      return MakeConst(arena_, v, x->dtype, sizes);
    }
    case SymKind::kLeaf: {
      Sym* out = CloneLeafish(arena_, x);
      int64_t off = x->offset;
      for (size_t i = 0; i < starts.size(); i++) off += starts[i] * x->strides[i];
      out->offset = off;
      out->shape = sizes;
      return out;
    }
    case SymKind::kElem: {
      std::vector<Sym*> args;
      for (Sym* a : x->args) {
        bool all_one = true;
        for (int64_t d : a->shape) all_one = all_one && d == 1;
        if (a->kind == SymKind::kConst || a->kind == SymKind::kCounter ||
            a->shape.empty() || all_one) {
          args.push_back(a);
          continue;
        }
        const size_t k = x->shape.size() - a->shape.size();
        std::vector<int64_t> asub;
        MslShape asz;
        for (size_t i = 0; i < a->shape.size(); i++) {
          const int64_t st = starts[k + i];
          const int64_t d = a->shape[i];
          asub.push_back(d == 1 ? std::min(st, std::max<int64_t>(0, d - 1))
                                : st);
          asz.push_back(d == 1 ? 1 : sizes[k + i]);
        }
        args.push_back(Sliced(a, asub, asz));
      }
      return MakeElem(arena_, x->op, args, x->dtype, sizes, x->extra);
    }
    case SymKind::kIota:
      return MakeIota(arena_, x->axis, x->start + starts[x->axis], sizes,
                      x->dtype);
    case SymKind::kDot: {
      int64_t reg_axis = -1;
      for (size_t i = 0; i < x->roles.size(); i++)
        if (x->roles[i].kind == RoleKind::kReg) {
          reg_axis = static_cast<int64_t>(i);
          break;
        }
      if (reg_axis >= 0) {
        bool rest_full = true;
        for (size_t i = 0; i < starts.size(); i++) {
          if (static_cast<int64_t>(i) == reg_axis) continue;
          rest_full = rest_full && starts[i] == 0 && sizes[i] == x->shape[i];
        }
        if (rest_full) {
          const int64_t lo = starts[reg_axis];
          const int64_t nsize = sizes[reg_axis];
          int64_t wd = -1;
          for (size_t i = 0; i < x->widx.size(); i++)
            if (x->widx[i].kind == RoleKind::kD) {
              wd = static_cast<int64_t>(i);
              break;
            }
          if (wd < 0) MslDecline("slice of a dot without a d dim");
          Sym* w = CloneLeafish(arena_, x->weight);
          w->offset = x->weight->offset + lo * x->weight->strides[wd];
          MslShape wshape(w->shape);
          wshape[wd] = nsize;
          w->shape = wshape;
          MslShape nshape(x->shape);
          nshape[reg_axis] = nsize;
          return MakeDot(arena_, x->data, w, x->roles, x->widx, x->csize, nsize,
                         x->dtype, nshape);
        }
      }
      MslDecline(absl::StrCat("slice of ", KindName(x)));
    }
    default:
      MslDecline(absl::StrCat("slice of ", KindName(x)));
  }
}

Sym* MslAnalyzer::Reshaped(Sym* x, const MslShape& out_shape) {
  if (x->shape == out_shape) return x;
  // Only dropping/adding size-1 dims is representable.
  MslShape ca, cb;
  for (int64_t d : x->shape)
    if (d != 1) ca.push_back(d);
  for (int64_t d : out_shape)
    if (d != 1) cb.push_back(d);
  if (ca != cb)
    MslDecline(absl::StrCat("reshape ", Join(x->shape), " -> ",
                            Join(out_shape)));
  switch (x->kind) {
    case SymKind::kConst: {
      ConstVal v{x->dval, x->ival, x->uns};
      return MakeConst(arena_, v, x->dtype, out_shape);
    }
    case SymKind::kElem: {
      std::vector<Sym*> args;
      for (Sym* a : x->args)
        args.push_back(a->kind == SymKind::kConst || a->kind == SymKind::kCounter
                           ? a
                           : Reshaped(a, out_shape));
      return MakeElem(arena_, x->op, args, x->dtype, out_shape, x->extra);
    }
    case SymKind::kAccRed:
      return MakeAccRed(arena_, x->inner, x->dims, out_shape, x->dtype,
                        &x->perm);
    case SymKind::kAccDot:
      return MakeAccDot(arena_, x->lhs, x->rhs, x->dims4, out_shape, x->dtype,
                        &x->perm);
    case SymKind::kRedReg:
      return MakeRedReg(arena_, x->inner, out_shape, x->dtype);
    case SymKind::kDot: {
      // Unit-dim insertions/removals: rebuild roles positionally.
      std::vector<MslRole> roles;
      size_t i = 0;
      const MslShape& xs = x->shape;
      MslRole one;
      one.kind = RoleKind::kOne;
      for (int64_t d : out_shape) {
        if (i < xs.size() && xs[i] == d) {
          roles.push_back(x->roles[i]);
          i++;
        } else if (d == 1) {
          roles.push_back(one);
        } else if (i < xs.size() && xs[i] == 1 &&
                   x->roles[i].kind == RoleKind::kOne) {
          i++;   // dropped unit dim; retry this out dim
          if (i < xs.size() && xs[i] == d) {
            roles.push_back(x->roles[i]);
            i++;
          } else {
            MslDecline(absl::StrCat("reshape ", Join(x->shape), " -> ",
                                    Join(out_shape), " of dot"));
          }
        } else {
          MslDecline(absl::StrCat("reshape ", Join(x->shape), " -> ",
                                  Join(out_shape), " of dot"));
        }
      }
      while (i < xs.size()) {
        if (xs[i] != 1 || x->roles[i].kind != RoleKind::kOne)
          MslDecline(absl::StrCat("reshape ", Join(x->shape), " -> ",
                                  Join(out_shape), " of dot"));
        i++;
      }
      return MakeDot(arena_, x->data, x->weight, roles, x->widx, x->csize,
                     x->dsize, x->dtype, out_shape);
    }
    case SymKind::kLeaf: {
      Sym* s = CloneLeafish(arena_, x);
      s->strides = RemapStrides(x->shape, x->strides, out_shape);
      s->shape = out_shape;
      return s;
    }
    default:
      MslDecline(absl::StrCat("reshape of ", KindName(x)));
  }
}

Sym* MslAnalyzer::Broadcasted(Sym* x, const MslShape& out_shape,
                              const std::vector<int64_t>& dims) {
  for (size_t i = 1; i < dims.size(); i++)
    if (dims[i] < dims[i - 1]) MslDecline("unsorted broadcast dims");
  switch (x->kind) {
    case SymKind::kConst: {
      ConstVal v{x->dval, x->ival, x->uns};
      return MakeConst(arena_, v, x->dtype, out_shape);
    }
    case SymKind::kElem: {
      // Broadcast commutes with elementwise: push down to the leaves.
      std::vector<Sym*> args;
      for (Sym* a : x->args) {
        bool all_one = true;
        for (int64_t d : a->shape) all_one = all_one && d == 1;
        if (a->kind == SymKind::kConst || a->kind == SymKind::kCounter ||
            a->shape.empty() || all_one) {
          args.push_back(a);
          continue;
        }
        std::vector<int64_t> adims(dims.begin() + (x->shape.size() -
                                                   a->shape.size()),
                                   dims.end());
        args.push_back(Broadcasted(a, out_shape, adims));
      }
      return MakeElem(arena_, x->op, args, x->dtype, out_shape, x->extra);
    }
    case SymKind::kDot: {
      // Only pure dim-insertion (sizes preserved) is representable.
      for (size_t i = 0; i < dims.size(); i++)
        if (x->shape[i] != out_shape[dims[i]])
          MslDecline("size-changing broadcast of dot");
      MslRole one;
      one.kind = RoleKind::kOne;
      std::vector<MslRole> roles(out_shape.size(), one);
      for (size_t i = 0; i < dims.size(); i++) roles[dims[i]] = x->roles[i];
      return MakeDot(arena_, x->data, x->weight, roles, x->widx, x->csize,
                     x->dsize, x->dtype, out_shape);
    }
    case SymKind::kRedReg:
      // Width-1 value: broadcasting is free at emission.
      return MakeRedReg(arena_, x->inner, out_shape, x->dtype);
    case SymKind::kIota:
      return MakeIota(arena_, dims[x->axis], x->start, out_shape, x->dtype);
    case SymKind::kStack: {
      bool same = true;
      for (size_t i = 0; i < dims.size(); i++)
        same = same && x->shape[i] == out_shape[dims[i]];
      if (same) {
        Sym* base = Broadcasted(x->sbase, out_shape, dims);
        return MakeStack(arena_, base, x->parts, out_shape, x->dtype,
                         dims[x->axis]);
      }
      MslDecline("size-changing broadcast of a stack");
    }
    case SymKind::kPad: {
      // The pad window lives on the last dim; a broadcast that keeps the last
      // dim mapped last commutes with it.
      if (!dims.empty() &&
          dims.back() == static_cast<int64_t>(out_shape.size()) - 1 &&
          x->shape.back() == out_shape.back()) {
        MslShape want(out_shape.begin(), out_shape.end() - 1);
        want.push_back(x->inner->shape.back());
        Sym* inner = Broadcasted(x->inner, want, dims);
        return MakePad(arena_, inner, x->lo, x->n, out_shape);
      }
      MslDecline("broadcast reshaping a pad window");
    }
    case SymKind::kAccDot:
    case SymKind::kAccRed: {
      // Pure unit-dim insertion commutes with the post-kernel accumulation.
      bool ok = true;
      for (size_t i = 0; i < dims.size(); i++)
        ok = ok && x->shape[i] == out_shape[dims[i]];
      std::set<int64_t> mapped(dims.begin(), dims.end());
      for (size_t d = 0; d < out_shape.size(); d++)
        if (!mapped.count(static_cast<int64_t>(d)))
          ok = ok && out_shape[d] == 1;
      if (ok) return Reshaped(x, out_shape);
      MslDecline(absl::StrCat("size-changing broadcast of accumulator ",
                              KindName(x), " ", Join(x->shape), "->",
                              Join(out_shape), " dims=",
                              absl::StrJoin(dims, ",")));
    }
    case SymKind::kLeaf: {
      Sym* s = CloneLeafish(arena_, x);
      MslShape new_strides(out_shape.size(), 0);
      for (size_t i = 0; i < dims.size(); i++) {
        if (x->shape[i] != 1) {
          if (x->shape[i] != out_shape[dims[i]])
            MslDecline("broadcast size mismatch");
          new_strides[dims[i]] = x->strides[i];
        }
      }
      s->shape = out_shape;
      s->strides = new_strides;
      return s;
    }
    default:
      MslDecline(absl::StrCat("broadcast of ", KindName(x), " ",
                              Join(x->shape), "->", Join(out_shape),
                              " dims=", absl::StrJoin(dims, ",")));
  }
}

std::optional<int64_t> MslAnalyzer::StaticIndex(const Sym* sym) {
  if (sym->kind == SymKind::kConst) return sym->ival;
  if (sym->kind == SymKind::kCounter && sym->a == 0) return sym->b;
  return std::nullopt;
}

Sym* MslAnalyzer::DynamicSlice(mlir::Operation* o,
                               const std::vector<Sym*>& ins) {
  Sym* x = ins[0];
  const std::vector<Sym*> starts(ins.begin() + 1, ins.end());
  const std::vector<int64_t> sizes = I64List(o, "slice_sizes");
  std::vector<std::optional<int64_t>> stat;
  bool all_static = true;
  for (const Sym* s : starts) {
    stat.push_back(StaticIndex(s));
    all_static = all_static && stat.back().has_value();
  }
  if (all_static) {
    // All-static dynamic_slice is a plain slice (nested-loop unrolling turns
    // inner-counter indices into constants).
    std::vector<int64_t> st;
    for (const auto& s : stat) st.push_back(*s);
    if (x->kind == SymKind::kStack) {
      const int64_t ax = x->axis;
      bool full_rest = true;
      for (size_t d = 0; d < x->shape.size(); d++) {
        if (static_cast<int64_t>(d) == ax)
          full_rest = full_rest && sizes[d] == 1;
        else
          full_rest = full_rest && st[d] == 0 && sizes[d] == x->shape[d];
      }
      auto it = x->parts.find(st[ax]);
      if (full_rest && it != x->parts.end()) {
        Sym* part = it->second;
        const MslShape want(sizes.begin(), sizes.end());
        if (part->shape != want) part = Reshaped(part, want);
        return part;
      }
      x = x->sbase;
    }
    return Sliced(x, st, MslShape(sizes.begin(), sizes.end()));
  }
  if (!(x->kind == SymKind::kLeaf &&
        (x->leaf == LeafKind::kArg || x->leaf == LeafKind::kWhole)))
    MslDecline(absl::StrCat("dynamic_slice of computed value: ", KindName(x),
                            " ", Dump(x)));
  const MslShape& shape = x->shape;
  if (shape.empty() || sizes[0] != 1) MslDecline("dynamic_slice shape");
  for (size_t d = 1; d < shape.size(); d++) {
    if (sizes[d] != shape[d]) MslDecline("partial dynamic_slice");
    const Sym* s = starts[d];
    const bool zero_const = s->kind == SymKind::kConst && s->ival == 0;
    const bool zero_counter =
        s->kind == SymKind::kCounter && s->a == 0 && s->b == 0;
    if (!zero_const && !zero_counter) MslDecline("nonzero inner start");
  }
  Sym* idx = starts[0];
  if (idx->kind == SymKind::kConst)
    idx = MakeCounter(arena_, 0, idx->ival, -1);
  if (idx->kind != SymKind::kCounter) MslDecline("non-affine slice index");
  MslShape leaf_shape{1};
  leaf_shape.insert(leaf_shape.end(), shape.begin() + 1, shape.end());
  Sym* leaf = MakeLeaf(arena_, LeafKind::kRead, leaf_shape, x->dtype, x->source);
  leaf->idx = idx;
  leaf->inner_shape = MslShape(shape.begin() + 1, shape.end());
  leaf->has_inner = true;
  return leaf;
}

Sym* MslAnalyzer::DynamicUpdate(mlir::Operation* o,
                                const std::vector<Sym*>& ins) {
  Sym* x = ins[0];
  Sym* upd = ins[1];
  const std::vector<Sym*> starts(ins.begin() + 2, ins.end());
  std::vector<std::optional<int64_t>> stat;
  bool all_static = true;
  for (const Sym* s : starts) {
    stat.push_back(StaticIndex(s));
    all_static = all_static && stat.back().has_value();
  }
  bool rest_zero = true;
  for (size_t i = 1; i < stat.size(); i++)
    rest_zero = rest_zero && stat[i].has_value() && *stat[i] == 0;
  auto tail = [](const MslShape& s) {
    return s.empty() ? MslShape{} : MslShape(s.begin() + 1, s.end());
  };
  const bool upd_block = !upd->shape.empty() && upd->shape[0] == 1 &&
                         tail(upd->shape) == tail(x->shape);
  if (all_static && rest_zero &&
      (x->kind == SymKind::kConst || x->kind == SymKind::kStack) && upd_block) {
    std::map<int64_t, Sym*> parts;
    Sym* base = x;
    if (x->kind == SymKind::kStack) {
      parts = x->parts;
      base = x->sbase;
    }
    parts[*stat[0]] = upd;
    return MakeStack(arena_, base, parts, x->shape, x->dtype);
  }
  if (!(x->kind == SymKind::kLeaf && x->leaf == LeafKind::kArg &&
        x->source.kind == SrcKind::kCarry))
    MslDecline(absl::StrCat("dynamic_update_slice target: ", KindName(x), " ",
                            Dump(x)));
  const MslShape& shape = x->shape;
  if (upd->shape.empty() || upd->shape[0] != 1 ||
      tail(upd->shape) != tail(shape))
    MslDecline("dus update shape");
  for (size_t i = 1; i < starts.size(); i++) {
    const Sym* s = starts[i];
    const bool zero_const = s->kind == SymKind::kConst && s->ival == 0;
    const bool zero_counter =
        s->kind == SymKind::kCounter && s->a == 0 && s->b == 0;
    if (!zero_const && !zero_counter) MslDecline("dus inner start");
  }
  Sym* idx = starts[0];
  if (idx->kind == SymKind::kConst)
    idx = MakeCounter(arena_, 0, idx->ival, -1);
  if (idx->kind != SymKind::kCounter) MslDecline("dus non-affine index");
  Sym* out = MakeLeaf(arena_, LeafKind::kUpdated, shape, x->dtype, x->source);
  out->idx = idx;
  updates[out] = {idx, upd};
  return out;
}

// ---------------------------------------------------------- canonicalizer
//
// msl_scan.py `_canonicalizer`: structural hash-consing over the Sym DAG.
// jax's AD-generated loops recompute gate subexpressions once per residual, so
// identical subtrees arrive as distinct ops; interning by structure collapses
// them so each dot/activation is emitted once, and the emitters' memoization
// by pointer then works structurally.

class Canonicalizer {
 public:
  Sym* operator()(Sym* s) {
    if (s == nullptr) return nullptr;
    auto hit = memo_.find(s);
    if (hit != memo_.end()) return hit->second;
    const std::string key = Key(s);
    auto it = table_.find(key);
    Sym* out = nullptr;
    if (it != table_.end()) {
      out = it->second;
    } else {
      table_[key] = s;
      out = s;
    }
    memo_[s] = out;
    return out;
  }

 private:
  std::string Key(Sym* s) {
    auto ids = [](const std::vector<Sym*>& v) {
      std::string out;
      for (const Sym* x : v)
        absl::StrAppend(&out, reinterpret_cast<uintptr_t>(x), ",");
      return out;
    };
    auto roles = [](const std::vector<MslRole>& v) {
      std::string out;
      for (const MslRole& r : v)
        absl::StrAppend(&out, static_cast<int>(r.kind), ":", r.idx, ",");
      return out;
    };
    switch (s->kind) {
      case SymKind::kElem: {
        for (Sym*& a : s->args) a = (*this)(a);
        return absl::StrCat("e|", s->op, "|", s->extra, "|", s->dtype, "|",
                            Join(s->shape), "|", ids(s->args));
      }
      case SymKind::kPad:
        s->inner = (*this)(s->inner);
        return absl::StrCat("pad|", reinterpret_cast<uintptr_t>(s->inner), "|",
                            s->lo, "|", s->n, "|", Join(s->shape));
      case SymKind::kPerm:
        s->inner = (*this)(s->inner);
        return absl::StrCat("perm|", reinterpret_cast<uintptr_t>(s->inner), "|",
                            absl::StrJoin(s->perm, ","));
      case SymKind::kDot:
        s->data = (*this)(s->data);
        s->weight = (*this)(s->weight);
        return absl::StrCat("dot|", reinterpret_cast<uintptr_t>(s->data), "|",
                            reinterpret_cast<uintptr_t>(s->weight), "|",
                            roles(s->roles), "|", roles(s->widx), "|",
                            Join(s->shape));
      case SymKind::kAccDot: {
        s->lhs = (*this)(s->lhs);
        s->rhs = (*this)(s->rhs);
        std::string d;
        for (const auto& v : s->dims4) absl::StrAppend(&d, Join(v), ";");
        return absl::StrCat("accdot|", reinterpret_cast<uintptr_t>(s->lhs), "|",
                            reinterpret_cast<uintptr_t>(s->rhs), "|", d, "|",
                            absl::StrJoin(s->perm, ","), "|", Join(s->shape));
      }
      case SymKind::kAccRed:
        s->inner = (*this)(s->inner);
        return absl::StrCat("accred|", reinterpret_cast<uintptr_t>(s->inner),
                            "|", Join(s->dims), "|",
                            absl::StrJoin(s->perm, ","), "|", Join(s->shape));
      case SymKind::kRedReg:
        s->inner = (*this)(s->inner);
        return absl::StrCat("redreg|", reinterpret_cast<uintptr_t>(s->inner),
                            "|", Join(s->shape));
      case SymKind::kIota:
        return absl::StrCat("iota|", s->axis, "|", s->start, "|",
                            Join(s->shape), "|", s->dtype);
      case SymKind::kStack: {
        s->sbase = (*this)(s->sbase);
        std::string p;
        for (auto& kv : s->parts) kv.second = (*this)(kv.second);
        for (const auto& kv : s->parts)
          absl::StrAppend(&p, kv.first, ":",
                          reinterpret_cast<uintptr_t>(kv.second), ",");
        return absl::StrCat("stack|", reinterpret_cast<uintptr_t>(s->sbase),
                            "|", s->axis, "|", p, "|", Join(s->shape));
      }
      case SymKind::kConst:
        return absl::StrCat("c|", s->dtype, "|", Join(s->shape), "|",
                            absl::StrFormat("%a", s->dval), "|", s->ival);
      case SymKind::kCounter:
        return absl::StrCat("t|", s->a, "|", s->b, "|", s->base);
      case SymKind::kLeaf: {
        if (s->idx != nullptr) s->idx = (*this)(s->idx);
        return absl::StrCat(
            "l|", static_cast<int>(s->leaf), "|",
            static_cast<int>(s->source.kind), "|", s->source.carry, "|",
            reinterpret_cast<uintptr_t>(s->source.value.getAsOpaquePointer()),
            "|", reinterpret_cast<uintptr_t>(s->idx), "|", Join(s->shape), "|",
            Join(s->strides), "|", s->offset);
      }
    }
    return absl::StrCat("?|", reinterpret_cast<uintptr_t>(s));
  }

  std::map<std::string, Sym*> table_;
  absl::flat_hash_map<Sym*, Sym*> memo_;
};

}  // namespace

// The two op tables, shared with the emitters (metal_msl_emit.cc).
const std::map<std::string, std::string>& MslUnaryOps() {
  static const auto* m = new std::map<std::string, std::string>{
      {"stablehlo.negate", "(-{0})"},
      {"stablehlo.abs", "metal::abs({0})"},
      {"stablehlo.exponential", "metal::precise::exp({0})"},
      {"stablehlo.log", "metal::precise::log({0})"},
      {"stablehlo.log_plus_one", "metal::precise::log(1.0f + {0})"},
      {"stablehlo.exponential_minus_one", "mj_expm1({0})"},
      {"stablehlo.tanh", "metal::precise::tanh({0})"},
      {"stablehlo.logistic", "(1.0f / (1.0f + metal::precise::exp(-({0}))))"},
      {"stablehlo.sqrt", "metal::precise::sqrt({0})"},
      {"stablehlo.rsqrt", "metal::precise::rsqrt({0})"},
      {"stablehlo.floor", "metal::floor({0})"},
      {"stablehlo.ceil", "metal::ceil({0})"},
      {"stablehlo.sign", "metal::sign({0})"},
      {"stablehlo.cosine", "metal::precise::cos({0})"},
      {"stablehlo.sine", "metal::precise::sin({0})"},
      {"stablehlo.not", "(!({0}))"},
  };
  return *m;
}

const std::map<std::string, std::string>& MslBinaryOps() {
  static const auto* m = new std::map<std::string, std::string>{
      {"stablehlo.add", "({0} + {1})"},
      {"stablehlo.subtract", "({0} - {1})"},
      {"stablehlo.multiply", "({0} * {1})"},
      {"stablehlo.divide", "({0} / {1})"},
      {"stablehlo.maximum", "metal::max({0}, {1})"},
      {"stablehlo.minimum", "metal::min({0}, {1})"},
      {"stablehlo.power", "metal::precise::pow({0}, {1})"},
      {"stablehlo.remainder", "metal::fmod({0}, {1})"},
      {"stablehlo.and", "({0} && {1})"},
      {"stablehlo.or", "({0} || {1})"},
      {"stablehlo.xor", "({0} != {1})"},
  };
  return *m;
}

const std::map<std::string, std::string>& MslCompareOps() {
  static const auto* m = new std::map<std::string, std::string>{
      {"EQ", "=="}, {"NE", "!="}, {"LT", "<"},
      {"LE", "<="}, {"GT", ">"},  {"GE", ">="}};
  return *m;
}

const std::string& MslTypeName(const std::string& dtype) {
  auto it = MslDtypes().find(dtype);
  if (it == MslDtypes().end()) MslDecline(absl::StrCat("dtype ", dtype));
  return it->second;
}

const std::string& MslTDecl() { return Flags().t_decl; }

// msl_scan.py `_vol_load`: a load the shader compiler must not cache across
// loop iterations.
std::string MslVolLoad(const std::string& type, const std::string& buf,
                       const std::string& off) {
  if (Flags().volatile_mode == "load")
    return absl::StrCat("(*(volatile device const ", type, "*)&", buf, "[", off,
                        "])");
  return absl::StrCat(buf, "[", off, "]");
}

int64_t MslRegLimit() { return Flags().reg_limit; }
int64_t MslNumel(const MslShape& s) { return Numel(s); }
MslShape MslRowmajor(const MslShape& shape) { return Rowmajor(shape); }
std::string MslDump(const Sym* s) { return Dump(s); }
std::vector<Sym*> MslCollectDots(const std::vector<Sym*>& roots) {
  return CollectDots(roots);
}
bool MslDebug() { return Flags().debug; }
bool MslEnabled() { return Flags().enabled; }
bool MslVolatileIsTmap() { return Flags().volatile_mode == "tmap"; }

// msl_scan.py `_literal`.
std::string MslLiteral(const Sym* s) {
  if (s->dtype == "i1") return s->ival != 0 ? "true" : "false";
  const bool is_int = s->dtype == "int" ||
                      (!s->dtype.empty() &&
                       (s->dtype[0] == 'i' ||
                        (s->dtype[0] == 'u' && s->dtype.size() > 1 &&
                         s->dtype[1] == 'i')) &&
                       s->dtype != "i1");
  if (is_int) {
    if (s->uns && s->ival < 0)
      return absl::StrCat(static_cast<uint64_t>(s->ival), "ull");
    return absl::StrCat(s->ival);
  }
  const double v = s->dval;
  if (v != v)
    return s->dtype == "bf16" ? "((bfloat16_t)NAN)" : "NAN";
  if (v > 0 && std::isinf(v))
    return s->dtype == "bf16" ? "((bfloat16_t)INFINITY)" : "INFINITY";
  if (v < 0 && std::isinf(v))
    return s->dtype == "bf16" ? "((bfloat16_t)(-INFINITY))" : "(-INFINITY)";
  // Python's `repr(float)` -- the shortest decimal that round-trips -- with a
  // decimal point forced, since `1f` is not a float literal in MSL.
  std::string out = absl::StrFormat("%.17g", v);
  for (int prec = 1; prec < 17; prec++) {
    const std::string cand = absl::StrFormat("%.*g", prec, v);
    if (std::strtod(cand.c_str(), nullptr) == v) {
      out = cand;
      break;
    }
  }
  if (out.find('.') == std::string::npos &&
      out.find('e') == std::string::npos &&
      out.find("inf") == std::string::npos &&
      out.find("nan") == std::string::npos)
    absl::StrAppend(&out, ".0");
  // A bf16 constant is emitted AS bf16: a bare float literal makes a call
  // like metal::max(x, 0.5f) ambiguous (the bfloat, float and half overloads
  // are all one implicit conversion away), and typing the literal is also
  // the HLO's own arithmetic -- a bf16 op computes on bf16 values.  f32 and
  // f16 literals keep Stage 1's exact spelling.
  if (s->dtype == "bf16")
    return absl::StrCat("((bfloat16_t)", out, "f)");
  return absl::StrCat(out, "f");
}

// msl_scan.py `_off_strided`: lane-offset expression from explicit element
// strides (scalar mode).
std::string MslOffStrided(const MslShape& shape, const MslShape& strides,
                          const MslShape& lane, int64_t base) {
  MslShape dims(shape), sts(strides);
  while (dims.size() < lane.size()) {
    dims.insert(dims.begin(), 1);
    sts.insert(sts.begin(), 0);
  }
  if (dims.size() > lane.size()) {
    const size_t extra = dims.size() - lane.size();
    for (size_t i = 0; i < extra; i++)
      if (dims[i] != 1)
        MslDecline(absl::StrCat("shape ", Join(shape), " vs lane ", Join(lane)));
    sts.erase(sts.begin(), sts.begin() + extra);
    dims.erase(dims.begin(), dims.begin() + extra);
  }
  std::vector<std::string> terms;
  if (base != 0) terms.push_back(absl::StrCat(base, "u"));
  for (size_t i = 0; i < dims.size(); i++) {
    if (dims[i] == 1 || sts[i] == 0) continue;
    if (dims[i] != lane[i])
      MslDecline(absl::StrCat("shape ", Join(shape), " vs lane ", Join(lane)));
    terms.push_back(absl::StrCat("c", i, " * ", sts[i], "u"));
  }
  return terms.empty() ? "0u" : absl::StrJoin(terms, " + ");
}

// msl_scan.py `_lane_offset`: a tensor's flat offset given lane coords.
std::string MslLaneOffset(const MslShape& shape, const MslShape& lane) {
  MslShape rs(shape);
  while (rs.size() < lane.size()) rs.insert(rs.begin(), 1);
  if (rs.size() > lane.size()) {
    const size_t extra = rs.size() - lane.size();
    for (size_t i = 0; i < extra; i++)
      if (rs[i] != 1)
        MslDecline(absl::StrCat("shape ", Join(shape), " vs lane ", Join(lane)));
    rs.erase(rs.begin(), rs.begin() + extra);
  }
  MslShape strides(rs.size(), 0);
  int64_t acc = 1;
  for (size_t i = rs.size(); i-- > 0;) {
    strides[i] = rs[i] != 1 ? acc : 0;
    acc *= rs[i];
  }
  for (size_t i = 0; i < rs.size(); i++)
    if (rs[i] != 1 && rs[i] != lane[i])
      MslDecline(absl::StrCat("shape ", Join(shape), " vs lane ", Join(lane)));
  std::vector<std::string> terms;
  for (size_t i = 0; i < strides.size(); i++)
    if (strides[i] != 0)
      terms.push_back(absl::StrCat("c", i, " * ", strides[i], "u"));
  return terms.empty() ? "0u" : absl::StrJoin(terms, " + ");
}

// msl_scan.py `_aliased_state_moves`: state slots that must be snapshotted
// before the per-iteration parallel assignment.  An update that IS verbatim
// another state register (a rotating carry list) would otherwise read a slot
// an earlier assignment has already overwritten.
std::vector<int64_t> MslAliasedStateMoves(
    const std::vector<std::string>& srcs) {
  std::map<std::string, int64_t> live;
  for (size_t k = 0; k < srcs.size(); k++)
    live[absl::StrCat("st", k)] = static_cast<int64_t>(k);
  std::set<int64_t> out;
  for (size_t j = 0; j < srcs.size(); j++) {
    auto it = live.find(srcs[j]);
    if (it != live.end() && it->second != static_cast<int64_t>(j))
      out.insert(it->second);
  }
  return std::vector<int64_t>(out.begin(), out.end());
}

// msl_scan.py `_reg_src`: the read expression for a value held in `R`
// registers when the write loop runs over a target of `tgt_R`.
std::string MslRegSrc(const std::string& name, int64_t R, int64_t tgt_R,
                      const std::string& what) {
  if (R > 1 && R != tgt_R)
    MslDecline(absl::StrCat(what, ": value holds ", R, " registers, target "
                            "holds ", tgt_R,
                            " (register axis mapped onto a lane dim)"));
  return R > 1 ? absl::StrCat(name, "[r]") : name;
}

// ================================================================ the plan

MslPlanned::MslPlanned(const MslEnv& env, mlir::Block& body,
                       int64_t counter_pos, int64_t trip_in, int64_t start_in,
                       bool no_coop_flip, bool no_inlane_dot, Fired* fired) {
  env_ = &env;
  trip = trip_in;
  start = start_in;
  const size_t nargs = body.getNumArguments();

  MslAnalyzer an(env, arena_, body, counter_pos, no_inlane_dot, fired);
  std::vector<Sym*> rets = an.Analyze();
  if (rets.size() != nargs) MslDecline("terminator mismatch");

  // Re-key stacked-update info by carry position, then intern the DAG.
  std::map<int64_t, std::pair<Sym*, Sym*>> upd;
  for (size_t i = 0; i < rets.size(); i++) {
    Sym* s = rets[i];
    if (s->kind == SymKind::kLeaf && s->leaf == LeafKind::kUpdated) {
      auto it = an.updates.find(s);
      if (it != an.updates.end())
        upd[static_cast<int64_t>(i)] = it->second;
    }
  }
  Canonicalizer canon;
  for (Sym*& s : rets) s = canon(s);
  for (auto& kv : upd) {
    kv.second.first = canon(kv.second.first);
    kv.second.second = canon(kv.second.second);
  }

  std::map<int64_t, int> state_pos_to_id;
  for (size_t i = 0; i < rets.size(); i++) {
    const int64_t pos = static_cast<int64_t>(i);
    Sym* s = rets[i];
    if (s->kind == SymKind::kLeaf && s->leaf == LeafKind::kArg &&
        s->source.kind == SrcKind::kCarry && s->source.carry == pos) {
      passthrough.push_back(static_cast<int>(pos));
    } else if (s->kind == SymKind::kCounter) {
      if (s->base != pos || s->a != 1) MslDecline("non-increment counter");
      counters.emplace_back(static_cast<int>(pos), s->b);
    } else if (s->kind == SymKind::kLeaf && s->leaf == LeafKind::kUpdated) {
      if (!(s->source.kind == SrcKind::kCarry && s->source.carry == pos))
        MslDecline("update aliasing");
      Sym* idx = upd[pos].first;
      Sym* val = upd[pos].second;
      if (ContainsAccDot(val))
        MslDecline(absl::StrCat("acc in stacked write: ", Dump(val)));
      stacked_.emplace_back(pos, idx, val);
    } else {
      if (an.counter_seeded.count(pos))
        MslDecline(absl::StrCat("i32 scalar carry ", pos,
                                " with non-affine update: ", KindName(s)));
      std::vector<Sym*> accs;
      if (MatchAccum(s, pos, &accs)) {
        accums_.emplace_back(pos, accs);
        continue;
      }
      if (ContainsAccDot(s))
        MslDecline(absl::StrCat("acc-dot outside accumulator update: ",
                                Dump(s)));
      state_pos_to_id[pos] = static_cast<int>(states_.size());
      states_.emplace_back(pos, s);
    }
  }

  for (mlir::BlockArgument a : body.getArguments()) {
    arg_shapes.push_back(ShapeOf(a));
    arg_dtypes.push_back(Dt(a));
  }

  // Squeeze interior unit dims out of classified values: row-major layout is
  // unchanged by them, and the lane-space unifier expects values shaped
  // (lane..., reg) without stray 1s.
  absl::flat_hash_map<Sym*, Sym*> squeeze_cache;
  auto squeeze_units = [&](Sym* sym) -> Sym* {
    const MslShape& shape = sym->shape;
    bool has_unit = false;
    for (size_t i = 0; i + 1 < shape.size(); i++)
      has_unit = has_unit || shape[i] == 1;
    if (shape.size() <= 1 || !has_unit) return sym;
    auto hit = squeeze_cache.find(sym);
    if (hit != squeeze_cache.end()) return hit->second;
    Sym* orig = sym;
    try {
      for (int64_t ax = static_cast<int64_t>(shape.size()) - 2; ax >= 0; ax--)
        if (sym->shape[ax] == 1) sym = an.DropDim(sym, ax);
    } catch (const MslUnsupported&) {
      // Keep whatever the walk reached, exactly as the Python's `pass` does.
    }
    squeeze_cache[orig] = sym;
    return sym;
  };
  for (auto& kv : states_) kv.second = squeeze_units(kv.second);
  for (auto& t : stacked_) std::get<2>(t) = squeeze_units(std::get<2>(t));

  // Resolve accumulator operands: reads of existing stacks are used directly
  // (with flips for reversed indexing); loop-computed values get hidden
  // per-step stacked outputs.
  std::function<MslAccSpec(Sym*)> resolve = [&](Sym* sym) -> MslAccSpec {
    MslAccSpec out;
    if (sym->kind == SymKind::kLeaf && sym->leaf == LeafKind::kRead &&
        sym->strides == Rowmajor(sym->shape) && sym->idx != nullptr &&
        (sym->idx->a == 1 || sym->idx->a == -1) && sym->offset == 0) {
      MslShape inner{1};
      if (sym->has_inner)
        inner.insert(inner.end(), sym->inner_shape.begin(),
                     sym->inner_shape.end());
      if (!sym->has_inner || Numel(sym->shape) == Numel(inner)) {
        // Only a FULL per-step block can be consumed directly as a buffer
        // slice; offset sub-windows need a hidden stack.
        out.kind = 1;
        out.source = sym->source;
        out.a = sym->idx->a;
        out.b = sym->idx->b;
        out.shape = sym->shape;
        return out;
      }
    }
    if (sym->kind == SymKind::kAccRed) {
      out.kind = 2;
      out.kids.push_back(resolve(sym->inner));
      out.dims = sym->dims;
      out.perm = sym->perm;
      out.shape = sym->shape;
      return out;
    }
    if (sym->kind == SymKind::kAccDot) {
      out.kind = 3;
      out.kids.push_back(resolve(sym->lhs));
      out.kids.push_back(resolve(sym->rhs));
      out.dims4 = sym->dims4;
      out.perm = sym->perm;
      out.lshape = sym->lhs->shape;
      out.rshape = sym->rhs->shape;
      return out;
    }
    if (ContainsAccDot(sym))
      MslDecline(absl::StrCat("acc under elementwise: ", Dump(sym)));
    // f32 or bf16: a hidden per-step stack is DECLARED f32 either way
    // (LowerMslPlan), so a bf16 term is upconverted at the kernel's store
    // and the post-kernel accumulation runs in f32 -- one rounding when the
    // total lands back on the carry (MslPlan::run).
    if (sym->dtype != "f32" && sym->dtype != "bf16")
      MslDecline(absl::StrCat("accumulator operand dtype ", sym->dtype));
    for (size_t hi = 0; hi < hidden_.size(); hi++) {
      if (hidden_[hi].first == sym) {
        out.kind = 0;
        out.hidden = static_cast<int>(hi);
        return out;
      }
    }
    sym = squeeze_units(sym);
    for (size_t hi = 0; hi < hidden_.size(); hi++) {
      if (hidden_[hi].first == sym) {
        out.kind = 0;
        out.hidden = static_cast<int>(hi);
        return out;
      }
    }
    hidden_.emplace_back(sym, absl::StrCat("hid", hidden_.size()));
    out.kind = 0;
    out.hidden = static_cast<int>(hidden_.size()) - 1;
    return out;
  };
  for (const auto& kv : accums_) {
    std::vector<MslAccSpec> specs;
    for (Sym* a : kv.second) specs.push_back(resolve(a));
    acc_plans.emplace_back(static_cast<int>(kv.first), std::move(specs));
  }

  // ---- mode choice
  std::vector<Sym*> exprs0;
  for (const auto& kv : states_) exprs0.push_back(kv.second);
  for (const auto& t : stacked_) exprs0.push_back(std::get<2>(t));
  for (const auto& h : hidden_) exprs0.push_back(h.first);
  const std::vector<Sym*> dots = CollectDots(exprs0);
  const int64_t reg_limit = Flags().reg_limit;
  const int64_t dcap = Flags().no_dsize ? reg_limit : 4 * reg_limit;
  // Vector mode holds both dot operands in registers: one dimension may be
  // wide, but if BOTH are wide the register tails collapse occupancy.
  bool vector_ok = !dots.empty();
  for (const Sym* d : dots)
    vector_ok = vector_ok && std::min(d->csize, d->dsize) <= reg_limit &&
                std::max(d->csize, d->dsize) <= dcap;
  if (vector_ok) {
    mode = "vector";
    // ...unless the dots are square multiples of the state width: then coop is
    // eligible and measured faster at every batch size.
    if (Flags().coop_pref && !no_coop_flip) {
      std::set<int64_t> widths;
      for (const auto& kv : states_) {
        const MslShape& sh = arg_shapes[kv.first];
        if (!sh.empty()) widths.insert(sh.back());
      }
      if (widths.size() == 1) {
        const int64_t f = *widths.begin();
        bool multiples = true;
        for (const Sym* d : dots)
          multiples = multiples && d->csize % f == 0 && d->dsize % f == 0;
        if (f >= Flags().coop_pref_min_f && multiples) {
          if (fired != nullptr) fired->flipped = true;
          mode = "coop";
        }
      }
    }
  } else if (!dots.empty() || !accums_.empty()) {
    mode = "coop";
  } else if (NeedsRegisters(exprs0)) {
    mode = "vector";
  } else {
    mode = "scalar";
  }

  if (mode == "coop") {
    std::set<int64_t> widths;
    for (const auto& kv : states_) {
      const MslShape& sh = arg_shapes[kv.first];
      if (!sh.empty()) widths.insert(sh.back());
    }
    if (widths.size() != 1)
      MslDecline(absl::StrCat("coop: mixed state widths ",
                              absl::StrJoin(widths, ",")));
    F = *widths.begin();
    if (!(16 < F && F <= 1024) && !dots.empty()) {
      if (F > 1024) MslDecline(absl::StrCat("coop: width ", F, " > 1024"));
    }
    int64_t shared_bytes = 0;
    std::set<const Sym*> seen_data;
    for (const Sym* d : dots) {
      if (d->csize % F != 0 || d->dsize % F != 0)
        MslDecline(absl::StrCat("coop: dot ", d->csize, "x", d->dsize,
                                " not a multiple of F=", F));
      if (d->csize > 4096 || d->dsize > 4096)
        MslDecline("coop: dot too large");
      if (seen_data.insert(d->data).second) shared_bytes += 4 * d->csize;
      for (const MslRole& r : d->widx)
        if (r.kind == RoleKind::kData) MslDecline("coop: batched dot");
    }
    if (shared_bytes > 32768)
      MslDecline(absl::StrCat("coop: threadgroup memory ", shared_bytes,
                              "B > 32KB"));
    // Every threadgroup streams each dot's weight matrix from device memory
    // every timestep; beyond ~2M weight elements per step the redundant
    // traffic loses to the compiled matmul path.
    int64_t work = 0;
    std::set<const Sym*> uniq;
    for (const Sym* d : dots)
      if (uniq.insert(d).second) work += d->csize * d->dsize;
    Debug(absl::StrCat("coop dot work/lane/step: ", work));
    if (work > Flags().coop_cap)
      MslDecline(absl::StrCat("coop: dot work ", work, " elems/step > ",
                              Flags().coop_cap, " (matmul path wins)"));
    // ... and the traffic is denominated in the FEATURE width, not in the
    // total: each threadgroup re-streams F elements of every weight row it
    // touches, so a square cell at F=1024 loses to the compiled matmul even
    // when its total work is under the cap.  `rnn.1024` (1.05M elems/step, so
    // admitted by the cap Stage 1 tuned on gru/lstm.1024 at 3.1M and 11.5M) is
    // 1.50x SLOWER as a kernel; measured 2026-08-15, notes/cpp-p22-release.md.
    // This width cap is a deliberate DIVERGENCE from msl_scan.py, which has
    // the work cap only: lowering the work cap instead would take coop away
    // from 22 suite configurations to fix 2, and costs up to 1.73x on the
    // gru/mgru/lstm.512 rows it would hit.  METALJAX_MSL_COOP_MAX_F=0 restores
    // Stage 1's policy exactly.
    // Gated on there BEING a dot: with no weights to re-stream a wide
    // elementwise cell has none of the cost this cap is about.
    if (!dots.empty() && Flags().coop_max_f > 0 && F >= Flags().coop_max_f)
      MslDecline(absl::StrCat("coop: state width F=", F, " >= ",
                              Flags().coop_max_f, " (matmul path wins)"));
  }

  // ---- lane space
  MslShape lane;
  if (mode == "coop") {
    for (const auto& kv : states_) {
      const MslShape& sh = arg_shapes[kv.first];
      lane = Bshape(lane, sh.empty() ? MslShape{}
                                     : MslShape(sh.begin(), sh.end() - 1));
    }
    for (const auto& t : stacked_) {
      const int64_t pos = std::get<0>(t);
      const MslShape& sh = arg_shapes[pos];
      if (sh.empty()) MslDecline("coop: rank-0 stacked output");
      if (sh.back() % F != 0) MslDecline("coop: width not multiple of F");
      const MslShape& vs = std::get<2>(t)->shape;
      lane = Bshape(lane, vs.size() >= 2
                              ? MslShape(vs.begin() + 1, vs.end() - 1)
                              : MslShape{});
    }
    for (const auto& h : hidden_) {
      const MslShape& sh = h.first->shape;
      lane = Bshape(lane, sh.empty() ? MslShape{}
                                     : MslShape(sh.begin(), sh.end() - 1));
    }
    lane.push_back(F);
  } else if (mode == "vector") {
    for (const auto& kv : states_) {
      const MslShape& sh = kv.second->shape;
      lane = Bshape(lane, sh.empty() ? MslShape{}
                                     : MslShape(sh.begin(), sh.end() - 1));
    }
    for (const auto& t : stacked_) {
      const MslShape& vs = std::get<2>(t)->shape;
      lane = Bshape(lane, vs.size() >= 2
                              ? MslShape(vs.begin() + 1, vs.end() - 1)
                              : MslShape{});
    }
    for (const auto& h : hidden_) {
      const MslShape& sh = h.first->shape;
      lane = Bshape(lane, sh.empty() ? MslShape{}
                                     : MslShape(sh.begin(), sh.end() - 1));
    }
  } else {
    for (const auto& kv : states_) lane = Bshape(lane, kv.second->shape);
    for (const auto& t : stacked_) {
      const MslShape& vs = std::get<2>(t)->shape;
      lane = Bshape(lane, vs.empty() ? MslShape{}
                                     : MslShape(vs.begin() + 1, vs.end()));
    }
  }
  lane_shape_ = lane;
  N = Numel(lane);
  if (N == 0) MslDecline("empty lane space");
  if (mode == "vector" && lane.size() == 1 && lane.back() > 1) {
    // A per-step value shaped exactly like the lane space is ONE SCALAR PER
    // LANE.  The emitter reads a value's trailing dim as its register width,
    // so such a value would look like `lane[-1]` registers and every lane
    // would broadcast-write all lanes' output slots (silently wrong results,
    // v0.2.0..v0.4.4).  The DESTINATION layout is what disambiguates.
    for (const auto& t : stacked_) {
      const MslShape& sh = arg_shapes[std::get<0>(t)];
      const MslShape per(sh.begin() + (sh.empty() ? 0 : 1), sh.end());
      if (per == lane && R(std::get<2>(t)) > 1)
        MslDecline(absl::StrCat(
            "stacked output is one scalar per lane (per-step shape ",
            Join(lane), ")"));
    }
  }

  // ---- device inputs
  for (const auto& kv : state_pos_to_id) state_args_[kv.first] = kv.second;
  Collect(exprs0);
  for (const auto& kv : weights_)
    if (wholes_.count(kv.first))
      MslDecline("weights also used elementwise");
  // Reads must come from pass-through carries or free captures -- never from
  // tensors this loop itself mutates.
  std::set<int64_t> mutated;
  for (const auto& t : stacked_) mutated.insert(std::get<0>(t));
  for (const auto& kv : states_) mutated.insert(kv.first);
  for (const auto& kv : accums_) mutated.insert(kv.first);
  for (const MslSrc& src : sources)
    if (src.kind == SrcKind::kCarry && mutated.count(src.carry))
      MslDecline("read of a mutated carry");
  for (int64_t s : an.hoisted_read_srcs)
    if (mutated.count(s)) MslDecline("hoisted read of a mutated carry");

  // Canonicalize weight layout to (data..., c, d): the output-feature dim at
  // unit stride makes threadgroup dot reads coalesced (the AD backward pass
  // contracts against W^T, which would otherwise read stride-F apart across
  // adjacent threads -- ~100x slower in coop mode).
  if (Flags().wnorm) {
    for (auto& kv : weights_) {
      const int sid = kv.first;
      Sym* wleaf = kv.second;
      auto dit = weight_dots_.find(sid);
      if (dit == weight_dots_.end() || dit->second.empty()) continue;
      const std::vector<Sym*>& ds = dit->second;
      const std::vector<MslRole> widx0 = ds[0]->widx;
      bool same = true;
      for (size_t i = 1; i < ds.size(); i++) same = same && ds[i]->widx == widx0;
      if (!same) continue;
      // All dots must read the SAME window of this source: the normalization
      // materializes one transformed buffer per source, which cannot serve two
      // different slices (gate splits of a fused weight index distinct
      // windows).
      const Sym* w0 = ds[0]->weight;
      bool one_window = true;
      for (size_t i = 1; i < ds.size(); i++)
        one_window = one_window && ds[i]->weight->shape == w0->shape &&
                     ds[i]->weight->strides == w0->strides &&
                     ds[i]->weight->offset == w0->offset;
      if (!one_window) continue;
      const size_t nd = wleaf->shape.size();
      std::vector<int64_t> perm;
      for (size_t i = 0; i < nd; i++)
        if (widx0[i].kind == RoleKind::kData) perm.push_back(i);
      for (size_t i = 0; i < nd; i++)
        if (widx0[i].kind == RoleKind::kC) perm.push_back(i);
      for (size_t i = 0; i < nd; i++)
        if (widx0[i].kind == RoleKind::kD) perm.push_back(i);
      bool identity = perm.size() == nd;
      for (size_t i = 0; i < perm.size() && identity; i++)
        identity = perm[i] == static_cast<int64_t>(i);
      if (identity && wleaf->strides == Rowmajor(wleaf->shape) &&
          wleaf->offset == 0)
        continue;
      MslNorm norm;
      norm.shape = wleaf->shape;
      norm.strides = wleaf->strides;
      norm.offset = wleaf->offset;
      norm.perm = perm;
      weight_norms[sid] = norm;
      MslShape new_shape;
      for (int64_t p : perm) new_shape.push_back(wleaf->shape[p]);
      wleaf->shape = new_shape;
      wleaf->strides = Rowmajor(new_shape);
      wleaf->offset = 0;
      std::vector<MslRole> new_widx;
      for (int64_t p : perm) new_widx.push_back(widx0[p]);
      for (Sym* d : ds) d->widx = new_widx;
    }
  }

  // bf16 dot weights fed as a per-call f32 copy (OFF by default -- see the
  // flag): flip the leaf to f32 -- the emitters and the packing pools
  // follow the dtype -- and the launch materializes the copy (bf16->f32 is
  // exact, so outputs are bit-identical).  f32 plans are untouched (the
  // pass only ever sees bf16 leaves).
  if (Flags().bf16_wconv > 0) {
    for (const auto& kv : weight_dots_) {
      const int sid = kv.first;
      if (kv.second.empty() || kv.second[0]->weight->dtype != "bf16") continue;
      // Read elsewhere at its own dtype (a time-indexed read or an
      // elementwise whole): the buffer cannot change type under it.  The
      // wholes_ overlap is already a decline; the reads_ one is caution.
      bool elsewhere = wholes_.count(sid) != 0;
      for (const ReadEntry& r : reads_) elsewhere = elsewhere || r.sid == sid;
      if (elsewhere) continue;
      // Distinct windows of one source are distinct interned leaves; the
      // gate charges their total.
      std::set<const Sym*> leaves;
      int64_t elems = 0;
      for (Sym* d : kv.second)
        if (leaves.insert(d->weight).second)
          elems += MslNumel(d->weight->shape);
      if (elems > Flags().bf16_wconv) continue;
      for (Sym* d : kv.second) d->weight->dtype = "f32";
      wconv.push_back(sid);
      Debug(absl::StrCat("msl bf16 wconv: source ", sid, " (", elems,
                         " elems) converted to f32 per call"));
    }
  }

  // ---- input packing: Metal caps kernels at 31 buffer bindings.  Deep fused
  // bodies (AD residual chains) can need far more inputs than that; pool
  // same-dtype inputs into one buffer per dtype with static element offsets
  // baked into the source.  0-dim inputs stay separate -- MLX passes those by
  // value, not as buffers.
  const int64_t raw_bind =
      static_cast<int64_t>(sources.size()) +
      2 * static_cast<int64_t>(states_.size()) +
      (MslVolatileIsTmap() ? 1 : 0) + static_cast<int64_t>(stacked_.size()) +
      static_cast<int64_t>(hidden_.size());
  if (raw_bind > Flags().pack_trigger) {
    std::map<int, Sym*> per_sid;
    // Any window of a source names the same BUFFER, and the pool slot is
    // sized from the buffer -- so the first entry stands for the source.
    for (const auto& kv : wholes_)
      if (!kv.second.empty()) per_sid.emplace(kv.first, kv.second.front().leaf);
    for (const ReadEntry& r : reads_) per_sid.emplace(r.sid, r.leaf);
    for (const auto& kv : weights_) per_sid.emplace(kv.first, kv.second);
    std::map<std::string, std::vector<std::pair<int, int64_t>>> by_dtype;
    for (const auto& kv : per_sid) {
      const MslShape bshape = BufferShape(kv.second);
      if (bshape.empty()) continue;
      // The slot size must equal what the launch actually concatenates:
      // weight-normalized sources are replaced by the materialized window
      // view, whose numel can differ from the source buffer's.
      auto norm = weight_norms.find(kv.first);
      const int64_t n = norm != weight_norms.end() ? Numel(norm->second.shape)
                                                   : Numel(bshape);
      by_dtype[kv.second->dtype].push_back({kv.first, n});
    }
    for (const auto& kv : by_dtype) {
      if (kv.second.size() < 2) continue;
      MslPackGroup g;
      g.name = absl::StrCat("pk", pack_groups.size());
      g.dtype = kv.first;
      int64_t off = 0;
      for (const auto& member : kv.second) {
        packed_[member.first] = {g.name, off};
        off += member.second;
        g.sids.push_back(member.first);
      }
      pack_groups.push_back(std::move(g));
    }
  }
  num_packed = static_cast<int64_t>(packed_.size());

  std::string src;
  if (mode == "coop") {
    src = EmitCoop();
  } else if (mode == "vector") {
    src = EmitVector();
  } else {
    src = EmitScalar();
  }
  if (ForceBuildFail())
    absl::StrAppend(&src, "\n_metaljax_forced_build_failure;\n");
  source = std::move(src);

  // Metal caps kernel buffer bindings at 31 (indices 0-30); MLX assigns them
  // sequentially to inputs then outputs.  If a plan still exceeds the cap,
  // reject so the loop takes the compiled-graph path instead of dying at
  // kernel build.  One slot for MLX.
  for (int i = 0; i < static_cast<int>(sources.size()); i++)
    if (packed_.count(i) == 0) unpacked.push_back(i);
  const int64_t n_bind = static_cast<int64_t>(unpacked.size()) +
                         static_cast<int64_t>(pack_groups.size()) +
                         2 * static_cast<int64_t>(states_.size()) +
                         (MslVolatileIsTmap() ? 1 : 0) +
                         static_cast<int64_t>(stacked_.size()) +
                         static_cast<int64_t>(hidden_.size());
  if (n_bind > 30)
    MslDecline(absl::StrCat("kernel needs ", n_bind,
                            " buffer bindings (Metal limit 31)"));

  // Kernel names must be process-unique: MLX caches compiled kernels by name.
  // The prefix differs from Stage 1's `mj_scan_` because a plan built here is
  // NOT the one that engine would have built for the same loop -- only the
  // source may share a cache entry, and only if it is identical.
  static std::atomic<int64_t> seq{0};
  kernel_name = absl::StrCat("mjn_scan_", ++seq);

  for (const auto& kv : states_) state_pos.push_back(static_cast<int>(kv.first));
  for (const auto& t : stacked_)
    stacked_pos.push_back(static_cast<int>(std::get<0>(t)));
  for (const auto& h : hidden_) {
    hidden_shapes.push_back(h.first->shape);
    hidden_names.push_back(h.second);
  }
}

// msl_scan.py `Plan._collect`: walk all expressions to collect leaves.
void MslPlanned::Collect(const std::vector<Sym*>& roots) {
  llvm::DenseSet<const Sym*> seen;
  std::function<void(Sym*)> walk = [&](Sym* s) {
    if (s == nullptr || !seen.insert(s).second) return;
    if (s->kind == SymKind::kElem) {
      for (Sym* a : s->args) walk(a);
    } else if (s->kind == SymKind::kPad || s->kind == SymKind::kRedReg ||
               s->kind == SymKind::kPerm || s->kind == SymKind::kAccRed) {
      // kPerm and kStack reach the emitters as the VALUE of a stacked write
      // (`_emit_vector`'s write-target arms), so their operands are emitted
      // and every invariant leaf under them has to be collected here.  While
      // an invariant leaf was keyed by source alone this was invisible: an
      // uncollected leaf simply resolved to whichever window of its source
      // was collected -- the same silent wrongness, one level down.
      walk(s->inner);
    } else if (s->kind == SymKind::kStack) {
      walk(s->sbase);
      for (const auto& kv : s->parts) walk(kv.second);
    } else if (s->kind == SymKind::kAccDot) {
      walk(s->lhs);
      walk(s->rhs);
    } else if (s->kind == SymKind::kDot) {
      walk(s->data);
      const int sid = SourceId(s->weight->source);
      weights_[sid] = s->weight;
      weight_dots_[sid].push_back(s);
    } else if (s->kind == SymKind::kLeaf) {
      if (s->leaf == LeafKind::kRead) {
        const int sid = SourceId(s->source);
        const std::string key =
            absl::StrCat(sid, "|", s->idx->a, "|", s->idx->b, "|",
                         Join(s->shape), "|", Join(s->strides), "|", s->offset);
        if (read_index_.find(key) == read_index_.end()) {
          read_index_[key] = reads_.size();
          reads_.push_back(ReadEntry{key, s, absl::StrCat("r", reads_.size()),
                                     sid, s->idx->a, s->idx->b});
        }
      } else if (s->leaf == LeafKind::kArg) {
        const int64_t pos = s->source.carry;
        if (state_args_.count(pos) == 0) AddWhole(SourceId(s->source), s);
      } else if (s->leaf == LeafKind::kWhole) {
        AddWhole(SourceId(s->source), s);
      } else if (s->leaf == LeafKind::kUpdated) {
        MslDecline("nested update use");
      }
    }
  };
  for (Sym* r : roots) walk(r);
}

namespace {
std::string SourceKeyString(const MslSrc& src) {
  if (src.kind == SrcKind::kCarry) return absl::StrCat("carry|", src.carry);
  const char* k = src.kind == SrcKind::kHoist ? "hoist" : "free";
  return absl::StrCat(
      k, "|", reinterpret_cast<uintptr_t>(src.value.getAsOpaquePointer()));
}
}  // namespace

int MslPlanned::SourceId(const MslSrc& src) {
  const std::string key = SourceKeyString(src);
  auto it = source_ids_.find(key);
  if (it != source_ids_.end()) return it->second;
  const int id = static_cast<int>(sources.size());
  source_ids_[key] = id;
  sources.push_back(src);
  return id;
}

int MslPlanned::SourceKey(const MslSrc& src) const {
  auto it = source_ids_.find(SourceKeyString(src));
  if (it == source_ids_.end()) MslDecline("source not collected");
  return it->second;
}

MslShape MslPlanned::BufferShape(const Sym* leaf) const {
  if (leaf->leaf == LeafKind::kArg) return arg_shapes[leaf->source.carry];
  if (leaf->source.kind == SrcKind::kCarry)
    return arg_shapes[leaf->source.carry];
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(leaf->source.value.getType());
  if (!t) MslDecline("a source that is not a ranked tensor");
  MslShape out;
  for (int64_t d : t.getShape()) out.push_back(d);
  return out;
}

// The source expression for input `sid`: the plain binding name, or a pointer
// into its dtype pool when packed.
std::string MslPlanned::Src(int sid) const {
  auto it = packed_.find(sid);
  if (it == packed_.end()) return absl::StrCat("inp", sid);
  return absl::StrCat("(", it->second.first, " + ", it->second.second, "u)");
}

// What makes two loop-invariant loads the SAME load: one source, one window.
std::string MslPlanned::WindowKey(int sid, const Sym* leaf) {
  return absl::StrCat(sid, "|", Join(leaf->shape), "|", Join(leaf->strides),
                      "|", leaf->offset);
}

// Register an invariant leaf, one entry per distinct window.  The first window
// of a source keeps the plain `w<sid>` name, so every plan that reads each of
// its sources at a single window generates exactly the source it did before.
void MslPlanned::AddWhole(int sid, Sym* leaf) {
  const std::string key = WindowKey(sid, leaf);
  if (whole_index_.find(key) != whole_index_.end()) return;
  std::vector<WholeEntry>& v = wholes_[sid];
  std::string name = v.empty() ? absl::StrCat("w", sid)
                               : absl::StrCat("w", sid, "_", v.size());
  // The census signal for the nested-scan P0: a source read at more than one
  // window is what the source-keyed map used to collapse.  Two windows can
  // differ as KEYS and still address the same elements (unit dims a reshape
  // left behind); only a difference in the addressed elements was wrong
  // before, so the line says which -- "alias" is the wrongness, "benign" is
  // an extra load of the same data.
  if (!v.empty()) {
    auto squeeze = [](const Sym* s) {
      std::string out;
      for (size_t i = 0; i < s->shape.size(); i++)
        if (s->shape[i] != 1)
          absl::StrAppend(&out, s->shape[i], ":",
                          i < s->strides.size() ? s->strides[i] : 0, ",");
      return out;
    };
    const Sym* first = v.front().leaf;
    const bool alias =
        first->offset != leaf->offset || squeeze(first) != squeeze(leaf);
    Debug(absl::StrCat("msl invariant window: source ", sid, " window ",
                       v.size(), " ", alias ? "ALIAS" : "benign", " (shape [",
                       Join(leaf->shape), "] strides [", Join(leaf->strides),
                       "] offset ", leaf->offset, " vs first offset ",
                       first->offset, ")"));
  }
  whole_index_[key] = {sid, v.size()};
  v.push_back(WholeEntry{leaf, std::move(name)});
}

const MslPlanned::WholeEntry& MslPlanned::Whole(const Sym* leaf) const {
  auto it = whole_index_.find(WindowKey(SourceKey(leaf->source), leaf));
  if (it == whole_index_.end()) MslDecline("invariant leaf not collected");
  return wholes_.at(it->second.first)[it->second.second];
}

std::string MslPlanned::ReadName(const Sym* leaf) const {
  const int sid = SourceKey(leaf->source);
  const std::string key = absl::StrCat(sid, "|", leaf->idx->a, "|",
                                       leaf->idx->b, "|", Join(leaf->shape),
                                       "|", Join(leaf->strides), "|",
                                       leaf->offset);
  auto it = read_index_.find(key);
  if (it == read_index_.end()) MslDecline("read not collected");
  return reads_[it->second].name;
}

// =============================================================== entry point

std::shared_ptr<MslPlanned> BuildMslPlan(const MslEnv& env, mlir::Block& body,
                                         int64_t counter_pos, int64_t trip,
                                         int64_t start, std::string* why) {
  if (!MslEnabled() || trip <= 0) {
    if (why != nullptr) *why = "disabled";
    return nullptr;
  }
  // msl_scan.py `build_plan`: if a build using the coop-over-vector preference
  // or the in-lane small-dot rewrite fails deeper in analysis or emission,
  // retry with the heuristics that fired disabled, so neither can lose a plan
  // that used to build.  Any exception counts, not just a decline (an emitter
  // crash too).
  MslPlanned::Fired fired;
  std::string first_error;
  try {
    return std::make_shared<MslPlanned>(env, body, counter_pos, trip, start,
                                        false, false, &fired);
  } catch (const MslUnsupported& e) {
    first_error = e.msg;
  } catch (const std::exception& e) {
    first_error = e.what();
  }
  if (!(fired.flipped || fired.inlane)) {
    if (why != nullptr) *why = first_error;
    return nullptr;
  }
  Debug(absl::StrCat("msl_scan: retry without ",
                     fired.flipped ? "coop-flip " : "",
                     fired.inlane ? "inlane-dot" : "", " (first build: ",
                     first_error, ")"));
  try {
    return std::make_shared<MslPlanned>(env, body, counter_pos, trip, start,
                                        fired.flipped, fired.inlane, nullptr);
  } catch (const MslUnsupported& e) {
    if (why != nullptr) *why = e.msg;
  } catch (const std::exception& e) {
    if (why != nullptr) *why = e.what();
  }
  return nullptr;
}

}  // namespace metaljax
