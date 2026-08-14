/* metaljax: msl_scan's three emitters, natively (src/metaljax/msl_scan.py).

`Plan._emit` (scalar/affine), `Plan._emit_vector` and `Plan._emit_coop`, line
for line.  Everything they generate is a string, so the transliteration is
readable against the Python side by side -- which is the point: a generated
kernel that differs by an index expression is silent wrongness, and the only
defence is that the two engines emit the same shader.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "metal_msl.h"

namespace metaljax {
namespace {

// One emitted value: the MSL name holding it, and how many registers wide it
// is (`R == 1` is a scalar, `R > 1` an array indexed by the loop var `r`).
using EVal = std::pair<std::string, int64_t>;

std::string Join(const MslShape& s) { return absl::StrJoin(s, ","); }

// `shape[1:]` -- the per-step block of a stacked carry.
MslShape Tail1(const MslShape& s) {
  return s.empty() ? MslShape{} : MslShape(s.begin() + 1, s.end());
}

const std::string& T(const std::string& dtype) { return MslTypeName(dtype); }

std::string Declare(const std::string& name, const std::string& dtype,
                    int64_t R) {
  return R > 1 ? absl::StrCat(T(dtype), " ", name, "[", R, "];")
               : absl::StrCat(T(dtype), " ", name, ";");
}

void Load(std::vector<std::string>* out, const std::string& dst,
          const std::string& dtype, int64_t R, const std::string& buf,
          const std::string& off, const std::string& indent = "",
          bool vol = false) {
  const std::string rd = vol ? MslVolLoad(T(dtype), buf, off)
                             : absl::StrCat(buf, "[", off, "]");
  if (R > 1) {
    out->push_back(absl::StrCat(indent, "for (int r = 0; r < ", R,
                                "; r++) ", dst, "[r] = ", rd, ";"));
  } else {
    out->push_back(absl::StrCat(indent, dst, " = ", rd, ";"));
  }
}

std::string Scalarize(const EVal& v) {
  return v.second > 1 ? absl::StrCat(v.first, "[r]") : v.first;
}

// The one place the op tables turn into an expression: `_UNARY`/`_BINARY`'s
// "{0}"/"{1}" placeholders, filled.
std::string Format(const std::string& pattern,
                   const std::vector<std::string>& args) {
  std::string out = pattern;
  for (size_t i = 0; i < args.size(); i++)
    out = absl::StrReplaceAll(out, {{absl::StrCat("{", i, "}"), args[i]}});
  return out;
}

// The elementwise arms every emitter shares (`convert`, `compare`, `select`,
// `clamp` and the two tables), given already-emitted operand expressions.
std::string ElemExpr(const Sym* s, const std::vector<std::string>& acc,
                     const char* where) {
  if (s->op == "convert")
    return absl::StrCat("((", T(s->extra), ")(", acc[0], "))");
  if (s->op == "compare") {
    auto it = MslCompareOps().find(s->extra);
    if (it == MslCompareOps().end()) MslDecline("compare direction");
    return absl::StrCat("(", acc[0], " ", it->second, " ", acc[1], ")");
  }
  if (s->op == "select")
    return absl::StrCat("(", acc[0], " ? ", acc[1], " : ", acc[2], ")");
  if (s->op == "clamp")
    return absl::StrCat("metal::min(metal::max(", acc[1], ", ", acc[0], "), ",
                        acc[2], ")");
  auto u = MslUnaryOps().find(s->op);
  if (u != MslUnaryOps().end()) return Format(u->second, acc);
  auto b = MslBinaryOps().find(s->op);
  if (b != MslBinaryOps().end()) return Format(b->second, acc);
  MslDecline(absl::StrCat(where, " ", s->op));
}

}  // namespace

// ------------------------------------------------------------ register widths

int64_t MslPlanned::R(const Sym* s) const {
  if (s->kind == SymKind::kConst || s->kind == SymKind::kCounter) return 1;
  if (s->kind == SymKind::kDot) return s->dsize;
  if (s->kind == SymKind::kPad) return s->shape.back();
  return s->shape.empty() ? 1 : s->shape.back();
}

int64_t MslPlanned::CoopR(const Sym* s) const {
  if (s->kind == SymKind::kConst || s->kind == SymKind::kCounter) return 1;
  if (s->kind == SymKind::kDot) return s->dsize / F;   // output chunks/thread
  const int64_t w = s->kind == SymKind::kPad
                        ? s->shape.back()
                        : (s->shape.empty() ? 1 : s->shape.back());
  if (w == 1) return 1;
  if (w % F != 0)
    MslDecline(absl::StrCat("coop width ", w, " not multiple of F=", F));
  return w / F;
}

int64_t MslPlanned::CoopRShape(const MslShape& shape) const {
  const int64_t w = shape.empty() ? 1 : shape.back();
  if (w == 1) return 1;
  if (w % F != 0)
    MslDecline(absl::StrCat("coop width ", w, " not multiple of F=", F));
  return w / F;
}

// ------------------------------------------------------------ lane offsets

// msl_scan.py `_vec_off`: the lane-part offset for a tensor whose last dim is
// the register tail (indexed by `r` when > 1), from explicit element strides.
std::string MslPlanned::VecOff(const MslShape& shape, const MslShape* strides,
                               int64_t base) const {
  const MslShape& lane = lane_shape_;
  const MslShape sts = strides != nullptr ? *strides : MslRowmajor(shape);
  const int64_t reg = shape.empty() ? 1 : shape.back();
  const int64_t reg_stride = shape.empty() ? 0 : sts.back();
  MslShape lane_dims, lane_sts;
  for (size_t i = 0; i + 1 < shape.size(); i++) {
    // Unit dims contribute no offset and may sit anywhere (broadcast
    // lowerings leave interior 1s); drop them before lane alignment.
    if (shape[i] == 1) continue;
    lane_dims.push_back(shape[i]);
    lane_sts.push_back(sts[i]);
  }
  int64_t pad = static_cast<int64_t>(lane.size()) -
                static_cast<int64_t>(lane_dims.size());
  if (pad < 0) {
    const size_t n = lane_dims.size() - lane.size();
    for (size_t i = 0; i < n; i++)
      if (lane_dims[i] != 1)
        MslDecline(absl::StrCat("vec shape ", Join(shape), " vs lane ",
                                Join(lane)));
    lane_dims.erase(lane_dims.begin(), lane_dims.begin() + n);
    lane_sts.erase(lane_sts.begin(), lane_sts.begin() + n);
    pad = 0;
  }
  std::vector<std::string> terms;
  if (base != 0) terms.push_back(absl::StrCat(base, "u"));
  for (size_t i = 0; i < lane_dims.size(); i++) {
    if (lane_dims[i] == 1 || lane_sts[i] == 0) continue;
    if (lane_dims[i] != lane[i + pad])
      MslDecline(absl::StrCat("vec shape ", Join(shape), " vs lane ",
                              Join(lane)));
    terms.push_back(absl::StrCat("c", i + pad, " * ", lane_sts[i], "u"));
  }
  std::string expr = terms.empty() ? "0u" : absl::StrJoin(terms, " + ");
  if (reg > 1 && reg_stride != 0)
    return absl::StrCat(expr, " + r * ", reg_stride, "u");
  return expr;
}

// msl_scan.py `_coop_off`: the offset for a tensor whose feature (last) dim is
// chunked -- thread f owns feature g*F + f for component g (loop var `r`).
std::string MslPlanned::CoopOff(const MslShape& shape, const MslShape* strides,
                                int64_t base) const {
  const MslShape& lane = lane_shape_;
  const std::string fcoord = absl::StrCat("c", lane.size() - 1);
  const MslShape sts = strides != nullptr ? *strides : MslRowmajor(shape);
  const int64_t w = shape.empty() ? 1 : shape.back();
  const int64_t fst = shape.empty() ? 0 : sts.back();
  MslShape lane_dims(shape.begin(),
                     shape.empty() ? shape.end() : shape.end() - 1);
  MslShape lane_sts(sts.begin(), shape.empty() ? sts.end() : sts.end() - 1);
  int64_t pad = static_cast<int64_t>(lane.size()) - 1 -
                static_cast<int64_t>(lane_dims.size());
  if (pad < 0) {
    const size_t n = static_cast<size_t>(-pad);
    for (size_t i = 0; i < n; i++)
      if (lane_dims[i] != 1)
        MslDecline(absl::StrCat("coop shape ", Join(shape), " vs lane ",
                                Join(lane)));
    lane_dims.erase(lane_dims.begin(), lane_dims.begin() + n);
    lane_sts.erase(lane_sts.begin(), lane_sts.begin() + n);
    pad = 0;
  }
  std::vector<std::string> terms;
  if (base != 0) terms.push_back(absl::StrCat(base, "u"));
  for (size_t i = 0; i < lane_dims.size(); i++) {
    if (lane_dims[i] == 1 || lane_sts[i] == 0) continue;
    if (lane_dims[i] != lane[i + pad])
      MslDecline(absl::StrCat("coop shape ", Join(shape), " vs lane ",
                              Join(lane)));
    terms.push_back(absl::StrCat("c", i + pad, " * ", lane_sts[i], "u"));
  }
  const std::string expr = terms.empty() ? "0u" : absl::StrJoin(terms, " + ");
  if (w > 1 && fst != 0) {
    if (w == F) return absl::StrCat(expr, " + ", fcoord, " * ", fst, "u");
    return absl::StrCat(expr, " + (r * ", F, "u + ", fcoord, ") * ", fst, "u");
  }
  return expr;
}

// msl_scan.py `_check_state_view`: reject a use of a state carry whose view is
// not the layout the carry was LOADED with.  A carry lives in registers,
// loaded once from the natural layout of its shape, and every use is emitted
// as that bare register name -- correct only when the use addresses exactly
// the same element per (lane, register).
void MslPlanned::CheckStateView(const Sym* s, int64_t pos, bool coop) const {
  const MslShape& shape = arg_shapes[pos];
  MslShape vshape, vstrides;
  for (size_t i = 0; i < s->shape.size(); i++) {
    // Interior unit dims address nothing; the trailing axis IS the register
    // axis and is compared as-is.
    if (s->shape[i] == 1 && i + 1 < s->shape.size()) continue;
    vshape.push_back(s->shape[i]);
    vstrides.push_back(s->strides[i]);
  }
  bool ok = false;
  try {
    const std::string want =
        coop ? CoopOff(shape, nullptr, 0) : VecOff(shape, nullptr, 0);
    const std::string got = coop ? CoopOff(vshape, &vstrides, s->offset)
                                 : VecOff(vshape, &vstrides, s->offset);
    int64_t reg_st = 1, reg_use = 1;
    if (coop) {
      reg_st = CoopRShape(shape);
      reg_use = CoopRShape(vshape);
    } else {
      reg_st = shape.empty() ? 1 : shape.back();
      reg_use = vshape.empty() ? 1 : vshape.back();
    }
    if (!vstrides.empty() && vstrides.back() == 0)
      reg_use = 1;   // broadcast along the trailing axis
    ok = got == want && reg_use == reg_st;
  } catch (const MslUnsupported&) {
    ok = false;
  }
  if (!ok)
    MslDecline(absl::StrCat("state carry ", pos, " ", Join(shape),
                            " used as ", Join(s->shape), " strides ",
                            Join(s->strides), " offset ", s->offset,
                            ": register axis mapped onto a lane coordinate"));
}

// ============================================================ scalar (affine)

std::string MslPlanned::EmitScalar() {
  std::vector<std::string> L;
  const MslShape& lane = lane_shape_;
  L.push_back("uint lane = thread_position_in_grid.x;");
  L.push_back(absl::StrCat("if (lane >= ", N, "u) return;"));
  int64_t tail = MslNumel(lane);
  for (size_t i = 0; i < lane.size(); i++) {
    tail /= lane[i];
    L.push_back(absl::StrCat("uint c", i, " = (lane / ", tail, "u) % ",
                             lane[i], "u;"));
  }

  // Whole (invariant) tensors: load once.  0-dim inputs arrive by value.
  for (const auto& kv : wholes_) {
    const int sid = kv.first;
    const Sym* leaf = kv.second.first;
    const std::string& name = kv.second.second;
    if (BufferShape(leaf).empty()) {
      L.push_back(absl::StrCat(T(leaf->dtype), " ", name, " = inp", sid, ";"));
    } else {
      const std::string off =
          MslOffStrided(leaf->shape, leaf->strides, lane, leaf->offset);
      L.push_back(absl::StrCat(T(leaf->dtype), " ", name, " = ", Src(sid), "[",
                               off, "];"));
    }
  }
  // State registers.
  for (size_t j = 0; j < states_.size(); j++) {
    const int64_t pos = states_[j].first;
    if (arg_shapes[pos].empty()) {
      L.push_back(absl::StrCat(T(arg_dtypes[pos]), " st", j, " = init", j, ";"));
    } else {
      const std::string off = MslLaneOffset(arg_shapes[pos], lane);
      L.push_back(absl::StrCat(T(arg_dtypes[pos]), " st", j, " = init", j, "[",
                               off, "];"));
    }
  }

  L.push_back(absl::StrCat("for (uint t_ = 0; t_ < ", trip, "u; t_++) {"));
  L.push_back(MslTDecl());

  // Per-iteration reads.
  for (const ReadEntry& r : reads_) {
    const Sym* leaf = r.leaf;
    const int64_t inner = MslNumel(leaf->inner_shape);
    const std::string off =
        MslOffStrided(leaf->shape, leaf->strides, lane, leaf->offset);
    const std::string idx =
        (r.a != 1 || r.b != 0 || start != 0)
            ? absl::StrCat("((int)t + ", start, ") * ", r.a, " + ", r.b)
            : "(int)t";
    L.push_back(absl::StrCat(
        "  ", T(leaf->dtype), " ", r.name, " = ",
        MslVolLoad(T(leaf->dtype), Src(r.sid),
                   absl::StrCat("(uint)(", idx, ") * ", inner, "u + (", off,
                                ")")),
        ";"));
  }

  memo_.clear();
  tmp_ = 0;
  body_.clear();
  std::function<std::string(const Sym*)> emit = [&](const Sym* s) {
    auto hit = memo_.find(s);
    if (hit != memo_.end()) return hit->second.first;
    std::string v;
    if (s->kind == SymKind::kConst) {
      v = MslLiteral(s);
    } else if (s->kind == SymKind::kCounter) {
      v = absl::StrCat("(", s->a, " * ((int)t + ", start, ") + ", s->b, ")");
    } else if (s->kind == SymKind::kLeaf) {
      if (s->leaf == LeafKind::kRead) {
        v = ReadName(s);
      } else if (s->leaf == LeafKind::kArg) {
        const int64_t pos = s->source.carry;
        auto st = state_args_.find(pos);
        if (st != state_args_.end()) {
          v = absl::StrCat("st", st->second);
        } else {
          v = wholes_.at(SourceKey(s->source)).second;
        }
      } else if (s->leaf == LeafKind::kWhole) {
        v = wholes_.at(SourceKey(s->source)).second;
      } else {
        MslDecline("leaf kind");
      }
    } else if (s->kind == SymKind::kElem) {
      std::vector<std::string> args;
      for (const Sym* a : s->args) args.push_back(emit(a));
      const std::string e = ElemExpr(s, args, "emit");
      const std::string name = absl::StrCat("v", tmp_++);
      body_.push_back(absl::StrCat("  ", T(s->dtype), " ", name, " = ", e, ";"));
      v = name;
    } else {
      MslDecline("emit type");
    }
    memo_[s] = {v, 1};
    return v;
  };

  // Stacked writes + state updates (compute all, then assign states).
  std::vector<std::string> writes;
  for (size_t q = 0; q < stacked_.size(); q++) {
    const int64_t pos = std::get<0>(stacked_[q]);
    const Sym* idx = std::get<1>(stacked_[q]);
    const std::string v = emit(std::get<2>(stacked_[q]));
    const MslShape per = Tail1(arg_shapes[pos]);
    const int64_t inner = MslNumel(per);
    const std::string off = MslLaneOffset(per, lane);
    const std::string ii =
        absl::StrCat("((int)t + ", start, ") * ", idx->a, " + ", idx->b);
    writes.push_back(absl::StrCat("  out", q, "[(uint)(", ii, ") * ", inner,
                                  "u + (", off, ")] = ", v, ";"));
  }
  std::vector<std::string> news;
  for (const auto& kv : states_) news.push_back(emit(kv.second));
  L.insert(L.end(), body_.begin(), body_.end());
  L.insert(L.end(), writes.begin(), writes.end());
  const std::vector<int64_t> moved = MslAliasedStateMoves(news);
  for (int64_t k : moved)
    L.push_back(absl::StrCat("  ", T(arg_dtypes[states_[k].first]), " sv", k,
                             " = st", k, ";"));
  std::map<std::string, std::string> ren;
  for (int64_t k : moved) ren[absl::StrCat("st", k)] = absl::StrCat("sv", k);
  for (size_t j = 0; j < news.size(); j++) {
    auto it = ren.find(news[j]);
    L.push_back(absl::StrCat("  st", j, " = ",
                             it == ren.end() ? news[j] : it->second, ";"));
  }
  L.push_back("}");
  for (size_t j = 0; j < states_.size(); j++) {
    const int64_t pos = states_[j].first;
    L.push_back(absl::StrCat("fin", j, "[", MslLaneOffset(arg_shapes[pos], lane),
                             "] = st", j, ";"));
  }
  return absl::StrJoin(L, "\n");
}

// ==================================================== vector (register tails)

std::string MslPlanned::EmitVector() {
  const MslShape& lane = lane_shape_;
  std::vector<std::string> out;
  out.push_back("uint lane = thread_position_in_grid.x;");
  out.push_back(absl::StrCat("if (lane >= ", N, "u) return;"));
  int64_t tail = MslNumel(lane);
  for (size_t i = 0; i < lane.size(); i++) {
    tail /= lane[i];
    out.push_back(absl::StrCat("uint c", i, " = (lane / ", tail, "u) % ",
                               lane[i], "u;"));
  }

  // Invariant tensors (not weights): preload.
  for (const auto& kv : wholes_) {
    const int sid = kv.first;
    const Sym* leaf = kv.second.first;
    const std::string& name = kv.second.second;
    const int64_t r = R(leaf);
    out.push_back(Declare(name, leaf->dtype, r));
    if (BufferShape(leaf).empty()) {
      out.push_back(r == 1 ? absl::StrCat(name, " = inp", sid, ";") : "");
    } else {
      Load(&out, name, leaf->dtype, r, Src(sid),
           VecOff(leaf->shape, &leaf->strides, leaf->offset));
    }
  }
  // States.
  for (size_t j = 0; j < states_.size(); j++) {
    const int64_t pos = states_[j].first;
    const MslShape& shape = arg_shapes[pos];
    const int64_t r = shape.empty() ? 1 : shape.back();
    out.push_back(Declare(absl::StrCat("st", j), arg_dtypes[pos], r));
    if (shape.empty()) {
      out.push_back(absl::StrCat("st", j, " = init", j, ";"));
    } else {
      Load(&out, absl::StrCat("st", j), arg_dtypes[pos], r,
           absl::StrCat("init", j), VecOff(shape, nullptr, 0));
    }
  }

  out.push_back(absl::StrCat("for (uint t_ = 0; t_ < ", trip, "u; t_++) {"));
  out.push_back(MslTDecl());

  // Reads.
  for (const ReadEntry& re : reads_) {
    const Sym* leaf = re.leaf;
    const int64_t r = R(leaf);
    const int64_t inner = MslNumel(leaf->inner_shape);
    const std::string idx =
        (re.a != 1 || re.b != 0 || start != 0)
            ? absl::StrCat("((int)t + ", start, ") * ", re.a, " + ", re.b)
            : "(int)t";
    const std::string off = VecOff(leaf->shape, &leaf->strides, leaf->offset);
    out.push_back(absl::StrCat("  ", Declare(re.name, leaf->dtype, r)));
    Load(&out, re.name, leaf->dtype, r, Src(re.sid),
         absl::StrCat("(uint)(", idx, ") * ", inner, "u + (", off, ")"), "  ",
         true);
  }

  memo_.clear();
  tmp_ = 0;
  body_.clear();
  writes_.clear();

  std::function<EVal(const Sym*)> emit;
  auto emit_dot = [&](const Sym* s) -> EVal {
    // Validate canonical orientation: lane dims ascending, reg last.
    std::vector<MslRole> roles;
    for (const MslRole& r : s->roles)
      if (r.kind != RoleKind::kOne) roles.push_back(r);
    if (roles.empty() || roles.back().kind != RoleKind::kReg)
      MslDecline("dot output reg dim not last");
    std::vector<int64_t> dorder;
    for (size_t i = 0; i + 1 < roles.size(); i++) dorder.push_back(roles[i].idx);
    if (!std::is_sorted(dorder.begin(), dorder.end()))
      MslDecline("dot output lane dims permuted");
    const EVal d = emit(s->data);
    if (d.second != s->csize) MslDecline("dot data register width mismatch");
    const int wsid = SourceKey(s->weight->source);
    const MslShape& wstrides = s->weight->strides;
    const int64_t pad = static_cast<int64_t>(lane_shape_.size()) -
                        (static_cast<int64_t>(s->data->shape.size()) - 1);
    std::vector<std::string> terms;
    for (size_t wdim = 0; wdim < s->widx.size(); wdim++) {
      const int64_t st = wstrides[wdim];
      const MslRole& role = s->widx[wdim];
      if (role.kind == RoleKind::kData) {
        if (s->data->shape[role.idx] == 1) continue;
        terms.push_back(absl::StrCat("c", role.idx + pad, " * ", st, "u"));
      } else if (role.kind == RoleKind::kC) {
        terms.push_back(absl::StrCat("(uint)cc * ", st, "u"));
      } else {
        terms.push_back(absl::StrCat("(uint)d * ", st, "u"));
      }
    }
    if (s->weight->offset != 0)
      terms.insert(terms.begin(), absl::StrCat(s->weight->offset, "u"));
    const std::string name = absl::StrCat("v", tmp_++);
    const std::string dacc =
        d.second > 1 ? absl::StrCat(d.first, "[cc]") : d.first;
    if (s->dsize > 1) {
      const std::string woff =
          terms.empty() ? "0u" : absl::StrJoin(terms, " + ");
      body_.push_back(absl::StrCat("  float ", name, "[", s->dsize, "];"));
      body_.push_back(absl::StrCat(
          "  for (int d = 0; d < ", s->dsize, "; d++) { float _a = 0.0f; ",
          "for (int cc = 0; cc < ", s->csize, "; cc++) _a += ", dacc, " * ",
          Src(wsid), "[", woff, "]; ", name, "[d] = _a; }"));
    } else {
      std::vector<std::string> t0;
      for (const std::string& t : terms)
        if (t.find("(uint)d") == std::string::npos) t0.push_back(t);
      const std::string woff = t0.empty() ? "0u" : absl::StrJoin(t0, " + ");
      body_.push_back(absl::StrCat("  float ", name, ";"));
      body_.push_back(absl::StrCat(
          "  { float _a = 0.0f; for (int cc = 0; cc < ", s->csize,
          "; cc++) _a += ", dacc, " * ", Src(wsid), "[", woff, "]; ", name,
          " = _a; }"));
    }
    return {name, s->dsize};
  };

  emit = [&](const Sym* s) -> EVal {
    auto hit = memo_.find(s);
    if (hit != memo_.end()) return hit->second;
    EVal v;
    switch (s->kind) {
      case SymKind::kConst:
        v = {MslLiteral(s), 1};
        break;
      case SymKind::kCounter:
        v = {absl::StrCat("(", s->a, " * ((int)t + ", start, ") + ", s->b, ")"),
             1};
        break;
      case SymKind::kLeaf: {
        if (s->leaf == LeafKind::kRead) {
          v = {ReadName(s), R(s)};
        } else if (s->leaf == LeafKind::kArg) {
          const int64_t pos = s->source.carry;
          auto st = state_args_.find(pos);
          if (st != state_args_.end()) {
            CheckStateView(s, pos, false);
            const MslShape& shape = arg_shapes[pos];
            v = {absl::StrCat("st", st->second),
                 shape.empty() ? 1 : shape.back()};
          } else {
            const auto& w = wholes_.at(SourceKey(s->source));
            v = {w.second, R(w.first)};
          }
        } else if (s->leaf == LeafKind::kWhole) {
          const auto& w = wholes_.at(SourceKey(s->source));
          v = {w.second, R(w.first)};
        } else {
          MslDecline("leaf kind in vector mode");
        }
        break;
      }
      case SymKind::kPad: {
        const EVal iv = emit(s->inner);
        const int64_t r = s->shape.back();
        const std::string name = absl::StrCat("v", tmp_++);
        body_.push_back(absl::StrCat("  ", T(s->dtype), " ", name, "[", r,
                                     "];"));
        const std::string src = iv.second > 1
                                    ? absl::StrCat(iv.first, "[r - ", s->lo,
                                                   "]")
                                    : iv.first;
        body_.push_back(absl::StrCat(
            "  for (int r = 0; r < ", r, "; r++) ", name, "[r] = (r >= ",
            s->lo, " && r < ", s->lo + s->n, ") ? (", src, ") : (",
            T(s->dtype), ")0;"));
        v = {name, r};
        break;
      }
      case SymKind::kRedReg: {
        const EVal iv = emit(s->inner);
        const int64_t true_w =
            s->inner->shape.empty() ? 1 : s->inner->shape.back();
        if (iv.second != true_w) {
          // The reduced dim is not (fully) register-resident: summing
          // registers would silently skip lane elements.
          MslDecline("register reduce over non-register dim");
        }
        const std::string name = absl::StrCat("v", tmp_++);
        body_.push_back(absl::StrCat("  ", T(s->dtype), " ", name, ";"));
        if (iv.second == 1) {
          body_.push_back(absl::StrCat("  ", name, " = ", iv.first, ";"));
        } else {
          body_.push_back(absl::StrCat(
              "  { ", T(s->dtype), " _s = (", T(s->dtype), ")0; ",
              "for (int r = 0; r < ", iv.second, "; r++) _s += ", iv.first,
              "[r]; ", name, " = _s; }"));
        }
        v = {name, 1};
        break;
      }
      case SymKind::kIota: {
        const std::string name = absl::StrCat("v", tmp_++);
        if (s->axis == static_cast<int64_t>(s->shape.size()) - 1 &&
            s->shape.back() > 1) {
          const int64_t r = s->shape.back();
          body_.push_back(absl::StrCat("  ", T(s->dtype), " ", name, "[", r,
                                       "];"));
          body_.push_back(absl::StrCat("  for (int r = 0; r < ", r, "; r++) ",
                                       name, "[r] = (", T(s->dtype), ")(r + ",
                                       s->start, ");"));
          v = {name, r};
        } else {
          std::string expr;
          if (s->shape[s->axis] == 1) {
            expr = absl::StrCat("(", T(s->dtype), ")", s->start);
          } else {
            const int64_t pad = static_cast<int64_t>(lane_shape_.size()) -
                                (static_cast<int64_t>(s->shape.size()) - 1);
            expr = absl::StrCat("(", T(s->dtype), ")(c", s->axis + pad, " + ",
                                s->start, ")");
          }
          body_.push_back(
              absl::StrCat("  ", T(s->dtype), " ", name, " = ", expr, ";"));
          v = {name, 1};
        }
        break;
      }
      case SymKind::kDot:
        v = emit_dot(s);
        break;
      case SymKind::kElem: {
        std::vector<EVal> args;
        for (const Sym* a : s->args) args.push_back(emit(a));
        int64_t r = 1;
        for (const EVal& a : args) r = std::max(r, a.second);
        r = std::max(r, R(s));
        for (const EVal& a : args)
          if (a.second != 1 && a.second != r)
            MslDecline("register width mismatch");
        const std::string name = absl::StrCat("v", tmp_++);
        body_.push_back(absl::StrCat("  ", Declare(name, s->dtype, r)));
        std::vector<std::string> acc;
        for (const EVal& a : args) acc.push_back(Scalarize(a));
        const std::string e = ElemExpr(s, acc, "vec emit");
        if (r > 1) {
          body_.push_back(absl::StrCat("  for (int r = 0; r < ", r, "; r++) ",
                                       name, "[r] = ", e, ";"));
        } else {
          body_.push_back(absl::StrCat("  ", name, " = ", e, ";"));
        }
        v = {name, r};
        break;
      }
      default:
        MslDecline(absl::StrCat("vec emit type ", MslDump(s)));
    }
    memo_[s] = v;
    return v;
  };

  // Emit writes of `val` into a buffer whose per-step layout is row-major
  // `per_shape`; absorbs top-level transposes into the write strides and
  // materializes SymStacks part by part.
  auto emit_write = [&](const Sym* val, const MslShape& per_shape,
                        const std::string& dest) {
    if (val->kind == SymKind::kStack) {
      MslShape vshape = val->shape;
      int64_t vaxis = val->axis;
      const int64_t extra = static_cast<int64_t>(vshape.size()) -
                            static_cast<int64_t>(per_shape.size());
      if (extra > 0 && vaxis >= extra) {
        bool lead_units = true;
        for (int64_t i = 0; i < extra; i++)
          lead_units = lead_units && vshape[i] == 1;
        if (lead_units) {
          vshape = MslShape(vshape.begin() + extra, vshape.end());
          vaxis -= extra;
        }
      }
      if (vshape.size() != per_shape.size())
        MslDecline(absl::StrCat("stack rank vs write target mismatch: stack",
                                Join(val->shape), " axis=", val->axis, " vs per",
                                Join(per_shape)));
      const MslShape st_t = MslRowmajor(per_shape);
      const int64_t ax_stride = st_t[vaxis];
      for (int64_t i = 0; i < vshape[vaxis]; i++)
        if (val->parts.find(i) == val->parts.end())
          MslDecline("partial stack write");
      if (static_cast<int64_t>(val->parts.size()) != vshape[vaxis])
        MslDecline("partial stack write");
      MslShape st_sub;
      for (size_t i2 = 0; i2 < st_t.size(); i2++)
        if (static_cast<int64_t>(i2) != vaxis) st_sub.push_back(st_t[i2]);
      for (const auto& kv : val->parts) {
        const EVal p = emit(kv.second);
        MslShape st_al = st_sub;
        while (st_al.size() < kv.second->shape.size())
          st_al.insert(st_al.begin(), 0);
        st_al.erase(st_al.begin(),
                    st_al.begin() + (st_al.size() - kv.second->shape.size()));
        const std::string off2 = VecOff(kv.second->shape, &st_al, 0);
        const int64_t base_off = kv.first * ax_stride;
        if (p.second > 1) {
          writes_.push_back(absl::StrCat("  for (int r = 0; r < ", p.second,
                                         "; r++) ", dest, "(", off2, ") + ",
                                         base_off, "u] = ", p.first, "[r];"));
        } else {
          writes_.push_back(absl::StrCat("  ", dest, "(", off2, ") + ",
                                         base_off, "u] = ", p.first, ";"));
        }
      }
      return;
    }
    const Sym* v2 = val;
    bool have_strides = false;
    MslShape wr_strides;
    if (val->kind == SymKind::kPerm) {
      const MslShape st_t = MslRowmajor(per_shape);
      for (size_t i = 0; i < val->perm.size(); i++) {
        const auto it =
            std::find(val->perm.begin(), val->perm.end(), static_cast<int64_t>(i));
        wr_strides.push_back(st_t[it - val->perm.begin()]);
      }
      have_strides = true;
      v2 = val->inner;
    }
    const EVal e = emit(v2);
    std::string off2;
    int64_t tgt_R = 1;
    if (have_strides) {
      off2 = VecOff(v2->shape, &wr_strides, 0);
      tgt_R = v2->shape.empty() ? 1 : v2->shape.back();
    } else if (e.second == 1 && !per_shape.empty() &&
               MslNumel(per_shape) == MslNumel(lane_shape_)) {
      // Width-1 value: every target dim is a lane dim; a fake unit register
      // dim maps them all to lane coordinates.
      MslShape fake(per_shape);
      fake.push_back(1);
      off2 = VecOff(fake, nullptr, 0);
      tgt_R = 1;
    } else {
      off2 = VecOff(v2->shape.empty() ? per_shape : v2->shape, nullptr, 0);
      tgt_R = v2->shape.empty() ? 1 : v2->shape.back();
    }
    const std::string src =
        MslRegSrc(e.first, e.second, tgt_R, "stacked write");
    if (tgt_R > 1) {
      writes_.push_back(absl::StrCat("  for (int r = 0; r < ", tgt_R,
                                     "; r++) ", dest, "(", off2, ")] = ", src,
                                     ";"));
    } else {
      writes_.push_back(
          absl::StrCat("  ", dest, "(", off2, ")] = ", src, ";"));
    }
  };

  for (size_t q = 0; q < stacked_.size(); q++) {
    const int64_t pos = std::get<0>(stacked_[q]);
    const Sym* idx = std::get<1>(stacked_[q]);
    const MslShape per = Tail1(arg_shapes[pos]);
    const int64_t inner = MslNumel(per);
    const std::string ii =
        absl::StrCat("((int)t + ", start, ") * ", idx->a, " + ", idx->b);
    emit_write(std::get<2>(stacked_[q]), per,
               absl::StrCat("out", q, "[(uint)(", ii, ") * ", inner, "u + "));
  }
  for (const auto& h : hidden_) {
    const int64_t numel = MslNumel(h.first->shape);
    emit_write(h.first, h.first->shape,
               absl::StrCat(h.second, "[t * ", numel, "u + "));
  }
  struct NewState {
    std::string name;
    int64_t r, sr;
  };
  std::vector<NewState> news;
  for (const auto& kv : states_) {
    const EVal e = emit(kv.second);
    const MslShape& shape = arg_shapes[kv.first];
    news.push_back({e.first, e.second, shape.empty() ? 1 : shape.back()});
  }
  out.insert(out.end(), body_.begin(), body_.end());
  out.insert(out.end(), writes_.begin(), writes_.end());
  std::vector<std::string> names;
  for (const NewState& n : news) names.push_back(n.name);
  const std::vector<int64_t> moved = MslAliasedStateMoves(names);
  for (int64_t k : moved) {
    const int64_t kpos = states_[k].first;
    const MslShape& kshape = arg_shapes[kpos];
    const int64_t kr = kshape.empty() ? 1 : kshape.back();
    out.push_back(absl::StrCat("  ",
                               Declare(absl::StrCat("sv", k), arg_dtypes[kpos],
                                       kr)));
    if (kr > 1) {
      out.push_back(absl::StrCat("  for (int r = 0; r < ", kr, "; r++) sv", k,
                                 "[r] = st", k, "[r];"));
    } else {
      out.push_back(absl::StrCat("  sv", k, " = st", k, ";"));
    }
  }
  std::map<std::string, std::string> ren;
  for (int64_t k : moved) ren[absl::StrCat("st", k)] = absl::StrCat("sv", k);
  for (size_t j = 0; j < news.size(); j++) {
    std::string nm = news[j].name;
    auto it = ren.find(nm);
    if (it != ren.end()) nm = it->second;
    const std::string src = MslRegSrc(
        nm, news[j].r, news[j].sr,
        absl::StrCat("state ", states_[j].first, " update"));
    if (news[j].sr > 1) {
      out.push_back(absl::StrCat("  for (int r = 0; r < ", news[j].sr,
                                 "; r++) st", j, "[r] = ", src, ";"));
    } else {
      out.push_back(absl::StrCat("  st", j, " = ", src, ";"));
    }
  }
  out.push_back("}");
  for (size_t j = 0; j < states_.size(); j++) {
    const MslShape& shape = arg_shapes[states_[j].first];
    const int64_t sr = shape.empty() ? 1 : shape.back();
    if (shape.empty()) {
      out.push_back(absl::StrCat("fin", j, "[0u] = st", j, ";"));
    } else if (sr > 1) {
      out.push_back(absl::StrCat("for (int r = 0; r < ", sr, "; r++) fin", j,
                                 "[", VecOff(shape, nullptr, 0), "] = st", j,
                                 "[r];"));
    } else {
      out.push_back(absl::StrCat("fin", j, "[", VecOff(shape, nullptr, 0),
                                 "] = st", j, ";"));
    }
  }
  return absl::StrJoin(out, "\n");
}

// ============================================ coop (threadgroup per batch el.)

std::string MslPlanned::EmitCoop() {
  const MslShape& lane = lane_shape_;
  const std::string fcoord = absl::StrCat("c", lane.size() - 1);
  std::vector<std::string> out;
  out.push_back("uint lane = thread_position_in_grid.x;");
  out.push_back(absl::StrCat("if (lane >= ", N, "u) return;"));
  int64_t tail = MslNumel(lane);
  for (size_t i = 0; i < lane.size(); i++) {
    tail /= lane[i];
    out.push_back(absl::StrCat("uint c", i, " = (lane / ", tail, "u) % ",
                               lane[i], "u;"));
  }

  // Threadgroup mirrors for dot data (one per distinct data sym).
  shared_.clear();
  shared_written_.clear();
  dot_memo_.clear();
  std::vector<std::pair<std::string, int64_t>> shared_sizes;
  {
    std::vector<Sym*> roots;
    for (const auto& kv : states_) roots.push_back(kv.second);
    for (const auto& t : stacked_) roots.push_back(std::get<2>(t));
    for (const auto& h : hidden_) roots.push_back(h.first);
    for (const Sym* d : MslCollectDots(roots)) {
      if (shared_.find(d->data) == shared_.end()) {
        const std::string nm = absl::StrCat("sh", shared_.size());
        shared_[d->data] = nm;
        shared_sizes.push_back({nm, d->csize});
      }
    }
  }
  for (const auto& kv : shared_sizes)
    out.push_back(
        absl::StrCat("threadgroup float ", kv.first, "[", kv.second, "];"));

  for (const auto& kv : wholes_) {
    const int sid = kv.first;
    const Sym* leaf = kv.second.first;
    const std::string& name = kv.second.second;
    const int64_t r = CoopR(leaf);
    out.push_back(Declare(name, leaf->dtype, r));
    if (BufferShape(leaf).empty()) {
      out.push_back(absl::StrCat(name, " = inp", sid, ";"));
    } else {
      Load(&out, name, leaf->dtype, r, Src(sid),
           CoopOff(leaf->shape, &leaf->strides, leaf->offset));
    }
  }
  for (size_t j = 0; j < states_.size(); j++) {
    const int64_t pos = states_[j].first;
    const MslShape& shape = arg_shapes[pos];
    const int64_t r = CoopRShape(shape);
    out.push_back(Declare(absl::StrCat("st", j), arg_dtypes[pos], r));
    if (shape.empty()) {
      out.push_back(absl::StrCat("st", j, " = init", j, ";"));
    } else {
      Load(&out, absl::StrCat("st", j), arg_dtypes[pos], r,
           absl::StrCat("init", j), CoopOff(shape, nullptr, 0));
    }
  }

  out.push_back(absl::StrCat("for (uint t_ = 0; t_ < ", trip, "u; t_++) {"));
  out.push_back(MslTDecl());

  for (const ReadEntry& re : reads_) {
    const Sym* leaf = re.leaf;
    const int64_t r = CoopR(leaf);
    const int64_t inner = MslNumel(leaf->inner_shape);
    const std::string idx =
        (re.a != 1 || re.b != 0 || start != 0)
            ? absl::StrCat("((int)t + ", start, ") * ", re.a, " + ", re.b)
            : "(int)t";
    const std::string off = CoopOff(leaf->shape, &leaf->strides, leaf->offset);
    out.push_back(absl::StrCat("  ", Declare(re.name, leaf->dtype, r)));
    Load(&out, re.name, leaf->dtype, r, Src(re.sid),
         absl::StrCat("(uint)(", idx, ") * ", inner, "u + (", off, ")"), "  ",
         true);
  }

  memo_.clear();
  tmp_ = 0;
  body_.clear();
  writes_.clear();

  std::function<EVal(const Sym*)> emit = [&](const Sym* s) -> EVal {
    auto hit = memo_.find(s);
    if (hit != memo_.end()) return hit->second;
    EVal v;
    switch (s->kind) {
      case SymKind::kConst:
        v = {MslLiteral(s), 1};
        break;
      case SymKind::kCounter:
        v = {absl::StrCat("(", s->a, " * ((int)t + ", start, ") + ", s->b, ")"),
             1};
        break;
      case SymKind::kLeaf: {
        if (s->leaf == LeafKind::kRead) {
          v = {ReadName(s), CoopR(s)};
        } else if (s->leaf == LeafKind::kArg) {
          const int64_t pos = s->source.carry;
          auto st = state_args_.find(pos);
          if (st != state_args_.end()) {
            CheckStateView(s, pos, true);
            v = {absl::StrCat("st", st->second), CoopRShape(arg_shapes[pos])};
          } else {
            const auto& w = wholes_.at(SourceKey(s->source));
            v = {w.second, CoopR(w.first)};
          }
        } else if (s->leaf == LeafKind::kWhole) {
          const auto& w = wholes_.at(SourceKey(s->source));
          v = {w.second, CoopR(w.first)};
        } else {
          MslDecline("coop leaf kind");
        }
        break;
      }
      case SymKind::kPad: {
        // Pad on the feature axis: component/feature-window shift.
        const EVal iv = emit(s->inner);
        const int64_t r = CoopR(s);
        if (s->lo % F != 0 || s->n % F != 0)
          MslDecline("coop pad not F-aligned");
        const int64_t glo = s->lo / F, gn = s->n / F;
        const std::string name = absl::StrCat("v", tmp_++);
        body_.push_back(absl::StrCat("  ", Declare(name, s->dtype, r)));
        const std::string src =
            iv.second > 1 ? absl::StrCat(iv.first, "[r - ", glo, "]")
                          : iv.first;
        if (r > 1) {
          body_.push_back(absl::StrCat(
              "  for (int r = 0; r < ", r, "; r++) ", name, "[r] = (r >= ", glo,
              " && r < ", glo + gn, ") ? (", src, ") : (", T(s->dtype), ")0;"));
        } else {
          body_.push_back(absl::StrCat("  ", name, " = ", src, ";"));
        }
        v = {name, r};
        break;
      }
      case SymKind::kDot: {
        // Structural CSE: the same dot may appear once per residual stack
        // differing only in unit dims; the per-thread scalar value is the
        // same.
        const int wsid = SourceKey(s->weight->source);
        std::string dkey = absl::StrCat(
            "dotval|", reinterpret_cast<uintptr_t>(s->data), "|", wsid, "|");
        for (const MslRole& r : s->widx)
          absl::StrAppend(&dkey, static_cast<int>(r.kind), ":", r.idx, ",");
        absl::StrAppend(&dkey, "|", s->weight->offset, "|",
                        Join(s->weight->shape));
        auto dhit = dot_memo_.find(dkey);
        if (dhit != dot_memo_.end()) return dhit->second;
        const std::string shname = shared_.at(s->data);
        const EVal d = emit(s->data);
        if (d.second * F != s->csize)
          MslDecline("coop dot data width mismatch");
        if (shared_written_.insert(s->data).second) {
          body_.push_back(
              "  threadgroup_barrier(metal::mem_flags::mem_threadgroup);");
          if (d.second > 1) {
            body_.push_back(absl::StrCat("  for (int r = 0; r < ", d.second,
                                         "; r++) ", shname, "[r * ", F, "u + ",
                                         fcoord, "] = ", d.first, "[r];"));
          } else {
            body_.push_back(absl::StrCat("  ", shname, "[", fcoord, "] = ",
                                         d.first, ";"));
          }
          body_.push_back(
              "  threadgroup_barrier(metal::mem_flags::mem_threadgroup);");
        }
        const MslShape& wst = s->weight->strides;
        const int64_t k_out = s->dsize / F;
        std::vector<std::string> terms;
        for (size_t wdim = 0; wdim < s->widx.size(); wdim++) {
          const int64_t st = wst[wdim];
          const MslRole& role = s->widx[wdim];
          if (role.kind == RoleKind::kC) {
            terms.push_back(absl::StrCat("(uint)cc * ", st, "u"));
          } else if (role.kind == RoleKind::kD) {
            if (k_out > 1) {
              terms.push_back(absl::StrCat("((uint)g * ", F, "u + ", fcoord,
                                           ") * ", st, "u"));
            } else {
              terms.push_back(absl::StrCat(fcoord, " * ", st, "u"));
            }
          }
        }
        if (s->weight->offset != 0)
          terms.insert(terms.begin(), absl::StrCat(s->weight->offset, "u"));
        const std::string woff =
            terms.empty() ? "0u" : absl::StrJoin(terms, " + ");
        const std::string name = absl::StrCat("v", tmp_++);
        if (k_out > 1) {
          body_.push_back(absl::StrCat("  float ", name, "[", k_out, "];"));
          body_.push_back(absl::StrCat(
              "  for (int g = 0; g < ", k_out,
              "; g++) { float _a = 0.0f; for (int cc = 0; cc < ", s->csize,
              "; cc++) _a += ", shname, "[cc] * ", Src(wsid), "[", woff, "]; ",
              name, "[g] = _a; }"));
        } else {
          body_.push_back(absl::StrCat("  float ", name, ";"));
          body_.push_back(absl::StrCat(
              "  { float _a = 0.0f; for (int cc = 0; cc < ", s->csize,
              "; cc++) _a += ", shname, "[cc] * ", Src(wsid), "[", woff, "]; ",
              name, " = _a; }"));
        }
        v = {name, k_out};
        dot_memo_[dkey] = v;
        break;
      }
      case SymKind::kElem: {
        std::vector<EVal> args;
        for (const Sym* a : s->args) args.push_back(emit(a));
        int64_t r = CoopR(s);
        for (const EVal& a : args) r = std::max(r, a.second);
        for (const EVal& a : args)
          if (a.second != 1 && a.second != r) MslDecline("coop width mismatch");
        const std::string name = absl::StrCat("v", tmp_++);
        body_.push_back(absl::StrCat("  ", Declare(name, s->dtype, r)));
        std::vector<std::string> acc;
        for (const EVal& a : args) acc.push_back(Scalarize(a));
        const std::string e = ElemExpr(s, acc, "coop emit");
        if (r > 1) {
          body_.push_back(absl::StrCat("  for (int r = 0; r < ", r, "; r++) ",
                                       name, "[r] = ", e, ";"));
        } else {
          body_.push_back(absl::StrCat("  ", name, " = ", e, ";"));
        }
        v = {name, r};
        break;
      }
      default:
        MslDecline(absl::StrCat("coop emit type ", MslDump(s)));
    }
    memo_[s] = v;
    return v;
  };

  for (size_t q = 0; q < stacked_.size(); q++) {
    const int64_t pos = std::get<0>(stacked_[q]);
    const Sym* idx = std::get<1>(stacked_[q]);
    const EVal e = emit(std::get<2>(stacked_[q]));
    const MslShape per = Tail1(arg_shapes[pos]);
    const int64_t inner = MslNumel(per);
    const std::string off = CoopOff(per, nullptr, 0);
    const std::string ii =
        absl::StrCat("((int)t + ", start, ") * ", idx->a, " + ", idx->b);
    const int64_t tgt_R = CoopRShape(per);
    const std::string src = MslRegSrc(e.first, e.second, tgt_R,
                                      absl::StrCat("stacked write ", pos));
    if (tgt_R > 1) {
      writes_.push_back(absl::StrCat("  for (int r = 0; r < ", tgt_R,
                                     "; r++) out", q, "[(uint)(", ii, ") * ",
                                     inner, "u + (", off, ")] = ", src, ";"));
    } else {
      writes_.push_back(absl::StrCat("  out", q, "[(uint)(", ii, ") * ", inner,
                                     "u + (", off, ")] = ", src, ";"));
    }
  }
  for (const auto& h : hidden_) {
    if (MslDebug())
      std::fprintf(stderr, "[metaljax] hidden %s: %s\n", h.second.c_str(),
                   MslDump(h.first).c_str());
    const EVal e = emit(h.first);
    const int64_t numel = MslNumel(h.first->shape);
    const std::string off = CoopOff(h.first->shape, nullptr, 0);
    const int64_t tgt_R = CoopRShape(h.first->shape);
    const std::string src = MslRegSrc(e.first, e.second, tgt_R,
                                      absl::StrCat("hidden stack ", h.second));
    if (tgt_R > 1) {
      writes_.push_back(absl::StrCat("  for (int r = 0; r < ", tgt_R,
                                     "; r++) ", h.second, "[t * ", numel,
                                     "u + (", off, ")] = ", src, ";"));
    } else {
      writes_.push_back(absl::StrCat("  ", h.second, "[t * ", numel, "u + (",
                                     off, ")] = ", src, ";"));
    }
  }
  struct NewState {
    std::string name;
    int64_t r, sr;
  };
  std::vector<NewState> news;
  for (const auto& kv : states_) {
    const EVal e = emit(kv.second);
    news.push_back({e.first, e.second, CoopRShape(arg_shapes[kv.first])});
  }
  out.insert(out.end(), body_.begin(), body_.end());
  out.insert(out.end(), writes_.begin(), writes_.end());
  std::vector<std::string> names;
  for (const NewState& n : news) names.push_back(n.name);
  const std::vector<int64_t> moved = MslAliasedStateMoves(names);
  for (int64_t k : moved) {
    const int64_t kpos = states_[k].first;
    const int64_t kr = CoopRShape(arg_shapes[kpos]);
    out.push_back(absl::StrCat(
        "  ", Declare(absl::StrCat("sv", k), arg_dtypes[kpos], kr)));
    if (kr > 1) {
      out.push_back(absl::StrCat("  for (int r = 0; r < ", kr, "; r++) sv", k,
                                 "[r] = st", k, "[r];"));
    } else {
      out.push_back(absl::StrCat("  sv", k, " = st", k, ";"));
    }
  }
  std::map<std::string, std::string> ren;
  for (int64_t k : moved) ren[absl::StrCat("st", k)] = absl::StrCat("sv", k);
  for (size_t j = 0; j < news.size(); j++) {
    std::string nm = news[j].name;
    auto it = ren.find(nm);
    if (it != ren.end()) nm = it->second;
    const std::string src = MslRegSrc(
        nm, news[j].r, news[j].sr,
        absl::StrCat("state ", states_[j].first, " update"));
    if (news[j].sr > 1) {
      out.push_back(absl::StrCat("  for (int r = 0; r < ", news[j].sr,
                                 "; r++) st", j, "[r] = ", src, ";"));
    } else {
      out.push_back(absl::StrCat("  st", j, " = ", src, ";"));
    }
  }
  out.push_back("}");
  for (size_t j = 0; j < states_.size(); j++) {
    const MslShape& shape = arg_shapes[states_[j].first];
    const int64_t sr = CoopRShape(shape);
    if (shape.empty()) {
      out.push_back(absl::StrCat("fin", j, "[0u] = st", j, ";"));
    } else if (sr > 1) {
      out.push_back(absl::StrCat("for (int r = 0; r < ", sr, "; r++) fin", j,
                                 "[", CoopOff(shape, nullptr, 0), "] = st", j,
                                 "[r];"));
    } else {
      out.push_back(absl::StrCat("fin", j, "[", CoopOff(shape, nullptr, 0),
                                 "] = st", j, ";"));
    }
  }
  return absl::StrJoin(out, "\n");
}

}  // namespace metaljax
