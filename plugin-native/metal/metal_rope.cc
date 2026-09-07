/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The rotate-half rope APPLY as a view (task B5, row 11's dispatch census).

Every rotary embedding in the model table applies its table the same way:

    x1, x2 = split(x, 2, axis=-1)
    rot    = concat(-x2, x1, axis=-1)          # keras: stack((-x2, x1), -2)
    out    = x * cos + rot * sin                #        then reshape back

and jax lowers `rot` as two slices, a negate, two unit-dim broadcasts (or
reshapes), a concatenate and a reshape.  MLX runs the negate as a kernel and
the concatenate as one copy kernel PER INPUT, so each apply costs a negate,
two copies and the fused multiply-add: four dispatches, three of them pure
data movement, for 4 KB of q.  Row 11's keras arm applies rope 56 times per
token (q and k of 28 layers): 168 dispatches per token, ~19 % of the token's
872, on a body whose device time is launch-bound.

The rewrite makes `rot` a VIEW.  Seen as [.., 2, h], the rotated half is the
same buffer with the pair axis REVERSED -- `[x2, x1]` -- which MLX expresses
as a slice of stride -1 (a stride rewrite, no kernel); the sign moves onto
the table, which is exact by IEEE 754's sign symmetry of multiplication:

    (-x2) * sin1  ==  x2 * (-sin1)        bit for bit, every rounding mode

so with `sgn = [-1, +1]` along the pair axis

    out5 = x5 * cos5 + rev(x5) * (sin5 * sgn)

is the SAME multiplications and the same addition on the same values, in the
same order, and the whole thing is one fused elementwise kernel over five
inputs.  Measured on the bench venv's MLX (logs/b5-row11/census/
probe_rope_view.py): eager and compiled answers bit-identical to the literal
spelling on random bf16 inputs; four kernels per apply become one.

What is matched (`AnalyzeRope`): an `add` of two `multiply`s, one reading x
and a cos operand, the other reading the rotate-half of x and a sin operand;
the rotate-half through either the keras form (unit-dim hops around the two
halves, a concatenate on the new pair axis, a reshape back to x's shape) or
the last-axis form (`concatenate(-x2, x1, -1)`); the two halves as slices of
one x covering exactly [0, h) and [h, 2h) of an even last axis, every other
axis full.  The table operands are read BEHIND a right-aligned
`broadcast_in_dim` when there is one (the broadcast itself is shared by
every layer after XLA's CSE, so it is peeled, not absorbed -- the DCE pass
drops it once no one reads it), and must be [.., 2h] so the same [.., 2, h]
view applies to them.  Any float dtype; any leading shape (prefill's
[1, L, H, D] as much as decode's), because nothing here depends on the
sequence extent -- unlike `mx::fast::rope`, which would also change the
arithmetic (fast cos/sin in f32 against the table's bf16 rounding) and is
the numeric-class option this rewrite exists to avoid.

A half-matched pattern lowers as ORDINARY ops, and a chain whose
intermediates escape declines whole: the emit reads x, cos and sin and
produces the root, so every absorbed op must be read by nothing else.
METALJAX_ROPE_VIEW=0 disables it; the literal tape is the A/B arm.

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
  std::fprintf(stderr, "[metaljax-native] rope: %s\n", line.c_str());
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

bool IsRealFloat(mlir::Type t) {
  return mlir::isa<mlir::Float16Type>(t) || mlir::isa<mlir::BFloat16Type>(t) ||
         mlir::isa<mlir::Float32Type>(t);
}

int64_t Numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

mlir::Operation* DefOf(mlir::Value v, const char* what) {
  mlir::Operation* op = v.getDefiningOp();
  if (op == nullptr) Bail(absl::StrCat(what, " is a block argument"));
  return op;
}

// A reshape, or a broadcast_in_dim that moves no data (same element count,
// monotone dimension map): the unit-dim hops jax puts around the halves.
bool IsUnitHop(mlir::Operation* op) {
  if (auto bc = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(op)) {
    if (Numel(ShapeOf(bc.getOperand())) != Numel(ShapeOf(bc.getResult())))
      return false;
    int64_t last = -1;
    for (int64_t d : bc.getBroadcastDimensions()) {
      if (d <= last) return false;
      last = d;
    }
    return true;
  }
  return mlir::isa_and_nonnull<mlir::stablehlo::ReshapeOp>(op);
}

// The value behind any chain of unit hops, the hops recorded.
mlir::Value PeelHops(mlir::Value v, std::vector<mlir::Operation*>* ops) {
  for (int guard = 0; guard < 6; guard++) {
    mlir::Operation* def = v.getDefiningOp();
    if (!IsUnitHop(def)) return v;
    ops->push_back(def);
    v = def->getOperand(0);
  }
  return v;
}

// A slice of `x` taking [lo, hi) of the last axis and all of every other
// axis, stride 1 throughout.
bool IsHalfSlice(mlir::Operation* op, mlir::Value* x, int64_t* lo,
                 int64_t* hi) {
  auto sl = mlir::dyn_cast_or_null<mlir::stablehlo::SliceOp>(op);
  if (!sl) return false;
  const std::vector<int64_t> src = ShapeOf(sl.getOperand());
  const size_t r = src.size();
  if (r == 0 || sl.getStartIndices().size() != r) return false;
  for (size_t i = 0; i + 1 < r; i++) {
    if (sl.getStartIndices()[i] != 0 || sl.getLimitIndices()[i] != src[i] ||
        sl.getStrides()[i] != 1)
      return false;
  }
  if (sl.getStrides()[r - 1] != 1) return false;
  *x = sl.getOperand();
  *lo = sl.getStartIndices()[r - 1];
  *hi = sl.getLimitIndices()[r - 1];
  return true;
}

// The table operand behind a right-aligned broadcast, if that is what it is.
// Not absorbed: XLA CSEs the broadcast across every layer that reads the
// same table, so it may have any number of users; peeled, the emit reads the
// operand the broadcast read and the DCE pass drops the broadcast when the
// last reader is gone.
mlir::Value PeelTableBroadcast(mlir::Value v) {
  auto bc = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      v.getDefiningOp());
  if (!bc) return v;
  const int64_t in_rank = static_cast<int64_t>(ShapeOf(bc.getOperand()).size());
  const int64_t out_rank = static_cast<int64_t>(ShapeOf(bc.getResult()).size());
  int64_t i = 0;
  for (int64_t d : bc.getBroadcastDimensions()) {
    if (d != out_rank - in_rank + i) return v;
    i++;
  }
  return bc.getOperand();
}

// The rotate-half of some x, rooted at `v`: the source x, the half width,
// and the ops the chain is made of.  Bails when `v` is not one.
struct Rot {
  mlir::Value x;
  int64_t h = 0;
  std::vector<mlir::Operation*> ops;
  const char* form = "";
};

Rot MatchRot(mlir::Value v) {
  Rot r;
  const std::vector<int64_t> shape = ShapeOf(v);
  mlir::Operation* def = DefOf(v, "the rotated half");
  mlir::Operation* reshape = nullptr;
  if (mlir::isa<mlir::stablehlo::ReshapeOp>(def)) {
    reshape = def;
    def = DefOf(def->getOperand(0), "the pair stack");
  }
  auto cat = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(def);
  if (!cat) Bail("the rotated half is not a concatenate");
  if (cat->getNumOperands() != 2) Bail("a concatenate of other than 2 parts");
  const int64_t ax = cat.getDimension();
  const std::vector<int64_t> cshape = ShapeOf(cat.getResult());

  // The negated half.  keras negates before its unit hop; other spellings
  // may negate after it.
  std::vector<mlir::Operation*> ops;
  mlir::Value p = PeelHops(cat->getOperand(0), &ops);
  mlir::Operation* neg = p.getDefiningOp();
  if (!mlir::isa_and_nonnull<mlir::stablehlo::NegOp>(neg))
    Bail("the first part is not negated");
  ops.push_back(neg);
  p = PeelHops(neg->getOperand(0), &ops);
  mlir::Value x_hi, x_lo;
  int64_t lo1 = 0, hi1 = 0, lo2 = 0, hi2 = 0;
  if (!IsHalfSlice(p.getDefiningOp(), &x_hi, &lo1, &hi1))
    Bail("the negated part is not a last-axis slice");
  ops.push_back(p.getDefiningOp());
  mlir::Value q = PeelHops(cat->getOperand(1), &ops);
  if (!IsHalfSlice(q.getDefiningOp(), &x_lo, &lo2, &hi2))
    Bail("the second part is not a last-axis slice");
  ops.push_back(q.getDefiningOp());
  if (x_hi != x_lo) Bail("the two halves are slices of different values");
  const std::vector<int64_t> xshape = ShapeOf(x_hi);
  const int64_t D = xshape.back();
  if (D % 2 != 0) Bail("an odd rotary width");
  const int64_t h = D / 2;
  if (lo1 != h || hi1 != D || lo2 != 0 || hi2 != h)
    Bail("the halves are not [h, 2h) and [0, h)");

  // The geometry: the pair stack (concat on a new axis right before the
  // half, reshaped back to x's shape) or the last-axis concatenate.
  std::vector<int64_t> stacked(xshape.begin(), xshape.end() - 1);
  stacked.push_back(2);
  stacked.push_back(h);
  const int64_t rank = static_cast<int64_t>(xshape.size());
  if (reshape != nullptr) {
    if (shape != xshape) Bail("the reshape does not restore x's shape");
    if (cshape != stacked || ax != rank - 1)
      Bail("the pair stack is not [.., 2, h] on the pair axis");
    ops.push_back(reshape);
    r.form = "stacked";
  } else {
    if (cshape != xshape || ax != rank - 1)
      Bail("the concatenate is not on the last axis");
    r.form = "concat";
  }
  ops.push_back(cat);
  r.x = x_hi;
  r.h = h;
  r.ops = std::move(ops);
  return r;
}

std::unique_ptr<RopeMatch> MatchRoot(mlir::Operation* root) {
  auto add = mlir::dyn_cast<mlir::stablehlo::AddOp>(root);
  if (!add) Bail("not an add");
  if (!IsRealFloat(ElemOf(add.getResult()))) Bail("not a float add");
  mlir::Operation* ma = add.getLhs().getDefiningOp();
  mlir::Operation* mb = add.getRhs().getDefiningOp();
  if (!mlir::isa_and_nonnull<mlir::stablehlo::MulOp>(ma) ||
      !mlir::isa_and_nonnull<mlir::stablehlo::MulOp>(mb))
    Bail("not a sum of two multiplies");

  // Which multiply holds the rotated half, and on which side.
  std::optional<Rot> rot;
  mlir::Operation* m_rot = nullptr;
  mlir::Value sin_b;
  for (mlir::Operation* m : {ma, mb}) {
    for (unsigned j = 0; j < 2 && !rot.has_value(); j++) {
      try {
        Rot cand = MatchRot(m->getOperand(j));
        rot = std::move(cand);
        m_rot = m;
        sin_b = m->getOperand(1 - j);
      } catch (const Reject&) {
      }
    }
    if (rot.has_value()) break;
  }
  if (!rot.has_value()) Bail("neither multiply reads a rotated half");
  mlir::Operation* m_x = m_rot == ma ? mb : ma;
  mlir::Value cos_b;
  if (m_x->getOperand(0) == rot->x) {
    cos_b = m_x->getOperand(1);
  } else if (m_x->getOperand(1) == rot->x) {
    cos_b = m_x->getOperand(0);
  } else {
    Bail("the other multiply does not read x");
  }

  auto m = std::make_unique<RopeMatch>();
  m->root = root;
  m->x = rot->x;
  m->h = rot->h;
  m->form = rot->form;
  m->cos = PeelTableBroadcast(cos_b);
  m->sin = PeelTableBroadcast(sin_b);
  const std::vector<int64_t> xshape = ShapeOf(m->x);
  for (mlir::Value t : {m->cos, m->sin}) {
    const std::vector<int64_t> ts = ShapeOf(t);
    if (ts.empty() || ts.back() != 2 * m->h)
      Bail("a table operand that is not rotary-wide");
    if (ts.size() > xshape.size()) Bail("a table operand of higher rank than x");
    if (ElemOf(t) != ElemOf(m->x)) Bail("a table operand of another dtype");
  }
  if (ElemOf(add.getResult()) != ElemOf(m->x)) Bail("the result dtype differs");
  m->ops = rot->ops;
  m->ops.push_back(ma);
  m->ops.push_back(mb);

  // Every absorbed op is read by the match and nothing else: the emit
  // produces the root from x, cos and sin, so an intermediate with another
  // reader would go unwritten.
  llvm::DenseSet<mlir::Operation*> inside(m->ops.begin(), m->ops.end());
  inside.insert(root);
  for (mlir::Operation* o : m->ops) {
    for (mlir::Value res : o->getResults())
      for (mlir::Operation* u : res.getUsers())
        if (!inside.contains(u)) Bail("an intermediate escapes the match");
  }
  m->name = absl::StrCat("rope.", m->form, ".h", m->h);
  return m;
}

}  // namespace

void AnalyzeRope(mlir::func::FuncOp fn, RewritePlan* plan) {
  if (EnvOff("METALJAX_ROPE_VIEW")) return;
  if (fn.getBody().getBlocks().size() != 1) return;
  auto module = fn->getParentOfType<mlir::ModuleOp>();
  if (!module) return;

  // Ops the other recognizers own (roots and absorbed alike), and the
  // absorbed ones on their own: a value the emit READS may be another
  // recognizer's root -- that binds a slot -- but never an absorbed op,
  // which has none.
  llvm::DenseSet<mlir::Operation*> taken = plan->skip;
  llvm::DenseSet<mlir::Operation*> absorbed_elsewhere = plan->skip;
  auto take = [&](mlir::Operation* root,
                  const std::vector<mlir::Operation*>& ops) {
    taken.insert(root);
    for (mlir::Operation* o : ops) {
      taken.insert(o);
      absorbed_elsewhere.insert(o);
    }
  };
  for (const auto& m : plan->qmm) take(m->root, m->ops);
  for (const auto& m : plan->sdpa) take(m->root, m->ops);
  for (const auto& m : plan->moe) take(m->root, m->ops);
  for (const auto& m : plan->ragged) take(m->root, m->ops);
  for (const auto& m : plan->stacked) take(m->root, m->ops);
  for (const auto& m : plan->mla) take(m->root, m->ops);
  for (const auto& m : plan->gdn) take(m->root, m->ops);
  for (const auto& m : plan->norm) take(m->root, m->ops);

  std::vector<std::unique_ptr<RopeMatch>> found;
  llvm::DenseSet<mlir::Operation*> visited_fns;
  std::function<void(mlir::Block&)> walk = [&](mlir::Block& block) {
    for (mlir::Operation& op : block) {
      const std::string name = OpName(&op);
      if (name == "stablehlo.add" && !taken.contains(&op)) {
        try {
          found.push_back(MatchRoot(&op));
        } catch (const Reject& e) {
          // Every float add in the module reaches here; only rejects past
          // the shape of a rope apply are worth a line.
          static const char* const kQuiet[] = {
              "not an add", "not a float add", "not a sum of two multiplies",
              "neither multiply reads a rotated half"};
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

  llvm::DenseSet<mlir::Operation*> used;
  int kept = 0, stacked = 0;
  for (auto& m : found) {
    bool overlaps = used.contains(m->root) || taken.contains(m->root);
    for (mlir::Operation* o : m->ops)
      overlaps = overlaps || taken.contains(o) || used.contains(o);
    for (mlir::Value v : {m->x, m->cos, m->sin}) {
      mlir::Operation* def = v.getDefiningOp();
      overlaps = overlaps || (def != nullptr && absorbed_elsewhere.contains(def));
    }
    if (overlaps) {
      if (kDebug) Debug("declined a rope apply that overlaps another match");
      continue;
    }
    used.insert(m->root);
    for (mlir::Operation* o : m->ops) used.insert(o);
    kept++;
    if (m->form == std::string("stacked")) stacked++;
    plan->rope.push_back(std::move(m));
  }
  if (kept > 0)
    Debug(absl::StrCat("matched ", kept, " rotate-half apply(ies) as views (",
                       stacked, " pair-stacked, ", kept - stacked,
                       " last-axis) in ", fn.getName().str()));
}

}  // namespace metaljax
