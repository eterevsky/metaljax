/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

msl_scan, natively: a counted loop whose body is a per-lane computation
becomes ONE generated Metal kernel -- a thread per lane, the carries in
registers, the whole time loop inside the shader.

`src/metaljax/msl_scan.py` is the specification, class for class and check
for check.  What this file holds is the PLANNING half: the symbolic IR the
loop body is evaluated into, the classification of each carry, the mode
choice, and the three emitters that generate MSL.  The LAUNCH half already
exists and is shared with Stage 1 -- `runtime/msl.cc`'s `MslPlan` -- so the
product of a plan is exactly what `src/metaljax/tape.py::_lower_msl` writes:
a kernel source, a binding order, a geometry, and the little recipe that
turns the kernel's outputs back into the loop's carries.

Three modes, in the order they were built (CLAUDE.md items 7, 8, 9):

* `scalar` (affine): a pure elementwise body, one thread per lane of the
  state, every carry in a scalar register.
* `vector`: small in-lane matvecs -- the trailing feature dim of every value
  lives in a register array, and cross-lane weight-gradient dots are fissioned
  out of the kernel into one post-kernel batched matmul.
* `coop`: full-width matvec cells -- one threadgroup per batch element, the
  feature dim as the thread axis, dot data staged through threadgroup memory.

Two disciplines are correctness rather than taste and are carried over
verbatim: the loop counter is VOLATILE (Apple's shader compiler miscompiles
multi-iteration kernel time loops without it -- CLAUDE.md item 12c), and a
plan that cannot be expressed is never an error, only a decline that leaves
the loop to the compiled-graph path.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_MSL_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_MSL_H_

#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace metaljax {

using MslShape = std::vector<int64_t>;

// msl_scan.py's `_Unsupported`: this loop does not qualify.  Thrown from
// anywhere in the analysis or emission and caught at `BuildMslPlan`, which is
// the only place that decides "no plan" -- exactly as `_msl_plan_for` catches
// every exception `build_plan` can raise.
struct MslUnsupported : std::exception {
  std::string msg;
  const char* what() const noexcept override { return msg.c_str(); }
};

[[noreturn]] void MslDecline(std::string what);

// ---------------------------------------------------------------- symbolic IR
//
// One class per Sym subclass of msl_scan.py, flattened into one node type:
// the analysis rewrites nodes in place (the canonicalizer interns them, the
// weight-layout pass rewrites a leaf's strides), and pointer identity is what
// every memo in the emitters keys on -- which is `id(sym)` there and a `Sym*`
// here.  Nodes live in the plan's arena, so a pointer is stable for the life
// of the plan (unlike CPython's `id`, whose reuse is CLAUDE.md item 16's trap).

enum class SymKind {
  kCounter,   // integer scalar, affine in the induction variable: a*iv + b
  kConst,
  kLeaf,      // a tensor materialized per lane out of a device buffer
  kElem,      // elementwise op
  kDot,       // small in-lane matvec: data (lane..., C) x invariant weights
  kPerm,      // deferred transpose of an elementwise subtree
  kPad,       // zero padding along the register (last) dim
  kStack,     // small tensor assembled by static-index update slices
  kIota,
  kRedReg,    // in-lane reduction over the register dim
  kAccRed,    // cross-lane reduce-sum, hoisted out of the kernel
  kAccDot,    // cross-lane dot, hoisted out as one post-kernel einsum
};

enum class LeafKind { kRead, kWhole, kArg, kUpdated };

// Which device array a leaf reads.  `kCarry` is a loop carry by position;
// `kFree` a value captured from the enclosing scope; `kHoist` the result of a
// loop-invariant op the analyzer lifted OUT of the body (msl_scan.py
// `_try_hoist`), which the tape lowers into the enclosing frame.
enum class SrcKind { kCarry, kFree, kHoist };

struct MslSrc {
  SrcKind kind = SrcKind::kCarry;
  int64_t carry = -1;
  mlir::Value value;

  bool operator==(const MslSrc& o) const {
    return kind == o.kind && carry == o.carry && value == o.value;
  }
};

// One dim's role in a `SymDot`: which lane dim of the data it follows, the
// output-feature ("register") dim, a unit dim, or -- on the weight side --
// the contracted and output dims.
enum class RoleKind { kData, kReg, kOne, kC, kD };
struct MslRole {
  RoleKind kind = RoleKind::kOne;
  int64_t idx = 0;   // kData only
  bool operator==(const MslRole& o) const {
    return kind == o.kind && (kind != RoleKind::kData || idx == o.idx);
  }
  bool operator!=(const MslRole& o) const { return !(*this == o); }
};

struct Sym {
  SymKind kind;
  MslShape shape;
  // The dtype names src/metaljax/tape.py spells ("f32", "i1", "ui32"), plus
  // msl_scan.py's own "int" for a counter.
  std::string dtype;

  // kCounter: value = a*iv + b, derived from carry `base` (-1 = none).
  int64_t a = 0, b = 0;
  int64_t base = -1;

  // kConst.  Both forms are kept because `_literal` formats by DTYPE and the
  // analysis compares by value (`float(pv.value) == 0.0`).
  double dval = 0.0;
  int64_t ival = 0;
  bool uns = false;

  // kLeaf
  LeafKind leaf = LeafKind::kArg;
  MslSrc source;
  Sym* idx = nullptr;              // kRead: the affine index, a SymCounter
  MslShape inner_shape;            // kRead: the buffer's per-step block shape
  bool has_inner = false;
  MslShape strides;
  int64_t offset = 0;

  // kElem
  std::string op;
  std::vector<Sym*> args;
  std::string extra;               // convert: dtype; compare: direction

  // kDot
  Sym* data = nullptr;
  Sym* weight = nullptr;
  std::vector<MslRole> roles, widx;
  int64_t csize = 0, dsize = 0;

  // kPerm / kPad / kRedReg / kAccRed: the operand
  Sym* inner = nullptr;
  std::vector<int64_t> perm;       // kPerm, and kAccRed/kAccDot's output perm
  int64_t lo = 0, n = 0;           // kPad

  // kStack
  Sym* sbase = nullptr;
  std::map<int64_t, Sym*> parts;
  int64_t axis = 0;                // kStack and kIota

  // kIota
  int64_t start = 0;

  // kAccRed: the reduced dims.  kAccDot: (lb, rb, lc, rc) and its operands.
  std::vector<int64_t> dims;
  std::vector<std::vector<int64_t>> dims4;
  Sym* lhs = nullptr;
  Sym* rhs = nullptr;
};

// Stable-address storage for the Sym graph of one plan.
class SymArena {
 public:
  Sym* make(SymKind k) {
    nodes_.emplace_back();
    nodes_.back().kind = k;
    return &nodes_.back();
  }
  size_t size() const { return nodes_.size(); }

 private:
  std::deque<Sym> nodes_;
};

// ------------------------------------------------------------- plan products
//
// Everything `metaljax.tape._lower_msl` reads off a Plan, which is what the
// native lowering writes into `MslPlan`'s launch recipe.

// A weight whose layout was canonicalized (Plan._weight_norms): the view to
// take of the source buffer, and the transpose to materialize.
struct MslNorm {
  MslShape shape, strides;
  int64_t offset = 0;
  std::vector<int64_t> perm;
};

struct MslPackGroup {
  std::string name;
  std::string dtype;
  std::vector<int> sids;
};

// One node of an accumulator's stacking recipe (`Plan.run`'s `stacked`, and
// tape.py's `_msl_spec`).  kinds: 0 a hidden per-step stack, 1 a slice of a
// device buffer, 2 a post-kernel sum, 3 a post-kernel batched matmul.
struct MslAccSpec {
  int kind = 0;
  int hidden = 0;                  // kind 0
  MslSrc source;                   // kind 1
  int64_t a = 0, b = 0;            // kind 1: the read's affine index
  MslShape shape;                  // kind 1/2: the per-step shape
  std::vector<int64_t> dims;       // kind 2
  std::vector<int64_t> perm;       // kind 2/3
  std::vector<std::vector<int64_t>> dims4;   // kind 3: lb, rb, lc, rc
  MslShape lshape, rshape;         // kind 3
  std::vector<MslAccSpec> kids;
};

class MslPlanned;

// What the analysis needs from the enclosing lowering, so this file depends on
// nothing of it: the module (to resolve `func.call`), the counted-loop
// analysis (for statically-counted NESTED loops, which are unrolled
// symbolically) and the op registry (`_try_hoist` only lifts an op the
// engine can actually execute).
struct MslEnv {
  mlir::ModuleOp module;
  // `_analyze_counted` restricted to a STATIC bound, which is the only kind
  // msl_scan accepts for a nested loop: (counter position, bound).
  std::function<std::optional<std::pair<int64_t, int64_t>>(mlir::Operation*)>
      counted;
  std::function<std::optional<int64_t>(mlir::Operation*, int64_t)> static_start;
  std::function<bool(llvm::StringRef)> op_supported;
};

// msl_scan.py's `Plan`: one compiled persistent-kernel execution plan.
class MslPlanned {
 public:
  // ---- what the lowering writes into the launch recipe
  std::string mode;                // "scalar" | "vector" | "coop"
  std::string kernel_name;
  std::string source;
  int64_t N = 0, F = 0, trip = 0, start = 0;

  std::vector<MslSrc> sources;
  std::vector<MslShape> arg_shapes;
  std::vector<std::string> arg_dtypes;

  std::vector<int> passthrough;
  std::vector<std::pair<int, int64_t>> counters;    // (pos, per-iter delta)
  std::vector<int> state_pos;                       // one per kernel state
  std::vector<int> stacked_pos;                     // one per stacked output
  std::vector<MslShape> hidden_shapes;
  std::vector<std::string> hidden_names;
  std::vector<std::pair<int, std::vector<MslAccSpec>>> acc_plans;

  std::map<int, MslNorm> weight_norms;
  std::vector<int> unpacked;
  std::vector<MslPackGroup> pack_groups;
  // Sources the kernel expects CONVERTED to f32 (bf16 dot weights): the leaf
  // dtype was flipped at plan time, and the launch materializes an f32 copy
  // once per call instead of the dot loop converting per element per timestep.
  std::vector<int> wconv;

  // Diagnostics only (METALJAX_DEBUG).
  int64_t num_packed = 0;

  size_t num_hidden() const { return hidden_shapes.size(); }

  // ---- construction (msl_scan.py `Plan.__init__`)
  //
  // `fired` records which of the two late heuristics the build used, so
  // `BuildMslPlan` can retry without them when it throws -- msl_scan.py's
  // `_last_flipped` / `_last_inlane`, which are module globals there because
  // that engine is single-threaded.  They are set the MOMENT the heuristic
  // fires, since the build may throw long afterwards.
  struct Fired {
    bool flipped = false;
    bool inlane = false;
  };
  MslPlanned(const MslEnv& env, mlir::Block& body, int64_t counter_pos,
             int64_t trip, int64_t start, bool no_coop_flip,
             bool no_inlane_dot, Fired* fired);

 private:
  friend class MslAnalyzer;

  // ---- emission
  std::string EmitScalar();
  std::string EmitVector();
  std::string EmitCoop();

  // Register width of a value, per mode.
  int64_t R(const Sym* s) const;
  int64_t CoopR(const Sym* s) const;
  int64_t CoopRShape(const MslShape& shape) const;
  std::string VecOff(const MslShape& shape, const MslShape* strides = nullptr,
                     int64_t base = 0) const;
  std::string CoopOff(const MslShape& shape, const MslShape* strides = nullptr,
                      int64_t base = 0) const;
  void CheckStateView(const Sym* s, int64_t pos, bool coop) const;

  int SourceKey(const MslSrc& src) const;
  MslShape BufferShape(const Sym* leaf) const;
  std::string Src(int sid) const;
  std::string ReadName(const Sym* leaf) const;

  void Collect(const std::vector<Sym*>& roots);
  int SourceId(const MslSrc& src);

  // ---- state carried from analysis into emission
  SymArena arena_;
  const MslEnv* env_ = nullptr;

  std::vector<std::pair<int64_t, Sym*>> states_;                  // (pos, expr)
  std::vector<std::tuple<int64_t, Sym*, Sym*>> stacked_;   // (pos, idx, value)
  std::vector<std::pair<Sym*, std::string>> hidden_;
  std::vector<std::pair<int64_t, std::vector<Sym*>>> accums_;

  // The reads, in discovery order (`Plan._reads`), and the name each got.
  struct ReadEntry {
    std::string key;
    Sym* leaf;
    std::string name;
    int sid;
    int64_t a, b;
  };
  std::vector<ReadEntry> reads_;
  absl::flat_hash_map<std::string, size_t> read_index_;

  std::map<int, std::pair<Sym*, std::string>> wholes_;
  std::map<int, Sym*> weights_;
  std::map<int, std::vector<Sym*>> weight_dots_;
  std::map<std::string, int> source_ids_;
  std::map<int64_t, int> state_args_;             // carry pos -> state index
  std::map<int, std::pair<std::string, int64_t>> packed_;
  MslShape lane_shape_;

  // Per-emitter scratch (one emitter runs per plan).
  absl::flat_hash_map<const Sym*, std::pair<std::string, int64_t>> memo_;
  absl::flat_hash_map<std::string, std::pair<std::string, int64_t>> dot_memo_;
  std::set<const Sym*> shared_written_;
  std::map<const Sym*, std::string> shared_;
  int64_t tmp_ = 0;
  std::vector<std::string> body_, writes_;
};

// The header every generated kernel carries (msl_scan.py `_KERNEL_HEADER`).
extern const char kMslKernelHeader[];

// ---- the pieces the two implementation files share (msl_scan.py's module
// level: the op tables, the literal and offset formatters, the knobs).
const std::map<std::string, std::string>& MslUnaryOps();
const std::map<std::string, std::string>& MslBinaryOps();
const std::map<std::string, std::string>& MslCompareOps();
const std::string& MslTypeName(const std::string& dtype);
const std::string& MslTDecl();
std::string MslVolLoad(const std::string& type, const std::string& buf,
                       const std::string& off);
std::string MslLiteral(const Sym* s);
std::string MslOffStrided(const MslShape& shape, const MslShape& strides,
                          const MslShape& lane, int64_t base);
std::string MslLaneOffset(const MslShape& shape, const MslShape& lane);
std::vector<int64_t> MslAliasedStateMoves(const std::vector<std::string>& srcs);
std::string MslRegSrc(const std::string& name, int64_t R, int64_t tgt_R,
                      const std::string& what);
int64_t MslNumel(const MslShape& s);
MslShape MslRowmajor(const MslShape& shape);
int64_t MslRegLimit();
// msl_scan.py `_dump`: a Sym as a one-line expression, for decline messages.
std::string MslDump(const Sym* s);
// msl_scan.py `_collect_dots`: every SymDot reachable from these roots.
std::vector<Sym*> MslCollectDots(const std::vector<Sym*>& roots);

// msl_scan.py `ENABLED` (METALJAX_MSL=0 turns the whole feature off).
bool MslEnabled();
// The volatile-counter workaround's setting, for the one variant the tape
// cannot express (METALJAX_MSL_VOLATILE=tmap binds an extra input).
bool MslVolatileIsTmap();
bool MslDebug();

// msl_scan.py `build_plan`: a plan for this loop body, or nullptr when it does
// not qualify.  Never throws: every decline is caught here, which is the
// contract `_msl_plan_for` has with the rest of the engine.  `why` receives
// the decline message when there is no plan.
std::shared_ptr<MslPlanned> BuildMslPlan(const MslEnv& env, mlir::Block& body,
                                         int64_t counter_pos, int64_t trip,
                                         int64_t start, std::string* why);

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_MSL_H_
