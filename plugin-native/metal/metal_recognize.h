/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The recognizers, as passes over the parsed StableHLO.

Stage 1 rewrites three graph shapes into one MLX call each -- a dequantize-
then-matmul chain into `quantized_matmul`, a softmax attention into
`fast::scaled_dot_product_attention`, an expert dispatch into `gather_qmm` --
and `src/metaljax/{qmm,sdpa,moe}.py` are the specification of what may be
rewritten.  This header is the phase-2 shape of that: the ANALYSIS produces a
plan of absorbed ops and roots, the lowering consults the plan, and the
executor's M4 emits (runtime/emits.cc) run what comes out.  Nothing about the
rewrite is re-invented here; where this and the Python could drift, the Python
is the specification and `runtime/emits.cc`'s `Cursor` reads are the ground
truth for the encoding.

Two rules the whole file exists to keep:

* A half-matched pattern lowers as ORDINARY ops.  Every rejection is a
  `_Reject` in the Python and a `nullopt`/skip here, and the consequence is a
  correct slow program -- never a wrong fused one.
* Packing needs concrete buffers, so it happens once, at the first execute,
  and the packed arrays travel as trailing INPUTS of the tape (never as
  constants a compiled graph could bake by value).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_RECOGNIZE_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_RECOGNIZE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlx/mlx.h"
#include "program.h"

namespace metaljax {

namespace mx = mlx::core;

// --------------------------------------------------------------------------
// qmm (src/metaljax/qmm.py `Match`)
// --------------------------------------------------------------------------

struct QmmMatch {
  // The op whose result the fused call produces, and the activation it reads.
  mlir::Operation* root = nullptr;
  mlir::Value lhs;
  // The three operand subtrees, as the multiply saw them (all of the weight's
  // pre-`post` shape, so one recipe re-shapes all three).  `zero` is null when
  // the quantization has no zero point, and in the MXFP4 form `codes` holds
  // the DECODED values -- the grid is non-uniform, so its codes have become
  // floats by the time the scale is applied.
  mlir::Value codes;
  mlir::Value zero;
  mlir::Value scale;
  // The reshape/transpose chain between the reconstruction and the dot.
  std::vector<mlir::Operation*> post;
  // The per-channel form: the graph divides the OUTPUT by a broadcast scale.
  bool recip = false;
  std::vector<int64_t> bcast_dims;

  // dot_general, resolved (qmm.py `_finish`).
  std::vector<int64_t> lperm, rperm;
  std::vector<int64_t> rshape, bshape, mshape, nshape;
  int64_t B = 1, M = 0, K = 0, N = 0;
  int out_dtype = 0;          // the tape's dtype code
  bool swapped = false;       // the weight was the dot's LHS
  int mode = 0;               // 0 affine, 1 mxfp4
  // The (lo, hi) an integer `codes - zero` is computed in, when the graph does
  // that subtraction in integer arithmetic (which wraps and the rewrite does
  // not).  Absent when the subtraction is done in floating point.
  bool has_sub_range = false;
  int64_t sub_lo = 0, sub_hi = 0;
  std::string name;

  // The ops this match absorbs (they never reach the tape), and the ones that
  // MUST be absorbed for the rewrite to be a win (`_prune`).
  std::vector<mlir::Operation*> ops;
  std::vector<mlir::Operation*> required;
  // Which of @main's arguments the reconstruction reads.  The pack is a pure
  // function of exactly these buffers, which is what makes it cacheable and
  // what says when it has to be rebuilt.
  std::vector<int> arg_indices;

  // Filled by the pack build, on concrete buffers.
  int64_t gs = 0, bits = 0;
  bool has_perm = false;
  int slot = -1;              // index of the first pack array in `packs`
  int nvals = 0;              // how many of them this match owns
  bool disabled = false;
  // Set when an expert-gather rewrite takes this dot over: the weight is
  // still packed here, but the dense `quantized_matmul` is never emitted --
  // `gather_qmm` replaces it (moe.py, `Match.absorbed`).
  bool absorbed = false;
};

// --------------------------------------------------------------------------
// sdpa (src/metaljax/sdpa.py `Match`)
// --------------------------------------------------------------------------

// sdpa.py `_apply`: an optional transpose then an optional reshape, which is
// how an operand of whatever layout the graph used reaches the [B, H, T, D]
// MLX's fused kernel wants.
struct Rec {
  bool has_perm = false;
  std::vector<int64_t> perm;
  bool has_shape = false;
  std::vector<int64_t> shape;
};

struct SdpaMatch {
  // The op whose result the fused attention produces: the probabilities-times
  // -values dot.
  mlir::Operation* root = nullptr;
  mlir::Value q, k, v;
  Rec q_rec, k_rec, v_rec;
  double scale = 1.0;

  // The mask, when the logits carry one.  `kind` 0 is a boolean SELECT (the
  // graph selects between the logits and a sentinel) and 1 an ADDITIVE mask;
  // `base` is the value the additive form is built from, and the two
  // constants are the sentinel and the scale folded into it.
  bool has_mask = false;
  int mask_kind = 0;
  mlir::Value mask_base;
  double mask_const = 0.0;
  double mask_mul = 1.0;
  Rec mask_rec;

  // The output recipe: reshape, transpose, reshape (sdpa.py `_out_recipe`).
  bool has_pre = false;
  std::vector<int64_t> pre;
  bool has_out_perm = false;
  std::vector<int64_t> out_perm;
  bool has_post = false;
  std::vector<int64_t> post;

  // The dtype the three operands are cast to, and the root's own.  Tape dtype
  // codes (metal_dtypes.h `TapeDtypeCode`).
  int dtype = 0;
  int out_dtype = 0;

  // The ops this match absorbs: the whole softmax chain, which never reaches
  // the tape.
  std::vector<mlir::Operation*> ops;
  std::string name;
};

// --------------------------------------------------------------------------
// moe (src/metaljax/moe.py: the pair-space plan)
// --------------------------------------------------------------------------

// One node of the plan `moe.emit` interprets.  The dense `(expert, token)`
// grid collapses onto the `P = T * K` selected pairs, so every node's value
// is `[P] + (its dense shape without the expert and token axes)`.  `ea` / `ta`
// are the axes of the DENSE shape that carry the expert and the token
// dimension, or -1 (Python's None).
struct MoeNode {
  enum Kind { kExt, kElem, kView, kDot };
  Kind kind = kExt;
  int ea = -1, ta = -1;
  std::vector<int64_t> shape;

  // kExt: a value computed outside the region and gathered on the way in,
  // or -- when `op` is set -- a nullary op bound inside a callee, which the
  // tape emits as the op it is.
  mlir::Value value;
  mlir::Operation* op = nullptr;

  // kElem: the op, over sources that are indices into `MoeMatch::order`.
  std::vector<int> srcs;

  // kView: squeeze / reorder / reshape / broadcast of the trailing axes.
  int src = -1;
  int view = 0;   // 0 bcast, 1 perm, 2 reshape, 3 slice, 4 dotperm
  std::vector<int64_t> keep;         // bcast: destination per source axis, -1
  std::vector<int64_t> trailing;     // bcast / reshape / dotperm
  std::vector<int64_t> order;        // perm / dotperm
  std::vector<int64_t> slices;       // slice: (start, stop, stride) triples

  // kDot: one per-expert matmul, `gather_qmm` when its weight was packed by
  // the quantized-matmul recognizer and `gather_mm` when it stays float.
  int data = -1;
  mlir::Value weight;
  QmmMatch* pack = nullptr;
  int64_t M = 0, N = 0, K = 0;
  std::vector<int64_t> mshape, nshape;
  int out_dtype = 0;
  bool n_first = false;
};

struct MoeMatch {
  mlir::Operation* root = nullptr;   // the weighted sum over the expert axis
  // The router's two tensors, both read at run time: `[T, K]` each.
  mlir::Value indices, weights;
  std::vector<MoeNode> order;        // the plan, in dependency order
  int out = -1;                      // which node is the region's output
  int64_t E = 0, T = 0, K = 0, P = 0;
  int64_t out_axis = 0;
  std::vector<int64_t> out_shape;
  int out_dtype = 0, sum_dtype = 0;
  std::vector<mlir::Operation*> ops; // the ops this match absorbs
  std::string name;

  // What the first-execute check reads: the rank-3 routing-score operand of
  // the product, and where its three axes are.  `moe.py` verifies the same
  // identity on SYNTHETIC logits; this plugin has no way to substitute an
  // input into a callee's frame, so it checks the same identity on the REAL
  // ones -- see `VerifyMoe`.
  mlir::Value score3;
  int64_t e_ax = 0, t_ax = 0;
  // The top-k's input, and its element type: what the check substitutes.
  // Null when the top-k sits inside a callee, whose values a cone cannot
  // bind -- the match is then left unverified and falls back.
  mlir::Value logits;
  int logits_dtype = 0;
  bool disabled = false;
};

// --------------------------------------------------------------------------
// the plan
// --------------------------------------------------------------------------

struct RewritePlan {
  std::vector<std::unique_ptr<QmmMatch>> qmm;
  std::vector<std::unique_ptr<SdpaMatch>> sdpa;
  std::vector<std::unique_ptr<MoeMatch>> moe;

  // Ops a recognizer absorbed: no entry, no slot, never executed.
  llvm::DenseSet<mlir::Operation*> skip;
  // Ops that lower to a fused call instead of themselves.
  llvm::DenseMap<mlir::Operation*, QmmMatch*> qmm_roots;
  llvm::DenseMap<mlir::Operation*, SdpaMatch*> sdpa_roots;
  llvm::DenseMap<mlir::Operation*, MoeMatch*> moe_roots;

  // The packed arrays, in the order the tape's trailing inputs take them.
  std::vector<mx::array> packs;
  // The @main arguments the packs were built from, and the identity of the
  // arrays they held when they were built: a later execute that hands over
  // different buffers has to repack (qmm.py `_Pack.matches`).
  std::vector<int> pack_args;
  std::vector<std::uintptr_t> pack_arg_ids;

  bool empty() const {
    return qmm_roots.empty() && sdpa_roots.empty() && moe_roots.empty();
  }
  // Recompute `skip` and the root maps from the matches that are still live.
  void rebuild();
  bool absorbed(mlir::Operation* op) const {
    return op != nullptr && skip.contains(op);
  }
};

// Evaluate operand subtrees on the concrete arguments of this execute.  The
// lowering supplies it (it is the only thing that can build a tape), and the
// pack build is written against this one capability.
// `bound` pins values in the MIDDLE of the graph to arrays of the caller's
// choosing: the cone then stops there instead of walking on into a loop
// carry.  It is what lets the router check run on SYNTHETIC logits, which is
// the only way to reach a dispatch inside a decode loop -- everything below
// the top-k depends on nothing else.
using SubtreeEval = std::function<absl::StatusOr<std::vector<mx::array>>(
    const std::vector<mlir::Value>& roots,
    const std::vector<std::pair<mlir::Value, mx::array>>& bound)>;

// P19: the same cone, NARROWED to one block of the weight's rows (qmm.py
// `_Source`).  `blocked` names every value of the cone whose leading axis IS
// the weight's row axis; the Program that comes back declares those values
// with `c` rows instead of the full extent, so running it on @main's
// arguments -- with every blocked ARGUMENT sliced to the same `c` rows --
// computes exactly that slice of the root.  Which values may be blocked is
// decided by the caller (`RowSource`, whose rules are qmm.py `_Source._op`'s);
// this builder only follows the set it is handed.
//
// It hands back the Program rather than its outputs because one Program
// serves every block of the same width: the blocks of a weight differ only in
// which rows the leaf slices name.
using BlockedConeBuilder =
    std::function<absl::StatusOr<std::shared_ptr<Program>>(
        const std::vector<mlir::Value>& roots,
        const llvm::DenseSet<mlir::Value>& blocked, int64_t c)>;

// Everything the pack build needs from the lowering that asked for it.
struct PackContext {
  // @main's single block, and this execute's buffers for its arguments.
  mlir::Block* main = nullptr;
  const std::vector<mx::array>* args = nullptr;
  // The module, for the fingerprint's callee lookup (a call is serialized
  // through its BODY: jax renumbers private helpers per program).
  mlir::ModuleOp module;
  SubtreeEval eval;
  BlockedConeBuilder blocked_cone;
};

// Structural analysis of one module: never touches a value.  Fills `plan`
// with the matches that are worth rewriting, in the order the walk found
// them.  `donated` are the argument positions the caller may reuse -- a
// quantized weight among them cannot be packed once and kept.
void AnalyzeQmm(mlir::func::FuncOp fn, const absl::flat_hash_set<int>& donated,
                RewritePlan* plan);

// The same for the fused attentions (src/metaljax/sdpa.py `analyze`).  Runs
// AFTER `AnalyzeQmm` and is disjoint from it by construction: a candidate
// that would absorb an op a quantized matmul already claimed is dropped, so
// no op has two owners.  Needs no buffers -- an attention has nothing to pack
// -- so this one runs at compile time as well.
void AnalyzeSdpa(mlir::func::FuncOp fn, RewritePlan* plan);

// The expert dispatches (src/metaljax/moe.py `analyze`).  Runs AFTER
// `AnalyzeQmm`, whose matches it may take over: a per-expert dot whose weight
// was packed there is dispatched by `gather_qmm` instead of by the dense
// `quantized_matmul`, and that match is marked `absorbed` rather than emitted.
void AnalyzeMoe(mlir::func::FuncOp fn, RewritePlan* plan);

// The first-execute check of a match's router, on the buffers of this
// execute: the scores must BE the top-k weights scattered at the matched
// indices, and no token may have more than K of them.  A match that fails is
// disabled and its dense dispatch runs.  METALJAX_MOE_VERIFY=0 skips it.
absl::Status VerifyMoe(RewritePlan* plan, const SubtreeEval& eval);

// Verify and repack every live match, at the first execute.  A match whose
// weight fails any exactness check is DISABLED (its chain then lowers
// literally) rather than packed approximately; `plan->packs` comes back
// holding the arrays the tape's trailing inputs take.
absl::Status BuildQmmPacks(RewritePlan* plan, const PackContext& ctx);

// The cross-executable build cache's counters (P19), for the tests: a pack is
// a pure function of the reconstruction and the buffers it reads, so two
// EXECUTABLES over one model share it.  `blocked` counts the weights that
// packed one row block at a time and `whole` the ones that fell back.
struct PackStats {
  int64_t build_hits = 0;      // a pack another executable had already built
  int64_t build_misses = 0;    // ...and one this process built here
  int64_t build_declines = 0;  // a reconstruction the fingerprint cannot cover
  int64_t blocked = 0;
  int64_t whole = 0;
  int64_t entries = 0;         // how many packs the cache holds right now
  // The highest CLAIMED device memory any one pack wave reached, read from
  // the plugin's own libmlx (the host process's `mlx.core` is a different
  // runtime and reads zero).  This is the figure row-blocking bounds.
  uint64_t peak_bytes = 0;
};
PackStats QmmPackStats();
void ResetQmmPackStats();

// P19: the ops a row-blocked prologue may narrow (qmm.py `_ROW_LOCAL` plus
// the five whose row-locality is a rule about their dimension attributes).
// The pack build decides which VALUES are narrowed; the lowering asserts
// against this that nothing outside the set ever is.
const absl::flat_hash_set<std::string>& RowLocalOps();

// The environment knobs, read once (METALJAX_QMM, METALJAX_SDPA,
// METALJAX_RECOGNIZE).
bool QmmEnabled();
bool SdpaEnabled();
bool RecognizeEnabled();

// qmm.py `_hoist`: follow a value out of the loops that merely carry it
// around.  jax lowers a while_loop's closed-over constants as loop-carried
// state rather than as region captures, so a decode loop's weights arrive as
// body block arguments and the value that is constant for the whole loop --
// the one a prologue can evaluate -- is the while's initial operand.
mlir::Value HoistInvariant(mlir::Value v);

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_RECOGNIZE_H_
