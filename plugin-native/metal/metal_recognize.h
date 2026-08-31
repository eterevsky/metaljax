/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The recognizers, as passes over the parsed StableHLO.

Stage 1 rewrote three graph shapes into one MLX call each -- a dequantize-
then-matmul chain into `quantized_matmul`, a softmax attention into
`fast::scaled_dot_product_attention`, an expert dispatch into `gather_qmm` --
and `src/metaljax/{qmm,sdpa,moe}.py` (deleted 0.11.6, ef5774d) were the
specification of what may be rewritten.  This header is the phase-2 shape of
that: the ANALYSIS produces a plan of absorbed ops and roots, the lowering
consults the plan, and the executor's M4 emits (runtime/emits.cc) run what
comes out.  Nothing about the rewrite was re-invented here; while both
engines lived, where this and the Python could drift, the Python was the
specification; `runtime/emits.cc`'s `Cursor` reads remain the ground truth
for the encoding.

Two rules the whole file exists to keep:

* A half-matched pattern lowers as ORDINARY ops.  Every rejection was a
  `_Reject` in the Python and is a `nullopt`/skip here, and the consequence is a
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
  // MLX's native affine form is `scales * q + biases` with an ARBITRARY float
  // bias per group, not `scale * (q - zero)` with an integer zero point -- and
  // the two are not interchangeable: matching slopes forces `zero = -bias/
  // scale`, which is not an integer, and rounding it costs up to half a
  // quantization step.  When this is set, `zero` holds that bias MAP instead
  // of a zero point: it is the same shape, rides the same blocking/regrouping
  // machinery, and only the two places that interpret it differ (the
  // integrality checks are skipped and the pack's biases are the map itself,
  // shifted by whatever offset made the codes unsigned).
  bool add_bias = false;
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
  // Just the CODES subtree's arguments.  A checkpoint that already ships
  // MLX's packing (mlx-community's, say) is unpacked in the graph only so the
  // recognizer can verify it, and repacks to bytes identical to the argument
  // it came from; when that is provably so the pack aliases the argument
  // instead of allocating a second copy of the whole model.
  std::vector<int> code_args;
  // Likewise for the scale and zero/bias subtrees.  These packs are a
  // sixteenth of the codes but they are RETAINED for the life of the
  // executable, which at 235B parameters is 7 GB apiece.
  std::vector<int> scale_args;
  std::vector<int> zero_args;

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
  // the product, and where its three axes are.  `moe.py` verified the same
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
// ragged dot (metal_ragged.cc): jax's `lax.ragged_dot` dense fallback
// --------------------------------------------------------------------------

// The graph shape jax emits for `lax.ragged_dot` on every backend without a
// native lowering (jax _ragged_dot_general_impl, "ragged_to_dense"): the
// [m, k] rows are broadcast to [g, m, k], masked by the half-open intervals
// a cumsum of `group_sizes` defines, and contracted against the [g, k, n]
// group weights over BOTH g and k.  That computes each row against its own
// group's matrix while streaming — and, at maxtext's 512-row padding,
// computing — all `g` of them.  The rewrite runs the row-vs-own-group form
// literally: one `gather_mm` whose per-row group index is derived from the
// same cumsum.  No runtime verification is needed; the equivalence is
// structural (see metal_ragged.cc for the two documented deviations:
// non-finite values in never-selected groups, and non-partition
// `group_sizes` no bincount can produce).
struct RaggedMatch {
  mlir::Operation* root = nullptr;  // the dot_general
  mlir::Value x;      // [m, k]: the real rows (the absorbed pad's input)
  mlir::Value w;      // [g, k, n]: the group weights, as the dot read them
  mlir::Value ends;   // [g]: cumsum(group_sizes), already in the graph
  int64_t g = 0, m = 0, M = 0, k = 0, n = 0;   // M = padded rows at the root
  int out_dtype = 0;
  // The stacked-weights form (maxtext's scan over layers): `w` was a
  // dynamic-index-in-dim out of a pass-through loop carry whose init is a
  // [1,0,2,3] transpose of a [g, L, k, n] stack.  MLX's dynamic slice is a
  // COPY (the offset is data), which re-materialized 100s of MB of weights
  // per layer per token; the gather reads the ORIGINAL buffer instead, at
  // matrix index `group * L + layer` — the transpose composed with the
  // row-major flattening.  When set, `w` is unused and the helper call that
  // computed it is absorbed; `w_stack` is the carried [L, g, k, n] view and
  // `layer` the loop counter the slice took.
  bool stacked = false;
  mlir::Value w_stack;
  mlir::Value layer;
  int64_t L = 0;
  // The helper call the stacked form absorbs.  AnalyzeRagged reverts the
  // match to the sliced form if the use-count fixpoint could not absorb it
  // (the slice would then run anyway, and the fused op must read its
  // result rather than gather a second copy of the same weights).
  mlir::Operation* helper_call = nullptr;
  std::vector<mlir::Operation*> ops;  // the ops this match absorbs
  std::string name;
};

// --------------------------------------------------------------------------
// stacked-weight dot (metal_stacked.cc): jax's scanned-layer weight reads
// --------------------------------------------------------------------------

// `dot_general(x, w)` where `w = dynamic_index_in_dim(stack, layer)` — the
// form every scanned-layer jax model (maxtext, flax scan) reads a layer's
// weights in.  MLX's dynamic slice is a COPY (the offset is data), so a
// 28-layer decode re-materialized every weight every token; the rewrite
// reads matrix `layer` straight out of the stack with `gather_mm`, whose
// kernels take the batch and leading strides from the array (no copy at
// all).  The geometry is proven at analysis: the stack (behind an optional
// loop-invariant transpose, carried or not) is contiguous, and the
// contracted / free axes collapse to an [L, K, N] `as_strided` view with
// the free stride 1 — anything else Bails to the ordinary slice chain.
struct StackedDotMatch {
  mlir::Operation* root = nullptr;  // the dot_general
  mlir::Value x;        // the activation, as the dot read it
  mlir::Value w_carry;  // the helper's stack operand (possibly a carried
                        // transposed view; the emit transposes it back)
  mlir::Value layer;    // the helper's index operand
  mlir::Operation* helper_call = nullptr;
  // Inverse of the transpose between `w_carry` and the contiguous stack
  // (identity when the stack is carried untransposed).
  std::vector<int64_t> back_perm;
  // The [L, K, N] view of the CONTIGUOUS stack, in elements.
  int64_t L = 0, K = 0, N = 0;
  int64_t sl = 0, sk = 0, sn = 0;
  int64_t M = 0;                    // product of the lhs free dims
  int out_dtype = 0;
  std::vector<int64_t> out_shape;   // the root's declared result shape
  std::vector<mlir::Operation*> ops;  // the ops this match absorbs
  std::string name;
};

// --------------------------------------------------------------------------
// multi-span decode attention (metal_mla.cc): maxtext's MLA decode
// --------------------------------------------------------------------------

// maxtext's MLA decode attention computes each cache span (prefill,
// autoregressive) as a separate masked softmax with running max/sum, then
// joins them with the flash-attention renormalization:
//   m = max(m_p, m_ar);  l = exp(m_p-m)*l_p + exp(m_ar-m)*l_ar
//   out = (exp(m_p-m)/l)*o_p + (exp(m_ar-m)/l)*o_ar
// which IS the softmax over the concatenated scores.  The rewrite emits
// concat(K), concat(V), one additive mask from the segment ids, and ONE
// `fast::scaled_dot_product_attention` — ~80 tape entries per layer become
// one, and the two-dot chain becomes the fused decode kernel (the vendored
// MLX supports the 192/128 MLA head geometry in its vector kernel).  The
// probabilities move from the literal bf16 max-subtract/exp chain into the
// kernel's f32 arithmetic: same class of reduction-order change as any fused
// attention, tolerance-level vs CPU, greedy near-ties may flip.
struct MlaSpan {
  mlir::Value k;    // [B, T, H, D], as the scores dot read it
  mlir::Value v;    // [B, T, H, Dv]
  mlir::Value seg;  // [B, T] integer segment ids
  int64_t T = 0;
};

struct MlaMatch {
  mlir::Operation* root = nullptr;  // the final combine add, [B, 1, H, Dv]
  mlir::Value q;                    // [B, 1, H, D], pre-scaled by the graph
  std::vector<MlaSpan> spans;
  int64_t B = 0, H = 0, D = 0, Dv = 0;
  int64_t seg_val = 1;              // the id the mask compares EQ against
  double mask_true = 0.0;           // the additive mask's two values
  double mask_false = 0.0;          // (0 and the -2.38e38 sentinel)
  int dtype = 0, out_dtype = 0;     // tape dtype codes
  std::vector<mlir::Operation*> ops;  // the ops this match absorbs
  std::string name;
};

// --------------------------------------------------------------------------
// rms norm (metal_norm.cc): the spelled-out root-mean-square norm
// --------------------------------------------------------------------------

// jax spells one RMS norm as ~13 ops (upcast, square, sum/N, +eps, rsqrt,
// scale, downcast, weight apply); the rewrite is MLX's fused
// `fast::rms_norm(x, w, eps)`, which accumulates in f32.  Each library
// spells the chain differently -- the square as a multiply or a `power`,
// the upcast before or after it, the eps add in f32 or the model dtype, the
// weight applied by a batching dot or a broadcast multiply, and sometimes no
// weight at all -- so metal_norm.cc reads BACKWARDS from one of three roots.
// The fused-vs-literal difference is 1 bf16 ULP for the keras families and 2
// for gemma's (whose chain rounds the mean to bf16, making the kernel the
// more accurate side); both measured, see metal_norm.cc.
struct RmsNormMatch {
  // The transpose (maxtext's dot form), multiply (broadcast weight apply, or
  // the bare normalize when there is no weight), or convert (keras' downcast)
  // the fused op takes the place of.
  mlir::Operation* root = nullptr;
  mlir::Value x;                    // [.., N], the normed input
  mlir::Value w;                    // [N] learned scale; null = no scale
  double eps = 0.0;
  // A splat folded off the weight -- maxtext's `w + 0`, gemma 2/3's
  // `1 + w`.  The emit forms `w + offset` once, [N] wide.
  double offset = 0.0;
  std::vector<mlir::Operation*> ops;
  std::string name;
};

// --------------------------------------------------------------------------
// the plan
// --------------------------------------------------------------------------

struct RewritePlan {
  std::vector<std::unique_ptr<QmmMatch>> qmm;
  std::vector<std::unique_ptr<SdpaMatch>> sdpa;
  std::vector<std::unique_ptr<MoeMatch>> moe;
  std::vector<std::unique_ptr<RaggedMatch>> ragged;
  std::vector<std::unique_ptr<StackedDotMatch>> stacked;
  std::vector<std::unique_ptr<MlaMatch>> mla;
  std::vector<std::unique_ptr<RmsNormMatch>> norm;

  // Ops a recognizer absorbed: no entry, no slot, never executed.
  llvm::DenseSet<mlir::Operation*> skip;
  // Ops that lower to a fused call instead of themselves.
  llvm::DenseMap<mlir::Operation*, QmmMatch*> qmm_roots;
  llvm::DenseMap<mlir::Operation*, SdpaMatch*> sdpa_roots;
  llvm::DenseMap<mlir::Operation*, MoeMatch*> moe_roots;
  llvm::DenseMap<mlir::Operation*, RaggedMatch*> ragged_roots;
  llvm::DenseMap<mlir::Operation*, StackedDotMatch*> stacked_roots;
  llvm::DenseMap<mlir::Operation*, MlaMatch*> mla_roots;
  llvm::DenseMap<mlir::Operation*, RmsNormMatch*> norm_roots;

  // The packed arrays, in the order the tape's trailing inputs take them.
  std::vector<mx::array> packs;
  // The @main arguments the packs were built from, and the identity of the
  // arrays they held when they were built: a later execute that hands over
  // different buffers has to repack (qmm.py `_Pack.matches`).
  std::vector<int> pack_args;
  std::vector<std::uintptr_t> pack_arg_ids;

  bool empty() const {
    return qmm_roots.empty() && sdpa_roots.empty() && moe_roots.empty() &&
           ragged_roots.empty() && stacked_roots.empty() &&
           mla_roots.empty() && norm_roots.empty();
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

// The ragged-dot dispatches (metal_ragged.cc).  Runs AFTER `AnalyzeMoe` and
// claims only dots no other recognizer did.  Purely structural — nothing to
// pack, nothing to verify at run time — so it runs at compile time like
// `AnalyzeSdpa`.  METALJAX_RAGGED=0 disables it.
void AnalyzeRagged(mlir::func::FuncOp fn, RewritePlan* plan);

// The stacked-weight dots (metal_stacked.cc).  Runs after AnalyzeRagged and
// claims only dots no other recognizer did.  Purely structural, like
// AnalyzeRagged.  METALJAX_STACKED_DOT=0 disables it.
void AnalyzeStackedDot(mlir::func::FuncOp fn, RewritePlan* plan);

// The multi-span decode attentions (metal_mla.cc).  Roots at the combine
// ADD, which no dot-rooted recognizer claims.  Purely structural.
// METALJAX_MLA=0 disables it.
void AnalyzeMla(mlir::func::FuncOp fn, RewritePlan* plan);

// The RMS norms (metal_norm.cc).  Runs LAST; roots at the weight-apply
// transpose.  Purely structural.  METALJAX_NORM=0 disables it.
void AnalyzeNorm(mlir::func::FuncOp fn, RewritePlan* plan);

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
