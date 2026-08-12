// metaljax native engine — the tape and its interface (Stage 2).
//
// src/metaljax/tape.py lowers an analyzed executable's main block into one
// of these Programs: a flat op tape with SSA slots resolved to indices,
// attributes decoded to integers, and constants already sitting on the
// device. `run` then walks it with no MLIR, no Python and no GIL — the
// point of the milestone, since decode is Python-dispatch-bound (~120 ms
// per token at gemma-31B scale, notes/cpp-migration-plan.md).
//
// The op set is small and grows monotonically. Anything tape.py cannot
// lower declines the WHOLE program, which then runs on the Python engine
// exactly as before, so a missing op is a performance question and never a
// correctness one. Every handler here is a transliteration of the Python
// handler in src/metaljax/ops/: where those carry a dtype branch or a
// zero-size guard, so does this file, because the differential test
// compares output BYTES and both engines call the same MLX kernels.
//
// C++ owns the opcode enum. Python asks for it via `opcodes()` and an op
// name that is not in the dict declines — so adding an op here is what
// makes it reachable, and there is no second table to forget.
//
// --- where the code lives -------------------------------------------------
//
// This header is the line a phase-2 native plugin builds against: the opcode
// enum, the entry format, the `Program` that replays it, and the runtime
// disciplines a replay runs under. It names no Python, and neither does any
// translation unit below it except `bindings.cc` and `metaljax_native.cc` --
// the tape is a plain C++ library that a process without an interpreter links
// and runs. Everything else is one translation unit per concern, and the op
// families mirror src/metaljax/ops/ one for one --
// the handlers ARE transliterations of those modules, so keeping the split
// identical is what lets a reader put the two engines side by side:
//
//   program.cc          Program itself: slots, the environment, the walk
//                       over the tape, and the recovery that wraps a run
//   config.cc           the registries and the cadences: the op-name table,
//                       `configure`, the counters, the recovery hook
//   bindings.cc         the nanobind adapter: the ONLY file besides
//                       metaljax_native.cc that knows Python exists
//   dtypes.cc           dtypes.py: the element-type table, the predicates
//                       handlers branch on, weak literals, complex64
//   runtime.cc          the cadences of interpreter.py + ops/control.py:
//                       eager flush, loop flush, host reads, recovery
//   compile.cc          the mx::compile integration and output anchoring
//   ops_elementwise.cc  ops/elementwise.py       (+ compare/select/convert)
//   ops_shape.cc        ops/shape.py             (+ constants)
//   ops_reduce.cc       ops/reduction.py         (reduce, argmax pair,
//                                                 general bodies, windows)
//   ops_index.cc        ops/gather.py + ops/sort.py
//   ops_linalg.cc       ops/linalg.py
//   ops_rng.cc          ops/rng.py
//   ops_conv.cc         ops/conv.py
//   emits.cc            the M4 recognizer emits: qmm, sdpa, moe
//   control.cc          ops/control.py: while/if/case and the body runners
//   msl.cc, msl.h       msl_scan.py's generated kernels (M5b)
//   host.cc             ops/callbacks.py: the entries that leave the tape
//
// `Program::step` owns no handler of its own: it offers the entry to one
// `step_*` per family until one claims it, so the partition of the opcode
// space over the files above is checked by the single `throw` at the end of
// that chain rather than by a table that could disagree with a switch.

#ifndef METALJAX_PROGRAM_H_
#define METALJAX_PROGRAM_H_

#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace metaljax {

// --------------------------------------------------------------------------
// opcodes
// --------------------------------------------------------------------------

enum Op : int {
  // unary
  kAbs = 0, kCbrt, kCeil, kCos, kErf, kErfInv, kExp, kExpm1, kFloor,
  kIsFinite, kLog, kLog1p, kLogistic, kNegate, kNot, kRoundAfz, kRoundEven,
  kRsqrt, kSign, kSin, kSqrt, kTan, kTanh, kSquare,
  // binary
  kAdd, kMultiply, kSubtract, kMaximum, kMinimum, kAnd, kOr, kXor,
  kDivide, kRemainder, kPower, kAtan2,
  kShiftLeft, kShiftRightLogical, kShiftRightArithmetic,
  // selection
  kCompare, kSelect, kClamp,
  // dtype
  kConvert, kReducePrecision,
  // complex64 (ops/elementwise.py _real / _imag / _complex / _fft)
  kReal, kImag, kMakeComplex, kFft,
  // bit counting (SWAR, ops/elementwise.py)
  kPopcnt, kClz,
  // shape
  kReshape, kTranspose, kBroadcastInDim, kSlice, kConcatenate, kIota, kPad,
  kReverse,
  // data / structured
  kConstant, kReduce, kArgReduce, kGenericReduce, kReduceWindow, kDotGeneral,
  kBitcastConvert, kDynamicSlice, kDynamicUpdateSlice, kSort, kLexSort, kTopK,
  kApproxTopK,
  kGather, kScatter, kSelectAndScatter, kRng, kConv,
  // control flow (M3): each carries its regions as sub-Programs
  kWhile, kIf, kCase,
  // recognizer emits (M4): a REWRITTEN program's roots. These are not
  // StableHLO ops — src/metaljax/tape.py resolves the recognizer's plan
  // (metaljax.interpreter._rewrite_plan) at lowering time and asks for
  // them by the pseudo-names below, exactly as it does for the argmax
  // reduce. The absorbed ops never reach the tape at all.
  kQmm, kSdpa, kSdpaMask,
  kMoeEIdx, kMoeTIdx, kMoeGather, kMoeConcat, kMoeView, kMoeDot, kMoeTail,
  // M5b: a counted loop msl_scan planned into one generated Metal kernel,
  // and a site where the handler computes on the HOST. Both are lowered by
  // src/metaljax/tape.py; the pseudo-names below are how it asks for them.
  kMslScan, kHostCall,
  // Ordered-effect tokens: no data, only order (dtypes.token_value).
  kToken,
};

// --------------------------------------------------------------------------
// dtypes (dtypes.cc)
// --------------------------------------------------------------------------
//
// The table itself is dtypes.cc's; what crosses is the decode and the
// predicates the handlers branch on. `weak`/`weak_int` are the reason a
// literal in a ported handler never spells `mx::array(0.5)`: MLX's python
// bindings give a scalar the ARRAY's dtype, and matching that is what keeps
// the two engines' result BITS equal.

mx::Dtype dtype_of(int64_t code);

// The emulated grids (dtypes.py `EMULATED`): i4/ui4 and the f8/f6/f4 formats,
// whose values live EXACTLY in a wider storage dtype (`dtype_of` returns that
// storage, so every handler that builds a result of the declared type gets it
// right without knowing the grid exists). What the grid is needed for is
// two things and only two: rounding a value onto the grid, and knowing that
// the code names one at all. The LOGICAL bit width bitcast_convert reads and
// the host transfer's encode/decode are the tape BUILDER's, and live there;
// nothing in a replay needs either.
bool is_emulated(int64_t code);
mx::array quantize_emulated(const mx::array& x, int64_t code);

bool is_bool(const mx::Dtype& d);
bool is_float(const mx::Dtype& d);
bool is_complex(const mx::Dtype& d);
bool is_unsigned(const mx::Dtype& d);
bool is_int(const mx::Dtype& d);
mx::Dtype unsigned_of(const mx::Dtype& d);
mx::array weak(double v, const mx::array& a);
mx::array weak_int(int64_t v, const mx::array& a);
mx::array fresh_copy(const mx::array& a);
mx::array total_order_key(const mx::array& x);

// The complex64 rearrangements (C99 Annex G where MLX's kernels are naive).
mx::array make_complex(mx::array re, mx::array im);
mx::array cabs(const mx::array& z);
mx::array expm1_f32(const mx::array& x);
mx::array csqrt(const mx::array& z);

// --------------------------------------------------------------------------
// runtime disciplines (M3)
// --------------------------------------------------------------------------
//
// Every cadence below is a MEASURED value with a crash or a corruption story
// behind it (see src/metaljax/interpreter.py and ops/control.py, which own
// the comments). None of them is re-derived here: `configure` copies them in
// from the Python modules that parse the environment, so the two engines can
// never drift apart on a number the command-buffer lottery is pinned to.

struct Config {
  int64_t eager_flush_bytes = 1024LL << 20;   // METALJAX_EAGER_FLUSH_MB
  int64_t flush_sync_every = 1;               // METALJAX_EAGER_FLUSH_SYNC
  int64_t flush_clear_bytes = 2048LL << 20;   // METALJAX_FLUSH_CLEAR_MB
  int64_t loop_clear_cost = 500000;           // METALJAX_LOOP_CLEAR_COST
  int64_t ingest_clear_bytes = 8LL << 30;     // METALJAX_INGEST_CLEAR_MB
  int64_t while_pipeline = 1;                 // METALJAX_WHILE_PIPELINE
  bool debug = false;                         // METALJAX_DEBUG
  bool memdbg = false;                        // METALJAX_MEMDBG
};

// The values, set once by `configure` (config.cc); the disciplines that
// spend them are runtime.cc's.
extern Config g_cfg;

void configure(int64_t eager_flush_bytes, int64_t flush_sync_every,
               int64_t flush_clear_bytes, int64_t loop_clear_cost,
               int64_t ingest_clear_bytes, int64_t while_pipeline, bool debug,
               bool memdbg);

struct Stats {
  int64_t flushes = 0;         // eager byte-denominated sync points
  int64_t cache_clears = 0;    // ...that returned cache memory to the OS
  int64_t loop_flushes = 0;    // loop sync points
  int64_t loop_clears = 0;     // ...that cleared on the op-unit cadence
  int64_t ingest_bytes = 0;    // device bytes taken in by transfers
  int64_t ingest_clears = 0;   // ...reclamation points the ingest cadence hit
  int64_t limit_retries = 0;   // Metal buffer-limit recoveries
  int64_t compiled_calls = 0;  // replays of a compiled tape (main or body)
  int64_t compiles = 0;        // mx::compile traces built
  int64_t compile_drops = 0;   // compiled paths abandoned after a failure
  int64_t chunk_drops = 0;     // chunked loop replays abandoned
  int64_t unrolls = 0;         // counted loops unrolled into a trace
  int64_t pipelined_loops = 0;  // dynamic whiles that ran pipelined
  int64_t pipelined_steps = 0;  // ...iterations they retired that way
  int64_t serial_loops = 0;     // dynamic whiles that could not pipeline
  int64_t msl_launches = 0;     // generated persistent kernels launched
  int64_t msl_failures = 0;     // ...plans retired to their loops
  int64_t host_calls = 0;       // entries that reacquired the GIL
};

extern Stats g_stats;

// --------------------------------------------------------------------------
// runtime disciplines (runtime.cc, host.cc)
// --------------------------------------------------------------------------

bool is_resource_limit(const std::exception& e);

// Breaking reference cycles before a buffer-limit retry is an embedder's
// business, not the tape's: what pins the buffers mx::clear_cache cannot free
// is dead Python objects in refcycles (CLAUDE.md item 19), and Python's cycle
// collector barely triggers under array workloads. The nanobind adapter sets
// this to gc.collect; in a process with no interpreter it stays empty and
// `gc_collect` is a no-op, which is the correct behaviour there — there are no
// cycles to break.
extern std::function<void()> g_gc_hook;
void gc_collect();                              // host.cc: runs the hook
void debug_line(const std::string& line);
void debug_print(const std::string& msg);
void flush_eval(const std::vector<mx::array>& arrays, bool hard);
void loop_eval(const std::vector<mx::array>& arrays);
void loop_account(int64_t cost_units);
void loop_flush(const std::vector<mx::array>& arrays, int64_t cost_units);
void ingest_account(int64_t bytes);
int64_t item_int(const mx::array& a);
bool item_bool(const mx::array& a);
bool loop_item_bool(const mx::array& a);

// A permutation that changes nothing; gather, scatter and msl all ask.
inline bool is_identity_perm(const std::vector<int>& p) {
  for (size_t i = 0; i < p.size(); i++)
    if (p[i] != static_cast<int>(i)) return false;
  return true;
}

// --------------------------------------------------------------------------
// the tape
// --------------------------------------------------------------------------
//
// Attribute layouts, by opcode (all int64; shapes/perms carry their own
// rank so a reader never needs the IR again):
//
//   unary / binary / select / clamp   (none)
//   kCompare            [direction]            0=EQ 1=NE 2=LT 3=LE 4=GT 5=GE
//   kConvert            [dtype, complex->real?]
//   kReducePrecision    [arm, exponent_bits, mantissa_bits]
//                       arm: 0 identity, 1 via bf16, 2 via f16, 3 general
//                       (the general arm's f16/bf16 operands compute in f32
//                       and cast back, which is where `orig` comes from)
//   kReshape            [rank, shape...]
//   kTranspose          [rank, perm...]
//   kBroadcastInDim     [transpose?, in_rank, perm...,
//                        out_rank, interim..., out_shape...]
//   kSlice              [rank, start..., stop..., strides...]
//   kConcatenate        [axis]
//   kIota               [dim, ramp_dtype, dtype, rank, shape...]
//   kConstant           (none, or [rank0_buffer]) — the value rides in
//                       `payload`; the flag says it is a one-element buffer
//                       standing in for a rank-0 f32 MLX would otherwise
//                       bake into Metal source as a lossy %.7g literal
//   kReduce             [kind, ndims, dims...]  kind: 0 sum 1 prod 2 max
//                                               3 min 4 any 5 all
//   kArgReduce          [is_max, dim]           two results: (value, index)
//   kShift*             [static?, amount]       see shift_guard
//   kDotGeneral         [lrank, lperm..., rrank, rperm..., B, M, K, N,
//                        out_dtype, out_rank, out_shape..., kind, chunk]
//                       kind: 0 float matmul, 1 exact-f32 K-chunks,
//                             2 int64 outer product, 3 the same in bool
//   kBitcastConvert     [dtype, kind]           kind: 0 same width,
//                                               1 narrowing, 2 widening
//   kDynamicSlice       [rank, clamp bounds..., sizes...]
//   kDynamicUpdateSlice [rank, clamp bounds...]  (sizes = update's shape)
//   kGather             [empty?, out dtype, [out shape],
//                        [batch shape], split?, index_vector_dim,
//                        [slice sizes], <index plan>, [reshape], [perm]]
//                       empty? stops the attrs right there: the result is
//                       zeros of the declared shape
//   kScatter            [method, strategy, [batch shape], split?,
//                        index_vector_dim, <index plan>,
//                        [update slice shape], [updates perm],
//                        [updates shape],
//                        strategy 1: [mask shape] (the neutral value is in
//                                    `payload`)
//                        strategy 2: pad position, pad width, extent]
//                       method: 0 set 1 add 2 mul 3 max 4 min 5 sub
//                               6 complex multiply, as gather-multiply-set
//                               7 an APPLY body under `unique_indices`,
//                                 likewise: gather the current values, run
//                                 region 0 on (old, update), set
//                               8 the same body with NO promise: one update
//                                 at a time, in row-major update order
//                       strategy: the OOB-drop rule — 0 none, 1 neutral
//                       value, 2 dummy pad. Second, ahead of everything
//                       variable-length, so it can be read at a glance.
//                       methods 7 and 8 append [ncaps] right after [updates
//                       shape]
//                       and ahead of the strategy's own extras, which is
//                       where the handler's cursor is when it needs it; its
//                       captures are the trailing `ins`, after the three
//                       scatter operands.
//   kSelectAndScatter   [rank, window dims..., strides..., pad lo/hi pairs...,
//                        is_max, comb]   comb: 0 add 1 or 2 and
//                       (the select is a compare, the scatter an add/or/and:
//                        both bodies are read structurally at lowering, as
//                        ops/reduction.py reads them)
//   <index plan>        n, then n quads (kind, a, b, operand axis):
//                       kind 0 = index-vector component `a`, clamped to
//                       [0, b]; kind 1 = an iota of length `a` at batch
//                       position `b`; kind 2 = a constant zero
//   kWhile              [ncarry, ncond_caps, nbody_caps, counted, k,
//                        bound_kind, bound, cost, period, chunkable, kmax,
//                        body_compile_max]
//                       regions [cond, body]; ins [carry..., cond caps...,
//                       body caps...]; bound_kind 0 static N, 1 carry index,
//                       2 index into the cond's captures
//   kIf / kCase         [ncaps_0, ncaps_1, ...] one per region
//                       regions = the branches; ins [pred/index, caps...]
//   kConv               documented beside its handler (ops_conv.cc): three
//                       layout permutations, the window attributes, and the
//                       arm the result dtype selected
//   kQmm/kSdpa/kMoe*    documented beside their handlers — the M4 emits'
//                       layouts are long enough to want reading in place,
//                       and they are read with a Cursor, not by index
//
// A region is a Program of its own, whose arguments are the region block's
// arguments followed by its CAPTURES -- the values it reads from enclosing
// scopes, resolved to parent slots at lowering. That is the whole of the
// nesting: no environment is shared, so a sub-program is compiled, replayed
// and reasoned about exactly like a top-level one.

class Program;
class MslPlan;

// The handler of an op that computes off the device (LAPACK, jax's callbacks:
// ops/callbacks.py). The tape holds it as an opaque callable and knows nothing
// about what is on the other side -- today the nanobind adapter wraps a Python
// handler into one of these and takes the GIL INSIDE the wrapper, which is the
// whole of what a host call costs the interpreter-free core.
using HostFn =
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

struct Entry {
  int op;
  std::vector<int> ins;
  std::vector<int> outs;
  std::vector<int64_t> attrs;
  // Float attributes. Only the recognizer emits have any: an attention's
  // scale and a mask sentinel are real numbers, and squeezing them through
  // the integer vector would be a bit-pattern encoding nobody could read.
  std::vector<double> fattrs;
  std::optional<mx::array> payload;  // kConstant only
  // An emulated dtype code whose grid every result of this entry is rounded
  // onto, or -1. Applied by `Program::step`, once, AFTER the family handler
  // has written its results -- deliberately not inside the handlers. The
  // Python engine spends this rule at three sites (the unary wrapper, the
  // binary wrapper and `_convert`, ops/elementwise.py `_regrid` and
  // `_maybe_wrap4`), and a per-site flag here would be a rule a new handler
  // can forget: forgetting it is a WRONG ANSWER, not a decline, since the
  // storage dtype is legal either way. One site cannot be forgotten.
  int64_t regrid = -1;
  std::vector<int> drops;            // slots whose last use is this op
  std::vector<std::shared_ptr<Program>> regions;  // control flow only
  int64_t bytes = 0;  // estimated result bytes, for the eager flush cadence
  // M5b. `msl`: a generated persistent kernel for this loop (the entry
  // still carries the interpreted loop in `regions`, as the fallback).
  // `host`: the handler of an op that computes on the host, the one place a
  // native run leaves the tape (empty for every other opcode).
  std::shared_ptr<MslPlan> msl;
  HostFn host;
};

// Sequential reader for the attribute vectors the M4 emits carry: they are
// long enough (three shape recipes for one attention) that positional
// indices would be unreadable on both sides. The layouts are documented
// beside each handler; tape.py writes them in the same order.
class Cursor {
 public:
  explicit Cursor(const std::vector<int64_t>& at) : at_(at) {}
  int64_t next() {
    if (p_ >= at_.size()) throw std::invalid_argument("tape: attrs underrun");
    return at_[p_++];
  }
  bool flag() { return next() != 0; }
  std::vector<int> vec() {
    int64_t n = next();
    std::vector<int> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) v[i] = static_cast<int>(next());
    return v;
  }
  std::vector<int64_t> vec64() {   // strides, which are int64 in MLX
    int64_t n = next();
    std::vector<int64_t> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) v[i] = next();
    return v;
  }
  bool done() const { return p_ >= at_.size(); }
  mx::Shape shp() {
    int64_t n = next();
    mx::Shape s(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++)
      s[i] = static_cast<mx::ShapeElem>(next());
    return s;
  }

 private:
  const std::vector<int64_t>& at_;
  size_t p_ = 0;
};

// Plans this thread has traced into a compiled graph and never proven. The
// list lives across the whole of one top-level Program::run, which settles
// the call synchronously while it is non-empty; thread-local because a
// trace happens on the calling thread and two threads may run different
// executables at once.
extern thread_local std::vector<MslPlan*> t_msl_pending;

class Program {
 public:
  explicit Program(int num_slots, int num_args);
  ~Program();

  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  void add(int op, std::vector<int> ins, std::vector<int> outs,
           std::vector<int64_t> attrs, std::optional<mx::array> payload,
           std::vector<int> drops,
           std::vector<std::shared_ptr<Program>> regions, int64_t bytes,
           std::vector<double> fattrs, std::shared_ptr<MslPlan> msl,
           HostFn host, int64_t regrid = -1);

  // `copies` are output POSITIONS whose array may not be handed out as it
  // is: it could be one of the program's own constants (which the Program
  // holds for the life of the executable, so two calls would share a
  // buffer) or an input's array reaching an output through no-ops. Which
  // ones those are is a static property tape.py works out; making the copy
  // here is XLA's no-alias contract, the half object identity cannot
  // express across the language boundary.
  void set_outputs(std::vector<int> outs, std::vector<int> copies);

  // Whether this program's tape is traced through mx::compile, and which of
  // its outputs need anchoring when it is. BOTH decided in Python: the cost
  // and byte estimators that gate compilation live there (ops/control.py),
  // and re-deriving them here would be a second opinion nothing keeps in
  // step with the first.
  void set_compile(bool on, std::vector<int> anchors, int64_t max_repeat);

  std::vector<mx::array> run(std::vector<mx::array> inputs);

  // One call of this program's tape: the compiled graph when it has one,
  // the op-by-op walk otherwise. `in_trace` says an enclosing mx::compile
  // is already tracing us, which forbids both a nested compile and any
  // sync point (there is nothing to evaluate: the values are tracers).
  std::vector<mx::array> call(const std::vector<mx::array>& inputs,
                              bool in_trace);

  // `repeat` applications of this program's tape as one compiled graph.
  // Bodies are compiled per repeat count (one chunked replay of K
  // iterations is a different graph from a single step), and each variant
  // gets a cache id of its own.
  const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
  compiled(int repeat);

  bool may_compile(int repeat) const;

  // A compiled path failed at CALL time. Every such failure is permanent
  // for this program (MLX will reject the same graph again), so drop the
  // compiled variants and never build another.
  void drop_compiled();

  bool compiled_dropped() const { return compile_disabled_; }
  bool no_chunk() const { return no_chunk_; }
  void set_no_chunk() { no_chunk_ = true; }

  size_t num_ops() const { return ops_.size(); }
  int num_slots() const { return nslots_; }
  int num_args() const { return nargs_; }

  // opcode -> how many entries carry it, THIS program and its regions.
  // For the tests: it is what says a recognizer's root really became its
  // fused op on the tape rather than the literal chain it replaces —
  // including inside a decode loop's body, which is its own Program.
  void tally(std::map<int, int64_t>& counts) const;

  // Does running this program read anything back to the HOST? Every
  // control-flow op does: a while reads its trip count or its condition, an
  // if/case its predicate. So does a host call: it hands its operands to a
  // handler off the device, and whatever that handler does (print, callback)
  // is done by the time the entry returns. Structure, not policy -- read off
  // the tape itself so it cannot drift from what the tape actually holds.
  //
  // It is what says whether a loop body may be BUILT ahead of the condition
  // that decides whether it runs (run_while's pipelined path). Building a
  // graph is free and pure; a nested host read is neither -- it would
  // evaluate real work at a carry the loop may be about to abandon, which
  // for a nested dynamic while is not even guaranteed to terminate, and for
  // an effect it would be a second print of a line the program never asked
  // for.
  bool reads_host() const;

  // The op-by-op walk. `in_trace` is threaded rather than stored: the same
  // Program is walked both ways (a counted loop unrolls its body into an
  // enclosing trace and replays it eagerly on the next call), and a stored
  // flag would make that a race with itself.
  std::vector<mx::array> interpret(const std::vector<mx::array>& inputs,
                                   bool in_trace);

  // Peak number of slots the environment holds at once, from the drop
  // lists alone. The tests assert on it: it is what says liveness pruning
  // is really running, and a chain whose intermediates are not dropped
  // shows up here as a count that grows with the chain.
  int max_live() const;

 private:
  void check_slot(int s) const;

  // One entry, applied to the environment: hand it to each family in turn
  // (program.cc), then release the slots whose last use it was.
  void step(const Entry& e, std::vector<std::optional<mx::array>>& env,
            bool in_trace) const;

  // The op families. Each switches on the opcodes it owns, returns false
  // for everything else, and lives in the file named beside it; between
  // them they partition the enum above, which is what the `throw` at the
  // end of `step` asserts.
  bool step_elementwise(const Entry& e,
                        std::vector<std::optional<mx::array>>& env,
                        bool in_trace) const;   // ops_elementwise.cc
  bool step_shape(const Entry& e,
                  std::vector<std::optional<mx::array>>& env,
                  bool in_trace) const;   // ops_shape.cc
  bool step_linalg(const Entry& e,
                   std::vector<std::optional<mx::array>>& env,
                   bool in_trace) const;   // ops_linalg.cc
  bool step_reduce(const Entry& e,
                   std::vector<std::optional<mx::array>>& env,
                   bool in_trace) const;   // ops_reduce.cc
  bool step_index(const Entry& e,
                  std::vector<std::optional<mx::array>>& env,
                  bool in_trace) const;   // ops_index.cc
  bool step_emit(const Entry& e,
                 std::vector<std::optional<mx::array>>& env,
                 bool in_trace) const;   // emits.cc
  bool step_control(const Entry& e,
                    std::vector<std::optional<mx::array>>& env,
                    bool in_trace) const;   // control.cc
  bool step_rng(const Entry& e,
                std::vector<std::optional<mx::array>>& env,
                bool in_trace) const;   // ops_rng.cc
  bool step_conv(const Entry& e,
                 std::vector<std::optional<mx::array>>& env,
                 bool in_trace) const;   // ops_conv.cc
  bool step_host(const Entry& e,
                 std::vector<std::optional<mx::array>>& env,
                 bool in_trace) const;   // host.cc

  // A reduce whose fold is an arbitrary body, run pairwise over whole
  // arrays; the schedule is documented at the definition (ops_reduce.cc).
  std::vector<mx::array> generic_reduce(
      const std::vector<mx::array>& inputs,
      const std::vector<mx::array>& inits,
      const std::vector<mx::array>& caps, const std::vector<int>& keep,
      const std::vector<int>& dims, Program* body, bool in_trace) const;

  // A counted loop msl_scan planned into one generated kernel, and the
  // loop it falls back to when the kernel dies (msl.cc, control.cc).
  void run_msl(const Entry& e, std::vector<std::optional<mx::array>>& env,
               bool in_trace) const;
  void run_while(const Entry& e, std::vector<std::optional<mx::array>>& env,
                 bool in_trace) const;

  static void write_results(const Entry& e,
                            std::vector<std::optional<mx::array>>& env,
                            const std::vector<mx::array>& vals);

  // Settle a call that traced a kernel MLX has never built, and retire
  // the plan if it cannot be (msl.cc).
  void settle_msl(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outs);

  // Whether anything in this program (or its regions) calls back into
  // Python. Structure, not policy -- read off the tape, like reads_host.
  bool has_host() const;

  // Drop every compiled graph here and in the regions (compile.cc).
  void drop_compiled_deep();

  // engine.execute's recovery policy, on the native side of the boundary:
  // a compiled path that fails hands the program to the eager one (which
  // is always correct), and buffer exhaustion clears and retries once.
  // Programs are pure, so a rerun is safe.
  std::vector<mx::array> run_recovering(
      const std::vector<mx::array>& inputs);

  // Readers for the shape and axis vectors an entry's attrs carry.
  static mx::Shape shape(const std::vector<int64_t>& at, size_t off,
                         int64_t n);
  static std::vector<int> axes(const std::vector<int64_t>& at, size_t off,
                               int64_t n);

  struct Compiled {
    std::uintptr_t id = 0;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> fn;
  };

  int nslots_;
  int nargs_;
  std::vector<Entry> ops_;
  std::vector<int> outputs_;
  std::vector<int> copies_;
  // M3: the compiled path. `compile_` and `anchors_` are Python's decision
  // (see set_compile); everything else is run-time state that only ever
  // moves one way -- toward the eager path, which is always correct.
  bool compile_ = false;
  bool compile_disabled_ = false;
  bool compile_probe_ = true;   // settle the first compiled call (run_recovering)
  bool no_chunk_ = false;
  mutable int reads_host_ = -1;   // lazily derived, never un-derived
  int64_t max_repeat_ = 1;
  std::vector<int> anchors_;
  std::map<int, Compiled> compiled_;
  // `run` mutates all of the run-time state above -- the compiled-graph
  // cache above all -- so two threads calling the SAME executable would race
  // on it (jax lets one jitted function be called from any number of
  // threads). One lock per program serializes those and nothing else;
  // distinct executables still run concurrently. An embedder that holds an
  // interpreter lock must drop it before calling `run`, never after: a
  // waiter here holding the GIL would deadlock against the holder's recovery
  // paths, which reacquire it through `g_gc_hook`. bindings.cc does exactly
  // that.
  std::mutex lock_;
};

// The registries a tape builder reads (config.cc, dtypes.cc): a name absent
// from either declines the program rather than guessing. Plain pairs, in
// table order; bindings.cc is what turns them into the dicts Python sees.
std::vector<std::pair<std::string, int>> opcodes();
std::vector<std::pair<std::string, int>> dtype_codes();

}  // namespace metaljax

#endif  // METALJAX_PROGRAM_H_
