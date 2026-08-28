// metaljax native engine — the registries and the cadences.
//
// Three things a tape builder asks of the engine, and nothing a replay
// touches: the opcode registry (an op name absent from it declines the WHOLE
// program, which is why there is no second table anywhere), the runtime
// cadences copied in from the modules that parse the environment, and the
// counters. bindings.cc is what puts a Python face on them.

#include "program.h"

namespace metaljax {

Config g_cfg;
Stats g_stats;
std::function<void()> g_gc_hook;

// StableHLO name -> opcode. Several names may share an opcode (chlo.erf is
// stablehlo.erf's handler verbatim). The emulated-dtype regrid the Python
// unary/binary wrappers apply is not here either: it is not an opcode but a
// per-entry field (`Entry::regrid`), so that the ONE site that spends it is
// `Program::step` rather than every handler that might produce a value on a
// grid.
struct NamedOp { const char* name; int op; };

const NamedOp kOpNames[] = {
    {"stablehlo.abs", kAbs},
    {"stablehlo.cbrt", kCbrt},
    {"stablehlo.ceil", kCeil},
    {"stablehlo.cosine", kCos},
    {"stablehlo.erf", kErf},
    {"chlo.erf", kErf},
    {"stablehlo.erf_inv", kErfInv},
    {"chlo.erf_inv", kErfInv},
    {"stablehlo.exponential", kExp},
    {"stablehlo.exponential_minus_one", kExpm1},
    {"stablehlo.floor", kFloor},
    {"stablehlo.is_finite", kIsFinite},
    {"stablehlo.log", kLog},
    {"stablehlo.log_plus_one", kLog1p},
    {"stablehlo.logistic", kLogistic},
    {"stablehlo.negate", kNegate},
    {"stablehlo.not", kNot},
    {"stablehlo.round_nearest_afz", kRoundAfz},
    {"stablehlo.round_nearest_even", kRoundEven},
    {"stablehlo.rsqrt", kRsqrt},
    {"stablehlo.sign", kSign},
    {"stablehlo.sine", kSin},
    {"stablehlo.sqrt", kSqrt},
    {"stablehlo.tan", kTan},
    {"stablehlo.tanh", kTanh},
    {"chlo.square", kSquare},
    {"stablehlo.add", kAdd},
    {"stablehlo.multiply", kMultiply},
    {"stablehlo.subtract", kSubtract},
    {"stablehlo.maximum", kMaximum},
    {"stablehlo.minimum", kMinimum},
    {"stablehlo.and", kAnd},
    {"stablehlo.or", kOr},
    {"stablehlo.xor", kXor},
    {"stablehlo.divide", kDivide},
    {"stablehlo.remainder", kRemainder},
    {"stablehlo.power", kPower},
    {"stablehlo.atan2", kAtan2},
    {"stablehlo.shift_left", kShiftLeft},
    {"stablehlo.shift_right_logical", kShiftRightLogical},
    {"stablehlo.shift_right_arithmetic", kShiftRightArithmetic},
    {"stablehlo.compare", kCompare},
    {"stablehlo.select", kSelect},
    {"stablehlo.clamp", kClamp},
    {"stablehlo.convert", kConvert},
    {"stablehlo.reduce_precision", kReducePrecision},
    {"stablehlo.real", kReal},
    {"stablehlo.imag", kImag},
    {"stablehlo.complex", kMakeComplex},
    {"stablehlo.fft", kFft},
    {"stablehlo.reshape", kReshape},
    {"stablehlo.transpose", kTranspose},
    {"stablehlo.broadcast_in_dim", kBroadcastInDim},
    {"stablehlo.slice", kSlice},
    {"stablehlo.concatenate", kConcatenate},
    {"stablehlo.iota", kIota},
    {"stablehlo.pad", kPad},
    {"stablehlo.reverse", kReverse},
    {"stablehlo.popcnt", kPopcnt},
    {"stablehlo.count_leading_zeros", kClz},
    {"stablehlo.constant", kConstant},
    {"stablehlo.reduce", kReduce},
    // ops/reduction.py read ONE stablehlo.reduce two ways depending on the
    // body — the single-operand monoid and the (values, indices) pair jax
    // lowers argmax/argmin to. tape.py decided which, then asked for the
    // opcode by this pseudo-name; C++ still owns both enum values.
    {"stablehlo.reduce.arg_pair", kArgReduce},
    // ...and a third way: a body neither table recognizes runs on whole
    // arrays, pairwise (ops/reduction.py _generic_reduce). The body is a
    // sub-Program; the halving schedule is this handler's.
    {"stablehlo.reduce.generic", kGenericReduce},
    {"stablehlo.reduce_window", kReduceWindow},
    {"stablehlo.dot_general", kDotGeneral},
    {"stablehlo.bitcast_convert", kBitcastConvert},
    {"stablehlo.dynamic_slice", kDynamicSlice},
    {"stablehlo.dynamic_update_slice", kDynamicUpdateSlice},
    {"stablehlo.sort", kSort},
    // ops/sort.py `_lex_sorted`: the LEXICOGRAPHIC sort, whose comparator is
    // a select tree over several key pairs rather than one compare. A
    // different execution shape (successive stable argsorts threaded through
    // a permutation), so a different opcode -- and a pseudo-name, like the
    // conv, because Stage 1's src/metaljax/tape.py (deleted 0.11.6, ef5774d)
    // had no lowering for it and kept declining the shape to the Python
    // engine.
    {"metaljax.lex_sort", kLexSort},
    // chlo.top_k survives a direct jax lowering; through a portable
    // artifact it arrives already decomposed into the sort above. Both
    // reach the plugin, so both are here.
    {"chlo.top_k", kTopK},
    // XLA's ApproxTopK custom call (jax.lax.approx_max_k/approx_min_k), whose
    // contract an EXACT top-k satisfies: recall 1.0 is >= any recall_target.
    // A pseudo-name, like the conv: the target's own name is what the plugin
    // matches on, and the attribute layout is this tape's.
    {"metaljax.approx_top_k", kApproxTopK},
    // StableHLO's gather/scatter go STRAIGHT to MLX's primitives (see the
    // handlers): the numpy-style indexing the Python handlers use lives in
    // MLX's python layer and has no C++ entry point, and routing through a
    // reimplementation of it would be a lossy second translation of
    // semantics the primitives already have.
    {"stablehlo.gather", kGather},
    {"stablehlo.scatter", kScatter},
    // ops/reduction.py `_select_and_scatter`: max/min-pool backward. Its two
    // regions are read STRUCTURALLY at lowering (a compare, then an
    // add/or/and), like the scatter combiner and unlike a generic reduce
    // body, so the entry carries no sub-Program.
    {"stablehlo.select_and_scatter", kSelectAndScatter},
    {"stablehlo.rng_bit_generator", kRng},
    // ops/conv.py, ported for phase 2 (P7) — under a PSEUDO-NAME, and that
    // is load-bearing. A tape builder with no `_lower_*` of its own emitted
    // an op it found in this table with an EMPTY attribute vector (tape.py's
    // `handler(...) if handler else ([], None)`), and this handler reads a
    // long one. src/metaljax/tape.py had no convolution lowering, so
    // `stablehlo.convolution` stayed absent from the table: Stage 1 kept
    // declining the op to the Python engine, at compile time, where it
    // belonged. The phase-2 plugin knows the layout and asks by this name.
    {"metaljax.conv", kConv},
    {"stablehlo.while", kWhile},
    {"stablehlo.if", kIf},
    {"stablehlo.case", kCase},
    // M4 recognizer emits. Pseudo-names: a build that predates a given
    // emit simply does not offer it, and tape.py declined the program —
    // which is why there is no version negotiation anywhere else.
    {"metaljax.qmm", kQmm},
    {"metaljax.sdpa", kSdpa},
    {"metaljax.sdpa.mask", kSdpaMask},
    {"metaljax.moe.eidx", kMoeEIdx},
    {"metaljax.moe.tidx", kMoeTIdx},
    {"metaljax.moe.gather", kMoeGather},
    {"metaljax.moe.concat", kMoeConcat},
    {"metaljax.moe.view", kMoeView},
    {"metaljax.moe.dot", kMoeDot},
    {"metaljax.moe.tail", kMoeTail},
    {"metaljax.ragged_dot", kRaggedDot},
    // M5b pseudo-names.
    {"metaljax.msl_scan", kMslScan},
    {"metaljax.host_call", kHostCall},
    {"stablehlo.create_token", kToken},
    {"stablehlo.after_all", kToken},
};

std::vector<std::pair<std::string, int>> opcodes() {
  std::vector<std::pair<std::string, int>> v;
  v.reserve(sizeof(kOpNames) / sizeof(kOpNames[0]));
  for (const NamedOp& n : kOpNames) v.emplace_back(n.name, n.op);
  return v;
}

// The runtime cadences, copied in from the Python modules that parse the
// environment (metaljax.interpreter and metaljax.ops.control). One source of
// truth per number: a second env-var reader here would be a second opinion,
// and these are values the command-buffer lottery is pinned to.
void configure(int64_t eager_flush_bytes, int64_t flush_sync_every,
               int64_t flush_clear_bytes, int64_t flush_footprint_bytes,
               int64_t flush_floor_bytes, int64_t flush_main_flushes,
               int64_t flush_earn_mult, int64_t loop_clear_cost,
               int64_t ingest_clear_bytes, int64_t while_pipeline, bool debug,
               bool memdbg) {
  g_cfg.eager_flush_bytes = eager_flush_bytes;
  g_cfg.flush_sync_every = flush_sync_every;
  g_cfg.flush_clear_bytes = flush_clear_bytes;
  g_cfg.flush_footprint_bytes = flush_footprint_bytes;
  g_cfg.flush_floor_bytes = flush_floor_bytes;
  g_cfg.flush_main_flushes = flush_main_flushes;
  g_cfg.flush_earn_mult = flush_earn_mult;
  g_cfg.loop_clear_cost = loop_clear_cost;
  g_cfg.ingest_clear_bytes = ingest_clear_bytes;
  g_cfg.while_pipeline = while_pipeline;
  g_cfg.debug = debug;
  g_cfg.memdbg = memdbg;

  // NOT DONE HERE, and measured (P25, notes/cpp-p25-cache-limit.md):
  // `mx::set_cache_limit(flush_clear_bytes)` at client construction -- a
  // GLOBAL pool bound -- is the shape the P20 root cause proposed, and it
  // costs more than it buys. It bounds every path, including the ones that
  // never reach an eager flush and were never bounded before: the E2B
  // keras-int4 decode row went **80.7 -> 190.0 ms/tok (2.35x)** under a
  // 2 GB global limit, because a compiled decode step whose transients
  // exceed the bound then re-allocates from the OS on every step (and the
  // texmo suite's `mid` class went with it, +11.8%). It bought 1.10x on the
  // row it was aimed at, where the trim below buys 1.17x.
  //
  // What ships instead is the same trim at the SAME cadence the clear had:
  // `program.cc::eager_flush` -> `trim_cache`, which bounds the pool where
  // an eager phase would otherwise claim its whole traffic and leaves every
  // other path exactly as it was.
  //
  // NOR IS THE WATERMARK ONE NUMBER (P27, notes/cpp-p27-flush-pressure.md).
  // P25's sweep left a straight memory-for-speed trade with no setting that
  // satisfied both ends: the maxtext training row reaches its anchor only at
  // a 32 GB pool, and handing that watermark to every program guard-kills
  // the LoRA row during its load. `runtime.cc::flush_bound` resolves it by
  // asking two questions the watermark alone could not -- whether the flush
  // belongs to an eager MAIN (`flush_main_flushes`, a program that has
  // flushed enough times to be reusing a pool rather than filling one), and
  // whether the process has the FOOTPRINT to spare (`flush_footprint_bytes`)
  // -- with P25's shipped 2048 MB as the floor under both, which is why all
  // four numbers arrive here rather than one.
  //
  // ...AND NEITHER OF THOSE ASKS WHETHER THE POOL IS BEING USED (P28,
  // notes/cpp-p28-benefit-gate.md). Both maxtext DECODE rows are past the
  // gate (their checkpoint load is one program taking 134 flushes in a single
  // call) and have footprint to spare (their live set is ~2 GB), so P27 hands
  // them the whole cap -- and the 14 GB of weights their load frees at its
  // last flush then stands in the pool for the rest of the process, 17 GB and
  // 11 GB of peak footprint for no speed at all. `flush_earn_mult` is the
  // third rule: a program may keep a multiple of the live-set SWING it has
  // demonstrated, because that swing is the only thing a trim can cost it.
}

}  // namespace metaljax
