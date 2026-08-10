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
// stablehlo.erf's handler verbatim); the emulated-dtype regrid the Python
// unary/binary wrappers apply is deliberately absent, because every element
// type that could trigger it is declined in tape.py.
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
    // ops/reduction.py reads ONE stablehlo.reduce two ways depending on the
    // body — the single-operand monoid and the (values, indices) pair jax
    // lowers argmax/argmin to. tape.py decides which, then asks for the
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
    // chlo.top_k survives a direct jax lowering; through a portable
    // artifact it arrives already decomposed into the sort above. Both
    // reach the plugin, so both are here.
    {"chlo.top_k", kTopK},
    // StableHLO's gather/scatter go STRAIGHT to MLX's primitives (see the
    // handlers): the numpy-style indexing the Python handlers use lives in
    // MLX's python layer and has no C++ entry point, and routing through a
    // reimplementation of it would be a lossy second translation of
    // semantics the primitives already have.
    {"stablehlo.gather", kGather},
    {"stablehlo.scatter", kScatter},
    {"stablehlo.rng_bit_generator", kRng},
    {"stablehlo.while", kWhile},
    {"stablehlo.if", kIf},
    {"stablehlo.case", kCase},
    // M4 recognizer emits. Pseudo-names: a build that predates a given
    // emit simply does not offer it, and tape.py declines the program —
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
               int64_t flush_clear_bytes, int64_t loop_clear_cost,
               int64_t while_pipeline, bool debug, bool memdbg) {
  g_cfg.eager_flush_bytes = eager_flush_bytes;
  g_cfg.flush_sync_every = flush_sync_every;
  g_cfg.flush_clear_bytes = flush_clear_bytes;
  g_cfg.loop_clear_cost = loop_clear_cost;
  g_cfg.while_pipeline = while_pipeline;
  g_cfg.debug = debug;
  g_cfg.memdbg = memdbg;
}

}  // namespace metaljax
