/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

StableHLO -> the executor's tape, in C++.

`LowerModule` walks the main function's single block once and builds the
`metaljax::Program` (native/program.h) that replays it: SSA values become
integer slots, attributes become the integer vectors the C++ handlers decode
with a `Cursor`, and constants cross to the device here, once.

The attribute layouts are NOT invented here.  Every one of them is the layout
src/metaljax/tape.py already writes and native/ops_*.cc already reads, because
both plugins share one executor: a disagreement would be a wrong answer rather
than a build error.  Where this file and tape.py could drift, tape.py is the
specification and the handler's `Cursor` reads are the ground truth.

The discipline is Stage 1's, unchanged: anything outside the supported set
declines the WHOLE program, naming the op, so a gap is a missing feature and
never a wrong number.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_LOWERING_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_LOWERING_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlx/dtype.h"
#include "program.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace metaljax {

// One argument or result of a lowered program: what PJRT calls it and what
// the device holds.  The two element types agree for everything that lowers
// (f64 has no dtype code, so an f64 program declines before it gets here).
struct ValueSpec {
  xla::PrimitiveType type;
  std::vector<int64_t> dims;
  mlx::core::Dtype dtype;
  // Which of the client's memory spaces this value belongs to: false for the
  // default "device", true for "pinned_host" (`mhlo.memory_kind` on main's
  // argument or result).  Apple silicon's pool is unified, so nothing about
  // the allocation changes -- the flag exists so the buffer handed back points
  // at the space the CALLER asked for, and jax's `sharding.memory_kind` and
  // `GetOutputMemoryKinds` answer with it instead of assuming "device".
  bool host_memory = false;

  xla::Shape shape() const;
  int64_t element_count() const;
};

struct LoweredProgram {
  std::shared_ptr<Program> program;
  std::vector<ValueSpec> parameters;
  std::vector<ValueSpec> results;
  // P13: the parameters the caller DONATES -- `tf.aliasing_output` (an input
  // XLA may write its output into) and `jax.buffer_donor` (one it may reuse
  // for anything).  XLA's contract is that the caller may not touch such a
  // buffer again, and jax relies on it: the executable deletes them.
  std::vector<int> donated_parameters;
  // Diagnostics: how many tape entries the program holds, and how many of
  // its outputs are copied out of the environment because they may alias an
  // argument or a constant the Program owns (XLA's no-alias contract).
  int64_t num_entries = 0;
  int64_t num_copies = 0;
  // Whether the whole tape is traced through mx::compile (the P5 decision;
  // while BODIES carry theirs in the entry's attrs instead).
  bool compiled = false;
  // The StableHLO this tape was built from, as MLIR bytecode -- the program
  // `GetHloModules` (PJRT's `OptimizedProgram`) hands back, converted to HLO
  // on demand.  Kept serialized rather than as a live module because the
  // module and its MLIRContext belong to the compile call, and kept at ALL
  // because a debugging surface that answers nothing is one jax turns into
  // `None` and every caller then trips over.
  std::string stablehlo;
};

absl::StatusOr<LoweredProgram> LowerModule(mlir::ModuleOp module);

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_LOWERING_H_
