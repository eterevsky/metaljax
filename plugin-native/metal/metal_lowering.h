/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

StableHLO -> the executor's tape, in C++.

`LowerModule` walks the main function's single block once and builds the
`metaljax::Program` (native/program.h) that replays it: SSA values become
integer slots, attributes become the integer vectors the C++ handlers decode
with a `Cursor`, and constants cross to the device here, once.

The attribute layouts are NOT invented here.  Every one of them is the layout
Stage 1's src/metaljax/tape.py (deleted 0.11.6, ef5774d) already wrote and
native/ops_*.cc (now runtime/) already read, because both plugins shared one
executor: a disagreement would be a wrong answer rather than a build error.
While tape.py lived, where this file and it could drift, tape.py was the
specification; the handler's `Cursor` reads remain the ground truth.

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
  // The outputs the copy list EXEMPTS because every argument they may alias
  // is donated (Stage 1 engine.py `_dealias`: "aliasing is exactly what
  // donation licenses"), each with those arguments.  Donation is retractable per CALL
  // through `non_donatable_input_indices`, and an output aliasing a retracted
  // argument has to be copied after all -- which is `RunOnce`'s job, since
  // only the call knows.  Empty on every program that donates nothing.
  std::vector<std::pair<int, std::vector<int>>> donated_output_aliases;
  // Whether the whole tape is traced through mx::compile (the P5 decision;
  // while BODIES carry theirs in the entry's attrs instead).
  bool compiled = false;
  // P17: the recognizer emits' packed weights, which are trailing INPUTS of
  // the tape (never constants -- mx::compile bakes a captured constant by
  // value, and a repack would then never be seen).  `Execute` appends them to
  // the caller's arguments; `pack_args` names the @main arguments they were
  // built from and `pack_arg_ids` the arrays those held at build time, which
  // is what says a later call has to repack.
  std::vector<mlx::core::array> packs;
  std::vector<int> pack_args;
  std::vector<std::uintptr_t> pack_arg_ids;
  // ...and a weak handle on each of those arrays' buffers with its raw
  // address.  `id()` is the address of a refcounted descriptor, and address
  // recycling applies to it exactly as it does to a CPython object: a caller
  // that hands over FRESH weights each call (training; a donated weight the
  // executable freed) can land the new array on the old descriptor's
  // address, and an id-only check then reused a stale pack -- measured on
  // the B6 contract's donated pair (calls 2 and 3 off by 4.5, silently).
  // The weak handle proves the buffer is still alive and the raw pointer
  // that it is the SAME buffer (qmm's BuiltCache `LeafMatches`, which this
  // repeats at the executable's own boundary).
  std::vector<std::weak_ptr<mlx::core::array::Data>> pack_arg_data;
  std::vector<const void*> pack_arg_raw;
  // The `pack_args` ONLY the projection packs (B6) key on, sorted: when a
  // call changes one of these and nothing else, the executable repacks a
  // bounded number of times and then re-lowers WITHOUT the projection
  // packs, keeping every other recognizer (`Tape`, `kMaxProjRepacks`).
  std::vector<int> proj_only_args;
  // How many fused entries the tape holds, per family (diagnostics, and what
  // the tests assert on).
  int64_t num_qmm = 0;
  int64_t num_moe = 0;
  int64_t num_sdpa = 0;
  int64_t num_ragged = 0;
  int64_t num_ragged_decode = 0;
  int64_t num_stacked = 0;
  int64_t num_stacked_pack = 0;
  int64_t num_gdn = 0;
  int64_t num_rope = 0;
  int64_t num_proj = 0;
  // The StableHLO this tape was built from, as MLIR bytecode -- the program
  // `GetHloModules` (PJRT's `OptimizedProgram`) hands back, converted to HLO
  // on demand.  Kept serialized rather than as a live module because the
  // module and its MLIRContext belong to the compile call, and kept at ALL
  // because a debugging surface that answers nothing is one jax turns into
  // `None` and every caller then trips over.
  std::string stablehlo;
};

absl::StatusOr<LoweredProgram> LowerModule(mlir::ModuleOp module);

// The same module lowered again with the recognizers ON, against the concrete
// arguments of an execute (P17).  Packing needs real buffers, so this runs at
// the FIRST call rather than at compile: the executable keeps the plain tape
// `LowerModule` built, asks for this one once, and keeps whichever it got --
// a module that recognizes nothing, or a pack that fails an exactness check,
// comes back `NotFound` and the plain tape stays.  `proj_packs` = false
// leaves the projection packs (B6) out: the executable asks for that after
// their weights changed too often (`kMaxProjRepacks`).
absl::StatusOr<LoweredProgram> LowerModuleFused(
    mlir::ModuleOp module, const std::vector<mlx::core::array>& args,
    bool proj_packs = true);

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_LOWERING_H_
