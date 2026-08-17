/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_executable.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "metal/metal_buffer.h"
#include "metal/metal_names.h"
#include "metal/metal_recognize.h"
#include "metal/metal_stream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlx/mlx.h"
#include "program.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/layout_util.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/utils.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace metaljax {

namespace {

namespace mx = mlx::core;

bool SameShape(const mx::array& a, const ValueSpec& spec) {
  if (a.shape().size() != spec.dims.size()) return false;
  for (size_t i = 0; i < spec.dims.size(); i++)
    if (static_cast<int64_t>(a.shape()[i]) != spec.dims[i]) return false;
  return true;
}

std::string ShapeString(const mx::array& a) {
  std::string s = "[";
  for (size_t i = 0; i < a.shape().size(); i++)
    absl::StrAppend(&s, i ? "," : "", a.shape()[i]);
  return absl::StrCat(s, "]");
}

// METALJAX_DEBUG=1 reports what one execute spent of the runtime's cadences.
// The counters are the executor's own (native/program.h `g_stats`), so this
// is the only window a process with no interpreter in it has on them -- and
// the loop counters are what say the flush discipline a long loop depends on
// really engaged.  Process-wide and unsynchronized, like the counters
// themselves: with two executes in flight the delta below attributes some of
// the other one's work, which is a diagnostic's business and not a run's.
const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();

// METALJAX_VERIFY_COMPILE=1 compares the two paths; =dump also prints the
// arguments and both answers of whatever diverged.
const char* const kVerifyCompileEnv = std::getenv("METALJAX_VERIFY_COMPILE");
const bool kVerifyCompile = kVerifyCompileEnv != nullptr;
const bool kVerifyDump =
    kVerifyCompile && std::string(kVerifyCompileEnv) == "dump";

std::string StatsDelta(const Stats& before, const Stats& after) {
  return absl::StrFormat(
      // `trim` is how many hard flushes trimmed MLX's pool back to
      // `METALJAX_FLUSH_CLEAR_MB` -- the excess only, where this used to
      // read `clear` and dump the whole pool (P25).
      "flushes=%d(+trim %d) loop_flushes=%d(+clear %d) ingest=%dMB(+clear %d) "
      "limit_retries=%d "
      "serial_loops=%d pipelined_loops=%d pipelined_steps=%d "
      "compiles=%d compiled_calls=%d unrolls=%d drops=%d/%d",
      after.flushes - before.flushes,
      after.cache_trims - before.cache_trims,
      after.loop_flushes - before.loop_flushes,
      after.loop_clears - before.loop_clears,
      // Not this execute's work -- transfers happen BETWEEN executes -- but
      // the delta says how much a load moved since the last program ran,
      // which is the only place a flight log can read it.
      (after.ingest_bytes - before.ingest_bytes) >> 20,
      after.ingest_clears - before.ingest_clears,
      after.limit_retries - before.limit_retries,
      after.serial_loops - before.serial_loops,
      after.pipelined_loops - before.pipelined_loops,
      after.pipelined_steps - before.pipelined_steps,
      // P5: the compiled path is otherwise invisible in a process with no
      // interpreter -- `compiles` counts traces BUILT, `compiled_calls`
      // replays (main, a while body, one chunk), and `drops` the two ways a
      // compiled path retires (compile_drops / chunk_drops).
      after.compiles - before.compiles,
      after.compiled_calls - before.compiled_calls,
      after.unrolls - before.unrolls,
      after.compile_drops - before.compile_drops,
      after.chunk_drops - before.chunk_drops);
}

}  // namespace

MetalLoadedExecutable::MetalLoadedExecutable(
    xla::PjRtClient* client, xla::PjRtDevice* device,
    xla::PjRtMemorySpace* memory_space, LoweredProgram lowered,
    xla::CompileOptions options, std::string name)
    : client_(client),
      device_(device),
      memory_space_(memory_space),
      lowered_(std::make_shared<const LoweredProgram>(std::move(lowered))),
      name_(std::move(name)),
      executable_(std::make_unique<MetalExecutable>(lowered_,
                                                    std::move(options), name_)),
      device_assignment_(/*replica_count=*/1, /*computation_count=*/1) {
  // The space a result annotated `mhlo.memory_kind = "pinned_host"` is handed
  // back on.  Same unified pool as `memory_space_`, so this is which SPACE the
  // buffer names and nothing else; if the client has no such space the result
  // stays where every other one goes.
  absl::StatusOr<xla::PjRtMemorySpace*> host =
      device->memory_space_by_kind(kHostMemoryKind);
  host_memory_space_ = host.ok() ? *host : memory_space;
  device_assignment_(0, 0) = device->id();
  logical_ids_.push_back(LogicalDeviceIds{/*replica=*/0, /*partition=*/0});
  devices_.push_back(device);
}

const std::shared_ptr<const LoweredProgram>& MetalLoadedExecutable::Tape(
    const std::vector<mx::array>& inputs) const {
  std::lock_guard<std::mutex> lock(fuse_mu_);
  if (fused_ != nullptr) {
    // The pack is a pure function of the buffers its reconstruction read: a
    // call that hands over the same arrays reuses it, and one that does not
    // has to repack (qmm.py `_Pack.matches`).
    bool same = true;
    for (size_t i = 0; i < fused_->pack_args.size() && same; i++) {
      const int a = fused_->pack_args[i];
      same = a < static_cast<int>(inputs.size()) &&
             inputs[a].id() == fused_->pack_arg_ids[i];
    }
    if (same) return fused_;
    fused_ = nullptr;
    if (++repacks_ > kMaxRepacks) {
      // Weights being trained, or a "scale" that is really per-call data.
      fuse_done_ = true;
      return lowered_;
    }
  } else if (fuse_done_) {
    return lowered_;
  }
  fuse_done_ = true;
  if (!RecognizeEnabled() || lowered_->stablehlo.empty()) return lowered_;
  // The module the compile was handed is long gone (it belongs to that call),
  // so the program is re-read from the bytecode this executable kept.  The
  // context lives only as long as the lowering: a tape holds no MLIR.
  mlir::MLIRContext context;
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      xla::ParseMlirModuleString(lowered_->stablehlo, context);
  if (!module.ok()) return lowered_;
  absl::StatusOr<LoweredProgram> fused;
  try {
    fused = LowerModuleFused(**module, inputs);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[metaljax-native] %s: recognizers failed (%s)\n",
                 name_.c_str(), e.what());
    std::fflush(stderr);
    return lowered_;
  }
  if (!fused.ok()) {
    if (kDebug && !absl::IsNotFound(fused.status())) {
      std::fprintf(stderr, "[metaljax-native] %s: no fused tape (%s)\n",
                   name_.c_str(),
                   std::string(fused.status().message()).c_str());
      std::fflush(stderr);
    }
    return lowered_;
  }
  // The fused tape must present the same boundary as the plain one: it is the
  // SAME program, and jax has already been told what this executable takes and
  // returns.  A disagreement would be a lowering bug, and handing back the
  // plain tape is the safe half of it.
  if (fused->parameters.size() != lowered_->parameters.size() ||
      fused->results.size() != lowered_->results.size())
    return lowered_;
  fused->stablehlo = lowered_->stablehlo;
  if (kDebug) {
    std::fprintf(stderr,
                 "[metaljax-native] %s: %lld fused quantized matmul(s), "
                 "%lld gathered expert dispatch(es), %lld fused "
                 "attention(s), %zu packed arrays\n",
                 name_.c_str(), static_cast<long long>(fused->num_qmm),
                 static_cast<long long>(fused->num_moe),
                 static_cast<long long>(fused->num_sdpa), fused->packs.size());
    std::fflush(stderr);
  }
  fused_ = std::make_shared<const LoweredProgram>(std::move(*fused));
  fuse_done_ = false;
  return fused_;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
MetalLoadedExecutable::RunOnce(
    absl::Span<xla::PjRtBuffer* const> argument_handles,
    const xla::ExecuteOptions& options) const {
  if (deleted_) {
    return absl::FailedPreconditionError(
        "metaljax-native: the executable has been deleted.");
  }
  if (argument_handles.size() != lowered_->parameters.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "metaljax-native: %s takes %d arguments, got %d", name_,
        lowered_->parameters.size(), argument_handles.size()));
  }
  BindThread();
  std::unique_lock<std::recursive_mutex> submission = SubmissionLock();

  // The governor's look at the machine before a program is entered (the
  // no-panic contract, runtime/memory.cc).  Sampled, so a decode loop paying
  // it once per token pays a compare; what it catches is the process that is
  // ALREADY past a hard line -- refusing to start is the difference between
  // an error and a program that allocates its way into a wedge with no sync
  // point in reach.  `want` is 0 because a tape's peak is not knowable here:
  // its flushes carry the check from then on.
  try {
    governor_admit(0, MemWhere::kExecute);
  } catch (const std::exception& e) {
    return absl::ResourceExhaustedError(e.what());
  }

  // XLA's donation contract is per CALL, not only per buffer: a caller that
  // hands the same buffer to a donated and to a plain position has asked for
  // an argument to be both consumed and read.  Every PjRtClient refuses that
  // (`f(donate(a), a)`), and this plugin used to accept it and delete the
  // buffer out from under the second use.  XLA's own bookkeeping is called
  // here, so the three messages are word for word the ones jax users see on
  // cpu/cuda/tpu.
  absl::flat_hash_map<const void*, std::pair<bool, int>> donation_clashes;
  donation_clashes.reserve(argument_handles.size());
  const absl::flat_hash_set<int> donated(lowered_->donated_parameters.begin(),
                                         lowered_->donated_parameters.end());

  std::vector<mx::array> inputs;
  inputs.reserve(argument_handles.size());
  for (size_t i = 0; i < argument_handles.size(); i++) {
    xla::PjRtBuffer* handle = argument_handles[i];
    if (handle == nullptr || handle->client() != client_) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "metaljax-native: argument %d is not a buffer of this client", i));
    }
    // Safe without RTTI: this client hands out nothing but MetalBuffers, and
    // the client identity above is what says the buffer came from here.
    const auto* buffer = static_cast<const MetalBuffer*>(handle);
    if (!buffer->array().has_value()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "metaljax-native: argument %d has been deleted", i));
    }
    const mx::array& a = *buffer->array();
    const ValueSpec& spec = lowered_->parameters[i];
    if (a.dtype() != spec.dtype || !SameShape(a, spec)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "metaljax-native: argument %d is %s, the program expects %s", i,
          ShapeString(a), xla::ShapeUtil::HumanString(spec.shape())));
    }
    const bool must_donate =
        donated.contains(static_cast<int>(i)) &&
        !options.non_donatable_input_indices.contains(static_cast<int>(i));
    RETURN_IF_ERROR(xla::TestBufferDonationClashes(
        handle, donation_clashes, must_donate, static_cast<int>(i),
        /*replica=*/0, /*partition=*/0));
    inputs.push_back(a);
  }

  // Which tape runs, and the packed weights it takes after the caller's own
  // arguments (P17).  The packs are INPUTS rather than constants: mx::compile
  // bakes a captured constant by value, and a repack would then never be seen.
  const std::shared_ptr<const LoweredProgram>& tape = Tape(inputs);
  for (const mx::array& pack : tape->packs) inputs.push_back(pack);

  std::vector<mx::array> outs;
  const Stats before = g_stats;
  try {
    const std::vector<mx::array> kept =
        kVerifyCompile ? inputs : std::vector<mx::array>{};
    outs = tape->program->run(std::move(inputs));
    // The half of the no-alias contract only a CALL can decide: an output the
    // lowering left uncopied because every argument it aliases is donated has
    // to be copied after all when this call takes the donation back
    // (`non_donatable_input_indices`).  Nothing to walk on the usual path --
    // jax passes the set empty, and a program that donates nothing has no
    // exemptions at all.
    if (!options.non_donatable_input_indices.empty()) {
      for (const auto& ex : tape->donated_output_aliases) {
        bool retracted = false;
        for (int a : ex.second)
          if (options.non_donatable_input_indices.contains(a)) {
            retracted = true;
            break;
          }
        if (retracted && ex.first >= 0 &&
            static_cast<size_t>(ex.first) < outs.size())
          outs[ex.first] = fresh_copy(outs[ex.first]);
      }
    }
    // P2 executes SYNCHRONOUSLY: correctness before pipelining.  Settling
    // here is what makes every buffer this returns honestly ready, and it is
    // the one thing to revisit first when the plugin is measured -- the Stage
    // 1 engine hands out lazy arrays and submits with async_eval instead.
    mx::eval(outs);
    // ...and MATERIALIZED.  A tape's output can be an MLX VIEW whose buffer is
    // smaller than the array says it is -- the plain case is a broadcast, which
    // comes back as 110 elements over a buffer of ONE (`strides=[0,0]`,
    // `data_size=1`).  That is fine inside MLX, which reads the strides, and it
    // is not fine as a PJRT buffer: this thing is handed to jax, kept, and
    // passed back as the ARGUMENT of the next executable, where its consumer
    // may be a kernel that reads it as dense memory.  `mx::scatter` under
    // `mx::compile` is one such consumer, and it silently dropped updates
    // (metaljax's tracked-open sparse `spdot_general` pair: the failure needed
    // one earlier program in the process to put such a buffer in jax's hands,
    // so it looked positional rather than structural).  `unsafe_buffer_pointer`
    // has the same stake -- P12 kept a settled view for it because a
    // re-gathered broadcast handed out a new address every call.
    //
    // The flags are only meaningful AFTER the eval above (M5c's `to_host`
    // reads them in the same order), and only the offenders pay.  It must be
    // `mx::contiguous` and not `fresh_copy`: the select `fresh_copy` builds
    // keeps the broadcast's strides, so it de-aliases (which is its job in
    // `Program::run`) without ever widening the buffer -- measured, the
    // `data_size` comes back 1.  `contiguous` is a pure re-layout, so bits
    // move unchanged and -0 and NaN payloads survive it.
    bool refreshed = false;
    for (mx::array& a : outs) {
      if (a.data_size() != a.size() || !a.flags().row_contiguous) {
        a = mx::contiguous(a);
        refreshed = true;
      }
    }
    if (refreshed) mx::eval(outs);
    // METALJAX_VERIFY_COMPILE=1: run the tape a SECOND time op by op and
    // compare.  The compiled path is the only place this plugin can be right
    // in one mode and wrong in the other, and a divergence is silent by
    // construction -- so the diagnostic that finds it lives here, next to the
    // one call that has both paths in reach.  Off by default and never taken
    // by a normal run; the sparse `spdot_general` pair was found with it.
    if (kVerifyCompile) {
      std::vector<mx::array> eager = tape->program->interpret(kept, false);
      mx::eval(eager);
      for (size_t j = 0; j < outs.size() && j < eager.size(); j++) {
        if (outs[j].shape() != eager[j].shape()) continue;
        mx::array d = mx::sum(mx::astype(
            mx::not_equal(mx::astype(outs[j], mx::float32),
                          mx::astype(eager[j], mx::float32)), mx::int32));
        mx::eval(d);
        if (d.item<int>() != 0) {
          std::fprintf(stderr,
                       "[metaljax-native] VERIFY %s result %zu: %d of %zu "
                       "elements differ between the compiled and the eager "
                       "walk\n",
                       name_.c_str(), j, d.item<int>(), (size_t)outs[j].size());
          if (kVerifyDump) {
            auto dump = [](const char* tag, const mx::array& a) {
              std::fprintf(stderr, "  %s %s ds=%zu rowc=%d: ", tag,
                           ShapeString(a).c_str(), (size_t)a.data_size(),
                           (int)a.flags().row_contiguous);
              mx::array f = mx::astype(mx::reshape(a, mx::Shape{-1}), mx::float32);
              mx::eval(f);
              for (int q = 0; q < f.size() && q < 60; q++)
                std::fprintf(stderr, "%g ", f.data<float>()[q]);
              std::fprintf(stderr, "\n");
            };
            for (const mx::array& in : kept) dump("in", in);
            dump("compiled", outs[j]);
            dump("eager", eager[j]);
          }
          std::fflush(stderr);
        }
      }
    }
  } catch (const std::exception& e) {
    // A governor refusal is not an internal failure: it is the machine
    // saying no, and jax's users read the status code.  RESOURCE_EXHAUSTED
    // is what XLA's own backends raise for it (jax turns it into
    // `XlaRuntimeError: RESOURCE_EXHAUSTED: ...`), and the message names what
    // was needed, what was there and which variable moves the line.
    if (is_oom(e)) return absl::ResourceExhaustedError(e.what());
    return absl::InternalError(
        absl::StrCat("metaljax-native: ", name_, " failed: ", e.what()));
  }
  if (kDebug) {
    std::fprintf(stderr, "[metaljax-native] %s: %s\n", name_.c_str(),
                 StatsDelta(before, g_stats).c_str());
    std::fflush(stderr);
  }

  if (outs.size() != tape->results.size()) {
    return absl::InternalError(absl::StrFormat(
        "metaljax-native: %s returned %d values, the module declares %d",
        name_, outs.size(), tape->results.size()));
  }
  std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
  buffers.reserve(outs.size());
  for (size_t j = 0; j < outs.size(); j++) {
    const ValueSpec& spec = tape->results[j];
    // A mismatch here would be a lowering bug handing back the wrong bytes
    // under the right shape, which is the one failure mode this plugin must
    // never have: check it on every call rather than trusting the tape.
    if (outs[j].dtype() != spec.dtype || !SameShape(outs[j], spec)) {
      return absl::InternalError(absl::StrFormat(
          "metaljax-native: %s result %d came back as %s, the module declares "
          "%s",
          name_, j, ShapeString(outs[j]),
          xla::ShapeUtil::HumanString(spec.shape())));
    }
    buffers.push_back(std::make_unique<MetalBuffer>(
        client_, spec.host_memory ? host_memory_space_ : memory_space_, device_,
        spec.shape(), outs[j]));
  }

  // The donation contract (P13), collected only after a run that succeeded:
  // an argument the module declares donated is the caller's to hand over, and
  // XLA's rule is that they may not touch it again.  This engine never writes
  // into an input -- the aliasing an output would need is exactly what
  // `Lowering::Run`'s copy rule refuses -- so the buffer is simply released,
  // which is the whole of what jax observes (a reuse raises, and the memory
  // goes back to MLX's pool one execute earlier than the caller's own
  // reference would have freed it).  `non_donatable_input_indices` is the
  // caller taking the promise back for one call, and it wins.
  for (int i : tape->donated_parameters) {
    if (options.non_donatable_input_indices.contains(i)) continue;
    if (static_cast<size_t>(i) < argument_handles.size() &&
        argument_handles[i] != nullptr)
      argument_handles[i]->Delete();
  }
  return buffers;
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>
MetalLoadedExecutable::Execute(
    absl::Span<const std::vector<xla::PjRtBuffer*>> argument_handles,
    const xla::ExecuteOptions& options,
    std::optional<std::vector<xla::Future<>>>& returned_futures) const {
  if (argument_handles.size() != 1) {
    return absl::UnimplementedError(absl::StrFormat(
        "metaljax-native: one device, got %d argument lists",
        argument_handles.size()));
  }
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers,
                   RunOnce(argument_handles[0], options));
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> result;
  result.push_back(std::move(buffers));
  if (returned_futures.has_value()) {
    // The run above already settled: the execution really is complete, so a
    // ready future is the truth and not a shortcut.
    returned_futures->clear();
    returned_futures->push_back(xla::Future<>(absl::OkStatus()));
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
MetalLoadedExecutable::ExecuteSharded(
    absl::Span<xla::PjRtBuffer* const> argument_handles,
    xla::PjRtDevice* device, const xla::ExecuteOptions& options,
    std::optional<xla::Future<>>& returned_future, bool fill_future) const {
  if (device != device_) {
    return absl::InvalidArgumentError(
        "metaljax-native: the executable is loaded on a different device.");
  }
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers,
                   RunOnce(argument_handles, options));
  if (fill_future) returned_future = xla::Future<>(absl::OkStatus());
  return buffers;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
MetalLoadedExecutable::ExecutePortable(
    absl::Span<xla::PjRtBuffer* const> argument_handles,
    xla::PjRtDevice* device, const xla::ExecuteOptions& options,
    std::optional<xla::Future<>>& returned_future, bool fill_future) const {
  return ExecuteSharded(argument_handles, device, options, returned_future,
                        fill_future);
}

// --------------------------------------------------------------------------
// the metadata half (see the header: the C-API wrapper asks THIS object)
// --------------------------------------------------------------------------

absl::StatusOr<std::vector<xla::Shape>> MetalExecutable::GetOutputShapes()
    const {
  std::vector<xla::Shape> shapes;
  shapes.reserve(lowered_->results.size());
  for (const ValueSpec& spec : lowered_->results) shapes.push_back(spec.shape());
  // One program, whose result is the TUPLE of the function's results: that is
  // the shape XLA's own implementations read out of an HloModule, and what the
  // element-type and dimension queries below are consistent with.
  return std::vector<xla::Shape>{xla::ShapeUtil::MakeTupleShape(shapes)};
}

absl::StatusOr<std::vector<std::vector<xla::PrimitiveType>>>
MetalExecutable::GetOutputElementTypes() const {
  std::vector<xla::PrimitiveType> types;
  types.reserve(lowered_->results.size());
  for (const ValueSpec& spec : lowered_->results) types.push_back(spec.type);
  return std::vector<std::vector<xla::PrimitiveType>>{std::move(types)};
}

absl::StatusOr<std::vector<std::vector<xla::DimensionVector>>>
MetalExecutable::GetOutputDimensions() const {
  std::vector<xla::DimensionVector> dims;
  dims.reserve(lowered_->results.size());
  for (const ValueSpec& spec : lowered_->results)
    dims.push_back(xla::DimensionVector(spec.dims.begin(), spec.dims.end()));
  return std::vector<std::vector<xla::DimensionVector>>{std::move(dims)};
}

absl::StatusOr<absl::flat_hash_map<std::string, xla::PjRtValueType>>
MetalExecutable::GetCostAnalysis() const {
  auto bytes = [](const std::vector<ValueSpec>& specs) {
    int64_t total = 0;
    for (const ValueSpec& spec : specs)
      total += spec.element_count() *
               static_cast<int64_t>(spec.dtype.size());
    return total;
  };
  // FLOAT, every one of them: the C API carries a cost property as a float and
  // `PJRT_Executable_GetCostAnalysis` CHECKs the variant's alternative, so an
  // int64 here aborts the process inside jaxlib rather than raising.
  return absl::flat_hash_map<std::string, xla::PjRtValueType>{
      {"metaljax_tape_entries", static_cast<float>(lowered_->num_entries)},
      {"metaljax_argument_bytes",
       static_cast<float>(bytes(lowered_->parameters))},
      {"metaljax_result_bytes", static_cast<float>(bytes(lowered_->results))},
      {"metaljax_output_copies", static_cast<float>(lowered_->num_copies)},
      {"metaljax_compiled", lowered_->compiled ? 1.0f : 0.0f},
  };
}

namespace {

std::vector<std::shared_ptr<const xla::PjRtLayout>> DefaultLayouts(
    const std::vector<ValueSpec>& specs) {
  // Dense row-major, which is the only layout this backend has: MLX's storage
  // is major-to-minor and the transfer paths refuse anything else.
  std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
  layouts.reserve(specs.size());
  for (const ValueSpec& spec : specs) {
    layouts.push_back(std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::GetDefaultLayoutForShape(spec.shape())));
  }
  return layouts;
}

}  // namespace

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
MetalExecutable::GetParameterLayouts() const {
  return DefaultLayouts(lowered_->parameters);
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
MetalExecutable::GetOutputLayouts() const {
  return DefaultLayouts(lowered_->results);
}

absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>>
MetalExecutable::GetHloModules() const {
  std::lock_guard<std::mutex> lock(hlo_mu_);
  if (hlo_ == nullptr) {
    if (lowered_->stablehlo.empty()) {
      return absl::UnimplementedError(
          "metaljax-native: this executable kept no program text.");
    }
    mlir::MLIRContext context;
    absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
        xla::ParseMlirModuleString(lowered_->stablehlo, context);
    if (!module.ok()) {
      return absl::UnimplementedError(absl::StrCat(
          "metaljax-native: cannot re-read this executable's program: ",
          module.status().message()));
    }
    absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo =
        xla::ConvertStablehloToHlo(**module);
    if (!hlo.ok()) {
      return absl::UnimplementedError(absl::StrCat(
          "metaljax-native: this executable's program has no HLO form: ",
          hlo.status().message()));
    }
    hlo_ = std::shared_ptr<xla::HloModule>(std::move(*hlo));
  }
  return std::vector<std::shared_ptr<xla::HloModule>>{hlo_};
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
MetalExecutable::GetParameterMemoryKinds() const {
  std::vector<absl::string_view> kinds;
  kinds.reserve(lowered_->parameters.size());
  for (const ValueSpec& spec : lowered_->parameters)
    kinds.push_back(spec.host_memory ? kHostMemoryKind : kMemoryKind);
  return std::vector<std::vector<absl::string_view>>{std::move(kinds)};
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
MetalExecutable::GetOutputMemoryKinds() const {
  // What the MODULE asked for, per result, which is what jax turns into the
  // output shardings' `memory_kind`.  One unified pool underneath, so the
  // answer is metadata -- but it is the caller's own annotation coming back,
  // rather than "device" for everything.
  std::vector<absl::string_view> kinds;
  kinds.reserve(lowered_->results.size());
  for (const ValueSpec& spec : lowered_->results)
    kinds.push_back(spec.host_memory ? kHostMemoryKind : kMemoryKind);
  return std::vector<std::vector<absl::string_view>>{std::move(kinds)};
}

}  // namespace metaljax
