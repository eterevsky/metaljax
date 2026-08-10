/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_client.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal/metal_buffer.h"
#include "metal/metal_dtypes.h"
#include "metal/metal_executable.h"
#include "metal/metal_lowering.h"
#include "metal/metal_stream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlx/mlx.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/runtime/device_id.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace metaljax {

namespace mx = mlx::core;

// METALJAX_DEBUG=1: narrate compiles.  Read here rather than through
// `g_cfg.debug` because it gates messages from `ConfigureFromEnv` itself.
const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();

namespace {

// The runtime cadences, from the environment (`ConfigureFromEnv` below).
//
// Stage 1 parses these in Python and copies them across through
// `metaljax::configure` -- so that two engines in ONE process cannot hold two
// opinions about a number the MLX command-buffer lottery is pinned to
// (notes/mlx-command-buffer-split.md).  This plugin is the only engine in its
// process: nothing Python-side is even imported, so there is no second reader
// to drift from, and leaving the cadences at their compiled-in defaults is
// what would drift -- from every documented default in
// src/metaljax/{interpreter,ops/control}.py, and from what a user setting the
// variable gets on the Stage 1 plugin.
//
// The defaults and the arithmetic below are copied from the module that owns
// each one, and the comment there is the explanation.  They are not
// preferences: the loop clear cadence is what keeps Metal's ~499k live-buffer
// limit at bay in a multi-hour loop (CLAUDE.md items 11/14), which is a
// correctness property of exactly the workloads Stage 2 exists for.
int64_t EnvInt(const char* name, int64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const long long parsed = std::strtoll(v, &end, 10);
  // Python's `int(os.environ[...])` raises on anything else; a plugin cannot
  // usefully raise from a static initializer, so garbage keeps the default and
  // says so under METALJAX_DEBUG.
  if (end == v || *end != '\0') {
    if (kDebug)
      std::fprintf(stderr, "[metaljax-native] ignoring %s=%s (not an integer)\n",
                   name, v);
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

bool EnvFlag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "1";
}

void ConfigureFromEnv() {
  const int64_t flush_mb = EnvInt("METALJAX_EAGER_FLUSH_MB", 1024);
  configure(
      /*eager_flush_bytes=*/std::max<int64_t>(flush_mb, 0) * (int64_t{1} << 20),
      /*flush_sync_every=*/EnvInt("METALJAX_EAGER_FLUSH_SYNC", 1),
      /*flush_clear_bytes=*/EnvInt("METALJAX_FLUSH_CLEAR_MB", 2048) *
          (int64_t{1} << 20),
      /*loop_clear_cost=*/EnvInt("METALJAX_LOOP_CLEAR_COST", 500000),
      /*while_pipeline=*/EnvInt("METALJAX_WHILE_PIPELINE", 1),
      /*debug=*/EnvFlag("METALJAX_DEBUG"),
      /*memdbg=*/EnvFlag("METALJAX_MEMDBG"));
}

}  // namespace

// ------------------------------------------------------------- description

MetalDeviceDescription::MetalDeviceDescription(int id, int process_index)
    : id_(id),
      process_index_(process_index),
      debug_string_(absl::StrFormat("MetalDevice(id=%d)", id)),
      to_string_(absl::StrFormat("MetalDevice(id=%d)", id)) {
  memory_space_ptrs_.push_back(&memory_space_);
}

// ------------------------------------------------------------ memory space

MetalMemorySpace::MetalMemorySpace(xla::PjRtClient* client, int id)
    : client_(client),
      id_(id),
      debug_string_(absl::StrFormat("MetalMemory(id=%d)", id)) {}

absl::Span<xla::PjRtDevice* const> MetalMemorySpace::devices() const {
  return client_->devices();
}

// ------------------------------------------------------------------ device

MetalDevice::MetalDevice(xla::PjRtClient* client, int id, int process_index)
    : client_(client), description_(id, process_index) {}

absl::Status MetalDevice::TransferToInfeed(const xla::LiteralSlice& literal) {
  return absl::UnimplementedError("metaljax does not support infeed.");
}

absl::Status MetalDevice::TransferFromOutfeed(
    xla::MutableBorrowingLiteral literal) {
  return absl::UnimplementedError("metaljax does not support outfeed.");
}

absl::Span<xla::PjRtMemorySpace* const> MetalDevice::memory_spaces() const {
  return client_->memory_spaces();
}

absl::StatusOr<xla::PjRtMemorySpace*> MetalDevice::default_memory_space()
    const {
  absl::Span<xla::PjRtMemorySpace* const> spaces = client_->memory_spaces();
  if (spaces.empty()) {
    return absl::InternalError("metaljax: device has no memory space.");
  }
  return spaces.front();
}

absl::StatusOr<xla::PjRtMemorySpace*> MetalDevice::memory_space_by_kind(
    absl::string_view memory_space_kind) const {
  for (xla::PjRtMemorySpace* space : client_->memory_spaces()) {
    if (space->kind() == memory_space_kind) {
      return space;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("metaljax: no memory space of kind ", memory_space_kind));
}

// ------------------------------------------------------------------ client

MetalClient::MetalClient() {
  // Once, here: the client is built when jax first asks for the platform, and
  // every execute in the process runs under these cadences.
  ConfigureFromEnv();
  // One GPU, one unified memory space.  Apple silicon exposes a single
  // integrated GPU per process; multi-device is out of scope.
  owned_memory_spaces_.push_back(
      std::make_unique<MetalMemorySpace>(this, /*id=*/0));
  owned_devices_.push_back(
      std::make_unique<MetalDevice>(this, /*id=*/0, /*process_index=*/0));
  for (auto& space : owned_memory_spaces_) {
    memory_spaces_.push_back(space.get());
  }
  for (auto& device : owned_devices_) {
    devices_.push_back(device.get());
  }
}

xla::PjRtPlatformId MetalClient::platform_id() const {
  static const uint64_t kId = tsl::Fingerprint64(kPlatformName);
  return kId;
}

absl::StatusOr<xla::PjRtDevice*> MetalClient::LookupDevice(
    xla::GlobalDeviceId global_device_id) const {
  for (xla::PjRtDevice* device : devices_) {
    if (device->global_device_id() == global_device_id) {
      return device;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "metaljax: no device with global id ", global_device_id.value()));
}

absl::StatusOr<xla::PjRtDevice*> MetalClient::LookupAddressableDevice(
    xla::LocalDeviceId local_device_id) const {
  for (xla::PjRtDevice* device : devices_) {
    if (device->local_device_id() == local_device_id) {
      return device;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "metaljax: no addressable device with local id ",
      local_device_id.value()));
}

namespace {

// Gather a strided host view into contiguous wire-format storage
// (native/metaljax_native.cc's `strided_gather`).  Element at a time with an
// odometer: a transfer happens once per buffer, the runs are usually
// contiguous anyway, and `data` may sit in the MIDDLE of the host allocation
// -- a flipped numpy view has negative strides and its logical [0,...,0] is
// not its lowest address.
void StridedGather(const char* base, char* dst, absl::Span<int64_t const> dims,
                   absl::Span<int64_t const> strides, size_t item) {
  int64_t total = 1;
  for (int64_t d : dims) total *= d;
  std::vector<int64_t> idx(dims.size(), 0);
  for (int64_t n = 0; n < total; n++) {
    int64_t off = 0;
    for (size_t i = 0; i < dims.size(); i++) off += idx[i] * strides[i];
    std::memcpy(dst + n * item, base + off, item);
    for (size_t i = dims.size(); i-- > 0;) {
      if (++idx[i] < dims[i]) break;
      idx[i] = 0;
    }
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>>
MetalClient::BufferFromHostBuffer(
    const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    xla::PjRtMemorySpace* memory_space, const xla::Layout* device_layout) {
  std::optional<WireType> wire = WireTypeOf(type);
  if (!wire.has_value()) {
    return absl::UnimplementedError(
        absl::StrCat("metaljax-native: no host transfer for ",
                     xla::PrimitiveType_Name(type)));
  }
  if (device_layout != nullptr &&
      !xla::LayoutUtil::IsMonotonicWithDim0Major(*device_layout)) {
    return absl::UnimplementedError(
        "metaljax-native handles row-major layouts only.");
  }
  if (byte_strides.has_value() && byte_strides->size() != dims.size()) {
    return absl::InvalidArgumentError(
        "metaljax: byte_strides rank does not match dims.");
  }

  BindThread();
  mx::Shape shape;
  shape.reserve(dims.size());
  int64_t total = 1;
  for (int64_t d : dims) {
    if (d < 0 || d > std::numeric_limits<mx::ShapeElem>::max()) {
      return absl::InvalidArgumentError(
          absl::StrCat("metaljax: dimension out of range: ", d));
    }
    shape.push_back(static_cast<mx::ShapeElem>(d));
    total *= d;
  }

  xla::PjRtMemorySpace* space =
      memory_space != nullptr ? memory_space : memory_spaces_.front();
  auto wrap = [&](mx::array array) {
    array.eval();  // honestly ready before the buffer is handed out
    if (on_done_with_host_buffer) std::move(on_done_with_host_buffer)();
    return std::make_unique<MetalBuffer>(this, space, devices_.front(),
                                         xla::ShapeUtil::MakeShape(type, dims),
                                         std::move(array));
  };

  if (data == nullptr || total == 0) {
    // Never touch a data pointer for an empty transfer: MLX 0.32 can hand out
    // null-backed zero-size buffers whose access segfaults.
    return wrap(mx::zeros(shape, wire->device));
  }

  // Stage into wire-format contiguous memory (gathering strides if any), then
  // narrow in place if the wire is wider than the device dtype.  The staging
  // block becomes the array's own storage -- one allocation, one copy, freed
  // by MLX when the array dies.  Ingest COPIES: PJRT only guarantees the host
  // memory for the duration of this call.
  const size_t item = wire->host_item;
  const size_t nbytes = static_cast<size_t>(total) * item;
  char* stage = static_cast<char*>(std::malloc(nbytes));
  if (stage == nullptr) {
    // The callback still runs: the caller's host buffer is no longer ours to
    // read whether or not we managed to copy it.
    if (on_done_with_host_buffer) std::move(on_done_with_host_buffer)();
    return absl::ResourceExhaustedError(
        absl::StrCat("metaljax: could not stage ", nbytes, " host bytes"));
  }
  const char* src = static_cast<const char*>(data);
  if (!byte_strides.has_value()) {
    std::memcpy(stage, src, nbytes);
  } else {
    StridedGather(src, stage, dims, *byte_strides, item);
  }
  if (wire->widen) {
    // f64 wire -> f32 device (or c128 -> c64): narrow in place.  Same
    // rounding as numpy's astype (C double->float, round-to-nearest-even).
    const double* from = reinterpret_cast<const double*>(stage);
    float* to = reinterpret_cast<float*>(stage);
    const int64_t scalars = total * wire->lanes;
    for (int64_t i = 0; i < scalars; i++) to[i] = static_cast<float>(from[i]);
  }
  return wrap(mx::array(static_cast<void*>(stage), shape, wire->device,
                        [](void* p) { std::free(p); }));
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
MetalClient::CompileAndLoad(xla::MaybeOwningMlirModule module,
                            xla::CompileOptions options) {
  // The C-API wrapper hands us a PARSED module: `stablehlo.*`, already
  // upgraded from the VHLO portable artifact, so nothing here touches
  // serialization (the whole of what src/metaljax/engine.py does by hand).
  mlir::ModuleOp mlir_module = module.mlir_module();
  if (!mlir_module) {
    return absl::InvalidArgumentError(
        "metaljax-native: CompileAndLoad received no module.");
  }
  const xla::ExecutableBuildOptions& build = options.executable_build_options;
  if (build.num_replicas() != 1 || build.num_partitions() != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "metaljax-native: one device only, asked for ", build.num_replicas(),
        " replicas x ", build.num_partitions(), " partitions"));
  }

  ASSIGN_OR_RETURN(LoweredProgram lowered, LowerModule(mlir_module));
  std::string name = "main";
  if (auto sym = mlir_module.getSymName(); sym.has_value())
    name = sym->str();
  if (kDebug) {
    std::fprintf(stderr,
                 "[metaljax-native] lowered %s: %lld entries, %zu args, "
                 "%zu results, %lld output copies\n",
                 name.c_str(), static_cast<long long>(lowered.num_entries),
                 lowered.parameters.size(), lowered.results.size(),
                 static_cast<long long>(lowered.num_copies));
    std::fflush(stderr);
  }
  return std::make_unique<MetalLoadedExecutable>(
      this, devices_.front(), memory_spaces_.front(), std::move(lowered),
      std::move(options), std::move(name));
}

std::unique_ptr<xla::PjRtClient> CreateMetalClient() {
  return std::make_unique<MetalClient>();
}

}  // namespace metaljax
