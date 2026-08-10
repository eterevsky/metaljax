// metaljax native engine — M0 scaffolding.
//
// Phase 1 of the C++ migration (notes/cpp-migration-plan.md): this
// extension will own the runtime replay path. M0 proves the build:
// nanobind module linked against the *installed MLX wheel's* libmlx,
// with a version handshake so ABI skew fails loudly at import instead
// of corrupting at runtime, and an mx::array round-trip demonstrating
// that arrays cross the boundary by handle, not by copy.
//
// Build: native/build.sh (same artisanal-clang pattern as plugin/).

#include <cstring>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mlx/mlx.h>
#include <mlx/version.h>

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

// The mlx version this extension was COMPILED against (headers) and the
// one it LINKED at runtime (libmlx's own version()). engine.py compares
// both to the imported mlx.__version__ and refuses the native engine on
// any mismatch: libmlx has no stable ABI and a silent skew is exactly
// the class of bug this project exists to avoid.
std::string compiled_mlx_version() {
  return std::to_string(MLX_VERSION_MAJOR) + "." +
         std::to_string(MLX_VERSION_MINOR) + "." +
         std::to_string(MLX_VERSION_PATCH);
}

std::string linked_mlx_version() { return mx::version(); }

// Round-trip proof for M0: accept the Python-side mx.array's underlying
// C++ object, do native work on it (a trivial computation forcing a real
// device dispatch), and hand a new array back. nanobind sees mx::array
// via MLX's own nanobind type caster, which the wheel's core module
// registers — arrays cross by shared handle, no host copy.
mx::array roundtrip_double(const mx::array& a) {
  auto out = mx::multiply(a, mx::array(2, a.dtype()));
  out.eval();
  return out;
}

// Device sanity for the handshake test: the default device seen from
// C++ must be the same GPU the Python side runs on.
std::string default_device() {
  auto& d = mx::default_device();
  return d == mx::Device::gpu ? "gpu" : "cpu";
}

// --------------------------------------------------------------------------
// M1: the buffer path (host <-> device transfers)
// --------------------------------------------------------------------------
//
// Ports engine.buffer_from_host / to_host with every hard-won rule intact:
//   * bf16 crosses by BITCAST in both directions. On the C++ side this is
//     free: the device layout of bf16 IS the wire format, so ingest and
//     egress are raw byte copies — the Python path's uint16-view dance
//     exists only because numpy cannot hold bf16. (History: the astype(f32)
//     detour doubled transfer bytes, transiently doubled device memory on a
//     62 GB load, and canonicalized NaN payload+sign.)
//   * f64/c128 pass through stored as f32/c64 (Metal has no doubles);
//     widened back on egress. Same double->float rounding as numpy astype
//     (C cast, round-to-nearest-even).
//   * negative-stride hosts arrive as (base offset, byte_strides) — the
//     logical [0,...,0] element sits at `offset` inside the buffer.
//   * 0-size arrays never touch a data pointer (MLX 0.32 can hand out
//     null-backed 0-size buffers whose access segfaults).
//   * ingest COPIES: PJRT only guarantees the host memory during the call
//     (the transient-address trap of CLAUDE.md item 20). Zero-copy with a
//     refcounting deleter is a future optimization gated on the plugin
//     plumbing PJRT_HostBufferSemantics through.
//
// The emulated dtypes (i4/u4/f8*/f6/f4/s2...) keep the Python path — their
// grid conversions live in dtypes.py and are not runtime-hot. engine.py
// routes by type_enum.

struct TypeInfo {
  mx::Dtype device;    // dtype stored on device
  size_t host_item;    // bytes per element on the PJRT wire
  bool widen;          // wire is f64/c128, device is f32/c64
};

// PJRT_Buffer_Type enum values (pjrt_c_api.h 0.114) for the natively
// handled set; everything else routes to the Python path in engine.py.
std::optional<TypeInfo> type_info(int type_enum) {
  switch (type_enum) {
    case 1:  return TypeInfo{mx::bool_, 1, false};      // PRED
    case 2:  return TypeInfo{mx::int8, 1, false};       // S8
    case 3:  return TypeInfo{mx::int16, 2, false};      // S16
    case 4:  return TypeInfo{mx::int32, 4, false};      // S32
    case 5:  return TypeInfo{mx::int64, 8, false};      // S64
    case 6:  return TypeInfo{mx::uint8, 1, false};      // U8
    case 7:  return TypeInfo{mx::uint16, 2, false};     // U16
    case 8:  return TypeInfo{mx::uint32, 4, false};     // U32
    case 9:  return TypeInfo{mx::uint64, 8, false};     // U64
    case 10: return TypeInfo{mx::float16, 2, false};    // F16
    case 11: return TypeInfo{mx::float32, 4, false};    // F32
    case 12: return TypeInfo{mx::float32, 8, true};     // F64 (stored f32)
    case 13: return TypeInfo{mx::bfloat16, 2, false};   // BF16 (bitcast)
    case 14: return TypeInfo{mx::complex64, 8, false};  // C64
    case 15: return TypeInfo{mx::complex64, 16, true};  // C128 (stored c64)
    default: return std::nullopt;
  }
}

bool native_type(int type_enum) { return type_info(type_enum).has_value(); }

// Gather a strided host view into contiguous wire-format storage.
// Element-at-a-time with an odometer: transfers happen once per buffer and
// the runs are usually contiguous anyway (the fast path above skips this).
void strided_gather(const char* base, char* dst, const int64_t* dims,
                    const int64_t* strides, size_t rank, size_t item) {
  int64_t total = 1;
  for (size_t i = 0; i < rank; i++) total *= dims[i];
  std::vector<int64_t> idx(rank, 0);
  for (int64_t n = 0; n < total; n++) {
    int64_t off = 0;
    for (size_t i = 0; i < rank; i++) off += idx[i] * strides[i];
    std::memcpy(dst + n * item, base + off, item);
    for (size_t i = rank; i-- > 0;) {
      if (++idx[i] < dims[i]) break;
      idx[i] = 0;
    }
  }
}

mx::array buffer_from_host(nb::handle data, int type_enum,
                           std::vector<int64_t> dims,
                           nb::handle byte_strides, int64_t offset) {
  auto info = type_info(type_enum);
  if (!info) throw std::invalid_argument("not a native dtype");
  mx::Shape shape(dims.begin(), dims.end());
  int64_t total = 1;
  for (auto d : dims) total *= d;

  if (data.is_none() || total == 0) {
    return mx::zeros(shape, info->device);
  }

  Py_buffer view;
  if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_SIMPLE) != 0)
    throw nb::python_error();
  const char* src = static_cast<const char*>(view.buf);

  // Stage into wire-format contiguous memory (gathering strides if any),
  // then narrow in place if the wire is wider than the device dtype. The
  // staging block becomes the array's own storage via the raw-pointer
  // constructor — one allocation, one copy, freed by MLX when the array
  // dies.
  const size_t item = info->host_item;
  char* stage = static_cast<char*>(std::malloc(total * item));
  if (stage == nullptr) {
    PyBuffer_Release(&view);
    throw std::bad_alloc();
  }
  {
    nb::gil_scoped_release nogil;
    if (byte_strides.is_none()) {
      std::memcpy(stage, src, total * item);
    } else {
      std::vector<int64_t> strides;
      {
        nb::gil_scoped_acquire gil;
        strides = nb::cast<std::vector<int64_t>>(byte_strides);
      }
      strided_gather(src + offset, stage, dims.data(), strides.data(),
                     dims.size(), item);
    }
    if (info->widen) {
      // f64 wire -> f32 device (or c128 -> c64): narrow in place, then
      // shrink the allocation's logical size. Same rounding as numpy's
      // astype (C double->float, round-to-nearest-even).
      const double* wide = reinterpret_cast<const double*>(stage);
      float* narrow = reinterpret_cast<float*>(stage);
      int64_t scalars = total * (type_enum == 15 ? 2 : 1);
      for (int64_t i = 0; i < scalars; i++)
        narrow[i] = static_cast<float>(wide[i]);
    }
  }
  PyBuffer_Release(&view);
  return mx::array(static_cast<void*>(stage), shape, info->device,
                   [](void* p) { std::free(p); });
}

nb::bytes to_host(const mx::array& a, int type_enum) {
  auto info = type_info(type_enum);
  if (!info) throw std::invalid_argument("not a native dtype");
  int64_t total = a.size();
  const size_t item = info->host_item;
  if (total == 0) return nb::bytes("", 0);

  // Settle the array, then read its raw device bytes: those ARE the wire
  // format for every non-widened dtype (including bf16 — the whole point).
  //
  // The settling is done on the array ITSELF and the `contiguous` copy is
  // made only when the layout needs one. Wrapping every result in a fresh
  // `contiguous` node and evaluating THAT costs a full round trip through
  // the stream — 20us, measured, even though the copy short-circuits to a
  // shared buffer — where `eval` on a value that is already available
  // returns in 0.1us. It is a per-OUTPUT cost, so a program handing back
  // 23 small tensors paid 0.5ms of it (the whole of the native engine's
  // deficit against the Python one on texmo's train chunks).
  //
  // `row_contiguous` is read after the eval because that is when MLX sets
  // it, and `data_size` is checked with it: the flag says the elements
  // start at `data` and run without gaps, the size says the buffer really
  // holds all of them (a short buffer read here is the conv-overread bug
  // of CLAUDE.md item 20, in a place with no test to catch it).
  mx::array c = a;
  {
    nb::gil_scoped_release nogil;
    c.eval();
    if (!c.flags().row_contiguous ||
        c.data_size() != static_cast<size_t>(total)) {
      c = mx::contiguous(c);
      c.eval();
    }
  }
  if (!info->widen) {
    return nb::bytes(c.data<char>(), total * item);
  }
  int64_t scalars = total * (type_enum == 15 ? 2 : 1);
  std::vector<double> wide(static_cast<size_t>(scalars));
  {
    nb::gil_scoped_release nogil;
    const float* narrow = c.data<float>();
    for (int64_t i = 0; i < scalars; i++)
      wide[i] = static_cast<double>(narrow[i]);
  }
  return nb::bytes(reinterpret_cast<const char*>(wide.data()), total * item);
}

}  // namespace

// M2: the prepared-program tape (native/tape.cc). Declared rather than
// header-included — one free function is the whole interface between the
// two translation units.
void register_tape(nb::module_& m);

NB_MODULE(metaljax_native, m) {
  m.doc() = "metaljax native replay engine (Stage 2, phase 1)";
  m.def("compiled_mlx_version", &compiled_mlx_version);
  m.def("linked_mlx_version", &linked_mlx_version);
  m.def("default_device", &default_device);
  m.def("roundtrip_double", &roundtrip_double,
        "Multiply by 2 natively; M0 boundary proof");
  // M1: the buffer path.
  m.def("native_type", &native_type,
        "Whether this PJRT type_enum transfers natively");
  m.def("buffer_from_host", &buffer_from_host,
        nb::arg("data").none(), nb::arg("type_enum"), nb::arg("dims"),
        nb::arg("byte_strides").none(), nb::arg("offset") = 0);
  m.def("to_host", &to_host);
  // M2: prepared programs and the eager core interpreter.
  register_tape(m);
}
