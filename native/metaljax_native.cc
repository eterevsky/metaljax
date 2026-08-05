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

}  // namespace

NB_MODULE(metaljax_native, m) {
  m.doc() = "metaljax native replay engine (Stage 2, phase 1)";
  m.def("compiled_mlx_version", &compiled_mlx_version);
  m.def("linked_mlx_version", &linked_mlx_version);
  m.def("default_device", &default_device);
  m.def("roundtrip_double", &roundtrip_double,
        "Multiply by 2 natively; M0 boundary proof");
}
