// The executor runtime, driven from a process that has no Python in it.
//
// Phase 2 P1's proof: `native/program.h` is a plain C++ interface, so a tape
// can be built and replayed on the GPU with no interpreter, no GIL and no
// nanobind anywhere in the image.  Until the plugin's CompileAndLoad grows an
// engine this is the only thing that exercises the runtime the way phase 2
// will, and it is deliberately end-to-end: real mx::arrays, real Metal
// kernels, exact numbers.
//
// Three things are checked, in order of what would hurt most to get wrong:
//
//   1. the image really is Python-free (dyld's loaded-image list, plus a
//      dlsym for CPython's entry points -- a dependency could arrive through
//      any of XLA's transitive deps, and this is the check that would catch
//      it);
//   2. a tape built through the C++ API interprets to the right bytes;
//   3. the same tape through mx::compile does too, which is what says the
//      engine-owned cache ids of compile.cc work outside the extension --
//      MLX's own convention is to key that cache by the address of a *Python*
//      function object.

#include <dlfcn.h>
#include <mach-o/dyld.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "program.h"

namespace {

namespace mx = mlx::core;
using metaljax::Program;

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
  if (!ok) g_failures++;
}

void check_close(float got, float want, const std::string& what) {
  const bool ok = std::fabs(got - want) <= 1e-5f * (1.0f + std::fabs(want));
  std::printf("%s %s (got %g, want %g)\n", ok ? "  ok  " : "  FAIL",
              what.c_str(), got, want);
  if (!ok) g_failures++;
}

// The registry, read as the plain C++ table it now is.  Looking the opcodes
// up by name rather than spelling the enum is the point: it is the same
// lookup src/metaljax/tape.py does, minus the dict.
int opcode(const std::string& name) {
  for (const auto& kv : metaljax::opcodes())
    if (kv.first == name) return kv.second;
  check(false, "opcode " + name + " is in the registry");
  return -1;
}

// out0 = a * b + a  (f32[2,3]);  out1 = sum(out0)  (f32 scalar).
//
// Small on purpose, but it covers the three shapes of entry a replay has:
// binary elementwise, a constant the Program owns for its lifetime (the
// reduce's init), and a structured op that reads an attribute vector.
void build(Program& p) {
  const int mul = opcode("stablehlo.multiply");
  const int add = opcode("stablehlo.add");
  const int cst = opcode("stablehlo.constant");
  const int red = opcode("stablehlo.reduce");
  // slots: 0 a, 1 b, 2 a*b, 3 a*b+a, 4 the reduce's init, 5 the sum
  p.add(mul, {0, 1}, {2}, {}, std::nullopt, {}, {}, 24, {}, nullptr, {});
  p.add(add, {2, 0}, {3}, {}, std::nullopt, {2}, {}, 24, {}, nullptr, {});
  p.add(cst, {}, {4}, {}, mx::array(0.0f), {}, {}, 4, {}, nullptr, {});
  // [kind=0 sum, ndims=2, dims 0 and 1]
  p.add(red, {3, 4}, {5}, {0, 2, 0, 1}, std::nullopt, {4}, {}, 4, {}, nullptr,
        {});
  p.set_outputs({3, 5}, {});
}

void verify(const std::vector<mx::array>& outs, const std::vector<float>& a,
            const std::vector<float>& b, const std::string& what) {
  if (outs.size() != 2) {
    check(false, what + ": two results");
    return;
  }
  mx::array elt = outs[0], sum = outs[1];
  elt.eval();
  sum.eval();
  bool shape_ok = elt.shape() == mx::Shape{2, 3} && sum.shape().empty() &&
                  elt.dtype() == mx::float32 && sum.dtype() == mx::float32;
  check(shape_ok, what + ": shapes and dtypes");
  if (!shape_ok) return;
  const float* got = elt.data<float>();
  float want_sum = 0.0f;
  bool all = true;
  for (size_t i = 0; i < a.size(); i++) {
    const float want = a[i] * b[i] + a[i];
    want_sum += want;
    if (std::fabs(got[i] - want) > 1e-5f * (1.0f + std::fabs(want)))
      all = false;
  }
  check(all, what + ": a*b+a elementwise");
  check_close(*sum.data<float>(), want_sum, what + ": sum of all six");
}

// Nothing in this process may be CPython.  Both halves matter: a static link
// would not show up in dyld's image list, and a dynamic one would not
// necessarily be findable by name if it were loaded with RTLD_LOCAL -- so ask
// both ways.
//
// Matched on the image's BASENAME, not its path: `libmlx.dylib` is loaded out
// of the build venv's site-packages (that is where the wheel keeps it, and
// what the @mlx rpath points at), so a substring match on the path flags
// "python3.14" in a directory name and proves nothing.
void check_no_python() {
  std::vector<std::string> offenders;
  for (uint32_t i = 0; i < _dyld_image_count(); i++) {
    const char* name = _dyld_get_image_name(i);
    if (name == nullptr) continue;
    const char* slash = std::strrchr(name, '/');
    const char* base = slash != nullptr ? slash + 1 : name;
    if (std::strncmp(base, "libpython", 9) == 0 ||
        std::strncmp(base, "Python", 6) == 0 ||
        std::strstr(base, "cpython-") != nullptr ||
        std::strstr(base, "nanobind") != nullptr ||
        std::strstr(name, "Python.framework") != nullptr)
      offenders.emplace_back(name);
  }
  if (!offenders.empty())
    for (const std::string& o : offenders)
      std::printf("       loaded image: %s\n", o.c_str());
  check(offenders.empty(), "no Python/nanobind image is loaded");
  check(dlsym(RTLD_DEFAULT, "Py_Initialize") == nullptr &&
            dlsym(RTLD_DEFAULT, "Py_IsInitialized") == nullptr &&
            dlsym(RTLD_DEFAULT, "PyGILState_Ensure") == nullptr,
        "CPython's entry points are not resolvable");
  check(!metaljax::g_gc_hook,
        "the gc hook is empty (nothing to collect without an interpreter)");
}

}  // namespace

int main() {
  std::printf("metaljax runtime, no interpreter in the process\n");
  std::printf("  device: %s\n",
              mx::default_device() == mx::Device::gpu ? "gpu" : "cpu");
  check(mx::default_device() == mx::Device::gpu, "MLX's default device is the GPU");

  check_no_python();

  // The cadences the Python engine would copy in; here they are just the
  // defaults, restated through the same entry point a phase-2 compile path
  // will use.
  metaljax::configure(/*eager_flush_bytes=*/1024LL << 20,
                      /*flush_sync_every=*/1,
                      /*flush_clear_bytes=*/2048LL << 20,
                      /*loop_clear_cost=*/500000, /*while_pipeline=*/1,
                      /*debug=*/false, /*memdbg=*/false);

  const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> b = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
  auto args = [&]() {
    return std::vector<mx::array>{
        mx::array(a.data(), mx::Shape{2, 3}, mx::float32),
        mx::array(b.data(), mx::Shape{2, 3}, mx::float32)};
  };

  // 1. the op-by-op walk.
  {
    Program p(/*num_slots=*/6, /*num_args=*/2);
    build(p);
    check(p.num_ops() == 4, "tape holds four entries");
    // Six slots, but a*b is dropped the moment the add has read it, so the
    // environment never holds more than five at once.
    check(p.max_live() == 5, "liveness pruning caps the environment at five");
    verify(p.run(args()), a, b, "interpreted");
  }

  // 2. the same tape as one compiled graph.
  {
    const int64_t compiles = metaljax::g_stats.compiles;
    const int64_t calls = metaljax::g_stats.compiled_calls;
    Program p(/*num_slots=*/6, /*num_args=*/2);
    build(p);
    p.set_compile(/*on=*/true, /*anchors=*/{}, /*max_repeat=*/1);
    check(p.may_compile(1), "the program says it may compile");
    verify(p.run(args()), a, b, "compiled (first call: traced)");
    verify(p.run(args()), a, b, "compiled (second call: replayed)");
    check(!p.compiled_dropped(), "the compiled path was never retired");
    check(metaljax::g_stats.compiles == compiles + 1,
          "exactly one mx::compile trace was built");
    check(metaljax::g_stats.compiled_calls == calls + 2,
          "both calls went through the compiled graph");
  }

  if (g_failures == 0) {
    std::printf("native runtime executed GIL-free: ok\n");
    return 0;
  }
  std::printf("native runtime executed GIL-free: FAILED (%d checks)\n",
              g_failures);
  return 1;
}
