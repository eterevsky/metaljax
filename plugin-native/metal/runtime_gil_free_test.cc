// The executor runtime, driven from a process that has no Python in it.
//
// Phase 2 P1's proof: `native/program.h` is a plain C++ interface, so a tape
// can be built and replayed on the GPU with no interpreter, no GIL and no
// nanobind anywhere in the image.  Until the plugin's CompileAndLoad grows an
// engine this is the only thing that exercises the runtime the way phase 2
// will, and it is deliberately end-to-end: real mx::arrays, real Metal
// kernels, exact numbers.
//
// Four things are checked, in order of what would hurt most to get wrong:
//
//   1. the image really is Python-free (dyld's loaded-image list, plus a
//      dlsym for CPython's entry points -- a dependency could arrive through
//      any of XLA's transitive deps, and this is the check that would catch
//      it);
//   2. a tape built through the C++ API interprets to the right bytes;
//   3. the same tape through mx::compile does too, which is what says the
//      engine-owned cache ids of compile.cc work outside the extension --
//      MLX's own convention is to key that cache by the address of a *Python*
//      function object;
//   4. a dynamic while runs its regions the RIGHT NUMBER OF TIMES -- the one
//      property the pipelined loop can break, and only for a region that
//      leaves the tape, which is why it is stated here (P8.5).

#include <dlfcn.h>
#include <mach-o/dyld.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

  // 3. a dynamic while whose COND has an effect, and the pure loops it must
  //    not slow down.  The pipelined loop builds iteration t+1 before it
  //    reads t's condition, which is free for a pure region and a second
  //    print (or a second callback) for one that leaves the tape -- so a
  //    host call anywhere in either region keeps the serial shape.  Counted
  //    HERE rather than in tests/: a host call cannot reach the runtime
  //    through the plugin's PJRT surface, so this is the only process in
  //    which the rule can be stated end to end.
  {
    const std::vector<int> want_calls{6, 0, 5};
    for (int variant = 0; variant < 3; variant++) {
      const bool cond_host = variant == 0, body_host = variant == 2;
      int calls = 0;
      metaljax::HostFn count =
          [&calls](const std::vector<mx::array>&) -> std::vector<mx::array> {
        calls++;
        return {};
      };
      auto cond = std::make_shared<Program>(/*num_slots=*/3, /*num_args=*/1);
      cond->add(opcode("stablehlo.constant"), {}, {1}, {}, mx::array(5),
                {}, {}, 4, {}, nullptr, {});
      if (cond_host)
        cond->add(opcode("metaljax.host_call"), {0}, {}, {}, std::nullopt, {},
                  {}, 0, {}, nullptr, count);
      // [direction] 2 = LT
      cond->add(opcode("stablehlo.compare"), {0, 1}, {2}, {2}, std::nullopt,
                {1}, {}, 1, {}, nullptr, {});
      cond->set_outputs({2}, {});

      auto body = std::make_shared<Program>(/*num_slots=*/3, /*num_args=*/1);
      body->add(opcode("stablehlo.constant"), {}, {1}, {}, mx::array(1), {},
                {}, 4, {}, nullptr, {});
      if (body_host)
        body->add(opcode("metaljax.host_call"), {0}, {}, {}, std::nullopt, {},
                  {}, 0, {}, nullptr, count);
      body->add(opcode("stablehlo.add"), {0, 1}, {2}, {}, std::nullopt, {1},
                {}, 4, {}, nullptr, {});
      body->set_outputs({2}, {});

      Program p(/*num_slots=*/2, /*num_args=*/1);
      // [ncarry, ncond_caps, nbody_caps, counted, k, bound_kind, bound,
      //  cost, period, chunkable, kmax, body_compile_max]
      p.add(opcode("stablehlo.while"), {0}, {1},
            {1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}, std::nullopt, {},
            {cond, body}, 4, {}, nullptr, {});
      p.set_outputs({1}, {});

      const int64_t serial = metaljax::g_stats.serial_loops;
      const int64_t piped = metaljax::g_stats.pipelined_loops;
      std::vector<mx::array> outs = p.run({mx::array(0)});
      outs[0].eval();
      const std::string what =
          cond_host ? "impure cond" : (body_host ? "impure body" : "pure");
      check(outs.size() == 1 && outs[0].item<int>() == 5,
            "dynamic while (" + what + "): five iterations");
      check(calls == want_calls[variant],
            "dynamic while (" + what + "): the host call ran " +
                std::to_string(calls) + " times, want " +
                std::to_string(want_calls[variant]));
      const bool pipelined = metaljax::g_stats.pipelined_loops == piped + 1 &&
                             metaljax::g_stats.serial_loops == serial;
      const bool went_serial =
          metaljax::g_stats.serial_loops == serial + 1 &&
          metaljax::g_stats.pipelined_loops == piped;
      check(variant == 1 ? pipelined : went_serial,
            "dynamic while (" + what + "): ran " +
                std::string(variant == 1 ? "pipelined" : "serial"));
    }
  }

  if (g_failures == 0) {
    std::printf("native runtime executed GIL-free: ok\n");
    return 0;
  }
  std::printf("native runtime executed GIL-free: FAILED (%d checks)\n",
              g_failures);
  return 1;
}
