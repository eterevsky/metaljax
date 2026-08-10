// metaljax native engine — the nanobind adapter.
//
// The tape below this file is a plain C++ library: program.h names no Python,
// and neither does any translation unit it covers. This one and
// metaljax_native.cc are the whole of the boundary, which is what lets the
// phase-2 plugin link the SAME sources into a process that has no interpreter
// at all (plugin-native/third_party/metaljax_runtime).
//
// Everything Python-shaped therefore lives here and nowhere else:
//
//   * the dicts. The core hands out tables and counters as plain C++
//     containers; the shapes Python reads are made at this line.
//   * the GIL. `Program::run` is entered with it RELEASED — the tape builds a
//     lazy MLX graph out of objects that are already C++, so there is nothing
//     to hold it for, and holding it would serialize concurrent executables
//     on the one thing this engine exists to get out from under. The arguments
//     are cast before the release and the results after it, in nanobind's own
//     code.
//   * the host handlers. `Entry::host` is an opaque callable; `wrap_host`
//     turns a Python one into it and takes the GIL back INSIDE the wrapper.
//   * `gc.collect`, installed into `g_gc_hook` (program.h says why the
//     recovery ladder wants one).

#include <nanobind/nanobind.h>
// The stl casters ride in this file for the same reason they used to ride in
// program.h: every conversion that crosses the boundary needs them visible,
// and a missing one is silent (nanobind falls back to an opaque type).
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "program.h"
#include "msl.h"

namespace nb = nanobind;

namespace {

using metaljax::HostFn;

// The Python handler of a host op, as the tape's opaque callable. The GIL is
// acquired HERE because the whole of Program::run is entered without it; a
// Python exception is formatted HERE for the same reason, since the recovery
// ladder above calls what() on whatever it catches and nanobind formats a
// python_error's message through the Python C API.
HostFn wrap_host(nb::object fn) {
  if (!fn.is_valid() || fn.is_none()) return {};
  return [fn = std::move(fn)](const std::vector<mx::array>& args)
             -> std::vector<mx::array> {
    nb::gil_scoped_acquire gil;
    try {
      return nb::cast<std::vector<mx::array>>(fn(nb::cast(args)));
    } catch (nb::python_error& err) {
      throw std::runtime_error(std::string("tape: host call raised: ") +
                               err.what());
    }
  };
}

nb::dict py_opcodes() {
  nb::dict d;
  for (const auto& kv : metaljax::opcodes()) d[kv.first.c_str()] = kv.second;
  return d;
}

nb::dict py_dtype_codes() {
  nb::dict d;
  for (const auto& kv : metaljax::dtype_codes())
    d[kv.first.c_str()] = kv.second;
  return d;
}

nb::dict py_stats() {
  const metaljax::Stats& s = metaljax::g_stats;
  nb::dict d;
  d["flushes"] = s.flushes;
  d["cache_clears"] = s.cache_clears;
  d["loop_flushes"] = s.loop_flushes;
  d["loop_clears"] = s.loop_clears;
  d["limit_retries"] = s.limit_retries;
  d["compiled_calls"] = s.compiled_calls;
  d["compiles"] = s.compiles;
  d["compile_drops"] = s.compile_drops;
  d["chunk_drops"] = s.chunk_drops;
  d["unrolls"] = s.unrolls;
  d["pipelined_loops"] = s.pipelined_loops;
  d["pipelined_steps"] = s.pipelined_steps;
  d["serial_loops"] = s.serial_loops;
  d["msl_launches"] = s.msl_launches;
  d["msl_failures"] = s.msl_failures;
  d["host_calls"] = s.host_calls;
  return d;
}

}  // namespace

// Outside the namespace: metaljax_native.cc declares this one function and
// nothing else, so the extension's two halves share exactly one symbol.
void register_tape(nb::module_& m) {
  using namespace metaljax;  // the tape's own names, unqualified
  // Python's cycle collector barely triggers under array workloads, and dead
  // refcycles pin buffers mx::clear_cache cannot free (CLAUDE.md item 19).
  // Captureless on purpose: a global std::function holding an nb::object
  // would decref it at process teardown, after the interpreter is gone.
  g_gc_hook = []() {
    nb::gil_scoped_acquire gil;
    try {
      nb::module_::import_("gc").attr("collect")();
    } catch (...) {
      // A collection that cannot run must not take the recovery down with
      // it -- and the exception must die here, while the GIL is held.
    }
  };
  m.def("opcodes", &py_opcodes,
        "StableHLO op name -> opcode; a name absent here declines");
  m.def("dtype_codes", &py_dtype_codes,
        "MLIR element type -> dtype code; absent means unsupported");
  m.def("configure", &configure, nb::arg("eager_flush_bytes"),
        nb::arg("flush_sync_every"), nb::arg("flush_clear_bytes"),
        nb::arg("loop_clear_cost"), nb::arg("while_pipeline"),
        nb::arg("debug"), nb::arg("memdbg"),
        "Copy the Python-side runtime cadences into the native engine");
  m.def("stats", &py_stats, "Native-engine counters (sync points, compiles)");
  // M5b: one generated persistent kernel's launch recipe. Built by
  // metaljax.tape._lower_msl from an msl_scan Plan; `layout` is the
  // Cursor-encoded rest of it (documented at MslPlan::parse).
  nb::class_<MslPlan>(m, "MslPlan")
      .def(nb::init<std::string, std::string, std::string,
                    std::vector<std::string>, std::vector<std::string>,
                    std::vector<std::vector<int>>, std::vector<int>,
                    std::vector<int64_t>>(),
           nb::arg("name"), nb::arg("source"), nb::arg("header"),
           nb::arg("input_names"), nb::arg("output_names"),
           nb::arg("out_shapes"), nb::arg("out_dtypes"), nb::arg("layout"))
      .def_prop_ro("dead", &MslPlan::dead)
      .def_prop_ro("validated", &MslPlan::validated)
      .def_prop_ro("name", &MslPlan::name);
  nb::class_<Program>(m, "Program")
      .def(nb::init<int, int>(), nb::arg("num_slots"), nb::arg("num_args"))
      .def("add",
           [](Program& self, int opcode, std::vector<int> operands,
              std::vector<int> results, std::vector<int64_t> attrs,
              std::optional<mx::array> payload, std::vector<int> drops,
              std::vector<std::shared_ptr<Program>> regions, int64_t bytes,
              std::vector<double> fattrs, std::shared_ptr<MslPlan> msl,
              nb::object host) {
             self.add(opcode, std::move(operands), std::move(results),
                      std::move(attrs), std::move(payload), std::move(drops),
                      std::move(regions), bytes, std::move(fattrs),
                      std::move(msl), wrap_host(std::move(host)));
           },
           nb::arg("opcode"), nb::arg("operands"),
           nb::arg("results"), nb::arg("attrs"),
           nb::arg("payload").none(), nb::arg("drops"),
           nb::arg("regions") = std::vector<std::shared_ptr<Program>>(),
           nb::arg("bytes") = 0,
           nb::arg("fattrs") = std::vector<double>(),
           nb::arg("msl").none() = nb::none(),
           nb::arg("host").none() = nb::none())
      .def("set_outputs", &Program::set_outputs, nb::arg("slots"),
           nb::arg("copies") = std::vector<int>())
      .def("set_compile", &Program::set_compile, nb::arg("compile"),
           nb::arg("anchors") = std::vector<int>(),
           nb::arg("max_repeat") = 1)
      .def("run",
           [](Program& self, std::vector<mx::array> inputs) {
             // Cast in, cast out, and nothing of Python in between.
             std::vector<mx::array> outs;
             {
               nb::gil_scoped_release nogil;
               outs = self.run(std::move(inputs));
             }
             return outs;
           },
           nb::arg("inputs"))
      .def_prop_ro("compiled_dropped", &Program::compiled_dropped)
      .def_prop_ro("num_ops", &Program::num_ops)
      .def_prop_ro("num_slots", &Program::num_slots)
      .def_prop_ro("num_args", &Program::num_args)
      .def("op_histogram",
           [](const Program& self) {
             std::map<int, int64_t> counts;
             self.tally(counts);
             nb::dict d;
             for (const auto& kv : counts) d[nb::int_(kv.first)] = kv.second;
             return d;
           },
           "opcode -> entry count, this program and its regions")
      .def_prop_ro("max_live", &Program::max_live);
}
