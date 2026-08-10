// metaljax native engine — the entries that take the GIL
// (src/metaljax/ops/callbacks.py).
//
// Two of them. A host call is the one place a native run reaches back into
// Python: the LAPACK targets, jax's debug/pure/io callbacks, anything whose
// handler computes on the host. It can never run inside a trace -- a block
// holding one is impure in the Python analysis, and a host handler reading
// tracers would compute on nothing at all. An ordered-effect token carries
// no data and exists only to be somewhere in the data flow.
//
// `gc_collect` lives here for the same reason: it is the other place the
// engine reacquires the GIL, from recovery paths that have released it.

#include "program.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

// Python's cycle collector barely triggers under array workloads, and dead
// refcycles pin buffers mx::clear_cache cannot free (CLAUDE.md item 19).
// Needs the GIL, which the hot path has released.
void gc_collect() {
  nb::gil_scoped_acquire gil;
  try {
    nb::module_::import_("gc").attr("collect")();
  } catch (...) {
    // A collection that cannot run must not take the recovery down with it.
  }
}

bool Program::step_host(const Entry& e,
                        std::vector<std::optional<mx::array>>& env,
                        bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };

  switch (e.op) {
    // ops/callbacks.py `_token`: an ordered-effect token carries no data
    // and exists only to be somewhere in the data flow. create_token
    // makes one; after_all joins several into one, which is the same
    // empty array.
    case kToken:
      env[e.outs[0]] = mx::zeros(mx::Shape{0}, mx::bool_);
      break;

    // --- host ops (M5b) --------------------------------------------
    case kHostCall: {
      // The one place a native run reacquires the GIL. Never inside a
      // trace: a block holding one of these is impure in the Python
      // analysis (interpreter._IMPURE_OPS, custom_call_host_hook), so
      // neither engine ever compiles it -- and a host handler reading
      // tracers would compute on nothing at all.
      if (in_trace)
        throw std::runtime_error("tape: a host call cannot run in a trace");
      std::vector<mx::array> args;
      args.reserve(e.ins.size());
      for (size_t i = 0; i < e.ins.size(); i++) args.push_back(in(i));
      std::vector<mx::array> res;
      {
        nb::gil_scoped_acquire gil;
        try {
          nb::object out = e.host(nb::cast(args));
          res = nb::cast<std::vector<mx::array>>(out);
        } catch (nb::python_error& err) {
          // A Python exception must not travel up a stack that has
          // released the GIL: the recovery paths above call what() on
          // whatever they catch, and nanobind formats a python_error's
          // message through the Python C API. Say it here, where the GIL
          // is held, and carry a plain string the rest of the way.
          throw std::runtime_error(std::string("tape: host call raised: ") +
                                   err.what());
        }
      }
      g_stats.host_calls++;
      if (res.size() != e.outs.size())
        throw std::runtime_error("tape: host call result count mismatch");
      for (size_t i = 0; i < res.size(); i++) env[e.outs[i]] = res[i];
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
