// metaljax native engine — the entries that leave the tape (ported from
// Stage 1's src/metaljax/ops/callbacks.py, deleted 0.11.6, ef5774d).
//
// Two of them. A host call is the one place a native run reaches back out of
// the tape: the LAPACK targets, jax's debug/pure/io callbacks, anything whose
// handler computes on the host. It can never run inside a trace -- a block
// holding one is impure in the Python analysis, and a host handler reading
// tracers would compute on nothing at all. An ordered-effect token carries
// no data and exists only to be somewhere in the data flow.
//
// What is on the other side of a host call is the embedder's business: the
// tape holds a `HostFn` and calls it (bindings.cc is what wraps a Python
// handler into one, GIL and all). `gc_collect` lives here for the symmetry —
// it is the other place the engine reaches out, and the hook it runs is set
// by the same adapter.

#include "program.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

// Run the embedder's cycle collector, if it has one; see g_gc_hook.
void gc_collect() {
  if (!g_gc_hook) return;
  try {
    g_gc_hook();
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
      // The one place a native run leaves the tape. Never inside a
      // trace: a block holding one of these is impure in the Python
      // analysis (interpreter._IMPURE_OPS, custom_call_host_hook), so
      // neither engine ever compiles it -- and a host handler reading
      // tracers would compute on nothing at all.
      if (in_trace)
        throw std::runtime_error("tape: a host call cannot run in a trace");
      std::vector<mx::array> args;
      args.reserve(e.ins.size());
      for (size_t i = 0; i < e.ins.size(); i++) args.push_back(in(i));
      // Whatever the handler needs to enter (an interpreter, a lock) it
      // enters and leaves inside itself, and whatever it raises arrives
      // here as a plain C++ exception -- the recovery ladder above calls
      // what() on it from a stack that holds nothing of the embedder's.
      std::vector<mx::array> res = e.host(args);
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
