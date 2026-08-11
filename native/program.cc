// metaljax native engine — the Program itself.
//
// A prepared program is a flat tape of entries over a slot environment:
// `interpret` walks it, `step` hands each entry to the family that owns its
// opcode (the files listed in program.h), and the drop lists free a slot the
// moment its last reader has run. Around that walk sit the two things a
// replay needs whatever the tape holds: the byte-denominated eager flush
// that keeps the pending MLX graph bounded, and the recovery ladder that
// retires a compiled path or clears the buffer cache and runs again.
//
// Nothing here knows what any op DOES; nothing in the op files knows how a
// program is entered. That line is the one a phase-2 compile path meets.

#include "program.h"

#include <mlx/compile_impl.h>

namespace metaljax {

namespace {

// interpreter._eager_flush: settle everything the environment still
// holds. Only that needs to survive -- pruned intermediates are already
// unreferenced, so MLX frees them as the evaluation walks the graph
// instead of materializing the whole chain at once.
void eager_flush(const std::vector<std::optional<mx::array>>& env) {
  std::vector<mx::array> live;
  for (const auto& v : env)
    if (v) live.push_back(*v);
  if (live.empty()) return;
  g_stats.flushes++;
  const bool hard = g_cfg.flush_sync_every > 0 &&
                    g_stats.flushes % g_cfg.flush_sync_every == 0;
  flush_eval(live, hard);
  // "Frees" into MLX's CACHE, whose byte bound is the memory limit -- so
  // an eager phase whose traffic is many times its live set claims the
  // traffic. Return it to the OS past the configured watermark.
  if (hard && g_cfg.flush_clear_bytes >= 0 &&
      static_cast<int64_t>(mx::get_cache_memory()) > g_cfg.flush_clear_bytes) {
    g_stats.cache_clears++;
    mx::clear_cache();
  }
}

}  // namespace

Program::Program(int num_slots, int num_args)
    : nslots_(num_slots), nargs_(num_args) {
  if (num_slots < 0 || num_args < 0 || num_args > num_slots)
    throw std::invalid_argument("tape: bad slot counts");
}

Program::~Program() {
  // MLX keeps a compiled graph per id for as long as nobody erases it.
  for (const auto& kv : compiled_) mx::detail::compile_erase(kv.second.id);
}

void Program::add(int op, std::vector<int> ins, std::vector<int> outs,
                  std::vector<int64_t> attrs, std::optional<mx::array> payload,
                  std::vector<int> drops,
                  std::vector<std::shared_ptr<Program>> regions, int64_t bytes,
                  std::vector<double> fattrs, std::shared_ptr<MslPlan> msl,
                  HostFn host) {
  for (int s : ins) check_slot(s);
  for (int s : outs) check_slot(s);
  for (int s : drops) check_slot(s);
  if ((op == kMslScan) != (msl != nullptr))
    throw std::invalid_argument("tape: msl plan without an msl entry");
  if ((op == kHostCall) != static_cast<bool>(host))
    throw std::invalid_argument("tape: host callable without a host entry");
  ops_.push_back(Entry{op, std::move(ins), std::move(outs),
                       std::move(attrs), std::move(fattrs),
                       std::move(payload), std::move(drops),
                       std::move(regions), bytes, std::move(msl),
                       std::move(host)});
}

void Program::set_outputs(std::vector<int> outs, std::vector<int> copies) {
  for (int s : outs) check_slot(s);
  outputs_ = std::move(outs);
  copies_ = std::move(copies);
}

void Program::set_compile(bool on, std::vector<int> anchors,
                          int64_t max_repeat) {
  compile_ = on;
  anchors_ = std::move(anchors);
  max_repeat_ = max_repeat;
}

std::vector<mx::array> Program::run(std::vector<mx::array> inputs) {
  if (static_cast<int>(inputs.size()) != nargs_)
    throw std::invalid_argument("tape: wrong number of inputs");
  std::vector<mx::array> outs;
  {
    // Everything below builds on the CALLING thread's default MLX
    // stream, which engine.execute has already pointed at a
    // cross-thread-evaluable stream of this thread's own
    // (engine.bind_thread) — it is the one entry point a native run can
    // be reached through, so a threadbare caller is bound before it gets
    // here.
    //
    // An embedder that has an interpreter drops its lock before calling
    // this (bindings.cc releases the GIL around it): nothing below needs
    // one, and a waiter on `lock_` that held the GIL would deadlock
    // against the recovery paths, which reacquire it through g_gc_hook.
    std::lock_guard<std::mutex> guard(lock_);
    t_msl_pending.clear();
    outs = run_recovering(inputs);
    settle_msl(inputs, outs);
    for (int i : copies_) {
      if (i >= 0 && static_cast<size_t>(i) < outs.size())
        outs[i] = fresh_copy(outs[i]);
    }
    // XLA's no-alias contract, the half object identity cannot express
    // across a language boundary: two outputs reading the SAME slot are
    // one array, and nanobind hands each of them a fresh Python wrapper,
    // so engine.execute's `seen_out` pass cannot see the
    // duplicate. (Input aliasing is a static property of the program and
    // is handled where it belongs — tape.py declines the shapes of
    // forwarding that would reach here.)
    for (size_t i = 1; i < outs.size(); i++) {
      for (size_t j = 0; j < i; j++) {
        if (outs[i].id() == outs[j].id()) {
          outs[i] = fresh_copy(outs[i]);
          break;
        }
      }
    }
  }
  return outs;
}

std::vector<mx::array> Program::call(const std::vector<mx::array>& inputs,
                                     bool in_trace) {
  if (in_trace) return interpret(inputs, true);
  if (compile_ && !compile_disabled_) {
    g_stats.compiled_calls++;
    return compiled(1)(inputs);
  }
  return interpret(inputs, false);
}

void Program::tally(std::map<int, int64_t>& counts) const {
  for (const Entry& e : ops_) {
    counts[e.op]++;
    for (const auto& r : e.regions) r->tally(counts);
  }
}

bool Program::reads_host() const {
  if (reads_host_ < 0) {
    reads_host_ = 0;
    for (const Entry& e : ops_) {
      if (e.op == kWhile || e.op == kIf || e.op == kCase) {
        reads_host_ = 1;
        break;
      }
      for (const auto& r : e.regions) {
        if (r->reads_host()) { reads_host_ = 1; break; }
      }
      if (reads_host_) break;
    }
  }
  return reads_host_ != 0;
}

std::vector<mx::array> Program::interpret(const std::vector<mx::array>& inputs,
                                          bool in_trace) {
  if (static_cast<int>(inputs.size()) != nargs_)
    throw std::invalid_argument("tape: wrong number of inputs");
  std::vector<std::optional<mx::array>> env(nslots_);
  for (size_t i = 0; i < inputs.size(); i++) env[i] = inputs[i];
  // interpreter.run_block's byte-denominated safety net: once this much
  // estimated result data has been produced with no sync point, settle
  // what is still live so the pending graph -- and the Metal buffers it
  // pins -- stay bounded. Never inside a trace: there is nothing to
  // evaluate there, and MLX manages the traced tape itself.
  const bool flushing = !in_trace && g_cfg.eager_flush_bytes > 0;
  int64_t acc = 0;
  for (const Entry& e : ops_) {
    step(e, env, in_trace);
    if (flushing) {
      acc += e.bytes;
      if (acc >= g_cfg.eager_flush_bytes) {
        acc = 0;
        eager_flush(env);
      }
    }
  }
  std::vector<mx::array> outs;
  outs.reserve(outputs_.size());
  for (int s : outputs_) {
    if (!env[s]) throw std::runtime_error("tape: output slot is empty");
    outs.push_back(*env[s]);
  }
  return outs;
}

int Program::max_live() const {
  std::vector<char> live(nslots_, 0);
  int cur = nargs_, peak = nargs_;
  for (int i = 0; i < nargs_; i++) live[i] = 1;
  for (const Entry& e : ops_) {
    for (int s : e.outs) {
      if (!live[s]) { live[s] = 1; cur++; }
    }
    if (cur > peak) peak = cur;
    for (int s : e.drops) {
      if (live[s]) { live[s] = 0; cur--; }
    }
  }
  return peak;
}

void Program::check_slot(int s) const {
  if (s < 0 || s >= nslots_)
    throw std::invalid_argument("tape: slot index out of range");
}

// One entry. The families are tried in rough order of how often a real tape
// holds them; each declines an opcode it does not own, and the tape's whole
// vocabulary is exactly what they accept between them.
void Program::step(const Entry& e, std::vector<std::optional<mx::array>>& env,
                   bool in_trace) const {
  if (!(step_elementwise(e, env, in_trace) || step_shape(e, env, in_trace) ||
        step_linalg(e, env, in_trace) || step_reduce(e, env, in_trace) ||
        step_index(e, env, in_trace) || step_emit(e, env, in_trace) ||
        step_control(e, env, in_trace) || step_rng(e, env, in_trace) ||
        step_conv(e, env, in_trace) || step_host(e, env, in_trace)))
    throw std::invalid_argument("tape: unknown opcode");

  for (int s : e.drops) env[s].reset();
}

bool Program::has_host() const {
  for (const Entry& e : ops_) {
    if (e.op == kHostCall) return true;
    for (const auto& r : e.regions)
      if (r->has_host()) return true;
  }
  return false;
}

std::vector<mx::array> Program::run_recovering(
    const std::vector<mx::array>& inputs) {
  for (int attempt = 0;; attempt++) {
    bool used_compiled = compile_ && !compile_disabled_;
    bool resource_limit = false;
    std::exception_ptr err;
    try {
      return call(inputs, false);
    } catch (const std::exception& ex) {
      resource_limit = is_resource_limit(ex);
      err = std::current_exception();
    }
    // Outside the handler, holding none of the failed attempt's arrays.
    if (used_compiled) {
      // Whatever went wrong, the compiled graph is what was running:
      // MLX's compiler rejects some traces outright, and buffer
      // exhaustion during a trace means the trace is too big to hold.
      debug_print("compiled native tape failed; running eagerly");
      drop_compiled();
      if (resource_limit) {
        gc_collect();
        mx::clear_cache();
      }
      continue;
    }
    if (resource_limit && attempt < 2) {
      debug_print("Metal buffer limit hit in the native tape; clearing "
                  "cache and retrying");
      g_stats.limit_retries++;
      gc_collect();
      mx::clear_cache();
      continue;
    }
    std::rethrow_exception(err);
  }
}

mx::Shape Program::shape(const std::vector<int64_t>& at, size_t off,
                         int64_t n) {
  mx::Shape s(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; i++)
    s[i] = static_cast<mx::ShapeElem>(at[off + static_cast<size_t>(i)]);
  return s;
}

std::vector<int> Program::axes(const std::vector<int64_t>& at, size_t off,
                               int64_t n) {
  std::vector<int> a(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; i++)
    a[i] = static_cast<int>(at[off + static_cast<size_t>(i)]);
  return a;
}

}  // namespace metaljax
