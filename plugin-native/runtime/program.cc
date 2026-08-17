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
void eager_flush(const std::vector<std::optional<mx::array>>& env,
                 int64_t& program_flushes) {
  std::vector<mx::array> live;
  for (const auto& v : env)
    if (v) live.push_back(*v);
  if (live.empty()) return;
  g_stats.flushes++;
  const bool hard = g_cfg.flush_sync_every > 0 &&
                    g_stats.flushes % g_cfg.flush_sync_every == 0;
  if (hard) program_flushes++;
  flush_eval(live, hard);
  // "Frees" into MLX's CACHE, whose own limit is MLX's memory limit -- so an
  // eager phase whose traffic is many times its live set claims the traffic
  // unless something bounds the pool here. Something does, at exactly the
  // cadence it always did: past the watermark, TRIM the pool back to it.
  //
  // This used to be `mx::clear_cache()`, which returns the WHOLE pool to the
  // OS: 7 dumps a step at ~70 ms each on the maxtext training row, 2.2x
  // against its anchor (STATUS row 19 / P20). `trim_cache` reclaims the
  // excess and leaves the rest reusable, which is worth 1.10x on that row
  // and nothing anywhere else -- see notes/cpp-p25-cache-limit.md for what
  // the remaining 1.85x is (the watermark itself) and for why the bound is
  // NOT set globally at client construction (1.64x on a compiled decode
  // row). The watermark is no longer one number for every program either:
  // `flush_bound` spends the cap where the process has the footprint to
  // spare and falls back to P25's shipped 2048 MB where it does not
  // (notes/cpp-p27-flush-pressure.md).
  int64_t cache_before = 0;
  int64_t bound = g_cfg.flush_clear_bytes;
  bool trimmed = false;
  if (hard && g_cfg.flush_clear_bytes >= 0) {
    cache_before = static_cast<int64_t>(mx::get_cache_memory());
    bound = flush_bound(cache_before, program_flushes);
    if (cache_before > bound) {
      g_stats.cache_trims++;
      trim_cache(bound);
      trimmed = true;
    }
  }
  // ...and the governor's look at the machine, at the same cadence and AFTER
  // the trim above, so it judges a process that has already given back what
  // it was holding for convenience (the no-panic contract, memory.cc). A hard
  // flush is the only sync point a long eager main reaches, which makes it
  // the only place a program that grows without transferring can be stopped
  // before the machine is.
  if (hard) governor_admit(0, MemWhere::kFlush);
  // METALJAX_MEMDBG: the eager path's meter, and the one the memory ladder
  // reads. It has to come from INSIDE the dylib because a flush point is
  // reachable from nowhere else: an embedder can only sample BETWEEN
  // executes, and this program spends its whole life inside one. (An
  // embedder's `mlx.core` does read the same counters -- one libmlx image is
  // loaded, shared by the plugin and the extension -- but it never gets to
  // look while a bound is being tested.) `cache=` is the pool as the flush
  // leaves it; `was=` what it held before the trim.
  if (g_cfg.memdbg && hard) {
    // `foot=` is the process footprint the bound was computed from (the
    // guard's metric, runtime.cc `phys_footprint`), `cap=` the ceiling it
    // was allowed to reach and `n=` this PROGRAM's own hard-flush count --
    // the three numbers `flush_bound` decided on, so a flight log says not
    // just what the bound was but which of its rules chose it. All appended
    // AFTER `bound=`, which the execute_test contracts parse.
    const int64_t foot = phys_footprint();
    debug_line("[metaljax-mem] flush #" + std::to_string(g_stats.flushes) +
               ": active=" + std::to_string(mx::get_active_memory() >> 20) +
               "MB cache=" + std::to_string(mx::get_cache_memory() >> 20) +
               "MB" +
               (trimmed ? " (was " + std::to_string(cache_before >> 20) + "MB)"
                        : "") +
               " bound=" + std::to_string(bound < 0 ? -1 : (bound >> 20)) +
               "MB foot=" + std::to_string(foot < 0 ? -1 : (foot >> 20)) +
               "MB cap=" +
               std::to_string(g_cfg.flush_clear_bytes < 0
                                  ? -1
                                  : (g_cfg.flush_clear_bytes >> 20)) +
               "MB n=" + std::to_string(program_flushes));
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
                  HostFn host, int64_t regrid) {
  for (int s : ins) check_slot(s);
  for (int s : outs) check_slot(s);
  for (int s : drops) check_slot(s);
  if ((op == kMslScan) != (msl != nullptr))
    throw std::invalid_argument("tape: msl plan without an msl entry");
  if ((op == kHostCall) != static_cast<bool>(host))
    throw std::invalid_argument("tape: host callable without a host entry");
  if (regrid >= 0 && !is_emulated(regrid))
    throw std::invalid_argument("tape: regrid onto a non-emulated dtype");
  ops_.push_back(Entry{op, std::move(ins), std::move(outs),
                       std::move(attrs), std::move(fattrs),
                       std::move(payload), regrid, std::move(drops),
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
      // Control flow reads a predicate or a trip count; a host call reads
      // its operands so a handler off the device can see them, and its
      // effect (a print, a callback) has already happened by the time the
      // entry returns. Both make "building" this program mean RUNNING it,
      // which is the property run_while's pipelined path needs.
      if (e.op == kWhile || e.op == kIf || e.op == kCase ||
          e.op == kHostCall) {
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
        eager_flush(env, flushes_);
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

  // The emulated grids, in ONE place (see Entry::regrid). An entry whose
  // result type is i4/ui4 or one of the OCP FP4/FP6 grids -- and the convert
  // onto any emulated type -- rounds here, after the handler that computed
  // in the wide storage dtype.
  if (e.regrid >= 0) {
    for (int s : e.outs) {
      if (env[s]) env[s] = quantize_emulated(*env[s], e.regrid);
    }
  }

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
      std::vector<mx::array> outs = call(inputs, false);
      // A compiled call only BUILDS the graph: MLX generates the fused
      // Metal kernels at EVAL, which is where it says "Too many
      // inputs/outputs fused" for a trace whose arguments exhaust the
      // buffers a kernel may bind. Prove the graph here, inside the
      // ladder, or that failure reaches a caller who can only report it --
      // where the eager path would have computed the program perfectly
      // well (engine.execute's `_can_compile = False` arm is the same
      // move on the Stage 1 side).
      //
      // Once per program, exactly like BodyRunner's probe: an
      // executable's shapes are fixed for its life, so one buildable call
      // proves every later one, and the steady state keeps handing back
      // lazy arrays. Not while an msl plan is unproven -- settle_msl owns
      // that eval, and a generated kernel that fails to build is retired
      // rather than blamed on the graph that traced it.
      if (used_compiled && compile_probe_ && t_msl_pending.empty()) {
        mx::eval(outs);
        compile_probe_ = false;
      }
      return outs;
    } catch (const std::exception& ex) {
      // A governor refusal is not a failure to recover FROM: the machine is
      // out of memory, and every rung of this ladder (retire the compiled
      // path, clear the cache, run again) would spend more of it to arrive at
      // the same answer more slowly. It leaves immediately, keeping the
      // program's compiled path -- the next call may well fit.
      if (is_oom(ex)) throw;
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
