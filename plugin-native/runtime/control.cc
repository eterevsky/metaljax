// metaljax native engine — control flow (src/metaljax/ops/control.py).
//
// while, if and case, and the machinery a while needs: chunked replays of a
// compiled body, the pipelined dynamic loop that keeps the device a token
// ahead of the host, and the runner that recovers from the ways MLX's
// compiled path can fail at CALL time. Every policy NUMBER here (cost,
// cadence, chunk size, which bodies may be compiled) is computed by the
// Python estimators and arrives in the entry's attrs -- re-deriving any of
// them would be a second opinion nothing keeps in step with the first.
//
// A branch and a loop bound are read on the HOST, which is what makes these
// ops the boundary of every trace: a block holding one is impure in the
// Python analysis, so no program containing one is ever compiled.

#include "program.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

namespace {

// One uncompiled application of a loop body.
std::vector<mx::array> run_body(Program* body,
                                const std::vector<mx::array>& vals,
                                const std::vector<mx::array>& caps,
                                bool in_trace) {
  std::vector<mx::array> flat(vals);
  flat.insert(flat.end(), caps.begin(), caps.end());
  return body->interpret(flat, in_trace);
}

// ops/control.py _run_chunked.
std::vector<mx::array> run_chunked(Program* body,
                                   const std::vector<mx::array>& ins,
                                   const std::vector<mx::array>& caps,
                                   int64_t trip, int64_t K, int64_t cost) {
  std::vector<mx::array> vals = ins;
  const int64_t sync_every =
      std::max<int64_t>(1, 75000 / std::max<int64_t>(K * cost, 1));
  auto chunk = [&](int64_t repeat, const std::vector<mx::array>& v) {
    std::vector<mx::array> flat(v);
    flat.insert(flat.end(), caps.begin(), caps.end());
    if (body->may_compile(static_cast<int>(repeat))) {
      g_stats.compiled_calls++;
      return body->compiled(static_cast<int>(repeat))(flat);
    }
    std::vector<mx::array> out = v;
    for (int64_t r = 0; r < repeat; r++) out = run_body(body, out, caps, false);
    return out;
  };
  for (int64_t i = 0; i < trip / K; i++) {
    vals = chunk(K, vals);
    // Async-flush each chunk (a blocking sync per chunk serializes CPU
    // and GPU); block only often enough to bound pending buffers.
    if ((i + 1) % sync_every == 0) {
      loop_flush(vals, sync_every * K * cost);
    } else {
      mx::async_eval(vals);
    }
  }
  const int64_t rem = trip % K;
  for (int64_t i = 0; i < rem; i++) vals = chunk(1, vals);
  loop_flush(vals, (trip % std::max<int64_t>(sync_every * K, 1)) * cost);
  return vals;
}

// ops/control.py _BodyRunner: runs a while body, recovering from the ways
// MLX's compiled path can fail at CALL time. Every recovery simply redoes
// the iteration -- bodies are pure, and a failed call leaves the caller's
// carry untouched.
class BodyRunner {
 public:
  BodyRunner(Program* body, const std::vector<mx::array>& caps, int repeat)
      : body_(body), caps_(caps), repeat_(repeat) {
    bind();
  }

  std::vector<mx::array> run_one(const std::vector<mx::array>& vals) {
    for (;;) {
      std::optional<std::vector<mx::array>> out = step(vals);
      if (out) return std::move(*out);
    }
  }

 private:
  void bind() {
    compiled_ = body_->may_compile(repeat_);
    // Probe the freshly bound compiled body once: a compiled call only
    // BUILDS the graph, and MLX generates the fused Metal kernels at
    // eval, so failures like "Too many inputs/outputs fused" land at the
    // next sync point -- by which time the carry has advanced past the
    // iteration a redo could repair. One sync per loop ENTRY: the shapes
    // are fixed for the life of the loop, so one buildable call proves
    // every later one.
    probe_ = compiled_;
  }

  std::optional<std::vector<mx::array>> step(
      const std::vector<mx::array>& vals) {
    bool resource_limit = false;
    std::exception_ptr err;
    try {
      std::vector<mx::array> flat(vals);
      flat.insert(flat.end(), caps_.begin(), caps_.end());
      std::vector<mx::array> out;
      if (compiled_) {
        g_stats.compiled_calls++;
        out = body_->compiled(repeat_)(flat);
      } else {
        out = body_->interpret(flat, false);
      }
      if (probe_) {
        // Inside the try on purpose: this is where a body MLX cannot
        // generate kernels for reports itself. Synchronous (not
        // async_eval) — a Metal build error raised on a worker thread
        // aborts the process.
        mx::eval(out);
        probe_ = false;
      }
      limit_retries_ = 0;
      return out;
    } catch (const std::exception& ex) {
      if (is_oom(ex)) throw;   // the governor's refusal: see run_recovering
      resource_limit = is_resource_limit(ex);
      err = std::current_exception();
    }
    // Recovery OUTSIDE the handler: the failed attempt's arrays must be
    // gone before anything tries to allocate again (the C++ analogue of
    // the traceback that pins a failed trace's buffers in Python).
    if (resource_limit) {
      debug_print("Metal buffer limit hit in while body; clearing cache "
                  "and retrying");
      g_stats.limit_retries++;
      gc_collect();
      mx::clear_cache();
      limit_retries_++;
      // BOUNDED: retrying an oversized compiled trace forever once
      // livelocked a worker for hours.
      if (limit_retries_ == 2 && compiled_) {
        body_->drop_compiled();
        bind();
      } else if (limit_retries_ > 3) {
        std::rethrow_exception(err);
      }
      return std::nullopt;
    }
    if (!compiled_) std::rethrow_exception(err);
    debug_print("compiled while body failed; retrying eagerly");
    body_->drop_compiled();
    bind();
    return std::nullopt;
  }

  Program* body_;
  const std::vector<mx::array>& caps_;
  int repeat_;
  bool compiled_ = false;
  bool probe_ = false;
  int limit_retries_ = 0;
};

}  // namespace

// ops/control.py _while, transliterated. Every branch here has a comment
// in that file explaining what it is for; the policy numbers (cost,
// cadence, chunk size, which bodies may be compiled) are computed by the
// same Python estimators and arrive in `attrs`.
void Program::run_while(const Entry& e,
                        std::vector<std::optional<mx::array>>& env,
                        bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;
  const int64_t ncarry = at[0], ncond_caps = at[1], nbody_caps = at[2];
  const bool counted = at[3] != 0;
  const int64_t k = at[4], bound_kind = at[5], bound = at[6];
  const int64_t cost = std::max<int64_t>(at[7], 1);
  const int64_t period = std::max<int64_t>(at[8], 1);
  const bool chunkable = at[9] != 0;
  const int64_t kmax = at[10];

  Program* cond = e.regions[0].get();
  Program* body = e.regions[1].get();

  std::vector<mx::array> ins, cond_caps, body_caps;
  ins.reserve(static_cast<size_t>(ncarry));
  for (int64_t i = 0; i < ncarry; i++) ins.push_back(in(i));
  for (int64_t i = 0; i < ncond_caps; i++)
    cond_caps.push_back(in(static_cast<size_t>(ncarry + i)));
  for (int64_t i = 0; i < nbody_caps; i++)
    body_caps.push_back(in(static_cast<size_t>(ncarry + ncond_caps + i)));

  std::vector<mx::array> vals;
  if (counted) {
    int64_t n;
    if (bound_kind == 0) {
      n = bound;
    } else if (bound_kind == 1) {
      n = item_int(ins[static_cast<size_t>(bound)]);
    } else {
      n = item_int(cond_caps[static_cast<size_t>(bound)]);
    }
    const int64_t start = item_int(ins[static_cast<size_t>(k)]);
    const int64_t trip = std::max<int64_t>(n - start, 0);
    if (in_trace) {
      // An enclosing mx::compile is tracing us: inline the iterations
      // into that graph. Past 64 of them the trace holds more
      // intermediates than Metal's buffer budget allows, and the answer
      // is the same as the Python engine's -- abort, and let the caller
      // fall back to the eager path (run_recovering does that here).
      // The lowering's `WhileTraceable` carries the same bound
      // (metal_lowering.cc kUnrollMax), so a body it let compile never
      // reaches this throw; keep the two numbers together.
      if (trip > 64)
        throw std::runtime_error(
            "metaljax: refusing to unroll trip=" + std::to_string(trip) +
            " inside a trace");
      g_stats.unrolls++;
      vals = ins;
      for (int64_t i = 0; i < trip; i++)
        vals = run_body(body, vals, body_caps, true);
      write_results(e, env, vals);
      return;
    }
    // Eager loop. Chained replays are expensive (a compiled call
    // evaluates its inputs), so unroll as many iterations as the trace
    // budget allows into each compiled chunk and replay trip/K chunks
    // instead of trip single steps -- while flushing often enough that
    // the buffers a pending replay pins stay bounded.
    int64_t K = 1;
    if (chunkable && !body->no_chunk())
      K = std::max<int64_t>(1, std::min<int64_t>(trip, kmax));
    if (K > 1) {
      bool failed = false;
      try {
        vals = run_chunked(body, ins, body_caps, trip, K, cost);
      } catch (const std::exception& ex) {
        // MLX's compiler can reject big fused traces ("Too many
        // inputs/outputs fused..."). Fall back to single-step replays,
        // from the ORIGINAL carries -- a failed chunk changed nothing.
        debug_print(std::string("chunked loop failed (") + ex.what() +
                    "); falling back to single-step");
        failed = true;
      }
      if (!failed) {
        write_results(e, env, vals);
        return;
      }
      body->set_no_chunk();
      g_stats.chunk_drops++;
    }
    BodyRunner runner(body, body_caps, 1);
    vals = ins;
    // `period` stays the SUBMISSION cadence, but the BLOCKING eval gets a
    // floor: a pessimistically-costed body (an inner plan-less scan charged
    // trip x cost by the estimator) collapses period to 1, and a blocking
    // mx::eval every iteration serializes host and device for the whole
    // loop.  Between blocking points each sync submits (async_eval) and
    // charges the same op-units, so the clear cadence is unchanged.  Safe
    // by construction: the unevaluated graph never exceeds `hard_floor`
    // submissions of one iteration each, the interpreter's own
    // byte-denominated eager_flush still fires INSIDE a big body, and a
    // body whose cost collapsed the period to 1 is big precisely because it
    // holds inner loops with sync points of their own.
    const int64_t hard_floor = 8;
    const int64_t hard_every =
        period * std::max<int64_t>(1, (hard_floor + period - 1) / period);
    for (int64_t i = 1; i <= trip; i++) {
      vals = runner.run_one(vals);
      if (i % period == 0) {
        if (i % hard_every == 0) {
          loop_flush(vals, period * cost);
        } else {
          mx::async_eval(vals);
          loop_account(period * cost);
        }
      }
    }
    write_results(e, env, vals);
    return;
  }

  // Dynamic (non-counted) loop: evaluate the condition each iteration.
  // The BODY still gets compiled -- a data-dependent trip count says
  // nothing about the body, and interpreting it op by op is what made
  // LLM decode Python-dispatch-bound. The cond stays eager: it ends in
  // a host read.
  if (in_trace)
    throw std::runtime_error(
        "metaljax: a dynamic while cannot run inside a trace");
  BodyRunner runner(body, body_caps, 1);
  vals = ins;
  // The condition of ONE carry, as a lazy array. `cargs` is scoped to die
  // here on purpose: it is a second handle on every carry, and a handle
  // still alive when the next iteration's update EVALUATES is the
  // difference between MLX writing a KV cache in place and copying the
  // whole thing (mx::array::is_donatable is a use_count test). Holding it
  // across the body cost 4.6 us per megabyte of cache per token.
  auto cond_of = [&](const std::vector<mx::array>& v) {
    std::vector<mx::array> cargs(v);
    cargs.insert(cargs.end(), cond_caps.begin(), cond_caps.end());
    std::vector<mx::array> pred = cond->interpret(cargs, false);
    if (pred.size() != 1)
      throw std::runtime_error("tape: while cond must return one value");
    return pred[0];
  };

  // Can the body be BUILT before its condition is known? Building an MLX
  // graph is pure and lazy, so a body built for an iteration that turns
  // out not to run is simply dropped -- unless the body reads something
  // back to the host, which would make "building" it mean RUNNING it.
  //
  // AND is it worth it? Building costs ~per-op Python-free but real
  // work; the saved host round trip is ~150 us. On a 4-matmul synthetic
  // body the build is noise and pipelining won 1.9x; on a whole-model
  // decode body (~2000 entries) the speculative build costs ~4.5 ms/tok
  // and LOSES (row 5 measured 65.1 vs 60.6 with pipeline off). Gate on
  // tape size: above ~256 entries the round trip is the cheaper side.
  // g_cfg.while_pipeline doubles as the threshold when > 1.
  const int64_t max_entries =
      g_cfg.while_pipeline > 1 ? g_cfg.while_pipeline : 256;
  const bool pipeline =
      g_cfg.while_pipeline > 0 &&
      static_cast<int64_t>(body->num_ops()) <= max_entries &&
      !body->reads_host() && !cond->reads_host();
  if (!pipeline) {
    g_stats.serial_loops++;
    for (;;) {
      mx::array pred = cond_of(vals);
      if (!item_bool(pred)) break;
      vals = runner.run_one(vals);
      // Flush the carry, not nothing: the cond only forces the values it
      // reads, so anything else in the carry (a KV cache) would pile up as
      // unevaluated graph across iterations.
      loop_flush(vals, cost);
    }
    write_results(e, env, vals);
    return;
  }

  // Pipelined dynamic loop. Two host round trips per iteration is what
  // made LLM decode stall (M4's verdict): the old shape submitted the
  // condition and waited, then submitted the body and waited, so the GPU
  // was idle for both decisions. Here the body and the NEXT condition are
  // built and submitted before the current condition is read back, so by
  // the time the host wakes up the device is already a token ahead.
  //
  // The order of what happens once the condition says "keep going" is
  // load-bearing:
  //   1. drop this iteration's carry, so the update ops in the body it
  //      feeds can donate their buffers (see cond_of);
  //   2. THEN submit -- the WHOLE carry, since the condition only forces
  //      what it reads and a KV cache would otherwise pile up as
  //      unevaluated graph;
  //   3. charge the iteration against the op-unit budget, exactly as the
  //      serial path's loop_flush does.
  // Speculation never touches the carry a loop may return: the body of
  // iteration t is only ever SUBMITTED once t's condition has said true.
  // The one place it is EVALUATED earlier is BodyRunner's probe after a
  // compiled body has been rebound mid-loop -- and that is still safe,
  // because `vals` is held across the build, which is precisely what
  // stops MLX donating (and therefore mutating) anything in it.
  g_stats.pipelined_loops++;
  // The first iteration stays unpipelined. BodyRunner probes a freshly
  // bound compiled body with a SYNCHRONOUS eval (a Metal build error
  // raised on an async worker aborts the process), and that probe should
  // land on an iteration the loop is known to want -- a zero-trip loop
  // must cost exactly one condition here, as it does on the serial path.
  {
    mx::array first = cond_of(vals);
    if (!loop_item_bool(first)) {
      write_results(e, env, vals);
      return;
    }
  }
  vals = runner.run_one(vals);
  loop_flush(vals, cost);
  mx::array pred = cond_of(vals);
  for (;;) {
    // Built, not run: `next` is a lazy graph until something asks for its
    // values, and nothing does until `pred` says this iteration happens.
    // Pure host work, and it overlaps whatever the device is still doing
    // for the carry and condition submitted last time round.
    std::vector<mx::array> next = runner.run_one(vals);
    mx::array npred = cond_of(next);
    const bool go = loop_item_bool(pred);    // the one blocking point
    // An evaluated condition is detached from the carry it was computed
    // from -- but only once MLX has actually walked it, and only if the
    // host read did not have to build a converted copy first. Dropping it
    // outright is one move and needs neither to be true.
    pred = npred;
    if (!go) break;                          // `vals` is intact: see above
    vals = std::move(next);                  // (1) release the old carry
    std::vector<mx::array> pending(vals);
    pending.push_back(pred);
    mx::async_eval(pending);                 // (2) submit, do not wait
    g_stats.pipelined_steps++;
    loop_account(cost);                      // (3) same cadence, same clears
  }
  write_results(e, env, vals);
}

void Program::write_results(const Entry& e,
                            std::vector<std::optional<mx::array>>& env,
                            const std::vector<mx::array>& vals) {
  if (vals.size() != e.outs.size())
    throw std::runtime_error("tape: loop result count mismatch");
  for (size_t i = 0; i < vals.size(); i++) env[e.outs[i]] = vals[i];
}

bool Program::step_control(const Entry& e,
                           std::vector<std::optional<mx::array>>& env,
                           bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;

  switch (e.op) {
    case kWhile:
      run_while(e, env, in_trace);
      break;

    case kMslScan:
      run_msl(e, env, in_trace);
      break;

    case kIf:
    case kCase: {
      // _if / _case: the branch is chosen on the HOST, so both make a
      // block impure in the Python analysis and no program containing
      // one is ever compiled -- which is why reading the predicate here
      // cannot be a sync point inside a trace.
      int64_t which;
      if (e.op == kIf) {
        which = item_bool(in(0)) ? 0 : 1;
      } else {
        which = item_int(in(0));
        which = std::min<int64_t>(
            std::max<int64_t>(which, 0),
            static_cast<int64_t>(e.regions.size()) - 1);
      }
      size_t base = 1;  // ins[0] is the predicate/index
      for (int64_t r = 0; r < which; r++)
        base += static_cast<size_t>(at[static_cast<size_t>(r)]);
      std::vector<mx::array> args;
      int64_t ncaps = at[static_cast<size_t>(which)];
      args.reserve(static_cast<size_t>(ncaps));
      for (int64_t i = 0; i < ncaps; i++) args.push_back(in(base + i));
      std::vector<mx::array> outs =
          e.regions[static_cast<size_t>(which)]->call(args, in_trace);
      if (outs.size() != e.outs.size())
        throw std::runtime_error("tape: branch result count mismatch");
      for (size_t i = 0; i < outs.size(); i++) env[e.outs[i]] = outs[i];
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
