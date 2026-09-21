// metaljax native engine — control flow (ported from Stage 1's
// src/metaljax/ops/control.py, deleted 0.11.6, ef5774d).
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metaljax {

// P49's two knobs, read the way every other perf switch in the engine is
// (`ops_shape.cc::StartPlanEnabled`): a static local, so a measurement pins
// ONE binary and the loader's fixed `configure` argument list does not move
// for a rewrite that is off by default.
bool LoopSpecializeEnabled() {
  static const bool on = [] {
    // Default ON (2026-09-21): bit-identical by construction, -1.2 ms/tok on
    // row 10 with rows 11/14 flat, texmo gate 106/106 in this mode; =0 replays
    // the single generic body as before.
    const char* v = std::getenv("METALJAX_LOOP_SPECIALIZE");
    return v == nullptr || std::strcmp(v, "0") != 0;
  }();
  return on;
}

int64_t LoopSpecializeMax() {
  static const int64_t n = [] {
    const char* v = std::getenv("METALJAX_LOOP_SPECIALIZE_MAX");
    if (v == nullptr) return int64_t{32};
    const int64_t parsed = std::strtoll(v, nullptr, 10);
    return parsed < 0 ? int64_t{0} : parsed;
  }();
  return n;
}

namespace {

// The allocation failures MLX's Metal allocator raises from inside an eval
// walk (backend/metal/allocator.cpp): a request past the device's maximum
// buffer, the live-buffer resource limit (`is_resource_limit`), or a plain
// refusal. A speculative submission that meets one of these is dropped, not
// propagated (run_while's submit-ahead step).
bool is_alloc_failure(const std::exception& e) {
  const std::string what = e.what();
  return what.find("[metal::malloc]") != std::string::npos ||
         what.find("[malloc] Unable to allocate") != std::string::npos;
}

// --------------------------------------------------------------------------
// P49: loop-position specialization (the rule is written out in program.h)
// --------------------------------------------------------------------------

// How many of a body's dynamic slice/update starts must fold before 13
// compile traces are worth building. A body with nothing to fold must not
// pay the trace time for a rewrite that would remove nothing.
constexpr int64_t kSpecMinFolds = 4;

// The most positions one body may keep a start table for, whatever the
// variant cap allows per call: a loop whose START moves from call to call
// (a decode loop's does not) would otherwise accumulate a table per
// position it is ever entered at.
constexpr size_t kSpecCacheMax = 4096;

// The most result bytes an entry may produce and still be run by the VALUE
// pass. A start index is a scalar; this is the guard that keeps a
// broadcast_in_dim of one to a million elements out of a walk that runs
// once per loop position.
constexpr int64_t kSpecFoldBytes = 1024;

// The opcodes a start index is spelled with: small integer arithmetic and
// the rearrangements around it. An entry outside this set is not folded --
// not because its handler would answer differently (the value pass runs the
// SAME handler the trace would) but because the pass runs it EAGERLY, once
// per loop position, and only these are cheap and free of effects enough to
// run speculatively.
bool IsFoldableOp(int op) {
  switch (op) {
    case kConstant:
    case kConvert:
    case kReshape:
    case kTranspose:
    case kBroadcastInDim:
    case kSlice:
    case kConcatenate:
    case kIota:
    case kAdd:
    case kSubtract:
    case kMultiply:
    case kDivide:
    case kRemainder:
    case kMaximum:
    case kMinimum:
    case kNegate:
    case kAbs:
    case kSign:
    case kClamp:
    case kSelect:
    case kCompare:
    case kAnd:
    case kOr:
    case kXor:
    case kNot:
    case kShiftLeft:
    case kShiftRightLogical:
    case kShiftRightArithmetic:
      return true;
    default:
      return false;
  }
}

// An index value, and nothing else: a start the value pass carries must be
// a handful of integers. Anything wider is either not an index or too big
// to fold, and the entry that produced it stays dynamic.
bool IsIndexArray(const mx::array& a) {
  return a.size() <= 64 &&
         (a.dtype() == mx::bool_ || mx::issubdtype(a.dtype(), mx::integer));
}

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
                                   int64_t trip, int64_t K, int64_t cost,
                                   int64_t start, int counter_slot,
                                   mx::Dtype counter_dtype) {
  std::vector<mx::array> vals = ins;
  const int64_t sync_every =
      std::max<int64_t>(1, 75000 / std::max<int64_t>(K * cost, 1));
  const int64_t nchunks = trip / K;
  const int64_t rem = trip % K;

  // P49: may this loop compile a body PER CHUNK, with the layer index folded
  // into the slice spellings?  Four things decide it, once per body:
  //
  //   * the knob (METALJAX_LOOP_SPECIALIZE, off by default),
  //   * the body compiles at all -- an interpreted body would pay the
  //     folding for nothing, and a chunked replay is the only caller here,
  //     which is what bounds the variant count by `trip/K + trip%K`,
  //   * that count is within METALJAX_LOOP_SPECIALIZE_MAX (32), and
  //   * the body really has starts to fold (>= kSpecMinFolds of them), read
  //     off the FIRST position's table rather than guessed from the tape.
  //
  // The verdict is sticky per body (`spec_state`): a decline must not redo
  // the probe once per token, and a loop that specializes narrates once.
  bool spec_on = LoopSpecializeEnabled() && body->spec_state() != 2;
  if (spec_on) {
    const int64_t variants = nchunks + rem;
    std::string why;
    if (!body->may_compile(static_cast<int>(K))) {
      why = "the body is not compiled";
    } else if (variants > LoopSpecializeMax()) {
      why = "variants=" + std::to_string(variants) + " > max=" +
            std::to_string(LoopSpecializeMax());
    } else {
      const LoopSpec* probe =
          body->loop_spec(counter_slot, counter_dtype, start);
      if (probe == nullptr) {
        why = "no start table";
      } else if (probe->folded() < kSpecMinFolds) {
        why = "only " + std::to_string(probe->folded()) + " of " +
              std::to_string(probe->ntotal) +
              " dynamic slice/update start(s) fold";
      } else if (body->spec_state() == 0) {
        body->set_spec_state(1);
        debug_print(
            "loop specialize: trip=" + std::to_string(trip) +
            " K=" + std::to_string(K) +
            " variants=" + std::to_string(variants) + " static starts=" +
            std::to_string(probe->folded()) + "/" +
            std::to_string(probe->ntotal) + " (" +
            std::to_string(probe->nslice) + " slice, " +
            std::to_string(probe->nupdate) + " update)");
      }
    }
    if (!why.empty()) {
      if (body->spec_state() != 2) {
        body->set_spec_state(2);
        g_stats.spec_declines++;
        debug_print("loop specialize: declined (" + why + ")");
      }
      spec_on = false;
    }
  }

  auto chunk = [&](int64_t repeat, int64_t pos0,
                   const std::vector<mx::array>& v) {
    std::vector<mx::array> flat(v);
    flat.insert(flat.end(), caps.begin(), caps.end());
    if (spec_on && body->may_compile(static_cast<int>(repeat))) {
      // The positions this call runs are `pos0 .. pos0 + repeat - 1`, from
      // the same `start` the probe used. A variant is therefore only ever
      // called at the position it was built for.
      std::vector<const LoopSpec*> specs;
      specs.reserve(static_cast<size_t>(repeat));
      for (int64_t r = 0; r < repeat; r++) {
        const LoopSpec* s =
            body->loop_spec(counter_slot, counter_dtype, pos0 + r);
        if (s == nullptr || s->folded() == 0) break;
        specs.push_back(s);
      }
      if (static_cast<int64_t>(specs.size()) == repeat) {
        bool failed = false;
        std::string what;
        try {
          std::vector<mx::array> out =
              body->compiled_spec(static_cast<int>(repeat), pos0, specs)(flat);
          // The same probe the generic chunk takes, for the same reason:
          // nothing unproven may reach an async submission.
          body->prove_spec(static_cast<int>(repeat), pos0, out);
          g_stats.compiled_calls++;
          g_stats.spec_calls++;
          return out;
        } catch (const std::exception& ex) {
          if (is_oom(ex)) throw;
          failed = true;
          what = ex.what();
        }
        // Recovery OUTSIDE the handler (BodyRunner's rule): the failed
        // trace's arrays must be gone before anything allocates again.
        if (failed) {
          debug_print(std::string("loop specialize: variant failed (") +
                      what + "); falling back to the generic body");
          body->drop_spec();
          spec_on = false;
        }
      }
    }
    if (body->may_compile(static_cast<int>(repeat))) {
      g_stats.compiled_calls++;
      std::vector<mx::array> out =
          body->compiled(static_cast<int>(repeat))(flat);
      // The probe this path was missing. Every other caller of a compiled
      // graph settles its first call (`BodyRunner::bind`, `run_recovering`,
      // and `MslPlan::run` for a kernel launched outside a trace); a chunk
      // went straight to the async submission below, so a chunk graph MLX
      // could not build -- including a generated kernel traced into it --
      // failed inside `mx::async_eval`, which abandons the events it had
      // already attached and wedges the next blocking eval forever
      // (Program::prove_compiled). It fails HERE now, where the catch below
      // already knows what to do with it: stop chunking, replay single-step.
      //
      // Once per variant per program: the loop's later chunks and every
      // later execute of the same executable skip straight past it.
      body->prove_compiled(static_cast<int>(repeat), out);
      return out;
    }
    std::vector<mx::array> out = v;
    for (int64_t r = 0; r < repeat; r++) out = run_body(body, out, caps, false);
    return out;
  };
  // The schedule, once per distinct (trip, K) per body under METALJAX_DEBUG:
  // `nchunks` compiled K-chunks, each submitted as it is built (a blocking
  // flush every `sync_every` of them), then `rem` single-step replays left
  // LAZY for the final flush.  `submits` counts `loop_submit`s, `flushes`
  // the blocking `loop_flush`es, the final one included.
  //
  // Why the singles trail lazily instead of being submitted one by one, and
  // why every submission is a cost in itself (T4, findings2): a submission
  // PINS every carry it is handed -- `mx::async_eval`'s Synchronizer node
  // keeps the inputs' buffers in its command-buffer completion handler
  // until the device is done -- so the NEXT replay's in-place update of a
  // carried accumulator (a `dynamic_update_slice` chain writing one slab
  // per iteration into a stacked output) finds its operand not donatable
  // and copies the whole stack.  Measured 0.35 ms per boundary per 64 MB of
  // accumulator; +80 ms per step on the maxtext 0.6B train row when its
  // 28-layer loops went from one submission to thirteen, and 2-5x on texmo's
  // lstm/gru.1024 rows at K=2-4 (logs/t4-chunksubmit/findings2.txt).  A
  // lazy chain of singles is evaluated by ONE eval at the flush, in place
  // all the way; the K-chunks stay submitted because a chunk is `K`
  // iterations of device work per boundary, and the lowering sizes K so
  // that boundary is cheap next to it (metal_lowering.cc kChunkBytesMb /
  // kChunkAccMb).
  // How far the host may run ahead of the device, in submitted chunks
  // (`g_cfg.chunk_inflight`, METALJAX_CHUNK_INFLIGHT).  MLX's own
  // back-pressure (transforms.cpp MAX_ACTIVE_TASKS) counts only the command
  // buffers its 800-op / 512 MB rule commits in the MIDDLE of an eval, never
  // the one `async_eval` commits at its end -- so a chunk that fits one
  // command buffer is never counted, and a loop of such chunks lets the
  // host submit every one of them before the device has finished the
  // first.  Each chunk in flight holds its transients (the completion
  // handlers pin them until the device is done), so the loop's whole
  // working set is allocated at once, and in a process whose buffer pool
  // already holds another program's leftovers every one of those
  // allocations misses the pool: +350 ms per 64-step chunk on texmo's
  // gru.256 row at K=2 inside the suite process, nothing standalone
  // (findings2 section 2d).  Waiting on the carries of the chunk
  // `inflight` behind keeps the host that many chunks ahead and no more:
  // the device never idles (inflight-1 chunks stay queued) and a finished
  // chunk's transients are back in the pool before the next one allocates.
  // The ring holds each submitted chunk's completion EVENTS, never its
  // CARRIES -- which is the whole of the back-pressure and none of its
  // former cost.
  //
  // Holding the arrays made the ring an observer of every carry for
  // `inflight` chunks, and `array::is_donatable` refuses on the array's
  // use count before the vendored fork's through-pins test is ever reached
  // (`array.cpp`: `array_desc_.use_count() != 1` -> false).  So the FIRST
  // update of chunk i+1 on a carried accumulator -- a decode loop's KV
  // cache, `metaljax.kv_update` or a `dynamic_update_slice` on the stack --
  // found its operand held by ring slot `i % inflight` and copied the whole
  // cache instead of writing its window in place.  Once per carry per chunk
  // boundary: 88 of row 10's 112 `slice_update_copied` per token.
  //
  // One `mx::async_eval` attaches ONE event per stream to every array it
  // schedules and signals it after the walk, so waiting on the events a
  // chunk left behind waits for exactly that chunk's device work -- what
  // `array::wait` did, by the same event.  A carry with no valid event was
  // never scheduled (a pass-through) or was already settled by a blocking
  // `loop_flush`; there is nothing to wait for either way, exactly as
  // before.
  //
  // `mx::Event::wait` is not an exported symbol of the library, so each
  // event is held by a STANDIN array -- no data, no primitive, no inputs,
  // and therefore no buffer and no claim on anything the loop carries --
  // whose `array::wait()` is that event's wait and nothing else.
  const int64_t inflight = std::max<int64_t>(1, g_cfg.chunk_inflight);
  auto completion_of = [](const std::vector<mx::array>& v) {
    std::vector<mx::array> waits;
    for (const mx::array& a : v) {
      if (!a.event().valid()) continue;
      mx::array w(mx::Shape{}, mx::bool_, nullptr, {});
      w.attach_event(a.event());
      waits.push_back(std::move(w));
    }
    return waits;
  };
  if (body->narrate_chunk_plan(trip, K)) {
    const int64_t blocking = nchunks / sync_every;
    debug_print("chunked loop: trip=" + std::to_string(trip) +
                " K=" + std::to_string(K) + " plan=" + std::to_string(nchunks) +
                "x" + std::to_string(K) + "+" + std::to_string(rem) +
                "x1 submits=" + std::to_string(nchunks - blocking) +
                " flushes=" + std::to_string(blocking + 1) +
                " inflight=" + std::to_string(inflight) +
                " (singles trail lazily)");
  }
  std::vector<std::vector<mx::array>> ring(static_cast<size_t>(inflight));
  for (int64_t i = 0; i < nchunks; i++) {
    vals = chunk(K, start + i * K, vals);
    // Async-flush each chunk (a blocking sync per chunk serializes CPU
    // and GPU); block only often enough to bound pending buffers.
    if ((i + 1) % sync_every == 0) {
      loop_flush(vals, sync_every * K * cost);
    } else {
      loop_submit(vals);
    }
    std::vector<mx::array>& slot = ring[static_cast<size_t>(i % inflight)];
    for (mx::array& a : slot) a.wait();   // chunk i - inflight, if any
    // Taken AFTER the submission: that is where the events are attached.
    slot = completion_of(vals);
  }
  for (int64_t i = 0; i < rem; i++)
    vals = chunk(1, start + nchunks * K + i, vals);
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

  // Is an iteration ONE compiled graph call, or `num_ops()` interpreted
  // entries? What a speculative build costs -- and therefore whether the
  // dynamic loop below should pipeline -- is that difference and nothing
  // else. Read once, at loop entry: a body that later loses its compiled
  // graph to `drop_compiled` (a recovery path, and a degraded one already)
  // keeps the layout the loop started with.
  bool compiled() const { return compiled_; }

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

// P49. The start indices this body's dynamic slices and updates take at ONE
// loop position, resolved on the host. Declared in program.h, where the rule
// and its bit-identity argument are written out.
//
// Two passes in one walk over the tape:
//
//   STRUCTURAL -- a slot is "static" if it is the loop counter, or the
//     result of an entry all of whose inputs are static (a constant's inputs
//     are none, so every small integer constant seeds too). Sound by
//     construction: a program input that is not the counter is never
//     static, so nothing that varies between iterations can be folded.
//
//   VALUE -- the static entries are RUN, through `Program::step`, with the
//     counter's slot holding `pos`. The handlers that compute the value are
//     therefore the same ones the traced body would use; nothing about
//     what a convert or a clamp means is written a second time here. The
//     drop lists are cleared for the pass (a slot released mid-walk would
//     be unavailable to a later slice) and the whole walk's arrays die with
//     the local env.
//
// The start itself is then the handler's own expression --
// `clip(astype(raw, int32), 0, bound)` on the same value -- evaluated once
// and read back as an integer. Costs one small eval per position, on the
// first call that asks; a decode loop's positions are the same every token.
const LoopSpec* Program::loop_spec(int counter_slot, mx::Dtype counter_dtype,
                                   int64_t pos) {
  if (counter_slot < 0 || counter_slot >= nslots_) return nullptr;
  if (spec_counter_ != counter_slot) {
    spec_cache_.clear();
    spec_counter_ = counter_slot;
  }
  auto cached = spec_cache_.find(pos);
  if (cached != spec_cache_.end()) return cached->second.get();
  if (spec_cache_.size() >= kSpecCacheMax) return nullptr;
  // A position the counter's dtype cannot hold exactly is not this loop's
  // position at all. Integer counters only: `AnalyzeCounted` proved the body
  // adds ONE to it, and `start + j` is that sum for every j.
  if (!mx::issubdtype(counter_dtype, mx::integer)) return nullptr;
  if (pos < 0 || pos > (int64_t{1} << 31) - 1) return nullptr;

  auto spec = std::make_unique<LoopSpec>();
  spec->pos = pos;

  struct PendingStart {
    const Entry* e;
    std::vector<mx::array> vals;
    bool update;
  };
  std::vector<PendingStart> pending;

  try {
    std::vector<std::optional<mx::array>> env(static_cast<size_t>(nslots_));
    std::vector<char> is_static(static_cast<size_t>(nslots_), 0);
    env[static_cast<size_t>(counter_slot)] = mx::array(pos, counter_dtype);
    is_static[static_cast<size_t>(counter_slot)] = 1;

    for (const Entry& e : ops_) {
      if (e.op == kDynamicSlice || e.op == kDynamicUpdateSlice) {
        // Never static itself: operand 0 is the stack. What is asked of it
        // is only whether each start the START PLAN kept is known here.
        spec->ntotal++;
        const std::vector<int64_t>& at = e.attrs;
        const bool update = e.op == kDynamicUpdateSlice;
        const size_t first = update ? 2 : 1;
        const int64_t rank = at[0];
        const size_t plan_at =
            static_cast<size_t>(update ? 1 + rank : 1 + 2 * rank);
        const int64_t nstart = at[plan_at];
        std::vector<mx::array> vals;
        bool ok = nstart > 0;
        for (int64_t j = 0; ok && j < nstart; j++) {
          const size_t p = plan_at + 1 + 3 * static_cast<size_t>(j);
          const int64_t axis = at[p];
          if (at[p + 1] != 0) {
            // Already resolved at lowering (`AppendStartPlan`): carry its
            // value through unchanged, so the folded spelling reads the
            // same start on every axis the plan kept.
            vals.push_back(
                mx::array(static_cast<int>(at[p + 2]), mx::int32));
            continue;
          }
          const size_t operand = first + static_cast<size_t>(axis);
          if (operand >= e.ins.size()) { ok = false; break; }
          const int slot = e.ins[operand];
          if (slot < 0 || slot >= nslots_ ||
              !is_static[static_cast<size_t>(slot)] ||
              !env[static_cast<size_t>(slot)]) {
            ok = false;
            break;
          }
          // The handler's own expression, on the same operand -- but on the
          // CPU stream. It is `clip(astype(raw, int32), 0, bound)` on ONE
          // integer, and running it on the device would hand Metal a command
          // buffer per loop position for arithmetic the host can do for
          // nothing (measured: +90 command buffers on a 90-step scan, for
          // 450 scalars). Integer clip and convert are exact on either
          // device, so the VALUE is the device's; only where it is computed
          // moves.
          const mx::StreamOrDevice cpu = mx::Device::cpu;
          vals.push_back(mx::clip(
              mx::reshape(mx::astype(*env[static_cast<size_t>(slot)],
                                     mx::int32, cpu),
                          mx::Shape{}, cpu),
              mx::array(0, mx::int32),
              mx::array(static_cast<int>(at[1 + axis]), mx::int32), cpu));
        }
        if (ok) {
          pending.push_back({&e, std::move(vals), update});
          if (update) {
            spec->nupdate++;
          } else {
            spec->nslice++;
          }
        }
        continue;
      }
      if (!IsFoldableOp(e.op) || !e.regions.empty() || e.host || e.msl)
        continue;
      if (e.bytes > kSpecFoldBytes) continue;
      bool ready = true;
      for (int s : e.ins) {
        if (s < 0 || s >= nslots_ || !is_static[static_cast<size_t>(s)] ||
            !env[static_cast<size_t>(s)]) {
          ready = false;
          break;
        }
      }
      if (!ready) continue;
      // Without the drop list: the pass's env is its own, and a slot the
      // real walk releases here may still be a later slice's start.
      Entry probe = e;
      probe.drops.clear();
      bool ok = true;
      try {
        step(probe, env, false);
      } catch (const std::exception&) {
        ok = false;
      }
      if (ok) {
        for (int s : probe.outs) {
          if (s < 0 || s >= nslots_ || !env[static_cast<size_t>(s)] ||
              !IsIndexArray(*env[static_cast<size_t>(s)])) {
            ok = false;
            break;
          }
        }
      }
      for (int s : probe.outs) {
        if (s < 0 || s >= nslots_) continue;
        if (ok) {
          is_static[static_cast<size_t>(s)] = 1;
        } else {
          env[static_cast<size_t>(s)].reset();
        }
      }
    }

    // One eval for the whole position: every start is a rank-0 int32.
    std::vector<mx::array> flat;
    for (const PendingStart& p : pending)
      flat.insert(flat.end(), p.vals.begin(), p.vals.end());
    if (!flat.empty()) mx::eval(flat);
    for (const PendingStart& p : pending) {
      std::vector<int32_t> starts;
      starts.reserve(p.vals.size());
      for (const mx::array& a : p.vals) starts.push_back(a.item<int32_t>());
      spec->starts.emplace(p.e, std::move(starts));
    }
  } catch (const std::exception& ex) {
    if (is_oom(ex)) throw;   // the governor's refusal is never swallowed
    pending.clear();
    debug_print(std::string("loop specialize: start table failed (") +
                ex.what() + ")");
    return nullptr;
  }

  const LoopSpec* out = spec.get();
  spec_cache_.emplace(pos, std::move(spec));
  return out;
}

// ops/control.py _while, transliterated. Every branch here had a comment
// in that file explaining what it is for; the policy numbers (cost,
// cadence, chunk size, which bodies may be compiled) are computed by the
// same estimators (Stage 1's Python ones, now metal_lowering.cc's ports)
// and arrive in `attrs`.
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
        vals = run_chunked(body, ins, body_caps, trip, K, cost, start,
                           static_cast<int>(k),
                           ins[static_cast<size_t>(k)].dtype());
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
          loop_submit(vals);
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
    // A bool, as part of the graph that is submitted with the carry: the
    // host read of an evaluated bool array is then a WAIT on the event
    // that submission attached and nothing else (MLX's `eval` of an
    // evaluated array is `wait`, `astype` to an array's own dtype is the
    // identity) -- which is what lets the submit-ahead step below read
    // t's condition without queueing behind the iteration it has already
    // submitted after it. A cond is `tensor<i1>` and this is a no-op; the
    // line is here so that stays a fact the loop does not depend on.
    return mx::astype(pred[0], mx::bool_);
  };

  // Can the body be BUILT before its condition is known? Building an MLX
  // graph is pure and lazy, so a body built for an iteration that turns
  // out not to run is simply dropped -- unless the body reads something
  // back to the host, which would make "building" it mean RUNNING it.
  //
  // AND is it worth it? That depends on ONE structural fact -- whether an
  // iteration is a compiled graph call or `num_ops()` interpreted entries
  // -- because that is what a SPECULATIVE build costs, against a saved
  // host round trip of ~150 us.
  //
  //   * A COMPILED body always pipelines. Building it is one
  //     `body->compiled()(flat)` call whatever its tape holds: measured
  //     ~0.7 ms on row 11's keras arm, a 1,844-entry whole-model decode
  //     body, where pipelining is worth -0.8 ms/token (9.0 -> 8.2, four
  //     runs, `pipelined_steps=127`, tokens identical).
  //   * An EAGER body keeps the entry-count rule. There the build really
  //     is per-op work: on a ~2000-entry decode body it cost ~4.5 ms/tok
  //     and LOST (row 5 measured 65.1 vs 60.6 with pipeline off), and
  //     above ~256 entries the round trip is the cheaper side.
  //
  // The old gate applied the eager rule to both, which is how the keras
  // arm -- one compiled graph replayed once per token -- ended up paying
  // two blocking host round trips per token to save a build it was not
  // doing (notes: row11-keras-diag/diagnosis.md 3a).
  //
  // g_cfg.while_pipeline doubles as the eager threshold when > 1, and 0
  // still disables pipelining outright for both kinds (METALJAX_WHILE_-
  // PIPELINE, the bisection knob).
  const int64_t max_entries =
      g_cfg.while_pipeline > 1 ? g_cfg.while_pipeline : 256;
  const bool cheap_to_speculate =
      runner.compiled() ||
      static_cast<int64_t>(body->num_ops()) <= max_entries;
  const bool pipeline = g_cfg.while_pipeline > 0 && cheap_to_speculate &&
                        !body->reads_host() && !cond->reads_host();
  // METALJAX_DEBUG=1: one line per dynamic loop with the vendored MLX's
  // dispatch accounting over the loop -- kernels and command buffers per
  // iteration, and the device's busy vs idle share of the loop's wall time.
  // The snapshot is taken at the last condition read, a blocking point, so
  // everything the loop submitted has been committed; the last buffer may
  // still be completing (`pending`).
  const mx::metal::DispatchStats loop_ds0 =
      g_cfg.debug ? mx::metal::dispatch_stats() : mx::metal::DispatchStats{};
  int64_t loop_steps = 0;
  auto narrate_loop = [&](const char* mode, const std::string& extra) {
    if (!g_cfg.debug) return;
    debug_line(std::string("[metaljax-native] while(") + mode +
               "): steps=" + std::to_string(loop_steps) + " " + extra +
               DispatchDelta(loop_ds0, DispatchSnapshotSettled(), loop_steps));
  };
  if (!pipeline) {
    g_stats.serial_loops++;
    for (;;) {
      mx::array pred = cond_of(vals);
      if (!item_bool(pred)) break;
      vals = runner.run_one(vals);
      loop_steps++;
      // Flush the carry, not nothing: the cond only forces the values it
      // reads, so anything else in the carry (a KV cache) would pile up as
      // unevaluated graph across iterations.
      loop_flush(vals, cost);
    }
    narrate_loop("serial", "");
    write_results(e, env, vals);
    return;
  }

  // Pipelined dynamic loop. Two host round trips per iteration is what
  // made LLM decode stall (M4's verdict): the old shape submitted the
  // condition and waited, then submitted the body and waited, so the GPU
  // was idle for both decisions. Here the body and the NEXT condition are
  // built before the current condition is read back -- and, from the third
  // iteration on, SUBMITTED before it too (submit-ahead, below) -- so by
  // the time the host wakes up the device is already a token ahead.
  //
  // The order of what happens once the condition says "keep going" is
  // load-bearing on the read-first step:
  //   1. drop this iteration's carry, so the update ops in the body it
  //      feeds can donate their buffers (see cond_of);
  //   2. THEN submit -- the WHOLE carry, since the condition only forces
  //      what it reads and a KV cache would otherwise pile up as
  //      unevaluated graph;
  //   3. charge the iteration against the op-unit budget, exactly as the
  //      serial path's loop_flush does.
  // Speculation never touches the carry a loop may return. On a read-first
  // step the body of iteration t is only ever SUBMITTED once t's condition
  // has said true; on a submit-ahead step it is submitted earlier, and
  // what keeps it off the carry is that `vals` is HELD across the
  // submission -- MLX donates only to a use_count of one (mx::array::
  // is_donatable; the vendored fork's through-pins variant keeps that
  // test), so nothing the speculation builds can write in place into
  // anything in `vals`. The one other place a body is EVALUATED early is
  // BodyRunner's probe after a compiled body has been rebound mid-loop,
  // and that is safe for the same reason: `vals` is held across it.
  g_stats.pipelined_loops++;

  // Submit-ahead (gap-rows item 8; METALJAX_WHILE_SUBMIT_AHEAD). The
  // condition read is the loop's one blocking point, and submitting t+1
  // only AFTER it leaves the device idle from the last kernel of t through
  // the host's wake-up, the encode of t+1's first command buffer and
  // Metal's commit-to-start latency: 1.1 ms of a 14 ms token on row 7,
  // 1.9-2.4 of 19.6 on row 4 (`gpu_idle_ms` in the narration). Submitting
  // t+1 first closes that gap; if t's condition then says stop, the work
  // is simply dropped. What it costs, and what keeps it inside the
  // no-panic contract:
  //
  //   * The held carry means the body's in-place chain (metaljax.kv_update
  //     / a slice_update on the carry) copies the cache ROOT into a fresh
  //     buffer once per speculative step and donates down the chain from
  //     there: cache bytes per token, `slice_update_copied` +1 per step.
  //     The carries that are not pass-throughs bound that copy, and
  //     METALJAX_WHILE_AHEAD_COPY_MB caps it -- past the cap the bubble
  //     hidden is worth less than the copy, and the loop reads first.
  //   * One more iteration's pinned transients are in flight. The first
  //     pipelined step measures them (active memory after its submission
  //     minus before its build) and every ahead step asks the governor --
  //     without stalling, reclaiming or refusing -- whether that much plus
  //     the copy still fits under its lines; when it does not, the step
  //     reads first, as before (`ahead_declined`).
  //   * A speculative submission that fails to allocate (the governor's
  //     OOM, Metal's buffer limit, MLX's malloc) is swallowed: the device
  //     is drained so the abandoned walk's encoded kernels retire before
  //     their buffers can be recycled, the speculation is dropped, the
  //     step is rebuilt and read first, and the loop stays read-first.
  //   * The condition read queues nothing behind the speculation: see
  //     cond_of. The host wakes when t is done, not when t+1 is.
  //   * Engaging from the THIRD iteration keeps a loop that stops after
  //     one or two from running a body for nothing: keras-hub's 1-token
  //     generate is a 1-step loop whose body is the whole prompt forward.
  //
  // What remains: when the loop stops, the dropped iteration's device time
  // runs ahead of whatever the program does after the loop -- one token
  // per generate call.
  const bool ahead_enabled = g_cfg.while_submit_ahead > 0;
  bool ahead = false;              // engaged for this loop
  bool ahead_decided = false;      // ...after the first pipelined step
  int64_t ahead_want = 0;          // bytes one more in-flight step adds
  int64_t ahead_copy = 0;          // bytes a speculation copies (a bound)
  int64_t ahead_inflight = 0;      // the measured in-flight set of a step
  int64_t ahead_steps = 0, ahead_declined = 0, ahead_failures = 0;
  auto mb = [](int64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", static_cast<double>(bytes) / 1048576.0);
    return std::string(buf);
  };

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
  loop_steps++;
  loop_flush(vals, cost);
  mx::array pred = cond_of(vals);
  for (;;) {
    // What this process holds before the step is built: the baseline the
    // first pipelined step's in-flight measurement is taken against
    // (nothing of the loop's is in flight here -- the first iteration was
    // flushed).
    const int64_t active_before =
        (ahead_enabled && !ahead_decided)
            ? static_cast<int64_t>(mx::get_active_memory()) : 0;
    // Built, not run: `next` is a lazy graph until something asks for its
    // values, and nothing does until `pred` says this iteration happens
    // (or, on an ahead step, until the submission below). Pure host work,
    // and it overlaps whatever the device is still doing for the carry
    // and condition submitted last time round.
    std::vector<mx::array> next = runner.run_one(vals);
    mx::array npred = cond_of(next);
    bool submitted = false;
    if (ahead) {
      if (!governor_fits(ahead_want)) {
        ahead_declined++;
        g_stats.ahead_declines++;
      } else {
        std::vector<mx::array> pending(next);
        pending.push_back(npred);
        bool failed = false;
        try {
          loop_submit(pending);            // `vals` held: nothing donates
          submitted = true;
        } catch (const std::exception& ex) {
          if (!is_oom(ex) && !is_alloc_failure(ex)) throw;
          debug_print(std::string("submit-ahead failed (") + ex.what() +
                      "); reading first for the rest of this loop");
          failed = true;
        }
        if (failed) {
          // Recovery OUTSIDE the handler, in this order: retire what the
          // abandoned walk had already encoded while its buffers are still
          // held (`pending`, `next`), THEN drop them, THEN allocate again.
          ahead = false;
          ahead_failures++;
          g_stats.ahead_declines++;
          mx::synchronize();
          pending.clear();
          next.clear();
          gc_collect();
          mx::clear_cache();
          next = runner.run_one(vals);
          npred = cond_of(next);
        }
      }
    }
    const bool go = loop_item_bool(pred);    // the one blocking point
    // An evaluated condition is detached from the carry it was computed
    // from -- but only once MLX has actually walked it, and only if the
    // host read did not have to build a converted copy first. Dropping it
    // outright is one move and needs neither to be true.
    pred = npred;
    if (!go) break;   // `vals` intact (see above); a speculation in flight
                      // runs to completion into buffers nothing reads
    if (ahead_enabled && !ahead_decided) {
      // The bytes a speculation would have to copy: every carry the body
      // does not pass through unchanged (a pass-through comes back as the
      // same array), which bounds the ones it updates in place.
      for (size_t i = 0; i < vals.size() && i < next.size(); i++)
        if (next[i].id() != vals[i].id())
          ahead_copy += static_cast<int64_t>(vals[i].nbytes());
    }
    vals = std::move(next);                  // (1) release the old carry
    if (!submitted) {
      std::vector<mx::array> pending(vals);
      pending.push_back(pred);
      loop_submit(pending);                  // (2) submit, do not wait
      if (ahead_enabled && !ahead_decided) {
        ahead_decided = true;
        ahead_inflight = std::max<int64_t>(
            0, static_cast<int64_t>(mx::get_active_memory()) - active_before);
        ahead_want = ahead_inflight + ahead_copy;
        ahead = ahead_copy <= g_cfg.while_ahead_copy_bytes;
        if (g_cfg.debug)
          debug_print(std::string("submit-ahead ") + (ahead ? "on" : "off") +
                      ": copy=" + mb(ahead_copy) + "MB (cap " +
                      mb(g_cfg.while_ahead_copy_bytes) + "MB) inflight=" +
                      mb(ahead_inflight) + "MB");
      }
    } else {
      ahead_steps++;
      g_stats.ahead_steps++;
    }
    g_stats.pipelined_steps++;
    loop_steps++;
    loop_account(cost);                      // (3) same cadence, same clears
  }
  narrate_loop(ahead_steps > 0 ? "pipelined+ahead" : "pipelined",
               ahead_enabled
                   ? "ahead_steps=" + std::to_string(ahead_steps) +
                         " ahead_declined=" + std::to_string(ahead_declined) +
                         " ahead_failures=" + std::to_string(ahead_failures) +
                         " ahead_copy_mb=" + mb(ahead_copy) +
                         " ahead_inflight_mb=" + mb(ahead_inflight) + " "
                   : std::string());
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
