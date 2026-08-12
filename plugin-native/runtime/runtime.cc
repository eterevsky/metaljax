// metaljax native engine — the runtime disciplines.
//
// Every cadence here has a crash or a corruption story behind it, and every
// NUMBER behind it belongs to Python (config.cc copies them in): what lives
// in this file is the mechanism -- when a replay blocks, what it evaluates,
// when it hands buffers back to the OS, and how it recovers from Metal's
// live-buffer limit. src/metaljax/interpreter.py and ops/control.py own the
// comments that explain the values themselves.

#include "program.h"

#include <string>
#include <vector>

#include <cstdio>

namespace metaljax {

namespace {

// The op-unit budget the loop cadence below spends.
int64_t g_flushed_cost = 0;

// The byte budget the ingest cadence below spends (see `ingest_account`).
int64_t g_ingested_since_clear = 0;

}  // namespace

bool is_resource_limit(const std::exception& e) {
  return std::string(e.what()).find("Resource limit") != std::string::npos;
}

// Narration. Deliberately not routed through Python: these fire from paths
// that have released the GIL, some of them while recovering from an
// allocation failure.
void debug_line(const std::string& line) {
  fputs((line + "\n").c_str(), stdout);
  fflush(stdout);
}

// METALJAX_DEBUG.
void debug_print(const std::string& msg) {
  if (g_cfg.debug) debug_line("[metaljax] " + msg);
}

// interpreter.flush_eval: settle `arrays`, recovering once from Metal buffer
// exhaustion. Programs are pure, so clearing and retrying is safe.
void flush_eval(const std::vector<mx::array>& arrays, bool hard) {
  try {
    if (hard) {
      mx::eval(arrays);
    } else {
      mx::async_eval(arrays);
    }
    return;
  } catch (const std::exception& e) {
    if (!is_resource_limit(e)) throw;
  }
  debug_print("Metal buffer limit hit at eager flush; clearing cache and "
              "retrying");
  g_stats.limit_retries++;
  gc_collect();
  mx::clear_cache();
  mx::eval(arrays);
}

// ops.control._loop_flush: a sync point inside a loop. Evaluates pending
// work, keeps the Metal buffer COUNT bounded (MLX's cache is bounded by
// bytes only, and long loops over small models accumulate tiny buffers until
// metal::malloc dies), and recovers once if the limit is hit anyway.
// The blocking half: settle `arrays`, recovering once from exhaustion.
void loop_eval(const std::vector<mx::array>& arrays) {
  bool retry = false;
  try {
    mx::eval(arrays);
  } catch (const std::exception& e) {
    if (!is_resource_limit(e)) throw;
    retry = true;
  }
  if (retry) {
    debug_print("Metal buffer limit hit at loop flush; clearing cache and "
                "retrying");
    g_stats.limit_retries++;
    gc_collect();  // dead refcycles pin buffers clear_cache cannot free
    mx::clear_cache();
    mx::eval(arrays);
  }
}

// The bookkeeping half: charge a sync point's work against the op-unit
// budget and return cached buffers to the OS when it is spent. Split out
// because a pipelined loop's blocking point is a HOST READ of the condition
// rather than an eval of the carry -- a different array to wait on, the same
// iteration of work to charge, and the same cadence either way.
void loop_account(int64_t cost_units) {
  g_stats.loop_flushes++;
  g_flushed_cost += cost_units;
  if (g_cfg.loop_clear_cost > 0 && g_flushed_cost >= g_cfg.loop_clear_cost) {
    g_flushed_cost = 0;
    g_stats.loop_clears++;
    mx::clear_cache();
    if (g_cfg.memdbg)
      debug_line("[metaljax-mem] loop clear: active=" +
                 std::to_string(mx::get_active_memory()) + "B cache=" +
                 std::to_string(mx::get_cache_memory()) + "B");
  }
}

void loop_flush(const std::vector<mx::array>& arrays, int64_t cost_units) {
  loop_eval(arrays);
  loop_account(cost_units);
}

// The INGEST cadence: a reclamation point every `ingest_clear_bytes` taken in
// by transfers (metal_client.cc `BufferFromHostBuffer`, and
// `MetalBuffer::CopyToMemorySpace`, which really allocates). The callers
// charge the bytes the DEVICE ends up holding, which is what the cadence is
// about; a wire that is wider than the device dtype (f64 -> f32, an emulated
// grid) moves more host bytes than it charges, transiently.
//
// WHY IT IS ITS OWN CADENCE. Every other reclamation point above is reached
// by EXECUTING something. A model load executes almost nothing: it is
// thousands of host->device transfers, tens of GB of them, with a handful of
// tiny programs in between -- so a process streaming a 65 GB checkpoint can
// run for minutes without crossing a single flush or loop boundary, while
// the OS page cache is simultaneously saturated by the mapped checkpoint
// (the VM-pressure wedge class: CLAUDE.md item 21 and the panic ledger in
// TASKS.md). Stage 1's big loads got a reclamation point from the BENCH
// HARNESS -- scripts/model_bench/adapter_keras_extra.py `_stream_clear`,
// gc.collect() + mx.clear_cache() every BENCH_STREAM_CLEAR_GB (8) of
// assigned weights, 16 of them in the R1-Distill-32B load -- and a plugin
// cannot depend on its embedder for a discipline the embedder does not know
// it is providing. The default is that harness's number.
//
// WHAT IT RECLAIMS, measured (2026-08-12, notes/data/ingest-cadence-*):
// MLX's cache holds NOTHING from the transfers themselves. The staging
// block becomes the array's storage through MLX's alien-buffer path, so it
// is freed to the C allocator and never enters the buffer cache -- probed at
// 4 KB, 256 KB and 256 MB per transfer, `mx::get_cache_memory()` stays at 0
// through 16 GB of churn. What it does reclaim is what the work AROUND the
// transfers left cached (a loader's casts and copies) plus, on the
// embedder's side, whatever `g_gc_hook` breaks loose. Cheap either way: an
// empty `clear_cache` is microseconds, and at 8 GB the cadence fires ~8
// times in a 65 GB load.
//
// The budget is a plain counter, like `g_flushed_cost`: both of this
// function's callers hold the plugin's submission lock, so it is serialized
// in every configuration but `METALJAX_CONCURRENT_EXECUTE=1`, where a racing
// transfer can cost the cadence a clear -- the same benign inexactness every
// other counter in `g_stats` has.
void ingest_account(int64_t bytes) {
  if (bytes <= 0) return;
  g_stats.ingest_bytes += bytes;
  if (g_cfg.ingest_clear_bytes <= 0) return;   // METALJAX_INGEST_CLEAR_MB=0
  g_ingested_since_clear += bytes;
  if (g_ingested_since_clear < g_cfg.ingest_clear_bytes) return;
  g_ingested_since_clear = 0;
  g_stats.ingest_clears++;
  const int64_t cache_before = mx::get_cache_memory();
  gc_collect();  // dead refcycles pin buffers clear_cache cannot free
  mx::clear_cache();
  if (g_cfg.debug || g_cfg.memdbg)
    debug_line("[metaljax-mem] ingest clear #" +
               std::to_string(g_stats.ingest_clears) + ": ingested=" +
               std::to_string(g_stats.ingest_bytes >> 20) + "MB active=" +
               std::to_string(mx::get_active_memory() >> 20) + "MB cache=" +
               std::to_string(cache_before >> 20) + "->" +
               std::to_string(mx::get_cache_memory() >> 20) + "MB");
}

// The loop counter / branch index of a control-flow op, on the host. Reading
// one is a sync point, exactly as the Python handlers' `.item()` is.
int64_t item_int(const mx::array& a) {
  mx::array v = mx::astype(a, mx::int64);
  v.eval();
  return v.item<int64_t>();
}

bool item_bool(const mx::array& a) {
  mx::array v = mx::astype(a, mx::bool_);
  v.eval();
  return v.item<bool>();
}

// A loop condition's host read. Same value as item_bool, but the wait goes
// through the loop's recovery path: on a pipelined loop this IS the sync
// point, so it is where Metal's buffer limit would surface.
bool loop_item_bool(const mx::array& a) {
  mx::array v = mx::astype(a, mx::bool_);
  loop_eval({v});
  return v.item<bool>();
}

}  // namespace metaljax
