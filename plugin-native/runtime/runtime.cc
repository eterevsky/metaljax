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

#include <algorithm>
#include <cstdio>

#include <mach/mach.h>       // phys_footprint: the guard's own metric
#include <mach/task_info.h>

#include <mlx/allocator.h>   // trim_cache pokes the allocator directly

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

// Bound MLX's buffer pool to `bytes` WITHOUT dumping it (P25).
//
// `mx::clear_cache()` returns the WHOLE pool to the OS, so the allocations
// that follow are cold Metal buffers; on an eager program whose traffic
// dwarfs its live set that cost the maxtext training row ~70 ms per clear,
// 7 clears a step. What is wanted at the same point is a trim: reclaim the
// excess, keep the rest reusable.
//
// MLX has no "trim to N" call -- it has a cache LIMIT, and reclaims down to
// it "on the next allocation" (mlx/memory.h). So the trim is: set the limit,
// poke the allocator, put the limit back. The poke is an allocator call and
// not a kernel (microseconds, and it takes the mutex the reclaim needs
// anyway); the limit is restored immediately, which is what keeps this a
// property of the FLUSH POINT rather than a global bound on every path --
// the global variant was measured and it costs 2.35x on a compiled decode
// row (see config.cc).
//
// That a ONE-BYTE allocation is enough to make MLX reclaim is a contract, not
// an assumption: `execute_test`'s "the pool stays under its bound" holds the
// pool at 255 MB of a 256 MB watermark through 552 flushes and 18 GB of
// traffic, and reads 4025 MB with the trim disabled. If MLX ever routes small
// allocations past the reclaim, that test says so.
//
// Concurrency (METALJAX_CONCURRENT_EXECUTE=1 only): another thread
// allocating inside this window sees the temporary limit and does the same
// reclaim, which is the reclaim we asked for; another thread's `NoCache`
// (metal_qmm.cc) overlapping it can end up restored one step late, and it
// restores itself on scope exit either way. Both are benign, and both are
// serialized away in every other configuration by the submission lock.
void trim_cache(int64_t bytes) {
  if (bytes < 0) return;
  const size_t prev = mx::set_cache_limit(static_cast<size_t>(bytes));
  mx::allocator::Buffer poke = mx::allocator::malloc(1);
  mx::allocator::free(poke);
  mx::set_cache_limit(prev);
}

// This process's PHYSICAL FOOTPRINT, or -1 if the kernel will not say.
//
// Deliberately the SAME metric the bench guard kills on
// (scripts/model_bench/mem_guard.sh samples `top -l 1 -o mem`, whose MEM
// column is this field) and therefore the metric every budget in STATUS.md
// is written in. It counts the wired Metal buffers MLX holds -- active AND
// cached, since a cached buffer is still an allocation the OS has handed
// out -- plus everything else the process owns; it does NOT count the
// mapped, clean checkpoint pages a streaming load reads through (those are
// RSS, which the guard caps separately).
//
// Two alternatives were rejected. `mx::get_active_memory()` sees only what
// MLX allocated, and the transfer path adopts its staging blocks through
// MLX's alien-buffer constructor (notes/ingest-cadence-2026-08-12.md §3), so
// a 10 GB checkpoint can be resident with MLX reporting almost none of it --
// exactly the phase where the bound has to be tight. `hw.memsize` minus free
// pages is the machine's number, not the process's, and would make one
// benchmark's discipline depend on another process's residency.
//
// task_info(TASK_VM_INFO) is a Mach call into the task's own accounting, a
// few microseconds, and it runs at most once per hard flush.
int64_t phys_footprint() {
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  const kern_return_t rc =
      task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count);
  if (rc != KERN_SUCCESS || count < TASK_VM_INFO_REV1_COUNT) return -1;
  return static_cast<int64_t>(info.phys_footprint);
}

// How much MLX may keep cached at THIS flush point (P27,
// notes/cpp-p27-flush-pressure.md).
//
// P25 left ONE watermark doing two jobs, and its sweep showed them pulling
// apart: the maxtext training row reaches its 0.11.3 anchor only with a
// ~26 GB pool (464 ms/step against 834 at 2 GB -- its step is ~105 GB of
// eager traffic over an 18 GB live set, so every megabyte the pool keeps is
// a Metal buffer it does not re-allocate from the OS), while handing that
// same watermark to every program guard-kills the LoRA row at 68 GB.
//
// WHAT THE LoRA ROW ACTUALLY DOES, measured with this file's meter: its
// blowout is not a pool at all. Somewhere in the keras build/convert phase
// its LIVE set goes 19.6 -> 37.5 -> 46.5 GB in three flushes (about a
// second), and the watermark decides only how much DEAD pool is standing
// beside that spike when it lands: 1.5 GB at P25's shipped 2048 (peak
// footprint 48.5 GB, survives) against 16.2 GB at 32768 (63.2 GB and rising
// when the guard fired). The pool it is standing on is genuinely being
// reused up to that point -- active and cache trade places while the
// footprint sits flat at 16.8 GB for 250 flushes -- so nothing about that
// phase looks wrong until the spike arrives, and no watermark that is high
// enough for the training row is low enough to make the spike safe.
//
// So the bound is not one number. Two things decide it:
//
//  1. WHOSE FLUSH IT IS. Only a program that has already taken
//     `flush_main_flushes` hard flushes -- an eager MAIN, whose traffic
//     dwarfs its live set and which therefore reuses a pool -- is allowed
//     past the floor at all. maxtext's training step qualifies inside its
//     first call (410 hard flushes per step). A program that has just
//     started does not, and on the LoRA row that is the load's own
//     protection: measured, the spike lands inside a program on its
//     SEVENTH flush, so the pool beside it is 1.5 GB rather than 16.2.
//  2. WHAT IT WOULD COST. Even for a main, the pool may claim only what the
//     footprint target has left after the program's own live set is paid
//     for:
//
//         bound = clamp(footprint_target - (footprint - cache), floor, cap)
//
//     `footprint - cache` rather than `footprint`, because the cache is what
//     this function is about to bound: charging a program for the pool it is
//     being asked to shrink makes the bound collapse the moment it is
//     granted (the first big bound would be the last one).
//
// Neither rule subsumes the other, and the LoRA row needs both within four
// flushes: rule 1 keeps the pool small where the spike lands, and rule 2 is
// what catches the same program two flushes later, when it IS past the gate
// and its live set has reached 38 GB -- the bound it gets there is 11.1 GB,
// then 2.2, and the 19.5 GB the spike frees is returned rather than held.
//
// The floor -- P25's shipped 2048 MB -- is the no-regression anchor: it is
// what every program got before, so no program can be trimmed HARDER than it
// was, and both rules above can only ever hand memory back.
//
// The floor is clamped by the cap, not the other way round, so a caller that
// asks for a SMALL cap still gets it: `execute_test`'s "the pool stays under
// its bound" runs at 256 MB and must stay there.
int64_t flush_bound(int64_t cache_now, int64_t program_flushes) {
  const int64_t cap = g_cfg.flush_clear_bytes;
  if (cap < 0) return cap;                     // METALJAX_FLUSH_CLEAR_MB=-1
  const int64_t floor = std::min(g_cfg.flush_floor_bytes, cap);
  // The governor's veto (the no-panic contract): while the MACHINE is being
  // squeezed -- the free list at a quarter of the governor's floor, or the
  // kernel itself reporting pressure -- no program keeps a pool for
  // convenience, whatever P27's two rules would have granted it. The HARD
  // line deliberately, not the soft one the ingest pacer uses: a warm page
  // cache puts free under the soft line routinely and this must not cost a
  // training step its pool for that (memory.cc `being_squeezed`). Cached, so
  // it costs a load and a branch; it can only ever LOWER the bound.
  if (governor_squeezed()) return floor;
  if (program_flushes < g_cfg.flush_main_flushes) return floor;
  if (g_cfg.flush_footprint_bytes <= 0) return cap;   // pressure half off
  const int64_t foot = phys_footprint();
  if (foot < 0) return floor;   // no reading: the conservative half of P25
  const int64_t live = std::max<int64_t>(foot - cache_now, 0);
  const int64_t room = g_cfg.flush_footprint_bytes - live;
  return std::min(cap, std::max(floor, room));
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
  // A multi-hour loop is the one execution shape that can walk the machine
  // into pressure without ever reaching a transfer or an eager flush, so the
  // governor is asked here too (sampled: the fast path is a compare).
  governor_admit(0, MemWhere::kFlush);
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
// NB the memory governor is NOT called from here, deliberately: this runs
// after the bytes are already resident, and both of its callers gate the
// transfer with `governor_admit` BEFORE staging it -- where a refusal is
// still a status a caller can be handed rather than an exception escaping
// through the PJRT C boundary.
void ingest_account(int64_t bytes) {
  if (bytes <= 0) return;
  g_stats.ingest_bytes += bytes;
  if (g_cfg.ingest_clear_bytes <= 0) return;   // METALJAX_INGEST_CLEAR_MB=0
  g_ingested_since_clear += bytes;
  if (g_ingested_since_clear < g_cfg.ingest_clear_bytes) return;
  g_ingested_since_clear = 0;
  g_stats.ingest_clears++;
  const int64_t cache_before = mx::get_cache_memory();
  // ...and the page cache the load has filled on its way here (the no-panic
  // contract). This cadence is the right one for it: it is denominated in
  // transferred bytes, which is exactly what a streaming load produces file
  // pages in proportion to, and the sweep costs one pass over this process's
  // VM regions per 8 GB moved.
  sweep_page_cache(g_gov.sweep_min);
  gc_collect();  // dead refcycles pin buffers clear_cache cannot free
  mx::clear_cache();
  if (g_cfg.debug || g_cfg.memdbg)
    debug_line("[metaljax-mem] ingest clear #" +
               std::to_string(g_stats.ingest_clears) + ": ingested=" +
               std::to_string(g_stats.ingest_bytes >> 20) + "MB active=" +
               std::to_string(mx::get_active_memory() >> 20) + "MB cache=" +
               std::to_string(cache_before >> 20) + "->" +
               std::to_string(mx::get_cache_memory() >> 20) + "MB" +
               // ...and the page-cache discipline's own meter (memory.cc):
               // how much of what has been read through a mapping was handed
               // straight back, which is the number a load's flight log is
               // argued on.
               " released=" + std::to_string(g_stats.pages_released >> 20) +
               "MB deactivated=" +
               std::to_string(g_stats.pages_deactivated >> 20) + "MB");
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
