// metaljax native engine — the memory governor (the no-panic contract).
//
// THE CONTRACT (Oleg, 2026-08-17, after kernel panic #9): metaljax must never
// panic the machine. Preferred behaviour under memory pressure is to DEGRADE
// -- trim, pace, stall; acceptable is a clean OOM error that surfaces as
// RESOURCE_EXHAUSTED through the PJRT boundary; a machine wedge is neither.
//
// WHY THE EXISTING DISCIPLINES ARE NOT ENOUGH. Everything else in this
// runtime bounds what METALJAX holds: `trim_cache` bounds MLX's pool at a
// flush, `ingest_account` reclaims per 8 GB of transfer, `flush_bound` (P27)
// spends the pool cap only where the process footprint has room. All of them
// read numbers that were HEALTHY at both machine-wedge panics:
//
//   panic #7 (row 8, 2026-08-04): footprint 53 GB, claimed 58.8 GB, every
//     guard sample "ok", watchdogd starved 91 s. RSS 101.9 GB -- the mapped
//     checkpoint.
//   panic #9 (row 9, 2026-08-17): footprint 53 GB, claimed 59.0 GB, guard
//     budget 80 GB, ingest already throttled to 0.30 GB/s, every sample "ok".
//     LAST row of a 34-row battery -- the page cache full of the previous
//     rows' checkpoints.
//
// Neither process was near any budget it was measured against. What both had
// is a machine whose physical memory was FULL of file-backed pages while a
// streaming load kept demanding more, i.e. a sustained reclaim storm. So the
// governor reads two things this runtime never read before -- the machine's
// own page counters (`host_statistics64`) and the kernel's pressure level --
// and it owns a lever nothing here had: it hands the pages of a consumed
// checkpoint range back to the OS (`release_page_cache`) instead of leaving
// them for the reclaimer to fight over.
//
// MEASURED, on this machine (macOS 26.5, 16 KB pages,
// ~/.cache/metaljax-bench/logs/no-panic-governor/{pagecache,region}_probe.c):
//
//   msync(MS_INVALIDATE) on a read-only MAP_SHARED file mapping DROPS the
//     pages: file-backed 16.39 -> 12.67 GB for a 3.72 GB shard, RSS 3.72 -> 0.
//   madvise(MADV_DONTNEED) only DEACTIVATES: file-backed unchanged, the
//     pages move to the inactive queue (+3.6 GB) -- worth doing where the
//     invalidate cannot be, since the reclaimer takes them first.
//   MADV_FREE_REUSABLE returns EPERM on a file mapping.
//   msync(MS_INVALIDATE) on a MAP_PRIVATE (copy-on-write) file mapping
//     returns 0 and drops nothing -- hence the share-mode test below.
//   msync(MS_INVALIDATE) on ANONYMOUS memory returns 0 and leaves the
//     contents intact (checked byte for byte), so a misfire on a staged host
//     copy would be harmless -- but the region test makes it impossible
//     anyway, because a WRITABLE file mapping is the one case where dropping
//     cached pages could lose data.

#include "program.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include <fcntl.h>
#include <libproc.h>          // proc_regionfilename: the shadow window's path
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include <set>

#include <mlx/mlx.h>

namespace metaljax {

namespace {

// The governor's own state. Guarded by `g_mu` on the slow path; the fast path
// reads `g_last_ns` and `g_pressured` without it, which can cost one extra
// sample under METALJAX_CONCURRENT_EXECUTE=1 and nothing else.
std::mutex g_mu;
uint64_t g_last_ns = 0;          // when the machine was last sampled
MemSample g_sample;              // ...and what it said
bool g_pressured = false;        // the cached verdict the ingest pacer asks
bool g_squeezed = false;         // ...and the harder one the flush asks
uint64_t g_throttle_ns = 0;      // the ingest pacer's deadline (see below)
uint64_t g_reclaim_ns = 0;       // when the ladder last reclaimed
uint64_t g_sweep_ns = 0;         // ...and last swept the page cache
double g_page = 16384.0;

uint64_t now_ns() {
  return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
}

int64_t sysctl_i64(const char* name, int64_t fallback) {
  int64_t v = 0;
  size_t len = sizeof(v);
  if (sysctlbyname(name, &v, &len, nullptr, 0) == 0 && len == sizeof(v))
    return v;
  int32_t v32 = 0;
  len = sizeof(v32);
  if (sysctlbyname(name, &v32, &len, nullptr, 0) == 0 && len == sizeof(v32))
    return v32;
  return fallback;
}

}  // namespace

// hw.memsize: the denominator of every default below.
int64_t machine_memory() {
  static const int64_t kMem = sysctl_i64("hw.memsize", 0);
  return kMem;
}

// The kernel's own opinion: kVMPressureNormal(1) / Warning(2) / Critical(4),
// the same ladder `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` delivers. Read by
// sysctl rather than by a dispatch source because the governor is called from
// paths that have released the GIL and must not depend on a run loop, and
// because a level that is polled at the decision point cannot be stale by a
// queue hop. It is a LAGGING signal on this class of failure (it read 1
// throughout the runs this file exists for), which is why it is one input of
// four rather than the trigger.
int memory_pressure_level() {
  return static_cast<int>(
      sysctl_i64("kern.memorystatus_vm_pressure_level", 1));
}

// The machine's page counters. `free` deliberately excludes speculative
// (read-ahead) pages: they are the page cache's own, and counting them as
// free would hide exactly the read-ahead a streaming load generates.
//
// `claimed` is wired + anonymous + compressor -- the memory that CANNOT be
// reclaimed by dropping a cache, and the same figure `mem_guard.sh` samples
// as "system used" (it learned in 2026-08-03 that top's PhysMem includes the
// file cache and false-tripped on a 60 GB checkpoint read).
MemSample read_machine() {
  MemSample s;
  vm_statistics64_data_t vm;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_size_t page = 0;
  host_page_size(mach_host_self(), &page);
  if (page) g_page = static_cast<double>(page);
  const int64_t p = static_cast<int64_t>(page ? page : 16384);
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm,
                        &count) == KERN_SUCCESS) {
    const int64_t anon =
        static_cast<int64_t>(vm.internal_page_count) -
        static_cast<int64_t>(vm.purgeable_count);
    s.free = (static_cast<int64_t>(vm.free_count) -
              static_cast<int64_t>(vm.speculative_count)) * p;
    s.file = static_cast<int64_t>(vm.external_page_count) * p;
    s.claimed = (static_cast<int64_t>(vm.wire_count) + (anon > 0 ? anon : 0) +
                 static_cast<int64_t>(vm.compressor_page_count)) * p;
    s.purgeable = static_cast<int64_t>(vm.purgeable_count) * p;
  }
  s.footprint = phys_footprint();
  s.total = machine_memory();
  s.pressure = memory_pressure_level();
  s.stamp_ns = now_ns();
  return s;
}

// --------------------------------------------------------------------------
// the configuration
// --------------------------------------------------------------------------
//
// Defaults are FRACTIONS of hw.memsize, not constants: every number here
// qualifies a footprint, which only means something against the memory the
// machine has. On the 128 GB machine this campaign ran on they are 96 / 96 /
// 8 GB, which sit BELOW the bench guard's own ceilings (GUARD_SYS_GB=100,
// budgets 20-80 GB per row) on purpose -- the governor is supposed to act
// while the guard is still watching, so a governor failure is caught by the
// guard rather than by the machine.

MemGovernor g_gov;

void configure_governor(bool on, int64_t budget_bytes, int64_t sys_bytes,
                        int64_t free_floor_bytes, int64_t stall_ms,
                        int64_t sample_us, int64_t advise_min_bytes,
                        int64_t sweep_min_bytes, int64_t throttle_kbps) {
  g_gov.on = on;
  g_gov.budget = budget_bytes;
  g_gov.sys_ceiling = sys_bytes;
  g_gov.free_floor = free_floor_bytes;
  g_gov.stall_ms = stall_ms;
  g_gov.sample_ns = sample_us * 1000;
  g_gov.advise_min = advise_min_bytes;
  g_gov.sweep_min = sweep_min_bytes;
  g_gov.throttle_bps = throttle_kbps * 1024;
}

// --------------------------------------------------------------------------
// the page-cache discipline
// --------------------------------------------------------------------------
//
// A model load reads its checkpoint through a MAPPING -- safetensors' Rust
// memmap2 and numpy's memmap both take the read-only MAP_SHARED path -- and
// every page it touches stays in the page cache afterwards, whether or not
// anybody will read it again. Nobody will: a checkpoint tensor is copied to
// the device once. Left alone, a 65 GB load leaves 65 GB of file-backed
// pages behind it, which is the condition BOTH machine-wedge panics were in
// (and, at the end of a 34-row battery, the condition row 9 STARTED in).
//
// So the transfer path returns them: after the staging copy, the consumed
// range is invalidated, and the pages go back to the free list instead of
// waiting for a reclaimer to find them under pressure.
//
// THE SAFETY TEST IS THE REGION, NOT THE POINTER. `data` belongs to the
// caller, and a PJRT client may hand us anything -- a numpy array, a staged
// host copy, a mapped shard. `mach_vm_region_recurse` says which: an
// `external_pager` region is file-backed, its `protection` says whether the
// caller could have dirtied it, and `share_mode` separates the MAP_SHARED
// case (where the invalidate works) from the copy-on-write one (where it is
// a no-op and MADV_DONTNEED's deactivation is the best available). A range
// that fails any test is left alone.
//
// Nothing here can change what the caller READS: invalidating a clean file
// page drops a cached copy, and the next reference faults it back from the
// file. The probe checks that too (`PRIVATE reread matches=1`).
int64_t release_page_cache(const void* data, int64_t bytes) {
  // METALJAX_MEMDBG narrates the first few decisions and nothing after: a
  // load makes thousands of these, and what a flight log needs is whether the
  // discipline ENGAGED on this loader's buffers -- which the first transfer
  // already answers. (`released=` on the ingest-clear line carries the total.)
  static int64_t narrated = 0;
  const bool say = g_cfg.memdbg && narrated < 4;
  auto decline = [&](const char* why) -> int64_t {
    if (say) {
      narrated++;
      debug_line("[metaljax-gov] page cache: not released (" +
                 std::string(why) + ")");
    }
    return 0;
  };
  if (!g_gov.on || g_gov.advise_min <= 0 || data == nullptr) return 0;
  if (bytes < g_gov.advise_min) return decline("range under the threshold");
  const uintptr_t page = static_cast<uintptr_t>(g_page > 0 ? g_page : 16384);
  const uintptr_t lo = reinterpret_cast<uintptr_t>(data);
  const uintptr_t hi = lo + static_cast<uintptr_t>(bytes);
  // Whole pages INSIDE the range only: a partial page at either end may hold
  // a neighbouring tensor the loader has not read yet, and dropping it would
  // charge that read a fault for nothing.
  uintptr_t start = (lo + page - 1) & ~(page - 1);
  uintptr_t end = hi & ~(page - 1);
  if (end <= start || static_cast<int64_t>(end - start) < g_gov.advise_min)
    return decline("no whole pages inside the range");

  mach_vm_address_t addr = start;
  mach_vm_size_t size = 0;
  vm_region_submap_info_data_64_t info;
  mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
  natural_t depth = 0;
  if (mach_vm_region_recurse(mach_task_self(), &addr, &size, &depth,
                             reinterpret_cast<vm_region_recurse_info_t>(&info),
                             &count) != KERN_SUCCESS)
    return decline("the VM would not describe the range");
  if (addr > start) return decline("no region at that address");
  if (!info.external_pager) return decline("anonymous memory, not a mapping");
  if (info.protection & VM_PROT_WRITE)
    return decline("the mapping is writable, so its pages may be dirty");
  // Clamp to the region: a caller's tensor never spans two mappings, but the
  // arithmetic must not depend on that.
  const uintptr_t region_end = static_cast<uintptr_t>(addr + size);
  if (end > region_end) end = region_end & ~(page - 1);
  if (end <= start) return decline("the region ends before the range does");

  const size_t len = static_cast<size_t>(end - start);
  void* p = reinterpret_cast<void*>(start);
  if (info.share_mode == SM_COW) {
    // Copy-on-write mapping: the invalidate is a measured no-op here, so ask
    // for the deactivation instead -- the pages stay counted but move to the
    // head of the reclaim queue, which is the difference between a reclaimer
    // that scans and one that takes.
    if (madvise(p, len, MADV_DONTNEED) != 0)
      return decline("madvise(MADV_DONTNEED) failed");
    g_stats.pages_deactivated += static_cast<int64_t>(len);
    return static_cast<int64_t>(len);
  }
  if (msync(p, len, MS_INVALIDATE) != 0)
    return decline("msync(MS_INVALIDATE) failed");
  if (say) {
    narrated++;
    debug_line("[metaljax-gov] page cache: released " +
               std::to_string(len >> 20) + "MB of a mapped range");
  }
  g_stats.pages_released += static_cast<int64_t>(len);
  return static_cast<int64_t>(len);
}

// THE OTHER HALF, and the one the real loaders need: a SWEEP of this
// process's own mappings.
//
// `release_page_cache` above only sees the pages a caller hands to the
// transfer path, and the measured fact is that the loaders do not hand them
// over: keras-hub's converter reads a shard, casts it to the variable's dtype
// and gives `device_put` an ANONYMOUS copy, so the mapping it read through is
// never named at the PJRT boundary. Measured on row 4 (gemma4-E2B, 10.3 GB):
// 8 GB ingested, `released=0MB`, and the machine's file-backed pages up by
// exactly the checkpoint -- 25.9 -> 36.0 GB.
//
// So the governor asks the VM what THIS PROCESS has mapped, and invalidates
// the large read-only file mappings it finds. That is blunt by construction,
// and it is safe by the same argument as above: a clean file page's contents
// are in the file, so dropping it costs a re-fault and nothing else. The
// tests it must pass are the ones that say a re-fault is rare -- a checkpoint
// tensor is read once, and a load that streams 65 GB reads no page twice.
//
// What it skips, and why:
//   * WRITABLE mappings -- their pages may be dirty (data loss);
//   * EXECUTABLE mappings -- the dylibs of this process, whose pages would be
//     re-faulted on the next call into them (the plugin's own text is 46 MB);
//   * copy-on-write mappings, where the invalidate is a measured no-op;
//   * anything under `min_region_bytes` (default 64 MB): a checkpoint shard is
//     gigabytes, and everything smaller is somebody's resource file.
int64_t sweep_page_cache(int64_t min_region_bytes) {
  if (!g_gov.on || min_region_bytes <= 0) return 0;
  // At most once a second: the walk is cheap but not free, and the pages it
  // would find twice in a row are the ones it just invalidated.
  const uint64_t now = now_ns();
  if (now - g_sweep_ns < 1000000000ull) return 0;
  g_sweep_ns = now;
  const int64_t page = static_cast<int64_t>(g_page > 0 ? g_page : 16384);
  mach_vm_address_t addr = 0;
  int64_t total = 0, biggest = 0, regions = 0, mapped = 0;
  std::set<std::string> seen;   // one shadow window per file per sweep
  // `mach_vm_region` with VM_REGION_EXTENDED_INFO, not the recursing variant:
  // the extended info carries exactly the three fields the decision needs
  // (`external_pager`, `share_mode`, `pages_resident`) and the flat walk
  // cannot stall on a submap -- the recursing one did, and stopped 256
  // regions in, before the 4 GB mapping the sweep exists for.
  for (int guard = 0; guard < (1 << 16); guard++) {
    mach_vm_size_t size = 0;
    vm_region_extended_info_data_t info;
    mach_msg_type_number_t count = VM_REGION_EXTENDED_INFO_COUNT;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &addr, &size, VM_REGION_EXTENDED_INFO,
                       reinterpret_cast<vm_region_info_t>(&info), &count,
                       &object) != KERN_SUCCESS)
      break;                                   // walked off the end
    regions++;
    const int64_t resident = static_cast<int64_t>(info.pages_resident) * page;
    if (info.external_pager) {
      mapped++;
      if (resident > biggest) biggest = resident;
    }
    const bool candidate =
        info.external_pager &&
        !(info.protection & (VM_PROT_WRITE | VM_PROT_EXECUTE)) &&
        static_cast<int64_t>(size) >= min_region_bytes && resident > 0;
    if (candidate && info.share_mode != SM_COW) {
      // A shared mapping answers for its own pages.  MS_SYNC before the
      // invalidate so that the call is lossless even if some OTHER mapper of
      // the same file has dirty pages: POSIX lets a bare MS_INVALIDATE
      // discard them, and nothing here knows who else has the file open.
      if (msync(reinterpret_cast<void*>(addr), static_cast<size_t>(size),
                MS_SYNC | MS_INVALIDATE) == 0)
        total += resident;
    } else if (candidate) {
      // ...and a COPY-ON-WRITE one does not: measured, MADV_FREE,
      // MADV_DONTNEED and MS_INVALIDATE all return 0 and drop nothing from a
      // MAP_PRIVATE file mapping (`private_probe.c`).  That is the mapping
      // safetensors gives keras-hub, i.e. the one every big load actually
      // uses -- 9.7 GB resident on row 4's 10.3 GB checkpoint.
      //
      // The pages belong to the FILE's object, though, not to the mapping, so
      // a MAP_SHARED window of our own onto the same file can invalidate them
      // and does: file-backed 32.21 -> 29.24 GB and the private reader's
      // residency to zero, with its contents intact and re-faultable
      // (`shadow_probe.c`).  The path comes from the kernel
      // (`proc_regionfilename`), the window is virtual (no read), and each
      // file is done once per sweep.
      char path[PATH_MAX] = {0};
      if (proc_regionfilename(getpid(), addr, path, sizeof(path)) > 0 &&
          seen.insert(std::string(path)).second) {
        const int fd = open(path, O_RDONLY);
        if (fd >= 0) {
          void* win = mmap(nullptr, static_cast<size_t>(size), PROT_READ,
                           MAP_FILE | MAP_SHARED, fd, 0);
          if (win != MAP_FAILED) {
            if (msync(win, static_cast<size_t>(size),
                      MS_SYNC | MS_INVALIDATE) == 0)
              total += resident;
            munmap(win, static_cast<size_t>(size));
          }
          close(fd);
        }
      }
    }
    addr += size;
  }
  if (total > 0) {
    g_stats.pages_released += total;
    g_stats.page_sweeps++;
  }
  if (g_cfg.memdbg)
    debug_line("[metaljax-gov] page cache: swept " +
               std::to_string(total >> 20) + "MB of mapped file pages (" +
               std::to_string(regions) + " regions, " +
               std::to_string(mapped) + " file-backed, biggest resident " +
               std::to_string(biggest >> 20) + "MB)");
  return total;
}

// --------------------------------------------------------------------------
// the ladder
// --------------------------------------------------------------------------

namespace {

std::string gb(int64_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.1f", bytes / 1073741824.0);
  return std::string(buf);
}

std::string where_name(MemWhere where) {
  switch (where) {
    case MemWhere::kIngest: return "transfer";
    case MemWhere::kFlush: return "flush";
    case MemWhere::kExecute: return "execute";
  }
  return "?";
}

void meter(const char* tag, const MemSample& s, int64_t want) {
  if (!g_cfg.memdbg && !g_cfg.debug) return;
  debug_line("[metaljax-gov] " + std::string(tag) + ": want=" + gb(want) +
             "G foot=" + gb(s.footprint) + "G claimed=" + gb(s.claimed) +
             "G free=" + gb(s.free) + "G file=" + gb(s.file) +
             "G press=" + std::to_string(s.pressure));
}

// Reclaim what this process can, at most once every 250 ms: a governor that
// dumped MLX's pool at every admitted transfer would be the P25 regression
// again (a whole-pool dump costs ~70 ms on an eager main).
void reclaim(const char* why) {
  const uint64_t now = now_ns();
  if (now - g_reclaim_ns < 250000000ull) return;
  g_reclaim_ns = now;
  g_stats.mem_reclaims++;
  gc_collect();
  mx::clear_cache();
  // ...and the page cache this process filled, which on a load is the larger
  // half by far: MLX's pool holds megabytes there, the mapped checkpoint
  // gigabytes.  Same lever as the ingest cadence's, asked for here because
  // pressure can arrive between two transfers (or in a phase that makes none).
  sweep_page_cache(g_gov.sweep_min);
  if (g_cfg.memdbg) debug_line("[metaljax-gov] reclaimed: " + std::string(why));
}

// The hard lines. `want` is what the caller is about to make resident.
bool over_hard_line(const MemSample& s, int64_t want, std::string* why) {
  if (g_gov.budget > 0 && s.footprint > 0 && s.footprint + want > g_gov.budget) {
    if (why)
      *why = "this process would hold " + gb(s.footprint + want) +
             " GB, over the " + gb(g_gov.budget) +
             " GB METALJAX_MEM_BUDGET_MB budget";
    return true;
  }
  if (g_gov.sys_ceiling > 0 && s.claimed + want > g_gov.sys_ceiling) {
    if (why)
      *why = "the machine would have " + gb(s.claimed + want) +
             " GB of unreclaimable memory claimed, over the " +
             gb(g_gov.sys_ceiling) + " GB METALJAX_MEM_SYS_MB ceiling";
    return true;
  }
  return false;
}

// The soft line: the regime both wedges happened in -- physical memory full,
// the free list drained, a load still pulling pages in. Not an error: the
// answer is to slow down and hand pages back.
bool under_pressure(const MemSample& s, int64_t want) {
  if (g_gov.free_floor > 0 && s.free - want < g_gov.free_floor) return true;
  return s.pressure > 1;
}

// ...and the HARDER soft line, for the one degrade step that costs a running
// program rather than a load: giving up MLX's buffer pool (`flush_bound`).
//
// The two are separated because "free is under the floor" is a NORMAL state
// on a machine with a warm page cache -- clean file pages are reclaimable on
// demand, and after a few big loads this machine sits at 50+ GB of them with
// single-digit free. Pacing a LOAD there is cheap and the contract prefers it;
// trimming a training step's pool there would cost the maxtext row 1.8x
// (P27's measurement, in reverse) for a machine that is not actually in
// trouble. So the pool goes back only when the free list is genuinely
// scraping -- a quarter of the floor -- or when the kernel itself says so.
bool being_squeezed(const MemSample& s) {
  if (g_gov.free_floor > 0 && s.free < g_gov.free_floor / 4) return true;
  return s.pressure > 1;
}

// Pace the caller so the reclaimer keeps up. The rate is CUMULATIVE, exactly
// like the bench harness's BENCH_STREAM_THROTTLE_GBPS (which row 9 ran under
// the night it panicked -- at 0.30 GB/s, from OUTSIDE the library, where it
// could not see that the page cache was already full): a caller that is
// already slower than the cap pays nothing, and one that is faster is held to
// it. Engages only while the soft line is crossed.
void pace(int64_t want) {
  if (g_gov.throttle_bps <= 0 || want <= 0) return;
  const uint64_t now = now_ns();
  if (g_throttle_ns < now) g_throttle_ns = now;   // fresh episode
  const uint64_t cost =
      static_cast<uint64_t>(1e9 * static_cast<double>(want) /
                            static_cast<double>(g_gov.throttle_bps));
  g_throttle_ns += cost;
  if (g_throttle_ns <= now) return;
  uint64_t sleep_ns = g_throttle_ns - now;
  if (sleep_ns > 1000000000ull) sleep_ns = 1000000000ull;   // 1 s per call
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(sleep_ns / 1000000000ull);
  ts.tv_nsec = static_cast<long>(sleep_ns % 1000000000ull);
  nanosleep(&ts, nullptr);
  g_stats.mem_throttle_ns += static_cast<int64_t>(sleep_ns);
  g_stats.mem_throttles++;
}

}  // namespace

bool is_oom(const std::exception& e) {
  return std::string(e.what()).find("metaljax out of memory") !=
         std::string::npos;
}

bool governor_pressured() { return g_gov.on && g_pressured; }
bool governor_squeezed() { return g_gov.on && g_squeezed; }

MemSample governor_sample(bool force) {
  std::lock_guard<std::mutex> lock(g_mu);
  const uint64_t now = now_ns();
  if (force || now - g_last_ns >= static_cast<uint64_t>(g_gov.sample_ns)) {
    g_sample = read_machine();
    g_last_ns = g_sample.stamp_ns;
    g_pressured = under_pressure(g_sample, 0);
    g_squeezed = being_squeezed(g_sample);
  }
  return g_sample;
}

// The gate. Called before a transfer stages, at every hard flush, at every
// loop sync point and at program entry.
//
//   1. TRIM     -- give back what this process is holding for convenience
//                  (MLX's pool, dead Python cycles), then look again.
//   2. PACE     -- past the soft line, hold the caller to a cumulative rate
//                  so the reclaimer is not racing a 1.2 GB/s streamer. This
//                  is the DEGRADE the contract prefers.
//   3. STALL    -- past a hard line, wait (bounded, narrated) for the
//                  machine to come back: the memory may belong to a phase
//                  that is about to end, and a slow answer beats an error.
//   4. REFUSE   -- still past the hard line: throw, which the PJRT boundary
//                  turns into RESOURCE_EXHAUSTED naming what was needed,
//                  what was available and which variable moves the line.
void governor_admit(int64_t want, MemWhere where) {
  if (!g_gov.on) return;
  const uint64_t now = now_ns();
  // The fast path: two loads and a compare. Sampling every call would put a
  // pair of syscalls (~4 us) on programs whose whole execute is 100 us.
  if (!g_pressured && want < (256LL << 20) &&
      now - g_last_ns < static_cast<uint64_t>(g_gov.sample_ns))
    return;

  MemSample s = governor_sample(/*force=*/want >= (256LL << 20));
  std::string why;
  if (!over_hard_line(s, want, &why)) {
    if (!under_pressure(s, want)) return;
    // The SOFT line, where the cheap half of the ladder lives -- and where
    // being cheap is the requirement, because a warm page cache puts the free
    // list under the floor on a machine that is not in trouble, and a decode
    // step must not pay for that. So:
    //
    //   * the page-cache sweep always (it costs a region walk and gives the
    //     machine back gigabytes),
    //   * MLX's pool only when genuinely squeezed -- dumping it is P25's
    //     70 ms-a-clear regression if it happens on a steady-state row,
    //   * the pace only on a TRANSFER, i.e. only a load can be slowed.
    sweep_page_cache(g_gov.sweep_min);
    if (being_squeezed(s)) reclaim("squeeze");
    s = governor_sample(/*force=*/true);
    if (!under_pressure(s, want)) return;
    g_stats.mem_degrades++;
    meter("pressure", s, want);
    if (where == MemWhere::kIngest) pace(want);
    return;
  }

  // Hard line. Reclaim, then stall, then refuse.
  reclaim("hard line");
  s = governor_sample(/*force=*/true);
  if (!over_hard_line(s, want, &why)) {
    g_stats.mem_degrades++;
    return;
  }
  meter("hard line", s, want);
  const uint64_t deadline = now_ns() +
      static_cast<uint64_t>(g_gov.stall_ms) * 1000000ull;
  uint64_t last_say = 0;
  while (now_ns() < deadline) {
    struct timespec ts = {0, 100000000L};   // 100 ms
    nanosleep(&ts, nullptr);
    g_stats.mem_stall_ns += 100000000LL;
    s = governor_sample(/*force=*/true);
    if (!over_hard_line(s, want, &why)) {
      g_stats.mem_stalls++;
      meter("stall cleared", s, want);
      return;
    }
    if (now_ns() - last_say > 1000000000ull) {
      last_say = now_ns();
      debug_line("[metaljax-gov] waiting for memory at " + where_name(where) +
                 ": " + why);
    }
  }
  g_stats.mem_refusals++;
  meter("refused", s, want);
  throw std::runtime_error(
      "metaljax out of memory at " + where_name(where) + ": " + why +
      ". Needed " + gb(want) + " GB more; the machine has " + gb(s.total) +
      " GB total, " + gb(s.free) + " GB free and " + gb(s.file) +
      " GB in the page cache, and this process holds " + gb(s.footprint) +
      " GB. Raise METALJAX_MEM_BUDGET_MB / METALJAX_MEM_SYS_MB to allow more "
      "(at the risk of paging the machine), or run a smaller model.");
}

}  // namespace metaljax
