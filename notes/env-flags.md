# Environment flags

Every `METALJAX_*` / `MJDBG_*` flag read by the code. Categories:
**user knob** (safe to tune), **workaround-retest** (revisit on
macOS/MLX updates), **debug-bisect** (diagnosis only).

| flag | default | category | purpose |
|---|---|---|---|
| `METALJAX_CHUNK_MAX` | `16` | user knob | max loop iterations per compiled chunk (default 16) |
| `METALJAX_CHUNK_MAX_COST` | `1500` | user knob | only chunk bodies cheaper than this (default 1500) |
| `METALJAX_CLEAR_PERIOD` | `50000` | user knob | engine cache clear every N executes (default 50000); 0 disables |
| `METALJAX_COMPILE` | `1` | debug-bisect | =0 disables mx.compile everywhere |
| `METALJAX_COMPILE_OPTIONS` | `` | user knob | =ignore skips XLA compile-option validation |
| `METALJAX_DEBUG` | `` | debug-bisect | =1 logs loop/compile/msl decisions |
| `METALJAX_EAGER_FLUSH_MB` | `1024` | user knob | eager sync point every N megabytes of estimated result data |
| `METALJAX_EAGER_FLUSH_SYNC` | `1` | user knob | every Nth eager flush is a blocking one (the others `async_eval`) |
| `METALJAX_F64` | `error` | user knob | f64 policy: unset=strict (compute errors), downcast=f32 emulation |
| `METALJAX_FLUSH_CLEAR_MB` | `32768` | user knob | CAP on the pool an eager flush may leave cached; -1 disables the trim (P25 shipped this at 2048 as a fixed watermark; P27 made it a cap) |
| `METALJAX_FLUSH_FLOOR_MB` | `2048` | user knob | plugin-native only: watermark a flush falls back to when neither rule below grants more — P25's shipped value, i.e. the no-regression floor |
| `METALJAX_FLUSH_FOOTPRINT_MB` | `3/8 of RAM` | user knob | plugin-native only: process footprint an eager main's pool is trimmed to stay under (48 GB on a 128 GB machine); 0 disables the pressure rule |
| `METALJAX_FLUSH_MAIN_FLUSHES` | `8` | user knob | plugin-native only: hard flushes a program must have taken before it counts as an eager MAIN and may pass the floor; 0 grants it from the first flush |
| `METALJAX_FLUSH_EARN_MULT` | `2` | user knob | plugin-native only: the BENEFIT gate (P28) — a program may keep this many times the live-set SWING it has demonstrated, and never more than its own live-set high-water; 0 disables the rule, restoring P27's two-rule bound (which cost the two maxtext decode rows 17 GB / 11 GB of peak for no speed) |
| `METALJAX_INGEST_CLEAR_MB` | `8192` | user knob | plugin-native only: reclamation cadence of the host->device TRANSFER path, in megabytes ingested (0 disables); a model load reaches no other sync point |
| `METALJAX_INGEST_ADVISE_KB` | `1024` | user knob | plugin-native only: smallest transfer whose consumed source range is handed back to the OS after the staging copy (the no-panic contract's page-cache discipline); 0 disables |
| `METALJAX_INGEST_SWEEP_MB` | `64` | user knob | plugin-native only: smallest MAPPING the ingest-cadence page-cache sweep invalidates (a checkpoint shard is GBs, a framework resource file is not); 0 disables the sweep |
| `METALJAX_MEM_BUDGET_MB` | `3/4 of RAM` | user knob | plugin-native only: the memory governor's hard line on THIS PROCESS's footprint; past it a transfer or a program is refused with RESOURCE_EXHAUSTED |
| `METALJAX_MEM_FREE_FLOOR_MB` | `1/16 of RAM` | user knob | plugin-native only: the governor's SOFT line — the machine free list below which a load is paced and the page cache swept (a quarter of it is where MLX's pool is given back) |
| `METALJAX_MEM_GOVERNOR` | `1` | user knob | plugin-native only: =0 turns the memory governor (and its page-cache discipline) off entirely |
| `METALJAX_MEM_SAMPLE_US` | `20000` | user knob | plugin-native only: how often the governor may re-read the machine; the fast path is a compare |
| `METALJAX_MEM_STALL_MS` | `5000` | user knob | plugin-native only: how long the governor waits at a hard line before refusing |
| `METALJAX_MEM_SYS_MB` | `3/4 of RAM` | user knob | plugin-native only: the governor's hard line on the MACHINE's unreclaimable memory (wired+anonymous+compressor) |
| `METALJAX_MEM_THROTTLE_KBPS` | `1048576` | user knob | plugin-native only: cumulative transfer rate a load is paced to while the free list is below the floor (1 GB/s) |
| `METALJAX_LOOP_CLEAR_COST` | `500000` | user knob | cache-clear cadence in loop op-units (default 500000) |
| `METALJAX_MATMUL_PRECISION` | `highest` | user knob | default=MLX default arch; unset picks the accurate g16g matmul path |
| `METALJAX_MEMDBG` | `` | debug-bisect | =1 logs active/cache memory at loop clears and execute end |
| `METALJAX_MOE` | `1` | debug-bisect | =0 disables the expert-gather rewrite (Stage 1 and plugin-native) |
| `METALJAX_MOE_VERIFY` | `1` | debug-bisect | =0 skips the router check the expert gather rests on -- a misread axis is then SILENT |
| `METALJAX_MOE_VERIFY_DRAWS` | `3` | user knob | synthetic-logit draws per router check |
| `METALJAX_MSL` | `1` | debug-bisect | =0 disables msl_scan kernel codegen |
| `METALJAX_MSL_COOP_CAP` | `2200000` | user knob | coop dot work cap in elems/step (default 2.2M) |
| `METALJAX_MSL_COOP_MIN_F` | `8` | user knob | min state width for the coop preference (default 8) |
| `METALJAX_MSL_COOP_PREF` | `1` | user knob | =0 restores the pre-0.4.3 vector-over-coop mode pick |
| `METALJAX_MSL_INLANE` | `1` | user knob | =0 disables the in-lane small-dot rewrite (matrix-state cells) |
| `METALJAX_MSL_PACK_TRIGGER` | `30` | user knob | bindings above which kernel inputs pool per dtype (default 30) |
| `METALJAX_MSL_REG` | `16` | user knob | vector-mode register width cap (default 16) |
| `METALJAX_MLX_COMPILE_MODE` | `None` | debug-bisect | plugin-native only: MLX's own compiler mode (`no_fuse`/`no_simplify`/`disabled`), which has no MLX env var; attributes a compiled-vs-eager divergence to MLX's fusion |
| `METALJAX_MSL_VOLATILE` | `t` | workaround-retest | Metal compiler loop miscompile workaround; t=default, tmap/tv/load/0 to retest on OS updates |
| `METALJAX_MSL_WNORM` | `1` | user knob | =0 keeps source weight layouts (skips coalescing materialization) |
| `METALJAX_PLUGIN_PATH` | `None` | user knob | override the plugin dylib location |
| `METALJAX_QMM` | `1` | debug-bisect | =0 disables the quantized-matmul rewrite (Stage 1 and plugin-native) |
| `METALJAX_QMM_BATCH` | `1` | debug-bisect | =0 rejects dots that carry batching dims (a stack of per-expert weights) |
| `METALJAX_QMM_SCALES` | `auto` | user knob | pack scale/bias width: auto keeps the source when lossless, `source` always narrows, `f32` never does |
| `METALJAX_RECOGNIZE` | `1` | debug-bisect | plugin-native only: =0 turns off ALL THREE recognizer emits, i.e. the second (fused) lowering is never built -- the control for what an emit is worth |
| `METALJAX_SDPA` | `1` | debug-bisect | =0 disables the fused-attention rewrite (Stage 1 and plugin-native) |
| `METALJAX_SYNC` | `0` | debug-bisect | (see src/metaljax/engine.py) |
| `METALJAX_TRACE_BUDGET` | `20000` | user knob | max ops in one mx.compile trace (default 20000) |
| `METALJAX_WHILE_PIPELINE` | `1` | debug-bisect | =0 restores the serial (two-round-trip) dynamic while on the native engine |
| `MJDBG_NODSIZE` | `None` | debug-bisect | narrow vector-mode dot cap for bisection |
| `MJDBG_NOHOIST` | `None` | debug-bisect | (see src/metaljax/msl_scan.py) |
| `MJDBG_NONESTED` | `None` | debug-bisect | (see src/metaljax/msl_scan.py) |
| `MJDBG_NOREDREG` | `None` | debug-bisect | disable in-lane register reduces for bisection |
| `METALJAX_VERIFY_COMPILE` | `None` | debug-bisect | plugin-native only: =1 runs every executable a second time op by op and reports outputs that differ from the compiled path; =dump also prints the arguments and both answers |
| `MJDBG_VERIFY_MSL` | `None` | debug-bisect | verify every msl plan against the raw loop; dumps mismatches |

metaljax also sets `MLX_METAL_GPU_ARCH` (accurate-matmul arch pin,
see METALJAX_MATMUL_PRECISION) and `MLX_MAX_OPS_PER_BUFFER` before
importing mlx.
