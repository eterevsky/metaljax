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
| `METALJAX_KV_INPLACE` | `1` | debug-bisect | plugin-native only: =0 turns off the KV in-place rewrite (a while body that rebuilds a stacked cache carry from slices of it becomes a chain of `slice_update`s ON the carry, `metal_lowering.cc kv::`); the literal tape is the A/B arm |
| `METALJAX_INGEST_SWEEP_MB` | `64` | user knob | plugin-native only: smallest MAPPING the ingest-cadence page-cache sweep invalidates (a checkpoint shard is GBs, a framework resource file is not); 0 disables the sweep |
| `METALJAX_MLA_KERNEL` | `1` | debug-bisect | plugin-native only: =0 keeps the multi-span attention recognizer but runs its emit on the concat path (concat K/V/mask + MLX fused sdpa) instead of the B3 two-span kernel (runtime/mla.cc) -- the arm that prices the kernel alone |
| `MJDBG_MLA_SOURCE` | `` | debug-bisect | plugin-native only: =1 dumps each generated two-span attention kernel source at build |
| `METALJAX_STACKED_RELAYOUT` | `1` | debug-bisect | plugin-native only: =0 declines the stacked dot's RELAYOUT form (a stack whose contracted axes straddle the layer axis -- maxtext's attention out-projection -- is packed once per executable in its carried layout, B3); off, the slice chain runs as before |
| `METALJAX_STACKED_RELAYOUT_MB` | `2048` | user knob | plugin-native only: cap on the device memory one executable's relaid stacks may hold in total (row 10: 218 MB); a stack past the cap stays sliced, 0 disables the form |
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
| `METALJAX_PROJ_PACK` | `auto` | user knob | plugin-native only: the projection PACK (`metal_proj.cc`, B6): sibling decode-step `dot_general`s over one activation with distinct loop-invariant weights become ONE dot over the weights concatenated along the output axis, packed once per executable (governor-admitted, keyed by the weights' identity like a stacked relayout) with per-consumer slice VIEWS. `0` declines every group; unset/`1`/`auto`: in a group of >= 3 with one member at least twice as wide as the next (q beside k and v) the wide one is left alone when the rest, packed, stay in their own gemv tile (gpt-oss 512+512 = 1024 keeps `bn4`: K+V, bit-identical, same PSO) and packed too when they would cross into a wider tile (Qwen3-0.6B 1024+1024 = 2048 is `bn16`: Q+K+V at 4096 shares q's tile) -- the tile rule is MLX's own `bn` steps at N >= 512 / 2048 (matmul.cpp 1099-1106); `kv` always leaves the wide member alone; `all` packs every eligible member (Q+K+V; k/v move to the wider PSO, expected identical, contract reports the bits). The pack's layout follows the weights' storage per group ([K, N] weights pack row-major [K, n_total] = `gemv_t`; [N, K] weights, `x @ W.T`, pack [n_total, K] = the non-T `gemv`; either way the members' own kernel). Decode-only (M == 1); members over 64 MiB, donated weights, weights the program UPDATES (a result of the weight's type derived from it, a training step), batching dims and middle-contracted (batched-arm) weights decline with the reason narrated under METALJAX_DEBUG=1 (`proj:` lines). A pack is keyed by its weights' identity: fresh weights repack twice (`kMaxProjRepacks`), then the executable re-lowers WITHOUT the projection packs and keeps every other recognizer (narrated `re-lowering without them`) |
| `METALJAX_PROJ_PACK_MB` | `2048` | user knob | plugin-native only: cap on the device memory one executable's projection packs may hold in total (row 7 K+V: 142 MB; Q+K+V: 708 MB); packs are built smallest first and one past the cap declines (its dots run as written), 0 disables the form |
| `METALJAX_PROJ_PACK_LAYOUT` | `` | debug-bisect | plugin-native only: unset = each group packs in its weights' own storage order (bit-identical to the members' dots); `=nk` / `=kn` force every pack to the row-major [n_total, K] (contracted on dim 1, MLX reads it transposed: the non-T `gemv`, 22 serial K steps instead of 90 on a [2880 -> 1024] bf16 pack, no in-loop barrier) / [K, n_total] (`gemv_t`) layout regardless of storage -- the A/B arm for the OTHER kernel, NOT bit-identical to the literal dots on the groups it flips (a different reduction order) |
| `METALJAX_ROPE_VIEW` | `1` | debug-bisect | plugin-native only: =0 turns off the rotate-half rope rewrite (`metal_rope.cc`: `x*cos + concat(-x2, x1)*sin` lowered as one fused kernel over a stride -1 view of x, the sign folded onto the table -- bit-exact, three dispatches fewer per apply); the literal tape is the A/B arm |
| `METALJAX_RECOGNIZE` | `1` | debug-bisect | plugin-native only: =0 turns off ALL THREE recognizer emits, i.e. the second (fused) lowering is never built -- the control for what an emit is worth |
| `METALJAX_SDPA` | `1` | debug-bisect | =0 disables the fused-attention rewrite (Stage 1 and plugin-native) |
| `METALJAX_SYNC` | `0` | debug-bisect | (see src/metaljax/engine.py) |
| `METALJAX_TRACE_BUDGET` | `20000` | user knob | max ops in one mx.compile trace (default 20000) |
| `METALJAX_WHILE_PIPELINE` | `1` | debug-bisect | =0 restores the serial (two-round-trip) dynamic while on the native engine. A dynamic while pipelines when its body COMPILES (one graph call to speculate) whatever its tape size; an EAGER body pipelines only up to 256 entries, and a value >1 replaces that eager threshold |
| `METALJAX_WHILE_SUBMIT_AHEAD` | `1` | debug-bisect | plugin-native: =0 makes a pipelined dynamic while READ iteration t's condition before it submits t+1 (the 0.11.6 shape). Default: from the third iteration on, t+1 is submitted first and dropped if the loop stops; the loop holds its carry across the submission, so a speculative step copies the KV cache root once (`slice_update_copied` +1/step) instead of writing in place, and the governor is asked (without stalling) whether one more in-flight step fits. Narrated as `while(pipelined+ahead): ... ahead_steps= ahead_declined= ahead_failures= ahead_copy_mb= ahead_inflight_mb=` |
| `METALJAX_WHILE_AHEAD_COPY_MB` | `128` | user knob | plugin-native: the most a submit-ahead step may have to copy per iteration (the bytes of the carries the body does not pass through unchanged, which bounds the ones it updates in place). Above the cap the loop reads first: past ~128 MB the copy costs more device time than the bubble it hides |
| `MJDBG_NODSIZE` | `None` | debug-bisect | narrow vector-mode dot cap for bisection |
| `MJDBG_NOHOIST` | `None` | debug-bisect | (see src/metaljax/msl_scan.py) |
| `MJDBG_NONESTED` | `None` | debug-bisect | (see src/metaljax/msl_scan.py) |
| `MJDBG_NOREDREG` | `None` | debug-bisect | disable in-lane register reduces for bisection |
| `METALJAX_VERIFY_COMPILE` | `None` | debug-bisect | plugin-native only: =1 runs every executable a second time op by op and reports outputs that differ from the compiled path; =dump also prints the arguments and both answers |
| `MJDBG_VERIFY_MSL` | `None` | debug-bisect | verify every msl plan against the raw loop; dumps mismatches |

metaljax also sets `MLX_METAL_GPU_ARCH` (accurate-matmul arch pin,
see METALJAX_MATMUL_PRECISION) and `MLX_MAX_OPS_PER_BUFFER` before
importing mlx.
