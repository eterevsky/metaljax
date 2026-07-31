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
| `METALJAX_F64` | `error` | user knob | f64 policy: unset=strict (compute errors), downcast=f32 emulation |
| `METALJAX_LOOP_CLEAR_COST` | `500000` | user knob | cache-clear cadence in loop op-units (default 500000) |
| `METALJAX_MATMUL_PRECISION` | `highest` | user knob | default=MLX default arch; unset picks the accurate g16g matmul path |
| `METALJAX_MEMDBG` | `` | debug-bisect | =1 logs active/cache memory at loop clears and execute end |
| `METALJAX_MSL` | `1` | debug-bisect | =0 disables msl_scan kernel codegen |
| `METALJAX_MSL_COOP_CAP` | `2200000` | user knob | coop dot work cap in elems/step (default 2.2M) |
| `METALJAX_MSL_COOP_MIN_F` | `8` | user knob | min state width for the coop preference (default 8) |
| `METALJAX_MSL_COOP_PREF` | `1` | user knob | =0 restores the pre-0.4.3 vector-over-coop mode pick |
| `METALJAX_MSL_INLANE` | `1` | user knob | =0 disables the in-lane small-dot rewrite (matrix-state cells) |
| `METALJAX_MSL_PACK_TRIGGER` | `30` | user knob | bindings above which kernel inputs pool per dtype (default 30) |
| `METALJAX_MSL_REG` | `16` | user knob | vector-mode register width cap (default 16) |
| `METALJAX_MSL_VOLATILE` | `t` | workaround-retest | Metal compiler loop miscompile workaround; t=default, tmap/tv/load/0 to retest on OS updates |
| `METALJAX_MSL_WNORM` | `1` | user knob | =0 keeps source weight layouts (skips coalescing materialization) |
| `METALJAX_PLUGIN_PATH` | `None` | user knob | override the plugin dylib location |
| `METALJAX_SYNC` | `0` | debug-bisect | (see src/metaljax/engine.py) |
| `METALJAX_TRACE_BUDGET` | `20000` | user knob | max ops in one mx.compile trace (default 20000) |
| `MJDBG_NODSIZE` | `None` | debug-bisect | narrow vector-mode dot cap for bisection |
| `MJDBG_NOHOIST` | `None` | debug-bisect | (see src/metaljax/msl_scan.py) |
| `MJDBG_NONESTED` | `None` | debug-bisect | (see src/metaljax/msl_scan.py) |
| `MJDBG_NOREDREG` | `None` | debug-bisect | disable in-lane register reduces for bisection |
| `MJDBG_VERIFY_MSL` | `None` | debug-bisect | verify every msl plan against the raw loop; dumps mismatches |

metaljax also sets `MLX_METAL_GPU_ARCH` (accurate-matmul arch pin,
see METALJAX_MATMUL_PRECISION) and `MLX_MAX_OPS_PER_BUFFER` before
importing mlx.
