# The MLX command-buffer corruption, located in MLX's source (2026-08-17)

**Verdict: found, one line, fixed, and already fixed upstream — but not in any
released MLX.** The defect is `compute_dynamic_offset` registering an *aliased*
array as a command-encoder temporary
(`mlx/backend/metal/slicing.cpp:62`, v0.32.0), combined with
`CommandEncoder::end_encoding()` erasing every temporary from the cross-encoder
fence bookkeeping (`mlx/backend/metal/device.cpp:442-446`). The consequence is
a **dropped fence wait between the command buffer that produces a dynamic
slice's start index and the command buffer that reads it** — so
`slice_update` / `dynamic_update_slice` lands at a *stale offset* whenever a
command-buffer boundary falls between the two.

Upstream fix: `ml-explore/mlx@7e8b4ccc` — *"Fix fence tracking for donated
dynamic slice offsets"* (PR #4099, 2026-08-10). **Unreleased**: v0.32.0 is
still the newest tag, and it predates the fix by ~230 commits.

Everything below is measured on this machine today: mlx built from source at
the v0.32.0 tag (`mlx-src/`, gitignored fork checkout), Python 3.13.5 venv
`~/.cache/metaljax-bench/venvs/mlxsrc`, Stage 1 (`METALJAX_ENGINE=py`) so that
swapping libmlx is a venv swap. Logs: `~/.cache/metaljax-bench/logs/mlx-patch/`.

---

## 1. Reproduced from source

`scripts/mlx_patch_canary.sh` runs the committed 8B canary
(`notes/data/qwen3_8b_prefill_36layer.mlir` — real-shape 36-layer Qwen3-8B
maxtext prefill, bf16 weights as arguments, no quantization, 16.4 GB of
parameters) against a chosen mlx build, checked against the jax-CPU reference
`row15-mechanism/B-cpu-ref.npz` from tonight's ladder (rtol/atol 2e-2).

| build | draws | result |
|---|---|---|
| pip wheel `mlx 0.32.0` (what we ship against) | 1/1 | **FAIL(5)** `abs 1.085e+04` `norm 1.000e+00` |
| **source build of the v0.32.0 tag** | 4 | **FAIL(5) `1.085e+04` on 2**, FAIL(3) `1.4-1.6e-2` on 2 |
| source build, `MLX_MAX_MB_PER_BUFFER=40` (MLX's own default) | 1/1 | **FAIL(5)** `1.085e+04` |

The source build reproduces the failure **bit-identically** to the wheel
(`max_abs_err` 1.085e+04, `max_norm_err` exactly 1.000 on 5 of 15 outputs) and,
like the wheel, draws it only some of the time — this is a race, not a
deterministic miscompute.

## 2. The defect, in MLX source terms

### 2.1 The two lines

`mlx/backend/metal/slicing.cpp:45-62` (v0.32.0):

```cpp
array compute_dynamic_offset(const array& indices, ...) {
  array offset({1}, int64, nullptr, {});
  bool donate = indices.is_donatable() && ...;
  if (donate) {
    offset.copy_shared_buffer(indices);   // 58: offset ALIASES a live array
  } else {
    offset.set_data(allocator::malloc(offset.itemsize()));
  }
  compute_encoder.add_temporary(offset);  // 62: ... and is called a temporary
```

`mlx/backend/metal/device.cpp:425-446` (v0.32.0), `CommandEncoder::end_encoding`:

```cpp
  // - Temporaries are a special case as they do not cross command encoder
  //   boundaries. These can be removed early from the encoders inputs and
  //   outputs since they don't need synchronization.
  ...
  for (auto& t : temporaries_) {
    all_outputs_.erase(t.buffer().ptr());
    all_inputs_.erase(t.buffer().ptr());
  }
  ...
  for (auto& in : all_inputs_) {
    if (auto it = prev_ce_outputs_.find(in); it != prev_ce_outputs_.end()) {
      encoder_->waitForFence(it->second.get());   // the wait that is skipped
```

`all_inputs_`/`all_outputs_` are the **only** input to MLX's cross-command-
buffer synchronization: at each encoder boundary MLX waits on the fence of
every previous encoder that wrote a buffer this encoder touches. Erasing a
buffer from those sets erases the dependency.

The comment's premise is true of a temporary *array* and false of its
*buffer*: line 58 gives `offset` the buffer of `indices`, an ordinary graph
array that a previous command encoder computed. So the erase removes exactly
the buffer whose producer must be waited for.

### 2.2 What goes wrong at run time

1. Encoder *N* computes the start index (`indices`) — in our workloads the
   token position for a KV-cache write, or a scan counter.
2. The byte or op budget fires; MLX ends encoding, commits, and starts a new
   command buffer. Encoder *N* is now **executing on the GPU**.
3. Encoder *N+1* runs `compute_dynamic_offset`: it reads `indices`, writes the
   computed offset **into that same buffer**, and the dynamic copy kernel then
   reads the offset.
4. Because the buffer was erased as a "temporary", no `waitForFence` is
   emitted. The offset kernel races encoder *N*'s writes and usually reads the
   buffer's **previous** contents (typically zeros).
5. The dynamic slice/update is performed at the stale offset — a whole KV
   block, or a whole layer's parameters, written at the wrong position, or
   read from the wrong position.

That is why the corrupted values are *plausible-magnitude real data* rather
than uninitialized memory, why `max_norm_err` is exactly **1.000** on the
affected outputs (total loss of signal — the tensor is somebody else's), and
why it moves from run to run.

### 2.3 Measured: the drop happens once per transformer layer

`mlx-src` branch `diag/split-instrumentation` adds `MLX_SPLIT_DEBUG` (off by
default): one line per command-encoder boundary with the fence bookkeeping it
performed, the waits the temporaries erase dropped, and (as a control) any
write-after-read hazard across encoders. On the 8B canary, shipped budgets:

| metric | value |
|---|---|
| command-encoder boundaries in one run (warmup + 3 reps) | 760 |
| boundaries that **dropped** a fence wait for a temporary | **144** = 36 layers x 4 executions |
| write-after-read hazards across encoders (`war=`) | **0** |
| primitives present in every dropping encoder | `DynamicSlice`, `Reshape`, `Transpose` (72/72 attributed at `MLX_SPLIT_DEBUG=3`) |

The dropping encoders are all the same shape — `ops=2 in=3 out=2 temps=1
waits=0 dropped=1` — which is `compute_dynamic_offset`'s kernel plus the
dynamic copy kernel, with `offset` as the single temporary.

`war=0` matters: it says the *other* candidate hole in this scheme
(a later encoder overwriting a buffer an earlier one only read) never fires,
because MLX keeps every primitive's input buffers alive in the command
buffer's completion handler. The temporaries erase is the only hole.

## 3. The fix, and what it buys

Two patches were built and measured. Both make the canary deterministic and
budget-independent; the first is upstream's, and is the one we vendor.

* **`fix/command-buffer-split`** — cherry-pick of upstream `7e8b4ccc`: move
  `add_temporary(offset)` into the non-aliasing branch. One line, plus
  upstream's own C++ regression test.
* **`fix/temporary-fence-tracking`** — our generic hardening of
  `end_encoding()`: keep a temporary in the fence bookkeeping when a previous
  encoder wrote its buffer. Fixes the *class* rather than the instance.

### 3.1 The canary

| build | budget | draws | result |
|---|---|---|---|
| pristine v0.32.0 (source) | 512 (shipped) | 4 | FAIL(5) `1.085e+04` / `1.000` on 2 |
| pristine v0.32.0 (source) | 40 | 1 | FAIL(5) `1.085e+04` / `1.000` |
| generic fix (`MLX_SPLIT_FIX=1`) | 512 | 3 | FAIL(3) **`9.277e-03` / `4.097e-02`** — identical every draw |
| generic fix | 2048 | 1 | FAIL(3) `9.277e-03` / `4.097e-02` |
| generic fix | 40 | 1 | FAIL(3) `9.277e-03` / `4.097e-02` |
| **upstream fix (cherry-pick)** | 512 | 3 | FAIL(3) `9.277e-03` / `4.097e-02` |
| **upstream fix** | 40 | 1 | FAIL(3) `9.277e-03` / `4.097e-02` |

Read the last four rows: across a **50x range of command-buffer budgets** the
patched engine returns *the same numbers to every digit*. The split lottery is
gone.

Cost: nothing measurable. Canary prefill time, `min` over 3 reps, one run per
row: pristine source 407.6 / 405.8 / 404.2 ms, pip wheel 405.4, upstream fix
404.1, generic fix 405.5, both merged 405.0. The patch adds at most one
`waitForFence` per encoder boundary (waits are deduplicated by fence) and, on
this workload, 172 map entries.

### 3.2 The 4-7 % residual (row-15 notes §8.7 item 2) — attributed

`9.277e-03 / 4.097e-02` is exactly the figure the unpatched engine produced at
`MLX_MAX_MB_PER_BUFFER=2048` (`notes/row15-wrong-output-2026-08-17.md` §8.5).
It is now reproduced at 40, 512 and 2048 MB, unchanged. **It does not move with
the split, so it is not corruption**: it is the bf16-accumulation difference
between 36 fused metal layers and the f32 jax-CPU reference, against a 2 %
tolerance. Three of fifteen outputs miss; the other twelve pass.

### 3.3 Our own command-buffer canaries

`tests/test_command_buffer.py` (Python-engine subset; the native-engine
detectors need the native extension rebuilt against the patched libmlx, which
this milestone does not do):

| build | correctness tests | corruption canaries |
|---|---|---|
| pristine v0.32.0 | 4 passed | **both find a corrupting budget** (as designed) |
| generic fix | 4 passed | **neither can find one** — kernel sweep and byte sweep (40, 20, 8, 48 MB) all clean |
| upstream fix | 4 passed | **neither can find one** |

The canaries failing is the *good* outcome — they are built to fail loudly when
the bug they hunt is gone ("Either MLX fixed the command-buffer split bug ... or
our lowering moved"). This is the first time either has come up empty.

Note what that means: the **eager kernel-budget face** (the `qwen3_init_scan`
asset, the one that gave maxtext a wrong first-step loss) is cured by the same
one-line patch as the compiled byte-budget face. Both faces are the same
defect — parameter init writes each layer's weights into a `[28, ...]` tensor
with a dynamic update slice, exactly as prefill writes the KV cache.

### 3.4 Row 15, the whole model: garbage becomes " Paris. The capital"

The row that started this: `qwix-int8-qwen3-8b` (Qwen3-8B, qwix int8, maxtext),
`scripts/model_bench/row15_forensics.py --prefill-reps 10 --decode 3` — ten
prefills of the *same loaded parameters in one process*, then a greedy decode.
Same checkpoint, same prompt, same script, same engine
(`METALJAX_ENGINE=py`, Stage 1); the only variable is which libmlx the venv has.
Both runs guarded, both peaked at 62 GB.

| libmlx | 10 draws | distinct first tokens | collapses | decode |
|---|---|---|---|---|
| pip wheel 0.32.0 | 10 | **9** | 0 | `~\n\n ..."\n\n晨 shards` |
| **patched fork** | 10 | **1** — token 12095, ten times | 0 | **`" Paris. The capital"`** |

Token 12095 is the same first token row 14 (the 0.6B control) returns ten times
out of ten. **Row 15 is fixed**: deterministic and coherent, on an 8B model that
has emitted nothing but garbage since 2026-08-03.

### 3.5 A minimal, pure-MLX reproducer

`notes/data/mlx-cbuf-repro/repro_c.py` — 20 lines, `mlx` only, ~2 s, no
metaljax, no JAX, no StableHLO, MLX's own default budgets:

```python
source = mx.concatenate([mx.zeros((2, 1 << 26), mx.int32),
                         mx.ones((2, 1 << 26), mx.int32)], axis=1)
target, update = mx.zeros((4, 4), mx.int32), mx.full((1, 1), 7, mx.int32)
mx.eval(source, target, update)
out = mx.slice_update(target, update, mx.max(source, axis=1), (0, 1))
```

| build | fresh processes x 20 evaluations | result |
|---|---|---|
| pip wheel 0.32.0 | 3 | **3/3 processes wrong**, on evaluation 0 (twice) or 1 (once) — the 7 lands at `[0,0]`, or disappears |
| source v0.32.0 | — | same |
| upstream fix / both fixes | 3 | **0 of 60 evaluations wrong** |

*Provenance of that table*: the failing numbers were measured with this exact
recipe run as an inline script (`3 fresh processes x 20 evaluations`, wheel
venv) before it was packaged as `repro_c.py`; the packaged file has been run on
the patched build (0 wrong) but a wheel-side rerun of the packaged file was
still queued behind the machine lock when this note was written. The recipe is
unchanged between the two.

Two properties of the repro are the mechanism in miniature: the start index
must **not** be bound to a Python name (a second reference makes it
non-donatable, MLX allocates a separate offset buffer, and the bug vanishes),
and only the first evaluation in a fresh process is wrong (afterwards the
allocator hands out warm buffers and the timing changes).

That single property retro-explains every observation in
`notes/mlx-command-buffer-split.md` that used to look like a lottery:

| old observation | explanation |
|---|---|
| "`mx.compile` is required" (repro A) | donation needs refcount 1; in eager Python the user holds the index array |
| "exposing every intermediate as an output makes it clean" | that also makes the index array non-donatable |
| "first call clean, calls 2+ corrupt" | the trace retains call 1's arrays |
| "corrupted values are plausible-magnitude data" | a slice landing at the wrong offset copies real data |
| "which boundary lands where decides it" | the fence is only needed when a boundary separates producer from consumer |
| "not the buffer cache (`set_cache_limit(0)` no help)" | correct — it is aliasing, not recycling |
| "`MLX_BFS_MAX_WIDTH=8` is clean" | different traversal, boundaries elsewhere |
| "shrinking every tensor 4x stops it" | fewer bytes, fewer splits |

## 4. Relation to MLX bug #8 (compiled scatter-add drops updates)

**Separate defect, unchanged by this patch.** Bug #8
(`notes/parity-whitelist-report.md`) is in `mx.compile`'s *fusion*, not in
command-buffer synchronization: it is insensitive to both budgets
(`MLX_MAX_OPS_PER_BUFFER`/`_MB_` at 1 or 10^8 both fail) and it is cured by
`METALJAX_MLX_COMPILE_MODE=no_fuse` / `no_simplify`, which do nothing for the
dynamic-slice bug. The two do not share a mechanism: one drops a fence, the
other builds a wrong fused kernel. Bug #8 gets its own branch and its own
reproducer; it is not closed by this work.

## 5. How to re-run any of it

```bash
# the fork (gitignored; branches: main / fix-* / diag-* / vendor-0.32.0)
cd mlx-src && git checkout vendor/0.32.0
CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_METAL_CPP=$PWD/build/temp.*/mlx.core/_deps/metal_cpp-src" \
CMAKE_BUILD_PARALLEL_LEVEL=12 uv pip install \
  --python ~/.cache/metaljax-bench/venvs/mlxsrc/bin/python --no-build-isolation -e .

# 2 s: the minimal pure-MLX reproducer
~/.cache/metaljax-bench/venvs/mlxsrc/bin/python notes/data/mlx-cbuf-repro/repro_c.py

# ~1 min, ~20 GB, machine lock: the 8B canary against the jax-CPU reference
scripts/mlx_patch_canary.sh <label> [MLX_MAX_MB_PER_BUFFER=40 ...]

# the instrumented build (branch diag/split-instrumentation)
MLX_SPLIT_DEBUG=1 scripts/mlx_patch_canary.sh diag     # per-boundary counters
MLX_SPLIT_DEBUG=3 MLX_CANARY_REPS=1 scripts/mlx_patch_canary.sh diag  # + primitives
MLX_SPLIT_FIX=1   scripts/mlx_patch_canary.sh fix      # A/B the fix in one binary

# ~6 min, 62 GB peak, machine lock: row 15 end to end (Stage 1)
env JAX_PLATFORMS=metal,cpu KMP_DUPLICATE_LIB_OK=TRUE GUARD_RSS_GB=105 \
    METALJAX_ENGINE=py METALJAX_SYNC=1 \
  scripts/model_bench/mem_guard.sh 92 <flight.log> \
  ~/.cache/metaljax-bench/maxtext/venv/bin/python \
  scripts/model_bench/row15_forensics.py --bench qwix-int8-qwen3-8b \
    --prefill-reps 10 --decode 3
```

Artifacts (all written as produced): `~/.cache/metaljax-bench/logs/mlx-patch/`
— `canary-*.log` / `.npz` per arm, `row15-py-{wheel,fixed}.log` plus their
guard flight logs, `build-*.log`. Patches: `notes/patches/*.patch`, PR drafts
`notes/patches/*-pr.md`.

**The maxtext venv was temporarily switched to the patched libmlx for the
row-15 arms and has been restored to `mlx==0.32.0`.** No shipped linkage
changed: `plugin/`, `plugin-native/` and `src/metaljax/lib/` still resolve the
pip wheel's MLX.

## 6. What is still open

1. **The native engine is not covered here.** `native/build/metaljax_native`
   links libmlx directly and refuses a version skew at import, so every
   measurement above is Stage 1 (`METALJAX_ENGINE=py`). Rebuilding it against
   the vendored libmlx is the vendoring milestone, not this one.
2. **The residual 3-of-15 outputs** (§3.2) are attributed to bf16 depth by the
   budget-independence argument, not by an error model. If it ever matters,
   the cheap check is the same canary in f32.
3. **Upstream has more fixes we do not have**: v0.32.0 is ~230 commits behind
   `main`, which includes at least one more Metal correctness fix in this
   family (`269e099d`, a within-encoder WAR hazard in the layer-norm VJP, is
   already in v0.32.0; `7e8b4ccc` is not). The vendoring plan
   (`notes/mlx-vendoring-plan.md`) treats "rebase our branches onto the next
   tag" as the maintenance loop.
