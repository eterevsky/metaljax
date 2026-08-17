# Draft: MLX upstream issue (ml-explore/mlx)

> **SUPERSEDED, 2026-08-17 — DO NOT FILE AS WRITTEN.** This draft describes the
> bug as an unexplained sensitivity to `MLX_MAX_*_PER_BUFFER`. The root cause
> has since been found in MLX's source and fixed:
> `compute_dynamic_offset` (`mlx/backend/metal/slicing.cpp:62`, v0.32.0)
> registers an array that ALIASES the caller's index buffer as a command-encoder
> temporary, and `CommandEncoder::end_encoding()` erases temporaries from the
> cross-encoder fence bookkeeping — so the wait on the index's producer is
> dropped and a dynamic slice can run at a stale offset. Upstream already fixed
> it in `7e8b4ccc` (PR #4099, 2026-08-10), which is **not in any release**.
>
> What to send upstream instead: `notes/patches/fix-command-buffer-split-pr.md`
> (a comment on #4099 asking for a release, with the 20-line pure-MLX
> reproducer `notes/data/mlx-cbuf-repro/repro_c.py`) and, if wanted, the
> hardening PR `notes/patches/fix-temporary-fence-tracking-pr.md`. Evidence and
> mechanism: `notes/mlx-patch-diagnosis.md`.
>
> Everything below is kept as the record of what was known before the source
> build; its *observations* are all still correct, and §"Observations that may
> help localize it" is explained item by item in the diagnosis note.

*Draft only — for Oleg to review and file. Everything below the rule is the
issue body as it should be pasted; this preamble is not part of it.*

**Attachments to upload with the issue** (regenerate with the recipe in the
issue's last appendix). The exact copies used for every measurement below are
in this session's scratchpad — **ephemeral, copy them somewhere durable before
filing**:

```
/private/tmp/claude-501/-Users-oleg-metaljax/43351818-f4b5-4809-87c8-2909e6e0e70e/scratchpad/
    qwen3_prefill.mlxfn      272 KB   (repro A)
    qwen3_init_scan.mlxfn    340 KB   (repro B)
    repro_a.py, repro_b.py            (extracted verbatim from this draft and re-run)
    export_graph.py, export_scan.py   (the generators)
```

* `qwen3_prefill.mlxfn` — exported MLX graph, repro A
* `qwen3_init_scan.mlxfn` — exported MLX graph, repro B
* `repro_a.py`, `repro_b.py` — the drivers, inlined in the body below

Both attachments are plain MLX graph exports: reproducing needs **only**
`mlx` and `numpy`. Nothing from our project, no JAX, no StableHLO.

Internal cross-references (do not paste): `notes/mlx-command-buffer-split.md`,
`tests/test_command_buffer.py`, `src/metaljax/__init__.py`, STATUS footnotes
7/8/9/11.

**Where today's measurements disagree with `notes/mlx-command-buffer-split.md`**
(all re-measured 2026-08-03 on this tree, mlx 0.32.0 unchanged; the issue body
uses the new numbers and cites the old ones as historical):

1. *The pinned thresholds moved.* `tests/test_command_buffer.py` passes at
   `MLX_MAX_MB_PER_BUFFER=80` (note says 80 corrupts, 160 is the first clean
   value) and **passes at `MLX_MAX_OPS_PER_BUFFER=400`** (note and
   `__init__.py` say 400 is the corrupting alignment). The eager-scan test now
   fails at **200** instead — swept 50/100/200/300/350/400/450/500/600/1000/
   2000, only 200 fails, 3/3. Most likely cause: `fdc7cde`'s static-splat
   shift peephole changed the threefry lowering (that test dropped from ~3 s
   to 1.3 s). **Consequence for us: the shipped `ops=800` is no longer pinned
   by anything — the test would pass at 400 too.** Worth re-pinning, or at
   least noting in the test, but I changed no files.
2. *"The first call of a compiled function is clean; calls 2+ are corrupt"*
   does not hold for the exported graph: at 40 MB the first compiled call
   already differs from the uncompiled evaluation, with or without other work
   before it. Left out of the issue.
3. *"It could not be reduced to a synthetic MLX program"* is still true for
   hand-written graphs (I re-tried a 16-layer GQA+RoPE+KV transformer in plain
   `mlx.core` at ops=1/2/4 and mb=1/4/40 — all clean), but the graph **export**
   route sidesteps it entirely: `mx.export_function` on our built graph gives a
   pure-MLX repro with no metaljax in the loop.

---

**Title:** `mx.compile`d graphs return wrong and nondeterministic results when
one `eval` is split across several Metal command buffers (0.32.0, M5 Max)

## Summary

On this machine a compiled MLX graph returns different results depending only
on how much work MLX packs into one Metal command buffer. With MLX's stock
budgets, an exported transformer-prefill graph returns all-NaN logits and
KV-cache tensors that differ from call to call; with `MLX_MAX_MB_PER_BUFFER`
raised, the same graph, the same input arrays and the same process return the
correct values. Evaluating the identical graph without `mx.compile` is correct
at every budget.

The same mechanism shows up a second way on the uncompiled path: iterating one
graph 10 times and calling `mx.eval` every iteration versus every 5 iterations
produces different results at some values of `MLX_MAX_OPS_PER_BUFFER`. Same
ops, same order, only different evaluation points.

Two self-contained repros are attached (repro A and repro B below). They are
MLX graph exports (`mx.export_function` / `mx.import_function`), so
reproducing needs only `mlx` and `numpy`.

Failures are silent: no error, no warning, and the corrupted values are of
plausible magnitude (`0.18` where `0.23` belongs) except when a NaN appears.

## Environment

* mlx **0.32.0** (pip wheel, no local build)
* macOS **26.5**, Xcode 26.6
* Apple **M5 Max**, 128 GB unified memory (`hw.model = Mac17,6`, GPU family
  `applegpu_g17s`)
* Python 3.14.4, numpy 2.x
* Reproduced with `MLX_METAL_GPU_ARCH` **unset** (native `applegpu_g17s`) and
  with `MLX_METAL_GPU_ARCH=applegpu_g16g` — the corruption is independent of
  the kernel-arch selection. Both repros as printed below run with it unset;
  repro A was additionally confirmed with the `g16g` pin, and repro B's
  failure was first seen through our backend, which pins `g16g`.

The graphs come from real workloads (Qwen3 prefill and Qwen3 parameter init,
lowered from JAX by a Metal PJRT backend we maintain on top of MLX), but
nothing about that backend is needed to run them — they are exported MLX
graphs.

## Repro A — a compiled graph, corrupted by the byte budget

`qwen3_prefill.mlxfn` (272 KB) is one prefill step of a small Qwen3 (8 layers,
d=1024, MLP 6144, vocab 2048, bf16 weights as graph inputs). A few hundred
GPU kernels; the script below takes ~2 s and a few GB of memory.

```python
# repro_a.py -- only mlx and numpy
import os, sys
import mlx.core as mx
import numpy as np

MODE = os.environ.get("MODE", "compiled")     # "compiled" | "eager"
SHAPES = [((16,), "i"), ((1024,), "b"), ((1024, 8, 6144), "b"),
          ((1024, 8, 6144), "b"), ((6144, 8, 1024), "b"), ((1024, 8), "b"),
          ((1024, 8), "b"), ((1024, 8, 8, 128), "b"), ((128, 8), "b"),
          ((16, 8, 128, 1024), "b"), ((1024, 8, 16, 128), "b"),
          ((128, 8), "b"), ((1024, 8, 8, 128), "b"), ((2048, 1024), "b"),
          ((), "i")]

rng = np.random.default_rng(0)
args = []
for shape, kind in SHAPES:
    if kind == "i":
        args.append(mx.array(rng.integers(0, 4, size=shape).astype(np.int32)))
    else:
        x = (rng.standard_normal(size=shape) * 0.1).astype(np.float32)
        args.append(mx.array(x).astype(mx.bfloat16))
mx.eval(*args)

fn = mx.import_function("qwen3_prefill.mlxfn")
if MODE == "compiled":
    fn = mx.compile(fn)

def run():
    outs = fn(*args)
    mx.eval(*outs)
    return [np.array(o.astype(mx.float32)) for o in outs]

runs = [run() for _ in range(3)]
for i in (1, 2):
    for j, (a, b) in enumerate(zip(runs[0], runs[i])):
        if not np.array_equal(a, b):
            print(f"call 1 vs call {i+1}: output {j} {a.shape} differs, "
                  f"max |diff| {np.nanmax(np.abs(a - b)):.4g}")
```

**Expected:** three calls of one compiled function on unchanged input arrays
return identical values (this graph contains no order-nondeterministic
reduction — no scatter-add, no atomics), and those values agree with the same
graph evaluated without `mx.compile`.

**Actual**, with no environment variables set at all (MLX's own defaults):

```
$ python repro_a.py
call 1 vs call 2: output 9 (1, 1, 2048) differs, max |diff| nan
call 1 vs call 2: output 12 (1, 1) differs, max |diff| 880
call 1 vs call 3: output 6 (8, 16, 8, 1, 128) differs, max |diff| 0.9141
call 1 vs call 3: output 7 (8, 16, 8, 1, 128) differs, max |diff| 1.375
```

(which outputs are affected, and by how much, varies from run to run; the
script has never been silent at this budget in any run we made)

Compared against the same graph run without `mx.compile` (`MODE=eager`, which
is correct at every budget we tried), the compiled run at 40 MB is wrong on
outputs 6, 7, 9, 12, 13 — every compiled call, including the first:

| output | uncompiled | compiled |
|---|---|---|
| 9 — logits `(1,1,2048)` | `0.5156, -0.6172, 0.0938, …` | `nan` (all 2048 entries) |
| 12 — argmax token `(1,1)` | `524` | `0` |
| 6, 7 — KV cache `(8,16,8,1,128)` | — | differs by up to `~0.95`, **differently on each call** |

**Raising the byte budget fixes it, changing nothing else:**

```
$ MLX_MAX_MB_PER_BUFFER=512 python repro_a.py     # no output: all three calls identical, and equal to uncompiled
```

Sweep (3 replays per run; "wrong" = replays disagree with each other and/or
with the uncompiled evaluation):

| `MLX_MAX_OPS_PER_BUFFER` | `MLX_MAX_MB_PER_BUFFER` | result |
|---|---|---|
| unset (default) | unset (default) | **wrong** (2/2 runs) |
| 800 | 40 | **wrong** (4/4 runs) |
| 800 | 80 | ok |
| 800 | 160 | ok |
| 800 | 512 | ok (3/3 runs) |
| 1 | 1000000 | **wrong** |
| 2 | 1000000 | **wrong** (one output all-NaN, another off by 1.8e37) |
| 4 | 1000000 | **wrong** |
| 8 | 1000000 | ok |
| 800 | 1000000 | ok |

So corruption appears once command buffers are cut every ~4 kernels or less,
and disappears by ~8. In practice the byte budget is what fires: an LLM
layer's tensors are tens of megabytes, so the 40 MB default commits every few
kernels.

`MODE=eager` (the imported function called directly, so the graph is rebuilt
per call and every intermediate is live for one big `eval`) is correct at
every budget tested, including the defaults.

## Repro B — an uncompiled loop, corrupted by the kernel-count budget

`qwen3_init_scan.mlxfn` (340 KB) is one iteration of a 28-layer parameter-init
body (Threefry key splits + normal draws into 28-slot parameter tensors,
f32, real shapes; ~1.6 GB of carries). The driver iterates it 10 times twice —
once calling `mx.eval` after every iteration, once after every 5 — and
compares the carries.

```python
# repro_b.py -- only mlx and numpy
import os, sys
import mlx.core as mx
import numpy as np

NCARRY = 27
SHAPES = [((28,), "u"), ((28, 2), "k"), ((28,), "u"), ((28, 2), "k"),
          ((28,), "u"), ((28, 2), "k"), ((), "i"),
          ((28, 1024, 3072), "f"), ((28, 1024, 3072), "f"),
          ((28, 3072, 1024), "f"), ((28, 1024), "f"), ((28, 1024), "f"),
          ((28, 1024, 8, 128), "f"), ((28, 128), "f"),
          ((28, 16, 128, 1024), "f"), ((28, 1024, 16, 128), "f"),
          ((28, 128), "f"), ((28, 1024, 8, 128), "f"),
          ((28,), "u"), ((28, 2), "u"), ((28,), "u"), ((28, 2), "u"),
          ((28,), "u"), ((28, 2), "u"), ((28,), "u"), ((28,), "u"),
          ((28,), "u"), ((), "one")]

rng = np.random.default_rng(0)
args = []
for shape, kind in SHAPES:
    if kind == "k":                      # PRNG keys: the only nonzero uint32 inputs
        args.append(mx.array(rng.integers(0, 2**32, size=shape, dtype=np.uint32)))
    elif kind == "u":
        args.append(mx.zeros(shape, dtype=mx.uint32))
    elif kind == "i":
        args.append(mx.array(np.int32(0)))
    elif kind == "one":
        args.append(mx.array(np.int32(1)))
    else:
        args.append(mx.zeros(shape, dtype=mx.float32))
mx.eval(*args)

body = mx.import_function("qwen3_init_scan.mlxfn")
captures = args[NCARRY:]

def run(flush_every):
    vals = list(args[:NCARRY])
    for i in range(1, 11):
        vals = list(body(*vals, *captures))
        if i % flush_every == 0:
            mx.eval(*vals)
    mx.eval(*vals)
    return vals

for j, (a, b) in enumerate(zip(run(1), run(5))):
    if not bool(mx.all(a == b).item()):
        d = mx.sum(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
        print(f"carry {j} {tuple(a.shape)} {a.dtype} differs: total |diff| {d}")
```

**Expected:** identical carries — the two runs issue the same ops in the same
order and differ only in when `eval` is called.

**Actual:**

```
$ MLX_MAX_OPS_PER_BUFFER=100 MLX_MAX_MB_PER_BUFFER=2048 python repro_b.py
carry 19 (28, 2) mlx.core.uint32 differs: total |diff| 5088055808.0
```

Reproduces 3/3 runs at `MLX_MAX_OPS_PER_BUFFER=100`; identical at 50, 200,
300, 400 and 800. The corrupted carry is a PRNG key, so downstream every
weight drawn from that stream is different — in the original workload this
made a training run start from different weights (first-step loss 208.78
instead of 247.81, versus 247.78 on CPU).

Two properties of this face worth noting:

* It is not "splits closer than N are unsafe": at a given budget only certain
  eval cadences are wrong. At 100 kernels/buffer, evaluating every 3 or every
  5 iterations is wrong while every 2, 4 or 6 is clean; and evaluating every 5
  is clean at 50, 200, 300, 400 and 800 kernels/buffer. In the original
  (pre-export) form of this workload the fatal pair was cadence 5 at 400
  kernels/buffer, with cadences 1, 2, 3, 4, 6, 8 clean there and cadence 5
  clean at 200, 800, 1600 and 5000. A particular boundary lands between a
  producer and a consumer; the distance between boundaries is not the
  predictor.
* Size matters: in the original workload, shrinking every tensor 4x stopped it
  reproducing at any cadence.

## What we ruled out

Each of these was tested, not assumed:

* **Not MLX's buffer cache** — `mx.set_cache_limit(0)` before the first call
  leaves repro A wrong at 40 MB (same five outputs).
* **Not confined to `mx.compile`** — repro A does need it (the same graph
  evaluated without `mx.compile` is correct at every budget we tried), but
  repro B has no `mx.compile` anywhere.
* **Not graph traversal order** — `MLX_BFS_MAX_WIDTH=8` (different traversal,
  same boundaries) is clean where the default order is wrong.
* **Not the kernel arch** — reproduces with `MLX_METAL_GPU_ARCH` unset and set
  to `applegpu_g16g`.
* **Not dtype-specific** — the original of repro A reproduces identically with
  every bf16 tensor in the program replaced by f32.
* **Not an async/eval race in the caller** — forcing a blocking `mx.eval` on
  every output of every call does not change it; the first executable in a
  chain is already wrong.
* **Not input mutation** — the compiled call does not modify its own inputs
  (checked by snapshotting every argument around the call).

## Observations that may help localize it

* Retaining intermediates hides it: when the same program is rewritten so that
  every intermediate value is also a graph **output**, results are correct.
  Likewise, evaluating the graph in one large uncompiled `eval` (all
  intermediates live) is correct.
* Corrupted values are plausible-magnitude data, not uninitialized garbage,
  which is what buffer reuse across a boundary would look like; and the
  corruption enters at a layer boundary that moves from run to run, then
  propagates through the hidden state.
* The corrupting budget is a property of the *graph*, not a constant. An
  unrelated change on our side to how shifts are lowered (fewer kernels per
  iteration, same values) moved repro B's fatal budget from 400 kernels/buffer
  to 200, and moved repro A's smallest safe byte budget from 160 MB to 80 MB —
  both measured through our interpreter, one day apart, same MLX build, no
  intended change in the computation. Exporting repro B's graph moved its
  fatal value again (200 → 100). Any given "safe" value is a property of the
  workload plus the MLX version, which is why we cannot ship one.

## Scale dependence — why raising the budget is not a general fix

The safe band scales with tensor size rather than absolute bytes. On a
Qwen3-8B-shaped workload (~100 MB weight slices, ~8-10 GB of tensors per
window), `MLX_MAX_MB_PER_BUFFER=512` still cuts every ~2-5 kernels and
produces garbage output — bf16 and int8-quantized alike, with per-layer
KV-cache deltas uncorrelated against a CPU reference — while the same program
is correct with compilation disabled. 2048 MB gives correct output on that
workload.

But the byte budget is bounded from above by memory, because every
intermediate of a command buffer stays allocated until that buffer completes:

* With an effectively unbounded budget, one commit of a diffusion model
  (SD3.5 MMDiT at 1024², unfused attention, ~400 kernels in one buffer)
  accumulated ~90 GB of transient attention logits as wired memory and
  **kernel-panicked the machine** — twice, reproducibly, with a flight
  recorder showing 18 GB → 90+ GB in the final second at memory pressure
  level 1.
* At 2048 MB, loading the 8B model wedged once at a ~120 GB wired footprint
  and, on a later attempt with the same configuration that had just succeeded,
  **panicked the machine** (watchdog timeout). Four machine panics on this
  hardware are attributable to this area.

So for that class of workload there is no value of `MLX_MAX_MB_PER_BUFFER`
that is both correct and stable: below ~2048 MB the results are silently
wrong, at/above it the machine is at risk. We have marked those models
unsupported pending a fix.

## Impact

Any long-running `mx.compile` workload whose per-kernel tensors are large
enough that MLX commits every few kernels — i.e. any LLM-sized or
diffusion-sized graph on this class of hardware — can return silently wrong
results at MLX's default settings, with no error and values that look
plausible. In our case it produced garbage tokens from a correct model, a
black image from a correct diffusion pipeline, and a training run started from
the wrong weights, each of which took days to attribute to the runtime rather
than to the model code.

The per-buffer budget environment variables are the only mitigation we have
found, they must be tuned per workload, the safe value moves when the graph
changes, and for large models the correct range and the memory-safe range do
not overlap.

## Appendix: full env-var matrix

All rows on mlx 0.32.0 / macOS 26.5 / M5 Max, measured 2026-08-03. "wrong" for
repro A = the three compiled replays are not bit-identical and/or disagree
with the uncompiled evaluation of the same graph; "wrong" for repro B = the
two flush cadences produce different carries.

**Repro A** — `qwen3_prefill.mlxfn`, `mx.compile`, 3 replays:

| `MLX_MAX_OPS_PER_BUFFER` | `MLX_MAX_MB_PER_BUFFER` | other | result |
|---|---|---|---|
| unset | unset | — | **wrong** 2/2 |
| 800 | 40 | — | **wrong** 4/4 |
| 800 | 40 | `MLX_METAL_GPU_ARCH=applegpu_g16g` | **wrong** |
| 800 | 40 | `mx.set_cache_limit(0)` | **wrong** |
| 800 | 40 | `MODE=eager` (no `mx.compile`) | ok |
| unset | unset | `MODE=eager` | ok |
| 800 | 80 | — | ok |
| 800 | 160 | — | ok |
| 800 | 512 | — | ok 3/3 |
| 1 / 2 / 4 | 1000000 | — | **wrong** |
| 8 / 800 | 1000000 | — | ok |

**Repro B** — `qwen3_init_scan.mlxfn`, 10 iterations, `eval` every 1 vs every 5
(`MLX_MAX_MB_PER_BUFFER=2048` throughout):

| `MLX_MAX_OPS_PER_BUFFER` | result |
|---|---|
| 50 | ok |
| 100 | **wrong** 3/3 |
| 200 | ok |
| 300 | ok |
| 400 | ok |
| 800 | ok |

For reference, the same computation driven op-by-op by our interpreter instead
of the exported graph is wrong at 200 (3/3) and clean at 50, 100, 300, 350,
400, 450, 500, 600, 1000, 2000 — the same workload, a slightly different op
sequence, a different fatal budget.

Historical values on the same machine and MLX version, before an unrelated
change to our lowering altered the op sequence (kept because they show the
fatal budget moving with the graph): repro A wrong at 40 and 80 MB, clean from
160 MB; repro B wrong at 400 kernels/buffer, clean at 200/800/1600/5000.

## Appendix: how the attached graphs were produced

The `.mlxfn` files are `mx.export_function` exports of the MLX graph our JAX
backend builds for two real programs (a maxtext Qwen3 prefill step and a
maxtext Qwen3 parameter-init scan body). They are attached so that reproducing
needs nothing but MLX. To regenerate them:

```python
# pip install --index-url https://test.pypi.org/simple/ metaljax==0.11.2
import mlx.core as mx, numpy as np
from metaljax.interpreter import Interpreter
from metaljax.ops import control

interp = Interpreter(open("qwen3_prefill_shrunk.mlir").read().encode())
rng = np.random.default_rng(0)
with interp.context:
    avals = interp.in_avals
args = [ ... seeded arrays for each (shape, dtype) in avals ... ]
with interp.context:
    underived = control._underived_outputs(interp._main_block(), [])

def traced(*a):
    interp._in_trace = True
    return control._anchor_outputs(tuple(interp(*a)), a, underived)

mx.export_function("qwen3_prefill.mlxfn", traced, *args)
```

(The `.mlir` inputs are StableHLO dumps of the two programs; we can attach
those and the full export script on request. Repro B's graph is the body
region of the program's `while` loop, exported the same way.)
