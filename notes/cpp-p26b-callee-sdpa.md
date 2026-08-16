# P26b: the fused lowering sees into callees (2026-08-16)

*The fix for what P26 diagnosed (`notes/data/p26-gemma-gap-2026-08-16.json`).
Code: `plugin-native/metal/{metal_sdpa,metal_lowering}.cc` + two
`execute_test.py` rows and two contracts. `src/` and `native/` are FROZEN and
untouched — this is a native-lowering change only, and it moves the native
stack TOWARDS Stage 1 rather than away from it.*

## What changed, in one line

`AnalyzeSdpa` walks **every block of every function in the module**
(`src/metaljax/sdpa.py::_all_blocks`) instead of `@main` and its nested
regions.

```c++
// metal_sdpa.cc, AnalyzeSdpa
if (module) {
  for (mlir::func::FuncOp f : module.getOps<mlir::func::FuncOp>())
    for (mlir::Region& r : f->getRegions())
      for (mlir::Block& b : r.getBlocks()) WalkBlocks(b, scan);
} else {
  WalkBlocks(main, scan);   // a detached function: itself and nothing else
}
```

That is the whole of the recognizer change. Nothing in the emission needed
touching: `Lowering::Inline` already splices a callee's block into the caller's
frame op by op (P17), and `LowerOp` consults `plan->sdpa_roots` by
`mlir::Operation*` wherever the op happens to live — so a match rooted inside a
callee fires at the point the callee is inlined, over the slots that call site
bound. The P17 note listed this as "not ported, deliberately"; it is ported
now, and the reason it had to be is below.

## Why that was the whole of P26

The gemma-lib sampler puts its decode step inside a private `func.call`
callee, and so does maxtext — and so does **any** jax loop whose body calls a
named function: a `fori_loop` over a plain `attn(...)` lowers to
`func.call @attn` inside the while body just as a non-inlined `jit` does
(verified on CPU while writing the tests; `while_loop` with a python body is
the one that inlines, which is the asymmetry `moe.py::analyze`'s docstring
already names).

So on row 1 the native stack fused **zero** of the 31B decode body's 60
attentions where Stage 1 fused all 60. That is not primarily a kernel
question — at `Tq=1` the fused attention is worth ~0 (P17's own micro row, and
row 2's emits changed its timing by 0.2 %). It is a **compile-gate** question:

```
by_cost = METALJAX_TRACE_BUDGET / BlockCost(body)      // integer division
```

`BlockCost` charges a fused root 3 units in place of the ~25-op chain it
replaces, so 60 attentions are worth ~1500 units. Native's body cost 20388
against a 20000 budget — `by_cost == 0`, no compiled body, ~20388 tape entries
dispatched per token, two eager flushes and a pool clear each time. Stage 1's
cost was 18888 for the same program, `by_cost == 1`, one graph replay per
token. Measured consequence: 301.6 vs 242.4 ms/tok, and the same 1.244x on
prefill.

## The hazard the scoping brings with it, and the fix for it

A callee is **inlined**, and one reached from two call sites lowers its block
twice, binding its arguments to different slots each time. The sdpa mask cache
was keyed by the mask base's `mlir::Value` — one entry per distinct mask per
frame, which is what stops 60 layers from each building their own
`[.., .., Tq, Tk]` additive mask. With callee-rooted matches recognized, that
key becomes wrong: `attn(q, k, v, mask_a)` and `attn(q, k, v, mask_b)` through
one helper are the SAME `mlir::Value` and two different arrays, so the second
attention would silently read the first one's mask.

The cache is now keyed by the base's **slot**, which is what the emission
actually reads. Where a value is lowered once — every case that existed before
this pass — the two keys are the same key.

## What did NOT widen: qmm and moe

Deliberately, and it is not laziness: `qmm.py` and `moe.py` both walk
`_walk_blocks(interp._main_block())`, so widening them natively would make the
native lowering recognize matches Stage 1 does not — and the match set is
exactly what `BlockCost` is computed from, which is what the compile decision
is computed from. The two stacks have to agree there above all (the same
argument P26 made against raising `METALJAX_TRACE_BUDGET` in one stack only).
There is a mechanical reason as well: a qmm pack is built by `LowerCone` over
**@main's** block and arguments, and a callee-rooted match's operands are the
callee's block arguments, which no cone rooted in @main can bind.

## The narration this pass added

`METALJAX_DEBUG=1` now prints the while gate's own inputs:

```
[metaljax-native] while gate: cost=20388 bytes=5567MB budget=20000 by_cost=0
                  by_bytes=11 pure=1 body_compile=0 chunkable=1 kmax=1 period=1
```

P26 had to reconstruct a body's cost from an enclosing program's, because
neither stack printed it (`trip=1024 x body + overhead`, exact but indirect).
Two integers say it directly, and the second new contract below is written
against them rather than against a hard-coded budget. Diagnostic only: nothing
in the line decides anything.

## The tests

Two numeric rows in `execute_test.py`, both differentials against jax-CPU like
every other row there, and both PASS on the new binary (1.2e-7 / 2.4e-7):

* **`sdpa inside a callee`** — a `fori_loop` whose body is a non-inlined jit
  holding the whole attention. This is the graph shape the recognizer was
  blind to. It passes on the OLD binary too, unfused, which is exactly why the
  contracts below exist: an unfused attention is not a wrong answer, it is a
  slow one, and no numeric row can tell the difference.
* **`sdpa two masks through one callee`** — ONE jitted helper, two call sites,
  two different additive masks. This is the mask-cache hazard, and it is a
  row that can only fail on a binary that has the scoping without the re-key.

Two contracts (`_p26_callee_sdpa`), both derived from the runs' own narration
rather than from hard-coded numbers:

* **`an attention in a callee fuses`** — the same program under
  `METALJAX_SDPA=1/0`: 2 fused with, 0 without, answers agreeing to 1e-5 (the
  fused attention is a different kernel, ~6e-8 relative here — it is not
  bit-identical and was never expected to be).
* **`the callee discount moves the gate`** — P26's cliff in miniature: read
  the fused and unfused body costs off the new `while gate` line, set
  `METALJAX_TRACE_BUDGET` strictly between them, and the same program then
  compiles its body in the fused arm and not in the unfused one.

**A trap the first draft of the second contract fell into, worth recording:**
a recognized program is lowered **twice** (P17's two-phase compile — a plain
tape at `CompileAndLoad`, then a fused one at the first execute, which is the
tape that runs), so there are two `while gate` lines per program and the FIRST
is undiscounted even in the fused arm. Reading the maximum gave "cost 47
fused / 47 unfused -- no discount" on a binary where the fusion was demonstrably
firing. The last line is the one that runs.

## Measured: row 1 (gemma4-31B), the row this was about

Protocol: machine lock held for the whole campaign (build inside the hold),
recovery precheck + settle before each row, `mem_guard.sh` budget 80 G /
`GUARD_RSS_GB=115` / system 100, one process per arm, `METALJAX_DEBUG=1`
throughout — P24's and P26's protocol exactly, so the cells are comparable to
theirs. Runner `p26b_run.sh`, logs
`~/.cache/metaljax-bench/logs/p26b-callee-sdpa/`.

**The narration first, because it is the acceptance criterion and it landed on
the predicted number:**

```
[metaljax-native] sdpa: 60 fused attention(s) recognized
[metaljax-native] while gate: cost=20388 ... by_cost=0 body_compile=0   <- plain tape
[metaljax-native] while gate: cost=18888 ... by_cost=1 body_compile=1   <- fused tape
```

60 fused attentions where the RC fused **zero**, and Stage 1 fuses exactly 60.
The decode body's cost falls 20388 -> **18888**, which is Stage 1's cost to the
unit, and `by_cost` goes 0 -> 1: **the body compiles at the shipped budget,
with no override**. P26 predicted "20388 -> ~18888"; the derivation was exact.

The sampler counters flip with it, exactly as P26's `TRACE_BUDGET=21000` probe
made them flip:

| | RC (P24) | P26b |
|---|---|---|
| `jit__sample_loop` | `flushes=257(+clear 129) compiles=0 compiled_calls=0` | `flushes=1(+trim 1) compiles=0 compiled_calls=128` |

128 compiled replays = one graph replay per decode token, against zero before.

**The timing, four arms, all measured today under one hold:**

| arm | decode ms/tok | vs Stage 1 | prefill ms | tokens vs Stage 1 |
|---|---:|---:|---:|---|
| Stage 1 (P24, clean) | 242.4 | 1.000 | 1906.5 | reference |
| native RC (P24) | 301.6 | 1.244 | 2372.5 | diverges at 34 |
| native, P25 only (control) | 270.8 | 1.117 | 2334.0 | diverges at 34 |
| **native, P25 + P26b** | **239.8** | **0.989** | 2311.8 | **identical, 64/64** |

Read three things out of that.

**The row is no longer a regression; it is ahead.** 301.6 -> 239.8 is 1.258x on
the same row, of which P25's trim-instead-of-dump is 1.114x (301.6 -> 270.8)
and this change is 1.129x (270.8 -> 239.8). Against the 0.11.3 anchor of 237.5
the native stack now sits at 1.010x, and against Stage 1 at 0.989x — the
acceptance bar was "~1.05x or better".

**The token stream is Stage-1-identical**, which P26 predicted and which is the
cleaner evidence that the divergence was the recognizer's: the RC arm and the
P25-only arm both still diverge at token 34, and the only difference in the
third arm is that native now computes attention the same way Stage 1 does.
Row 1's entry in the token-divergence ladder is closed — and note it changes
the published stream for this row, which is the warning P26 attached to the
prediction.

**Prefill barely moved, and that is a finding rather than a disappointment.**
The 60 attentions are all in the DECODE program; the prefill program
(`jit___call`) takes no fusion on either stack, and its narration is now
identical on both:

```
native  : main: pure=1 cost=20839 bytes=37125.4MB compile=0
Stage 1 : exec jit___call: pure=True cost=20839 bytes=37125.4MB compile=False
```

Same cost, same bytes, same refusal — 839 units over the same 20000 budget,
4.2 % — so the two stacks now agree on **every** compile decision this row
makes (the sample-loop main matches too: 19341367 both sides). What is left on
prefill is not a decision, it is the eager path itself: 2311.8 vs 1906.5 =
**1.21x, native slower than Python, on a program neither compiles**. That is
carried to `notes/framework-gap-gemma31b.md` as its own item.

## Measured: rows 2 and 4

| row | RC (P24) | P26b | Stage 1 (P24) | vs Stage 1 | tokens |
|---|---:|---:|---:|---:|---|
| 2, gemma4-12B | 98.6 | **93.9** | 93.8 | 1.001 (was 1.051) | identical |
| 4, gemma4-E2B | 27.0 | 27.2 | 27.0 | 1.007 | identical |

Row 2 is the control that proves the mechanism rather than the timing, and its
narration is the cleanest evidence in the pass: **48 fused attentions in the
decode body and 8 in the prefill program — Stage 1's own counts exactly — and
the decode body's cost 16306 -> 15130, which is Stage 1's 15130 to the unit**
(P26 recorded both numbers from Stage 1's side before this existed). The 12B
was never over the budget, so nothing here was about the cliff; the row still
moved 98.6 -> 93.9, which retires P26's "12B gap = the native stack's baseline
overhead of 1.051x" — it was not a baseline, it was this plus P25's pool dumps.

Row 4 (E2B, keras path) takes no fusion on either binary and is unchanged
inside its noise band (27.0 / 27.2 across today's and P24's runs).

## Blast radius, counted

`census.sh`: on texmo's two attention configurations (`mid05-b16l256`,
`big05-b8l512`) **both** binaries fuse **zero** attentions — the change moves
nothing in the texmo suite, which is why the gate is bit-for-bit the same 106.
`decline_census` is 35 of 35 on both. The rewrite reaches exactly the shape it
was built for: an attention rooted inside a `func.call` callee.

## Battery

| check | result |
|---|---|
| `execute_test.py` (frozen p26b dylib) | **540 ok, 0 failures**, all four arms (compiled / eager / msl-off / forced build failure); 471 cases each |
| ...the two new rows | `sdpa inside a callee` 1.2e-7, `sdpa two masks through one callee` 2.4e-7 vs jax-CPU |
| ...the two new contracts | 2 fused / 0 with `METALJAX_SDPA=0`; cost 19 fused / 47 unfused, `body_compile=1/0` at the budget between them |
| `texmo_gate.py` | **106 ok** (19 via sensitivity scaling), 0 decline, 0 FAIL, 0 error, of 106 |
| `smoke_test.py` | all checkpoints passed |
| `decline_census.py` | 35 of 35 programs lower |
| `bazel test //...` | PASSED |
| model rows | 1, 2, 4 above; row 1 also with a P25-only control |

Artifacts: `~/.cache/metaljax-bench/logs/p26b-callee-sdpa/` (runners
`p26b_battery.sh`, `p26b_run.sh`, `p26b_tail.sh`, `census.sh`, the frozen
dylib `frozen-p26b.dylib`), data
`notes/data/p26b-callee-sdpa-2026-08-16.json`.

## Scrutiny

* **`bazel test //...` proves less than it looks.** The only target is
  `//metal:runtime_gil_free_test` and it reported `(cached)` after this change,
  i.e. it does not depend on the two files edited here. The coverage that
  matters is `execute_test` + the gate.
* **The mask re-key is a hazard fix without a failing test on the shipped
  binary.** `sdpa two masks through one callee` passes on the OLD binary too
  (unfused), and can only fail on a binary with the scoping and the old key —
  a build that does not exist. `maskkey_toggle.py` in the session scratch
  builds exactly that variant if someone wants the proof; it was not run, to
  keep the campaign inside one hold.
* **Unreachable functions are analysed too.** The walk is Stage 1's — every
  function in the module, not the ones reachable from @main — so a match in a
  dead function would be counted and never emitted. Harmless (nothing lowers
  it, and `BlockCost` only recurses through real calls) but it is why the
  narration's count is "recognized", not "emitted".
* **`futures_` is still keyed by `mlir::Value`** (the async-op slot map), which
  is the same shape of hazard the mask cache had. It is pre-existing and
  unrelated to attention — an `async_start` inside a twice-inlined callee would
  need it — and it is left alone deliberately rather than fixed blind.
* **This is the third time an `id()`-like key has bitten** (kernel names in
  0.2.0, sort-comparator dep walks in 0.4.1, this). The pattern is always the
  same: a key that is stable in the IR is not stable in the LOWERING.
* Row 1's published token stream changes with this commit (it becomes
  Stage-1-identical). That is an improvement and a change; it should be said in
  the release notes rather than discovered.
