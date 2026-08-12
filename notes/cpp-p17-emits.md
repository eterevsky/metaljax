# P17: the recognizers emit, natively (2026-08-12)

The phase-2 plugin's lowering now recognizes all three graph shapes Stage 1
rewrites and emits the M4 opcodes the shared runtime has been able to execute
since May: `metaljax.qmm`, `metaljax.moe.*`, `metaljax.sdpa`.  This closes the
gap [benchmarks/perf-2026-08-native-baseline.md](../benchmarks/perf-2026-08-native-baseline.md)
measured — items 2, 3 and 4 of its frontier (qmm 434x on gpt-oss, MoE 6.9x on
26B-A4B, sdpa 3.1x on SD 3.5).  msl_scan (item 1) is a milestone of its own and
is untouched.

Nothing about the rewrites was re-designed: `src/metaljax/{qmm,sdpa,moe}.py`
are the specification, `runtime/emits.cc`'s `Cursor` reads are the ground truth
for the encodings, and `src/metaljax/tape.py`'s `_lower_qmm` / `_lower_sdpa` /
`_lower_moe` are what the new `Lowering::Lower*` methods transliterate.

## The shape of it

    plugin-native/metal/metal_recognize.h   the plan: QmmMatch, SdpaMatch,
                                            MoeMatch/MoeNode, RewritePlan
    plugin-native/metal/metal_qmm.cc        qmm.py's matching AND its packing
    plugin-native/metal/metal_sdpa.cc       sdpa.py's matching
    plugin-native/metal/metal_moe.cc        moe.py's router proof, pair-space
                                            planner and router check
    plugin-native/metal/metal_lowering.cc   the emission + the two-phase entry

**The two-phase compile is the whole architectural point.**  A quantized
weight is an ARGUMENT of the executable, not a constant, so it cannot be packed
at compile time — and the packed arrays must be tape INPUTS rather than
captures, because `mx::compile` bakes a captured constant by value and a repack
would then never be seen.  So:

* `CompileAndLoad` lowers the program exactly as it did before, with no
  recognizers.  That tape is always valid and is the fallback.
* The FIRST `Execute` re-parses the StableHLO the executable kept (the module
  the compile was handed belongs to that call), runs the three analyses,
  builds the packs and the router checks on the real buffers, and lowers a
  SECOND tape with the emits.  `MetalLoadedExecutable::Tape` keeps whichever it
  got; a decline of any kind leaves the plain tape in place.
* A later call that hands over different weight buffers repacks (the pack is a
  pure function of the buffers its reconstruction read, so buffer identity is
  the test), bounded by `kMaxRepacks` = 8 — past that, repacking costs more
  than the chain it replaces and the plain tape stays for good.

`Lowering::LowerCone` is what evaluates an operand subtree on concrete
buffers: the cone of a set of values over @main's arguments, lowered into one
frame as a Program of its own and run.  It gets qmm.py's `_eval` staging for
free — the tape's drop lists release every intermediate at its last use, which
matters when the intermediates are full weight size — and its `bound` argument
pins values in the MIDDLE of the graph, which is how the MoE router check runs
on synthetic logits.

## qmm

`_try_affine` (both operand sides), `_try_perchannel`, `_finish`, `_prune`,
then the first-execute half: `_to_nk`, the group-constancy checks, the group
size, `pack_codes`, `_scale_bias` with the `METALJAX_QMM_SCALES` policy, the
MXFP4 grid and E8M0 verifications, and `_regroup` (the column-digest clustering
that un-interleaves a contraction axis, which is what keras' EinsumDense
needs).  Every exactness check is exact and every failure disables that one
dot.

Measured on `plugin-native/execute_test.py`'s own rows: int4 sub-channel
(f32/bf16), 8-bit codes, per-channel, the interleaved EinsumDense projection
(packs `regrouped`, four pack arrays), MXFP4 dense and batched-expert, and a
projection inside a decode loop (the packs cross into the while body as extra
region captures).

NOT ported, deliberately: `_Source`'s row-blocked evaluation and the
cross-executable build cache.  Both are memory/latency optimizations over an
evaluation this tape already stages op by op; leaving them out costs peak
memory on a 20 GB weight set, never an answer.  `_NoCache` — MLX's buffer cache
off for the duration of a pack — IS ported, because what it bounds is the
CLAIMED memory a watchdog reads.

## sdpa

The atom algebra (`_roles_*`, `_recipe`), both chains (`_logits` to the first
dot, `_probs` to the `exp`), the two reductions, the deferred normalization and
the survivor rules.  Both operand orders, normalize-first and normalize-last,
optional scale, boolean-select and additive masks, `[B,T,H,D]` and `[B,H,T,D]`,
the rank-5 spelling `jax.nn.dot_product_attention` emits, reassociation through
reshape/transpose/convert/sharding constraints/clamps/calls.  The mask cache is
the lowering's: one `metaljax.sdpa.mask` entry per distinct mask per frame,
because one causal mask is shared by every layer and building it costs a full
`[.., .., Tq, Tk]` tensor.

Not ported: `_all_blocks`, i.e. an attention rooted WHOLLY inside a `func.call`
callee is not a candidate (chains that merely pass through a call are).

## moe

The call-frame machinery (`_Scope.deref`), `_peel`, the router proof
(`_match_router`: the one-hot down to its `compare EQ (indices, iota)` form,
the top-k, the `[T, K]` layout), the pair-space planner with all four node
kinds, the use-count discipline and the dead sweep.  A per-expert dot whose
weight the quantized recognizer packed is dispatched by `gather_qmm` and that
qmm match is marked absorbed rather than emitted; everything else takes
`gather_mm` over the dense weight in place.

`VerifyMoe` is `_verify`, and it runs where the Python's runs — the eager
prologue of the first execute, before any trace, because it syncs with the
host.  Three draws of synthetic logits, coarse from the second on (exact ties
are where a top-k and a hand-rolled scatter would most easily disagree).  The
one case it cannot reach is a top-k bound inside a CALLEE, whose input a cone
has no way to pin; such a match stays unverified and runs dense.

**Substituting the logits is not a nicety.**  The first draft checked the
identity on the real logits of the first execute, and the decode-loop dispatch
— the one that matters, since it is what a sampler runs per token — declined
every time, because its logits are a loop carry no prologue can evaluate.  With
the substitution both dispatches of a prefill+decode program verify.

## Measured

`notes/data/p17-emits-micro.jsonl` (best of 20, ms, machine lock held,
sequential; `scripts/bench_recognizers.py`).  "no emits" is the SAME native
plugin under `METALJAX_RECOGNIZE=0`, so the programs, shapes and arithmetic are
identical and only the rewrite differs.

| benchmark | shape | native+emits | native, no emits | emit is worth | Stage 1 |
|---|---|---:|---:|---:|---:|
| qmm mxfp4 decode | gpt-oss gate_up, `[1,2880] x mxfp4[5760,2880]` | 0.250 | 1.910 | **7.63x** | 0.162 |
| qmm mxfp4 prefill | the same at T=64 | 0.906 | 2.001 | **2.21x** | 0.459 |
| qmm int4 decode | keras sub-channel `[1,4096] x int4[4096,4096]` | 0.172 | 1.141 | **6.62x** | 0.190 |
| moe decode | dense dispatch E=32 K=4 T=1 d=h=1024 | 0.246 | 1.087 | **4.43x** | 0.254 |
| moe prefill | the same at T=32 | 0.757 | 1.093 | 1.44x | 0.734 |
| sdpa 1024x1024 | B=2 H=16 Tq=Tk=1024 D=64 | 1.138 | 2.974 | **2.61x** | 1.217 |
| sdpa decode | B=1 H=32 Tq=1 Tk=2048 D=128 | 0.419 | 0.356 | 0.85x | 0.499 |

Two things to read out of the last two columns.  Against Stage 1 the native
path is now at or ahead of it on four of seven rows (int4 decode 0.91x, moe
decode 0.97x, moe prefill 1.03x, sdpa 1024 0.94x, sdpa decode 0.84x) and behind
on the two MXFP4 rows (1.55x and 1.98x) — sub-millisecond programs where the
difference is one program's dispatch, not the kernel.  And `sdpa decode` is the
one row where FUSING LOSES (0.85x): at `Tq=1` MLX's fused attention is slower
than the plain matmul-softmax-matmul, on both stacks (Stage 1 fuses it too and
is slower still).  Worth a look during the perf phase; it is not a P17
regression, since Stage 1 has the same property.

**The model rows were not run.**  Every one of them (keras directly,
gemma-lib through `kauldron`, maxtext through `array_record_module.so`) imports
TensorFlow, and the native dylib cannot be dlopened into such a process without
the exported-symbols relink the baseline campaign found and deliberately left
out of the tree.  Landing that linker change is a separate deliberate commit
with its own validation; the micro-benchmarks above stand in, on the exact
shapes those rows are made of.  The nearest thing to the campaign's headline
target (gpt-oss within ~2x of Stage 1's 21.9 ms/tok) is the qmm mxfp4 decode
row: 1.55x of Stage 1 at gpt-oss's own gate_up shape.

## Battery

`notes/data/p17-emits-battery.txt`.  `execute_test` 502 -> **520 checks**, all
matching jax-CPU (18 new rows: 9 qmm, 5 moe, 4 sdpa — every one of them a
FUSED answer against the literal chain the CPU backend runs); `texmo_gate`
**106 ok / 0 decline / 0 FAIL**; `smoke_test`; `decline_census` 35 of 35;
`bazel test //...`; `tests/` through the native plugin 1187 passed / 71 failed,
the same 71 as before this milestone.

**The 70 recognizer-family rows in `tests/` are unmoved, and cannot move.**
Every one of them asserts a counter in the Stage 1 PYTHON module
(`qmm.stats()`, `moe.stats()`), which a plugin with no interpreter in it has no
way to tick — and `src/` and `tests/` are frozen.  What those rows also contain
is real numeric content, and it now runs: with the counters neutralized by a
pytest plugin (a measuring instrument, not a fix), the three files go **70 ->
8**, and each of the 8 asserts a Stage 1 internal this port deliberately did
not bring over (the row-blocked packer, the build cache, the dead-sweep
worklist).  That is why the 17 execute_test rows exist: they are the same
graphs and the same references, in a harness the native plugin can satisfy.

## Gotchas found on the way

* **`RewritePlan::rebuild` forgot to clear `moe_roots`.**  A match disabled by
  the router check stayed in the root map with its object freed underneath —
  a segfault at the next lowering, and one that only appears when one match of
  several is disabled.  Every map a rebuild touches must be cleared in it.
* **`_regroup`'s column keys are per COLUMN, not per map.**  The first draft
  laid the digest words out map-major and read them column-major, so the
  clustering was garbage; the exact re-verification caught it (the pack
  declined instead of being wrong), which is the design working, but the
  EinsumDense projection lost its fusion until it was fixed.
* **The moe router check needs synthetic logits**, see above.
* **`METALJAX_MOE_VERIFY=0` is not a safe default** for anything: with it off
  the decode-loop match fused and computed the right answer, which is exactly
  the situation where a misread axis would be silent.
* A `select` mask sentinel must be at least `_MASK_FRACTION` (0.1) of the
  dtype's finite max; `-1e30` in f32 is NOT one, and a test that writes it gets
  no fusion (rightly — `select` and `add` are then different functions).
