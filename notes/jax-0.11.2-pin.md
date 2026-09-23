# The jax 0.11.2 pin (2026-09-22)

jax/jaxlib 0.11.2 (released 2026-09-17) replaces 0.11.0 as the pinned release: the pinned test suite is
`jax-v0.11.2/tests` (a `--depth 1` clone of the `jax-v0.11.2` tag, gitignored), the whitelist is
`notes/data/pinned-0.11.2-failures.txt`, and main's `.venv` plus the bench and gemma benchmark venvs run
jax/jaxlib 0.11.2.  The wheel's dependency range (`jax>=0.11,<0.12`) is unchanged: the plugin still runs
on 0.11.0.

## The plugin's XLA and the maxtext venv

- **The plugin's XLA moved to 91888df6** (jax 0.11.2's, the `archive_override` in
  `jax-v0.11.2/MODULE.bazel`; 2026-09-23, checkout `metaljax/xla-91888df6`; `metaljax/xla` keeps 0.11.0's
  131bf41 so older commits still build).  Still a WORKSPACE build: jax 0.11.2 is bzlmod-only and XLA
  gates WORKSPACE mode behind `--config=workspace` ("to be removed"), so the next pin move is likely the
  bzlmod migration.  The move needed `compatibility_proxy_repo()` (rules_cc 0.2.20), an absolute
  `--experimental_downloader_config` (tensorflow.bazelrc's is relative), `ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS`
  on the four plugin-shell targets (XLA deleted `xla/tsl/platform/status_macros.h`, which did exactly
  that), and `device_assignment` for the removed `computation_placer_hdr`; no source change otherwise.
  The plugin now reports PJRT C API 0.115 (was 0.114) and VHLO 1.20.0 (was 1.18.0), so jaxlib 0.11.2 no
  longer downgrades programs: two-operand `collective_broadcast` (VHLO 1.20) now compiles, and
  `collective_reduce` (1.19; jax does not emit it) reaches the plugin and declines by name.  jaxlib 0.11.0
  still loads the new plugin (identical `execute_test`).  Old vs new binary on jax 0.11.2: execute_test,
  ingest_test, pytest (526 + 1 xfail) and texmo_gate (106/106) identical, compile cost unchanged
  (`logs/xla918/`).  Scout: `logs/jax0112/xla-scout/findings.md`.
- **The maxtext benchmark venv stays on jax 0.11.0.**  jax 0.11.2 renamed `jax.experimental.hijax.
  HiPrimitive` to `HiPrim`; every released flax (latest 0.12.9) still imports the old name, so
  `flax.nnx` cannot load.  flax main fixed it on 2026-08-21 (unreleased).  The maxtext rows (10, 11's
  maxtext arm, 14, 15, 19) move when a flax release carries the rename.

## The suite on 0.11.2

Release binary of 0.11.8 (7434ae86): 28,677 passed / 137 failed.  After e96f861 (12c5a8d1): **28,679
passed / 135 failed / 6,120 skipped / 35 collection errors (99.53 %)**.  Against the 0.11.0 set (129 ids,
paths mapped): 5 ids no longer fail (4 in files jax removed, `linalg_test::testQrInvalidDtypeCPU`
changed), 11 are added:

| test | class |
|---|---|
| `lax_test.py::FunctionAccuracyTest::testSuccessOnComplexPlane_{arccos,arccosh,arcsin,arcsinh,arctan,arctanh,log1p,sqrt,tan}_complex64` (9) | jax-CPU 0.11.2 fails them identically -- same region, same points (e.g. the reference expects `arccos(-inf + 1e-15j) == 0j`); the test's reference, not a backend |
| `lax_autodiff_test.py::testAcoshGradLargeValuesFloat64` | new test; computes in f64 under `jax.enable_x64()` -- the strict f64 policy declines f64 compute (METALJAX_F64=downcast opts in, and 1e200 overflows f32 anyway) |
| `layout_test.py::test_host_auto_layout` | new test, TPU-only: skips on cpu/gpu by platform name, then asserts TPU memory space S(5) in the compiled HLO text; since e96f861 it lowers (LayoutConstraint is an alias) and fails only at the HLO-text assertion |

Fixed by e96f861 rather than whitelisted: `lax_test.py::LaxTest::testOpAgainstNumpy594` (float remainder
was inexact in every float dtype -- a silent-wrongness bug in every release) and
`overlap_test.py::OverlapTest::test_avoid_excess_precision` (int2/uint2).  The new `tests/numerics/`
accuracy suite is Bazel-only (it `pytest.skip`s itself) and needs mpmath; it is outside the pinned
suite exactly as it is for jax's own pytest runs.

## jax 0.11.2 behaviour changes our own tests had to follow

- The `lax.linalg` primitives gained dtype rules (f32/f64/c64/c128): jax refuses bf16/f16
  factorizations at trace time on every backend, so metaljax's half-precision linalg (host handlers
  that upcast) is unreachable on jax >= 0.11.2.  It still serves 0.11.0/0.11.1.
- `exec_time_optimization_effort` / `memory_fitting_effort` left jax's build options: passed as
  `compiler_options` they now reach XLA's validator and are refused, by jax-CPU and by this plugin alike.
- `fori_loop` bodies are no longer wrapped in `closed_call`, so the KV in-place rewrite now also sees
  (and rewrites) that case.

Model rows on the release binary, jax 0.11.2 vs 0.11.0 in the benchmark venvs: rows 2 (gemma library),
4 (keras-hub) and 11 keras read the same (56.5 / 16.8 / 5.1) with identical token streams.
