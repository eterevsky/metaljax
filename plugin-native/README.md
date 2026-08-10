# plugin-native — the fully-native metaljax PJRT plugin (Stage 2)

An `xla::PjRtClient` subclass; XLA's `pjrt_c_api_wrapper_impl` manufactures the
whole PJRT C API around it. No Python, no nanobind — unlike `plugin/`, which
trampolines into `metaljax.engine`.

```sh
cd plugin-native
bazel build //metal:libmetal_pjrt_native.dylib
bazel test //...          # incl. the GIL-free runtime test below

cd .. && ./.venv/bin/python plugin-native/smoke_test.py    # it works at all
./.venv/bin/python plugin-native/execute_test.py           # vs jax-CPU
```

`execute_test.py` is the differential suite: every expression is run through
this plugin and through jax on the CPU backend (in a subprocess of its own,
since a process with `JAX_PLATFORMS=metal` can see no other), and the CPU
answer is the bar. `METALJAX_DUMP_TAPE=1` prints the lowered tape, which is
what a reviewer diffs against the Stage 1 lowering's.

The executor runtime (`../native`) builds here too, as
`@metaljax_runtime//:runtime` — the same sources the nanobind extension
compiles, shared through a `new_local_repository` whose build file is
`third_party/metaljax_runtime/BUILD.runtime`. `CompileAndLoad` lowers the
parsed StableHLO into one of its `Program`s and `Execute` replays it.
`//metal:runtime_gil_free_test` builds a tape through the same C++ API and
runs it on the GPU in a process with no interpreter in it.

XLA comes from the read-only `metaljax/xla` checkout via `local_repository`
(pinned to jax 0.11.0's XLA revision); MLX comes from the venv's wheel via
`third_party/mlx`. First build is ~7 minutes, everything after that is seconds
(see `--disk_cache` in `.bazelrc`).

Status, measurements, gotchas and the route decision:
[`../notes/pjrt-native-p0.md`](../notes/pjrt-native-p0.md), then
[`../notes/cpp-p1-runtime.md`](../notes/cpp-p1-runtime.md) for the runtime and
[`../notes/cpp-p2-lowering.md`](../notes/cpp-p2-lowering.md) for the lowering
and the executable — which ops lower, which decline, and why.
