# plugin-native — the fully-native metaljax PJRT plugin (Stage 2)

An `xla::PjRtClient` subclass; XLA's `pjrt_c_api_wrapper_impl` manufactures the
whole PJRT C API around it. No Python, no nanobind — unlike `plugin/`, which
trampolines into `metaljax.engine`.

```sh
cd plugin-native
bazel build //metal:libmetal_pjrt_native.dylib

cd .. && ./.venv/bin/python plugin-native/smoke_test.py
```

XLA comes from the read-only `metaljax/xla` checkout via `local_repository`
(pinned to jax 0.11.0's XLA revision); MLX comes from the venv's wheel via
`third_party/mlx`. First build is ~7 minutes, everything after that is seconds
(see `--disk_cache` in `.bazelrc`).

Status, measurements, gotchas and the route decision:
[`../notes/pjrt-native-p0.md`](../notes/pjrt-native-p0.md).
