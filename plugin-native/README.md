# plugin-native — the metaljax PJRT plugin

An `xla::PjRtClient` subclass; XLA's `pjrt_c_api_wrapper_impl` manufactures the
whole PJRT C API around it. No Python, no nanobind. Since the Stage-1
retirement (0.11.6) this is the only engine — the Python trampoline
(`plugin/` + `metaljax.engine`) is gone.

```sh
cd plugin-native
bazel build //metal:libmetal_pjrt_native.dylib
bazel test //...          # incl. the GIL-free runtime test below

cd .. && ./.venv/bin/python plugin-native/smoke_test.py    # it works at all
./.venv/bin/python plugin-native/execute_test.py           # vs jax-CPU
./.venv/bin/python plugin-native/texmo_gate.py             # the whole suite
./.venv/bin/python plugin-native/decline_census.py         # what still declines
./.venv/bin/python plugin-native/ingest_test.py            # the transfer path

# coexistence with a static-protobuf/LLVM carrier -- needs a python that has
# TensorFlow or array_record, so not the repo venv:
~/.cache/metaljax-bench/venvs/bench/bin/python plugin-native/coexist_test.py
```

`execute_test.py` is the differential suite: every expression is run through
this plugin and through jax on the CPU backend (in a subprocess of its own,
since a process with `JAX_PLATFORMS=metal` can see no other), and the CPU
answer is the bar. `texmo_gate.py` is the same doctrine on real workloads —
every configuration in `benchmarks/texmo-suite.csv` trains one chunk through
both backends, compared with a sensitivity-scaled tolerance (inherited from
the retired `scripts/texmo_check.py`) — and it is the standing gate: it exits
nonzero if any
configuration computes a different answer, while a program the plugin still
declines is reported and forgiven.

`METALJAX_DUMP_TAPE=1` prints the lowered tape; start reading it from
`METALJAX_DUMP_MODULE=1` (the module XLA's parse hands us is not the one jax
printed — chlo is legalized and constants are hoisted).

The executor runtime is `runtime/` — forked from the pre-PJRT `native/` tree
at 6c2bb5e and diverged since; `native/` itself was deleted with the Stage-1
retirement. `CompileAndLoad` lowers the parsed StableHLO into one of its
`Program`s and `Execute` replays it. `//metal:runtime_gil_free_test` builds a
tape through the same C++ API and runs it on the GPU in a process with no
interpreter in it.

The dylib exports two symbols, and `metal/exported_symbols.exp` is what holds it
to exactly those: everything else — XLA, MLIR/LLVM, StableHLO, protobuf, absl —
is private extern, because dyld coalesces weak definitions across images and an
unrestricted export table makes this plugin SIGSEGV at `dlopen` in a process
that already holds TensorFlow or array_record (`coexist_test.py` is the
contract, both load orders). The list is also why the dylib is 46 MB and not
166. `GetPjrtApi` is the plugin; the other,
`metaljax_native_set_callback_trampoline`, is the callback bridge (P13):
`src/jax_plugins/metal/__init__.py` keeps the registry of Python callables that
`jax.debug.print` / `pure_callback` / `io_callback` lower to and installs a
ctypes callback here, so the GIL enters this plugin inside a user callback and
nowhere else. Its C ABI is `runtime/host_callback.h`.

XLA comes from the read-only `metaljax/xla` checkout via `local_repository`
(pinned to jax 0.11.0's XLA revision); MLX is our vendored build, staged in
`src/metaljax/lib/mlx` by `scripts/vendor_mlx.sh` and picked up through
`third_party/mlx` (`METALJAX_MLX_DIR` overrides the location — that is also
the build-level A/B lever between MLX trees). First build is ~7 minutes,
everything after that is seconds (see `--disk_cache` in `.bazelrc`).

`METALJAX_VERIFY_COMPILE=1` runs every executable a SECOND time op by op and
reports any output that differs from the compiled path — the one divergence
this plugin can have that is silent by construction (`=dump` also prints the
arguments and both answers). `METALJAX_MLX_COMPILE_MODE=no_fuse` then says
whether such a divergence is MLX's kernel fusion rather than this tape; MLX's
own `set_compile_mode` has no environment variable, which is why the knob is
here. Both are off by default and cost nothing when unset.

Status, measurements, gotchas and the route decision:
[`../notes/pjrt-native-p0.md`](../notes/pjrt-native-p0.md), then
[`../notes/cpp-p1-runtime.md`](../notes/cpp-p1-runtime.md) for the runtime and
[`../notes/cpp-p2-lowering.md`](../notes/cpp-p2-lowering.md) for the lowering
and the executable — which ops lower, which decline, and why — then
[`../notes/cpp-p3-control.md`](../notes/cpp-p3-control.md) for control flow,
[`../notes/cpp-p4-gather-scatter.md`](../notes/cpp-p4-gather-scatter.md) for
gather/scatter, the RNG and the gate, and
[`../notes/cpp-p5-compile.md`](../notes/cpp-p5-compile.md) for the compile
decisions — which programs and loop bodies are traced through `mx::compile`,
and why that turned out to be correctness rather than tuning.
