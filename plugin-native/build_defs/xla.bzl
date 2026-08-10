"""Shared bazel helpers for every cc target in this workspace.

Kept out of //metal so the future executor-runtime package (and any test
binary) can use the same rules without depending on the plugin shell.
"""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

# XLA's own xla_cc_binary() macro cannot be reused out of tree: half of its
# _XLA_SHARED_OBJECT_SENSITIVE_DEPS entries are Label() objects (repo-anchored,
# fine) and half are bare "//xla/..." strings, which Starlark resolves against
# the *calling* package's repository -- i.e. against this workspace, where
# //xla/tsl/... does not exist.  This is the same list from
# xla/xla.default.bzl, re-spelled with @xla// / @tsl// prefixes; diff it
# against XLA's whenever the pin moves.
XLA_SHARED_OBJECT_SENSITIVE_DEPS = [
    "@xla//xla:autotune_results_proto_cc_impl",
    "@xla//xla:autotuning_proto_cc_impl",
    "@xla//xla:xla_data_proto_cc_impl",
    "@xla//xla:xla_proto_cc_impl",
    "@xla//xla/service:buffer_assignment_proto_cc_impl",
    "@xla//xla/service:hlo_proto_cc_impl",
    "@xla//xla/service:metrics_proto_cc_impl",
    "@xla//xla/service/gpu:backend_configs_cc_impl",
    "@xla//xla/service/gpu/model:hlo_op_profile_proto_cc_impl",
    "@xla//xla/service/memory_space_assignment:memory_space_assignment_proto_cc_impl",
    "@xla//xla/stream_executor:device_description_proto_cc_impl",
    "@xla//xla/stream_executor:stream_executor_impl",
    "@xla//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl",
    "@xla//xla/backends/cpu/runtime:thunk_proto_cc_impl",
    "@xla//xla/tsl/framework:allocator_registry_impl",
    "@xla//xla/tsl/framework:allocator",
    "@xla//xla/tsl/platform:env_impl",
    "@xla//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
    "@xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
    "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
    "@tsl//tsl/profiler/protobuf:xplane_proto_cc_impl",
    "@xla//xla/tsl/profiler/utils:time_utils_impl",
    "@xla//xla/tsl/protobuf:protos_all_cc_impl",
]

def metaljax_cc_binary(name, deps = [], **kwargs):
    """cc_binary that links XLA's shared-object-sensitive impl targets."""
    cc_binary(
        name = name,
        deps = deps + XLA_SHARED_OBJECT_SENSITIVE_DEPS,
        **kwargs
    )
