# The metaljax native PJRT plugin: an xla::PjRtClient subclass wrapped into a
# full PJRT C API by XLA's pjrt_c_api_wrapper_impl.
#
# This package holds the plugin shell (client / device / memory space / buffer
# plumbing) and the StableHLO -> tape lowering that stands between it and the
# shared executor runtime (//runtime).  The bazel helpers
# live in //build_defs.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//build_defs:xla.bzl", "metaljax_cc_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "metal_names",
    hdrs = ["metal_names.h"],
    deps = ["@com_google_absl//absl/strings:string_view"],
)

cc_library(
    name = "metal_stream",
    srcs = ["metal_stream.cc"],
    hdrs = ["metal_stream.h"],
    deps = ["@mlx"],
)

cc_library(
    name = "metal_dtypes",
    srcs = ["metal_dtypes.cc"],
    hdrs = ["metal_dtypes.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "//runtime",
        "@mlx",
        "@xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "metal_buffer",
    srcs = ["metal_buffer.cc"],
    hdrs = ["metal_buffer.h"],
    deps = [
        ":metal_dtypes",
        ":metal_stream",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@mlx",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:statusor",
    ],
)

# StableHLO -> the executor's tape.  Depends on the runtime (for the opcode
# and dtype registries and the Program itself) and on the StableHLO dialect,
# and on no part of the plugin shell: a lowering is a pure function of the
# module.
cc_library(
    name = "metal_lowering",
    srcs = ["metal_lowering.cc"],
    hdrs = ["metal_lowering.h"],
    deps = [
        ":metal_dtypes",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "//runtime",
        "@mlx",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "metal_executable",
    srcs = ["metal_executable.cc"],
    hdrs = ["metal_executable.h"],
    deps = [
        ":metal_buffer",
        ":metal_lowering",
        ":metal_names",
        ":metal_stream",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime",
        "@mlx",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/service:computation_placer_hdr",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "metal_client",
    srcs = ["metal_client.cc"],
    hdrs = ["metal_client.h"],
    deps = [
        ":metal_buffer",
        ":metal_dtypes",
        ":metal_executable",
        ":metal_lowering",
        ":metal_names",
        ":metal_stream",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@mlx",
        "@tsl//tsl/platform:fingerprint",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_device_description",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "metal_c_pjrt",
    srcs = ["metal_c_pjrt.cc"],
    hdrs = ["metal_c_pjrt.h"],
    deps = [
        ":metal_client",
        "@com_google_absl//absl/status",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_status_utils",
        "@xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
    ],
    alwayslink = 1,
)

# The loadable plugin.  METALJAX_PLUGIN_PATH should point at this file.
#
# The executor runtime rides along before anything calls it: it is alwayslink,
# so every one of its objects is really in this dylib, and that is what proves
# the tape, libmlx and XLA's statically-linked LLVM/MLIR/absl can share one
# image.  Discovering otherwise while wiring up CompileAndLoad would be a much
# worse place to find out.
metaljax_cc_binary(
    name = "libmetal_pjrt_native.dylib",
    linkshared = True,
    deps = [
        ":metal_c_pjrt",
        "//runtime",
    ],
)

# Phase 2 P1: the executor runtime driven from a process with no interpreter
# in it.  Depends on //runtime and @mlx and on nothing else -- no XLA,
# no plugin -- because the claim being tested is exactly that the tape stands
# on its own.
#
# `local`: the test talks to the Metal device and MLX writes its compiled
# kernels to the user's cache directory, neither of which survives bazel's
# darwin sandbox.
cc_test(
    name = "runtime_gil_free_test",
    size = "small",
    srcs = ["runtime_gil_free_test.cc"],
    tags = ["local"],
    deps = [
        "//runtime",
        "@mlx",
    ],
)
