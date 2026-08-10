# The metaljax native PJRT plugin: an xla::PjRtClient subclass wrapped into a
# full PJRT C API by XLA's pjrt_c_api_wrapper_impl.
#
# This package holds only the *plugin shell* (client / device / memory space /
# buffer plumbing).  The execution engine will land in a sibling package so it
# can be built and tested on its own; the bazel helpers both need live in
# //build_defs.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//build_defs:xla.bzl", "metaljax_cc_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "metal_buffer",
    srcs = ["metal_buffer.cc"],
    hdrs = ["metal_buffer.h"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@mlx",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "metal_client",
    srcs = ["metal_client.cc"],
    hdrs = ["metal_client.h"],
    deps = [
        ":metal_buffer",
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
metaljax_cc_binary(
    name = "libmetal_pjrt_native.dylib",
    linkshared = True,
    deps = [":metal_c_pjrt"],
)
