# The metaljax executor runtime -- FORKED from metaljax/native at 6c2bb5e
# (P8.5), per Oleg's freeze decision: the Stage 1 implementation is no longer
# modified, so `native/` is frozen with it and every pure-native runtime
# change from P9 on lands HERE and only here.  Backporting a fix to
# `native/` is an explicit decision (it re-opens the Stage 1 battery), not a
# reflex.  History before the fork point is shared; `git log native/<file>`
# is this tree's archaeology.
#
# The two nanobind translation units (`bindings.cc`, `metaljax_native.cc`)
# stayed behind with Stage 1 -- program.h names no Python, and keeping it
# that way is what this package's build enforces.
#
# `srcs` is spelled out rather than globbed on purpose, in native/build.sh's
# order (which no longer compiles these copies -- it compiles the frozen
# originals for the Stage 1 extension).

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "runtime",
    srcs = [
        "compile.cc",
        "config.cc",
        "control.cc",
        "dtypes.cc",
        "emits.cc",
        "host.cc",
        "msl.cc",
        "ops_conv.cc",
        "ops_elementwise.cc",
        "ops_index.cc",
        "ops_linalg.cc",
        "ops_reduce.cc",
        "ops_rng.cc",
        "ops_shape.cc",
        "program.cc",
        "runtime.cc",
    ],
    hdrs = [
        "msl.h",
        "program.h",
    ],
    copts = [
        # MLX's half/bfloat16 headers declare a copy constructor without a
        # copy assignment and default the comparison operators; both are
        # noisy under XLA's warning set and neither is ours to fix.
        "-Wno-deprecated-copy",
        "-Wno-c++20-extensions",
    ],
    includes = ["."],
    # The plugin dylib's link forces every object in (symbol-collision
    # trouble between the runtime, libmlx and XLA's static LLVM/absl must be
    # a build failure, not a phase-3 mystery), and the gil-free test's
    # opcode-by-name lookups need the registry even before anything calls it.
    alwayslink = 1,
    deps = ["@mlx"],
)
