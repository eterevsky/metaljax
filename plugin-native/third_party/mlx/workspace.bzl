"""Makes the MLX C++ library (headers + the dylib) available to Bazel.

Since the vendoring milestone (2026-08-17) the default is OUR OWN BUILD of
OUR OWN FORK: `scripts/vendor_mlx.sh` builds `mlx-src/` and stages
`src/metaljax/lib/mlx/{lib,include}`, and that tree is what the plugin links
and what `hatch_build.py` copies into the native wheel.  Three consequences,
all of them the reason to bother:

  * the plugin carries the command-buffer fence fix (`notes/mlx-patch-
    diagnosis.md`) that no released MLX has;
  * the native wheel needs **no `mlx` on PyPI at all** -- the Metal runtime
    is inside it;
  * the ABI handshake stops being a runtime hope.  Plugin and library are
    built from one tree in one step, so "which mlx is this?" is a build-time
    fact, recorded in `src/metaljax/lib/mlx/VENDOR_STAMP`.

The vendored dylibs are RENAMED (`libmlx_metaljax.dylib`,
`libjaccl_metaljax.dylib`).  A consumer process may also import the public
`mlx` wheel -- Stage 1 does -- and two images with the same install name in
one process is a symbol-interposition lottery over one GPU.  Renamed, they
are merely two independent libraries.

Override the location with METALJAX_MLX_DIR.  A directory holding an
*unrenamed* `lib/libmlx.dylib` (a pip wheel's `site-packages/mlx`) still
works, so the pre-vendoring linkage can be rebuilt for an A/B; that path
keeps the wheel-relative rpath the pip layout needs.
"""

_DEFAULT_MLX_DIR = "/Users/oleg/metaljax/src/metaljax/lib/mlx"

_BUILD = """\
package(default_visibility = ["//visibility:public"])

cc_import(
    name = "libmlx_import",
    shared_library = "lib/{dylib}",
)

cc_library(
    name = "mlx",
    hdrs = glob([
        "include/mlx/**/*.h",
        "include/mlx/**/*.hpp",
    ]),
    includes = ["include"],
    # The dylib's install name is @rpath-relative (and it pulls its
    # libjaccl from the same directory), so every binary that links it
    # needs a run-path that finds it.  Two are baked in:
    #
    #   * a RELATIVE one, resolved against the plugin dylib itself, so it
    #     follows the consumer's venv wherever that is.  Vendored, the
    #     library lives INSIDE the wheel -- site-packages/metaljax/lib/mlx/
    #     lib/ next to site-packages/metaljax/lib/libmetal_pjrt_native.dylib
    #     -- hence `@loader_path/mlx/lib`.  (Against a pip wheel's mlx the
    #     layout is the old one, `@loader_path/../../mlx/lib`.)
    #   * the absolute path of the staged tree -- what bazel-bin binaries
    #     (smoke_test.py, any cc_test) load, from any process, even one
    #     that has not imported mlx.
    #
    # ORDER IS LOAD-BEARING.  dyld walks LC_RPATHs in order and takes the
    # first hit, and this machine's absolute path still exists on this
    # machine -- with it first, a wheel installed into another venv
    # silently loaded *the build tree's* library (observed with
    # DYLD_PRINT_LIBRARIES, in the pip era: a 3.13 venv running the 3.14
    # venv's mlx).  The wheel's own copy must win; the absolute path is
    # only a fallback for bazel-bin, where the relative one misses.
    linkopts = [
        "-Wl,-rpath,{relative_rpath}",
        "-Wl,-rpath,{mlx_dir}/lib",
    ],
    deps = [":libmlx_import"],
)
"""

# (file name, rpath that finds it from the installed plugin dylib)
_PRIVATE = ("libmlx_metaljax.dylib", "@loader_path/mlx/lib")
_PUBLIC = ("libmlx.dylib", "@loader_path/../../mlx/lib")

def _mlx_repository_impl(ctx):
    mlx_dir = ctx.os.environ.get("METALJAX_MLX_DIR", _DEFAULT_MLX_DIR)
    dylib, relative_rpath = _PRIVATE
    if not ctx.path(mlx_dir + "/lib/" + dylib).exists:
        dylib, relative_rpath = _PUBLIC
        if not ctx.path(mlx_dir + "/lib/" + dylib).exists:
            fail(("MLX not found at {}: neither lib/{} (the vendored build -- " +
                  "run scripts/vendor_mlx.sh) nor lib/{} (a pip wheel's mlx). " +
                  "Set METALJAX_MLX_DIR to point elsewhere.").format(
                mlx_dir,
                _PRIVATE[0],
                _PUBLIC[0],
            ))
    ctx.symlink(mlx_dir + "/include", "include")
    ctx.symlink(mlx_dir + "/lib", "lib")
    ctx.file("BUILD", _BUILD.format(
        dylib = dylib,
        relative_rpath = relative_rpath,
        mlx_dir = mlx_dir,
    ))

mlx_repository = repository_rule(
    implementation = _mlx_repository_impl,
    environ = ["METALJAX_MLX_DIR"],
    local = True,
)
