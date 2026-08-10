"""Makes the MLX C++ library (headers + libmlx.dylib) available to Bazel.

MLX ships a full C++ SDK inside its Python wheel: `<site-packages>/mlx/include`
and `<site-packages>/mlx/lib/libmlx.dylib`.  We consume that copy directly so
the native plugin is guaranteed to be ABI-identical to the mlx the Python side
loads in the same process.

Override the location with METALJAX_MLX_DIR if the venv moves.
"""

_DEFAULT_MLX_DIR = "/Users/oleg/metaljax/.venv/lib/python3.14/site-packages/mlx"

_BUILD = """\
package(default_visibility = ["//visibility:public"])

cc_import(
    name = "libmlx_import",
    shared_library = "lib/libmlx.dylib",
)

cc_library(
    name = "mlx",
    hdrs = glob([
        "include/mlx/**/*.h",
        "include/mlx/**/*.hpp",
    ]),
    includes = ["include"],
    # libmlx.dylib's install name is @rpath/libmlx.dylib (and it pulls
    # @rpath/libjaccl.dylib from the same directory), so every binary that
    # links it needs a run-path that finds it.  Two are baked in:
    #
    #   * @loader_path/../../mlx/lib -- the *wheel* layout, where the plugin
    #     sits at site-packages/metaljax/lib/ and mlx at site-packages/mlx/lib/.
    #     Resolved relative to the dylib itself, so it follows the consumer's
    #     venv wherever that is;
    #   * the absolute path of the *build* venv -- what bazel-bin binaries
    #     (smoke_test.py, any cc_test) load, from any process, even one that
    #     has not imported mlx.
    #
    # ORDER IS LOAD-BEARING.  dyld walks LC_RPATHs in order and takes the
    # first hit, and the build venv's absolute path still exists on the
    # build machine -- with it first, a wheel installed into another venv on
    # this machine silently loaded *the build venv's* libmlx (observed with
    # DYLD_PRINT_LIBRARIES: a 3.13 venv running the 3.14 venv's mlx).  The
    # consumer's own mlx must win; the absolute path is only a fallback for
    # bazel-bin, where the relative one misses.
    linkopts = [
        "-Wl,-rpath,@loader_path/../../mlx/lib",
        "-Wl,-rpath,{mlx_dir}/lib",
    ],
    deps = [":libmlx_import"],
)
"""

def _mlx_repository_impl(ctx):
    mlx_dir = ctx.os.environ.get("METALJAX_MLX_DIR", _DEFAULT_MLX_DIR)
    if not ctx.path(mlx_dir + "/lib/libmlx.dylib").exists:
        fail("MLX not found at {} (set METALJAX_MLX_DIR)".format(mlx_dir))
    ctx.symlink(mlx_dir + "/include", "include")
    ctx.symlink(mlx_dir + "/lib", "lib")
    ctx.file("BUILD", _BUILD.format(mlx_dir = mlx_dir))

mlx_repository = repository_rule(
    implementation = _mlx_repository_impl,
    environ = ["METALJAX_MLX_DIR"],
    local = True,
)
