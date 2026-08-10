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
    linkopts = ["-Wl,-rpath,{mlx_dir}/lib"],
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
