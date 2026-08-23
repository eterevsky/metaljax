"""Hatch build hook: put the native PJRT plugin and its Metal runtime into
the wheel.

The plugin (`plugin-native/`, an xla::PjRtClient subclass built with bazel)
is a single dylib; the wheel also CARRIES ITS OWN METAL RUNTIME: our patched
MLX build (`scripts/vendor_mlx.sh` -> `src/metaljax/lib/mlx/lib/`), renamed
to a private install name and placed inside the wheel at
`metaljax/lib/mlx/lib/`, which is where the plugin's `@loader_path/mlx/lib`
run-path looks.  So the wheel runs in a venv with no `mlx` installed at all,
and in one that has the public `mlx` the two libraries coexist instead of
colliding.

The plugin embeds no CPython, so one wheel tagged
py3-none-macosx_14_0_arm64 serves every supported interpreter.

Release rule 1 (every number in a release table comes from the release
binary) is why a PREBUILT dylib may be named via METALJAX_NATIVE_DYLIB
instead of rebuilding: pointing this at the frozen, gated binary makes "the
wheel ships the binary that was measured" a build-time fact rather than a
hope.  Without it, bazel builds from `plugin-native/` -- a dev-only path.
"""

import os
import pathlib
import re
import shutil
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

NATIVE_DYLIB = "libmetal_pjrt_native.dylib"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        root = pathlib.Path(self.root)
        self._check_version(root)
        libdir = root / "src" / "metaljax" / "lib"
        libdir.mkdir(parents=True, exist_ok=True)
        self._place_dylib(root, libdir / NATIVE_DYLIB)
        # The vendored Metal runtime rides along.  force_include rather than
        # an `artifacts` glob in pyproject.toml: which files ship is decided
        # here, per build, next to the code that requires them.
        build_data.setdefault("force_include", {}).update(
            self._vendored_mlx(libdir))
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-macosx_14_0_arm64"

    # -- version consistency ----------------------------------------------

    def _check_version(self, root):
        """`metaljax.__version__` must equal the pyproject version.  They
        drifted once (the 0.11.3-in-0.11.4-wheel cosmetic bug); refuse to
        build anything mismatched."""
        init = (root / "src" / "metaljax" / "__init__.py").read_text()
        m = re.search(r'^__version__ = "(.*)"$', init, re.M)
        if not m:
            raise RuntimeError("no __version__ in src/metaljax/__init__.py")
        if m.group(1) != self.metadata.version:
            raise RuntimeError(
                f"__version__ ({m.group(1)}) != pyproject version "
                f"({self.metadata.version}) -- bump both, they drifted once")

    # -- the vendored Metal runtime ---------------------------------------

    # What the wheel carries beside the plugin.  `mlx.metallib` is located
    # by MLX at run time by dladdr'ing itself and looking in its OWN
    # directory, so it must sit next to the dylib in every layout, wheel
    # included.  `include/` is deliberately NOT shipped: it is a build input
    # (~4 MB of C++ headers), not a runtime one.
    # (path under the staged mlx dir, path under metaljax/ in the wheel)
    _MLX_RUNTIME = (
        ("lib/libmlx_metaljax.dylib", "lib/mlx/lib/libmlx_metaljax.dylib"),
        ("lib/libjaccl_metaljax.dylib", "lib/mlx/lib/libjaccl_metaljax.dylib"),
        ("lib/mlx.metallib", "lib/mlx/lib/mlx.metallib"),
        ("VENDOR_STAMP", "lib/mlx/VENDOR_STAMP"),
    )

    def _vendored_mlx(self, libdir):
        """{staged file -> path inside the wheel} for the vendored runtime."""
        # METALJAX_VENDORED_MLX lets the runtime live outside the tree being
        # packaged; the default is where scripts/vendor_mlx.sh stages it.
        env = os.environ.get("METALJAX_VENDORED_MLX")
        staged = pathlib.Path(env) if env else libdir / "mlx"
        files = {}
        for rel, dst in self._MLX_RUNTIME:
            src = staged / rel
            if not src.exists():
                raise RuntimeError(
                    f"the vendored MLX runtime is missing ({src}). Run "
                    f"scripts/vendor_mlx.sh -- the plugin links "
                    f"libmlx_metaljax.dylib and the wheel has to carry it.")
            files[str(src)] = f"metaljax/{dst}"
        return files

    # -- the plugin dylib --------------------------------------------------

    def _place_dylib(self, root, out):
        prebuilt = os.environ.get("METALJAX_NATIVE_DYLIB")
        if prebuilt:
            src = pathlib.Path(prebuilt)
            if not src.exists():
                raise RuntimeError(
                    f"METALJAX_NATIVE_DYLIB={prebuilt!r} does not exist")
            _unlink_forced(out)
            shutil.copyfile(src, out)      # not copy2: frozen copies are r-x
            out.chmod(0o755)
            return
        # Reuse a dylib already in place (a dev tree carries one so the
        # editable install works); rebuild only when there is none.
        if out.exists():
            return
        workspace = root / "plugin-native"
        if not (workspace / "WORKSPACE").exists():
            raise RuntimeError(
                f"{workspace} is not a bazel workspace and no prebuilt "
                "dylib was named (METALJAX_NATIVE_DYLIB) or already in "
                "place -- build from a git checkout")
        bazel = (os.environ.get("METALJAX_BAZEL")
                 or shutil.which("bazel")
                 or str(pathlib.Path.home() / ".local" / "bin" / "bazel"))
        if not pathlib.Path(bazel).exists():
            raise RuntimeError(
                "building the plugin needs bazel on PATH (or "
                f"METALJAX_BAZEL=<path>); tried {bazel!r}")
        target = "//metal:" + NATIVE_DYLIB
        subprocess.run([bazel, "build", target], cwd=str(workspace),
                       check=True)
        built = workspace / "bazel-bin" / "metal" / NATIVE_DYLIB
        if not built.exists():
            raise RuntimeError(f"bazel reported success but {built} is missing")
        _unlink_forced(out)
        shutil.copyfile(built, out)      # not copy2: bazel outputs are r-x
        out.chmod(0o755)


def _unlink_forced(path):
    """unlink() a possibly read-only file (bazel outputs are mode 0555, and
    copyfile onto one fails with EACCES)."""
    if path.exists():
        path.chmod(0o644)
        path.unlink()
