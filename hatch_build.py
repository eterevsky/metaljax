"""Hatch build hook: put the PJRT plugin dylib into the wheel.

Two plugins exist and the wheel carries exactly one of them; which one is
chosen by the **build-time** environment variable `METALJAX_WHEEL_PLUGIN`:

  unset / anything else  the Stage 1 trampoline (`plugin/metal_pjrt.cc`),
                         compiled here with clang.  This is the released
                         wheel and this path must stay exactly as it was.
  "native"               the Stage 2 fully-native plugin
                         (`plugin-native/`, an xla::PjRtClient subclass),
                         built with bazel.  ~160 MB of statically-linked
                         LLVM/MLIR/absl; experimental.

The trampoline is a plain dylib (not a Python extension module) built
against CPython's limited API (>=3.12), and the native plugin embeds no
CPython at all, so in both cases a single wheel tagged
py3-none-macosx_14_0_arm64 serves every supported interpreter. Building the
trampoline from sdist requires the Xcode command-line tools (clang++); the
native variant additionally requires bazel and the `plugin-native/`
workspace, which the sdist does not ship.

Since the vendoring milestone the native variant also CARRIES ITS OWN METAL
RUNTIME: our patched MLX build (`scripts/vendor_mlx.sh` ->
`src/metaljax/lib/mlx/lib/`), renamed to a private install name and placed
inside the wheel at `metaljax/lib/mlx/lib/`, which is where the plugin's
`@loader_path/mlx/lib` run-path looks.  So the native wheel runs in a venv
with no `mlx` installed at all, and in one that has the public `mlx` the two
libraries coexist instead of colliding.  The default (trampoline) wheel is
untouched: Stage 1's Python engine imports the public `mlx`, and nothing
about that changes here.
"""

import os
import pathlib
import shutil
import subprocess
import sysconfig

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

TRAMPOLINE_DYLIB = "libmetal_pjrt.dylib"
NATIVE_DYLIB = "libmetal_pjrt_native.dylib"
# Where the trampoline dylib is parked while a native wheel is built, so the
# native wheel cannot pick it up (the `artifacts` glob in pyproject.toml is
# `src/metaljax/lib/*.dylib`) and so the repo's own dev setup -- which loads
# this very file -- is restored afterwards. Not a *.dylib name on purpose.
_PARKED = TRAMPOLINE_DYLIB + ".parked"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        root = pathlib.Path(self.root)
        libdir = root / "src" / "metaljax" / "lib"
        libdir.mkdir(parents=True, exist_ok=True)
        native = os.environ.get("METALJAX_WHEEL_PLUGIN") == "native"

        if native:
            # Keep the trampoline out of this wheel, but do not destroy it:
            # src/metaljax/lib/ is also what an editable install of this repo
            # loads from.  finalize() puts it back.
            tramp = libdir / TRAMPOLINE_DYLIB
            if tramp.exists():
                tramp.replace(libdir / _PARKED)
            self._build_native(root, libdir / NATIVE_DYLIB)
            # The vendored Metal runtime rides along.  force_include rather
            # than the `artifacts` glob in pyproject.toml on purpose: the
            # glob applies to BOTH variants and the default wheel must stay
            # a 288 KB trampoline, so which files the native wheel adds is
            # decided here, per build, from the same env var.
            build_data.setdefault("force_include", {}).update(
                self._vendored_mlx(libdir))
        else:
            (libdir / NATIVE_DYLIB).unlink(missing_ok=True)
            self._build_trampoline(root, libdir / TRAMPOLINE_DYLIB)

        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-macosx_14_0_arm64"

    def finalize(self, version, build_data, artifact_path):
        """Undo the native variant's staging: the repo tree goes back to
        carrying the trampoline dylib (and only that), which is what a
        dev/editable checkout expects to load."""
        if self.target_name != "wheel":
            return
        if os.environ.get("METALJAX_WHEEL_PLUGIN") != "native":
            return
        libdir = pathlib.Path(self.root) / "src" / "metaljax" / "lib"
        _unlink_forced(libdir / NATIVE_DYLIB)
        parked = libdir / _PARKED
        if parked.exists():
            parked.replace(libdir / TRAMPOLINE_DYLIB)

    # -- the vendored Metal runtime ---------------------------------------

    # What the native wheel carries beside the plugin.  `mlx.metallib` is
    # located by MLX at run time by dladdr'ing itself and looking in its OWN
    # directory, so it must sit next to the dylib in every layout, wheel
    # included.  `include/` is deliberately NOT shipped: it is a build input
    # (~4 MB of C++ headers), not a runtime one.
    # (path under src/metaljax/lib/mlx/, path under metaljax/ in the wheel)
    _MLX_RUNTIME = (
        ("lib/libmlx_metaljax.dylib", "lib/mlx/lib/libmlx_metaljax.dylib"),
        ("lib/libjaccl_metaljax.dylib", "lib/mlx/lib/libjaccl_metaljax.dylib"),
        ("lib/mlx.metallib", "lib/mlx/lib/mlx.metallib"),
        ("VENDOR_STAMP", "lib/mlx/VENDOR_STAMP"),
    )

    def _vendored_mlx(self, libdir):
        """{staged file -> path inside the wheel} for the vendored runtime."""
        # METALJAX_VENDORED_MLX lets the runtime live outside the source
        # tree being packaged, which is what a staged native-only build
        # needs: the 183 MB tree stays in the repo and is force_included
        # from there, so it is never copied and never walked twice.
        env = os.environ.get("METALJAX_VENDORED_MLX")
        staged = pathlib.Path(env) if env else libdir / "mlx"
        files = {}
        for rel, dst in self._MLX_RUNTIME:
            src = staged / rel
            if not src.exists():
                raise RuntimeError(
                    f"METALJAX_WHEEL_PLUGIN=native, but the vendored MLX "
                    f"runtime is missing ({src}). Run scripts/vendor_mlx.sh "
                    f"-- the native plugin links libmlx_metaljax.dylib and "
                    f"the wheel has to carry it.")
            files[str(src)] = f"metaljax/{dst}"
        return files

    # -- variants ---------------------------------------------------------

    def _build_trampoline(self, root, out):
        _unlink_forced(out)
        subprocess.run(
            [
                "clang++", "-std=c++20", "-O2", "-fPIC", "-shared",
                "-Wno-deprecated-declarations",
                "-mmacosx-version-min=14.0",
                "-undefined", "dynamic_lookup",
                "-I", str(root / "plugin" / "vendor"),
                "-I", sysconfig.get_paths()["include"],
                str(root / "plugin" / "metal_pjrt.cc"),
                "-o", str(out),
            ],
            check=True,
        )

    def _build_native(self, root, out):
        # A PREBUILT plugin may be named instead of rebuilding.  Release
        # rule 1 is the reason: the numbers in a release table come from one
        # gated binary, and a wheel that re-runs bazel carries whatever the
        # tree produces at packaging time instead.  Pointing this at the
        # frozen, gated dylib makes "the wheel ships the binary that was
        # measured" a build-time fact rather than a hope -- and it lets the
        # wheel be built from a staged source tree that has no bazel
        # workspace in it at all (scripts/build_native_wheel.sh).
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
        workspace = root / "plugin-native"
        if not (workspace / "WORKSPACE").exists():
            raise RuntimeError(
                f"METALJAX_WHEEL_PLUGIN=native, but {workspace} is not a bazel "
                "workspace (the sdist does not ship it -- build the native "
                "wheel from a git checkout)")
        bazel = (os.environ.get("METALJAX_BAZEL")
                 or shutil.which("bazel")
                 or str(pathlib.Path.home() / ".local" / "bin" / "bazel"))
        if not pathlib.Path(bazel).exists():
            raise RuntimeError(
                "METALJAX_WHEEL_PLUGIN=native needs bazel on PATH (or "
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
