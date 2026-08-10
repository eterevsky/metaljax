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
