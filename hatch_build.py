"""Hatch build hook: compile the PJRT plugin dylib into the wheel.

The plugin is a plain dylib (not a Python extension module) built against
CPython's limited API (>=3.12), so a single wheel tagged
py3-none-macosx_14_0_arm64 serves every supported interpreter. Building
from sdist requires the Xcode command-line tools (clang++).
"""

import pathlib
import subprocess
import sysconfig

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        root = pathlib.Path(self.root)
        out = root / "src" / "metaljax" / "lib" / "libmetal_pjrt.dylib"
        out.parent.mkdir(parents=True, exist_ok=True)
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
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-macosx_14_0_arm64"
