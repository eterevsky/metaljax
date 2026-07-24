# Releasing metaljax to PyPI

One wheel covers all supported Pythons: the PJRT dylib is built against
CPython's limited API (>=3.12), so artifacts are
`metaljax-X.Y.Z-py3-none-macosx_14_0_arm64.whl` + an sdist (which
compiles the plugin at install time; needs Xcode CLT).

## 1. Bump the version

- `pyproject.toml` → `[project] version`
- `src/metaljax/__init__.py` → `__version__`

## 2. Build and check

```bash
rm -rf dist
uv build
uvx twine check dist/*
```

The wheel tag must be `py3-none-macosx_14_0_arm64` (the hatch build hook
in `hatch_build.py` compiles the dylib and sets it).

## 3. Smoke-test the built wheel

Install the wheel into a fresh venv on a *different* Python version than
the one that built it (validates the limited-API build):

```bash
uv venv --python 3.12 /tmp/mj-check
uv pip install -p /tmp/mj-check/bin/python dist/*.whl
JAX_PLATFORMS=metal /tmp/mj-check/bin/python -c \
  "import jax, jax.numpy as jnp; print(jax.devices()); print(2 * jnp.array([1,2,3]))"
```

## 4. (Recommended) TestPyPI dry run

```bash
uv publish --index testpypi --token <testpypi-token>
uv venv --python 3.13 /tmp/mj-testpypi
uv pip install -p /tmp/mj-testpypi/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ metaljax
```

(`uv publish --index testpypi` needs `[[tool.uv.index]]` config; the
plain form is `uv publish --publish-url https://test.pypi.org/legacy/`.)

## 5. Publish

```bash
uv publish --token <pypi-token>   # or export UV_PUBLISH_TOKEN
```

Tokens come from https://pypi.org/manage/account/token/ (create the
project-scoped token after the first upload).

## 6. Tag

```bash
git tag v0.1.0
git push origin v0.1.0
```
