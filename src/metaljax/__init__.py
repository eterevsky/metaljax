"""metaljax: a Metal backend for JAX (native PJRT plugin).

The package carries the native plugin and its private Metal runtime under
`metaljax/lib/`, and nothing else: the Stage-1 Python engine -- the
StableHLO interpreter, tape, msl_scan codegen and op handlers -- was
retired in 0.11.6 (the engine is `plugin-native/`, C++).
`jax_plugins.metal` locates `lib/` with find_spec and never imports this
package, so nothing here is on any hot path; `__version__` is kept because
it is a published surface.

The MLX environment defaults (MLX_MAX_OPS_PER_BUFFER, MLX_MAX_MB_PER_BUFFER,
MLX_METAL_GPU_ARCH) that used to be set here are owned by the plugin itself:
plugin-native/metal/metal_client.cc, which carries the measurements behind
the numbers.
"""

__version__ = "0.11.6.dev0"
__all__ = ["__version__"]
