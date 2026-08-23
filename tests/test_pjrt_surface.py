"""PJRT-boundary behaviour that is not about computing anything:
unsafe_buffer_pointer's aliasing contract, and compile-option validation.
Both run through the real backend.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp


def _metal():
    return jax.devices("metal")[0]


# --- unsafe_buffer_pointer --------------------------------------------


@pytest.mark.parametrize(
    "dtype", ["float32", "bfloat16", "float16", "int8", "complex64", "bool"])
def test_buffer_pointer_of_broadcast(dtype):
    """jnp.ones is a broadcast: stride 0, and no contiguous memoryview.
    The pointer must still exist and be stable across calls."""
    with jax.default_device(_metal()):
        x = jnp.ones(10, dtype=dtype)
        p = x.unsafe_buffer_pointer()
        assert p != 0
        assert x.unsafe_buffer_pointer() == p


def test_buffer_pointer_distinguishes_views_from_copies():
    with jax.default_device(_metal()):
        x = jnp.ones(10)
        ptr = lambda a: a.unsafe_buffer_pointer()
        # asarray of a jax array is the same array
        assert ptr(jnp.asarray(x)) == ptr(x)
        # a jitted identity and any copy must own fresh buffers
        assert ptr(jax.jit(jnp.asarray)(x)) != ptr(x)
        assert ptr(jnp.copy(x)) != ptr(x)
        assert ptr(jax.jit(jnp.copy)(x)) != ptr(x)


def test_buffer_pointer_of_strided_result():
    with jax.default_device(_metal()):
        x = jax.jit(lambda a: a[::2])(jnp.arange(20.0))
        assert x.unsafe_buffer_pointer() == x.unsafe_buffer_pointer()


# --- compiler_options -------------------------------------------------


def test_compile_options_accepts_xla_flags():
    with jax.default_device(_metal()):
        f = lambda x: x * 2.0 + 1.0
        jax.jit(f, compiler_options={
            "xla_embed_ir_in_executable": True,
            "xla_dump_max_hlo_modules": 200,
            "xla_gpu_auto_spmd_partitioning_memory_budget_ratio": 0.5,
            # A string-valued option must not be mistaken for a bad bool.
            "xla_dump_to": "",
        })(1.0)


def test_compile_options_rejects_unknown_name():
    with jax.default_device(_metal()):
        with pytest.raises(jax.errors.JaxRuntimeError,
                           match="No such compile option: 'invalid_key'"):
            jax.jit(lambda x: x * 2.0,
                    compiler_options={"invalid_key": "invalid_value"})(1.0)


def test_compile_options_rejects_bad_bool():
    with jax.default_device(_metal()):
        with pytest.raises(jax.errors.JaxRuntimeError,
                           match="is not a valid bool value."):
            jax.jit(lambda x: x * 2.0, compiler_options={
                "xla_embed_ir_in_executable": "invalid_value"})(1.0)


def test_effort_levels_are_not_env_overrides():
    """jax applies these on ExecutableBuildOptions itself, so they must
    never reach (and be refused by) the option validator."""
    from jax._src import config as jconfig
    with jax.default_device(_metal()):
        jax.jit(lambda x: x * 2.0, compiler_options={
            "exec_time_optimization_effort": 0.0})(1.0)
        jax.jit(lambda x: x * 2.0, compiler_options={
            "optimization_level": jconfig.EffortLevel.O1.value})(1.0)
