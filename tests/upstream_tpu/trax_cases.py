"""Pinned Trax LayerNorm.forward, with explicit method state and AI candidates.

The entire original method is selected unchanged. This exercises its computation
under Trax's JAX NumPy backend, not base.Layer initialization or training state.
"""

from functools import cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tests.upstream_tpu.sources import load_nested_definition

REVISION = "31022d6cd7dd525ed11a04d84cd3936228499173"
CASES = (
    ("direct", True, True),
    ("reciprocal_root", False, True),
    ("wrong_scale", False, False),
)


@cache
def original():
    return load_nested_definition("trax", "normalization", "LayerNorm.forward", {"jnp": jnp})


def function(variant=None, *, center=True):
    if variant is None:
        method = original()

        def baseline(value, scale, bias):
            state = SimpleNamespace(weights=(scale, bias), _center=center, _epsilon=0.125)
            return method(state, value)

        return baseline
    if variant not in {row[0] for row in CASES}:
        raise ValueError(variant)

    def candidate(value, scale, bias):
        mean = jnp.mean(value, axis=-1, keepdims=True)
        centered = value - mean if center else value
        variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
        if variant == "reciprocal_root":
            normalized = centered * jax.lax.rsqrt(variance + 0.125)
        else:
            normalized = centered / jnp.sqrt(variance + 0.125)
        scaled = normalized * (-scale if variant == "wrong_scale" else scale)
        return scaled + bias if center else scaled

    return candidate


def inputs(seed, *, shape=(2, 3)):
    rng = np.random.default_rng(seed)
    values = (
        rng.normal(size=shape),
        rng.uniform(0.4, 1.4, size=shape[-1]),
        rng.uniform(-0.6, 0.6, size=shape[-1]),
    )
    return tuple(jnp.asarray(value, dtype=jnp.float32) for value in values)


def reference(args, *, center=True):
    value, scale, bias = (np.asarray(arg, dtype=np.float64) for arg in args)
    centered = value - value.mean(axis=-1, keepdims=True) if center else value
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    scaled = centered / np.sqrt(variance + 0.125) * scale
    return scaled + bias if center else scaled
