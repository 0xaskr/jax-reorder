"""Unmodified DALL·E Mini RMSNorm and fixed AI candidates for that real layer."""

from functools import cache
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tests.upstream_tpu.sources import load_definitions


@cache
def layer_type():
    return load_definitions(
        "dalle_mini", "model", ("RMSNorm",), dict(Any=Any, nn=nn, jax=jax, jnp=jnp, lax=lax)
    )["RMSNorm"]


def original(x, scale):
    return layer_type()().apply({"params": {"scale": scale}}, x)


def reordered(x, scale):
    """Positive: prepare the independent scale broadcast before the statistics."""
    scale = scale.reshape((1,) * (x.ndim - 1) + (x.shape[-1],))
    promoted = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
    mean_square = jnp.mean(lax.square(promoted), (-1,))
    mean_square = mean_square.reshape(x.shape[:-1] + (1,))
    multiplier = lax.rsqrt(mean_square + 1e-6)
    multiplier *= scale
    return jnp.asarray(multiplier * x, jnp.float32)


def reassociated(x, scale):
    """Positive over ideal reals: multiply x and scale before the inverse RMS."""
    mean_square = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    return (x * scale) * lax.rsqrt(mean_square + 1e-6)


def missing_scale(x, scale):
    """Negative: omit the learned scale while preserving the input contract."""
    del scale
    mean_square = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    return x * lax.rsqrt(mean_square + 1e-6)


def samples(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape).astype(np.float32)
    scale = np.linspace(-1.5, 2.0, shape[-1], dtype=np.float32)
    return x, scale


def reference(x, scale):
    x = np.asarray(x, dtype=np.float64)
    return x * np.asarray(scale) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
