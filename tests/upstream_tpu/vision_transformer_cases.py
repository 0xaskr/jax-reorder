"""Original ViT MLP block with fixed weights and independent AI implementations."""

import typing
from functools import cache

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax import lax

from tests.upstream_tpu.sources import load_definitions


@cache
def layer_type():
    # Preserve the actual upstream typing names rather than rewriting its annotations.
    namespace = {name: vars(typing)[name] for name in ("Any", "Callable", "Optional", "Tuple")}
    namespace.update(nn=nn, jnp=jnp)
    return load_definitions(
        "vision_transformer",
        "model",
        ("Array", "PRNGKey", "Shape", "Dtype", "MlpBlock"),
        namespace,
    )["MlpBlock"]


def original(x, w1, b1, w2, b2):
    model = layer_type()(mlp_dim=w1.shape[-1], out_dim=w2.shape[-1])
    params = {"Dense_0": {"kernel": w1, "bias": b1}, "Dense_1": {"kernel": w2, "bias": b2}}
    return model.apply({"params": params}, x, deterministic=True)


def reordered(x, w1, b1, w2, b2):
    """Positive: prepare both independent bias broadcasts before the contractions."""
    bias1 = b1.reshape((1,) * (x.ndim - 1) + (b1.shape[0],))
    bias2 = b2.reshape((1,) * (x.ndim - 1) + (b2.shape[0],))
    dimensions = (((x.ndim - 1,), (0,)), ((), ()))
    hidden = nn.gelu(lax.dot_general(x, w1, dimensions) + bias1)
    return lax.dot_general(hidden, w2, dimensions) + bias2


def split_output_bias(x, w1, b1, w2, b2):
    """Positive over ideal reals: split the output bias into two equal terms."""
    hidden = nn.gelu(jnp.dot(x, w1) + b1)
    return (jnp.dot(hidden, w2) + b2 * 0.5) + b2 * 0.5


def wrong_output_bias(x, w1, b1, w2, b2):
    """Negative: subtract the output bias, keeping all inputs and output shapes."""
    hidden = nn.gelu(jnp.dot(x, w1) + b1)
    return jnp.dot(hidden, w2) - b2


def samples(seed):
    rng = np.random.default_rng(seed)
    shapes = ((2, 3, 4), (4, 6), (6,), (6, 4), (4,))
    return tuple(rng.normal(scale=0.4, size=shape).astype(np.float32) for shape in shapes)


def reference(x, w1, b1, w2, b2):
    x, w1, b1, w2, b2 = [np.asarray(value, np.float64) for value in (x, w1, b1, w2, b2)]
    hidden = x @ w1 + b1
    # Flax/JAX's default GELU uses the tanh approximation, not scipy's exact erf.
    hidden = 0.5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden**3)))
    return hidden @ w2 + b2
