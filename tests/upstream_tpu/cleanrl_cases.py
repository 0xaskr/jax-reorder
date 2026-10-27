"""Pinned CleanRL DDPG Actor, including its original 256-wide hidden layers."""

from functools import cache

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax import lax

from tests.upstream_tpu.sources import load_definitions


@cache
def actor_type():
    return load_definitions("cleanrl", "ddpg", ("Actor",), dict(nn=nn, jnp=jnp))["Actor"]


def original(x, w1, b1, w2, b2, w3, b3, scale, bias):
    actor = actor_type()(action_dim=w3.shape[-1], action_scale=scale, action_bias=bias)
    params = {
        "Dense_0": {"kernel": w1, "bias": b1},
        "Dense_1": {"kernel": w2, "bias": b2},
        "Dense_2": {"kernel": w3, "bias": b3},
    }
    return actor.apply({"params": params}, x)


def _hidden(x, w1, b1, w2, b2, w3, b3):
    return jnp.tanh(nn.relu(nn.relu(x @ w1 + b1) @ w2 + b2) @ w3 + b3)


def reordered(x, w1, b1, w2, b2, w3, b3, scale, bias):
    """Positive: prepare the three independent Dense bias broadcasts first."""
    b1, b2, b3 = (b.reshape((1, b.shape[0])) for b in (b1, b2, b3))
    dims = (((1,), (0,)), ((), ()))
    hidden = nn.relu(lax.dot_general(x, w1, dims) + b1)
    hidden = nn.relu(lax.dot_general(hidden, w2, dims) + b2)
    return jnp.tanh(lax.dot_general(hidden, w3, dims) + b3) * scale + bias


def split_bias(x, w1, b1, w2, b2, w3, b3, scale, bias):
    """Positive over ideal reals: split the final action bias in equal halves."""
    y = _hidden(x, w1, b1, w2, b2, w3, b3) * scale
    return (y + bias * 0.5) + bias * 0.5


def wrong_bias(x, w1, b1, w2, b2, w3, b3, scale, bias):
    """Negative: reverse the action bias without changing the array interface."""
    return _hidden(x, w1, b1, w2, b2, w3, b3) * scale - bias


def samples(seed, batch=2):
    rng = np.random.default_rng(seed)
    shapes = ((batch, 3), (3, 256), (256,), (256, 256), (256,), (256, 2), (2,))
    args = tuple(rng.normal(scale=0.08, size=shape).astype(np.float32) for shape in shapes)
    return (*args, np.array([1.5, 0.75], np.float32), np.array([0.25, -0.5], np.float32))


def reference(*args):
    x, w1, b1, w2, b2, w3, b3, scale, bias = [np.asarray(a, np.float64) for a in args]
    hidden = np.maximum(x @ w1 + b1, 0)
    hidden = np.maximum(hidden @ w2 + b2, 0)
    return np.tanh(hidden @ w3 + b3) * scale + bias
