"""Pinned MNIST forward, preserving its parameter pytree and original layer sizes."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from tests.upstream_tpu.sources import load_definitions


@cache
def definitions():
    return load_definitions(
        "ai_engineering",
        "mlp",
        ("init_params", "forward"),
        dict(jax=jax, jnp=jnp, random=jax.random),
    )


def original(params, x):
    return definitions()["forward"](params, x)


def reordered(params, x):
    """Positive: schedule the independent bias broadcasts before the first dot."""
    b1 = jax.lax.broadcast_in_dim(params["layer1"]["b"], (1, params["layer1"]["b"].size), (1,))
    b2 = jax.lax.broadcast_in_dim(params["layer2"]["b"], (1, params["layer2"]["b"].size), (1,))
    b3 = jax.lax.broadcast_in_dim(params["layer3"]["b"], (1, params["layer3"]["b"].size), (1,))
    x = jax.nn.relu(jnp.dot(x, params["layer1"]["w"]) + b1)
    x = jax.nn.relu(jnp.dot(x, params["layer2"]["w"]) + b2)
    return jnp.dot(x, params["layer3"]["w"]) + b3


def split_bias(params, x):
    """Positive over ideal reals: split the final layer bias in equal halves."""
    x = jax.nn.relu(jnp.dot(x, params["layer1"]["w"]) + params["layer1"]["b"])
    x = jax.nn.relu(jnp.dot(x, params["layer2"]["w"]) + params["layer2"]["b"])
    x = jnp.dot(x, params["layer3"]["w"])
    return (x + params["layer3"]["b"] * 0.5) + params["layer3"]["b"] * 0.5


def wrong_bias(params, x):
    """Negative: subtract the final bias while retaining every input leaf."""
    x = jax.nn.relu(jnp.dot(x, params["layer1"]["w"]) + params["layer1"]["b"])
    x = jax.nn.relu(jnp.dot(x, params["layer2"]["w"]) + params["layer2"]["b"])
    return jnp.dot(x, params["layer3"]["w"]) - params["layer3"]["b"]


def samples(seed, batch=2):
    rng = np.random.default_rng(seed)
    widths = (784, 256, 128, 10)
    params = {
        f"layer{index}": {
            "w": rng.normal(scale=np.sqrt(2 / source), size=(source, target)).astype(np.float32),
            "b": rng.uniform(-0.25, 0.25, size=target).astype(np.float32),
        }
        for index, (source, target) in enumerate(zip(widths, widths[1:], strict=False), 1)
    }
    # Nonzero trained-style biases ensure the wrong-bias candidate has a witness;
    # the unmodified upstream initializer is exercised separately with zero biases.
    params["layer3"]["b"] = np.linspace(-0.5, 0.5, 10, dtype=np.float32)
    return params, rng.uniform(size=(batch, 784)).astype(np.float32)


def reference(params, x):
    result = np.asarray(x, dtype=np.float64)
    for index in (1, 2, 3):
        layer = params[f"layer{index}"]
        result = result @ np.asarray(layer["w"], np.float64) + np.asarray(layer["b"], np.float64)
        if index < 3:
            result = np.maximum(result, 0)
    return result
