"""Show two equivalent calculations with different equation orders.

Run with: uv run python examples/inspect_jaxpr.py
"""

import jax
import jax.numpy as jnp

from jax_reorder import __version__


def first(x, y):
    a = jnp.sin(x)
    b = jnp.cos(y)
    return a + b


def second(x, y):
    b = jnp.cos(y)
    a = jnp.sin(x)
    return a + b


def main():
    x = jax.ShapeDtypeStruct((4,), jnp.float32)
    y = jax.ShapeDtypeStruct((4,), jnp.float32)
    print(f"jax-reorder {__version__}; JAX {jax.__version__}")
    print("First trace:")
    print(jax.make_jaxpr(first)(x, y))
    print("Second trace:")
    print(jax.make_jaxpr(second)(x, y))
    print("Canonicalization is the next implementation milestone.")


if __name__ == "__main__":
    main()
