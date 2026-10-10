"""Show two equivalent calculations with different equation orders.

Run with: uv run python examples/inspect_jaxpr.py
"""

import jax
import jax.numpy as jnp

from jax_reorder import __version__, canonicalize, compare_jaxprs


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
    left = jax.make_jaxpr(first)(x, y)
    print(left)
    print("Second trace:")
    right = jax.make_jaxpr(second)(x, y)
    print(right)
    for label, graph in (("First", left), ("Second", right)):
        result = canonicalize(graph)
        print(f"{label}: BFS={result.bfs_order}, equation order={result.equation_order}")
        print(result.jaxpr)
    print(f"Graph comparison: {compare_jaxprs(left, right).status.value}")


if __name__ == "__main__":
    main()
