# jax-reorder

[English](README.md) | [简体中文](README.zh.md)

Compare AI-generated JAX code with the original through structural canonicalization of jaxprs.

`jax-reorder` aims to check whether JAX code generated or rewritten by AI is equivalent to the original code. The planned workflow traces both versions with the same input signature and static arguments, converts their jaxprs to a canonical structural representation, then compares the results.

## Developer Quickstart

Development uses Python 3.12 and [uv](https://docs.astral.sh/uv/) to manage the environment and dependencies.

```bash
git clone https://github.com/0xaskr/jax-reorder.git
cd jax-reorder

# Install the locked environment, including development tools
uv sync --locked

# Inspect two computations with different equation orders
uv run python examples/inspect_jaxpr.py
```

`uv sync` creates a local `.venv` with CPU JAX and development dependencies. To activate the environment directly, run `source .venv/bin/activate`.

Common checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

`pytest` is included as a development dependency. Add meaningful tests under `tests/` when implementing canonicalization, then enable them in CI.

## Example and Project Navigation

[`examples/inspect_jaxpr.py`](examples/inspect_jaxpr.py) traces two functions that compute `sin(x) + cos(y)` with independent operations in a different order:

```python
import jax.numpy as jnp


def first(x, y):
    a = jnp.sin(x)
    b = jnp.cos(y)
    return a + b


def second(x, y):
    b = jnp.cos(y)
    a = jnp.sin(x)
    return a + b
```

In this example, `first` represents the original implementation and `second` represents a candidate rewrite of the kind AI might produce.
