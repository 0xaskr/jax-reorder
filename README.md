# jax-reorder

[English](README.md) | [简体中文](README.zh.md)

Compare AI-generated or rewritten JAX code with the original through structural canonicalization of jaxprs.

The first round of graph comparison is implemented: assign IDs using reverse BFS from outputs, then reorder independent equations by dependency depth. Operand positions, parameters, constants, dtypes, shapes, sharing, and nested regions are preserved. Commutativity, associativity, and other algebraic transformations are not applied.

## Developer Quickstart

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The locked environment uses JAX/jaxlib 0.11.2 on CPU.

```bash
git clone https://github.com/0xaskr/jax-reorder.git
cd jax-reorder
uv sync --locked

# Inspect the original jaxprs, BFS IDs, and reordered equations
uv run python examples/inspect_jaxpr.py

# Check graph comparisons, reordered execution, and numerical counterexamples
uv run python examples/check_correctness.py

# Replay AI-authored classic examples and SGLang-JAX LlamaMLP variants
uv run python examples/bfs_reordering.py
```

`uv sync` creates a local `.venv` and installs development dependencies. If the default cache is not writable, set `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`.

The [manual check script](examples/check_correctness.py) accepts `--show-jaxpr` to display
original and reordered graphs, and `--trials 100 --seed 42` to adjust numerical samples.
Edit `original`, the candidate functions, and `make_inputs` to try your own computations;
`CASES` records expected labels. Failed checks return a nonzero exit code.

## Usage

```python
import jax
import jax.numpy as jnp
from jax_reorder import compare_jaxprs


def original(x, y):
    a = jnp.sin(x)
    b = jnp.cos(y)
    return a + b


def candidate(x, y):
    b = jnp.cos(y)
    a = jnp.sin(x)
    return a + b


signature = jax.ShapeDtypeStruct((4,), jnp.float32)
left = jax.make_jaxpr(original)(signature, signature)
right = jax.make_jaxpr(candidate)(signature, signature)
result = compare_jaxprs(left, right)
print(result.status.value)  # equivalent
```

Results are `equivalent` (matching graphs), `different` (different graphs), or `unknown` (inconclusive). `different` includes a difference location, but does not prove mathematical inequivalence. Unused branches or unsupported semantics produce `unknown` with a reason.

Use `canonicalize(jaxpr)` to obtain the reordered jaxpr, canonical structure, IDs, depths, and variable references. Callers must keep tracing contexts consistent and check function input/output pytrees; this interface accepts flattened jaxprs only.

## Validation and Documentation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python tests/lax_inventory.py
```

Tests cover public lax primitives, independent branch combinations, nested graphs, 19 dtypes and several shapes, and 18 saved AI-labelled examples with numerical evidence. Collective tests default to two logical CPU devices. Finite samples neither restrict the legal parameter domain nor exhaust all combinations.

- [Architecture and comparison contract](ARCHITECTURE.md)
- [BFS design and review](docs/development-docs/bfs-reordering.md)
- [Primitive coverage and CPU limitations](docs/development-docs/lax-coverage.md)
- [AI examples and model provenance](docs/development-docs/bfs-cases.md)
- [Documentation index](docs/index.md)
