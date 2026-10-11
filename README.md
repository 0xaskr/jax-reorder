# jax-reorder

[English](README.md) | [简体中文](README.zh.md)

Compare AI-generated or rewritten JAX code with the original through structural canonicalization of jaxprs.

The first round of graph comparison is implemented: assign IDs using reverse BFS from outputs, then reorder independent equations by dependency depth. Ordinary operator operand positions, parameters, constants, dtypes, shapes, sharing, and nested regions are preserved. Commutativity, associativity, and other algebraic transformations are not applied.

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

# Multi-device communication: reordered collectives, parameter changes, and sharded matmul
uv run python examples/collective_jaxpr.py

# Replay AI-authored classic examples and SGLang-JAX LlamaMLP variants
uv run python examples/bfs_reordering.py

# Replay hardcoded, AI-authored business-layer rewrites and save numerical witnesses
uv run python examples/business_layer_cases.py --report /tmp/business-layer-cases.json

# Check reordered cond captures and incorrect argument bindings
uv run python examples/cond_bindings.py --show-jaxpr

# Replay AI rewrites of selected layers across all registered SGLang-JAX models
uv run python -m examples.sglang_layers.runner --report /tmp/sglang-layers.json
```

`uv sync` creates a local `.venv` and installs development dependencies. If the default cache is not writable, set `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`.

The [manual check script](examples/check_correctness.py) accepts `--show-jaxpr` to display
original and reordered graphs, and `--trials 100 --seed 42` to adjust numerical samples.
Edit `original`, the candidate functions, and `make_inputs` to try your own computations;
`CASES` records expected labels. Failed checks return a nonzero exit code.

The [communication example](examples/collective_jaxpr.py) runs 22 combinations on two logical
CPU devices by default. It accepts `--devices 4`, `--trials 10 --seed 42`, and `--show-jaxpr`.
It checks graph judgments separately from execution before and after reordering on both sides,
and prints numerical counterexamples for changed parameters or omitted reductions. Validation
covers collectives within one CPU process; see the [communication record](docs/development-docs/collective-jaxpr.md)
for the precise boundaries.

The [business cases script](examples/business_layer_cases.py) contains five bases—gated RMSNorm,
RoPE, causal attention, KV cache updates, and an MoE router—and 20 independently written
candidate functions. AI predictions, reasons, and candidate source are hardcoded; `--seed`
and `--trials` change input samples only. Use `--scenario causal_attention --show-jaxpr` to
inspect one scenario. Each case checks an independent NumPy reference, the graph verdict,
and execution before and after canonicalization. See the [sources and case record](docs/development-docs/business-layer-cases.md).

The [SGLang-JAX model-layer tests](docs/development-docs/sglang-model-layers.md) select 36 real layers across all registration sources at a pinned upstream revision, executing original source against hardcoded AI positive and negative candidates. They cover FP32, BF16, and different shapes, and check each model's construction or inheritance path even when aliases share a layer. Use `--layer llama --profile bf16 --show-jaxpr` to inspect one case; see the [model-to-layer inventory](docs/references/sglang-model-coverage.md) for the full mapping and scope.

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

For `cond` / `lax.switch`, internal arguments and all branch parameters are permuted together, while external function argument order stays fixed. This recognizes closure capture changes caused by reordered independent computations and still detects incorrect bindings. Multiple operands unused by every branch currently produce `unknown`. See the [binding rules and regression tests](docs/development-docs/cond-bindings.md).

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
- [Communication combinations and multi-device validation](docs/development-docs/collective-jaxpr.md)
- [Business-layer bases and fixed AI rewrites](docs/development-docs/business-layer-cases.md)
- [Documentation index](docs/index.md)
