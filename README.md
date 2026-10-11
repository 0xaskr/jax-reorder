# jax-reorder

[English](README.md) | [简体中文](README.zh.md)

Compare AI-generated or rewritten JAX code with the original through structural canonicalization of jaxprs.

The implementation supports complete graph comparison, function contracts, and separately traced first-order JVP/VJP modes. It assigns IDs using reverse BFS from outputs and reorders independent equations by dependency depth, synchronizing private region captures, returns, and loop feedback. Ordinary operator operand positions, parameters, constants, dtypes, shapes, sharing, and nested regions are preserved. Commutativity, associativity, and other algebraic transformations are not applied.

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

# Check original complete attention/blocks, derivatives, indexers, and backend boundaries
uv run python -m examples.full_layers.runner --supplemental --report /tmp/full-layers.json
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

Results are `equivalent` (matching graphs), `different` (different graphs), or `unknown` (inconclusive). `different` includes a difference location, but does not prove mathematical inequivalence. Unused computations and constants remain part of the complete graph. Unsupported semantics or exhausted exact-search budgets produce `unknown` with a reason. Every equivalent result carries an independently checked correspondence between the original graphs.

Use `canonicalize(jaxpr)` to obtain the reordered jaxpr, canonical structure, IDs, depths, and variable references. The low-level interface accepts flattened jaxprs, so callers must check function contracts. Alternatively, use the function interface:

```python
from jax_reorder import compare_functions

report = compare_functions(
    original,
    candidate,
    signature,
    signature,
    modes=("forward", "jvp", "vjp"),
    check_cpu=True,
)
for mode in report.modes:
    print(mode.mode, mode.status.value, mode.difference, mode.failure_kind)
```

The function interface checks input/output pytrees, static arguments, abstract signatures, and tracing context. JVP/VJP use full dynamic tangent/cotangent inputs; unrequested modes are explicitly unverified. Low-level `compare_jaxprs` materializes and compares registered first-order `custom_jvp_call` / `custom_vjp_call` rules by default, while the function interface compares the actual requested transformed programs. Matching forward graphs do not imply matching derivative graphs: a different gradient accumulation order produces a VJP graph difference. See the [function and derivative contract](docs/development-docs/function-and-derivative-contract.md).

Audited private interfaces for `cond` / `lax.switch`, `jit`, `scan`, `while_loop`, and `shard_map` support synchronized permutations. External inputs/outputs and ordinary operand positions stay fixed. Ambiguous unused slots and unrooted subgraphs use bounded exact search with a default `search_budget=10000`; exhaustion yields `unknown`. See [region bindings](docs/development-docs/region-bindings.md) and [complete graphs](docs/development-docs/complete-graph.md).

Original GLM5 MHA, Llama GQA, CLIP encoder, and Flux joint attention have complete-layer positive and negative cases. Pallas kernels, CPU rejections, and unverified configurations are listed separately in [complete-layer coverage](docs/development-docs/full-attention.md). Conclusions apply to the declared signature, pure-function assumption, locked CPU environment, and requested modes; they do not prove general Python equivalence or bitwise floating-point equality across compilation environments.

## Validation and Documentation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python tests/lax_inventory.py
uv run python tests/adapter_inventory.py
uv run python tools/check_docs.py --changed
```

Tests cover public lax primitives, independent branch combinations, nested graphs, 19 dtypes and several shapes, and 18 saved AI-labelled examples with numerical evidence. Collective tests default to two logical CPU devices. Finite samples neither restrict the legal parameter domain nor exhaust all combinations.

The [system validation matrix](docs/development-docs/verification-matrix.md) lists additional complete-graph, derivative, region, convolution/dot/linear-algebra, and stress cases. CI separates quick, business, stress, and two/four-device jobs. Unaudited JAX/jaxlib versions yield `unknown`; upgrades are inspected in an isolated workflow.

- [Architecture and comparison contract](ARCHITECTURE.md)
- [BFS design and review](docs/development-docs/bfs-reordering.md)
- [Primitive coverage and CPU limitations](docs/development-docs/lax-coverage.md)
- [AI examples and model provenance](docs/development-docs/bfs-cases.md)
- [Communication combinations and multi-device validation](docs/development-docs/collective-jaxpr.md)
- [Business-layer bases and fixed AI rewrites](docs/development-docs/business-layer-cases.md)
- [Documentation index](docs/index.md)
