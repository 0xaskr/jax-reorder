"""Inventory actual GLM graphs without compiling or evaluating array operations."""

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax._src import compiler, dispatch

from examples import glm52_decode_dsa_mla as layer
from jax_reorder import trace_function


def nested_graphs(value, path="root"):
    if hasattr(value, "eqns"):
        yield path, value
        for index, equation in enumerate(value.eqns):
            for name, child in equation.params.items():
                yield from nested_graphs(child, f"{path}/{index}:{equation.primitive.name}/{name}")
    elif hasattr(value, "jaxpr"):
        yield from nested_graphs(value.jaxpr, path)
    elif isinstance(value, dict):
        for name, child in value.items():
            yield from nested_graphs(child, f"{path}/{name}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from nested_graphs(child, f"{path}/{index}")


def forbidden(*args, **kwargs):
    raise AssertionError("inventory must not compile or evaluate JAX array operations")


def inventory(graph):
    counts = Counter()
    parameters = {}
    regions = []
    for path, child in nested_graphs(graph):
        regions.append({"path": path, "equations": len(child.eqns)})
        for equation in child.eqns:
            name = equation.primitive.name
            counts[name] += 1
            selected = {
                key: repr(value)
                for key, value in equation.params.items()
                if key
                in {
                    "axes",
                    "dimension",
                    "dimension_numbers",
                    "fill_value",
                    "k",
                    "length",
                    "mode",
                    "precision",
                    "recall_target",
                    "reduction_dimension",
                    "reverse",
                    "slice_sizes",
                }
            }
            if selected and selected not in parameters.setdefault(name, []):
                parameters[name].append(selected)
    return {
        "equation_occurrences": counts.total(),
        "primitives": dict(sorted(counts.items())),
        "parameter_examples": parameters,
        "regions": regions,
    }


def run(profile, dtype):
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        args = layer.signatures(profile, dtype)
        traces = {"base": trace_function(layer.function(profile), *args)}
        for case in layer.CASES:
            traces[case.name] = trace_function(layer.function(profile, case), *args)
    graphs = {name: inventory(trace.jaxpr) for name, trace in traces.items()}
    return {
        "jax_version": jax.__version__,
        "source_revision": layer.REVISION,
        "profile": profile,
        "dtype": str(jnp.dtype(dtype)),
        "compiled": False,
        "executed": False,
        "interpretation": "recursive syntactic occurrences, not executed iteration counts",
        "all_primitives": sorted({op for graph in graphs.values() for op in graph["primitives"]}),
        "graphs": graphs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=layer.PROFILES, default="glm")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--json", type=Path)
    options = parser.parse_args()
    report = run(options.profile, jnp.dtype(options.dtype))
    if options.json:
        options.json.parent.mkdir(parents=True, exist_ok=True)
        options.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "profile": report["profile"],
                "graphs": len(report["graphs"]),
                "base_equation_occurrences": report["graphs"]["base"]["equation_occurrences"],
                "primitives": report["all_primitives"],
                "compiled": False,
                "executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
