"""Pinned JAX activations and genuine TPU Pallas matmul, with fixed AI candidates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core

from jax_reorder import (
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_jaxprs,
)
from tests.pallas_oracle import run_worker
from tests.upstream_tpu import jax_cases as cases
from tests.upstream_tpu.sources import read_manifest, read_source

pytestmark = pytest.mark.business


def test_pinned_sources_and_complete_native_launcher():
    manifest = read_manifest("jax")
    assert manifest["revision"] == cases.REVISION
    for key in manifest["files"]:
        read_source("jax", key)
    assert "def squareplus(" in read_source("jax", "nn")
    assert "scratch_shapes=[pltpu.VMEM" in read_source("jax", "matmul")
    assert "Apache License" in read_source("jax", "license")


@pytest.mark.parametrize("seed", [7, 23])
def test_squareplus_independent_numerical_oracle(seed):
    values = cases.squareplus_samples(seed)
    expected = cases.squareplus_reference(values)
    for function in (cases.squareplus_original(), cases.squareplus_half):
        np.testing.assert_allclose(function(values), expected, rtol=3e-6, atol=3e-6)
    assert not np.allclose(cases.squareplus_wrong_scale(values), expected)


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (cases.squareplus_half, Status.EQUIVALENT),
        (cases.squareplus_wrong_scale, Status.INEQUIVALENT),
    ],
)
def test_squareplus_structural_and_mathematical_results(candidate, expected):
    signature = jax.ShapeDtypeStruct((16,), jnp.float32)
    original = cases.squareplus_original()
    assert (
        compare_functions(original, candidate, signature, equivalence="structural").status
        == Status.DIFFERENT
    )
    result = compare_functions(original, candidate, signature, algebra_timeout=20)
    assert result.status == expected
    assert result.modes[0].algebra.warnings
    if expected == Status.EQUIVALENT:
        assert result.equivalence_kind == "algebraic"


def _pallas_calls(value):
    if isinstance(value, core.ClosedJaxpr):
        value = value.jaxpr
    if isinstance(value, core.Jaxpr):
        for equation in value.eqns:
            if equation.primitive.name == "pallas_call":
                yield equation
            yield from _pallas_calls(equation.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _pallas_calls(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _pallas_calls(child)


def _matmul_graph(variant):
    graph = jax.make_jaxpr(cases.matmul_function(variant))(*cases.matmul_signatures())
    (call,) = tuple(_pallas_calls(graph))
    assert call.params["interpret"] is False
    mapping = call.params["grid_mapping"]
    assert tuple(mapping.grid) == (2, 2, 2)
    assert mapping.num_scratch_operands == 1
    return graph


@pytest.mark.parametrize(
    "variant,expected", [("reordered", Status.EQUIVALENT), ("overwrite", Status.INEQUIVALENT)]
)
@pytest.mark.parametrize("scope", ["function", "device_local"])
def test_real_matmul_complete_output_comparison(variant, expected, scope):
    left, right = _matmul_graph("original"), _matmul_graph(variant)
    assert compare_jaxprs(left, right, include_derivative_rules=False).status == Status.DIFFERENT
    result = compare_algebraic_jaxprs(left, right, scope=scope, algebra_timeout=30)
    assert result.status == expected
    assert result.scope == scope
    assert result.backend_result is not None and result.warnings
    if expected == Status.EQUIVALENT:
        assert result.equivalence_kind == "algebraic"
    else:
        assert any(query.answer is False for query in result.backend_result.queries)


@pytest.mark.parametrize("scope", ["function", "device_local"])
def test_missing_scratch_initialization_is_rejected(scope):
    left, right = _matmul_graph("original"), _matmul_graph("missing_init")
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        compare_algebraic_jaxprs(left, right, scope=scope, algebra_timeout=30)


@pytest.mark.pallas_communication
def test_official_tpu_interpreter_against_numpy_and_wrong_tile_witness():
    run_worker(
        "tests.upstream_tpu.jax_cases",
        entry="interpreter_evidence",
        expected={"case": "jax_matmul", "status": "ok", "races": False},
        startup_timeout=120,
        execution_timeout=90,
    )
