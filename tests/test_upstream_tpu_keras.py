"""Actual Keras functions, fixed AI labels, and independent numeric witnesses."""

import json
import subprocess
import sys

import jax
import numpy as np
import pytest
from jax._src import core

from jax_reorder import (
    ComparisonContractError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_jaxprs,
)
from tests.upstream_tpu.keras_cases import (
    ATTENTION_CASES,
    BATCHNORM_CASES,
    REVISION,
    attention_function,
    attention_inputs,
    attention_reference,
    batchnorm_function,
    batchnorm_inputs,
    batchnorm_reference,
    splash_function,
    splash_signatures,
)
from tests.upstream_tpu.sources import read_manifest, read_source


def test_keras_sources_are_pinned_and_tpu_backend_is_real():
    manifest = read_manifest("keras")
    assert manifest["revision"] == REVISION
    for key in manifest["files"]:
        assert read_source("keras", key)
    source = read_source("keras", "nn")
    assert "jax.experimental.pallas.ops.tpu.splash_attention" in source
    assert "jax.default_backend() == 'tpu'" in read_source("keras", "tpu_ci")


@pytest.mark.parametrize("variant,structural_equal,mathematical_equal", BATCHNORM_CASES)
@pytest.mark.parametrize("axis", [-1, 1])
def test_keras_batchnorm_comparison_and_independent_reference(
    variant, structural_equal, mathematical_equal, axis
):
    left = batchnorm_function(axis=axis)
    right = batchnorm_function(variant, axis=axis)
    args = batchnorm_inputs(7, axis=axis)
    graphs = tuple(jax.make_jaxpr(function)(*args) for function in (left, right))
    structure = compare_jaxprs(*graphs, include_derivative_rules=False)
    assert structure.status == (Status.EQUIVALENT if structural_equal else Status.DIFFERENT)
    for seed in (7, 23):
        inputs = batchnorm_inputs(seed, axis=axis)
        original, candidate = np.asarray(left(*inputs)), np.asarray(right(*inputs))
        reference = batchnorm_reference(inputs, axis=axis)
        np.testing.assert_allclose(original, reference, rtol=2e-6, atol=2e-6)
        if mathematical_equal:
            np.testing.assert_allclose(candidate, reference, rtol=2e-6, atol=2e-6)
        else:
            assert np.max(np.abs(candidate - reference)) > 0.1


@pytest.mark.parametrize("variant,structural_equal,mathematical_equal", BATCHNORM_CASES)
@pytest.mark.parametrize("axis", [-1, 1])
def test_keras_batchnorm_mathematical_comparison(
    variant, structural_equal, mathematical_equal, axis
):
    args = batchnorm_inputs(7, axis=axis)
    graphs = tuple(
        jax.make_jaxpr(batchnorm_function(mode, axis=axis))(*args) for mode in (None, variant)
    )
    algebra = compare_algebraic_jaxprs(*graphs, algebra_timeout=20)
    assert algebra.status == (Status.EQUIVALENT if mathematical_equal else Status.INEQUIVALENT)
    if mathematical_equal and not structural_equal:
        assert algebra.warnings


@pytest.mark.parametrize("variant,structural_equal,mathematical_equal", ATTENTION_CASES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_keras_attention_comparison_and_independent_reference(
    variant, structural_equal, mathematical_equal, is_causal
):
    left = attention_function(is_causal=is_causal)
    right = attention_function(variant, is_causal=is_causal)
    args = attention_inputs(7)
    graphs = tuple(jax.make_jaxpr(function)(*args) for function in (left, right))
    structure = compare_jaxprs(*graphs, include_derivative_rules=False)
    assert structure.status == (Status.EQUIVALENT if structural_equal else Status.DIFFERENT)
    for seed in (7, 23):
        inputs = attention_inputs(seed)
        original, candidate = np.asarray(left(*inputs)), np.asarray(right(*inputs))
        reference = attention_reference(inputs, is_causal=is_causal)
        np.testing.assert_allclose(original, reference, rtol=2e-6, atol=2e-6)
        if mathematical_equal:
            np.testing.assert_allclose(candidate, reference, rtol=2e-6, atol=2e-6)
        else:
            assert np.max(np.abs(candidate - reference)) > 1e-4


@pytest.mark.parametrize(
    "variant,structural_equal,mathematical_equal", [case for case in ATTENTION_CASES if case[2]]
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_keras_attention_mathematical_comparison(
    variant, structural_equal, mathematical_equal, is_causal
):
    args = attention_inputs(7)
    graphs = tuple(
        jax.make_jaxpr(attention_function(mode, is_causal=is_causal))(*args)
        for mode in (None, variant)
    )
    algebra = compare_algebraic_jaxprs(*graphs, algebra_timeout=20)
    assert algebra.status == (Status.EQUIVALENT if mathematical_equal else Status.INEQUIVALENT)
    if mathematical_equal and not structural_equal:
        assert algebra.warnings


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("wrong_value", "SymPy did not establish output identity"),
        ("wrong_scale", "original conditions are not established"),
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_keras_attention_negative_math_remains_undecided(variant, reason, is_causal):
    # The separate NumPy checks establish these fixed negatives. This assertion
    # records an incomplete comparison, never a mathematical negative proof.
    args = attention_inputs(7)
    graphs = tuple(
        jax.make_jaxpr(attention_function(mode, is_causal=is_causal))(*args)
        for mode in (None, variant)
    )
    with pytest.raises(MathematicalUndecidedError, match=reason):
        compare_algebraic_jaxprs(*graphs, algebra_timeout=20)


def test_keras_finite_mask_sentinel_preserves_all_masked_rows():
    args = list(attention_inputs(7))
    args[-1] = np.zeros_like(np.asarray(args[-1]))
    actual = np.asarray(attention_function()(*args))
    reference = attention_reference(args)
    expected_mean = np.broadcast_to(np.asarray(args[2]).mean(axis=1, keepdims=True), actual.shape)
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual, expected_mean, rtol=2e-6, atol=2e-6)


def _primitives(graph):
    if isinstance(graph, core.ClosedJaxpr):
        graph = graph.jaxpr
    if isinstance(graph, core.Jaxpr):
        for equation in graph.eqns:
            yield equation.primitive.name
            yield from _primitives(equation.params)
    elif isinstance(graph, dict):
        for value in graph.values():
            yield from _primitives(value)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from _primitives(value)


@pytest.mark.parametrize("variant,equal", [("direct", True), ("wrong_value", False)])
def test_keras_actual_splash_structural_comparison(variant, equal):
    graphs = tuple(
        jax.make_jaxpr(splash_function(mode))(*splash_signatures()) for mode in (None, variant)
    )
    for graph in graphs:
        assert "pallas_call" in set(_primitives(graph))
    result = compare_jaxprs(*graphs, include_derivative_rules=False)
    assert result.status == (Status.EQUIVALENT if equal else Status.DIFFERENT)
    if equal:
        assert compare_algebraic_jaxprs(*graphs).equivalence_kind == "structural"


@pytest.mark.parametrize(
    "scope,error,reason",
    [
        ("function", UnsupportedJaxprError, "outer postprocessing needs output composition"),
        ("device_local", ComparisonContractError, "local inputs, outputs or axis bindings differ"),
    ],
)
def test_keras_actual_splash_mathematical_projection_reports_remaining_boundary(
    scope, error, reason
):
    graphs = tuple(
        jax.make_jaxpr(splash_function(mode))(*splash_signatures())
        for mode in (None, "wrong_value")
    )
    with pytest.raises(error, match=reason):
        compare_algebraic_jaxprs(*graphs, scope=scope, algebra_timeout=10)


def test_keras_actual_splash_cpu_lowering_reports_source_backend_boundary():
    with pytest.raises(ValueError, match="Only interpret mode is supported on CPU backend"):
        jax.jit(splash_function()).lower(*splash_signatures())


@pytest.mark.pallas_communication
def test_keras_actual_splash_official_interpreter_and_numpy_reference():
    # The official TPU interpreter maintains process-global race tracking state.
    code = (
        "import json; "
        "from tests.upstream_tpu.keras_cases import splash_interpreter_evidence; "
        "print(json.dumps(splash_interpreter_evidence()))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert process.returncode == 0, process.stdout + process.stderr
    evidence = json.loads(process.stdout.splitlines()[-1])
    assert evidence["execution"] == "official_tpu_interpreter_on_cpu"
    assert evidence["original"] < 2e-6 and evidence["direct"] < 2e-6
    assert evidence["wrong_value"] > 0.1
