"""Actual AlphaFold 3 masked means preserve maximum instead of additive epsilon."""

import jax
import numpy as np
import pytest

from jax_reorder import Status, compare_functions
from tests.upstream_tpu import alphafold3_cases as cases
from tests.upstream_tpu.sources import read_manifest, read_source

pytestmark = pytest.mark.business


def test_alphafold3_sources_are_pinned_and_contain_tpu_specializations():
    manifest = read_manifest("alphafold3")
    assert manifest["revision"] == cases.REVISION
    for key in manifest["files"]:
        assert read_source("alphafold3", key)
    assert "On TPU small matrix multiplications are very suboptimal" in read_source(
        "alphafold3", "geometry"
    )
    linear = read_source("alphafold3", "linear")
    assert "linear Layer does not use the matmul units on the tpu" in linear
    assert "if num_input_dims == 0 and self.fast_scalar_mode:" in linear
    assert cases.original().__code__.co_firstlineno == 60


@pytest.mark.parametrize("scenario", cases.SCENARIOS)
@pytest.mark.parametrize("variant,structural_equal,mathematical_equal", cases.CASES)
def test_alphafold3_mask_mean_structure_and_independent_numeric_reference(
    scenario, variant, structural_equal, mathematical_equal
):
    left, right = cases.function(scenario=scenario), cases.function(variant, scenario=scenario)
    for seed in (7, 23):
        args = cases.inputs(seed, scenario=scenario)
        reference = cases.reference(*args, scenario=scenario)
        np.testing.assert_allclose(left(*args), reference, rtol=2e-6, atol=2e-6)
        actual = np.asarray(right(*args))
        if mathematical_equal:
            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)
        else:
            assert np.max(np.abs(actual - reference)) > 1e-3
    signatures = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in args)
    result = compare_functions(left, right, *signatures, equivalence="structural")
    assert result.status == (Status.EQUIVALENT if structural_equal else Status.DIFFERENT)


@pytest.mark.parametrize("variant,structural_equal,mathematical_equal", cases.CASES)
def test_alphafold3_mask_mean_math(variant, structural_equal, mathematical_equal):
    args = cases.inputs(7)
    signatures = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in args)
    result = compare_functions(
        cases.function(), cases.function(variant), *signatures, algebra_timeout=20
    )
    assert result.status == (Status.EQUIVALENT if mathematical_equal else Status.INEQUIVALENT)
    if mathematical_equal and not structural_equal:
        assert result.modes[0].equivalence_kind == "algebraic"


@pytest.mark.parametrize("scenario", cases.SCENARIOS)
@pytest.mark.parametrize("epsilon", [1e-10, cases.EPSILON])
def test_alphafold3_mask_mean_zero_and_tiny_masks_keep_the_clamp(scenario, epsilon):
    for mode in ("positive", "zero", "tiny"):
        args = cases.inputs(7, scenario=scenario, zero=mode == "zero", tiny=mode == "tiny")
        reference = cases.reference(*args, scenario=scenario, epsilon=epsilon)
        actual = cases.function(scenario=scenario, epsilon=epsilon)(*args)
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)
        if mode == "zero":
            assert np.all(np.asarray(actual) == 0)
