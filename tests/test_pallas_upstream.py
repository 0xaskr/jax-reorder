"""Pinned real-kernel baselines, AI candidates, and independent numeric witnesses."""

import jax
import numpy as np
import pytest
from jax._src import core

from examples.pallas_kernels import (
    CASES,
    MANIFEST,
    expected_status,
    function,
    inputs,
    oracle,
    source,
)
from jax_reorder import Status, compare_jaxprs, verify_correspondence


@pytest.mark.parametrize("index", range(len(MANIFEST["files"])))
def test_original_sources_match_commit_blob_and_content_digest(index):
    assert source(index)


@pytest.mark.parametrize("family,mode,numerical_equal", CASES)
def test_pinned_kernel_against_locally_authored_ai_candidate(family, mode, numerical_equal):
    args = inputs(family, 7)
    left, right = (jax.make_jaxpr(function(family, variant))(*args) for variant in (None, mode))
    result = compare_jaxprs(left, right, include_derivative_rules=False)
    assert result.status == expected_status(mode)
    if mode == "reorder":
        schedules = [
            tuple(eq.primitive for eq in g.eqns[0].params["jaxpr"].eqns) for g in (left, right)
        ]
        assert schedules[0] != schedules[1]
    core.check_jaxpr(result.left.jaxpr)
    core.check_jaxpr(result.right.jaxpr)
    if result.status == Status.EQUIVALENT:
        verify_correspondence(left, right, result.correspondence, include_derivative_rules=False)
    for seed in (7, 23):
        args = inputs(family, seed)
        original = function(family, interpret=True)(*args)
        candidate = function(family, mode, interpret=True)(*args)
        reference = oracle(family, args)
        for actual, expected in zip(
            jax.tree.leaves(original), jax.tree.leaves(reference), strict=True
        ):
            # Integer quantization may differ by one at a float rounding boundary.
            atol = 1 if np.issubdtype(actual.dtype, np.integer) else 2e-5
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=atol)
        equal = all(
            np.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(candidate), strict=True)
        )
        assert equal is numerical_equal, (family, mode, seed)
