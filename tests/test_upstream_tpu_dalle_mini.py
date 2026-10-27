"""Pinned real RMSNorm, AI schedule/math positives and a learned-scale negative."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import Status, compare_functions
from tests.upstream_tpu import dalle_mini_cases as cases

pytestmark = pytest.mark.business


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
def test_upstream_rmsnorm_and_independent_numerical_witnesses(shape):
    for seed in (7, 23):
        x, scale = cases.samples(shape, seed)
        expected = cases.reference(x, scale)
        for function in (cases.original, cases.reordered, cases.reassociated):
            np.testing.assert_allclose(function(x, scale), expected, atol=2e-6, rtol=2e-6)
        assert not np.allclose(cases.missing_scale(x, scale), expected)


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
@pytest.mark.parametrize(
    "candidate,structural,mathematical",
    [
        (cases.reordered, Status.EQUIVALENT, Status.EQUIVALENT),
        (cases.reassociated, Status.DIFFERENT, Status.EQUIVALENT),
        (cases.missing_scale, Status.DIFFERENT, Status.INEQUIVALENT),
    ],
)
def test_real_rmsnorm_comparison(candidate, structural, mathematical, shape):
    args = (
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct((shape[-1],), jnp.float32),
    )
    structure = compare_functions(cases.original, candidate, *args, equivalence="structural")
    assert structure.status == structural
    result = compare_functions(cases.original, candidate, *args, algebra_timeout=15)
    assert result.status == mathematical
    if candidate is cases.reassociated:
        assert result.modes[0].equivalence_kind == "algebraic"
        assert result.modes[0].algebra.warnings
