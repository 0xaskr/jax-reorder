"""Real ViT MLP: schedule and algebra positives, bias negative, NumPy oracle."""

import jax
import numpy as np
import pytest

from jax_reorder import Status, compare_functions
from tests.upstream_tpu import vision_transformer_cases as cases

pytestmark = pytest.mark.business


def test_upstream_mlp_and_independent_numerical_witnesses():
    for seed in (7, 23):
        args = cases.samples(seed)
        expected = cases.reference(*args)
        for function in (cases.original, cases.reordered, cases.split_output_bias):
            np.testing.assert_allclose(function(*args), expected, rtol=3e-6, atol=3e-6)
        assert not np.allclose(cases.wrong_output_bias(*args), expected)


@pytest.mark.parametrize(
    "candidate,structural,mathematical",
    [
        (cases.reordered, Status.EQUIVALENT, Status.EQUIVALENT),
        (cases.split_output_bias, Status.DIFFERENT, Status.EQUIVALENT),
        (cases.wrong_output_bias, Status.DIFFERENT, Status.INEQUIVALENT),
    ],
)
def test_real_mlp_comparison(candidate, structural, mathematical):
    args = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in cases.samples(7))
    structure = compare_functions(cases.original, candidate, *args, equivalence="structural")
    assert structure.status == structural
    result = compare_functions(cases.original, candidate, *args, algebra_timeout=15)
    assert result.status == mathematical
    if candidate is cases.split_output_bias:
        assert result.modes[0].equivalence_kind == "algebraic"
        assert result.modes[0].algebra.warnings
