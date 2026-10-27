"""Real Performer and distributed ALX source, independent candidates and witnesses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import Status, UnsupportedJaxprError, compare_functions
from tests.upstream_tpu import google_research_cases as cases

pytestmark = pytest.mark.business


@pytest.mark.parametrize("is_query", [False, True])
def test_real_performer_with_independent_numeric_reference(is_query):
    for seed in (7, 23):
        args = cases.performer_samples(seed)
        expected = cases.performer_reference(*args, is_query=is_query)
        np.testing.assert_allclose(
            cases.performer(*args, is_query=is_query), expected, rtol=2e-6, atol=2e-6
        )
        for variant in ("reordered", "distributed_scale"):
            np.testing.assert_allclose(
                cases.candidate(*args, is_query=is_query, variant=variant),
                expected,
                rtol=2e-6,
                atol=2e-6,
            )
        assert not np.allclose(
            cases.candidate(*args, is_query=is_query, variant="wrong_epsilon"), expected
        )


@pytest.mark.parametrize("is_query", [False, True])
@pytest.mark.parametrize(
    "variant,structural,mathematical",
    [
        ("reordered", Status.EQUIVALENT, Status.EQUIVALENT),
        ("distributed_scale", Status.DIFFERENT, Status.EQUIVALENT),
        ("wrong_epsilon", Status.DIFFERENT, Status.INEQUIVALENT),
    ],
)
def test_real_performer_comparison(is_query, variant, structural, mathematical):
    functions = cases.performer_functions(is_query, variant)
    args = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype) for value in cases.performer_samples(7)
    )
    assert compare_functions(*functions, *args, equivalence="structural").status == structural
    result = compare_functions(*functions, *args, algebra_timeout=20)
    assert result.status == mathematical
    if variant == "distributed_scale":
        assert result.modes[0].equivalence_kind == "algebraic"


def test_actual_collective_gramian_and_mean_negative():
    assert len(jax.devices("cpu")) >= 2
    values = np.arange(24, dtype=np.float32).reshape(6, 4) / 8 - 1
    expected = values.T @ values
    original = cases.distributed_gramian("original")
    positive = cases.distributed_gramian("einsum")
    negative = cases.distributed_gramian("wrong_average")
    np.testing.assert_allclose(original(values), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(positive(values), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(negative(values), expected / 2, rtol=2e-6, atol=2e-6)
    assert not np.allclose(negative(values), expected)
    signature = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    assert (
        compare_functions(original, negative, signature, equivalence="structural").status
        == Status.DIFFERENT
    )
    assert (
        compare_functions(original, positive, signature, equivalence="structural").status
        == Status.DIFFERENT
    )


@pytest.mark.parametrize("variant", ["einsum", "wrong_average"])
def test_alx_collective_math_is_explicitly_not_yet_supported(variant):
    # The positive has a numerical oracle; neither numerical equality nor this
    # known rejection counts as a successful mathematical equivalence proof.
    with pytest.raises(UnsupportedJaxprError, match="ordered or effectful common call"):
        compare_functions(
            cases.distributed_gramian("original"),
            cases.distributed_gramian(variant),
            jax.ShapeDtypeStruct((6, 4), jnp.float32),
            algebra_timeout=20,
        )
