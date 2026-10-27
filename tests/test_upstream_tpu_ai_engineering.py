"""Real tutorial MLP, its original dimensions, and fixed AI comparison candidates."""

import jax
import numpy as np
import pytest

from jax_reorder import ComparisonFailureKind, MathematicalUndecidedError, Status, compare_functions
from tests.upstream_tpu import ai_engineering_cases as cases
from tests.upstream_tpu.sources import read_manifest, read_source

pytestmark = pytest.mark.business


def test_pinned_mnist_source_and_tpu_installation_evidence():
    manifest = read_manifest("ai_engineering")
    for key in manifest["files"]:
        read_source("ai_engineering", key)
    assert "def forward(params, x):" in read_source("ai_engineering", "mlp")
    assert "pip install jax[tpu]" in read_source("ai_engineering", "tpu_evidence")
    assert "MIT License" in read_source("ai_engineering", "license")


def test_unmodified_upstream_initializer_and_full_width_forward():
    params = cases.definitions()["init_params"](jax.random.PRNGKey(7))
    shapes = ((784, 256), (256, 128), (128, 10))
    for index, shape in enumerate(shapes, 1):
        assert params[f"layer{index}"]["w"].shape == shape
        np.testing.assert_array_equal(params[f"layer{index}"]["b"], np.zeros(shape[-1]))
    _, x = cases.samples(23)
    np.testing.assert_allclose(cases.original(params, x), cases.reference(params, x), atol=3e-6)


@pytest.mark.parametrize("batch", [1, 128])
def test_full_width_mlp_numerical_witnesses(batch):
    for seed in (7, 23):
        args = cases.samples(seed, batch)
        expected = cases.reference(*args)
        for fn in (cases.original, cases.reordered, cases.split_bias):
            np.testing.assert_allclose(fn(*args), expected, rtol=3e-6, atol=3e-6)
        wrong = np.asarray(cases.wrong_bias(*args))
        assert not np.allclose(wrong, expected)
        np.testing.assert_allclose(
            wrong - expected,
            np.broadcast_to(-2 * args[0]["layer3"]["b"], expected.shape),
            rtol=3e-5,
            atol=3e-6,
        )


@pytest.mark.parametrize(
    "candidate,structural,mathematical",
    [
        (cases.reordered, Status.EQUIVALENT, Status.EQUIVALENT),
        (cases.split_bias, Status.DIFFERENT, Status.EQUIVALENT),
    ],
)
def test_full_width_mlp_comparison(candidate, structural, mathematical):
    args = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), cases.samples(7))
    structure = compare_functions(cases.original, candidate, *args, equivalence="structural")
    assert structure.status == structural
    result = compare_functions(cases.original, candidate, *args, algebra_timeout=20)
    assert result.status == mathematical
    if candidate is cases.split_bias:
        assert result.modes[0].equivalence_kind == "algebraic"
        assert result.modes[0].algebra.warnings


def test_wrong_bias_retains_the_common_call_definedness_boundary():
    args = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), cases.samples(7))
    structure = compare_functions(cases.original, cases.wrong_bias, *args, equivalence="structural")
    assert structure.status == Status.DIFFERENT
    # The numerical counterexamples above do not substitute for a symbolic proof:
    # shared ReLU JIT call domains still prevent an exact mathematical witness.
    with pytest.raises(MathematicalUndecidedError) as exc:
        compare_functions(cases.original, cases.wrong_bias, *args, algebra_timeout=20)
    assert exc.value.failure_kind == ComparisonFailureKind.CAS_UNDECIDED
    assert "common_jit" in exc.value.diagnostics["rules"]
