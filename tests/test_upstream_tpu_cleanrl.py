"""Real TPU-tested CleanRL Actor and independent AI positive/negative candidates."""

import jax
import numpy as np
import pytest

from jax_reorder import ComparisonFailureKind, MathematicalUndecidedError, Status, compare_functions
from tests.upstream_tpu import cleanrl_cases as cases
from tests.upstream_tpu.sources import read_manifest, read_source

pytestmark = pytest.mark.business


def test_pinned_actor_source_and_tpu_evidence():
    manifest = read_manifest("cleanrl")
    for key in manifest["files"]:
        read_source("cleanrl", key)
    assert "VM w/ TPU" in read_source("cleanrl", "tpu_evidence")
    assert "class Actor(nn.Module):" in read_source("cleanrl", "ddpg")
    assert "MIT License" in read_source("cleanrl", "license")


@pytest.mark.parametrize("batch", [1, 3])
def test_actor_numerical_witnesses(batch):
    for seed in (7, 23):
        args = cases.samples(seed, batch)
        expected = cases.reference(*args)
        for fn in (cases.original, cases.reordered, cases.split_bias):
            np.testing.assert_allclose(fn(*args), expected, rtol=3e-6, atol=3e-6)
        assert not np.allclose(cases.wrong_bias(*args), expected)


@pytest.mark.parametrize(
    "candidate,structural,mathematical",
    [
        (cases.reordered, Status.EQUIVALENT, Status.EQUIVALENT),
        (cases.split_bias, Status.DIFFERENT, Status.EQUIVALENT),
    ],
)
def test_actor_comparison(candidate, structural, mathematical):
    args = tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in cases.samples(7))
    structure = compare_functions(cases.original, candidate, *args, equivalence="structural")
    assert structure.status == structural
    result = compare_functions(cases.original, candidate, *args, algebra_timeout=20)
    assert result.status == mathematical
    if candidate is cases.split_bias:
        assert result.modes[0].equivalence_kind == "algebraic"
        assert result.modes[0].algebra.warnings


def test_actor_negative_retains_the_common_call_definedness_boundary():
    args = tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in cases.samples(7))
    structure = compare_functions(cases.original, cases.wrong_bias, *args, equivalence="structural")
    assert structure.status == Status.DIFFERENT
    # Shared ReLU JITs become opaque call_*/defined_call_* terms. Their outputs
    # cancel in the residual, but the exact witness cannot establish every call's
    # original domain. Independent numerical witnesses above still reject the AI
    # candidate; they are not substituted for a mathematical nonidentity proof.
    with pytest.raises(MathematicalUndecidedError) as exc:
        compare_functions(cases.original, cases.wrong_bias, *args, algebra_timeout=20)
    assert exc.value.failure_kind == ComparisonFailureKind.CAS_UNDECIDED
    assert "common_jit" in exc.value.diagnostics["rules"]
