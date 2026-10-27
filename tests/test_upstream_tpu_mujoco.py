"""Real MJX-JAX quaternion multiplication, without substituting Warp or Pallas."""

import jax
import numpy as np
import pytest

from jax_reorder import Status, compare_functions
from tests.upstream_tpu import mujoco_cases as cases
from tests.upstream_tpu.sources import read_manifest, read_source

pytestmark = pytest.mark.business


def test_pinned_quaternion_source_and_mjx_jax_tpu_support():
    manifest = read_manifest("mujoco")
    assert manifest["revision"] == cases.REVISION
    for key in manifest["files"]:
        read_source("mujoco", key)
    source = read_source("mujoco", "math")
    assert "def quat_mul(u: jax.Array, v: jax.Array)" in source
    assert "(4,) quaternion (w,x,y,z)" in source
    evidence = read_source("mujoco", "tpu_evidence")
    assert "Google Cloud TPUs" in evidence and "MJX-JAX" in evidence
    assert "optimizes performance specifically for NVIDIA GPUs" in evidence
    assert "Apache License" in read_source("mujoco", "license")
    assert cases.original().__code__.co_firstlineno == 148


@pytest.mark.parametrize("batch", [None, 3], ids=["source_signature", "external_vmap"])
@pytest.mark.parametrize("seed", [7, 23])
def test_hamilton_product_against_independent_numpy_matrix(batch, seed):
    args = cases.samples(seed, batch=batch)
    expected = cases.reference(*args)
    for variant in ("original", "reordered", "regrouped"):
        actual = cases.function(variant, batch=batch)(*args)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    wrong = cases.function("wrong_cross", batch=batch)(*args)
    np.testing.assert_allclose(wrong, cases.reference(*args[::-1]), rtol=2e-6, atol=2e-6)
    assert np.max(np.abs(wrong - expected)) > 1e-2


@pytest.mark.parametrize("batch", [None, 3], ids=["source_signature", "external_vmap"])
@pytest.mark.parametrize("variant,structural_equal,mathematical_equal", cases.CASES)
def test_quaternion_structure_and_complete_mathematical_output(
    batch, variant, structural_equal, mathematical_equal
):
    args = cases.samples(7, batch=batch)
    signatures = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in args)
    left, right = cases.function(batch=batch), cases.function(variant, batch=batch)
    structure = compare_functions(left, right, *signatures, equivalence="structural")
    assert structure.status == (Status.EQUIVALENT if structural_equal else Status.DIFFERENT)
    result = compare_functions(left, right, *signatures, algebra_timeout=20)
    assert result.status == (Status.EQUIVALENT if mathematical_equal else Status.INEQUIVALENT)
    if mathematical_equal and not structural_equal:
        assert result.equivalence_kind == "algebraic"
        assert result.modes[0].algebra.warnings


def test_unit_quaternions_identity_and_noncommuting_basis():
    identity, i, j, k = np.eye(4, dtype=np.float32)
    source = cases.original()
    np.testing.assert_array_equal(source(identity, j), j)
    np.testing.assert_array_equal(source(j, identity), j)
    np.testing.assert_array_equal(source(i, j), k)
    np.testing.assert_array_equal(source(j, i), -k)
    np.testing.assert_array_equal(cases.wrong_cross(i, j), -k)
    for quaternion in (identity, i, j, k):
        np.testing.assert_array_equal(
            source(quaternion, identity), cases.reference(quaternion, identity)
        )
