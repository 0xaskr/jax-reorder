"""Real nested Diffusers functions, fixed AI rewrites and independent CPU evidence."""

import json
import os
import shutil

import jax
import numpy as np
import pytest

from jax_reorder import (
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    Status,
    compare_functions,
)
from tests.upstream_tpu import diffusers_cases as cases
from tests.upstream_tpu import sources

pytestmark = pytest.mark.business


def test_diffusers_sources_are_pinned_legacy_tpu_training_code():
    manifest = sources.read_manifest("diffusers")
    assert manifest["revision"] == cases.REVISION
    for key in manifest["files"]:
        assert sources.read_source("diffusers", key)
    readme = sources.read_source("diffusers", "readme")
    assert "Running on Google Cloud TPU" in readme
    assert "scripts require `diffusers<=0.39.x`" in readme
    training = sources.read_source("diffusers", "training")
    assert 'jax.pmap(train_step, "batch"' in training
    snr, norm = cases.originals()
    assert snr.__code__.co_firstlineno == 837
    assert norm.__code__.co_firstlineno == 982
    assert snr.__code__.co_filename.endswith("train_controlnet_flax.py.txt")


@pytest.mark.parametrize("name", ["main.l2", "main.train_step.compute_snr", "main.missing"])
def test_nested_source_loader_requires_the_exact_lexical_scope(name):
    with pytest.raises(ValueError, match="missing upstream nested definition"):
        sources.load_nested_definition("diffusers", "training", name, {})


@pytest.mark.parametrize("digest", ["sha256", "git_blob"])
def test_nested_source_loader_rejects_unverified_source(tmp_path, monkeypatch, digest):
    directory = tmp_path / "diffusers"
    shutil.copytree(sources.FIXTURES / "diffusers", directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if digest == "sha256":
        snapshot = directory / manifest["files"]["training"]["snapshot"]
        snapshot.write_bytes(snapshot.read_bytes() + b"\nraise RuntimeError('altered source')\n")
        message = "upstream SHA-256 mismatch"
    else:
        manifest["files"]["training"]["git_blob"] = "0" * 40
        path.write_text(json.dumps(manifest))
        message = "upstream Git blob mismatch"
    monkeypatch.setattr(sources, "FIXTURES", tmp_path)
    with pytest.raises(ValueError, match=message):
        sources.load_nested_definition("diffusers", "training", "main.compute_snr", {})


@pytest.mark.parametrize("variant,structural_equal,mathematical_equal", cases.SNR_CASES)
def test_diffusers_snr_comparison_and_independent_reference(
    variant, structural_equal, mathematical_equal
):
    left, right = cases.snr_function(), cases.snr_function(variant)
    for seed in (7, 23):
        inputs = cases.snr_inputs(seed)
        reference = cases.snr_reference(inputs)
        np.testing.assert_allclose(left(inputs), reference, rtol=2e-6, atol=2e-6)
        candidate = np.asarray(right(inputs))
        if mathematical_equal:
            np.testing.assert_allclose(candidate, reference, rtol=2e-6, atol=2e-6)
        else:
            assert np.max(np.abs(candidate - reference)) > 1.0
    signature = jax.ShapeDtypeStruct((4,), np.int32)
    structural = compare_functions(left, right, signature, equivalence="structural")
    assert structural.status == (Status.EQUIVALENT if structural_equal else Status.DIFFERENT)
    if structural_equal:
        assert compare_functions(left, right, signature).modes[0].equivalence_kind == "structural"


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("ratio_of_squares", "original conditions are not established"),
        ("wrong_scale", "SymPy did not establish output identity"),
    ],
)
def test_diffusers_snr_scalar_math_is_explicitly_undecided(variant, reason):
    # A scalar timestep exposes the current dynamic-index condition/query boundary
    # without spending 60 seconds on the batch gather's Piecewise expansion.
    # Separate numeric witnesses label the candidates, not this failed proof.
    with pytest.raises(MathematicalUndecidedError, match=reason):
        compare_functions(
            cases.snr_function(),
            cases.snr_function(variant),
            jax.ShapeDtypeStruct((), np.int32),
            algebra_timeout=20,
        )


@pytest.mark.stress
@pytest.mark.skipif(
    os.environ.get("JAX_REORDER_RUN_DIFFUSERS_SNR_BATCH") != "1",
    reason="opt in to the two measured 60-second batch SNR timeout checks",
)
@pytest.mark.parametrize("variant", ["ratio_of_squares", "wrong_scale"])
def test_diffusers_snr_batch_math_reports_default_timeout(variant):
    with pytest.raises(ComparisonTimeoutError, match="symbolic worker exceeded 60s"):
        compare_functions(
            cases.snr_function(),
            cases.snr_function(variant),
            jax.ShapeDtypeStruct((4,), np.int32),
        )


@pytest.mark.parametrize("variant,structural_equal,mathematical_equal", cases.NORM_CASES)
def test_diffusers_gradient_norm_comparison_and_independent_reference(
    variant, structural_equal, mathematical_equal
):
    left, right = cases.norm_function(), cases.norm_function(variant)
    for seed in (7, 23):
        inputs = cases.norm_inputs(seed)
        reference = cases.norm_reference(inputs)
        np.testing.assert_allclose(left(inputs), reference, rtol=2e-6, atol=2e-6)
        candidate = np.asarray(right(inputs))
        if mathematical_equal:
            np.testing.assert_allclose(candidate, reference, rtol=2e-6, atol=2e-6)
        else:
            assert abs(candidate - reference) > 0.1
    signature = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), cases.norm_inputs(7)
    )
    structural = compare_functions(left, right, signature, equivalence="structural")
    assert structural.status == (Status.EQUIVALENT if structural_equal else Status.DIFFERENT)
    mathematical = compare_functions(left, right, signature, algebra_timeout=20)
    assert mathematical.status == (Status.EQUIVALENT if mathematical_equal else Status.INEQUIVALENT)
    if mathematical_equal and not structural_equal:
        assert mathematical.modes[0].equivalence_kind == "algebraic"
