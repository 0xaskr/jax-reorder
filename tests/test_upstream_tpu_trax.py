"""Real Trax LayerNorm computation with frozen AI labels and NumPy witnesses."""

import jax
import numpy as np
import pytest

from jax_reorder import Status, compare_algebraic_jaxprs, compare_jaxprs
from tests.upstream_tpu.sources import read_manifest, read_source
from tests.upstream_tpu.trax_cases import CASES, REVISION, function, inputs, reference


def test_trax_sources_and_jax_backend_are_pinned():
    manifest = read_manifest("trax")
    assert manifest["revision"] == REVISION
    assert manifest["repository"] == "google/trax"
    source = read_source("trax", "normalization")
    assert "class LayerNorm(base.Layer):" in source
    assert "from trax.fastmath import numpy as jnp" in source
    backend = read_source("trax", "backend")
    assert "import jax.numpy as jnp" in backend and "'np': jnp" in backend
    assert "TPU" in read_source("trax", "readme")
    assert "Apache License" in read_source("trax", "license")


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("variant,structural,mathematical", CASES)
def test_actual_trax_layernorm_and_fixed_ai_candidates(center, variant, structural, mathematical):
    args = inputs(3)
    original, candidate = function(center=center), function(variant, center=center)
    graphs = jax.make_jaxpr(original)(*args), jax.make_jaxpr(candidate)(*args)
    result = compare_jaxprs(*graphs)
    assert result.status is (Status.EQUIVALENT if structural else Status.DIFFERENT)
    result = compare_algebraic_jaxprs(*graphs, algebra_timeout=10)
    assert result.status is (Status.EQUIVALENT if mathematical else Status.INEQUIVALENT)
    if mathematical and not structural:
        assert result.equivalence_kind == "algebraic"
    for seed in (3, 29):
        args = inputs(seed)
        expected = reference(args, center=center)
        actual = np.asarray(original(*args))
        changed = np.asarray(candidate(*args))
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        if mathematical:
            np.testing.assert_allclose(changed, expected, rtol=2e-6, atol=2e-6)
        else:
            assert np.max(np.abs(changed - expected)) > 0.1
