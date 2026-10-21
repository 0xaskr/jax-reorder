"""Full-value, initialization, proof replay and footprint mask regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import ComparisonBudgetError, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget, integer, modulo
from jax_reorder.core.masked_access import mask_regions
from jax_reorder.core.memory import Predicate
from jax_reorder.core.normalization import (
    NormalizedEvidence,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.tensor_domain import coordinate
from jax_reorder.core.tensor_storage import full_box
from tests.pallas_masked_access_cases import dma_function, expected, function
from tests.pallas_projection_oracle import at_point

MODES = (
    "slice",
    "holes",
    "cover",
    "no_other",
    "read_where",
    "data",
    "data_no_other",
    "data_initialize",
    "empty",
)


def project(fn, shape):
    view = adapt(jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))).view
    proof = normalize_view(view, budget=AnalysisBudget(10000))
    verify_normalized_evidence(
        view, view, NormalizedEvidence(proof, proof, ((0, 0),)), budget=AnalysisBudget(10000)
    )
    return proof


def check(proof, x, target):
    # The independent oracle evaluates select children eagerly. Sentinel padded
    # buffers make both children evaluable while detecting incorrect predicates.
    padded = np.pad(
        x,
        tuple((0, 16 - n) if i == x.ndim - 1 else (0, 0) for i, n in enumerate(x.shape)),
        constant_values=123456,
    )
    actual = np.empty_like(target)
    for point in np.ndindex(target.shape):
        actual[point] = at_point(proof.projection.outputs[0], (padded,), point)
    np.testing.assert_array_equal(actual, target)


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("mode", MODES)
def test_complete_masked_outputs_and_replay(mode, rank):
    fn, shape = function(mode, rank=rank)
    x = (np.arange(np.prod(shape), dtype=np.float32) - np.prod(shape) // 2).reshape(shape)
    check(project(fn, shape), x, expected(x, mode))


@pytest.mark.parametrize("mode", ["missing", "bad_no_other", "read_bad", "data_bad"])
def test_invalid_lane_or_unwritten_output_is_rejected(mode):
    fn, shape = function(mode)
    with pytest.raises(UnsupportedJaxprError, match="undefined|uninitialized"):
        project(fn, shape)


def test_local_dma_masked_store_uses_only_active_lanes():
    x = np.arange(8 * 256, dtype=np.float32).reshape(8, 256)
    proof = project(dma_function(), x.shape)
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == x[point] + (
            2 if point[-1] >= 128 else 0
        )
    assert len(proof.projection.communication) == 1


@pytest.mark.parametrize("mode", ["overlap", "missing_wait"])
def test_local_dma_mask_does_not_remove_real_conflicts_or_waits(mode):
    with pytest.raises(UnsupportedJaxprError):
        project(dma_function(mode), (8, 256))


def test_large_static_mask_is_a_rectangle_not_a_lane_expansion():
    budget = AnalysisBudget(256)
    regions = mask_regions(
        Predicate("lt", (coordinate("local", 0), integer(1 << 29))),
        (full_box((1 << 30,)),),
        {},
        budget,
        "large",
    )
    assert len(regions) == 1 and regions[0].box.shape == (1 << 29,)
    assert budget.used < 16


def test_fragmented_mask_fails_with_its_explicit_budget():
    with pytest.raises(ComparisonBudgetError):
        mask_regions(
            Predicate("eq", (modulo(coordinate("local", 0), 2), integer(0))),
            (full_box((1 << 20,)),),
            {},
            AnalysisBudget(50),
            "fragmented",
        )


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", MODES)
def test_native_cpu_interpreter_matches_complete_oracle(mode):
    fn, shape = function(mode, interpret=True)
    x = (np.arange(np.prod(shape), dtype=np.float32) - np.prod(shape) // 2).reshape(shape)
    np.testing.assert_array_equal(np.asarray(fn(x)), expected(x, mode))
