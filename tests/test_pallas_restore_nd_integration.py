"""Restoration preserves source versions across local and remote matrix paths."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import CorrespondenceError, Status, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_projection_oracle import at_point
from tests.pallas_restore_nd_cases import (
    ROWS,
    WIDTH,
    local_padded_expected,
    local_padded_revisit,
    matrix_expected,
    matrix_revisit,
    padded_temporary_lifetime,
)


def source(function, shape):
    return adapt(jax.make_jaxpr(function)(jax.ShapeDtypeStruct(shape, jnp.float32))).view


def complete_output(expression, x, expected):
    actual = np.empty(expected.shape, dtype=expected.dtype)
    for point in np.ndindex(expected.shape):
        actual[point] = at_point(expression, (x,), point)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("wrong", [False, True])
def test_local_padded_revisit_and_fixed_accumulator_have_independent_lifetimes(wrong):
    shape = (ROWS, 3 * WIDTH)
    view = source(local_padded_revisit(wrong=wrong), shape)
    proof = normalize_view(view, budget=AnalysisBudget(200000))
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    for expression, expected in zip(
        proof.projection.outputs, local_padded_expected(x, wrong=wrong), strict=True
    ):
        complete_output(expression, x, expected)
    if wrong:
        assert not np.array_equal(
            local_padded_expected(x)[0], local_padded_expected(x, wrong=True)[0]
        )


def test_local_any_does_not_initialize_a_first_visit_hole():
    view = source(local_padded_revisit(missing_init=True), (ROWS, 3 * WIDTH))
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(view, budget=AnalysisBudget(200000))


@pytest.mark.parametrize("wrong", [False, True])
def test_local_padded_source_cpu_execution_matches_complete_numpy_outputs(wrong):
    x = np.arange(ROWS * 3 * WIDTH, dtype=np.float32).reshape(ROWS, 3 * WIDTH)
    actual = jax.jit(local_padded_revisit(wrong=wrong, interpret=True))(x)
    for value, expected in zip(actual, local_padded_expected(x, wrong=wrong), strict=True):
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("wrong", [False, True])
def test_nonleading_shard_axis_matrix_dma_restores_only_the_old_half(wrong):
    shape = (ROWS, 6 * WIDTH)
    view = source(matrix_revisit(wrong=wrong), shape)
    proof = normalize_view(view, budget=AnalysisBudget(200000))
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    complete_output(proof.projection.outputs[0], x, matrix_expected(x, wrong=wrong))
    assert proof.remote.protocol.io.ports[1].restore_on_revisit
    if wrong:
        assert not np.array_equal(matrix_expected(x), matrix_expected(x, wrong=True))


def test_nd_restore_does_not_supply_peer_synchronization():
    view = source(matrix_revisit(synchronize=False), (ROWS, 6 * WIDTH))
    with pytest.raises(UnsupportedJaxprError):
        normalize_view(view, budget=AnalysisBudget(200000))


def test_nd_wait_reordering_replays_exact_restore_evidence():
    shape = (ROWS, 6 * WIDTH)
    a, b = (source(matrix_revisit(receive_first=order), shape) for order in (False, True))
    result = compare_program_views(a, b, analysis_budget=200000)
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget(200000))
    proof = result.evidence.left.remote
    ports = proof.protocol.io.ports
    altered = replace(ports[1], restore_on_revisit=False)
    forged_remote = replace(
        proof,
        protocol=replace(proof.protocol, io=replace(proof.protocol.io, ports=(ports[0], altered))),
    )
    forged = replace(result.evidence, left=replace(result.evidence.left, remote=forged_remote))
    with pytest.raises(CorrespondenceError, match="does not replay"):
        verify_normalized_evidence(a, b, forged, budget=AnalysisBudget(200000))


def test_local_and_self_remote_matrix_paths_match_the_same_complete_reference():
    shape = (ROWS, 6 * WIDTH)
    local = source(matrix_revisit(remote=False, sharded=False), (ROWS, 3 * WIDTH))
    remote = source(matrix_revisit(self_target=True), shape)
    a, b = (normalize_view(view, budget=AnalysisBudget(200000)) for view in (local, remote))
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = matrix_expected(x, self_target=True)
    complete_output(b.projection.outputs[0], x, expected)
    for shard, reference in zip(np.split(x, 2, axis=1), np.split(expected, 2, axis=1), strict=True):
        complete_output(a.projection.outputs[0], shard, reference)


def test_nonconsecutive_revisit_does_not_restore_unwritten_hbm_padding():
    view = source(padded_temporary_lifetime(), (ROWS, 3 * WIDTH))
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(view, budget=AnalysisBudget(200000))


def test_consecutive_visit_retains_initialized_temporary_padding():
    shape = (ROWS, 3 * WIDTH)
    view = source(padded_temporary_lifetime(consecutive=True), shape)
    proof = normalize_view(view, budget=AnalysisBudget(200000))
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.concatenate((x[:, 512:], np.full((ROWS, 128), 8, np.float32)), axis=1)
    complete_output(proof.projection.outputs[0], x, expected)
