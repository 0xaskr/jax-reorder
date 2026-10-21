"""Local ANY restores only committed values, keeping each output's own timeline."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    NormalizedEvidence,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_local_any_cases import WIDTH, dma_function, expected, function, upstream_function
from tests.pallas_projection_oracle import at_point


def trace(fn, shape):
    return jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))


def project(source, *, limit=30000):
    return normalize_view(adapt(source).view, budget=AnalysisBudget(limit))


def verify_rewrite_evidence(view, proof, *, budget):
    evidence = NormalizedEvidence(
        proof, proof, tuple((i, i) for i in range(len(proof.projection.outputs)))
    )
    verify_normalized_evidence(view, view, evidence, budget=budget)


def assert_outputs(projection, x, outputs):
    for value, expected_output in zip(projection.outputs, outputs, strict=True):
        # Evaluate every output coordinate, not a selection of boundary lanes.
        actual = np.asarray(
            [at_point(value, (x,), point) for point in np.ndindex(expected_output.shape)]
        ).reshape(expected_output.shape)
        np.testing.assert_array_equal(actual, expected_output)


def test_pinned_upstream_non_immediate_reduction_restores_each_column_block():
    source = trace(upstream_function(), (24, 384))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, result)
    proof = project(source)
    assert sum("output_restore" in v.origin.path for v in proof.projection.versions) == 6
    x = np.arange(24 * 384, dtype=np.float32).reshape(24, 384)
    oracle = x.reshape(3, 8, 384).sum(axis=0)
    np.testing.assert_array_equal(jax.jit(lambda a: a.reshape(3, 8, 384).sum(axis=0))(x), oracle)
    assert_outputs(proof.projection, x, (oracle,))


@pytest.mark.parametrize("mode", ["accumulate", "partial", "overwrite"])
@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("consecutive,steps", [(1, 5), (2, 6)])
def test_revisited_values_have_complete_independent_numpy_oracles(
    mode, multiple, consecutive, steps
):
    source = trace(
        function(mode, multiple=multiple, consecutive=consecutive, steps=steps), (16, steps * WIDTH)
    )
    proof = project(source)
    verify_rewrite_evidence(adapt(source).view, proof, budget=AnalysisBudget())
    for seed in (7, 23):
        x = np.random.default_rng(seed).integers(-8, 9, size=(16, steps * WIDTH)).astype(np.float32)
        assert_outputs(
            proof.projection, x, expected(x, mode, consecutive=consecutive, multiple=multiple)
        )
    restore = [v for v in proof.projection.versions if "output_restore" in v.origin.path]
    assert len(restore) == (steps + consecutive - 1) // consecutive - 2
    assert all(v.storage == 1 for v in restore)
    for slot in {v.storage for v in proof.projection.versions}:
        numbers = [v.version for v in proof.projection.versions if v.storage == slot]
        assert numbers == sorted(set(numbers))


@pytest.mark.parametrize("mode", ["wrong_map", "reset_each", "overwrite"])
def test_wrong_mapping_or_lost_history_has_a_complete_output_counterexample(mode):
    a, b = trace(function(), (16, 5 * WIDTH)), trace(function(mode), (16, 5 * WIDTH))
    x = np.arange(16 * 5 * WIDTH, dtype=np.float32).reshape(16, 5 * WIDTH)
    changed = project(b).projection
    assert_outputs(changed, x, expected(x, mode))
    actual = np.asarray(
        [at_point(changed.outputs[0], (x,), point) for point in np.ndindex((16, 2 * WIDTH))]
    ).reshape(16, 2 * WIDTH)
    assert np.any(actual != expected(x)[0])
    assert compare_pallas_kernels(a, b, check_protocol=True).status is Status.DIFFERENT


@pytest.mark.parametrize("revisit", [None, "immediate"])
def test_default_and_immediate_nonconsecutive_revisits_remain_rejected(revisit):
    with pytest.raises(UnsupportedJaxprError, match="nonconsecutively"):
        project(trace(function(revisit=revisit), (16, 5 * WIDTH)))


def test_first_visit_is_not_initialized_by_any():
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        project(trace(function("missing_init"), (16, 5 * WIDTH)))


def test_local_dma_to_restored_output_waits_and_keeps_latest_full_version():
    source = trace(dma_function(), (8, 3 * WIDTH))
    proof = project(source)
    x = np.arange(8 * 3 * WIDTH, dtype=np.float32).reshape(8, 3 * WIDTH)
    assert_outputs(
        proof.projection, x, (np.concatenate((x[:, 2 * WIDTH :], x[:, WIDTH : 2 * WIDTH]), axis=1),)
    )
    assert len(proof.projection.communication) == 3
    verify_rewrite_evidence(adapt(source).view, proof, budget=AnalysisBudget())


@pytest.mark.parametrize("mode", ["missing_wait", "write_pending"])
def test_restoration_never_supplies_missing_dma_completion(mode):
    with pytest.raises(UnsupportedJaxprError, match="DMA destination|completion wait"):
        project(trace(dma_function(mode), (8, 3 * WIDTH)))


def test_replay_rejects_forged_restoration_version():
    source = trace(function(), (16, 5 * WIDTH))
    view = adapt(source).view
    proof = project(source)
    versions = list(proof.projection.versions)
    index = next(i for i, value in enumerate(versions) if "output_restore" in value.origin.path)
    versions[index] = replace(versions[index], version=0)
    changed = replace(proof, projection=replace(proof.projection, versions=tuple(versions)))
    with pytest.raises(CorrespondenceError):
        verify_rewrite_evidence(view, changed, budget=AnalysisBudget())


def test_any_replay_has_an_explicit_resource_budget():
    with pytest.raises(ComparisonBudgetError):
        project(trace(function(steps=100), (16, 100 * WIDTH)), limit=20)


@pytest.mark.parametrize("mode", ["parallel_repeated", "owner_swap"])
def test_any_does_not_create_order_or_change_a_parallel_owner(mode):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu

    def kernel(x, y):
        y[...] = x[...]

    def mapping(owner, p):
        if mode == "parallel_repeated":
            return owner, p % 2
        return jnp.where(p == 2, 1 - owner, owner), p % 2

    source = trace(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
            grid=(2, 3),
            in_specs=[pl.BlockSpec((8, 128), lambda owner, p: (owner, p))],
            out_specs=pl.BlockSpec(
                (8, 128),
                mapping,
                pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
            ),
            compiler_params=tpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel" if mode == "parallel_repeated" else "arbitrary",
                )
            ),
        ),
        (16, 384),
    )
    with pytest.raises(UnsupportedJaxprError, match="bijection|redundant|overlap"):
        project(source)
