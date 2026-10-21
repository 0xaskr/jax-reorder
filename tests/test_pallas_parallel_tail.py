"""ANY restoration partitions symbolic parallel owners without losing history."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    NormalizedEvidence,
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_parallel_tail_cases import (
    dma_function,
    expected,
    function,
    two_tail_expected,
    two_tail_function,
)
from tests.pallas_projection_oracle import at_point


def source(fn, shape=(24, 512)):
    return adapt(jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))).view


def project(view, *, budget=None):
    return normalize_view(view, budget=AnalysisBudget(30000) if budget is None else budget)


def complete_outputs(proof, x, outputs):
    for expression, oracle in zip(proof.projection.outputs, outputs, strict=True):
        actual = np.empty(oracle.shape, oracle.dtype)
        for point in np.ndindex(oracle.shape):
            actual[point] = at_point(expression, (x,), point)
        np.testing.assert_array_equal(actual, oracle)


def replay(view, proof):
    evidence = NormalizedEvidence(
        proof, proof, tuple((i, i) for i in range(len(proof.projection.outputs)))
    )
    verify_normalized_evidence(view, view, evidence, budget=AnalysisBudget(30000))


@pytest.mark.parametrize("mode", ["partial", "full"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_full_and_tail_owners_restore_independent_complete_outputs(mode, reverse, multiple):
    options = dict(reverse=reverse, multiple=multiple, different_tail=multiple)
    view = source(function(mode, **options))
    proof = project(view)
    replay(view, proof)
    x = np.random.default_rng(302157).integers(-9, 10, (24, 512)).astype(np.float32)
    complete_outputs(proof, x, expected(x, mode, **options))
    versions = proof.projection.versions
    assert any("output_restore" in version.origin.path for version in versions)
    assert all(".owners[" in version.origin.path for version in versions)
    assert len({(v.storage, v.version, v.origin.path) for v in versions}) == len(versions)


@pytest.mark.parametrize("mode", ["partial", "full", "reset"])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_source_generic_cpu_execution_matches_full_numpy_oracle(mode, reverse):
    options = dict(reverse=reverse, multiple=True, different_tail=True)
    x = np.arange(24 * 512, dtype=np.float32).reshape(24, 512)
    actual = jax.jit(function(mode, interpret=True, **options))(x)
    for value, oracle in zip(actual, expected(x, mode, **options), strict=True):
        np.testing.assert_array_equal(value, oracle)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_tail_axes_and_mixed_radix_owner_mapping(mixed, reverse):
    options = dict(mixed=mixed, reverse=reverse)
    view = source(two_tail_function(**options), (3, 16, 512))
    proof = project(view)
    replay(view, proof)
    x = np.random.default_rng(701).integers(-9, 10, (3, 16, 512)).astype(np.float32)
    complete_outputs(proof, x, two_tail_expected(x))
    np.testing.assert_array_equal(
        jax.jit(two_tail_function(interpret=True, **options))(x), two_tail_expected(x)[0]
    )


def test_resetting_revisited_owner_has_a_complete_output_counterexample():
    original, changed = (source(function(mode)) for mode in ("partial", "reset"))
    x = np.arange(24 * 512, dtype=np.float32).reshape(24, 512)
    complete_outputs(project(changed), x, expected(x, "reset"))
    assert np.any(expected(x)[0] != expected(x, "reset")[0])
    with pytest.raises(MathematicalUndecidedError):
        compare_program_views(original, changed, analysis_budget=30000)


@pytest.mark.parametrize("mode", ["missing_init", "padding_leak"])
def test_partition_neither_initializes_a_first_visit_nor_restores_padding(mode):
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        project(source(function(mode)))


@pytest.mark.parametrize("mode", ["overlap", "owner_swap"])
def test_partition_cannot_repair_global_parallel_owner_conflicts(mode):
    with pytest.raises(UnsupportedJaxprError, match="bijection|redundant|overlap|owner"):
        project(source(function(mode)))


def test_tail_owner_dma_preserves_restored_unmodified_columns():
    view = source(dma_function())
    proof = project(view)
    x = np.arange(24 * 512, dtype=np.float32).reshape(24, 512)
    complete_outputs(proof, x, expected(x, "reset"))
    replay(view, proof)


def test_tail_owner_restore_does_not_complete_a_pending_dma():
    with pytest.raises(UnsupportedJaxprError, match="DMA destination|completion wait|pending DMA"):
        project(source(dma_function(missing_wait=True)))


def test_partition_restoration_evidence_rejects_a_forged_memory_version():
    view = source(function())
    proof = project(view)
    versions = list(proof.projection.versions)
    index = next(i for i, version in enumerate(versions) if "output_restore" in version.origin.path)
    versions[index] = replace(versions[index], version=0)
    forged = replace(proof, projection=replace(proof.projection, versions=tuple(versions)))
    with pytest.raises(CorrespondenceError):
        replay(view, forged)


def test_large_parallel_grid_uses_owner_regions_instead_of_enumerating_programs():
    costs = []
    for groups in (8, 1_000_000):
        budget = AnalysisBudget(10000)
        project(source(function(columns=groups * 256 - 128), (24, groups * 256)), budget=budget)
        costs.append(budget.used)
    assert costs[1] <= costs[0] + 1000
    with pytest.raises(ComparisonBudgetError):
        project(source(function()), budget=AnalysisBudget(20))


def test_public_kernel_comparison_reaches_parallel_tail_resource_proof():
    traced = jax.make_jaxpr(function())(jax.ShapeDtypeStruct((24, 512), jnp.float32))
    result = compare_pallas_kernels(traced, traced, check_protocol=True, analysis_budget=30000)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(traced, traced, result, analysis_budget=30000)


def test_separate_and_reversed_mixed_radix_owners_match_semantically():
    original = source(two_tail_function(), (3, 16, 512))
    candidate = source(two_tail_function(mixed=True, reverse=True), (3, 16, 512))
    result = compare_program_views(original, candidate, analysis_budget=30000)
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(original, candidate, result.evidence, budget=AnalysisBudget(30000))


def test_tail_revisit_does_not_change_an_ordinary_full_width_output_lifetime():
    view = source(function(multiple=True))
    proof = project(view)
    x = np.arange(24 * 512, dtype=np.float32).reshape(24, 512)
    complete_outputs(proof, x, expected(x, multiple=True))
    assert all(
        version.storage == 1
        for version in proof.projection.versions
        if "output_restore" in version.origin.path
    )
