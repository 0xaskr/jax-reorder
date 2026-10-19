"""DMA splitting, count consumption and interval version safety."""

from dataclasses import replace

import jax.numpy as jnp
import pytest
from jax._src.state import indexing
from jax._src.state import types as state_types
from jax.experimental.pallas import tpu as pltpu

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    MathematicalUndecidedError,
    UnsupportedJaxprError,
)
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_dma_cases import disjoint_reuse, relayed, split_kernel
from tests.pallas_oracle import run_oracle
from tests.test_pallas_dma import copy, direct, view


def region_view(kernel, *, size=256, **options):
    return view(kernel, size=size, block=size, **options)


@pytest.mark.parametrize("waits", ["total", "parts", "serial"])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_split_and_whole_copy_have_same_projection(waits, order):
    a = region_view(split_kernel(waits=waits, order=order))
    b = region_view(copy)
    result = compare_program_views(a, b)
    assert result.equivalence_kind == "normalized"
    proof = result.evidence.left.projection
    assert len(proof.communication) == 2
    assert sum(w.logical_bytes for w in proof.waits) == 1024
    if waits == "parts":
        assert [w.released for w in proof.waits] == [False, True]
        assert proof.communication[0].wait == proof.communication[1].wait == proof.waits[-1].origin
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget())


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.int32])
def test_four_chunks_merge_independently_of_dtype_and_start_order(dtype):
    a = region_view(split_kernel(size=512, chunks=4, order=(2, 0, 3, 1)), size=512, dtype=dtype)
    b = region_view(direct, semaphores=0, size=512, dtype=dtype)
    assert compare_program_views(a, b).equivalence_kind == "normalized"


def test_partial_wait_must_not_publish_a_particular_producer():
    def kernel(x, y, z, semaphore):
        a = pltpu.async_copy(x.at[:128], y.at[:128], semaphore)
        b = pltpu.async_copy(x.at[128:], y.at[128:], semaphore)
        a.wait()
        z[:] = y[:]
        b.wait()

    with pytest.raises(UnsupportedJaxprError, match="destination accessed"):
        normalize_view(region_view(kernel, outputs=2), budget=AnalysisBudget())


def test_partial_consumption_cannot_mix_a_new_generation():
    def kernel(x, y, semaphore):
        transfer = pltpu.async_copy(x, y, semaphore)
        pltpu.make_async_copy(x.at[:128], y.at[:128], semaphore).wait()
        pltpu.async_copy(x.at[128:], y.at[128:], semaphore)
        transfer.wait()

    with pytest.raises(UnsupportedJaxprError, match="generation mixed"):
        normalize_view(region_view(kernel), budget=AnalysisBudget())


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("gap", "uninitialized interval"),
        ("underwait", "unconsumed semaphore"),
        ("overwait", "more than the issued"),
        ("overlap", "destination accessed"),
        ("bounds", "out of bounds"),
        ("stride", "unit stride"),
        ("extents", "extents differ"),
        ("unaligned", "128-element aligned"),
    ],
)
def test_invalid_region_and_count_contracts_fail(mode, reason):
    def kernel(x, y, semaphore):
        if mode == "bounds":
            # The public .at API rejects this before tracing. Preserve the bad
            # source interval explicitly to test our semantic admission boundary.
            indexer = indexing.NDIndexer((indexing.Slice(128, 256, 1),), (256,), (), validate=False)
            bad = state_types.TransformedRef(x, (indexer,))
            pltpu.async_copy(bad, y, semaphore).wait()
        elif mode == "stride":
            pltpu.async_copy(x.at[::2], y.at[::2], semaphore).wait()
        elif mode == "extents":
            pltpu.async_copy(x.at[:128], y, semaphore).wait()
        elif mode == "underwait":
            pltpu.async_copy(x, y, semaphore)
            pltpu.make_async_copy(x.at[:128], y.at[:128], semaphore).wait()
        elif mode == "unaligned":
            pltpu.async_copy(x.at[:64], y.at[:64], semaphore)
            pltpu.async_copy(x.at[64:], y.at[64:], semaphore)
            pltpu.make_async_copy(x, y, semaphore).wait()
        else:
            transfer = pltpu.async_copy(x.at[:128], y.at[:128], semaphore)
            if mode == "overlap":
                pltpu.async_copy(x.at[128:], y.at[:128], semaphore)
            if mode == "overwait":
                pltpu.make_async_copy(x, y, semaphore).wait()
            else:
                transfer.wait()

    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(region_view(kernel), budget=AnalysisBudget())


def test_disjoint_write_does_not_invalidate_a_pending_sources_interval_version():
    a = region_view(disjoint_reuse, arrays=1, semaphores=2)
    result = compare_program_views(a, region_view(copy))
    proof = result.evidence.left.projection
    # The full temporary storage advances to v2 while its first half remains v1.
    first = next(p for p in proof.communication if p.transfer.source == 2)
    assert first.transfer.source_version == 1
    assert first.transfer.source_versions[0].version == 1


def test_split_scratch_relay_preserves_values_and_source_versions():
    result = compare_program_views(region_view(relayed, arrays=1), region_view(copy))
    transfer = result.evidence.left.projection.communication[-1].transfer
    assert len(transfer.source_versions) == 2
    assert [v.version for v in transfer.source_versions] == [1, 2]


def test_same_read_set_does_not_erase_different_destination_correspondence():
    a = region_view(split_kernel(swapped=True))
    b = region_view(split_kernel())
    with pytest.raises(MathematicalUndecidedError):
        compare_program_views(a, b)
    # Two spellings of the same half exchange do match exactly.
    c = region_view(split_kernel(swapped=True, order=(1, 0), waits="parts"))
    assert compare_program_views(a, c).equivalence_kind == "normalized"


def test_count_evidence_and_written_intervals_are_replayed():
    a, b = region_view(split_kernel()), region_view(copy)
    evidence = compare_program_views(a, b).evidence
    projection = evidence.left.projection
    wait = projection.waits[0]
    for bad in (replace(wait, consumed_after=0), replace(wait, producers=(0,))):
        forged = replace(
            evidence, left=replace(evidence.left, projection=replace(projection, waits=(bad,)))
        )
        with pytest.raises(CorrespondenceError, match="does not replay"):
            verify_normalized_evidence(a, b, forged, budget=AnalysisBudget())


def test_large_split_uses_boundaries_not_element_enumeration():
    size = 2**24
    result = compare_program_views(
        region_view(split_kernel(size=size), size=size),
        region_view(copy, size=size),
        analysis_budget=256,
    )
    assert result.equivalence_kind == "normalized"
    with pytest.raises(ComparisonBudgetError):
        compare_program_views(region_view(split_kernel()), region_view(copy), analysis_budget=1)


def test_counter_bound_is_checked_without_allocating_tensor_data():
    size = 2**29  # float32 payload exceeds the reviewed signed 32-bit bound.
    with pytest.raises(UnsupportedJaxprError, match="counter bound"):
        normalize_view(region_view(copy, size=size), budget=AnalysisBudget())


@pytest.mark.pallas_communication
@pytest.mark.parametrize("case", ["split", "parts", "swapped", "relay", "disjoint_reuse"])
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_split_and_relay_against_official_interpreter_in_isolated_process(case, mode):
    run_oracle(case, mode)
