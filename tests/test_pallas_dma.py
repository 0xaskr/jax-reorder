"""Local DMA source facts, memory hazards, replay and independent CPU execution."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core
from jax._src.pallas import core as pallas_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jax_reorder import CorrespondenceError, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import SemaphoreType
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_oracle import run_oracle


def function(
    kernel,
    *,
    outputs=1,
    semaphores=1,
    programs=1,
    size=128,
    interpret=False,
    dtype=jnp.float32,
    arrays=0,
    manual_axes=None,
    block=128,
):
    spec = pl.BlockSpec((block,), lambda p: (p,))
    shape = jax.ShapeDtypeStruct((size,), dtype, manual_axis_type=manual_axes)
    return pl.pallas_call(
        kernel,
        out_shape=shape if outputs == 1 else (shape,) * outputs,
        grid=(programs,),
        in_specs=(spec,),
        out_specs=spec if outputs == 1 else (spec,) * outputs,
        scratch_shapes=(pltpu.VMEM((block,), dtype),) * arrays
        + (pltpu.SemaphoreType.DMA,) * semaphores,
        interpret=interpret,
    )


def view(kernel, **options):
    return adapt(
        jax.make_jaxpr(function(kernel, **options))(
            jax.ShapeDtypeStruct((options.get("size", 128),), options.get("dtype", jnp.float32))
        )
    ).view


def copy(x, y, semaphore):
    pltpu.async_copy(x, y, semaphore).wait()


def direct(x, y):
    y[:] = x[:]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.int32])
def test_local_copy_projects_to_same_values_as_load_store_with_progress_assumption(dtype):
    a, b = view(copy, dtype=dtype), view(direct, semaphores=0, dtype=dtype)
    call = a.root.operations[0].call
    assert isinstance(call.scratch[0].type.inner, SemaphoreType)
    assert call.scratch[0].initial_value == 0 and call.scratch[0].lifetime == "call"
    assert [op.dma.phase for op in call.kernel.operations] == ["start", "wait"]
    result = compare_program_views(a, b)
    assert result.equivalence_kind == "normalized"
    assert any("eventually completes" in s for s in result.assumptions)
    (proof,) = result.evidence.left.projection.communication
    assert proof.transfer.source_version == 0 and proof.destination_version == 1
    assert proof.transfer.elements == 128
    assert proof.transfer.logical_bytes == 128 * np.dtype(dtype).itemsize
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget())


def test_exact_integer_difference_also_records_dma_completion_assumption():
    def constant(offset):
        def kernel(x, y, temporary, semaphore):
            temporary[:] = jnp.broadcast_to(pl.program_id(0) + offset, (128,))
            pltpu.async_copy(temporary, y, semaphore).wait()

        return kernel

    result = compare_program_views(
        view(constant(0), arrays=1, dtype=jnp.int32), view(constant(1), arrays=1, dtype=jnp.int32)
    )
    assert result.status.value == "inequivalent" and result.evidence.kind == "index_difference"
    assert (result.evidence.coordinate, result.evidence.left, result.evidence.right) == (0, 0, 1)
    assert any("eventually completes" in s for s in result.assumptions)


def scheduled(early):
    def kernel(x, y, z, semaphore):
        transfer = pltpu.make_async_copy(x, y, semaphore)
        if early:
            transfer.start()
        z[:] = x[:] + 1
        if not early:
            transfer.start()
        transfer.wait()

    return kernel


def test_start_can_move_across_independent_work_and_source_reads():
    a, b = view(scheduled(True), outputs=2), view(scheduled(False), outputs=2)
    assert compare_program_views(a, b).equivalence_kind == "normalized"


def chained(x, y, z, semaphore):
    pltpu.async_copy(x, y, semaphore).wait()
    pltpu.async_copy(y, z, semaphore).wait()


def test_wait_publishes_version_and_semaphore_can_be_reused_after_consumption():
    def plain(x, y, z):
        y[:] = x[:]
        z[:] = y[:]

    result = compare_program_views(view(chained, outputs=2), view(plain, outputs=2, semaphores=0))
    transfers = result.evidence.left.projection.communication
    assert [p.transfer.generation for p in transfers] == [1, 2]
    assert [p.transfer.source_version for p in transfers] == [0, 1]


def concurrent(reverse_waits):
    def kernel(x, y, z, s, t):
        first = pltpu.async_copy(x, y, s)
        second = pltpu.async_copy(x, z, t)
        for transfer in (second, first) if reverse_waits else (first, second):
            transfer.wait()

    return kernel


def test_disjoint_local_transfers_can_complete_in_either_order():
    a = view(concurrent(False), outputs=2, semaphores=2)
    b = view(concurrent(True), outputs=2, semaphores=2)
    assert compare_program_views(a, b).equivalence_kind == "normalized"


def unsafe(mode):
    def kernel(x, y, z, semaphore):
        if mode == "source_write":
            y[:] = x[:]
            transfer = pltpu.async_copy(y, z, semaphore)
            y[:] = x[:] + 1
            transfer.wait()
            return
        if mode == "uninitialized_source":
            pltpu.async_copy(z, y, semaphore).wait()
            z[:] = x[:]
            return
        if mode == "input_write":
            y[:] = x[:]
            pltpu.async_copy(y, x, semaphore).wait()
            z[:] = x[:]
            return
        if mode == "overlap":
            y[:] = x[:]
            pltpu.async_copy(y, y, semaphore).wait()
            z[:] = x[:]
            return
        transfer = pltpu.make_async_copy(x, y, semaphore)
        if mode != "wait_first":
            transfer.start()
        if mode == "destination_read":
            z[:] = y[:]
        elif mode == "destination_write":
            y[:] = x[:]
        elif mode == "cumulative":
            pltpu.async_copy(x, z, semaphore)
        elif mode == "wrong_wait":
            pltpu.make_async_copy(x, z, semaphore).wait()
        if mode != "no_wait":
            transfer.wait()
        if mode == "double_wait":
            transfer.wait()
        if mode != "destination_read":
            z[:] = x[:]

    return kernel


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("source_write", "source overwritten"),
        ("uninitialized_source", "uninitialized source"),
        ("input_write", "read-only inputs"),
        ("overlap", "overlapping DMA"),
        ("wait_first", "no pending start"),
        ("destination_read", "destination accessed"),
        ("destination_write", "destination accessed"),
        ("cumulative", "cumulative DMA"),
        ("wrong_wait", "does not match"),
        ("no_wait", "pending DMA"),
        ("double_wait", "no pending start"),
    ],
)
def test_dma_hazards_and_progress_obligations_cannot_be_erased_by_projection(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(view(unsafe(mode), outputs=2), budget=AnalysisBudget())


def test_two_semaphores_do_not_protect_overlapping_destinations():
    def kernel(x, y, s, t):
        a = pltpu.async_copy(x, y, s)
        b = pltpu.async_copy(x, y, t)
        a.wait()
        b.wait()

    with pytest.raises(UnsupportedJaxprError, match="destination accessed"):
        normalize_view(view(kernel, semaphores=2), budget=AnalysisBudget())


@pytest.mark.parametrize("programs,size,reason", [(2, 256, "lifetime proof"), (1, 100, "unpadded")])
def test_grid_scratch_reuse_and_padding_need_additional_proofs(programs, size, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(view(copy, programs=programs, size=size), budget=AnalysisBudget())


@pytest.mark.parametrize("options", [{"priority": 1}, {"add": True}])
def test_nondefault_dma_options_are_source_facts_but_not_normalization_rules(options):
    def kernel(x, y, semaphore):
        pltpu.async_copy(x, y, semaphore, **options).wait()

    a = view(kernel)
    assert compare_program_views(a, a).equivalence_kind == "structural"
    with pytest.raises(UnsupportedJaxprError, match="priority/add"):
        normalize_view(a, budget=AnalysisBudget())


def test_sliced_dma_preserves_indexer_facts_but_missing_output_coverage_is_rejected():
    def kernel(x, y, semaphore):
        pltpu.async_copy(x.at[:64], y.at[:64], semaphore).wait()

    a = view(kernel)
    dma = a.root.operations[0].call.kernel.operations[0].dma
    assert dma.source_slice.start == dma.destination_slice.start == 0
    assert dma.source_slice.size == dma.destination_slice.size == 64
    with pytest.raises(UnsupportedJaxprError, match="uninitialized interval"):
        normalize_view(a, budget=AnalysisBudget())


def test_transfer_evidence_is_replayed_including_semaphore_generation():
    a, b = view(copy), view(direct, semaphores=0)
    evidence = compare_program_views(a, b).evidence
    proof = evidence.left.projection.communication[0]
    for communication in ((), (replace(proof, transfer=replace(proof.transfer, generation=2)),)):
        forged = replace(
            evidence,
            left=replace(
                evidence.left,
                projection=replace(evidence.left.projection, communication=communication),
            ),
        )
        with pytest.raises(CorrespondenceError, match="does not replay"):
            verify_normalized_evidence(a, b, forged, budget=AnalysisBudget())


def relayed(x, y, temporary, semaphore):
    pltpu.async_copy(x, temporary, semaphore).wait()
    pltpu.async_copy(temporary, y, semaphore).wait()


def reused(x, y, z, temporary, semaphore):
    pltpu.async_copy(x, temporary, semaphore).wait()
    pltpu.async_copy(temporary, y, semaphore).wait()
    temporary[:] = x[:] + 1
    pltpu.async_copy(temporary, z, semaphore).wait()


def test_local_scratch_relay_has_same_values_as_direct_transfer():
    a, b = view(relayed, arrays=1), view(copy)
    assert a.root.operations[0].call.scratch[0].initial_value is None
    assert compare_program_views(a, b).equivalence_kind == "normalized"


@pytest.mark.parametrize("check_vma", [True, False])
def test_local_dma_and_scratch_remain_shard_local_under_manual_axis_context(check_vma):
    mesh = Mesh(np.array(jax.devices("cpu")[:2]), ("d",))

    def trace(kernel, arrays):
        def local(x):
            return function(kernel, arrays=arrays, manual_axes=jax.typeof(x).manual_axis_type)(x)

        f = jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P("d"),
            out_specs=P("d"),
            check_vma=check_vma,
        )
        return adapt(jax.make_jaxpr(f)(jax.ShapeDtypeStruct((256,), jnp.float32))).view

    a, b = trace(copy, 0), trace(relayed, 1)
    scratch = b.root.operations[0].regions[0].region.operations[0].call.scratch[0]
    assert scratch.type.inner.sharding.mesh.axes == ()
    assert scratch.bound_array.sharding.mesh.axes == (("d", 2, "Manual"),)
    result = compare_program_views(a, b)
    assert result.equivalence_kind == "normalized"
    assert result.evidence.right.shard_dependencies == (((0,),),)


def test_named_scratch_mesh_is_not_silently_rebound_to_another_kernel():
    other_mesh = Mesh(np.array(jax.devices("cpu")[:2]), ("other",)).abstract_mesh
    temporary = pallas_core.MemoryRef(
        core.ShapedArray((128,), jnp.float32, sharding=NamedSharding(other_mesh, P())),
        pltpu.VMEM,
    )
    spec = pl.BlockSpec((128,), lambda p: (p,))
    f = pl.pallas_call(
        relayed,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(temporary, pltpu.SemaphoreType.DMA),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((128,), jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match="different abstract mesh"):
        adapt(source)


def test_scratch_reuse_keeps_the_version_of_each_transfer():
    def plain(x, y, z):
        y[:] = x[:]
        z[:] = x[:] + 1

    a, b = view(reused, arrays=1, outputs=2), view(plain, semaphores=0, outputs=2)
    projection = compare_program_views(a, b).evidence.left.projection
    assert [p.transfer.source_version for p in projection.communication] == [0, 1, 2]
    assert [(v.storage, v.version) for v in projection.versions] == [(3, 1), (1, 1), (3, 2), (2, 1)]


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("uninitialized", "uninitialized"),
        ("early_source_write", "source overwritten"),
        ("early_destination_read", "destination accessed"),
        ("early_destination_write", "destination accessed"),
    ],
)
def test_scratch_buffers_require_initialization_and_completed_transfers(mode, reason):
    def kernel(x, y, temporary, semaphore):
        if mode == "uninitialized":
            y[:] = temporary[:]
            return
        first = pltpu.async_copy(x, temporary, semaphore)
        if mode == "early_destination_read":
            y[:] = temporary[:]
        elif mode == "early_destination_write":
            temporary[:] = x[:]
        first.wait()
        second = pltpu.async_copy(temporary, y, semaphore)
        if mode == "early_source_write":
            temporary[:] = x[:] + 1
        second.wait()

    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(view(kernel, arrays=1), budget=AnalysisBudget())


def test_wait_for_one_reader_does_not_release_another_readers_source():
    def kernel(x, y, z, temporary, first, second):
        temporary[:] = x[:]
        a = pltpu.async_copy(temporary, y, first)
        b = pltpu.async_copy(temporary, z, second)
        a.wait()
        temporary[:] = x[:] + 1
        b.wait()

    with pytest.raises(UnsupportedJaxprError, match="source overwritten"):
        normalize_view(view(kernel, arrays=1, outputs=2, semaphores=2), budget=AnalysisBudget())


def test_many_scratch_versions_do_not_expand_into_an_execution_search():
    def kernel(x, y, temporary, semaphore):
        for _ in range(100):
            pltpu.async_copy(x, temporary, semaphore).wait()
        pltpu.async_copy(temporary, y, semaphore).wait()

    result = compare_program_views(
        view(kernel, arrays=1), view(direct, semaphores=0), analysis_budget=512
    )
    transfers = result.evidence.left.projection.communication
    assert len(transfers) == 101 and transfers[-1].transfer.source_version == 100


@pytest.mark.parametrize("execution", ["eager", "on_wait"])
@pytest.mark.parametrize("case", ["chain", "schedule", "reuse"])
@pytest.mark.pallas_communication
def test_official_tpu_interpreter_as_independent_local_dma_oracle(case, execution):
    run_oracle(case, execution)
