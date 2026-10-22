"""Exact ordinary mask footprints must preserve remote ordering and initialized lanes."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_pallas_kernels,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    _call,
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.protocol import Access
from jax_reorder.core.remote_memory import functionalize_remote_call, same_remote_evidence
from jax_reorder.core.storage import Span
from tests.pallas_projection_oracle import at_point
from tests.pallas_remote_mask_cases import expected, masked_function


def trace(mode="store", **options):
    return jax.make_jaxpr(masked_function(mode, **options))(
        jax.ShapeDtypeStruct((16, 256), jnp.float32)
    )


def graph(mode="store", **options):
    return adapt(trace(mode, **options)).view


def projection(mode="store", **options):
    return normalize_view(graph(mode, **options), budget=AnalysisBudget(100000))


@pytest.mark.parametrize(
    "mode",
    [
        "store",
        "swap",
        "vector_other",
        "transpose_store",
        "holes",
        "load",
        "where",
        "device_mask",
        "disjoint",
        "empty",
        "empty_data",
        "partial_init",
        "squeezed",
        "transpose",
        "data_load",
        "data_store",
        "data_where",
        "conditional_store",
        "conditional_slices",
        "conditional_read",
        "conditional_dma_complete",
    ],
)
@pytest.mark.parametrize("identity_indices", [False, True])
def test_remote_mask_complete_numpy_projection(mode, identity_indices):
    proof = projection(mode, identity_indices=identity_indices)
    x = np.arange(4096, dtype=np.float32).reshape(16, 256)
    result = np.asarray(
        [at_point(proof.projection.outputs[0], (x,), p) for p in np.ndindex(x.shape)]
    ).reshape(x.shape)
    np.testing.assert_array_equal(result, expected(x, mode))


@pytest.mark.parametrize("mode", ["conditional_store", "conditional_slices"])
def test_conditional_remote_initialization_reaches_public_comparison(mode):
    source = trace(mode)
    assert compare_pallas_kernels(source, source, check_protocol=True).protocol_checked
    assert compare_algebraic_jaxprs(trace("data_store"), source).status is Status.EQUIVALENT


def test_conditional_remote_initialization_keeps_the_actual_second_write():
    result = compare_algebraic_jaxprs(trace("conditional_store"), trace("conditional_wrong"))
    assert result.status is Status.INEQUIVALENT


@pytest.mark.parametrize(
    "mode",
    [
        "conditional_gap",
        "conditional_same",
        "conditional_read_wrong",
        "conditional_overwritten_read",
        "conditional_mask",
        "conditional_control",
        "conditional_index",
        "conditional_dma",
        "conditional_missing_wait",
    ],
)
def test_conditional_initialization_cannot_hide_an_actual_undefined_consumer(mode):
    with pytest.raises(UnsupportedJaxprError):
        projection(mode)


def local_projection(mode="conditional_store", **options):
    return _local_projection(graph(mode, **options))


def _local_projection(view):
    call, boundaries = _call(view)
    boundary = next(b for b in boundaries if b.shard_map is not None)
    return functionalize_remote_call(
        call,
        boundary.shard_map,
        boundary.inputs,
        boundary.outputs,
        scope="device_local",
        budget=AnalysisBudget(100000),
    )


@pytest.mark.parametrize("mode", ["data_store", "conditional_store"])
def test_remote_local_outputs_do_not_claim_parent_replication(mode):
    local, proof = local_projection(mode, replicated_output=True)
    partitioned, _ = local_projection(mode)
    assert proof.local_outputs == ((0, 0), (0, 1))
    assert local.outputs == partitioned.outputs
    x = np.arange(4096, dtype=np.float32).reshape(16, 256)
    expected_shards = np.split(expected(x, mode), 2)
    for output, reference in zip(local.outputs, expected_shards, strict=True):
        actual = np.asarray(
            [at_point(output, (x,), point) for point in np.ndindex((8, 256))]
        ).reshape(8, 256)
        np.testing.assert_array_equal(actual, reference)
    # The separate full-function contract still checks claimed replication.
    with pytest.raises(UnsupportedJaxprError, match="replicated output values differ"):
        projection(mode, replicated_output=True)
    assert not same_remote_evidence(proof, replace(proof, local_outputs=((0, 1), (0, 0))))


def test_remote_local_outputs_keep_the_received_peer_identity():
    correct, left = local_projection()
    wrong, right = local_projection(wrong_peer=True)
    x = np.arange(4096, dtype=np.float32).reshape(16, 256)
    reference = np.where(x > 2500, x + 100, x)
    for output, shard in zip(wrong.outputs, np.split(reference, 2), strict=True):
        actual = np.asarray(
            [at_point(output, (x,), point) for point in np.ndindex((8, 256))]
        ).reshape(8, 256)
        np.testing.assert_array_equal(actual, shard)
    assert at_point(correct.outputs[0], (x,), (0, 0)) != at_point(wrong.outputs[0], (x,), (0, 0))
    assert not same_remote_evidence(left, right)


def test_remote_local_outputs_preserve_port_then_device_order():
    from jax.experimental.pallas import tpu

    from tests.pallas_remote_cases import projection_array, remote_graph

    def kernel(x, y, z, send, recv):
        tpu.async_remote_copy(x, y, send, recv, (1 - jax.lax.axis_index("d"),)).wait()
        z[:] = x[:] + 1

    local, proof = _local_projection(adapt(remote_graph(kernel, outputs=2)).view)
    assert proof.local_outputs == ((0, 0), (0, 1), (1, 0), (1, 1))
    x = np.arange(256, dtype=np.float32)
    references = (x[128:], x[:128], x[:128] + 1, x[128:] + 1)
    for output, reference in zip(local.outputs, references, strict=True):
        actual = projection_array(output, (x,), 128)
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("overlap", "completion order"),
        ("no_wait", "completion order|unfinished|unconsumed|no consumer"),
        ("empty_no_wait", "completion order|unfinished|unconsumed|no consumer"),
        ("uninitialized", "uninitialized"),
        ("leak", "undefined"),
        ("swap_leak", "undefined"),
    ],
)
def test_remote_mask_rejects_unsafe_active_accesses(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        projection(mode)


def test_remote_mask_empty_accesses_do_not_create_protocol_events():
    proof = projection("empty")
    assert len(proof.remote.transfers) == 2
    assert all(
        len([c for c in program.commands if isinstance(c, Access)]) == 4
        for program in proof.remote.protocol.programs
    )


def test_remote_mask_evidence_replays_exact_footprint():
    a, b = (graph("store", receive_first=first) for first in (False, True))
    result = compare_program_views(a, b, analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget(100000))
    left = result.evidence.left
    remote = left.remote
    programs = list(remote.protocol.programs)
    commands = list(programs[0].commands)
    index = next(
        i
        for i, c in enumerate(commands)
        if isinstance(c, Access) and len(c.footprint) == 8 and c.footprint[0].size == 192
    )
    commands[index] = replace(commands[index], footprint=(Span(0, 2048),))
    programs[0] = replace(programs[0], commands=tuple(commands))
    forged = replace(remote, protocol=replace(remote.protocol, programs=tuple(programs)))
    with pytest.raises(CorrespondenceError, match="replay"):
        verify_normalized_evidence(
            a,
            b,
            replace(result.evidence, left=replace(left, remote=forged)),
            budget=AnalysisBudget(100000),
        )


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case",
    [
        "store",
        "holes",
        "disjoint",
        "data_store",
        "tail",
        "conditional_store",
        "conditional_slices",
        "conditional_dma_complete",
        "conditional_wrong",
    ],
)
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_remote_masks_official_cpu_interpreter(case, mode):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.pallas_remote_mask_cases",
        case,
        mode,
        entry="run_cpu",
        expected=dict(case=case, mode=mode, status="ok"),
    )


@pytest.mark.parametrize("mode", ["tail", "tail_mask"])
def test_remote_input_padding_is_eliminated_before_full_dma(mode):
    from tests.pallas_remote_mask_cases import tail_function

    x = np.arange(8 * 768, dtype=np.float32).reshape(8, 768)
    source = adapt(jax.make_jaxpr(tail_function(mode))(x)).view
    proof = normalize_view(source, budget=AnalysisBudget(100000))
    result = np.asarray(
        [at_point(proof.projection.outputs[0], (x,), p) for p in np.ndindex(x.shape)]
    ).reshape(x.shape)
    np.testing.assert_array_equal(result, np.concatenate((x[:, 384:], x[:, :384]), axis=1))
    assert len(proof.remote.transfers) == 4


def test_remote_input_padding_cannot_escape_into_dma_source():
    from tests.pallas_remote_mask_cases import tail_function

    source = adapt(
        jax.make_jaxpr(tail_function("tail_leak"))(jax.ShapeDtypeStruct((8, 768), jnp.float32))
    ).view
    with pytest.raises(UnsupportedJaxprError, match="undefined"):
        normalize_view(source, budget=AnalysisBudget(100000))


def test_remote_constant_mask_loaded_from_memory_keeps_conservative_event_alignment():
    from jax._src.pallas import primitives as pp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    shape = (8, 256)
    spec = pl.BlockSpec(shape, lambda _: (0, 0))

    def kernel(x, y, known_mask, send, recv):
        tpu.async_remote_copy(x, y, send, recv, (1 - jax.lax.axis_index("d"),)).wait()
        known_mask[...] = jnp.zeros(shape, jnp.bool_)
        y[...] = pp.load(y, (slice(None), slice(None)), mask=known_mask[...], other=7)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.VMEM(shape, jnp.bool_), tpu.SemaphoreType.DMA, tpu.SemaphoreType.DMA),
        compiler_params=tpu.CompilerParams(),
    )
    f = jax.shard_map(
        call,
        mesh=Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)),
        in_specs=P("d", None),
        out_specs=P("d", None),
        check_vma=False,
    )
    x = np.arange(4096, dtype=np.float32).reshape(16, 256)
    proof = normalize_view(adapt(jax.make_jaxpr(f)(x)).view, budget=AnalysisBudget(100000))
    output = np.asarray(
        [at_point(proof.projection.outputs[0], (x,), p) for p in np.ndindex(x.shape)]
    )
    np.testing.assert_array_equal(output, np.full(x.size, 7, np.float32))
