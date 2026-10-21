"""Remote DMA uses original tensor coordinates and exact row footprints."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import CorrespondenceError, Status, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_projection_oracle import at_point
from tests.pallas_remote_nd_cases import expected, nd_function


def graph(**options):
    shape = options.get("shape", (16, 256))
    axis = options.get("partition_axis", 0)
    shape = tuple(n * (2 if i == axis else 1) for i, n in enumerate(shape))
    return jax.make_jaxpr(nd_function(**options))(jax.ShapeDtypeStruct(shape, jnp.float32))


def projected(**options):
    return normalize_view(adapt(graph(**options)).view, budget=AnalysisBudget(100000))


def values(projection, x):
    return np.asarray(
        [at_point(projection.outputs[0], (x,), point) for point in np.ndindex(x.shape)]
    ).reshape(x.shape)


@pytest.mark.parametrize(
    "shape,partition_axis,mode",
    [
        ((16, 256), 0, "full"),
        ((16, 256), 1, "full"),
        ((2, 8, 256), 2, "full"),
        ((16, 256), 1, "columns"),
        ((2, 8, 256), 1, "columns"),
        ((16, 256), 0, "reshape"),
        ((2, 8, 128), 1, "transpose"),
        ((16, 256), 1, "ref_access"),
        ((16, 256), 1, "local_dma"),
    ],
)
@pytest.mark.parametrize("wrong", [False, True])
def test_nd_original_storage_complete_projection(shape, partition_axis, mode, wrong):
    options = dict(shape=shape, partition_axis=partition_axis, mode=mode, wrong=wrong)
    normalized = projected(**options)
    global_shape = tuple(n * (2 if i == partition_axis else 1) for i, n in enumerate(shape))
    x = np.arange(np.prod(global_shape), dtype=np.float32).reshape(global_shape)
    np.testing.assert_array_equal(values(normalized.projection, x), expected(x, **options))
    assert len(normalized.remote.transfers) == (4 if mode == "local_dma" else 2)


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("no_wait", "completion order"),
        ("gap", "uninitialized"),
        ("uninitialized", "uninitialized|completion order"),
        ("inflight_overlap", "completion order"),
        ("shared_dma_counter", "completion order"),
    ],
)
def test_nd_invalid_memory_and_sync(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        projected(mode=mode)


def test_nd_holes_are_not_false_conflicts():
    normalized = projected(mode="inflight_hole")
    starts = [
        c for p in normalized.remote.protocol.programs for c in p.commands if hasattr(c, "transfer")
    ]
    assert all(len(c.transfer.source_spans) == 16 for c in starts)
    assert all(c.transfer.logical_bytes == 16 * 128 * 4 for c in starts)


@pytest.mark.parametrize(
    "shape,block_shape,partition_axis",
    [((16, 256), (8, 256), 0), ((16, 256), (16, 128), 1), ((2, 8, 256), (1, 8, 256), 2)],
)
def test_nd_staging_windows(shape, block_shape, partition_axis):
    options = dict(shape=shape, block_shape=block_shape, partition_axis=partition_axis)
    normalized = projected(**options)
    global_shape = tuple(n * (2 if i == partition_axis else 1) for i, n in enumerate(shape))
    x = np.arange(np.prod(global_shape), dtype=np.float32).reshape(global_shape)
    np.testing.assert_array_equal(values(normalized.projection, x), expected(x, **options))


def test_nd_protocol_and_values_replay():
    a, b = (adapt(graph(mode="columns", receive_first=order)).view for order in (False, True))
    result = compare_program_views(a, b, analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget(100000))
    left = result.evidence.left
    forged = replace(
        left.remote,
        assembly=tuple(
            replace(p, offsets=tuple(0 for _ in p.offsets)) for p in left.remote.assembly
        ),
    )
    with pytest.raises(CorrespondenceError, match="replay"):
        verify_normalized_evidence(
            a,
            b,
            replace(result.evidence, left=replace(left, remote=forged)),
            budget=AnalysisBudget(100000),
        )


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case", ["matrix", "tensor", "columns", "wrong_columns", "tensor_columns", "staging", "hole"]
)
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_nd_official_cpu_interpreter(case, mode):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.pallas_remote_nd_cases",
        case,
        mode,
        entry="run_cpu",
        expected=dict(case=case, mode=mode, status="ok"),
    )


@pytest.mark.parametrize("field", ["footprint", "lane_map"])
def test_nd_replay_rejects_forged_access_and_lane_maps(field):
    from jax_reorder.core.index_domain import integer
    from jax_reorder.core.protocol import Access, Start

    a, b = (adapt(graph(mode="columns", receive_first=order)).view for order in (False, True))
    result = compare_program_views(a, b, analysis_budget=100000)
    left = result.evidence.left
    remote = left.remote
    protocol = remote.protocol
    programs = list(protocol.programs)
    commands = list(programs[0].commands)
    pos = next(
        i
        for i, c in enumerate(commands)
        if isinstance(c, Access if field == "footprint" else Start)
    )
    command = commands[pos]
    commands[pos] = (
        replace(command, footprint=(command.span,))
        if field == "footprint"
        else replace(command, transfer=replace(command.transfer, destination_to_source=integer(0)))
    )
    # Pick an actual holey ordinary access; a full input staging read is already contiguous.
    if field == "footprint":
        pos = next(
            i
            for i, c in enumerate(programs[0].commands)
            if isinstance(c, Access) and len(c.spans) > 1
        )
        command = programs[0].commands[pos]
        commands[pos] = replace(command, footprint=(command.span,))
    programs[0] = replace(programs[0], commands=tuple(commands))
    altered = replace(remote, protocol=replace(protocol, programs=tuple(programs)))
    with pytest.raises(CorrespondenceError, match="replay"):
        verify_normalized_evidence(
            a,
            b,
            replace(result.evidence, left=replace(left, remote=altered)),
            budget=AnalysisBudget(100000),
        )


@pytest.mark.parametrize(
    "footprint", [((-1, 1),), ((0, 0),), ((0, 2), (1, 3)), ((2, 3), (0, 1)), ((0, 5),)]
)
def test_nd_ordinary_access_rejects_invalid_footprints(footprint):
    from jax_reorder.core.model import Origin
    from jax_reorder.core.protocol import Access, DeviceProgram, Instance, analyze_protocol
    from jax_reorder.core.storage import Span

    command = Access(
        Instance(0, 0),
        Span(0, 4),
        True,
        Origin("invalid.footprint"),
        tuple(Span(a, b) for a, b in footprint),
    )
    with pytest.raises(UnsupportedJaxprError, match="ordinary access footprint"):
        analyze_protocol((DeviceProgram(0, (command,)),), (), budget=AnalysisBudget())


def test_nd_large_contiguous_and_column_tiles_never_enumerate_lanes():
    from jax_reorder.core.model import SliceIndex, TensorIndex
    from jax_reorder.core.remote_coordinates import remote_view

    budgets = []
    for width in (256, 2**24):
        budget = AnalysisBudget(128)
        shape = (16, width)
        selected = TensorIndex((SliceIndex(0, 16, 1), SliceIndex(0, 128, 1)), shape)
        view = remote_view(shape, selected, (), lambda operand: 0, budget, "large.column")
        assert len(view.spans) == 16 and sum(s.size for s in view.spans) == 16 * 128
        budgets.append(budget.used)
    assert budgets[0] == budgets[1]


def padded_matrix(*, wrong=False, mesh=None):
    def kernel(x, y, send, recv, ack):
        peer = 1 - jax.lax.axis_index("d")
        pl.semaphore_signal(ack, 1, device_id=(peer,))
        pl.semaphore_wait(ack, 1)

        @pl.when(pl.program_id(0) == 0)
        def full():
            tpu.async_remote_copy(x, y, send, recv, (peer,)).wait()

        @pl.when(pl.program_id(0) == 1)
        def tail():
            if wrong:
                tpu.async_remote_copy(x, y, send, recv, (peer,)).wait()
            else:
                tpu.async_remote_copy(x.at[:, :128], y.at[:, :128], send, recv, (peer,)).wait()

    block = pl.BlockSpec((8, 256), lambda p: (0, p))
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 384), jnp.float32),
        grid=(2,),
        in_specs=(block,),
        out_specs=block,
        scratch_shapes=(tpu.SemaphoreType.DMA, tpu.SemaphoreType.DMA, tpu.SemaphoreType.REGULAR),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
    )
    mesh = mesh or Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",))
    return jax.shard_map(
        call, mesh=mesh, in_specs=P(None, "d"), out_specs=P(None, "d"), check_vma=False
    )


def test_nd_remote_tail_does_not_initialize_or_transfer_padding():
    graph = jax.make_jaxpr(padded_matrix())(jax.ShapeDtypeStruct((8, 768), jnp.float32))
    proof = normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))
    x = np.arange(8 * 768, dtype=np.float32).reshape(8, 768)
    result = np.asarray(
        [at_point(proof.projection.outputs[0], (x,), point) for point in np.ndindex(x.shape)]
    ).reshape(x.shape)
    np.testing.assert_array_equal(result, np.concatenate((x[:, 384:], x[:, :384]), axis=1))
    assert len(proof.remote.transfers) == 4
    output = proof.remote.protocol.io.ports[1]
    assert len(output.segments(1)) == 8
    assert sum(a.size for a, _ in output.segments(1)) == 8 * 128


def test_nd_remote_full_tail_dma_rejects_uninitialized_padding():
    graph = jax.make_jaxpr(padded_matrix(wrong=True))(jax.ShapeDtypeStruct((8, 768), jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))
