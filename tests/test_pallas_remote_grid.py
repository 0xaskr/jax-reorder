"""Remote grid expansion preserves all coordinates and only real program order."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import (
    CorrespondenceError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_projection_oracle import at_point
from tests.pallas_remote_cases import projection_array
from tests.pallas_remote_grid_cases import LOCAL_SIZE, binding, expected, graph


def normalized(**options):
    return normalize_view(
        adapt(graph(**options)).view,
        budget=AnalysisBudget(100000),
        **(
            {"execution": binding(**options)}
            if options.get("parallel") or options.get("all_parallel")
            else {}
        ),
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("flat,transpose_grid", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("receive_first", [False, True])
def test_serial_multidimensional_grid_complete_projection(
    explicit, flat, transpose_grid, receive_first
):
    proof = normalized(
        explicit=explicit, flat=flat, transpose_grid=transpose_grid, receive_first=receive_first
    )
    x = np.arange(2 * LOCAL_SIZE, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), x.size), expected(x)
    )
    assert len(proof.remote.transfers) == (24 if explicit else 12)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("transpose_grid", [False, True])
def test_serial_grid_flattening_and_wait_order_replay(explicit, transpose_grid):
    left = adapt(graph(explicit=explicit, transpose_grid=transpose_grid)).view
    right = adapt(graph(explicit=explicit, flat=True, receive_first=True)).view
    result = compare_program_views(left, right, analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("parallel", [False, True])
def test_three_dimensional_grid_binds_every_program_axis(parallel):
    left = adapt(graph(three_dimensional=True, parallel=parallel)).view
    right = adapt(graph(flat=True, explicit=parallel, receive_first=True)).view
    result = compare_program_views(
        left,
        right,
        analysis_budget=100000,
        **(
            {"execution": (binding(three_dimensional=True), binding(flat=True))} if parallel else {}
        ),
    )
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("mode", ["wrong_index_map", "self_target"])
@pytest.mark.parametrize("parallel", [False, True])
def test_remote_grid_wrong_coordinate_or_peer_changes_values(mode, parallel):
    left, right = (adapt(graph(parallel=parallel, mode=m)).view for m in ("normal", mode))
    execution = {"execution": binding()} if parallel else {}
    with pytest.raises(MathematicalUndecidedError):
        compare_program_views(left, right, analysis_budget=100000, **execution)
    x = np.arange(2 * LOCAL_SIZE, dtype=np.float32)
    outputs = []
    for view, variant in ((left, "normal"), (right, mode)):
        proof = normalize_view(view, budget=AnalysisBudget(100000), **execution)
        output = projection_array(proof.projection.outputs[0], (x,), x.size)
        np.testing.assert_array_equal(output, expected(x, mode=variant))
        outputs.append(output)
    assert not np.array_equal(*outputs)


@pytest.mark.parametrize("mode", ["missing_send", "missing_receive"])
@pytest.mark.parametrize("parallel", [False, True])
def test_remote_grid_does_not_invent_missing_dma_completion(mode, parallel):
    with pytest.raises(UnsupportedJaxprError):
        normalized(mode=mode, parallel=parallel)


@pytest.mark.parametrize("transpose_grid", [False, True])
@pytest.mark.parametrize("receive_first", [False, True])
@pytest.mark.parametrize("all_parallel", [False, True])
def test_parallel_owners_with_serial_inner_work_have_independent_streams(
    transpose_grid, receive_first, all_parallel
):
    proof = normalized(
        parallel=True,
        all_parallel=all_parallel,
        transpose_grid=transpose_grid,
        receive_first=receive_first,
    )
    x = np.arange(2 * LOCAL_SIZE, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), x.size), expected(x)
    )
    streams = proof.remote.protocol.programs
    stream_count = 12 if all_parallel else 4
    assert len(streams) == stream_count
    assert len({(program.device, program.stream) for program in streams}) == stream_count
    protocol = proof.remote.protocol.protocol
    for a in protocol.events:
        for b in protocol.events:
            if a.device == b.device and a.stream != b.stream:
                assert not protocol.precedes(a.id, b.id)


@pytest.mark.parametrize("transpose_grid", [False, True])
@pytest.mark.parametrize("all_parallel", [False, True])
def test_parallel_grid_is_equivalent_to_explicit_serial_execution(transpose_grid, all_parallel):
    left = adapt(
        graph(parallel=True, all_parallel=all_parallel, transpose_grid=transpose_grid)
    ).view
    right = adapt(graph(flat=True, explicit=True, receive_first=True)).view
    result = compare_program_views(
        left,
        right,
        analysis_budget=100000,
        execution=(binding(transpose_grid=transpose_grid), binding(flat=True)),
    )
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("mode", ["overlapping_output", "shared_scratch"])
@pytest.mark.parametrize("transpose_grid", [False, True])
def test_parallel_grid_neither_serializes_conflicts_nor_privatizes_real_storage(
    mode, transpose_grid
):
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        normalized(parallel=True, mode=mode, transpose_grid=transpose_grid)


def test_remote_parallel_stream_identity_is_replayed():
    left = adapt(graph(parallel=True)).view
    right = adapt(graph(parallel=True, receive_first=True)).view
    result = compare_program_views(left, right, analysis_budget=100000, execution=binding())
    proof = result.evidence.left
    programs = proof.remote.protocol.programs
    forged_programs = tuple(replace(program, stream=0) for program in programs)
    forged = replace(
        result.evidence,
        left=replace(
            proof,
            remote=replace(
                proof.remote, protocol=replace(proof.remote.protocol, programs=forged_programs)
            ),
        ),
    )
    with pytest.raises(CorrespondenceError, match="replay"):
        verify_normalized_evidence(left, right, forged, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("parallel,transpose_grid", [(False, False), (True, False), (True, True)])
def test_dma_waits_can_complete_starts_from_previous_grid_iterations(parallel, transpose_grid):
    options = dict(parallel=parallel, transpose_grid=transpose_grid, deferred=True)
    left = adapt(graph(**options)).view
    right = adapt(graph(flat=True, explicit=True, receive_first=True)).view
    result = compare_program_views(
        left,
        right,
        analysis_budget=100000,
        **({"execution": (binding(**options), binding(flat=True))} if parallel else {}),
    )
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))


def test_deferred_grid_dma_still_requires_every_receive_completion():
    with pytest.raises(UnsupportedJaxprError, match="semaphore count has no consumer"):
        normalized(parallel=True, deferred=True, mode="missing_receive")


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case",
    [
        "serial",
        "flat",
        "serial_transposed",
        "three_dimensional",
        "parallel",
        "all_parallel",
        "parallel_transposed",
        "wrong_index_map",
        "self_target",
        "deferred",
    ],
)
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_remote_grid_official_cpu_interpreter(case, mode):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.pallas_remote_grid_cases",
        case,
        mode,
        entry="run_cpu",
        expected=dict(
            case=case,
            mode=mode,
            status="interpreter_grid_limitation" if case == "parallel_transposed" else "ok",
        ),
    )


def source(*, flat=False, revisit=False, initialize=True, wrong=False):
    def point(*coordinates):
        return (coordinates[0] // 3, coordinates[0] % 3) if flat else coordinates

    def kernel(x, y, temporary, send, receive, ack):
        i, j = point(pl.program_id(0)) if flat else (pl.program_id(0), pl.program_id(1))
        peer = 1 - jax.lax.axis_index("d")
        pl.semaphore_signal(ack, 1, device_id=(peer,))
        pl.semaphore_wait(ack, 1)
        if revisit:
            if initialize:

                @pl.when(j < 2)
                def first_visit():
                    y[...] = jnp.zeros((8, 128), jnp.float32)

            tpu.async_remote_copy(x, temporary, send, receive, (peer,)).wait()
            y[...] += temporary[...]
        else:
            tpu.async_remote_copy(x, y, send, receive, (peer,)).wait()

    def input_map(*coordinates):
        i, j = point(*coordinates)
        return i, (j + int(wrong)) % 3

    def output_map(*coordinates):
        i, j = point(*coordinates)
        return i, j % 2 if revisit else j

    output_shape = (16, 256 if revisit else 384)
    out = pl.BlockSpec(
        (8, 128),
        output_map,
        pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY) if revisit else None,
    )
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, jnp.float32),
        grid=(6,) if flat else (2, 3),
        in_specs=(pl.BlockSpec((8, 128), input_map),),
        out_specs=out,
        scratch_shapes=(
            tpu.VMEM((8, 128), jnp.float32),
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.REGULAR,
        ),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",) * (1 if flat else 2)),
    )
    mapped = jax.shard_map(
        call,
        mesh=Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)),
        in_specs=P("d"),
        out_specs=P("d"),
        check_vma=False,
    )
    return adapt(jax.make_jaxpr(mapped)(jax.ShapeDtypeStruct((32, 384), jnp.float32))).view


@pytest.mark.parametrize("revisit", [False, True])
@pytest.mark.parametrize("wrong", [False, True])
def test_multidimensional_staged_output_values_and_flattening(revisit, wrong):
    views = [source(revisit=revisit, wrong=wrong, flat=flat) for flat in (False, True)]
    result = compare_program_views(*views, analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    proof = normalize_view(views[0], budget=AnalysisBudget(100000))
    x = np.arange(32 * 384, dtype=np.float32).reshape(32, 384)
    incoming = x.reshape(2, 16, 3, 128)[::-1]
    if wrong:
        incoming = np.roll(incoming, -1, axis=2)
    expected = (
        np.stack((incoming[:, :, 0] + incoming[:, :, 2], incoming[:, :, 1]), axis=2)
        if revisit
        else incoming
    ).reshape(32, -1)
    actual = np.asarray(
        [at_point(proof.projection.outputs[0], (x,), point) for point in np.ndindex(expected.shape)]
    ).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    if revisit:
        assert (
            sum(
                "restore_read" in command.origin.path
                for program in proof.remote.protocol.programs
                for command in program.commands
            )
            == 4
        )


def test_multidimensional_restoration_does_not_initialize_first_visit():
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(source(revisit=True, initialize=False), budget=AnalysisBudget(100000))
