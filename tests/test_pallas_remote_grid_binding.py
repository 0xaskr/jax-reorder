"""Remote grid ownership, execution conditions and whole-core protocol progress."""

from dataclasses import replace
from itertools import product
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_function_traces,
    compare_functions,
    compare_pallas_kernels,
    trace_function,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.core_execution import CoreExecutionBinding
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import GridAxis
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.remote import analyze_remote_call
from jax_reorder.core.remote_grid import plan_remote_grid
from tests.pallas_remote_cases import projection_array
from tests.pallas_remote_grid_cases import LOCAL_SIZE, binding, function, graph


@pytest.fixture(scope="module")
def parallel_sources():
    return graph(parallel=True), graph(parallel=True, receive_first=True)


def normalized_pair(sources):
    left, right = (adapt(source).view for source in sources)
    result = compare_program_views(left, right, analysis_budget=100000, execution=binding())
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    return left, right, result


@pytest.mark.parametrize("cores", [None, 1])
def test_parallel_remote_requires_an_ordered_binding_for_the_resource_proof(
    parallel_sources, cores
):
    source, reordered = parallel_sources
    assert compare_pallas_kernels(source, source).status is Status.EQUIVALENT
    with pytest.raises(UnsupportedJaxprError, match="binding|execution"):
        compare_pallas_kernels(
            source, source, check_protocol=True, analysis_budget=100000, execution_cores=cores
        )
    with pytest.raises(UnsupportedJaxprError, match="binding|execution"):
        normalize_view(adapt(source).view, budget=AnalysisBudget(100000), execution_cores=cores)
    with pytest.raises(UnsupportedJaxprError, match="binding|execution"):
        compare_program_views(
            adapt(source).view, adapt(reordered).view, analysis_budget=100000, execution_cores=cores
        )


def test_public_local_entry_records_and_replays_remote_program_order(parallel_sources):
    source = parallel_sources[0]
    result = compare_pallas_kernels(
        source, source, check_protocol=True, execution=binding(), analysis_budget=100000
    )
    assert result.status is Status.EQUIVALENT and result.protocol_checked
    assert result.execution == binding() and result.execution_cores is None
    assert any("every device" in assumption for assumption in result.assumptions)
    for proof in (result.left_protocol, result.right_protocol):
        assert proof.execution == proof.remote.protocol.execution == binding()
        assert proof.execution_cores is proof.remote.protocol.execution_cores is None
    verify_pallas_kernel_comparison(source, source, result, analysis_budget=100000)


def test_normalized_remote_comparison_records_and_replays_program_order(parallel_sources):
    left, right, result = normalized_pair(parallel_sources)
    assert any("caller binds" in assumption for assumption in result.assumptions)
    for proof in (result.evidence.left, result.evidence.right):
        assert proof.execution == proof.remote.protocol.execution == binding()
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("binding", [False, True, 0, 2, 1.0])
def test_normalization_rejects_invalid_core_conditions_before_structural_shortcut(
    parallel_sources, binding
):
    view = adapt(parallel_sources[0]).view
    with pytest.raises(ValueError, match="single-core"):
        normalize_view(view, budget=AnalysisBudget(100000), execution_cores=binding)
    with pytest.raises(ValueError, match="single-core"):
        compare_program_views(view, view, execution_cores=binding)


@pytest.mark.parametrize("field", ["rewrite", "remote", "boolean", "order", "all_conditions"])
def test_normalized_replay_rejects_omitted_or_forged_execution_conditions(parallel_sources, field):
    left, right, result = normalized_pair(parallel_sources)

    def alter(proof):
        if field == "rewrite":
            return replace(proof, execution=None)
        condition = (
            True
            if field == "boolean"
            else CoreExecutionBinding((tuple(reversed(binding().programs[0])),))
            if field == "order"
            else None
        )
        altered = replace(
            proof,
            remote=replace(
                proof.remote, protocol=replace(proof.remote.protocol, execution=condition)
            ),
        )
        return replace(altered, execution=None) if field == "all_conditions" else altered

    evidence = replace(
        result.evidence,
        left=alter(result.evidence.left),
        right=alter(result.evidence.right) if field == "all_conditions" else result.evidence.right,
    )
    with pytest.raises(CorrespondenceError, match="conditions|does not replay"):
        verify_normalized_evidence(left, right, evidence, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("field", ["result", "rewrite", "remote", "boolean", "assumptions"])
def test_public_entry_replay_rejects_execution_condition_tampering(parallel_sources, field):
    source = parallel_sources[0]
    result = compare_pallas_kernels(
        source, source, check_protocol=True, execution=binding(), analysis_budget=100000
    )
    proof = result.left_protocol
    if field == "result":
        forged = replace(result, execution=None)
    elif field == "assumptions":
        forged = replace(result, assumptions=())
    elif field == "rewrite":
        forged = replace(result, left_protocol=replace(proof, execution=None))
    else:
        protocol = replace(proof.remote.protocol, execution=True if field == "boolean" else None)
        forged = replace(
            result, left_protocol=replace(proof, remote=replace(proof.remote, protocol=protocol))
        )
    with pytest.raises(CorrespondenceError):
        verify_pallas_kernel_comparison(source, source, forged, analysis_budget=100000)


def test_local_core_mapping_does_not_silently_bind_remote_core_routes(parallel_sources):
    source = parallel_sources[0]
    two_cores = CoreExecutionBinding((((0, 0), (0, 1), (0, 2)), ((1, 0), (1, 1), (1, 2))))
    with pytest.raises(UnsupportedJaxprError, match="single.core|one core"):
        compare_pallas_kernels(source, source, check_protocol=True, execution=two_cores)
    with pytest.raises(UnsupportedJaxprError, match="single.core|one core"):
        compare_algebraic_jaxprs(source, source, execution=two_cores)


def test_serial_remote_keeps_unconditional_default_and_accepts_explicit_single_core():
    source = graph()
    for cores in (None, 1):
        result = compare_pallas_kernels(
            source, source, check_protocol=True, execution_cores=cores, analysis_budget=100000
        )
        assert result.execution_cores is cores
        assert result.left_protocol.execution_cores is cores
        assert result.left_protocol.remote.protocol.execution_cores is cores
        verify_pallas_kernel_comparison(source, source, result, analysis_budget=100000)


def test_distinct_source_grids_replay_their_own_explicit_program_streams():
    left, right = (adapt(graph(**options)).view for options in ({"parallel": True}, {"flat": True}))
    executions = binding(), binding(flat=True)
    result = compare_program_views(left, right, analysis_budget=100000, execution=executions)
    assert result.equivalence_kind == "normalized"
    assert (result.evidence.left.execution, result.evidence.right.execution) == executions
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))
    swapped = replace(
        result.evidence,
        left=replace(result.evidence.left, execution=executions[1]),
        right=replace(result.evidence.right, execution=executions[0]),
    )
    with pytest.raises(CorrespondenceError, match="conditions"):
        verify_normalized_evidence(left, right, swapped, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("kind", ["duplicate", "missing", "reversed", "wrong_grid"])
def test_remote_execution_stream_must_cover_the_exact_grid_in_order(parallel_sources, kind):
    points = binding().programs[0]
    points = {
        "duplicate": (*points[:-1], points[0]),
        "missing": points[:-1],
        "reversed": tuple(reversed(points)),
        "wrong_grid": binding(flat=True).programs[0],
    }[kind]
    source = parallel_sources[0]
    view = adapt(source).view
    condition = CoreExecutionBinding((points,))
    for check in (
        lambda: normalize_view(view, budget=AnalysisBudget(100000), execution=condition),
        lambda: compare_program_views(view, view, analysis_budget=100000, execution=condition),
        lambda: compare_algebraic_jaxprs(
            source, source, algebra_budget=100000, execution=condition
        ),
    ):
        with pytest.raises(UnsupportedJaxprError, match="binding|grid"):
            check()


def test_execution_mapping_and_core_count_are_mutually_exclusive(parallel_sources):
    view = adapt(parallel_sources[0]).view
    with pytest.raises(ValueError, match="no execution_cores"):
        normalize_view(view, budget=AnalysisBudget(100000), execution=binding(), execution_cores=1)
    with pytest.raises(ValueError, match="no execution_cores"):
        compare_program_views(view, view, execution=binding(), execution_cores=1)


@pytest.mark.parametrize("execution", [True, (), (binding(),), (binding(), True)])
def test_comparison_rejects_malformed_execution_pairs(parallel_sources, execution):
    view = adapt(parallel_sources[0]).view
    with pytest.raises(ValueError, match="binding pair"):
        compare_program_views(view, view, execution=execution)
    with pytest.raises(ValueError, match="binding pair"):
        compare_algebraic_jaxprs(parallel_sources[0], parallel_sources[0], execution=execution)


@pytest.mark.parametrize("entry", ["jaxpr", "function", "trace"])
@pytest.mark.parametrize("same_source", [False, True])
def test_public_algebra_entries_preserve_remote_execution_proofs(entry, same_source, monkeypatch):
    left = function(parallel=True)
    right = left if same_source else function(flat=True, explicit=True, receive_first=True)
    argument = jax.ShapeDtypeStruct((2 * LOCAL_SIZE,), jnp.float32)
    executions = binding() if same_source else (binding(), binding(flat=True))
    if same_source:

        def unexpected_cas(*args, **kwargs):
            raise AssertionError("structural equality with a checked binding needs no CAS")

        monkeypatch.setattr("jax_reorder.algebra.trusted.isolated_compare", unexpected_cas)
    options = dict(execution=executions, algebra_budget=100000)
    if entry == "jaxpr":
        result = compare_algebraic_jaxprs(
            jax.make_jaxpr(left)(argument), jax.make_jaxpr(right)(argument), **options
        )
    elif entry == "function":
        compared = compare_functions(left, right, argument, **options)
        assert compared.status is Status.EQUIVALENT
        result = compared.modes[0].algebra
    else:
        compared = compare_function_traces(
            trace_function(left, argument), trace_function(right, argument), **options
        )
        assert compared.status is Status.EQUIVALENT
        result = compared.algebra
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == ("structural" if same_source else "algebraic")
    assert bool(result.warnings) is not same_source
    assert result.work > 0
    expected = (executions, executions) if same_source else executions
    for proof, condition in zip(result.normalizations, expected, strict=True):
        assert proof.execution == proof.remote.protocol.execution == condition
        assert proof.remote.protocol.ordered_protocol is not None
    assert any("caller binds" in value for value in result.assumptions)
    assert any("all devices execute the exact same" in value for value in result.assumptions)
    assert any("does not enforce" in value for value in result.assumptions)


def test_identical_program_views_replay_explicit_execution_proofs(parallel_sources):
    view = adapt(parallel_sources[0]).view
    result = compare_program_views(view, view, execution=binding(), analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    assert result.evidence.left.execution == result.evidence.right.execution == binding()
    verify_normalized_evidence(view, view, result.evidence, budget=AnalysisBudget(100000))


def test_public_algebra_remote_parallel_requires_execution():
    with pytest.raises(UnsupportedJaxprError, match="execution binding"):
        compare_algebraic_jaxprs(
            graph(parallel=True),
            graph(flat=True, explicit=True, receive_first=True),
            algebra_budget=100000,
        )


@pytest.mark.parametrize("execution", [binding(), (binding(), binding())])
def test_public_algebra_rejects_wrong_grid_binding_on_the_right(execution):
    with pytest.raises(UnsupportedJaxprError, match="right.*core binding.*grid point"):
        compare_algebraic_jaxprs(
            graph(parallel=True),
            graph(flat=True, explicit=True),
            algebra_budget=100000,
            execution=execution,
        )


@pytest.mark.parametrize("programs", [[], (), ([],)])
def test_public_algebra_rejects_malformed_binding_streams(parallel_sources, programs):
    with pytest.raises(ValueError, match="tuple|at least one"):
        compare_algebraic_jaxprs(
            parallel_sources[0], parallel_sources[0], execution=CoreExecutionBinding(programs)
        )


@pytest.mark.parametrize("mode", ["shared_scratch", "missing_receive"])
def test_public_algebra_structural_equality_still_checks_the_bound_protocol(mode):
    source = graph(parallel=True, mode=mode)
    with pytest.raises(UnsupportedJaxprError):
        compare_algebraic_jaxprs(source, source, algebra_budget=100000, execution=binding())


@pytest.mark.parametrize("mode", ["wrong_index_map", "self_target"])
def test_public_algebra_execution_does_not_hide_remote_value_changes(mode):
    result = compare_algebraic_jaxprs(
        graph(parallel=True),
        graph(flat=True, explicit=True, mode=mode),
        execution=(binding(), binding(flat=True)),
        algebra_budget=100000,
    )
    assert result.status is Status.INEQUIVALENT
    assert result.backend_result is not None
    assert len(result.normalizations) == 2


@pytest.mark.parametrize("local_pallas", [False, True])
def test_public_algebra_rejects_execution_without_remote_protocol(local_pallas):
    if local_pallas:

        def kernel(x, y):
            y[:] = x[:]

        source = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((128,), jnp.float32))
    else:

        def source(x):
            return x + 1

    traced = jax.make_jaxpr(source)(jax.ShapeDtypeStruct((128,), jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match="requires a remote"):
        compare_algebraic_jaxprs(traced, traced, execution=binding())


SHAPE = (2, 2, 2)


WIDTH = 128


SIZE = prod(SHAPE) * WIDTH


def execution(shape=SHAPE):
    return CoreExecutionBinding((tuple(product(*(range(n) for n in shape))),))


def source(semantics, *, cross_owner_signal=False):
    def kernel(x, y, temporary, local, send, receive, ack):
        index = (pl.program_id(0) * pl.num_programs(1) + pl.program_id(1)) * pl.num_programs(2)
        index += pl.program_id(2)
        peer = 1 - jax.lax.axis_index("d")
        signal_index = (index + 1) % prod(SHAPE) if cross_owner_signal else index
        pl.semaphore_signal(ack.at[signal_index], 1, device_id=(peer,))
        pl.semaphore_wait(ack.at[index], 1)
        block = temporary.at[index]
        tpu.async_copy(x.at[pl.ds(index * WIDTH, WIDTH)], block, local.at[index]).wait()
        tpu.async_remote_copy(
            block,
            y.at[pl.ds(index * WIDTH, WIDTH)],
            send.at[index],
            receive.at[index],
            (peer,),
        ).wait()

    full = pl.BlockSpec((SIZE,), lambda *coordinates: (0,), memory_space=pl.ANY)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((SIZE,), jnp.float32),
        grid=SHAPE,
        in_specs=(full,),
        out_specs=full,
        scratch_shapes=(
            tpu.VMEM((prod(SHAPE), WIDTH), jnp.float32),
            tpu.SemaphoreType.DMA((prod(SHAPE),)),
            tpu.SemaphoreType.DMA((prod(SHAPE),)),
            tpu.SemaphoreType.DMA((prod(SHAPE),)),
            tpu.SemaphoreType.REGULAR((prod(SHAPE),)),
        ),
        compiler_params=tpu.CompilerParams(dimension_semantics=semantics),
    )
    mapped = jax.shard_map(
        call,
        mesh=Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)),
        in_specs=P("d"),
        out_specs=P("d"),
        check_vma=False,
    )
    return adapt(jax.make_jaxpr(mapped)(jax.ShapeDtypeStruct((2 * SIZE,), jnp.float32))).view


def bindings(view):
    outer = view.root.operations[0]
    return outer.regions[0].region.operations[0].call, outer.shard_map.mesh


@pytest.mark.parametrize("semantics", list(product(("arbitrary", "parallel"), repeat=3)))
def test_three_dimensional_grid_values_and_owner_partition(semantics):
    view = source(semantics)
    proof = normalize_view(view, budget=AnalysisBudget(100000), execution=execution())
    x = np.arange(2 * SIZE, dtype=np.float32)
    actual = projection_array(proof.projection.outputs[0], (x,), x.size)
    np.testing.assert_array_equal(actual, x.reshape(2, SIZE)[::-1].reshape(-1))
    plan = proof.remote.protocol.io.grid
    assert plan.shape == SHAPE
    streams = {point: owner for owner, points in enumerate(plan.streams) for point in points}
    for first, a in enumerate(plan.points):
        for second, b in enumerate(plan.points):
            same_owner = all(
                x == y for x, y, axis in zip(a, b, semantics, strict=True) if axis == "parallel"
            )
            assert (streams[first] == streams[second]) == same_owner
    assert len(proof.remote.protocol.programs) == 2 * 2 ** semantics.count("parallel")
    for point, previous in enumerate(plan.previous):
        if previous is not None:
            assert streams[previous] == streams[point]
            assert plan.following[previous] == point


def test_cross_owner_signal_cannot_assume_single_core_switches_blocked_invocations():
    call, mesh = bindings(source(("parallel",) * 3, cross_owner_signal=True))
    with pytest.raises(UnsupportedJaxprError, match="wait dependency cycle"):
        analyze_remote_call(call, mesh, budget=AnalysisBudget(100000), execution=execution())


def interleaved_source(*, safe_cross_owner=False):
    def kernel(x, y, temporary, local, send, receive, ack):
        device = jax.lax.axis_index("d")
        row, column = pl.program_id(0), pl.program_id(1)
        peer = 1 - device

        if safe_cross_owner:

            @pl.when((row == 0) & (column == 0))
            def signal_later_owner():
                pl.semaphore_signal(ack.at[0], 1, device_id=(peer,))

            @pl.when((row == 0) & (column == 1))
            def wait_earlier_owner():
                pl.semaphore_wait(ack.at[0], 1)

        else:

            @pl.when((device == 0) & (row == 0) & (column == 0))
            def wait_first_owner():
                pl.semaphore_wait(ack.at[0], 1)

            @pl.when((device == 1) & (row == 1) & (column == 0))
            def signal_first_owner():
                pl.semaphore_signal(ack.at[0], 1, device_id=(peer,))

            @pl.when((device == 0) & (row == 0) & (column == 1))
            def signal_second_owner():
                pl.semaphore_signal(ack.at[1], 1, device_id=(peer,))

            @pl.when((device == 1) & (row == 0) & (column == 1))
            def wait_second_owner():
                pl.semaphore_wait(ack.at[1], 1)

        @pl.when((row == 1) & (column == 1))
        def transfer():
            tpu.async_copy(x, temporary, local).wait()
            tpu.async_remote_copy(temporary, y, send, receive, (peer,)).wait()

    full = pl.BlockSpec((WIDTH,), lambda *coordinates: (0,), memory_space=pl.ANY)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((WIDTH,), jnp.float32),
        grid=(2, 2),
        in_specs=(full,),
        out_specs=full,
        scratch_shapes=(
            tpu.VMEM((WIDTH,), jnp.float32),
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.REGULAR((2,)),
        ),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary", "parallel")),
    )
    mapped = jax.shard_map(
        call,
        mesh=Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)),
        in_specs=P("d"),
        out_specs=P("d"),
        check_vma=False,
    )
    return adapt(jax.make_jaxpr(mapped)(jax.ShapeDtypeStruct((2 * WIDTH,), jnp.float32))).view


def test_interleaved_single_core_deadlock_is_not_hidden_by_independent_owner_progress(monkeypatch):
    from jax_reorder.core import remote

    call, mesh = bindings(interleaved_source())
    successful = []
    analyze = remote.analyze_protocol

    def record(programs, semaphores, *, budget):
        proof = analyze(programs, semaphores, budget=budget)
        successful.append((programs, proof))
        return proof

    monkeypatch.setattr(remote, "analyze_protocol", record)
    with pytest.raises(UnsupportedJaxprError, match="wait dependency cycle"):
        analyze_remote_call(call, mesh, budget=AnalysisBudget(100000), execution=execution((2, 2)))
    # The relaxed model really did advance, and every counter used only the
    # same owner on peers. Explicit whole-core order supplies the missing cycle.
    assert successful
    assert len(successful[0][0]) == 4
    relaxed = successful[0][1]
    for counter in relaxed.counters:
        owner = relaxed.events[counter.consumer].stream
        assert all(relaxed.events[event].stream == owner for event, _ in counter.producers)


def test_cross_owner_signals_with_valid_whole_core_order_are_supported():
    view = interleaved_source(safe_cross_owner=True)
    proof = normalize_view(view, budget=AnalysisBudget(100000), execution=execution((2, 2)))
    x = np.arange(2 * WIDTH, dtype=np.float32)
    actual = projection_array(proof.projection.outputs[0], (x,), x.size)
    np.testing.assert_array_equal(actual, x.reshape(2, WIDTH)[::-1].reshape(-1))
    protocol = proof.remote.protocol.protocol
    assert proof.remote.protocol.ordered_protocol is not None
    assert any(
        protocol.events[event].stream != protocol.events[counter.consumer].stream
        for counter in protocol.counters
        for event, _ in counter.producers
    )


def test_grid_budget_is_checked_before_cartesian_product_allocation(monkeypatch):
    call, _ = bindings(source(("arbitrary",) * 3))
    call = replace(call, grid=(GridAxis(10**9, "arbitrary"), GridAxis(10**9, "arbitrary")))

    def forbidden_product(*args, **kwargs):
        raise AssertionError("the Cartesian product must not be allocated")

    monkeypatch.setattr("jax_reorder.core.remote_grid.product", forbidden_product)
    with pytest.raises(ComparisonBudgetError):
        plan_remote_grid(call, AnalysisBudget(100))


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_grid_planner_rejects_nonpositive_or_noninteger_extents(size):
    call, _ = bindings(source(("arbitrary",) * 3))
    with pytest.raises(UnsupportedJaxprError, match="positive static grid"):
        plan_remote_grid(replace(call, grid=(GridAxis(size, "arbitrary"),)), AnalysisBudget())
