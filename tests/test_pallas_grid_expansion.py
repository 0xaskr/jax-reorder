"""Grid expansion retains concrete windows, resource hazards and symbolic owners."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.call_bindings import extract_call
from jax_reorder.core.index_domain import AnalysisBudget, axis
from jax_reorder.core.memory import _substitute
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.tensor_memory import functionalize_tensor_call
from tests.pallas_grid_expansion_cases import (
    counter,
    dot_groups,
    grouped,
    grouped_dma,
    grouped_loop,
    inputs,
    parallel_counter,
    reference,
)
from tests.pallas_oracle import run_worker
from tests.pallas_projection_oracle import at_point


def trace(function, *, group_size=2, groups=2):
    return jax.make_jaxpr(function)(
        jax.ShapeDtypeStruct((8 * groups, 256 * group_size), np.float32)
    )


def projection(source):
    return normalize_view(adapt(source).view, budget=AnalysisBudget()).projection


@pytest.mark.parametrize("group_size", [2, 3])
@pytest.mark.parametrize("mode", ["base", "reduce"])
def test_grouped_serial_windows_have_complete_output_coordinates(group_size, mode):
    source = trace(grouped(group_size=group_size, mode=mode), group_size=group_size)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, result)
    proof = result.left_protocol.projection
    for seed in (7, 23):
        args = inputs(group_size=group_size, seed=seed)
        for value, expected in zip(
            proof.outputs, reference(args, group_size=group_size, mode=mode), strict=True
        ):
            for row in (0, 7, 8, 15):
                for col in range(expected.shape[1]):
                    np.testing.assert_equal(at_point(value, args, (row, col)), expected[row, col])


@pytest.mark.parametrize("group_size", [2, 3])
def test_flat_grouped_axis_matches_a_split_grid(group_size):
    left = adapt(trace(grouped(group_size=group_size), group_size=group_size)).view
    right = adapt(trace(grouped(group_size=group_size, split=True), group_size=group_size)).view
    result = compare_program_views(left, right)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget())


@pytest.mark.parametrize("group_size", [2, 3])
def test_internal_loop_projection_preserves_grouped_output_values(group_size):
    source = trace(grouped_loop(group_size), group_size=group_size)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    # This validates the loop's value projection against the same independent
    # reference as the grid case. It does not claim a proof between the graphs.
    for seed in (7, 23):
        args = inputs(group_size=group_size, seed=seed)
        for value, expected in zip(
            result.left_protocol.projection.outputs,
            reference(args, group_size=group_size),
            strict=True,
        ):
            for row in (0, 7, 8, 15):
                for col in range(expected.shape[1]):
                    np.testing.assert_equal(at_point(value, args, (row, col)), expected[row, col])


@pytest.mark.parametrize("mode", ["wrong_map", "overwrite"])
def test_changed_mapping_or_update_preserves_a_concrete_counterexample(mode):
    proof = projection(trace(grouped(mode=mode)))
    args = inputs()
    expected = reference(args)[1]
    assert any(
        at_point(proof.outputs[1], args, (0, col)) != expected[0, col]
        for col in range(expected.shape[1])
    )


@pytest.mark.parametrize(
    "mode,reason",
    [("missing_init", "uninitialized"), ("revisit", "nonconsecutively"), ("gap", "uncovered")],
)
def test_invalid_grouped_window_is_refused(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        projection(trace(grouped(mode=mode)))


def test_parallel_repeated_writers_are_not_given_a_serial_order():
    call = extract_call(adapt(trace(grouped())).view, budget=AnalysisBudget()).operation.call
    call = replace(call, grid=tuple(replace(g, semantics="parallel") for g in call.grid))
    with pytest.raises(UnsupportedJaxprError, match="bijection|redundant"):
        functionalize_tensor_call(call, budget=AnalysisBudget())


def test_shifted_parallel_owners_cannot_overlap_between_serial_windows():
    def body(x, y, total):
        y[...] = x[...]
        total[...] = x[...]

    full = pl.BlockSpec((8, 128), lambda p, j: (p, j))
    fn = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((16, 512), jnp.float32),
            jax.ShapeDtypeStruct((24, 128), jnp.float32),
        ),
        grid=(2, 4),
        in_specs=[full],
        out_specs=(full, pl.BlockSpec((8, 128), lambda p, j: (p + j // 2, 0))),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )
    with pytest.raises(UnsupportedJaxprError, match="windows overlap"):
        projection(trace(fn))


def test_dma_is_carried_across_instances_and_reused_in_the_next_window():
    proof = projection(trace(grouped_dma(), groups=1))
    assert len(proof.communication) == 2
    assert len({t.transfer.generation for t in proof.communication}) == 2
    args = inputs(groups=1)
    x = args[0].reshape(8, 4, 128)
    expected = np.repeat(x[:, ::2], 2, axis=1).reshape(8, 512)
    for row in (0, 7):
        for col in range(512):
            assert at_point(proof.outputs[0], args, (row, col)) == expected[row, col]


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("input_source", "DMA source"),
        ("overwrite_source", "DMA source"),
        ("missing_wait", "before completion wait"),
    ],
)
def test_expansion_never_invents_a_dma_wait(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        projection(trace(grouped_dma(mode), groups=1))


def test_counter_without_array_scratch_survives_output_window_switches():
    proof = projection(trace(counter(), groups=1))
    args = inputs(groups=1)
    for col in (0, 127, 128, 255, 256, 383, 384, 511):
        assert at_point(proof.outputs[0], args, (0, col)) == args[0][0, col] * 2
    with pytest.raises(UnsupportedJaxprError, match="unsatisfied"):
        projection(trace(counter("wrong_count"), groups=1))


@pytest.mark.parametrize("semaphore", [False, True])
def test_rank_one_single_output_can_carry_grouped_or_counter_state(semaphore):
    def body(x, y, *scratch):
        j = pl.program_id(0)
        if semaphore:

            @pl.when(j % 2 == 0)
            def signal():
                pl.semaphore_signal(scratch[0], 1)

            @pl.when(j % 2 == 1)
            def wait():
                pl.semaphore_wait(scratch[0], 1)

            y[...] = x[...]
        else:

            @pl.when(j % 2 == 0)
            def initialize():
                y[...] = jnp.zeros((128,), jnp.float32)

            y[...] += x[...]

    fn = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((512 if semaphore else 256,), jnp.float32),
        grid=(4,),
        in_specs=[pl.BlockSpec((128,), lambda j: (j,))],
        out_specs=pl.BlockSpec((128,), lambda j: (j if semaphore else j // 2,)),
        scratch_shapes=[tpu.SemaphoreType.REGULAR] if semaphore else [],
    )
    x = np.arange(512, dtype=np.float32)
    proof = projection(jax.make_jaxpr(fn)(x))
    expected = x if semaphore else x.reshape(2, 2, 128).sum(axis=1).reshape(256)
    for i in range(expected.size):
        assert at_point(proof.outputs[0], (x,), (i,)) == expected[i]


@pytest.mark.parametrize("rank_one", [False, True])
@pytest.mark.parametrize("core_index", [None, 0])
def test_parallel_counters_require_a_recorded_core_binding(rank_one, core_index):
    args = (np.arange(512, dtype=np.float32),) if rank_one else inputs(groups=1)
    source = jax.make_jaxpr(parallel_counter(core_index=core_index, rank_one=rank_one))(*args)
    with pytest.raises(UnsupportedJaxprError, match="core target binding"):
        compare_pallas_kernels(source, source, check_protocol=True)
    result = compare_pallas_kernels(source, source, check_protocol=True, execution_cores=1)
    assert result.execution_cores == 1
    assert "caller binds" in result.assumptions[-1]
    verify_pallas_kernel_comparison(source, source, result)
    point = (511,) if rank_one else (7, 511)
    value = result.left_protocol.projection.outputs[0]
    if rank_one:
        value = _substitute(value, {"global": axis("output:0")})
    assert at_point(value, args, point) == args[0][point] * 2
    with pytest.raises(CorrespondenceError):
        verify_pallas_kernel_comparison(source, source, replace(result, assumptions=()))


@pytest.mark.parametrize(
    "mode,target,reason",
    [("base", 1, "bound core 0"), ("wrong_count", 0, "unsatisfied"), ("leftover", 0, "conserved")],
)
def test_single_core_binding_does_not_hide_wrong_targets_or_counts(mode, target, reason):
    source = trace(parallel_counter(core_index=target, mode=mode), groups=1)
    with pytest.raises(UnsupportedJaxprError, match=reason):
        compare_pallas_kernels(source, source, check_protocol=True, execution_cores=1)


def test_core_targets_remain_part_of_the_source_graph():
    left = trace(parallel_counter(core_index=0), groups=1)
    right = trace(parallel_counter(core_index=1), groups=1)
    assert compare_pallas_kernels(left, right).status is Status.DIFFERENT
    for invalid in (0, 2, True):
        with pytest.raises(ValueError, match="single-core"):
            compare_pallas_kernels(left, left, check_protocol=True, execution_cores=invalid)


def test_single_core_binding_does_not_serialize_parallel_counter_dependencies():
    def body(x, y, sem):
        @pl.when(pl.program_id(0) == 0)
        def signal():
            pl.semaphore_signal(sem, 1)

        @pl.when(pl.program_id(0) == 1)
        def wait():
            pl.semaphore_wait(sem, 1)

        y[...] = x[...]

    full = pl.BlockSpec((8, 128), lambda j: (0, j))
    source = jax.make_jaxpr(
        pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((8, 512), jnp.float32),
            grid=(4,),
            in_specs=[full],
            out_specs=full,
            scratch_shapes=[tpu.SemaphoreType.REGULAR],
            compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
        )
    )(*inputs(groups=1))
    with pytest.raises(UnsupportedJaxprError, match="different counter states"):
        compare_pallas_kernels(source, source, check_protocol=True, execution_cores=1)


def test_parallel_program_coordinates_do_not_become_dot_contraction_axes():
    x = inputs()[0]
    w = np.arange(128 * 128, dtype=np.float32).reshape(128, 128) % 3
    source = jax.make_jaxpr(dot_groups())(x, w)
    proof = projection(source)
    tiles = x.reshape(16, 4, 128) @ w
    expected = (tiles.reshape(16, 512), tiles.reshape(16, 2, 2, 128).sum(axis=2).reshape(16, 256))
    for value, output in zip(proof.outputs, expected, strict=True):
        for row in (0, 7, 8, 15):
            for col in (0, 127, 128, output.shape[1] - 1):
                assert at_point(value, (x, w), (row, col)) == output[row, col]


def test_symbolic_parallel_prefix_has_constant_analysis_cost():
    costs = []
    for groups in (4, 2**24):
        source = trace(grouped(groups=groups), groups=groups)
        call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
        budget = AnalysisBudget(2048)
        proof = functionalize_tensor_call(call, budget=budget)
        assert proof.domains[0].shape[0] == groups * 8
        costs.append(budget.used)
    assert costs[0] == costs[1]


def test_serial_expansion_is_budgeted_before_allocating_instances():
    source = trace(grouped(group_size=2**20), group_size=2**20)
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    with pytest.raises(ComparisonBudgetError, match="serial_grid"):
        functionalize_tensor_call(call, budget=AnalysisBudget(256))


def test_aligned_window_coverage_does_not_compare_every_pair():
    costs = []
    for size in (32, 64):
        source = trace(grouped(group_size=size), group_size=size)
        call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
        budget = AnalysisBudget(20000)
        functionalize_tensor_call(call, budget=budget)
        costs.append(budget.used)
    assert costs[1] < 2.5 * costs[0]


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case", ["group2", "group3", "loop2", "loop3", "reduce", "dma", "counter", "parallel_counter"]
)
def test_grid_expansion_matches_official_cpu_execution(case):
    run_worker(
        "tests.pallas_grid_expansion_cases",
        case,
        expected=dict(case=case, status="ok"),
        entry="execute",
        devices=1,
        execution_timeout=60,
    )
