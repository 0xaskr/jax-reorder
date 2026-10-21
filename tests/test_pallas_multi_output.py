"""Different output index maps retain and finish their own serial windows."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from jax_reorder import (
    ComparisonBudgetError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.call_bindings import extract_call
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.tensor_memory import functionalize_tensor_call
from tests.pallas_multi_output_cases import (
    dma_function,
    function,
    inputs,
    loop_function,
    nested_function,
    nested_reference,
    reference,
    trace,
    upstream_function,
)
from tests.pallas_oracle import run_worker
from tests.pallas_projection_oracle import at_point


def assert_projection(projection, args, expected):
    for value, output in zip(projection.outputs, expected, strict=True):
        for row in (0, 7, 8, output.shape[0] - 1):
            if row >= output.shape[0]:
                continue
            for start in range(0, output.shape[1], 128):
                for col in (start, start + 127):
                    np.testing.assert_equal(at_point(value, args, (row, col)), output[row, col])


@pytest.mark.parametrize("summary_first", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_each_output_retains_its_own_index_map_window(summary_first, reverse, prefix):
    kwargs = dict(summary_first=summary_first, reverse=reverse)
    source = trace("prefix" if prefix else "base", **kwargs)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, result)
    for seed in (7, 23):
        args = inputs(seed)
        assert_projection(
            result.left_protocol.projection, args, reference(args, prefix=prefix, **kwargs)
        )


def test_three_outputs_finish_at_three_different_grid_depths():
    source = jax.make_jaxpr(nested_function())(*inputs(steps=6))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    domains = result.left_protocol.projection.domains
    assert [d.serial_axes for d in domains[-3:]] == [(), (2,), (1, 2)]
    for seed in (7, 23):
        args = inputs(seed, steps=6)
        assert_projection(result.left_protocol.projection, args, nested_reference(args))


def test_upstream_scratch_kernel_can_emit_history_and_a_final_only_output():
    source = jax.make_jaxpr(upstream_function())(*inputs(groups=1, steps=1))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    for seed in (7, 23):
        args = inputs(seed, groups=1, steps=1)
        (x,) = args
        assert_projection(
            result.left_protocol.projection,
            args,
            (np.concatenate([x + 1, x + 2, x + 3]), x + 3),
        )


def test_different_output_windows_match_an_inner_kernel_loop():
    left = trace()
    right = jax.make_jaxpr(loop_function())(*inputs())
    a, b = adapt(left).view, adapt(right).view
    result = compare_program_views(a, b)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget())
    assert compare_algebraic_jaxprs(left, right).status is Status.EQUIVALENT


def test_different_lifetimes_can_also_have_different_local_block_shapes():
    def body(x, y, total):
        @pl.when(pl.program_id(1) == 0)
        def initialize():
            total[...] = jnp.zeros((8, 1), jnp.float32)

        y[...] = x[...] * 2
        total[...] += jnp.sum(x[...], axis=1, keepdims=True)

    full = pl.BlockSpec((8, 128), lambda p, j: (p, j))
    fn = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((16, 384), jnp.float32),
            jax.ShapeDtypeStruct((16, 1), jnp.float32),
        ),
        grid=(2, 3),
        in_specs=[full],
        out_specs=(full, pl.BlockSpec((8, 1), lambda p, j: (p, 0))),
    )
    source = jax.make_jaxpr(fn)(*inputs())
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    args = inputs()
    for row in (0, 7, 8, 15):
        assert (
            at_point(result.left_protocol.projection.outputs[1], args, (row, 0))
            == args[0].sum(axis=1)[row]
        )


@pytest.mark.parametrize("mode", ["reset_each", "overwrite", "wrong_map"])
def test_changing_only_the_summary_has_a_complete_output_counterexample(mode):
    result = compare_algebraic_jaxprs(trace(), trace(mode))
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )
    changed = normalize_view(adapt(trace(mode)).view, budget=AnalysisBudget()).projection
    args = inputs()
    y, s = reference(args)
    assert_projection(replace(changed, outputs=changed.outputs[:1]), args, (y,))
    assert any(
        at_point(changed.outputs[1], args, (i, j)) != s[i, j]
        for i, j in [(0, 0), (7, 127), (8, 0), (15, 127)]
    )


@pytest.mark.parametrize("mode", ["missing_init", "missing_tile", "stale_tile"])
def test_each_output_requires_initialization_in_every_owning_window(mode):
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(adapt(trace(mode)).view, budget=AnalysisBudget())


@pytest.mark.parametrize("mode", ["long_source", "long_destination"])
def test_dma_can_use_a_long_lived_output_while_another_output_changes_blocks(mode):
    source = jax.make_jaxpr(dma_function(mode))(*inputs(groups=1, steps=1))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    projection = result.left_protocol.projection
    assert len(projection.communication) == 1
    for seed in (7, 23):
        args = inputs(seed, groups=1, steps=1)
        (x,) = args
        assert_projection(projection, args, (np.concatenate([x, x]), x))


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("short_source", "DMA source"),
        ("short_destination", "DMA destination"),
        ("missing_wait", "before completion wait"),
    ],
)
def test_output_specific_window_boundaries_do_not_weaken_dma_hazards(mode, reason):
    source = jax.make_jaxpr(dma_function(mode))(*inputs(groups=1, steps=1))
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(adapt(source).view, budget=AnalysisBudget())


@pytest.mark.parametrize("accumulate", [False, True])
def test_rank_one_multi_output_routes_to_the_same_window_proof(accumulate):
    def body(x, y, s):
        if accumulate:

            @pl.when(pl.program_id(0) == 0)
            def initialize():
                s[...] = jnp.zeros_like(x[...])

            s[...] += x[...]
        else:
            s[...] = x[...]
        y[...] = x[...] * 2

    tiles = pl.BlockSpec((128,), lambda i: (i,))
    fn = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((384,), jnp.float32),
            jax.ShapeDtypeStruct((128,), jnp.float32),
        ),
        grid=(3,),
        in_specs=[tiles],
        out_specs=(tiles, pl.BlockSpec((128,), lambda i: (0,))),
    )
    args = (np.arange(384, dtype=np.float32),)
    source = jax.make_jaxpr(fn)(*args)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    proof = normalize_view(adapt(source).view, budget=AnalysisBudget()).projection
    expected = (
        args[0] * 2,
        args[0].reshape(3, 128).sum(axis=0) if accumulate else args[0][-128:],
    )
    for value, output in zip(proof.outputs, expected, strict=True):
        for point in range(output.size):
            assert at_point(value, args, (point,)) == output[point]


def test_each_output_keeps_the_contiguous_serial_writer_requirement():
    call = extract_call(adapt(trace()).view, budget=AnalysisBudget()).operation.call
    parallel = replace(call, grid=(call.grid[0], replace(call.grid[1], semantics="parallel")))
    with pytest.raises(UnsupportedJaxprError, match="trailing serial"):
        functionalize_tensor_call(parallel, budget=AnalysisBudget())

    def body(x, y, s):
        y[...] = x[...]
        s[...] = x[...]

    full = pl.BlockSpec((8, 128), lambda k, i: (i, k))
    source = jax.make_jaxpr(
        pl.pallas_call(
            body,
            out_shape=(
                jax.ShapeDtypeStruct((16, 256), jnp.float32),
                jax.ShapeDtypeStruct((16, 128), jnp.float32),
            ),
            grid=(2, 2),
            in_specs=[full],
            out_specs=(full, pl.BlockSpec((8, 128), lambda k, i: (i, 0))),
        )
    )(*inputs(steps=2))
    with pytest.raises(UnsupportedJaxprError, match="trailing serial"):
        normalize_view(adapt(source).view, budget=AnalysisBudget())


def test_serial_union_is_budgeted_before_expanding_any_output():
    source = jax.make_jaxpr(function(steps=2**20))(
        jax.ShapeDtypeStruct((16, 128 * 2**20), jnp.float32)
    )
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    with pytest.raises(ComparisonBudgetError, match="serial_grid"):
        functionalize_tensor_call(call, budget=AnalysisBudget(128))


def test_independent_parallel_prefix_is_not_enumerated_for_multiple_lifetimes():
    source = jax.make_jaxpr(function(groups=2**24))(
        jax.ShapeDtypeStruct((8 * 2**24, 384), jnp.float32)
    )
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    projection = functionalize_tensor_call(call, budget=AnalysisBudget(256))
    point = (8 * 2**24 - 1, 383)
    # The input is symbolic; huge parallel ownership must fit a fixed budget.
    assert projection.domains[1].shape == (point[0] + 1, point[1] + 1)
    assert projection.domains[2].serial_axes == (1,)


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case",
    [
        "base",
        "prefix",
        "summary_first",
        "reverse",
        "nested",
        "loop",
        "upstream",
        "dma_long_source",
        "dma_long_destination",
        "reset_each",
        "wrong_map",
    ],
)
def test_multi_output_cases_execute_in_official_tpu_cpu_interpreter(case):
    run_worker(
        "tests.pallas_multi_output_cases",
        case,
        expected=dict(case=case, status="ok"),
        entry="execute",
        devices=1,
        execution_timeout=60,
    )
