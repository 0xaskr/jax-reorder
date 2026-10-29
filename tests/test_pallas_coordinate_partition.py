"""Exact coordinate partitions close grid/loop gaps without floating algebra."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    MathematicalUndecidedError,
    Status,
    compare_pallas_kernels,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.coordinate_partition import equal_on_output_domain
from jax_reorder.core.index_domain import AnalysisBudget, add, axis, integer, scale
from jax_reorder.core.memory import (
    Predicate,
    ScalarType,
    ValueExpr,
    _share_expression,
    expressions_equal,
)
from jax_reorder.core.model import ParameterView
from jax_reorder.core.normalization import compare_program_views, verify_normalized_evidence
from jax_reorder.core.tensor_values import project_coordinates
from tests.pallas_grid_expansion_cases import grouped, grouped_loop
from tests.pallas_projection_oracle import at_point
from tests.test_pallas_grid_expansion import trace
from tests.upstream_tpu.kda_cases import source_namespace


@pytest.mark.parametrize("size", [2, 3])
def test_grouped_grid_matches_internal_loop_without_a_mathematical_backend(size):
    a, b = (
        adapt(trace(fn, group_size=size)).view
        for fn in (grouped(group_size=size), grouped_loop(size))
    )
    result = compare_program_views(a, b)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    assert result.work < 10000
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget())
    with pytest.raises(ComparisonBudgetError):
        verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget(1))
    with pytest.raises(CorrespondenceError, match="terminal"):
        verify_normalized_evidence(
            a,
            b,
            replace(result.evidence, terminal_outputs=((0, 1), (1, 0))),
            budget=AnalysisBudget(),
        )


@pytest.mark.parametrize("mode", ["wrong_map", "overwrite"])
def test_coordinate_partition_does_not_hide_changed_values(mode):
    a, b = (adapt(trace(fn)).view for fn in (grouped(mode=mode), grouped_loop()))
    with pytest.raises(MathematicalUndecidedError):
        compare_program_views(a, b)


F32 = ScalarType("float32", False, ())


def literal(n):
    return ValueExpr("literal", F32, data=np.asarray(n, dtype=np.float32).tobytes())


def converted(n):
    return ValueExpr(
        "convert_element_type",
        F32,
        (integer(n),),
        (ParameterView("new_dtype", "float32"), ParameterView("weak_type", False)),
    )


@pytest.mark.parametrize("n", [0, -7, 2**24 + 1])
def test_integer_cast_uses_actual_target_dtype_rounding(n):
    actual = project_coordinates(converted(n), {}, {}, {}, AnalysisBudget())
    assert actual == literal(n)


def test_host_intermediate_rounding_is_not_assumed_for_large_integers():
    value = converted(2**53 + 1)
    assert project_coordinates(value, {}, {}, {}, AnalysisBudget()) == value


@pytest.mark.parametrize("factor", [-3, 2])
def test_affine_predicate_partitions_cover_complete_integer_intervals(factor):
    x = axis("output:0")
    left = ValueExpr(
        "select_n",
        F32,
        (Predicate("gt", (add(scale(x, factor), integer(4)), integer(1))), literal(0), literal(1)),
    )
    right = ValueExpr(
        "select_n",
        F32,
        (Predicate("ge", (scale(x, factor), integer(-2))), converted(0), converted(1)),
    )
    assert equal_on_output_domain(left, right, (13,), AnalysisBudget())
    changed = replace(right, operands=(*right.operands[:2], literal(2)))
    assert not equal_on_output_domain(left, changed, (13,), AnalysisBudget())


def test_floating_operation_order_remains_observable():
    x, y, z = (ValueExpr("input", F32, (axis("output:0"),), input=i) for i in range(3))
    left = ValueExpr("add", F32, (ValueExpr("add", F32, (x, y)), z))
    right = ValueExpr("add", F32, (x, ValueExpr("add", F32, (y, z))))
    assert not equal_on_output_domain(left, right, (2**24,), AnalysisBudget())


def test_partition_comparison_has_a_shared_budget():
    a, b = (adapt(trace(fn)).view for fn in (grouped(), grouped_loop()))
    with pytest.raises(ComparisonBudgetError):
        compare_program_views(a, b, analysis_budget=1000)


def test_inactive_piece_is_not_evaluated_outside_its_domain():
    # The unselected local piece is intentionally undefined in this cell.
    invalid = axis("unbound.local")
    value = ValueExpr("select_n", F32, (True, invalid, converted(3)))
    assert project_coordinates(value, {}, {}, {}, AnalysisBudget()) == literal(3)


def test_unknown_device_range_preserves_an_unproved_result():
    device = axis("device:x")
    assert not equal_on_output_domain(device, add(device, integer(1)), (8,), AnalysisBudget())


def test_coordinate_projection_recovers_exact_sharing_between_copied_subtrees():
    def copied(depth):
        if depth == 0:
            return ValueExpr("input", F32, (axis("output:0"),), input=0)
        return ValueExpr("add", F32, (copied(depth - 1), copied(depth - 1)))

    value = copied(8)
    shared = _share_expression(value)
    assert expressions_equal((value,), (shared,))
    budget = AnalysisBudget(16)
    projected = project_coordinates(value, {}, {}, {"output:0": (0, 2)}, budget)
    inputs = (np.asarray([-2, 0, 3], np.float32),)
    for i in range(3):
        assert at_point(projected, inputs, (i,)) == 256 * inputs[0][i]
    with pytest.raises(ComparisonBudgetError):
        project_coordinates(value, {}, {}, {"output:0": (0, 2)}, AnalysisBudget(1))


@pytest.mark.parametrize(
    "change",
    [
        {"operation": "sub"},
        {"type": ScalarType("float64", False, ())},
        {"type": ScalarType("float32", True, ())},
        {"type": ScalarType("float32", False, (("tag", "different"),))},
        {"parameters": (ParameterView("precision", "highest"),)},
        {"data": b"other literal bits"},
        {"input": 1},
        {"operands": (literal(2), literal(1))},
    ],
)
def test_value_sharing_retains_every_descriptor_field_and_operand_order(change):
    left = ValueExpr("add", F32, (literal(1), literal(2)))
    right = replace(left, **change)
    root = ValueExpr("add", F32, (left, right))
    shared = _share_expression(root)
    assert expressions_equal((root,), (shared,))
    assert shared.operands[0] is not shared.operands[1]


def test_value_sharing_does_not_merge_signed_zero_literals():
    left, right = literal(0.0), literal(-0.0)
    shared = _share_expression(ValueExpr("add", F32, (left, right)))
    assert shared.operands[0] is not shared.operands[1]
    assert shared.operands[0].data != shared.operands[1].data


def test_real_reverse_prefix_sum_projection_keeps_values_within_default_budget():
    reverse_sum = source_namespace()["compute_reverse_cumsum_dg_scan"]

    def function(*, interpret=False):
        def kernel(x, y):
            value = x[...]
            for _ in range(12):
                value = jnp.exp2(value * 0.25) - 1
            y[...] = reverse_sum(value[None, ...])[0]

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
            interpret=interpret,
        )

    signature = jax.ShapeDtypeStruct((64, 128), jnp.float32)
    source = jax.make_jaxpr(function())(signature)
    result = compare_pallas_kernels(source, source, equivalence="semantic")
    assert result.status is Status.EQUIVALENT and result.protocol_checked
    x = np.linspace(-0.1, 0.1, 64 * 128, dtype=np.float32).reshape(64, 128)
    expected = np.asarray(function(interpret=True)(x))
    for point in ((0, 0), (1, 127), (31, 63), (63, 127)):
        actual = at_point(result.left_protocol.projection.outputs[0], (x,), point)
        np.testing.assert_allclose(actual, expected[point], rtol=1e-5, atol=1e-7)
