"""Source intervals, signed guards, finite carry, and retained compiler hints."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu

from jax_reorder import ComparisonBudgetError, Status, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget, IndexExpr, integer
from jax_reorder.core.memory import ScalarType, ValueExpr
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.value_operations import integer_value_bounds
from jax_reorder.core.view_graph import match_views
from tests.pallas_dynamic_range_cases import (
    SHAPE,
    expected,
    function,
    graph,
    inputs,
    layout_function,
)
from tests.pallas_projection_oracle import at_point

MODES = ("and", "mod", "div", "lax_div", "negative_div", "carry", "alias_carry", "wrong_guard")


def project(mode, **options):
    return normalize_view(
        adapt(graph(mode, **options)).view, budget=AnalysisBudget(100000)
    ).projection


@pytest.mark.parametrize("mode", MODES)
def test_bounded_sources_match_every_output_and_cpu(mode):
    projection = project(mode)
    execute = jax.jit(function(mode, interpret=True))
    for raw in (-2147483648, -17, -5, -1, 0, 1, 3, 4, 7, 15, 16, 2147483632):
        x = inputs(raw)
        reference = expected(x, mode)
        actual = np.asarray(
            [at_point(projection.outputs[0], (x,), point) for point in np.ndindex(SHAPE)],
            dtype=np.int32,
        ).reshape(SHAPE)
        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(execute(x), reference)


@pytest.mark.parametrize("mode", ("and", "mod", "div", "lax_div", "carry", "alias_carry"))
def test_wrong_mask_divisor_and_loop_update_preserve_counterexamples(mode):
    left, right = project(mode), project(mode, wrong=True)
    observed = False
    for raw in (-5, 3, 4, 7, 12):
        x = inputs(raw)
        reference = expected(x, mode, wrong=True)
        for row in range(SHAPE[0]):
            actual = at_point(right.outputs[0], (x,), (row, 1))
            assert actual == reference[row, 1]
            observed |= actual != at_point(left.outputs[0], (x,), (row, 1))
    assert observed


@pytest.mark.parametrize("mode", ("signed_rem", "oob_and", "zero_divisor", "overflow", "unbounded"))
def test_unknown_or_invalid_integer_domains_remain_rejected(mode):
    with pytest.raises(UnsupportedJaxprError, match="bounds|finite"):
        project(mode)


def test_a_wrong_modulo_guard_is_not_recognized_as_python_remainder():
    correct, wrong = project("mod"), project("wrong_guard")
    x = inputs(3)
    assert at_point(correct.outputs[0], (x,), (3, 1)) != at_point(wrong.outputs[0], (x,), (3, 1))
    assert at_point(correct.outputs[0], (x,), (7, 1)) != at_point(wrong.outputs[0], (x,), (7, 1))


@pytest.mark.parametrize("mode", ("mod", "carry", "alias_carry", "no_progress"))
def test_dynamic_domain_and_progress_remain_budgeted(mode):
    with pytest.raises(ComparisonBudgetError):
        normalize_view(adapt(graph(mode)).view, budget=AnalysisBudget(150))


def test_zero_iteration_path_does_not_gain_initialization_from_other_carries():
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        project("missing_init")


@pytest.mark.parametrize("op", ("and", "or", "xor", "not", "div", "rem", "sign"))
def test_numpy_projection_oracle_matches_integer_jax_semantics(op):
    from jax import lax

    types = ScalarType("int32", False, ())
    for a in (-31, -5, -1, 0, 1, 5, 31):
        for b in (-4, -1, 1, 4):
            args = (integer(a),) if op in ("not", "sign") else (integer(a), integer(b))
            expr = ValueExpr(op, types, args)
            operation = {
                "and": lax.bitwise_and,
                "or": lax.bitwise_or,
                "xor": lax.bitwise_xor,
                "not": lax.bitwise_not,
                "div": lax.div,
                "rem": lax.rem,
                "sign": lax.sign,
            }[op]
            values = (jnp.int32(a),) if len(args) == 1 else (jnp.int32(a), jnp.int32(b))
            assert at_point(expr, (), ()) == np.asarray(operation(*values))


def test_branch_ranges_include_every_signed_result_and_keep_guard_changes():
    types, boolean = ScalarType("int32", False, ()), ScalarType("bool", False, ())
    raw = ValueExpr("input", types, (integer(0),), input=0)
    clipped = ValueExpr("min", types, (ValueExpr("max", types, (raw, integer(-15))), integer(15)))
    rem = ValueExpr("rem", types, (clipped, integer(4)))
    negative = ValueExpr("lt", boolean, (rem, integer(0)))
    nonzero = ValueExpr("ne", boolean, (rem, integer(0)))
    guards = (
        negative,
        nonzero,
        ValueExpr("and", boolean, (negative, nonzero)),
        ValueExpr("or", boolean, (negative, nonzero)),
        ValueExpr("not", boolean, (negative,)),
    )
    for guard in guards:
        expr = ValueExpr(
            "select_n", types, (guard, rem, ValueExpr("add", types, (rem, integer(4))))
        )
        low, high = integer_value_bounds(expr, {})
        values = [int(at_point(expr, (np.asarray([v], np.int32),), ())) for v in range(-35, 36)]
        assert low <= min(values) <= max(values) <= high
    correct = ValueExpr(
        "select_n", types, (negative, rem, ValueExpr("add", types, (rem, integer(4))))
    )
    assert integer_value_bounds(correct, {}) == (0, 3)


@pytest.mark.parametrize("variant", ("original", "equivalent", "wrong"))
def test_remote_data_route_is_not_treated_as_a_set_of_uncorrelated_peers(variant):
    from tests.pallas_remote_cases import remote_graph

    def kernel(x, y, send, receive):
        peer = jnp.sum(x[:1]).astype(jnp.int32) & 1
        if variant == "equivalent":
            peer += 0
        elif variant == "wrong":
            peer ^= 1
        tpu.async_remote_copy(x, y, send, receive, (peer,)).wait()

    with pytest.raises(UnsupportedJaxprError, match="static integer"):
        normalize_view(adapt(remote_graph(kernel)).view, budget=AnalysisBudget(100000))


def layout_view(**options):
    typ = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    return adapt(jax.make_jaxpr(layout_function(**options))(typ, typ)).view


def test_reviewed_transpose_hint_keeps_source_attributes_and_complete_values():
    default, hinted = layout_view(hint=False), layout_view(hint=True)
    assert match_views(default, hinted) is None
    projection = normalize_view(hinted, budget=AnalysisBudget(100000)).projection
    call = hinted.root.operations[0].call
    assert not call.compiler.default_execution_options and call.compiler.reviewed_layout_options
    assert next(
        p.value for p in call.compiler.parameters if p.name == "fuse_transposed_lhs_in_matmul"
    ) == (("type", "builtins", "bool"), True)
    x = np.arange(128 * 128, dtype=np.float32).reshape((128, 128)) / 16384
    z = np.eye(128, dtype=np.float32)
    reference = x.T @ z
    for point in ((0, 0), (0, 127), (17, 63), (127, 127)):
        np.testing.assert_allclose(at_point(projection.outputs[0], (x, z), point), reference[point])
    np.testing.assert_allclose(layout_function(interpret=True)(x, z), reference)
    assert not np.array_equal(layout_function(interpret=True, wrong=True)(x, z), reference)


@pytest.mark.parametrize("option", ("disable_semaphore_checks", "skip_device_barrier"))
def test_hint_does_not_admit_unreviewed_execution_or_safety_options(option):
    with pytest.raises(UnsupportedJaxprError, match="reviewed native execution contract"):
        normalize_view(layout_view(**{option: True}), budget=AnalysisBudget(100000))


@pytest.mark.parametrize("mode", ("and", "mod", "div", "carry", "alias_carry"))
def test_dynamic_source_reordering_replays_complete_evidence(mode):
    left, right = (adapt(graph(mode, initialize_first=first)).view for first in (False, True))
    result = compare_program_views(left, right, analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))


def test_transpose_hint_load_order_replays_complete_evidence():
    left, right = (layout_view(reverse_loads=reverse) for reverse in (False, True))
    result = compare_program_views(left, right, analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))


def test_equal_intervals_of_distinct_inputs_are_not_the_same_expression():
    types, boolean = ScalarType("int32", False, ()), ScalarType("bool", False, ())

    def bounded(port):
        value = ValueExpr("input", types, (integer(0),), input=port)
        return ValueExpr("min", types, (ValueExpr("max", types, (value, integer(-3))), integer(3)))

    a, b = bounded(0), bounded(1)
    guard = ValueExpr("lt", boolean, (a, integer(0)))
    expr = ValueExpr("select_n", types, (guard, b, ValueExpr("add", types, (b, integer(4)))))
    low, high = integer_value_bounds(expr, {})
    values = [
        int(at_point(expr, (np.asarray([x], np.int32), np.asarray([y], np.int32)), ()))
        for x in range(-5, 6)
        for y in range(-5, 6)
    ]
    assert (min(values), max(values)) == (-3, 7)
    assert low <= -3 and high >= 7


def test_wrong_sign_guard_retains_its_negative_result():
    types, boolean = ScalarType("int32", False, ()), ScalarType("bool", False, ())
    raw = ValueExpr("input", types, (integer(0),), input=0)
    clipped = ValueExpr("min", types, (ValueExpr("max", types, (raw, integer(0))), integer(15)))
    quotient = ValueExpr("div", types, (clipped, integer(4)))
    guard = ValueExpr("ne", boolean, (ValueExpr("sign", types, (clipped,)), integer(-1)))
    expr = ValueExpr(
        "select_n", types, (guard, quotient, ValueExpr("sub", types, (quotient, integer(1))))
    )
    low, high = integer_value_bounds(expr, {})
    values = [int(at_point(expr, (np.asarray([x], np.int32),), ())) for x in range(-20, 21)]
    assert low <= min(values) == -1 and high >= max(values)


def test_deep_and_branching_range_queries_have_a_global_work_limit():
    types, boolean = ScalarType("int32", False, ()), ScalarType("bool", False, ())
    raw = ValueExpr("input", types, (integer(0),), input=0)
    guard = ValueExpr("lt", boolean, (raw, integer(0)))
    deep = integer(0)
    for _ in range(5000):
        deep = ValueExpr("select_n", types, (guard, deep, integer(1)))
    assert integer_value_bounds(deep, {}) == (-2147483648, 2147483647)
    branching = ValueExpr("min", types, (raw, integer(3)))
    for port in range(1, 31):
        other = ValueExpr("input", types, (integer(0),), input=port)
        guard = ValueExpr("lt", boolean, (other, integer(0)))
        branching = ValueExpr("select_n", types, (guard, branching, branching))
    result = integer_value_bounds(branching, {})
    assert result == (-2147483648, 2147483647)


def test_shared_index_expression_expansion_is_charged_before_exact_bounds():
    shared = integer(1)
    for _ in range(30):
        shared = IndexExpr("sum", (shared, shared))
    result = integer_value_bounds(shared, {})
    assert result is None
