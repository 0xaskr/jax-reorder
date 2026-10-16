"""完整输出的 SymPy 不恒等判定、定义域和合法坐标查询。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    Status,
    compare_algebraic_jaxprs,
)


def graphs(left, right, *shapes, dtype=jnp.float32):
    """只构造抽象 jaxpr，不编译、不运行函数或抽样张量。"""
    args = tuple(jax.ShapeDtypeStruct(shape, dtype) for shape in shapes)
    return jax.make_jaxpr(left)(*args), jax.make_jaxpr(right)(*args)


@pytest.mark.parametrize(
    "left,right,shapes",
    [
        (lambda x: x + 1, lambda x: x + 2, ((),)),
        (lambda x: 2 * x, lambda x: x, ((3,),)),
        (lambda x, y: x - y, lambda x, y: y - x, ((2,), (2,))),
        (lambda x, y: (x, y), lambda x, y: (y, x), ((2,), (2,))),
        (lambda x: x * x, lambda x: x, ((),)),
        (lambda x: jnp.sum(x, axis=0), lambda x: jnp.sum(x, axis=1), ((2, 2),)),
        (lambda x: x[:2], lambda x: x[1:], ((3,),)),
        (lambda x: x, lambda x: x.T, ((2, 2),)),
        (lambda x, y: x @ y, lambda x, y: y @ x, ((2, 2), (2, 2))),
        (
            lambda x, y: jnp.sum(x) * jnp.sum(y),
            lambda x, y: jnp.sum(x * y),
            ((2,), (2,)),
        ),
        (lambda x: x.reshape((2, 3)), lambda x: x.reshape((3, 2)).T, ((6,),)),
        (lambda x: (x, x + 1), lambda x: (x, x + 2), ((),)),
    ],
    ids=[
        "offset",
        "gain",
        "sign",
        "output_order",
        "degree",
        "reduce_axis",
        "slice",
        "transpose",
        "matmul",
        "cross_terms",
        "reshape",
        "second_output",
    ],
)
def test_complete_output_difference_is_confirmed_by_sympy(left, right, shapes):
    first, second = graphs(left, right, *shapes)
    result = compare_algebraic_jaxprs(first, second)
    assert result.status == Status.INEQUIVALENT, result
    assert result.correspondence is None
    assert result.warnings and result.assumptions
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate")
        for q in result.backend_result.queries
    )
    reverse = compare_algebraic_jaxprs(second, first)
    assert reverse.status == Status.INEQUIVALENT


def test_output_slot_and_later_coordinate_are_observable_not_local():
    first, second = graphs(lambda x: (x, x), lambda x: (x, x.T), (2, 2))
    result = compare_algebraic_jaxprs(first, second)
    assert result.status == Status.INEQUIVALENT
    assert result.output == 1
    assert any(
        q.answer is False and q.scope == "output_coordinate=(0, 1)"
        for q in result.backend_result.queries
    )


@pytest.mark.parametrize(
    "left,right,shapes",
    [
        (lambda x, y: x + y, lambda x, y: y + x, ((2,), (2,))),
        (lambda x: (x + 1) * 0 + x, lambda x: (x + 2) * 0 + x, ((2,),)),
        (lambda x: jnp.sum(x - x), lambda x: jnp.sum(x) - jnp.sum(x), ((3,),)),
        (lambda x: x.T, lambda x: x, ((1, 1),)),
        (lambda x: x + 1, lambda x: x + 2, ((0,),)),
        (lambda x: jnp.sum(x) + 0, lambda x: jnp.sum(x) * 2, ((0,),)),
        (
            lambda x: jnp.sum(jnp.broadcast_to(x, (4,))) ** 2,
            lambda x: 16 * x * x,
            ((),),
        ),
    ],
    ids=[
        "commute",
        "erased_local_difference",
        "cancellation",
        "unit_transpose",
        "empty_output",
        "empty_sum",
        "reused_input_element",
    ],
)
def test_no_false_inequivalence_for_equal_outputs(left, right, shapes):
    result = compare_algebraic_jaxprs(*graphs(left, right, *shapes))
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize(
    "left,right,dtype,expected",
    [
        (lambda x: jnp.sin(x), lambda x: x + 1, jnp.float32, Status.INEQUIVALENT),
        (lambda x: jnp.exp(x), lambda x: x, jnp.float32, Status.INEQUIVALENT),
        (lambda x: jnp.log(x), lambda x: x, jnp.float32, Status.UNKNOWN),
        (lambda x: x / x, lambda x: x + 1, jnp.float32, Status.UNKNOWN),
        (lambda x: x + (-jnp.inf), lambda x: x + 1, jnp.float32, Status.UNKNOWN),
        (
            lambda x: (x + 1, jnp.sin(x)),
            lambda x: (x + 2, jnp.sin(x)),
            jnp.float32,
            Status.INEQUIVALENT,
        ),
        (lambda x: x + 1, lambda x: x + 2, jnp.int32, Status.UNKNOWN),
        (lambda x: x + 1, lambda x: x + 2, jnp.complex64, Status.UNKNOWN),
    ],
    ids=[
        "opaque",
        "transcendental",
        "partial",
        "denominator",
        "nonfinite",
        "other_output_opaque",
        "integer",
        "complex",
    ],
)
def test_total_real_functions_and_domain_boundaries(left, right, dtype, expected):
    first, second = graphs(left, right, (), dtype=dtype)
    result = compare_algebraic_jaxprs(first, second)
    assert result.status == expected, result
    assert result.backend_result is not None


def test_captured_tensor_is_not_fresh_independent_input():
    value = np.asarray([1.0, 2.0], dtype=np.float32)
    first, second = graphs(lambda x: x + value, lambda x: x + 1, (2,))
    assert compare_algebraic_jaxprs(first, second).status == Status.INEQUIVALENT


@pytest.mark.parametrize("operation", [jnp.log, lambda x: x + (-jnp.inf)])
def test_dead_partial_or_nonfinite_work_does_not_make_domain_assumptions(operation):
    def first(x):
        operation(x)
        return x + 1

    left, right = graphs(first, lambda x: x + 2, ())
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.UNKNOWN, result
    assert result.backend_result.failure_kind in {"condition_undecided", "unsupported_conversion"}


def test_dead_region_cannot_be_discarded_to_claim_inequivalence():
    def first(x):
        jax.lax.while_loop(lambda c: c < 3, lambda c: c + 1, x)
        return x + 1

    left, right = graphs(first, lambda x: x + 2, ())
    assert compare_algebraic_jaxprs(left, right).status == Status.UNKNOWN


def test_huge_output_shape_does_not_allocate_coordinate_pools():
    first, second = graphs(lambda x: x + 1, lambda x: x + 2, (10**9,))
    result = compare_algebraic_jaxprs(first, second)
    assert result.status == Status.INEQUIVALENT, result
    assert any(
        q.answer is False and q.scope == "output_coordinate=(0,)"
        for q in result.backend_result.queries
    )


def test_large_sum_does_not_trigger_unbounded_equals_expansion():
    first, second = graphs(lambda x: jnp.sum(x), lambda x: 2 * jnp.sum(x), (10**9,))
    result = compare_algebraic_jaxprs(first, second, algebra_budget=1000)
    assert result.status == Status.UNKNOWN
    assert result.backend_result.failure_kind == "cas_undecided"
    assert any("unevaluated_sum" in q.method for q in result.backend_result.queries)


def test_different_contract_is_not_a_polynomial_inequivalence():
    first, second = graphs(lambda x: x, lambda x: x[:1], (2,))
    assert compare_algebraic_jaxprs(first, second).status == Status.UNKNOWN


def test_high_degree_coefficient_uses_correct_variable_multiplicity():
    left, right = graphs(lambda x, y: (x + y) ** 4, lambda x, y: x**4 + y**4, (), ())
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.INEQUIVALENT, result


def test_rational_constant_difference_is_exact():
    left, right = graphs(lambda x: x + 0.5, lambda x: x + 0.25, ())
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.INEQUIVALENT, result
    assert any("1/2" in q.left and "1/4" in q.right for q in result.backend_result.queries)


def test_canceled_intermediate_difference_is_not_final_inequality():
    left, right = graphs(lambda x: (x + 1) * 0, lambda x: (x + 2) * 0, ())
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.EQUIVALENT, result
