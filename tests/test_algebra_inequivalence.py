"""完整输出的 SymPy 不恒等判定、定义域和合法坐标查询。

覆盖：
- 反例（正结论）：常量偏移、倍数、减法/输出槽位互换、非对称乘法、归约轴、切片、转置、
  矩阵乘法顺序、独立求和乘积漏交叉项、reshape 后转置、多输出中的一项不同，都应得到完整
  输出坐标上的负结论。
- 不误报：两侧输出相同时必须判等效；全实数函数与定义域边界组合按预期给出等价或未决。
- 边界：后一个输出坐标、捕获张量不是独立输入、死代码中的部分函数/非有限值、被删除的死
  区域、巨大输出 shape（不分配坐标池）、大求和（不触发无界 equals 展开）、不同契约、
  高次项重数、有理常量差、被约掉的中间差值。
- 前置条件：CPU 平台与 SymPy 1.14；graphs 只构造抽象 jaxpr，不编译、不运行、不抽样。
- 结论是“存在输入使两函数不同”，不是“每个输入都不同”。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    ComparisonContractError,
    ComparisonError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
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
        (lambda x: jnp.log(x), lambda x: x, jnp.float32, Status.INEQUIVALENT),
        (lambda x: x / x, lambda x: x + 1, jnp.float32, Status.INEQUIVALENT),
        (lambda x: x + (-jnp.inf), lambda x: x + 1, jnp.float32, MathematicalUndecidedError),
        (
            lambda x: (x + 1, jnp.sin(x)),
            lambda x: (x + 2, jnp.sin(x)),
            jnp.float32,
            Status.INEQUIVALENT,
        ),
        (lambda x: x + 1, lambda x: x + 2, jnp.int32, MathematicalUndecidedError),
        (lambda x: x + 1, lambda x: x + 2, jnp.complex64, UnsupportedJaxprError),
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
    if isinstance(expected, type) and issubclass(expected, ComparisonError):
        with pytest.raises(expected):
            compare_algebraic_jaxprs(first, second)
        return
    result = compare_algebraic_jaxprs(first, second)
    assert result.status == expected, result
    assert result.backend_result is not None


def test_captured_tensor_is_not_fresh_independent_input():
    value = np.asarray([1.0, 2.0], dtype=np.float32)
    first, second = graphs(lambda x: x + value, lambda x: x + 1, (2,))
    assert compare_algebraic_jaxprs(first, second).status == Status.INEQUIVALENT


@pytest.mark.parametrize("operation", [jnp.log, lambda x: x / x])
def test_partial_negative_uses_a_shared_defined_input(operation):
    result = compare_algebraic_jaxprs(*graphs(operation, lambda x: x + 1, ()))
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.method == "exact representable input witness: one"
        for q in result.backend_result.queries
    )
    # x=1 belongs to both original domains; this is independent of the
    # symbolic witness construction and does not require a domain assumption.
    assert float(operation(jnp.float32(1))) != 2


@pytest.mark.parametrize("same_output", [False, True])
def test_dead_partial_work_preserves_its_domain_in_complete_output_comparison(same_output):
    def first(x):
        jnp.log(x)
        return x + 1

    left, right = graphs(first, lambda x: x + (1 if same_output else 2), ())
    if same_output:
        # Ordinary JAX lowering still retains dead partial-function domains.
        with pytest.raises(MathematicalUndecidedError, match="condition_undecided"):
            compare_algebraic_jaxprs(left, right)
        return
    result = compare_algebraic_jaxprs(left, right)
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.method == "exact representable input witness: one" and (q.left, q.right) == ("2", "3")
        for q in result.backend_result.queries
    )


def test_dead_nonfinite_work_does_not_make_domain_assumptions():
    def first(x):
        x + (-jnp.inf)
        return x + 1

    left, right = graphs(first, lambda x: x + 2, ())
    with pytest.raises(MathematicalUndecidedError) as error:
        compare_algebraic_jaxprs(left, right)
    result = error.value
    assert result.failure_kind in {"condition_undecided", "unsupported_conversion"}


def test_dead_region_cannot_be_discarded_to_claim_inequivalence():
    def first(x):
        jax.lax.while_loop(lambda c: c < 3, lambda c: c + 1, x)
        return x + 1

    left, right = graphs(first, lambda x: x + 2, ())
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare_algebraic_jaxprs(left, right)


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
    assert result.status is Status.INEQUIVALENT
    assert any(
        query.answer is False and "exact representable input witness" in query.method
        for query in result.backend_result.queries
    )


def test_different_contract_is_not_a_polynomial_inequivalence():
    first, second = graphs(lambda x: x, lambda x: x[:1], (2,))
    with pytest.raises(ComparisonContractError):
        compare_algebraic_jaxprs(first, second)


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


@pytest.mark.parametrize("size", [2, 1024])
def test_partial_multi_output_negative_has_a_shared_valid_input(size):
    left, right = graphs(
        lambda x: (jnp.log(x), jnp.float32(0)),
        lambda x: (jnp.log(x), jnp.float32(1)),
        (size,),
    )
    result = compare_algebraic_jaxprs(left, right)
    assert result.status is Status.INEQUIVALENT
    assert any(
        "exact representable input witness: one" in q.method for q in result.backend_result.queries
    )
