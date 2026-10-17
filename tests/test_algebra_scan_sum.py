"""独立 scan 累加与有限求和互认，以及反馈、序列输出和资源反例。

覆盖：
- 正例：独立累加 scan（含初值、reverse、unroll、零长度与多维序列）与 ``initial + sum``
  在 forward/jvp/vjp 下等价，命中 ``scan:independent_accumulation`` 规则；独立 carry、
  捕获的线性增量、序列输出与常量增量（长度 0/1/9）也保持。
- 反例：减法、换输入、双倍、错误初值、错误轴、交叉 carry 反馈、序列历史/输出顺序改变、
  死代码体与错误自定义梯度都不能匹配。
- 资源：求和长度与查询工作量保持符号表示并受预算限制。

前置条件：CPU 平台与 SymPy 1.14；JVP/VJP 只覆盖实际追踪出的一阶程序。
"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax

from jax_reorder import (
    ComparisonBudgetError,
    ComparisonContractError,
    MathematicalUndecidedError,
    Status,
    compare_algebraic_jaxprs,
    compare_functions,
)


def _scan(initial, x, *, reverse=False, unroll=1):
    """包含初值的简单累加，统一覆盖标量和向量反馈。"""
    return lax.scan(lambda c, a: (c + a, None), initial, x, reverse=reverse, unroll=unroll)[0]


@pytest.mark.parametrize(
    "shape,reverse,unroll",
    [
        # 长度为 0/1/7，reverse 与多种 unroll（含 0 表示自动）组合；形状含多维序列。
        ((0,), False, 1),
        ((1,), True, 1),
        ((7,), False, 2),
        ((5,), True, 0),
        ((0, 2), True, 1),
        ((1, 2), False, 2),
        ((7, 2), True, 3),
        ((3, 2, 3), False, 1),
    ],
)
@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_scan_equals_sum_with_initial_and_all_derivative_modes(shape, reverse, unroll, mode):
    """独立累加与 initial+sum 等价；reverse/unroll 不改变累加语义。"""
    initial = jnp.full(shape[1:], 2.0, dtype=jnp.float32)
    x = jnp.ones(shape, dtype=jnp.float32)
    result = compare_functions(
        lambda initial, x: _scan(initial, x, reverse=reverse, unroll=unroll),
        lambda initial, x: initial + jnp.sum(x, axis=0),
        initial,
        x,
        modes=(mode,),
        check_cpu=True,
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.cpu_compiled
    evidence = result.modes[0]
    # 只有实际走了代数路径时才要求规则名；结构等价时 algebra 为 None。
    if evidence.algebra is not None:
        assert evidence.algebra.equivalence_kind == "algebraic"
        assert "scan:independent_accumulation" in evidence.algebra.rules


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_independent_carries_and_captured_linear_increment(mode):
    def left(x, scale, initial):
        return lax.scan(lambda c, a: ((c[0] + scale * a, c[1] - a), None), (initial, initial), x)[0]

    def right(x, scale, initial):
        summed = jnp.sum(x, axis=0)
        return initial + scale * summed, initial - summed

    result = compare_functions(
        left, right, jnp.ones((4, 2)), jnp.ones(2), jnp.ones(2), modes=(mode,)
    )
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_independent_sequence_outputs_are_preserved(mode):
    def left(x):
        return lax.scan(lambda c, a: (c + a, a * 2), jnp.zeros(x.shape[1:]), x)

    result = compare_functions(
        left,
        lambda x: (jnp.sum(x, axis=0), x * 2),
        jnp.ones((3, 2)),
        modes=(mode,),
    )
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("length", [0, 1, 9])
def test_constant_increment_without_sequence_input(length):
    def left(initial, delta):
        return lax.scan(lambda c, _: (c + delta, None), initial, None, length=length)[0]

    result = compare_functions(
        left,
        lambda initial, delta: initial + length * delta,
        jnp.ones(2),
        jnp.ones(2),
        modes=("forward", "jvp", "vjp"),
    )
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("wrong", ["subtract", "input", "double", "initial", "axis", "feedback"])
def test_non_equivalent_accumulators_never_match_sum(wrong):
    def left(x, y):
        initial = jnp.ones(3) if wrong == "initial" else jnp.zeros(3)
        source = y if wrong == "input" else x

        def body(c, a):
            if wrong == "subtract":
                return c - a, None
            if wrong == "double":
                return c + c + a, None
            if wrong == "feedback":
                return c + c * a, None
            return c + a, None

        return lax.scan(body, initial, source)[0]

    right = (
        (lambda x, y: jnp.sum(x, axis=1)) if wrong == "axis" else (lambda x, y: jnp.sum(x, axis=0))
    )
    result = compare_functions(left, right, jnp.ones((3, 3)), jnp.ones((3, 3)))
    assert result.status != Status.EQUIVALENT, result


def test_cross_carry_dependency_is_not_independent_reduction():
    def left(x):
        def body(c, a):
            return (c[0] + a, c[1] + c[0]), None

        return lax.scan(body, (0.0, 0.0), x)[0][1]

    with pytest.raises(ComparisonContractError, match="output_avals"):
        compare_functions(left, lambda x: jnp.sum(x), jnp.ones(4))


@pytest.mark.parametrize("wrong", ["history", "output", "order"])
def test_sequence_values_and_output_order_are_not_erased(wrong):
    def left(x):
        return lax.scan(lambda c, a: (c + a, c if wrong == "history" else a), 0.0, x)

    def right(x):
        if wrong == "order":
            return x, jnp.sum(x)
        return jnp.sum(x), 2 * x if wrong == "output" else x

    if wrong in ("history", "order"):
        with pytest.raises(ComparisonContractError, match="output_avals"):
            compare_functions(left, right, jnp.ones(4))
    else:
        result = compare_functions(left, right, jnp.ones(4))
        assert result.status == Status.INEQUIVALENT


def test_partial_dead_body_is_rejected_and_scalar_time_capture_is_supported():
    def partial(x):
        def body(c, a):
            _ = jnp.log(a)
            return c + a, None

        return lax.scan(body, 0.0, x)[0]

    def opaque(x):
        return lax.scan(lambda c, a: (c + jnp.sin(a), None), 0.0, x)[0]

    x = jnp.ones(4)
    with pytest.raises(MathematicalUndecidedError, match="condition_undecided"):
        compare_functions(partial, lambda x: jnp.sum(x), x)
    result = compare_functions(opaque, lambda x: jnp.sum(jnp.sin(x)), x)
    assert result.status == Status.EQUIVALENT, result


def test_scan_sum_uses_symbolic_queries():
    x = jax.ShapeDtypeStruct((4, 2), jnp.float32)
    left = jax.make_jaxpr(lambda x: _scan(jnp.zeros(2), x))(x)
    right = jax.make_jaxpr(lambda x: jnp.sum(x, axis=0))(x)
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.EQUIVALENT, result
    assert result.backend_result.backend == "sympy"
    assert "scan:independent_accumulation" in result.rules


def test_scan_length_is_symbolic_and_query_work_is_budgeted():
    results = []
    for length in (5, 1000000):
        x = jax.ShapeDtypeStruct((length, 2), jnp.float32)
        left = jax.make_jaxpr(lambda x: _scan(jnp.zeros(2), x))(x)
        right = jax.make_jaxpr(lambda x: jnp.sum(x, axis=0))(x)
        result = compare_algebraic_jaxprs(left, right)
        assert result.status == Status.EQUIVALENT, result
        results.append(result)
        with pytest.raises(ComparisonBudgetError) as error:
            compare_algebraic_jaxprs(left, right, algebra_budget=1)
        denied = error.value
        assert "budget exceeded" in denied.reason
    assert results[0].work == results[1].work


def test_wrong_custom_gradient_is_not_accepted_by_forward_scan_rule():
    @jax.custom_jvp
    def wrong(x):
        return _scan(jnp.float32(0), x)

    @wrong.defjvp
    def derivative(primals, tangents):
        return wrong(primals[0]), jnp.sum(tangents[0]) * 2

    result = compare_functions(wrong, jnp.sum, jnp.ones(4), modes=("jvp", "vjp"))
    assert all(mode.status != Status.EQUIVALENT for mode in result.modes), result
