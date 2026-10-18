"""公共数学路径完全信任 SymPy，且不执行 JAX 数组。

覆盖：
- 正例：标量恒等式、张量求和线性与空域、大求和保持惰性、矩阵结合、嵌套收缩与独立双重
  求和一致、dot 广播与 reshape、条件与 jit 绑定、共享调用的捕获与界限。
- 反例与拒绝：负结论必须覆盖完整最终输出；整数 dtype 值域、原始定义条件（x/x、
  exp(log(x))*0+1）、循环拒绝与 false 循环谓词、None 查询、布尔/数值 Piecewise 分支顺序、
  超时/预算、非法 timeout 值、转换异常都按契约分类，不把未决当作不等价。
- 恢复性：超时与远端异常之后的下一次查询仍可继续；worker 本身不执行 JAX。
- 前置条件：CPU 平台与 SymPy 1.14；helper 在比较期间 patch 掉编译与执行入口。
- 这些是固定公式与边界回归，不证明 SymPy 对所有程序都能在预算内判定。
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
import sympy as sp
from jax._src import compiler, dispatch

from jax_reorder import (
    ComparisonBudgetError,
    ComparisonInternalError,
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_functions,
)
from jax_reorder.algebra.adapter.transport import freeze
from jax_reorder.algebra.budget import Budget
from jax_reorder.algebra.symbolic import SymbolicBuilder
from jax_reorder.algebra.sympy_ops import equal, finite_expansion
from jax_reorder.algebra.trusted import compare_symbolic


def signature(shape=()):
    return jax.ShapeDtypeStruct(shape, jnp.float32)


def forbidden(*args, **kwargs):
    raise AssertionError("the trusted path must not execute JAX")


def compare(left, right, *args, **kwargs):
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        return compare_functions(left, right, *(args or (signature(),)), **kwargs).modes[0]


@pytest.mark.parametrize(
    "left,right",
    [
        (lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2, jnp.ones_like),
        (lambda x: jnp.log(jnp.exp(x)), lambda x: x),
        (lambda x: jnp.sqrt(x * x), jnp.abs),
        (lambda x: jnp.exp(x) * jnp.exp(2 * x), lambda x: jnp.exp(3 * x)),
        (lambda x: (x + 2) * 3, lambda x: 3 * x + 6),
        (lambda x: jnp.exp(x) / jnp.exp(x), jnp.ones_like),
    ],
)
def test_complete_scalar_queries(left, right):
    result = compare(left, right, signature((3,)))
    assert result.status == Status.EQUIVALENT, result
    assert result.algebra.correspondence is None
    assert result.algebra.backend_result.backend == "sympy"
    assert result.algebra.backend_result.queries
    assert result.equivalence_kind == "algebraic"
    assert result.algebra.warnings


@pytest.mark.parametrize(
    "left,right",
    [
        (lambda x: x + x, lambda x: x),
        (lambda x: jnp.exp(x), lambda x: jnp.exp(2 * x)),
        (lambda x: jnp.sqrt(x * x), lambda x: x),
        (lambda x: (x, x + 1), lambda x: (x + 1, x)),
    ],
)
def test_negative_is_a_complete_final_output(left, right):
    result = compare(left, right, signature((3,)))
    assert result.status == Status.INEQUIVALENT, result
    assert result.algebra.correspondence is None
    assert result.difference == f"algebra.outputs[{result.algebra.output}]"
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate")
        for q in result.algebra.backend_result.queries
    )


@pytest.mark.parametrize("left", [lambda x: x / x, lambda x: jnp.exp(jnp.log(x)) * 0 + 1])
def test_original_conditions_survive_cancellation(left):
    with pytest.raises(MathematicalUndecidedError) as error:
        compare(left, jnp.ones_like)
    result = error.value
    assert result.failure_kind == "condition_undecided"


@pytest.mark.parametrize("shape", [(), (3,)])
def test_integer_dtype_bounds_cannot_be_forgotten_in_negative_queries(shape):
    with pytest.raises(MathematicalUndecidedError) as error:
        compare(
            lambda x: jnp.where(x <= 127, jnp.float32(1), jnp.float32(2)),
            lambda x: jnp.ones_like(x, dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape, jnp.int8),
        )
    assert error.value.failure_kind == "cas_undecided"


def test_intermediate_difference_can_be_eliminated():
    assert compare(lambda x: (jnp.sin(x) + 1) * 0 + x, lambda x: x).status == Status.EQUIVALENT


@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (3, 4), (8, 32)])
def test_tensor_sum_linearity_and_empty_domains(shape):
    result = compare(lambda x: jnp.sum(2 * x), lambda x: 2 * jnp.sum(x), signature(shape))
    assert result.status == Status.EQUIVALENT, result


def test_large_sum_scale_remains_lazy_under_an_opaque_operation():
    result = compare(
        lambda x: jax.lax.top_k(jnp.sum(x, axis=1) / 16, 2),
        lambda x: jax.lax.top_k(jnp.sum(x / 16, axis=1), 2),
        signature((4, 576)),
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.algebra.backend_result.work < 1000
    assert any("Sum(" in definition for _, definition in result.algebra.backend_result.definitions)


def test_matrix_association_and_wrong_axis():
    args = [signature(shape) for shape in ((2, 3), (3, 4), (4, 5))]
    result = compare(lambda a, b, c: (a @ b) @ c, lambda a, b, c: a @ (b @ c), *args)
    assert result.status == Status.EQUIVALENT, result
    result = compare(lambda a, b: a @ b, lambda a, b: a @ b.T, signature((3, 3)), signature((3, 3)))
    assert result.status == Status.INEQUIVALENT, result


@pytest.mark.parametrize("function", [lambda a, b, c: (a @ b) @ c, lambda a, b, c: a @ (b @ c)])
def test_nested_contraction_conversion_matches_an_independent_double_sum(function):
    args = [signature(shape) for shape in ((2, 3), (3, 4), (4, 2))]
    output = SymbolicBuilder().graph(freeze(jax.make_jaxpr(function)(*args)))[0]
    a, b, c = [sp.IndexedBase(f"x{i}", shape=arg.shape, real=True) for i, arg in enumerate(args)]
    expected = sum(a[0, j] * b[j, k] * c[k, 0] for j in range(3) for k in range(4))
    actual = output.read((0, 0))[0].doit()
    assert sp.expand(actual - expected) == 0


def test_dot_broadcast_and_reshape():
    result = compare(
        lambda a, b: a @ b,
        lambda a, b: jnp.sum(a[:, :, None] * b[None, :, :], axis=1),
        signature((2, 3)),
        signature((3, 4)),
    )
    assert result.status == Status.EQUIVALENT, result
    result = compare(lambda x: x.reshape(12).reshape(3, 4), lambda x: x, signature((3, 4)))
    assert result.status == Status.EQUIVALENT, result


def test_conditions_and_jit_bindings():
    result = compare(
        lambda x: jax.lax.cond(x > 0, lambda y: y + y, lambda y: -y, x),
        lambda x: jnp.where(x > 0, 2 * x, -x),
    )
    assert result.status == Status.EQUIVALENT, result
    result = compare(jax.jit(lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2), jnp.ones_like)
    assert result.status == Status.EQUIVALENT, result


def test_unsupported_loop_is_not_an_inequality():
    with pytest.raises(UnsupportedJaxprError) as error:
        compare(lambda x: jax.lax.while_loop(lambda y: y < 10, lambda y: y + 1, x), lambda x: x + 1)
    result = error.value
    assert result.failure_kind == "unsupported_conversion"


def test_false_loop_predicate_keeps_its_original_conditions():
    with pytest.raises(UnsupportedJaxprError) as error:
        compare(
            lambda x: jax.lax.while_loop(lambda y: jnp.log(y) * 0 > 0, lambda y: y + 1, x),
            lambda x: x,
        )
    result = error.value
    assert result.failure_kind == "unsupported_conversion"


def test_timeout_budget_and_next_query_recovery():
    with pytest.raises(ComparisonTimeoutError) as error:
        compare(lambda x: x + x, lambda x: 2 * x, algebra_timeout=0.001)
    result = error.value
    assert result.failure_kind == "timeout"
    with pytest.raises(ComparisonBudgetError):
        compare(lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2, jnp.ones_like, algebra_budget=1)
    assert compare(lambda x: x + x, lambda x: 2 * x).status == Status.EQUIVALENT


def test_none_is_not_inequality(monkeypatch):
    monkeypatch.setattr(sp.Expr, "equals", lambda *a, **k: None)
    assert equal(sp.Symbol("x"), sp.S.Zero, Budget(100)) is None


@pytest.mark.parametrize("reverse", [False, True])
def test_boolean_piecewise_uses_logic_in_both_operand_orders(reverse):
    x = sp.Symbol("x", real=True)
    condition = sp.Piecewise((sp.false, x <= 0), (sp.Ne(x, 0), True))
    left, right = (sp.true, condition) if reverse else (condition, sp.true)
    methods = []
    assert equal(left, right, Budget(1000), methods=methods) is None
    assert "Piecewise.rewrite(ITE)" in methods
    assert "simplify(left-right)" not in methods
    simple_condition = sp.Piecewise((sp.true, x > 0), (sp.false, True))
    assert equal(simple_condition, x > 0, Budget(1000)) is True


def test_numeric_piecewise_keeps_arithmetic_comparison():
    x = sp.Symbol("x", real=True)
    expression = sp.Piecewise((-x, x < 0), (x, True))
    assert equal(expression, sp.Abs(x), Budget(1000)) is True


def test_tensor_construction_size_does_not_enumerate_input_elements():
    counts = []
    for size in (8, 32, 128, 512):
        graph = freeze(jax.make_jaxpr(lambda x: jnp.sum(x * x))(signature((size,))))
        output = SymbolicBuilder().graph(graph)[0]
        counts.append(len(set(sp.preorder_traversal(output.expression))))
        assert len(output.expression.atoms(sp.Indexed)) == 1
    assert len(set(counts)) == 1, counts


def test_nested_sum_expansion_accounts_for_multiplicative_work():
    i, j = sp.symbols("i j", integer=True)
    x = sp.IndexedBase("x", shape=(128, 128), real=True)
    expression = sp.Sum(sp.Sum(x[i, j], (j, 0, 127)) + x[i, 0], (i, 0, 127))
    assert not finite_expansion(expression, Budget(10000))


def test_worker_engine_itself_never_executes_jax():
    a = freeze(jax.make_jaxpr(lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2)(signature((3,))))
    b = freeze(jax.make_jaxpr(jnp.ones_like)(signature((3,))))
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        assert compare_symbolic(a, b).status == "equivalent"


def test_shared_call_keeps_captured_scan_time_and_sum_bounds():
    transform = jax.jit(jnp.sin)

    def scan(x):
        return jax.lax.scan(lambda carry, row: (carry + transform(row), None), 0.0, x)[0]

    with pytest.raises(MathematicalUndecidedError, match="condition_undecided"):
        compare(lambda x: 2 * scan(x[:2]), scan, signature((4,)))


def test_expansion_is_budgeted_before_large_polynomial_construction():
    with pytest.raises(ComparisonBudgetError) as error:
        compare(lambda x: (x + 1) ** 100000, lambda x: x, algebra_budget=100)
    result = error.value
    assert result.failure_kind == "budget_exhausted"


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), None])
def test_timeout_contract_applies_before_the_structural_fast_path(value):
    with pytest.raises(ValueError, match="algebra_timeout"):
        compare_functions(lambda x: x, lambda x: x, signature(), algebra_timeout=value)


def test_unexpected_conversion_exception_is_reported_and_next_call_recovers(monkeypatch):
    a = freeze(jax.make_jaxpr(lambda x: x + x)(signature()))
    b = freeze(jax.make_jaxpr(lambda x: 2 * x)(signature()))

    def fail(*args):
        raise RuntimeError("conversion failure")

    with monkeypatch.context() as context:
        context.setattr(SymbolicBuilder, "graph", fail)
        with pytest.raises(ComparisonInternalError) as error:
            compare_symbolic(a, b)
        result = error.value
        assert result.failure_kind == "internal_error"
    assert compare_symbolic(a, b).status == "equivalent"
