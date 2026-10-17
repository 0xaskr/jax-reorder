"""从真实 jaxpr 查询 SymPy 标量恒等式，保留定义域与嵌套求和作用域。

覆盖：
- 正例：exp 的乘法/除法/零点、常数乘除、嵌套 exp、相同部分函数相除、求和相除等恒等式
  由可信 SymPy 直接判定；相同部分定义域的两侧仍能给出正结论。
- 反例与拒绝：部分函数（0*log、x/x 等缺少非零证据）、非有限值、缺失的非零事实不能消项；
  内外求和的作用域必须保持，不能因重名把外层读取捕获到内层。
- 规模：大归约保持符号表示，不逐元素展开；定义域已知的 exp 可以与归约组合推理。
- 前置条件：CPU 平台与 SymPy 1.14；helper 在比较期间 patch 掉编译与 primitive 执行，
  因此这些结论不依赖 JAX 数值执行。
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from jax._src import compiler, dispatch

from jax_reorder import (
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
)


def forbidden(*args, **kwargs):
    """任何真实数组执行都应失败：本文件的结论只允许来自符号查询。"""
    raise AssertionError("symbolic comparison must not execute JAX")


def compare(left, right, shape=(5,), **options):
    """在禁止执行的环境里追踪两侧函数并做代数比较。"""
    value = jax.ShapeDtypeStruct(shape, jnp.float32)
    # 先枚举设备，避免把“初始化后端”误判成执行。
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        a, b = jax.make_jaxpr(left)(value), jax.make_jaxpr(right)(value)
        result = compare_algebraic_jaxprs(a, b, **options)
    return a, b, result


@pytest.mark.parametrize(
    "left,right",
    [
        (lambda x: jnp.exp(x) * jnp.exp(2 * x), lambda x: jnp.exp(3 * x)),
        (lambda x: jnp.exp(x) * jnp.exp(-x), jnp.ones_like),
        (lambda x: jnp.exp(x * 0), jnp.ones_like),
        (lambda x: jnp.exp(x * 0 + 0.5) ** 2, lambda x: jnp.exp(x * 0 + 1)),
        (lambda x: x / 2, lambda x: x * 0.5),
        (lambda x: (x + 2) / -2, lambda x: -0.5 * x - 1),
        (lambda x: jnp.exp(x / 2), lambda x: jnp.exp(x * 0.5)),
        (lambda x: jnp.exp(jnp.exp(x) / jnp.exp(2 * x)), lambda x: jnp.exp(jnp.exp(-x))),
        (lambda x: jnp.exp(x) / jnp.exp(2 * x), lambda x: jnp.exp(-x)),
        (lambda x: jnp.exp(x) / jnp.exp(x), jnp.ones_like),
        (
            lambda x: jnp.sum(jnp.exp(x)) / jnp.sum(jnp.exp(x)),
            lambda x: jnp.array(1.0, dtype=jnp.float32),
        ),
        (
            lambda x: -jnp.sum(jnp.exp(x)) / -jnp.sum(jnp.exp(x)),
            lambda x: jnp.array(1.0, dtype=jnp.float32),
        ),
        (lambda x: jnp.sum(jnp.exp(x) * jnp.exp(2 * x)), lambda x: jnp.sum(jnp.exp(3 * x))),
        (
            lambda x: jnp.exp(jnp.sum(x)) * jnp.exp(jnp.sum(2 * x)),
            lambda x: jnp.exp(jnp.sum(3 * x)),
        ),
    ],
)
def test_scalar_identities_use_trusted_sympy_from_actual_jaxprs(left, right):
    a, b, result = compare(left, right)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "algebraic"
    assert result.backend_result.backend == "sympy"
    assert result.backend_result.queries


@pytest.mark.parametrize(
    "left,right,shape",
    [
        (lambda x: x / x, jnp.ones_like, (5,)),
        (lambda x: jnp.sin(x) / jnp.sin(x), jnp.ones_like, (5,)),
        (lambda x: jnp.maximum(x, 0) / jnp.maximum(x, 0), jnp.ones_like, (5,)),
        (lambda x: x / 0, jnp.zeros_like, (5,)),
        (lambda x: jnp.exp(x / x), lambda x: jnp.exp(jnp.ones_like(x)), (5,)),
        (lambda x: jnp.exp(x / 0), lambda x: jnp.exp(x * 0), (5,)),
        (lambda x: jnp.exp((x - jnp.inf) / 2), lambda x: jnp.exp((x - jnp.inf) * 0.5), (5,)),
        (
            lambda x: jnp.sum(jnp.exp(x)) / jnp.sum(jnp.exp(x)),
            lambda x: jnp.array(1.0, dtype=jnp.float32),
            (0,),
        ),
        (
            lambda x: jnp.exp(x - jnp.inf) * jnp.exp(-x + jnp.inf),
            jnp.ones_like,
            (5,),
        ),
    ],
)
def test_partial_nonfinite_and_missing_nonzero_facts_are_not_erased(left, right, shape):
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare(left, right, shape)


def test_inner_and_outer_summation_scopes_are_preserved():
    def left(x):
        return jnp.sum(jnp.exp(jnp.sum(x, axis=0)) * jnp.exp(x), axis=1)

    def right(x):
        return jnp.sum(jnp.exp(jnp.sum(x, axis=0) + x), axis=1)

    a, b, result = compare(left, right, (3, 4))
    assert result.status == Status.EQUIVALENT, result
    _, _, wrong = compare(
        left, lambda x: jnp.sum(jnp.exp(jnp.sum(x, axis=1)[:, None] + x), axis=1), (3, 4)
    )
    assert wrong.status != Status.EQUIVALENT


def test_defined_exponential_composes_with_reductions():
    a, b, result = compare(
        lambda x: jnp.exp(jnp.sum(x, axis=0) / 2),
        lambda x: jnp.exp(jnp.sum(x * 0.5, axis=0)),
        (3, 4),
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.backend_result.queries


def test_large_reductions_remain_symbolic():
    reports = []
    for size in (20, 1_000_000):
        _, _, result = compare(
            lambda x: jnp.exp(jnp.sum(x)) * jnp.exp(jnp.sum(2 * x)),
            lambda x: jnp.exp(jnp.sum(3 * x)),
            (size,),
        )
        assert result.status == Status.EQUIVALENT
        reports.append(result)
    assert reports[0].work == reports[1].work


@pytest.mark.parametrize(
    "left,right",
    [
        (lambda x: jnp.exp(jnp.log(x) / 2), lambda x: jnp.exp(jnp.log(x) * 0.5)),
        (lambda x: jnp.exp(jnp.log(x)) ** 2, lambda x: jnp.exp(2 * jnp.log(x))),
    ],
)
def test_equal_partial_domains_are_preserved_in_positive_queries(left, right):
    _, _, result = compare(left, right)
    assert result.status == Status.EQUIVALENT, result
    domains = [query for query in result.backend_result.queries if query.scope == "definedness"]
    assert domains and all(query.answer is True for query in domains)
    assert all(query.left != "True" and query.right != "True" for query in domains)
