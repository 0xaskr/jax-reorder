"""SymPy 比较的源图绑定、确定性查询与资源停止路径。

覆盖：
- 正例：x+2 与 2+x 等价；矩阵乘法与“广播乘积再求和”等价；重复平方的 30 层链保持紧凑
  幂表示；同一对图反向比较得到同一结论，查询列表逐项可复现。
- 反例：源码常量 x+2 改为 x+3 后判 inequivalent，证明结论绑定原始常量。
- 资源路径：300 层加法链在 algebra_budget=100 时抛带路径的 ComparisonBudgetError，
  错误定位在 algebra 阶段。
- 附加不变量：比较前后规范图不变，说明代数路径不修改原图。

前置条件：CPU 平台与 SymPy 1.14；纯符号查询不要求执行 JAX 数值。
预算阈值只证明这些固定规模下的行为，不给出通用的复杂度界。
"""

import jax
import jax.numpy as jnp
import pytest

from jax_reorder import (
    ComparisonBudgetError,
    Status,
    canonicalize,
    compare_algebraic_jaxprs,
)


def test_changed_source_constant_changes_mathematical_result():
    """等价证明绑定源图常量：换掉 x+2 中的一个常量就应判 inequivalent。"""
    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    left = jax.make_jaxpr(lambda x: x + 2)(x)
    right = jax.make_jaxpr(lambda x: 2 + x)(x)
    result = compare_algebraic_jaxprs(left, right)
    changed = jax.make_jaxpr(lambda x: x + 3)(x)
    assert result.status == Status.EQUIVALENT
    assert compare_algebraic_jaxprs(changed, right).status == Status.INEQUIVALENT


def test_all_outputs_and_original_graph_are_preserved():
    """结论不依赖比较方向；原图在比较前后保持不变，查询列表可复现。"""
    args = (jax.ShapeDtypeStruct((2, 3), jnp.float32), jax.ShapeDtypeStruct((3, 4), jnp.float32))
    left = jax.make_jaxpr(lambda a, b: a @ b)(*args)
    right = jax.make_jaxpr(lambda a, b: jnp.sum(a[:, :, None] * b[None, :, :], axis=1))(*args)
    # before 保存规范化后的图，比较结束后再取一次用于对比是否被修改。
    before = canonicalize(left).graph, canonicalize(right).graph
    first = compare_algebraic_jaxprs(left, right)
    second = compare_algebraic_jaxprs(right, left)
    assert first.status == second.status == Status.EQUIVALENT
    assert before == (canonicalize(left).graph, canonicalize(right).graph)
    # 同一对图重复比较应产生完全相同的查询记录，说明没有跨请求的隐藏状态。
    assert (
        first.backend_result.queries == compare_algebraic_jaxprs(left, right).backend_result.queries
    )


def test_repeated_squaring_keeps_compact_powers():
    """30 次平方用紧凑幂表示，工作量远小于预算，不需要逐项展开。"""

    def huge(x):
        for _ in range(30):
            x = x * x
        return x

    x = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(huge)(x)
    # other 只多乘一个 1，等价性应能由紧凑表示直接确认。
    other = jax.make_jaxpr(lambda x: huge(x) * 1)(x)
    result = compare_algebraic_jaxprs(graph, other, algebra_budget=1000)
    assert result.status == Status.EQUIVALENT, result
    assert result.work < 1000


def test_deep_chain_ends_with_a_located_budget_failure():
    """预算不足时抛错并给出 algebra 阶段定位，不返回“未决等于不等价”的结论。"""

    def chain(x):
        for _ in range(300):
            x = x + 1
        return x

    x = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(chain)(x)
    other = jax.make_jaxpr(lambda x: chain(x) + 0)(x)
    with pytest.raises(ComparisonBudgetError) as error:
        compare_algebraic_jaxprs(graph, other, algebra_budget=100)
    result = error.value
    assert result.path.startswith("algebra")
    assert "budget" in result.reason
