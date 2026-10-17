"""ArrayExpr 转换必须保留坐标映射与有限求和域，不能把下标简化为轴标签。

覆盖：
- 反例：偏移、步长、翻转、部分求和、坐标权重、广播轴六种改写都改变输出坐标映射，
  必须判 inequivalent，且负查询落在 ``output_coordinate`` 作用域。
- 正例：切片/步长/翻转视图保持同一坐标映射时，``view(x) + view(x)`` 与 ``2 * view(x)``
  是代数恒等。
- 规模边界：512×512 的合法 Indexed/Sum 仍走 SymPy 的 ArrayExpr（``doit``）查询路径，
  不因规模退化为不可判定。

前置条件：CPU 平台与 SymPy 1.14；纯符号查询不要求 JAX 后端执行。
这些是有限改写样本，不覆盖全部索引/广播组合。
"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jax_reorder import Status, compare_functions
from jax_reorder.algebra_budget import Budget
from jax_reorder.algebra_trusted import equal


@pytest.mark.parametrize(
    "left,right",
    [
        (lambda x: x[:3], lambda x: x[1:4]),
        (lambda x: x[:3], lambda x: x[::2]),
        (lambda x: x, lambda x: x[::-1]),
        (lambda x: jnp.sum(x[:3]), lambda x: jnp.sum(x)),
        (lambda x: jnp.sum(x), lambda x: jnp.sum(jnp.arange(6, dtype=jnp.float32) * x)),
        (
            lambda x: jnp.broadcast_to(x[:, None], (6, 6)),
            lambda x: jnp.broadcast_to(x[None, :], (6, 6)),
        ),
    ],
    ids=["offset", "stride", "reverse", "partial_sum", "coordinate_weight", "broadcast_axis"],
)
def test_changed_coordinate_maps_are_inequivalent(left, right):
    """下标映射改变即为不同函数；负结论必须由输出坐标查询支撑。"""
    spec = jax.ShapeDtypeStruct((6,), jnp.float32)
    mode = compare_functions(left, right, spec).modes[0]
    assert mode.status == Status.INEQUIVALENT, mode.issues
    # 要求至少一条 output_coordinate 作用域的查询返回 False，排除靠不相关查询蒙对。
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate")
        for q in mode.algebra.backend_result.queries
    )


@pytest.mark.parametrize("view", [lambda x: x[1:4], lambda x: x[::2], lambda x: x[::-1]])
def test_same_coordinate_maps_still_allow_algebraic_rewrites(view):
    """只要两侧视图坐标映射相同，加法的代数改写仍应被接受。"""
    spec = jax.ShapeDtypeStruct((6,), jnp.float32)
    mode = compare_functions(lambda x: view(x) + view(x), lambda x: 2 * view(x), spec).modes[0]
    assert mode.status == Status.EQUIVALENT, mode.issues


def test_large_matrix_association_still_uses_array_expressions():
    """大矩阵结合律必须在 ArrayExpr 路径内完成，而不是退化成逐元素展开。"""
    i, j, k, column = sp.symbols("i j k column", integer=True, nonnegative=True)
    a, b, c = (sp.IndexedBase(name, shape=(512, 512), real=True) for name in "abc")
    left = sp.Sum(sp.Sum(a[i, j] * b[j, k], (j, 0, 511)) * c[k, column], (k, 0, 511))
    right = sp.Sum(a[i, j] * sp.Sum(b[j, k] * c[k, column], (k, 0, 511)), (j, 0, 511))
    methods = []
    # methods 由查询过程原地追加，用来确认实际走的是 ArrayExpr.doit。
    assert equal(left, right, Budget(100), indices=(i, column), methods=methods) is True
    assert methods == ["ArrayExpr.doit"]
