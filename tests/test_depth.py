"""深度压力与分支数量分开：共享子区域不展开为指数规模的树。

覆盖：
- 正例：8/16/32 层受控共享区域（jit/cond 交替外壳）中的换序仍判 equivalent，且 lineage
  规模随深度线性增长（``< 5 * (depth + 1)``），证明没有把共享子图展开成树。
- 反例：同一外壳下换成 sin(x)+cos(x) 的候选判 DIFFERENT，说明深层比较没有退化成只看外壳。
- 边界：2000 层参数编码/差异/替换按迭代执行，不触发 Python 递归上限。

前置条件：CPU 平台；属于 stress 分组（pytest -m stress）。
这些固定深度只覆盖到 32 层，不给出任意深度的复杂度证明。
"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax._src import core

from jax_reorder import Status, compare_jaxprs
from jax_reorder.core.canonical import difference
from jax_reorder.core.semantics import Encoder, replace_nested


def first(x):
    a = lax.sin(x)
    b = lax.cos(x)
    return a - b


def second(x):
    b = lax.cos(x)
    a = lax.sin(x)
    return a - b


def nest(graph, depth):
    """构造受控 IR 压力夹具；两份候选函数由 AI 手写，夹具只增加相同外壳。"""
    template = jax.make_jaxpr(lambda x: lax.cond(True, lax.neg, lax.neg, x))(jnp.float32(1))
    call = jax.make_jaxpr(jax.jit(lax.neg))(jnp.float32(1))
    for index in range(depth):
        wrapper = call if index % 3 == 0 else template
        equations = list(wrapper.eqns)
        eq = equations[-1]
        params = (
            {**eq.params, "jaxpr": graph}
            if wrapper is call
            else {**eq.params, "branches": (graph, graph)}
        )
        equations[-1] = eq.replace(params=params)
        graph = wrapper.replace(eqns=equations)
    return graph


@pytest.mark.parametrize("depth", [8, 16, 32])
def test_deep_mixed_shared_regions(depth):
    a, b = (nest(jax.make_jaxpr(fn)(jnp.float32(1)), depth) for fn in (first, second))
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT, result
    assert len(result.left.lineage) < 5 * (depth + 1)
    core.check_jaxpr(result.left.jaxpr)
    wrong = nest(jax.make_jaxpr(lambda x: lax.sin(x) + lax.cos(x))(jnp.float32(1)), depth)
    assert compare_jaxprs(a, wrong).status == Status.DIFFERENT


def test_deep_parameter_encoding_comparison_and_replacement_are_iterative():
    encoder = Encoder(lambda *args: None, lambda *args: None)
    value, changed = 1, 2
    for _ in range(2000):
        value, changed = (value,), (changed,)
    left = encoder.encode(value, "parameter")
    right = encoder.encode(changed, "parameter")
    assert difference(left, right) is not None
    assert difference(left, encoder.encode(value, "parameter")) is None
    assert replace_nested(value, lambda x: x) is value
