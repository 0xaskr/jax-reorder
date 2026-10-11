"""深度压力与分支数量分开：共享子区域不展开为指数规模的树。"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax._src import core

from jax_reorder import Status, compare_jaxprs
from jax_reorder.canonical import _difference
from jax_reorder.semantics import Encoder, replace_nested


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
    assert _difference(left, right) is not None
    assert _difference(left, encoder.encode(value, "parameter")) is None
    assert replace_nested(value, lambda x: x) is value
