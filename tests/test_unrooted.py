"""AI 固定的无根组件、全图共享、对称搜索及资源边界。

覆盖：
- 正例：菱形/环等无根图在重排后保留每个节点并规范化到同一结果，对应关系可独立复核；
  未使用常量保留数量、内容与共享绑定；仅由无根节点使用的 jit/cond/scan/while/shard_map
  私有输入被共同规范化，donation 仍附着在实际值上。
- 反例：数值相等的不同节点、细化后颜色相同的图都不能证明图相等；cond 有根分支的锚点
  必须覆盖无根分支的旧编号。
- 资源：搜索预算随图规模不足时抛 ComparisonBudgetError，不用任意顺序兜底。

前置条件：CPU 平台；shard_map 组合需要 2 个逻辑设备（conftest.py）。
固定图不覆盖所有对称接口与无根拓扑。
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import ComparisonBudgetError, Status, canonicalize, compare_jaxprs
from jax_reorder.core.correspondence import verify_correspondence


def diamond(x, y):
    result = lax.sin(x)
    a = lax.cos(y)
    b = lax.sin(y)
    c = a - b
    _ = c * a
    return result


def diamond_reordered(x, y):
    b = lax.sin(y)
    a = lax.cos(y)
    c = a - b
    result = lax.sin(x)
    _ = c * a
    return result


def diamond_wrong_sharing(x, y):
    result = lax.sin(x)
    a = lax.cos(y)
    b = lax.sin(y)
    c = a - b
    _ = c * b
    return result


def ring(x):
    a = lax.sin(x)
    b = lax.sin(x)
    c = lax.sin(x)
    d = lax.sin(x)
    lax.sub(a, b)
    lax.sub(b, c)
    lax.sub(c, d)
    lax.sub(d, a)
    return x


def ring_reordered(x):
    c = lax.sin(x)
    d = lax.sin(x)
    b = lax.sin(x)
    a = lax.sin(x)
    lax.sub(d, a)
    lax.sub(b, c)
    lax.sub(a, b)
    lax.sub(c, d)
    return x


def two_pairs(x):
    a = lax.sin(x)
    b = lax.sin(x)
    c = lax.sin(x)
    d = lax.sin(x)
    lax.sub(a, b)
    lax.sub(b, a)
    lax.sub(c, d)
    lax.sub(d, c)
    return x


@pytest.mark.parametrize(
    "pair,args", [((diamond, diamond_reordered), (0.7, -0.3)), ((ring, ring_reordered), (0.7,))]
)
def test_unrooted_graph_preserves_every_node_and_is_canonical(pair, args):
    args = tuple(map(jnp.float32, args))
    a, b = (jax.make_jaxpr(fn)(*args) for fn in pair)
    original = str(a), str(b)
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT, result
    assert result.left.unused_equations
    assert result.left.complete
    assert original == (str(a), str(b))
    assert compare_jaxprs(b, a).status == Status.EQUIVALENT
    for raw, normalized in ((a, result.left), (b, result.right)):
        assert len(raw.eqns) == len(normalized.jaxpr.eqns)
        core.check_jaxpr(normalized.jaxpr)
        assert canonicalize(normalized.jaxpr).graph == normalized.graph
        np.testing.assert_array_equal(
            core.eval_jaxpr(raw, raw.consts, *args),
            core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, *args),
        )


def test_equal_values_or_equal_refined_colors_do_not_prove_graph_equality():
    a, b = (jax.make_jaxpr(fn)(jnp.float32(0.7)) for fn in (ring, two_pairs))
    assert compare_jaxprs(a, b).status == Status.DIFFERENT
    a, b = (
        jax.make_jaxpr(fn)(jnp.float32(0.7), jnp.float32(-0.3))
        for fn in (diamond, diamond_wrong_sharing)
    )
    assert compare_jaxprs(a, b).status == Status.DIFFERENT


def test_search_budget_raises_with_graph_size():
    a, b = (jax.make_jaxpr(fn)(jnp.float32(0.7)) for fn in (ring, ring_reordered))
    with pytest.raises(ComparisonBudgetError) as error:
        compare_jaxprs(a, b, search_budget=1)
    result = error.value
    assert "8 nodes" in result.reason
    assert "budget exceeded" in result.reason
    assert compare_jaxprs(a, b, search_budget=100).status == Status.EQUIVALENT


def test_unused_constants_preserve_count_content_and_shared_bindings():
    a = jax.make_jaxpr(lambda x: x)(jnp.float32(1))
    aval = a.invars[0].aval
    p, q = core.Var(aval), core.Var(aval)
    first = a.replace(constvars=[p, q], consts=[np.float32(2), np.float32(3)])
    second = a.replace(constvars=[q, p], consts=[np.float32(3), np.float32(2)])
    result = compare_jaxprs(first, second)
    assert result.status == Status.EQUIVALENT, result
    assert len(result.left.graph.constants) == 2
    assert (
        compare_jaxprs(first, second.replace(consts=[np.float32(3), np.float32(4)])).status
        == Status.DIFFERENT
    )
    assert compare_jaxprs(first, a).status == Status.DIFFERENT


def dead_inputs_left(a, b, live):
    lax.sin(a)
    lax.sin(b)
    return live


def dead_inputs_right(b, a, live):
    lax.sin(b)
    lax.sin(a)
    return live


def private_dead_inputs(region):
    """固定源码的同步换序，经各区域的实际 JAX 接口追踪。"""
    if region == "jit":
        left = jax.jit(dead_inputs_left, keep_unused=True)
        right = jax.jit(dead_inputs_right, keep_unused=True)
        return left, lambda x, y, z: right(y, x, z)
    if region == "cond":
        return (
            lambda x, y, z: lax.cond(z > 0, dead_inputs_left, dead_inputs_left, x, y, z),
            lambda x, y, z: lax.cond(z > 0, dead_inputs_right, dead_inputs_right, y, x, z),
        )
    if region == "scan":

        def left(x, y, z):
            return lax.scan(lambda c, _: (dead_inputs_left(x, y, c), ()), z, None, length=2)[0]

        def right(x, y, z):
            return lax.scan(lambda c, _: (dead_inputs_right(y, x, c), ()), z, None, length=2)[0]

        return left, right
    if region == "while":

        def left(x, y, z):
            return lax.while_loop(lambda c: c < 2, lambda c: dead_inputs_left(x, y, c) + 1, z)

        def right(x, y, z):
            return lax.while_loop(lambda c: c < 2, lambda c: dead_inputs_right(y, x, c) + 1, z)

        return left, right
    assert region == "shard_map"
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("devices",))
    spec = jax.sharding.PartitionSpec()
    wrap = partial(jax.shard_map, mesh=mesh, in_specs=(spec, spec, spec), out_specs=spec)
    left, right = wrap(dead_inputs_left), wrap(dead_inputs_right)
    return left, lambda x, y, z: right(y, x, z)


@pytest.mark.parametrize("region", ["jit", "cond", "scan", "while", "shard_map"])
def test_private_inputs_used_only_by_unrooted_nodes_are_jointly_canonicalized(region):
    left, right = private_dead_inputs(region)
    args = jnp.float32(0.7), jnp.float32(-0.3), jnp.float32(0)
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (left, right))
    before = str(a), str(b)
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT, result
    verify_correspondence(a, b, result.correspondence)
    assert compare_jaxprs(b, a).status == Status.EQUIVALENT
    assert before == (str(a), str(b))
    for raw, normalized in ((a, result.left), (b, result.right)):
        core.check_jaxpr(normalized.jaxpr)
        assert canonicalize(normalized.jaxpr).graph == normalized.graph
        np.testing.assert_array_equal(
            core.eval_jaxpr(raw, raw.consts, *args),
            core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, *args),
        )
    wrong = jax.make_jaxpr(lambda x, y, z: left(x, x, z))(*args)
    # 外部输出相同也不能合并两条独立输入与共享同一输入的完整图。
    assert compare_jaxprs(a, wrong).status == Status.DIFFERENT
    with pytest.raises(ComparisonBudgetError) as error:
        compare_jaxprs(a, b, search_budget=1)
    limited = error.value
    assert "interface search budget exceeded" in limited.reason


def test_unrooted_private_input_search_keeps_donation_attached_to_actual_value():
    left, right = private_dead_inputs("jit")
    args = jnp.float32(0.7), jnp.float32(-0.3), jnp.float32(0)
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (left, right))

    def donated(graph, slots):
        eq = graph.eqns[0]
        return graph.replace(eqns=[eq.replace(params={**eq.params, "donated_invars": slots})])

    a = donated(a, (True, False, False))
    assert compare_jaxprs(a, donated(b, (False, True, False))).status == Status.EQUIVALENT
    assert compare_jaxprs(a, donated(b, (True, False, False))).status == Status.DIFFERENT


def test_cond_rooted_branch_anchors_override_unrooted_branch_input_numbers():
    def left(x, y, z):
        return lax.cond(z > 0, lambda a, b, c: a - b + c, dead_inputs_left, x, y, z)

    def right(x, y, z):
        return lax.cond(z > 0, lambda b, a, c: a - b + c, dead_inputs_right, y, x, z)

    args = jnp.float32(3), jnp.float32(1), jnp.float32(0)
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (left, right))
    result = compare_jaxprs(a, b, search_budget=1)
    assert result.status == Status.EQUIVALENT, result
    verify_correspondence(a, b, result.correspondence)
