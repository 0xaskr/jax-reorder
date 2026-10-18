"""规范化的图不变量，以及对遍历实现保持独立的负例。

覆盖：
- 正例：BFS 编号与依赖深度是两种顺序；所有合法执行调度叠加 alpha 重命名后得到同一
  规范图；不可区分的重复调用仍是独立节点；未使用方程保留并参与完整比较；源图与空输出
  图也能正常规范化；闭合常量按使用顺序而非原始槽位编号。
- 反例：操作数位置、输出端口、primitive、共享关系、多结果端口交换、常量内容、
  参数 dtype/shape、有符号零与正负无穷的差异都判为 different。
- 拒绝路径：未定义变量、重复输出、前向引用等非法图抛 InvalidJaxprError；
  jax.debug.print 这类未支持 effect 抛 UnsupportedJaxprError。

这是有限样本上的回归，不证明任意 JAX 程序的等价性；不涉及多设备、上游快照或数值抽样。
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import (
    InvalidJaxprError,
    Status,
    UnsupportedJaxprError,
    canonicalize,
    compare_jaxprs,
)


def trace(fn, *args):
    """追踪一个具体调用的 jaxpr；所有用例都基于同一入口，便于互换原图与变体。"""
    return jax.make_jaxpr(fn)(*args)


def alpha_rename(graph):
    """按对象身份替换全部 Var，保留 aval 和 effects，用来验证结论不依赖变量名。"""
    variables = (*graph.constvars, *graph.invars, *(v for e in graph.eqns for v in e.outvars))
    names = {v: core.Var(v.aval) for v in variables}

    def rename(atom):
        return atom if isinstance(atom, core.Literal) else names[atom]

    return core.Jaxpr(
        [names[v] for v in graph.constvars],
        [names[v] for v in graph.invars],
        [rename(v) for v in graph.outvars],
        [
            e.replace(invars=[rename(v) for v in e.invars], outvars=[names[v] for v in e.outvars])
            for e in graph.eqns
        ],
        effects=graph.effects,
        consts=graph.consts,
        debug_info=graph.debug_info,
    )


def assert_preserved(graph, result, *args):
    """核对规范化没有删除或改写语义：方程映射、接口列表、执行结果与幂等性。

    equation_order 把规范化后的每个位置映射回原图方程下标，因此可以逐项比较
    primitive、操作数、结果端口和 ctx；随后实际 eval_jaxpr 两份图做数值对照。
    """
    core.check_jaxpr(result.jaxpr)
    assert len(graph.eqns) == len(result.jaxpr.eqns)
    for index, eq in zip(result.equation_order, result.jaxpr.eqns, strict=True):
        old = graph.eqns[index]
        assert eq.primitive is old.primitive
        assert eq.invars == old.invars
        assert eq.outvars == old.outvars
        assert eq.ctx is old.ctx
    assert result.jaxpr.invars == graph.invars
    assert result.jaxpr.outvars == graph.outvars
    assert result.jaxpr.constvars == graph.constvars
    for before, after in zip(
        core.eval_jaxpr(graph, graph.consts, *args),
        core.eval_jaxpr(result.jaxpr, result.jaxpr.consts, *args),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)
    again = canonicalize(result.jaxpr)
    assert again.graph == result.graph
    assert again.jaxpr.eqns == result.jaxpr.eqns


def test_bfs_and_depth_are_different_orders():
    def fn(x):
        a = lax.sin(x)
        b = lax.cos(a)
        return lax.add(a, b)

    graph = trace(fn, 1.0)
    result = canonicalize(graph)
    # sin 从输出倒推先被访问，所以 BFS 编号 (2, 0, 1) 与依赖深度 (1, 2, 3) 不同；
    # 可执行顺序仍按依赖深度取 (0, 1, 2)。
    assert result.bfs_order == (2, 0, 1)
    assert result.depths == (1, 2, 3)
    assert result.equation_order == (0, 1, 2)
    assert result.value_references[graph.outvars[0]] == ("result", 0, 0)
    assert_preserved(graph, result, 1.0)


def test_all_legal_schedules_and_alpha_renaming():
    def fn(x, y):
        a = lax.sin(x)
        b = lax.cos(y)
        c = lax.neg(a)
        return lax.add(a, b), lax.mul(c, b)

    graph = trace(fn, 1.0, 2.0)
    expected = canonicalize(graph)
    count = 0
    # 枚举方程的全部排列，只接受满足“先定义后使用”的合法调度；每个合法调度都应与
    # 原图规范化到同一结果，并与原图比较为 equivalent。
    for permutation in itertools.permutations(graph.eqns):
        bound = set(graph.invars)
        valid = True
        for equation in permutation:
            if any(isinstance(v, core.Var) and v not in bound for v in equation.invars):
                valid = False
                break
            bound.update(equation.outvars)
        if not valid:
            continue
        candidate = alpha_rename(graph.replace(eqns=permutation))
        normalized = canonicalize(candidate)
        assert normalized.graph == expected.graph
        assert compare_jaxprs(graph, candidate).status == Status.EQUIVALENT
        assert_preserved(candidate, normalized, 1.0, 2.0)
        count += 1
    # 断言至少覆盖多个合法调度，防止枚举逻辑退化成只检查原顺序。
    assert count > 4


@pytest.mark.parametrize("change", ["operand", "output", "primitive", "sharing"])
def test_real_structural_differences(change):
    def original(x, y):
        a = lax.sin(x)
        return lax.sub(a, y), a

    variants = {
        "operand": lambda x, y: (lax.sub(y, lax.sin(x)), lax.sin(x)),
        "output": lambda x, y: (lax.sin(x), lax.sub(lax.sin(x), y)),
        "primitive": lambda x, y: (lax.add(lax.sin(x), y), lax.sin(x)),
        # sharing 变体调用两次 sin，数值结果相同但不再共享同一个节点。
        "sharing": lambda x, y: (lax.sub(lax.sin(x), y), lax.sin(x)),
    }
    result = compare_jaxprs(trace(original, 1.0, 2.0), trace(variants[change], 1.0, 2.0))
    assert result.status == Status.DIFFERENT
    assert result.difference.startswith("graph.")


def test_indistinguishable_calls_remain_distinct_nodes():
    def fn(x):
        a, b = lax.sin(x), lax.sin(x)
        return lax.sub(a, b)

    graph = trace(fn, 1.0)
    result = canonicalize(graph)
    # 两次 sin 内容相同但不是同一节点：结果端口分别由两次调用产生。
    assert len(result.graph.nodes) == 3
    assert result.graph.nodes[0].inputs == (("result", 1, 0), ("result", 2, 0))
    # 只交换这两次调用不改变依赖，因此仍应判为 equivalent。
    reverse = graph.replace(eqns=[graph.eqns[1], graph.eqns[0], graph.eqns[2]])
    assert compare_jaxprs(graph, reverse).status == Status.EQUIVALENT


def test_multiple_results_repeated_outputs_and_ports():
    """多输出原语：交换输出端口是语义差异，重复输出同一 Var 则合法且保留重复。"""
    graph = trace(lambda x: lax.top_k(x, 2), jnp.array([1.0, 3.0, 2.0]))
    result = canonicalize(graph)
    assert result.bfs_order == (0,)
    assert result.graph.outputs == (("result", 0, 0), ("result", 0, 1))
    # top_k 的两个结果不可交换：数值和下标角色不同。
    swapped = graph.replace(outvars=list(reversed(graph.outvars)))
    assert compare_jaxprs(graph, swapped).status == Status.DIFFERENT
    # 同一结果返回两次是合法接口，规范化后保留两个相同的输出引用。
    repeated = graph.replace(outvars=[graph.outvars[0], graph.outvars[0]])
    assert canonicalize(repeated).graph.outputs == (("result", 0, 0),) * 2


def test_unused_equations_are_retained_and_fully_compared():
    """未使用分支不能被 DCE 删除：它仍是完整图的一部分，少一个节点就应判 different。"""

    def fn(x, y):
        lax.cos(y)
        return lax.sin(x)

    graph = trace(fn, 1.0, 2.0)
    result = canonicalize(graph)
    assert result.unused_equations == (0,)
    assert result.equation_order == (1, 0)
    assert_preserved(graph, result, 1.0, 2.0)
    assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT
    without_unused = trace(lambda x, y: lax.sin(x), 1.0, 2.0)
    assert compare_jaxprs(graph, without_unused).status == Status.DIFFERENT


@pytest.mark.parametrize("fn", [lambda x: x, lambda x: 3, lambda x: (), lambda x: (x, x)])
def test_source_and_empty_outputs(fn):
    """没有方程的图（直通、常量、空输出、重复输出）也必须可以规范化并保持接口。"""
    graph = trace(fn, 1.0)
    result = canonicalize(graph)
    assert result.graph.nodes == ()
    assert_preserved(graph, result, 1.0)


def test_closed_constants_are_numbered_by_use_not_original_slots():
    """闭合常量按首次使用顺序编号；两份图只是定义顺序不同，常量槽位互换。"""
    a, b = np.array([2.0, 3.0], np.float32), np.array([4.0, 5.0], np.float32)

    def first(x):
        u = lax.mul(x, a)
        v = lax.add(x, b)
        return lax.sub(u, v)

    def second(x):
        v = lax.add(x, b)
        u = lax.mul(x, a)
        return lax.sub(u, v)

    x = jnp.ones(2)
    left, right = trace(first, x), trace(second, x)
    # 先被 mul 使用的 a 在两张图里落到不同的常量槽位，但比较仍应等价。
    assert np.array_equal(left.consts[0], right.consts[1])
    assert compare_jaxprs(left, right).status == Status.EQUIVALENT
    # 改常量内容后即使形状相同也必须判 different。
    changed = right.replace(consts=[right.consts[0], np.array([2.0, 9.0], np.float32)])
    assert compare_jaxprs(left, changed).status == Status.DIFFERENT


def test_parameter_dtype_shape_and_signed_zero_differences():
    """参数与常量差异必须精确判出，不能被数值容差掩盖。

    覆盖有符号零、正负无穷、归约轴（静态参数）、输入 shape 和 dtype：这些差异在
    equivalent 判定里都不可忽略。
    """
    for a, b in [(0.0, -0.0), (np.inf, -np.inf)]:
        assert compare_jaxprs(trace(lambda a=a: a), trace(lambda b=b: b)).status == Status.DIFFERENT
    x = jnp.arange(4.0).reshape(2, 2)
    a = trace(lambda x: lax.reduce_sum(x, (0,)), x)
    b = trace(lambda x: lax.reduce_sum(x, (1,)), x)
    assert compare_jaxprs(a, b).status == Status.DIFFERENT
    assert (
        compare_jaxprs(trace(lambda x: x, jnp.ones(2)), trace(lambda x: x, jnp.ones(3))).status
        == Status.DIFFERENT
    )
    assert (
        compare_jaxprs(
            trace(lambda x: x, jnp.ones(2)), trace(lambda x: x, jnp.ones(2, dtype=jnp.int32))
        ).status
        == Status.DIFFERENT
    )


@pytest.mark.parametrize("kind", ["undefined", "duplicate", "forward"])
def test_invalid_graphs_fail(kind):
    """三类非法图必须在规范化入口就被拒绝，且比较入口也抛同一异常。"""
    graph = trace(lambda x: lax.cos(lax.sin(x)), 1.0)
    # undefined：输出引用未绑定的 Var；duplicate：重复定义同一输出；
    # forward：把方程倒序，造成使用先于定义。
    if kind == "undefined":
        graph = graph.replace(outvars=[core.Var(graph.invars[0].aval)])
    elif kind == "duplicate":
        graph = graph.replace(eqns=[graph.eqns[0], graph.eqns[0]])
    else:
        graph = graph.replace(eqns=list(reversed(graph.eqns)))
    with pytest.raises(InvalidJaxprError):
        canonicalize(graph)
    with pytest.raises(InvalidJaxprError):
        compare_jaxprs(graph, graph)


def test_unsupported_effect_even_in_unused_branch():
    """未支持的 effect 即使位于未使用分支也必须拒绝，不能靠删除节点跳过语义。"""

    def fn(x):
        jax.debug.print("{x}", x=x)
        return lax.sin(x)

    graph = trace(fn, 1.0)
    # 抛出的是带路径和原因的异常对象，测试读取其 reason 确认拒绝理由与 effect 有关。
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "effect" in result.reason


def test_long_chain_does_not_recurse_on_dependencies():
    """1200 层链用于确认规范化按迭代而不是递归实现，不会触发 Python 递归上限。"""

    def fn(x):
        for _ in range(1200):
            x = lax.neg(x)
        return x

    graph = trace(fn, 1.0)
    result = canonicalize(graph)
    assert max(result.depths) == 1200
    assert compare_jaxprs(graph, alpha_rename(graph)).status == Status.EQUIVALENT
