"""cond 绑定重排的独立正反例；候选源码固定，不由比较器生成。"""

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


def producers_candidate(flag, x, y):
    b = lax.cos(y)
    a = lax.sin(x)

    def selected():
        negative_b = -b
        negative_a = -a
        return negative_a - negative_b

    # 假分支保持原定义顺序；真分支仅重排两个独立 neg，捕获 b 的语句提前。
    def selected_original():
        negative_a = -a
        negative_b = -b
        return negative_a - negative_b

    return lax.cond(flag, selected, selected_original)


def producers_base(flag, x, y):
    a = lax.sin(x)
    b = lax.cos(y)

    def selected():
        negative_a = -a
        negative_b = -b
        return negative_a - negative_b

    return lax.cond(flag, selected, selected)


def nested_original(flag, inner, x, y):
    def branch():
        return lax.cond(inner, lambda: lax.sin(x) - lax.cos(y), lambda: x)

    return lax.cond(flag, branch, lambda: y)


def nested_candidate(flag, inner, x, y):
    def branch():
        def selected():
            b = lax.cos(y)
            a = lax.sin(x)
            return a - b

        return lax.cond(inner, selected, lambda: x)

    return lax.cond(flag, branch, lambda: y)


def switch_original(index, x, y, count):
    return lax.switch(
        index,
        (
            lambda args: (args[0], args[1], args[2]),
            lambda args: (args[0] - args[1], args[1], args[2]),
            lambda args: (args[0] + args[1], -args[1], args[2] + 1),
        ),
        (x, y, count),
    )


def switch_candidate(index, x, y, count):
    return lax.switch(
        index,
        (
            lambda args: (args[2], args[0], args[1]),
            lambda args: (args[2] - args[0], args[0], args[1]),
            lambda args: (args[2] + args[0], -args[0], args[1] + 1),
        ),
        (y, count, x),
    )


def shared_original(flag, x, y):
    def yes(a, b):
        return a - b

    def no(a, b):
        return b - a

    return lax.cond(flag, yes, no, x, y), lax.cond(flag, yes, no, y, x)


def shared_candidate(flag, x, y):
    def yes(b, a):
        return a - b

    def no(b, a):
        return b - a

    return lax.cond(flag, yes, no, y, x), lax.cond(flag, yes, no, x, y)


def repeated_original(flag, x):
    return lax.cond(flag, lambda a: (a[0] - a[1]) + a[2], lambda a: a[1], (x, x, 2.0))


def repeated_candidate(flag, x):
    return lax.cond(flag, lambda a: (a[2] - a[1]) + a[0], lambda a: a[1], (2.0, x, x))


BIAS = np.array([0.5, -0.75], np.float32)


def constant_original(flag, x):
    return lax.cond(flag, lambda: x - BIAS, lambda: BIAS)


def constant_candidate(flag, x):
    return lax.cond(flag, lambda args: args[1] - args[0], lambda args: args[0], (BIAS, x))


def assert_match_and_preserved(left, right, samples):
    """验证完整比较、两侧执行、幂等及原始图不变，不以数值替代图结论。"""
    graphs = [jax.make_jaxpr(fn)(*samples[0]) for fn in (left, right)]
    snapshots = [str(graph) for graph in graphs]
    comparison = compare_jaxprs(*graphs)
    assert comparison.status == Status.EQUIVALENT, comparison
    for graph in graphs:
        normalized = canonicalize(graph)
        core.check_jaxpr(normalized.jaxpr)
        assert normalized.jaxpr.invars == graph.invars
        assert normalized.jaxpr.outvars == graph.outvars
        assert canonicalize(normalized.jaxpr).graph == normalized.graph
        before = jax.jit(lambda *args, g=graph: core.eval_jaxpr(g, g.consts, *args))
        after = jax.jit(lambda *args, g=normalized.jaxpr: core.eval_jaxpr(g, g.consts, *args))
        for args in samples:
            for a, b in zip(before(*args), after(*args), strict=True):
                np.testing.assert_array_equal(a, b)
    for args in samples:
        for a, b in zip(jax.tree.leaves(left(*args)), jax.tree.leaves(right(*args)), strict=True):
            np.testing.assert_array_equal(a, b)
    assert [str(graph) for graph in graphs] == snapshots


@pytest.mark.parametrize(
    "left,right",
    [(producers_base, producers_candidate), (shared_original, shared_candidate)],
)
def test_producer_numbering_and_shared_branches(left, right):
    samples = [(jnp.bool_(flag), jnp.float32(0.7), jnp.float32(-0.3)) for flag in (False, True)]
    assert_match_and_preserved(left, right, samples)


def test_recursive_cond_bindings():
    samples = [
        (jnp.bool_(a), jnp.bool_(b), jnp.float32(0.7), jnp.float32(-0.3))
        for a in (False, True)
        for b in (False, True)
    ]
    assert_match_and_preserved(nested_original, nested_candidate, samples)


def test_switch_direct_inputs_multiple_outputs_and_mixed_types():
    samples = [
        (jnp.int32(i), jnp.array([1.0, 3.0]), jnp.float32(0.5), jnp.int32(8))
        for i in (-1, 0, 1, 2, 3)
    ]
    assert_match_and_preserved(switch_original, switch_candidate, samples)


@pytest.mark.parametrize(
    "left,right", [(repeated_original, repeated_candidate), (constant_original, constant_candidate)]
)
def test_repeated_actuals_literals_and_captured_constants(left, right):
    samples = [(jnp.bool_(flag), jnp.array([1.0, 3.0])) for flag in (False, True)]
    assert_match_and_preserved(left, right, samples)


def binding_original(flag, x, y):
    return lax.cond(flag, lambda a, b: a - b, lambda a, b: a + b, x, y)


def binding_wrong_arguments(flag, x, y):
    return lax.cond(flag, lambda a, b: a - b, lambda a, b: a + b, y, x)


def binding_wrong_branch(flag, x, y):
    return lax.cond(flag, lambda b, a: a - b, lambda a, b: a + b, x, y)


def binding_swapped_branches(flag, x, y):
    return lax.cond(flag, lambda a, b: a + b, lambda a, b: a - b, x, y)


def binding_changed_predicate(flag, x, y):
    return lax.cond(~flag, lambda a, b: a - b, lambda a, b: a + b, x, y)


def binding_changed_sharing(flag, x, y):
    return lax.cond(flag, lambda a, b: a - b, lambda a, b: a + b, x, x)


def binding_changed_constant(flag, x, y):
    return lax.cond(flag, lambda a, b: (a - b) + 1.0, lambda a, b: a + b, x, y)


@pytest.mark.parametrize(
    "candidate",
    [
        binding_wrong_arguments,
        binding_wrong_branch,
        binding_swapped_branches,
        binding_changed_predicate,
        binding_changed_sharing,
        binding_changed_constant,
    ],
)
def test_wrong_binding_is_different_with_witness(candidate):
    args = (jnp.bool_(True), jnp.float32(3), jnp.float32(1))
    graphs = [jax.make_jaxpr(fn)(*args) for fn in (binding_original, candidate)]
    assert compare_jaxprs(*graphs).status == Status.DIFFERENT
    assert float(binding_original(*args)) != float(candidate(*args))


def test_unused_branch_computations_are_fully_compared():
    def unused(flag, x, y):
        def selected():
            lax.sin(y)
            return x

        return lax.cond(flag, selected, lambda: y)

    args = (jnp.bool_(True), jnp.float32(3), jnp.float32(1))
    graph = jax.make_jaxpr(unused)(*args)
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.EQUIVALENT
    normalized = next(eq for eq in result.left.jaxpr.eqns if eq.primitive is lax.cond_p)
    assert sum(len(branch.eqns) for branch in normalized.params["branches"]) == 1


def test_multiple_unused_slots_are_retained_and_matched():
    def unused(flag, x, y, z):
        return lax.cond(flag, lambda a, b, c: a, lambda a, b, c: -a, x, y, z)

    args = (jnp.bool_(True), jnp.float32(3), jnp.float32(1), jnp.float32(2))
    graph = jax.make_jaxpr(unused)(*args)
    result = canonicalize(graph)
    assert result.complete
    assert len(result.graph.nodes[0].inputs) == 4
    assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT


@pytest.mark.parametrize("corruption", ["arity", "branches", "parameter", "cycle"])
def test_malformed_or_unsupported_cond_is_explicit(corruption):
    args = (jnp.bool_(True), jnp.float32(3), jnp.float32(1))
    original = jax.make_jaxpr(binding_original)(*args)
    snapshot = str(original)
    # make_jaxpr 可能复用缓存图；只破坏副本，避免污染其他参数化案例。
    graph = original.replace(eqns=list(original.eqns))
    index = next(i for i, eq in enumerate(graph.eqns) if eq.primitive is lax.cond_p)
    eq = graph.eqns[index]
    params = dict(eq.params)
    if corruption == "arity":
        branches = params["branches"]
        params["branches"] = (branches[0].replace(invars=branches[0].invars[:-1]), branches[1])
    elif corruption == "branches":
        params["branches"] = ()
    elif corruption == "parameter":
        params["unreviewed_operand_metadata"] = (True, False)
    else:
        branch = params["branches"][0]
        cyclic = branch.replace(eqns=list(branch.eqns))
        predicate = core.Literal(np.int32(1), core.ShapedArray((), np.dtype(np.int32)))
        cyclic.eqns[0] = cyclic.eqns[0].replace(
            primitive=lax.cond_p,
            invars=[predicate, *cyclic.invars],
            params={"branches": (cyclic, cyclic)},
        )
        params["branches"] = (cyclic, params["branches"][1])
    graph.eqns[index] = eq.replace(params=params)
    with pytest.raises(
        UnsupportedJaxprError if corruption == "parameter" else InvalidJaxprError
    ) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    if corruption == "parameter":
        assert "unsupported cond parameters" in result.reason
    elif corruption == "cycle":
        assert "cyclic nested jaxpr" in result.reason
    assert str(original) == snapshot


def test_debug_argument_names_follow_their_variables():
    args = (jnp.bool_(True), jnp.float32(3), jnp.float32(1))
    graph = jax.make_jaxpr(shared_original)(*args)
    normalized = canonicalize(graph)
    old = [eq for eq in graph.eqns if eq.primitive is lax.cond_p]
    new = [eq for eq in normalized.jaxpr.eqns if eq.primitive is lax.cond_p]
    assert old[0].invars != new[0].invars
    for before, after in zip(old, new, strict=True):
        for a, b in zip(before.params["branches"], after.params["branches"], strict=True):
            assert a.debug_info.arg_names is not None
            labels = dict(zip(a.invars, a.debug_info.arg_names, strict=True))
            assert b.debug_info.arg_names == tuple(labels[v] for v in b.invars)


def test_invalid_non_graph_input_has_explicit_error():
    with pytest.raises(InvalidJaxprError):
        canonicalize([])
