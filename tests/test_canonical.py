"""Graph invariants and counterexamples independent of the traversal code."""

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
    return jax.make_jaxpr(fn)(*args)


def alpha_rename(graph):
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
    assert len(result.graph.nodes) == 3
    assert result.graph.nodes[0].inputs == (("result", 1, 0), ("result", 2, 0))
    reverse = graph.replace(eqns=[graph.eqns[1], graph.eqns[0], graph.eqns[2]])
    assert compare_jaxprs(graph, reverse).status == Status.EQUIVALENT


def test_multiple_results_repeated_outputs_and_ports():
    graph = trace(lambda x: lax.top_k(x, 2), jnp.array([1.0, 3.0, 2.0]))
    result = canonicalize(graph)
    assert result.bfs_order == (0,)
    assert result.graph.outputs == (("result", 0, 0), ("result", 0, 1))
    swapped = graph.replace(outvars=list(reversed(graph.outvars)))
    assert compare_jaxprs(graph, swapped).status == Status.DIFFERENT
    repeated = graph.replace(outvars=[graph.outvars[0], graph.outvars[0]])
    assert canonicalize(repeated).graph.outputs == (("result", 0, 0),) * 2


def test_unused_equations_are_retained_and_fully_compared():
    def fn(x, y):
        lax.cos(y)
        return lax.sin(x)

    graph = trace(fn, 1.0, 2.0)
    result = canonicalize(graph)
    assert result.complete
    assert result.unused_equations == (0,)
    assert result.equation_order == (1, 0)
    assert_preserved(graph, result, 1.0, 2.0)
    assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT
    without_unused = trace(lambda x, y: lax.sin(x), 1.0, 2.0)
    assert compare_jaxprs(graph, without_unused).status == Status.DIFFERENT


@pytest.mark.parametrize("fn", [lambda x: x, lambda x: 3, lambda x: (), lambda x: (x, x)])
def test_source_and_empty_outputs(fn):
    graph = trace(fn, 1.0)
    result = canonicalize(graph)
    assert result.complete
    assert result.graph.nodes == ()
    assert_preserved(graph, result, 1.0)


def test_closed_constants_are_numbered_by_use_not_original_slots():
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
    assert np.array_equal(left.consts[0], right.consts[1])
    assert compare_jaxprs(left, right).status == Status.EQUIVALENT
    changed = right.replace(consts=[right.consts[0], np.array([2.0, 9.0], np.float32)])
    assert compare_jaxprs(left, changed).status == Status.DIFFERENT


def test_parameter_dtype_shape_and_signed_zero_differences():
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
    graph = trace(lambda x: lax.cos(lax.sin(x)), 1.0)
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
    def fn(x):
        jax.debug.print("{x}", x=x)
        return lax.sin(x)

    graph = trace(fn, 1.0)
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "effect" in result.reason


def test_long_chain_does_not_recurse_on_dependencies():
    def fn(x):
        for _ in range(1200):
            x = lax.neg(x)
        return x

    graph = trace(fn, 1.0)
    result = canonicalize(graph)
    assert result.complete and max(result.depths) == 1200
    assert compare_jaxprs(graph, alpha_rename(graph)).status == Status.EQUIVALENT
