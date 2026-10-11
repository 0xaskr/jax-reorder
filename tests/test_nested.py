"""Nested regions keep their interfaces and are normalized independently."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import Status, canonicalize, compare_jaxprs


def first(x):
    a = lax.sin(x)
    b = lax.cos(x)
    return lax.sub(a, b)


def second(x):
    b = lax.cos(x)
    a = lax.sin(x)
    return lax.sub(a, b)


def changed(x):
    a = lax.sin(x)
    b = lax.cos(x)
    return lax.add(a, b)


def wrapped(kind, body):
    if kind == "jit":
        return jax.jit(body)
    if kind == "cond":
        return lambda x: lax.cond(x > 0, body, lambda z: -z, x)
    if kind == "scan":
        return lambda x: lax.scan(lambda carry, _: (body(carry), carry), x, None, length=3)
    if kind == "while":

        def run(x):
            return lax.while_loop(
                lambda carry: carry[0] < 2,
                lambda carry: (carry[0] + 1, body(carry[1])),
                (jnp.int32(0), x),
            )[1]

        return run
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["jit", "cond", "scan", "while"])
def test_nested_equivalent_and_changed(kind):
    x = jnp.float32(0.7)
    left = jax.make_jaxpr(wrapped(kind, first))(x)
    right = jax.make_jaxpr(wrapped(kind, second))(x)
    wrong = jax.make_jaxpr(wrapped(kind, changed))(x)
    result = compare_jaxprs(left, right)
    assert result.status == Status.EQUIVALENT, result
    assert compare_jaxprs(left, wrong).status == Status.DIFFERENT
    for graph in (left, right):
        normalized = canonicalize(graph)
        core.check_jaxpr(normalized.jaxpr)
        assert normalized.graph == canonicalize(normalized.jaxpr).graph
        for a, b in zip(
            core.eval_jaxpr(graph, graph.consts, x),
            core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, x),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)


def test_nested_unused_work_is_preserved_and_compared():
    @jax.jit
    def body(x):
        lax.cos(x)
        return lax.sin(x)

    graph = jax.make_jaxpr(body)(jnp.float32(1))
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.EQUIVALENT
    assert len(result.left.jaxpr.eqns[0].params["jaxpr"].eqns) == 2


def test_branch_positions_and_scan_parameters_are_semantic():
    x = jnp.float32(1)
    a = jax.make_jaxpr(lambda x: lax.cond(x > 0, first, changed, x))(x)
    b = jax.make_jaxpr(lambda x: lax.cond(x > 0, changed, first, x))(x)
    assert compare_jaxprs(a, b).status == Status.DIFFERENT
    graphs = [
        jax.make_jaxpr(lambda x, n=n: lax.scan(lambda c, _: (lax.neg(c), c), x, None, length=n))(x)
        for n in (2, 3)
    ]
    assert compare_jaxprs(*graphs).status == Status.DIFFERENT
