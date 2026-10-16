"""固定循环改写的回归：图结论、执行、幂等及错误参数路径分别核对。"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core

from examples import region_bindings as cases
from jax_reorder import (
    InvalidJaxprError,
    Status,
    UnsupportedJaxprError,
    canonicalize,
    compare_jaxprs,
)


def inputs(dtype=jnp.float32, shape=()):
    """非对称值避免错误绑定恰好输出相同。"""
    return tuple(jnp.full(shape, value, dtype) for value in (0.7, 1.25, -0.3))


def assert_equivalent(left, right, samples):
    graphs = tuple(jax.make_jaxpr(fn)(*samples[0]) for fn in (left, right))
    snapshots = tuple(str(graph) for graph in graphs)
    result = compare_jaxprs(*graphs)
    assert result.status == Status.EQUIVALENT, result
    for graph in graphs:
        normalized = canonicalize(graph)
        core.check_jaxpr(normalized.jaxpr)
        assert normalized.graph == canonicalize(normalized.jaxpr).graph
        assert normalized.jaxpr.invars == graph.invars
        assert normalized.jaxpr.outvars == graph.outvars
        for sample in samples:
            before = core.eval_jaxpr(graph, graph.consts, *sample)
            after = core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, *sample)
            for a, b in zip(before, after, strict=True):
                np.testing.assert_array_equal(a, b)
    for sample in samples:
        for a, b in zip(
            jax.tree.leaves(left(*sample)), jax.tree.leaves(right(*sample)), strict=True
        ):
            np.testing.assert_array_equal(a, b)
    assert tuple(str(graph) for graph in graphs) == snapshots


@pytest.mark.parametrize(
    "dtype,shape", [(jnp.float32, ()), (jnp.float32, (2, 3)), (jnp.bfloat16, ())]
)
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize(
    "original,reordered",
    [
        (cases.scan_original, cases.scan_reordered),
        (cases.scan_producers_original, cases.scan_producers_reordered),
        (cases.while_original, cases.while_reordered),
    ],
)
def test_loop_capture_reordering(original, reordered, steps, dtype, shape):
    sample = inputs(dtype, shape)
    assert_equivalent(partial(original, steps=steps), partial(reordered, steps=steps), [sample])


def test_while_condition_and_body_captures():
    samples = [(*inputs(), jnp.int32(n), jnp.float32(100)) for n in (0, 1, 4)]
    assert_equivalent(
        cases.while_both_regions_original, cases.while_both_regions_reordered, samples
    )


@pytest.mark.parametrize(
    "original,candidate",
    [
        (cases.scan_original, cases.scan_wrong_source),
        (cases.scan_original, cases.scan_wrong_scale),
        (cases.while_original, cases.while_wrong_source),
        (cases.while_original, cases.while_wrong_scale),
    ],
)
def test_wrong_loop_binding_has_graph_difference_and_witness(original, candidate):
    sample = inputs()
    graphs = tuple(jax.make_jaxpr(fn)(*sample) for fn in (original, candidate))
    result = compare_jaxprs(*graphs)
    assert result.status == Status.DIFFERENT, result
    assert any(
        not np.allclose(a, b)
        for a, b in zip(
            jax.tree.leaves(original(*sample)), jax.tree.leaves(candidate(*sample)), strict=True
        )
    )


@pytest.mark.parametrize(
    "original,reordered",
    [(cases.scan_original, cases.scan_reordered), (cases.while_original, cases.while_reordered)],
)
def test_raw_capture_order_really_changes(original, reordered):
    graphs = tuple(jax.make_jaxpr(fn)(*inputs()) for fn in (original, reordered))
    orders = []
    for graph in graphs:
        equation = next(eq for eq in graph.eqns if eq.primitive.name in {"scan", "while"})
        orders.append(tuple(graph.invars.index(v) for v in equation.invars if v in graph.invars))
    assert orders == [(0, 1, 2), (2, 0, 1)]


@pytest.mark.parametrize("fn", [cases.scan_original, cases.while_original])
def test_unknown_loop_metadata_is_rejected(fn):
    graph = jax.make_jaxpr(fn)(*inputs())
    equations = list(graph.eqns)
    index = next(i for i, eq in enumerate(equations) if eq.primitive.name in {"scan", "while"})
    equations[index] = equations[index].replace(
        params={**equations[index].params, "future_slots": (1, 0)}
    )
    malformed = graph.replace(eqns=equations)
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(malformed, malformed)
    result = error.value
    assert "unsupported" in result.reason


@pytest.mark.parametrize("fn", [cases.scan_original, cases.while_original])
def test_missing_loop_operand_is_rejected(fn):
    graph = jax.make_jaxpr(fn)(*inputs())
    equations = list(graph.eqns)
    index = next(i for i, eq in enumerate(equations) if eq.primitive.name in {"scan", "while"})
    equations[index] = equations[index].replace(invars=equations[index].invars[:-1])
    malformed = graph.replace(eqns=equations)
    with pytest.raises(InvalidJaxprError):
        compare_jaxprs(malformed, malformed)
