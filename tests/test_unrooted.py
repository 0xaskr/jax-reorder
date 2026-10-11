"""AI 固定的无根组件、全图共享、对称搜索及资源边界。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import Status, canonicalize, compare_jaxprs


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


def test_search_budget_reports_unknown_with_graph_size():
    a, b = (jax.make_jaxpr(fn)(jnp.float32(0.7)) for fn in (ring, ring_reordered))
    result = compare_jaxprs(a, b, search_budget=1)
    assert result.status == Status.UNKNOWN
    assert "8 nodes" in result.issues[0].reason
    assert "budget exceeded" in result.issues[0].reason
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
