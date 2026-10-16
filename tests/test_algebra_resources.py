"""SymPy 比较的源图绑定、确定性查询与资源停止路径。"""

import jax
import jax.numpy as jnp
import pytest

from jax_reorder import (
    ComparisonBudgetError,
    Status,
    canonicalize,
    compare_algebraic_jaxprs,
)


def test_changed_source_constant_changes_mathematical_result():
    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    left = jax.make_jaxpr(lambda x: x + 2)(x)
    right = jax.make_jaxpr(lambda x: 2 + x)(x)
    result = compare_algebraic_jaxprs(left, right)
    changed = jax.make_jaxpr(lambda x: x + 3)(x)
    assert result.status == Status.EQUIVALENT
    assert compare_algebraic_jaxprs(changed, right).status == Status.INEQUIVALENT


def test_all_outputs_and_original_graph_are_preserved():
    args = (jax.ShapeDtypeStruct((2, 3), jnp.float32), jax.ShapeDtypeStruct((3, 4), jnp.float32))
    left = jax.make_jaxpr(lambda a, b: a @ b)(*args)
    right = jax.make_jaxpr(lambda a, b: jnp.sum(a[:, :, None] * b[None, :, :], axis=1))(*args)
    before = canonicalize(left).graph, canonicalize(right).graph
    first = compare_algebraic_jaxprs(left, right)
    second = compare_algebraic_jaxprs(right, left)
    assert first.status == second.status == Status.EQUIVALENT
    assert before == (canonicalize(left).graph, canonicalize(right).graph)
    assert (
        first.backend_result.queries == compare_algebraic_jaxprs(left, right).backend_result.queries
    )


def test_repeated_squaring_keeps_compact_powers():
    def huge(x):
        for _ in range(30):
            x = x * x
        return x

    x = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(huge)(x)
    other = jax.make_jaxpr(lambda x: huge(x) * 1)(x)
    result = compare_algebraic_jaxprs(graph, other, algebra_budget=1000)
    assert result.status == Status.EQUIVALENT, result
    assert result.work < 1000


def test_deep_chain_ends_with_a_located_budget_failure():
    def chain(x):
        for _ in range(300):
            x = x + 1
        return x

    x = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(chain)(x)
    other = jax.make_jaxpr(lambda x: chain(x) + 0)(x)
    with pytest.raises(ComparisonBudgetError) as error:
        compare_algebraic_jaxprs(graph, other, algebra_budget=100)
    result = error.value
    assert result.path.startswith("algebra")
    assert "budget" in result.reason
