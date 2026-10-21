"""Private-port invariants reduce search without discarding feedback or dead code."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import ComparisonBudgetError, Status, canonicalize, compare_jaxprs
from jax_reorder.core.correspondence import verify_correspondence


def _unused_carries(order, *, wrong_feedback=False):
    def permute(values):
        return tuple(values[i] for i in order)

    def restore(values):
        return tuple(values[order.index(i)] for i in range(8))

    def function(x):
        def condition(values):
            i, bound, *_ = restore(values)
            return i < bound

        def body(values):
            i, bound, k, ell, _, _, a, b = restore(values)
            next_k, next_ell = k + 1, ell + 1
            if wrong_feedback:
                next_k, next_ell = next_ell, next_k
            return permute(
                (
                    i + 1,
                    bound,
                    next_k,
                    next_ell,
                    a,
                    b,
                    lax.dynamic_slice(x, (k,), (2,)),
                    lax.dynamic_slice(x, (k,), (3,)),
                )
            )

        initial = (jnp.int32(0), jnp.int32(2), jnp.int32(0), jnp.int32(1))
        lax.while_loop(
            condition,
            body,
            permute((*initial, x[:2], x[:3], x[:2], x[:3])),
        )
        return x

    return function


@pytest.mark.parametrize(
    "order",
    [(7, 6, 5, 4, 3, 2, 1, 0), (2, 3, 0, 1, 6, 7, 4, 5), (1, 5, 3, 7, 0, 4, 2, 6)],
)
def test_types_and_feedback_consumers_refine_eight_unused_carries(order):
    x = jnp.arange(8, dtype=jnp.float32)
    left, right = (
        jax.make_jaxpr(_unused_carries(permutation))(x) for permutation in (tuple(range(8)), order)
    )
    before = str(left), str(right)
    result = compare_jaxprs(left, right, search_budget=1)
    assert result.status == Status.EQUIVALENT
    assert compare_jaxprs(right, left, search_budget=1).status == Status.EQUIVALENT
    verify_correspondence(left, right, result.correspondence)
    assert (str(left), str(right)) == before
    for raw, normalized in ((left, result.left), (right, result.right)):
        assert normalized.unused_equations
        assert canonicalize(normalized.jaxpr, search_budget=1).graph == normalized.graph
        core.check_jaxpr(normalized.jaxpr)
        np.testing.assert_array_equal(
            core.eval_jaxpr(raw, raw.consts, x),
            core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, x),
        )


def test_refinement_keeps_wrong_feedback_in_an_unused_loop():
    x = jnp.arange(8, dtype=jnp.float32)
    left, right = (
        jax.make_jaxpr(_unused_carries(tuple(range(8)), wrong_feedback=wrong))(x)
        for wrong in (False, True)
    )
    assert compare_jaxprs(left, right, search_budget=1).status == Status.DIFFERENT


def test_consumer_multiplicity_is_not_an_equivalence_proof():
    def function(x):
        lax.while_loop(
            lambda values: values[0] < 2,
            lambda values: tuple(value + 1 for value in values),
            (jnp.int32(0), x, x),
        )
        return x

    graph = jax.make_jaxpr(function)(jnp.int32(0))
    with pytest.raises(ComparisonBudgetError, match="interface search budget exceeded"):
        compare_jaxprs(graph, graph, search_budget=1)
    assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT


@pytest.mark.parametrize("kind", ["jit", "cond", "scan"])
def test_unused_result_types_remain_invariant_under_private_slot_permutation(kind):
    def function(order):
        def kernel(x):
            values = tuple(x[:size] for size in range(1, 9))

            def results(values):
                return tuple(value + 1 for value in values)

            private = tuple(values[i] for i in order)
            if kind == "jit":
                jax.jit(results)(private)
            elif kind == "cond":
                lax.cond(x[0] > 0, results, lambda xs: tuple(v - 1 for v in xs), private)
            else:
                lax.scan(lambda carry, _: (results(carry), None), private, None, length=2)
            return x

        return kernel

    x = jnp.arange(8, dtype=jnp.float32)
    left, right = (
        jax.make_jaxpr(function(order))(x) for order in (tuple(range(8)), (1, 7, 3, 5, 0, 6, 2, 4))
    )
    result = compare_jaxprs(left, right, search_budget=1)
    assert result.status == Status.EQUIVALENT
    assert canonicalize(result.right.jaxpr, search_budget=1).graph == result.right.graph
