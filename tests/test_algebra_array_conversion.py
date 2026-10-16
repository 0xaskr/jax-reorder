"""ArrayExpr adaptation must preserve coordinate maps and finite summation domains."""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jax_reorder import Status, compare_functions
from jax_reorder.algebra_budget import Budget
from jax_reorder.algebra_trusted import equal


@pytest.mark.parametrize(
    "left,right",
    [
        (lambda x: x[:3], lambda x: x[1:4]),
        (lambda x: x[:3], lambda x: x[::2]),
        (lambda x: x, lambda x: x[::-1]),
        (lambda x: jnp.sum(x[:3]), lambda x: jnp.sum(x)),
        (lambda x: jnp.sum(x), lambda x: jnp.sum(jnp.arange(6, dtype=jnp.float32) * x)),
        (
            lambda x: jnp.broadcast_to(x[:, None], (6, 6)),
            lambda x: jnp.broadcast_to(x[None, :], (6, 6)),
        ),
    ],
    ids=["offset", "stride", "reverse", "partial_sum", "coordinate_weight", "broadcast_axis"],
)
def test_changed_coordinate_maps_are_inequivalent(left, right):
    spec = jax.ShapeDtypeStruct((6,), jnp.float32)
    mode = compare_functions(left, right, spec).modes[0]
    assert mode.status == Status.INEQUIVALENT, mode.issues
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate")
        for q in mode.algebra.backend_result.queries
    )


@pytest.mark.parametrize("view", [lambda x: x[1:4], lambda x: x[::2], lambda x: x[::-1]])
def test_same_coordinate_maps_still_allow_algebraic_rewrites(view):
    spec = jax.ShapeDtypeStruct((6,), jnp.float32)
    mode = compare_functions(lambda x: view(x) + view(x), lambda x: 2 * view(x), spec).modes[0]
    assert mode.status == Status.EQUIVALENT, mode.issues


def test_large_matrix_association_still_uses_array_expressions():
    i, j, k, column = sp.symbols("i j k column", integer=True, nonnegative=True)
    a, b, c = (sp.IndexedBase(name, shape=(512, 512), real=True) for name in "abc")
    left = sp.Sum(sp.Sum(a[i, j] * b[j, k], (j, 0, 511)) * c[k, column], (k, 0, 511))
    right = sp.Sum(a[i, j] * sp.Sum(b[j, k] * c[k, column], (k, 0, 511)), (j, 0, 511))
    methods = []
    assert equal(left, right, Budget(100), indices=(i, column), methods=methods) is True
    assert methods == ["ArrayExpr.doit"]
