"""可安全扩展的零填充及分块 VJP；边界、缺口与越界不能被掩盖。"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax

from jax_reorder import Status, compare_functions
from jax_reorder.algebra_ir import Budget, Factor, Index
from jax_reorder.algebra_lower import Lowerer
from jax_reorder.algebra_padding import simplify_prefix_factors


@pytest.mark.parametrize("cut", [0, 1, 2, 5])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_vjp_of_partitioned_reduction_reassembles_zero_extended_parts(cut, dtype):
    def whole(x, w):
        return jnp.sum(x.astype(jnp.float32) * w.astype(jnp.float32))

    def split(x, w):
        x, w = x.astype(jnp.float32), w.astype(jnp.float32)
        return jnp.sum(x[:cut] * w[:cut]) + jnp.sum(x[cut:] * w[cut:])

    a = jax.ShapeDtypeStruct((5,), dtype)
    result = compare_functions(jax.jit(whole), jax.jit(split), a, a, modes=("vjp",))
    assert result.status == Status.EQUIVALENT, result.modes[0].issues


@pytest.mark.parametrize("cut", [1, 3])
def test_wrong_partition_gap_or_overlap_is_not_a_complete_gradient(cut):
    def whole(x):
        return jnp.sum(x * x)

    def wrong(x):
        return jnp.sum(x[:2] * x[:2]) + jnp.sum(x[cut:] * x[cut:])

    result = compare_functions(whole, wrong, jnp.ones(5), modes=("vjp",))
    assert result.status != Status.EQUIVALENT


@pytest.mark.parametrize("kind", ["out_of_bounds", "nonzero", "interior", "partial"])
def test_zero_extension_rejects_unproved_conditions(kind):
    def padded(x):
        if kind == "out_of_bounds":
            return lax.pad(x, jnp.float32(0), ((1, 1, 0),))
        if kind == "nonzero":
            return lax.pad(x[:2], jnp.float32(1), ((0, 3, 0),))
        if kind == "interior":
            return lax.pad(x[:2], jnp.float32(0), ((0, 2, 1),))
        return lax.pad(jnp.log(x[:2]), jnp.float32(0), ((0, 3, 0),))

    graph = jax.make_jaxpr(padded)(jax.ShapeDtypeStruct((5,), jnp.float32))
    lowerer = Lowerer(Budget(10000))
    lowerer.graph(graph)
    assert "safe_zero_extension" not in lowerer.rules


@pytest.mark.parametrize("stop", [1, 2, 3])
def test_prefix_singleton_rule_keeps_support_and_agrees_on_integer_points(stop):
    index = Index("free", 0)
    factors = (Factor(("index_prefix", stop), (index,)), Factor(("value",), (index,)))
    simplified = simplify_prefix_factors(factors)

    def evaluate(items, point):
        product = 1
        for factor in items:
            coordinate = point if factor.indices[0] == index else factor.indices[0].number
            value = (
                int(0 <= coordinate < factor.semantic[1])
                if factor.semantic[0] == "index_prefix"
                else 2 + coordinate
            )
            product *= value
        return product

    for point in range(-3, 6):
        assert evaluate(factors, point) == evaluate(simplified, point)
    assert simplified[0] == factors[0]
    if stop > 1:
        assert simplified == factors


def test_prefix_singleton_does_not_rewrite_partial_values():
    index = Index("free", 0)
    factors = (Factor(("index_prefix", 1), (index,)), Factor(("partial",), (index,), total=False))
    assert simplify_prefix_factors(factors) == factors
