"""可安全扩展的零填充及分块 VJP；边界、缺口与越界不能被掩盖。"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp
from jax import lax

from jax_reorder import Status, compare_functions
from jax_reorder.algebra_symbolic import SymbolicBuilder
from jax_reorder.algebra_transport import freeze


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
def test_padding_preserves_support_fill_and_partial_conditions(kind):
    def padded(x):
        if kind == "out_of_bounds":
            return lax.pad(x, jnp.float32(0), ((1, 1, 0),))
        if kind == "nonzero":
            return lax.pad(x[:2], jnp.float32(1), ((0, 3, 0),))
        if kind == "interior":
            return lax.pad(x[:2], jnp.float32(0), ((0, 2, 1),))
        return lax.pad(jnp.log(x[:2]), jnp.float32(0), ((0, 3, 0),))

    graph = jax.make_jaxpr(padded)(jax.ShapeDtypeStruct((5,), jnp.float32))
    output = SymbolicBuilder().graph(freeze(graph))[0]
    if kind == "partial":
        assert output.defined != sp.true
    else:
        x = sp.IndexedBase(sp.Symbol("x0", real=True), shape=(5,), real=True)
        expected = {
            "out_of_bounds": [0, x[0], x[1], x[2], x[3], x[4], 0],
            "nonzero": [x[0], x[1], 1, 1, 1],
            "interior": [x[0], 0, x[1], 0, 0],
        }[kind]
        for i, target in enumerate(expected):
            expression = output.read((sp.Integer(i),))[0]
            assert expression == target
