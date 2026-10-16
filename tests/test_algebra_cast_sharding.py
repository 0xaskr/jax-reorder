"""求导生成的默认空网格 cast 描述不应阻断逐点索引映射。"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax._src import core

from jax_reorder import Status, compare_functions
from jax_reorder.algebra_ir import Budget
from jax_reorder.algebra_lower import Lowerer
from jax_reorder.algebra_scalar import lower_scalar


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_vjp_casts_with_default_rank_dependent_sharding_are_scalar(dtype):
    def broadcast(x, w):
        result = jnp.sum(w[..., None] * x[:, None, :].astype(jnp.float32), axis=0)
        return result.astype(dtype)

    def matrix(x, w):
        return (w.T @ x.astype(jnp.float32)).astype(dtype)

    args = jax.ShapeDtypeStruct((4, 3), dtype), jax.ShapeDtypeStruct((4, 2), jnp.float32)
    result = compare_functions(jax.jit(broadcast), jax.jit(matrix), *args, modes=("vjp",))
    assert result.status == Status.EQUIVALENT, result.modes[0].issues
    mode = result.modes[0]
    assert mode.algebra.backend_result is not None


@pytest.mark.parametrize("memory_kind", [None, "device", "pinned_host", "unpinned_host"])
def test_cast_only_normalizes_the_complete_default_sharding(memory_kind):
    graph = jax.make_jaxpr(lambda x: lax.convert_element_type(x, jnp.float32))(
        jax.ShapeDtypeStruct((4, 3), jnp.bfloat16)
    )
    eq = graph.eqns[0]
    default = core.ShapedArray((4, 3), jnp.float32).sharding
    sharding = default if memory_kind is None else default.with_memory_kind(memory_kind)
    params = {**eq.params, "sharding": sharding}
    lowerer = Lowerer(Budget(10000))
    value = lowerer._input(eq.invars[0].aval, ("argument", 0))
    converted = lower_scalar(lowerer, eq, (value,), params)
    assert (converted is not None) == (memory_kind is None)
