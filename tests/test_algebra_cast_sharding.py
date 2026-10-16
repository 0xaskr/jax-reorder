"""求导生成的默认空网格 cast 描述不应阻断逐点索引映射。"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax._src import core

from jax_reorder import Status, UnsupportedJaxprError, compare_functions
from jax_reorder.algebra_transport import freeze


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
    changed = graph.jaxpr.replace(eqns=[eq.replace(params=params)])
    if memory_kind is None:
        frozen = freeze(changed)
        assert frozen.eqns[0].params["sharding"] is None
    else:
        with pytest.raises(UnsupportedJaxprError, match="non-default cast sharding"):
            freeze(changed)
