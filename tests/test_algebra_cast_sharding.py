"""求导生成的 cast 描述处理：默认空网格可归一化，非默认 sharding 必须拒绝。

覆盖：
- 正例：VJP 里由广播与矩阵乘两条不同路径生成的 float16/bfloat16 cast，其默认依赖 rank
  的 sharding 被归一化为标量描述，两条程序比较为 equivalent。
- 边界：只有与默认 ``ShapedArray.sharding`` 完全一致（含 memory_kind）的 cast 描述才
  归一化；device/pinned_host/unpinned_host 等非默认 memory kind 抛 UnsupportedJaxprError。

前置条件：CPU 平台；不涉及多设备，也不证明其他 primitive 的 sharding 参数处理。
"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax._src import core

from jax_reorder import Status, UnsupportedJaxprError, compare_functions
from jax_reorder.algebra_transport import freeze


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_vjp_casts_with_default_rank_dependent_sharding_are_scalar(dtype):
    """broadcast 与 matrix 两段代码等价，只是 cast 来自不同 VJP 路径。"""

    def broadcast(x, w):
        result = jnp.sum(w[..., None] * x[:, None, :].astype(jnp.float32), axis=0)
        return result.astype(dtype)

    def matrix(x, w):
        return (w.T @ x.astype(jnp.float32)).astype(dtype)

    args = jax.ShapeDtypeStruct((4, 3), dtype), jax.ShapeDtypeStruct((4, 2), jnp.float32)
    # 只请求 vjp 模式，差异恰好出现在求导生成的 cast 参数上。
    result = compare_functions(jax.jit(broadcast), jax.jit(matrix), *args, modes=("vjp",))
    assert result.status == Status.EQUIVALENT, result.modes[0].issues
    mode = result.modes[0]
    # algebra backend 记录非空，说明结构不同后由代数路径接受；结构路径时该字段为 None。
    assert mode.algebra.backend_result is not None


@pytest.mark.parametrize("memory_kind", [None, "device", "pinned_host", "unpinned_host"])
def test_cast_only_normalizes_the_complete_default_sharding(memory_kind):
    """只有完整等于默认 sharding（memory_kind 也相同）的 cast 参数才归一化。"""
    graph = jax.make_jaxpr(lambda x: lax.convert_element_type(x, jnp.float32))(
        jax.ShapeDtypeStruct((4, 3), jnp.bfloat16)
    )
    eq = graph.eqns[0]
    default = core.ShapedArray((4, 3), jnp.float32).sharding
    # None 表示直接使用默认描述；其余为在默认描述上替换 memory_kind 的非默认变体。
    sharding = default if memory_kind is None else default.with_memory_kind(memory_kind)
    params = {**eq.params, "sharding": sharding}
    changed = graph.jaxpr.replace(eqns=[eq.replace(params=params)])
    if memory_kind is None:
        # 默认描述在冻结后应归一化为 None，不进入后续比较。
        frozen = freeze(changed)
        assert frozen.eqns[0].params["sharding"] is None
    else:
        with pytest.raises(UnsupportedJaxprError, match="non-default cast sharding"):
            freeze(changed)
