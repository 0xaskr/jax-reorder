"""可安全扩展的零填充及分块 VJP；边界、缺口与越界不能被掩盖。

覆盖：
- 正例：分块归约（cut=0/1/2/5）的 VJP 在 float32/bfloat16 下与整体归约等价，转置允许
  静态零扩展；padding 的位置、填充值与非零填充都能由符号读取正确重建（越界、非零、内部
  空缺三种形态）。
- 反例：分块缺口或重叠（cut=1/3，前两块固定为 [0:2]）不是完整梯度，不能判 equivalent。
- 部分函数：``log`` 之后的输出保留 ``defined != True``，填充不能无条件设置有限性。

前置条件：CPU 平台与 SymPy 1.14；VJP 用例只请求 vjp 模式。
固定 4 个 cut 与 4 类 padding 不覆盖任意分块/填充配置。
"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp
from jax import lax

from jax_reorder import Status, compare_functions
from jax_reorder.algebra.adapter.transport import freeze
from jax_reorder.algebra.symbolic import SymbolicBuilder


@pytest.mark.parametrize("cut", [0, 1, 2, 5])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_vjp_of_partitioned_reduction_reassembles_zero_extended_parts(cut, dtype):
    """分块归约的 VJP 等于整体归约的 VJP；cut 边界包含空块与整块两种极端。"""

    def whole(x, w):
        return jnp.sum(x.astype(jnp.float32) * w.astype(jnp.float32))

    # split 把归约拆成两段，中间没有缺口也没有重叠。
    def split(x, w):
        x, w = x.astype(jnp.float32), w.astype(jnp.float32)
        return jnp.sum(x[:cut] * w[:cut]) + jnp.sum(x[cut:] * w[cut:])

    a = jax.ShapeDtypeStruct((5,), dtype)
    result = compare_functions(jax.jit(whole), jax.jit(split), a, a, modes=("vjp",))
    assert result.status == Status.EQUIVALENT, result.modes[0].issues


@pytest.mark.parametrize("cut", [1, 3])
def test_wrong_partition_gap_or_overlap_is_not_a_complete_gradient(cut):
    """缺口（cut=3 漏掉元素）或重叠（cut=1 重复元素）都会改变梯度。"""

    def whole(x):
        return jnp.sum(x * x)

    # wrong 的第一段固定为 [0:2]：cut=1 时元素 1 重复，cut=3 时元素 2 缺失。
    def wrong(x):
        return jnp.sum(x[:2] * x[:2]) + jnp.sum(x[cut:] * x[cut:])

    result = compare_functions(whole, wrong, jnp.ones(5), modes=("vjp",))
    assert result.status != Status.EQUIVALENT


@pytest.mark.parametrize("kind", ["out_of_bounds", "nonzero", "interior", "partial"])
def test_padding_preserves_support_fill_and_partial_conditions(kind):
    """padding 的符号读取必须保留支持域位置、填充值与部分性条件。"""

    def padded(x):
        if kind == "out_of_bounds":
            # 两侧各补一个 0，输出长度 7。
            return lax.pad(x, jnp.float32(0), ((1, 1, 0),))
        if kind == "nonzero":
            # 非零填充值必须原样进入表达式，而不是被当作默认 0。
            return lax.pad(x[:2], jnp.float32(1), ((0, 3, 0),))
        if kind == "interior":
            # interior padding 在元素之间插入空缺，读取下标要跳过这些位置。
            return lax.pad(x[:2], jnp.float32(0), ((0, 2, 1),))
        # log 在 0 处无定义，部分性必须传播到输出。
        return lax.pad(jnp.log(x[:2]), jnp.float32(0), ((0, 3, 0),))

    graph = jax.make_jaxpr(padded)(jax.ShapeDtypeStruct((5,), jnp.float32))
    # 直接对冻结图构造逐输出符号表示，检查每个下标的读取表达式。
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
