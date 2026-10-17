"""纯 JIT 组合及 VJP 私有残差变化；非默认调用和部分函数仍保留边界。

覆盖：
- 正例：广播乘积归约与转置矩阵乘在 forward/jvp/vjp 下等价，JIT 的私有残差接口差异由
  代数路径吸收；JIT 捕获的实际常量（weights）保持绑定；嵌套 while 循环的 barrier 与
  x + body(x, 3) 的换序等价。
- 反例：捕获常量改成 weights+1、循环长度 4 改为与 3 不同时判 inequivalent。
- 拒绝与未决：donation/inline/keep_unused 等非默认 JIT 元数据抛 UnsupportedJaxprError；
  未使用的部分调用实参或输出（log 在 0 处无定义）抛 condition_undecided，不能消失。

前置条件：CPU 平台与 SymPy 1.14；不涉及多设备或上游快照。
固定组合不覆盖任意 JIT 嵌套与全部执行属性。
"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax

from jax_reorder import MathematicalUndecidedError, Status, UnsupportedJaxprError, compare_functions


def broadcast_sum(x, w):
    """参考实现：广播乘积后沿行求和；给 JIT 使用。"""
    return jnp.sum(w[..., None] * x[:, None, :], axis=0)


def matrix_sum(x, w):
    """等价实现：等价于 (w.T @ x)，只是索引表达式不同。"""
    return w.T @ x


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_default_jit_composes_different_private_residual_interfaces(mode):
    """默认执行属性的 JIT 允许组合；两侧残差接口不同时仍走代数路径。"""
    args = jax.ShapeDtypeStruct((4, 3), jnp.float32), jax.ShapeDtypeStruct((4, 2), jnp.float32)
    result = compare_functions(jax.jit(broadcast_sum), jax.jit(matrix_sum), *args, modes=(mode,))
    assert result.status == Status.EQUIVALENT, result.modes[0].issues
    for report in result.modes:
        # correspondence 为 None 表示结构路径没通过，结论来自代数查询。
        assert report.algebra and report.algebra.backend_result
        assert report.algebra.correspondence is None


def test_default_jit_keeps_constants_and_actual_bindings():
    """JIT 捕获的常量必须保持精确值；改成 weights+1 就不再等价。"""
    weights = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    left = jax.jit(lambda x: broadcast_sum(x, weights))

    def right(x):
        return matrix_sum(x, weights)

    assert compare_functions(left, right, x).status == Status.EQUIVALENT

    def wrong(x):
        return matrix_sum(x, weights + 1)

    assert compare_functions(left, wrong, x).status != Status.EQUIVALENT


@pytest.mark.parametrize("metadata", ["donation", "inline", "keep_unused"])
def test_jit_composition_keeps_non_default_execution_metadata(metadata):
    """非默认 JIT 执行属性会改变语义边界，必须显式拒绝而不是当作默认纯函数。"""
    settings = {
        "donation": {"donate_argnums": (0,)},
        "inline": {"inline": True},
        "keep_unused": {"keep_unused": True},
    }
    left = jax.jit(lambda x: x + x, **settings[metadata])

    # right 用等价的 2*x，差异只在 JIT 调用元数据上。
    def right(x):
        return 2 * x

    with pytest.raises(UnsupportedJaxprError, match="non-default JIT"):
        compare_functions(left, right, jnp.ones(3))


def test_unused_partial_call_argument_cannot_disappear():
    """部分函数的实参（log(x)）即使 JIT 内部未使用也不能被静默删除。"""
    wrapped = jax.jit(lambda x, unused: x + x)

    def left(x):
        return wrapped(x, jnp.log(x))

    def right(x):
        return 2 * x

    with pytest.raises(MathematicalUndecidedError, match="condition_undecided"):
        compare_functions(left, right, jnp.ones(3))


def test_unused_partial_call_output_cannot_disappear():
    """部分函数的输出被丢弃时同样保留部分性条件，不能当作纯无用节点删除。"""
    wrapped = jax.jit(jnp.log)

    def left(x):
        wrapped(x)
        return x + x

    with pytest.raises(MathematicalUndecidedError, match="condition_undecided"):
        compare_functions(left, lambda x: 2 * x, jnp.ones(3))


def test_jit_composition_retains_nested_loop_barriers():
    """外层 JIT 内部嵌套 while 的 barrier 保留；循环长度改变则判不等价。"""

    def body(x, length):
        return lax.while_loop(lambda c: c[0] < length, lambda c: (c[0] + 1, c[1] + x), (0, x))[1]

    x = jnp.ones(2)
    left = jax.jit(lambda x: body(x, 3) + x)

    # right 只交换独立加法项的顺序，while 循环体不变。
    def right(x):
        return x + body(x, 3)

    assert compare_functions(left, right, x).status == Status.EQUIVALENT

    # wrong 把循环长度从 3 改成 4，循环语义改变。
    def wrong(x):
        return x + body(x, 4)

    assert compare_functions(left, wrong, x).status != Status.EQUIVALENT
