"""共同纯函数的完整语义键、原始端口映射及符号定义。

覆盖：
- 正例：两侧共享同一个 jit 归一化函数时，``y*a + y*b`` 与 ``y*(a+b)`` 可由公共调用证据
  证明等价：命中 ``common_jit`` 规则、工作量受限，并记录符号定义。三种模式（forward/jvp/
  vjp）下公共调用的输出端口在区域重排后恢复到原始端口。
- 反例：把 ``a*u + v`` 写成 ``u + v*a`` 会改变端口映射，抛 MathematicalUndecidedError。
- 拒绝与未决：共享调用内部常量改变（offset 0.1→0.2）、``0*log(x)`` 的部分函数消项都拒绝，
  不能靠公共调用证据跳过语义。
- 前置条件：CPU 平台与 SymPy 1.14；归一化函数含 20 次迭代的固定循环，不依赖随机数。
- 有限用例不证明任意公共调用组合。
"""

import jax
import jax.numpy as jnp
import pytest

from jax_reorder import (
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    Status,
    compare_functions,
)


def normalization(iterations=20, offset=0.1):
    """构造两侧共享的 jit 归一化函数；iterations 与 offset 决定其语义。"""

    @jax.jit
    def run(x):
        y = jnp.exp(x)
        for _ in range(iterations):
            # 每次迭代沿最后一轴和倒数第二轴各归一化一次，形成固定的循环体。
            y = y / (jnp.sum(y, axis=-1, keepdims=True) + offset)
            y = y / (jnp.sum(y, axis=-2, keepdims=True) + offset)
        return y

    return run


def test_common_nonlinear_call_preserves_definition_and_limits_work():
    """共享调用只展开一次：两边都引用同一符号定义，工作量保持在预算内。"""
    normalize = normalization()

    def left(x, a, b):
        y = normalize(x)
        return y * a + y * b

    # right 用分配律合并为 y * (a + b)，依赖同一份 normalize 定义。
    def right(x, a, b):
        return normalize(x) * (a + b)

    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 12
    result = compare_functions(left, right, x, jnp.float32(2), jnp.float32(3))
    assert result.status == Status.EQUIVALENT, result.modes[0].issues
    mode = result.modes[0]
    # common_jit 表示公共调用被整体当作一个符号定义，而不是逐次展开。
    assert "common_jit" in mode.algebra.rules
    assert mode.algebra.work < 2000
    assert mode.algebra.backend_result.definitions


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_common_call_original_ports_are_restored_after_region_reordering(mode):
    """区域内部重排后，公共调用的输出端口必须映射回原始 (u, v) 端口。"""

    @jax.jit
    def pair(x, y):
        return jnp.sin(x + 2 * y), jnp.cos(x - 3 * y)

    def left(x, y, a):
        u, v = pair(x, y)
        return a * u + v

    # right 只交换独立项顺序（v + u*a），u/v 的端口语义不变。
    def right(x, y, a):
        u, v = pair(x, y)
        return v + u * a

    # wrong 把 u 与 v 的角色对调，端口映射不再成立。
    def wrong(x, y, a):
        u, v = pair(x, y)
        return u + v * a

    args = (jnp.arange(3, dtype=jnp.float32), jnp.ones(3), jnp.float32(2))
    assert compare_functions(left, right, *args, modes=(mode,)).status == Status.EQUIVALENT
    with pytest.raises(MathematicalUndecidedError):
        compare_functions(left, wrong, *args, modes=(mode,))


def test_common_call_cannot_hide_changed_constant_or_missing_partial_evaluation():
    """共享调用的常量改变或部分函数消项都必须拒绝。"""
    first, changed = normalization(2, 0.1), normalization(2, 0.2)

    def left(x):
        return first(x) + first(x)

    # wrong 看似把重复调用合并为 2*changed(x)，但 changed 的 offset 常量不同。
    def wrong(x):
        return 2 * changed(x)

    with pytest.raises((MathematicalUndecidedError, ComparisonTimeoutError)):
        compare_functions(left, wrong, jnp.ones((2, 3)))
    log = jax.jit(jnp.log)

    # log 在 0 处无定义，0*log(x) 不能无条件消为 0。
    def partial(x):
        return 0 * log(x)

    with pytest.raises(MathematicalUndecidedError):
        compare_functions(partial, jnp.zeros_like, jnp.ones(3))
