"""循环捕获槽位的固定正反例；全部候选由 AI 手写，不生成改写源码。"""

import jax.numpy as jnp
from jax import lax


def scan_original(x, scale, gate, steps=3):
    """两个独立计算捕获 x/scale/gate，循环 carry 与扫描输出均保留。"""

    def body(carry, item):
        a = jnp.sin(x * scale)
        b = jnp.cos(gate)
        return (carry + a) + b, item + x

    return lax.scan(body, jnp.zeros_like(x), jnp.arange(steps, dtype=x.dtype))


def scan_reordered(x, scale, gate, steps=3):
    """提前 cos 只改变捕获顺序，普通算子的输入位置和运算分组不变。"""

    def body(carry, item):
        b = jnp.cos(gate)
        a = jnp.sin(x * scale)
        return (carry + a) + b, item + x

    return lax.scan(body, jnp.zeros_like(x), jnp.arange(steps, dtype=x.dtype))


def scan_wrong_source(x, scale, gate, steps=3):
    """反例：sin 和 cos 所依赖的实际输入被交换。"""

    def body(carry, item):
        b = jnp.cos(x)
        a = jnp.sin(gate * scale)
        return (carry + a) + b, item + x

    return lax.scan(body, jnp.zeros_like(x), jnp.arange(steps, dtype=x.dtype))


def scan_wrong_scale(x, scale, gate, steps=3):
    """反例：类型相同不代表 scale/gate 可以互换。"""

    def body(carry, item):
        b = jnp.cos(scale)
        a = jnp.sin(x * gate)
        return (carry + a) + b, item + x

    return lax.scan(body, jnp.zeros_like(x), jnp.arange(steps, dtype=x.dtype))


def scan_producers_original(x, scale, gate, steps=3):
    """捕获值本身来自父图计算，子图还包含 cond。"""
    projected = x * scale
    shifted = gate + 0.25

    def body(carry, item):
        a = jnp.sin(projected)
        b = jnp.cos(shifted)
        value = lax.cond(item > 0, lambda: a - b, lambda: b - a)
        return carry + value, value + projected

    return lax.scan(body, jnp.zeros_like(x), jnp.arange(steps, dtype=x.dtype))


def scan_producers_reordered(x, scale, gate, steps=3):
    shifted = gate + 0.25
    projected = x * scale

    def body(carry, item):
        b = jnp.cos(shifted)
        a = jnp.sin(projected)
        value = lax.cond(item > 0, lambda: a - b, lambda: b - a)
        return carry + value, value + projected

    return lax.scan(body, jnp.zeros_like(x), jnp.arange(steps, dtype=x.dtype))


def while_original(x, scale, gate, steps=3):
    def body(state):
        index, carry = state
        a = jnp.sin(x * scale)
        b = jnp.cos(gate)
        return index + 1, (carry + a) + b

    return lax.while_loop(lambda state: state[0] < steps, body, (0, jnp.zeros_like(x)))


def while_reordered(x, scale, gate, steps=3):
    def body(state):
        index, carry = state
        b = jnp.cos(gate)
        a = jnp.sin(x * scale)
        return index + 1, (carry + a) + b

    return lax.while_loop(lambda state: state[0] < steps, body, (0, jnp.zeros_like(x)))


def while_wrong_source(x, scale, gate, steps=3):
    def body(state):
        index, carry = state
        b = jnp.cos(x)
        a = jnp.sin(gate * scale)
        return index + 1, (carry + a) + b

    return lax.while_loop(lambda state: state[0] < steps, body, (0, jnp.zeros_like(x)))


def while_wrong_scale(x, scale, gate, steps=3):
    def body(state):
        index, carry = state
        b = jnp.cos(scale)
        a = jnp.sin(x * gate)
        return index + 1, (carry + a) + b

    return lax.while_loop(lambda state: state[0] < steps, body, (0, jnp.zeros_like(x)))


def while_both_regions_original(x, scale, gate, limit, bound):
    """condition 和 body 分别捕获数据，两个区域都需要同步调整自己的接口。"""
    projected = x * scale
    shifted = gate + 0.25

    def condition(state):
        index, carry = state
        count_ok = index < limit
        value_ok = jnp.sum(carry) < bound
        return count_ok & value_ok

    def body(state):
        index, carry = state
        a = jnp.sin(projected)
        b = jnp.cos(shifted)
        return index + 1, (carry + a) + b

    return lax.while_loop(condition, body, (0, jnp.zeros_like(x)))


def while_both_regions_reordered(x, scale, gate, limit, bound):
    shifted = gate + 0.25
    projected = x * scale

    def condition(state):
        index, carry = state
        value_ok = jnp.sum(carry) < bound
        count_ok = index < limit
        return count_ok & value_ok

    def body(state):
        index, carry = state
        b = jnp.cos(shifted)
        a = jnp.sin(projected)
        return index + 1, (carry + a) + b

    return lax.while_loop(condition, body, (0, jnp.zeros_like(x)))
