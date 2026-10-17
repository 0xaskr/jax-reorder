"""循环反馈与 scan 结果槽位的固定正反例；全部候选由 AI 手写。

运行：uv run pytest tests/test_feedback_bindings.py -q
本模块没有独立入口，函数由测试导入。while 的 carry 初始值、condition/body
形参、body 返回和父节点结果必须使用同一个置换；scan 还要保留 carry 与
xs/ys 的角色。正例只同步调整内部槽位，外部参数和返回顺序固定。负例只改
一侧，使反馈边或结果端口实际错接。scan_reordered 另外改写了 ys 槽位：前向
等价，但 VJP 按返回槽位累加余切，结构模式会保留这一差异。
"""

import jax.numpy as jnp
from jax import lax


def while_original(x, y, count):
    """基准：state 为 (i, a, b)，每轮用 sin(a)、cos(b) 更新后按原槽位返回。"""

    def condition(state):
        i, a, b = state
        return i < count

    def body(state):
        i, a, b = state
        p = lax.sin(a)
        q = lax.cos(b)
        return i + 1, p, q

    _, a, b = lax.while_loop(condition, body, (jnp.int32(0), x, y))
    return a - b, a, b


def while_reordered(x, y, count):
    """正例：carry 顺序改为 (b, i, a)，初值与 condition/body 的形参同步调整。"""

    def condition(state):
        b, i, a = state
        return i < count

    def body(state):
        b, i, a = state
        q = lax.cos(b)
        p = lax.sin(a)
        return q, i + 1, p

    b, _, a = lax.while_loop(condition, body, (y, jnp.int32(0), x))
    return a - b, a, b


def while_reordered_again(x, y, count):
    """正例：carry 再改为 (i, b, a)，反馈位置与新槽位一一对应，循环语义不变。"""

    def condition(state):
        i, b, a = state
        return i < count

    def body(state):
        i, b, a = state
        p = lax.sin(a)
        q = lax.cos(b)
        return i + 1, q, p

    _, b, a = lax.while_loop(condition, body, (jnp.int32(0), y, x))
    return a - b, a, b


def while_wrong_initial(x, y, count):
    """负例：外部实参未交换就调用置换版，carry 初值接到错误槽位。"""
    return while_reordered(y, x, count)


def while_wrong_feedback(x, y, count):
    """负例：body 把 p、q 写回对方槽位，下一步循环读到错误的反馈值。"""

    def condition(state):
        b, i, a = state
        return i < count

    def body(state):
        b, i, a = state
        q = lax.cos(b)
        p = lax.sin(a)
        return p, i + 1, q

    b, _, a = lax.while_loop(condition, body, (y, jnp.int32(0), x))
    return a - b, a, b


def scan_original(x, y, u, v):
    """基准：carry 为 (a, b)，xs 为 (s, t)，每步输出 (p - q, q + p)。"""

    def body(state, item):
        a, b = state
        s, t = item
        p = lax.sin(a + s)
        q = lax.cos(b + t)
        return (p, q), (p - q, q + p)

    return lax.scan(body, (x, y), (u, v))


def scan_reordered(x, y, u, v):
    """正例：carry 改为 (b, a)、xs 改为 (t, s)，外层返回时换回原顺序。

    体内 ys 写成 (q + p, p - q)（基准为 (p - q, q + p)）：前向仍等价，但 VJP
    按返回槽位累加余切，切向量的 add_any 顺序与基准不同，结构模式能看到差异。
    """

    def body(state, item):
        b, a = state
        t, s = item
        q = lax.cos(b + t)
        p = lax.sin(a + s)
        return (q, p), (q + p, p - q)

    (b, a), (total, difference) = lax.scan(body, (y, x), (v, u))
    return (a, b), (difference, total)


def scan_reordered_again(x, y, u, v):
    """正例：只置换 carry/xs，ys 仍在体内按基准顺序计算，求导累加顺序不变。"""

    def body(state, item):
        b, a = state
        s, t = item
        q = lax.cos(b + t)
        p = lax.sin(a + s)
        return (q, p), (p - q, q + p)

    (b, a), outputs = lax.scan(body, (y, x), (u, v))
    return (a, b), outputs


def scan_wrong_initial(x, y, u, v):
    """负例：直接交换报给 scan 的外部实参，carry 与 xs 的初始槽位错位。"""
    return scan_reordered(y, x, u, v)


def scan_wrong_output(x, y, u, v):
    """负例：ys 第二个结果由 q + p 改成 q - p，与基准的 p - q 数值不同。"""

    def body(state, item):
        b, a = state
        t, s = item
        q = lax.cos(b + t)
        p = lax.sin(a + s)
        return (q, p), (q + p, q - p)

    (b, a), (total, difference) = lax.scan(body, (y, x), (v, u))
    return (a, b), (difference, total)
