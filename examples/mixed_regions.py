"""混合控制流的固定正反例：cond→scan→while→cond、scan→cond→scan、jit→cond→scan。

运行：uv run pytest tests/test_mixed_regions.py -q
本模块没有独立入口，函数由测试导入。候选同步调整 cond 实参与分支形参、
scan 的 carry/xs 槽位以及 jit 私有端口，每一层的实际计算保持不变。测试用
追踪出的 Primitive 顺序确认两图确实不同，再分别执行原图与重排图核对数值。
"""

import jax
import jax.numpy as jnp
from jax import lax

from examples.feedback_bindings import scan_original, scan_reordered


def cond_scan_while_cond(flag, x, y, u, v):
    """基准：cond 真分支里 scan 每步再跑 while，while 体内再嵌套一层 cond。"""

    def selected(seed):
        def scan_body(carry, item):
            def condition(state):
                return state[0] < 2

            def body(state):
                i, value = state
                a = lax.sin(seed)
                b = lax.cos(y)
                new = lax.cond(value > 0, lambda: (value + a) - b, lambda: (value - a) + b)
                return i + 1, new

            _, new = lax.while_loop(condition, body, (jnp.int32(0), carry + item))
            return new, new

        return lax.scan(scan_body, seed, u)[0]

    return lax.cond(flag, selected, lambda seed: -seed, x)


def cond_scan_while_cond_reordered(flag, x, y, u, v):
    """正例：交换 scan carry 与 while state 的槽位，闭包捕获顺序随之改变。"""

    def selected(seed):
        def scan_body(carry, item):
            def condition(state):
                return state[1] < 2

            def body(state):
                value, i = state
                b = lax.cos(y)
                a = lax.sin(seed)
                new = lax.cond(value > 0, lambda: (value + a) - b, lambda: (value - a) + b)
                return new, i + 1

            new, _ = lax.while_loop(condition, body, (carry + item, jnp.int32(0)))
            return new, new

        return lax.scan(scan_body, seed, u)[0]

    return lax.cond(flag, selected, lambda seed: -seed, x)


def scan_cond_scan(flag, x, y, u, v):
    """基准：外层 scan 的每一步把 scan_original 同时当作 cond 的真假分支。"""

    def body(state, item):
        a, b = state
        s, t = item
        result = lax.cond(flag, scan_original, scan_original, a, b, s.reshape(1), t.reshape(1))
        return result

    return lax.scan(body, (x, y), (u, v))


def scan_cond_scan_reordered(flag, x, y, u, v):
    """正例：同步置换外层 scan 的 carry/xs 与 cond 的实参槽位。"""

    def body(state, item):
        b, a = state
        t, s = item
        (p, q), outputs = lax.cond(
            flag, scan_reordered, scan_reordered, a, b, s.reshape(1), t.reshape(1)
        )
        return (q, p), outputs

    (b, a), outputs = lax.scan(body, (y, x), (v, u))
    return (a, b), outputs


def jit_cond_scan(flag, x, y, u, v):
    """基准：jit 包裹 cond→scan，外部参数顺序固定为 (flag, x, y, u, v)。"""

    @jax.jit
    def call(p, a, b, s, t):
        return lax.cond(p, scan_original, scan_original, a, b, s, t)

    return call(flag, x, y, u, v)


def jit_cond_scan_reordered(flag, x, y, u, v):
    """正例：只重排 jit 内部形参，调用处按新槽位传实参，外部接口和计算不变。"""

    @jax.jit
    def call(b, t, p, s, a):
        return lax.cond(p, scan_reordered, scan_reordered, a, b, s, t)

    return call(y, v, flag, u, x)
