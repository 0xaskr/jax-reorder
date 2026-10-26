"""jit 调用私有输入/结果槽位的固定正反例；外部函数接口保持不变。

运行：uv run pytest tests/test_call_bindings.py -q
本模块没有独立入口，函数由测试导入并追踪成 jaxpr。正例只交换内层 jit 的
实参/形参，或同步调整返回值槽位；调用双方的外部输入顺序、普通算子端口和
最终输出保持不变。负例只改一侧绑定，内层计算结果因此与实际槽位错配。
"""

import jax
from jax import lax


def original(x, y):
    """基准：先 sin 后 cos，内层按 (a, b) 接收并返回 (sin, cos)，外层 a - b、a + b。"""

    @jax.jit
    def inner(a, b):
        sine = lax.sin(a)
        cosine = lax.cos(b)
        return sine, cosine

    a, b = inner(x, y)
    return a - b, a + b


def reordered(x, y):
    """正例：内层形参改为 (b, a)、实参按 (y, x) 传入，返回槽位同步变成 (cos, sin)。"""

    @jax.jit
    def inner(b, a):
        cosine = lax.cos(b)
        sine = lax.sin(a)
        return cosine, sine

    b, a = inner(y, x)
    return a - b, a + b


def output_reordered(x, y):
    """正例：调用实参不变，只交换内层返回槽位，调用处按新顺序接收并赋回 a/b。"""

    @jax.jit
    def inner(a, b):
        cosine = lax.cos(b)
        sine = lax.sin(a)
        return cosine, sine

    b, a = inner(x, y)
    return a - b, a + b


def wrong_input(x, y):
    """负例：内层形参声明为 (b, a)，调用处却仍按 (x, y) 传入，sin/cos 实际输入互换。"""

    @jax.jit
    def inner(b, a):
        cosine = lax.cos(b)
        sine = lax.sin(a)
        return cosine, sine

    b, a = inner(x, y)
    return a - b, a + b


def wrong_output(x, y):
    """负例：实参按置换传入，调用处仍按 (a, b) 接收，返回槽位与实际值错配。"""

    @jax.jit
    def inner(b, a):
        cosine = lax.cos(b)
        sine = lax.sin(a)
        return cosine, sine

    a, b = inner(y, x)
    return a - b, a + b
