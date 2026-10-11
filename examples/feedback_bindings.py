"""AI 手写的循环反馈及扫描接口同步置换，外部参数和返回顺序固定。"""

import jax.numpy as jnp
from jax import lax


def while_original(x, y, count):
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
    return while_reordered(y, x, count)


def while_wrong_feedback(x, y, count):
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
    def body(state, item):
        a, b = state
        s, t = item
        p = lax.sin(a + s)
        q = lax.cos(b + t)
        return (p, q), (p - q, q + p)

    return lax.scan(body, (x, y), (u, v))


def scan_reordered(x, y, u, v):
    def body(state, item):
        b, a = state
        t, s = item
        q = lax.cos(b + t)
        p = lax.sin(a + s)
        return (q, p), (q + p, p - q)

    (b, a), (total, difference) = lax.scan(body, (y, x), (v, u))
    return (a, b), (difference, total)


def scan_reordered_again(x, y, u, v):
    def body(state, item):
        b, a = state
        s, t = item
        q = lax.cos(b + t)
        p = lax.sin(a + s)
        return (q, p), (p - q, q + p)

    (b, a), outputs = lax.scan(body, (y, x), (u, v))
    return (a, b), outputs


def scan_wrong_initial(x, y, u, v):
    return scan_reordered(y, x, u, v)


def scan_wrong_output(x, y, u, v):
    def body(state, item):
        b, a = state
        t, s = item
        q = lax.cos(b + t)
        p = lax.sin(a + s)
        return (q, p), (q + p, q - p)

    (b, a), (total, difference) = lax.scan(body, (y, x), (v, u))
    return (a, b), (difference, total)
