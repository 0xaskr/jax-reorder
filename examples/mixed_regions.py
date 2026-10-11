"""AI 手写的混合控制流与调用场景；候选仅同步接口或重排独立计算。"""

import jax
import jax.numpy as jnp
from jax import lax

from examples.feedback_bindings import scan_original, scan_reordered


def cond_scan_while_cond(flag, x, y, u, v):
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
    def body(state, item):
        a, b = state
        s, t = item
        result = lax.cond(flag, scan_original, scan_original, a, b, s.reshape(1), t.reshape(1))
        return result

    return lax.scan(body, (x, y), (u, v))


def scan_cond_scan_reordered(flag, x, y, u, v):
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
    @jax.jit
    def call(p, a, b, s, t):
        return lax.cond(p, scan_original, scan_original, a, b, s, t)

    return call(flag, x, y, u, v)


def jit_cond_scan_reordered(flag, x, y, u, v):
    @jax.jit
    def call(b, t, p, s, a):
        return lax.cond(p, scan_reordered, scan_reordered, a, b, s, t)

    return call(y, v, flag, u, x)
