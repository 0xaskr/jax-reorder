"""AI 手写的内部调用槽位改写；外部函数接口与普通算子端口保持不变。"""

import jax
from jax import lax


def original(x, y):
    @jax.jit
    def inner(a, b):
        sine = lax.sin(a)
        cosine = lax.cos(b)
        return sine, cosine

    a, b = inner(x, y)
    return a - b, a + b


def reordered(x, y):
    @jax.jit
    def inner(b, a):
        cosine = lax.cos(b)
        sine = lax.sin(a)
        return cosine, sine

    b, a = inner(y, x)
    return a - b, a + b


def output_reordered(x, y):
    @jax.jit
    def inner(a, b):
        cosine = lax.cos(b)
        sine = lax.sin(a)
        return cosine, sine

    b, a = inner(x, y)
    return a - b, a + b


def wrong_input(x, y):
    @jax.jit
    def inner(b, a):
        cosine = lax.cos(b)
        sine = lax.sin(a)
        return cosine, sine

    b, a = inner(x, y)
    return a - b, a + b


def wrong_output(x, y):
    @jax.jit
    def inner(b, a):
        cosine = lax.cos(b)
        sine = lax.sin(a)
        return cosine, sine

    a, b = inner(y, x)
    return a - b, a + b
