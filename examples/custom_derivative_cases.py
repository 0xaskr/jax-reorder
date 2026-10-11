"""AI 直接编写的求导正反例；残差和候选源码全部固定，不生成程序改写。"""

from functools import partial

import jax
from jax import lax
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero


@jax.custom_vjp
def vjp_original(x, y):
    a = lax.sin(x)
    b = lax.cos(y)
    return a * b, a + b


def vjp_fwd(x, y):
    a = lax.sin(x)
    da = lax.cos(x)
    b = lax.cos(y)
    db = -lax.sin(y)
    return (a * b, a + b), (a, b, da, db)


def vjp_bwd(residuals, cotangents):
    a, b, da, db = residuals
    c, d = cotangents
    dx = (c * b + d) * da
    dy = (c * a + d) * db
    return dx, dy


vjp_original.defvjp(vjp_fwd, vjp_bwd)


@jax.custom_vjp
def vjp_reordered(x, y):
    b = lax.cos(y)
    a = lax.sin(x)
    return a * b, a + b


def reordered_fwd(x, y):
    b = lax.cos(y)
    db = -lax.sin(y)
    a = lax.sin(x)
    da = lax.cos(x)
    return (a * b, a + b), (db, b, da, a)


def reordered_bwd(residuals, cotangents):
    db, b, da, a = residuals
    c, d = cotangents
    dy = (c * a + d) * db
    dx = (c * b + d) * da
    return dx, dy


vjp_reordered.defvjp(reordered_fwd, reordered_bwd)


@jax.custom_vjp
def vjp_wrong_residual(x, y):
    a = lax.sin(x)
    b = lax.cos(y)
    return a * b, a + b


def wrong_bwd(residuals, cotangents):
    b, a, da, db = residuals
    c, d = cotangents
    return (c * b + d) * da, (c * a + d) * db


vjp_wrong_residual.defvjp(vjp_fwd, wrong_bwd)


@jax.custom_vjp
def vjp_wrong_fwd(x, y):
    a = lax.sin(x)
    b = lax.cos(y)
    return a * b, a + b


def wrong_fwd(x, y):
    a = lax.sin(x)
    da = lax.cos(x)
    b = lax.cos(y)
    db = -lax.sin(y)
    return (a * b, a - b), (a, b, da, db)


vjp_wrong_fwd.defvjp(wrong_fwd, vjp_bwd)


@jax.custom_vjp
def forwarded_residual(x, y):
    return x * y


def forwarded_fwd(x, y):
    return x * y, (x, y)


def forwarded_bwd(residuals, c):
    x, y = residuals
    return c * y, c * x


forwarded_residual.defvjp(forwarded_fwd, forwarded_bwd)


@jax.custom_vjp
def forwarded_reordered(x, y):
    return x * y


def forwarded_reordered_fwd(x, y):
    return x * y, (y, x)


def forwarded_reordered_bwd(residuals, c):
    y, x = residuals
    dy = c * x
    dx = c * y
    return dx, dy


forwarded_reordered.defvjp(forwarded_reordered_fwd, forwarded_reordered_bwd)


@jax.custom_jvp
def jvp_original(x, y):
    a = x * x
    b = y * y
    return a - b


@jvp_original.defjvp
def jvp_rule(primals, tangents):
    x, y = primals
    tx, ty = tangents
    a = x * x
    b = y * y
    dx = (2 * x) * tx
    dy = (2 * y) * ty
    return a - b, dx - dy


@jax.custom_jvp
def jvp_reordered(x, y):
    b = y * y
    a = x * x
    return a - b


@jvp_reordered.defjvp
def reordered_jvp_rule(primals, tangents):
    x, y = primals
    tx, ty = tangents
    dy = (2 * y) * ty
    b = y * y
    dx = (2 * x) * tx
    a = x * x
    return a - b, dx - dy


@jax.custom_jvp
def jvp_wrong_gradient(x, y):
    return x * x - y * y


@jvp_wrong_gradient.defjvp
def wrong_jvp_rule(primals, tangents):
    x, y = primals
    tx, ty = tangents
    return x * x - y * y, (3 * x) * tx - (2 * y) * ty


@jax.custom_jvp
def symbolic_jvp(x, y):
    return x * y


@partial(symbolic_jvp.defjvp, symbolic_zeros=True)
def symbolic_jvp_rule(primals, tangents):
    x, y = primals
    tx, ty = tangents
    if isinstance(tx, SymbolicZero):
        tangent = x * ty
    elif isinstance(ty, SymbolicZero):
        tangent = tx * y
    else:
        tangent = tx * y + x * ty
    return x * y, tangent


@jax.custom_vjp
def symbolic_vjp(x, y):
    return x * y


def symbolic_fwd(x: CustomVJPPrimal, y: CustomVJPPrimal):
    # 记录 perturbed 的各个模式；本例公式在这些模式下均相同。
    return x.value * y.value, (x.value, y.value)


symbolic_vjp.defvjp(symbolic_fwd, forwarded_bwd, symbolic_zeros=True)
