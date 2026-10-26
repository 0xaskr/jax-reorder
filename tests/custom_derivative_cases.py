"""custom_jvp / custom_vjp 正反例：规则内容、残差槽位与 symbolic zeros。

运行：uv run pytest tests/test_custom_derivatives.py -q
本模块没有独立入口，函数由测试导入。所有规则均由 AI 手写并固定在源码中，
不生成改写程序。正例只调整残差的记录/解包顺序，或改变独立计算的书写顺序，
让前向规则与求导规则继续对应；负例改变梯度系数、前向输出或残差顺序。
"""

from functools import partial

import jax
from jax import lax
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero


@jax.custom_vjp
def vjp_original(x, y):
    """基准 custom_vjp：前向为 (a * b, a + b)，反传按残差 (a, b, da, db) 组合。"""
    a = lax.sin(x)
    b = lax.cos(y)
    return a * b, a + b


def vjp_fwd(x, y):
    """记录前向中间值与两个局部导数；残差顺序即反传解包顺序。"""
    a = lax.sin(x)
    da = lax.cos(x)
    b = lax.cos(y)
    db = -lax.sin(y)
    return (a * b, a + b), (a, b, da, db)


def vjp_bwd(residuals, cotangents):
    """两个输出各有一条余切：dx 用 c，dy 用 d，再乘各自的局部导数。"""
    a, b, da, db = residuals
    c, d = cotangents
    dx = (c * b + d) * da
    dy = (c * a + d) * db
    return dx, dy


vjp_original.defvjp(vjp_fwd, vjp_bwd)


@jax.custom_vjp
def vjp_reordered(x, y):
    """正例：前向只交换 a/b 的定义顺序，输出表达式与基准相同。"""
    b = lax.cos(y)
    a = lax.sin(x)
    return a * b, a + b


def reordered_fwd(x, y):
    """残差改为 (db, b, da, a)：顺序改变，但每个值仍对应同一数学量。"""
    b = lax.cos(y)
    db = -lax.sin(y)
    a = lax.sin(x)
    da = lax.cos(x)
    return (a * b, a + b), (db, b, da, a)


def reordered_bwd(residuals, cotangents):
    """按新残差顺序解包，dx/dy 的公式与基准完全一致。"""
    db, b, da, a = residuals
    c, d = cotangents
    dy = (c * a + d) * db
    dx = (c * b + d) * da
    return dx, dy


vjp_reordered.defvjp(reordered_fwd, reordered_bwd)


@jax.custom_vjp
def vjp_wrong_residual(x, y):
    """负例：前向与基准相同，反传却按 (b, a, da, db) 解包基准残差。"""
    a = lax.sin(x)
    b = lax.cos(y)
    return a * b, a + b


def wrong_bwd(residuals, cotangents):
    """错误反传：a/b 的位置对调，与局部导数 da/db 的配对随之错位。"""
    b, a, da, db = residuals
    c, d = cotangents
    return (c * b + d) * da, (c * a + d) * db


vjp_wrong_residual.defvjp(vjp_fwd, wrong_bwd)


@jax.custom_vjp
def vjp_wrong_fwd(x, y):
    """负例：前向第二个输出改成 a - b，函数本身已与基准不同。"""
    a = lax.sin(x)
    b = lax.cos(y)
    return a * b, a + b


def wrong_fwd(x, y):
    """错误前向规则：返回 (a * b, a - b)，反传仍使用基准公式处理余切。"""
    a = lax.sin(x)
    da = lax.cos(x)
    b = lax.cos(y)
    db = -lax.sin(y)
    return (a * b, a - b), (a, b, da, db)


vjp_wrong_fwd.defvjp(wrong_fwd, vjp_bwd)


@jax.custom_vjp
def forwarded_residual(x, y):
    """正例基准：前向只是 x * y，残差原样记录输入供反传使用。"""
    return x * y


def forwarded_fwd(x, y):
    """透传前向：残差就是两个输入本身，没有额外中间量。"""
    return x * y, (x, y)


def forwarded_bwd(residuals, c):
    """乘法的标准梯度：对 x 的余切是 c * y，对 y 的余切是 c * x。"""
    x, y = residuals
    return c * y, c * x


forwarded_residual.defvjp(forwarded_fwd, forwarded_bwd)


@jax.custom_vjp
def forwarded_reordered(x, y):
    """正例：前向不变，残差槽位改为 (y, x) 并同步反传解包。"""
    return x * y


def forwarded_reordered_fwd(x, y):
    """残差记录顺序改为 (y, x)；这属于接口约定，不改变前向计算。"""
    return x * y, (y, x)


def forwarded_reordered_bwd(residuals, c):
    """按 (y, x) 解包后仍返回 dx = c * y、dy = c * x。"""
    y, x = residuals
    dy = c * x
    dx = c * y
    return dx, dy


forwarded_reordered.defvjp(forwarded_reordered_fwd, forwarded_reordered_bwd)


@jax.custom_jvp
def jvp_original(x, y):
    """基准 custom_jvp：前向为 x² - y²，规则显式给出 2x·tx - 2y·ty。"""
    a = x * x
    b = y * y
    return a - b


@jvp_original.defjvp
def jvp_rule(primals, tangents):
    """重复前向表达式，再按链式法则返回切向量；两者必须使用同一公式。"""
    x, y = primals
    tx, ty = tangents
    a = x * x
    b = y * y
    dx = (2 * x) * tx
    dy = (2 * y) * ty
    return a - b, dx - dy


@jax.custom_jvp
def jvp_reordered(x, y):
    """正例：前向只交换 a/b 定义顺序，输出表达式仍为 x² - y²。"""
    b = y * y
    a = x * x
    return a - b


@jvp_reordered.defjvp
def reordered_jvp_rule(primals, tangents):
    """数学上与基准规则相同，仅改变了独立切向量的定义顺序。"""
    x, y = primals
    tx, ty = tangents
    dy = (2 * y) * ty
    b = y * y
    dx = (2 * x) * tx
    a = x * x
    return a - b, dx - dy


@jax.custom_jvp
def jvp_wrong_gradient(x, y):
    """负例：前向保持 x² - y²，JVP 规则把 2x 误写成 3x。"""
    return x * x - y * y


@jvp_wrong_gradient.defjvp
def wrong_jvp_rule(primals, tangents):
    """错误规则：切向量系数 3x 与真实导数 2x 不符，前向输出仍相同。"""
    x, y = primals
    tx, ty = tangents
    return x * x - y * y, (3 * x) * tx - (2 * y) * ty


@jax.custom_jvp
def symbolic_jvp(x, y):
    """symbolic_zeros 正例：结构为 x * y，某种输入的切向量可能被标记为零。"""
    return x * y


@partial(symbolic_jvp.defjvp, symbolic_zeros=True)
def symbolic_jvp_rule(primals, tangents):
    """按缺失切向量的位置分支，分别计算 x*ty、tx*y 或完整乘积和。"""
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
    """custom_vjp 的 symbolic_zeros 正例：前向值在规则中按 CustomVJPPrimal 读取。"""
    return x * y


def symbolic_fwd(x: CustomVJPPrimal, y: CustomVJPPrimal):
    """显式读取 perturbed 输入的值；本例残差公式在三种激活模式下相同。"""
    # 记录 perturbed 的各个模式；本例公式在这些模式下均相同。
    return x.value * y.value, (x.value, y.value)


# 反传沿用 forwarded_bwd：两个输入都激活时公式与单边激活一致。
symbolic_vjp.defvjp(symbolic_fwd, forwarded_bwd, symbolic_zeros=True)
