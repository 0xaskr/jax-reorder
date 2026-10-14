"""AI 手写的完整 hc_post 候选；不从 base 的 AST 或 jaxpr 自动生成。

原始公式见固定上游 reference.py:113，Apache-2.0 许可证随快照保留。
矩阵乘法与分块只在理想实数解释下等价，实际舍入须另外测量。
"""

import jax
import jax.numpy as jnp


@jax.jit
def post_matmul(x, residual, post, comb):
    """把流间广播乘积归约变为批量矩阵乘法，避免显式 hc²D 中间张量。"""
    x32, residual32 = x.astype(jnp.float32), residual.astype(jnp.float32)
    post32, comb32 = post.astype(jnp.float32), comb.astype(jnp.float32)
    mixed = jnp.matmul(jnp.swapaxes(comb32, -1, -2), residual32)
    return (post32[..., None] * x32[..., None, :] + mixed).astype(x.dtype)


@jax.jit
def post_blocked(x, residual, post, comb):
    """按来源流分两块归约，缩小每块广播乘积并重新结合两段求和。"""
    cut = residual.shape[-2] // 2
    x32, residual32 = x.astype(jnp.float32), residual.astype(jnp.float32)
    post32, comb32 = post.astype(jnp.float32), comb.astype(jnp.float32)
    first = jnp.sum(comb32[..., :cut, :, None] * residual32[..., :cut, None, :], axis=-3)
    second = jnp.sum(comb32[..., cut:, :, None] * residual32[..., cut:, None, :], axis=-3)
    return (first + (post32[..., None] * x32[..., None, :] + second)).astype(x.dtype)


@jax.jit
def wrong_transpose(x, residual, post, comb):
    """反例：交换 comb 的来源流和目标流，形状相同但连接错误。"""
    mixed = jnp.matmul(comb.astype(jnp.float32), residual.astype(jnp.float32))
    return (
        post[..., None].astype(jnp.float32) * x[..., None, :].astype(jnp.float32) + mixed
    ).astype(x.dtype)


@jax.jit
def wrong_scale(x, residual, post, comb):
    """反例：遗漏 post 的完整系数，输出接口仍与 base 相同。"""
    mixed = jnp.matmul(jnp.swapaxes(comb.astype(jnp.float32), -1, -2), residual.astype(jnp.float32))
    return (
        jnp.float32(0.5) * post[..., None].astype(jnp.float32) * x[..., None, :].astype(jnp.float32)
        + mixed
    ).astype(x.dtype)


@jax.jit
def wrong_rounding(x, residual, post, comb):
    """反例：把 FP32 的流间累加提前舍入，不能当作普通实数结合律。"""
    mixed = jnp.matmul(jnp.swapaxes(comb.astype(jnp.float32), -1, -2), residual.astype(jnp.float32))
    mixed = mixed.astype(jnp.bfloat16).astype(jnp.float32)
    return (
        post[..., None].astype(jnp.float32) * x[..., None, :].astype(jnp.float32) + mixed
    ).astype(x.dtype)
