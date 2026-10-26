"""AI 手写的完整 hc_post 候选；不从 base 的 AST 或 jaxpr 自动生成。

原始公式见固定上游 reference.py:113，Apache-2.0 许可证随快照保留。
post_matmul 与 post_blocked 只在理想实数解释下等价：其输入输出形状、
dtype 和调用接口不变，但临时张量大小与舍入顺序改变，实际误差和性能必须
由独立数值参照、编译器报告与计时分别测量。wrong_* 是形状相同而计算
错误的反例，用于确认比较器不会把数值接近或接口相同当成等价。

运行方式与报告字段见 `python -m tests.deepseek_v41.runner`。
"""

import jax
import jax.numpy as jnp


@jax.jit
def post_matmul(x, residual, post, comb):
    """把流间广播乘积归约变为批量矩阵乘法，避免显式 hc²D 中间张量。

    输入 x [B,S,D]、residual [B,S,H,D]、post [B,S,H]、comb [B,S,H,H]；
    返回 [B,S,H,D]，dtype 与 x 相同。中间量统一用 FP32 计算，最后一次
    转换与原始 hc_post 保持相同的舍入边界。逻辑中间张量由 H²D 降到 HD，
    但实际临时内存和延迟以编译报告与计时为准，不能由本函数推断。
    """
    x32, residual32 = x.astype(jnp.float32), residual.astype(jnp.float32)
    post32, comb32 = post.astype(jnp.float32), comb.astype(jnp.float32)
    # comb[..., source, target]；转置后沿来源流收缩，输出下标仍为目标流。
    mixed = jnp.matmul(jnp.swapaxes(comb32, -1, -2), residual32)
    return (post32[..., None] * x32[..., None, :] + mixed).astype(x.dtype)


@jax.jit
def post_blocked(x, residual, post, comb):
    """按来源流分两块归约，缩小每块广播乘积并重新结合两段求和。

    输入与 post_matmul 相同；返回 [B,S,H,D] 且 dtype 与 x 相同。cut 取
    来源流数的一半；两段区间不重叠且覆盖全部来源流，逐项索引绑定不变。
    分块只改变有限求和的结合顺序，浮点舍入差异仍须独立测量。
    """
    # 来源流在倒数第二维；只切分来源维，目标维和隐藏维保持原下标。
    cut = residual.shape[-2] // 2
    x32, residual32 = x.astype(jnp.float32), residual.astype(jnp.float32)
    post32, comb32 = post.astype(jnp.float32), comb.astype(jnp.float32)
    first = jnp.sum(comb32[..., :cut, :, None] * residual32[..., :cut, None, :], axis=-3)
    second = jnp.sum(comb32[..., cut:, :, None] * residual32[..., cut:, None, :], axis=-3)
    # 两段求和与 post 分支的相加顺序固定，避免把浮点重结合混入候选。
    return (first + (post32[..., None] * x32[..., None, :] + second)).astype(x.dtype)


@jax.jit
def wrong_transpose(x, residual, post, comb):
    """反例：交换 comb 的来源流和目标流，形状相同但连接错误。

    comb 未转置就与 residual 相乘，等价于把 comb[source,target] 当作
    comb[target,source]；输出接口不变，必须由精确图比较拒绝并给出数值见证。
    """
    mixed = jnp.matmul(comb.astype(jnp.float32), residual.astype(jnp.float32))
    return (
        post[..., None].astype(jnp.float32) * x[..., None, :].astype(jnp.float32) + mixed
    ).astype(x.dtype)


@jax.jit
def wrong_scale(x, residual, post, comb):
    """反例：遗漏 post 的完整系数，输出接口仍与 base 相同。

    post 被固定乘以 0.5，形状与 dtype 不变；比较器应给出 different，
    数值参照应在抵消输入下暴露约 0.5 的差值。
    """
    mixed = jnp.matmul(jnp.swapaxes(comb.astype(jnp.float32), -1, -2), residual.astype(jnp.float32))
    return (
        jnp.float32(0.5) * post[..., None].astype(jnp.float32) * x[..., None, :].astype(jnp.float32)
        + mixed
    ).astype(x.dtype)


@jax.jit
def wrong_rounding(x, residual, post, comb):
    """反例：把 FP32 的流间累加提前舍入，不能当作普通实数结合律。

    mixed 先降到 BF16 再回到 FP32；对实数公式没有影响，但改变实际 dtype
    转换边界。用于确认比较器不会把低精度舍入当作允许的代数改写。
    """
    mixed = jnp.matmul(jnp.swapaxes(comb.astype(jnp.float32), -1, -2), residual.astype(jnp.float32))
    mixed = mixed.astype(jnp.bfloat16).astype(jnp.float32)
    return (
        post[..., None].astype(jnp.float32) * x[..., None, :].astype(jnp.float32) + mixed
    ).astype(x.dtype)
