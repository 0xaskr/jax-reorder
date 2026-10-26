"""真实 mHC 接口、固定测试形状及独立 NumPy 数值/一阶参照。

PROFILES 描述 (batch, sequence, streams, hidden)，其中 (1, 1, 4, 5120) 是
模型真实 H=4、D=5120 的配置，其余为小形状快速回归。signatures/inputs 只
提供声明签名与采样输入；forward/jvp/vjp_reference 按逐流定义独立实现，
不调用候选、上游 hc_post 或 jaxpr 比较器。in_context 使用上游 hc_mixes
产生真实混合系数，把 hc_post 或候选放回 20 次 Sinkhorn 的完整上下文。

复现入口见 `python -m tests.deepseek_v41.runner`；这里的容差是采样
回归阈值，不是浮点误差界或逐位等价承诺。
"""

import jax
import jax.numpy as jnp
import numpy as np

from .upstream import original

# (batch, sequence, streams, hidden)；最后一项是模型真实的 H=4、D=5120 配置。
PROFILES = ((1, 1, 4, 8), (2, 3, 4, 7), (1, 2, 3, 9), (1, 1, 4, 5120))


def signatures(profile, dtype):
    """返回 hc_post 四个输入口的抽象签名，顺序为 x、residual、post、comb。

    前两个张量跟随 profile 的 dtype（真实激活精度）；post 与 comb 由 FP32
    混合系数产生，固定为 float32。只构造 ShapeDtypeStruct，不分配数组、
    不执行计算，也不检查实际取值。
    """
    batch, sequence, streams, hidden = profile
    shapes = (
        (batch, sequence, hidden),
        (batch, sequence, streams, hidden),
        (batch, sequence, streams),
        (batch, sequence, streams, streams),
    )
    # post/comb 是 Sinkhorn 归一化后的 FP32 系数，不随激活 dtype 变化。
    return tuple(
        jax.ShapeDtypeStruct(shape, dtype if i < 2 else jnp.float32)
        for i, shape in enumerate(shapes)
    )


def inputs(profile, dtype, seed):
    """按给定种子生成四口随机输入，尺度 0.5 使各流贡献保持同量级。

    返回与 signatures 顺序一致的 tuple；不同 seed 得到不同样本，调用方
    负责记录 seed 以便精确重放。
    """
    rng = np.random.default_rng(seed)
    return tuple(
        jnp.asarray(rng.normal(0, 0.5, spec.shape), spec.dtype)
        for spec in signatures(profile, dtype)
    )


def cancellation_inputs(profile, dtype):
    """构造 post 分支与流混合近似抵消的样本，放大连接错误与舍入影响。

    返回四元组，dtype 约定与 signatures 相同。x 全为 -1，仅 0 号来源流
    非零、comb[0, :]=1.003，使 base 每个目标流约为 0.003；遗漏转置时差值
    约 1.003，提前 BF16 舍入则约 0.003，便于分别识别两类错误。
    """
    batch, sequence, streams, hidden = profile
    x = -jnp.ones((batch, sequence, hidden), dtype)
    residual = np.zeros((batch, sequence, streams, hidden), np.float32)
    residual[..., 0, :] = 1
    post = jnp.ones((batch, sequence, streams), jnp.float32)
    comb = np.zeros((batch, sequence, streams, streams), np.float32)
    # 略大于 1 的系数让 base 的 post*x 与混合项不完全抵消，保留可测差值。
    comb[..., 0, :] = np.float32(1.003)
    return x, jnp.asarray(residual, dtype), post, jnp.asarray(comb)


def forward_reference(args):
    """按来源流逐项累加，不复用候选 matmul 或广播乘积。

    args 为 (x, residual, post, comb)；返回 [B,S,H,D]，dtype 取 args[0]。
    外层循环固定目标流、内层固定来源流，累加顺序与公式一致，作为独立
    数值参照而不是逐位标准。
    """
    x, residual, post, comb = [np.asarray(v, np.float32) for v in args]
    result = np.zeros(residual.shape, np.float32)
    for target in range(residual.shape[-2]):
        mixed = np.zeros(x.shape, np.float32)
        for source in range(residual.shape[-2]):
            mixed += comb[..., source, target, None] * residual[..., source, :]
        result[..., target, :] = post[..., target, None] * x + mixed
    return result.astype(args[0].dtype)


def jvp_reference(args, tangents):
    """显式微分每一项，最后按原始输出 dtype 转换。

    args 与 tangents 都是四元组（x, residual, post, comb 的取值/切向量）；
    返回 [B,S,H,D]，dtype 取 args[0]。对 post*x 用乘积法则，对流混合项
    同时计入 comb 与 residual 的切向量。
    """
    x, residual, post, comb = [np.asarray(v, np.float32) for v in args]
    dx, dr, dp, dc = [np.asarray(v, np.float32) for v in tangents]
    result = np.zeros(residual.shape, np.float32)
    for target in range(residual.shape[-2]):
        result[..., target, :] = dp[..., target, None] * x + post[..., target, None] * dx
        for source in range(residual.shape[-2]):
            result[..., target, :] += dc[..., source, target, None] * residual[..., source, :]
            result[..., target, :] += comb[..., source, target, None] * dr[..., source, :]
    return result.astype(args[0].dtype)


def vjp_reference(args, cotangent):
    """按四个输入端口独立写出伴随公式，所有流/隐藏维均有显式归约。

    args 为四元组，cotangent 为输出 [B,S,H,D] 的余切；返回按 (x, residual,
    post, comb) 顺序的梯度 tuple，各自 dtype 与原输入一致。该参照独立于
    JAX 转置规则，用于核对候选的实际 VJP 接口。
    """
    x, residual, post, comb = [np.asarray(v, np.float32) for v in args]
    cotangent = np.asarray(cotangent, np.float32)
    dx, dr, dp, dc = [np.zeros_like(v) for v in (x, residual, post, comb)]
    for target in range(residual.shape[-2]):
        g = cotangent[..., target, :]
        dx += post[..., target, None] * g
        dp[..., target] = np.sum(x * g, axis=-1)
        for source in range(residual.shape[-2]):
            dr[..., source, :] += comb[..., source, target, None] * g
            dc[..., source, target] = np.sum(residual[..., source, :] * g, axis=-1)
    return tuple(v.astype(a.dtype) for v, a in zip((dx, dr, dp, dc), args, strict=True))


def check_close(actual, expected):
    """采样阈值只用于执行回归；报告实际误差，不替代实数证明。

    先按低精度/FP32 选择 rtol/atol（低精度 0.016，FP32 为 2e-5/2e-4），
    超限由 assert_allclose 抛出 AssertionError；返回 FP32 下的最大绝对
    误差，供报告记录。比较在 FP32 上进行，不承诺逐位一致。
    """
    a, b = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    low_precision = np.dtype(actual.dtype) in (np.dtype(jnp.bfloat16), np.dtype(jnp.float16))
    rtol, atol = (0.016, 0.016) if low_precision else (2e-5, 2e-4)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return float(np.max(np.abs(a - b), initial=0))


def check_numerics(function, args, seed=0):
    """对给定函数一次性核对前向、JVP 与 VJP 的实际数值。

    用 seed 生成切向量与余切向量，分别调用 jax.jvp 和 jax.vjp；返回
    forward_max_abs、jvp_max_abs、vjp_max_abs 三个最大绝对误差。阈值
    不足时抛出 AssertionError，不修改输入、不写报告。
    """
    rng = np.random.default_rng(seed)
    tangents = tuple(jnp.asarray(rng.normal(0, 0.3, a.shape), a.dtype) for a in args)
    value, tangent = jax.jvp(function, args, tangents)
    cotangent = jnp.asarray(rng.normal(0, 0.3, value.shape), value.dtype)
    _, backward = jax.vjp(function, *args)
    gradients = backward(cotangent)
    return {
        "forward_max_abs": check_close(value, forward_reference(args)),
        "jvp_max_abs": check_close(tangent, jvp_reference(args, tangents)),
        "vjp_max_abs": max(
            check_close(a, b)
            for a, b in zip(gradients, vjp_reference(args, cotangent), strict=True)
        ),
    }


def in_context(post_function, *, streams=4, sinkhorn_iters=20):
    """原始混合系数生成、20 次 Sinkhorn、pre 与指定完整 post 的连接。

    返回 run(x, residual, weight, scale, bias)：先用上游 hc_mixes 生成
    (pre, post, comb)，再执行 post_function 和上游 hc_pre。两个输出都参与
    后续比较，因此候选若改变混合系数或 pre 路径同样会被发现。streams
    与 sinkhorn_iters 是上游静态参数，默认保持模型真实值。
    """
    layer = original()

    def run(x, residual, weight, scale, bias):
        """固定调用顺序：先产生系数，再分别执行 post 分支和 pre 分支。"""
        pre, post, comb = layer.hc_mixes(
            residual, weight, scale, bias, hc_mult=streams, sinkhorn_iters=sinkhorn_iters
        )
        return post_function(x, residual, post, comb), layer.hc_pre(residual, pre)

    return run


def context_inputs(profile, dtype, seed):
    """生成完整上下文所需的激活与 hc_mixes 参数，形状随 profile 变化。

    返回 (x, residual, weight, scale, bias)。weight 的行数为
    streams²+2·streams（comb、pre、post 三组打包），列为展平后的
    streams·hidden；scale 的 3 个元素对应三组系数。所有系数固定为 FP32。
    """
    x, residual, _, _ = inputs(profile, dtype, seed)
    _, _, streams, hidden = profile
    width = streams * streams + 2 * streams
    # 与上游初始化一致：权重小尺度随机、偏置小尺度随机，scale 固定为三个值。
    rng = np.random.default_rng(seed + 1000)
    weight = jnp.asarray(rng.normal(0, 0.1, (width, streams * hidden)), jnp.float32)
    scale = jnp.asarray([0.9, 1.1, 0.8], jnp.float32)
    bias = jnp.asarray(rng.normal(0, 0.2, (width,)), jnp.float32)
    return x, residual, weight, scale, bias
