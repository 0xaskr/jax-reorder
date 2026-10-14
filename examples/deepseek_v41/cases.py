"""真实 mHC 接口、固定测试形状及独立 NumPy 数值/一阶参照。"""

import jax
import jax.numpy as jnp
import numpy as np

from .upstream import original

PROFILES = ((1, 1, 4, 8), (2, 3, 4, 7), (1, 2, 3, 9), (1, 1, 4, 5120))


def signatures(profile, dtype):
    batch, sequence, streams, hidden = profile
    shapes = (
        (batch, sequence, hidden),
        (batch, sequence, streams, hidden),
        (batch, sequence, streams),
        (batch, sequence, streams, streams),
    )
    return tuple(
        jax.ShapeDtypeStruct(shape, dtype if i < 2 else jnp.float32)
        for i, shape in enumerate(shapes)
    )


def inputs(profile, dtype, seed):
    rng = np.random.default_rng(seed)
    return tuple(
        jnp.asarray(rng.normal(0, 0.5, spec.shape), spec.dtype)
        for spec in signatures(profile, dtype)
    )


def cancellation_inputs(profile, dtype):
    batch, sequence, streams, hidden = profile
    x = -jnp.ones((batch, sequence, hidden), dtype)
    residual = np.zeros((batch, sequence, streams, hidden), np.float32)
    residual[..., 0, :] = 1
    post = jnp.ones((batch, sequence, streams), jnp.float32)
    comb = np.zeros((batch, sequence, streams, streams), np.float32)
    comb[..., 0, :] = np.float32(1.003)
    return x, jnp.asarray(residual, dtype), post, jnp.asarray(comb)


def forward_reference(args):
    """按来源流逐项累加，不复用候选 matmul 或广播乘积。"""
    x, residual, post, comb = [np.asarray(v, np.float32) for v in args]
    result = np.zeros(residual.shape, np.float32)
    for target in range(residual.shape[-2]):
        mixed = np.zeros(x.shape, np.float32)
        for source in range(residual.shape[-2]):
            mixed += comb[..., source, target, None] * residual[..., source, :]
        result[..., target, :] = post[..., target, None] * x + mixed
    return result.astype(args[0].dtype)


def jvp_reference(args, tangents):
    """显式微分每一项，最后按原始输出 dtype 转换。"""
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
    """按四个输入端口独立写出伴随公式，所有流/隐藏维均有显式归约。"""
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
    """采样阈值只用于执行回归；报告实际误差，不替代实数证明。"""
    a, b = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    low_precision = np.dtype(actual.dtype) in (np.dtype(jnp.bfloat16), np.dtype(jnp.float16))
    rtol, atol = (0.016, 0.016) if low_precision else (2e-5, 2e-4)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return float(np.max(np.abs(a - b), initial=0))


def check_numerics(function, args, seed=0):
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
    """原始混合系数生成、20 次 Sinkhorn、pre 与指定完整 post 的连接。"""
    layer = original()

    def run(x, residual, weight, scale, bias):
        pre, post, comb = layer.hc_mixes(
            residual, weight, scale, bias, hc_mult=streams, sinkhorn_iters=sinkhorn_iters
        )
        return post_function(x, residual, post, comb), layer.hc_pre(residual, pre)

    return run


def context_inputs(profile, dtype, seed):
    x, residual, _, _ = inputs(profile, dtype, seed)
    _, _, streams, hidden = profile
    width = streams * streams + 2 * streams
    rng = np.random.default_rng(seed + 1000)
    weight = jnp.asarray(rng.normal(0, 0.1, (width, streams * hidden)), jnp.float32)
    scale = jnp.asarray([0.9, 1.1, 0.8], jnp.float32)
    bias = jnp.asarray(rng.normal(0, 0.2, (width,)), jnp.float32)
    return x, residual, weight, scale, bias
