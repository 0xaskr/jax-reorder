"""真实业务基准与 AI 直接编写的固定改写。

运行：uv run python examples/business_layer_cases.py
也可用 --scenario 筛选场景、--trials/--seed 调整数值输入、--show-jaxpr 查看
图、--report 保存 JSON。所有候选均为下方独立函数，源码与预判 hardcode；
随机数只用于测试输入。每个候选先按预判检查图结论，再用独立 NumPy 参照核对
base 数值；负例必须找到实际数值反例。采样不证明一般等价，也不能替代图判定。
基准提取自 Apache-2.0 项目 pallas-kernel / SGLang-JAX，来源与配置见
docs/exec-plans/completed/validation/business-layer-cases.md，许可证见 examples/LICENSE.*。
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import core

from jax_reorder import Status, canonicalize, compare_jaxprs


def norm_base(x, gate, scale):
    """Pallas gated RMSNorm：归一化先转换回输入 dtype，再执行 FP32 sigmoid 门控。"""
    x32 = x.astype(jnp.float32)
    rstd = lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + 1e-6)
    normalized = (x32 * rstd * scale.astype(jnp.float32)).astype(x.dtype)
    activated = jax.nn.sigmoid(gate.astype(jnp.float32))
    return (normalized.astype(jnp.float32) * activated).astype(x.dtype)


def norm_gate_first(x, gate, scale):
    """正例：先计算独立的门控，保留归一化路径上的转换边界。"""
    activated = jax.nn.sigmoid(gate.astype(jnp.float32))
    x32 = x.astype(jnp.float32)
    rstd = lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + 1e-6)
    normalized = (x32 * rstd * scale.astype(jnp.float32)).astype(x.dtype)
    return (normalized.astype(jnp.float32) * activated).astype(x.dtype)


def norm_prepare_scale(x, gate, scale):
    """正例：提前准备 scale 和 gate 的类型转换，独立分支交错执行。"""
    scale32 = scale.astype(jnp.float32)
    gate32 = gate.astype(jnp.float32)
    x32 = x.astype(jnp.float32)
    rstd = lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + 1e-6)
    activated = jax.nn.sigmoid(gate32)
    normalized = (x32 * rstd * scale32).astype(x.dtype)
    return (normalized.astype(jnp.float32) * activated).astype(x.dtype)


def norm_changed_epsilon(x, gate, scale):
    """负例：将稳定项从 1e-6 改成 0.1，改变归一化数值。"""
    x32 = x.astype(jnp.float32)
    rstd = lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + 0.1)
    normalized = (x32 * rstd * scale.astype(jnp.float32)).astype(x.dtype)
    activated = jax.nn.sigmoid(gate.astype(jnp.float32))
    return (normalized.astype(jnp.float32) * activated).astype(x.dtype)


def norm_wrong_gate(x, gate, scale):
    """负例：把 sigmoid 门控误改成 SiLU。"""
    x32 = x.astype(jnp.float32)
    rstd = lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + 1e-6)
    normalized = (x32 * rstd * scale.astype(jnp.float32)).astype(x.dtype)
    activated = jax.nn.silu(gate.astype(jnp.float32))
    return (normalized.astype(jnp.float32) * activated).astype(x.dtype)


def rope_base(x, cos, sin):
    """SGLang-JAX 的 Neox RoPE，输入为 token×head×dimension。"""
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate((o1, o2), axis=-1)


def rope_second_half_first(x, cos, sin):
    """正例：先计算后半输出，最终拼接端口顺序不变。"""
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    o2 = x2 * cos + x1 * sin
    o1 = x1 * cos - x2 * sin
    return jnp.concatenate((o1, o2), axis=-1)


def rope_products_first(x, cos, sin):
    """正例：交错准备四个独立乘积，保留加减法操作数及最终拼接。"""
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    b = x2 * sin
    d = x1 * sin
    a = x1 * cos
    c = x2 * cos
    o1 = a - b
    o2 = c + d
    return jnp.concatenate((o1, o2), axis=-1)


def rope_wrong_sign(x, cos, sin):
    """负例：旋转前半部分的减号改成加号。"""
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    o1 = x1 * cos + x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate((o1, o2), axis=-1)


def rope_swapped_outputs(x, cos, sin):
    """负例：计算公式保留，但两半向量的输出位置互换。"""
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate((o2, o1), axis=-1)


def attention_base(q, k, v):
    """Pallas MLA 的普通 causal attention 路径，支持非等长的 Q 与 KV。"""
    q32 = q.astype(jnp.float32)
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    qt = jnp.transpose(q32, (0, 2, 1, 3))
    kt = jnp.transpose(k32, (0, 2, 1, 3))
    vt = jnp.transpose(v32, (0, 2, 1, 3))
    scores = jnp.matmul(qt, jnp.transpose(kt, (0, 1, 3, 2))) * q.shape[-1] ** -0.5
    qpos = jnp.arange(q.shape[1])
    kpos = jnp.arange(k.shape[1])
    mask = (qpos[:, None] + (k.shape[1] - q.shape[1])) >= kpos[None, :]
    scores = jnp.where(mask[None, None, :, :], scores, float("-inf"))
    shifted = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    out = jnp.matmul(weights, vt)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def attention_mask_first(q, k, v):
    """正例：mask 只依赖 shape，可在 QK 点积前准备。"""
    qpos = jnp.arange(q.shape[1])
    kpos = jnp.arange(k.shape[1])
    mask = (qpos[:, None] + (k.shape[1] - q.shape[1])) >= kpos[None, :]
    q32 = q.astype(jnp.float32)
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    qt = jnp.transpose(q32, (0, 2, 1, 3))
    kt = jnp.transpose(k32, (0, 2, 1, 3))
    vt = jnp.transpose(v32, (0, 2, 1, 3))
    scores = jnp.matmul(qt, jnp.transpose(kt, (0, 1, 3, 2))) * q.shape[-1] ** -0.5
    scores = jnp.where(mask[None, None, :, :], scores, float("-inf"))
    shifted = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    out = jnp.matmul(weights, vt)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def attention_value_late(q, k, v):
    """正例：先计算 K 分支，V 的转换与转置推迟到 softmax 之后。"""
    k32 = k.astype(jnp.float32)
    q32 = q.astype(jnp.float32)
    kt = jnp.transpose(k32, (0, 2, 1, 3))
    qt = jnp.transpose(q32, (0, 2, 1, 3))
    scores = jnp.matmul(qt, jnp.transpose(kt, (0, 1, 3, 2))) * q.shape[-1] ** -0.5
    qpos = jnp.arange(q.shape[1])
    kpos = jnp.arange(k.shape[1])
    mask = (qpos[:, None] + (k.shape[1] - q.shape[1])) >= kpos[None, :]
    scores = jnp.where(mask[None, None, :, :], scores, float("-inf"))
    shifted = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    v32 = v.astype(jnp.float32)
    vt = jnp.transpose(v32, (0, 2, 1, 3))
    out = jnp.matmul(weights, vt)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def attention_wrong_scale(q, k, v):
    """负例：错误地乘 sqrt(head_dim)，而不是乘其倒数。"""
    q32 = q.astype(jnp.float32)
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    qt = jnp.transpose(q32, (0, 2, 1, 3))
    kt = jnp.transpose(k32, (0, 2, 1, 3))
    vt = jnp.transpose(v32, (0, 2, 1, 3))
    scores = jnp.matmul(qt, jnp.transpose(kt, (0, 1, 3, 2))) * q.shape[-1] ** 0.5
    qpos = jnp.arange(q.shape[1])
    kpos = jnp.arange(k.shape[1])
    mask = (qpos[:, None] + (k.shape[1] - q.shape[1])) >= kpos[None, :]
    scores = jnp.where(mask[None, None, :, :], scores, float("-inf"))
    shifted = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    out = jnp.matmul(weights, vt)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def attention_future_mask(q, k, v):
    """负例：反转因果 mask，错误关注当前及未来位置。"""
    q32 = q.astype(jnp.float32)
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    qt = jnp.transpose(q32, (0, 2, 1, 3))
    kt = jnp.transpose(k32, (0, 2, 1, 3))
    vt = jnp.transpose(v32, (0, 2, 1, 3))
    scores = jnp.matmul(qt, jnp.transpose(kt, (0, 1, 3, 2))) * q.shape[-1] ** -0.5
    qpos = jnp.arange(q.shape[1])
    kpos = jnp.arange(k.shape[1])
    mask = (qpos[:, None] + (k.shape[1] - q.shape[1])) <= kpos[None, :]
    scores = jnp.where(mask[None, None, :, :], scores, float("-inf"))
    shifted = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    out = jnp.matmul(weights, vt)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def cache_base(cache, loc, k, v):
    """SGLang KV 更新的展平数值核心：交错 K/V，丢弃负索引及越界写入。"""
    safe_loc = jnp.where(loc >= 0, loc, jnp.int32(cache.shape[0]))
    fused = jnp.stack((k, v), axis=2)
    updates = fused.reshape(k.shape[0], k.shape[1] * 2, k.shape[2])
    return cache.at[safe_loc].set(updates, mode="drop")


def cache_pack_first(cache, loc, k, v):
    """正例：先交错打包更新值，再计算安全索引。"""
    fused = jnp.stack((k, v), axis=2)
    updates = fused.reshape(k.shape[0], k.shape[1] * 2, k.shape[2])
    safe_loc = jnp.where(loc >= 0, loc, jnp.int32(cache.shape[0]))
    return cache.at[safe_loc].set(updates, mode="drop")


def cache_interleaved_preparation(cache, loc, k, v):
    """正例：索引条件和 K/V 打包交错，scatter 的输入位置不变。"""
    valid = loc >= 0
    fused = jnp.stack((k, v), axis=2)
    safe_loc = jnp.where(valid, loc, jnp.int32(cache.shape[0]))
    updates = fused.reshape(k.shape[0], k.shape[1] * 2, k.shape[2])
    return cache.at[safe_loc].set(updates, mode="drop")


def cache_invalid_to_zero(cache, loc, k, v):
    """负例：把无效 token 写到第零槽，覆盖仍然有效的缓存。"""
    safe_loc = jnp.where(loc >= 0, loc, jnp.int32(0))
    fused = jnp.stack((k, v), axis=2)
    updates = fused.reshape(k.shape[0], k.shape[1] * 2, k.shape[2])
    return cache.at[safe_loc].set(updates, mode="drop")


def cache_swapped_kv(cache, loc, k, v):
    """负例：将缓存内 K/V 的交错顺序写反。"""
    safe_loc = jnp.where(loc >= 0, loc, jnp.int32(cache.shape[0]))
    fused = jnp.stack((v, k), axis=2)
    updates = fused.reshape(k.shape[0], k.shape[1] * 2, k.shape[2])
    return cache.at[safe_loc].set(updates, mode="drop")


def router_base(hidden, kernel, bias):
    """SGLang MoE router：FP32 投影、sigmoid、带偏置选专家，权重取未加偏置的分数。"""
    h32 = hidden.astype(jnp.float32)
    w32 = kernel.astype(jnp.float32)
    logits = jnp.dot(h32, w32, precision=lax.Precision.HIGHEST)
    scores = jax.nn.sigmoid(logits)
    expanded = jnp.expand_dims(bias, axis=0)
    choice = scores + expanded
    ids = lax.top_k(choice, 2)[1]
    weights = jnp.take_along_axis(scores, ids, axis=1)
    return weights, ids


def router_bias_first(hidden, kernel, bias):
    """正例：专家偏置广播提前到投影之前。"""
    expanded = jnp.expand_dims(bias, axis=0)
    h32 = hidden.astype(jnp.float32)
    w32 = kernel.astype(jnp.float32)
    logits = jnp.dot(h32, w32, precision=lax.Precision.HIGHEST)
    scores = jax.nn.sigmoid(logits)
    choice = scores + expanded
    ids = lax.top_k(choice, 2)[1]
    weights = jnp.take_along_axis(scores, ids, axis=1)
    return weights, ids


def router_prepare_weights(hidden, kernel, bias):
    """正例：先转权重，再交错准备偏置，点积操作数与输出端口不变。"""
    w32 = kernel.astype(jnp.float32)
    expanded = jnp.expand_dims(bias, axis=0)
    h32 = hidden.astype(jnp.float32)
    logits = jnp.dot(h32, w32, precision=lax.Precision.HIGHEST)
    scores = jax.nn.sigmoid(logits)
    choice = scores + expanded
    ids = lax.top_k(choice, 2)[1]
    weights = jnp.take_along_axis(scores, ids, axis=1)
    return weights, ids


def router_biased_weights(hidden, kernel, bias):
    """负例：错误地把用于选择的 correction bias 也加入路由权重。"""
    h32 = hidden.astype(jnp.float32)
    w32 = kernel.astype(jnp.float32)
    logits = jnp.dot(h32, w32, precision=lax.Precision.HIGHEST)
    scores = jax.nn.sigmoid(logits)
    expanded = jnp.expand_dims(bias, axis=0)
    choice = scores + expanded
    ids = lax.top_k(choice, 2)[1]
    weights = jnp.take_along_axis(choice, ids, axis=1)
    return weights, ids


def router_wrong_correction_sign(hidden, kernel, bias):
    """负例：选择分数错误地减去 correction bias，专家身份可能改变。"""
    h32 = hidden.astype(jnp.float32)
    w32 = kernel.astype(jnp.float32)
    logits = jnp.dot(h32, w32, precision=lax.Precision.HIGHEST)
    scores = jax.nn.sigmoid(logits)
    expanded = jnp.expand_dims(bias, axis=0)
    choice = scores - expanded
    ids = lax.top_k(choice, 2)[1]
    weights = jnp.take_along_axis(scores, ids, axis=1)
    return weights, ids


def norm_numpy(x, gate, scale):
    """独立参照：显式计算 RMS，并保留中间转换回输入 dtype 的边界。"""
    x32 = x.astype(np.float32)
    variance = np.sum(x32 * x32, axis=-1, keepdims=True) / x.shape[-1]
    y = (x32 / np.sqrt(variance + 1e-6) * scale.astype(np.float32)).astype(x.dtype)
    return (y.astype(np.float32) / (1 + np.exp(-gate.astype(np.float32)))).astype(x.dtype)


def rope_numpy(x, cos, sin):
    """独立参照：分别给每个头应用二维旋转，低精度乘积按输入 dtype 舍入。"""
    half = x.shape[-1] // 2
    result = np.empty_like(x)
    for token in range(x.shape[0]):
        for head in range(x.shape[1]):
            a, b = (
                x[token, head, :half].astype(np.float32),
                x[token, head, half:].astype(np.float32),
            )
            c, s = (
                cos[token].astype(x.dtype).astype(np.float32),
                sin[token].astype(x.dtype).astype(np.float32),
            )
            ac = (a * c).astype(x.dtype).astype(np.float32)
            bs = (b * s).astype(x.dtype).astype(np.float32)
            bc = (b * c).astype(x.dtype).astype(np.float32)
            ass = (a * s).astype(x.dtype).astype(np.float32)
            result[token, head, :half] = ac - bs
            result[token, head, half:] = bc + ass
    return result


def attention_numpy(q, k, v):
    """独立参照：逐个 query 只读取可见前缀，避免复用 JAX mask 和转置路径。"""
    batch, queries, heads, dim = q.shape
    result = np.empty((batch, queries, heads, v.shape[-1]), dtype=v.dtype)
    for b in range(batch):
        for h in range(heads):
            for i in range(queries):
                end = k.shape[1] - queries + i + 1
                logits = k[b, :end, h].astype(np.float64) @ q[b, i, h].astype(np.float64)
                logits /= np.sqrt(dim)
                weights = np.exp(logits - np.max(logits))
                weights /= weights.sum()
                result[b, i, h] = weights @ v[b, :end, h].astype(np.float64)
    return result


def cache_numpy(cache, loc, k, v):
    """独立参照：按 token 与 head 写入 K/V，不复用 stack、reshape 或 scatter。"""
    result = cache.copy()
    for token, index in enumerate(loc):
        if 0 <= index < len(cache):
            for head in range(k.shape[1]):
                result[index, 2 * head] = k[token, head]
                result[index, 2 * head + 1] = v[token, head]
    return result


def router_numpy(hidden, kernel, bias):
    """独立参照：完整排序取前两个专家，分数并列时保留较小的专家索引。"""
    logits = hidden.astype(np.float32) @ kernel.astype(np.float32)
    scores = 1 / (1 + np.exp(-logits))
    ids = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :2].astype(np.int32)
    weights = np.array([[scores[row, expert] for expert in ids[row]] for row in range(len(ids))])
    return weights.astype(np.float32), ids


@dataclass(frozen=True)
class Variant:
    """AI 运行前预判；图标签与数值要求分开保存，不能由比较器生成真值。

    expected 是预期的图结论；numerical 为 "same" 时所有样本输出必须一致，
    为 "witness" 时必须找到实际数值反例；reason 记录改写理由供报告使用。
    """

    fn: Callable
    expected: Status
    numerical: str
    reason: str


@dataclass(frozen=True)
class Scenario:
    """固定场景，绑定 base、独立 NumPy 参照和手写候选列表。"""

    name: str
    base: Callable
    oracle: Callable
    variants: tuple[Variant, ...]


# 以下 20 个源码引用、标签和理由均由 AI 直接填写；没有候选生成器。
SCENARIOS = (
    Scenario(
        "gated_rmsnorm",
        norm_base,
        norm_numpy,
        (
            Variant(norm_gate_first, Status.EQUIVALENT, "same", "只提前独立的 sigmoid 门控分支"),
            Variant(
                norm_prepare_scale,
                Status.EQUIVALENT,
                "same",
                "独立类型转换与门控交错，保留中间低精度舍入",
            ),
            Variant(
                norm_changed_epsilon, Status.DIFFERENT, "witness", "RMS 稳定项从 1e-6 改为 0.1"
            ),
            Variant(
                norm_wrong_gate, Status.DIFFERENT, "witness", "sigmoid 错换成 SiLU，门控计算改变"
            ),
        ),
    ),
    Scenario(
        "rope",
        rope_base,
        rope_numpy,
        (
            Variant(
                rope_second_half_first,
                Status.EQUIVALENT,
                "same",
                "先计算后半结果，拼接顺序仍为 o1、o2",
            ),
            Variant(
                rope_products_first,
                Status.EQUIVALENT,
                "same",
                "四个乘积换序，操作数与加减分组保持不变",
            ),
            Variant(rope_wrong_sign, Status.DIFFERENT, "witness", "旋转前半部分的减号改为加号"),
            Variant(rope_swapped_outputs, Status.DIFFERENT, "witness", "拼接端口顺序改为 o2、o1"),
        ),
    ),
    Scenario(
        "causal_attention",
        attention_base,
        attention_numpy,
        (
            Variant(
                attention_mask_first, Status.EQUIVALENT, "same", "不依赖输入数值的 mask 提前准备"
            ),
            Variant(
                attention_value_late,
                Status.EQUIVALENT,
                "same",
                "K/Q 独立准备换序，V 分支推迟到 softmax 后",
            ),
            Variant(
                attention_wrong_scale,
                Status.DIFFERENT,
                "witness",
                "点积缩放从 dim**-0.5 改为 dim**0.5",
            ),
            Variant(attention_future_mask, Status.DIFFERENT, "witness", "因果 mask 的比较方向反转"),
        ),
    ),
    Scenario(
        "kv_cache",
        cache_base,
        cache_numpy,
        (
            Variant(cache_pack_first, Status.EQUIVALENT, "same", "K/V 打包与安全索引计算独立换序"),
            Variant(
                cache_interleaved_preparation,
                Status.EQUIVALENT,
                "same",
                "交错准备索引与更新值，scatter 连接不变",
            ),
            Variant(
                cache_invalid_to_zero, Status.DIFFERENT, "witness", "负位置不再丢弃，错误覆盖第零槽"
            ),
            Variant(cache_swapped_kv, Status.DIFFERENT, "witness", "K/V 在缓存内的槽位写反"),
        ),
    ),
    Scenario(
        "moe_router",
        router_base,
        router_numpy,
        (
            Variant(
                router_bias_first, Status.EQUIVALENT, "same", "独立的 correction bias 广播提前"
            ),
            Variant(
                router_prepare_weights, Status.EQUIVALENT, "same", "独立输入转换与偏置广播交错执行"
            ),
            Variant(
                router_biased_weights,
                Status.DIFFERENT,
                "witness",
                "返回带 correction bias 的分数，错误改变路由权重",
            ),
            Variant(
                router_wrong_correction_sign,
                Status.DIFFERENT,
                "witness",
                "选择分数减去偏置，改变专家选择",
            ),
        ),
    ),
)

# 三组固定签名：FP32、BF16 与 FP32 小形状；每组另有两个边界样本。
PROFILES = ("float32", "bfloat16", "small")


def make_samples(scenario, profile, trials, seed):
    """两个固定边界样本加 trials 个随机数值样本，不改变任何函数源码或配置规则。"""
    if trials < 1 or profile not in PROFILES:
        raise ValueError("trials 必须大于零，profile 必须是已声明的输入签名")
    rng = np.random.default_rng(seed)
    dtype = np.dtype(jnp.bfloat16) if profile == "bfloat16" else np.dtype(np.float32)
    small = profile == "small"

    def normal(shape):
        return rng.normal(size=shape).astype(dtype)

    samples = []
    for index in range(trials + 2):
        if scenario.name == "gated_rmsnorm":
            shape = (1, 1, 1, 4) if small else (2, 3, 2, 8)
            x, gate, scale = normal(shape), normal(shape), normal((shape[-1],))
            if index == 0:
                x.fill(0.01)
                gate.fill(0)
                scale.fill(1)
            elif index == 1:
                x.fill(0)
            args = (x, gate, scale)
        elif scenario.name == "rope":
            shape = (1, 3, 4) if small else (3, 2, 8)
            x = normal(shape)
            angles = rng.uniform(-np.pi, np.pi, (shape[0], shape[-1] // 2))
            cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
            if index == 0:
                cos.fill(0)
                sin.fill(1)
            elif index == 1:
                cos.fill(1)
                sin.fill(0)
            args = (x, cos, sin)
        elif scenario.name == "causal_attention":
            # small 为单 query decode，BF16 为已有 prefix 的非等长 attention。
            tq, tk = (1, 6) if small else ((2, 5) if profile == "bfloat16" else (4, 4))
            q, k, v = normal((1, tq, 2, 4)), normal((1, tk, 2, 4)), normal((1, tk, 2, 3))
            if index == 0:
                q.fill(0)
                k.fill(0)
                v[:] = np.arange(tk)[None, :, None, None]
            elif index == 1:
                q.fill(1)
                k[:] = np.linspace(-0.5, 0.5, tk)[None, :, None, None]
                v[:] = np.arange(tk)[None, :, None, None]
            args = (q, k, v)
        elif scenario.name == "kv_cache":
            tokens, heads, dim = (1, 1, 4) if small else (3, 2, 3)
            cache, k, v = (
                normal((8, heads * 2, dim)),
                normal((tokens, heads, dim)),
                normal((tokens, heads, dim)),
            )
            # 有效位置互异且避开零槽；负例也不会形成重复 scatter 索引的不确定写入。
            loc = rng.choice(np.arange(1, 8), tokens, replace=False).astype(np.int32)
            if index == 0:
                loc[0] = -1
                cache.fill(-3)
                k.fill(1)
                v.fill(2)
            elif index == 1:
                loc[:] = np.arange(1, tokens + 1)
                k.fill(1)
                v.fill(2)
            elif index % 2 == 0:
                loc[0] = 9
            args = (cache, loc, k, v)
        elif scenario.name == "moe_router":
            tokens, hidden_dim, experts = (1, 3, 4) if small else (3, 5, 6)
            hidden, kernel = normal((tokens, hidden_dim)), normal((hidden_dim, experts))
            bias = rng.uniform(-0.5, 0.5, experts).astype(np.float32)
            if index == 0:
                hidden.fill(0)
                bias[:] = np.linspace(-0.8, 0.8, experts)
            elif index == 1:
                hidden.fill(0)
                bias.fill(0)
            args = (hidden, kernel, bias)
        else:
            raise ValueError(f"未知场景：{scenario.name}")
        samples.append(args)
    return samples


def check_values(expected, actual):
    """逐叶检查签名与数值；索引必须精确一致，浮点用与输出 dtype 对应的容差。"""
    left, right = jax.tree.leaves(expected), jax.tree.leaves(actual)
    assert len(left) == len(right), "输出叶子数改变"
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, "输出 shape/dtype 改变"
        if np.issubdtype(a.dtype, np.integer):
            np.testing.assert_array_equal(a, b)
        else:
            tol = 2e-2 if a.dtype == np.dtype(jnp.bfloat16) else 2e-5
            # 转成 FP32 后再交给 NumPy，避免 BF16 的 isclose 类型提升问题。
            assert np.all(np.isfinite(a.astype(np.float32)))
            assert np.all(np.isfinite(b.astype(np.float32)))
            np.testing.assert_allclose(
                a.astype(np.float32), b.astype(np.float32), rtol=tol, atol=tol
            )


def values_match(left, right):
    """使用同一数值标准寻找反例，不能用结构不同代替输出不同。"""
    try:
        check_values(left, right)
        return True
    except AssertionError:
        return False


def serialize_arrays(value):
    """报告保存完整小样本、dtype 与 shape，便于复查而不是只打印哈希。"""
    return [
        {"dtype": str(x.dtype), "shape": list(x.shape), "values": np.asarray(x).tolist()}
        for x in jax.tree.leaves(value)
    ]


def verify_variant(scenario, variant, profile, samples, *, show_jaxpr=False):
    """先读取固定预判，再做独立参照、精确图比较和两侧重排执行验证。"""
    # 图结构只依赖输入签名，因此用第一个样本追踪一次；数值检查仍遍历全部样本。
    graphs = [jax.make_jaxpr(fn)(*samples[0]) for fn in (scenario.base, variant.fn)]
    comparison = compare_jaxprs(*graphs)
    assert comparison.status == variant.expected, (
        scenario.name,
        variant.fn.__name__,
        profile,
        variant.reason,
        comparison,
    )
    normalized = [canonicalize(graph) for graph in graphs]
    for item in normalized:
        core.check_jaxpr(item.jaxpr)
        assert canonicalize(item.jaxpr).graph == item.graph, "规范化不幂等"

    originals = [jax.jit(scenario.base), jax.jit(variant.fn)]
    reordered = [
        jax.jit(lambda *args, g=item.jaxpr: core.eval_jaxpr(g, g.consts, *args))
        for item in normalized
    ]
    witness = None
    for index, args in enumerate(samples):
        a, b = [fn(*args) for fn in originals]
        assert jax.tree.structure(a) == jax.tree.structure(b), "函数输出 pytree 改变"
        check_values(scenario.oracle(*args), a)
        for original, fn in zip((a, b), reordered, strict=True):
            check_values(original, fn(*jax.tree.leaves(args)))
        if variant.numerical == "same":
            check_values(a, b)
        elif variant.numerical == "witness":
            if witness is None and not values_match(a, b):
                witness = {
                    "sample_index": index,
                    "inputs": serialize_arrays(args),
                    "base_outputs": serialize_arrays(a),
                    "candidate_outputs": serialize_arrays(b),
                }
        else:
            raise AssertionError(f"未知数值要求：{variant.numerical}")
    assert variant.numerical != "witness" or witness is not None, "没有找到预判要求的数值反例"
    if show_jaxpr:
        for label, graph, item in zip(("base", "candidate"), graphs, normalized, strict=True):
            print(f"{variant.fn.__name__}/{profile} {label}：\n{graph}\n重排后：\n{item.jaxpr}")
    return {
        "scenario": scenario.name,
        "base": scenario.base.__name__,
        "candidate": variant.fn.__name__,
        "profile": profile,
        "ai_expected": variant.expected.value,
        "ai_reason": variant.reason,
        "actual": comparison.status.value,
        "difference": comparison.difference,
        "samples": len(samples),
        "numpy_reference": "passed",
        "reordered_execution": "passed",
        "numerical_requirement": variant.numerical,
        "witness": witness,
    }


def main():
    """重放 hardcode 的候选，可筛选场景、查看图或保存完整数值反例报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[s.name for s in SCENARIOS])
    parser.add_argument(
        "--trials", type=int, default=3, help="每个签名的随机输入数，另有两个边界输入"
    )
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--show-jaxpr", action="store_true")
    parser.add_argument("--report", type=Path, help="保存 JSON 报告，包含完整数值反例")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials 必须大于零")
    jax.config.update("jax_platforms", "cpu")
    results = []
    for scenario in SCENARIOS:
        if args.scenario and args.scenario != scenario.name:
            continue
        for variant in scenario.variants:
            for profile in PROFILES:
                samples = make_samples(scenario, profile, args.trials, args.seed)
                result = verify_variant(
                    scenario, variant, profile, samples, show_jaxpr=args.show_jaxpr
                )
                results.append(result)
            print(
                f"PASS {scenario.name}/{variant.fn.__name__}: "
                f"AI={variant.expected.value}；3 个签名均通过"
            )
    if args.report:
        report = {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "seed": args.seed,
            "random_trials": args.trials,
            "boundary_samples": 2,
            "candidate_origin": "AI 直接编写并 hardcode；运行时不生成改写",
            "results": results,
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"报告已保存：{args.report}")
    print(f"全部通过：{len(results)} 个候选/签名组合。数值样本不替代图等价判定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
