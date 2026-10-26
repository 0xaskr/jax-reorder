"""AI 直接编写并固定保存的候选；没有随机改写器或根据比较结果选择预期。

正例只移动独立运算，负例改变真实依赖或常量。共享函数仅用于计算步骤
本来相同的层家族；每个基准仍执行其所属上游类的原方法。FAMILIES 为每个
家族保存 (正例, 反例)，REASONS 保存人工预判依据；标签由写候选时的依赖
关系决定，不由比较器输出推导。
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .upstream import dependencies


def up_first(m, x):
    """正例：先算独立 up 投影，gate 激活后与 up 相乘的分组保持不变。"""
    up, _ = m.up_proj(x)
    gate, _ = m.gate_proj(x)
    return m.down_proj(up * m.act_fn(gate))[0]


def up_wrong_gate(m, x):
    """反例：把激活接到 up 投影、与 gate 相乘，形状相同但数据连接错误。"""
    up, _ = m.up_proj(x)
    gate, _ = m.gate_proj(x)
    return m.down_proj(gate * m.act_fn(up))[0]


def gate_activation_early(m, x):
    """正例：SiLU 提前到独立 up 投影之前，乘法分组不变。"""
    gate, _ = m.gate_proj(x)
    active = jax.nn.silu(gate)
    up, _ = m.up_proj(x)
    return m.down_proj(active * up)[0]


def gate_wrong_activation(m, x):
    """反例：门控激活由 SiLU 错换成 sigmoid，接口与 dtype 不变。"""
    gate, _ = m.gate_proj(x)
    up, _ = m.up_proj(x)
    return m.down_proj(jax.nn.sigmoid(gate) * up)[0]


def qwen_second_first(m, x):
    """正例：先算 w2 及其 SiLU 激活，再算 w1，保留 w1 乘激活的分组。"""
    second, _ = m.w2(x)
    active = jax.nn.silu(second)
    first, _ = m.w1(x)
    return m.c_proj(first * active)[0]


def qwen_wrong_gate(m, x):
    """反例：激活错误地应用到 w1，w2 直接作为乘数。"""
    first, _ = m.w1(x)
    second, _ = m.w2(x)
    return m.c_proj(second * jax.nn.silu(first))[0]


def vision_up_first(m, x):
    """正例：交换独立投影执行顺序，并保留 batch/tensor 分片输出参数。"""
    s = m.specs
    col = s.sharding(s.batch_axis, s.tensor_axis)
    up, _ = m.up_proj(x, out_sharding=col)
    gate, _ = m.gate_proj(x, out_sharding=col)
    return m.down_proj(m.act_fn(gate) * up, out_sharding=s.sharding(s.batch_axis))[0]


def vision_wrong_gate(m, x):
    """反例：交换门控与被门控的投影，其余分片与乘法位置不变。"""
    s = m.specs
    col = s.sharding(s.batch_axis, s.tensor_axis)
    gate, _ = m.gate_proj(x, out_sharding=col)
    up, _ = m.up_proj(x, out_sharding=col)
    return m.down_proj(m.act_fn(up) * gate, out_sharding=s.sharding(s.batch_axis))[0]


def gemma_vision_up_first(m, x):
    """正例：三维 token 布局下交换投影顺序，保留 gelu 近似与三维分片。"""
    s = m.specs
    col = s.sharding(s.batch_axis, None, s.tensor_axis)
    up, _ = m.up_proj(x, out_sharding=col)
    gate, _ = m.gate_proj(x, out_sharding=col)
    return m.down_proj(
        jax.nn.gelu(gate, approximate=True) * up, out_sharding=s.sharding(s.batch_axis)
    )[0]


def gemma_vision_wrong_gate(m, x):
    """反例：同一三维布局下把 gelu 接到 up 投影，破坏门控绑定。"""
    s = m.specs
    col = s.sharding(s.batch_axis, None, s.tensor_axis)
    gate, _ = m.gate_proj(x, out_sharding=col)
    up, _ = m.up_proj(x, out_sharding=col)
    return m.down_proj(
        jax.nn.gelu(up, approximate=True) * gate, out_sharding=s.sharding(s.batch_axis)
    )[0]


def grok_up_first(m, x):
    """正例：up 提前；GELU 门控、乘法与 reduce sharding 保持不变。"""
    col = NamedSharding(m.mesh, P("data", "tensor"))
    up, _ = m.up_proj(x, out_sharding=col)
    gate, _ = m.gate_proj(x, out_sharding=col)
    active, _ = m.act_fn(gate, up)
    row = dependencies()["make_reduce_sharding"](active, m.mesh, enable_sp=False)
    return m.down_proj(active, out_sharding=row)[0]


def grok_wrong_gate(m, x):
    """反例：激活输入改为 up 投影，偏置与分片参数保持不变。"""
    col = NamedSharding(m.mesh, P("data", "tensor"))
    gate, _ = m.gate_proj(x, out_sharding=col)
    up, _ = m.up_proj(x, out_sharding=col)
    active, _ = m.act_fn(up, gate)
    row = dependencies()["make_reduce_sharding"](active, m.mesh, enable_sp=False)
    return m.down_proj(active, out_sharding=row)[0]


def gated_norm_gate_first(m, x, gate):
    """正例：独立门控 sigmoid 与权重转换提前，FP32 归约顺序不变。"""
    active = jax.nn.sigmoid(gate.astype(jnp.float32))
    scale = m.weight[...].astype(jnp.float32)
    full = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(full), axis=-1, keepdims=True)
    norm = full * lax.rsqrt(variance + m.epsilon)
    return ((norm * scale) * active).astype(x.dtype)


def gated_norm_wrong_epsilon(m, x, gate):
    """反例：RMSNorm 稳定项从 1e-6 改为 0.1，其余路径不变。"""
    full = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(full), axis=-1, keepdims=True)
    norm = full * lax.rsqrt(variance + 0.1)
    norm = norm * m.weight[...].astype(jnp.float32)
    return (norm * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(x.dtype)


def qk_scale_first(m, x):
    """正例：scale 转换提前；默认 FP32 权重下该转换会在追踪中消去。"""
    scale = jnp.asarray(m.scale, jnp.float32)
    full = jnp.asarray(x, jnp.float32)
    var = jnp.mean(lax.square(full), axis=-1, keepdims=True)
    y = full * lax.rsqrt(var + m.epsilon)
    return (scale * y).astype(x.dtype)


def qk_wrong_epsilon(m, x):
    """反例：QK norm 稳定项改为 0.1，输出接口与 dtype 不变。"""
    full = jnp.asarray(x, jnp.float32)
    var = jnp.mean(lax.square(full), axis=-1, keepdims=True)
    y = full * lax.rsqrt(var + 0.1)
    return (jnp.asarray(m.scale, jnp.float32) * y).astype(x.dtype)


def clipped_up_first(m, x):
    """正例：up 提前，swigluoai 裁剪、alpha/limit 参数与乘法分组不变。"""
    up, _ = m.up_proj(x)
    gate, _ = m.gate_proj(x)
    active = dependencies()["swigluoai"](gate, up, m.swiglu_alpha, m.swiglu_limit)
    return m.down_proj(active)[0]


def clipped_wrong_alpha(m, x):
    """反例：swigluoai 的 sigmoid 缩放从 1.702 改为 0.5。"""
    gate, _ = m.gate_proj(x)
    up, _ = m.up_proj(x)
    active = dependencies()["swigluoai"](gate, up, 0.5, m.swiglu_limit)
    return m.down_proj(active)[0]


def nnx_up_first(m, x):
    """正例：NNX 投影顺序交换，保留 bias 与该层激活函数。"""
    up = m.up_proj(x)
    gate = m.act_fn(m.gate_proj(x))
    return m.down_proj(gate * up)


def nnx_wrong_gate(m, x):
    """反例：门控投影绑定错误，激活改接到 up 投影。"""
    up = m.up_proj(x)
    gate = m.gate_proj(x)
    return m.down_proj(m.act_fn(up) * gate)


def t5_second_first(m, x):
    """正例：wi_1 提前，激活、乘法分组与 dtype 边界均保留。"""
    h1, _ = m.wi_1(x)
    h0, _ = m.wi_0(x)
    h = (m.act(h0) * h1).astype(x.dtype)
    if hasattr(m, "dropout"):
        h = m.dropout(h, deterministic=True)
    return m.wo(h)[0]


def t5_wrong_gate(m, x):
    """反例：激活应用到错误投影，wi_0 与 wi_1 的角色互换。"""
    h0, _ = m.wi_0(x)
    h1, _ = m.wi_1(x)
    h = (m.act(h1) * h0).astype(x.dtype)
    if hasattr(m, "dropout"):
        h = m.dropout(h, deterministic=True)
    return m.wo(h)[0]


def markov_unpacked(m, logits, ids):
    """正例：展开 apply_step_logits，保留 embedding 与 bias 的共享 gather。

    get_prev_embeddings 的 gather 同时供 markov_w2 与返回值使用；展开后
    仍只做一次 gather，并保持 logits + bias 的加法方向。
    """
    # 返回 embedding 与 bias 共享同一次 gather；展开方法但保留共享关系。
    embedding = m.markov_w1.value.at[ids.astype(jnp.int32)].get(
        out_sharding=NamedSharding(m.mesh, P("data", None))
    )
    bias, _ = m.markov_w2(embedding)
    result = logits + bias
    return result, embedding


def markov_wrong_sign(m, logits, ids):
    """反例：Markov bias 加法错成减法，嵌入与输出结构不变。"""
    embedding = m.get_prev_embeddings(ids)
    bias, _ = m.markov_w2(embedding)
    return logits - bias, embedding


def router_weight_first(m, x):
    """正例：交换独立的类型/权重准备；dot 的 HIGHEST precision 保留。"""
    weight = m.kernel.value.astype(m.compute_dtype)
    active = x.astype(m.compute_dtype)
    logits = jnp.dot(active, weight, precision=lax.Precision.HIGHEST)
    return jax.nn.softmax(logits, axis=-1)


def router_wrong_axis(m, x):
    """反例：softmax 从专家轴改到 token 轴，形状不变但归一化语义错误。"""
    logits = jnp.dot(
        x.astype(m.compute_dtype),
        m.kernel.value.astype(m.compute_dtype),
        precision=lax.Precision.HIGHEST,
    )
    return jax.nn.softmax(logits, axis=0)


def _linear_without_bias(layer, x, target=None):
    """手写展开真实 LinearBase 的点积，bias 在候选开头独立准备。

    layer 提供 weight、params_dtype、mesh、kernel_axes；x 为输入；target
    为可选输出分片，缺省按 (data, ..., tensor) 规则构造。返回不含 bias
    的 dot_general 结果，bias 由调用方显式相加。
    """
    if target is None:
        target = NamedSharding(
            layer.mesh, P("data", *([None] * (x.ndim - 2)), layer.kernel_axes[-1])
        )
    return lax.dot_general(
        x,
        layer.weight.value,
        (((x.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=layer.params_dtype,
        out_sharding=target,
    )


def _early_bias(layer, rank):
    """把 bias 提前 reshape 成可广播到 rank 维输出的形状。"""
    return jnp.expand_dims(layer.bias.value, tuple(range(rank - 1)))


def wan_bias_first(m, x):
    """正例：末层 bias 广播提前，GELU 与 dot 参数完全保留。"""
    bias = _early_bias(m.fc_out, x.ndim)
    hidden, _ = m.fc_in(x)
    return _linear_without_bias(m.fc_out, m.act(hidden)) + bias


def wan_wrong_activation(m, x):
    """反例：GELU 错换成 SiLU，接口与输出形状不变。"""
    hidden, _ = m.fc_in(x)
    return m.fc_out(jax.nn.silu(hidden))[0]


def flux_bias_first(m, x):
    """正例：末层 bias 广播提前，保留推理 dropout 与 GELU 分组。"""
    bias = _early_bias(m.net[2], x.ndim)
    hidden, _ = m.net[0](x)
    hidden = m.net[1](m.act(hidden), deterministic=True)
    return _linear_without_bias(m.net[2], hidden) + bias


def flux_wrong_activation(m, x):
    """反例：GELU 错换成 SiLU，推理 dropout 位置不变。"""
    hidden, _ = m.net[0](x)
    hidden = m.net[1](jax.nn.silu(hidden), deterministic=True)
    return m.net[2](hidden)[0]


def clip_bias_first(m, x):
    """正例：末层 bias 广播提前，quick GELU 与线性层参数不变。"""
    bias = _early_bias(m.fc2, x.ndim)
    hidden, _ = m.fc1(x)
    return _linear_without_bias(m.fc2, m.act_fn(hidden)) + bias


def clip_wrong_activation(m, x):
    """反例：quick GELU 错换成 SiLU。"""
    hidden, _ = m.fc1(x)
    return m.fc2(jax.nn.silu(hidden))[0]


def omni_bias_first(m, x):
    """正例：末层 bias 广播提前，保留原 GELU 与投影参数。"""
    bias = _early_bias(m.fc2, x.ndim)
    hidden, _ = m.fc1(x)
    return _linear_without_bias(m.fc2, m.act(hidden)) + bias


def omni_wrong_activation(m, x):
    """反例：GELU 错换成 SiLU。"""
    hidden, _ = m.fc1(x)
    return m.fc2(jax.nn.silu(hidden))[0]


def vision_bias_first(m, x):
    """正例：末层 bias 广播提前，保留 gelu approximate 与分片参数。"""
    bias = _early_bias(m.fc2, x.ndim)
    s = m.specs
    hidden, _ = m.fc1(x, out_sharding=s.sharding(s.batch_axis, s.tensor_axis))
    active = jax.nn.gelu(hidden, approximate=m.approximate)
    return _linear_without_bias(m.fc2, active, s.sharding(s.batch_axis)) + bias


def vision_wrong_activation(m, x):
    """反例：GELU 错换成 SiLU，输出分片保持不变。"""
    s = m.specs
    hidden, _ = m.fc1(x, out_sharding=s.sharding(s.batch_axis, s.tensor_axis))
    return m.fc2(jax.nn.silu(hidden), out_sharding=s.sharding(s.batch_axis))[0]


def kimi_bias_first(m, x):
    """正例：down_proj bias reshape 提前，保留 promote_dtype 与 dot_general 参数。"""
    bias = jnp.reshape(m.down_proj.bias[...], (1, -1))
    hidden = jax.nn.gelu(m.up_proj(x), approximate=True)
    kernel = m.down_proj.kernel[...]
    hidden, kernel, _ = m.down_proj.promote_dtype((hidden, kernel, None), dtype=m.down_proj.dtype)
    output = m.down_proj.dot_general(
        hidden,
        kernel,
        (((hidden.ndim - 1,), (0,)), ((), ())),
        precision=m.down_proj.precision,
        out_sharding=None,
    )
    return output + bias


def kimi_wrong_activation(m, x):
    """反例：GELU 错换成 SiLU，仍走原 down_proj 投影。"""
    return m.down_proj(jax.nn.silu(m.up_proj(x)))


def rvq_gathers_first(m, codes):
    """正例：先 gather 第二级码本，仍按 first + second 的原次序累加。"""
    flat = codes.reshape(codes.shape[0], -1)
    second = m.codebooks[1].embedding.value[flat[1]]
    first = m.codebooks[0].embedding.value[flat[0]]
    decoded = jnp.zeros((flat.shape[1], m.dimension), dtype=jnp.float32)
    decoded = decoded + first
    decoded = decoded + second
    return decoded.reshape(*codes.shape[1:], m.dimension)


def rvq_wrong_codebook(m, codes):
    """反例：第二级误用第一级码本，量化和求和结构不变。"""
    flat = codes.reshape(codes.shape[0], -1)
    decoded = jnp.zeros((flat.shape[1], m.dimension), dtype=jnp.float32)
    decoded = decoded + m.codebooks[0].embedding.value[flat[0]]
    decoded = decoded + m.codebooks[0].embedding.value[flat[1]]
    return decoded.reshape(*codes.shape[1:], m.dimension)


def resnet_shortcut_first(m, x):
    """正例：独立 shortcut 卷积提前，norm/conv 分组与残差结构不变。"""
    residual = m.conv_shortcut(x)
    hidden = m.conv1(jax.nn.swish(m.norm1(x)))
    hidden = jax.nn.swish(m.norm2(hidden))
    hidden = m.dropout_layer(hidden, deterministic=True)
    hidden = m.conv2(hidden)
    return hidden + residual


def resnet_wrong_residual(m, x):
    """反例：残差加法错成减法，卷积与归一化路径不变。"""
    hidden = m.conv1(jax.nn.swish(m.norm1(x)))
    hidden = jax.nn.swish(m.norm2(hidden))
    hidden = m.conv2(m.dropout_layer(hidden, deterministic=True))
    return hidden - m.conv_shortcut(x)


def causal_cache_first(m, x, cache):
    """正例：返回缓存的切片提前，保留因果左填充与卷积分组。

    x 为新帧、cache 为时间历史；返回 (卷积输出, 新缓存)。先拼接缓存与
    新帧，再按 temporal_padding 截取最新窗口，padding 只补时间方向左侧。
    """
    joined = jnp.concatenate([cache, x], axis=1)
    new_cache = joined[:, -2 * m.temporal_padding :, :, :, :]
    padding = list(m.padding)
    padding[1] = (max(0, m.padding[1][0] - cache.shape[1]), 0)
    padded = jnp.pad(joined, tuple(padding), mode="constant")
    return m.conv(padded), new_cache


def causal_wrong_cache(m, x, cache):
    """反例：把历史缓存拼到新帧后面，返回的仍是同一段但时间顺序错误。"""
    joined = jnp.concatenate([x, cache], axis=1)
    padding = list(m.padding)
    padding[1] = (max(0, m.padding[1][0] - cache.shape[1]), 0)
    output = m.conv(jnp.pad(joined, tuple(padding), mode="constant"))
    return output, joined[:, -2 * m.temporal_padding :, :, :, :]


# 标签是写候选时基于依赖关系人工预判的，不由比较器输出推导。
FAMILIES = {
    "up_silu": (up_first, up_wrong_gate),
    "up_gelu": (up_first, up_wrong_gate),
    "gate_silu": (gate_activation_early, gate_wrong_activation),
    "qwen": (qwen_second_first, qwen_wrong_gate),
    "vision_gate_silu": (vision_up_first, vision_wrong_gate),
    "vision_gate_gelu": (gemma_vision_up_first, gemma_vision_wrong_gate),
    "grok": (grok_up_first, grok_wrong_gate),
    "gated_norm": (gated_norm_gate_first, gated_norm_wrong_epsilon),
    "qk_norm": (qk_scale_first, qk_wrong_epsilon),
    "swigluoai": (clipped_up_first, clipped_wrong_alpha),
    "nnx_gate_silu": (nnx_up_first, nnx_wrong_gate),
    "t5_gated": (t5_second_first, t5_wrong_gate),
    "markov": (markov_unpacked, markov_wrong_sign),
    "router": (router_weight_first, router_wrong_axis),
    "dit_mlp": (wan_bias_first, wan_wrong_activation),
    "flux_ffn": (flux_bias_first, flux_wrong_activation),
    "clip_mlp": (clip_bias_first, clip_wrong_activation),
    "omni_vision": (omni_bias_first, omni_wrong_activation),
    "vision_gelu": (vision_bias_first, vision_wrong_activation),
    "nnx_gelu": (kimi_bias_first, kimi_wrong_activation),
    "rvq_decode": (rvq_gathers_first, rvq_wrong_codebook),
    "resnet": (resnet_shortcut_first, resnet_wrong_residual),
    "causal_conv": (causal_cache_first, causal_wrong_cache),
}

# 与函数一起保存 AI 预判依据；共享家族仍逐个执行不同的上游类。
REASONS = {
    "up_silu": ("独立 up 投影提前，乘法输入位置保持不变", "把 SiLU 门控应用到错误投影"),
    "up_gelu": ("独立 up 投影提前，GELU 和乘法位置不变", "把 GELU 门控应用到错误投影"),
    "gate_silu": ("SiLU 提前到独立 up 投影之前", "SiLU 错换成 sigmoid"),
    "qwen": ("先计算 w2 与激活，保留 w1 乘激活的分组", "激活错误地应用于 w1"),
    "vision_gate_silu": ("交换独立投影的执行顺序，保留分片参数", "交换门控与被门控的投影"),
    "vision_gate_gelu": ("交换独立投影的执行顺序，保留三维分片", "交换门控与被门控的投影"),
    "grok": ("up 提前，GELU、乘法和分片不变", "激活输入改为 up"),
    "gated_norm": ("独立门控和权重转换提前", "稳定项从 1e-6 改为 0.1"),
    "qk_norm": ("权重转换提前；FP32 时转换会消去", "稳定项从 1e-6 改为 0.1"),
    "swigluoai": ("up 提前，裁剪及乘法分组不变", "sigmoid 缩放从 1.702 改为 0.5"),
    "nnx_gate_silu": ("up 提前，保留 NNX 投影与偏置", "门控投影绑定错误"),
    "t5_gated": ("wi_1 提前，保留激活、乘法及 dtype 边界", "激活应用于错误投影"),
    "markov": ("展开方法调用，保留共享 gather；原图顺序不变", "Markov bias 加法错成减法"),
    "router": ("交换独立类型准备；默认 FP32 权重转换消去", "softmax 从专家轴改到 token 轴"),
    "dit_mlp": ("末层 bias 广播提前，dot 参数完全保留", "GELU 错换成 SiLU"),
    "flux_ffn": ("末层 bias 广播提前，保留推理 dropout", "GELU 错换成 SiLU"),
    "clip_mlp": ("末层 bias 广播提前", "quick GELU 错换成 SiLU"),
    "omni_vision": ("末层 bias 广播提前", "GELU 错换成 SiLU"),
    "vision_gelu": ("末层 bias 广播提前，保留分片", "GELU 错换成 SiLU"),
    "nnx_gelu": ("末层 bias reshape 提前，保留 NNX 点积参数", "GELU 错换成 SiLU"),
    "rvq_decode": ("先 gather 第二级码本，仍按原次序累加", "第二级误用第一级码本"),
    "resnet": ("独立 shortcut 卷积提前", "残差加法错成减法"),
    "causal_conv": ("返回缓存的切片提前，保留因果填充", "把历史缓存拼到新帧后面"),
}
