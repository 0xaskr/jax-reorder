"""AI 直接编写并固定保存的候选；没有随机改写器或根据比较结果选择预期。

正例只移动独立运算，负例改变真实依赖或常量。共享函数仅用于计算步骤
本来相同的层家族；每个基准仍执行其所属上游类的原方法。
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .upstream import dependencies


def up_first(m, x):
    up, _ = m.up_proj(x)
    gate, _ = m.gate_proj(x)
    return m.down_proj(up * m.act_fn(gate))[0]


def up_wrong_gate(m, x):
    up, _ = m.up_proj(x)
    gate, _ = m.gate_proj(x)
    return m.down_proj(gate * m.act_fn(up))[0]


def gate_activation_early(m, x):
    gate, _ = m.gate_proj(x)
    active = jax.nn.silu(gate)
    up, _ = m.up_proj(x)
    return m.down_proj(active * up)[0]


def gate_wrong_activation(m, x):
    gate, _ = m.gate_proj(x)
    up, _ = m.up_proj(x)
    return m.down_proj(jax.nn.sigmoid(gate) * up)[0]


def qwen_second_first(m, x):
    second, _ = m.w2(x)
    active = jax.nn.silu(second)
    first, _ = m.w1(x)
    return m.c_proj(first * active)[0]


def qwen_wrong_gate(m, x):
    first, _ = m.w1(x)
    second, _ = m.w2(x)
    return m.c_proj(second * jax.nn.silu(first))[0]


def vision_up_first(m, x):
    s = m.specs
    col = s.sharding(s.batch_axis, s.tensor_axis)
    up, _ = m.up_proj(x, out_sharding=col)
    gate, _ = m.gate_proj(x, out_sharding=col)
    return m.down_proj(m.act_fn(gate) * up, out_sharding=s.sharding(s.batch_axis))[0]


def vision_wrong_gate(m, x):
    s = m.specs
    col = s.sharding(s.batch_axis, s.tensor_axis)
    gate, _ = m.gate_proj(x, out_sharding=col)
    up, _ = m.up_proj(x, out_sharding=col)
    return m.down_proj(m.act_fn(up) * gate, out_sharding=s.sharding(s.batch_axis))[0]


def gemma_vision_up_first(m, x):
    s = m.specs
    col = s.sharding(s.batch_axis, None, s.tensor_axis)
    up, _ = m.up_proj(x, out_sharding=col)
    gate, _ = m.gate_proj(x, out_sharding=col)
    return m.down_proj(
        jax.nn.gelu(gate, approximate=True) * up, out_sharding=s.sharding(s.batch_axis)
    )[0]


def gemma_vision_wrong_gate(m, x):
    s = m.specs
    col = s.sharding(s.batch_axis, None, s.tensor_axis)
    gate, _ = m.gate_proj(x, out_sharding=col)
    up, _ = m.up_proj(x, out_sharding=col)
    return m.down_proj(
        jax.nn.gelu(up, approximate=True) * gate, out_sharding=s.sharding(s.batch_axis)
    )[0]


def grok_up_first(m, x):
    col = NamedSharding(m.mesh, P("data", "tensor"))
    up, _ = m.up_proj(x, out_sharding=col)
    gate, _ = m.gate_proj(x, out_sharding=col)
    active, _ = m.act_fn(gate, up)
    row = dependencies()["make_reduce_sharding"](active, m.mesh, enable_sp=False)
    return m.down_proj(active, out_sharding=row)[0]


def grok_wrong_gate(m, x):
    col = NamedSharding(m.mesh, P("data", "tensor"))
    gate, _ = m.gate_proj(x, out_sharding=col)
    up, _ = m.up_proj(x, out_sharding=col)
    active, _ = m.act_fn(up, gate)
    row = dependencies()["make_reduce_sharding"](active, m.mesh, enable_sp=False)
    return m.down_proj(active, out_sharding=row)[0]


def gated_norm_gate_first(m, x, gate):
    active = jax.nn.sigmoid(gate.astype(jnp.float32))
    scale = m.weight[...].astype(jnp.float32)
    full = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(full), axis=-1, keepdims=True)
    norm = full * lax.rsqrt(variance + m.epsilon)
    return ((norm * scale) * active).astype(x.dtype)


def gated_norm_wrong_epsilon(m, x, gate):
    full = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(full), axis=-1, keepdims=True)
    norm = full * lax.rsqrt(variance + 0.1)
    norm = norm * m.weight[...].astype(jnp.float32)
    return (norm * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(x.dtype)


def qk_scale_first(m, x):
    scale = jnp.asarray(m.scale, jnp.float32)
    full = jnp.asarray(x, jnp.float32)
    var = jnp.mean(lax.square(full), axis=-1, keepdims=True)
    y = full * lax.rsqrt(var + m.epsilon)
    return (scale * y).astype(x.dtype)


def qk_wrong_epsilon(m, x):
    full = jnp.asarray(x, jnp.float32)
    var = jnp.mean(lax.square(full), axis=-1, keepdims=True)
    y = full * lax.rsqrt(var + 0.1)
    return (jnp.asarray(m.scale, jnp.float32) * y).astype(x.dtype)


def clipped_up_first(m, x):
    up, _ = m.up_proj(x)
    gate, _ = m.gate_proj(x)
    active = dependencies()["swigluoai"](gate, up, m.swiglu_alpha, m.swiglu_limit)
    return m.down_proj(active)[0]


def clipped_wrong_alpha(m, x):
    gate, _ = m.gate_proj(x)
    up, _ = m.up_proj(x)
    active = dependencies()["swigluoai"](gate, up, 0.5, m.swiglu_limit)
    return m.down_proj(active)[0]


def nnx_up_first(m, x):
    up = m.up_proj(x)
    gate = m.act_fn(m.gate_proj(x))
    return m.down_proj(gate * up)


def nnx_wrong_gate(m, x):
    up = m.up_proj(x)
    gate = m.gate_proj(x)
    return m.down_proj(m.act_fn(up) * gate)


def t5_second_first(m, x):
    h1, _ = m.wi_1(x)
    h0, _ = m.wi_0(x)
    h = (m.act(h0) * h1).astype(x.dtype)
    if hasattr(m, "dropout"):
        h = m.dropout(h, deterministic=True)
    return m.wo(h)[0]


def t5_wrong_gate(m, x):
    h0, _ = m.wi_0(x)
    h1, _ = m.wi_1(x)
    h = (m.act(h1) * h0).astype(x.dtype)
    if hasattr(m, "dropout"):
        h = m.dropout(h, deterministic=True)
    return m.wo(h)[0]


def markov_unpacked(m, logits, ids):
    # 返回 embedding 与 bias 共享同一次 gather；展开方法但保留共享关系。
    embedding = m.markov_w1.value.at[ids.astype(jnp.int32)].get(
        out_sharding=NamedSharding(m.mesh, P("data", None))
    )
    bias, _ = m.markov_w2(embedding)
    result = logits + bias
    return result, embedding


def markov_wrong_sign(m, logits, ids):
    embedding = m.get_prev_embeddings(ids)
    bias, _ = m.markov_w2(embedding)
    return logits - bias, embedding


def router_weight_first(m, x):
    weight = m.kernel.value.astype(m.compute_dtype)
    active = x.astype(m.compute_dtype)
    logits = jnp.dot(active, weight, precision=lax.Precision.HIGHEST)
    return jax.nn.softmax(logits, axis=-1)


def router_wrong_axis(m, x):
    logits = jnp.dot(
        x.astype(m.compute_dtype),
        m.kernel.value.astype(m.compute_dtype),
        precision=lax.Precision.HIGHEST,
    )
    return jax.nn.softmax(logits, axis=0)


def _linear_without_bias(layer, x, target=None):
    """手写展开真实 LinearBase 的点积，bias 在候选开头独立准备。"""
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
    return jnp.expand_dims(layer.bias.value, tuple(range(rank - 1)))


def wan_bias_first(m, x):
    bias = _early_bias(m.fc_out, x.ndim)
    hidden, _ = m.fc_in(x)
    return _linear_without_bias(m.fc_out, m.act(hidden)) + bias


def wan_wrong_activation(m, x):
    hidden, _ = m.fc_in(x)
    return m.fc_out(jax.nn.silu(hidden))[0]


def flux_bias_first(m, x):
    bias = _early_bias(m.net[2], x.ndim)
    hidden, _ = m.net[0](x)
    hidden = m.net[1](m.act(hidden), deterministic=True)
    return _linear_without_bias(m.net[2], hidden) + bias


def flux_wrong_activation(m, x):
    hidden, _ = m.net[0](x)
    hidden = m.net[1](jax.nn.silu(hidden), deterministic=True)
    return m.net[2](hidden)[0]


def clip_bias_first(m, x):
    bias = _early_bias(m.fc2, x.ndim)
    hidden, _ = m.fc1(x)
    return _linear_without_bias(m.fc2, m.act_fn(hidden)) + bias


def clip_wrong_activation(m, x):
    hidden, _ = m.fc1(x)
    return m.fc2(jax.nn.silu(hidden))[0]


def omni_bias_first(m, x):
    bias = _early_bias(m.fc2, x.ndim)
    hidden, _ = m.fc1(x)
    return _linear_without_bias(m.fc2, m.act(hidden)) + bias


def omni_wrong_activation(m, x):
    hidden, _ = m.fc1(x)
    return m.fc2(jax.nn.silu(hidden))[0]


def vision_bias_first(m, x):
    bias = _early_bias(m.fc2, x.ndim)
    s = m.specs
    hidden, _ = m.fc1(x, out_sharding=s.sharding(s.batch_axis, s.tensor_axis))
    active = jax.nn.gelu(hidden, approximate=m.approximate)
    return _linear_without_bias(m.fc2, active, s.sharding(s.batch_axis)) + bias


def vision_wrong_activation(m, x):
    s = m.specs
    hidden, _ = m.fc1(x, out_sharding=s.sharding(s.batch_axis, s.tensor_axis))
    return m.fc2(jax.nn.silu(hidden), out_sharding=s.sharding(s.batch_axis))[0]


def kimi_bias_first(m, x):
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
    return m.down_proj(jax.nn.silu(m.up_proj(x)))


def rvq_gathers_first(m, codes):
    flat = codes.reshape(codes.shape[0], -1)
    second = m.codebooks[1].embedding.value[flat[1]]
    first = m.codebooks[0].embedding.value[flat[0]]
    decoded = jnp.zeros((flat.shape[1], m.dimension), dtype=jnp.float32)
    decoded = decoded + first
    decoded = decoded + second
    return decoded.reshape(*codes.shape[1:], m.dimension)


def rvq_wrong_codebook(m, codes):
    flat = codes.reshape(codes.shape[0], -1)
    decoded = jnp.zeros((flat.shape[1], m.dimension), dtype=jnp.float32)
    decoded = decoded + m.codebooks[0].embedding.value[flat[0]]
    decoded = decoded + m.codebooks[0].embedding.value[flat[1]]
    return decoded.reshape(*codes.shape[1:], m.dimension)


def resnet_shortcut_first(m, x):
    residual = m.conv_shortcut(x)
    hidden = m.conv1(jax.nn.swish(m.norm1(x)))
    hidden = jax.nn.swish(m.norm2(hidden))
    hidden = m.dropout_layer(hidden, deterministic=True)
    hidden = m.conv2(hidden)
    return hidden + residual


def resnet_wrong_residual(m, x):
    hidden = m.conv1(jax.nn.swish(m.norm1(x)))
    hidden = jax.nn.swish(m.norm2(hidden))
    hidden = m.conv2(m.dropout_layer(hidden, deterministic=True))
    return hidden - m.conv_shortcut(x)


def causal_cache_first(m, x, cache):
    joined = jnp.concatenate([cache, x], axis=1)
    new_cache = joined[:, -2 * m.temporal_padding :, :, :, :]
    padding = list(m.padding)
    padding[1] = (max(0, m.padding[1][0] - cache.shape[1]), 0)
    padded = jnp.pad(joined, tuple(padding), mode="constant")
    return m.conv(padded), new_cache


def causal_wrong_cache(m, x, cache):
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
