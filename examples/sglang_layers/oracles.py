"""独立 NumPy 参照：不调用上游 forward、AI 候选或 jaxpr 比较器。

矩阵与卷积累加使用较高精度，层边界按真实输出 dtype 舍入。这里的容差
检查是数值证据；图是否相等仍必须精确比较，不能由 allclose 代替。
"""

import itertools

import numpy as np


def array(value):
    if hasattr(value, "get_value"):
        value = value.get_value()
    return np.asarray(value)


def act(x, kind):
    full = x.astype(np.float64)
    if kind == "silu":
        out = full / (1 + np.exp(-full))
    elif kind == "quick":
        out = full / (1 + np.exp(-1.702 * full))
    elif kind == "sigmoid":
        out = 1 / (1 + np.exp(-full))
    elif kind == "gelu":
        out = 0.5 * full * (1 + np.tanh(np.sqrt(2 / np.pi) * (full + 0.044715 * full**3)))
    else:
        raise ValueError(kind)
    return out.astype(x.dtype)


def dense(layer, x):
    kernel = array(layer.weight if hasattr(layer, "weight") else layer.kernel)
    # LinearBase 显式请求参数 dtype；NNX 路径按输入/权重共同类型计算。
    dtype = (
        np.dtype(layer.params_dtype)
        if hasattr(layer, "params_dtype")
        else np.result_type(x, kernel)
    )
    out = (x.astype(np.float64) @ kernel.astype(np.float64)).astype(dtype)
    if layer.bias is not None:
        out = (out.astype(np.float64) + array(layer.bias).astype(np.float64)).astype(dtype)
    return out


def product(a, b):
    return (a.astype(np.float64) * b.astype(np.float64)).astype(np.result_type(a, b))


def norm(layer, x, weight):
    full = x.astype(np.float64)
    inverse = 1 / np.sqrt(np.sum(full * full, axis=-1, keepdims=True) / x.shape[-1] + layer.epsilon)
    return full * inverse * array(weight).astype(np.float64)


def conv(layer, x, padding):
    """逐位置求和的独立 NHWC/NDHWC 卷积，不复用 lax 或 JAX 编译结果。"""
    kernel, bias = array(layer.kernel), array(layer.bias)
    dims = kernel.ndim - 2
    padded = np.pad(x.astype(np.float64), ((0, 0), *padding, (0, 0)))
    spatial = tuple(padded.shape[d + 1] - kernel.shape[d] + 1 for d in range(dims))
    result = np.empty((x.shape[0], *spatial, kernel.shape[-1]), np.float64)
    for batch in range(x.shape[0]):
        for pos in itertools.product(*(range(n) for n in spatial)):
            patch = padded[
                (
                    batch,
                    *(slice(i, i + k) for i, k in zip(pos, kernel.shape[:dims], strict=True)),
                    slice(None),
                )
            ]
            result[(batch, *pos)] = np.sum(
                patch[..., None] * kernel.astype(np.float64), axis=tuple(range(dims + 1))
            )
    # 卷积结果和 bias 相加分开舍入，保留低精度计算边界。
    dtype = np.result_type(x, kernel)
    result = result.astype(dtype)
    return (result.astype(np.float64) + bias.astype(np.float64)).astype(dtype)


def group_norm(layer, x):
    groups = layer.num_groups
    full = x.astype(np.float64).reshape(*x.shape[:-1], groups, -1)
    axes = (*range(1, full.ndim - 2), full.ndim - 1)
    mean = full.mean(axis=axes, keepdims=True)
    var = ((full - mean) ** 2).mean(axis=axes, keepdims=True)
    out = ((full - mean) / np.sqrt(var + layer.epsilon)).reshape(x.shape)
    out = out * array(layer.scale).astype(np.float64) + array(layer.bias).astype(np.float64)
    return out.astype(x.dtype)


def reference(spec, m, *inputs):
    """依据层的数学定义及 dtype 边界进行独立计算，拒绝未登记家族。"""
    xs = [array(x) for x in inputs]
    x = xs[0]
    family = spec.family
    if family in (
        "up_silu",
        "up_gelu",
        "gate_silu",
        "vision_gate_silu",
        "vision_gate_gelu",
        "grok",
        "nnx_gate_silu",
    ):
        gate, up = dense(m.gate_proj, x), dense(m.up_proj, x)
        kind = "gelu" if family in ("up_gelu", "vision_gate_gelu", "grok") else "silu"
        return dense(m.down_proj, product(act(gate, kind), up))
    if family == "qwen":
        return dense(m.c_proj, product(dense(m.w1, x), act(dense(m.w2, x), "silu")))
    if family == "swigluoai":
        gate = np.minimum(dense(m.gate_proj, x), m.swiglu_limit).astype(x.dtype)
        up = np.clip(dense(m.up_proj, x), -m.swiglu_limit, m.swiglu_limit).astype(x.dtype)
        sigmoid = act((gate.astype(np.float64) * m.swiglu_alpha).astype(x.dtype), "sigmoid")
        fused = product(product((up.astype(np.float64) + 1).astype(x.dtype), gate), sigmoid)
        return dense(m.down_proj, fused)
    if family == "gated_norm":
        normalized = norm(m, x, m.weight)
        sigmoid = 1 / (1 + np.exp(-xs[1].astype(np.float64)))
        return (normalized * sigmoid).astype(x.dtype)
    if family == "qk_norm":
        return norm(m, x, m.scale).astype(x.dtype)
    if family == "nnx_gelu":
        return dense(m.down_proj, act(dense(m.up_proj, x), "gelu"))
    if family in ("vision_gelu", "omni_vision", "clip_mlp"):
        return dense(m.fc2, act(dense(m.fc1, x), "quick" if family == "clip_mlp" else "gelu"))
    if family == "t5_gated":
        return dense(m.wo, product(act(dense(m.wi_0, x), "gelu"), dense(m.wi_1, x)))
    if family == "dit_mlp":
        return dense(m.fc_out, act(dense(m.fc_in, x), "gelu"))
    if family == "flux_ffn":
        return dense(m.net[2], act(dense(m.net[0], x), "gelu"))
    if family == "router":
        logits = x.astype(np.float64) @ array(m.kernel).astype(np.float64)
        exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return (exponent / exponent.sum(axis=-1, keepdims=True)).astype(np.float32)
    if family == "markov":
        embedding = array(m.markov_w1)[xs[1].astype(np.int32)]
        bias = dense(m.markov_w2, embedding)
        return (x.astype(np.float64) + bias.astype(np.float64)).astype(x.dtype), embedding
    if family == "rvq_decode":
        out = np.zeros((*x.shape[1:], m.dimension), np.float32)
        for level in range(x.shape[0]):
            out += array(m.codebooks[level].embedding)[x[level]].astype(np.float32)
        return out
    if family == "resnet":
        hidden = conv(m.conv1, act(group_norm(m.norm1, x), "silu"), ((1, 1), (1, 1)))
        hidden = conv(m.conv2, act(group_norm(m.norm2, hidden), "silu"), ((1, 1), (1, 1)))
        residual = conv(m.conv_shortcut, x, ((0, 0), (0, 0)))
        return (hidden.astype(np.float64) + residual.astype(np.float64)).astype(hidden.dtype)
    if family == "causal_conv":
        cache = xs[1]
        joined = np.concatenate((cache, x), axis=1)
        temporal = max(0, 2 * m.temporal_padding - cache.shape[1])
        return conv(m.conv, joined, ((temporal, 0), m.padding[2], m.padding[3])), joined[
            :, -2 * m.temporal_padding :
        ]
    raise ValueError(f"没有独立数值参照：{family}")
