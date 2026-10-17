"""NumPy 完整层参照：独立计算投影、旋转、mask、softmax 和 cache 更新。

不导入上游类、AI 候选或 jaxpr 比较器；各层按数学定义逐步实现，并在每个
运算边界按真实 dtype 舍入。这里只提供数值证据，图是否相等仍由比较器精确
判定，不能用 allclose 代替，也不能把参照当作逐位标准。
"""

import numpy as np

from examples.sglang_layers.oracles import act, array, dense, norm


def rotary(x, positions, base):
    """独立实现 RoPE：按相邻分量配对旋转，保留 dtype 的舍入边界。

    x 的形状为 [..., 2n]，positions 为 token 位置；返回与 x 同 shape、
    同 dtype 的旋转结果。cos/sin 与乘积先舍入到输入 dtype，再在 float64
    中做加减并写回，复现低精度实现的分步舍入，而不是纯实数公式。
    """
    dtype = x.dtype
    frequency = np.arange(0, x.shape[-1], 2, dtype=np.float32) / x.shape[-1]
    angles = np.asarray(positions, np.float32)[:, None] * (1.0 / base**frequency)
    cos = np.cos(angles).astype(dtype)[:, None, :].astype(np.float64)
    sin = np.sin(angles).astype(dtype)[:, None, :].astype(np.float64)
    even, odd = x[..., ::2].astype(np.float64), x[..., 1::2].astype(np.float64)
    ec, es = (even * cos).astype(dtype), (even * sin).astype(dtype)
    oc, os = (odd * cos).astype(dtype), (odd * sin).astype(dtype)
    result = np.empty_like(x)
    # 旋转公式：out_even = e·cos - o·sin，out_odd = o·cos + e·sin。
    result[..., ::2] = ec.astype(np.float64) - os.astype(np.float64)
    result[..., 1::2] = oc.astype(np.float64) + es.astype(np.float64)
    return result


def glm5(case, args):
    """独立重算 Glm5Attention 前向（含 KV 写入），不调用原 forward。

    args 为 builders.Glm5Case.sample 返回的九元组；返回 (attn 输出, 更新
    后的 cache)。维度常量 256/448/192/64 是上游构造默认值，builder 只缩小
    hidden 与 token，因此这些维度必须保持一致。
    """
    weights, hidden, positions, initial_cache, lengths, locations, writes, prefix, _ = args
    m = case.module(weights)
    hidden, dtype = array(hidden), array(hidden).dtype
    q = dense(m.q_a_proj, hidden)
    q = norm(m.q_a_layernorm, q, m.q_a_layernorm.scale).astype(dtype)
    q = dense(m.q_b_proj, q).reshape(-1, m.num_heads, 256)
    latent = dense(m.kv_a_proj_with_mqa, hidden)
    # MLA latent 拆成 KV 压缩部分 512 与 rope 部分 64；qk_nope=192、v=256。
    compressed, key_rope = latent[:, :512], latent[:, 512:].reshape(-1, 1, 64)
    compressed = norm(m.kv_a_layernorm, compressed, m.kv_a_layernorm.scale).astype(dtype)
    kv = dense(m.kv_b_proj, compressed).reshape(-1, m.num_heads, 448)
    q = np.concatenate((q[..., :192], rotary(q[..., 192:], positions, m.rotary_emb.base)), -1)
    key_rope = rotary(key_rope, positions, m.rotary_emb.base)
    key = np.concatenate((kv[..., :192], np.broadcast_to(key_rope, (*kv.shape[:2], 64))), -1)
    value = kv[..., 192:]
    attended, updated = native(case, args, q, key, value, m.scaling)
    return dense(m.o_proj, attended), updated


def native(case, args, q, key, value, scaling):
    """独立实现分页 KV 写入、按 cache_loc 读取与 causal/decode 掩码。

    q/key/value 为已投影张量；返回 (attention 输出, 更新后 cache)。只用
    writes 中的合法目标写入 K/V，按 locations 聚集历史；0 号页与超出
    length 的槽视为无效。EXTEND 模式按 prefix+query 施加因果约束。
    """
    _, hidden, _, initial_cache, lengths, locations, writes, prefix, _ = args
    dtype = key.dtype
    kv_heads, dim = key.shape[1:]
    query_heads = q.shape[1]
    updated = array(initial_cache).copy()
    # cache 页布局为 [token, kv_head, 2, dim]：0 存 K、1 存 V，与上游合并写一致。
    rows = updated.reshape(-1, kv_heads, 2, dim)
    for i, target in enumerate(array(writes)):
        if 0 <= target < len(rows):
            rows[target, :, 0] = key[i]
            rows[target, :, 1] = value[i]
    loc = array(locations)
    gathered = np.zeros((len(loc), kv_heads, 2, dim), dtype=dtype)
    for i, index in enumerate(loc):
        # 0 号页保留不读；越界位置同样按无效处理。
        if 0 < index < len(rows):
            gathered[i] = rows[index]
    k, v = gathered[:, :, 0], gathered[:, :, 1]
    # GQA：K/V head 数少于 Q head 时按组复制，与上游 repeat 语义一致。
    k = np.repeat(k, query_heads // kv_heads, axis=1)
    v = np.repeat(v, query_heads // kv_heads, axis=1)
    logits = np.einsum("qhd,khd->qhk", q.astype(np.float64), k.astype(np.float64)).astype(dtype)
    logits = (logits.astype(np.float64) * np.asarray(scaling, dtype=dtype)).astype(dtype)
    scores = logits.astype(np.float64)
    for query in range(len(hidden)):
        for column in range(len(loc)):
            # 无效页、超过真实长度，以及 EXTEND 下的未来 token 全部屏蔽。
            valid = loc[column] > 0 and column < int(array(lengths)[0])
            if case.profile.mode == "EXTEND":
                valid = valid and column <= int(array(prefix)[0]) + query
            if not valid:
                scores[query, :, column] = -np.inf
    # softmax 的每一步都回到 dtype 再继续，模拟低精度归约，而非纯 float64。
    shifted = (scores - scores.max(axis=-1, keepdims=True)).astype(dtype).astype(np.float64)
    exps = np.exp(shifted).astype(dtype).astype(np.float64)
    denominators = exps.sum(axis=-1, keepdims=True).astype(dtype).astype(np.float64)
    probability = (exps / denominators).astype(dtype).astype(np.float64)
    attn = np.einsum("qhk,khd->qhd", probability, v.astype(np.float64)).astype(dtype)
    return attn.reshape(len(hidden), -1), updated


def llama(case, args):
    """独立重算 LlamaAttention GQA 前向，返回 (attn 输出, 更新后 cache)。

    只使用 q/k/v/o 投影、RoPE 与 attention 参数，K/V 写入与掩码交给
    native；不调用候选或原 forward。
    """
    weights, hidden, positions, *_ = args
    m = case.module(weights)
    q = dense(m.q_proj, array(hidden)).reshape(-1, m.q_head_num, m.head_dim)
    k = dense(m.k_proj, array(hidden)).reshape(-1, m.kv_head_num, m.head_dim)
    v = dense(m.v_proj, array(hidden)).reshape(-1, m.kv_head_num, m.head_dim)
    q, k = rotary(q, positions, m.rope_theta), rotary(k, positions, m.rope_theta)
    attended, cache = native(case, args, q, k, v, m.scaling)
    return dense(m.o_proj, attended), cache


def layer_norm(layer, x):
    """用 float64 统计量实现 LayerNorm，再按层的声明 dtype 舍入。

    layer 需提供 epsilon、可选 scale/bias 与 dtype；返回与 x 同 shape 的
    归一化结果。不读取上游 module 的 forward。
    """
    f = x.astype(np.float64)
    mean = f.mean(axis=-1, keepdims=True)
    var = ((f - mean) ** 2).mean(axis=-1, keepdims=True)
    result = (f - mean) / np.sqrt(var + layer.epsilon)
    if layer.scale is not None:
        result *= array(layer.scale).astype(np.float64)
    if layer.bias is not None:
        result += array(layer.bias).astype(np.float64)
    return result.astype(layer.dtype or x.dtype)


def attention(q, k, v, scale, mask=None):
    """独立实现多头缩放点积 attention；mask 为布尔可见性矩阵。

    q/k/v 形状为 [B,Q,H,D]/[B,K,H,D]；mask 为 None 时不做因果或填充
    屏蔽。返回 [B,Q,H,D] 的 float64 结果，由调用方决定 dtype。
    """
    logits = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64)) * scale
    if mask is not None:
        logits = np.where(mask, logits, -np.inf)
    # 先减去行最大值再求指数，避免指数溢出；该稳定性变换不改变数学值。
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probability = exponent / exponent.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", probability, v.astype(np.float64))


def clip(case, args):
    """独立重算 CLIPEncoderLayer：norm→MHA→残差→norm→MLP→残差。

    args 为 (weights, x, mask)；mask 为 None 表示因果 tril，否则按有效位
    屏蔽。返回与上游 forward 同 shape/dtype 的张量。
    """
    weights, x, mask = args
    m, x = case.module(weights), array(x)
    z = layer_norm(m.layer_norm1, x)
    attn = m.self_attn
    shape = (*z.shape[:2], attn.num_heads, attn.head_dim)
    q, k, v = (dense(layer, z).reshape(shape) for layer in (attn.q_proj, attn.k_proj, attn.v_proj))
    valid = (
        np.tril(np.ones((x.shape[1], x.shape[1]), bool))[None, None, :, :]
        if mask is None
        else array(mask)[:, None, None, :] > 0
    )
    z = attention(q, k, v, attn.scale, valid).reshape(x.shape).astype(x.dtype)
    attended = dense(attn.out_proj, z)
    # 残差在输入 dtype 内相加，保留上游的舍入边界。
    residual = (x.astype(np.float64) + attended.astype(np.float64)).astype(x.dtype)
    z = layer_norm(m.layer_norm2, residual)
    mlp = dense(m.mlp.fc2, act(dense(m.mlp.fc1, z), "quick"))
    return (residual.astype(np.float64) + mlp.astype(np.float64)).astype(x.dtype)


def flux(case, args):
    """独立重算 FluxSingleTransformerBlock：调制、joint attention、MLP 与门控。

    args 为 (weights, hidden, context, temb)；返回 (图像输出, 文本输出)。
    modulation 由 SiLU(temb) 线性映射后拆成 shift/scale/gate 三份，文本与
    图像 token 在残差维拼接；不调用上游 block。
    """
    weights, hidden, context, temb = args
    m = case.module(weights)
    residual = np.concatenate((array(context), array(hidden)), axis=1)
    dtype = residual.dtype
    z = layer_norm(m.norm.norm, residual)
    modulation = dense(m.norm.linear, act(array(temb), "silu"))
    shift, scale, gate = np.split(modulation, 3, axis=-1)
    scale = (1 + scale.astype(np.float64)).astype(dtype)
    z = (z.astype(np.float64) * scale[:, None, :].astype(np.float64)).astype(dtype)
    z = (z.astype(np.float64) + shift[:, None, :].astype(np.float64)).astype(dtype)
    mlp = act(dense(m.proj_mlp, z), "gelu")
    attn = m.attn
    shape = (*z.shape[:2], attn.heads, attn.head_dim)
    q, k, v = (dense(layer, z).reshape(shape) for layer in (attn.to_q, attn.to_k, attn.to_v))
    q = norm(attn.norm_q, q, attn.norm_q.scale).astype(np.float32)
    k = norm(attn.norm_k, k, attn.norm_k.scale).astype(np.float32)
    attended = attention(q, k, v, attn.head_dim**-0.5).reshape(z.shape).astype(np.float32)
    projected = dense(m.proj_out, np.concatenate((attended, mlp.astype(np.float32)), axis=-1))
    gated = (gate[:, None, :].astype(np.float64) * projected.astype(np.float64)).astype(dtype)
    output = (residual.astype(np.float64) + gated.astype(np.float64)).astype(dtype)
    text_length = context.shape[1]
    return output[:, :text_length, :], output[:, text_length:, :]


def flux_attention(case, args):
    """独立重算 FluxAttention joint 分支，返回 (图像输出, 文本输出)。

    args 为 (weights, hidden, context)。文本与图像分别投影、各自做 QK
    norm，拼接后执行非 causal SDPA；最后分别经 to_out 与 to_add_out。
    """
    weights, hidden, context = args
    m = case.module(weights)
    hidden, context = array(hidden), array(context)
    shape = (*hidden.shape[:2], m.heads, m.head_dim)
    eshape = (*context.shape[:2], m.heads, m.head_dim)
    q, k, v = (dense(layer, hidden).reshape(shape) for layer in (m.to_q, m.to_k, m.to_v))
    eq, ek, ev = (
        dense(layer, context).reshape(eshape)
        for layer in (m.add_q_proj, m.add_k_proj, m.add_v_proj)
    )
    q = norm(m.norm_q, q, m.norm_q.scale).astype(np.float32)
    k = norm(m.norm_k, k, m.norm_k.scale).astype(np.float32)
    eq = norm(m.norm_added_q, eq, m.norm_added_q.scale).astype(np.float32)
    ek = norm(m.norm_added_k, ek, m.norm_added_k.scale).astype(np.float32)
    # 文本 token 在前、图像 token 在后，与上游拼接顺序一致。
    q, k, v = (np.concatenate((a, b), axis=1) for a, b in ((eq, q), (ek, k), (ev, v)))
    out = attention(q, k, v, m.head_dim**-0.5).astype(np.float32)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    text_len = context.shape[1]
    return dense(m.to_out[0], out[:, text_len:, :]), dense(m.to_add_out, out[:, :text_len, :])


def moe(args):
    """按 token/专家逐项计算原 EPMoE，不复用其分发或 GMM 内核。

    args 为 (weights, hidden, gate, indices)；weights 叶子顺序为
    (wi_0, wi_1, wo)，indices 为每个 token 的专家编号。返回 [tokens, dim]
    输出。循环显式展开 top-k 与加权求和，便于独立核对 CPU interpret 分支。
    """
    weights, hidden, gate, indices = args
    w0, w1, wo = (array(w) for w in weights)
    x, routing, ids = array(hidden), array(gate), array(indices)
    dtype = x.dtype
    out = np.zeros_like(x)
    for i in range(len(x)):
        total = np.zeros(x.shape[-1], np.float64)
        for j, expert in enumerate(ids[i]):
            # 每个专家的两路投影、SiLU 门控与输出投影都按真实 dtype 分步舍入。
            a = (x[i].astype(np.float64) @ w0[expert].astype(np.float64)).astype(dtype)
            b = (x[i].astype(np.float64) @ w1[expert].astype(np.float64)).astype(dtype)
            h = (act(a, "silu").astype(np.float64) * b.astype(np.float64)).astype(dtype)
            y = (h.astype(np.float64) @ wo[expert].astype(np.float64)).astype(dtype)
            total += y.astype(np.float64) * float(routing[i, j])
        out[i] = total
    return out
