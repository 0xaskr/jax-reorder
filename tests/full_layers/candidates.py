"""AI 直接手写的完整层候选，不使用脚本或 AST 生成改写。

按原始类分为 GLM5 MHA、Llama GQA、CLIP encoder 与 Flux joint attention 四组；
每组用 *_CASES 保存 (候选, 人工预判, 理由)。正例只移动独立运算并保留算子
参数与操作数位置；负例改变位置、cache、残差、mask 或调制输入。预期标签
在比较前固定，候选不读取比较器输出。完整 Flux block 的原图追踪在锁定
JAX 上被拒绝，其 FLUX_CASES 只作源码记录，不计入等价覆盖。
"""

import jax
import jax.numpy as jnp

from .upstream import block_classes


def glm5_kv_first(m, positions, hidden, batch, pool):
    """先完成独立 KV 投影；Q 路径、RoPE、原 attention 后端与 cache 全部保留。

    m/positions/hidden/batch/pool 与上游 Glm5Attention.forward 相同；返回
    (output, cache) 二元组。索引器（若存在）在投影后按原位置调用，其结果
    虽不参与主输出，但调用本身必须保留以匹配完整图。该候选只重排独立定义。
    """
    latent, _ = m.kv_a_proj_with_mqa(hidden)
    compressed, k_rope = jnp.split(latent, [m.kv_lora_rank], axis=-1)
    compressed = m.kv_a_layernorm(compressed)
    k_rope = k_rope.reshape(-1, 1, m.qk_rope_head_dim)
    q_compressed, _ = m.q_a_proj(hidden)
    q_compressed = m.q_a_layernorm(q_compressed)
    q, _ = m.q_b_proj(q_compressed)
    q = q.reshape(-1, m.num_heads, m.qk_head_dim)
    if m.indexer is not None:
        # 索引器结果被丢弃，但调用必须保留：完整图不做无根子图删除。
        _ = m.indexer(hidden, q_compressed, positions, m.rotary_emb)
    q_nope = q[:, :, : m.qk_nope_head_dim]
    q_rope = q[:, :, m.qk_nope_head_dim :]
    q_rope, k_rope = m.rotary_emb(positions, q_rope, k_rope)
    attn, cache = m._forward_mha(q_nope, q_rope, compressed, k_rope, batch, pool)
    output, _ = m.o_proj(attn)
    return output, cache


def glm5_interleaved(m, positions, hidden, batch, pool):
    """交错计算两个投影和两个归一化；每个算子的操作数顺序不变。

    输入输出与 glm5_kv_first 相同；Q 与 KV 的独立投影/归一化跨深度交错，
    形成实际调度换序，同时保持每个调用的参数绑定和 RoPE 位置处理。
    """
    q_compressed, _ = m.q_a_proj(hidden)
    latent, _ = m.kv_a_proj_with_mqa(hidden)
    compressed, k_rope = jnp.split(latent, [m.kv_lora_rank], axis=-1)
    q_compressed = m.q_a_layernorm(q_compressed)
    compressed = m.kv_a_layernorm(compressed)
    q, _ = m.q_b_proj(q_compressed)
    k_rope = k_rope.reshape(-1, 1, m.qk_rope_head_dim)
    q = q.reshape(-1, m.num_heads, m.qk_head_dim)
    if m.indexer is not None:
        _ = m.indexer(hidden, q_compressed, positions, m.rotary_emb)
    q_nope = q[:, :, : m.qk_nope_head_dim]
    q_rope = q[:, :, m.qk_nope_head_dim :]
    q_rope, k_rope = m.rotary_emb(positions, q_rope, k_rope)
    attn, cache = m._forward_mha(q_nope, q_rope, compressed, k_rope, batch, pool)
    output, _ = m.o_proj(attn)
    return output, cache


def glm5_wrong_position(m, positions, hidden, batch, pool):
    """负例：错误地将所有 RoPE 位置设为零，改变输出与持久 cache。

    只替换位置输入，不改变接口；用于确认比较器能定位到位置依赖。
    """
    return m(jnp.zeros_like(positions), hidden, batch, pool)


def glm5_wrong_cache(m, positions, hidden, batch, pool):
    """负例：返回更新前的 cache；即使当前激活相同，状态契约也不满足。

    输出元组的第二个分量来自 pool 的旧内容，报告应定位到 cache 端口。
    """
    output, _ = m(positions, hidden, batch, pool)
    return output, pool.get_fused_kv_buffer(0)


# (候选, 预期结论, 理由)；预期在写候选时人工固定，不读取比较器输出。
GLM5_CASES = (
    (glm5_kv_first, "equivalent", "独立 KV 路径提前"),
    (glm5_interleaved, "equivalent", "独立 Q/KV 计算跨深度交错"),
    (glm5_wrong_position, "different", "RoPE 位置改变"),
    (glm5_wrong_cache, "different", "遗漏持久 KV cache 更新"),
)


def glm5_drop_indexer(m, positions, hidden, batch, pool):
    """图差异反例：删除未参与输出的完整索引器，数值输出仍可相同。

    其余计算与 glm5_kv_first 完全一致，只省略索引器调用；该函数只在
    evidence.py 的索引器证据中使用，属于“图不同但数值一致”的独立一列。
    """
    latent, _ = m.kv_a_proj_with_mqa(hidden)
    compressed, k_rope = jnp.split(latent, [m.kv_lora_rank], axis=-1)
    compressed = m.kv_a_layernorm(compressed)
    k_rope = k_rope.reshape(-1, 1, m.qk_rope_head_dim)
    q_compressed, _ = m.q_a_proj(hidden)
    q_compressed = m.q_a_layernorm(q_compressed)
    q, _ = m.q_b_proj(q_compressed)
    q = q.reshape(-1, m.num_heads, m.qk_head_dim)
    q_nope = q[:, :, : m.qk_nope_head_dim]
    q_rope = q[:, :, m.qk_nope_head_dim :]
    q_rope, k_rope = m.rotary_emb(positions, q_rope, k_rope)
    attn, cache = m._forward_mha(q_nope, q_rope, compressed, k_rope, batch, pool)
    output, _ = m.o_proj(attn)
    return output, cache


def llama_kv_first(m, positions, hidden, batch, pool):
    """GQA 正例：K/V 投影与 reshape 提前，Q 路径及其余调用保持原顺序。

    输入输出与 LlamaAttention.forward 相同；返回 (output, cache)。两个
    KV 投影独立于 Q，提前执行只改变调度，不改变操作数或分组。
    """
    k, _ = m.k_proj(hidden)
    v, _ = m.v_proj(hidden)
    k = k.reshape(-1, m.kv_head_num, m.head_dim)
    v = v.reshape(-1, m.kv_head_num, m.head_dim)
    q, _ = m.q_proj(hidden)
    q = q.reshape(-1, m.q_head_num, m.head_dim)
    q, k = m.rotary_emb(positions, q, k)
    attn, cache = m.attn(q, k, v, forward_batch=batch, token_to_kv_pool=pool)
    output, _ = m.o_proj(attn)
    return output, cache


# 负例复用 GLM5 的错误函数：LlamaAttention 的调用签名与 Glm5Attention 相同，
# 位置错误和旧 cache 返回仍是同一类状态契约违背。
LLAMA_CASES = (
    (llama_kv_first, "equivalent", "GQA 的 K/V 投影及 reshape 提前"),
    (glm5_wrong_position, "different", "改变 RoPE 位置"),
    (glm5_wrong_cache, "different", "未返回新 KV 状态"),
)


def clip_attention_kv_first(m, x, mask):
    """正例：展开 CLIPAttention 前向，只把 Q/K/V 独立投影提前。

    m 为原始 CLIPAttention，x 为 [B,L,E]，mask 可能为 None；返回
    [B,L,E] 的 attention 输出。内部保持原投影、缩放、mask、softmax、
    dropout（推理模式）与 out_proj 的数值边界，不调用原 forward。
    """
    b, length, _ = x.shape
    k, _ = m.k_proj(x)
    v, _ = m.v_proj(x)
    q, _ = m.q_proj(x)
    k = jnp.transpose(k.reshape(b, length, m.num_heads, m.head_dim), (0, 2, 1, 3))
    v = jnp.transpose(v.reshape(b, length, m.num_heads, m.head_dim), (0, 2, 1, 3))
    q = jnp.transpose(q.reshape(b, length, m.num_heads, m.head_dim), (0, 2, 1, 3))
    scores = (
        jnp.matmul(q.astype(jnp.float32), jnp.transpose(k.astype(jnp.float32), (0, 1, 3, 2)))
        * m.scale
    )
    if mask is None:
        lower = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
        scores = scores + jnp.where(lower, 0.0, jnp.finfo(scores.dtype).min)
    else:
        mask = jnp.expand_dims(mask, axis=(1, 2)) if mask.ndim == 2 else mask
        scores = scores + jnp.where(mask > 0, 0.0, jnp.finfo(scores.dtype).min)
    weights = m.dropout(jax.nn.softmax(scores, axis=-1), deterministic=True)
    out = jnp.matmul(weights, v.astype(jnp.float32))
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, length, m.embed_dim)
    return m.out_proj(out.astype(x.dtype))[0]


def clip_reordered(m, x, mask):
    """正例：完整 CLIPEncoderLayer 内 Q/K/V 独立投影重排，残差结构不变。

    m 为原始 encoder 层，x 为 [B,L,E]；返回与上游 forward 相同的张量。
    """
    attended = clip_attention_kv_first(m.self_attn, m.layer_norm1(x), mask)
    residual = x + attended
    return residual + m.mlp(m.layer_norm2(residual))


def clip_wrong_residual(m, x, mask):
    """负例：遗漏 attention 残差边，直接对 attended 做归一化与 MLP。"""
    attended = m.self_attn(m.layer_norm1(x), mask, True)
    return x + m.mlp(m.layer_norm2(attended))


# (候选, 预期结论, 理由)；正例要求原方程次序确实改变，负例必须有数值见证。
CLIP_CASES = (
    (clip_reordered, "equivalent", "完整 encoder 内 Q/K/V 独立投影重排"),
    (clip_wrong_residual, "different", "遗漏 attention 残差边"),
)


def flux_attention_first(m, hidden, context, temb):
    """完整 diffusion block 的重排：Attention 与 MLP 消费同一归一化输出。

    输入与 FluxSingleTransformerBlock.forward 一致；返回 (图像输出, 文本
    输出)。两条独立分支换序，gate、proj_out 与残差相加保持原分组。完整
    block 的原图追踪在锁定 JAX 上被拒绝，本组只作源码记录，不进入 runner。
    """
    text_len = context.shape[1]
    residual = jnp.concatenate([context, hidden], axis=1)
    normed, gate = m.norm(residual, emb=temb)
    # MLP 与 Attention 消费同一归一化输出，两条分支互不依赖。
    attention = m.attn(hidden_states=normed, image_rotary_emb=None, req=None)
    mlp, _ = m.proj_mlp(normed)
    mlp = m.act_mlp(mlp)
    joined = jnp.concatenate([attention, mlp], axis=-1)
    gate = gate[:, None, :]
    out, _ = m.proj_out(joined)
    out = residual + gate * out
    return out[:, :text_len, :], out[:, text_len:, :]


def flux_wrong_gate(m, hidden, context, temb):
    """负例：把调制输入清零，接口与输出树不变但 timestep 条件被抹掉。"""
    # 反例只改调制输入，不改变接口和输出树。
    return m(hidden, context, jnp.zeros_like(temb))


# (候选, 预期结论, 理由)；完整 Flux block 尚未得到可比较原图，故不在 FAMILIES 中。
FLUX_CASES = (
    (flux_attention_first, "equivalent", "完整 diffusion block 内 Attention 与 MLP 换序"),
    (flux_wrong_gate, "different", "遗漏 timestep/text 调制信号"),
)


def flux_context_first(m, hidden, context):
    """Joint Attention：先算独立文本分支，保留两种输出及所有 QK norm。

    m 为原始 FluxAttention，hidden 为图像 token，context 为文本 token；
    返回 (图像输出, 文本输出)，顺序与上游 forward 一致。文本与图像投影
    相互独立，只提前文本分支，其余 QK norm、拼接和 sdpa 保持原样。
    """
    ek, _ = m.add_k_proj(context)
    ev, _ = m.add_v_proj(context)
    eq, _ = m.add_q_proj(context)
    k, _ = m.to_k(hidden)
    v, _ = m.to_v(hidden)
    q, _ = m.to_q(hidden)
    shape = (*hidden.shape[:2], m.heads, m.head_dim)
    eshape = (*context.shape[:2], m.heads, m.head_dim)
    q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
    q, k = m.norm_q(q), m.norm_k(k)
    eq, ek, ev = eq.reshape(eshape), ek.reshape(eshape), ev.reshape(eshape)
    eq, ek = m.norm_added_q(eq), m.norm_added_k(ek)
    # 上游 joint 注意力把文本 token 拼在图像 token 之前，划分点即 text_len。
    q = jnp.concatenate([eq, q], axis=1)
    k = jnp.concatenate([ek, k], axis=1)
    v = jnp.concatenate([ev, v], axis=1)
    out = block_classes().sdpa(q, k, v, is_causal=False)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    out = block_classes().output_sharding(out.astype(q.dtype), m.mesh)
    text_len = context.shape[1]
    encoded, image = out[:, :text_len, :], out[:, text_len:, :]
    image, _ = m.to_out[0](image)
    encoded, _ = m.to_add_out(encoded)
    return image, encoded


def flux_wrong_context(m, hidden, context):
    """负例：把文本上下文清零，保留接口但破坏 joint attention 的输入。"""
    return m(hidden, jnp.zeros_like(context))


# (候选, 预期结论, 理由)；该组正例与负例都要求实际依赖排列改变。
FLUX_ATTENTION_CASES = (
    (flux_context_first, "equivalent", "文本与图像投影独立换序"),
    (flux_wrong_context, "different", "错误地抹掉文本上下文"),
)
