"""AI 直接手写的完整 GLM5 MHA 候选，不使用脚本或 AST 生成改写。"""

import jax
import jax.numpy as jnp

from .upstream import block_classes


def glm5_kv_first(m, positions, hidden, batch, pool):
    """先完成独立 KV 投影；Q 路径、RoPE、原 attention 后端与 cache 全部保留。"""
    latent, _ = m.kv_a_proj_with_mqa(hidden)
    compressed, k_rope = jnp.split(latent, [m.kv_lora_rank], axis=-1)
    compressed = m.kv_a_layernorm(compressed)
    k_rope = k_rope.reshape(-1, 1, m.qk_rope_head_dim)
    q_compressed, _ = m.q_a_proj(hidden)
    q_compressed = m.q_a_layernorm(q_compressed)
    q, _ = m.q_b_proj(q_compressed)
    q = q.reshape(-1, m.num_heads, m.qk_head_dim)
    if m.indexer is not None:
        _ = m.indexer(hidden, q_compressed, positions, m.rotary_emb)
    q_nope = q[:, :, : m.qk_nope_head_dim]
    q_rope = q[:, :, m.qk_nope_head_dim :]
    q_rope, k_rope = m.rotary_emb(positions, q_rope, k_rope)
    attn, cache = m._forward_mha(q_nope, q_rope, compressed, k_rope, batch, pool)
    output, _ = m.o_proj(attn)
    return output, cache


def glm5_interleaved(m, positions, hidden, batch, pool):
    """交错计算两个投影和两个归一化；每个算子的操作数顺序不变。"""
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
    """负例：错误地将所有 RoPE 位置设为零，改变输出与持久 cache。"""
    return m(jnp.zeros_like(positions), hidden, batch, pool)


def glm5_wrong_cache(m, positions, hidden, batch, pool):
    """负例：返回更新前的 cache；即使当前激活相同，状态契约也不满足。"""
    output, _ = m(positions, hidden, batch, pool)
    return output, pool.get_fused_kv_buffer(0)


GLM5_CASES = (
    (glm5_kv_first, "equivalent", "独立 KV 路径提前"),
    (glm5_interleaved, "equivalent", "独立 Q/KV 计算跨深度交错"),
    (glm5_wrong_position, "different", "RoPE 位置改变"),
    (glm5_wrong_cache, "different", "遗漏持久 KV cache 更新"),
)


def glm5_drop_indexer(m, positions, hidden, batch, pool):
    """图差异反例：删除未参与输出的完整索引器，数值输出仍可相同。"""
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


LLAMA_CASES = (
    (llama_kv_first, "equivalent", "GQA 的 K/V 投影及 reshape 提前"),
    (glm5_wrong_position, "different", "改变 RoPE 位置"),
    (glm5_wrong_cache, "different", "未返回新 KV 状态"),
)


def clip_attention_kv_first(m, x, mask):
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
    attended = clip_attention_kv_first(m.self_attn, m.layer_norm1(x), mask)
    residual = x + attended
    return residual + m.mlp(m.layer_norm2(residual))


def clip_wrong_residual(m, x, mask):
    attended = m.self_attn(m.layer_norm1(x), mask, True)
    return x + m.mlp(m.layer_norm2(attended))


CLIP_CASES = (
    (clip_reordered, "equivalent", "完整 encoder 内 Q/K/V 独立投影重排"),
    (clip_wrong_residual, "different", "遗漏 attention 残差边"),
)


def flux_attention_first(m, hidden, context, temb):
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
    # 反例只改调制输入，不改变接口和输出树。
    return m(hidden, context, jnp.zeros_like(temb))


FLUX_CASES = (
    (flux_attention_first, "equivalent", "完整 diffusion block 内 Attention 与 MLP 换序"),
    (flux_wrong_gate, "different", "遗漏 timestep/text 调制信号"),
)


def flux_context_first(m, hidden, context):
    """Joint Attention：先算独立文本分支，保留两种输出及所有 QK norm。"""
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
    return m(hidden, jnp.zeros_like(context))


FLUX_ATTENTION_CASES = (
    (flux_context_first, "equivalent", "文本与图像投影独立换序"),
    (flux_wrong_context, "different", "错误地抹掉文本上下文"),
)
