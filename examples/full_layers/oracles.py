"""NumPy 完整层参照：独立计算投影、旋转、mask、softmax 和 cache 更新。"""

import numpy as np

from examples.sglang_layers.oracles import act, array, dense, norm


def rotary(x, positions, base):
    dtype = x.dtype
    frequency = np.arange(0, x.shape[-1], 2, dtype=np.float32) / x.shape[-1]
    angles = np.asarray(positions, np.float32)[:, None] * (1.0 / base**frequency)
    cos = np.cos(angles).astype(dtype)[:, None, :].astype(np.float64)
    sin = np.sin(angles).astype(dtype)[:, None, :].astype(np.float64)
    even, odd = x[..., ::2].astype(np.float64), x[..., 1::2].astype(np.float64)
    ec, es = (even * cos).astype(dtype), (even * sin).astype(dtype)
    oc, os = (odd * cos).astype(dtype), (odd * sin).astype(dtype)
    result = np.empty_like(x)
    result[..., ::2] = ec.astype(np.float64) - os.astype(np.float64)
    result[..., 1::2] = oc.astype(np.float64) + es.astype(np.float64)
    return result


def glm5(case, args):
    weights, hidden, positions, initial_cache, lengths, locations, writes, prefix, _ = args
    m = case.module(weights)
    hidden, dtype = array(hidden), array(hidden).dtype
    q = dense(m.q_a_proj, hidden)
    q = norm(m.q_a_layernorm, q, m.q_a_layernorm.scale).astype(dtype)
    q = dense(m.q_b_proj, q).reshape(-1, m.num_heads, 256)
    latent = dense(m.kv_a_proj_with_mqa, hidden)
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
    _, hidden, _, initial_cache, lengths, locations, writes, prefix, _ = args
    dtype = key.dtype
    kv_heads, dim = key.shape[1:]
    query_heads = q.shape[1]
    updated = array(initial_cache).copy()
    rows = updated.reshape(-1, kv_heads, 2, dim)
    for i, target in enumerate(array(writes)):
        if 0 <= target < len(rows):
            rows[target, :, 0] = key[i]
            rows[target, :, 1] = value[i]
    loc = array(locations)
    gathered = np.zeros((len(loc), kv_heads, 2, dim), dtype=dtype)
    for i, index in enumerate(loc):
        if 0 < index < len(rows):
            gathered[i] = rows[index]
    k, v = gathered[:, :, 0], gathered[:, :, 1]
    k = np.repeat(k, query_heads // kv_heads, axis=1)
    v = np.repeat(v, query_heads // kv_heads, axis=1)
    logits = np.einsum("qhd,khd->qhk", q.astype(np.float64), k.astype(np.float64)).astype(dtype)
    logits = (logits.astype(np.float64) * np.asarray(scaling, dtype=dtype)).astype(dtype)
    scores = logits.astype(np.float64)
    for query in range(len(hidden)):
        for column in range(len(loc)):
            valid = loc[column] > 0 and column < int(array(lengths)[0])
            if case.profile.mode == "EXTEND":
                valid = valid and column <= int(array(prefix)[0]) + query
            if not valid:
                scores[query, :, column] = -np.inf
    shifted = (scores - scores.max(axis=-1, keepdims=True)).astype(dtype).astype(np.float64)
    exps = np.exp(shifted).astype(dtype).astype(np.float64)
    denominators = exps.sum(axis=-1, keepdims=True).astype(dtype).astype(np.float64)
    probability = (exps / denominators).astype(dtype).astype(np.float64)
    attn = np.einsum("qhk,khd->qhd", probability, v.astype(np.float64)).astype(dtype)
    return attn.reshape(len(hidden), -1), updated


def llama(case, args):
    weights, hidden, positions, *_ = args
    m = case.module(weights)
    q = dense(m.q_proj, array(hidden)).reshape(-1, m.q_head_num, m.head_dim)
    k = dense(m.k_proj, array(hidden)).reshape(-1, m.kv_head_num, m.head_dim)
    v = dense(m.v_proj, array(hidden)).reshape(-1, m.kv_head_num, m.head_dim)
    q, k = rotary(q, positions, m.rope_theta), rotary(k, positions, m.rope_theta)
    attended, cache = native(case, args, q, k, v, m.scaling)
    return dense(m.o_proj, attended), cache


def layer_norm(layer, x):
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
    logits = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64)) * scale
    if mask is not None:
        logits = np.where(mask, logits, -np.inf)
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probability = exponent / exponent.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", probability, v.astype(np.float64))


def clip(case, args):
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
    residual = (x.astype(np.float64) + attended.astype(np.float64)).astype(x.dtype)
    z = layer_norm(m.layer_norm2, residual)
    mlp = dense(m.mlp.fc2, act(dense(m.mlp.fc1, z), "quick"))
    return (residual.astype(np.float64) + mlp.astype(np.float64)).astype(x.dtype)


def flux(case, args):
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
    q, k, v = (np.concatenate((a, b), axis=1) for a, b in ((eq, q), (ek, k), (ev, v)))
    out = attention(q, k, v, m.head_dim**-0.5).astype(np.float32)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    text_len = context.shape[1]
    return dense(m.to_out[0], out[:, text_len:, :]), dense(m.to_add_out, out[:, :text_len, :])


def moe(args):
    """按 token/专家逐项计算原 EPMoE，不复用其分发或 GMM 内核。"""
    weights, hidden, gate, indices = args
    w0, w1, wo = (array(w) for w in weights)
    x, routing, ids = array(hidden), array(gate), array(indices)
    dtype = x.dtype
    out = np.zeros_like(x)
    for i in range(len(x)):
        total = np.zeros(x.shape[-1], np.float64)
        for j, expert in enumerate(ids[i]):
            a = (x[i].astype(np.float64) @ w0[expert].astype(np.float64)).astype(dtype)
            b = (x[i].astype(np.float64) @ w1[expert].astype(np.float64)).astype(dtype)
            h = (act(a, "silu").astype(np.float64) * b.astype(np.float64)).astype(dtype)
            y = (h.astype(np.float64) @ wo[expert].astype(np.float64)).astype(dtype)
            total += y.astype(np.float64) * float(routing[i, j])
        out[i] = total
    return out
