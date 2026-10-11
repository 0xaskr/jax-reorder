"""完整层的声明配置及显式参数、cache、位置与 mask 输入。"""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .upstream import block_classes, classes


@dataclass(frozen=True)
class Profile:
    name: str
    dtype: object
    mode: str
    heads: int
    tokens: int
    qk_norm: bool = False
    indexer: bool = False


PROFILES = (
    Profile("prefill-f32", jnp.float32, "EXTEND", 2, 3),
    Profile("prefill-bf16", jnp.bfloat16, "EXTEND", 2, 2, True),
    Profile("decode-f32", jnp.float32, "DECODE", 2, 1, True),
    Profile("decode-bf16", jnp.bfloat16, "DECODE", 2, 1),
)


class Glm5Case:
    """原始构造函数固定的 2048/512/256/64 维度均保留。

    仅缩小合法的 hidden、head 和 token 配置；不替换算子、数值核心或 KV
    更新方法。权重使用纯数组 tuple 输入，避免依赖 Flax 状态包装的序列化。
    """

    def __init__(self, profile, mesh):
        self.profile, self.mesh = profile, mesh
        c = classes()
        self.model = c.Glm5Attention(
            hidden_size=8,
            num_heads=profile.heads,
            num_kv_heads=profile.heads,
            max_position_embeddings=64,
            mesh=mesh,
            rms_norm_eps=1e-6,
            use_qk_norm=profile.qk_norm,
            dtype=profile.dtype,
            use_absorbed=False,
            has_indexer=profile.indexer,
        )
        self.graphdef, params, self.rest = nnx.split(self.model, nnx.Param, ...)
        self.weights, self.param_tree = jax.tree.flatten(params)
        self.pool = c.KVPool(8, 2, profile.dtype, profile.heads, 256, 1, mesh)
        _, self.pool_meta = self.pool.tree_flatten()
        self.backend = c.NativeAttention(profile.heads, profile.heads, mesh)

    def module(self, weights):
        return nnx.merge(self.graphdef, self.param_tree.unflatten(weights), self.rest)

    def function(self, candidate=None):
        """包装只准备原始 ForwardBatch/KVPool 元数据；原方法负责全部计算。"""
        c = classes()

        def run(weights, hidden, positions, cache, lengths, locations, writes, prefix, extends):
            model = self.module(weights)
            pool = c.KVPool.tree_unflatten(self.pool_meta, ([cache],))
            batch = c.ForwardBatch(
                bid=0,
                forward_mode=getattr(c.ForwardMode, self.profile.mode),
                batch_size=1,
                input_ids=None,
                req_pool_indices=None,
                seq_lens=lengths,
                out_cache_loc=writes,
                positions=positions,
                attn_backend=self.backend,
                cache_loc=locations,
                extend_prefix_lens=prefix,
                extend_seq_lens=extends,
            )
            if candidate is None:
                return model(positions, hidden, batch, pool)
            return candidate(model, positions, hidden, batch, pool)

        return run

    def sample(self, seed):
        rng = np.random.default_rng(6300 + seed)
        p = self.profile

        def array(raw, dtype=p.dtype, sharding=None):
            return jax.device_put(
                jnp.asarray(raw, dtype), sharding or NamedSharding(self.mesh, P())
            )

        weights = tuple(
            array(
                rng.normal(0, 0.06, w.shape) + (0.9 if w.ndim == 1 else 0),
                w.dtype,
                w.sharding,
            )
            for w in self.weights
        )
        hidden = rng.normal(0, 0.7, (p.tokens, 8))
        if seed == 0:
            hidden[0] = 0
        cache = rng.normal(0, 0.2, self.pool.kv_buffer[0].shape)
        # 0 号 cache 页保留；有效 token 地址从 2 开始。包含一个无效读取槽，
        # prefill 使用真实 prefix，decode 保留两个历史 token 后写入当前 token。
        prefix = 1 if p.mode == "EXTEND" else 2
        length = prefix + p.tokens
        positions = np.arange(prefix, length, dtype=np.int32) + seed * 2
        locations = np.array([*range(2, 2 + length), 0], np.int32)
        writes = np.arange(2 + prefix, 2 + length, dtype=np.int32)
        return (
            weights,
            array(hidden),
            array(positions, jnp.int32),
            array(cache, sharding=self.pool.kv_buffer[0].sharding),
            array([length], jnp.int32),
            array(locations, jnp.int32),
            array(writes, jnp.int32),
            array([prefix], jnp.int32),
            array([p.tokens], jnp.int32),
        )


class LlamaCase(Glm5Case):
    """完整 GQA：4 个 Q heads、2 个 KV heads，保留 128 对齐的实际 KV 后端。"""

    def __init__(self, profile, mesh):
        self.profile, self.mesh = profile, mesh
        c = classes()
        self.model = c.LlamaAttention(
            hidden_size=8,
            num_heads=4,
            num_kv_heads=2,
            mesh=mesh,
            head_dim=128,
            rope_is_neox_style=False,
            max_position_embeddings=64,
            dtype=profile.dtype,
            attention_bias=True,
        )
        self.graphdef, params, self.rest = nnx.split(self.model, nnx.Param, ...)
        self.weights, self.param_tree = jax.tree.flatten(params)
        self.pool = c.KVPool(8, 2, profile.dtype, 2, 128, 1, mesh)
        _, self.pool_meta = self.pool.tree_flatten()
        self.backend = c.NativeAttention(4, 2, mesh)


@dataclass(frozen=True)
class BlockProfile:
    name: str
    dtype: object
    tokens: int
    causal: bool


BLOCK_PROFILES = (
    BlockProfile("masked-f32", jnp.float32, 3, False),
    BlockProfile("causal-f32", jnp.float32, 2, True),
    BlockProfile("masked-bf16", jnp.bfloat16, 3, False),
    BlockProfile("causal-bf16", jnp.bfloat16, 2, True),
)

FLUX_PROFILES = (
    BlockProfile("tokens3-f32", jnp.float32, 3, False),
    BlockProfile("tokens2-f32", jnp.float32, 2, False),
    BlockProfile("tokens3-bf16", jnp.bfloat16, 3, False),
    BlockProfile("tokens2-bf16", jnp.bfloat16, 2, False),
)


class BlockCase:
    """完整视觉 encoder 或 diffusion transformer，状态参数均为显式数组。"""

    def __init__(self, profile, mesh, family):
        self.profile, self.mesh, self.family = profile, mesh, family
        c = block_classes()
        if family == "clip":
            config = SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                intermediate_size=12,
                attention_dropout=0.0,
                layer_norm_eps=1e-5,
                hidden_act="quick_gelu",
            )
            self.model = c.CLIPEncoderLayer(config, mesh, dtype=profile.dtype)
        elif family == "flux_attention":
            self.model = c.FluxAttention(
                query_dim=8,
                heads=2,
                dim_head=4,
                bias=True,
                added_kv_proj_dim=8,
                attention_impl="sdpa",
                params_dtype=profile.dtype,
                mesh=mesh,
                rngs=nnx.Rngs(31),
            )
        elif family == "flux":
            self.model = c.FluxSingleTransformerBlock(
                dim=8,
                num_attention_heads=2,
                attention_head_dim=4,
                mesh=mesh,
                mlp_ratio=1.5,
                attention_impl="sdpa",
                params_dtype=profile.dtype,
                rngs=nnx.Rngs(31),
            )
        else:
            raise ValueError(family)
        self.graphdef, params, self.rest = nnx.split(self.model, nnx.Param, ...)
        self.weights, self.param_tree = jax.tree.flatten(params)

    def module(self, weights):
        return nnx.merge(self.graphdef, self.param_tree.unflatten(weights), self.rest)

    def function(self, candidate=None):
        def run(weights, *inputs):
            model = self.module(weights)
            return model(*inputs) if candidate is None else candidate(model, *inputs)

        return run

    def sample(self, seed):
        rng = np.random.default_rng(6400 + seed)

        def put(x, dtype, sharding=None):
            return jax.device_put(jnp.asarray(x, dtype), sharding or NamedSharding(self.mesh, P()))

        weights = tuple(
            put(rng.normal(0, 0.15, w.shape) + (0.8 if w.ndim == 1 else 0), w.dtype, w.sharding)
            for w in self.weights
        )
        x = rng.normal(0, 0.7, (2, self.profile.tokens, 8))
        if seed == 0:
            x[0, 0] = 0
        hidden = put(x, self.profile.dtype)
        if self.family == "clip":
            mask = np.ones((2, self.profile.tokens), np.int32)
            mask[0, -1] = 0
            return weights, hidden, None if self.profile.causal else put(mask, jnp.int32)
        context = put(rng.normal(0, 0.4, (2, 2, 8)), self.profile.dtype)
        if self.family == "flux_attention":
            return weights, hidden, context
        temb = put(rng.normal(0, 0.6, (2, 8)), self.profile.dtype)
        return weights, hidden, context, temb
