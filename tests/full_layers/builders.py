"""完整层的声明配置及显式参数、cache、位置与 mask 输入。

Profile/BlockProfile 固定 dtype、前向模式与合法小尺寸；权重取自上游构造
函数的真实参数形状，但以普通数组 tuple 作为动态输入，不依赖 Flax 状态
序列化。sample() 为每个种子生成独立非对称权重，避免上游同一 PRNGKey 让
gate/up 等投影偶然相等。本模块只准备模型和输入，不执行图比较，也不改写
上游计算。
"""

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
    """GLM5/Llama 的单次前向配置：名称、精度、模式、head 数与 token 数。

    qk_norm 与 indexer 只切换构造开关；部分开关在锁定源码的前向路径中
    并不生效，报告里会按实际行为单独说明，不能据此宣称两种算法都验证过。
    """

    name: str
    dtype: object
    mode: str
    heads: int
    tokens: int
    qk_norm: bool = False
    indexer: bool = False


# prefill 至少 2 个 token 才有 prefix/mask 行为；decode 固定 1 个 token。
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
        """构造原始 Glm5Attention、配套 KV pool 与 attention 后端。

        profile 决定 dtype/模式/head 数；mesh 为显式单设备 data/tensor
        mesh。构造期间分配的权重随后被拆成动态参数，不进入计算图常量。
        """
        self.profile, self.mesh = profile, mesh
        c = classes()
        self.model = c.Glm5Attention(
            # 小 hidden 只压缩尺寸；LoRA/head/rope 维度沿用上游构造默认值。
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
        # 参数作为动态输入，rest 保存非参数静态状态；两者分开以便逐样本替换权重。
        self.graphdef, params, self.rest = nnx.split(self.model, nnx.Param, ...)
        self.weights, self.param_tree = jax.tree.flatten(params)
        # 构造参数依次为 size=8、page_size=2、dtype、head_num、head_dim=256、layer_num=1、mesh。
        self.pool = c.KVPool(8, 2, profile.dtype, profile.heads, 256, 1, mesh)
        _, self.pool_meta = self.pool.tree_flatten()
        # MHA 配置下 Q/KV head 数相同，后端保留原始 NativeAttention 调用。
        self.backend = c.NativeAttention(profile.heads, profile.heads, mesh)

    def module(self, weights):
        """用给定动态权重重建 NNX 模块；不修改 self.model 或共享状态。"""
        return nnx.merge(self.graphdef, self.param_tree.unflatten(weights), self.rest)

    def function(self, candidate=None):
        """包装只准备原始 ForwardBatch/KVPool 元数据；原方法负责全部计算。

        candidate 为 None 时调用上游 model，否则调用 AI 候选并阻止其读取
        额外状态。返回的 run 接收 (weights, hidden, positions, cache,
        lengths, locations, writes, prefix, extends) 九个动态输入。
        """
        c = classes()

        def run(weights, hidden, positions, cache, lengths, locations, writes, prefix, extends):
            """重建模块/pool/batch，再按 candidate 选择实际执行路径。"""
            model = self.module(weights)
            pool = c.KVPool.tree_unflatten(self.pool_meta, ([cache],))
            batch = c.ForwardBatch(
                # 单请求、单 batch；extend/decode 分别由 prefix 与 lengths 表达。
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
        """按 seed 生成一组独立权重与动态输入。

        返回调用 run 所需的九元组。seed=0 把首个 hidden 行置零，覆盖零激活；
        权重每叶独立采样并给 1 维权重加 0.9 偏移，避免 gate/up 等偏置全零。
        cache 保留 0 号无效页，地址从 2 开始，prefill 使用真实 prefix。
        """
        rng = np.random.default_rng(6300 + seed)
        p = self.profile

        def array(raw, dtype=p.dtype, sharding=None):
            """把 NumPy 数据放到 mesh 上，默认使用完全复制的空 PartitionSpec。"""
            return jax.device_put(
                jnp.asarray(raw, dtype), sharding or NamedSharding(self.mesh, P())
            )

        weights = tuple(
            array(
                # 1 维参数（偏置/scale）加常量偏移，避免与零激活同时消去信号。
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
        # locations 末尾追加 0，覆盖“读取到保留页”的无效槽路径。
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
        """构造原始 LlamaAttention 及匹配的 GQA KV pool 与后端。

        head_dim=128 保持原后端对齐要求，仅缩小 hidden 与 token；权重、
        输入和 forward 包装复用 Glm5Case 的 sample/function 逻辑。
        """
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
        # 与 GQA 配置一致：head_num=2（KV heads）、head_dim=128、单层。
        self.pool = c.KVPool(8, 2, profile.dtype, 2, 128, 1, mesh)
        _, self.pool_meta = self.pool.tree_flatten()
        self.backend = c.NativeAttention(4, 2, mesh)


@dataclass(frozen=True)
class BlockProfile:
    """CLIP/Flux block 的配置：名称、精度、token 数与是否走 causal mask。"""

    name: str
    dtype: object
    tokens: int
    causal: bool


# causal=True 时向 CLIP 传 None mask，走原 forward 的 tril 因果分支；
# 否则传显式有效 mask，覆盖 masked 路径。
BLOCK_PROFILES = (
    BlockProfile("masked-f32", jnp.float32, 3, False),
    BlockProfile("causal-f32", jnp.float32, 2, True),
    BlockProfile("masked-bf16", jnp.bfloat16, 3, False),
    BlockProfile("causal-bf16", jnp.bfloat16, 2, True),
)

# Flux 只走非 causal 的 sdpa 路径；两种 token 数分别覆盖文本/图像拆分。
FLUX_PROFILES = (
    BlockProfile("tokens3-f32", jnp.float32, 3, False),
    BlockProfile("tokens2-f32", jnp.float32, 2, False),
    BlockProfile("tokens3-bf16", jnp.bfloat16, 3, False),
    BlockProfile("tokens2-bf16", jnp.bfloat16, 2, False),
)


class BlockCase:
    """完整视觉 encoder 或 diffusion transformer，状态参数均为显式数组。

    family 取 "clip"、"flux_attention" 或 "flux"（完整 block）。后者在
    锁定 JAX 上原图追踪被拒绝，只用于边界记录；不替换任何算子。
    """

    def __init__(self, profile, mesh, family):
        """按 family 构造原始 CLIPEncoderLayer、FluxAttention 或 Flux block。

        配置只缩小 hidden/head/token；dropout 固定为推理模式。构造用固定
        RNG 仅生成参数形状，随后权重被拆成动态输入并由 sample 替换。
        """
        self.profile, self.mesh, self.family = profile, mesh, family
        c = block_classes()
        if family == "clip":
            config = SimpleNamespace(
                # 与 GLM5 相同的小 hidden；2 个 head，head_dim=4。
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
                # 固定纯 JAX 的 sdpa 实现，不启用 USP/Flash/Pallas。
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
        # 参数作为动态输入，rest 保存静态结构与状态；两者分开以便重放。
        self.graphdef, params, self.rest = nnx.split(self.model, nnx.Param, ...)
        self.weights, self.param_tree = jax.tree.flatten(params)

    def module(self, weights):
        """用给定动态权重重建 NNX 模块；不修改共享 graphdef/rest。"""
        return nnx.merge(self.graphdef, self.param_tree.unflatten(weights), self.rest)

    def function(self, candidate=None):
        """返回接收 (weights, *inputs) 的包装；candidate=None 时调用原模块。

        inputs 的顺序和数量由 family 决定：CLIP 为 (hidden, mask)，
        FluxAttention 为 (hidden, context)，完整 Flux block 多一个 temb。
        """

        def run(weights, *inputs):
            """重建模块后选择原始前向或 AI 候选。"""
            model = self.module(weights)
            return model(*inputs) if candidate is None else candidate(model, *inputs)

        return run

    def sample(self, seed):
        """按 seed 生成权重与 family 对应的输入。

        seed=0 把首行激活置零；给 1 维权重加 0.8 偏移以保留偏置信号。
        CLIP 的 mask 含一个无效位置，Flux 的 context 为两个文本 token，
        完整 block 另加 timestep 调制向量 temb。
        """
        rng = np.random.default_rng(6400 + seed)

        def put(x, dtype, sharding=None):
            """把 NumPy 数据放到 mesh 上；默认完全复制。"""
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
            # 显式 mask 的最后一个位置无效，覆盖 masked 与无效槽路径。
            mask = np.ones((2, self.profile.tokens), np.int32)
            mask[0, -1] = 0
            return weights, hidden, None if self.profile.causal else put(mask, jnp.int32)
        # Joint attention 的文本上下文固定 2 个 token。
        context = put(rng.normal(0, 0.4, (2, 2, 8)), self.profile.dtype)
        if self.family == "flux_attention":
            return weights, hidden, context
        temb = put(rng.normal(0, 0.6, (2, 8)), self.profile.dtype)
        return weights, hidden, context, temb
