"""加载固定上游的真实计算依赖，避免以简化 attention 代替原实现。

类与函数从 tests/fixtures/sglang_jax 的已校验快照按 AST 抽取，只注入所选
层实际读取的依赖；模型服务器、权重下载和无关模块不执行。本模块只准备
比较基准，不生成候选，也不证明数值正确性。

classes() 覆盖 GLM5、Llama、NativeAttention 及 KV pool 依赖；
block_classes() 覆盖 CLIP encoder 与 Flux 层，其中 FluxSingleTransformerBlock
的原图追踪在锁定版本上被拒绝，只作边界记录。入口见
`python -m examples.full_layers.runner`。
"""

import abc
import enum
import functools
import logging
import sys
import time
import types
from collections.abc import Iterable
from contextlib import contextmanager

import jax
from flax.nnx.nn import dtypes
from jax._src import dtypes as jax_dtypes

from examples.sglang_layers.upstream import dependencies, layer_class, load


@functools.cache
def classes():
    """按需装配 GLM5/Llama/NativeAttention 的原始类，并缓存结果。

    返回 SimpleNamespace；首次调用执行多次 AST 抽取，不运行模型前向。
    依赖通过 load 的显式关键字传入，缺依赖即失败，不做替代实现。结果
    缓存供同进程内多次构造复用；模块级状态的隔离由调用方负责。
    """
    deps = dependencies()
    scope = deps["named_scope"]
    norm_path = "layers/layernorm.py"
    # 上游模块之间存在隐式全局依赖，逐个注入选中类实际引用的函数。
    axes = load(norm_path, "_canonicalize_axes", Iterable=Iterable)
    norm_deps = dict(_canonicalize_axes=axes, dtypes=dtypes)
    rms = load(
        norm_path,
        "RMSNorm",
        named_scope=scope,
        _compute_stats=load(
            norm_path, "_compute_stats", **norm_deps, _abs_sq=load(norm_path, "_abs_sq")
        ),
        _normalize=load(norm_path, "_normalize", **norm_deps),
    )
    rope_path = "layers/embeddings.py"
    rope = load(
        rope_path,
        "RotaryEmbedding",
        named_scope=scope,
        # 固定 RoPE 环境为非交错（Neox 半维）实现，与验证记录一致。
        apply_rotary_emb=load(rope_path, "apply_rotary_emb", _ROTARY_INTERLEAVED=False),
    )
    batch_path = "model_executor/forward_batch_info.py"
    mode = load(batch_path, "ForwardMode", IntEnum=enum.IntEnum, auto=enum.auto)
    batch = load(
        batch_path,
        "ForwardBatch",
        ForwardMode=mode,
        register_pytree_node_class=jax.tree_util.register_pytree_node_class,
    )
    radix_path = "layers/radix_attention.py"
    kind = load(radix_path, "AttentionType", Enum=enum.Enum)
    radix = load(radix_path, "RadixAttention", AttentionType=kind)
    backend_path = "layers/attention/native_backend.py"
    backend_base = load(
        "layers/attention/base_attn_backend.py",
        "AttentionBackend",
        abstractmethod=abc.abstractmethod,
    )
    attention = load(
        backend_path,
        "forward_attention",
        ForwardMode=mode,
        _apply_extend_mask=load(backend_path, "_apply_extend_mask"),
        _apply_decode_mask=load(backend_path, "_apply_decode_mask"),
    )
    native = load(
        backend_path,
        "NativeAttention",
        AttentionBackend=backend_base,
        AttentionType=kind,
        ForwardMode=mode,
        named_scope=scope,
        forward_attention=attention,
        is_tpu_runtime=load("utils/jax_utils.py", "is_tpu_runtime"),
    )
    util_path = "kernels/ragged_paged_attention/util.py"
    packing = load(util_path, "get_dtype_packing", dtypes=jax_dtypes)
    align = load(util_path, "align_to", cdiv=load(util_path, "cdiv"))
    # 原 merge_kv 在函数内部 import util；original_imports() 负责把该模块注册进 sys.modules。
    util = types.ModuleType("sgl_jax.srt.kernels.ragged_paged_attention.util")
    util.align_to, util.get_dtype_packing = align, packing
    pool_path = "mem_cache/memory_pool.py"
    pool = load(
        pool_path,
        "MHATokenToKVPool",
        KVCache=load(
            pool_path,
            "KVCache",
            abc=abc,
            register_pytree_node_class=jax.tree_util.register_pytree_node_class,
        ),
        register_pytree_node_class=jax.tree_util.register_pytree_node_class,
        get_dtype_packing=packing,
        merge_kv=load(pool_path, "merge_kv", get_dtype_packing=packing),
        # 零填充分配器缓存随本次加载隔离，保持原类语义但不跨测试共享。
        _get_kv_zero_allocator=load(
            pool_path, "_get_kv_zero_allocator", _KV_ZERO_ALLOCATOR_CACHE={}
        ),
        logger=logging.getLogger(__name__),
        time=time,
        GB=1024**3,
    )
    indexer = load(
        "models/glm5_moe.py",
        "GlmDsaIndexer",
        LinearBase=deps["LinearBase"],
        GlmNorm=load("models/glm5_moe.py", "GlmNorm"),
        get_hadamard_matrix=load("models/glm5_moe.py", "get_hadamard_matrix"),
        # 索引器固定 at[].set 的 RoPE 写入路径；use_dsa_sparse 默认关闭。
        _INDEXER_ROPE_CONCAT=False,
    )
    glm = load(
        "models/glm5_moe.py",
        "Glm5Attention",
        LinearBase=deps["LinearBase"],
        RMSNorm=rms,
        RotaryEmbedding=rope,
        RadixAttention=radix,
        GlmDsaIndexer=indexer,
    )
    llama = load(
        "models/llama.py",
        "LlamaAttention",
        LinearBase=deps["LinearBase"],
        DtypeConfig=deps["DtypeConfig"],
        RadixAttention=radix,
        # 清空进程级 RoPE 缓存，避免跨构造复用同一份频率表。
        get_rope=load(rope_path, "get_rope", RotaryEmbedding=rope, _ROPE_DICT={}),
    )
    return types.SimpleNamespace(
        Glm5Attention=glm,
        LlamaAttention=llama,
        RMSNorm=rms,
        RotaryEmbedding=rope,
        RadixAttention=radix,
        NativeAttention=native,
        ForwardBatch=batch,
        ForwardMode=mode,
        KVPool=pool,
        util=util,
    )


@functools.cache
def block_classes():
    """装配 CLIP 编码层与 Flux 层；Flux block 只用于边界记录。

    返回包含 CLIPEncoderLayer、FluxSingleTransformerBlock、FluxAttention、
    sdpa 与输出分片辅助的 SimpleNamespace。sdpa 为纯 JAX 路径，不启用
    Pallas；FluxSingleTransformerBlock 在锁定 JAX 上原图追踪被拒绝，因此
    只在 evidence 中单独列出，不计入等价通过数。
    """
    deps = dependencies()
    clip_path = "multimodal/models/encoders/clip.py"
    clip = load(
        clip_path,
        "CLIPEncoderLayer",
        CLIPAttention=load(
            clip_path,
            "CLIPAttention",
            LinearBase=deps["LinearBase"],
            _create_causal_mask=load(clip_path, "_create_causal_mask"),
        ),
        CLIPMLP=layer_class(clip_path, "CLIPMLP"),
    )
    flux_path = "multimodal/models/dits/flux.py"
    ada_path = "multimodal/layers/adalayernorm.py"
    resolve = load(ada_path, "_resolve_rngs")
    ada = load(
        ada_path,
        "FluxAdaLayerNormZeroSingle",
        _resolve_rngs=resolve,
        _build_zero_norm=load(ada_path, "_build_zero_norm", _resolve_rngs=resolve),
        _no_shard=load(ada_path, "_no_shard"),
        ACT2FN=deps["ACT2FN"],
        LinearBase=deps["LinearBase"],
    )
    flux_attn = load(
        flux_path,
        "FluxAttention",
        _resolve_rngs=resolve,
        # 只允许上游声明的两种实现；本实验固定为纯 JAX 的 sdpa。
        _SUPPORTED_ATTENTION_IMPLS=("usp", "sdpa"),
        RMSNorm=classes().RMSNorm,
        LinearBase=deps["LinearBase"],
        _get_qkv_projections=load(
            flux_path,
            "_get_qkv_projections",
            _get_projections=load(flux_path, "_get_projections"),
        ),
        _sdpa_attention=load(flux_path, "_sdpa_attention"),
        _data_seq_tensor_shard=load(flux_path, "_data_seq_tensor_shard"),
        logger=logging.getLogger(__name__),
    )
    flux = load(
        flux_path,
        "FluxSingleTransformerBlock",
        _resolve_rngs=resolve,
        FluxAdaLayerNormZeroSingle=ada,
        FluxAttention=flux_attn,
        LinearBase=deps["LinearBase"],
        ACT2FN=deps["ACT2FN"],
    )
    return types.SimpleNamespace(
        CLIPEncoderLayer=clip,
        FluxSingleTransformerBlock=flux,
        FluxAttention=flux_attn,
        sdpa=load(flux_path, "_sdpa_attention"),
        output_sharding=load(flux_path, "_data_seq_tensor_shard"),
    )


@contextmanager
def original_imports():
    """原 merge_kv 内部的局部 import 仍指向已校验的原工具函数。

    作为上下文管理器，临时在 sys.modules 注册只含 align_to 与
    get_dtype_packing 的同名模块；退出时恢复原模块表，不向缺失的计算依赖
    注入替代实现。副作用仅限进程内模块表，不改变数值。
    """
    util = classes().util
    previous = sys.modules.get(util.__name__)
    sys.modules[util.__name__] = util
    try:
        yield
    finally:
        if previous is None:
            del sys.modules[util.__name__]
        else:
            sys.modules[util.__name__] = previous
