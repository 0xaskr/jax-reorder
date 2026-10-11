"""加载固定上游的真实计算依赖，避免以简化 attention 代替原实现。"""

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
    deps = dependencies()
    scope = deps["named_scope"]
    norm_path = "layers/layernorm.py"
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

    测试结束恢复模块表；不向缺失的计算依赖注入替代实现。
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
