"""原始硬件路径的可重放边界探测；拒绝结果不计入等价通过数。

针对 Pallas absorbed MLA、Wan USP/Flash cross attention 与 megablox GMM
EPMoE，加载未经改写的原内核与后端，分别记录结构比较、搜索预算以及 CPU 的
实际编译/追踪拒绝点。这里不启用替代 interpret（EPMoE 的 interpret 是原
源码的 CPU 分支），也不把“可执行”当作“可比较”。入口由
evidence.verify_boundaries 调用，最终经 runner --supplemental 报告。
"""

import abc
import functools
import logging
import sys
import types
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec as P

from examples.sglang_layers.upstream import dependencies, load, source

from .upstream import classes


@functools.cache
def mla_dependencies():
    """加载 absorbed MLA 的真实 Pallas 内核、后端、pool 与 decoder 依赖。

    返回 SimpleNamespace；首次调用执行已校验文件并缓存。内核文件按原
    AST 整体执行，不改写或替换计算；只注入后端构造实际读取的符号。
    """
    # 原内核只有 JAX/Pallas 标准导入；直接执行整个已校验文件，不改写 AST。
    name = "sgl_jax.srt.kernels.mla.v2.kernel"
    kernel = types.ModuleType(name)
    exec(compile(source("kernels/mla/v2/kernel.py"), name, "exec"), kernel.__dict__)
    # 后端、metadata 与 pool 均来自同一锁定快照，维度保持原模型值。
    path = "layers/attention/mla_backend.py"
    metadata = load(
        path,
        "MLAAttentionMetadata",
        register_pytree_node_class=jax.tree_util.register_pytree_node_class,
    )
    backend = load(
        path,
        "MLAAttentionBackend",
        AttentionBackend=load(
            "layers/attention/base_attn_backend.py",
            "AttentionBackend",
            abstractmethod=abc.abstractmethod,
        ),
        MLAAttentionMetadata=metadata,
        named_scope=dependencies()["named_scope"],
        mla_ragged_paged_attention=kernel.mla_ragged_paged_attention,
    )
    pool_path = "mem_cache/memory_pool.py"
    pool = load(
        pool_path,
        "MLATokenToKVPool",
        # 复用已加载 MHA pool 的基类，保证与真实 KV 池实现同一来源。
        KVCache=classes().KVPool.__bases__[0],
        register_pytree_node_class=jax.tree_util.register_pytree_node_class,
        _get_kv_zero_allocator=load(
            pool_path, "_get_kv_zero_allocator", _KV_ZERO_ALLOCATOR_CACHE={}
        ),
        logger=logging.getLogger(__name__),
        GB=1024**3,
    )
    glm_path = "models/glm5_moe.py"
    # 完整 Decoder 复用同一 Glm5Attention/RMSNorm；MLP 走非融合实现。
    decoder = load(
        glm_path,
        "Glm5DecoderLayer",
        Glm5Attention=classes().Glm5Attention,
        Glm5MLP=load(glm_path, "Glm5MLP", LinearBase=dependencies()["LinearBase"]),
        RMSNorm=classes().RMSNorm,
    )
    return types.SimpleNamespace(
        kernel=kernel, backend=backend, metadata=metadata, pool=pool, decoder=decoder
    )


@contextmanager
def mla_original_imports():
    """把已执行的 MLA 内核模块临时注册到 sys.modules，退出时恢复。

    原后端在函数内部按模块名 import 内核；副作用仅限进程内模块表。
    """
    kernel = mla_dependencies().kernel
    previous = sys.modules.get(kernel.__name__)
    sys.modules[kernel.__name__] = kernel
    try:
        yield
    finally:
        if previous is None:
            del sys.modules[kernel.__name__]
        else:
            sys.modules[kernel.__name__] = previous


def glm5_absorbed(mesh, *, decoder=False, mode="DECODE"):
    """实际 absorbed Attention 或 Decoder；显式内核配置只绕过 TPU 自动调参。

    正常 Pallas lowering 不启用 interpret，也不将内核换成 NumPy/普通 JAX。
    权重、hidden 与 KV 状态仍是动态输入；本函数用于证明 CPU 的实际拒绝点。
    decoder=True 时构造完整 Glm5DecoderLayer（非 MoE 层），否则只构造
    Glm5Attention。返回 (run, args)，run 每次调用重建 metadata/pool/module。
    """
    deps = mla_dependencies()
    if decoder:
        cfg = types.SimpleNamespace(
            # 配置沿用固定快照的 head/LoRA 维度；只缩小 hidden/intermediate。
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=12,
            rms_norm_eps=1e-6,
            rope_parameters={"rope_theta": 1000000},
            indexer_types=["shared"],
            first_k_dense_replace=1,
            _sgl_use_fused_mlp=False,
        )
        model = deps.decoder(cfg, mesh, dtype=jnp.bfloat16)
        # 确认走 absorbed MHA 且不是 MoE 层，确保探测目标明确。
        assert model.self_attn.use_absorbed and not model.is_moe_layer
    else:
        model = classes().Glm5Attention(
            hidden_size=8,
            num_heads=2,
            num_kv_heads=1,
            max_position_embeddings=64,
            mesh=mesh,
            rms_norm_eps=1e-6,
            use_qk_norm=False,
            dtype=jnp.bfloat16,
            use_absorbed=True,
            has_indexer=False,
        )
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    weights, tree = jax.tree.flatten(params)
    pool = deps.pool(256, 128, jnp.bfloat16, 512, 64, 1, mesh)
    _, pool_meta = pool.tree_flatten()
    backend = deps.backend(
        # 显式 block/VMEM 参数避免依赖 TPU 自动调参；数值随快照固定。
        num_attn_heads=2,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        page_size=128,
        mesh=mesh,
        vmem_limit_bytes=16 * 1024 * 1024,
        num_kv_pages_per_block=(1, 1, 1),
        num_queries_per_block=(16, 16, 16),
    )

    def run(weights, hidden, positions, cache, lengths, pages, q_lens, kv_lens, distribution):
        """用本次动态参数重建后端 metadata/pool/module，再执行原始层。"""
        # 每次调用都重建原 metadata/pool，避免跨追踪保留动态状态。
        # NNX Module 对象需经自己的构造逻辑初始化；后端的静态配置可共享，
        # 用原 tree_unflatten 复原包含本次动态 metadata 的后端。
        _, auxiliary = backend.tree_flatten()
        metadata = deps.metadata(q_lens, kv_lens, pages, lengths, distribution)
        b = deps.backend.tree_unflatten(auxiliary, (metadata,))
        batch = classes().ForwardBatch(
            bid=0,
            forward_mode=getattr(classes().ForwardMode, mode),
            batch_size=1,
            input_ids=None,
            req_pool_indices=None,
            seq_lens=lengths,
            out_cache_loc=None,
            positions=positions,
            attn_backend=b,
        )
        state = deps.pool.tree_unflatten(pool_meta, ([cache], []))
        module = nnx.merge(graphdef, tree.unflatten(weights), rest)
        return module(positions, hidden, batch, state)

    # 一个合法序列，cache 第 1 页存放当前 token；head/LoRA 维度保持原值。
    args = (
        tuple(weights),
        jnp.ones((1, 8), jnp.bfloat16),
        jnp.array([0], jnp.int32),
        pool.kv_buffer[0],
        jnp.array([1], jnp.int32, out_sharding=P("data")),
        jnp.array([1], jnp.int32, out_sharding=P("data")),
        jnp.array([0, 1], jnp.int32, out_sharding=P("data")),
        jnp.array([0, 128], jnp.int32, out_sharding=P("data")),
        jnp.array([1, 1, 1] if mode == "DECODE" else [0, 0, 1], jnp.int32, out_sharding=P("data")),
    )
    return run, args


@contextmanager
def _original_modules(paths):
    """仅给完整原文件建立导入命名空间，结束时还原；不注入计算替代函数。

    paths 为快照内相对路径；按原路径生成 sgl_jax.srt.* 模块名并在
    sys.modules 中临时注册（含缺失的父包），退出时按逆序还原。副作用仅限
    进程内模块表，内核代码本身按原文执行。
    """
    saved = {}
    modules = {}
    try:
        for path in paths:
            name = "sgl_jax.srt." + path.removesuffix(".py").replace("/", ".")
            parts = name.split(".")
            for depth in range(1, len(parts)):
                parent_name = ".".join(parts[:depth])
                if parent_name not in sys.modules:
                    saved[parent_name] = None
                    parent = types.ModuleType(parent_name)
                    parent.__path__ = []
                    sys.modules[parent_name] = parent
            saved[name] = sys.modules.get(name)
            module = types.ModuleType(name)
            sys.modules[name] = module
            exec(compile(source(path), name, "exec"), module.__dict__)
            modules[path] = module
        yield modules
    finally:
        for name, previous in reversed(tuple(saved.items())):
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@contextmanager
def cross_attention(mesh):
    """加载 Wan 的 USP/Flash cross attention 原始调用链。

    paths 为原工具、调度与 Flash 内核文件；上下文产出 (run, args)，run
    执行原始 WanT2VCrossAttention。该路径最终落到 Pallas，用于记录 CPU
    边界而不是替换注意力核心。副作用是临时模块表，退出时恢复。
    """
    paths = (
        "utils/jax_utils.py",
        "multimodal/kernels/tuned_block_sizes.py",
        "multimodal/kernels/flash_attention.py",
    )
    with _original_modules(paths) as modules:
        kernel = modules[paths[-1]]
        backend = load(
            "multimodal/layers/attention/flash_attention_backend.py",
            "FlashAttentionBackend",
            AttentionBackend=load(
                "layers/attention/base_attn_backend.py",
                "AttentionBackend",
                abstractmethod=abc.abstractmethod,
            ),
            flash_attention=kernel.flash_attention,
        )
        import jax.experimental.pallas as pl

        path = "multimodal/layers/attention/layer.py"
        usp = load(
            path,
            "USPAttention",
            FlashAttentionBackend=backend,
            SegmentIds=kernel.SegmentIds,
            align_to=load(path, "align_to", pl=pl),
        )
        path = "multimodal/models/wan/diffusion/wan_dit.py"
        parent = load(
            path,
            "WanSelfAttention",
            LinearBase=dependencies()["LinearBase"],
            RMSNorm=classes().RMSNorm,
            USPAttention=usp,
        )
        cls = load(path, "WanT2VCrossAttention", WanSelfAttention=parent)
        model = cls(dim=256, num_heads=2, mesh=mesh)
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        weights, tree = jax.tree.flatten(params)

        def run(weights, hidden, context):
            """用动态权重重建原始 WanT2VCrossAttention 并执行 cross attention。"""
            m = nnx.merge(graphdef, tree.unflatten(weights), rest)
            return m(hidden, context, None)

        yield (
            run,
            (
                tuple(weights),
                jnp.ones((2, 2, 256), jnp.bfloat16),
                jnp.ones((2, 3, 256), jnp.bfloat16),
            ),
        )


@contextmanager
def original_moe(mesh):
    """加载 megablox GMM 与原始 EPMoE，构造 2 专家、top-1 的 CPU 路径。

    CPU 上原后端自动使用 interpret 分支，该路径可以结构自比较。
    上下文产出 (run, args)；权重与输入使用固定
    非对称值，便于 oracles.moe 逐专家独立核对。退出时恢复模块表。
    """
    paths = (
        "utils/jax_utils.py",
        "kernels/gmm/megablox_gmm_kernel/common.py",
        "kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py",
        "kernels/gmm/megablox_gmm_kernel/gmm.py",
    )
    with _original_modules(paths) as modules:
        is_tpu = modules[paths[0]].is_tpu_runtime
        gmm = load(
            "kernels/gmm/megablox_gmm_backend.py",
            "gmm",
            is_tpu_runtime=is_tpu,
            gmm_v1_kernel=modules[paths[-1]].gmm,
        )
        cls = load(
            "layers/moe.py",
            "EPMoE",
            named_scope=dependencies()["named_scope"],
            shard_map=jax.shard_map,
            gmm=gmm,
            get_global_expert_location_metadata=load(
                "eplb/expert_location.py",
                "get_global_expert_location_metadata",
                _GLOBAL_EXPERT_LOCATION_METADATA=None,
            ),
        )
        model = cls(
            hidden_size=128,
            num_experts=2,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=128,
            use_sc_permute=False,
        )
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        weights, tree = jax.tree.flatten(params)

        def run(weights, hidden, gate, indices):
            """用动态权重重建原 EPMoE；hidden/gate/indices 与上游接口一致。"""
            return nnx.merge(graphdef, tree.unflatten(weights), rest)(hidden, gate, indices)

        rng = np.random.default_rng(6500)
        weights = tuple(
            jax.device_put(jnp.asarray(rng.normal(0, 0.04, w.shape), w.dtype), w.sharding)
            for w in weights
        )
        # 参照按明确的参数叶子顺序逐专家计算，输入也使用非对称值。
        assert tuple(params.keys()) == ("wi_0", "wi_1", "wo")
        yield (
            run,
            (
                tuple(weights),
                jnp.asarray(rng.normal(0, 0.5, (2, 128)), jnp.bfloat16),
                jnp.array([[0.7], [0.3]], jnp.bfloat16),
                jnp.array([[0], [1]], jnp.int32),
            ),
        )
