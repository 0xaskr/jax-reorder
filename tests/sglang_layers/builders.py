"""实例化原始上游层，并将参数作为显式输入；构造期间的 RNG 不进入计算图。

Profile 固定小尺寸与精度；build() 按层 key 选择上游构造签名，只缩小
hidden/intermediate/tokens，保留原算法与配置分支。split() 把参数拆成动态
输入；sample() 为每个种子独立采样权重与输入，供 runner 与 NumPy 参照使用。
这里不执行图比较，也不改写候选。
"""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .upstream import dependencies, layer_class


@dataclass(frozen=True)
class Profile:
    """单层签名配置：名称、精度、hidden/intermediate 尺寸与 token 数。"""

    name: str
    dtype: object
    hidden: int
    intermediate: int
    tokens: int


# 三种签名：FP32、BF16 与更宽的 FP32；token 数固定为 2–3。
PROFILES = (
    Profile("f32", jnp.float32, 4, 8, 3),
    Profile("bf16", jnp.bfloat16, 4, 8, 3),
    Profile("wide", jnp.float32, 6, 10, 2),
)


def cpu_mesh():
    """单 CPU 设备、显式 data/tensor mesh，作为上游分片参数的最小环境。"""
    return Mesh(
        np.array(jax.devices("cpu")[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def build(spec, profile, mesh):
    """按层 key 构造原始类实例，保留其构造签名与算法分支。

    spec/profile/mesh 决定层类、尺寸与精度；返回实例。配置字段一次性覆盖
    各家族读取的同义 hidden/intermediate/激活开关，只有构造签名确实不同
    的家族才分支；不修改参数，也不调用上游 forward。
    """
    cls = layer_class(spec.path, spec.class_name)
    h, mid, dtype = profile.hidden, profile.intermediate, profile.dtype
    # 字段覆盖不同层读取的同义配置名；取值都映射到小尺寸。
    cfg = SimpleNamespace(
        hidden_size=h,
        intermediate_size=mid,
        hidden_act="silu",
        vt_hidden_size=h,
        vt_intermediate_size=mid,
        d_model=h,
        d_ff=mid,
        is_gated_act=True,
        dense_act_fn="gelu_new",
        feed_forward_proj="gated-gelu",
        dropout_rate=0.1,
    )
    # 构造期 RNG 只影响初始参数，随后被 sample 的独立权重替换。
    rngs = nnx.Rngs(17)
    specs = dependencies()["VisionShardSpecs"](mesh, True)
    key = spec.key
    # 各家族构造签名不同：按 key 显式分支，禁止按类名或路径猜测。
    if key == "dflash":
        model = cls(cfg, mesh, dtype=dtype)
    elif key in ("gemma4_vision", "qwen3_vision"):
        cfg.hidden_act = "gelu_pytorch_tanh"
        model = cls(cfg, dtype, mesh, specs)
    elif key in ("kimi_vision", "qwen25_legacy"):
        model = cls(cfg, dtype, rngs)
    elif key == "mimo_vision":
        model = cls(cfg, dtype, rngs, mesh, True)
    elif key == "qwen25_vision":
        model = cls(cfg, dtype, mesh, rngs, True)
    elif key == "kimi_norm":
        # GatedRMSNorm 只接收 hidden 与参数精度；epsilon 用类默认值。
        model = cls(h, param_dtype=dtype)
    elif key == "minimax2":
        # QK norm 固定 epsilon=1e-6，与覆盖清单一致。
        model = cls(h, epsilon=1e-6, param_dtype=dtype)
    elif key in ("umt5", "t5", "clip"):
        if key == "clip":
            cfg.hidden_act = "quick_gelu"
        model = cls(cfg, mesh, dtype=dtype)
    elif key == "dspark":
        model = cls(vocab_size=mid, markov_rank=h, mesh=mesh, dtype=dtype)
    elif key == "qwen3_moe_gate":
        model = cls(h, num_experts=mid, weight_dtype=dtype)
    elif key == "wan":
        model = cls(h, mid, mesh=mesh, dtype=dtype)
    elif key == "flux":
        model = cls(h, mesh, inner_dim=mid, params_dtype=dtype, rngs=rngs)
    elif key == "qwen_omni_vision":
        model = cls(h, mid, "gelu_pytorch_tanh", mesh, dtype, rngs)
    elif key == "audio_quantizer":
        # 两级码本（n_q=2），bins 使用 small intermediate 尺寸。
        model = cls(h, n_q=2, bins=[mid, mid], dtype=dtype, mesh=mesh)
    elif key == "flux_vae":
        # 带 1×1 shortcut，推理模式下 dropout 关闭。
        model = cls(h, h, groups=2, use_nin_shortcut=True, dtype=dtype, rngs=rngs)
    elif key == "wan_vae":
        # 时间方向 padding=(1,1,1)，保留因果卷积的输入缓存。
        model = cls(h, h, rngs=rngs, padding=(1, 1, 1))
    elif key == "minimax3":
        model = cls(h, mid, 1.702, 7.0, mesh, dtype)
    elif key == "grok":
        model = cls(h, mid, layer_id=0, mesh=mesh, dtype=dtype)
    elif key == "glm5":
        # CPU 非融合路径 use_fused=False。
        model = cls(h, mid, mesh=mesh, dtype=dtype, use_fused=False)
    else:
        model = cls(h, mid, mesh=mesh, dtype=dtype)
    return model


def split(model):
    """把 NNX 模块拆成 (graphdef, 动态参数, 其余静态状态)。"""
    return nnx.split(model, nnx.Param, ...)


def baseline(spec, model, *inputs):
    """调用原始层的实际入口；rvq/markov 使用其公开方法而不是 __call__。"""
    if spec.family == "rvq_decode":
        return model.decode(*inputs)
    if spec.family == "markov":
        return model.apply_step_logits(*inputs)
    return model(*inputs)


def sample(spec, profile, mesh, params, seed):
    """每个权重独立采样，避免上游固定 PRNGKey 令 gate/up 偶然相等。

    0 为零激活，1 为交替正负边界；另三个固定种子覆盖非对称一般输入。
    所有样本都有独立权重，不能依赖同一组权重得出候选正确的结论。

    返回 (采样权重, *输入)；输入形状按 spec.family 的真实接口构造。
    """
    rng = np.random.default_rng(9100 + seed)

    def weight(value):
        """为单个参数叶子采样独立权重；1 维叶子加偏移保留偏置信号。"""
        raw = rng.normal(0, 0.35, value.shape).astype(np.float32)
        if value.ndim == 1:
            raw += 0.7
        return jax.device_put(jnp.asarray(raw, value.dtype), value.sharding)

    sampled = jax.tree.map(weight, params)
    h, t, mid = profile.hidden, profile.tokens, profile.intermediate

    def floating(shape):
        """按种子生成浮点输入：0 全零，1 为交替正负，其余为一般随机值。"""
        raw = rng.normal(0, 0.8, shape).astype(np.float32)
        if seed == 0:
            raw.fill(0)
        elif seed == 1:
            raw = (np.arange(np.prod(shape)).reshape(shape) % 7 - 3).astype(np.float32)
        return jax.device_put(jnp.asarray(raw, profile.dtype), NamedSharding(mesh, P()))

    # 输入形状按家族真实接口：三维视觉、NDHWC 缓存、整数码本等。
    if spec.family == "rvq_decode":
        # 码本索引为两级整数；seed 0 全零，其余随机。
        values = (
            np.zeros((2, t), np.int32)
            if seed == 0
            else rng.integers(0, mid, (2, t), dtype=np.int32)
        )
        inputs = (jax.device_put(values, NamedSharding(mesh, P())),)
    elif spec.family == "markov":
        # Markov 头同时接收浮点 logits 与整数 token 索引。
        tokens = jax.device_put(
            rng.integers(0, mid, (t,), dtype=np.int32), NamedSharding(mesh, P())
        )
        inputs = (floating((t, mid)), tokens)
    elif spec.family == "gated_norm":
        inputs = (floating((t, h)), floating((t, h)))
    elif spec.family == "resnet":
        inputs = (floating((1, t, 3, h)),)
    elif spec.family == "causal_conv":
        # NDHWC 新帧与时间历史缓存，缓存长度 1。
        inputs = (floating((1, t, 3, 3, h)), floating((1, 1, 3, 3, h)))
    elif spec.family == "vision_gate_gelu":
        inputs = (floating((1, t, h)),)
    else:
        inputs = (floating((t, h)),)
    return (sampled, *inputs)
