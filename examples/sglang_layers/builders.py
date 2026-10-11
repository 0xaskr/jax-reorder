"""实例化原始上游层，并将参数作为显式输入；构造期间的 RNG 不进入计算图。"""

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
    name: str
    dtype: object
    hidden: int
    intermediate: int
    tokens: int


PROFILES = (
    Profile("f32", jnp.float32, 4, 8, 3),
    Profile("bf16", jnp.bfloat16, 4, 8, 3),
    Profile("wide", jnp.float32, 6, 10, 2),
)


def cpu_mesh():
    return Mesh(
        np.array(jax.devices("cpu")[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def build(spec, profile, mesh):
    """小尺寸保留层的原始算法和构造逻辑，配置分支写在覆盖清单中。"""
    cls = layer_class(spec.path, spec.class_name)
    h, mid, dtype = profile.hidden, profile.intermediate, profile.dtype
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
    rngs = nnx.Rngs(17)
    specs = dependencies()["VisionShardSpecs"](mesh, True)
    key = spec.key
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
        model = cls(h, param_dtype=dtype)
    elif key == "minimax2":
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
        model = cls(h, n_q=2, bins=[mid, mid], dtype=dtype, mesh=mesh)
    elif key == "flux_vae":
        model = cls(h, h, groups=2, use_nin_shortcut=True, dtype=dtype, rngs=rngs)
    elif key == "wan_vae":
        model = cls(h, h, rngs=rngs, padding=(1, 1, 1))
    elif key == "minimax3":
        model = cls(h, mid, 1.702, 7.0, mesh, dtype)
    elif key == "grok":
        model = cls(h, mid, layer_id=0, mesh=mesh, dtype=dtype)
    elif key == "glm5":
        model = cls(h, mid, mesh=mesh, dtype=dtype, use_fused=False)
    else:
        model = cls(h, mid, mesh=mesh, dtype=dtype)
    return model


def split(model):
    return nnx.split(model, nnx.Param, ...)


def baseline(spec, model, *inputs):
    if spec.family == "rvq_decode":
        return model.decode(*inputs)
    if spec.family == "markov":
        return model.apply_step_logits(*inputs)
    return model(*inputs)


def sample(spec, profile, mesh, params, seed):
    """每个权重独立采样，避免上游固定 PRNGKey 令 gate/up 偶然相等。

    0 为零激活，1 为交替正负边界；另三个固定种子覆盖非对称一般输入。
    所有样本都有独立权重，不能依赖同一组权重得出候选正确的结论。
    """
    rng = np.random.default_rng(9100 + seed)

    def weight(value):
        raw = rng.normal(0, 0.35, value.shape).astype(np.float32)
        if value.ndim == 1:
            raw += 0.7
        return jax.device_put(jnp.asarray(raw, value.dtype), value.sharding)

    sampled = jax.tree.map(weight, params)
    h, t, mid = profile.hidden, profile.tokens, profile.intermediate

    def floating(shape):
        raw = rng.normal(0, 0.8, shape).astype(np.float32)
        if seed == 0:
            raw.fill(0)
        elif seed == 1:
            raw = (np.arange(np.prod(shape)).reshape(shape) % 7 - 3).astype(np.float32)
        return jax.device_put(jnp.asarray(raw, profile.dtype), NamedSharding(mesh, P()))

    if spec.family == "rvq_decode":
        values = (
            np.zeros((2, t), np.int32)
            if seed == 0
            else rng.integers(0, mid, (2, t), dtype=np.int32)
        )
        inputs = (jax.device_put(values, NamedSharding(mesh, P())),)
    elif spec.family == "markov":
        tokens = jax.device_put(
            rng.integers(0, mid, (t,), dtype=np.int32), NamedSharding(mesh, P())
        )
        inputs = (floating((t, mid)), tokens)
    elif spec.family == "gated_norm":
        inputs = (floating((t, h)), floating((t, h)))
    elif spec.family == "resnet":
        inputs = (floating((1, t, 3, h)),)
    elif spec.family == "causal_conv":
        inputs = (floating((1, t, 3, 3, h)), floating((1, 1, 3, 3, h)))
    elif spec.family == "vision_gate_gelu":
        inputs = (floating((1, t, h)),)
    else:
        inputs = (floating((t, h)),)
    return (sampled, *inputs)
