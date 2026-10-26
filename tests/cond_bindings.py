"""重放 RMSNorm 的 cond 闭包误判修复，候选和预期均由 AI 手写并固定。

运行：uv run python tests/cond_bindings.py --show-jaxpr
对每个签名执行真假两个分支：闭包参数同步换序与显式传参对照必须 equivalent，
实际输入绑定错误必须 different 并找到数值反例。脚本只验证 cond 数据实参与
分支形参的同步置换，不声称一般 cond 程序或代数改写等价。
"""

import argparse

import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_reorder import Status

if __package__:
    from . import business_layer_cases as business
else:
    # 支持直接执行 tests/cond_bindings.py，无需修改 PYTHONPATH。
    import business_layer_cases as business


def original(x, gate, scale, flag):
    """原版按归一化、门控顺序计算，通过闭包向分支传值。"""
    return lax.cond(flag, lambda: business.norm_base(x, gate, scale), lambda: jnp.zeros_like(x))


def gate_first(x, gate, scale, flag):
    """只提前独立门控，JAX 自动捕获的实参/形参位置随之改变。"""
    return lax.cond(
        flag, lambda: business.norm_gate_first(x, gate, scale), lambda: jnp.zeros_like(x)
    )


def explicit_arguments(x, gate, scale, flag):
    """显式传参对照，外部接口和分支实际计算不变。"""
    return lax.cond(
        flag,
        lambda args: business.norm_gate_first(*args),
        lambda args: jnp.zeros_like(args[0]),
        (x, gate, scale),
    )


def wrong_binding(x, gate, scale, flag):
    """负例：归一化 gate、用 x 门控，不能因输入集合相同而获得匹配。"""
    return lax.cond(flag, lambda: business.norm_base(gate, x, scale), lambda: jnp.zeros_like(x))


def reference(x, gate, scale, flag):
    """分支外的 NumPy 参照，不依赖规范化器或 JAX cond。"""
    return business.norm_numpy(x, gate, scale) if bool(flag) else np.zeros_like(x)


# 场景固定三个候选：两个正例分别覆盖闭包换序和显式传参，一个绑定错误负例。
SCENARIO = business.Scenario(
    "cond_gated_rmsnorm",
    original,
    reference,
    (
        business.Variant(gate_first, Status.EQUIVALENT, "same", "闭包参数同步换序"),
        business.Variant(explicit_arguments, Status.EQUIVALENT, "same", "显式传参对照"),
        business.Variant(wrong_binding, Status.DIFFERENT, "witness", "实际输入绑定错误"),
    ),
)


def make_samples(profile, trials=3, seed=42):
    """仅生成数值输入，每个样本分别执行真假两个分支。"""
    return [
        (*args, jnp.bool_(flag))
        for args in business.make_samples(business.SCENARIOS[0], profile, trials, seed)
        for flag in (False, True)
    ]


def main():
    """三种签名分别重放三个候选；断言失败即抛错退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-jaxpr", action="store_true")
    args = parser.parse_args()
    for profile in business.PROFILES:
        samples = make_samples(profile)
        for variant in SCENARIO.variants:
            result = business.verify_variant(
                SCENARIO, variant, profile, samples, show_jaxpr=args.show_jaxpr
            )
            print(f"PASS {variant.fn.__name__}/{profile}: {result['actual']}")


if __name__ == "__main__":
    main()
