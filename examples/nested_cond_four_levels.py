"""手写四层 cond：完整分支树、独立计算换序、显式绑定和最深层错误。

运行：uv run python examples/nested_cond_four_levels.py [--show-jaxpr] [--report PATH]
运行时只生成数值样本，不生成候选或修改 AST/jaxpr。四个布尔输入决定
16 条路径，结果中的整数编号能检查每条路径确实被执行。SCENARIO 含一个
equivalent 正例和两个 different 负例；入口先断言 15 个 cond 节点、深度 4，
避免写成四次连续浅层 cond 却声称四层嵌套。
"""

import argparse
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_reorder import Status

if __package__:
    from . import business_layer_cases as business
else:
    import business_layer_cases as business


BIAS = np.array([0.25, -0.5, 0.75], dtype=np.float32)


def leaf_true(u, v, scale, code):
    """第四层真分支：共享 score 同时产生矩阵输出与行统计。"""
    a = lax.sin(u)
    b = lax.cos(v)
    score = (a - b) * scale
    stats = jnp.sum(score, axis=-1) + code.astype(jnp.float32)
    return score, stats, code + jnp.int32(1)


def leaf_wrong_binding(u, v, scale, code):
    """固定负例：最深真分支的 sin/cos 实际输入绑定交换，其余层保持原样。"""
    a = lax.sin(v)
    b = lax.cos(u)
    score = (a - b) * scale
    stats = jnp.sum(score, axis=-1) + code.astype(jnp.float32)
    return score, stats, code + jnp.int32(1)


def _original(true_leaf, x, y, scale, f0, f1, f2, f3):
    """外三层共用固定源码，负例只替换人工写好的第四层函数。"""
    scale32 = scale.astype(jnp.float32)
    shared = x.astype(jnp.float32) * scale32
    other = y.astype(jnp.float32) + 0.25

    def fourth(u, v, code):
        def yes():
            return true_leaf(u, v, scale32, code)

        def no():
            a = lax.sin(u)
            b = lax.cos(v)
            score = (a + b) * scale32
            stats = jnp.sum(score, axis=-1) + code.astype(jnp.float32)
            return score, stats, code

        return lax.cond(f3, yes, no)

    def third(a, b, code):
        def yes():
            first = lax.tanh(a)
            second = lax.square(b)
            return fourth(first + second, second - first, code + jnp.int32(2))

        def no():
            first = lax.square(a)
            second = lax.tanh(b)
            return fourth(first - second, first + second, code)

        return lax.cond(f2, yes, no)

    def second(p, q, code):
        def yes():
            left = p + BIAS
            right = q * scale32
            out, stats, tag = third(left - right, right, code + jnp.int32(4))
            return out + left, stats + jnp.sum(right, axis=-1), tag

        def no():
            left = p * scale32
            right = q - BIAS
            out, stats, tag = third(left, right - left, code)
            return out - left, stats + jnp.sum(right, axis=-1), tag

        return lax.cond(f1, yes, no)

    def yes():
        a = lax.sin(shared)
        b = lax.cos(other)
        return second(a, b, jnp.int32(8))

    def no():
        a = lax.cos(shared)
        b = lax.sin(other)
        return second(a, b, jnp.int32(0))

    return lax.cond(f0, yes, no)


def original(x, y, scale, f0, f1, f2, f3):
    """基准入口：外三层共用 _original，第四层真分支使用 leaf_true。"""
    return _original(leaf_true, x, y, scale, f0, f1, f2, f3)


def reordered(x, y, scale, f0, f1, f2, f3):
    """固定正例：四层均改变独立计算顺序；第四层还同步置换显式实参。

    输入/输出接口、减法方向、乘法位置、分组与共享节点均保持不变。
    各层是直接写出的 JAX 程序，没有通过遍历原始程序生成候选。
    """
    other = y.astype(jnp.float32) + 0.25
    scale32 = scale.astype(jnp.float32)
    shared = x.astype(jnp.float32) * scale32

    def fourth(u, v, code):
        # 顺序改为 (v, scale, code, u)，两个分支同时遵守新的绑定关系。
        def yes(second, weight, tag, first):
            b = lax.cos(second)
            a = lax.sin(first)
            score = (a - b) * weight
            code32 = tag.astype(jnp.float32)
            stats = jnp.sum(score, axis=-1) + code32
            return score, stats, tag + jnp.int32(1)

        def no(second, weight, tag, first):
            b = lax.cos(second)
            a = lax.sin(first)
            score = (a + b) * weight
            code32 = tag.astype(jnp.float32)
            stats = jnp.sum(score, axis=-1) + code32
            return score, stats, tag

        return lax.cond(f3, yes, no, v, scale32, code, u)

    def third(a, b, code):
        def yes():
            second = lax.square(b)
            first = lax.tanh(a)
            return fourth(first + second, second - first, code + jnp.int32(2))

        def no():
            second = lax.tanh(b)
            first = lax.square(a)
            return fourth(first - second, first + second, code)

        return lax.cond(f2, yes, no)

    def second(p, q, code):
        def yes():
            right = q * scale32
            left = p + BIAS
            # 不依赖嵌套结果的行归约也提前，保留输出相加的操作数位置。
            reduced = jnp.sum(right, axis=-1)
            out, stats, tag = third(left - right, right, code + jnp.int32(4))
            return out + left, stats + reduced, tag

        def no():
            right = q - BIAS
            left = p * scale32
            reduced = jnp.sum(right, axis=-1)
            out, stats, tag = third(left, right - left, code)
            return out - left, stats + reduced, tag

        return lax.cond(f1, yes, no)

    def yes():
        b = lax.cos(other)
        a = lax.sin(shared)
        return second(a, b, jnp.int32(8))

    def no():
        b = lax.sin(other)
        a = lax.cos(shared)
        return second(a, b, jnp.int32(0))

    return lax.cond(f0, yes, no)


def wrong_deep_binding(x, y, scale, f0, f1, f2, f3):
    """负例：只替换最深层真分支的 sin/cos 输入绑定，其余层保持基准源码。"""
    return _original(leaf_wrong_binding, x, y, scale, f0, f1, f2, f3)


def wrong_third_predicate(x, y, scale, f0, f1, f2, f3):
    """负例：第三层条件取反（传入 ~f2），实际执行路径与基准不同。"""
    return _original(leaf_true, x, y, scale, f0, f1, ~f2, f3)


def reference(x, y, scale, f0, f1, f2, f3):
    """用顺序执行的 NumPy/Python 分支作参照，不调用 JAX 候选或比较器。"""
    weight = np.asarray(scale, np.float32)
    shared = np.asarray(x, np.float32) * weight
    other = np.asarray(y, np.float32) + np.float32(0.25)
    if bool(f0):
        p, q, code = np.sin(shared), np.cos(other), 8
    else:
        p, q, code = np.cos(shared), np.sin(other), 0
    if bool(f1):
        left, right = p + BIAS, q * weight
        a, b = left - right, right
        code += 4
    else:
        left, right = p * weight, q - BIAS
        a, b = left, right - left
    if bool(f2):
        first, second = np.tanh(a), b * b
        u, v = first + second, second - first
        code += 2
    else:
        first, second = a * a, np.tanh(b)
        u, v = first - second, first + second
    a, b = np.sin(u), np.cos(v)
    score = (a - b if bool(f3) else a + b) * weight
    stats = np.sum(score, axis=-1) + np.float32(code)
    if bool(f3):
        code += 1
    out = score + left if bool(f1) else score - left
    return out, stats + np.sum(right, axis=-1), np.int32(code)


SCENARIO = business.Scenario(
    "four_nested_cond",
    original,
    reference,
    (
        business.Variant(
            reordered, Status.EQUIVALENT, "same", "四层独立运算换序，第四层实参/形参同步置换"
        ),
        business.Variant(
            wrong_deep_binding, Status.DIFFERENT, "witness", "第四层真分支的 sin/cos 输入绑定错误"
        ),
        business.Variant(
            wrong_third_predicate, Status.DIFFERENT, "witness", "第三层条件取反，实际执行路径改变"
        ),
    ),
)


def make_samples(profile, trials=3, seed=20261011):
    """5 组数值输入分别执行全部 16 条路径；首组全假不能掩盖深层错误。"""
    if profile not in business.PROFILES:
        raise ValueError(profile)
    rng = np.random.default_rng(seed)
    shape = (1, 3) if profile == "small" else (2, 3)
    dtype = jnp.bfloat16 if profile == "bfloat16" else jnp.float32
    samples = []
    for index in range(trials + 2):
        x, y = [rng.normal(0, 0.7, shape).astype(np.float32) for _ in range(2)]
        scale = rng.uniform(0.3, 1.1, shape[-1]).astype(np.float32)
        if index == 0:
            x.fill(0)
            y.fill(0)
        elif index == 1:
            x[:] = (-1.0, 0.0, 1.0)
            y[:] = (0.5, -0.75, 0.25)
        arrays = tuple(jnp.asarray(value, dtype) for value in (x, y, scale))
        for flags in itertools.product((False, True), repeat=4):
            samples.append((*arrays, *(jnp.bool_(flag) for flag in flags)))
    return samples


def structure(graph):
    """统计实际追踪到的条件节点和深度，防止四次顺序 cond 冒充四层嵌套。"""
    levels = {}
    count = 0
    deepest = 0
    pending = [(graph, 0)]
    while pending:
        region, depth = pending.pop()
        deepest = max(deepest, depth)
        levels.setdefault(depth, []).append(tuple(eq.primitive.name for eq in region.eqns))
        for eq in region.eqns:
            if eq.primitive is lax.cond_p:
                count += 1
                pending.extend((branch, depth + 1) for branch in eq.params["branches"])
    return count, deepest, levels


def main():
    """检查嵌套结构后逐签名重放三个候选，并打印是否找到数值反例。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-jaxpr", action="store_true")
    parser.add_argument("--report", type=Path, help="保存结构判断与完整数值反例")
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")
    results = []
    for profile in business.PROFILES:
        samples = make_samples(profile)
        graph = jax.make_jaxpr(original)(*samples[0])
        count, depth, _ = structure(graph)
        assert (count, depth) == (15, 4), (count, depth)
        print(f"{profile}: cond 节点={count}，嵌套深度={depth}，16 条路径 × 5 组输入")
        for variant in SCENARIO.variants:
            result = business.verify_variant(
                SCENARIO, variant, profile, samples, show_jaxpr=args.show_jaxpr
            )
            results.append(result)
            witness = result["witness"]
            detail = (
                f"，数值反例样本={witness['sample_index']}" if witness else "，全部样本输出一致"
            )
            print(f"PASS {variant.fn.__name__}: {result['actual']}{detail}")
    if args.report:
        report = {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "cond_depth": 4,
            "cond_nodes": 15,
            "results": results,
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"通过 {len(results)} 个候选/签名组合，共 {sum(r['samples'] for r in results)} 组检查。")


if __name__ == "__main__":
    main()
