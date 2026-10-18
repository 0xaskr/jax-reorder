"""逐层验收：固定 AI 预判、精确图比较、独立参照、数值反例与重排执行。

对 LAYERS × PROFILES 的每个组合，检查原图 IR、图预判、规范化幂等、原图
不变、两侧规范化前后执行、独立 NumPy 参照，并为负例保存数值见证。
--show-jaxpr 打印原图与候选图，--report 写 JSON（含四类注册映射）。

运行：
  uv run python -m examples.sglang_layers.runner --report /tmp/sglang-layers.json
  uv run python -m examples.sglang_layers.runner --layer llama --profile bf16 --show-jaxpr

成功时逐层打印 PASS 并返回 0；断言失败或参数非法时退出码非零。结论只
覆盖选定层、声明签名和 CPU 环境。
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from jax._src import core

from jax_reorder import Status, canonicalize, compare_jaxprs

from .builders import PROFILES, baseline, build, cpu_mesh, sample, split
from .candidates import FAMILIES, REASONS
from .catalog import COMPONENT_LAYERS, CONFIG_LAYERS, LAYERS, MODEL_LAYERS, MULTIMODAL_LAYERS
from .oracles import reference
from .upstream import REVISION


def values(tree):
    """把 pytree 转成 NumPy 数组列表，供逐叶比较与快照使用。"""
    return [np.asarray(value) for value in jax.tree.leaves(tree)]


def close(left, right, *, low_precision=False, oracle=False):
    """结构、shape 和 dtype 先独立检查；容差仅用于数值验证。

    oracle=True 时 base 与独立参照比较，容差为 FP32 3e-5 / BF16 0.035；
    否则两侧执行结果比较，容差固定 2e-5。任何 shape/dtype 不一致直接
    断言失败，不能由容差掩盖。
    """
    a, b = values(left), values(right)
    assert len(a) == len(b)
    rtol, atol = ((0.035, 0.035) if low_precision else (3e-5, 3e-5)) if oracle else (2e-5, 2e-5)
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_allclose(x.astype(np.float64), y.astype(np.float64), rtol=rtol, atol=atol)


def snapshot(tree):
    """把 pytree 的 shape/dtype/数值转成可写入 JSON 的快照。"""
    return [
        {"shape": list(v.shape), "dtype": str(v.dtype), "values": v.astype(np.float64).tolist()}
        for v in values(tree)
    ]


def verify(spec, profile, *, show_jaxpr=False):
    """对单个层/签名组合执行两个固定候选的完整验收。

    spec/profile 决定层与尺寸；show_jaxpr 只控制调试打印。返回每个候选
    一行的记录。使用 5 组独立权重/输入；base 先与独立 NumPy 参照核对，
    正例要求两侧执行一致，负例要求至少一组输入出现数值反例。
    """
    mesh = cpu_mesh()
    results = []
    with jax.set_mesh(mesh):
        model = build(spec, profile, mesh)
        graphdef, params, rest = split(model)

        def base(p, *x):
            """用给定动态参数重建模块并调用原始入口。"""
            return baseline(spec, nnx.merge(graphdef, p, rest), *x)

        samples = [sample(spec, profile, mesh, params, seed) for seed in range(5)]
        base_graph = jax.make_jaxpr(base)(*samples[0])
        compiled_base = jax.jit(base)
        original = str(base_graph)
        # 候选顺序与预期结论固定：第一个正例、第二个反例。
        expected_statuses = (Status.EQUIVALENT, Status.DIFFERENT)
        for fn, expected, reason in zip(
            FAMILIES[spec.family], expected_statuses, REASONS[spec.family], strict=True
        ):

            def candidate(p, *x, fn=fn):
                """用给定动态参数重建模块并调用 AI 候选。"""
                return fn(nnx.merge(graphdef, p, rest), *x)

            candidate_graph = jax.make_jaxpr(candidate)(*samples[0])
            if show_jaxpr:
                print(
                    f"\n{spec.key}/{profile.name}/{fn.__name__}\nBASE\n{base_graph}\nCANDIDATE\n{candidate_graph}"
                )
            report = compare_jaxprs(base_graph, candidate_graph)
            # 预期标签在写候选时人工固定，与比较器输出无关。
            assert report.status == expected, (spec.key, profile.name, fn.__name__, report)
            assert str(base_graph) == original
            normalized = (report.left, report.right)
            for item in normalized:
                assert item is not None
                core.check_jaxpr(item.jaxpr)
                # 规范化幂等：再次规范化必须得到同一图。
                assert canonicalize(item.jaxpr).graph == item.graph
            executables = [
                jax.jit(lambda *args, g=item.jaxpr: core.eval_jaxpr(g, g.consts, *args))
                for item in normalized
            ]
            compiled_candidate = jax.jit(candidate)
            witness = None
            for seed, args in enumerate(samples):
                base_output = compiled_base(*args)
                output = compiled_candidate(*args)
                # 每次比较都重新构造参照实例，避免参照侧复用候选的模块状态。
                oracle = reference(spec, nnx.merge(graphdef, args[0], rest), *args[1:])
                close(base_output, oracle, low_precision=profile.name == "bf16", oracle=True)
                assert jax.tree.structure(base_output) == jax.tree.structure(output)
                flat = jax.tree.leaves(args)
                # 两侧规范化图分别实际执行，证明重排前后数值一致。
                close(base_output, executables[0](*flat))
                close(output, executables[1](*flat))
                if expected == Status.EQUIVALENT:
                    close(base_output, output)
                elif witness is None:
                    a, b = values(base_output), values(output)
                    if any(
                        not np.allclose(
                            x.astype(np.float64), y.astype(np.float64), rtol=2e-3, atol=2e-3
                        )
                        for x, y in zip(a, b, strict=True)
                    ):
                        # 保存第一组实际数值反例：输入、两侧输出与种子。
                        witness = {
                            "seed": seed,
                            "inputs": snapshot(args),
                            "base_outputs": snapshot(base_output),
                            "candidate_outputs": snapshot(output),
                        }
            if expected == Status.DIFFERENT:
                assert witness is not None, (spec.key, profile.name, fn.__name__, "缺少数值反例")
            results.append(
                {
                    "layer": spec.key,
                    "profile": profile.name,
                    "candidate": fn.__name__,
                    "expected": expected.value,
                    "reason": reason,
                    "actual": report.status.value,
                    "difference": report.difference,
                    "samples": len(samples),
                    # raw_order_changed 只说明原始表示变化，不等同于等价结论。
                    "raw_order_changed": str(candidate_graph) != original,
                    "witness": witness,
                }
            )
    return results


def main():
    """解析参数，逐层/签名运行 verify 并输出或写入报告。

    --layer/--profile 限定单个组合，--show-jaxpr 打印图，--report 写 JSON
    （含四类注册映射）。参数非法时 parser.error 以退出码 2 结束；验收失败
    时异常向上传播，退出码非零；全部通过返回 0。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=[s.key for s in LAYERS])
    parser.add_argument("--profile", choices=[p.name for p in PROFILES])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--show-jaxpr", action="store_true")
    args = parser.parse_args()
    # 显式固定 CPU，避免本地默认后端改变结论或报告字段。
    jax.config.update("jax_platforms", "cpu")
    results = []
    for spec in LAYERS:
        if args.layer and spec.key != args.layer:
            continue
        for profile in PROFILES:
            if args.profile and profile.name != args.profile:
                continue
            results.extend(verify(spec, profile, show_jaxpr=args.show_jaxpr))
            print(
                f"PASS {spec.key}/{profile.name}: equivalent + different，5 组独立输入/权重",
                flush=True,
            )
    if args.report:
        report = {
            "revision": REVISION,
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            # 同时保存四类注册映射，便于从模型名/配置名追溯到所选层。
            "models": MODEL_LAYERS,
            "multimodal_models": MULTIMODAL_LAYERS,
            "configs": CONFIG_LAYERS,
            "components": COMPONENT_LAYERS,
            "results": results,
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"通过 {len(results)} 个候选/签名，{sum(r['samples'] for r in results)} 组数值验证。")


if __name__ == "__main__":
    main()
