"""完整层验收与可重放报告；只运行预先 hardcode 的 AI 候选。

对 FAMILIES 中的 GLM5、Llama、CLIP、Flux joint attention，逐配置执行原图
IR 检查、原图对应证据、对称性、规范化幂等、两侧规范化前后 CPU 执行、
独立 NumPy 参照及数值见证。报告保存 cases/counts；--supplemental 另加
derivatives、indexer、boundaries、unverified 与 supplemental_counts。

运行：
  uv run python -m examples.full_layers.runner --supplemental --report /tmp/full-layers.json
  uv run python -m examples.full_layers.runner --family glm5 --samples 3

成功时打印 counts 并返回 0；断言失败、参数非法或 original 源码边界改变时
抛出异常/parser.error，退出码非零。报告是本地 CPU 证据，不声明全模型或
所有配置的通过，也不把 unknown/拒绝路径计入等价数。
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import jax
import jaxlib
import numpy as np
from jax._src import core

from examples.sglang_layers.builders import cpu_mesh
from examples.sglang_layers.runner import close
from examples.sglang_layers.upstream import REVISION
from jax_reorder import Status, canonicalize, compare_jaxprs

from . import oracles
from .builders import BLOCK_PROFILES, FLUX_PROFILES, PROFILES, BlockCase, Glm5Case, LlamaCase
from .candidates import CLIP_CASES, FLUX_ATTENTION_CASES, GLM5_CASES, LLAMA_CASES
from .upstream import original_imports


def signature(args):
    """把 pytree 叶子编码为 shape、dtype 与内容 SHA-256，供精确重放。

    返回列表，每项对应一个叶子。大权重不嵌入报告，只保留摘要；重放时
    按报告记录的 family/profile/seed 重新构造输入并核对摘要。
    """
    return [
        {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            # 摘要用于确认重放输入一致，不代替数值比较。
            "sha256": hashlib.sha256(np.asarray(x).tobytes()).hexdigest(),
        }
        for x in jax.tree.leaves(args)
    ]


def raw_schedule(graph):
    """保留原方程次序和有序绑定，忽略仅用于显示的变量名。

    连续三个 dense 的算子名序列可能完全相同；实际 Q/K/V 权重实参改变
    已经构成真实换序，不能只用 Primitive 名称序列作覆盖证据。

    返回 tuple((primitive, inputs))，inputs 把每个操作数编码为输入、常量、
    字面量或前序结果端口。该结果只用于判断实际换序，不参与等价判定。
    """
    values = {v: ("input", i) for i, v in enumerate(graph.invars)}
    values.update({v: ("constant", i) for i, v in enumerate(graph.constvars)})
    schedule = []
    for index, eq in enumerate(graph.eqns):
        inputs = tuple(
            values[v]
            if isinstance(v, core.Var)
            else ("literal", str(v.aval), np.asarray(v.val).tobytes())
            for v in eq.invars
        )
        schedule.append((eq.primitive, inputs))
        values.update({v: ("result", index, port) for port, v in enumerate(eq.outvars)})
    return tuple(schedule)


def numerical_witness(left, right):
    """保存反例的差异位置与两个实际值；输入由报告的 seed/config 精确重放。

    返回第一个超出 rtol=atol=2e-3 的端口、索引、两侧值与最大绝对差；
    全部在容差内时返回 None。该阈值只用于标记数值见证，不构成图结论。
    """
    for port, (a, b) in enumerate(zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)):
        a, b = np.asarray(a).astype(np.float64), np.asarray(b).astype(np.float64)
        wrong = ~np.isclose(a, b, rtol=2e-3, atol=2e-3, equal_nan=True)
        if wrong.any():
            index = tuple(int(i) for i in np.argwhere(wrong)[0])
            return {
                "output_port": port,
                "index": index,
                "base_value": float(a[index]),
                "candidate_value": float(b[index]),
                "max_absolute_difference": float(np.max(np.abs(a - b))),
            }
    return None


# family -> (profiles, hardcode 候选组, 独立 NumPy 参照)。完整 Flux block 的
# 原图追踪被拒绝，不在表内；其候选只在 candidates.py 中保留源码。
FAMILIES = {
    "glm5": (PROFILES, GLM5_CASES, oracles.glm5),
    "llama": (PROFILES, LLAMA_CASES, oracles.llama),
    "clip": (BLOCK_PROFILES, CLIP_CASES, oracles.clip),
    "flux_attention": (FLUX_PROFILES, FLUX_ATTENTION_CASES, oracles.flux_attention),
}


def build_case(family, profile, mesh):
    """按 family 构造对应原始层，返回提供 sample/function 的 case 对象。"""
    if family == "glm5":
        return Glm5Case(profile, mesh)
    if family == "llama":
        return LlamaCase(profile, mesh)
    return BlockCase(profile, mesh, family)


def verify(profile, *, family="glm5", samples=3):
    """对单个 family/profile 执行全部候选的图比较与数值验收。

    返回每个候选一行的报告记录。samples 为独立随机种子数，每个种子使用
    独立权重与输入。不打印、不写文件；失败时抛出断言异常。验证内容包括
    原图 IR、NumPy 参照、图预判、对应证据、对称性、幂等、原图不变、
    规范化前后执行及负例数值见证。
    """
    profiles, candidates, oracle = FAMILIES[family]
    mesh = cpu_mesh()
    with jax.set_mesh(mesh), original_imports():
        case = build_case(family, profile, mesh)
        inputs = [case.sample(seed) for seed in range(samples)]
        base = case.function()
        graph = jax.make_jaxpr(base)(*inputs[0])
        before = str(graph)
        core.check_jaxpr(graph)
        base_compiled = jax.jit(base)
        expected_outputs = [base_compiled(*args) for args in inputs]
        # 每个样本的 base 输出先与独立 NumPy 参照核对，防止两侧同时算错。
        for args, expected in zip(inputs, expected_outputs, strict=True):
            close(
                expected, oracle(case, args), low_precision=profile.dtype != np.float32, oracle=True
            )
        rows = []
        for fn, expected, reason in candidates:
            candidate = case.function(fn)
            other = jax.make_jaxpr(candidate)(*inputs[0])
            core.check_jaxpr(other)
            report = compare_jaxprs(graph, other)
            # 预判是写候选时人工固定的标签，不相符即失败。
            assert report.status.value == expected, (fn.__name__, report)
            if expected == "equivalent":
                assert report.correspondence is not None
                # 正例还要求反向比较，确认不是单向偶然匹配。
                assert compare_jaxprs(other, graph).status == Status.EQUIVALENT
                # 确认实际依赖排列改变，排除仅改变 Python 函数名称的伪覆盖。
                assert raw_schedule(graph) != raw_schedule(other)
            normalized = (report.left, report.right)
            executables = []
            for form in normalized:
                assert form is not None
                core.check_jaxpr(form.jaxpr)
                # 幂等：再次规范化必须得到同一图，避免表示不稳定。
                assert canonicalize(form.jaxpr).graph == form.graph
                executables.append(
                    jax.jit(lambda *xs, g=form.jaxpr: core.eval_jaxpr(g, g.consts, *xs))
                )
            compiled = jax.jit(candidate)
            witness = None
            for seed, (args, value) in enumerate(zip(inputs, expected_outputs, strict=True)):
                output = compiled(*args)
                assert jax.tree.structure(value) == jax.tree.structure(output)
                flat = jax.tree.leaves(args)
                # 两侧规范化图分别执行，证明重排前后数值一致。
                close(value, executables[0](*flat))
                close(output, executables[1](*flat))
                if expected == "equivalent":
                    close(value, output)
                elif witness is None:
                    evidence = numerical_witness(value, output)
                    if evidence is not None:
                        witness = {"seed": seed, "inputs": signature(args), **evidence}
            # 负例必须有数值见证，不能用“图不同”代替数值证据。
            assert expected == "equivalent" or witness is not None
            rows.append(
                {
                    "family": family,
                    "layer": type(case.model).__name__,
                    "upstream_revision": REVISION,
                    "profile": profile.name,
                    "mode": "forward",
                    "backend": (
                        "NativeAttention/MHATokenToKVPool.set_kv_buffer_legacy"
                        if family in ("glm5", "llama")
                        else "original pure JAX/CPU XLA"
                    ),
                    # 记录静态构造开关，说明该结论对应的实际运行分支。
                    "static_configuration": (
                        {
                            "forward_mode": profile.mode,
                            "indexer": profile.indexer,
                            "absorbed": False,
                            "qk_norm_constructor_flag": profile.qk_norm,
                            "rotary_interleaved": False,
                        }
                        if family == "glm5"
                        else {"forward_mode": profile.mode, "query_heads": 4, "kv_heads": 2}
                        if family == "llama"
                        else {"causal": profile.causal, "dropout": "deterministic"}
                        if family == "clip"
                        else {
                            "attention_impl": "sdpa",
                            "joint_context": True,
                            "image_rotary_emb": None,
                            "is_causal": False,
                        }
                    ),
                    "candidate": fn.__name__,
                    "expected": expected,
                    "actual": report.status.value,
                    "reason": reason,
                    "difference": report.difference,
                    "signature": signature(inputs[0]),
                    "primitives": sorted({eq.primitive.name for eq in graph.eqns}),
                    "numerical_samples": samples,
                    "raw_order_changed": raw_schedule(graph) != raw_schedule(other),
                    "witness": witness,
                    "replay": {
                        "family": family,
                        "profile_index": profiles.index(profile),
                        "sample": "build_case(family, profile, mesh).sample(seed)",
                    },
                }
            )
        # 所有候选跑完后原图字符串必须不变，防止验收过程意外修改基准。
        assert str(graph) == before
        return rows


def verify_glm5(profile, *, samples=3):
    """兼容旧调用：等价于 family="glm5" 的 verify。"""
    return verify(profile, samples=samples)


def main():
    """解析参数，逐 family/profile 运行 verify 并输出报告与 counts。

    --family 限定单个 family，--samples 为每组数值样本数，--supplemental
    另列求导、索引器与硬件边界证据（要求不指定 --family）。--report 写 JSON；
    始终把 counts 打印到标准输出。参数非法时 parser.error 以退出码 2 结束；
    验证失败时异常向上传播，退出码非零。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--family", choices=tuple(FAMILIES))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--supplemental", action="store_true", help="另列求导、索引器与硬件拒绝证据"
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    families = (args.family,) if args.family else tuple(FAMILIES)
    rows = [
        row
        for family in families
        for profile in FAMILIES[family][0]
        for row in verify(profile, family=family, samples=args.samples)
    ]
    result = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "devices": [str(d) for d in jax.devices()],
            "mesh": {"data": 1, "tensor": 1, "axis_types": "Explicit"},
            "jax_enable_x64": jax.config.jax_enable_x64,
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "jax_default_prng_impl": jax.config.jax_default_prng_impl,
        },
        "cases": rows,
        # counts 只统计前向主表；unknown/拒绝路径由 supplemental 单独列出。
        "counts": {
            "source_classes": len(families),
            "candidates": sum(len(FAMILIES[family][1]) for family in families),
            "distinct_candidate_functions": len(
                {fn for family in families for fn, _, _ in FAMILIES[family][1]}
            ),
            "candidate_signatures": len(rows),
            "numerical_samples": sum(row["numerical_samples"] for row in rows),
            "equivalent": sum(row["actual"] == "equivalent" for row in rows),
            "different": sum(row["actual"] == "different" for row in rows),
            "actually_reordered_positive_signatures": sum(
                row["expected"] == "equivalent" and row["raw_order_changed"] for row in rows
            ),
        },
    }
    if args.supplemental:
        if args.family:
            parser.error("--supplemental checks all families; omit --family")
        from .evidence import supplemental_report

        # supplemental 证据不并入 cases/counts，避免把 unknown 或拒绝计作通过。
        result["supplemental"] = supplemental_report()
        result["supplemental_counts"] = {
            group: {
                "cases": len(result["supplemental"][group]),
                "errors": sum("error" in row for row in result["supplemental"][group]),
                "statuses": {
                    status: sum(
                        row.get("actual") == status for row in result["supplemental"][group]
                    )
                    for status in ("equivalent", "different", "trace_rejected")
                },
            }
            for group in ("derivatives", "indexer", "boundaries")
        }
    if args.report:
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
