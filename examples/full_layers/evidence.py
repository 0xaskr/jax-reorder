"""完整层的求导、未使用索引器和硬件边界证据；各类结果分开统计。

verify_derivatives 对四类层首个 FP32 配置跟踪 forward/JVP/VJP（argnums=(1,)），
VJP 预期图不同但数值一致；verify_indexer 验证未参与输出的索引器仍属完整图；
verify_boundaries 记录 Pallas/absorbed 路径的 unknown 与 CPU compile/trace
拒绝。supplemental_report 汇总为 runner --supplemental 报告；这些证据不
并入主 cases 的等价计数，unknown 与拒绝不能算作通过。
"""

from dataclasses import replace

import jax
import numpy as np
from jax._src import core
from jax.sharding import NamedSharding

from examples.sglang_layers.builders import cpu_mesh
from examples.sglang_layers.runner import close
from examples.sglang_layers.upstream import REVISION
from jax_reorder import (
    Status,
    UnsupportedJaxprError,
    canonicalize,
    compare_functions,
    compare_jaxprs,
)

from . import oracles
from .blocked import cross_attention, glm5_absorbed, mla_original_imports, original_moe
from .builders import BLOCK_PROFILES, PROFILES, BlockCase, Glm5Case
from .candidates import glm5_drop_indexer, glm5_kv_first
from .runner import FAMILIES, build_case, raw_schedule, signature
from .upstream import original_imports


def verify_derivatives(family):
    """全部输出的随机余切与全部选定输入的切向量，保留严格的累加顺序。

    family 必须是 FAMILIES 中的键；返回每个模式一行的证据记录。只对
    hidden states（argnums=(1,)）求导，权重和其他状态仍为动态输入。
    预期前向/JVP 等价、VJP 图不同（add_any 累加顺序改变），两者都要求
    数值一致；VJP 图差异不能被数值接近掩盖。
    """
    profiles, candidates, _ = FAMILIES[family]
    mesh = cpu_mesh()
    with jax.set_mesh(mesh), original_imports():
        case = build_case(family, profiles[0], mesh)
        args = case.sample(1)
        result = compare_functions(
            case.function(),
            case.function(candidates[0][0]),
            *args,
            # 仅对 hidden states 求导；其余输入保持动态普通输入。
            argnums=(1,),
            modes=("forward", "jvp", "vjp"),
            equivalence="structural",
        )
        assert [m.status for m in result.modes] == [
            Status.EQUIVALENT,
            Status.EQUIVALENT,
            Status.DIFFERENT,
        ]
        assert result.status == Status.DIFFERENT and not result.unverified_modes
        # 图不同必须能归因到 add_any 累加顺序，而不是其他语义差异。
        assert any(eq.primitive.name == "add_any" for eq in result.modes[2].left.jaxpr.eqns)
        rows = []
        for mode in result.modes:
            # 每个模式使用自己的完整动态切向量/余切向量种子。
            rng = np.random.default_rng(6700)
            flat = tuple(jax.tree.leaves(args))
            seeds = tuple(
                jax.device_put(
                    np.asarray(rng.normal(size=a.shape), a.dtype),
                    NamedSharding(mesh, a.sharding.spec),
                )
                for a in mode.left.abstract_inputs[len(flat) :]
            )
            actual = []
            for trace, normalized in (
                (mode.left, mode.comparison.left),
                (mode.right, mode.comparison.right),
            ):
                g = trace.jaxpr
                # 原图与规范化图都必须先通过 IR 校验，再实际执行比较。
                core.check_jaxpr(g)
                core.check_jaxpr(normalized.jaxpr)
                value = jax.jit(lambda *xs, g=g: core.eval_jaxpr(g, g.consts, *xs))(*flat, *seeds)
                ng = normalized.jaxpr
                reordered = jax.jit(lambda *xs, g=ng: core.eval_jaxpr(g, g.consts, *xs))(
                    *flat, *seeds
                )
                close(value, reordered)
                actual.append(value)
            close(*actual)
            rows.append(
                {
                    "family": family,
                    "profile": profiles[0].name,
                    "candidate": candidates[0][0].__name__,
                    "mode": mode.mode,
                    "argnums": [1],
                    "expected": "different" if mode.mode == "vjp" else "equivalent",
                    "actual": mode.status.value,
                    "difference": mode.difference,
                    "signature": signature(args),
                    "seed_signature": signature(seeds),
                    "numerical_agreement": True,
                    "numerical_samples": 1,
                    # replay 说明如何从固定种子重建同一切向量，避免大数组入报告。
                    "replay": (
                        "sample(1); NumPy default_rng(6700) normal seeds in abstract_inputs order"
                    ),
                    "reason": "VJP 梯度贡献的 add_any 累加顺序改变；数值接近不覆盖图差异",
                }
            )
        return rows


def verify_indexer(profile):
    """真实未使用分支也属于图：保留时接受，删除时拒绝，即使数值一致。

    profile 会被复制并把 indexer 打开；返回两行证据（保留索引器、删除
    索引器）。Glm5Attention 的索引器输出不进入最终结果，但 top_k 等方程
    属于完整图，因此保留时 equivalent、删除时 different。
    """
    mesh = cpu_mesh()
    profile = replace(profile, indexer=True)
    with jax.set_mesh(mesh), original_imports():
        case = Glm5Case(profile, mesh)
        args = case.sample(1)
        base = case.function()
        graph = jax.make_jaxpr(base)(*args)
        # 确认该配置确实构造出索引器路径（top_k 为索引器核心原语）。
        assert any(eq.primitive.name == "top_k" for eq in graph.eqns)
        value = jax.jit(base)(*args)
        close(value, oracles.glm5(case, args), low_precision=True, oracle=True)
        rows = []
        for fn, status in (
            (glm5_kv_first, Status.EQUIVALENT),
            (glm5_drop_indexer, Status.DIFFERENT),
        ):
            other = jax.make_jaxpr(case.function(fn))(*args)
            report = compare_jaxprs(graph, other)
            assert report.status == status, report
            # 实际方程列表必须变化；名称相同不算覆盖。
            assert raw_schedule(graph) != raw_schedule(other)
            for form in (report.left, report.right):
                assert form is not None and form.complete
                core.check_jaxpr(form.jaxpr)
                # 规范化幂等；两次规范化必须得到同一图。
                assert canonicalize(form.jaxpr).graph == form.graph
            close(value, jax.jit(case.function(fn))(*args))
            normalized = report.left.jaxpr
            # 规范化后的图单独执行，确认重排没改变数值。
            close(
                value,
                jax.jit(lambda *xs, g=normalized: core.eval_jaxpr(g, g.consts, *xs))(
                    *jax.tree.leaves(args)
                ),
            )
            rows.append(
                {
                    "layer": "Glm5Attention",
                    "profile": profile.name,
                    "indexer": True,
                    "indexer_rope_concat": False,
                    "use_dsa_sparse": False,
                    "mode": "forward",
                    "candidate": fn.__name__,
                    "expected": status.value,
                    "actual": report.status.value,
                    "difference": report.difference,
                    "numerical_agreement": True,
                    "numerical_samples": 1,
                    "signature": signature(args),
                    "reason": "删除未使用索引器改变完整图，但不改变本配置的返回值",
                }
            )
        return rows


def _pallas_boundary(fn, args, *, layer, configuration, executable=False):
    """记录一条硬件路径的实际拒绝点，不认证任何候选。

    fn 为原始 absorbed/MLA/USP/GMM 调用；先确认比较器以 pallas_call 拒绝，
    再按 executable 选择：CPU interpret 可执行时用独立参照核对，否则要求
    CPU 编译给出 “Only interpret mode is supported on CPU”。返回边界记录；
    任何预期拒绝消失都会抛出断言，提示重新审核。
    """
    graph = jax.make_jaxpr(fn)(*args)
    try:
        compare_jaxprs(graph, graph)
    except UnsupportedJaxprError as error:
        # 只接受明确的 pallas_call 不支持；其他异常不能被当作边界证据。
        assert "unsupported primitive pallas_call" in error.reason
        failure = {"type": type(error).__name__, "path": error.path, "reason": error.reason}
    else:
        raise AssertionError("Pallas comparison boundary changed; review its semantics")
    compilation = "executed" if executable else "cpu_compile_rejected"
    rejection = None
    if executable:
        # 原 EPMoE 在 CPU 自动走 interpret；这是原源码分支，不是替代实现。
        close(jax.jit(fn)(*args), oracles.moe(args), low_precision=True, oracle=True)
    else:
        try:
            jax.jit(fn).lower(*args).compile()
        except ValueError as exc:
            assert "Only interpret mode is supported on CPU" in str(exc), str(exc)
            rejection = str(exc)
        else:
            raise AssertionError("原始硬件路径的 CPU 边界已改变，必须重新审核")
    return {
        "layer": layer,
        "configuration": configuration,
        "error": failure,
        "comparison_scope": "original against itself; no candidate certified",
        "cpu": compilation,
        "compile_rejection": rejection,
        "signature": signature(args),
        "numerical_samples": int(executable),
    }


def verify_boundaries():
    """逐项重放 Pallas/absorbed/CPU 编译与原始追踪拒绝路径。

    返回边界记录列表：除最后两行 trace_rejected 外，其余是 unknown 图探测
    （含 EPMoE 的 CPU interpret 执行证据）。这些行不计入等价通过数，只用于
    说明当前环境实际拒绝位置。
    """
    mesh = cpu_mesh()
    rows = []
    with jax.set_mesh(mesh), mla_original_imports():
        for decoder in (False, True):
            for mode in ("EXTEND", "DECODE"):
                fn, args = glm5_absorbed(mesh, decoder=decoder, mode=mode)
                rows.append(
                    _pallas_boundary(
                        fn,
                        args,
                        layer="Glm5DecoderLayer" if decoder else "Glm5Attention",
                        configuration={"absorbed": True, "mode": mode, "moe": False},
                    )
                )
    with jax.set_mesh(mesh), cross_attention(mesh) as (fn, args):
        rows.append(_pallas_boundary(fn, args, layer="WanT2VCrossAttention", configuration={}))
    with jax.set_mesh(mesh), original_moe(mesh) as (fn, args):
        # EPMoE 的 interpret 分支可执行，因此同时保存逐专家 NumPy 参照证据。
        rows.append(
            _pallas_boundary(
                fn,
                args,
                layer="EPMoE",
                configuration={"experts": 2, "top_k": 1, "cpu_interpret": True},
                executable=True,
            )
        )
    with jax.set_mesh(mesh), original_imports():
        for layer, case, exception, message in (
            (
                "FluxSingleTransformerBlock",
                BlockCase(BLOCK_PROFILES[0], mesh, "flux"),
                AssertionError,
                "with_sharding_constraint",
            ),
            (
                "Glm5Attention(single head)",
                Glm5Case(replace(PROFILES[0], heads=1), mesh),
                core.ShardingTypeError,
                "same sharding",
            ),
        ):
            # 原追踪必须继续拒绝；若不再拒绝，说明锁定版本行为已变，需要重新审核。
            try:
                jax.make_jaxpr(case.function())(*case.sample(1))
            except exception as exc:
                assert message in str(exc), str(exc)
                rows.append({"layer": layer, "actual": "trace_rejected", "reason": str(exc)})
            else:
                raise AssertionError("原始追踪边界改变，必须重新审核")
    return rows


def supplemental_report():
    """汇总求导、索引器、硬件边界和未验证项，供 --supplemental 报告使用。

    返回 dict；各组分开保存，调用方另算 supplemental_counts，不并入主表。
    """
    return {
        "upstream_revision": REVISION,
        "derivatives": [row for family in FAMILIES for row in verify_derivatives(family)],
        "indexer": [
            row for profile in (PROFILES[0], PROFILES[3]) for row in verify_indexer(profile)
        ],
        "boundaries": verify_boundaries(),
        # 未验证项显式保留，避免把“没测”误读成“通过”。
        "unverified": [
            "GLM5 use_dsa_sparse=True 与其他稀疏后端",
            "GLM5 MoE Decoder 的完整原始调用链",
            "FluxSingleTransformerBlock 的 AI 候选（原图追踪被拒绝）",
            "上述完整层的其他 dtype/shape/配置、权重求导、高阶求导与多主机",
        ],
    }
