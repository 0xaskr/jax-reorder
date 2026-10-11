"""完整层的求导、未使用索引器和硬件边界证据；各类结果分开统计。"""

from dataclasses import replace

import jax
import numpy as np
from jax._src import core
from jax.sharding import NamedSharding

from examples.sglang_layers.builders import cpu_mesh
from examples.sglang_layers.runner import close
from examples.sglang_layers.upstream import REVISION
from jax_reorder import Status, canonicalize, compare_functions, compare_jaxprs

from . import oracles
from .blocked import cross_attention, glm5_absorbed, mla_original_imports, original_moe
from .builders import BLOCK_PROFILES, PROFILES, BlockCase, Glm5Case
from .candidates import glm5_drop_indexer, glm5_kv_first
from .runner import FAMILIES, build_case, raw_schedule, signature
from .upstream import original_imports


def verify_derivatives(family):
    """全部输出的随机余切与全部选定输入的切向量，保留严格的累加顺序。"""
    profiles, candidates, _ = FAMILIES[family]
    mesh = cpu_mesh()
    with jax.set_mesh(mesh), original_imports():
        case = build_case(family, profiles[0], mesh)
        args = case.sample(1)
        result = compare_functions(
            case.function(),
            case.function(candidates[0][0]),
            *args,
            argnums=(1,),
            modes=("forward", "jvp", "vjp"),
        )
        assert [m.status for m in result.modes] == [
            Status.EQUIVALENT,
            Status.EQUIVALENT,
            Status.DIFFERENT,
        ]
        assert result.status == Status.DIFFERENT and not result.unverified_modes
        assert any(eq.primitive.name == "add_any" for eq in result.modes[2].left.jaxpr.eqns)
        rows = []
        for mode in result.modes:
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
                    "replay": (
                        "sample(1); NumPy default_rng(6700) normal seeds in abstract_inputs order"
                    ),
                    "reason": "VJP 梯度贡献的 add_any 累加顺序改变；数值接近不覆盖图差异",
                }
            )
        return rows


def verify_indexer(profile):
    """真实未使用分支也属于图：保留时接受，删除时拒绝，即使数值一致。"""
    mesh = cpu_mesh()
    profile = replace(profile, indexer=True)
    with jax.set_mesh(mesh), original_imports():
        case = Glm5Case(profile, mesh)
        args = case.sample(1)
        base = case.function()
        graph = jax.make_jaxpr(base)(*args)
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
            assert raw_schedule(graph) != raw_schedule(other)
            for form in (report.left, report.right):
                assert form is not None and form.complete
                core.check_jaxpr(form.jaxpr)
                assert canonicalize(form.jaxpr).graph == form.graph
            close(value, jax.jit(case.function(fn))(*args))
            normalized = report.left.jaxpr
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
    graph = jax.make_jaxpr(fn)(*args)
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.UNKNOWN
    assert any("unsupported primitive pallas_call" in issue.reason for issue in result.issues)
    compilation = "executed" if executable else "cpu_compile_rejected"
    rejection = None
    if executable:
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
        "actual": result.status.value,
        "comparison_scope": "original against itself; no candidate certified",
        "cpu": compilation,
        "compile_rejection": rejection,
        "unknown_reasons": [issue.reason for issue in result.issues],
        "signature": signature(args),
        "numerical_samples": int(executable),
    }


def verify_boundaries():
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
            try:
                jax.make_jaxpr(case.function())(*case.sample(1))
            except exception as exc:
                assert message in str(exc), str(exc)
                rows.append({"layer": layer, "actual": "trace_rejected", "reason": str(exc)})
            else:
                raise AssertionError("原始追踪边界改变，必须重新审核")
    return rows


def supplemental_report():
    return {
        "upstream_revision": REVISION,
        "derivatives": [row for family in FAMILIES for row in verify_derivatives(family)],
        "indexer": [
            row for profile in (PROFILES[0], PROFILES[3]) for row in verify_indexer(profile)
        ],
        "boundaries": verify_boundaries(),
        "unverified": [
            "GLM5 use_dsa_sparse=True 与其他稀疏后端",
            "GLM5 MoE Decoder 的完整原始调用链",
            "FluxSingleTransformerBlock 的 AI 候选（原图追踪被拒绝）",
            "上述完整层的其他 dtype/shape/配置、权重求导、高阶求导与多主机",
        ],
    }
