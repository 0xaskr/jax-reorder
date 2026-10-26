"""原始完整 Attention、AI 固定改写、状态反例及可复现的 CPU 边界。

覆盖：
- 正例：GLM5 完整层在多个 profile 下给出固定标签序列（两正两负），负例带可序列化的数值
  见证；Llama/CLIP/Flux 等代表性完整层同样重放；索引器即使输出未使用也保留。
- 边界：Flux 的显式 sharding constraint、单头配置拒绝追踪；GLM5 Pallas、Wan/EPMoE
  完成结构比较。CPU 正常 lowering 与原 MoE interpret 分别验证。
- 独立证据：attention 依赖文件与 git blob 摘要对照，确认使用的是固定上游源码。
- 求导边界：forward 与 JVP 相同不表示 VJP 累加顺序一致，四种层族分别检查。
- 前置条件：业务标记（pytest -m business）、CPU 平台、tests/fixtures 上游快照。
- 这些是固定层与配置的验收，不覆盖全部模型或硬件后端。
"""

import hashlib
import json
from dataclasses import replace

import jax
import pytest
from jax._src import core

from tests.full_layers.builders import BLOCK_PROFILES, PROFILES, BlockCase, Glm5Case
from tests.full_layers.runner import FAMILIES, verify
from tests.full_layers.upstream import original_imports
from tests.sglang_layers.builders import cpu_mesh
from tests.sglang_layers.upstream import ROOT
from jax_reorder import Status, TraceError, compare_functions, compare_jaxprs

pytestmark = pytest.mark.filterwarnings(
    r"ignore:'\.value' access is now deprecated:DeprecationWarning"
)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_complete_original_glm5_attention(profile):
    rows = verify(profile)
    assert [row["actual"] for row in rows] == ["equivalent", "equivalent", "different", "different"]
    assert rows[3]["witness"]["output_port"] == 1
    assert json.loads(json.dumps(rows))[2]["witness"]["inputs"]


@pytest.mark.parametrize(
    ("family", "profile"),
    [
        (family, p)
        for family, (profiles, _, _) in FAMILIES.items()
        if family != "glm5"
        for p in profiles
    ],
    ids=lambda p: p.name if hasattr(p, "name") else p,
)
def test_representative_complete_layers(family, profile):
    rows = verify(profile, family=family)
    assert rows[0]["actual"] == "equivalent" and rows[0]["raw_order_changed"]
    assert all(row["witness"] is not None for row in rows if row["actual"] == "different")


def test_original_flux_block_rejects_its_explicit_sharding_constraint():
    mesh = cpu_mesh()
    with jax.set_mesh(mesh):
        case = BlockCase(BLOCK_PROFILES[0], mesh, "flux")
        with pytest.raises(AssertionError, match="with_sharding_constraint.*assert"):
            jax.make_jaxpr(case.function())(*case.sample(1))
        with pytest.raises(TraceError) as error:
            compare_functions(case.function(), case.function(), *case.sample(1))
        result = error.value
        assert result.failure_kind == "trace_rejected"
        assert "with_sharding_constraint" in result.reason


def test_original_single_head_configuration_is_upstream_trace_rejection():
    # 原 MHA 代码在 singleton head reshape 后保留了不同的显式 sharding。
    # 不篡改原方法让此配置“通过”，也不将 JAX 拒绝追踪计作比较器 unknown。
    mesh = cpu_mesh()
    with jax.set_mesh(mesh), original_imports():
        case = Glm5Case(replace(PROFILES[0], heads=1), mesh)
        with pytest.raises(core.ShardingTypeError, match="same sharding"):
            jax.make_jaxpr(case.function())(*case.sample(1))


def test_attention_dependencies_match_independent_git_blob_digests():
    tree = json.loads((ROOT / "upstream-attention-tree.json").read_text())
    for entry in tree["files"]:
        data = (ROOT / (entry["path"] + ".txt")).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == entry["sha"]
        )


@pytest.mark.parametrize("decoder", [False, True], ids=["attention", "decoder"])
@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
def test_original_glm5_absorbed_path_rejects_cpu_lowering(decoder, mode):
    from tests.full_layers.blocked import glm5_absorbed, mla_original_imports

    mesh = cpu_mesh()
    with jax.set_mesh(mesh), mla_original_imports():
        fn, args = glm5_absorbed(mesh, decoder=decoder, mode=mode)
        left, right = (jax.make_jaxpr(fn)(*args) for _ in range(2))
        result = compare_jaxprs(left, right, search_budget=1)
        assert result.status == Status.EQUIVALENT
        assert result.correspondence is not None
        with pytest.raises(ValueError, match="Only interpret mode is supported on CPU"):
            jax.jit(fn).lower(*args).compile()


def test_original_cross_attention_is_structurally_comparable_but_cpu_rejected():
    from tests.full_layers.blocked import cross_attention

    mesh = cpu_mesh()
    with jax.set_mesh(mesh), cross_attention(mesh) as (fn, args):
        result = compare_jaxprs(jax.make_jaxpr(fn)(*args), jax.make_jaxpr(fn)(*args))
        assert result.status == Status.EQUIVALENT
        with pytest.raises(ValueError, match="Only interpret mode is supported on CPU"):
            jax.jit(fn).lower(*args).compile()


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_and_jvp_do_not_imply_identical_vjp_accumulation(family):
    from tests.full_layers.evidence import verify_derivatives

    rows = verify_derivatives(family)
    assert [row["actual"] for row in rows] == ["equivalent", "equivalent", "different"]
    assert all(row["numerical_agreement"] for row in rows)


@pytest.mark.parametrize("profile", (PROFILES[0], PROFILES[3]), ids=lambda p: p.name)
def test_complete_indexer_is_retained_even_when_outputs_do_not_use_it(profile):
    from tests.full_layers.evidence import verify_indexer

    rows = verify_indexer(profile)
    assert [row["actual"] for row in rows] == ["equivalent", "different"]
    assert all(row["numerical_agreement"] for row in rows)


def test_original_moe_executes_in_cpu_interpret_and_matches_structurally():
    from tests.full_layers.blocked import original_moe
    from tests.full_layers.oracles import moe
    from tests.sglang_layers.runner import close

    mesh = cpu_mesh()
    with jax.set_mesh(mesh), original_moe(mesh) as (fn, args):
        graph = jax.make_jaxpr(fn)(*args)
        assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT
        close(jax.jit(fn)(*args), moe(args), low_precision=True, oracle=True)
