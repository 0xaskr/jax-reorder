"""通信必须在多设备上执行，图标签、参数差异和数值证据分别核对。"""

import jax
import numpy as np
import pytest
from jax import lax
from jax._src import core
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from examples.collective_jaxpr import (
    AXIS,
    Case,
    cases,
    check_values,
    equations,
    independent,
    make_samples,
    map_body,
    parallel_matmul,
    reordered,
    run_case,
)
from jax_reorder import Status, canonicalize, compare_jaxprs


def cpu_mesh():
    """设备不足直接失败，防止 CI 以单设备退化结果冒充通信验证。"""
    devices = jax.devices("cpu")
    assert len(devices) >= 2, "请在启动 JAX 前设置至少两个逻辑 CPU 设备"
    return Mesh(np.array(devices), (AXIS,))


@pytest.mark.parametrize("check_vma", [False, True])
@pytest.mark.parametrize("index", range(20))
def test_communication_combinations(index, check_vma):
    mesh = cpu_mesh()
    all_cases = cases(mesh, check_vma=check_vma)
    assert len(all_cases) == 20
    # 正负输入均执行：cond 的通信分支与本地分支各有样本证据。
    run_case(all_cases[index], make_samples(mesh.size, 2, seed=0))


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.float32])
def test_independent_collectives_have_actual_reordered_equations(dtype):
    mesh = cpu_mesh()
    sample = make_samples(mesh.size, 1, seed=3, dtype=dtype)[0]
    case = Case(
        "独立通信及 pvary",
        map_body(independent, mesh),
        map_body(reordered, mesh),
        Status.EQUIVALENT,
        True,
        frozenset({"psum_invariant", "pmax", "pvary"}),
    )
    result = run_case(case, [sample])
    left = jax.make_jaxpr(case.original)(*sample)
    right = jax.make_jaxpr(case.candidate)(*sample)
    before = [eq.primitive for eq in equations(left)]
    candidate_before = [eq.primitive for eq in equations(right)]
    after = [eq.primitive for eq in equations(result.right.jaxpr)]
    assert before != candidate_before, "用例没有产生不同的原方程顺序"
    assert after != candidate_before, "规范化没有真正移动通信节点"
    # 除 jaxpr 中出现通信节点外，确认 CPU lowering 也产生了 all-reduce。
    lowered = jax.jit(case.original).lower(*sample).as_text()
    assert "stablehlo.all_reduce" in lowered


@pytest.mark.parametrize("missing_sum", [False, True])
def test_tensor_parallel_matmul_against_numpy(missing_sum):
    mesh = cpu_mesh()
    rng = np.random.default_rng(42)
    samples = [
        tuple(
            rng.normal(size=shape).astype(np.float32)
            for shape in ((2, mesh.size * 2), (mesh.size * 2, 3), (3,))
        )
        for _ in range(3)
    ]
    original = parallel_matmul(mesh)
    case = Case(
        "分片矩阵乘法",
        original,
        parallel_matmul(mesh, reverse=True, missing_sum=missing_sum),
        Status.DIFFERENT if missing_sum else Status.EQUIVALENT,
        not missing_sum,
        frozenset({"dot_general", "psum_invariant", "pvary"}),
    )
    run_case(case, samples)
    for x, weight, bias in samples:
        # 使用独立的完整矩阵乘法作参照，避免原函数与重排图同时实现错误仍通过。
        expected = np.tile(x @ weight + bias, (mesh.size, 1))
        check_values(expected, jax.jit(original)(x, weight, bias))


@pytest.mark.parametrize("change", ["axis_name", "devices", "out_specs"])
def test_shard_map_context_is_semantic(change):
    mesh = cpu_mesh()
    sample = make_samples(mesh.size, 1, seed=1)[0][0]

    def trace(target_mesh, axis, out_spec):
        fn = jax.shard_map(
            lambda x: lax.psum(x, axis),
            mesh=target_mesh,
            in_specs=P(axis, None),
            out_specs=out_spec,
        )
        return jax.make_jaxpr(fn)(sample)

    original = trace(mesh, AXIS, P(AXIS, None))
    if change == "axis_name":
        candidate = trace(Mesh(mesh.devices, ("renamed",)), "renamed", P("renamed", None))
    elif change == "devices":
        candidate = trace(Mesh(mesh.devices[::-1], (AXIS,)), AXIS, P(AXIS, None))
    else:
        candidate = trace(mesh, AXIS, P())
    result = compare_jaxprs(original, candidate)
    assert result.status == Status.DIFFERENT, result
    assert result.difference is not None


def test_pvary_axis_parameter_and_abstract_type_are_preserved():
    mesh = cpu_mesh()
    sample = make_samples(mesh.size, 1, seed=1)[0]
    graph = jax.make_jaxpr(map_body(independent, mesh))(*sample)
    normalized = canonicalize(graph)
    original_pvary = [eq for eq in equations(graph) if eq.primitive is core.pvary_p]
    normalized_pvary = [eq for eq in equations(normalized.jaxpr) if eq.primitive is core.pvary_p]
    assert original_pvary and len(original_pvary) == len(normalized_pvary)
    for a, b in zip(original_pvary, normalized_pvary, strict=True):
        assert a.params == b.params == {"axes": (AXIS,)}
        assert a.outvars[0].aval == b.outvars[0].aval
        assert AXIS in b.outvars[0].aval.mat.varying
    # 名字相同但身份不同的自定义原语，不能借新增的支持绕过检查。
    fake = core.Primitive("pvary")
    inner = graph.eqns[0].params["jaxpr"]
    forged = inner.replace(
        eqns=[
            eq.replace(primitive=fake) if eq.primitive is core.pvary_p else eq for eq in inner.eqns
        ]
    )
    outer = graph.replace(
        eqns=[graph.eqns[0].replace(params={**graph.eqns[0].params, "jaxpr": forged})]
    )
    result = compare_jaxprs(graph, outer)
    assert result.status == Status.UNKNOWN
    assert any("unsupported primitive pvary" in issue.reason for issue in result.issues)
