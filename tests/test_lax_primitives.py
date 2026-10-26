"""逐原语执行与结构保留：先证明清单中的原语确实被追踪出来，再声明 CPU 覆盖。

覆盖方式（每个 lax_cases 用例各一条）：
- 正例：canonicalize 完整、规范化图与原图比较为 equivalent、重排后图仍是合法 jaxpr、
  重新 jit 执行得到相同结果（非确定性用例外只比较 shape/dtype 与有限性）；
  另有一条“独立分支换序”用例，验证原语在任何合法调度下都能匹配。
- 拒绝路径：ragged_all_to_all 与 pbroadcast 断言 XLA:CPU 明确拒绝执行，psend/precv
  断言 JAX 只在 GPU 实现；这些不计为 CPU 执行覆盖。
- 追踪消去：stage 被 JAX 0.11.2 自身消去，因此断言图中没有它。

前置条件：CPU 平台并暴露 2 个 host 设备（conftest.py）。有限用例集不证明所有
dtype/shape/参数组合，也不证明 GPU/TPU 行为；执行证据只对该锁定 JAX 版本成立。
"""

import jax
import numpy as np
import pytest
from jax._src import core
from tests.lax_cases import cases, trace_case

from jax_reorder import Status, canonicalize, compare_jaxprs

CPU_REJECTIONS = {
    "ragged_all_to_all": (jax.errors.JaxRuntimeError, "ragged-all-to-all.*not supported"),
    "pbroadcast": (NotImplementedError, "pbroadcast.*not found for platform cpu"),
}


def nested_primitives(graph):
    """递归收集图及其子图参数中出现的全部 primitive，用于核对用例实际追踪结果。"""
    found = set()
    pending = [graph]
    while pending:
        value = pending.pop()
        if isinstance(value, core.Jaxpr):
            for equation in value.eqns:
                found.add(equation.primitive)
                pending.extend(equation.params.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return found


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_lax_cpu_primitive(case):
    original = trace_case(case)
    names = {p.name for p in nested_primitives(original)}
    if case.name == "stage":
        # 公开的 stage 调用被 JAX 0.11.2 的追踪本身消去；它的抽象求值规则也不合法，
        # 因此不能算作 IR 执行覆盖。
        assert not names
    else:
        assert case.expected in names, f"fixture did not trace {case.expected}: {names}"
    normalized = canonicalize(original)
    assert compare_jaxprs(original, normalized.jaxpr).status == Status.EQUIVALENT
    core.check_jaxpr(normalized.jaxpr)
    # 用 jit 编译完整图，让控制流和命名轴环境走各自的 CPU lowering；
    # 上面已经确认目标 primitive 在追踪结果中。
    flat_args = jax.tree.leaves(case.args)
    if case.name in CPU_REJECTIONS:
        # 这是真实的 CPU 后端拒绝，不是被跳过的项目限制。
        error, message = CPU_REJECTIONS[case.name]
        with pytest.raises(error, match=message):
            jax.jit(lambda *args: core.eval_jaxpr(original, original.consts, *args))(*flat_args)
        return
    before = jax.jit(lambda *args: core.eval_jaxpr(original, original.consts, *args))(*flat_args)
    after = jax.jit(
        lambda *args: core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, *args)
    )(*flat_args)
    assert len(before) == len(after)
    for a, b in zip(before, after, strict=True):
        if isinstance(a, core.Token):
            assert isinstance(b, core.Token)
        elif case.deterministic:
            np.testing.assert_array_equal(a, b)
        else:
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))


@pytest.mark.parametrize(
    "case", [c for c in cases() if c.name not in CPU_REJECTIONS], ids=lambda case: case.name
)
def test_each_primitive_with_an_independent_branch(case):
    def before(extra, *args):
        value = case.fn(*args)
        side = jax.lax.neg(extra)
        return value, side

    def after(extra, *args):
        side = jax.lax.neg(extra)
        value = case.fn(*args)
        return value, side

    args = (np.float32(2), *case.args)
    # before/after 只交换“用例计算”与独立 neg 分支的定义顺序，依赖关系不变，
    # 因此这是重排正例而不是数值等价断言。
    left, right = jax.make_jaxpr(before)(*args), jax.make_jaxpr(after)(*args)
    comparison = compare_jaxprs(left, right)
    assert comparison.status == Status.EQUIVALENT, comparison
    assert canonicalize(comparison.right.jaxpr).graph == comparison.right.graph


def test_public_inventory_has_no_unexplained_omissions():
    from jax_reorder.core.support import public_primitives

    found = set().union(*(nested_primitives(trace_case(c)) for c in cases()))
    # stage 被 JAX 追踪消去；ragged_all_to_all 虽然出现，但它的 CPU 拒绝在上面的
    # 用例里被显式断言，因此不计为 CPU 执行。
    assert set(public_primitives()) - found == {jax.lax.stage_p}


@pytest.mark.parametrize("which", ["psend", "precv"])
def test_send_receive_are_rejected_by_jax_cpu(which):
    """psend/precv 在 CPU 上没有任何 lowering；这里断言的是 JAX 自身拒绝，不是项目跳过。"""
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices("cpu")), ("workers",))
    if which == "psend":

        def body(x):
            jax.lax.psend(x, "workers", [(0, 1)])
            return x
    else:

        def body(x):
            return jax.lax.precv(
                jax.lax.create_token(x), jax.ShapeDtypeStruct((1,), np.float32), "workers", [(0, 1)]
            )

    fn = jax.shard_map(body, mesh=mesh, in_specs=P("workers"), out_specs=P(), check_vma=False)
    with pytest.raises(NotImplementedError, match=f"{which} is currently only implemented on GPU"):
        jax.jit(fn)(np.ones(len(jax.devices("cpu")), dtype=np.float32))
