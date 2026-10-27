"""故意损坏原图对应证据，验证独立校验器不能被规范化器的假匹配绕过。

覆盖：
- 正例：合法换序的对应证据可以被独立校验器接受；scan/while 嵌套区域有完整的原始对应关系，
  且子区域证据不能缺省。
- 反例：删除或重复方程映射、重命名外部输入、交换结果端口/输出顺序、改 primitive 或操作数、
  改常量字节、丢失规范化参数差异、单个分支错绑、相同操作之间的隐式顺序、省略 effect
  字段、两侧同一张畸形图、未审查的 primitive 身份、版本不符都被拒绝。
- 独立性：校验器只用两张原图和映射，不调用规范化器；该路径被 monkeypatch 成失败哨兵。
- 前置条件：CPU 平台；证据是进程内对象，不定义跨进程持久化格式。
- 这些是篡改回归，不证明规范化器对所有图都产生完备证据。
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jax_reorder import (
    CorrespondenceError,
    Status,
    compare_jaxprs,
    verify_correspondence,
)
from jax_reorder.core import canonical as implementation
from tests import region_bindings


def original(x, y):
    a = lax.sin(x)
    b = lax.cos(y)
    return a - b, a + b


def candidate(x, y):
    b = lax.cos(y)
    a = lax.sin(x)
    return a - b, a + b


def pair():
    args = (jnp.float32(0.7), jnp.float32(-0.3))
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (original, candidate))
    report = compare_jaxprs(a, b)
    assert report.status == Status.EQUIVALENT, report
    return a, b, report.correspondence


def test_verifier_uses_only_original_graphs_and_mapping(monkeypatch):
    a, b, evidence = pair()

    def forbidden(*args, **kwargs):
        pytest.fail("独立校验不应调用规范化器")

    monkeypatch.setattr(implementation, "canonicalize", forbidden)
    assert verify_correspondence(a, b, evidence) is None


def test_missing_or_duplicate_equation_mapping_is_rejected():
    a, b, evidence = pair()
    with pytest.raises(CorrespondenceError, match="missing equation"):
        verify_correspondence(a, b, replace(evidence, equations=evidence.equations[:-1]))
    repeated = (evidence.equations[0], *evidence.equations[:-1])
    with pytest.raises(CorrespondenceError, match="bijection"):
        verify_correspondence(a, b, replace(evidence, equations=repeated))


def test_external_inputs_cannot_be_renamed_by_certificate():
    a, b, evidence = pair()
    with pytest.raises(CorrespondenceError, match="external input order"):
        verify_correspondence(a, b, replace(evidence, inputs=(1, 0)))


def test_result_ports_and_output_order_are_checked():
    a, b, evidence = pair()
    with pytest.raises(CorrespondenceError, match="binding"):
        verify_correspondence(a, b.replace(outvars=b.outvars[::-1]), evidence)


def test_changed_primitive_and_changed_operand_are_checked():
    a, b, evidence = pair()
    equations = list(b.eqns)
    index = next(i for i, eq in enumerate(equations) if eq.primitive is lax.sin_p)
    equations[index] = equations[index].replace(primitive=lax.cos_p)
    with pytest.raises(CorrespondenceError, match="primitive identity"):
        verify_correspondence(a, b.replace(eqns=equations), evidence)
    equations = list(b.eqns)
    index = next(i for i, eq in enumerate(equations) if eq.primitive is lax.sub_p)
    equations[index] = equations[index].replace(invars=equations[index].invars[::-1])
    with pytest.raises(CorrespondenceError, match="binding"):
        verify_correspondence(a, b.replace(eqns=equations), evidence)


def test_constants_are_checked_against_original_bytes():
    weight = np.array([1.0, -2.0], dtype=np.float32)
    graph = jax.make_jaxpr(lambda x: x * weight)(jnp.ones(2))
    evidence = compare_jaxprs(graph, graph).correspondence
    changed = graph.replace(consts=[np.array([1.0, 2.0], np.float32)])
    with pytest.raises(CorrespondenceError, match="constant content"):
        verify_correspondence(graph, changed, evidence)


def test_equivalent_claim_is_blocked_when_normalized_parameter_difference_is_lost(monkeypatch):
    a = jax.make_jaxpr(lambda x: lax.integer_pow(x, 2))(jnp.float32(2))
    b = jax.make_jaxpr(lambda x: lax.integer_pow(x, 3))(jnp.float32(2))
    # 模拟规范图比较漏掉参数差异，独立原图校验必须阻止假阳性。
    monkeypatch.setattr(implementation, "difference", lambda *args: None)
    with pytest.raises(CorrespondenceError) as error:
        compare_jaxprs(a, b)
    result = error.value
    assert "primitive parameters differ" in result.reason
    assert result.failure_kind == "internal_error"


@pytest.mark.parametrize(
    "left,right",
    [
        (region_bindings.scan_original, region_bindings.scan_reordered),
        (region_bindings.while_original, region_bindings.while_reordered),
    ],
)
def test_nested_interfaces_have_complete_original_correspondence(left, right):
    args = tuple(map(jnp.float32, (0.7, 1.25, -0.3)))
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (left, right))
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT, result
    evidence = result.correspondence
    assert verify_correspondence(a, b, evidence) is None
    index = next(i for i, eq in enumerate(evidence.equations) if eq.children)
    equations = list(evidence.equations)
    equations[index] = replace(equations[index], children=())
    with pytest.raises(CorrespondenceError, match="child correspondence"):
        verify_correspondence(a, b, replace(evidence, equations=tuple(equations)))


def test_one_branch_wrong_binding_is_rejected():
    a = jax.make_jaxpr(lambda p, x, y: lax.cond(p, lambda: x - y, lambda: y - x))(
        True, jnp.float32(0.7), jnp.float32(-0.3)
    )
    evidence = compare_jaxprs(a, a).correspondence
    index = next(i for i, eq in enumerate(a.eqns) if eq.primitive is lax.cond_p)
    equations = list(evidence.equations)
    children = list(equations[index].children)
    key, branch = children[0]
    children[0] = (key, replace(branch, inputs=(1, 0)))
    equations[index] = replace(equations[index], children=tuple(children))
    with pytest.raises(CorrespondenceError, match="binding"):
        verify_correspondence(a, a, replace(evidence, equations=tuple(equations)))


def test_implicit_order_is_checked_even_for_identical_operations():
    def ordered(x):
        lax.dce_sink(x)
        lax.dce_sink(x)
        return x

    a = jax.make_jaxpr(ordered)(jnp.float32(1))
    result = compare_jaxprs(a, a)
    assert result.status == Status.EQUIVALENT, result
    evidence = result.correspondence
    assert len(evidence.equations) == 2
    swapped = replace(evidence, equations=evidence.equations[::-1])
    with pytest.raises(CorrespondenceError, match="implicit operation order"):
        verify_correspondence(a, a, swapped)


def test_effect_fields_cannot_be_omitted():
    def ordered(x):
        lax.dce_sink(x)
        return x

    a = jax.make_jaxpr(ordered)(jnp.float32(1))
    evidence = compare_jaxprs(a, a).correspondence
    changed = a.replace(eqns=[a.eqns[0].replace(effects=frozenset())])
    with pytest.raises(CorrespondenceError, match="equation effects"):
        verify_correspondence(a, changed, evidence)


def test_standalone_verifier_rejects_same_malformed_graph_on_both_sides():
    graph = jax.make_jaxpr(lambda x: lax.cos(lax.sin(x)))(jnp.float32(0.7))
    evidence = compare_jaxprs(graph, graph).correspondence
    malformed = graph.replace(eqns=list(reversed(graph.eqns)))
    with pytest.raises(CorrespondenceError, match="undefined value"):
        verify_correspondence(malformed, malformed, evidence)


def test_standalone_verifier_does_not_accept_unreviewed_primitive_identity():
    from jax._src import core

    graph = jax.make_jaxpr(lax.sin)(jnp.float32(0.7))
    evidence = compare_jaxprs(graph, graph).correspondence
    other = graph.replace(eqns=[graph.eqns[0].replace(primitive=core.Primitive("sin"))])
    with pytest.raises(CorrespondenceError, match="unsupported primitive"):
        verify_correspondence(other, other, evidence)


def test_standalone_verifier_checks_version(monkeypatch):
    graph = jax.make_jaxpr(lax.sin)(jnp.float32(0.7))
    evidence = compare_jaxprs(graph, graph).correspondence
    monkeypatch.setattr(jax, "__version__", "0.11.3")
    with pytest.raises(CorrespondenceError, match="requires audited"):
        verify_correspondence(graph, graph, evidence)
