"""调用私有端口可同步置换，但不能改变外部位置、绑定或位置元数据。

覆盖：
- 正例：jit 调用的输入/输出端口同步置换（reordered / output_reordered）在 float32 与
  bfloat16 下判 equivalent，对应关系可复核，原图文字不被修改，规范化图幂等且执行结果一致。
- 反例：错误输入/输出绑定给出数值见证；伪造子区域结果槽位的证书被拒绝。
- 保护：sharding/layout/donation 等调用元数据不能丢失或挪到另一个操作数；未知元数据显式
  抛 UnsupportedJaxprError。
- 前置条件：CPU 平台；example 中的调用固定单设备接口，不要求多设备。
- 有限调用形状不证明所有私有端口排列。
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core, pjit

from jax_reorder import (
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    canonicalize,
    compare_jaxprs,
)
from jax_reorder.core.correspondence import verify_correspondence
from jax_reorder.functions import compare_functions
from tests import call_bindings


@pytest.mark.parametrize("candidate", [call_bindings.reordered, call_bindings.output_reordered])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_call_input_and_output_permutations(candidate, dtype):
    args = (jnp.asarray([0.7, 1.1], dtype), jnp.asarray([-0.3, 0.2], dtype))
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (call_bindings.original, candidate))
    before = (str(a), str(b))
    report = compare_jaxprs(a, b)
    assert report.status == Status.EQUIVALENT, report
    assert report.correspondence.equations[0].outputs == (1, 0)
    assert before == (str(a), str(b))
    for raw, result in ((a, report.left), (b, report.right)):
        core.check_jaxpr(result.jaxpr)
        assert canonicalize(result.jaxpr).graph == result.graph
        for expected, actual in zip(
            core.eval_jaxpr(raw, raw.consts, *args),
            core.eval_jaxpr(result.jaxpr, result.jaxpr.consts, *args),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
    assert compare_jaxprs(b, a).status == Status.EQUIVALENT
    assert (
        compare_functions(
            call_bindings.original,
            candidate,
            *args,
            modes=("forward", "jvp", "vjp"),
            check_cpu=True,
        ).status
        == Status.EQUIVALENT
    )


@pytest.mark.parametrize("candidate", [call_bindings.wrong_input, call_bindings.wrong_output])
def test_call_wrong_binding_has_a_numerical_witness(candidate):
    args = (jnp.float32(0.7), jnp.float32(-0.3))
    report = compare_functions(call_bindings.original, candidate, *args, equivalence="structural")
    assert report.status == Status.DIFFERENT, report
    assert call_bindings.original(*args)[0] != candidate(*args)[0]


def test_call_certificate_cannot_forge_child_result_slots():
    args = (jnp.float32(0.7), jnp.float32(-0.3))
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (call_bindings.original, call_bindings.reordered))
    evidence = compare_jaxprs(a, b).correspondence
    equations = list(evidence.equations)
    equation = equations[0]
    key, child = equation.children[0]
    equations[0] = replace(equation, children=((key, replace(child, outputs=(0, 1))),))
    with pytest.raises(CorrespondenceError, match="region result binding"):
        verify_correspondence(a, b, replace(evidence, equations=tuple(equations)))


def test_call_metadata_is_not_lost_or_assigned_to_another_operand():
    args = (jnp.float32(0.7), jnp.float32(-0.3))
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (call_bindings.original, call_bindings.reordered))
    assert a.eqns[0].primitive is pjit.jit_p

    def donation(graph, slots):
        eq = graph.eqns[0]
        return graph.replace(
            eqns=[eq.replace(params={**eq.params, "donated_invars": slots}), *graph.eqns[1:]]
        )

    left = donation(a, (True, False))
    right = donation(b, (False, True))
    assert compare_jaxprs(left, right).status == Status.EQUIVALENT
    assert compare_jaxprs(left, donation(b, (True, False))).status == Status.DIFFERENT


def test_call_unknown_metadata_fails_explicitly():
    graph = jax.make_jaxpr(call_bindings.original)(jnp.float32(0.7), jnp.float32(-0.3))
    eq = graph.eqns[0]
    bad = graph.replace(
        eqns=[eq.replace(params={**eq.params, "new_semantic_flag": True}), *graph.eqns[1:]]
    )
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(bad, bad)
    result = error.value
    assert "unsupported jit parameters" in result.reason
