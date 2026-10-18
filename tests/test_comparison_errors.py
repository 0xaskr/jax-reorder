"""公共比较异常、worker 失败传输和三种成功结论的契约。

覆盖：
- 三种成功结论：结构等价、数学等价、数学不等价正常返回；不再存在 Status.UNKNOWN；
  ``different`` 只是结构差异，不宣称数学不等价。
- 异常通道：未识别 effect、函数契约不符（auto 与 structural 两种模式）、None 查询、预算
  失败、内部错误、worker 启动/崩溃/损坏响应都抛出带阶段的分类异常，不能被错标成
  unsupported；成功载荷不能编码失败；结构实现故障不能伪装成“不支持输入”。
- 恢复性：远端内部错误保留原始 traceback，且后续请求仍可继续。
- 前置条件：CPU 平台；worker 失败通过 monkeypatch 模拟，不依赖真实资源故障。
- 这些是异常契约回归，不评估数学判定的完备性。
"""

import json
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from jax_reorder import (
    ComparisonBudgetError,
    ComparisonContractError,
    ComparisonError,
    ComparisonFailureKind,
    ComparisonInternalError,
    ComparisonStage,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_jaxprs,
)
from jax_reorder.algebra.adapter.transport import freeze
from jax_reorder.algebra.trusted import (
    RemoteWorkerError,
    SympyResult,
    compare_symbolic,
    isolated_compare,
)


def graphs():
    argument = jax.ShapeDtypeStruct((), jnp.float32)
    return tuple(jax.make_jaxpr(fn)(argument) for fn in (lambda x: x + x, lambda x: 2 * x))


@pytest.mark.parametrize(
    "right,status,kind",
    [
        (lambda x: x + x, Status.EQUIVALENT, "structural"),
        (lambda x: 2 * x, Status.EQUIVALENT, "algebraic"),
        (lambda x: 3 * x, Status.INEQUIVALENT, None),
    ],
)
def test_complete_comparison_has_three_successful_outcomes(right, status, kind):
    argument = jax.ShapeDtypeStruct((), jnp.float32)
    result = compare_functions(lambda x: x + x, right, argument)
    assert (result.status, result.equivalence_kind) == (status, kind)
    assert not hasattr(Status, "UNKNOWN")


def test_structural_diagnostic_does_not_claim_mathematical_inequality():
    left, right = graphs()
    assert compare_jaxprs(left, right).status == Status.DIFFERENT
    assert compare_algebraic_jaxprs(left, right).status == Status.EQUIVALENT


def test_unrecognized_effect_raises_before_the_structural_fast_path():
    def function(x):
        jax.debug.print("{}", x)
        return x

    graph = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((), jnp.float32))
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_algebraic_jaxprs(graph, graph)
    assert error.value.stage == "structural"
    assert error.value.path.startswith("left.")
    assert error.value.failure_kind == "unsupported_semantics"
    assert isinstance(error.value.__cause__, UnsupportedJaxprError)


@pytest.mark.parametrize("equivalence", ["auto", "structural"])
def test_function_contract_failure_is_an_exception_in_both_modes(equivalence):
    with pytest.raises(ComparisonContractError, match="output_tree"):
        compare_functions(
            lambda x: (x,),
            lambda x: [x],
            jax.ShapeDtypeStruct((), jnp.float32),
            equivalence=equivalence,
        )


def test_none_queries_raise_and_preserve_output_query_records(monkeypatch):
    import jax_reorder.algebra.trusted as trusted

    original = trusted.equal

    def undecided(left, right, *args, **kwargs):
        if left == right:
            return original(left, right, *args, **kwargs)
        return None

    monkeypatch.setattr(trusted, "equal", undecided)
    with pytest.raises(MathematicalUndecidedError) as error:
        right = jax.make_jaxpr(lambda x: 3 * x)(jax.ShapeDtypeStruct((), jnp.float32))
        compare_symbolic(freeze(graphs()[0]), freeze(right))
    assert error.value.failure_kind == "cas_undecided"
    assert any(query.answer is None for query in error.value.diagnostics["queries"])
    assert error.value.stage == "algebra"


def test_budget_failure_is_not_mislabeled_as_unsupported_conversion():
    with pytest.raises(ComparisonBudgetError) as error:
        isolated_compare(*(freeze(graph) for graph in graphs()), algebra_budget=1)
    assert error.value.failure_kind == "budget_exhausted"
    assert error.value.diagnostics["limit"] == 1
    assert error.value.diagnostics["used"] <= 1
    assert isinstance(error.value.__cause__, RemoteWorkerError)


def test_remote_internal_error_retains_original_traceback_and_next_call_recovers():
    with pytest.raises(ComparisonInternalError) as error:
        isolated_compare(None, None)
    assert isinstance(error.value.__cause__, RemoteWorkerError)
    assert "AttributeError" in str(error.value.__cause__)
    assert "graph" in str(error.value.__cause__)
    assert isolated_compare(*(freeze(graph) for graph in graphs())).status == "equivalent"


@pytest.mark.parametrize("response", [b"", b"not a pickle", pickle.dumps(None)])
def test_damaged_worker_response_is_an_internal_error(monkeypatch, response):
    import jax_reorder.algebra.trusted as trusted

    process = SimpleNamespace(
        communicate=lambda *args, **kwargs: (response, b""),
        poll=lambda: 0,
        returncode=0,
    )
    monkeypatch.setattr(trusted.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(ComparisonInternalError, match="invalid worker response") as error:
        isolated_compare(None, None)
    assert error.value.__cause__ is not None


def test_worker_start_failure_retains_original_exception(monkeypatch):
    import jax_reorder.algebra.trusted as trusted

    def fail(*args, **kwargs):
        raise OSError("cannot spawn worker")

    monkeypatch.setattr(trusted.subprocess, "Popen", fail)
    with pytest.raises(ComparisonInternalError) as error:
        isolated_compare(None, None)
    assert isinstance(error.value.__cause__, OSError)


def test_worker_crash_is_not_a_mathematical_result(monkeypatch):
    import jax_reorder.algebra.trusted as trusted

    process = SimpleNamespace(
        communicate=lambda *args, **kwargs: (b"", b"worker fault"),
        poll=lambda: 3,
        returncode=3,
    )
    monkeypatch.setattr(trusted.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(ComparisonInternalError, match="worker exited 3: worker fault"):
        isolated_compare(None, None)


def test_conversion_bug_is_not_reclassified_as_an_unsupported_primitive(monkeypatch):
    import jax_reorder.algebra.adapter.transport as transport

    def fail(*args, **kwargs):
        raise TypeError("conversion bug")

    monkeypatch.setattr(transport, "freeze", fail)
    with pytest.raises(ComparisonInternalError) as error:
        compare_algebraic_jaxprs(*graphs())
    assert isinstance(error.value.__cause__, TypeError)
    assert error.value.path == "algebra.convert"


def test_success_payload_cannot_encode_a_failure():
    for status in ("unknown", "different", "error"):
        with pytest.raises(ValueError, match="invalid mathematical result status"):
            SympyResult(status)
    assert issubclass(MathematicalUndecidedError, ComparisonError)


def test_structural_implementation_failure_is_not_an_unsupported_input(monkeypatch):
    import jax_reorder.core.canonical as canonical

    def fail(*args, **kwargs):
        raise RuntimeError("broken structural checker")

    monkeypatch.setattr(canonical, "build_correspondence", fail)
    graph = graphs()[0]
    with pytest.raises(ComparisonInternalError) as error:
        compare_jaxprs(graph, graph)
    assert error.value.stage == "structural"
    assert isinstance(error.value.__cause__, RuntimeError)


def test_exception_enums_preserve_string_and_transport_compatibility():
    error = ComparisonError(
        "eqns[0]", "reason", stage="algebra", failure_kind="condition_undecided", query_count=2
    )
    assert error.stage is ComparisonStage.ALGEBRA
    assert error.failure_kind is ComparisonFailureKind.CONDITION_UNDECIDED
    assert str(error) == "algebra:eqns[0] [condition_undecided]: reason"
    fields = {"stage": error.stage, "failure_kind": error.failure_kind}
    assert json.loads(json.dumps(fields)) == {
        "stage": "algebra",
        "failure_kind": "condition_undecided",
    }
    assert pickle.loads(pickle.dumps(fields)) == fields
    assert pickle.loads(pickle.dumps(error.stage)) is ComparisonStage.ALGEBRA


def test_exception_relocation_keeps_subclass_defaults_and_diagnostics():
    class CustomError(ComparisonError):
        default_stage = ComparisonStage.TRACE
        default_kind = ComparisonFailureKind.TRACE_REJECTED

    payload = {"queries": [1, 2]}
    original = CustomError("jaxpr", "reason", payload=payload)
    relocated = original.at("left").at("forward")
    assert type(relocated) is CustomError
    assert relocated.path == "forward.left.jaxpr"
    assert original.path == "jaxpr"
    assert relocated.stage is ComparisonStage.TRACE
    assert relocated.failure_kind is ComparisonFailureKind.TRACE_REJECTED
    assert relocated.diagnostics["payload"] is payload


@pytest.mark.parametrize("field", ["stage", "failure_kind"])
def test_exception_rejects_unknown_classification(field):
    with pytest.raises(ValueError, match="not a valid"):
        ComparisonError("jaxpr", "reason", **{field: "misspelled"})
