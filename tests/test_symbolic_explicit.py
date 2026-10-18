"""保留的显式转换器公式基线可复现；不参与生产数学判等。

覆盖：
- 23 项固定探针：默认映射连续两次调用一致；除 conditional_region 外的用例都转换为公式，
  并逐条对照独立参照；changed_coefficient 与 ordered_outputs 是对照失败的反例。
- 拒绝：区域构造（cond）不走显式公式基线，报告为 error 而不是静默跳过。
- 前置条件：纯 SymPy 1.14 转换；测试期间 monkeypatch 禁止编译与 JAX 执行，因此结果只
  来自公式转换。
- 该基线是历史转换代码的回归，不代表生产数学后端的支持范围。
"""

import jax
from jax._src import compiler, dispatch

from jax_reorder.algebra.explicit import jaxpr_to_sympy_expressions, primitive_to_sympy_op
from tools.experiments.equivalence_backends.architecture_reuse_probe import jax2sympy_probe


def test_explicit_converter_retains_the_23_case_baseline(monkeypatch):
    jax.devices()

    def forbidden(*args, **kwargs):
        raise AssertionError("conversion baseline must not execute JAX")

    monkeypatch.setattr(compiler, "backend_compile_and_load", forbidden)
    monkeypatch.setattr(dispatch, "apply_primitive", forbidden)
    report = jax2sympy_probe(convert=jaxpr_to_sympy_expressions, mappings=primitive_to_sympy_op)
    assert report["default_mapping_repeat"] == "two_calls_succeeded"
    assert len(report["records"]) == 23
    for row in report["records"]:
        if row["case"] == "conditional_region":
            # Regions use SymbolicBuilder, not the formula-only explicit baseline.
            assert row["status"] == "error"
            assert "cond" in row["error"]
        else:
            assert row["status"] == "converted", row
            expected = row["case"] not in ("changed_coefficient", "ordered_outputs")
            assert all(q["all_equal"] is expected for q in row["reference_checks"]), row
