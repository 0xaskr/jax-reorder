"""精确 CAS 能力探针；不执行 JAX、不做浮点采样、不构造模型数组。

运行（隔离环境，见 docs/exec-plans/completed/research/equivalence-backends/cas.md）：
  /tmp/jax-reorder-equivalence-eval/cas/venv/bin/python \
      tools/experiments/equivalence_backends/cas_probe.py \
      > /tmp/jax-reorder-equivalence-eval/cas/result.json

在 SymPy 1.14.0 与 SymEngine 0.14.1 上验证标量恒等式、符号矩阵/指标求和、
分段条件、负例多项式与域义务。它是能力与边界探针，不是 jaxpr 转换器或
完备判定器：未化零不等于不等价，有限维反例也不构成任意维度正例。输出
JSON 到 stdout；`Expr.equals` 与随机探测被显式禁用，数值启发式不参与证明。
"""

from __future__ import annotations

import json
from unittest.mock import patch


def record(case, status, evidence, assumptions=()):
    """构造一条探针记录：case、status、evidence 与 assumptions。"""
    return {
        "case": case,
        "status": status,
        "evidence": str(evidence),
        "assumptions": list(assumptions),
    }


def reject_numeric_heuristic(*args, **kwargs):
    """拦截被禁用的数值启发式 API，防止采样判等混入证明。"""
    raise RuntimeError("Numerical equality heuristics are excluded from this probe")


def sympy_probe():
    """运行 SymPy 精确变换用例，返回 version 与逐 case 结果。

    覆盖分配律、矩阵缩放、指标求和转矩阵/数组、exp 恒等式、softmax
    shift、全遮蔽常量事实、负例多项式、ReLU/边界/混合导数与域义务。
    只调用确定接口（expand/cancel/Poly 等），不调用 equals 或随机采样。
    """
    import sympy as sp
    from sympy.tensor.array.expressions import (
        ArraySymbol,
        convert_indexed_to_array,
        convert_matrix_to_array,
    )

    x, y, z, scale = sp.symbols("x y z scale", real=True)
    rows, tokens, width = sp.symbols("rows tokens width", integer=True, positive=True)
    i, j, k = sp.symbols("i j k", integer=True)
    q = sp.MatrixSymbol("Q", rows, width)
    keys = sp.MatrixSymbol("K", tokens, width)
    p = sp.MatrixSymbol("P", rows, tokens)
    v = sp.MatrixSymbol("V", tokens, width)
    results = []

    # E2：分配律在独立实标量上展开后差为零。
    difference = sp.expand((x + y) * z - (x * z + y * z))
    assert difference == 0
    results.append(record("E2", "equivalent", difference, ["finite real scalars"]))

    left = (q * keys.T) * scale
    right = (scale * q) * keys.T
    assert left == right
    results.append(
        record("E3", "equivalent", left, ["symbolic positive dimensions", "real entries"])
    )

    # E4：指标求和转换为矩阵/数组表达式，不枚举元素。
    indexed = sp.Sum(p[i, j] * v[j, k], (j, 0, tokens - 1))
    as_matrix = sp.MatrixExpr.from_index_summation(indexed)
    as_array = convert_indexed_to_array(indexed, [i, k])
    assert as_matrix == p * v
    assert as_array == convert_matrix_to_array(p * v)
    results.append(
        record(
            "E4",
            "equivalent",
            {"matrix": str(as_matrix), "array": str(as_array)},
            ["symbolic positive dimensions; no element expansion", "free index order [i,k]"],
        )
    )

    # E4_batched：真实 batch 形式的上游转换在本版本报错，结论保留为 unknown。
    batches, heads = sp.symbols("batches heads", integer=True, positive=True)
    batch = sp.Symbol("batch", integer=True)
    batched_p = ArraySymbol("batched_P", (batches, heads, tokens))
    batched_v = ArraySymbol("batched_V", (batches, tokens, width))
    batched_sum = sp.Sum(batched_p[batch, i, j] * batched_v[batch, j, k], (j, 0, tokens - 1))
    try:
        converted = convert_indexed_to_array(batched_sum, [batch, i, k])
    except (ValueError, NotImplementedError) as exc:
        results.append(record("E4_batched", "unknown", f"{type(exc).__name__}: {exc}"))
    else:
        results.append(record("E4_batched", "conversion_only", converted))

    difference = sp.expand_power_exp(sp.exp(x + y)) - sp.exp(x) * sp.exp(y)
    assert difference == 0
    results.append(record("E5", "equivalent", difference, ["finite real x,y"]))

    logits = sp.IndexedBase("logits", real=True)
    shift = sp.Symbol("shift", real=True)
    denominator = sp.Sum(sp.exp(logits[j]), (j, 0, tokens - 1))
    shifted_denominator = sp.Sum(sp.exp(logits[j] - shift), (j, 0, tokens - 1))
    difference = sp.cancel(
        sp.exp(x - shift) / sp.expand_power_exp(shifted_denominator) - sp.exp(x) / denominator
    )
    assert difference == 0
    results.append(
        record(
            "E6_finite_shift",
            "equivalent_under_conditions",
            difference,
            ["positive finite token count", "all logits and shift finite real", "denominator > 0"],
        )
    )
    # E6：全遮蔽行涉及 inf/nan 常量事实，本轮未实现完整 masked-softmax 转换。
    assert sp.exp(-sp.oo) == 0
    all_masked_exponent = sp.exp(-sp.oo - (-sp.oo))
    assert all_masked_exponent is sp.nan
    outer_select = sp.Piecewise((all_masked_exponent, False), (0, True))
    assert outer_select == 0
    results.append(
        record(
            "E6_full_mask",
            "unknown",
            "exp(-oo)=0; exp(-oo-(-oo))=nan; a false outer select yields 0. "
            "No complete masked-softmax conversion or proof has been implemented.",
        )
    )

    # N1：差为非零多项式，说明独立实变量下不是恒等式。
    polynomial = sp.Poly((scale - 1) * x * y, scale, x, y, domain=sp.QQ)
    assert not polynomial.is_zero
    results.append(
        record(
            "N1",
            "inequivalent",
            polynomial.as_expr(),
            ["independent real scale,x,y; nonzero polynomial is not identically zero"],
        )
    )

    entries = sp.symbols("p00 p01 p10 p11 v00 v01 v10 v11", real=True)
    p2, v2 = sp.Matrix(2, 2, entries[:4]), sp.Matrix(2, 2, entries[4:])
    polynomial = sp.Poly((p2 * v2 - p2 * v2.T)[0, 0], *entries, domain=sp.QQ)
    assert not polynomial.is_zero
    results.append(
        record(
            "N2",
            "inequivalent",
            polynomial.as_expr(),
            ["2x2 symbolic counterexample family only; not an arbitrary-shape positive proof"],
        )
    )

    negative = sp.Symbol("negative", negative=True)
    relu_difference = sp.Max(0, negative) - negative
    assert relu_difference.is_positive is True
    results.append(record("N3", "inequivalent", relu_difference, ["negative < 0"]))

    length = sp.Symbol("length", integer=True, positive=True)
    boundary_difference = sp.Lt(length, length) != sp.Le(length, length)
    assert boundary_difference
    results.append(
        record(
            "N4",
            "inequivalent",
            "at index=length: strict mask=False, inclusive mask=True",
            ["capacity=length+1; length is a positive integer"],
        )
    )

    wrong_exp = sp.exp(x + y) - sp.exp(x) - sp.exp(y)
    mixed_derivative = sp.diff(wrong_exp, x, y)
    assert mixed_derivative.is_positive is True
    results.append(
        record(
            "N5",
            "inequivalent",
            f"mixed derivative of the difference is {mixed_derivative} > 0",
            ["real x,y; an identically zero smooth function has zero mixed derivative"],
        )
    )

    # U1：不透明算子的非零差不是不等价证据，保留 unknown。
    unknown = sp.Function("unsupported_primitive")
    remaining = sp.expand(unknown(x + y) - unknown(x) - unknown(y))
    assert remaining != 0
    results.append(
        record(
            "U1",
            "unknown",
            remaining,
            ["uninterpreted operator; a nonzero-looking residual is not inequality evidence"],
        )
    )

    # 域义务：cancel 得到 1 会丢失原式 x!=0 的前提，必须由调用方另行维护。
    quotient = sp.Mul(x, sp.Pow(x, -1, evaluate=False), evaluate=False)
    assert sp.cancel(quotient) == 1
    results.append(
        record(
            "domain_guard",
            "requires_external_guard",
            "cancel(x/x)=1 drops the original x!=0 domain obligation",
        )
    )
    return {"version": sp.__version__, "results": results}


def symengine_probe():
    """检查 SymEngine 的标量能力与张量 API 缺口，返回 version 与结果。

    E2/E5 精确为零；E3/E4 因缺少 MatrixSymbol/Sum 记为 unknown，不通过
    固定维度展开冒充符号形状支持；另检查假设传递与全遮蔽常量事实。
    """
    import symengine as se

    x, y, z = se.symbols("x y z")
    results = []
    difference = se.expand((x + y) * z - x * z - y * z)
    assert difference == 0
    results.append(record("E2", "equivalent", difference))
    difference = se.exp(x + y) - se.exp(x) * se.exp(y)
    assert difference == 0
    results.append(record("E5", "equivalent", difference, ["scalar mathematical exp"]))

    available = {name: hasattr(se, name) for name in ("MatrixSymbol", "Sum", "Piecewise", "Max")}
    for case in ("E3", "E4"):
        results.append(
            record(case, "unknown", available, ["no symbolic-shape tensor lowering in this probe"])
        )
    assumption = se.Symbol("assumed_real", real=True).is_real
    results.append(
        record(
            "assumption_transfer",
            "requires_external_guard",
            {"Symbol(real=True).is_real": assumption},
        )
    )
    assert se.exp(-se.oo) == 0
    results.append(
        record(
            "E6_full_mask",
            "unknown",
            f"exp(-oo)={se.exp(-se.oo)}, oo-oo={se.oo - se.oo}; no full mask proof",
        )
    )
    return {"version": se.__version__, "results": results}


def main():
    """在禁用随机/数值判等的补丁下运行两组探针并打印 JSON。

    无参数；副作用仅为导入隔离环境依赖与打印报告。SymPy 的 equals 可能
    内部走随机替换/evalf，这里显式禁用；这不表示所有内部计算都已被形式化。
    """
    import sympy as sp

    # Expr.equals 可能内部使用随机替换与 evalf。禁用该 API；精确变换内部
    # 为排列系数仍可能调用 evalf，这与“以浮点采样判等”是不同路径。
    with (
        patch.object(sp.Expr, "_random", reject_numeric_heuristic),
        patch.object(sp.Expr, "equals", reject_numeric_heuristic),
    ):
        result = {"sympy": sympy_probe(), "symengine": symengine_probe()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
