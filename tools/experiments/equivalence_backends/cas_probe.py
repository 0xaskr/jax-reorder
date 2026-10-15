"""Exact symbolic CAS probes; no JAX execution, floating samples, or model arrays.

Run in an isolated environment with sympy==1.14.0 and symengine==0.14.1.
These are capability probes, not a jaxpr translator or a complete decision procedure.
"""

from __future__ import annotations

import json
from unittest.mock import patch


def record(case, status, evidence, assumptions=()):
    return {
        "case": case,
        "status": status,
        "evidence": str(evidence),
        "assumptions": list(assumptions),
    }


def reject_numeric_heuristic(*args, **kwargs):
    raise RuntimeError("Numerical equality heuristics are excluded from this probe")


def sympy_probe():
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

    difference = sp.expand((x + y) * z - (x * z + y * z))
    assert difference == 0
    results.append(record("E2", "equivalent", difference, ["finite real scalars"]))

    left = (q * keys.T) * scale
    right = (scale * q) * keys.T
    assert left == right
    results.append(
        record("E3", "equivalent", left, ["symbolic positive dimensions", "real entries"])
    )

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
    import sympy as sp

    # Expr.equals may use random substitutions and evalf internally. Exclude that
    # API; internal coefficient ordering in exact transforms may still use evalf.
    with (
        patch.object(sp.Expr, "_random", reject_numeric_heuristic),
        patch.object(sp.Expr, "equals", reject_numeric_heuristic),
    ):
        result = {"sympy": sympy_probe(), "symengine": symengine_probe()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
