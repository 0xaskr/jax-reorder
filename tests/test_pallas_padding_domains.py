"""The algebra interval transformation preserves exact integer-lane values."""

from dataclasses import replace

import numpy as np
import pytest
import sympy as sp

from jax_reorder import ComparisonBudgetError
from jax_reorder.algebra.expression import SymbolicTensor, coordinates
from jax_reorder.algebra.pallas_domains import lower_padding_domains
from jax_reorder.algebra.symbolic import SymbolicBuilder


@pytest.mark.parametrize("comparison", [sp.Lt, sp.Le, sp.Gt, sp.Ge, sp.Eq, sp.Ne])
@pytest.mark.parametrize("coefficient,offset", [(1, -3), (-1, 3), (2, -5), (-3, 4)])
def test_affine_mask_interval_split_preserves_nonzero_other(comparison, coefficient, offset):
    k = sp.Symbol("k", integer=True)
    condition = comparison(coefficient * k + offset, 0)
    expression = sp.Sum(sp.Piecewise((k**2 + 1, condition), (7, True)), (k, -2, 6))
    original = SymbolicTensor((), np.dtype("float32"), expression)
    lowered = lower_padding_domains(SymbolicBuilder(10000), original)
    expected = sum(i**2 + 1 if bool(condition.subs(k, i)) else 7 for i in range(-2, 7))
    assert lowered.expression.doit() == expected


@pytest.mark.parametrize("divisor,extent", [(2, 7), (3, 8), (5, 5), (5, 3)])
def test_output_quotient_partitions_include_last_partial_interval(divisor, extent):
    (axis,) = coordinates((extent,))
    k = sp.Symbol("k", integer=True)
    quotient = sp.floor(axis / divisor)
    expression = sp.Sum(sp.Piecewise((k + 10 * quotient, k + quotient < 3), (7, True)), (k, 0, 4))
    value = SymbolicTensor((extent,), np.dtype("float32"), expression)
    lowered = lower_padding_domains(SymbolicBuilder(10000), value)
    for i in range(extent):
        expected = sum(k + 10 * (i // divisor) if k + i // divisor < 3 else 7 for k in range(5))
        assert lowered.read((i,))[0].doit() == expected


@pytest.mark.parametrize("incomplete", ["defined", "memory", "finite"])
def test_partial_or_nonfinite_values_do_not_gain_extra_prefix_reads(incomplete):
    k = sp.Symbol("k", integer=True)
    x = sp.IndexedBase("x", shape=(8,), real=True)
    expression = sp.Sum(sp.Piecewise((sp.log(x[k + 2]), k >= 2), (0, True)), (k, 0, 3))
    value = SymbolicTensor((), np.dtype("float32"), expression)
    if incomplete == "defined":
        value = replace(value, defined=sp.And(x[4] > 0, x[5] > 0))
    elif incomplete == "memory":
        value = replace(value, memory_validity=sp.false)
    else:
        value = replace(value, finite=False)
    lowered = lower_padding_domains(SymbolicBuilder(10000), value)
    assert lowered.expression == expression
    assert lowered.defined == value.defined
    assert lowered.memory_validity == value.memory_validity


def test_data_dependent_mask_is_not_given_an_invented_interval():
    k, limit = sp.symbols("k limit", integer=True)
    expression = sp.Sum(sp.Piecewise((k, k < limit), (7, True)), (k, 0, 8))
    value = SymbolicTensor((), np.dtype("float32"), expression)
    lowered = lower_padding_domains(SymbolicBuilder(10000), value)
    assert lowered.expression == expression


def test_output_partition_budget_is_charged_before_large_allocation():
    (axis,) = coordinates((2**24,))
    k = sp.Symbol("k", integer=True)
    expression = sp.Sum(sp.Piecewise((k, k + sp.floor(axis / 2) < 8), (7, True)), (k, 0, 8))
    value = SymbolicTensor((2**24,), np.dtype("float32"), expression)
    with pytest.raises(ComparisonBudgetError, match="budget"):
        lower_padding_domains(SymbolicBuilder(32), value)


@pytest.mark.parametrize(
    "function", [lambda k: sp.log(k - 2), lambda k: 1 / (k - 2), lambda k: sp.sqrt(k - 2)]
)
def test_a_total_masked_sum_does_not_make_its_partial_branch_total(function):
    k = sp.Symbol("k", integer=True)
    expression = sp.Sum(sp.Piecewise((function(k), k >= 3), (0, True)), (k, 0, 7))
    value = SymbolicTensor((), np.dtype("float32"), expression)
    lowered = lower_padding_domains(SymbolicBuilder(10000), value)
    expected = sum(function(sp.Integer(i)) for i in range(3, 8))
    assert sp.simplify(lowered.expression.doit() - expected) == 0
    assert not lowered.expression.doit().has(sp.nan, sp.zoo, sp.oo, -sp.oo)
