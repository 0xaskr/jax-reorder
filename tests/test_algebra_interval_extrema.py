"""Independent exact oracles for interval merging and finite extrema queries."""

from itertools import product

import numpy as np
import pytest
import sympy as sp

from jax_reorder import ComparisonBudgetError, MathematicalUndecidedError
from jax_reorder.algebra.budget import Budget
from jax_reorder.algebra.expression import SymbolicTensor, coordinates
from jax_reorder.algebra.extrema import FlatMax, maximum, minimum
from jax_reorder.algebra.pallas_domains import lower_padding_domains
from jax_reorder.algebra.symbolic import SymbolicBuilder
from jax_reorder.algebra.sympy_ops import compact_sums, equal


def test_overlapping_intervals_match_an_independent_signed_coefficient_oracle():
    k = sp.Symbol("k", integer=True)
    x = sp.IndexedBase("x", shape=(9,), real=True)
    for a, b, c in product(range(1, 6), range(1, 6), (-2, -1, 1, 2)):
        original = sp.Sum(x[k], (k, 0, a)) + c * sp.Sum(x[k], (k, 2, b + 2))
        lowered = compact_sums(original, Budget(1000)).doit()
        for lane in range(9):
            expected = int(lane <= a) + c * int(2 <= lane <= b + 2)
            assert lowered.coeff(x[lane]) == expected


def test_large_prefix_difference_only_expands_its_actual_residual():
    k = sp.Symbol("k", integer=True)
    x = sp.IndexedBase("x", shape=(100000,), real=True)
    expression = sp.Sum(x[k], (k, 0, 99999)) - sp.Sum(x[k], (k, 0, 99998))
    assert compact_sums(expression, Budget(64)) == x[99999]
    with pytest.raises(ComparisonBudgetError):
        compact_sums(expression, Budget(1))


def test_partial_function_intervals_never_acquire_prefix_points():
    k = sp.Symbol("k", integer=True)
    for function in (sp.log(k - 2), 1 / (k - 2), sp.sqrt(k - 2)):
        expression = sp.Sum(function, (k, 3, 8)) - sp.Sum(function, (k, 5, 8))
        lowered = compact_sums(expression, Budget(100))
        assert all(
            low >= 3 for summation in lowered.atoms(sp.Sum) for _, low, _ in summation.limits
        )
        assert sp.simplify(lowered.doit() - sum(function.subs(k, i) for i in (3, 4))) == 0


def test_initialization_partition_matches_exhaustive_lane_truth():
    k = sp.Symbol("k", integer=True)
    (axis,) = coordinates((7,))
    for coefficient, limit, valid in product((-2, -1, 1, 2), range(-2, 7), (2, 4)):
        address = k + 3 * sp.floor(axis / 3)
        mask = coefficient * address < limit
        condition = sp.Or(sp.Not(mask), address < valid)
        obligation = sp.Eq(sp.Product(sp.Piecewise((1, condition), (0, True)), (k, 0, 4)), 1)
        tensor = SymbolicTensor((7,), np.dtype("float32"), sp.S.Zero, memory_validity=obligation)
        lowered = lower_padding_domains(SymbolicBuilder(10000), tensor)
        for point in range(7):
            expected = all(
                not coefficient * (lane + 3 * (point // 3)) < limit
                or lane + 3 * (point // 3) < valid
                for lane in range(5)
            )
            assert bool(lowered.memory_validity.subs(axis, point)) is expected


@pytest.mark.parametrize("function", [maximum, minimum])
def test_large_extrema_preserve_flattening_constants_and_reconstruction(function):
    symbols = sp.symbols("x:40", real=True)
    node = function(*symbols, -3, 7, function(*symbols[:35]))
    assert set(node.args) == set(symbols) | {sp.Integer(7 if function is maximum else -3)}
    replacement = {symbol: sp.Integer(i - 20) for i, symbol in enumerate(symbols)}
    assert node.xreplace(replacement) == (19 if function is maximum else -20)


def test_extrema_abstraction_does_not_invent_negative_answers():
    symbols = sp.symbols("x:40", real=True)
    shifted = FlatMax(*(symbol + 1 for symbol in symbols))
    original = FlatMax(*symbols)
    assert equal(shifted, original + 1, Budget(10000)) is None
    # An independent exact assignment can disprove a changed leaf set.
    assert equal(original, FlatMax(*symbols, 0), Budget(10000)) is False
    with pytest.raises(ComparisonBudgetError):
        equal(original, FlatMax(*symbols, 0), Budget(8))


def test_extremum_witness_does_not_violate_symbol_or_query_assumptions():
    positive = sp.symbols("p:40", positive=True)
    node = FlatMax(*positive)
    assert equal(node, FlatMax(*positive, 0), Budget(10000)) is None
    symbols = sp.symbols("x:40", real=True)
    node = FlatMax(*symbols)
    assert equal(node, FlatMax(*symbols, 0), Budget(10000), assumptions=symbols[0] > 0) is None


def test_mixed_plain_and_flat_extrema_share_exact_operand_identity():
    symbols = sp.symbols("x:40", real=True)
    flat = FlatMax(*symbols, 0)
    plain = sp.Max(sp.Max(*symbols[:20], evaluate=False), *symbols[20:], 0, evaluate=False)
    assert equal(flat, plain, Budget(10000)) is True
    assert equal(flat, sp.Max(*symbols, 1, evaluate=False), Budget(10000)) is False


def test_interval_merging_preserves_nested_binding_and_distinct_integrands():
    k, lane = sp.symbols("k lane", integer=True)
    x = sp.IndexedBase("x", shape=(3, 3), real=True)
    original = sp.Sum(sp.Sum(x[k, lane], (k, 0, 2)) - sp.Sum(x[lane, k], (k, 1, 2)), (lane, 0, 2))
    lowered = compact_sums(original, Budget(1000)).doit()
    for row, column in product(range(3), repeat=2):
        assert lowered.coeff(x[row, column]) == int(column == 0)
    original = sp.Sum(x[k, 0], (k, 0, 2)) - sp.Sum(x[k, 1], (k, 0, 2))
    lowered = compact_sums(original, Budget(1000)).doit()
    for row in range(3):
        assert lowered.coeff(x[row, 0]) == 1
        assert lowered.coeff(x[row, 1]) == -1


@pytest.mark.parametrize("periodic", [True, False])
def test_unproved_boolean_piecewise_coverage_is_undecided_not_internal(periodic):
    k = sp.Symbol("k", integer=True)
    condition = sp.And(k < 384, sp.Mod(k, 512) < 384) if periodic else k < 0
    expression = sp.Piecewise((sp.true, condition), (sp.false, k >= 384))
    with pytest.raises(MathematicalUndecidedError, match="coverage") as error:
        equal(expression, sp.true, Budget(10000))
    assert error.value.path == "algebra.conditions"


@pytest.mark.parametrize("contraction", [sp.Sum, sp.Product])
@pytest.mark.parametrize("extremum", [maximum, minimum])
def test_extrema_congruence_never_erases_contraction_index_dependence(contraction, extremum):
    k = sp.Symbol("k", integer=True)
    x = sp.IndexedBase("x", shape=(4, 40), real=True)
    term = extremum(*(x[k, column] for column in range(40)))
    left, right = contraction(term, (k, 0, 1)), contraction(term, (k, 2, 3))
    assert equal(left, right, Budget(10000)) is None
    expanded_left, expanded_right = left.doit(), right.doit()
    assignment = {
        read: read.indices[0] + (1 if contraction is sp.Product else 0)
        for read in expanded_left.atoms(sp.Indexed) | expanded_right.atoms(sp.Indexed)
    }
    expected = (2, 12) if contraction is sp.Product else (1, 5)
    assert (expanded_left.xreplace(assignment), expanded_right.xreplace(assignment)) == expected


@pytest.mark.parametrize("context", ["derivative", "integral", "limit", "subs", "lambda"])
def test_extrema_abstraction_preserves_other_variable_dependent_contexts(context):
    x = sp.Symbol("x", real=True)
    leaves = sp.symbols("a:39", real=True)
    term = maximum(x, *leaves)
    left, right = {
        "derivative": (sp.Derivative(term, x, evaluate=False), sp.S.Zero),
        "integral": (sp.Integral(term, (x, 0, 1)), sp.Integral(term, (x, 1, 2))),
        "limit": (sp.Limit(term, x, 1), sp.Limit(term, x, 2)),
        "subs": (sp.Subs(term, x, 1), sp.Subs(term, x, 2)),
        "lambda": (sp.Lambda(x, term), sp.Lambda(x, sp.S.Zero)),
    }[context]
    assert equal(left, right, Budget(10000)) is None
    # Setting all a_i to zero gives the ordinary exact function Max(x, 0).
    a, b = (value.xreplace({term: sp.Max(x, 0)}).doit() for value in (left, right))
    if context == "lambda":
        assert a(1) != b(1)
    else:
        assert a.subs(x, 1) != b.subs(x, 1)


def test_extrema_lambda_keeps_parameter_binding_identity():
    x, y = sp.symbols("x y", real=True)
    term = maximum(x, *sp.symbols("a:39", real=True))
    left, right = sp.Lambda(x, term), sp.Lambda(y, term)
    assert equal(left, right, Budget(10000)) is None
    # a_i = 0, free x = 1, actual function argument = 2.
    a, b = (value.xreplace({term: sp.Max(x, 0)})(2).subs(x, 1) for value in (left, right))
    assert (a, b) == (2, 1)
