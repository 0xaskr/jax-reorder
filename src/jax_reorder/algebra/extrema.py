"""Exact finite extrema without quadratic symbolic order searches.

Large extrema retain every distinct operand. Only associativity, idempotence,
identity elements and exact rational constant order are used in construction.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import sympy as sp


def _arguments(function: type[sp.Max] | type[sp.Min], values: Iterable[sp.Basic]) -> list[sp.Basic]:
    flat = []
    for value in values:
        value = sp.sympify(value)
        flat.extend(value.args if isinstance(value, function) else (value,))
    constants = [v for v in flat if v.is_Rational or v in (sp.oo, -sp.oo)]
    others = [v for v in flat if not v.is_Rational and v not in (sp.oo, -sp.oo)]
    if constants:
        constant = function(*constants)
        if constant == function.zero:
            return [constant]
        if constant != function.identity:
            others.append(constant)
    return others


class FlatMax(sp.Max):
    """SymPy Max with exact flattening but no pairwise symbolic comparisons."""

    def __new__(cls, *args: sp.Basic, **kwargs: Any) -> sp.Basic:
        return _flat(cls, sp.Max, args)


class FlatMin(sp.Min):
    """SymPy Min with the same bounded construction policy as FlatMax."""

    def __new__(cls, *args: sp.Basic, **kwargs: Any) -> sp.Basic:
        return _flat(cls, sp.Min, args)


def _flat(
    cls: type[sp.Max] | type[sp.Min],
    kind: type[sp.Max] | type[sp.Min],
    args: Iterable[sp.Basic],
) -> sp.Basic:
    """Use SymPy's exact argument set/order with memoized DAG node counts.

    MinMaxBase(evaluate=False) orders operands by recursive tree size before
    default_sort_key. Shared paged-address subexpressions make that naive tree
    traversal exponential. Memoization computes the same sizes and order,
    preserving the complete argument set and the standard MinMaxBase layout.
    """
    values = frozenset(_arguments(kind, args))
    if not values:
        return kind.identity
    if len(values) == 1:
        return next(iter(values))
    counts: dict[sp.Basic, int] = {}

    def count(node: sp.Basic) -> int:
        if node not in counts:
            # SymPy counts Float as one half; scale all sizes by two to keep
            # exact integer keys while retaining precisely the same ordering.
            counts[node] = 1 if node.is_Float else 2 + sum(count(arg) for arg in node.args)
        return counts[node]

    ordered = sp.ordered(values, keys=[count, sp.default_sort_key], default=False)
    result = sp.Expr.__new__(cls, *ordered)
    result._argset = values
    return result


def maximum(*args: sp.Basic) -> sp.Basic:
    flat = _arguments(sp.Max, args)
    return FlatMax(*flat) if len(flat) > 32 else sp.Max(*flat)


def minimum(*args: sp.Basic) -> sp.Basic:
    flat = _arguments(sp.Min, args)
    return FlatMin(*flat) if len(flat) > 32 else sp.Min(*flat)


def distinct_extrema(left: sp.Basic, right: sp.Basic) -> bool:
    """Find an exact rational counterexample for independent real leaf extrema.

    Used after concrete input reads become independent symbols. The returned
    fact is a verified witness, never an inference from unequal operand sets.
    """
    if not any(isinstance(left, kind) and isinstance(right, kind) for kind in (sp.Max, sp.Min)):
        return False
    operands = set(left.args) | set(right.args)
    if any(
        not value.is_Rational
        and not (
            isinstance(value, sp.Symbol)
            and value.assumptions0 == sp.Symbol("real_leaf", real=True).assumptions0
        )
        for value in operands
    ):
        return False
    symbols = {value for value in operands if isinstance(value, sp.Symbol)}
    constants = {value for value in operands if value.is_Rational} | {sp.S.Zero}
    low, high = min(constants) - 1, max(constants) + 1
    differing = (set(left.args) ^ set(right.args)) & symbols
    assignments = [{symbol: low for symbol in symbols}, {symbol: high for symbol in symbols}]
    if differing:
        chosen = min(differing, key=sp.default_sort_key)
        assignment = {symbol: low if isinstance(left, sp.Max) else high for symbol in symbols}
        assignment[chosen] = high if isinstance(left, sp.Max) else low
        assignments.append(assignment)
    return any(left.xreplace(values) != right.xreplace(values) for values in assignments)
