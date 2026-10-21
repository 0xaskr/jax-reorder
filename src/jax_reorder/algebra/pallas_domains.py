"""Budgeted local Sum-domain lowering for allocated Pallas input padding.

This constructs formulas, not equality decisions. It partitions only the known
output coordinate domain and a Sum's original integer bounds; grid execution,
addresses and memory versions have already been projected by core.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import product
from math import prod
from typing import TYPE_CHECKING

import sympy as sp
from sympy.core.relational import Relational

from .expression import SymbolicTensor, coordinates

if TYPE_CHECKING:
    from .symbolic import SymbolicBuilder


def _total_polynomial(builder: SymbolicBuilder, expression: sp.Basic) -> bool:
    """A branch may gain prefix points only when arithmetic stays total there."""
    pending, seen = [expression], set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        builder.budget.spend(path="algebra.pallas.prefix_domain")
        if isinstance(node, sp.Indexed | sp.Symbol) and node.is_finite is True:
            continue
        if node.is_Rational:
            continue
        if node.is_Add or node.is_Mul:
            pending.extend(node.args)
        elif node.is_Pow and node.exp.is_Integer and node.exp >= 0:
            pending.append(node.base)
        else:
            return False
    return True


def _split_reduction(builder: SymbolicBuilder, node: sp.Basic) -> sp.Basic:
    if not isinstance(node, (sp.Sum, sp.Product)) or len(node.limits) != 1:
        return node
    body = node.function
    variable, low, high = node.limits[0]
    if not low.is_Integer or not high.is_Integer or high < low:
        return node
    if isinstance(node, sp.Product) and body in (sp.S.Zero, sp.S.One):
        builder.budget.spend(path="algebra.pallas.masked_product_constant")
        return body
    if not isinstance(body, sp.Piecewise):
        return node
    cuts = {int(low), int(high) + 1}
    for _, condition in body.args:
        if condition.free_symbols - {variable}:
            return node
        for relation in condition.atoms(Relational):
            builder.budget.spend(path="algebra.pallas.masked_sum_guards")
            try:
                polynomial = sp.Poly(relation.lhs - relation.rhs, variable)
            except sp.PolynomialError:
                return node
            if polynomial.degree() != 1:
                return node
            coefficient, constant = polynomial.all_coeffs()
            if not coefficient.is_Rational or not constant.is_Rational:
                return node
            # All six comparison relations can change truth only at the root
            # or the next integer. Including both also isolates equality.
            root = int(sp.floor(-constant / coefficient))
            cuts.update(n for n in (root, root + 1) if low < n <= high)
        if condition not in (sp.true, sp.false) and not condition.atoms(Relational):
            return node
    boundaries = sorted(cuts)
    builder.budget.spend(len(boundaries) - 1, path="algebra.pallas.masked_sum_intervals")
    terms = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = None
        for expression, condition in body.args:
            truth = condition.xreplace({variable: sp.Integer(start)})
            if truth == sp.true:
                selected = expression
                break
            if truth != sp.false:
                return node
        if selected is None:
            return node
        if isinstance(node, sp.Product):
            if variable in selected.free_symbols:
                return node
            terms.append(selected ** (stop - start))
        elif variable not in selected.free_symbols:
            terms.append(selected * (stop - start))
        else:
            # Prefix canonicalization may read earlier coordinates before
            # canceling them. Permit it only with a complete real/memory domain;
            # a partial log/sqrt must keep the exact restricted interval.
            terms.append(
                builder.summation(
                    selected,
                    ((variable, start, stop - 1),),
                    defined=sp.true if _total_polynomial(builder, selected) else sp.false,
                )
            )
    if isinstance(node, sp.Product):
        builder.rules.add("pallas.initialization-intervals:v1")
        return sp.Mul(*terms)
    builder.rules.add("pallas.masked-sum-domains:v1")
    return sp.Add(*terms)


def _output_regions(
    builder: SymbolicBuilder,
    expression: sp.Basic,
    shape: tuple[int, ...],
    transform: Callable[[sp.Basic], sp.Basic],
) -> list[tuple[sp.Basic, sp.Basic]]:
    """Cover the complete finite output with exact constant-quotient boxes."""
    axes = coordinates(shape)
    cuts = {axis: {0, extent} for axis, extent in zip(axes, shape, strict=True) if axis != 0}
    quotients = {}
    for node in expression.atoms(sp.floor):
        for axis, extent in zip(axes, shape, strict=True):
            if axis == 0:
                continue
            coefficient = node.args[0].coeff(axis)
            if (
                not coefficient.is_Rational
                or coefficient <= 0
                or coefficient.p != 1
                or node.args[0] != coefficient * axis
            ):
                continue
            divisor = int(coefficient.q)
            builder.budget.spend((extent - 1) // divisor, path="algebra.pallas.output_domain_cuts")
            cuts[axis].update(range(divisor, extent, divisor))
            quotients[node] = (axis, divisor)
    ordered = [(axis, sorted(points)) for axis, points in cuts.items()]
    count = prod(len(points) - 1 for _, points in ordered)
    extrema_size = sum(len(node.args) for node in expression.atoms(sp.Max, sp.Min))
    builder.budget.spend(count * max(1, extrema_size), path="algebra.pallas.output_domain_boxes")
    intervals = [tuple(zip(points[:-1], points[1:], strict=True)) for _, points in ordered]
    expressions = []
    for box in product(*intervals):
        starts = dict(zip((axis for axis, _ in ordered), (span[0] for span in box), strict=True))
        substitution = {
            node: starts[axis] // divisor for node, (axis, divisor) in quotients.items()
        }
        local = expression.xreplace(substitution)
        local = transform(local)
        guard = sp.And(
            *(
                sp.And(axis >= start, axis < stop)
                for (axis, _), (start, stop) in zip(ordered, box, strict=True)
            )
        )
        expressions.append((local, guard))
    return expressions


def lower_padding_domains(builder: SymbolicBuilder, value: SymbolicTensor) -> SymbolicTensor:
    """Lower memory duties and masked values independently on exact integer boxes.

    Product guards use their original bounds; no memory obligation is removed by
    value cancellation. Partial mathematical branches retain restricted bounds.
    """
    if value.memory_validity != sp.true:
        regions = _output_regions(
            builder,
            value.memory_validity,
            value.shape,
            lambda local: local.replace(
                lambda node: isinstance(node, sp.Product),
                lambda node: _split_reduction(builder, node),
            ),
        )
        validity = (
            sp.true
            if all(local == sp.true for local, _ in regions)
            else sp.Or(*(sp.And(local, guard) for local, guard in regions))
        )
        value = replace(value, memory_validity=validity)
    if value.defined != sp.true or value.memory_validity != sp.true or not value.finite:
        return value
    expression = value.expression
    masked_sum = any(isinstance(node.function, sp.Piecewise) for node in expression.atoms(sp.Sum))
    masked_extremum = expression.has(sp.Max, sp.Min) and expression.has(sp.Piecewise)
    if not masked_sum and not masked_extremum:
        return value
    if expression.has(sp.oo, -sp.oo, sp.nan, sp.zoo):
        return value
    regions = _output_regions(
        builder,
        expression,
        value.shape,
        lambda local: local.replace(
            lambda node: isinstance(node, sp.Sum),
            lambda node: _split_reduction(builder, node),
        ),
    )
    return replace(value, expression=sp.Piecewise(*regions, (sp.S.Zero, True)))
