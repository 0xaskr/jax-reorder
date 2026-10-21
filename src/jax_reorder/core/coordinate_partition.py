"""Exact piecewise comparison over complete rectangular output domains.

Partitions refine integer coordinate formulas only. Floating value operations
retain their order and types, and every point in every cell must be covered.
"""

from itertools import pairwise, product
from math import prod
from typing import cast

from .index_domain import AnalysisBudget, IndexExpr, add, bounds, scale
from .memory import Expression, Predicate, ValueExpr, expressions_equal
from .tensor_values import project_coordinates


def _affine(value: IndexExpr) -> tuple[str, int, int] | None:
    coefficients: dict[str, int] = {}
    offset = 0
    pending = [(value, 1)]
    while pending:
        item, factor = pending.pop()
        if item.kind == "axis":
            name = str(item.value)
            coefficients[name] = coefficients.get(name, 0) + factor
        elif item.kind == "constant":
            offset += factor * int(item.value)
        elif item.kind == "sum":
            pending.extend((child, factor) for child in item.args)
        elif item.kind == "scale":
            pending.append((item.args[0], factor * int(item.value)))
        else:
            return None
    coefficients = {name: c for name, c in coefficients.items() if c}
    if len(coefficients) != 1:
        return None
    name, coefficient = next(iter(coefficients.items()))
    return name, coefficient, offset


def equal_on_output_domain(
    left: Expression,
    right: Expression,
    shape: tuple[int, ...],
    budget: AnalysisBudget,
) -> bool:
    """Prove exact DAG equality on every cell of a finite coordinate partition."""
    ranges = {f"output:{i}": (0, n - 1) for i, n in enumerate(shape)}
    cuts = {name: {lo, hi + 1} for name, (lo, hi) in ranges.items()}

    def cut(value: IndexExpr, threshold: int) -> None:
        affine = _affine(value)
        if affine is None:
            return
        name, factor, offset = affine
        if name not in cuts:
            return
        boundary = (
            -((offset - threshold) // factor) if factor > 0 else (threshold - offset) // factor + 1
        )
        lo, hi = ranges[name]
        if lo < boundary <= hi:
            cuts[name].add(boundary)

    pending, seen = [left, right], set()
    axes: set[str] = set()
    bound_axes = set(ranges)
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        budget.spend("normalized.partition.scan")
        if isinstance(item, IndexExpr):
            pending.extend(item.args)
            if item.kind == "axis":
                axes.add(str(item.value))
            if item.kind in ("div", "mod") and _affine(item.args[0]) is not None:
                affine = _affine(item.args[0])
                assert affine is not None
                if affine[0] not in ranges:
                    continue
                lo, hi = bounds(item.args[0], ranges)
                divisor = int(item.value)
                first, stop = lo // divisor + 1, hi // divisor + 1
                budget.spend("normalized.partition.cuts", stop - first)
                for quotient in range(first, stop):
                    cut(item.args[0], quotient * divisor)
        elif isinstance(item, Predicate | ValueExpr):
            pending.extend(item.operands)
            if isinstance(item, ValueExpr) and item.operation.startswith("tensor."):
                parameters = {p.name: p.value for p in item.parameters}
                for names in cast(tuple[tuple[str, ...], ...], parameters["__input_axes"]):
                    bound_axes.update(names)
            if (
                isinstance(item, Predicate)
                and item.operation in ("lt", "le", "gt", "ge", "eq", "ne")
                and all(isinstance(v, IndexExpr) for v in item.operands)
            ):
                a, b = item.operands
                assert isinstance(a, IndexExpr) and isinstance(b, IndexExpr)
                difference = add(a, scale(b, -1))
                if item.operation in ("lt", "ge", "eq", "ne"):
                    cut(difference, 0)
                if item.operation in ("le", "gt", "eq", "ne"):
                    cut(difference, 1)
    if not axes <= bound_axes:
        # Device and other free coordinates need their own proved domains.
        return False
    intervals = [tuple(pairwise(sorted(values))) for values in cuts.values()]
    budget.spend("normalized.partition.cells", prod(len(parts) for parts in intervals))
    for point in product(*intervals):
        cell = {name: (lo, hi - 1) for name, (lo, hi) in zip(ranges, point, strict=True)}
        a = project_coordinates(left, {}, {}, cell, budget)
        b = project_coordinates(right, {}, {}, cell, budget)
        if not expressions_equal((a,), (b,)):
            return False
    return True
