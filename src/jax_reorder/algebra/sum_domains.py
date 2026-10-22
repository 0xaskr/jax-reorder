"""Exact mixed-radix bijections for independent finite summation domains.

This is a coordinate change, not an expansion of the summands. Every reference
to a collapsed coordinate follows its quotient/remainder inverse; offset reads,
strides, masks and scalar uses therefore retain their original meaning.
"""

from __future__ import annotations

import sympy as sp

from .budget import Budget
from .expression import finite_sum, substitute


def compact_domain(node: sp.Sum, budget: Budget) -> sp.Sum:
    """Collapse axes joined by an exact mixed-radix input coordinate."""
    while len(node.limits) > 1:
        limits = {v: (start, stop) for v, start, stop in node.limits}
        if len(limits) != len(node.limits) or any(
            not start.is_Integer or not stop.is_Integer or stop < start
            for start, stop in limits.values()
        ):
            return node
        candidates = []
        for read in sorted(node.function.atoms(sp.Indexed), key=sp.default_sort_key):
            budget.spend(len(read.indices), path="algebra.sum_domains")
            for index in read.indices:
                axes = [v for v in limits if index.has(v)]
                if len(axes) < 2:
                    continue
                coefficients = {v: index.coeff(v) for v in axes}
                if any(not c.is_Integer or c <= 0 for c in coefficients.values()):
                    continue
                remainder = index - sum(c * v for v, c in coefficients.items())
                if remainder.has(*axes):
                    continue
                axes.sort(key=lambda v: int(coefficients[v]))
                width, unit = 1, coefficients[axes[0]]
                for axis in axes:
                    if coefficients[axis] != unit * width:
                        break
                    start, stop = limits[axis]
                    width *= int(stop - start + 1)
                else:
                    candidates.append(axes)
        if not candidates:
            return node
        axes = max(candidates, key=len)
        # A nested binder may shadow an outer variable in hand-built formulas.
        # Source lowering normally uses unique Dummies, but never rely on that.
        if any(
            set(axes) & {limit[0] for limit in child.limits}
            for child in node.function.atoms(sp.Sum, sp.Product)
        ):
            return node
        variable = sp.Dummy("linear", integer=True, nonnegative=True)
        weights, inverse, width = {}, {}, 1
        for axis in axes:
            start, stop = limits[axis]
            extent = int(stop - start + 1)
            weights[axis] = width
            inverse[axis] = start + sp.Mod(sp.floor(variable / width), extent)
            width *= extent
        offset = sum(weights[v] * limits[v][0] for v in axes)
        cache: dict[sp.Basic, sp.Basic] = {}

        def rewrite(
            value: sp.Basic,
            *,
            cache: dict[sp.Basic, sp.Basic] = cache,
            axes: list[sp.Symbol] = axes,
            weights: dict[sp.Symbol, int] = weights,
            variable: sp.Symbol = variable,
            offset: sp.Expr | int = offset,
        ) -> sp.Basic:
            if value in cache:
                return cache[value]
            budget.spend(path="algebra.sum_domains")
            result = value
            if value.is_Add:
                coefficients = {v: value.coeff(v) for v in axes}
                coefficient = coefficients[axes[0]]
                if (
                    coefficient.is_Integer
                    and coefficient != 0
                    and all(coefficients[v] == coefficient * weights[v] for v in axes)
                ):
                    remainder = value - sum(c * v for v, c in coefficients.items())
                    if not remainder.has(*axes):
                        result = remainder + coefficient * (variable + offset)
            if result is value and value.args:
                args = tuple(rewrite(arg) for arg in value.args)
                if args != value.args:
                    result = value.func(*args)
            cache[value] = result
            return result

        function = substitute(rewrite(node.function), inverse)
        remaining = [limit for limit in node.limits if limit[0] not in axes]
        budget.spend(len(axes), path="algebra.sum_domains")
        node = finite_sum(function, (variable, 0, width - 1), *remaining)
    return node
