"""Bounded integer interval queries with exact source-expression path constraints."""

from __future__ import annotations

from typing import cast

import numpy as np

from .index_domain import IndexExpr
from .index_domain import bounds as index_bounds
from .memory import Expression, Predicate, ValueExpr

type Assumptions = tuple[tuple[Expression, tuple[int, int]], ...]


class _RangeLimit(Exception):
    pass


class _RangeQuery:
    def __init__(self) -> None:
        self.remaining = 20000

    def spend(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise _RangeLimit

    def check_depth(self, value: Expression) -> None:
        pending = [(value, 0)]
        depths: dict[int, int] = {}
        while pending:
            self.spend()
            item, depth = pending.pop()
            if depth > 48:
                raise _RangeLimit
            if depths.get(id(item), -1) >= depth:
                continue
            depths[id(item)] = depth
            args = (
                item.args
                if isinstance(item, IndexExpr)
                else item.operands
                if isinstance(item, Predicate | ValueExpr)
                else ()
            )
            pending.extend((child, depth + 1) for child in args)

    def same(self, left: Expression, right: Expression) -> bool:
        # Complete structural pairing, including typed operands/parameters. Two
        # unrelated expressions with the same interval are never interchangeable.
        pending = [(left, right)]
        seen = set()
        while pending:
            self.spend()
            a, b = pending.pop()
            if (id(a), id(b)) in seen:
                continue
            seen.add((id(a), id(b)))
            if type(a) is not type(b):
                return False
            if isinstance(a, ValueExpr) and isinstance(b, ValueExpr):
                if (a.operation, a.type, a.parameters, a.data, a.input) != (
                    b.operation,
                    b.type,
                    b.parameters,
                    b.data,
                    b.input,
                ):
                    return False
                aa, bb = a.operands, b.operands
            elif isinstance(a, Predicate) and isinstance(b, Predicate):
                if a.operation != b.operation:
                    return False
                aa, bb = a.operands, b.operands
            elif isinstance(a, IndexExpr) and isinstance(b, IndexExpr):
                if (a.kind, a.value) != (b.kind, b.value):
                    return False
                aa, bb = a.args, b.args
            else:
                if a != b:
                    return False
                continue
            if len(aa) != len(bb):
                return False
            pending.extend(zip(aa, bb, strict=True))
        return True

    def condition(
        self, value: Expression, ranges: dict[str, tuple[int, int]], assumptions: Assumptions = ()
    ) -> bool | None:
        """Infer a selector only from integer intervals, without changing its source expression."""
        self.spend()
        if type(value) is bool:
            return value
        if not isinstance(value, Predicate | ValueExpr):
            return None
        if isinstance(value, ValueExpr) and value.type.dtype != "bool":
            return None
        name, args = value.operation, value.operands
        if name in ("not", "and", "or", "xor"):
            parts = tuple(self.condition(v, ranges, assumptions) for v in args)
            if name == "not":
                return None if parts[0] is None else not parts[0]
            if name == "and":
                return False if False in parts else True if all(v is True for v in parts) else None
            if name == "or":
                return True if True in parts else False if all(v is False for v in parts) else None
            return parts[0] != parts[1] if None not in parts else None
        if name not in ("eq", "ne", "lt", "le", "gt", "ge"):
            return None
        if name in ("eq", "ne"):
            left, right = (self.condition(v, ranges, assumptions) for v in args)
            if left is not None and right is not None:
                return left == right if name == "eq" else left != right
        left, right = (self.bounds(v, ranges, assumptions) for v in args)
        if left is None or right is None:
            return None
        low, high = left[0] - right[1], left[1] - right[0]
        if name == "lt":
            return True if high < 0 else False if low >= 0 else None
        if name == "le":
            return True if high <= 0 else False if low > 0 else None
        if name == "gt":
            return True if low > 0 else False if high <= 0 else None
        if name == "ge":
            return True if low >= 0 else False if high < 0 else None
        equal = True if low == high == 0 else False if high < 0 or low > 0 else None
        return equal if name == "eq" or equal is None else not equal

    def assume(
        self,
        value: Expression,
        interval: tuple[int, int],
        ranges: dict[str, tuple[int, int]],
        assumptions: Assumptions,
    ) -> Assumptions | None:
        self.spend()
        previous = self.bounds(value, ranges, assumptions)
        if previous is None:
            return assumptions
        low, high = max(interval[0], previous[0]), min(interval[1], previous[1])
        if low > high:
            return None
        if (low, high) == previous:
            return assumptions
        result = ((value, (low, high)),) + tuple(
            (expression, limits)
            for expression, limits in assumptions
            if not self.same(expression, value)
        )
        # sign has a precise monotone inverse on integers. This also retains the
        # correlation between sign(x) and rem(x, d) in Python's floor-division guard.
        if isinstance(value, ValueExpr) and value.operation == "sign":
            source = value.operands[0]
            original = self.bounds(source, ranges, assumptions)
            if original is not None:
                lower = max(original[0], 1 if low > 0 else 0) if low >= 0 else original[0]
                upper = min(original[1], -1 if high < 0 else 0) if high <= 0 else original[1]
                return self.assume(source, (lower, upper), ranges, result)
        return result

    def paths(
        self,
        value: Expression,
        expected: bool,
        ranges: dict[str, tuple[int, int]],
        assumptions: Assumptions,
    ) -> tuple[Assumptions, ...]:
        """Conservative path intervals; at most eight alternatives per selector."""
        self.spend()
        known = self.condition(value, ranges, assumptions)
        if known is not None:
            return (assumptions,) if known == expected else ()
        if not isinstance(value, Predicate | ValueExpr):
            return (assumptions,)
        if isinstance(value, ValueExpr) and value.type.dtype != "bool":
            return (assumptions,)
        name, args = value.operation, value.operands
        if name == "not":
            return self.paths(args[0], not expected, ranges, assumptions)
        if name in ("and", "or"):
            conjunction = (name == "and") == expected
            cases = (assumptions,) if conjunction else ()
            for operand in args:
                if conjunction:
                    cases = tuple(
                        refined
                        for case in cases
                        for refined in self.paths(operand, expected, ranges, case)
                    )
                else:
                    cases += self.paths(operand, expected, ranges, assumptions)
                if len(cases) > 8 or any(len(case) > 8 for case in cases):
                    return (assumptions,)
            return cases
        if name in ("eq", "ne"):
            for symbolic, fixed in (args, args[::-1]):
                boolean = self.condition(fixed, ranges, assumptions)
                if boolean is not None:
                    target = boolean if (name == "eq") == expected else not boolean
                    return self.paths(symbolic, target, ranges, assumptions)
        if name not in ("eq", "ne", "lt", "le", "gt", "ge"):
            return (assumptions,)
        if not expected:
            name = {"eq": "ne", "ne": "eq", "lt": "ge", "le": "gt", "gt": "le", "ge": "lt"}[name]
        left, right = (self.bounds(v, ranges, assumptions) for v in args)
        if left is None or right is None:
            return (assumptions,)
        a, b = args
        if name in ("gt", "ge"):
            a, b, left, right = b, a, right, left
            name = "lt" if name == "gt" else "le"
        if name == "ne":
            if right[0] != right[1]:
                if left[0] != left[1]:
                    return (assumptions,)
                a, left, right = b, right, left
            excluded = right[0]
            intervals = ((left[0], excluded - 1), (excluded + 1, left[1]))
            return tuple(
                case
                for interval in intervals
                if (case := self.assume(a, interval, ranges, assumptions)) is not None
            )
        if name == "eq":
            lower, upper = max(left[0], right[0]), min(left[1], right[1])
            intervals = ((a, (lower, upper)), (b, (lower, upper)))
        else:
            gap = int(name == "lt")
            intervals = (
                (a, (left[0], min(left[1], right[1] - gap))),
                (b, (max(right[0], left[0] + gap), right[1])),
            )
        result = assumptions
        for expression, interval in intervals:
            narrowed = self.assume(expression, interval, ranges, result)
            if narrowed is None:
                return ()
            result = narrowed
        return (result,)

    def bounds(
        self, value: Expression, ranges: dict[str, tuple[int, int]], assumptions: Assumptions = ()
    ) -> tuple[int, int] | None:
        """Conservative, dtype-checked intervals derived from integer source operations."""
        self.spend()
        for expression, interval in assumptions:
            if self.same(expression, value):
                return interval
        if isinstance(value, IndexExpr):
            # index_domain.bounds recursively expands shared index DAGs. Charge
            # that exact expansion before entering the existing integer rule.
            pending = [value]
            while pending:
                self.spend()
                pending.extend(pending.pop().args)
            return index_bounds(value, ranges)
        if not isinstance(value, ValueExpr) or np.dtype(value.type.dtype).kind not in "iu":
            return None
        dtype = np.iinfo(value.type.dtype)
        fallback = (int(dtype.min), int(dtype.max))
        name = value.operation
        if name == "select_n":
            if len(value.operands) == 3:
                selected = self.condition(value.operands[0], ranges, assumptions)
                if selected is not None:
                    return self.bounds(value.operands[1 + int(selected)], ranges, assumptions)
            if len(value.operands) == 3:
                parts = tuple(
                    self.bounds(value.operands[1 + int(selected)], ranges, case)
                    for selected in (False, True)
                    for case in self.paths(value.operands[0], selected, ranges, assumptions)
                )
            else:
                parts = tuple(self.bounds(v, ranges, assumptions) for v in value.operands[1:])
            if parts and all(part is not None for part in parts):
                cases = cast(tuple[tuple[int, int], ...], parts)
                return min(p[0] for p in cases), max(p[1] for p in cases)
            return fallback
        if name not in (
            "min",
            "max",
            "clamp",
            "add",
            "sub",
            "mul",
            "neg",
            "convert_element_type",
            "rem",
            "div",
            "and",
            "sign",
        ):
            return fallback
        parts = tuple(self.bounds(v, ranges, assumptions) for v in value.operands)
        if any(part is None for part in parts):
            return fallback
        known = cast(tuple[tuple[int, int], ...], parts)
        if name == "sign":
            low, high = known[0]
            return (low > 0) - (low < 0), (high > 0) - (high < 0)
        if name == "and":
            nonnegative = [p[1] for p in known if p[0] >= 0]
            return (0, min(nonnegative)) if nonnegative else fallback
        if name == "min":
            return min(p[0] for p in known), min(p[1] for p in known)
        if name == "max":
            return max(p[0] for p in known), max(p[1] for p in known)
        if name == "clamp":
            lo, arg, hi = known
            if lo[0] == lo[1] and hi[0] == hi[1] and lo[0] <= hi[0]:
                return max(lo[0], min(arg[0], hi[0])), max(lo[0], min(arg[1], hi[0]))
            return fallback
        a = known[0]
        if name == "convert_element_type":
            interval = a
        elif name == "neg":
            interval = -a[1], -a[0]
        else:
            b = known[1]
            if name == "add":
                interval = a[0] + b[0], a[1] + b[1]
            elif name == "sub":
                interval = a[0] - b[1], a[1] - b[0]
            elif name == "mul":
                products = tuple(x * y for x in a for y in b)
                interval = min(products), max(products)
            elif name == "div":
                if b[0] != b[1] or b[0] == 0:
                    return fallback
                divisor = b[0]
                values = [
                    (abs(v) // abs(divisor)) * (-1 if (v < 0) != (divisor < 0) else 1) for v in a
                ]
                interval = min(values), max(values)
            elif b[0] == b[1] and b[0] > 0:
                extent = b[0] - 1
                interval = (
                    max(a[0], -extent) if a[0] < 0 else 0,
                    min(a[1], extent) if a[1] > 0 else 0,
                )
            else:
                return fallback
        if fallback[0] <= interval[0] <= interval[1] <= fallback[1]:
            return interval
        return fallback


def integer_value_bounds(
    value: Expression, ranges: dict[str, tuple[int, int]]
) -> tuple[int, int] | None:
    """Return conservative source bounds; exhaustion discards the entire query.

    A query has a fixed 20,000-step allowance, depth 48, and at most eight path
    alternatives per selector. The caller separately budgets any concrete
    address/loop enumeration. The source expression and its parameters survive.
    """
    query = _RangeQuery()
    try:
        query.check_depth(value)
        return query.bounds(value, ranges)
    except _RangeLimit:
        if isinstance(value, ValueExpr) and np.dtype(value.type.dtype).kind in "iu":
            limits = np.iinfo(value.type.dtype)
            return int(limits.min), int(limits.max)
        return None
