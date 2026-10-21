"""Budgeted geometric mask footprints and defined-lane proofs, without algebra."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, IndexExpr, axis, bounds, integer
from .memory import Expression, Predicate, ValueExpr, expressions_equal, truth
from .storage import Span
from .tensor_storage import Box


@dataclass(frozen=True)
class MaskedRegion:
    box: Box
    guard: Expression
    ranges: dict[str, tuple[int, int]]


def _axes(value: Expression) -> set[str]:
    result, pending, seen = set(), [value], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, IndexExpr):
            if item.kind == "axis":
                result.add(str(item.value))
            pending.extend(item.args)
        elif isinstance(item, Predicate | ValueExpr):
            pending.extend(item.operands)
    return result


def _truth(value: Expression, ranges: dict[str, tuple[int, int]]) -> bool | None:
    cache: dict[int, bool | None] = {}

    def visit(item: Expression) -> bool | None:
        if id(item) not in cache:
            if (
                isinstance(item, Predicate | ValueExpr)
                and item.operation in ("and", "or", "not")
                and (isinstance(item, Predicate) or item.type.dtype == "bool")
            ):
                parts = tuple(visit(child) for child in item.operands)
                if item.operation == "not":
                    result = None if parts[0] is None else not parts[0]
                elif item.operation == "and":
                    result = (
                        False
                        if False in parts
                        else True
                        if all(part is True for part in parts)
                        else None
                    )
                else:
                    result = (
                        True
                        if True in parts
                        else False
                        if all(part is False for part in parts)
                        else None
                    )
            else:
                result = truth(item, ranges)
            cache[id(item)] = result
        return cache[id(item)]

    return visit(value)


def _geometric(value: Expression) -> bool:
    cache: dict[int, bool] = {}

    def visit(item: Expression) -> bool:
        if id(item) not in cache:
            if isinstance(item, ValueExpr):
                result = False
            elif isinstance(item, Predicate):
                result = all(visit(child) for child in item.operands)
            else:
                result = isinstance(item, IndexExpr) or type(item) is bool
            cache[id(item)] = result
        return cache[id(item)]

    return visit(value)


def _static_envelope(value: Expression) -> IndexExpr | Predicate | bool:
    cache: dict[int, IndexExpr | Predicate | bool] = {}

    def visit(item: Expression) -> IndexExpr | Predicate | bool:
        if id(item) not in cache:
            result: IndexExpr | Predicate | bool
            if isinstance(item, Predicate | ValueExpr) and item.operation in ("and", "or"):
                result = Predicate(item.operation, tuple(visit(child) for child in item.operands))
            elif _geometric(item):
                result = cast(IndexExpr | Predicate | bool, item)
            else:
                result = True
            cache[id(item)] = result
        return cache[id(item)]

    return visit(value)


def _split(
    value: Expression,
    ranges: dict[str, tuple[int, int]],
    path: str,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    unresolved_cache: dict[int, set[str]] = {}

    def unresolved(item: Expression) -> set[str]:
        if id(item) not in unresolved_cache:
            if _truth(item, ranges) is not None:
                result: set[str] = set()
            elif isinstance(item, Predicate) and item.operation in ("and", "or", "not"):
                result = set().union(*(unresolved(child) for child in item.operands))
            else:
                result = _axes(item)
            unresolved_cache[id(item)] = result
        return unresolved_cache[id(item)]

    candidates = [
        key for key in unresolved(value) if key in ranges and ranges[key][0] < ranges[key][1]
    ]
    if not candidates:
        raise UnsupportedJaxprError(path, "masked access needs an exact static geometric predicate")
    key = max(
        candidates,
        key=lambda k: (k.startswith(("program:", "device:")), ranges[k][1] - ranges[k][0], k),
    )
    low, high = ranges[key]
    middle = (low + high) // 2
    return {**ranges, key: (low, middle)}, {**ranges, key: (middle + 1, high)}


def range_guard(
    ranges: dict[str, tuple[int, int]],
    initial: dict[str, tuple[int, int]],
    *,
    include_local: bool = False,
) -> Expression:
    terms = []
    for key, (low, high) in sorted(ranges.items()):
        if (include_local or not key.startswith("local:")) and initial.get(key) != (low, high):
            terms.extend(
                (
                    Predicate("ge", (axis(key), integer(low))),
                    Predicate("le", (axis(key), integer(high))),
                )
            )
    return Predicate("and", tuple(terms)) if terms else True


def mask_regions(
    mask: Expression,
    boxes: tuple[Box, ...],
    ranges: dict[str, tuple[int, int]],
    budget: AnalysisBudget,
    path: str,
) -> tuple[MaskedRegion, ...]:
    """Return exactly active storage rectangles and their nonlocal owner guards."""
    if mask is True:
        # Unmasked accesses already have exact, budgeted view footprints. They
        # need no predicate partition or coalescing proof; retain those boxes.
        return tuple(
            MaskedRegion(
                box,
                True,
                {
                    **ranges,
                    **{
                        f"local:{i}": (span.start, span.stop - 1) for i, span in enumerate(box.axes)
                    },
                },
            )
            for box in boxes
        )
    result = []
    for box in boxes:
        initial = {
            **ranges,
            **{f"local:{i}": (span.start, span.stop - 1) for i, span in enumerate(box.axes)},
        }
        geometric = _geometric(mask)
        envelope = mask if geometric else _static_envelope(mask)
        pending = [initial]
        while pending:
            current = pending.pop()
            budget.spend(path + ".mask_regions")
            known = _truth(envelope, current)
            if known is False:
                continue
            if known is None:
                pending.extend(reversed(_split(envelope, current, path)))
                continue
            selected = Box(
                tuple(
                    Span(current[f"local:{i}"][0], current[f"local:{i}"][1] + 1)
                    for i in range(len(box.axes))
                )
            )
            guard = range_guard(current, initial)
            if not geometric:
                guard = (
                    mask
                    if guard is True
                    else (
                        ValueExpr("and", mask.type, (guard, mask))
                        if isinstance(mask, ValueExpr)
                        else Predicate(
                            "and", cast(tuple[IndexExpr | Predicate | bool, ...], (guard, mask))
                        )
                    )
                )
            result.append(MaskedRegion(selected, guard, current))
    # Binary proof cuts are not memory boundaries. Coalesce adjacent rectangles
    # for the same owner domain before charging storage/version work.
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(result):
            for j in range(i + 1, len(result)):
                budget.spend(path + ".mask_merge")
                right = result[j]
                if not expressions_equal((left.guard,), (right.guard,)):
                    continue
                different = [
                    d
                    for d, (a, b) in enumerate(zip(left.box.axes, right.box.axes, strict=True))
                    if a != b
                ]
                if len(different) != 1:
                    continue
                d = different[0]
                a, b = left.box.axes[d], right.box.axes[d]
                if a.stop != b.start and b.stop != a.start:
                    continue
                merged = Span(min(a.start, b.start), max(a.stop, b.stop))
                axes = list(left.box.axes)
                axes[d] = merged
                result[i] = MaskedRegion(
                    Box(tuple(axes)),
                    left.guard,
                    {**left.ranges, f"local:{d}": (merged.start, merged.stop - 1)},
                )
                result.pop(j)
                changed = True
                break
            if changed:
                break
    return tuple(result)


def require_true(
    value: Expression, ranges: dict[str, tuple[int, int]], budget: AnalysisBudget, path: str
) -> None:
    pending = [ranges]
    while pending:
        current = pending.pop()
        budget.spend(path + ".valid_lanes")
        known = _truth(value, current)
        if known is True:
            continue
        if known is False:
            raise UnsupportedJaxprError(path, "tensor access has an unguarded out-of-bounds tail")
        pending.extend(_split(value, current, path))


def require_defined(
    value: Expression, ranges: dict[str, tuple[int, int]], budget: AnalysisBudget, path: str
) -> None:
    """Prove every evaluated branch defined; select alternatives are lazy."""
    has_cache: dict[int, bool] = {}

    def has_undefined(item: Expression) -> bool:
        if id(item) not in has_cache:
            has_cache[id(item)] = isinstance(item, ValueExpr | Predicate) and (
                isinstance(item, ValueExpr)
                and item.operation == "undefined"
                or any(has_undefined(child) for child in item.operands)
            )
        return has_cache[id(item)]

    def visit(
        item: Expression,
        current: dict[str, tuple[int, int]],
        assumptions: tuple[tuple[Expression, int], ...],
    ) -> None:
        if not has_undefined(item):
            return
        budget.spend(path + ".defined_lanes")
        # A static subdomain can make a formerly unknown mixed predicate
        # contradict a branch assumption; that branch has no consumer lanes.
        if any(
            (known := _truth(assumed, current)) is not None and int(known) != selected
            for assumed, selected in assumptions
        ):
            return
        if not isinstance(item, ValueExpr | Predicate):
            return
        if isinstance(item, ValueExpr):
            if item.operation == "undefined":
                raise UnsupportedJaxprError(
                    path, "undefined masked lane or uninitialized tensor output/scratch escapes"
                )
            if item.operation == "select_n":
                selector = item.operands[0]
                visit(selector, current, assumptions)
                for assumed, selected in assumptions:
                    budget.spend(path + ".selector_assumption")
                    if expressions_equal((selector,), (assumed,)):
                        visit(item.operands[1 + selected], current, assumptions)
                        return
                known = _truth(selector, current)
                if known is not None:
                    visit(item.operands[1 + int(known)], current, assumptions)
                    return
                if isinstance(selector, IndexExpr):
                    low, high = bounds(selector, current)
                    if low == high and 0 <= low < len(item.operands) - 1:
                        visit(item.operands[1 + low], current, assumptions)
                        return
                if _geometric(selector):
                    for part in _split(selector, current, path):
                        visit(item, part, assumptions)
                else:
                    # The exact same pure selector has the same value in each
                    # nested select. No relation between different data masks
                    # is assumed, and every possible alternative is checked.
                    for index, branch in enumerate(item.operands[1:]):
                        visit(branch, current, assumptions + ((selector, index),))
                return
            if item.operation.startswith("tensor."):
                parameters = {p.name: p.value for p in item.parameters}
                current = dict(current)
                for names, shape in zip(
                    cast(tuple[tuple[str, ...], ...], parameters["__input_axes"]),
                    cast(tuple[tuple[int, ...], ...], parameters["__input_shapes"]),
                    strict=True,
                ):
                    current.update(
                        (name, (0, size - 1)) for name, size in zip(names, shape, strict=True)
                    )
        for child in item.operands:
            visit(child, current, assumptions)

    visit(value, ranges, ())


def resolve_defined(
    value: Expression, ranges: dict[str, tuple[int, int]], budget: AnalysisBudget, path: str
) -> Expression:
    """Remove only markers proven unreachable on the entire consumer domain."""
    require_defined(value, ranges, budget, path)
    cache: dict[int, Expression] = {}

    def clean(item: Expression) -> Expression:
        if id(item) not in cache:
            if isinstance(item, ValueExpr) and item.operation == "undefined":
                # This arbitrary replacement is made only after proving the
                # branch unreachable, never to initialize a memory lane.
                result: Expression = ValueExpr(
                    "literal", item.type, data=np.zeros((), dtype=item.type.dtype).tobytes()
                )
            elif isinstance(item, ValueExpr | Predicate):
                result = replace(item, operands=tuple(clean(child) for child in item.operands))
            else:
                result = item
            cache[id(item)] = result
        return cache[id(item)]

    return clean(value)
