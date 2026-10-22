"""Split symbolic parallel owners only where output restoration validity changes.

The caller must first prove ownership on the original complete grid. Each
rectangle below denotes a set of independent owners, not an execution order.
No tensor coordinates or complete parallel grid are expanded.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

from ..errors import UnsupportedJaxprError
from .grid_state import GridPlan
from .index_domain import AnalysisBudget, IndexExpr, bounds, integer, substitute
from .memory import (
    Expression,
    MemoryProjection,
    Predicate,
    ValueExpr,
    expressions_equal,
    scalar_type,
)
from .model import PallasCallView
from .tensor_domain import coordinate, coordinate_ranges


@dataclass(frozen=True)
class OwnerPartition:
    intervals: tuple[tuple[str, tuple[int, int]], ...]
    tails: tuple[bool, ...]


def partition_owner_ranges(
    requirements: tuple[tuple[IndexExpr, int], ...],
    ranges: dict[str, tuple[int, int]],
    axes: tuple[str, ...],
    budget: AnalysisBudget,
    path: str,
) -> tuple[OwnerPartition, ...]:
    """Cover the owner rectangle by disjoint, fixed full/tail validity classes.

    Interval subdivision keeps all already classified owners symbolic. Adjacent
    equal classes are coalesced before any kernel replay. Nonmonotone mappings
    may require more rectangles; the same explicit analysis budget bounds both
    their construction and subsequent executions.
    """
    requirements = tuple(dict.fromkeys(requirements))
    dependencies, costs = [], []
    for expr, _ in requirements:
        names: set[str] = set()
        pending = [expr]
        cost = 0
        while pending:
            budget.spend(path + ".expression")
            node = pending.pop()
            cost += 1
            if node.kind == "axis":
                names.add(str(node.value))
            pending.extend(node.args)
        dependencies.append(names)
        costs.append(cost)
    initial = tuple((name, ranges[name]) for name in axes)
    pending = [initial]
    result: list[OwnerPartition] = []
    while pending:
        rectangle = pending.pop()
        current = {**ranges, **dict(rectangle)}
        tails: list[bool] = []
        for index, (expr, tail) in enumerate(requirements):
            budget.spend(path + ".classify", costs[index])
            lo, hi = bounds(expr, current)
            if 0 <= lo <= hi < tail:
                tails.append(False)
            elif lo == hi == tail:
                tails.append(True)
            else:
                candidates = tuple(
                    (high - low, name, low, high)
                    for name, (low, high) in rectangle
                    if high > low and name in dependencies[index]
                )
                if not candidates:
                    raise UnsupportedJaxprError(
                        path, "padded output restoration cannot classify its parallel owner"
                    )
                _, name, low, high = max(candidates)
                middle = (low + high) // 2
                budget.spend(path + ".split", 2)
                for interval in ((middle + 1, high), (low, middle)):
                    pending.append(
                        tuple((key, interval if key == name else old) for key, old in rectangle)
                    )
                break
        else:
            result.append(OwnerPartition(rectangle, tuple(tails)))
    # Only complete rectangular unions are merged: all other axes and every
    # output/window validity component must agree exactly.
    changed = True
    while changed:
        changed = False
        for axis in range(len(axes)):
            groups: dict[
                tuple[tuple[bool, ...], tuple[tuple[str, tuple[int, int]], ...]],
                list[OwnerPartition],
            ] = {}
            for item in result:
                budget.spend(path + ".coalesce")
                other = item.intervals[:axis] + item.intervals[axis + 1 :]
                groups.setdefault((item.tails, other), []).append(item)
            merged = []
            for group in groups.values():
                group.sort(key=lambda item: item.intervals[axis][1])
                previous = group[0]
                for item in group[1:]:
                    name, (low, high) = previous.intervals[axis]
                    begin, end = item.intervals[axis][1]
                    if high + 1 == begin:
                        previous = OwnerPartition(
                            previous.intervals[:axis]
                            + ((name, (low, end)),)
                            + previous.intervals[axis + 1 :],
                            previous.tails,
                        )
                        changed = True
                    else:
                        merged.append(previous)
                        previous = item
                merged.append(previous)
            result = merged
    return tuple(sorted(result, key=lambda item: item.intervals))


def functionalize_partitioned_owners(
    call: PallasCallView,
    plan: GridPlan,
    *,
    budget: AnalysisBudget,
    coordinates: dict[str, IndexExpr] | None,
    axis_ranges: dict[str, tuple[int, int]] | None,
    project: Callable[[dict[str, tuple[int, int]], str], MemoryProjection],
) -> MemoryProjection | None:
    """Replay each validity class using the previously proved global plan."""
    restoring = tuple(
        j
        for j, block in enumerate(call.blocks[call.num_inputs :])
        if block.pipeline_mode is not None
        and block.pipeline_mode.revisit == "any"
        and plan.domains[call.num_inputs + j].padded
    )
    axes = tuple(
        f"program:{i}"
        for i, dim in enumerate(call.grid)
        if dim.size > 1 and dim.semantics == "parallel"
    )
    if not restoring or not axes:
        return None
    path = call.kernel.origin.path + ".parallel_restoration"
    grid = tuple(dim.size for dim in call.grid)
    ranges = {**coordinate_ranges("program", grid), **(axis_ranges or {})}
    requirements = []
    for output in restoring:
        domain = plan.domains[call.num_inputs + output]
        if plan.windows:
            mappings = tuple(window.mapping for window in plan.windows[output])
        else:
            mappings = []
            for point in product(*(range(grid[i]) for i in plan.serial_axes)):
                budget.spend(path + ".window")
                fixed = {
                    f"program:{i}": integer(k) for i, k in zip(plan.serial_axes, point, strict=True)
                }
                mappings.append(tuple(substitute(value, fixed) for value in domain.mapping))
        for mapping in dict.fromkeys(mappings):
            for size, width, value in zip(domain.shape, domain.block_shape, mapping, strict=True):
                if size % width:
                    requirements.append((substitute(value, coordinates or {}), size // width))
    partitions = partition_owner_ranges(tuple(requirements), ranges, axes, budget, path)
    if len(partitions) == 1:
        return None
    projections = []
    for partition in partitions:
        label = ",".join(f"{name}={lo}..{hi}" for name, (lo, hi) in partition.intervals)
        projections.append(
            project({**(axis_ranges or {}), **dict(partition.intervals)}, f".owners[{label}]")
        )

    def owner_predicate(output: int, partition: OwnerPartition) -> Predicate:
        def intervals(inverse: dict[str, IndexExpr]) -> tuple[Predicate, ...]:
            return tuple(
                predicate
                for name, (low, high) in partition.intervals
                if (low, high) != ranges[name]
                for predicate in (
                    Predicate("ge", (inverse[name], integer(low))),
                    Predicate("le", (inverse[name], integer(high))),
                )
            )

        domain = plan.domains[call.num_inputs + output]
        if not plan.windows:
            return Predicate("and", intervals(domain.inverse))
        pieces = []
        # Expanded plans prove a separate inverse for every serial window;
        # the unproved top-level domain cannot map output coordinates to owners.
        for window in plan.windows[output]:
            budget.spend(path + ".output_window")
            predicates = intervals(window.inverse) + tuple(
                predicate
                for dimension, span in enumerate(window.box.axes)
                for predicate in (
                    Predicate("ge", (coordinate("output", dimension), integer(span.start))),
                    Predicate("lt", (coordinate("output", dimension), integer(span.stop))),
                )
            )
            piece = Predicate("and", predicates)
            if piece not in pieces:
                pieces.append(piece)
        return Predicate("or", tuple(pieces))

    outputs = []
    for output in range(call.num_outputs):
        result: Expression = projections[-1].outputs[output]
        for partition, projection in reversed(
            tuple(zip(partitions[:-1], projections[:-1], strict=True))
        ):
            value = projection.outputs[output]
            if not expressions_equal((value,), (result,)):
                budget.spend(path + ".output_piece")
                result = ValueExpr(
                    "select_n",
                    scalar_type(call.blocks[call.num_inputs + output].array),
                    (owner_predicate(output, partition), result, value),
                )
        outputs.append(result)
    return MemoryProjection(
        tuple(outputs),
        plan.domains,
        tuple(version for proof in projections for version in proof.versions),
        tuple(origin for proof in projections for origin in proof.origins),
        (
            *dict.fromkeys(condition for proof in projections for condition in proof.conditions),
            "parallel owner intervals partition the original globally proved writer domain; "
            "each class replays its own serial history with fixed valid restoration rectangles; "
            "no ordering or shared state is introduced between parallel owners",
        ),
        tuple(transfer for proof in projections for transfer in proof.communication),
        tuple(wait for proof in projections for wait in proof.waits),
        assumptions=tuple(dict.fromkeys(a for proof in projections for a in proof.assumptions)),
    )
