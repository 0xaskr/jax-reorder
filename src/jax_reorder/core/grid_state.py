"""Prove output ownership and ordered reuse of a block window.

Default output reuse must be consecutive; synchronous ANY restores exact prior
windows of the same parallel owner. With persistent array scratch, a bounded
serial suffix may also traverse distinct output windows while the parallel
prefix remains symbolic. An arbitrary dimension never implies addition.
"""

from dataclasses import dataclass, replace
from itertools import product
from math import prod

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, IndexExpr, add, bounds, integer, substitute
from .model import PallasCallView
from .storage import Span
from .tensor_domain import (
    TensorDomain,
    coordinate,
    coordinate_ranges,
    mutable_prefetch,
    prove_tensor_domain,
)
from .tensor_storage import Box


@dataclass(frozen=True)
class OutputWindow:
    first: int
    stop: int
    mapping: tuple[IndexExpr, ...]
    owner: TensorDomain
    box: Box
    coordinates: tuple[tuple[str, IndexExpr], ...]

    @property
    def inverse(self) -> dict[str, IndexExpr]:
        offsets = {
            f"output:{i}": add(coordinate("output", i), integer(-span.start))
            for i, span in enumerate(self.box.axes)
        }
        return {
            **{key: substitute(value, offsets) for key, value in self.owner.inverse.items()},
            **dict(self.coordinates),
        }


@dataclass(frozen=True)
class GridPlan:
    domains: tuple[TensorDomain, ...]
    serial_axes: tuple[int, ...]
    windows: tuple[tuple[OutputWindow, ...], ...] = ()
    independent: bool = False


def _serial_axes(call: PallasCallView) -> tuple[int, ...]:
    ordered = tuple(i for i, g in enumerate(call.grid) if g.size > 1 and g.semantics == "arbitrary")
    if ordered and any(g.size > 1 and g.semantics == "parallel" for g in call.grid[ordered[0] :]):
        raise UnsupportedJaxprError(
            call.kernel.origin.path, "stateful grid requires a parallel prefix and serial suffix"
        )
    return ordered


def state_domains(
    call: PallasCallView,
    budget: AnalysisBudget,
    *,
    banked_interleaving: bool = False,
    input_mappings: dict[int, tuple[IndexExpr, ...]] | None = None,
) -> GridPlan:
    grid = tuple(g.size for g in call.grid)
    domains = tuple(
        prove_tensor_domain(
            block,
            grid,
            budget,
            writer=False,
            mapping=(input_mappings or {}).get(i) if i < call.num_inputs else None,
        )
        for i, block in enumerate(call.blocks)
    )
    if banked_interleaving:
        serial = tuple(
            i for i, g in enumerate(call.grid) if g.size > 1 and g.semantics == "arbitrary"
        )
        return _expanded_domains(call, domains, serial, budget)
    try:
        return _axis_domains(call, domains, budget)
    except UnsupportedJaxprError:
        try:
            ordered = _serial_axes(call)
        except UnsupportedJaxprError:
            # Native lowering accepts interleaved semantics. A fresh-state
            # proof of each complete program is independent of core assignment.
            # Merely replaying arbitrary axes would instead privatize scratch
            # across the intervening parallel axis, which is not justified.
            last_parallel = max(
                i for i, g in enumerate(call.grid) if g.size > 1 and g.semantics == "parallel"
            )
            suffix = tuple(
                i
                for i, g in enumerate(call.grid)
                if i > last_parallel and g.size > 1 and g.semantics == "arbitrary"
            )
            if suffix:
                # This suffix is contiguous in the original native grid. The
                # symbolic prefix can include arbitrary dimensions too, but its
                # scratch starts unknown, so no state across groups is assumed.
                return replace(_expanded_domains(call, domains, suffix, budget), independent=True)
            return independent_domains(call, budget)
        if not ordered:
            raise
        return _expanded_domains(call, domains, ordered, budget)


def independent_domains(call: PallasCallView, budget: AnalysisBudget) -> GridPlan:
    grid = tuple(g.size for g in call.grid)
    return GridPlan(
        tuple(
            prove_tensor_domain(block, grid, budget, writer=i >= call.num_inputs)
            for i, block in enumerate(call.blocks)
        ),
        (),
        independent=True,
    )


def _axis_domains(
    call: PallasCallView, domains: tuple[TensorDomain, ...], budget: AnalysisBudget
) -> GridPlan:
    grid = tuple(g.size for g in call.grid)
    serial: set[int] = set()
    outputs = []
    for block, domain in zip(
        call.blocks[call.num_inputs :], domains[call.num_inputs :], strict=True
    ):
        used = set()
        pending: list[IndexExpr] = list(domain.mapping)
        while pending:
            expr = pending.pop()
            if expr.kind == "axis" and str(expr.value).startswith("program:"):
                used.add(int(str(expr.value).split(":")[1]))
            pending.extend(expr.args)
        missing = tuple(i for i, n in enumerate(grid) if n > 1 and i not in used)
        if missing and (
            any(i > missing[0] for i in used)
            or any(call.grid[i].semantics != "arbitrary" for i in missing)
        ):
            raise UnsupportedJaxprError(
                block.origin.path, "redundant writers require a trailing serial grid suffix"
            )
        serial.update(missing)
        owner_grid = tuple(1 if i in missing else n for i, n in enumerate(grid))
        owner = prove_tensor_domain(block, owner_grid, budget, writer=True)
        if owner.mapping != domain.mapping:
            raise UnsupportedJaxprError(block.origin.path, "output window changes in serial suffix")
        outputs.append(replace(owner, grid=grid, serial_axes=missing))
    # Outputs may stop changing at different grid depths. Their individually
    # proved trailing suffixes are nested; the union retains every dependency.
    replay = tuple(sorted(serial))
    if call.scratch or mutable_prefetch(call):
        # Call-lifetime scratch also carries state between *different* output
        # windows. Replay the serial suffix while retaining symbolic owners in
        # the parallel prefix; this does not relax output staging continuity.
        replay = _serial_axes(call)
    return GridPlan(domains[: call.num_inputs] + tuple(outputs), replay)


class OutputWindowTracker:
    """Prove output coverage as each serial program binds its output windows."""

    def __init__(
        self,
        call: PallasCallView,
        domains: tuple[TensorDomain, ...],
        serial: tuple[int, ...],
        budget: AnalysisBudget,
    ) -> None:
        self.call, self.domains, self.serial, self.budget = call, domains, serial, budget
        self.grid = tuple(g.size for g in call.grid)
        self.owner_grid = tuple(1 if i in serial else n for i, n in enumerate(self.grid))
        self.ranges = coordinate_ranges("program", self.owner_grid)
        self.windows: list[list[OutputWindow]] = [[] for _ in range(call.num_outputs)]
        self.lattices: list[tuple[int, ...] | None] = [None] * call.num_outputs
        self.occupied: list[set[Box]] = [set() for _ in range(call.num_outputs)]
        self.seen_mappings: list[set[tuple[IndexExpr, ...]]] = [
            set() for _ in range(call.num_outputs)
        ]
        self.steps = 0

    def append(
        self,
        mappings: tuple[tuple[IndexExpr, ...], ...],
        coordinates: dict[str, IndexExpr],
    ) -> None:
        """Bind this program before its kernel updates any scalar-prefetch state."""
        outputs = self.domains[self.call.num_inputs :]
        for j, (block, domain, mapping) in enumerate(
            zip(self.call.blocks[self.call.num_inputs :], outputs, mappings, strict=True)
        ):
            self.budget.spend(block.origin.path + ".window")
            if self.windows[j] and self.windows[j][-1].mapping == mapping:
                self.windows[j][-1] = replace(self.windows[j][-1], stop=self.steps + 1)
                continue
            limits = tuple(bounds(v, self.ranges) for v in mapping)
            box = Box(
                tuple(
                    Span(lo * b, (hi + 1) * b)
                    for (lo, hi), b in zip(limits, domain.block_shape, strict=True)
                )
            )
            shifted = tuple(
                add(v, integer(-lo)) for v, (lo, _) in zip(mapping, limits, strict=True)
            )
            owner = prove_tensor_domain(
                replace(block, array=replace(block.array, shape=tuple(s.size for s in box.axes))),
                self.owner_grid,
                self.budget,
                writer=True,
                mapping=shifted,
            )
            # Equal-sized aligned rectangles form a disjoint lattice: hashing
            # proves overlap without comparing every pair of serial windows.
            shape = tuple(span.size for span in box.axes)
            aligned = all(span.start % span.size == 0 for span in box.axes)
            if not self.windows[j] and aligned:
                self.lattices[j] = shape
            if self.lattices[j] == shape and aligned:
                overlaps = box in self.occupied[j]
            else:
                self.lattices[j] = None
                overlaps = False
                for previous in self.windows[j]:
                    self.budget.spend(block.origin.path + ".window_overlap")
                    if box.intersection(previous.box) is not None:
                        overlaps = True
                        break
            if overlaps and not (
                block.pipeline_mode is not None
                and block.pipeline_mode.revisit == "any"
                and mapping in self.seen_mappings[j]
            ):
                raise UnsupportedJaxprError(
                    block.origin.path,
                    "output windows overlap or revisit a block nonconsecutively; "
                    "redundant writers require a trailing serial grid suffix",
                )
            self.occupied[j].add(box)
            self.seen_mappings[j].add(mapping)
            self.windows[j].append(
                OutputWindow(
                    self.steps, self.steps + 1, mapping, owner, box, tuple(coordinates.items())
                )
            )
        self.steps += 1

    def finish(self) -> tuple[tuple[OutputWindow, ...], ...]:
        path = self.call.kernel.origin.path
        if self.steps != prod(self.grid[i] for i in self.serial):
            raise UnsupportedJaxprError(path, "output window replay is incomplete")
        for domain, boxes in zip(self.domains[self.call.num_inputs :], self.occupied, strict=True):
            padded_size = prod(
                ((n + b - 1) // b) * b
                for n, b in zip(domain.shape, domain.block_shape, strict=True)
            )
            if sum(box.size for box in boxes) != padded_size:
                raise UnsupportedJaxprError(path, "expanded grid leaves an uncovered output block")
        return tuple(tuple(windows) for windows in self.windows)


def _expanded_domains(
    call: PallasCallView,
    domains: tuple[TensorDomain, ...],
    serial: tuple[int, ...],
    budget: AnalysisBudget,
) -> GridPlan:
    """Prove concrete serial windows while leaving parallel owners symbolic."""
    grid = tuple(g.size for g in call.grid)
    budget.spend(call.kernel.origin.path + ".serial_grid", prod(grid[i] for i in serial))
    tracker = OutputWindowTracker(call, domains, serial, budget)
    for point in product(*(range(grid[i]) for i in serial)):
        coordinates = {f"program:{i}": integer(k) for i, k in zip(serial, point, strict=True)}
        tracker.append(
            tuple(
                tuple(substitute(v, coordinates) for v in domain.mapping)
                for domain in domains[call.num_inputs :]
            ),
            coordinates,
        )
    return GridPlan(domains, serial, tracker.finish())
