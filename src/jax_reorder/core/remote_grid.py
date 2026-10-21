"""Finite grid coordinates and independent owner streams for remote protocols."""

from dataclasses import dataclass
from itertools import product
from math import prod

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, IndexExpr, integer
from .model import PallasCallView


@dataclass(frozen=True)
class RemoteGridPlan:
    shape: tuple[int, ...]
    points: tuple[tuple[int, ...], ...]
    streams: tuple[tuple[int, ...], ...]
    previous: tuple[int | None, ...]
    following: tuple[int | None, ...]

    def coordinates(self, program: int) -> dict[str, IndexExpr]:
        return {f"program:{i}": integer(k) for i, k in enumerate(self.points[program])}

    def suffix(self, program: int) -> str:
        return f".program[{program}]" if len(self.points) > 1 else ""


def plan_remote_grid(call: PallasCallView, budget: AnalysisBudget) -> RemoteGridPlan:
    path = call.kernel.origin.path
    if not call.grid or any(
        type(g.size) is not int or g.size < 1 or g.semantics not in ("arbitrary", "parallel")
        for g in call.grid
    ):
        raise UnsupportedJaxprError(path, "remote protocol requires a positive static grid")
    shape = tuple(g.size for g in call.grid)
    budget.spend(path + ".grid_points", prod(shape) * len(shape))
    points = tuple(product(*(range(n) for n in shape)))
    parallel = tuple(i for i, g in enumerate(call.grid) if g.semantics == "parallel")
    owners: dict[tuple[int, ...], list[int]] = {}
    for program, point in enumerate(points):
        owners.setdefault(tuple(point[i] for i in parallel), []).append(program)
    streams = tuple(tuple(programs) for programs in owners.values())
    previous: list[int | None] = [None] * len(points)
    following: list[int | None] = [None] * len(points)
    for stream in streams:
        for before, after in zip(stream, stream[1:], strict=False):
            following[before] = after
            previous[after] = before
    return RemoteGridPlan(shape, points, streams, tuple(previous), tuple(following))
