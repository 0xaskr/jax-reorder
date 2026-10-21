"""Exact rectangular storage partitioned by source write boundaries, not elements."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget
from .model import Origin
from .storage import Span


@dataclass(frozen=True, order=True)
class Box:
    axes: tuple[Span, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(a.size for a in self.axes)

    @property
    def size(self) -> int:
        return prod(self.shape)

    def intersection(self, other: Box) -> Box | None:
        if len(self.axes) != len(other.axes):
            raise ValueError("rectangle ranks differ")
        axes = tuple(
            Span(max(a.start, b.start), min(a.stop, b.stop))
            for a, b in zip(self.axes, other.axes, strict=True)
        )
        return Box(axes) if all(a.size > 0 for a in axes) else None

    def subtract(self, other: Box) -> tuple[Box, ...]:
        overlap = self.intersection(other)
        if overlap is None:
            return (self,)
        remaining, result = list(self.axes), []
        for i, clip in enumerate(overlap.axes):
            original = remaining[i]
            for edge in (Span(original.start, clip.start), Span(clip.stop, original.stop)):
                if edge.size > 0:
                    piece = remaining.copy()
                    piece[i] = edge
                    result.append(Box(tuple(piece)))
            remaining[i] = clip
        return tuple(result)

    def translated(self, delta: tuple[int, ...]) -> Box:
        return Box(
            tuple(Span(a.start + d, a.stop + d) for a, d in zip(self.axes, delta, strict=True))
        )

    def linear_bounds(self, shape: tuple[int, ...]) -> Span:
        strides = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
        start = sum(a.start * s for a, s in zip(self.axes, strides, strict=True))
        last = sum((a.stop - 1) * s for a, s in zip(self.axes, strides, strict=True))
        return Span(start, last + 1)


def full_box(shape: tuple[int, ...]) -> Box:
    return Box(tuple(Span(0, n) for n in shape))


@dataclass(frozen=True)
class BoxPiece[T]:
    box: Box
    value: T | None
    version: int
    origin: Origin


class TensorStorage[T]:
    def __init__(
        self, shape: tuple[int, ...], value: T | None, origin: Origin, budget: AnalysisBudget
    ) -> None:
        self.extent = full_box(shape)
        self.pieces = [BoxPiece(self.extent, value, 0, origin)]
        self.version, self.budget = 0, budget

    def read(self, box: Box, path: str, *, initialized: bool = True) -> tuple[BoxPiece[T], ...]:
        if self.extent.intersection(box) != box:
            raise UnsupportedJaxprError(path, "tensor rectangle is empty or out of bounds")
        result = []
        for piece in self.pieces:
            self.budget.spend(path)
            overlap = piece.box.intersection(box)
            if overlap is not None:
                if initialized and piece.value is None:
                    raise UnsupportedJaxprError(path, "read of uninitialized tensor output/scratch")
                result.append(BoxPiece(overlap, piece.value, piece.version, piece.origin))
        return tuple(result)

    def write(self, box: Box, value: T, origin: Origin) -> int:
        self.read(box, origin.path, initialized=False)
        self.version += 1
        result = []
        for piece in self.pieces:
            self.budget.spend(origin.path)
            result.extend(
                BoxPiece(part, piece.value, piece.version, piece.origin)
                for part in piece.box.subtract(box)
            )
        result.append(BoxPiece(box, value, self.version, origin))
        self.pieces = sorted(result, key=lambda p: p.box)
        return self.version

    def invalidate(self, origin: Origin) -> None:
        """A new staging window cannot inherit the preceding output block."""
        self.budget.spend(origin.path)
        self.version += 1
        self.pieces = [BoxPiece(self.extent, None, self.version, origin)]
