"""Exact interval storage. Work follows source boundaries, never element counts."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget
from .model import Origin


@dataclass(frozen=True, order=True)
class Span:
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.stop and other.start < self.stop

    def contains(self, other: Span) -> bool:
        return self.start <= other.start < other.stop <= self.stop


@dataclass(frozen=True)
class VersionSlice:
    span: Span
    version: int


@dataclass(frozen=True)
class Piece[T]:
    span: Span
    value: T | None
    version: int
    origin: Origin


class Storage[T]:
    def __init__(self, size: int, value: T | None, origin: Origin, budget: AnalysisBudget) -> None:
        self.extent = Span(0, size)
        self.pieces = [Piece(self.extent, value, 0, origin)]
        self.version = 0
        self.budget = budget

    def read(self, span: Span, path: str, *, initialized: bool = True) -> tuple[Piece[T], ...]:
        if not self.extent.contains(span):
            raise UnsupportedJaxprError(path, "storage interval is empty or out of bounds")
        result = []
        for piece in self.pieces:
            # A single full-storage value remains constant-cost on the M1 path.
            if len(self.pieces) > 1:
                self.budget.spend(path)
            if piece.span.overlaps(span):
                if initialized and piece.value is None:
                    raise UnsupportedJaxprError(
                        path, "read of uninitialized output/scratch storage"
                    )
                result.append(
                    Piece(
                        Span(max(span.start, piece.span.start), min(span.stop, piece.span.stop)),
                        piece.value,
                        piece.version,
                        piece.origin,
                    )
                )
        return tuple(result)

    def versions(self, span: Span, path: str) -> tuple[VersionSlice, ...]:
        # Coalesce equal version numbers so unrelated writes outside this range
        # cannot change its version certificate merely by adding boundaries.
        result: list[VersionSlice] = []
        for piece in self.read(span, path):
            if result and result[-1].version == piece.version:
                result[-1] = VersionSlice(
                    Span(result[-1].span.start, piece.span.stop), piece.version
                )
            else:
                result.append(VersionSlice(piece.span, piece.version))
        return tuple(result)

    def write(self, span: Span, value: T, origin: Origin, *, overwrite: bool) -> int:
        covered = self.read(span, origin.path, initialized=False)
        if not overwrite and any(p.value is not None for p in covered):
            raise UnsupportedJaxprError(origin.path, "multiple writes overlap an output interval")
        self.version += 1
        result = []
        for piece in self.pieces:
            if not piece.span.overlaps(span):
                result.append(piece)
            else:
                if piece.span.start < span.start:
                    result.append(
                        Piece(
                            Span(piece.span.start, span.start),
                            piece.value,
                            piece.version,
                            piece.origin,
                        )
                    )
                if span.stop < piece.span.stop:
                    result.append(
                        Piece(
                            Span(span.stop, piece.span.stop),
                            piece.value,
                            piece.version,
                            piece.origin,
                        )
                    )
        result.append(Piece(span, value, self.version, origin))
        self.pieces = sorted(result, key=lambda p: p.span.start)
        return self.version
