"""Exact logical element correspondence for rectangular, rank-reducing DMA."""

from __future__ import annotations

from dataclasses import dataclass

from .index_domain import IndexExpr
from .memory import Expression, _substitute
from .storage import Span
from .tensor_storage import Box, BoxPiece


@dataclass(frozen=True)
class DmaRectangle:
    box: Box
    axes: tuple[int, ...]
    to_ref: dict[str, IndexExpr]
    to_value: dict[str, IndexExpr]
    shape: tuple[int, ...]
    transformed: bool = False
    footprint: tuple[Box, ...] = ()

    @property
    def boxes(self) -> tuple[Box, ...]:
        return self.footprint or (self.box,)

    @property
    def size(self) -> int:
        return sum(box.size for box in self.boxes)

    def copy_piece(
        self, piece: BoxPiece[Expression], destination: DmaRectangle
    ) -> tuple[Box, Expression]:
        axes = list(destination.box.axes)
        for a, b in zip(self.axes, destination.axes, strict=True):
            delta = destination.box.axes[b].start - self.box.axes[a].start
            axes[b] = Span(piece.box.axes[a].start + delta, piece.box.axes[a].stop + delta)
        assert piece.value is not None
        value = _substitute(_substitute(piece.value, self.to_ref), destination.to_value)
        return Box(tuple(axes)), value
