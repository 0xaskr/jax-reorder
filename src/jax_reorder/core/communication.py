"""Local DMA interval hazards and counted waits, independent of JAX and values.

A partial wait consumes progress, but does not identify which producer finished.
All intervals stay protected until the generation's total count is consumed.
Logical bytes describe payloads, not a hardware register encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget
from .model import DmaView, Origin
from .storage import Span, VersionSlice
from .tensor_storage import Box

DMA_PROGRESS_ASSUMPTION = "the TPU runtime eventually completes each issued, valid local DMA"
MAX_DMA_BYTES = 2**31 - 1
type Region = Box | tuple[Box, ...]


def _boxes(region: Region) -> tuple[Box, ...]:
    return (region,) if isinstance(region, Box) else region


def _overlap(left: Region, right: Region, budget: AnalysisBudget, path: str) -> bool:
    a_boxes, b_boxes = _boxes(left), _boxes(right)
    budget.spend(path + ".dma_overlap", len(a_boxes) * len(b_boxes))
    return any(a.intersection(b) is not None for a in a_boxes for b in b_boxes)


@dataclass(frozen=True)
class PendingTransfer:
    source: int
    destination: int
    semaphore: int
    generation: int
    source_version: int
    elements: int
    logical_bytes: int
    start: Origin
    id: int
    source_span: Span
    destination_span: Span
    source_versions: tuple[VersionSlice, ...]
    source_region: Region | None = None
    destination_region: Region | None = None


@dataclass(frozen=True)
class TransferEvidence:
    transfer: PendingTransfer
    wait: Origin
    destination_version: int
    initial_semaphore: int = 0
    final_semaphore: int = 0


@dataclass(frozen=True)
class WaitEvidence:
    semaphore: int
    generation: int
    origin: Origin
    source_span: Span
    destination_span: Span
    logical_bytes: int
    total_bytes: int
    consumed_before: int
    consumed_after: int
    producers: tuple[int, ...]
    released: bool


@dataclass
class _Group:
    dtype: str
    transfers: list[PendingTransfer] = field(default_factory=list)
    total: int = 0
    consumed: int = 0
    aligned: bool = True


def _aligned(*spans: Span) -> bool:
    return all(s.start % 128 == s.size % 128 == 0 for s in spans)


class LocalDmaState:
    """Prove all allowed completion orders safe without searching schedules."""

    def __init__(self, semaphores: set[int], budget: AnalysisBudget) -> None:
        self.pending: dict[int, _Group] = {}
        self.generations = dict.fromkeys(semaphores, 0)
        self.completed: list[TransferEvidence] = []
        self.waits: list[WaitEvidence] = []
        self.readers: dict[int, dict[int, Span | Region]] = {}
        self.writers: dict[int, dict[int, Span | Region]] = {}
        self.next_id = 0
        self.budget = budget

    def access(
        self, storage: int, span: Span, *, write: bool, path: str, region: Region | None = None
    ) -> None:
        for table, active, reason in (
            (self.writers, True, "DMA destination accessed before completion wait"),
            (self.readers, write, "DMA source overwritten before completion wait"),
        ):
            if not active:
                continue
            intervals = table.get(storage, {})
            if len(intervals) > 1:
                self.budget.spend(path, len(intervals) - 1)
            if any(
                _overlap(region, other, self.budget, path)
                if not isinstance(other, Span) and region is not None
                else span.overlaps(other)
                if isinstance(other, Span)
                else True
                for other in intervals.values()
            ):
                raise UnsupportedJaxprError(path, reason)

    def start(
        self,
        dma: DmaView,
        origin: Origin,
        *,
        source_version: int,
        source_span: Span,
        destination_span: Span,
        source_versions: tuple[VersionSlice, ...],
        dtype: str,
        logical_bytes: int,
        source_region: Region | None = None,
        destination_region: Region | None = None,
        aligned_regions: bool | None = None,
    ) -> PendingTransfer:
        path = origin.path
        if dma.priority != 0 or dma.add is not False:
            raise UnsupportedJaxprError(path, "non-default DMA priority/add needs a separate rule")
        if dma.semaphore.id not in self.generations:
            raise UnsupportedJaxprError(path, "DMA semaphore is not a zero-initialized scratch Ref")
        group = self.pending.get(dma.semaphore.id)
        aligned = (
            _aligned(source_span, destination_span) if aligned_regions is None else aligned_regions
        )
        if group is not None:
            first = group.transfers[0]
            if group.consumed:
                raise UnsupportedJaxprError(
                    path, "new DMA generation mixed with partial wait consumption"
                )
            if (dma.source.id, dma.destination.id, dtype) != (
                first.source,
                first.destination,
                group.dtype,
            ):
                raise UnsupportedJaxprError(
                    path, "cumulative DMA group requires matching storage and dtype"
                )
            if not group.aligned or not aligned:
                raise UnsupportedJaxprError(
                    path, "cumulative DMA requires 128-element aligned regions"
                )
        else:
            group = _Group(dtype)
            self.generations[dma.semaphore.id] += 1
            self.pending[dma.semaphore.id] = group
        if not 0 < group.total + logical_bytes <= MAX_DMA_BYTES:
            raise UnsupportedJaxprError(
                path, "DMA logical count exceeds the reviewed counter bound"
            )
        self.access(dma.source.id, source_span, write=False, path=path, region=source_region)
        self.access(
            dma.destination.id, destination_span, write=True, path=path, region=destination_region
        )
        if dma.source == dma.destination and (
            _overlap(source_region, destination_region, self.budget, path)
            if source_region is not None and destination_region is not None
            else source_span.overlaps(destination_span)
        ):
            raise UnsupportedJaxprError(path, "overlapping DMA source and destination")
        transfer = PendingTransfer(
            dma.source.id,
            dma.destination.id,
            dma.semaphore.id,
            self.generations[dma.semaphore.id],
            source_version,
            sum(box.size for box in _boxes(source_region))
            if source_region is not None
            else source_span.size,
            logical_bytes,
            origin,
            self.next_id,
            source_span,
            destination_span,
            source_versions,
            source_region,
            destination_region,
        )
        self.next_id += 1
        group.transfers.append(transfer)
        group.total += logical_bytes
        group.aligned &= aligned
        self.readers.setdefault(dma.source.id, {})[transfer.id] = source_region or source_span
        self.writers.setdefault(dma.destination.id, {})[transfer.id] = (
            destination_region or destination_span
        )
        return transfer

    def release_scope(self, resources: set[int], path: str) -> None:
        if any(
            resource in self.pending or self.readers.get(resource) or self.writers.get(resource)
            for resource in resources
        ):
            raise UnsupportedJaxprError(path, "scope exits with DMA using a local allocation")

    def wait(
        self,
        dma: DmaView,
        origin: Origin,
        *,
        source_span: Span,
        destination_span: Span,
        dtype: str,
        logical_bytes: int,
        aligned_regions: bool | None = None,
    ) -> tuple[PendingTransfer, ...]:
        group = self.pending.get(dma.semaphore.id)
        if group is None:
            raise UnsupportedJaxprError(origin.path, "DMA wait has no pending start")
        first = group.transfers[0]
        if (dma.source.id, dma.destination.id, dtype) != (
            first.source,
            first.destination,
            group.dtype,
        ):
            raise UnsupportedJaxprError(
                origin.path, "DMA wait does not match pending transfer storage/dtype"
            )
        consumed = group.consumed + logical_bytes
        if logical_bytes <= 0 or consumed > group.total:
            raise UnsupportedJaxprError(
                origin.path, "DMA wait consumes more than the issued transfer count"
            )
        if (len(group.transfers) > 1 or group.consumed or consumed != group.total) and not (
            group.aligned
            and (
                _aligned(source_span, destination_span)
                if aligned_regions is None
                else aligned_regions
            )
        ):
            raise UnsupportedJaxprError(
                origin.path, "partial/cumulative wait requires aligned regions"
            )
        released = consumed == group.total
        self.budget.spend(origin.path, max(len(group.transfers) - 1, 0))
        self.waits.append(
            WaitEvidence(
                dma.semaphore.id,
                first.generation,
                origin,
                source_span,
                destination_span,
                logical_bytes,
                group.total,
                group.consumed,
                consumed,
                tuple(p.id for p in group.transfers),
                released,
            )
        )
        group.consumed = consumed
        if not released:
            return ()
        del self.pending[dma.semaphore.id]
        for transfer in group.transfers:
            del self.readers[transfer.source][transfer.id]
            del self.writers[transfer.destination][transfer.id]
        return tuple(group.transfers)

    def finish(self, path: str) -> tuple[TransferEvidence, ...]:
        if self.pending:
            raise UnsupportedJaxprError(
                path, "kernel exits with pending DMA or unconsumed semaphore"
            )
        return tuple(self.completed)
