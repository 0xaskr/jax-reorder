"""Finite remote DMA progress and interval ordering, without source objects.

This component does not establish value equivalence or allocation safety. Its
caller must establish the storage/route types, entry synchronization, initialized
values and output coverage. Each semaphore has one zero-initialized generation,
one consumer and a whole-count wait; no schedule search is used in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal

from ..errors import UnsupportedJaxprError
from .communication import MAX_DMA_BYTES
from .index_domain import AnalysisBudget
from .model import Origin
from .storage import Span

REMOTE_PROGRESS_ASSUMPTIONS = (
    "participating devices and links do not fail",
    "enabled participants are not permanently starved",
    "the TPU runtime eventually completes each issued, valid remote DMA",
)


@dataclass(frozen=True, order=True)
class Instance:
    device: int
    ref: int


@dataclass(frozen=True)
class Semaphore:
    instance: Instance
    initial: int


@dataclass(frozen=True)
class Transfer:
    source: Instance
    destination: Instance
    source_span: Span
    destination_span: Span
    send: Instance
    receive: Instance
    logical_bytes: int


@dataclass(frozen=True)
class Start:
    transfer: Transfer
    origin: Origin


@dataclass(frozen=True)
class Wait:
    semaphore: Instance
    kind: Literal["send", "receive"]
    logical_bytes: int
    origin: Origin


@dataclass(frozen=True)
class Access:
    storage: Instance
    span: Span
    write: bool
    origin: Origin


type Command = Start | Wait | Access


@dataclass(frozen=True)
class DeviceProgram:
    device: int
    commands: tuple[Command, ...]


@dataclass(frozen=True)
class Event:
    id: int
    device: int
    kind: str
    command: int
    origin: Origin


@dataclass(frozen=True)
class CounterEvidence:
    semaphore: Instance
    kind: str
    producers: tuple[tuple[int, int], ...]
    consumer: int
    logical_bytes: int
    initial: int = 0
    final: int = 0


@dataclass(frozen=True)
class AccessInterval:
    storage: Instance
    span: Span
    write: bool
    begin: int
    end: int
    origin: Origin


@dataclass(frozen=True)
class ProtocolEvidence:
    events: tuple[Event, ...]
    edges: tuple[tuple[int, int], ...]
    counters: tuple[CounterEvidence, ...]
    accesses: tuple[AccessInterval, ...]
    order: tuple[int, ...]
    # Bits are event IDs. This relation, not the chosen topological order,
    # justifies every conflicting pair's order under all allowed schedules.
    ancestors: tuple[int, ...]
    assumptions: tuple[str, ...] = REMOTE_PROGRESS_ASSUMPTIONS

    def precedes(self, first: int, second: int) -> bool:
        return bool(self.ancestors[second] & (1 << first))


def analyze_protocol(
    programs: tuple[DeviceProgram, ...],
    semaphores: tuple[Semaphore, ...],
    *,
    budget: AnalysisBudget,
) -> ProtocolEvidence:
    """Check whole-count progress and all-schedule conflict ordering only."""
    path = "protocol"
    devices = {p.device for p in programs}
    if not devices or len(devices) != len(programs) or any(d < 0 for d in devices):
        raise UnsupportedJaxprError(path, "invalid or duplicate device program")
    sems = {s.instance: s for s in semaphores}
    if len(sems) != len(semaphores) or any(
        s.initial != 0 or s.instance.device not in devices for s in semaphores
    ):
        raise UnsupportedJaxprError(path, "semaphore instances require known zero initial state")
    events: list[Event] = []
    edges: set[tuple[int, int]] = set()
    accesses: list[AccessInterval] = []
    producers: dict[Instance, list[tuple[int, int, str]]] = {}
    consumers: dict[Instance, tuple[int, Wait]] = {}

    def node(device: int, kind: str, command: int, origin: Origin) -> int:
        budget.spend(origin.path)
        event = Event(len(events), device, kind, command, origin)
        events.append(event)
        return event.id

    def edge(before: int, after: int) -> None:
        budget.spend(path)
        edges.add((before, after))

    def produce(sem: Instance, event: int, count: int, kind: str, origin: Origin) -> None:
        if sem not in sems:
            raise UnsupportedJaxprError(origin.path, "unknown semaphore instance")
        producers.setdefault(sem, []).append((event, count, kind))

    def access(
        storage: Instance, span: Span, write: bool, begin: int, end: int, origin: Origin
    ) -> None:
        if storage.device not in devices or storage in sems or span.start < 0 or span.size <= 0:
            raise UnsupportedJaxprError(origin.path, "invalid protocol storage access")
        accesses.append(AccessInterval(storage, span, write, begin, end, origin))

    for program in sorted(programs, key=lambda p: p.device):
        previous = None
        for i, command in enumerate(program.commands):
            event = node(program.device, type(command).__name__.lower(), i, command.origin)
            if previous is not None:
                edge(previous, event)
            previous = event
            if isinstance(command, Start):
                t = command.transfer
                if (
                    t.source.device != program.device
                    or t.send.device != program.device
                    or t.destination.device not in devices
                    or t.receive.device != t.destination.device
                    or t.send == t.receive
                    or t.source_span.size != t.destination_span.size
                    or not 0 < t.logical_bytes <= MAX_DMA_BYTES
                ):
                    raise UnsupportedJaxprError(
                        command.origin.path, "invalid transfer endpoints/count"
                    )
                sent = node(program.device, "send_complete", i, command.origin)
                received = node(t.destination.device, "receive_complete", i, command.origin)
                edge(event, sent)
                edge(sent, received)
                produce(t.send, sent, t.logical_bytes, "send", command.origin)
                produce(t.receive, received, t.logical_bytes, "receive", command.origin)
                access(t.source, t.source_span, False, event, sent, command.origin)
                access(t.destination, t.destination_span, True, event, received, command.origin)
            elif isinstance(command, Wait):
                if (
                    command.semaphore not in sems
                    or command.semaphore.device != program.device
                    or command.kind not in ("send", "receive")
                    or not 0 < command.logical_bytes <= MAX_DMA_BYTES
                ):
                    raise UnsupportedJaxprError(command.origin.path, "invalid wait instance/count")
                if command.semaphore in consumers:
                    raise UnsupportedJaxprError(
                        command.origin.path,
                        "partial waits or multiple consumers require another rule",
                    )
                consumers[command.semaphore] = (event, command)
            else:
                if command.storage.device != program.device:
                    raise UnsupportedJaxprError(command.origin.path, "ordinary access is not local")
                access(command.storage, command.span, command.write, event, event, command.origin)

    counters = []
    for sem in sorted(producers.keys() | consumers.keys()):
        produced, consumer = producers.get(sem, []), consumers.get(sem)
        if not produced or consumer is None:
            raise UnsupportedJaxprError(path, "semaphore count has no producer or no consumer")
        event, wait = consumer
        total = sum(count for _, count, _ in produced)
        if total != wait.logical_bytes or total > MAX_DMA_BYTES:
            raise UnsupportedJaxprError(wait.origin.path, "semaphore count is not conserved")
        if any(kind != wait.kind for _, _, kind in produced):
            raise UnsupportedJaxprError(
                wait.origin.path, "mixed send/receive semaphore consumption"
            )
        for producer, _, _ in produced:
            edge(producer, event)
        counters.append(
            CounterEvidence(
                sem, wait.kind, tuple((p, count) for p, count, _ in produced), event, total
            )
        )

    successors: list[list[int]] = [[] for _ in events]
    incoming = [0] * len(events)
    for before, after in sorted(edges):
        successors[before].append(after)
        incoming[after] += 1
    ready = [e.id for e in events if not incoming[e.id]]
    order: list[int] = []
    ancestors = [0] * len(events)
    while ready:
        current = heappop(ready)
        order.append(current)
        for successor in successors[current]:
            budget.spend(events[successor].origin.path)
            ancestors[successor] |= ancestors[current] | (1 << current)
            incoming[successor] -= 1
            if incoming[successor] == 0:
                heappush(ready, successor)
    if len(order) != len(events):
        blocked = tuple(events[i].origin.path for i, degree in enumerate(incoming) if degree)
        raise UnsupportedJaxprError(path, "protocol has a wait dependency cycle", blocked=blocked)
    evidence = ProtocolEvidence(
        tuple(events),
        tuple(sorted(edges)),
        tuple(counters),
        tuple(accesses),
        tuple(order),
        tuple(ancestors),
    )
    by_storage: dict[Instance, list[AccessInterval]] = {}
    for current in accesses:
        previous = by_storage.setdefault(current.storage, [])
        for other in previous:
            budget.spend(current.origin.path)
            if not (current.write or other.write) or not current.span.overlaps(other.span):
                continue
            if not (
                evidence.precedes(other.end, current.begin)
                or evidence.precedes(current.end, other.begin)
            ):
                raise UnsupportedJaxprError(
                    current.origin.path,
                    "conflicting protocol accesses lack a proven completion order",
                    other=other.origin.path,
                    storage=current.storage,
                )
        previous.append(current)
    return evidence
