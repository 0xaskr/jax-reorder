"""Finite counter progress and necessary interval ordering, without source objects.

This component does not establish value equivalence or allocation safety. Its
caller must establish the storage/route types, entry synchronization, initialized
values and output coverage. Each semaphore has one ordered consumer stream.
DMA credits may arrive incrementally; no producer identity is assigned to a wait.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal

from ..errors import UnsupportedJaxprError
from .communication import DMA_PROGRESS_ASSUMPTION, MAX_DMA_BYTES
from .index_domain import AnalysisBudget, IndexExpr
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
    kind: Literal["dma", "regular"] = "dma"


@dataclass(frozen=True)
class Transfer:
    source: Instance
    destination: Instance
    source_span: Span
    destination_span: Span
    send: Instance | None
    receive: Instance
    logical_bytes: int
    source_footprint: tuple[Span, ...] = ()
    destination_footprint: tuple[Span, ...] = ()
    destination_to_source: IndexExpr | None = None

    @property
    def source_spans(self) -> tuple[Span, ...]:
        return self.source_footprint or (self.source_span,)

    @property
    def destination_spans(self) -> tuple[Span, ...]:
        return self.destination_footprint or (self.destination_span,)


@dataclass(frozen=True)
class Start:
    transfer: Transfer
    origin: Origin


@dataclass(frozen=True)
class Wait:
    semaphore: Instance
    kind: Literal["send", "receive", "local", "regular"]
    logical_bytes: int
    origin: Origin


@dataclass(frozen=True)
class Access:
    storage: Instance
    span: Span
    write: bool
    origin: Origin
    footprint: tuple[Span, ...] = ()

    @property
    def spans(self) -> tuple[Span, ...]:
        return self.footprint or (self.span,)


@dataclass(frozen=True)
class Signal:
    semaphore: Instance
    amount: int
    origin: Origin


@dataclass(frozen=True)
class Read:
    semaphore: Instance
    origin: Origin


type Command = Start | Wait | Access | Signal | Read


@dataclass(frozen=True)
class DeviceProgram:
    device: int
    commands: tuple[Command, ...]
    # Program order applies only inside this stream. Storage and semaphores
    # remain device instances shared by all streams on the same device.
    stream: int = 0


@dataclass(frozen=True)
class Event:
    id: int
    device: int
    kind: str
    command: int
    origin: Origin
    stream: int = 0


@dataclass(frozen=True)
class CounterEvidence:
    semaphore: Instance
    kind: str
    producers: tuple[tuple[int, int], ...]
    consumer: int
    logical_bytes: int
    # Whole-instance entry/exit invariant, not the count just before/after this wait.
    initial: int = 0
    final: int = 0
    consumed_before: int = 0
    threshold: int = 0
    excluded: tuple[int, ...] = ()
    required_completions: tuple[int, ...] = ()
    required_issues: tuple[int, ...] = ()


@dataclass(frozen=True)
class Producer:
    semaphore: Instance
    issue: int
    complete: int
    amount: int
    kind: str


@dataclass(frozen=True)
class CounterObservation:
    event: int
    semaphore: Instance
    value: int


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
    producers: tuple[Producer, ...] = ()
    observations: tuple[CounterObservation, ...] = ()

    def precedes(self, first: int, second: int) -> bool:
        return bool(self.ancestors[second] & (1 << first))


def analyze_protocol(
    programs: tuple[DeviceProgram, ...],
    semaphores: tuple[Semaphore, ...],
    *,
    budget: AnalysisBudget,
) -> ProtocolEvidence:
    """Prove monotone progress, necessary causality and all-schedule access order.

    Nonnegative production and a single ordered consumer per instance make
    cumulative thresholds monotone. A complete saturation witnesses progress;
    necessary edges are inferred separately, without assuming atomic DMA credits.
    """
    path = "protocol"
    devices = {p.device for p in programs}
    if (
        not devices
        or any(d < 0 for d in devices)
        or any(type(p.stream) is not int or p.stream < 0 for p in programs)
        or len({(p.device, p.stream) for p in programs}) != len(programs)
    ):
        raise UnsupportedJaxprError(path, "invalid or duplicate device program stream")
    sems = {s.instance: s for s in semaphores}
    if len(sems) != len(semaphores) or any(
        s.initial != 0 or s.instance.device not in devices or s.kind not in ("dma", "regular")
        for s in semaphores
    ):
        raise UnsupportedJaxprError(path, "semaphore instances require known zero initial state")
    events: list[Event] = []
    edges: set[tuple[int, int]] = set()
    accesses: list[AccessInterval] = []
    producers: list[Producer] = []
    waits: dict[int, tuple[Wait, int, int]] = {}
    reads: dict[int, tuple[Read, int]] = {}
    consumed: dict[Instance, int] = {}
    observers: dict[Instance, tuple[int, int]] = {}

    def node(device: int, kind: str, command: int, origin: Origin, stream: int) -> int:
        budget.spend(origin.path)
        event = Event(len(events), device, kind, command, origin, stream)
        events.append(event)
        return event.id

    def edge(before: int, after: int) -> None:
        if (before, after) not in edges:
            budget.spend(path)
            edges.add((before, after))

    def produce(sem: Instance, issue: int, complete: int, count: int, kind: str) -> None:
        if sem not in sems:
            raise UnsupportedJaxprError(path, "unknown semaphore instance")
        expected = "regular" if kind == "signal" else "dma"
        if sems[sem].kind != expected:
            raise UnsupportedJaxprError(path, "incompatible semaphore counter kind")
        if count:
            producers.append(Producer(sem, issue, complete, count, kind))

    def access(
        storage: Instance, span: Span, write: bool, begin: int, end: int, origin: Origin
    ) -> None:
        if storage.device not in devices or storage in sems or span.start < 0 or span.size <= 0:
            raise UnsupportedJaxprError(origin.path, "invalid protocol storage access")
        accesses.append(AccessInterval(storage, span, write, begin, end, origin))

    for program in sorted(programs, key=lambda p: (p.device, p.stream)):
        previous = None
        for i, command in enumerate(program.commands):
            event = node(
                program.device, type(command).__name__.lower(), i, command.origin, program.stream
            )
            if previous is not None:
                edge(previous, event)
            previous = event
            if isinstance(command, Start):
                t = command.transfer
                if (
                    t.source.device != program.device
                    or (t.send is not None and t.send.device != program.device)
                    or (t.send is None and t.destination.device != program.device)
                    or t.destination.device not in devices
                    or t.receive.device != t.destination.device
                    or t.send == t.receive
                    or sum(s.size for s in t.source_spans)
                    != sum(s.size for s in t.destination_spans)
                    or not 0 < t.logical_bytes <= MAX_DMA_BYTES
                ):
                    raise UnsupportedJaxprError(
                        command.origin.path, "invalid transfer endpoints/count"
                    )
                sent = node(program.device, "send_complete", i, command.origin, program.stream)
                received = node(
                    t.destination.device, "receive_complete", i, command.origin, program.stream
                )
                edge(event, sent)
                edge(sent, received)
                if t.send is not None:
                    produce(t.send, event, sent, t.logical_bytes, "send")
                produce(t.receive, event, received, t.logical_bytes, "receive")
                for storage, spans, bound, writing, end in (
                    (t.source, t.source_spans, t.source_span, False, sent),
                    (t.destination, t.destination_spans, t.destination_span, True, received),
                ):
                    budget.spend(command.origin.path, len(spans))
                    if (
                        bound.start < 0
                        or bound.size <= 0
                        or any(
                            s.size <= 0 or s.start < bound.start or s.stop > bound.stop
                            for s in spans
                        )
                        or any(a.stop > b.start for a, b in zip(spans, spans[1:], strict=False))
                    ):
                        raise UnsupportedJaxprError(
                            command.origin.path, "invalid transfer footprint"
                        )
                    for selected in spans:
                        access(storage, selected, writing, event, end, command.origin)
            elif isinstance(command, Signal):
                if command.semaphore not in sems or not 0 <= command.amount <= MAX_DMA_BYTES:
                    raise UnsupportedJaxprError(
                        command.origin.path, "invalid signal instance/count"
                    )
                complete = event
                if command.semaphore.device != program.device:
                    complete = node(
                        command.semaphore.device,
                        "signal_complete",
                        i,
                        command.origin,
                        program.stream,
                    )
                    edge(event, complete)
                produce(command.semaphore, event, complete, command.amount, "signal")
            elif isinstance(command, Wait | Read):
                sem = command.semaphore
                if sem not in sems or sem.device != program.device:
                    raise UnsupportedJaxprError(
                        command.origin.path, "invalid local semaphore instance"
                    )
                stream = (program.device, program.stream)
                if observers.setdefault(sem, stream) != stream:
                    raise UnsupportedJaxprError(
                        command.origin.path,
                        "semaphore consumption and reads require one ordered stream",
                    )
                before = consumed.get(sem, 0)
                if isinstance(command, Read):
                    reads[event] = (command, before)
                    continue
                expected = "regular" if command.kind == "regular" else "dma"
                if (
                    command.kind not in ("send", "receive", "local", "regular")
                    or sems[sem].kind != expected
                    or not 0 <= command.logical_bytes <= MAX_DMA_BYTES
                ):
                    raise UnsupportedJaxprError(command.origin.path, "invalid wait instance/count")
                threshold = before + command.logical_bytes
                waits[event] = (command, before, threshold)
                consumed[sem] = threshold
            else:
                if command.storage.device != program.device:
                    raise UnsupportedJaxprError(command.origin.path, "ordinary access is not local")
                budget.spend(command.origin.path, len(command.spans))
                if (
                    command.span.start < 0
                    or command.span.size <= 0
                    or any(s.size <= 0 or not command.span.contains(s) for s in command.spans)
                    or any(
                        a.stop > b.start
                        for a, b in zip(command.spans, command.spans[1:], strict=False)
                    )
                ):
                    raise UnsupportedJaxprError(
                        command.origin.path, "invalid ordinary access footprint"
                    )
                for selected in command.spans:
                    access(command.storage, selected, command.write, event, event, command.origin)

    by_sem: dict[Instance, list[Producer]] = {sem: [] for sem in sems}
    for producer in producers:
        by_sem[producer.semaphore].append(producer)
    for sem, ps in by_sem.items():
        total = sum(p.amount for p in ps)
        if total and sem not in consumed:
            raise UnsupportedJaxprError(path, "semaphore count has no consumer")
        if total != consumed.get(sem, 0):
            raise UnsupportedJaxprError(path, "semaphore count is not conserved")

    def closure() -> tuple[int, ...]:
        successors: list[list[int]] = [[] for _ in events]
        incoming = [0] * len(events)
        for before, after in sorted(edges):
            successors[before].append(after)
            incoming[after] += 1
        ready = [e.id for e in events if not incoming[e.id]]
        ancestors = [0] * len(events)
        done = 0
        while ready:
            current = heappop(ready)
            done += 1
            for successor in successors[current]:
                budget.spend(events[successor].origin.path)
                ancestors[successor] |= ancestors[current] | (1 << current)
                incoming[successor] -= 1
                if incoming[successor] == 0:
                    heappush(ready, successor)
        if done != len(events):
            raise UnsupportedJaxprError(
                path,
                "protocol has a wait dependency cycle",
                blocked=tuple(events[i].origin.path for i, n in enumerate(incoming) if n),
            )
        return tuple(ancestors)

    counters: list[CounterEvidence] = []
    while True:
        ancestors = closure()
        before_edges = len(edges)
        counters = []
        for event, (wait, before, threshold) in waits.items():
            ps = by_sem[wait.semaphore]
            budget.spend(wait.origin.path, len(ps))
            eligible = tuple(p for p in ps if not ancestors[p.issue] & (1 << event))
            total = sum(p.amount for p in eligible)
            if threshold > total:
                raise UnsupportedJaxprError(
                    wait.origin.path,
                    "protocol has a wait dependency cycle",
                    blocked=(wait.origin.path,),
                )
            # A partial contribution proves issue, never full DMA completion.
            completions = (
                tuple(p.complete for p in eligible) if threshold and threshold == total else ()
            )
            issues = tuple(
                p.issue for p in eligible if p.kind == "signal" and threshold > total - p.amount
            )
            for required in (*completions, *issues):
                edge(required, event)
            counters.append(
                CounterEvidence(
                    wait.semaphore,
                    wait.kind,
                    tuple((p.complete, p.amount) for p in eligible),
                    event,
                    wait.logical_bytes,
                    consumed_before=before,
                    threshold=threshold,
                    excluded=tuple(p.complete for p in ps if ancestors[p.issue] & (1 << event)),
                    required_completions=completions,
                    required_issues=issues,
                )
            )
        if len(edges) == before_edges:
            break

    # Bound the counter in every interval between its ordered consumptions. The
    # cumulative total may exceed the hardware range over several safe generations.
    for counter in counters:
        if sum(n for _, n in counter.producers) - counter.consumed_before > MAX_DMA_BYTES:
            raise UnsupportedJaxprError(path, "semaphore live count may overflow")

    # This witness respects actual credits. A topo sort of only the necessary
    # edges can place a partial wait before any sufficient producer combination.
    available: dict[Instance, int] = {}
    completed = 0
    order: list[int] = []
    completion = {p.complete: p for p in producers}
    pending = set(range(len(events)))
    while pending:
        advanced = False
        for event in sorted(pending):
            budget.spend(events[event].origin.path)
            if ancestors[event] & ~completed:
                continue
            if event in waits:
                wait, _, threshold = waits[event]
                if available.get(wait.semaphore, 0) < threshold:
                    continue
            if event in completion:
                producer = completion[event]
                available[producer.semaphore] = (
                    available.get(producer.semaphore, 0) + producer.amount
                )
            order.append(event)
            completed |= 1 << event
            pending.remove(event)
            advanced = True
        if not advanced:
            raise UnsupportedJaxprError(
                path,
                "protocol has a wait dependency cycle",
                blocked=tuple(events[e].origin.path for e in sorted(pending)),
            )

    observations = []
    for event, (read, before) in reads.items():
        arrived = 0
        for p in by_sem[read.semaphore]:
            budget.spend(read.origin.path)
            if ancestors[event] & (1 << p.complete):
                arrived += p.amount
            elif not ancestors[p.issue] & (1 << event):
                raise UnsupportedJaxprError(
                    read.origin.path, "semaphore read depends on scheduling"
                )
        value = arrived - before
        if value < 0 or (sems[read.semaphore].kind == "dma" and value != 0):
            raise UnsupportedJaxprError(
                read.origin.path, "DMA counter observation requires stable zero"
            )
        observations.append(CounterObservation(event, read.semaphore, value))
    assumptions = REMOTE_PROGRESS_ASSUMPTIONS[:2]
    if any(
        isinstance(c, Start) and c.transfer.send is not None for p in programs for c in p.commands
    ):
        assumptions += REMOTE_PROGRESS_ASSUMPTIONS[2:]
    if any(isinstance(c, Start) and c.transfer.send is None for p in programs for c in p.commands):
        assumptions += (DMA_PROGRESS_ASSUMPTION,)
    if any(e.kind == "signal_complete" for e in events):
        assumptions += ("the TPU runtime eventually delivers each issued, valid remote signal",)
    evidence = ProtocolEvidence(
        tuple(events),
        tuple(sorted(edges)),
        tuple(counters),
        tuple(accesses),
        tuple(order),
        ancestors,
        assumptions,
        tuple(producers),
        tuple(observations),
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
