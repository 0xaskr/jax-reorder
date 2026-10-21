"""Parallel device streams preserve real storage sharing and partial order."""

from dataclasses import replace
from itertools import permutations

import pytest

from jax_reorder import UnsupportedJaxprError
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import Origin
from jax_reorder.core.protocol import (
    Access,
    DeviceProgram,
    Instance,
    Read,
    Semaphore,
    Signal,
    Start,
    Transfer,
    Wait,
    analyze_protocol,
)
from jax_reorder.core.storage import Span


def prove(programs, semaphores=()):
    return analyze_protocol(programs, semaphores, budget=AnalysisBudget())


def access(*, offset=0, write=True, label="access"):
    return Access(Instance(0, 0), Span(offset, offset + 4), write, Origin(label))


def regular_commands(sem=None):
    sem = Instance(0, 10) if sem is None else sem
    return (
        Signal(sem, 1, Origin("signal")),
        Wait(sem, "regular", 1, Origin("wait")),
        Read(sem, Origin("read")),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_disjoint_parallel_streams_have_only_their_own_program_order(reverse):
    programs = (
        DeviceProgram(0, (access(label="a0"), access(label="a1")), stream=0),
        DeviceProgram(0, (access(offset=4, label="b0"), access(offset=4, label="b1")), stream=1),
    )
    if reverse:
        programs = tuple(reversed(programs))
    proof = prove(programs)
    events = {(e.stream, e.command): e.id for e in proof.events}
    assert proof.precedes(events[0, 0], events[0, 1])
    assert proof.precedes(events[1, 0], events[1, 1])
    assert not proof.precedes(events[0, 1], events[1, 0])
    assert not proof.precedes(events[1, 1], events[0, 0])
    assert {a.storage for a in proof.accesses} == {Instance(0, 0)}


@pytest.mark.parametrize("first_write,second_write", [(True, True), (False, True), (True, False)])
def test_parallel_streams_cannot_hide_shared_storage_conflicts(first_write, second_write):
    programs = (
        DeviceProgram(0, (access(write=first_write),), stream=0),
        DeviceProgram(0, (access(write=second_write),), stream=1),
    )
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        prove(programs)


def test_parallel_shared_reads_need_no_order():
    proof = prove(
        (
            DeviceProgram(0, (access(write=False),), stream=0),
            DeviceProgram(0, (access(write=False),), stream=1),
        )
    )
    assert not proof.edges


def test_shared_semaphore_establishes_order_between_streams():
    signal, wait, read = regular_commands()
    proof = prove(
        (
            DeviceProgram(0, (access(label="write"), signal), stream=4),
            DeviceProgram(0, (wait, access(write=False, label="read"), read), stream=1),
        ),
        (Semaphore(signal.semaphore, 0, "regular"),),
    )
    writer = next(e for e in proof.events if e.kind == "access" and e.stream == 4)
    reader = next(e for e in proof.events if e.kind == "access" and e.stream == 1)
    assert proof.precedes(writer.id, reader.id)
    assert proof.observations[0].value == 0


def test_parallel_producers_can_feed_one_ordered_consumer():
    signal, wait, read = regular_commands()
    proof = prove(
        (
            DeviceProgram(0, (signal,), stream=0),
            DeviceProgram(0, (signal,), stream=1),
            DeviceProgram(0, (replace(wait, logical_bytes=2), read), stream=2),
        ),
        (Semaphore(signal.semaphore, 0, "regular"),),
    )
    assert len(proof.counters[0].required_completions) == 2
    assert proof.observations[0].value == 0


@pytest.mark.parametrize("other", ["wait", "read"])
@pytest.mark.parametrize("consumer_stream", [0, 1])
def test_cross_stream_counter_consumption_and_observation_are_rejected(other, consumer_stream):
    signal, wait, read = regular_commands()
    second = wait if other == "wait" else read
    signals = (signal, signal) if other == "wait" else (signal,)
    programs = (
        DeviceProgram(0, (*signals, wait), stream=consumer_stream),
        DeviceProgram(0, (second,), stream=1 - consumer_stream),
    )
    with pytest.raises(UnsupportedJaxprError, match="one ordered stream"):
        prove(programs, (Semaphore(signal.semaphore, 0, "regular"),))


def test_cross_stream_wait_cycle_is_rejected():
    first, second = Instance(0, 10), Instance(0, 11)
    a_signal, a_wait, _ = regular_commands(first)
    b_signal, b_wait, _ = regular_commands(second)
    with pytest.raises(UnsupportedJaxprError, match="wait dependency cycle"):
        prove(
            (
                DeviceProgram(0, (a_wait, b_signal), stream=0),
                DeviceProgram(0, (b_wait, a_signal), stream=1),
            ),
            (Semaphore(first, 0, "regular"), Semaphore(second, 0, "regular")),
        )


def dma_programs(*, overlapping=False):
    programs, semaphores = [], []
    for stream in range(2):
        span = Span(stream * 4, (stream + 1) * 4)
        target = Span(0, 4) if overlapping else span
        send, receive = Instance(0, 10 + stream), Instance(1, 10 + stream)
        transfer = Transfer(Instance(0, 0), Instance(1, 0), span, target, send, receive, 16)
        programs.extend(
            (
                DeviceProgram(
                    0,
                    (Start(transfer, Origin("start")), Wait(send, "send", 16, Origin("send"))),
                    stream=stream,
                ),
                DeviceProgram(
                    1,
                    (
                        Wait(receive, "receive", 16, Origin("receive")),
                        Access(transfer.destination, target, False, Origin("read")),
                    ),
                    stream=stream,
                ),
            )
        )
        semaphores.extend((Semaphore(send, 0), Semaphore(receive, 0)))
    return tuple(programs), tuple(semaphores)


def test_parallel_remote_dmas_keep_stream_identity_and_actual_device_storage():
    programs, semaphores = dma_programs()
    proof = prove(programs, semaphores)
    completions = [e for e in proof.events if e.kind == "receive_complete"]
    assert {(e.device, e.stream) for e in completions} == {(1, 0), (1, 1)}
    assert len(proof.counters) == 4
    assert {a.storage for a in proof.accesses} == {Instance(0, 0), Instance(1, 0)}
    assert not proof.precedes(completions[0].id, completions[1].id)
    assert not proof.precedes(completions[1].id, completions[0].id)


def test_parallel_remote_dmas_to_overlapping_target_are_rejected():
    programs, semaphores = dma_programs(overlapping=True)
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        prove(programs, semaphores)


@pytest.mark.parametrize("stream", [-1, True, 0.5, "0", []])
def test_invalid_stream_ids_are_rejected(stream):
    with pytest.raises(UnsupportedJaxprError, match="program stream"):
        prove((DeviceProgram(0, (), stream=stream),))


def test_duplicate_stream_id_on_one_device_is_rejected():
    with pytest.raises(UnsupportedJaxprError, match="duplicate device program stream"):
        prove((DeviceProgram(0, (), stream=2), DeviceProgram(0, (), stream=2)))


def test_default_stream_preserves_legacy_single_program_behavior():
    proof = prove((DeviceProgram(0, (access(label="first"), access(label="second"))),))
    assert all(event.stream == 0 for event in proof.events)
    assert proof.precedes(0, 1)


def interleavings(programs):
    """Small independent counter interpreter, without protocol events or edges."""
    pending = [((0,) * len(programs), {}, ())]
    finished, deadlocked = [], False
    while pending:
        positions, counts, trace = pending.pop()
        if all(pc == len(p.commands) for pc, p in zip(positions, programs, strict=True)):
            finished.append(trace)
            continue
        advanced = False
        for stream, (pc, program) in enumerate(zip(positions, programs, strict=True)):
            if pc == len(program.commands):
                continue
            command = program.commands[pc]
            updated = counts.copy()
            if isinstance(command, Wait):
                available = counts.get(command.semaphore, 0)
                if available < command.logical_bytes:
                    continue
                updated[command.semaphore] = available - command.logical_bytes
            elif isinstance(command, Signal):
                updated[command.semaphore] = counts.get(command.semaphore, 0) + command.amount
            elif not isinstance(command, Access):
                raise AssertionError("unsupported oracle command")
            following = list(positions)
            following[stream] += 1
            pending.append((tuple(following), updated, (*trace, command.origin.path)))
            advanced = True
        deadlocked |= not advanced
    return finished, deadlocked


@pytest.mark.parametrize("writer_order", list(permutations(range(2))))
@pytest.mark.parametrize("reader_order", list(permutations(range(2))))
def test_stream_access_order_agrees_with_all_counter_interleavings(writer_order, reader_order):
    signal, wait, _ = regular_commands()
    writer = (access(label="write"), signal)
    reader = (wait, access(write=False, label="read"))
    programs = (
        DeviceProgram(0, tuple(writer[i] for i in writer_order), stream=0),
        DeviceProgram(0, tuple(reader[i] for i in reader_order), stream=1),
    )
    traces, deadlocked = interleavings(programs)
    assert traces and not deadlocked
    ordered = all(t.index("write") < t.index("read") for t in traces)
    semaphores = (Semaphore(signal.semaphore, 0, "regular"),)
    if ordered:
        prove(programs, semaphores)
    else:
        with pytest.raises(UnsupportedJaxprError, match="completion order"):
            prove(programs, semaphores)


@pytest.mark.parametrize("first_order", list(permutations(range(2))))
@pytest.mark.parametrize("second_order", list(permutations(range(2))))
def test_stream_progress_agrees_with_all_counter_interleavings(first_order, second_order):
    first, second = Instance(0, 10), Instance(0, 11)
    a_signal, a_wait, _ = regular_commands(first)
    b_signal, b_wait, _ = regular_commands(second)
    a_commands, b_commands = (a_wait, b_signal), (b_wait, a_signal)
    programs = (
        DeviceProgram(0, tuple(a_commands[i] for i in first_order), stream=0),
        DeviceProgram(0, tuple(b_commands[i] for i in second_order), stream=1),
    )
    traces, deadlocked = interleavings(programs)
    semaphores = (Semaphore(first, 0, "regular"), Semaphore(second, 0, "regular"))
    if deadlocked:
        with pytest.raises(UnsupportedJaxprError, match="wait dependency cycle"):
            prove(programs, semaphores)
    else:
        assert traces
        prove(programs, semaphores)
