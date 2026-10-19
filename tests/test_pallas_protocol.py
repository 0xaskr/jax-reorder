"""Finite remote event analysis and an independent counter-machine oracle."""

from dataclasses import replace
from itertools import permutations, product

import pytest

from jax_reorder import ComparisonBudgetError, UnsupportedJaxprError
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import Origin
from jax_reorder.core.protocol import (
    Access,
    DeviceProgram,
    Instance,
    Semaphore,
    Start,
    Transfer,
    Wait,
    analyze_protocol,
)
from jax_reorder.core.storage import Span


def transfer(sender, receiver, sem=10, *, source=0, destination=1, span=None):
    span = Span(0, 128) if span is None else span
    return Transfer(
        Instance(sender, source),
        Instance(receiver, destination),
        span,
        span,
        Instance(sender, sem),
        Instance(receiver, sem + 1),
        span.size * 4,
    )


def start(t):
    return Start(t, Origin(f"{t.source.device}.start.{t.send.ref}"))


def wait(t, kind):
    sem = t.send if kind == "send" else t.receive
    return Wait(sem, kind, t.logical_bytes, Origin(f"{sem.device}.wait.{sem.ref}"))


def semaphores(*transfers):
    return tuple(
        Semaphore(s, 0) for s in sorted({s for t in transfers for s in (t.send, t.receive)})
    )


def prove(programs, *transfers, budget=10000):
    return analyze_protocol(programs, semaphores(*transfers), budget=AnalysisBudget(budget))


def request_reply(*, delayed=False, relay=True):
    request = transfer(0, 1)
    reply = transfer(1, 0, 12, source=1 if relay else 0)
    a = (start(request), wait(reply, "receive"), wait(request, "send"))
    if delayed:
        a = (a[1], a[0], a[2])
    b = (wait(request, "receive"), start(reply), wait(reply, "send"))
    return (DeviceProgram(0, a), DeviceProgram(1, b)), (request, reply)


def test_request_reply_preserves_start_before_remote_progress():
    programs, transfers = request_reply()
    proof = prove(programs, *transfers)
    assert len(proof.counters) == 4
    assert all(c.initial == c.final == 0 for c in proof.counters)
    request_received = next(e.id for e in proof.events if e.kind == "receive_complete")
    reply_start = next(e.id for e in proof.events if e.kind == "start" and e.device == 1)
    assert proof.precedes(request_received, reply_start)
    bad, transfers = request_reply(delayed=True)
    with pytest.raises(UnsupportedJaxprError, match="wait dependency cycle") as caught:
        prove(bad, *transfers)
    assert caught.value.diagnostics["blocked"]


def test_three_device_direct_and_relay_progress_are_both_supported():
    direct, first, second = transfer(0, 2), transfer(0, 1), transfer(1, 2, 12, source=1)
    a = (
        DeviceProgram(0, (start(direct), wait(direct, "send"))),
        DeviceProgram(1, ()),
        DeviceProgram(2, (wait(direct, "receive"),)),
    )
    b = (
        DeviceProgram(0, (start(first), wait(first, "send"))),
        DeviceProgram(1, (wait(first, "receive"), start(second), wait(second, "send"))),
        DeviceProgram(2, (wait(second, "receive"),)),
    )
    assert len(prove(a, direct).counters) == 2
    assert len(prove(b, first, second).counters) == 4


def test_send_wait_releases_source_without_requiring_receive_wait():
    t = transfer(0, 1)
    write = Access(t.source, t.source_span, True, Origin("overwrite"))
    a = DeviceProgram(0, (start(t), wait(t, "send"), write))
    b = DeviceProgram(1, (wait(t, "receive"),))
    prove((a, b), t)
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        prove((replace(a, commands=(start(t), write, wait(t, "send"))), b), t)


def test_remote_destination_needs_receive_completion_before_read():
    t = transfer(0, 1)
    read = Access(t.destination, t.destination_span, False, Origin("read"))
    a = DeviceProgram(0, (start(t), wait(t, "send")))
    b = DeviceProgram(1, (wait(t, "receive"), read))
    prove((a, b), t)
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        prove((a, replace(b, commands=(read, wait(t, "receive")))), t)


def test_unordered_overlapping_writes_fail_even_when_one_topological_sort_looks_safe():
    a, b = transfer(0, 2), transfer(1, 2, 12)
    programs = (
        DeviceProgram(0, (start(a), wait(a, "send"))),
        DeviceProgram(1, (start(b), wait(b, "send"))),
        DeviceProgram(2, (wait(a, "receive"), wait(b, "receive"))),
    )
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        prove(programs, a, b)
    separate = replace(b, destination_span=Span(128, 256))
    prove(
        (
            programs[0],
            replace(programs[1], commands=(start(separate), wait(b, "send"))),
            programs[2],
        ),
        a,
        b,
    )


def test_split_transfers_consume_total_without_pairing_waits_to_starts():
    a, b = transfer(0, 1), transfer(0, 1, span=Span(128, 256))
    send, receive = (
        replace(wait(a, "send"), logical_bytes=1024),
        replace(wait(a, "receive"), logical_bytes=1024),
    )
    programs = (DeviceProgram(0, (start(b), start(a), send)), DeviceProgram(1, (receive,)))
    proof = prove(programs, a, b)
    assert all(len(c.producers) == 2 and c.logical_bytes == 1024 for c in proof.counters)
    partial = replace(programs[1], commands=(wait(a, "receive"), wait(b, "receive")))
    with pytest.raises(UnsupportedJaxprError, match="partial waits or multiple consumers"):
        prove((programs[0], partial), a, b)


@pytest.mark.parametrize(
    "change,reason",
    [
        (0, "not conserved"),
        (1, "no consumer"),
        (2, "mixed send/receive"),
        (3, "zero initial"),
        (4, "unknown semaphore"),
    ],
)
def test_counter_preconditions_are_checked_before_topology(change, reason):
    t = transfer(0, 1)
    receive = wait(t, "receive")
    programs = (DeviceProgram(0, (start(t), wait(t, "send"))), DeviceProgram(1, (receive,)))
    sems = semaphores(t)
    if change == 0:
        programs = (
            programs[0],
            replace(programs[1], commands=(replace(receive, logical_bytes=256),)),
        )
    elif change == 1:
        programs = (programs[0], replace(programs[1], commands=()))
    elif change == 2:
        programs = (programs[0], replace(programs[1], commands=(replace(receive, kind="send"),)))
    elif change == 3:
        sems = (replace(sems[0], initial=1), *sems[1:])
    else:
        sems = ()
    with pytest.raises(UnsupportedJaxprError, match=reason):
        analyze_protocol(programs, sems, budget=AnalysisBudget())


def counter_machine_deadlocks(programs, transfers):
    """Exhaustive small oracle: execute counters and DMA phases, never graph edges."""
    slots = {sem: i for i, sem in enumerate(s.instance for s in semaphores(*transfers))}
    ids = {t: i for i, t in enumerate(transfers)}
    initial = ((0,) * len(programs), (-1,) * len(transfers), (0,) * len(slots))
    pending, seen = [initial], set()
    deadlock = False
    while pending:
        state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        pcs, phases, counts = state
        next_states = []
        for i, (pc, program) in enumerate(zip(pcs, programs, strict=True)):
            if pc == len(program.commands):
                continue
            command = program.commands[pc]
            ps, cs, positions = list(phases), list(counts), list(pcs)
            if isinstance(command, Start):
                ps[ids[command.transfer]] = 0
            elif isinstance(command, Wait):
                slot = slots[command.semaphore]
                if cs[slot] < command.logical_bytes:
                    continue
                cs[slot] -= command.logical_bytes
            positions[i] += 1
            next_states.append((tuple(positions), tuple(ps), tuple(cs)))
        for i, (phase, t) in enumerate(zip(phases, transfers, strict=True)):
            if phase not in (0, 1):
                continue
            ps, cs = list(phases), list(counts)
            ps[i] += 1
            cs[slots[t.send if phase == 0 else t.receive]] += t.logical_bytes
            next_states.append((pcs, tuple(ps), tuple(cs)))
        if not next_states:
            finished = all(pc == len(p.commands) for pc, p in zip(pcs, programs, strict=True))
            deadlock |= not (finished and all(phase == 2 for phase in phases) and not any(counts))
        pending.extend(next_states)
    return deadlock


def test_progress_matches_all_interleavings_of_two_device_command_permutations():
    programs, transfers = request_reply(relay=False)
    for a, b in product(permutations(programs[0].commands), permutations(programs[1].commands)):
        candidate = (replace(programs[0], commands=a), replace(programs[1], commands=b))
        if counter_machine_deadlocks(candidate, transfers):
            with pytest.raises(UnsupportedJaxprError, match="wait dependency cycle"):
                prove(candidate, *transfers)
        else:
            prove(candidate, *transfers)


def test_event_budget_is_independent_of_tensor_elements():
    t = transfer(0, 1, span=Span(0, 2**24))
    programs = (
        DeviceProgram(0, (start(t), wait(t, "send"))),
        DeviceProgram(1, (wait(t, "receive"),)),
    )
    assert len(prove(programs, t, budget=32).events) == 5
    with pytest.raises(ComparisonBudgetError):
        prove(programs, t, budget=1)
