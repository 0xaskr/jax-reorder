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
    split = prove((programs[0], partial), a, b)
    recv = [c for c in split.counters if c.kind == "receive"]
    assert not recv[0].required_completions
    assert len(recv[1].required_completions) == 2


@pytest.mark.parametrize(
    "change,reason",
    [
        (0, "not conserved"),
        (1, "no consumer"),
        (2, "invalid wait instance/count"),
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
        programs = (programs[0], replace(programs[1], commands=(replace(receive, kind="regular"),)))
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
    assert len(prove(programs, t, budget=128).events) == 5
    with pytest.raises(ComparisonBudgetError):
        prove(programs, t, budget=1)


def test_incremental_dma_credits_do_not_complete_an_indispensable_producer():
    from tests.protocol_counter_oracle import audit

    a = replace(transfer(0, 1), logical_bytes=1)
    b = replace(transfer(0, 1, span=Span(128, 256)), logical_bytes=2)
    first, second = Start(a, Origin("a")), Start(b, Origin("b"))
    partial = replace(wait(b, "receive"), origin=Origin("partial"))
    final = replace(wait(a, "receive"), origin=Origin("final"))
    programs = (
        DeviceProgram(0, (first, second, replace(wait(a, "send"), logical_bytes=3))),
        DeviceProgram(1, (partial, final)),
    )
    proof = prove(programs, a, b)
    first_wait = next(c for c in proof.counters if c.kind == "receive")
    assert not first_wait.required_completions
    assert audit(programs, proof) > 30
    read = Access(b.destination, b.destination_span, False, Origin("early read of second DMA"))
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        prove((programs[0], replace(programs[1], commands=(partial, read, final))), a, b)


def test_regular_signal_does_not_join_outstanding_dma_completion():
    from jax_reorder.core.protocol import Signal

    t = replace(transfer(0, 1), logical_bytes=2)
    sem = Instance(1, 20)
    programs = (
        DeviceProgram(0, (start(t), Signal(sem, 1, Origin("notify")), wait(t, "send"))),
        DeviceProgram(
            1,
            (
                Wait(sem, "regular", 1, Origin("notified")),
                Access(t.destination, t.destination_span, False, Origin("unsafe read")),
                wait(t, "receive"),
            ),
        ),
    )
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        analyze_protocol(
            programs, (*semaphores(t), Semaphore(sem, 0, "regular")), budget=AnalysisBudget()
        )


def test_acknowledgement_proves_reuse_and_missing_ack_fails():
    from jax_reorder.core.protocol import Signal
    from tests.protocol_counter_oracle import audit

    t = replace(transfer(0, 1), logical_bytes=2)
    sem = Instance(0, 20)
    ack = Wait(sem, "regular", 1, Origin("ack"))
    signal = Signal(sem, 1, Origin("release buffer"))
    read = Access(t.destination, t.destination_span, False, Origin("first read"))
    a = DeviceProgram(
        0,
        (
            Start(t, Origin("first")),
            wait(t, "send"),
            ack,
            Start(t, Origin("second")),
            wait(t, "send"),
        ),
    )
    b = DeviceProgram(1, (wait(t, "receive"), read, signal, wait(t, "receive")))
    sems = (*semaphores(t), Semaphore(sem, 0, "regular"))
    proof = analyze_protocol((a, b), sems, budget=AnalysisBudget())
    assert audit((a, b), proof) > 20
    unsafe = replace(a, commands=(a.commands[0], a.commands[1], a.commands[3], ack, a.commands[4]))
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        analyze_protocol((unsafe, b), sems, budget=AnalysisBudget())


def test_mixed_local_remote_and_partial_regular_credits_have_no_role_identity():
    from jax_reorder.core.protocol import Signal
    from tests.protocol_counter_oracle import audit

    remote = replace(transfer(0, 1), logical_bytes=2)
    local = replace(transfer(1, 1, span=Span(128, 256)), send=None, logical_bytes=1)
    regular = Instance(1, 20)
    a = DeviceProgram(
        0, (start(remote), Signal(regular, 2, Origin("signal a")), wait(remote, "send"))
    )
    b = DeviceProgram(
        1,
        (
            Start(local, Origin("local")),
            Signal(regular, 1, Origin("signal b")),
            Wait(regular, "regular", 2, Origin("partial signal")),
            Wait(regular, "regular", 1, Origin("remaining signal")),
            replace(wait(remote, "receive"), logical_bytes=3, kind="local"),
        ),
    )
    sems = (*semaphores(remote), Semaphore(regular, 0, "regular"))
    proof = analyze_protocol((a, b), sems, budget=AnalysisBudget())
    assert audit((a, b), proof) > 100


def test_counter_reads_require_schedule_invariance_and_dma_stable_zero():
    from jax_reorder.core.protocol import Read, Signal
    from tests.protocol_counter_oracle import audit

    sem = Instance(0, 20)
    a = DeviceProgram(
        0,
        (
            Signal(sem, 3, Origin("signal")),
            Read(sem, Origin("read")),
            Wait(sem, "regular", 3, Origin("wait")),
        ),
    )
    proof = analyze_protocol((a,), (Semaphore(sem, 0, "regular"),), budget=AnalysisBudget())
    assert proof.observations[0].value == 3
    audit((a,), proof)
    t = replace(transfer(0, 1), logical_bytes=2)
    a = DeviceProgram(0, (start(t), wait(t, "send")))
    b = DeviceProgram(1, (wait(t, "receive"), Read(t.receive, Origin("stable zero"))))
    proof = prove((a, b), t)
    assert proof.observations[0].value == 0
    audit((a, b), proof)
    b = replace(b, commands=tuple(reversed(b.commands)))
    with pytest.raises(UnsupportedJaxprError, match="depends on scheduling"):
        prove((a, b), t)


def test_regular_zero_wait_has_no_acquire_and_repeated_generations_can_exceed_cumulative_max():
    from jax_reorder.core.communication import MAX_DMA_BYTES
    from jax_reorder.core.protocol import Signal

    sem = Instance(0, 20)
    commands = (
        Wait(sem, "regular", 0, Origin("zero")),
        Signal(sem, MAX_DMA_BYTES, Origin("first")),
        Wait(sem, "regular", MAX_DMA_BYTES, Origin("first wait")),
        Signal(sem, MAX_DMA_BYTES, Origin("second")),
        Wait(sem, "regular", MAX_DMA_BYTES, Origin("second wait")),
    )
    proof = analyze_protocol(
        (DeviceProgram(0, commands),), (Semaphore(sem, 0, "regular"),), budget=AnalysisBudget()
    )
    assert not proof.counters[0].required_completions and not proof.counters[0].required_issues
    unsafe = (commands[1], commands[3], commands[2], commands[4])
    with pytest.raises(UnsupportedJaxprError, match="live count may overflow"):
        analyze_protocol(
            (DeviceProgram(0, unsafe),), (Semaphore(sem, 0, "regular"),), budget=AnalysisBudget()
        )


def test_incremental_progress_audits_all_small_partial_wait_permutations():
    from tests.protocol_counter_oracle import audit

    a = replace(transfer(0, 1), logical_bytes=1)
    b = replace(transfer(0, 1, span=Span(128, 256)), logical_bytes=2)
    commands = (
        Start(a, Origin("a")),
        Start(b, Origin("b")),
        replace(wait(a, "send"), origin=Origin("send one")),
        replace(wait(b, "send"), origin=Origin("send two")),
    )
    receive = (
        replace(wait(a, "receive"), origin=Origin("recv one")),
        replace(wait(b, "receive"), origin=Origin("recv two")),
    )
    admitted = 0
    for sender, receiver in product(permutations(commands), permutations(receive)):
        programs = (DeviceProgram(0, sender), DeviceProgram(1, receiver))
        try:
            proof = prove(programs, a, b)
        except UnsupportedJaxprError as error:
            assert "wait dependency cycle" in error.reason
            continue
        audit(programs, proof)
        admitted += 1
    assert admitted > 0


@pytest.mark.parametrize(
    "footprint",
    [
        (Span(0, 0),),
        (Span(1, 0),),
        (Span(0, 128), Span(200, 128)),
        (Span(0, 128), Span(64, 192)),
        (Span(0, 128), Span(256, 384)),
        (Span(128, 256), Span(0, 128)),
    ],
)
def test_transfer_footprint_requires_positive_ordered_disjoint_in_bounds_spans(footprint):
    t = replace(
        transfer(0, 1, span=Span(0, 256)),
        source_footprint=footprint,
        destination_footprint=footprint,
    )
    programs = (
        DeviceProgram(0, (start(t), wait(t, "send"))),
        DeviceProgram(1, (wait(t, "receive"),)),
    )
    with pytest.raises(UnsupportedJaxprError, match="invalid transfer footprint"):
        prove(programs, t)
