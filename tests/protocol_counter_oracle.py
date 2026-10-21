"""Exhaustive tiny counter machine with incremental DMA/signal progress.

The machine never schedules by proof edges. It checks each claimed edge against
all reachable executions, including receive progress before full send completion.
"""

from jax_reorder.core.protocol import Access, Read, Signal, Start, Wait


def audit(programs, proof):
    commands = {(p.device, i): c for p in programs for i, c in enumerate(p.commands)}
    commands_to_event = {
        (e.device, e.command): e.id
        for e in proof.events
        if e.kind in ("access", "read", "signal", "start", "wait")
    }
    slots = {s: i for i, s in enumerate(sorted({p.semaphore for p in proof.producers}))}
    transfers = [(k, c.transfer) for k, c in commands.items() if isinstance(c, Start)]
    signals = [
        (k, c) for k, c in commands.items() if isinstance(c, Signal) and c.semaphore.device != k[0]
    ]
    transfer_ids = {k: i for i, (k, _) in enumerate(transfers)}
    signal_ids = {k: i for i, (k, _) in enumerate(signals)}
    completions = {(e.origin, e.kind): e.id for e in proof.events if e.kind.endswith("_complete")}
    observations = {o.event: o.value for o in proof.observations}
    initial = (
        (0,) * len(programs),
        (-1,) * (2 * len(transfers)),
        (-1,) * len(signals),
        (0,) * len(slots),
        0,
    )
    pending, seen = [initial], set()
    finished = 0

    def occur(event, mask):
        assert not proof.ancestors[event] & ~mask, (proof.events[event], mask)
        return mask | (1 << event)

    while pending:
        state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        pcs, progress, deliveries, counts, mask = state
        next_states = []
        for n, p in enumerate(programs):
            pc = pcs[n]
            if pc == len(p.commands):
                continue
            key = (p.device, pc)
            command = p.commands[pc]
            ps, ds, cs, positions = list(progress), list(deliveries), list(counts), list(pcs)
            if isinstance(command, Start):
                i = transfer_ids[key]
                ps[2 * i] = ps[2 * i + 1] = 0
            elif isinstance(command, Wait):
                slot = slots[command.semaphore]
                if cs[slot] < command.logical_bytes:
                    continue
                cs[slot] -= command.logical_bytes
            elif isinstance(command, Signal):
                if key in signal_ids:
                    ds[signal_ids[key]] = 0
                else:
                    cs[slots[command.semaphore]] += command.amount
            else:
                assert isinstance(command, Access | Read)
                if isinstance(command, Read):
                    assert cs[slots[command.semaphore]] == observations[commands_to_event[key]]
            positions[n] += 1
            next_states.append(
                (
                    tuple(positions),
                    tuple(ps),
                    tuple(ds),
                    tuple(cs),
                    occur(commands_to_event[key], mask),
                )
            )
        for i, (key, t) in enumerate(transfers):
            sent, received = progress[2 * i : 2 * i + 2]
            for receiving in (False, True):
                value = received if receiving else sent
                if value < 0 or value == t.logical_bytes or (receiving and received >= sent):
                    continue
                ps, cs = list(progress), list(counts)
                ps[2 * i + int(receiving)] += 1
                sem = t.receive if receiving else t.send
                if sem is not None:
                    cs[slots[sem]] += 1
                next_mask = mask
                if value + 1 == t.logical_bytes:
                    origin = commands[key].origin
                    event = completions[
                        origin, "receive_complete" if receiving else "send_complete"
                    ]
                    next_mask = occur(event, mask)
                next_states.append((pcs, tuple(ps), deliveries, tuple(cs), next_mask))
        for i, (key, signal) in enumerate(signals):
            value = deliveries[i]
            if value < 0 or value == signal.amount:
                continue
            ds, cs = list(deliveries), list(counts)
            ds[i] += 1
            cs[slots[signal.semaphore]] += 1
            next_mask = mask
            if ds[i] == signal.amount:
                next_mask = occur(completions[commands[key].origin, "signal_complete"], mask)
            next_states.append((pcs, progress, tuple(ds), tuple(cs), next_mask))
        if not next_states:
            assert all(pc == len(p.commands) for p, pc in zip(programs, pcs, strict=True)), state
            assert all(
                progress[2 * i] == progress[2 * i + 1] == t.logical_bytes
                for i, (_, t) in enumerate(transfers)
            ), state
            assert all(n == s.amount for n, (_, s) in zip(deliveries, signals, strict=True)), state
            assert not any(counts), state
            finished += 1
        pending.extend(next_states)
    assert finished
    return len(seen)
