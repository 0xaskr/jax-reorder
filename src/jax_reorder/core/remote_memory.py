"""Eliminate remote state only after proving all-schedule conflict ordering.

Device programs are finite; array coordinates remain symbolic. A legal counter
execution evaluates already ordered accesses, never supplies missing
dependencies. This module consumes source-free views, not JAX or SymPy objects.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, replace
from math import prod
from typing import Literal, cast

from ..errors import UnsupportedJaxprError
from .core_execution import CoreExecutionBinding
from .index_domain import (
    AnalysisBudget,
    IndexExpr,
    add,
    axis,
    integer,
)
from .masked_access import require_defined, resolve_defined
from .memory import (
    Expression,
    MemoryProjection,
    Predicate,
    Uninitialized,
    ValueExpr,
    _assemble,
    _literal,
    _substitute,
    expressions_equal,
    scalar_type,
)
from .model import (
    ArrayType,
    Operand,
    OperationView,
    Origin,
    PallasCallView,
    RefType,
    RegionView,
    ShardMapView,
    ValueRef,
    ValueType,
)
from .protocol import Access, Instance, Start
from .remote import RemoteProtocolEvidence, _constant, _coordinate_value, analyze_remote_call
from .remote_coordinates import box_spans, coordinates, linear, remote_access_view
from .remote_io import StagingAccess
from .storage import Piece, Span, Storage, VersionSlice
from .tensor_domain import coordinate_ranges
from .tensor_storage import Box

type Binding = Expression | Instance | Uninitialized


@dataclass(frozen=True)
class RemoteVersion:
    storage: Instance
    version: int
    span: Span
    value: Expression
    origin: Origin


@dataclass(frozen=True)
class RemoteTransferEvidence:
    start: int
    send_complete: int
    receive_complete: int
    source_versions: tuple[VersionSlice, ...]
    destination_versions: tuple[VersionSlice, ...]


@dataclass(frozen=True)
class ShardPlacement:
    role: str
    port: int
    storage: Instance
    span: Span
    shape: tuple[int, ...] = ()
    offsets: tuple[int, ...] = ()


@dataclass(frozen=True)
class RemoteValueEvidence:
    protocol: RemoteProtocolEvidence
    versions: tuple[RemoteVersion, ...]
    transfers: tuple[RemoteTransferEvidence, ...]
    assembly: tuple[ShardPlacement, ...]
    discharged_obligations: tuple[str, ...]
    # Device-local projection outputs are ordered by (kernel port, device).
    # Empty retains the existing parent-function/global assembly contract.
    local_outputs: tuple[tuple[int, int], ...] = ()


def same_remote_evidence(a: RemoteValueEvidence | None, b: RemoteValueEvidence | None) -> bool:
    if a is None or b is None:
        return a is b
    return (
        (a.protocol, a.transfers, a.assembly, a.discharged_obligations, a.local_outputs)
        == (b.protocol, b.transfers, b.assembly, b.discharged_obligations, b.local_outputs)
        and tuple((v.storage, v.version, v.span, v.origin) for v in a.versions)
        == tuple((v.storage, v.version, v.span, v.origin) for v in b.versions)
        and expressions_equal(
            tuple(v.value for v in a.versions), tuple(v.value for v in b.versions)
        )
    )


def has_protocol(region: RegionView) -> bool:
    pending = [region]
    while pending:
        for op in pending.pop().operations:
            if op.semaphore or (
                op.dma and (op.dma.source_semaphore is not None or op.dma.device_coordinates)
            ):
                return True
            pending.extend(binding.region for binding in op.regions)
    return False


def _value(
    op: OperationView,
    args: tuple[Expression, ...],
    device: int,
    name: str | None,
    *,
    program: tuple[int, ...] = (0,),
    programs: tuple[int, ...] = (1,),
    input_types: tuple[ArrayType, ...] = (),
    budget: AnalysisBudget | None = None,
) -> Expression:
    if len(op.results) != 1 or not isinstance(op.results[0].type, ArrayType):
        raise UnsupportedJaxprError(op.origin.path, "remote projection requires one array result")
    assert budget is not None
    value = _coordinate_value(
        op,
        args,
        device,
        name,
        program=program,
        programs=programs,
        input_types=input_types,
        budget=budget,
    )
    assert value is not None and not isinstance(value, Instance)
    return value


def functionalize_remote_call(
    call: PallasCallView,
    shard: ShardMapView | None,
    inputs: tuple[ValueType, ...],
    outputs: tuple[ValueType, ...],
    *,
    budget: AnalysisBudget,
    execution_cores: int | None = None,
    execution: CoreExecutionBinding | None = None,
    scope: Literal["global", "device_local"] = "global",
) -> tuple[MemoryProjection, RemoteValueEvidence]:
    """Prove remote state, optionally retaining separate device-local outputs.

    Local scope keeps the same peer program/protocol and globally bound input
    identities. Each (port, device) output has the kernel's local array shape;
    parent output partitioning and cross-device replication are not its claims.
    """
    if scope not in ("global", "device_local"):
        raise ValueError("remote projection scope must be global or device_local")
    path = call.kernel.origin.path
    proof = analyze_remote_call(
        call,
        shard.mesh if shard is not None else None,
        budget=budget,
        execution_cores=execution_cores,
        execution=execution,
    )
    if call.num_outputs == 0:
        raise UnsupportedJaxprError(path, "remote projection requires outputs")
    axis_name, devices = shard.mesh.axes[0][:2] if shard is not None else (None, 1)
    input_partitions = shard.input_partitions if shard is not None else ((),) * len(inputs)
    output_partitions = shard.output_partitions if shard is not None else ((),) * len(outputs)
    if scope == "device_local":
        outputs = tuple(block.array for block in call.blocks[call.num_inputs :])
        output_partitions = ((),) * len(outputs)
    io, domains = proof.io, proof.io.domains
    access_commands = {
        command.origin: command
        for program in proof.programs
        for command in program.commands
        if isinstance(command, Access)
    }
    refs = call.kernel.inputs
    count = call.num_inputs + call.num_outputs
    input_resources = {r for p in io.ports if p.input for r in (p.ref, p.hbm)}
    types: dict[int, ArrayType] = {}
    assembly: list[ShardPlacement] = []
    memories: dict[Instance, Storage[Expression]] = {}
    versions: list[RemoteVersion] = []
    origins: list[Origin] = []

    def placements(
        role: str,
        contracts: tuple[ValueType, ...],
        partitions: tuple[tuple[tuple[str, ...], ...], ...],
        begin: int,
        end: int,
    ) -> None:
        if len(contracts) != end - begin or len(partitions) != len(contracts):
            raise UnsupportedJaxprError(path, "invalid global shard port bindings")
        for port, (typ, part, block) in enumerate(
            zip(contracts, partitions, call.blocks[begin:end], strict=True)
        ):
            local = block.array
            axes = part + ((),) * (len(local.shape) - len(part))
            if (
                len(axes) != len(local.shape)
                or any(a not in ((), (axis_name,)) for a in axes)
                or sum(bool(a) for a in axes) > 1
            ):
                raise UnsupportedJaxprError(path, "global assembly requires one tensor mesh axis")
            shape = tuple(n * (devices if a else 1) for n, a in zip(local.shape, axes, strict=True))
            if (
                not isinstance(typ, ArrayType)
                or typ.shape != shape
                or (typ.dtype, typ.weak_type) != (local.dtype, local.weak_type)
            ):
                raise UnsupportedJaxprError(path, "global/local shard contracts differ")
            for device in range(devices):
                budget.spend(path)
                offsets = tuple(
                    device * n if a else 0 for n, a in zip(local.shape, axes, strict=True)
                )
                offset = sum(k * prod(shape[i + 1 :]) for i, k in enumerate(offsets))
                assembly.append(
                    ShardPlacement(
                        role,
                        port,
                        Instance(device, io.ports[begin + port].hbm),
                        Span(
                            offset,
                            offset
                            + 1
                            + sum(
                                (n - 1) * prod(shape[i + 1 :]) for i, n in enumerate(local.shape)
                            ),
                        ),
                        local.shape,
                        offsets,
                    )
                )

    placements("input", inputs, input_partitions, 0, call.num_inputs)
    placements("output", outputs, output_partitions, call.num_inputs, count)
    inputs_by_instance = {p.storage: p for p in assembly if p.role == "input"}
    for slot, ref in enumerate(refs):
        assert isinstance(ref.type, RefType)  # Checked by source protocol instantiation.
        typ = ref.type.inner
        if not isinstance(typ, ArrayType):
            continue
        if slot >= count:
            scratch = call.scratch[slot - count]
            if scratch.initial_value is not None or scratch.bound_array is None:
                raise UnsupportedJaxprError(
                    path, "remote scratch must be an uninitialized call allocation"
                )
            typ = scratch.bound_array
        if typ.sharding is None or any(typ.sharding.partitions):
            raise UnsupportedJaxprError(path, "remote storage requires local vectors")
        types[ref.id] = typ
        for device in range(devices):
            budget.spend(path)
            storage = Instance(device, ref.id)
            placement = inputs_by_instance.get(storage)
            value = (
                ValueExpr(
                    "input",
                    scalar_type(typ),
                    tuple(
                        add(c, integer(k))
                        for c, k in zip(
                            coordinates(axis("global"), typ.shape), placement.offsets, strict=True
                        )
                    ),
                    input=placement.port,
                )
                if placement is not None
                else None
            )
            memories[storage] = Storage(prod(typ.shape), value, call.kernel.origin, budget)

    for port in io.ports:
        if not port.buffered:
            continue
        typ = replace(types[port.ref], shape=port.shape)
        types[port.hbm] = typ
        for device in range(devices):
            storage = Instance(device, port.hbm)
            placement = inputs_by_instance.get(storage)
            value = (
                ValueExpr(
                    "input",
                    scalar_type(typ),
                    tuple(
                        add(c, integer(k))
                        for c, k in zip(
                            coordinates(axis("global"), typ.shape), placement.offsets, strict=True
                        )
                    ),
                    input=placement.port,
                )
                if placement is not None
                else None
            )
            memories[storage] = Storage(port.size, value, call.kernel.origin, budget)

    def invalidate(storage: Instance, origin: Origin) -> None:
        memory = memories[storage]
        memory.version += 1
        memory.pieces = [Piece(memory.extent, None, memory.version, origin)]

    def write(
        storage: Instance,
        span: Span,
        value: Expression,
        origin: Origin,
        *,
        internal: bool = False,
        conditional: bool = False,
    ) -> None:
        if storage.ref in input_resources and not internal:
            raise UnsupportedJaxprError(origin.path, "input storage must remain read-only")
        budget.spend(origin.path)
        # The protocol proof orders every conflicting access under all allowed
        # schedules. Replay therefore publishes each real write's latest version,
        # including repeated output HBM commits; it never supplies missing order.
        if not conditional:
            value = resolve_defined(
                value, {"global": (span.start, span.stop - 1)}, budget, origin.path
            )
        version = memories[storage].write(span, value, origin, overwrite=True)
        versions.append(RemoteVersion(storage, version, span, value, origin))

    def select(
        mask: Expression, inactive: Expression, active: Expression, typ: ArrayType, loc: str
    ) -> Expression:
        # A literal boolean complement is the same predicate with exchanged
        # branches. This does not infer relationships between distinct masks.
        while (
            isinstance(mask, Predicate | ValueExpr)
            and mask.operation == "not"
            and (isinstance(mask, Predicate) or mask.type.dtype == "bool")
        ):
            budget.spend(loc + ".mask_complement")
            (mask,) = mask.operands
            inactive, active = active, inactive
        branches = [inactive, active]
        for index, branch in enumerate(branches):
            while (
                isinstance(branch, ValueExpr)
                and branch.operation == "select_n"
                and len(branch.operands) == 3
                and not branch.parameters
                and expressions_equal((mask,), (branch.operands[0],))
            ):
                budget.spend(loc + ".mask_assumption")
                branch = branch.operands[index + 1]
            branches[index] = branch
        return ValueExpr("select_n", scalar_type(typ), (mask, *branches))

    def defined_pieces(
        pieces: tuple[Piece[Expression], ...], loc: str
    ) -> tuple[Piece[Expression], ...]:
        result = []
        for piece in pieces:
            assert piece.value is not None
            result.append(
                replace(
                    piece,
                    value=resolve_defined(
                        piece.value,
                        {"global": (piece.span.start, piece.span.stop - 1)},
                        budget,
                        loc,
                    ),
                )
            )
        return tuple(result)

    def evaluate(
        source: RegionView,
        arguments: tuple[Binding, ...],
        device: int,
        suffix: str,
        program: int,
    ) -> Generator[Origin, None, tuple[Binding, ...]]:
        env = dict(zip((v.id for v in source.inputs), arguments, strict=True))
        value_types = {v.id: v.type for v in source.inputs}
        value_types.update((v.id, v.type) for v, _ in source.constants)
        value_types.update((v.id, v.type) for op in source.operations for v in op.results)
        env.update((v.id, _literal(value, source.origin.path)) for v, value in source.constants)

        def read(operand: Operand) -> Binding:
            value = (
                env[operand.id]
                if isinstance(operand, ValueRef)
                else _literal(operand, source.origin.path)
            )
            if isinstance(value, Uninitialized):
                raise UnsupportedJaxprError(
                    value.origin.path, "uninitialized old swap value is consumed"
                )
            return value

        def expression(operand: Operand) -> Expression:
            value = read(operand)
            if isinstance(value, Instance | Uninitialized):
                raise UnsupportedJaxprError(source.origin.path, "expected a value, not a Ref")
            return value

        for op in source.operations:
            budget.spend(op.origin.path)
            origin = Origin(f"{op.origin.path}.device[{device}]" + suffix)
            origins.append(origin)
            if op.semaphore is not None:
                yield origin
                if op.semaphore.phase == "read":
                    env[op.results[0].id] = integer(observations[origin])
            elif op.dma is not None:
                yield origin
            elif op.primitive == "jit" and len(op.regions) == 1:
                binding = op.regions[0]
                result = yield from evaluate(
                    binding.region,
                    tuple(read(a) for a in binding.arguments),
                    device,
                    suffix,
                    program,
                )
                env.update((v.id, x) for v, x in zip(op.results, result, strict=True))
            elif op.primitive == "cond":
                selector = expression(op.operands[0])
                selected = _constant(
                    selector if isinstance(selector, IndexExpr | bool) else None, origin.path
                )
                if not 0 <= selected < len(op.regions):
                    raise UnsupportedJaxprError(origin.path, "invalid static branch")
                branch = op.regions[selected]
                result = yield from evaluate(
                    branch.region, tuple(read(v) for v in branch.arguments), device, suffix, program
                )
                env.update((v.id, x) for v, x in zip(op.results, result, strict=True))
            elif op.primitive == "scan":
                params = {p.name: p.value for p in op.parameters}
                const_slots, carry_slots, xs_slots = cast(
                    tuple[tuple[int, ...], ...], params["input_groups"]
                )
                carry_out, ys_out = cast(tuple[tuple[int, ...], ...], params["output_groups"])
                if xs_slots or ys_out:
                    raise UnsupportedJaxprError(
                        origin.path, "remote scan requires finite scalar carry without history"
                    )
                consts = tuple(read(op.operands[i]) for i in const_slots)
                carry = tuple(read(op.operands[i]) for i in carry_slots)
                length = cast(int, params["length"])
                for iteration in range(length - 1, -1, -1) if params["reverse"] else range(length):
                    budget.spend(origin.path + ".iteration")
                    result = yield from evaluate(
                        op.regions[0].region,
                        consts + carry,
                        device,
                        suffix + f".iteration[{iteration}]",
                        program,
                    )
                    carry = tuple(result[i] for i in carry_out)
                env.update((v.id, x) for v, x in zip(op.results, carry, strict=True))
            elif op.primitive == "while":
                params = {p.name: p.value for p in op.parameters}
                nc, nb = cast(int, params["cond_nconsts"]), cast(int, params["body_nconsts"])
                consts = tuple(read(a) for a in op.operands[: nc + nb])
                carry = tuple(read(a) for a in op.operands[nc + nb :])
                iteration = 0
                while True:
                    budget.spend(origin.path + ".iteration")
                    tag = suffix + f".iteration[{iteration}]"
                    (condition,) = yield from evaluate(
                        op.regions[0].region, consts[:nc] + carry, device, tag, program
                    )
                    if type(condition) is not bool:
                        raise UnsupportedJaxprError(
                            origin.path, "remote while requires static finite control"
                        )
                    if not condition:
                        break
                    carry = yield from evaluate(
                        op.regions[1].region, consts[nc:] + carry, device, tag, program
                    )
                    iteration += 1
                env.update((v.id, x) for v, x in zip(op.results, carry, strict=True))
            elif op.access is not None:
                access = op.access
                storage = read(access.ref)
                if not isinstance(storage, Instance):
                    raise UnsupportedJaxprError(origin.path, "invalid remote access binding")
                mask = expression(access.mask) if access.mask is not None else None
                if mask is not None:
                    assert access.mask is not None
                    mask_type = cast(
                        ArrayType,
                        value_types[access.mask.id]
                        if isinstance(access.mask, ValueRef)
                        else access.mask.type,
                    )
                    require_defined(
                        mask, {"global": (0, prod(mask_type.shape) - 1)}, budget, origin.path
                    )
                selected = remote_access_view(
                    types[storage.ref].shape,
                    access.index,
                    access.transforms,
                    expression,
                    budget,
                    origin.path,
                    mask,
                    lambda v: cast(
                        ArrayType, value_types[v.id] if isinstance(v, ValueRef) else v.type
                    ),
                )
                # A protocol data mask can overapproximate the exact value mask.
                # Replay its conservative event even when the actual mask is empty.
                command = access_commands.get(origin)
                coverage = iter(command.spans if command is not None else ())
                covered = next(coverage, None)
                for span in selected.spans:
                    cursor = span.start
                    while cursor < span.stop:
                        budget.spend(origin.path + ".access_coverage")
                        if covered is not None and covered.stop <= cursor:
                            covered = next(coverage, None)
                            continue
                        if covered is None or covered.start > cursor:
                            raise UnsupportedJaxprError(
                                origin.path, "value access exceeds the proven protocol footprint"
                            )
                        cursor = min(span.stop, covered.stop)
                if command is not None:
                    yield origin
                result_type = cast(ArrayType, op.results[0].type)
                undefined = ValueExpr("undefined", scalar_type(result_type))
                pieces = tuple(
                    piece
                    for span in selected.spans
                    for piece in memories[storage].read(span, origin.path, initialized=False)
                )
                buffered_input = any(
                    p.input and p.buffered and p.ref == storage.ref for p in io.ports
                )
                pieces = tuple(
                    replace(p, value=undefined) if p.value is None else p for p in pieces
                )
                previous = _assemble(pieces, types[storage.ref]) if pieces else undefined
                assert not isinstance(previous, Uninitialized)
                previous = _substitute(previous, {"global": selected.to_storage})
                old = previous
                if mask is not None:
                    other = expression(access.other) if access.other is not None else undefined
                    old = select(mask, other, old, result_type, origin.path)
                if access.value is None and not buffered_input:
                    # Inactive load results may remain undefined, but active
                    # reads must already be initialized even when unused later.
                    checked = (
                        previous
                        if mask is None
                        else select(mask, True, previous, result_type, origin.path)
                    )
                    require_defined(
                        checked,
                        {"global": (0, prod(result_type.shape) - 1)},
                        budget,
                        origin.path,
                    )
                if access.value is not None:
                    value = expression(access.value)
                    if selected.conditional and selected.spans:
                        # The conservative footprint covers every possible write;
                        # untouched lanes retain both the previous value and its
                        # conditional initialization. They are not read by store.
                        assert mask is not None
                        require_defined(
                            select(mask, True, value, result_type, origin.path),
                            {"global": (0, prod(result_type.shape) - 1)},
                            budget,
                            origin.path,
                        )
                        value = select(mask, previous, value, result_type, origin.path)
                    value = _substitute(value, {"global": selected.to_view})
                    for span in selected.spans:
                        write(storage, span, value, origin, conditional=selected.conditional)
                if len(op.results) != 1:
                    raise UnsupportedJaxprError(origin.path, "invalid access result arity")
                env[op.results[0].id] = old
            else:
                if op.call is not None or op.regions:
                    raise UnsupportedJaxprError(origin.path, "unsupported remote value region")
                value = _value(
                    op,
                    tuple(expression(v) for v in op.operands),
                    device,
                    axis_name,
                    program=io.grid.points[program],
                    programs=io.grid.shape,
                    input_types=tuple(
                        cast(ArrayType, value_types[v.id] if isinstance(v, ValueRef) else v.type)
                        for v in op.operands
                    ),
                    budget=budget,
                )
                env[op.results[0].id] = value
        return tuple(read(v) for v in source.outputs)

    observations = {
        proof.protocol.events[observation.event].origin: observation.value
        for observation in proof.protocol.observations
    }

    io_origins = set()

    def execute_stream(device: int, stream: int) -> Generator[Origin, None, None]:
        staged: dict[int, tuple[Piece[Expression], ...]] = {}

        def perform(accesses: tuple[StagingAccess, ...]) -> Generator[Origin, None, None]:
            for event in accesses:
                command, port = event.command, io.ports[event.port]
                io_origins.add(command.origin)
                yield command.origin
                pairs = port.segments(event.program)
                if event.phase in ("input_read", "restore_read", "output_read"):
                    incoming = event.phase != "output_read"
                    result = []
                    for local, global_ in pairs:
                        source, target = (global_, local) if incoming else (local, global_)
                        delta = target.start - source.start
                        for piece in defined_pieces(
                            memories[command.storage].read(source, command.origin.path),
                            command.origin.path,
                        ):
                            assert piece.value is not None
                            result.append(
                                Piece(
                                    Span(piece.span.start + delta, piece.span.stop + delta),
                                    _substitute(
                                        piece.value,
                                        {"global": add(axis("global"), integer(-delta))},
                                    ),
                                    piece.version,
                                    piece.origin,
                                )
                            )
                    staged[event.port] = tuple(result)
                    if not incoming:
                        invalidate(command.storage, command.origin)
                else:
                    if event.phase in ("input_write", "restore_write"):
                        invalidate(command.storage, command.origin)
                    for piece in staged.pop(event.port):
                        assert piece.value is not None
                        write(
                            command.storage, piece.span, piece.value, command.origin, internal=True
                        )

        for program in io.grid.streams[stream]:
            budget.spend(path + ".program")
            yield from perform(io.accesses(program, device, path, before=True))
            yield from evaluate(
                call.kernel,
                tuple(Instance(device, v.id) for v in refs),
                device,
                io.grid.suffix(program),
                program,
            )
            yield from perform(io.accesses(program, device, path, before=False))

    programs = {(p.device, p.stream): p for p in proof.programs}
    machines = {key: execute_stream(*key) for key in programs}
    pending = {key: next(machine, None) for key, machine in machines.items()}
    starts = {
        c.origin: (c, event.id)
        for event in proof.protocol.events
        if event.kind == "start"
        for c in (programs[event.device, event.stream].commands[event.command],)
        if isinstance(c, Start)
    }
    payloads: dict[Origin, tuple[Piece[Expression], ...]] = {}
    snapshots: dict[Origin, tuple[VersionSlice, ...]] = {}
    sent: dict[Origin, int] = {}
    transfers = []
    for event_id in proof.protocol.order:
        event = proof.protocol.events[event_id]
        budget.spend(event.origin.path)
        if event.kind == "signal_complete":
            continue
        if event.kind in ("send_complete", "receive_complete"):
            start, start_id = starts[event.origin]
            transfer = start.transfer
            if event.kind == "send_complete":
                if (
                    tuple(
                        version
                        for span in transfer.source_spans
                        for version in memories[transfer.source].versions(span, event.origin.path)
                    )
                    != snapshots[event.origin]
                ):
                    raise UnsupportedJaxprError(
                        event.origin.path, "remote source versions changed in flight"
                    )
                sent[event.origin] = event.id
            else:
                incoming_pieces = payloads.pop(event.origin)
                if transfer.destination_to_source is not None:
                    incoming = _assemble(incoming_pieces, types[transfer.source.ref])
                    assert not isinstance(incoming, Uninitialized)
                    incoming = _substitute(incoming, {"global": transfer.destination_to_source})
                    for span in transfer.destination_spans:
                        write(transfer.destination, span, incoming, event.origin)
                else:
                    delta = transfer.destination_span.start - transfer.source_span.start
                    for piece in incoming_pieces:
                        assert piece.value is not None
                        value = _substitute(
                            piece.value, {"global": add(axis("global"), integer(-delta))}
                        )
                        write(
                            transfer.destination,
                            Span(piece.span.start + delta, piece.span.stop + delta),
                            value,
                            event.origin,
                        )
                transfers.append(
                    RemoteTransferEvidence(
                        start_id,
                        sent.pop(event.origin),
                        event.id,
                        snapshots.pop(event.origin),
                        tuple(
                            version
                            for span in transfer.destination_spans
                            for version in memories[transfer.destination].versions(
                                span, event.origin.path
                            )
                        ),
                    )
                )
            continue
        key = (event.device, event.stream)
        command = programs[key].commands[event.command]
        if pending[key] != event.origin:
            raise UnsupportedJaxprError(
                event.origin.path, "value execution differs from proven device program"
            )
        if isinstance(command, Start):
            transfer = command.transfer
            payloads[event.origin] = defined_pieces(
                tuple(
                    piece
                    for span in transfer.source_spans
                    for piece in memories[transfer.source].read(span, event.origin.path)
                ),
                event.origin.path,
            )
            snapshots[event.origin] = tuple(
                version
                for span in transfer.source_spans
                for version in memories[transfer.source].versions(span, event.origin.path)
            )
        elif (
            isinstance(command, Access)
            and command.write
            and command.storage.ref in input_resources
            and command.origin not in io_origins
        ):
            raise UnsupportedJaxprError(event.origin.path, "input storage must remain read-only")
        pending[key] = next(machines[key], None)
    if any(origin is not None for origin in pending.values()) or payloads or snapshots or sent:
        raise UnsupportedJaxprError(path, "remote execution has unfinished operations")

    def output_value(value: Expression, shape: tuple[int, ...]) -> Expression:
        if len(shape) != 1:
            value = _substitute(
                value,
                {"global": linear(tuple(axis(f"output:{i}") for i in range(len(shape))), shape)},
            )
        return resolve_defined(
            value,
            coordinate_ranges("output", shape)
            if len(shape) != 1
            else {"global": (0, prod(shape) - 1)},
            budget,
            path,
        )

    global_outputs = []
    local_outputs = []
    for port, contract in enumerate(outputs):
        assert isinstance(contract, ArrayType)
        pieces: list[Piece[Expression]] = []
        replicas: list[Expression] = []
        partitioned = any(output_partitions[port])
        for placement in (p for p in assembly if p.role == "output" and p.port == port):
            memory = memories[placement.storage]
            local = memory.read(memory.extent, path)
            value = _assemble(local, types[placement.storage.ref])
            assert not isinstance(value, Uninitialized)
            if scope == "device_local":
                global_outputs.append(output_value(value, contract.shape))
                local_outputs.append((port, placement.storage.device))
                continue
            if partitioned:
                global_coordinates = coordinates(axis("global"), contract.shape)
                to_local = linear(
                    tuple(
                        add(c, integer(-k))
                        for c, k in zip(global_coordinates, placement.offsets, strict=True)
                    ),
                    placement.shape,
                )
                value = _substitute(value, {"global": to_local})
                box = Box(
                    tuple(
                        Span(k, k + n)
                        for k, n in zip(placement.offsets, placement.shape, strict=True)
                    )
                )
                for span in box_spans(box, contract.shape, budget, path):
                    pieces.append(Piece(span, value, 0, call.kernel.origin))
            else:
                replicas.append(value)
        if scope == "device_local":
            continue
        if partitioned:
            output = _assemble(tuple(sorted(pieces, key=lambda p: p.span)), contract)
            assert not isinstance(output, Uninitialized)
        else:
            if not all(expressions_equal((replicas[0],), (v,)) for v in replicas[1:]):
                raise UnsupportedJaxprError(path, "replicated output values differ across devices")
            output = replicas[0]
        global_outputs.append(output_value(output, contract.shape))
    evidence = RemoteValueEvidence(
        proof,
        tuple(versions),
        tuple(transfers),
        tuple(assembly),
        proof.unproved_obligations,
        tuple(local_outputs),
    )
    return MemoryProjection(
        tuple(global_outputs),
        domains,
        (),
        tuple(origins),
        (
            "all conflicting accesses are ordered by original protocol completion dependencies",
            "remote start snapshots initialized source versions, protected until send completion",
            "receive completion publishes destination versions; waits consume counts",
            "inputs are read-only; scratch has call lifetime; no transfer is outstanding at exit",
            "output windows retain versions until their individual automatic writeback; "
            "ordered HBM overwrites publish the latest version without restoring staging",
            "each ordered (kernel output port, device) retains its complete local value; "
            "parent output assembly and replica equality are not claimed"
            if scope == "device_local"
            else "global shard placement is exact; replicated outputs agree across devices",
        ),
    ), evidence
