"""Instantiate finite device protocols from source views, without tensor expansion.

This is a progress/access analysis entry, not a value-equivalence entry. Branches
and routes must depend only on static scalar device/program coordinates. All
branches were admitted by the source adapter, including inactive ones.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Literal, cast

import numpy as np

from ..errors import ComparisonBudgetError, CorrespondenceError, UnsupportedJaxprError
from .core_execution import CoreExecutionBinding, validate_binding
from .index_domain import (
    AnalysisBudget,
    IndexExpr,
    check_integer_range,
    integer,
    literal_integer,
    substitute,
)
from .index_domain import (
    axis as coordinate_axis,
)
from .memory import Expression, ValueExpr, _substitute, scalar_type
from .model import (
    ArrayType,
    LiteralValue,
    MeshView,
    Operand,
    OperationView,
    Origin,
    PallasCallView,
    RefTransform,
    RefType,
    RegionView,
    SemaphoreType,
    SliceIndex,
    TensorIndex,
    ValueRef,
)
from .protocol import (
    Access,
    Command,
    DeviceProgram,
    Instance,
    ProtocolEvidence,
    Read,
    Semaphore,
    Signal,
    Start,
    Transfer,
    Wait,
    analyze_protocol,
)
from .remote_coordinates import RemoteView, coordinates, localize, remote_access_view, remote_view
from .remote_io import RemoteIOPlan, plan_remote_io
from .tensor_domain import POINTWISE_PRIMITIVES, TensorDomain, coordinate_ranges
from .tensor_values import tensor_value
from .value_operations import pure_value


@dataclass(frozen=True)
class RemoteProtocolEvidence:
    programs: tuple[DeviceProgram, ...]
    semaphores: tuple[Semaphore, ...]
    protocol: ProtocolEvidence
    mesh: MeshView | None
    io: RemoteIOPlan
    entry_synchronization: str = "default TPU device barrier; no custom barrier or skipped barrier"
    unproved_obligations: tuple[str, ...] = (
        "storage lifetimes and input read-only policy",
        "initialized reads/source versions and complete output coverage",
        "global device-coordinate and value correspondence",
    )
    execution_cores: int | None = None
    execution: CoreExecutionBinding | None = None
    ordered_protocol: ProtocolEvidence | None = None


type StaticValue = Expression | Instance | None


def _literal(value: LiteralValue) -> StaticValue:
    if value.type.shape:
        return None
    kind = np.dtype(value.type.dtype).kind
    if kind in "iu":
        return literal_integer(value, "remote.literal")
    if kind == "b":
        return bool(np.frombuffer(value.data, dtype=value.type.dtype)[0])
    return None


def _constant(value: StaticValue, path: str) -> int:
    if not isinstance(value, IndexExpr) or value.kind != "constant":
        raise UnsupportedJaxprError(path, "protocol control/route must be a static integer")
    return int(value.value)


def _pure(
    op: OperationView,
    args: tuple[StaticValue, ...],
    device: int,
    axis: str | None,
    *,
    program: tuple[int, ...] = (0,),
    programs: tuple[int, ...] = (1,),
) -> StaticValue:
    path = op.origin.path
    if len(op.results) != 1 or not isinstance(op.results[0].type, ArrayType):
        raise UnsupportedJaxprError(path, "unexpected protocol scalar operation")
    typ = op.results[0].type
    if typ.shape:
        return None
    name = op.primitive
    params = {p.name: p.value for p in op.parameters}
    if name == "axis_index":
        if params["axis_name"] != axis:
            raise UnsupportedJaxprError(path, "device axis differs from protocol mesh")
        value = integer(device)
    elif name in ("program_id", "num_programs"):
        dimension = params["axis"]
        if type(dimension) is not int or not 0 <= dimension < len(programs):
            raise UnsupportedJaxprError(path, "unknown program axis")
        value = integer(program[dimension] if name == "program_id" else programs[dimension])
    elif any(v is None or isinstance(v, Instance) for v in args):
        return None
    elif name in ("eq", "ne", "lt", "le", "gt", "ge"):
        a, b = (int(v) if type(v) is bool else _constant(v, path) for v in args)
        return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[
            name
        ]
    elif name == "convert_element_type" and type(args[0]) is bool:
        if np.dtype(typ.dtype).kind == "b":
            return args[0]
        if np.dtype(typ.dtype).kind not in "iu":
            # Floating values belong to the value projection, not the integer
            # domain used for protocol routes and semaphore counts.
            return None
        value = integer(int(args[0]))
    elif name == "clamp":
        lower, item, upper = (_constant(v, path) for v in args)
        if lower > upper:
            raise UnsupportedJaxprError(path, "protocol clamp requires ordered static bounds")
        value = integer(min(max(item, lower), upper))
    elif name == "select_n":
        selected = int(args[0]) if type(args[0]) is bool else _constant(args[0], path)
        if not 0 <= selected < len(args) - 1:
            raise UnsupportedJaxprError(path, "protocol selector outside branch range")
        return args[selected + 1]
    elif name in ("and", "or", "xor", "not") and all(type(v) is bool for v in args):
        if name == "not":
            return not args[0]
        a, b = args
        return bool(a and b) if name == "and" else bool(a or b) if name == "or" else a != b
    elif all(isinstance(v, IndexExpr) for v in args) and np.dtype(typ.dtype).kind in "iu":
        result = pure_value(op, cast(tuple[Expression, ...], args), {}, {})
        return result
    else:
        return None
    check_integer_range(value, typ.dtype, {}, path)
    return value


def _coordinate_value(
    op: OperationView,
    args: tuple[StaticValue, ...],
    device: int,
    name: str | None,
    *,
    program: tuple[int, ...],
    programs: tuple[int, ...],
    input_types: tuple[ArrayType, ...],
    budget: AnalysisBudget,
) -> StaticValue:
    """Track static tensor coordinates for masks without treating loaded data as static."""
    if op.primitive in ("axis_index", "program_id", "num_programs"):
        return _pure(op, args, device, name, program=program, programs=programs)
    if any(isinstance(value, Instance) for value in args):
        return None
    args = tuple(
        ValueExpr("protocol.unknown", scalar_type(typ)) if value is None else value
        for value, typ in zip(args, input_types, strict=True)
    )
    typ = op.results[0].type
    if not isinstance(typ, ArrayType):
        return None
    if (
        not typ.shape
        and op.primitive in POINTWISE_PRIMITIVES
        and all(isinstance(value, IndexExpr | bool) for value in args)
    ):
        static = _pure(op, args, device, name, program=program, programs=programs)
        if static is not None:
            return static
    localized = tuple(
        localize(cast(Expression, value), t.shape, budget)
        for value, t in zip(args, input_types, strict=True)
    )
    value = tensor_value(
        op,
        localized,
        input_types,
        coordinate_ranges("local", typ.shape),
        {str(name): integer(device), **{f"program:{i}": integer(k) for i, k in enumerate(program)}},
        programs,
        budget,
    )
    return _substitute(
        value,
        dict(
            zip(
                (f"local:{i}" for i in range(len(typ.shape))),
                coordinates(coordinate_axis("global"), typ.shape),
                strict=True,
            )
        ),
    )


def remote_domains(call: PallasCallView, budget: AnalysisBudget) -> tuple[TensorDomain, ...]:
    return plan_remote_io(call, budget).domains


def _dma_element_type(typ: ArrayType, path: str) -> ArrayType:
    # Ref indexing may drop dimensions. Rank-dependent empty partitions do not
    # change a shard-local element, but mesh, memory kind and scalar metadata do.
    if typ.sharding is None or any(typ.sharding.partitions):
        raise UnsupportedJaxprError(path, "remote DMA requires shard-local Ref elements")
    return replace(typ, shape=(), sharding=replace(typ.sharding, partitions=()))


def analyze_remote_call(
    call: PallasCallView,
    mesh: MeshView | None,
    *,
    budget: AnalysisBudget,
    execution_cores: int | None = None,
    execution: CoreExecutionBinding | None = None,
) -> RemoteProtocolEvidence:
    """Bind finite full-Ref programs per device and prove protocol progress/order.

    The default backend barrier establishes participant entry. Allocation lifetime,
    initialized source versions and the global output value proof are separate work;
    this evidence is deliberately not admitted as normalized equivalence.
    """
    path = call.kernel.origin.path
    if (
        call.aliases
        or call.interpret
        or call.compiler is None
        or not call.compiler.default_execution_options
    ):
        raise UnsupportedJaxprError(path, "remote protocol needs a default, non-aliased execution")
    if execution_cores is not None and (type(execution_cores) is not int or execution_cores != 1):
        raise ValueError("remote execution_cores requires an explicit single-core binding")
    if execution is not None:
        if execution_cores is not None:
            raise ValueError("execution and execution_cores are mutually exclusive")
        validate_binding(execution, call, budget)
        if len(execution.programs) != 1:
            raise UnsupportedJaxprError(
                path, "remote protocol needs a single-core execution binding"
            )
    if any(g.size > 1 and g.semantics == "parallel" for g in call.grid) and execution is None:
        raise UnsupportedJaxprError(
            path,
            "parallel remote programs require an explicit single-core execution binding "
            "with the same complete grid order on every device",
        )
    if mesh is not None and (
        not mesh.concrete or len(mesh.axes) != 1 or mesh.axes[0][1] != len(mesh.devices)
    ):
        raise UnsupportedJaxprError(path, "remote protocol requires one concrete mesh axis")
    devices = len(mesh.devices) if mesh is not None else 1
    axis_name = mesh.axes[0][0] if mesh is not None else None
    io = plan_remote_io(call, budget)
    domains = io.domains
    if not domains:
        raise UnsupportedJaxprError(path, "remote protocol requires data ports")
    types: dict[int, ArrayType] = {}
    count_dtypes: dict[Instance, str] = {}
    sem_refs: dict[int, str] = {}
    sem_arrays: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    next_resource = io.next_resource
    count = call.num_inputs + call.num_outputs
    if len(call.kernel.inputs) != count + len(call.scratch):
        raise UnsupportedJaxprError(path, "invalid remote scratch bindings")
    for i, v in enumerate(call.kernel.inputs):
        if not isinstance(v.type, RefType):
            raise UnsupportedJaxprError(path, "protocol arguments must be Refs")
        typ = v.type.inner
        if i < count:
            if not isinstance(typ, ArrayType) or typ.shape != domains[i].local_shape:
                raise UnsupportedJaxprError(path, "invalid data Ref domain binding")
            types[v.id] = typ
        else:
            scratch = call.scratch[i - count]
            if scratch.type != v.type or scratch.lifetime != "call":
                raise UnsupportedJaxprError(path, "invalid protocol scratch lifetime/type")
            if isinstance(typ, SemaphoreType):
                if typ.kind not in ("dma", "regular") or scratch.initial_value != 0:
                    raise UnsupportedJaxprError(path, "protocol semaphore lacks zero initial state")
                size = prod(typ.shape)
                budget.spend(path + ".semaphore_slots", size)
                elements = (
                    tuple(range(next_resource, next_resource + size)) if typ.shape else (v.id,)
                )
                if typ.shape:
                    next_resource += size
                sem_arrays[v.id] = (typ.shape, elements)
                sem_refs.update(dict.fromkeys(elements, typ.kind))
            elif scratch.bound_array is not None:
                types[v.id] = scratch.bound_array
            else:
                raise UnsupportedJaxprError(path, "unsupported protocol scratch")

    def region(
        source: RegionView,
        arguments: tuple[StaticValue, ...],
        device: int,
        commands: list[Command],
        suffix: str,
        program: int,
    ) -> tuple[StaticValue, ...]:
        env = dict(zip((v.id for v in source.inputs), arguments, strict=True))
        value_types = {v.id: v.type for v in source.inputs}
        value_types.update((v.id, v.type) for v, _ in source.constants)
        value_types.update((v.id, v.type) for op in source.operations for v in op.results)
        env.update((v.id, _literal(value)) for v, value in source.constants)

        def read(operand: Operand) -> StaticValue:
            return env[operand.id] if isinstance(operand, ValueRef) else _literal(operand)

        def ref(operand: Operand) -> Instance:
            value = read(operand)
            if not isinstance(value, Instance):
                raise UnsupportedJaxprError(source.origin.path, "invalid protocol Ref binding")
            return value

        def mapped_view(
            resource: Instance,
            index: SliceIndex | TensorIndex | Operand | None,
            transforms: tuple[RefTransform, ...],
            loc: str,
        ) -> RemoteView:
            if resource.ref not in types:
                raise UnsupportedJaxprError(loc, "DMA data operand is not an array Ref")
            return remote_view(
                types[resource.ref].shape,
                index,
                transforms,
                lambda value: _constant(read(value), loc),
                budget,
                loc,
            )

        def semaphore(operand: ValueRef, index: TensorIndex | None, loc: str) -> Instance:
            root = ref(operand)
            if root.ref not in sem_arrays:
                raise UnsupportedJaxprError(loc, "protocol semaphore lacks an allocation")
            shape, elements = sem_arrays[root.ref]
            if (index is None and shape) or (index is not None and index.shape != shape):
                raise UnsupportedJaxprError(loc, "protocol semaphore must select a scalar slot")
            indices = (
                ()
                if index is None
                else tuple(_constant(read(cast(Operand, i)), loc) for i in index.indices)
            )
            if len(indices) != len(shape) or any(
                not 0 <= i < n for i, n in zip(indices, shape, strict=True)
            ):
                raise UnsupportedJaxprError(loc, "protocol semaphore slot is out of bounds")
            offset = 0
            for i, n in zip(indices, shape, strict=True):
                offset = offset * n + i
            return Instance(root.device, elements[offset])

        def target(coordinates: tuple[Operand, ...] | None, loc: str) -> int:
            if coordinates is None:
                return device
            if mesh is None or len(coordinates) != 1:
                raise UnsupportedJaxprError(loc, "remote target requires one mesh coordinate")
            result = _constant(read(coordinates[0]), loc)
            if not 0 <= result < devices:
                raise UnsupportedJaxprError(loc, "remote target is outside the mesh")
            return result

        for op in source.operations:
            loc = op.origin.path
            origin = Origin(f"{loc}.device[{device}]" + suffix)
            budget.spend(loc)
            if op.dma is not None:
                dma = op.dma
                destination = target(dma.device_coordinates, loc)
                src, dst, sem = (
                    ref(dma.source),
                    ref(dma.destination),
                    semaphore(dma.semaphore, dma.semaphore_index, loc),
                )
                other = (
                    semaphore(dma.source_semaphore, dma.source_semaphore_index, loc)
                    if dma.source_semaphore is not None
                    else None
                )
                source_view = mapped_view(
                    src, dma.source_tensor_slice or dma.source_slice, dma.source_transforms, loc
                )
                destination_view = mapped_view(
                    dst,
                    dma.destination_tensor_slice or dma.destination_slice,
                    dma.destination_transforms,
                    loc,
                )
                if source_view.shape != destination_view.shape or (
                    _dma_element_type(types[src.ref], loc) != _dma_element_type(types[dst.ref], loc)
                ):
                    raise UnsupportedJaxprError(loc, "remote transfer region/type contracts differ")
                if any(
                    s.start % 128 or s.size % 128
                    for s in (*source_view.spans, *destination_view.spans)
                ):
                    raise UnsupportedJaxprError(
                        loc, "remote DMA requires 128-element aligned intervals"
                    )
                amount = (
                    sum(s.size for s in source_view.spans) * np.dtype(types[src.ref].dtype).itemsize
                )
                counters = (
                    (other, Instance(destination, sem.ref)) if dma.phase == "start" else (sem,)
                )
                for counter in counters:
                    if counter is None:
                        continue
                    if sem_refs.get(counter.ref) != "dma":
                        raise UnsupportedJaxprError(loc, "DMA requires DMA semaphore instances")
                    dtype = types[src.ref].dtype
                    if count_dtypes.setdefault(counter, dtype) != dtype:
                        raise UnsupportedJaxprError(
                            loc, "mixed DMA count dtypes require another rule"
                        )
                if dma.phase == "start":
                    if dma.add is not False or dma.priority != 0:
                        raise UnsupportedJaxprError(loc, "non-default remote DMA options")
                    commands.append(
                        Start(
                            Transfer(
                                src,
                                Instance(destination, dst.ref),
                                source_view.bound,
                                destination_view.bound,
                                other,
                                Instance(destination, sem.ref),
                                amount,
                                source_view.spans,
                                destination_view.spans,
                                substitute(
                                    source_view.to_storage, {"global": destination_view.to_view}
                                )
                                if (
                                    len(types[src.ref].shape) != 1
                                    or len(types[dst.ref].shape) != 1
                                    or dma.source_transforms
                                    or dma.destination_transforms
                                )
                                else None,
                            ),
                            origin,
                        )
                    )
                else:
                    commands.append(
                        Wait(sem, "send" if dma.is_wait_send else "receive", amount, origin)
                    )
            elif op.semaphore is not None:
                sem = op.semaphore
                if sem.core_index is not None:
                    raise UnsupportedJaxprError(loc, "remote semaphore core target needs a binding")
                instance = semaphore(sem.semaphore, sem.index, loc)
                if sem.phase == "read":
                    commands.append(Read(instance, origin))
                    # Counter observations used as control are not static routing.
                    env.update((v.id, None) for v in op.results)
                else:
                    if sem.amount is None:
                        raise UnsupportedJaxprError(loc, "semaphore count is missing")
                    amount = _constant(read(sem.amount), loc)
                    if sem.phase == "signal":
                        commands.append(
                            Signal(
                                Instance(target(sem.device_coordinates, loc), instance.ref),
                                amount,
                                origin,
                            )
                        )
                    else:
                        commands.append(Wait(instance, "regular", amount, origin))
            elif op.primitive == "jit" and len(op.regions) == 1:
                binding = op.regions[0]
                outputs = region(
                    binding.region,
                    tuple(read(a) for a in binding.arguments),
                    device,
                    commands,
                    suffix,
                    program,
                )
                env.update((v.id, value) for v, value in zip(op.results, outputs, strict=True))
            elif op.primitive == "cond":
                selector = _constant(read(op.operands[0]), loc)
                if not 0 <= selector < len(op.regions):
                    raise UnsupportedJaxprError(
                        loc, "conditional selector outside reviewed branch range"
                    )
                binding = op.regions[selector]
                outputs = region(
                    binding.region,
                    tuple(read(a) for a in binding.arguments),
                    device,
                    commands,
                    suffix,
                    program,
                )
                env.update((v.id, value) for v, value in zip(op.results, outputs, strict=True))
            elif op.primitive == "scan":
                params = {p.name: p.value for p in op.parameters}
                const_slots, carry_slots, xs_slots = cast(
                    tuple[tuple[int, ...], ...], params["input_groups"]
                )
                carry_out, ys_out = cast(tuple[tuple[int, ...], ...], params["output_groups"])
                if xs_slots or ys_out:
                    raise UnsupportedJaxprError(
                        loc, "remote scan requires finite scalar carry without history"
                    )
                consts = tuple(read(op.operands[i]) for i in const_slots)
                carry = tuple(read(op.operands[i]) for i in carry_slots)
                length = cast(int, params["length"])
                for iteration in range(length - 1, -1, -1) if params["reverse"] else range(length):
                    budget.spend(loc + ".iteration")
                    result = region(
                        op.regions[0].region,
                        consts + carry,
                        device,
                        commands,
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
                    budget.spend(loc + ".iteration")
                    tag = suffix + f".iteration[{iteration}]"
                    (condition,) = region(
                        op.regions[0].region, consts[:nc] + carry, device, commands, tag, program
                    )
                    if type(condition) is not bool:
                        raise UnsupportedJaxprError(
                            loc, "remote while requires static finite control"
                        )
                    if not condition:
                        break
                    carry = region(
                        op.regions[1].region, consts[nc:] + carry, device, commands, tag, program
                    )
                    iteration += 1
                env.update((v.id, x) for v, x in zip(op.results, carry, strict=True))
            elif op.access is not None:
                access = op.access
                storage = ref(access.ref)

                def access_expression(operand: Operand, loc: str = loc) -> Expression:
                    result = read(operand)
                    if result is None or isinstance(result, Instance):
                        raise UnsupportedJaxprError(
                            loc, "remote access mask/index must use static coordinates"
                        )
                    return result

                selected = remote_access_view(
                    types[storage.ref].shape,
                    access.index,
                    access.transforms,
                    access_expression,
                    budget,
                    loc,
                    (
                        cast(Expression | None, read(access.mask))
                        if access.mask is not None
                        else None
                    ),
                    lambda v: cast(
                        ArrayType, value_types[v.id] if isinstance(v, ValueRef) else v.type
                    ),
                )
                if selected.spans:
                    commands.append(
                        Access(
                            storage,
                            selected.bound,
                            access.value is not None,
                            origin,
                            selected.spans,
                        )
                    )
                env.update(
                    (v.id, ValueExpr("protocol.unknown", scalar_type(cast(ArrayType, v.type))))
                    for v in op.results
                )
            else:
                if op.regions or op.call is not None:
                    raise UnsupportedJaxprError(loc, "unsupported nested protocol operation")
                result = _coordinate_value(
                    op,
                    tuple(read(a) for a in op.operands),
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
                env[op.results[0].id] = result
        return tuple(read(v) for v in source.outputs)

    programs = []
    semaphores = []
    invocations: dict[tuple[int, int], tuple[Command, ...]] = {}
    try:
        for device in range(devices):
            budget.spend(path)
            for stream, points in enumerate(io.grid.streams):
                commands: list[Command] = []
                for program in points:
                    start = len(commands)
                    budget.spend(path + ".program")
                    commands.extend(
                        a.command for a in io.accesses(program, device, path, before=True)
                    )
                    region(
                        call.kernel,
                        tuple(Instance(device, v.id) for v in call.kernel.inputs),
                        device,
                        commands,
                        io.grid.suffix(program),
                        program,
                    )
                    commands.extend(
                        a.command for a in io.accesses(program, device, path, before=False)
                    )
                    invocations[device, program] = tuple(commands[start:])
                programs.append(DeviceProgram(device, tuple(commands), stream=stream))
            semaphores.extend(
                Semaphore(Instance(device, ref), 0, cast(Literal["dma", "regular"], kind))
                for ref, kind in sem_refs.items()
            )
    except RecursionError as error:
        raise ComparisonBudgetError(path, "protocol region nesting limit exceeded") from error
    program_tuple, semaphore_tuple = tuple(programs), tuple(semaphores)
    proof = analyze_protocol(program_tuple, semaphore_tuple, budget=budget)
    ordered = None
    if execution is not None:
        # Concurrent owners can switch away from a blocked invocation, which
        # the bound single core cannot do. Prove its actual order separately;
        # these additional edges never justify shared-memory race freedom.
        point_ids = {point: i for i, point in enumerate(io.grid.points)}
        ordered = analyze_protocol(
            tuple(
                DeviceProgram(
                    device,
                    tuple(
                        command
                        for point in execution.programs[0]
                        for command in invocations[device, point_ids[point]]
                    ),
                )
                for device in range(devices)
            ),
            semaphore_tuple,
            budget=budget,
        )
    if execution_cores == 1 or execution is not None:
        proof = replace(
            proof,
            assumptions=proof.assumptions
            + ("caller binds execution to one local core (core 0) on every device",)
            + (
                ("all devices execute the exact same explicitly bound grid point sequence",)
                if execution is not None
                else ()
            ),
        )
    return RemoteProtocolEvidence(
        program_tuple,
        semaphore_tuple,
        proof,
        mesh,
        io,
        execution_cores=execution_cores,
        execution=execution,
        ordered_protocol=ordered,
        entry_synchronization=(
            "default TPU device barrier; no custom barrier or skipped barrier"
            if mesh is not None
            else "one local participant"
        ),
    )


def verify_remote_protocol(
    call: PallasCallView,
    mesh: MeshView | None,
    evidence: RemoteProtocolEvidence,
    *,
    budget: AnalysisBudget,
) -> None:
    """Replay source bindings, counts and all-schedule ordering; not an equivalence proof."""
    if evidence.execution_cores is not None and (
        type(evidence.execution_cores) is not int or evidence.execution_cores != 1
    ):
        raise CorrespondenceError(call.kernel.origin.path, "invalid remote execution binding")
    try:
        replay = analyze_remote_call(
            call,
            mesh,
            budget=budget,
            execution_cores=evidence.execution_cores,
            execution=evidence.execution,
        )
    except (UnsupportedJaxprError, ValueError) as error:
        raise CorrespondenceError(
            call.kernel.origin.path, "remote protocol evidence does not replay"
        ) from error
    if replay != evidence:
        raise CorrespondenceError(
            call.kernel.origin.path, "remote protocol evidence does not replay"
        )
