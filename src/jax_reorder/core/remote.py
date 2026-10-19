"""Instantiate finite device protocols from source views, without tensor expansion.

This is a progress/access analysis entry, not a value-equivalence entry. Branches
and routes must depend only on static scalar device/program coordinates. All
branches were admitted by the source adapter, including inactive ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ..errors import ComparisonBudgetError, CorrespondenceError, UnsupportedJaxprError
from .index_domain import (
    AnalysisBudget,
    IndexExpr,
    check_integer_range,
    integer,
    integer_operation,
    literal_integer,
    prove_block_domain,
)
from .model import (
    ArrayType,
    LiteralValue,
    MeshView,
    Operand,
    OperationView,
    Origin,
    PallasCallView,
    RefType,
    RegionView,
    SemaphoreType,
    SliceIndex,
    ValueRef,
)
from .protocol import (
    Access,
    Command,
    DeviceProgram,
    Instance,
    ProtocolEvidence,
    Semaphore,
    Start,
    Transfer,
    Wait,
    analyze_protocol,
)
from .storage import Span


@dataclass(frozen=True)
class RemoteProtocolEvidence:
    programs: tuple[DeviceProgram, ...]
    semaphores: tuple[Semaphore, ...]
    protocol: ProtocolEvidence
    mesh: MeshView
    entry_synchronization: str = "default TPU device barrier; no custom barrier or skipped barrier"
    unproved_obligations: tuple[str, ...] = (
        "storage lifetimes and input read-only policy",
        "initialized reads/source versions and complete output coverage",
        "global device-coordinate and value correspondence",
    )


type StaticValue = IndexExpr | bool | Instance | None


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


def _pure(op: OperationView, args: tuple[StaticValue, ...], device: int, axis: str) -> StaticValue:
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
        if params["axis"] != 0:
            raise UnsupportedJaxprError(path, "unknown program axis")
        value = integer(0 if name == "program_id" else 1)
    elif any(v is None or isinstance(v, Instance) for v in args):
        return None
    elif name in ("eq", "ne", "lt", "le", "gt", "ge"):
        a, b = (_constant(v, path) for v in args)
        return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[
            name
        ]
    elif name == "convert_element_type" and type(args[0]) is bool:
        value = integer(int(args[0]))
    elif name == "clamp":
        lower, item, upper = (_constant(v, path) for v in args)
        if lower > upper:
            raise UnsupportedJaxprError(path, "protocol clamp requires ordered static bounds")
        value = integer(min(max(item, lower), upper))
    elif name in ("and", "or", "xor", "not") and all(type(v) is bool for v in args):
        if name == "not":
            return not args[0]
        a, b = args
        return bool(a and b) if name == "and" else bool(a or b) if name == "or" else a != b
    elif all(isinstance(v, IndexExpr) for v in args) and np.dtype(typ.dtype).kind in "iu":
        value = integer_operation(name, cast(tuple[IndexExpr, ...], args), {}, path)
    else:
        return None
    check_integer_range(value, typ.dtype, {}, path)
    return value


def analyze_remote_call(
    call: PallasCallView, mesh: MeshView, *, budget: AnalysisBudget
) -> RemoteProtocolEvidence:
    """Bind one full-block program to each device and prove protocol progress/order.

    The default backend barrier establishes participant entry. Allocation lifetime,
    initialized source versions and the global output value proof are separate work;
    this evidence is deliberately not admitted as normalized equivalence.
    """
    path = call.kernel.origin.path
    if (
        call.aliases
        or call.interpret
        or len(call.grid) != 1
        or call.grid[0].size != 1
        or call.grid[0].semantics not in ("arbitrary", "parallel")
        or call.compiler is None
        or not call.compiler.default_execution_options
    ):
        raise UnsupportedJaxprError(path, "remote protocol needs one default, non-aliased program")
    if not mesh.concrete or len(mesh.axes) != 1 or mesh.axes[0][1] != len(mesh.devices):
        raise UnsupportedJaxprError(path, "remote protocol requires one concrete mesh axis")
    domains = tuple(prove_block_domain(block, 1, budget) for block in call.blocks)
    if not domains or any(d.block != d.size for d in domains):
        raise UnsupportedJaxprError(path, "remote protocol requires full unpadded blocks")
    types: dict[int, ArrayType] = {}
    count_dtypes: dict[Instance, str] = {}
    sem_refs = []
    count = call.num_inputs + call.num_outputs
    if len(call.kernel.inputs) != count + len(call.scratch):
        raise UnsupportedJaxprError(path, "invalid remote scratch bindings")
    for i, v in enumerate(call.kernel.inputs):
        if not isinstance(v.type, RefType):
            raise UnsupportedJaxprError(path, "protocol arguments must be Refs")
        typ = v.type.inner
        if i < count:
            if not isinstance(typ, ArrayType) or typ.shape != (domains[i].block,):
                raise UnsupportedJaxprError(path, "invalid data Ref domain binding")
            types[v.id] = typ
        else:
            scratch = call.scratch[i - count]
            if scratch.type != v.type or scratch.lifetime != "call":
                raise UnsupportedJaxprError(path, "invalid protocol scratch lifetime/type")
            if isinstance(typ, SemaphoreType):
                if typ.kind != "dma" or scratch.initial_value != 0:
                    raise UnsupportedJaxprError(path, "protocol semaphore lacks zero initial state")
                sem_refs.append(v.id)
            elif scratch.bound_array is not None and len(typ.shape) == 1:
                types[v.id] = scratch.bound_array
            else:
                raise UnsupportedJaxprError(path, "unsupported protocol scratch")

    def span(ref: Instance, index: SliceIndex | None, loc: str) -> Span:
        if ref.ref not in types:
            raise UnsupportedJaxprError(loc, "DMA data operand is not an array Ref")
        extent = types[ref.ref].shape[0]
        index = index or SliceIndex(0, extent, 1)
        if (
            index.stride != 1
            or index.size <= 0
            or index.start < 0
            or index.start + index.size > extent
        ):
            raise UnsupportedJaxprError(loc, "invalid remote contiguous interval")
        return Span(index.start, index.start + index.size)

    def region(
        source: RegionView, arguments: tuple[StaticValue, ...], device: int, commands: list[Command]
    ) -> tuple[StaticValue, ...]:
        env = dict(zip((v.id for v in source.inputs), arguments, strict=True))
        env.update((v.id, _literal(value)) for v, value in source.constants)

        def read(operand: Operand) -> StaticValue:
            return env[operand.id] if isinstance(operand, ValueRef) else _literal(operand)

        def ref(operand: Operand) -> Instance:
            value = read(operand)
            if not isinstance(value, Instance):
                raise UnsupportedJaxprError(source.origin.path, "invalid protocol Ref binding")
            return value

        for op in source.operations:
            loc = op.origin.path
            origin = Origin(f"{loc}.device[{device}]")
            budget.spend(loc)
            if op.dma is not None:
                dma = op.dma
                if dma.source_semaphore is None or dma.device_coordinates is None:
                    raise UnsupportedJaxprError(
                        loc, "mixed local/remote protocol is not yet instantiated"
                    )
                if len(dma.device_coordinates) != 1:
                    raise UnsupportedJaxprError(loc, "remote target must have one mesh coordinate")
                target = _constant(read(dma.device_coordinates[0]), loc)
                if not 0 <= target < len(mesh.devices):
                    raise UnsupportedJaxprError(loc, "remote target is outside the mesh")
                src, dst, sem, other = (
                    ref(dma.source),
                    ref(dma.destination),
                    ref(dma.semaphore),
                    ref(dma.source_semaphore),
                )
                a, b = span(src, dma.source_slice, loc), span(dst, dma.destination_slice, loc)
                if a.size != b.size or types[src.ref] != types[dst.ref]:
                    raise UnsupportedJaxprError(loc, "remote transfer region/type contracts differ")
                if any(s.start % 128 or s.size % 128 for s in (a, b)):
                    raise UnsupportedJaxprError(
                        loc, "remote DMA requires 128-element aligned intervals"
                    )
                amount = a.size * np.dtype(types[src.ref].dtype).itemsize
                counters = (other, Instance(target, sem.ref)) if dma.phase == "start" else (sem,)
                for counter in counters:
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
                                Instance(target, dst.ref),
                                a,
                                b,
                                other,
                                Instance(target, sem.ref),
                                amount,
                            ),
                            origin,
                        )
                    )
                else:
                    commands.append(
                        Wait(sem, "send" if dma.is_wait_send else "receive", amount, origin)
                    )
            elif op.primitive == "cond":
                selector = _constant(read(op.operands[0]), loc)
                if not 0 <= selector < len(op.regions):
                    raise UnsupportedJaxprError(
                        loc, "conditional selector outside reviewed branch range"
                    )
                binding = op.regions[selector]
                outputs = region(
                    binding.region, tuple(read(a) for a in binding.arguments), device, commands
                )
                env.update((v.id, value) for v, value in zip(op.results, outputs, strict=True))
            elif op.access is not None:
                access = op.access
                if not isinstance(access.index, SliceIndex) or access.mask is not None:
                    raise UnsupportedJaxprError(
                        loc, "remote access requires an unmasked static slice"
                    )
                storage = ref(access.ref)
                commands.append(
                    Access(
                        storage, span(storage, access.index, loc), access.value is not None, origin
                    )
                )
                env.update((v.id, None) for v in op.results)
            else:
                if op.regions or op.call is not None:
                    raise UnsupportedJaxprError(loc, "unsupported nested protocol operation")
                result = _pure(op, tuple(read(a) for a in op.operands), device, mesh.axes[0][0])
                env[op.results[0].id] = result
        return tuple(read(v) for v in source.outputs)

    programs = []
    semaphores = []
    try:
        for device in range(len(mesh.devices)):
            budget.spend(path)
            commands: list[Command] = []
            region(
                call.kernel,
                tuple(Instance(device, v.id) for v in call.kernel.inputs),
                device,
                commands,
            )
            programs.append(DeviceProgram(device, tuple(commands)))
            semaphores.extend(Semaphore(Instance(device, ref), 0) for ref in sem_refs)
    except RecursionError as error:
        raise ComparisonBudgetError(path, "protocol region nesting limit exceeded") from error
    program_tuple, semaphore_tuple = tuple(programs), tuple(semaphores)
    proof = analyze_protocol(program_tuple, semaphore_tuple, budget=budget)
    return RemoteProtocolEvidence(program_tuple, semaphore_tuple, proof, mesh)


def verify_remote_protocol(
    call: PallasCallView,
    mesh: MeshView,
    evidence: RemoteProtocolEvidence,
    *,
    budget: AnalysisBudget,
) -> None:
    """Replay source bindings, counts and all-schedule ordering; not an equivalence proof."""
    if analyze_remote_call(call, mesh, budget=budget) != evidence:
        raise CorrespondenceError(
            call.kernel.origin.path, "remote protocol evidence does not replay"
        )
