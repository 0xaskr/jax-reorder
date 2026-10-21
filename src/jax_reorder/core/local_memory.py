"""Bounded local control and resource proofs over independent blocked programs.

Every dynamic branch is checked. Scratch is initially unknown on every iteration
and counters must return to zero, proving independence despite call-long storage.
Remote endpoints are deliberately left to the cross-device protocol analyzer.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from ..errors import UnsupportedJaxprError
from .communication import MAX_DMA_BYTES, LocalDmaState, TransferEvidence
from .index_domain import (
    AnalysisBudget,
    IndexExpr,
    add,
    axis,
    bounds,
    integer,
    prove_block_domain,
    substitute,
)
from .memory import (
    Expression,
    MemoryProjection,
    MemoryVersion,
    Predicate,
    Uninitialized,
    ValueExpr,
    _assemble,
    _literal,
    _substitute,
    expressions_equal,
    scalar_type,
    truth,
)
from .model import (
    ArrayType,
    Operand,
    Origin,
    PallasCallView,
    RefType,
    RegionView,
    SemaphoreType,
    SliceIndex,
    TensorIndex,
    ValueRef,
)
from .storage import Piece, Span, Storage
from .tensor_domain import multidimensional, restoring_outputs
from .tensor_memory import functionalize_tensor_call
from .value_operations import pure_value, static_integer


@dataclass(frozen=True)
class Ref:
    slot: int


type Binding = Expression | Ref | Uninitialized


def has_unmasked_padding(call: PallasCallView) -> bool:
    """Select the richer lane-validity projector only when it is needed."""
    if not any(
        any(n % width for n, width in zip(b.array.shape, b.block_shape, strict=True))
        for b in call.blocks
    ):
        return False
    pending = [call.kernel]
    while pending:
        for op in pending.pop().operations:
            if op.access and op.access.value is None and op.access.mask is None:
                return True
            pending.extend(r.region for r in op.regions)
    return False


def functionalize_local_call(
    call: PallasCallView,
    *,
    budget: AnalysisBudget,
    inputs: tuple[Expression, ...] | None = None,
    coordinates: dict[str, IndexExpr] | None = None,
    axis_ranges: dict[str, tuple[int, int]] | None = None,
    execution_cores: int | None = None,
    defer_padding: bool = False,
) -> MemoryProjection:
    if execution_cores is not None and (type(execution_cores) is not int or execution_cores != 1):
        raise ValueError("execution_cores currently requires an explicit single-core binding")
    if (
        restoring_outputs(call)
        or multidimensional(call)
        or defer_padding
        and has_unmasked_padding(call)
    ):
        return functionalize_tensor_call(
            call,
            budget=budget,
            inputs=inputs,
            coordinates=coordinates,
            axis_ranges=axis_ranges,
            execution_cores=execution_cores,
            defer_padding=defer_padding,
        )
    try:
        return _functionalize_independent_call(
            call,
            budget=budget,
            inputs=inputs,
            coordinates=coordinates,
            axis_ranges=axis_ranges,
            execution_cores=execution_cores,
        )
    except UnsupportedJaxprError as error:
        if not (
            any(g.size > 1 and g.semantics == "arbitrary" for g in call.grid)
            and error.reason
            in (
                "grid leaves a coverage gap or redundant programs",
                "local programs require aligned block extents",
                "local control requires full disjoint output blocks",
                "read of uninitialized output/scratch storage",
                "dynamic branches leave different counter states",
                "dynamic branch exits with outstanding DMA",
                "local semaphore counts are not conserved at program exit",
            )
        ):
            raise
        # Keep the symbolic independent-program proof (including huge grids)
        # when it succeeds. Persistent scratch or output windows with different
        # lifetimes need the stricter, budgeted serial proof on this fallback.
        return functionalize_tensor_call(
            call,
            budget=budget,
            inputs=inputs,
            coordinates=coordinates,
            axis_ranges=axis_ranges,
            execution_cores=execution_cores,
            defer_padding=defer_padding,
        )


def _functionalize_independent_call(
    call: PallasCallView,
    *,
    budget: AnalysisBudget,
    inputs: tuple[Expression, ...] | None,
    coordinates: dict[str, IndexExpr] | None,
    axis_ranges: dict[str, tuple[int, int]] | None,
    execution_cores: int | None,
) -> MemoryProjection:
    path = call.kernel.origin.path
    if call.interpret or len(call.grid) != 1 or call.num_outputs == 0:
        raise UnsupportedJaxprError(path, "unsupported local kernel execution contract")
    if call.compiler is None or not call.compiler.default_execution_options:
        raise UnsupportedJaxprError(path, "non-default compiler options need a local proof")
    if call.grid[0].semantics not in ("arbitrary", "parallel"):
        raise UnsupportedJaxprError(path, "core/subcore scheduling needs a hardware binding proof")
    programs = call.grid[0].size
    domains = tuple(prove_block_domain(b, programs, budget) for b in call.blocks)
    target = domains[call.num_inputs]
    if any((d.size, d.block) != (target.size, target.block) for d in domains):
        raise UnsupportedJaxprError(path, "local programs require aligned block extents")
    if target.size != target.block * programs or any(
        d.mapping != target.mapping for d in domains[call.num_inputs :]
    ):
        raise UnsupportedJaxprError(path, "local control requires full disjoint output blocks")
    npre = len(call.prefetch)
    if inputs is not None and len(inputs) != npre + call.num_inputs:
        raise UnsupportedJaxprError(path, "local input expression arity mismatch")
    count = npre + call.num_inputs + call.num_outputs
    if len(call.kernel.inputs) != count + len(call.scratch):
        raise UnsupportedJaxprError(path, "invalid local Ref bindings")
    # Blocked kernel Refs are separate staging buffers. input_output_aliases
    # aliases the external buffers at copy-out, not these local Ref identities.
    for i, o in call.aliases:
        if domains[i - npre] != domains[call.num_inputs + o]:
            raise UnsupportedJaxprError(
                path, "alias requires identical input/output block mappings"
            )
    ranges = {"global": (0, target.block - 1), "program": (0, programs - 1), **(axis_ranges or {})}
    coordinate_values = {
        "program_id": axis("program"),
        "num_programs": integer(programs),
        **(coordinates or {}),
    }
    memories: dict[int, Storage[Expression]] = {}
    types: dict[int, ArrayType] = {}
    regular: dict[int, int] = {}
    dma_semaphores: set[int] = set()
    readonly = set(range(npre + call.num_inputs))
    for slot, ref in enumerate(call.kernel.inputs):
        if not isinstance(ref.type, RefType):
            raise UnsupportedJaxprError(path, "local argument is not a Ref")
        typ = ref.type.inner
        if isinstance(typ, SemaphoreType):
            scratch = call.scratch[slot - count]
            if scratch.initial_value != 0 or scratch.lifetime != "call":
                raise UnsupportedJaxprError(
                    path, "semaphore requires call lifetime and zero initial state"
                )
            if typ.kind == "regular":
                regular[slot] = 0
            elif typ.kind == "dma":
                dma_semaphores.add(slot)
            else:
                raise UnsupportedJaxprError(path, "unknown local counter kind")
            continue
        if slot >= count:
            scratch = call.scratch[slot - count]
            if (
                scratch.bound_array is None
                or scratch.initial_value is not None
                or scratch.lifetime != "call"
            ):
                raise UnsupportedJaxprError(path, "scratch requires uninitialized call allocation")
            typ = scratch.bound_array
        if len(typ.shape) > 1 or typ.sharding is None or any(typ.sharding.partitions):
            raise UnsupportedJaxprError(path, "local storage requires shard-local scalars/vectors")
        types[slot] = typ
        value = None
        if slot < npre + call.num_inputs:
            address = (
                axis("global")
                if slot < npre
                else substitute(domains[slot - npre].address, {"lane": axis("global")})
            )
            value = (
                _substitute(inputs[slot], {"global": address})
                if inputs is not None
                else ValueExpr(
                    "input", scalar_type(typ), (address,) if typ.shape else (), input=slot
                )
            )
        memories[slot] = Storage(
            typ.shape[0] if typ.shape else 1, value, call.kernel.origin, budget
        )
    dma_state = LocalDmaState(dma_semaphores, budget)
    if programs > 1 and call.grid[0].semantics == "parallel" and regular and execution_cores != 1:
        raise UnsupportedJaxprError(
            path, "parallel regular semaphores require an explicit core target binding"
        )
    payloads: dict[int, tuple[Piece[Expression], ...]] = {}
    versions: list[MemoryVersion] = []
    origins: list[Origin] = []

    def write(slot: int, span: Span, value: Expression, origin: Origin) -> int:
        if slot in readonly:
            raise UnsupportedJaxprError(origin.path, "input storage must remain read-only")
        version = memories[slot].write(span, value, origin, overwrite=True)
        versions.append(MemoryVersion(slot, version, value, origin, span))
        return version

    def expression(value: Binding, loc: str) -> Expression:
        if isinstance(value, Uninitialized):
            raise UnsupportedJaxprError(loc, "read of uninitialized output/scratch storage")
        if isinstance(value, Ref):
            raise UnsupportedJaxprError(loc, "expected a value instead of a Ref")
        return value

    def clone_memories() -> dict[int, Storage[Expression]]:
        result = {slot: copy(memory) for slot, memory in memories.items()}
        for memory in result.values():
            memory.pieces = list(memory.pieces)
        return result

    def clone_dma() -> LocalDmaState:
        if dma_state.pending:
            raise UnsupportedJaxprError(path, "dynamic branch requires quiescent DMA at entry")
        result = LocalDmaState(dma_semaphores, budget)
        result.generations = dma_state.generations.copy()
        result.next_id = dma_state.next_id
        result.completed = dma_state.completed.copy()
        result.waits = dma_state.waits.copy()
        return result

    def select(selector: Expression, values: tuple[Expression, ...], typ: ArrayType) -> Expression:
        if all(expressions_equal((values[0],), (v,)) for v in values[1:]):
            return values[0]
        return ValueExpr("select_n", scalar_type(typ), (selector, *values))

    def evaluate(
        source: RegionView, arguments: tuple[Binding, ...], suffix: str = ""
    ) -> tuple[Binding, ...]:
        nonlocal memories, regular, dma_state
        env = dict(zip((v.id for v in source.inputs), arguments, strict=True))
        env.update((v.id, _literal(vv, source.origin.path)) for v, vv in source.constants)
        value_types = {v.id: v.type for v in source.inputs}
        value_types.update((v.id, v.type) for v, _ in source.constants)
        value_types.update((v.id, v.type) for op in source.operations for v in op.results)

        def read(operand: Operand) -> Binding:
            return (
                env[operand.id]
                if isinstance(operand, ValueRef)
                else _literal(operand, source.origin.path)
            )

        def value(operand: Operand) -> Expression:
            return expression(read(operand), source.origin.path)

        def slot(operand: Operand) -> int:
            ref = read(operand)
            if not isinstance(ref, Ref):
                raise UnsupportedJaxprError(source.origin.path, "invalid local resource binding")
            return ref.slot

        for op in source.operations:
            origin = Origin(op.origin.path + suffix)
            loc = origin.path
            budget.spend(loc)
            origins.append(origin)
            if op.semaphore is not None:
                sem = op.semaphore
                resource = slot(sem.semaphore)
                if sem.device_coordinates is not None:
                    raise UnsupportedJaxprError(loc, "remote signal needs peer protocol analysis")
                if sem.core_index is not None and static_integer(value(sem.core_index), loc) != 0:
                    raise UnsupportedJaxprError(loc, "semaphore target differs from bound core 0")
                if sem.phase == "read":
                    if resource in regular:
                        result = regular[resource]
                    elif resource in dma_semaphores and resource not in dma_state.pending:
                        result = 0
                    else:
                        raise UnsupportedJaxprError(loc, "semaphore read depends on DMA scheduling")
                    env[op.results[0].id] = integer(result)
                else:
                    if resource not in regular or sem.amount is None:
                        raise UnsupportedJaxprError(
                            loc, "ordinary counter operation needs a regular semaphore"
                        )
                    amount = static_integer(value(sem.amount), loc)
                    if not 0 <= amount <= MAX_DMA_BYTES:
                        raise UnsupportedJaxprError(loc, "counter amount outside reviewed range")
                    updated = regular[resource] + (amount if sem.phase == "signal" else -amount)
                    if not 0 <= updated <= MAX_DMA_BYTES:
                        raise UnsupportedJaxprError(
                            loc, "local counter wait is unsatisfied or count overflows"
                        )
                    regular[resource] = updated
                continue
            if op.dma is not None:
                dma = op.dma
                if dma.device_coordinates is not None or dma.source_semaphore is not None:
                    raise UnsupportedJaxprError(loc, "remote DMA needs peer protocol analysis")
                src, dst, sem = slot(dma.source), slot(dma.destination), slot(dma.semaphore)
                if (
                    src not in types
                    or dst not in types
                    or dst in readonly
                    or types[src] != types[dst]
                    or not types[src].shape
                ):
                    raise UnsupportedJaxprError(loc, "invalid local DMA data binding")
                a = dma.source_slice or SliceIndex(0, types[src].shape[0], 1)
                b = dma.destination_slice or SliceIndex(0, types[dst].shape[0], 1)
                if a.stride != 1 or b.stride != 1 or a.size != b.size:
                    raise UnsupportedJaxprError(loc, "DMA requires matching contiguous intervals")
                source_span, dest_span = (
                    Span(a.start, a.start + a.size),
                    Span(b.start, b.start + b.size),
                )
                bound_dma = replace(
                    dma, source=ValueRef(src), destination=ValueRef(dst), semaphore=ValueRef(sem)
                )
                dtype = types[src].dtype
                size = a.size * np.dtype(dtype).itemsize
                if dma.phase == "start":
                    pieces = memories[src].read(source_span, loc)
                    memories[dst].read(dest_span, loc, initialized=False)
                    transfer = dma_state.start(
                        bound_dma,
                        origin,
                        source_version=memories[src].version,
                        source_span=source_span,
                        destination_span=dest_span,
                        source_versions=memories[src].versions(source_span, loc),
                        dtype=dtype,
                        logical_bytes=size,
                    )
                    payloads[transfer.id] = pieces
                else:
                    completed = dma_state.wait(
                        bound_dma,
                        origin,
                        source_span=source_span,
                        destination_span=dest_span,
                        dtype=dtype,
                        logical_bytes=size,
                    )
                    for transfer in completed:
                        if (
                            memories[transfer.source].versions(transfer.source_span, loc)
                            != transfer.source_versions
                        ):
                            raise UnsupportedJaxprError(loc, "DMA source changed in flight")
                        delta = transfer.destination_span.start - transfer.source_span.start
                        version = 0
                        for piece in payloads.pop(transfer.id):
                            assert piece.value is not None
                            version = write(
                                transfer.destination,
                                Span(piece.span.start + delta, piece.span.stop + delta),
                                _substitute(
                                    piece.value, {"global": add(axis("global"), integer(-delta))}
                                ),
                                origin,
                            )
                        dma_state.completed.append(TransferEvidence(transfer, origin, version))
                continue
            if op.primitive == "jit" and len(op.regions) == 1:
                binding = op.regions[0]
                results = evaluate(
                    binding.region, tuple(read(a) for a in binding.arguments), suffix
                )
                env.update((v.id, x) for v, x in zip(op.results, results, strict=True))
                continue
            if op.primitive == "cond":
                selector = value(op.operands[0])
                known = truth(selector, ranges)
                if known is not None:
                    selector = integer(int(known))
                if isinstance(selector, IndexExpr) and selector.kind == "constant":
                    index = max(0, min(int(selector.value), len(op.regions) - 1))
                    branch = op.regions[index]
                    results = evaluate(
                        branch.region, tuple(read(a) for a in branch.arguments), suffix
                    )
                else:
                    if (
                        isinstance(selector, ValueExpr)
                        and selector.operation == "convert_element_type"
                        and (
                            isinstance(selector.operands[0], Predicate | bool)
                            or isinstance(selector.operands[0], ValueExpr)
                            and selector.operands[0].type.dtype == "bool"
                        )
                    ):
                        selector = selector.operands[0]
                    if not (
                        isinstance(selector, Predicate | bool)
                        or isinstance(selector, ValueExpr)
                        and selector.type.dtype == "bool"
                    ):
                        operand = op.operands[0]
                        typ = (
                            value_types[operand.id]
                            if isinstance(operand, ValueRef)
                            else operand.type
                        )
                        assert isinstance(typ, ArrayType)
                        selector = ValueExpr(
                            "clamp",
                            scalar_type(typ),
                            (integer(0), selector, integer(len(op.regions) - 1)),
                        )
                    before_mem, before_counts, before_dma = memories, regular, dma_state
                    baseline = len(dma_state.completed), len(dma_state.waits)
                    states, branch_results = [], []
                    for branch in op.regions:
                        memories, regular, dma_state = before_mem, before_counts, before_dma
                        memories, regular, dma_state = clone_memories(), regular.copy(), clone_dma()
                        branch_results.append(
                            evaluate(
                                branch.region, tuple(read(a) for a in branch.arguments), suffix
                            )
                        )
                        if dma_state.pending:
                            raise UnsupportedJaxprError(
                                loc, "dynamic branch exits with outstanding DMA"
                            )
                        states.append((memories, regular, dma_state))
                    if any(counts != states[0][1] for _, counts, _ in states[1:]):
                        raise UnsupportedJaxprError(
                            loc, "dynamic branches leave different counter states"
                        )
                    memories, regular, dma_state = before_mem, states[0][1], before_dma
                    for resource, memory in memories.items():
                        pieces = []
                        points = sorted(
                            {
                                n
                                for m, _, _ in states
                                for p in m[resource].pieces
                                for n in (p.span.start, p.span.stop)
                            }
                        )
                        for begin, end in zip(points[:-1], points[1:], strict=True):
                            span = Span(begin, end)
                            values = tuple(
                                _assemble(
                                    m[resource].read(span, loc, initialized=False), types[resource]
                                )
                                for m, _, _ in states
                            )
                            merged = (
                                None
                                if any(isinstance(v, Uninitialized) for v in values)
                                else select(
                                    selector, cast(tuple[Expression, ...], values), types[resource]
                                )
                            )
                            memory.version += 1
                            pieces.append(Piece(span, merged, memory.version, origin))
                        memory.pieces = pieces
                    for _, _, state in states:
                        dma_state.completed.extend(state.completed[baseline[0] :])
                        dma_state.waits.extend(state.waits[baseline[1] :])
                        dma_state.next_id = max(dma_state.next_id, state.next_id)
                        for resource, generation in state.generations.items():
                            dma_state.generations[resource] = max(
                                dma_state.generations[resource], generation
                            )
                    if any(not isinstance(v.type, ArrayType) for v in op.results):
                        raise UnsupportedJaxprError(
                            loc, "conditional Ref results need an alias proof"
                        )
                    results = tuple(
                        select(
                            selector,
                            tuple(expression(result[i], loc) for result in branch_results),
                            cast(ArrayType, var.type),
                        )
                        for i, var in enumerate(op.results)
                    )
                env.update((v.id, x) for v, x in zip(op.results, results, strict=True))
                continue
            if op.primitive == "scan":
                p = {v.name: v.value for v in op.parameters}
                const_slots, carry_slots, xs_slots = cast(
                    tuple[tuple[int, ...], ...], p["input_groups"]
                )
                carry_out, ys_out = cast(tuple[tuple[int, ...], ...], p["output_groups"])
                body = op.regions[0].region
                consts = tuple(read(op.operands[i]) for i in const_slots)
                carry = tuple(read(op.operands[i]) for i in carry_slots)
                xs = tuple(value(op.operands[i]) for i in xs_slots)
                length = cast(int, p["length"])
                sequence: dict[int, tuple[Binding, ...]] = {}
                indices = range(length - 1, -1, -1) if p["reverse"] else range(length)
                for iteration in indices:
                    budget.spend(loc)
                    sliced = tuple(_substitute(x, {"global": integer(iteration)}) for x in xs)
                    result = evaluate(
                        body, consts + carry + sliced, suffix + f".iteration[{iteration}]"
                    )
                    carry = tuple(result[i] for i in carry_out)
                    sequence[iteration] = tuple(result[i] for i in ys_out)
                result_values = list(carry)
                for i, out in enumerate(ys_out):
                    typ = op.results[out].type
                    if not isinstance(typ, ArrayType) or typ.shape != (length,) or length == 0:
                        raise UnsupportedJaxprError(
                            loc, "scan stacked output needs a nonempty scalar sequence"
                        )
                    pieces = tuple(
                        Piece(Span(j, j + 1), expression(sequence[j][i], loc), 0, origin)
                        for j in range(length)
                    )
                    result_values.append(_assemble(pieces, typ))
                env.update((v.id, x) for v, x in zip(op.results, result_values, strict=True))
                continue
            if op.primitive == "while":
                p = {v.name: v.value for v in op.parameters}
                nc, nb = cast(int, p["cond_nconsts"]), cast(int, p["body_nconsts"])
                consts = tuple(read(a) for a in op.operands[: nc + nb])
                carry = tuple(read(a) for a in op.operands[nc + nb :])
                iteration = 0
                while True:
                    budget.spend(loc)
                    tag = suffix + f".iteration[{iteration}]"
                    (condition,) = evaluate(op.regions[0].region, consts[:nc] + carry, tag)
                    decision = truth(expression(condition, loc), ranges)
                    if decision is None:
                        raise UnsupportedJaxprError(
                            loc, "while termination needs a static finite carry proof"
                        )
                    if not decision:
                        break
                    carry = evaluate(op.regions[1].region, consts[nc:] + carry, tag)
                    iteration += 1
                env.update((v.id, x) for v, x in zip(op.results, carry, strict=True))
                continue
            if op.access is not None:
                access = op.access
                resource = slot(access.ref)
                if resource not in memories or access.mask is not None:
                    raise UnsupportedJaxprError(loc, "local access requires an unmasked array Ref")
                extent = types[resource].shape[0] if types[resource].shape else 1
                if isinstance(access.index, SliceIndex):
                    index = access.index
                    if index.stride != 1:
                        raise UnsupportedJaxprError(loc, "local access requires a contiguous slice")
                    span = Span(index.start, index.start + index.size)
                    address = add(axis("global"), integer(index.start))
                else:
                    if isinstance(access.index, TensorIndex):
                        raise UnsupportedJaxprError(
                            loc, "tensor index requires the tensor memory rule"
                        )
                    index_value = value(access.index)
                    if not isinstance(index_value, IndexExpr):
                        raise UnsupportedJaxprError(
                            loc, "data-dependent scalar index needs a domain proof"
                        )
                    low, high = bounds(index_value, ranges)
                    if low != high or not 0 <= low < extent:
                        raise UnsupportedJaxprError(
                            loc, "scalar Ref index must be statically in bounds"
                        )
                    span, address = Span(low, low + 1), integer(low)
                dma_state.access(resource, span, write=access.value is not None, path=loc)
                old = _assemble(
                    memories[resource].read(span, loc, initialized=False), types[resource]
                )
                if not isinstance(old, Uninitialized):
                    old = _substitute(old, {"global": address})
                if access.value is None:
                    expression(old, loc)
                else:
                    incoming = value(access.value)
                    incoming = _substitute(
                        incoming, {"global": add(axis("global"), integer(-span.start))}
                    )
                    write(resource, span, incoming, origin)
                if access.other is not None:
                    value(access.other)
                if len(op.results) != 1:
                    raise UnsupportedJaxprError(loc, "invalid local access result arity")
                env[op.results[0].id] = old
                continue
            if op.regions or op.call is not None:
                raise UnsupportedJaxprError(loc, "unsupported local value region")
            env[op.results[0].id] = pure_value(
                op, tuple(value(a) for a in op.operands), ranges, coordinate_values
            )
        return tuple(read(o) for o in source.outputs)

    evaluate(call.kernel, tuple(Ref(i) for i in range(len(call.kernel.inputs))))
    if any(regular.values()):
        raise UnsupportedJaxprError(
            path, "local semaphore counts are not conserved at program exit"
        )
    transfers = dma_state.finish(path)
    outputs = tuple(
        expression(_assemble(memories[i].read(memories[i].extent, path), types[i]), path)
        for i in range(npre + call.num_inputs, count)
    )
    program, lane = target.inverse
    bindings = {"global": lane, "program": program}
    return MemoryProjection(
        tuple(_substitute(v, bindings) for v in outputs),
        domains,
        tuple(versions),
        tuple(origins),
        (
            "each output block has one independent program; sequential writes retain versions",
            "all dynamic branches satisfy initialization, bounds and resource obligations",
            "each program initializes scratch before use; exit has zero counters and quiescent DMA",
            "blocked aliases have identical external coordinates; staging inputs remain read-only",
        ),
        transfers,
        tuple(dma_state.waits),
    )
