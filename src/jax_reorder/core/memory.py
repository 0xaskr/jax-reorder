"""Versioned Ref elimination over adapted facts, with symbolic access checks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from ..errors import UnsupportedJaxprError
from .communication import LocalDmaState, TransferEvidence, WaitEvidence
from .index_domain import (
    AnalysisBudget,
    BlockDomain,
    IndexExpr,
    add,
    axis,
    bounds,
    check_integer_range,
    integer,
    integer_operation,
    literal_integer,
    prove_block_domain,
    scale,
    substitute,
)
from .model import (
    ArrayType,
    Attribute,
    LiteralValue,
    Operand,
    Origin,
    PallasCallView,
    ParameterView,
    RefType,
    SemaphoreType,
    SliceIndex,
    TensorIndex,
    ValueRef,
)
from .storage import Piece, Span, Storage
from .tensor_domain import POINTWISE_PRIMITIVES, TensorDomain
from .value_usage import consumed_values


@dataclass(frozen=True)
class Predicate:
    operation: str
    operands: tuple[IndexExpr | Predicate | bool, ...]


@dataclass(frozen=True)
class ScalarType:
    dtype: str
    weak_type: bool
    metadata: tuple[tuple[str, Attribute], ...]


@dataclass(frozen=True)
class ValueExpr:
    operation: str
    type: ScalarType
    operands: tuple[Expression, ...] = ()
    parameters: tuple[ParameterView, ...] = ()
    data: bytes = b""
    input: int | None = None


type Expression = IndexExpr | ValueExpr | Predicate | bool


@dataclass(frozen=True)
class MemoryVersion:
    storage: int
    version: int
    value: Expression
    origin: Origin
    span: Span
    rectangle: tuple[Span, ...] = ()


@dataclass(frozen=True)
class Uninitialized:
    origin: Origin


@dataclass(frozen=True)
class MemoryProjection:
    outputs: tuple[Expression, ...]
    domains: tuple[BlockDomain | TensorDomain, ...]
    versions: tuple[MemoryVersion, ...]
    origins: tuple[Origin, ...]
    conditions: tuple[str, ...]
    communication: tuple[TransferEvidence, ...] = ()
    waits: tuple[WaitEvidence, ...] = ()
    assumptions: tuple[str, ...] = ()


def scalar_type(typ: ArrayType) -> ScalarType:
    # Rank-specific sharding is already checked at admission. Remaining metadata
    # describes each scalar operation and must survive the projection.
    return ScalarType(
        typ.dtype, typ.weak_type, tuple((k, v) for k, v in typ.metadata if k != "sharding")
    )


def _literal(value: LiteralValue, path: str) -> Expression:
    if value.type.shape:
        raise UnsupportedJaxprError(
            path, "non-scalar constants need an indexed constant projection"
        )
    kind = np.dtype(value.type.dtype).kind
    if kind in "iu":
        return literal_integer(value, path)
    if kind == "b":
        return bool(np.frombuffer(value.data, dtype=value.type.dtype)[0])
    return ValueExpr("literal", scalar_type(value.type), data=value.data)


def truth(value: Expression, ranges: dict[str, tuple[int, int]]) -> bool | None:
    if type(value) is bool:
        return value
    if not isinstance(value, Predicate):
        return None
    if value.operation in ("and", "or", "not"):
        parts = tuple(truth(cast(Expression, v), ranges) for v in value.operands)
        if value.operation == "not":
            return None if parts[0] is None else not parts[0]
        if value.operation == "and":
            return False if False in parts else True if all(p is True for p in parts) else None
        return True if True in parts else False if all(p is False for p in parts) else None
    a, b = value.operands
    if not isinstance(a, IndexExpr) or not isinstance(b, IndexExpr):
        return None
    lo, hi = bounds(add(a, scale(b, -1)), ranges)
    if value.operation == "lt":
        return True if hi < 0 else False if lo >= 0 else None
    if value.operation == "le":
        return True if hi <= 0 else False if lo > 0 else None
    if value.operation == "gt":
        return True if lo > 0 else False if hi <= 0 else None
    if value.operation == "ge":
        return True if lo >= 0 else False if hi < 0 else None
    if value.operation == "eq":
        return True if lo == hi == 0 else False if hi < 0 or lo > 0 else None
    if value.operation == "ne":
        eq = truth(Predicate("eq", (a, b)), ranges)
        return None if eq is None else not eq
    return None


def _substitute(value: Expression, bindings: dict[str, IndexExpr]) -> Expression:
    # Expressions are DAGs (e.g. repeated squaring), not trees. Preserve sharing
    # and avoid recursion/exponential copying when changing coordinate systems.
    done: dict[int, Expression] = {}
    pending = [(value, False)]
    while pending:
        item, ready = pending.pop()
        if id(item) in done:
            continue
        if isinstance(item, IndexExpr):
            projected = substitute(item, bindings)
            done[id(item)] = item if projected == item else projected
        elif isinstance(item, (Predicate, ValueExpr)):
            if not ready:
                pending.append((item, True))
                pending.extend((child, False) for child in item.operands)
                continue
            children = tuple(done[id(child)] for child in item.operands)
            if all(a is b for a, b in zip(children, item.operands, strict=True)):
                done[id(item)] = item
            elif isinstance(item, Predicate):
                done[id(item)] = Predicate(
                    item.operation, cast(tuple[IndexExpr | Predicate | bool, ...], children)
                )
            else:
                done[id(item)] = ValueExpr(
                    item.operation, item.type, children, item.parameters, item.data, item.input
                )
        else:
            done[id(item)] = item
    return done[id(value)]


def _share_expression(value: Expression) -> Expression:
    """Restore exact DAG sharing after coordinate substitution copies subtrees."""
    done: dict[int, Expression] = {}
    labels: dict[int, int] = {}
    interned: dict[tuple[object, ...], Expression] = {}
    pending = [(value, False)]
    while pending:
        item, ready = pending.pop()
        if id(item) in done:
            continue
        if type(item) not in (IndexExpr, Predicate, ValueExpr, bool):
            raise UnsupportedJaxprError("tensor.sharing", "unreviewed value expression")
        original: tuple[Expression, ...] = (
            item.args
            if isinstance(item, IndexExpr)
            else item.operands
            if isinstance(item, (ValueExpr, Predicate))
            else ()
        )
        if original and not ready:
            pending.append((item, True))
            pending.extend((child, False) for child in original)
            continue
        children = tuple(done[id(child)] for child in original)
        child_labels = tuple(labels[id(child)] for child in children)
        # Flat child labels avoid recursively hashing an expanded expression
        # tree. Dictionary lookup still compares the complete exact key;
        # neither digests nor algebraic identities establish equality here.
        if isinstance(item, ValueExpr):
            key = (
                ValueExpr,
                item.operation,
                item.type,
                child_labels,
                item.parameters,
                item.data,
                item.input,
            )
        elif isinstance(item, Predicate):
            key = (Predicate, item.operation, child_labels)
        elif isinstance(item, IndexExpr):
            key = (IndexExpr, item.kind, child_labels, item.value)
        else:
            key = (bool, item)
        shared = interned.get(key)
        if shared is None:
            if all(a is b for a, b in zip(children, original, strict=True)):
                shared = item
            elif isinstance(item, IndexExpr):
                shared = replace(item, args=cast(tuple[IndexExpr, ...], children))
            else:
                assert isinstance(item, (Predicate, ValueExpr))
                shared = replace(item, operands=children)
            interned[key] = shared
            labels[id(shared)] = len(labels)
        done[id(item)] = shared
    return done[id(value)]


def expressions_equal(left: tuple[Expression, ...], right: tuple[Expression, ...]) -> bool:
    """Exact DAG equality with sharing-aware traversal, without digest equality."""
    if len(left) != len(right):
        return False
    pending = list(zip(left, right, strict=True))
    seen = set()
    while pending:
        a, b = pending.pop()
        pair = (id(a), id(b))
        if pair in seen:
            continue
        seen.add(pair)
        if type(a) is not type(b):
            return False
        if isinstance(a, ValueExpr) and isinstance(b, ValueExpr):
            if (a.operation, a.type, a.parameters, a.data, a.input) != (
                b.operation,
                b.type,
                b.parameters,
                b.data,
                b.input,
            ):
                return False
            aa, bb = a.operands, b.operands
        elif isinstance(a, Predicate) and isinstance(b, Predicate):
            if a.operation != b.operation:
                return False
            aa, bb = a.operands, b.operands
        elif isinstance(a, IndexExpr) and isinstance(b, IndexExpr):
            if (a.kind, a.value) != (b.kind, b.value):
                return False
            aa, bb = a.args, b.args
        else:
            if a != b:
                return False
            continue
        if len(aa) != len(bb):
            return False
        pending.extend(zip(aa, bb, strict=True))
    return True


def _assemble(pieces: tuple[Piece[Expression], ...], typ: ArrayType) -> Expression | Uninitialized:
    # Coordinates are global already. Multiple pieces currently occur only in
    # the single-program interval rule, where local and global extents agree.
    merged: list[tuple[int, Expression]] = []
    for piece in pieces:
        if piece.value is None:
            return Uninitialized(piece.origin)
        if merged and expressions_equal((merged[-1][1],), (piece.value,)):
            merged[-1] = (piece.span.stop, merged[-1][1])
        else:
            merged.append((piece.span.stop, piece.value))
    result = merged[-1][1]
    for stop, value in reversed(merged[:-1]):
        condition = Predicate("lt", (axis("global"), integer(stop)))
        result = ValueExpr("select_n", scalar_type(typ), (condition, result, value))
    return result


def _dma_span(index: SliceIndex | None, size: int, path: str) -> Span:
    if index is None:
        return Span(0, size)
    if index.stride != 1:
        raise UnsupportedJaxprError(path, "DMA region requires unit stride")
    result = Span(index.start, index.start + index.size)
    if not Span(0, size).contains(result):
        raise UnsupportedJaxprError(path, "DMA region is empty or out of bounds")
    return result


def _check_access(address: IndexExpr, mask: Expression, size: int, padded: int, path: str) -> None:
    ranges = {"global": (0, padded - 1)}
    lo, hi = bounds(address, ranges)
    if lo >= 0 and hi < size:
        return
    # Restricted tail rule: after the proven inverse, this access is the
    # identity address and the mask excludes every padded lane. No sampling.
    if address == axis("global") and padded >= size:
        if padded == size or truth(mask, {"global": (size, padded - 1)}) is False:
            return
    raise UnsupportedJaxprError(
        path, "read/write address may be out of bounds; tail guard not proved"
    )


def functionalize_call(call: PallasCallView, *, budget: AnalysisBudget) -> MemoryProjection:
    """Prove independent blocked programs, then eliminate straight-line state.

    Each output has one full valid-domain writer. Reads of an output use its
    current version, and a swap's old value stays uninitialized until consumed.
    All operations, including dead operations, pass admission and access checks.
    """
    path = call.kernel.origin.path
    used = consumed_values(call.kernel)
    if call.aliases or call.interpret or len(call.grid) != 1 or call.num_outputs == 0:
        raise UnsupportedJaxprError(path, "unsupported kernel execution contract")
    if call.compiler is None or not call.compiler.default_execution_options:
        raise UnsupportedJaxprError(
            path, "non-default compiler execution options need a normalization proof"
        )
    if any(axis.semantics not in ("arbitrary", "parallel") for axis in call.grid):
        raise UnsupportedJaxprError(path, "core/subcore scheduling needs a hardware binding proof")
    for block in call.blocks:
        for typ in (block.array, block.ref.inner):
            if (
                not isinstance(typ, ArrayType)
                or typ.sharding is None
                or any(typ.sharding.partitions)
            ):
                raise UnsupportedJaxprError(
                    path, "kernel projection requires shard-local array values"
                )
    domains = tuple(prove_block_domain(b, call.grid[0].size, budget) for b in call.blocks)
    target = domains[call.num_inputs]
    if any((d.size, d.block) != (target.size, target.block) for d in domains):
        raise UnsupportedJaxprError(path, "memory projection requires aligned block extents")
    if any(d.mapping != target.mapping for d in domains[call.num_inputs :]):
        raise UnsupportedJaxprError(path, "multiple outputs need the same writer coordinates")
    program, lane = target.inverse
    bindings = {"program": program, "lane": lane}
    ranges = {"program": (0, target.programs - 1), "lane": (0, target.block - 1)}
    valid = {"global": (0, target.size - 1)}
    padded = target.programs * target.block
    ranges["global"] = (0, padded - 1)
    values: dict[int, Expression | Uninitialized] = {}
    memories: dict[int, Storage[Expression]] = {}
    data_count = call.num_inputs + call.num_outputs
    data_refs = call.kernel.inputs[:data_count]
    if len(call.kernel.inputs) != data_count + len(call.scratch):
        raise UnsupportedJaxprError(path, "invalid scratch bindings")
    if call.scratch and call.grid[0].size != 1:
        raise UnsupportedJaxprError(
            path, "scratch/DMA across multiple programs needs a lifetime proof"
        )
    semaphores = set()
    ref_slots = {v.id: i for i, v in enumerate(data_refs)}
    array_types = {v.id: cast(ArrayType, cast(RefType, v.type).inner) for v in data_refs}
    for i, (var, scratch) in enumerate(
        zip(call.kernel.inputs[data_count:], call.scratch, strict=True), start=data_count
    ):
        typ = scratch.type.inner
        if var.type != scratch.type or scratch.lifetime != "call":
            raise UnsupportedJaxprError(path, "unsupported scratch binding/lifetime")
        if isinstance(typ, SemaphoreType):
            if typ.kind != "dma" or scratch.initial_value != 0:
                raise UnsupportedJaxprError(path, "unsupported scratch semaphore initialization")
            semaphores.add(var.id)
        else:
            if (
                scratch.initial_value is not None
                or scratch.bound_array is None
                or typ.shape != (target.block,)
                or typ.sharding is None
                or any(typ.sharding.partitions)
                or padded != target.size
            ):
                raise UnsupportedJaxprError(
                    path, "scratch requires a full unpadded local array block"
                )
            ref_slots[var.id] = i
            array_types[var.id] = scratch.bound_array
            memories[var.id] = Storage(target.block, None, call.kernel.origin, budget)
    dma_state = LocalDmaState(semaphores, budget)
    payloads: dict[int, tuple[Piece[Expression], ...]] = {}
    for var, const in call.kernel.constants:
        values[var.id] = _literal(const, path)
    for var, domain, block in zip(data_refs, domains, call.blocks, strict=True):
        slot = ref_slots[var.id]
        value = (
            ValueExpr(
                "input",
                scalar_type(block.array),
                (substitute(domain.address, bindings),),
                input=slot,
            )
            if slot < call.num_inputs
            else None
        )
        memories[var.id] = Storage(target.block, value, call.kernel.origin, budget)
    versions: list[MemoryVersion] = []

    def read(operand: Operand) -> Expression:
        result = values[operand.id] if isinstance(operand, ValueRef) else _literal(operand, path)
        if isinstance(result, Uninitialized):
            raise UnsupportedJaxprError(
                result.origin.path, "uninitialized output/old swap value is consumed"
            )
        return result

    def global_value(value: Expression) -> Expression:
        return _substitute(value, bindings)

    def write_storage(storage: int, span: Span, value: Expression, origin: Origin) -> int:
        slot = ref_slots[storage]
        if slot < call.num_inputs:
            raise UnsupportedJaxprError(origin.path, "input storage must remain read-only")
        version = memories[storage].write(span, value, origin, overwrite=slot >= data_count)
        versions.append(MemoryVersion(slot, version, value, origin, span))
        return version

    for op in call.kernel.operations:
        loc = op.origin.path
        budget.spend(loc)
        if op.dma is not None:
            dma = op.dma
            if dma.source_semaphore is not None or dma.device_coordinates is not None:
                raise UnsupportedJaxprError(loc, "remote DMA needs global protocol/value analysis")
            if op.results or op.call is not None or op.regions:
                raise UnsupportedJaxprError(loc, "invalid DMA operation bindings")
            if call.grid[0].size != 1 or padded != target.size:
                raise UnsupportedJaxprError(loc, "DMA requires one complete, unpadded program")
            src, dst = ref_slots.get(dma.source.id), ref_slots.get(dma.destination.id)
            if src is None or dst is None:
                raise UnsupportedJaxprError(loc, "DMA source/destination must be array Refs")
            if dst < call.num_inputs:
                raise UnsupportedJaxprError(loc, "DMA destination must not alias read-only inputs")
            a, b = array_types[dma.source.id], array_types[dma.destination.id]
            if a != b:
                raise UnsupportedJaxprError(loc, "DMA source/destination abstract contracts differ")
            source_span = _dma_span(dma.source_slice, target.block, loc)
            destination_span = _dma_span(dma.destination_slice, target.block, loc)
            if source_span.size != destination_span.size:
                raise UnsupportedJaxprError(loc, "DMA source/destination region extents differ")
            if (
                dma.phase == "start"
                and dma.source == dma.destination
                and source_span.overlaps(destination_span)
            ):
                raise UnsupportedJaxprError(loc, "overlapping DMA source and destination")
            size = source_span.size * np.dtype(a.dtype).itemsize
            if dma.phase == "start":
                source = memories[dma.source.id]
                pieces = source.read(source_span, loc, initialized=False)
                if any(p.value is None for p in pieces):
                    raise UnsupportedJaxprError(loc, "DMA reads uninitialized source storage")
                if dst < data_count and any(
                    p.value is not None
                    for p in memories[dma.destination.id].read(
                        destination_span, loc, initialized=False
                    )
                ):
                    raise UnsupportedJaxprError(loc, "multiple writes overlap an output interval")
                transfer = dma_state.start(
                    dma,
                    op.origin,
                    source_version=source.version,
                    source_span=source_span,
                    destination_span=destination_span,
                    source_versions=source.versions(source_span, loc),
                    dtype=a.dtype,
                    logical_bytes=size,
                )
                payloads[transfer.id] = pieces
            else:
                released = dma_state.wait(
                    dma,
                    op.origin,
                    source_span=source_span,
                    destination_span=destination_span,
                    dtype=a.dtype,
                    logical_bytes=size,
                )
                for transfer in released:
                    if (
                        memories[transfer.source].versions(transfer.source_span, loc)
                        != transfer.source_versions
                    ):
                        raise UnsupportedJaxprError(
                            loc, "DMA source interval versions changed in flight"
                        )
                for transfer in released:
                    delta = transfer.destination_span.start - transfer.source_span.start
                    destination_version = 0
                    for piece in payloads.pop(transfer.id):
                        value = _substitute(
                            cast(Expression, piece.value),
                            {"global": add(axis("global"), integer(-delta))},
                        )
                        span = Span(piece.span.start + delta, piece.span.stop + delta)
                        destination_version = write_storage(
                            transfer.destination, span, value, op.origin
                        )
                    dma_state.completed.append(
                        TransferEvidence(transfer, op.origin, destination_version)
                    )
            continue
        if len(op.results) != 1 or op.call is not None or op.regions:
            raise UnsupportedJaxprError(loc, "unsupported kernel operation arity/region")
        result = op.results[0]
        if not isinstance(result.type, ArrayType) or result.type.shape not in ((), (target.block,)):
            raise UnsupportedJaxprError(loc, "projection requires scalar or full block values")
        if result.type.sharding is None or any(result.type.sharding.partitions):
            raise UnsupportedJaxprError(loc, "partitioned kernel operations are not yet projected")
        access = op.access
        if access is not None:
            if access.transforms:
                raise UnsupportedJaxprError(loc, "Ref coordinate transforms require tensor memory")
            slot = ref_slots.get(access.ref.id)
            if slot is None:
                raise UnsupportedJaxprError(loc, "unknown storage binding")
            dma_state.access(
                access.ref.id, Span(0, target.block), write=access.value is not None, path=loc
            )
            domain = domains[slot] if slot < data_count else target
            index = access.index
            if isinstance(index, SliceIndex):
                if index != SliceIndex(0, target.block, 1):
                    raise UnsupportedJaxprError(
                        loc, "partial Ref slices are not yet functionalized"
                    )
                local = axis("lane")
            else:
                if isinstance(index, TensorIndex):
                    raise UnsupportedJaxprError(loc, "tensor index requires the tensor memory rule")
                local = read(index)
            if local != axis("lane"):
                raise UnsupportedJaxprError(
                    loc, "non-identity lane access is not yet functionalized"
                )
            mask = global_value(read(access.mask)) if access.mask is not None else True
            address = substitute(domain.address, bindings)
            _check_access(address, mask, domain.size, padded, loc)
            if truth(mask, valid) is not True:
                raise UnsupportedJaxprError(
                    loc, "mask leaves a coverage gap in the valid output domain"
                )
            if access.other is not None:
                read(access.other)  # Preserve and validate the complete source, even if inactive.
            old = _assemble(
                memories[access.ref.id].read(Span(0, target.block), loc, initialized=False),
                array_types[access.ref.id],
            )
            if access.value is None:
                if result.id in used and isinstance(old, Uninitialized):
                    raise UnsupportedJaxprError(loc, "read of uninitialized output/scratch storage")
                value = old
            else:
                new = global_value(read(access.value))
                write_storage(access.ref.id, Span(0, target.block), new, op.origin)
                value = old
            values[result.id] = value
            continue
        args = tuple(read(v) for v in op.operands)
        params = {p.name: p.value for p in op.parameters}
        name = op.primitive
        if name in ("program_id", "num_programs"):
            if params["axis"] != 0:
                raise UnsupportedJaxprError(loc, "invalid program axis")
            value = axis("program") if name == "program_id" else integer(target.programs)
        elif name == "iota":
            if params["dimension"] != 0 or params["shape"] != (target.block,):
                raise UnsupportedJaxprError(loc, "iota must enumerate the block lanes")
            value = axis("lane")
        elif name == "broadcast_in_dim":
            if params["broadcast_dimensions"] != () or len(args) != 1:
                raise UnsupportedJaxprError(loc, "only scalar-to-block broadcast is projected")
            value = args[0]
        elif name not in POINTWISE_PRIMITIVES:
            raise UnsupportedJaxprError(loc, "value operation requires a tensor coordinate rule")
        elif name in ("lt", "le", "gt", "ge", "eq", "ne") and all(
            isinstance(v, IndexExpr) for v in args
        ):
            value = Predicate(name, cast(tuple[IndexExpr, ...], args))
        elif name in ("and", "or", "not") and all(
            isinstance(v, Predicate) or type(v) is bool for v in args
        ):
            value = Predicate(name, cast(tuple[Predicate | bool, ...], args))
        elif (
            all(isinstance(v, IndexExpr) for v in args) and np.dtype(result.type.dtype).kind in "iu"
        ):
            value = integer_operation(name, cast(tuple[IndexExpr, ...], args), ranges, loc)
        else:
            value = ValueExpr(name, scalar_type(result.type), args, op.parameters)
        if isinstance(value, IndexExpr):
            check_integer_range(value, result.type.dtype, ranges, loc)
        values[result.id] = value
    transfers = dma_state.finish(path)
    outputs = []
    for var in data_refs[call.num_inputs :]:
        value = _assemble(
            memories[var.id].read(Span(0, target.block), path, initialized=False),
            array_types[var.id],
        )
        if isinstance(value, Uninitialized):
            raise UnsupportedJaxprError(
                path, "output has no writer or contains an uninitialized interval"
            )
        outputs.append(value)
    return MemoryProjection(
        tuple(outputs),
        domains,
        tuple(versions),
        tuple(op.origin for op in call.kernel.operations),
        (
            "blocked forward/reverse maps have exact inverses",
            "every valid output coordinate has exactly one writer",
            "all active accesses are in bounds; every tail access is guarded",
            "inputs are read-only; no aliases or inter-program state",
            "every used output read observes an initialized current version",
            "all operations, including unused operations, were admitted",
            "local DMA sources stay stable; destinations are inaccessible until a matching wait",
            "each scratch semaphore starts at zero and has no outstanding transfer at exit",
        ),
        transfers,
        tuple(dma_state.waits),
    )
