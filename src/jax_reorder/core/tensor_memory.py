"""Multidimensional grid/address and rectangular state proofs, without a CAS."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from itertools import product
from math import prod
from typing import cast

import numpy as np

from ..errors import UnsupportedJaxprError
from .communication import MAX_DMA_BYTES, LocalDmaState, TransferEvidence
from .grid_execution import functionalize_banked_grid
from .grid_state import GridPlan, independent_domains, state_domains
from .index_domain import AnalysisBudget, IndexExpr, add, bounds, divide, integer, scale, substitute
from .masked_access import mask_regions, range_guard, require_defined, require_true, resolve_defined
from .memory import (
    Expression,
    MemoryProjection,
    MemoryVersion,
    Predicate,
    ScalarType,
    Uninitialized,
    ValueExpr,
    _literal,
    _substitute,
    expressions_equal,
    scalar_type,
    truth,
)
from .model import (
    ArrayType,
    DynamicSliceIndex,
    Operand,
    OperationView,
    Origin,
    PallasCallView,
    ParameterView,
    RefTransform,
    RefType,
    RegionView,
    SemaphoreType,
    SliceIndex,
    TensorIndex,
    ValueDef,
    ValueRef,
)
from .ref_views import view_coordinates, view_footprint
from .storage import Span, VersionSlice
from .tensor_dma import DmaRectangle
from .tensor_domain import coordinate, coordinate_ranges
from .tensor_owner_partition import functionalize_partitioned_owners
from .tensor_storage import Box, BoxPiece, TensorStorage, full_box
from .tensor_values import project_coordinates, tensor_value
from .value_operations import integer_value_bounds, static_integer


@dataclass(frozen=True)
class TensorRef:
    slot: int


type Binding = Expression | TensorRef | Uninitialized


def box_predicate(box: Box) -> Predicate | bool:
    parts = tuple(
        Predicate(
            "and",
            (
                Predicate("ge", (coordinate("local", i), integer(a.start))),
                Predicate("lt", (coordinate("local", i), integer(a.stop))),
            ),
        )
        for i, a in enumerate(box.axes)
    )
    return Predicate("and", parts) if parts else True


def assemble(
    pieces: tuple[BoxPiece[Expression], ...], typ: ArrayType
) -> Expression | Uninitialized:
    if not pieces or any(p.value is None for p in pieces):
        return Uninitialized(pieces[0].origin if pieces else Origin("tensor.memory"))
    result = cast(Expression, pieces[-1].value)
    for piece in reversed(pieces[:-1]):
        value = cast(Expression, piece.value)
        if not expressions_equal((value,), (result,)):
            result = ValueExpr(
                "select_n", scalar_type(typ), (box_predicate(piece.box), result, value)
            )
    return result


def partial_value(pieces: tuple[BoxPiece[Expression], ...], typ: ArrayType) -> Expression:
    """Assemble storage without making unwritten lanes appear initialized."""
    if pieces and all(piece.value is not None for piece in pieces):
        return cast(Expression, assemble(pieces, typ))
    result: Expression = ValueExpr("undefined", scalar_type(typ))
    for piece in reversed(pieces):
        if piece.value is not None:
            result = ValueExpr(
                "select_n", scalar_type(typ), (box_predicate(piece.box), result, piece.value)
            )
    return result


def functionalize_tensor_call(
    call: PallasCallView,
    *,
    budget: AnalysisBudget,
    inputs: tuple[Expression, ...] | None = None,
    coordinates: dict[str, IndexExpr] | None = None,
    axis_ranges: dict[str, tuple[int, int]] | None = None,
    execution_cores: int | None = None,
    program_point: tuple[int, ...] | None = None,
    semaphore_observer: Callable[[str, int, int, int, Origin], None] | None = None,
    defer_padding: bool = False,
) -> MemoryProjection:
    try:
        return _functionalize_tensor_call(
            call,
            budget=budget,
            inputs=inputs,
            coordinates=coordinates,
            axis_ranges=axis_ranges,
            execution_cores=execution_cores,
            program_point=program_point,
            semaphore_observer=semaphore_observer,
            defer_padding=defer_padding,
        )
    except UnsupportedJaxprError:
        serial = [i for i, g in enumerate(call.grid) if g.size > 1 and g.semantics == "arbitrary"]
        if (
            program_point is not None
            or semaphore_observer is not None
            or not call.scratch
            or not serial
            or not any(g.size > 1 and g.semantics == "parallel" for g in call.grid[serial[0] :])
        ):
            raise

        def project_owner(
            owner_coordinates: dict[str, IndexExpr],
            owner_ranges: dict[str, tuple[int, int]],
        ) -> MemoryProjection:
            return _functionalize_tensor_call(
                call,
                budget=budget,
                inputs=inputs,
                coordinates=owner_coordinates,
                axis_ranges=owner_ranges,
                execution_cores=execution_cores,
                banked_interleaving=True,
                defer_padding=defer_padding,
            )

        return functionalize_banked_grid(
            call,
            budget=budget,
            coordinates=coordinates,
            axis_ranges=axis_ranges,
            execution_cores=execution_cores,
            project=project_owner,
        )


def _functionalize_tensor_call(
    call: PallasCallView,
    *,
    budget: AnalysisBudget,
    inputs: tuple[Expression, ...] | None = None,
    coordinates: dict[str, IndexExpr] | None = None,
    axis_ranges: dict[str, tuple[int, int]] | None = None,
    execution_cores: int | None = None,
    program_point: tuple[int, ...] | None = None,
    semaphore_observer: Callable[[str, int, int, int, Origin], None] | None = None,
    defer_padding: bool = False,
    banked_interleaving: bool = False,
    owner_plan: GridPlan | None = None,
    owner_suffix: str = "",
) -> MemoryProjection:
    if execution_cores is not None and (type(execution_cores) is not int or execution_cores != 1):
        raise ValueError("execution_cores currently requires an explicit single-core binding")
    path = call.kernel.origin.path + owner_suffix
    if (
        call.interpret
        or not call.num_outputs
        or call.compiler is None
        or not (call.compiler.default_execution_options or call.compiler.reviewed_layout_options)
    ):
        raise UnsupportedJaxprError(
            path, "tensor projection needs a reviewed native execution contract"
        )
    if any(g.semantics not in ("arbitrary", "parallel") for g in call.grid):
        raise UnsupportedJaxprError(path, "tensor grid requires reviewed program scheduling")
    grid = tuple(g.size for g in call.grid)
    plan = owner_plan or (
        state_domains(call, budget, banked_interleaving=banked_interleaving)
        if program_point is None
        else independent_domains(call, budget)
    )
    domains, serial_axes = plan.domains, plan.serial_axes
    if owner_plan is None and program_point is None and semaphore_observer is None:

        def project_owner(
            owner_ranges: dict[str, tuple[int, int]], suffix: str
        ) -> MemoryProjection:
            return _functionalize_tensor_call(
                call,
                budget=budget,
                inputs=inputs,
                coordinates=coordinates,
                axis_ranges=owner_ranges,
                execution_cores=execution_cores,
                defer_padding=defer_padding,
                banked_interleaving=banked_interleaving,
                owner_plan=plan,
                owner_suffix=suffix,
            )

        partitioned = functionalize_partitioned_owners(
            call,
            plan,
            budget=budget,
            coordinates=coordinates,
            axis_ranges=axis_ranges,
            project=project_owner,
        )
        if partitioned is not None:
            return partitioned
    if program_point is not None:
        if len(program_point) != len(grid) or any(
            type(p) is not int or not 0 <= p < n for p, n in zip(program_point, grid, strict=True)
        ):
            raise ValueError("program point must be in the complete grid")
        serial_axes = tuple(range(len(grid)))
    if serial_axes:
        if call.aliases:
            raise UnsupportedJaxprError(path, "serial aliases need an input staging lifetime proof")
        if program_point is None:
            budget.spend(path + ".serial_grid", prod(grid[i] for i in serial_axes))
    npre = len(call.prefetch)
    count = npre + call.num_inputs + call.num_outputs
    if len(call.kernel.inputs) != count + len(call.scratch) or (
        inputs is not None and len(inputs) != npre + call.num_inputs
    ):
        raise UnsupportedJaxprError(path, "invalid tensor kernel port bindings")
    for i, o in call.aliases:
        if call.blocks[i - npre].unbuffered or call.blocks[call.num_inputs + o].unbuffered:
            raise UnsupportedJaxprError(path, "unbuffered aliases need a shared HBM version proof")
        a, b = domains[i - npre], domains[call.num_inputs + o]
        if (a.shape, a.block_shape, a.mapping) != (b.shape, b.block_shape, b.mapping):
            raise UnsupportedJaxprError(path, "tensor alias requires identical block mappings")
    padded = any(d.padded for d in domains)
    if padded and call.aliases:
        raise UnsupportedJaxprError(path, "padded tensor aliases require a separate lifetime proof")
    ranges = {**coordinate_ranges("program", grid), **(axis_ranges or {})}
    coordinates = dict(coordinates or {})
    iteration_coordinates: dict[str, IndexExpr] = {}
    types: dict[int, ArrayType] = {}
    memories: dict[int, TensorStorage[Expression]] = {}
    regular: dict[int, int] = {}
    semaphores: set[int] = set()
    semaphore_shapes: dict[int, tuple[int, ...]] = {}
    semaphore_elements: dict[int, tuple[int, ...]] = {}
    next_resource = len(call.kernel.inputs)

    def allocate_semaphore(resource: int, typ: SemaphoreType) -> None:
        nonlocal next_resource
        size = prod(typ.shape)
        budget.spend(path + ".semaphore_slots", size)
        semaphore_shapes[resource] = typ.shape
        elements = tuple(range(next_resource, next_resource + size)) if typ.shape else (resource,)
        if typ.shape:
            next_resource += size
        semaphore_elements[resource] = elements
        if typ.kind == "regular":
            if (
                semaphore_observer is None
                and execution_cores != 1
                and any(g.size > 1 and g.semantics == "parallel" for g in call.grid)
            ):
                raise UnsupportedJaxprError(
                    path, "parallel regular semaphores require an explicit core target binding"
                )
            regular.update(dict.fromkeys(elements, 0))
        elif typ.kind == "dma":
            semaphores.update(elements)
        else:
            raise UnsupportedJaxprError(path, "unknown tensor semaphore allocation")

    readonly = set(range(npre + call.num_inputs))
    restoring = {
        npre + i
        for i, block in enumerate(call.blocks)
        if i >= call.num_inputs
        and block.pipeline_mode is not None
        and block.pipeline_mode.revisit == "any"
    }
    unbuffered = {npre + i for i, block in enumerate(call.blocks) if block.unbuffered}
    for slot, var in enumerate(call.kernel.inputs):
        if not isinstance(var.type, RefType):
            raise UnsupportedJaxprError(path, "tensor kernel argument must be a Ref")
        typ = var.type.inner
        if slot >= count:
            scratch = call.scratch[slot - count]
            if scratch.lifetime != "call":
                raise UnsupportedJaxprError(path, "tensor scratch needs call lifetime")
            if isinstance(typ, SemaphoreType):
                if scratch.initial_value != 0:
                    raise UnsupportedJaxprError(path, "tensor semaphore must start at zero")
                allocate_semaphore(slot, typ)
                continue
            if scratch.bound_array is None or scratch.initial_value is not None:
                raise UnsupportedJaxprError(path, "tensor scratch must be initially uninitialized")
            typ = scratch.bound_array
        if not isinstance(typ, ArrayType) or typ.sharding is None or any(typ.sharding.partitions):
            raise UnsupportedJaxprError(path, "tensor memory must be shard-local")
        types[slot] = typ
        value = None
        if slot in readonly:
            address = (
                tuple(coordinate("local", i) for i in range(len(typ.shape)))
                if slot < npre
                else domains[slot - npre].address
            )
            if inputs is None:
                value = ValueExpr("input", scalar_type(typ), address, input=slot)
            else:
                bindings = {f"local:{i}": a for i, a in enumerate(address)}
                if len(address) == 1:
                    bindings["global"] = address[0]
                value = _substitute(inputs[slot], bindings)
        memories[slot] = TensorStorage(typ.shape, value, call.kernel.origin, budget)
    dma_state = LocalDmaState(semaphores, budget)
    payloads: dict[int, tuple[DmaRectangle, DmaRectangle, tuple[BoxPiece[Expression], ...]]] = {}
    versions: list[MemoryVersion] = []
    origins: list[Origin] = []
    # Final output values retain executed control even after writes are replaced.
    control_guard: Expression = True

    def expression(value: Binding, loc: str) -> Expression:
        if isinstance(value, Uninitialized):
            if (plan.independent or banked_interleaving) and program_point is None:
                raise UnsupportedJaxprError(
                    loc,
                    "stateful grid requires a parallel prefix and serial suffix; "
                    "interleaved owner groups must initialize scratch independently",
                )
            raise UnsupportedJaxprError(loc, "read of uninitialized tensor output/scratch")
        if isinstance(value, TensorRef):
            raise UnsupportedJaxprError(loc, "expected tensor value instead of Ref")
        return value

    def write(slot: int, box: Box, value: Expression, origin: Origin) -> int:
        if slot in readonly:
            raise UnsupportedJaxprError(origin.path, "input storage must remain read-only")
        version = memories[slot].write(box, value, origin)
        versions.append(
            MemoryVersion(
                slot, version, value, origin, box.linear_bounds(types[slot].shape), box.axes
            )
        )
        return version

    def valid_output_box(resource: int, mapping: tuple[IndexExpr, ...]) -> Box:
        domain = domains[resource - npre]
        axes = []
        for size, width, tile in zip(domain.shape, domain.block_shape, mapping, strict=True):
            if size % width == 0:
                stop = width
            else:
                low, high = bounds(
                    tile, {**coordinate_ranges("program", grid), **(axis_ranges or {})}
                )
                if low != high and high == size // width:
                    raise UnsupportedJaxprError(
                        path, "padded output restoration needs a fixed tail owner"
                    )
                stop = min(width, size - high * width)
            axes.append(Span(0, stop))
        return Box(tuple(axes))

    def evaluate(
        source: RegionView, arguments: tuple[Binding, ...], suffix: str = ""
    ) -> tuple[Binding, ...]:
        nonlocal memories, regular, dma_state, next_resource
        env = dict(zip((v.id for v in source.inputs), arguments, strict=True))
        env.update((v.id, _literal(c, source.origin.path)) for v, c in source.constants)
        value_types = {v.id: v.type for v in source.inputs}
        value_types.update((v.id, v.type) for v, _ in source.constants)
        value_types.update((v.id, v.type) for op in source.operations for v in op.results)

        def read(v: Operand) -> Binding:
            return env[v.id] if isinstance(v, ValueRef) else _literal(v, source.origin.path)

        def value(v: Operand) -> Expression:
            return expression(read(v), source.origin.path)

        def array_type(v: Operand) -> ArrayType:
            typ = value_types[v.id] if isinstance(v, ValueRef) else v.type
            if not isinstance(typ, ArrayType):
                raise UnsupportedJaxprError(source.origin.path, "non-array tensor value")
            return typ

        def shape(v: Operand) -> tuple[int, ...]:
            return array_type(v).shape

        def slot(v: Operand) -> int:
            ref = read(v)
            if not isinstance(ref, TensorRef):
                raise UnsupportedJaxprError(source.origin.path, "invalid tensor Ref binding")
            return ref.slot

        def access_region(
            index: SliceIndex | TensorIndex | Operand,
            resource: int,
            loc: str,
            *,
            readonly_access: bool = False,
        ) -> tuple[tuple[Box, ...], dict[str, IndexExpr], dict[str, IndexExpr], tuple[int, ...]]:
            storage_shape = types[resource].shape
            if not storage_shape:
                return (full_box(()),), {}, {}, ()
            indices = index.indices if isinstance(index, TensorIndex) else (index,)
            if len(indices) != len(storage_shape):
                raise UnsupportedJaxprError(loc, "tensor access rank differs from storage")
            vector = (isinstance(index, TensorIndex) and bool(index.integer_shape)) or (
                not isinstance(index, TensorIndex | SliceIndex) and bool(shape(index))
            )
            if (
                vector
                and not isinstance(index, TensorIndex)
                and shape(cast(Operand, index)) != storage_shape
            ):
                raise UnsupportedJaxprError(
                    loc, "tensor vector index shape differs from its storage"
                )
            axes, to_ref, to_value, output_shape = [], {}, {}, []
            if (
                isinstance(index, TensorIndex)
                and vector
                and (
                    index.integer_shape != storage_shape
                    or any(isinstance(i, SliceIndex | DynamicSliceIndex) for i in indices)
                )
            ):
                raise UnsupportedJaxprError(
                    loc, "tensor array indexing requires a full identity coordinate grid"
                )
            for i, (ix, extent) in enumerate(zip(indices, storage_shape, strict=True)):
                if isinstance(ix, DynamicSliceIndex):
                    start = value(ix.start)
                    if readonly_access and isinstance(start, IndexExpr):
                        low, high = bounds(start, ranges)
                        if low != high:
                            if ix.stride < 1 or ix.size < 1:
                                raise UnsupportedJaxprError(
                                    loc, "tensor slice needs positive size/stride"
                                )
                            axes.append((Span(low, high + (ix.size - 1) * ix.stride + 1),))
                            d = len(output_shape)
                            output_shape.append(ix.size)
                            to_ref[f"local:{i}"] = add(
                                scale(coordinate("local", d), ix.stride), start
                            )
                            continue
                        start = integer(low)
                    ix = SliceIndex(static_integer(start, loc), ix.size, ix.stride)
                if isinstance(ix, SliceIndex):
                    if ix.stride < 1 or ix.size < 1:
                        raise UnsupportedJaxprError(loc, "tensor slice needs positive size/stride")
                    if ix.stride == 1:
                        parts = (Span(ix.start, ix.start + ix.size),)
                    else:
                        budget.spend(loc + ".strided_regions", ix.size)
                        parts = tuple(
                            Span(k, k + 1)
                            for k in range(ix.start, ix.start + ix.size * ix.stride, ix.stride)
                        )
                    d = len(output_shape)
                    output_shape.append(ix.size)
                    to_ref[f"local:{i}"] = add(
                        scale(coordinate("local", d), ix.stride), integer(ix.start)
                    )
                    to_value[f"local:{d}"] = divide(
                        add(coordinate("local", i), integer(-ix.start)), ix.stride
                    )
                elif vector:
                    if value(ix) != coordinate("local", i):
                        raise UnsupportedJaxprError(
                            loc, "tensor vector index is not the identity lane coordinate"
                        )
                    parts = (Span(0, extent),)
                    output_shape.append(extent)
                    to_ref[f"local:{i}"] = to_value[f"local:{i}"] = coordinate("local", i)
                else:
                    position = value(ix)
                    if readonly_access and isinstance(position, IndexExpr):
                        low, high = bounds(position, ranges)
                        parts = (Span(low, high + 1),)
                        to_ref[f"local:{i}"] = position
                    else:
                        fixed = static_integer(position, loc)
                        parts = (Span(fixed, fixed + 1),)
                        to_ref[f"local:{i}"] = integer(fixed)
                axes.append(parts)
            region_count = prod(len(parts) for parts in axes)
            # The operation and storage accesses already charge the original
            # rectangle. Charge only additional regions before expanding a
            # strided Cartesian product, preserving the unit-stride budget.
            if region_count > 1:
                budget.spend(loc + ".access_regions", region_count - 1)
            boxes = tuple(Box(tuple(parts)) for parts in product(*axes))
            for box in boxes:
                memories[resource].read(box, loc, initialized=False)
            return boxes, to_ref, to_value, tuple(output_shape)

        def fork_address(
            operand: ValueRef,
            candidate: Expression,
            low: int,
            high: int,
            loc: str,
            origin: Origin,
            position: int,
        ) -> tuple[Binding, ...]:
            budget.spend(loc + ".address_cases", high - low + 1)
            variables = tuple(ValueDef(i, value_types[i]) for i in env)
            region = RegionView(
                variables,
                (),
                source.operations[position:],
                source.outputs,
                source.effects,
                source.is_high,
                source.origin,
            )

            def at_address(
                position: int,
                case: str,
                region: RegionView = region,
                operand: ValueRef = operand,
                variables: tuple[ValueDef, ...] = variables,
            ) -> tuple[Binding, ...]:
                return evaluate(
                    region,
                    tuple(
                        integer(position) if v.id == operand.id else env[v.id] for v in variables
                    ),
                    suffix + case,
                )

            selector = (
                add(candidate, integer(-low))
                if isinstance(candidate, IndexExpr)
                else candidate
                if low == 0
                else ValueExpr("sub", cast(ValueExpr, candidate).type, (candidate, integer(low)))
            )
            # Execute the remainder under each choice. This preserves
            # write-then-read initialization when the same scalar
            # address is reused later in the region.
            return fork(
                selector,
                tuple(partial(at_address, i) for i in range(low, high + 1)),
                tuple(
                    ValueDef(i, value_types[v.id] if isinstance(v, ValueRef) else v.type)
                    for i, v in enumerate(source.outputs)
                ),
                loc,
                origin,
            )

        def view_indices(
            storage_shape: tuple[int, ...], transforms: tuple[RefTransform, ...]
        ) -> tuple[tuple[Operand, int, int], ...]:
            current = storage_shape
            result = []
            for transform in transforms:
                if transform.kind == "reshape":
                    current = transform.dimensions
                elif transform.kind == "transpose":
                    current = tuple(current[i] for i in transform.dimensions)
                else:
                    for index, extent in zip(transform.indices, current, strict=True):
                        if isinstance(index, DynamicSliceIndex):
                            result.append((index.start, extent, (index.size - 1) * index.stride))
                        elif isinstance(index, ValueRef):
                            result.append((index, extent, 0))
                    current = tuple(
                        i.size
                        for i in transform.indices
                        if isinstance(i, SliceIndex | DynamicSliceIndex)
                    )
            return tuple(result)

        def fixed_views(transforms: tuple[RefTransform, ...], loc: str) -> tuple[RefTransform, ...]:
            return tuple(
                replace(
                    t,
                    indices=tuple(
                        SliceIndex(static_integer(value(i.start), loc), i.size, i.stride)
                        if isinstance(i, DynamicSliceIndex)
                        else static_integer(value(i), loc)
                        if isinstance(i, ValueRef)
                        else i
                        for i in t.indices
                    ),
                )
                for t in transforms
            )

        def dynamic_view_case(
            candidates: tuple[tuple[Operand, int, int], ...],
            loc: str,
        ) -> tuple[ValueRef, Expression, tuple[int, int]] | None:
            for operand, extent, tail in candidates:
                candidate = value(operand)
                if isinstance(candidate, IndexExpr) and candidate.kind == "constant":
                    continue
                interval = integer_value_bounds(candidate, ranges)
                if interval is None:
                    raise UnsupportedJaxprError(
                        loc, "dynamic Ref address needs finite integer bounds"
                    )
                if interval[0] < 0 or interval[1] + tail >= extent:
                    raise UnsupportedJaxprError(
                        loc, "dynamic Ref address bounds leave its intermediate view"
                    )
                if not isinstance(operand, ValueRef) or shape(operand):
                    raise UnsupportedJaxprError(loc, "dynamic Ref address must be a scalar value")
                return operand, candidate, interval
            return None

        def dma_rectangle(
            index: SliceIndex | TensorIndex | None,
            resource: int,
            loc: str,
            transforms: tuple[RefTransform, ...] = (),
        ) -> DmaRectangle:
            shape = types[resource].shape
            if index is None:
                indices = tuple(SliceIndex(0, n, 1) for n in shape)
                index = TensorIndex(indices, shape)
            indices = index.indices if isinstance(index, TensorIndex) else (index,)
            if any(
                isinstance(ix, SliceIndex | DynamicSliceIndex) and ix.stride != 1 for ix in indices
            ):
                raise UnsupportedJaxprError(loc, "tensor DMA slice requires unit stride")
            (box,), to_ref, to_value, output_shape = access_region(index, resource, loc)
            footprint = (box,)
            if transforms:
                if box != full_box(shape):
                    raise UnsupportedJaxprError(
                        loc, "DMA transforms require a full original-storage base"
                    )
                transforms = fixed_views(transforms, loc)
                output_shape, to_ref, to_value = view_coordinates(shape, transforms, loc, budget)
                footprint = view_footprint(shape, output_shape, to_ref, transforms, loc, budget)
                box = Box(
                    tuple(
                        Span(
                            min(b.axes[i].start for b in footprint),
                            max(b.axes[i].stop for b in footprint),
                        )
                        for i in range(len(shape))
                    )
                )

            if any(
                part.linear_bounds(shape).size != part.size
                and not (
                    len(shape) >= 2
                    and np.dtype(types[resource].dtype).itemsize == 4
                    and shape[-1] % 128 == 0
                    and part.axes[-1].start % 128 == part.axes[-1].size % 128 == 0
                    and part.axes[-2].start % 8 == part.axes[-2].size % 8 == 0
                )
                for part in footprint
            ):
                raise UnsupportedJaxprError(
                    loc, "noncontiguous DMA requires tile-aligned 32-bit rectangles"
                )
            kept = tuple(
                i for i, ix in enumerate(indices) if isinstance(ix, SliceIndex | DynamicSliceIndex)
            )
            return DmaRectangle(
                box, kept, to_ref, to_value, output_shape, bool(transforms), footprint
            )

        def semaphore_slot(ref: ValueRef, index: TensorIndex | None, loc: str) -> int:
            resource = slot(ref)
            dims = semaphore_shapes.get(resource)
            if (
                dims is None
                or (index is None and dims)
                or (index is not None and index.shape != dims)
            ):
                raise UnsupportedJaxprError(loc, "semaphore must select a bound scalar slot")
            positions = (
                ()
                if index is None
                else tuple(static_integer(value(cast(Operand, ix)), loc) for ix in index.indices)
            )
            if len(positions) != len(dims) or any(
                not 0 <= i < n for i, n in zip(positions, dims, strict=True)
            ):
                raise UnsupportedJaxprError(loc, "semaphore slot is out of bounds")
            offset = 0
            for i, n in zip(positions, dims, strict=True):
                offset = offset * n + i
            return semaphore_elements[resource][offset]

        def fork(
            selector: Expression,
            actions: tuple[Callable[[str], tuple[Binding, ...]], ...],
            result_defs: tuple[ValueDef, ...],
            loc: str,
            origin: Origin,
        ) -> tuple[Binding, ...]:
            nonlocal memories, regular, dma_state, control_guard
            if semaphore_observer is not None:
                raise UnsupportedJaxprError(
                    loc, "explicit core protocol requires statically resolved control"
                )
            if dma_state.pending:
                raise UnsupportedJaxprError(loc, "tensor branch requires quiescent DMA at entry")
            before_mem, before_counts, before_dma = memories, regular, dma_state
            before_guard = control_guard
            baseline = len(dma_state.completed), len(dma_state.waits)
            states, branch_results, branch_guards = [], [], []
            next_transfer = before_dma.next_id
            for case, action in enumerate(actions):
                control_guard = before_guard
                memories = {i: copy(m) for i, m in before_mem.items()}
                for memory in memories.values():
                    memory.pieces = list(memory.pieces)
                regular = before_counts.copy()
                dma_state = LocalDmaState(semaphores, budget)
                dma_state.generations = before_dma.generations.copy()
                dma_state.next_id = next_transfer
                dma_state.completed = before_dma.completed.copy()
                dma_state.waits = before_dma.waits.copy()
                branch_results.append(action(f".case[{case}]"))
                if dma_state.pending:
                    raise UnsupportedJaxprError(loc, "tensor branch exits with outstanding DMA")
                states.append((memories, regular, dma_state))
                branch_guards.append(control_guard)
                next_transfer = dma_state.next_id
            # Equal branch values may disappear, but their evaluated selector
            # must keep its domain and initialization requirements.
            control_guard = ValueExpr(
                "select_n",
                ScalarType("bool", False, ()),
                (selector, *branch_guards),
                parameters=(ParameterView("__control_guard", True),),
            )
            if any(c != states[0][1] for _, c, _ in states):
                raise UnsupportedJaxprError(loc, "tensor branches leave different counter states")
            memories, regular, dma_state = before_mem, states[0][1], before_dma

            def select(
                items: tuple[Expression, ...],
                typ: ArrayType,
                selector: Expression = selector,
            ) -> Expression:
                return (
                    items[0]
                    if all(expressions_equal((items[0],), (v,)) for v in items[1:])
                    else ValueExpr("select_n", scalar_type(typ), (selector, *items))
                )

            for resource, memory in memories.items():
                pieces = [(p.box, (p.value,)) for p in states[0][0][resource].pieces]
                for state, _, _ in states[1:]:
                    combined = []
                    for box, values in pieces:
                        for piece in state[resource].pieces:
                            budget.spend(loc)
                            overlap = box.intersection(piece.box)
                            if overlap is not None:
                                combined.append((overlap, (*values, piece.value)))
                    pieces = combined
                memory.version += 1
                memory.pieces = [
                    BoxPiece(
                        box,
                        None
                        if any(v is None for v in values)
                        else select(cast(tuple[Expression, ...], values), types[resource]),
                        memory.version,
                        origin,
                    )
                    for box, values in pieces
                ]
            for _, _, state in states:
                dma_state.completed.extend(state.completed[baseline[0] :])
                dma_state.waits.extend(state.waits[baseline[1] :])
                dma_state.next_id = max(dma_state.next_id, state.next_id)
                for resource, generation in state.generations.items():
                    dma_state.generations[resource] = max(
                        dma_state.generations[resource], generation
                    )
            if any(not isinstance(v.type, ArrayType) for v in result_defs):
                raise UnsupportedJaxprError(loc, "tensor branch Ref results require an alias proof")
            return tuple(
                select(
                    tuple(expression(r[i], loc) for r in branch_results),
                    cast(ArrayType, v.type),
                )
                for i, v in enumerate(result_defs)
            )

        def run_while(
            op: OperationView,
            consts: tuple[Binding, ...],
            carry: tuple[Binding, ...],
            nc: int,
            loc: str,
            origin: Origin,
        ) -> tuple[Binding, ...]:
            def run_loop(
                consts: tuple[Binding, ...],
                carry: tuple[Binding, ...],
                iteration: int,
                context: str,
            ) -> tuple[Binding, ...]:
                def at_value(
                    position: int,
                    selected: ValueExpr,
                    consts: tuple[Binding, ...],
                    carry: tuple[Binding, ...],
                    iteration: int,
                    context: str,
                    case: str,
                ) -> tuple[Binding, ...]:
                    cache: dict[int, Binding] = {}

                    def bind(item: Binding) -> Binding:
                        if id(item) in cache:
                            return cache[id(item)]
                        budget.spend(loc + ".while_values")
                        result: Binding = item
                        if item is selected:
                            result = integer(position)
                        elif isinstance(item, ValueExpr):
                            result = replace(
                                item,
                                operands=tuple(expression(bind(v), loc) for v in item.operands),
                            )
                        cache[id(item)] = result
                        return result

                    return run_loop(
                        tuple(bind(c) for c in consts),
                        tuple(bind(c) for c in carry),
                        iteration,
                        context + case,
                    )

                def at_coordinate(
                    position: int,
                    name: str,
                    low: int,
                    high: int,
                    consts: tuple[Binding, ...],
                    carry: tuple[Binding, ...],
                    iteration: int,
                    context: str,
                    case: str,
                ) -> tuple[Binding, ...]:
                    old_coordinate = coordinates.get(name)
                    ranges[name] = (position, position)
                    coordinates[name] = integer(position)

                    def bind(item: Binding) -> Binding:
                        return (
                            item
                            if isinstance(item, TensorRef | Uninitialized)
                            else _substitute(item, {name: integer(position)})
                        )

                    try:
                        return run_loop(
                            tuple(bind(c) for c in consts),
                            tuple(bind(c) for c in carry),
                            iteration,
                            context + case,
                        )
                    finally:
                        ranges[name] = (low, high)
                        if old_coordinate is None:
                            coordinates.pop(name, None)
                        else:
                            coordinates[name] = old_coordinate

                while True:
                    budget.spend(loc)
                    tag = suffix + context + f".iteration[{iteration}]"
                    (condition,) = evaluate(op.regions[0].region, consts[:nc] + carry, tag)
                    require_defined(
                        expression(condition, loc), ranges, budget, loc + ".while_condition"
                    )
                    decision = truth(expression(condition, loc), ranges)
                    if decision is None:
                        pending = [expression(condition, loc)]
                        axes = set()
                        seen = set()
                        while pending:
                            item = pending.pop()
                            if id(item) in seen:
                                continue
                            seen.add(id(item))
                            budget.spend(loc + ".while_domain")
                            if isinstance(item, IndexExpr):
                                if item.kind == "axis":
                                    axes.add(str(item.value))
                                pending.extend(item.args)
                            elif isinstance(item, Predicate | ValueExpr):
                                pending.extend(item.operands)
                        candidates = sorted(
                            a
                            for a in axes
                            if a.startswith("program:") and ranges[a][0] < ranges[a][1]
                        )
                        guarded = tuple(
                            (item, interval)
                            for item in (*consts, *carry)
                            if isinstance(item, ValueExpr)
                            and (interval := integer_value_bounds(item, ranges)) is not None
                            and interval[0] < interval[1]
                            and interval[1] - interval[0] < budget.limit
                        )
                        if (not candidates and not guarded) or op.regions[0].region.effects:
                            raise UnsupportedJaxprError(
                                loc, "tensor while needs a static finite carry proof"
                            )
                        if not candidates:
                            selected, (low, high) = guarded[0]
                            budget.spend(loc + ".while_values", high - low + 1)

                            selector = (
                                selected
                                if low == 0
                                else ValueExpr("sub", selected.type, (selected, integer(low)))
                            )
                            return fork(
                                selector,
                                tuple(
                                    partial(
                                        at_value,
                                        position,
                                        selected,
                                        consts,
                                        carry,
                                        iteration,
                                        context,
                                    )
                                    for position in range(low, high + 1)
                                ),
                                op.results,
                                loc,
                                origin,
                            )
                        name = candidates[0]
                        low, high = ranges[name]
                        budget.spend(loc + ".while_domain", high - low + 1)

                        return fork(
                            add(IndexExpr("axis", value=name), integer(-low)),
                            tuple(
                                partial(
                                    at_coordinate,
                                    position,
                                    name,
                                    low,
                                    high,
                                    consts,
                                    carry,
                                    iteration,
                                    context,
                                )
                                for position in range(low, high + 1)
                            ),
                            op.results,
                            loc,
                            origin,
                        )
                    if not decision:
                        return carry
                    carry = evaluate(op.regions[1].region, consts[nc:] + carry, tag)
                    iteration += 1

            return run_loop(consts, carry, 0, "")

        for operation_position, op in enumerate(source.operations):
            loc = op.origin.path + suffix
            origin = Origin(loc)
            origins.append(origin)
            budget.spend(loc)
            if op.primitive == "run_scoped":
                allocated, resources = [], set()
                for allocation in op.allocations:
                    budget.spend(loc + ".allocate")
                    resource = next_resource
                    next_resource += 1
                    allocated.append(TensorRef(resource))
                    resources.add(resource)
                    typ = allocation.type.inner
                    if isinstance(typ, SemaphoreType):
                        allocate_semaphore(resource, typ)
                        resources.update(semaphore_elements[resource])
                        if typ.kind == "dma":
                            dma_state.generations.update(
                                dict.fromkeys(semaphore_elements[resource], 0)
                            )
                    else:
                        if allocation.bound_array is None or allocation.lifetime != "scope":
                            raise UnsupportedJaxprError(loc, "invalid scoped array binding")
                        if typ.sharding is None or any(typ.sharding.partitions):
                            raise UnsupportedJaxprError(loc, "scoped storage must be shard-local")
                        types[resource] = allocation.bound_array
                        memories[resource] = TensorStorage(typ.shape, None, origin, budget)
                child = op.regions[0]
                results = evaluate(
                    child.region, tuple(read(a) for a in child.arguments) + tuple(allocated), suffix
                )
                if any(isinstance(v, TensorRef) for v in results):
                    raise UnsupportedJaxprError(loc, "Ref cannot escape a local allocation scope")
                dma_state.release_scope(resources, loc)
                if any(regular.get(r, 0) for r in resources):
                    raise UnsupportedJaxprError(loc, "scope exits with unconsumed semaphore count")
                for resource in resources:
                    memories.pop(resource, None)
                    types.pop(resource, None)
                    regular.pop(resource, None)
                    semaphores.discard(resource)
                    semaphore_shapes.pop(resource, None)
                    semaphore_elements.pop(resource, None)
                env.update((v.id, result) for v, result in zip(op.results, results, strict=True))
                continue
            if op.semaphore is not None:
                sem = op.semaphore
                resource = semaphore_slot(sem.semaphore, sem.index, loc)
                if sem.device_coordinates is not None:
                    raise UnsupportedJaxprError(
                        loc, "tensor remote signal needs peer protocol analysis"
                    )
                if semaphore_observer is not None:
                    if (
                        resource not in regular
                        or sem.phase == "read"
                        or sem.amount is None
                        or slot(sem.semaphore) >= len(call.kernel.inputs)
                    ):
                        raise UnsupportedJaxprError(
                            loc, "explicit core protocol requires call-lifetime regular signal/wait"
                        )
                    if sem.phase == "signal" and sem.core_index is None:
                        raise UnsupportedJaxprError(
                            loc, "explicit core protocol requires an explicit signal target"
                        )
                    target = (
                        0 if sem.core_index is None else static_integer(value(sem.core_index), loc)
                    )
                    semaphore_observer(
                        sem.phase, resource, target, static_integer(value(sem.amount), loc), origin
                    )
                    continue
                if sem.core_index is not None and static_integer(value(sem.core_index), loc) != 0:
                    raise UnsupportedJaxprError(loc, "semaphore target differs from bound core 0")
                if sem.phase == "read":
                    if resource in regular:
                        observed = regular[resource]
                    elif resource in semaphores and resource not in dma_state.pending:
                        observed = 0
                    else:
                        raise UnsupportedJaxprError(loc, "semaphore read depends on DMA scheduling")
                    env[op.results[0].id] = integer(observed)
                else:
                    if resource not in regular or sem.amount is None:
                        raise UnsupportedJaxprError(
                            loc, "counter operation requires a regular semaphore"
                        )
                    amount = static_integer(value(sem.amount), loc)
                    updated = regular[resource] + (amount if sem.phase == "signal" else -amount)
                    if not 0 <= amount <= MAX_DMA_BYTES or not 0 <= updated <= MAX_DMA_BYTES:
                        raise UnsupportedJaxprError(
                            loc, "tensor counter wait is unsatisfied or outside its range"
                        )
                    regular[resource] = updated
                continue
            if op.dma is not None:
                dma = op.dma
                if dma.device_coordinates is not None or dma.source_semaphore is not None:
                    raise UnsupportedJaxprError(loc, "tensor DMA requires local storage")
                src, dst = slot(dma.source), slot(dma.destination)
                sem = semaphore_slot(dma.semaphore, dma.semaphore_index, loc)
                if (
                    src not in types
                    or dst not in types
                    or dst in readonly
                    or types[src].dtype != types[dst].dtype
                ):
                    raise UnsupportedJaxprError(loc, "invalid tensor DMA data binding")
                selected = dynamic_view_case(
                    view_indices(types[src].shape, dma.source_transforms)
                    + view_indices(types[dst].shape, dma.destination_transforms),
                    loc,
                )
                if selected is not None:
                    operand, candidate, (low, high) = selected
                    return fork_address(
                        operand, candidate, low, high, loc, origin, operation_position
                    )
                boxes = []
                for resource, single, tensor, transforms in (
                    (src, dma.source_slice, dma.source_tensor_slice, dma.source_transforms),
                    (
                        dst,
                        dma.destination_slice,
                        dma.destination_tensor_slice,
                        dma.destination_transforms,
                    ),
                ):
                    index = tensor if tensor is not None else single
                    boxes.append(dma_rectangle(index, resource, loc, transforms))
                a, b = boxes
                if a.shape != b.shape:
                    raise UnsupportedJaxprError(loc, "tensor DMA region shapes differ")
                source_span, dest_span = (
                    a.box.linear_bounds(types[src].shape),
                    b.box.linear_bounds(types[dst].shape),
                )
                bound = replace(
                    dma, source=ValueRef(src), destination=ValueRef(dst), semaphore=ValueRef(sem)
                )
                count_bytes = a.size * np.dtype(types[src].dtype).itemsize
                aligned_regions = all(
                    span.start % 128 == span.size % 128 == 0
                    for endpoint, resource in ((a, src), (b, dst))
                    for box in endpoint.boxes
                    for span in (box.linear_bounds(types[resource].shape),)
                )
                if dma.phase == "start":
                    if npre <= src < npre + call.num_inputs:
                        domain = domains[src - npre]
                        if domain.padded:
                            # Automatic input staging leaves the out-of-bounds
                            # tail undefined. A DMA has no mask: every source
                            # element must belong to the true input domain.
                            for source_box in a.boxes:
                                source_ranges = {
                                    **ranges,
                                    **{
                                        f"local:{i}": (span.start, span.stop - 1)
                                        for i, span in enumerate(source_box.axes)
                                    },
                                }
                                for address, size in zip(domain.address, domain.shape, strict=True):
                                    low, high = bounds(
                                        substitute(address, iteration_coordinates), source_ranges
                                    )
                                    if low < 0 or high >= size:
                                        raise UnsupportedJaxprError(
                                            loc,
                                            "tensor DMA source includes undefined input padding",
                                        )
                    pieces = tuple(p for box in a.boxes for p in memories[src].read(box, loc))
                    for piece in pieces:
                        assert piece.value is not None
                        require_defined(
                            piece.value,
                            {
                                **ranges,
                                **{
                                    f"local:{i}": (span.start, span.stop - 1)
                                    for i, span in enumerate(piece.box.axes)
                                },
                            },
                            budget,
                            loc + ".dma_source",
                        )
                    transfer = dma_state.start(
                        bound,
                        origin,
                        source_version=memories[src].version,
                        source_span=source_span,
                        destination_span=dest_span,
                        source_versions=tuple(
                            VersionSlice(p.box.linear_bounds(types[src].shape), p.version)
                            for p in pieces
                        ),
                        dtype=types[src].dtype,
                        logical_bytes=count_bytes,
                        source_region=a.boxes,
                        destination_region=b.boxes,
                        aligned_regions=aligned_regions,
                    )
                    payloads[transfer.id] = (
                        a,
                        b,
                        tuple(
                            replace(p, value=_substitute(p.value, iteration_coordinates))
                            if p.value is not None and src in readonly
                            else p
                            for p in pieces
                        ),
                    )
                else:
                    completed = dma_state.wait(
                        bound,
                        origin,
                        source_span=source_span,
                        destination_span=dest_span,
                        dtype=types[src].dtype,
                        logical_bytes=count_bytes,
                        aligned_regions=aligned_regions,
                    )
                    for transfer in completed:
                        a, b, pieces = payloads.pop(transfer.id)
                        current = tuple(
                            VersionSlice(p.box.linear_bounds(types[src].shape), p.version)
                            for box in a.boxes
                            for p in memories[src].read(box, loc)
                        )
                        if current != transfer.source_versions:
                            raise UnsupportedJaxprError(loc, "tensor DMA source changed in flight")
                        version = 0
                        if a.transformed or b.transformed:
                            incoming = expression(assemble(pieces, types[src]), loc)
                            incoming = _substitute(_substitute(incoming, a.to_ref), b.to_value)
                            for box in b.boxes:
                                version = write(dst, box, incoming, origin)
                        else:
                            for piece in pieces:
                                box, incoming = a.copy_piece(piece, b)
                                version = write(dst, box, incoming, origin)
                        dma_state.completed.append(TransferEvidence(transfer, origin, version))
                continue
            if op.primitive == "jit" and len(op.regions) == 1:
                child = op.regions[0]
                results = evaluate(child.region, tuple(read(a) for a in child.arguments), suffix)
                env.update((v.id, a) for v, a in zip(op.results, results, strict=True))
                continue
            if op.primitive == "cond":
                selector = value(op.operands[0])
                require_defined(selector, ranges, budget, loc + ".condition")
                known = truth(selector, ranges)
                if known is not None:
                    selector = integer(int(known))
                if isinstance(selector, IndexExpr) and selector.kind == "constant":
                    branch = op.regions[max(0, min(int(selector.value), len(op.regions) - 1))]
                    results = evaluate(
                        branch.region, tuple(read(a) for a in branch.arguments), suffix
                    )
                else:
                    if dma_state.pending:
                        raise UnsupportedJaxprError(
                            loc, "tensor branch requires quiescent DMA at entry"
                        )
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
                        selector = ValueExpr(
                            "clamp",
                            scalar_type(
                                cast(ArrayType, value_types[cast(ValueRef, op.operands[0]).id])
                            ),
                            (integer(0), selector, integer(len(op.regions) - 1)),
                        )

                    results = fork(
                        selector,
                        tuple(
                            lambda case, branch=branch: evaluate(
                                branch.region,
                                tuple(read(a) for a in branch.arguments),
                                suffix + case,
                            )
                            for branch in op.regions
                        ),
                        op.results,
                        loc,
                        origin,
                    )
                env.update((v.id, a) for v, a in zip(op.results, results, strict=True))
                continue
            if op.primitive == "scan":
                params = {p.name: p.value for p in op.parameters}
                const_slots, carry_slots, xs_slots = cast(
                    tuple[tuple[int, ...], ...], params["input_groups"]
                )
                carry_out, ys_out = cast(tuple[tuple[int, ...], ...], params["output_groups"])
                consts = tuple(read(op.operands[i]) for i in const_slots)
                carry = tuple(read(op.operands[i]) for i in carry_slots)
                length = cast(int, params["length"])
                sequences: dict[int, tuple[Binding, ...]] = {}
                for iteration in range(length - 1, -1, -1) if params["reverse"] else range(length):
                    budget.spend(loc)
                    sliced = tuple(
                        _substitute(
                            value(op.operands[i]),
                            {
                                "local:0": integer(iteration),
                                **{
                                    f"local:{j}": coordinate("local", j - 1)
                                    for j in range(1, len(shape(op.operands[i])))
                                },
                            },
                        )
                        for i in xs_slots
                    )
                    result = evaluate(
                        op.regions[0].region,
                        consts + carry + sliced,
                        suffix + f".iteration[{iteration}]",
                    )
                    carry = tuple(result[i] for i in carry_out)
                    sequences[iteration] = tuple(result[i] for i in ys_out)
                output_values = list(carry)
                for i, out in enumerate(ys_out):
                    typ = op.results[out].type
                    if not isinstance(typ, ArrayType) or not length or typ.shape[0] != length:
                        raise UnsupportedJaxprError(
                            loc, "tensor scan requires a nonempty static history"
                        )
                    items = tuple(
                        _substitute(
                            expression(sequences[j][i], loc),
                            {
                                f"local:{k}": coordinate("local", k + 1)
                                for k in range(len(typ.shape) - 1)
                            },
                        )
                        for j in range(length)
                    )
                    output_values.append(
                        ValueExpr("select_n", scalar_type(typ), (coordinate("local", 0), *items))
                    )
                env.update((v.id, a) for v, a in zip(op.results, output_values, strict=True))
                continue
            if op.primitive == "while":
                params = {p.name: p.value for p in op.parameters}
                nc, nb = cast(int, params["cond_nconsts"]), cast(int, params["body_nconsts"])
                consts = tuple(read(a) for a in op.operands[: nc + nb])
                carry = tuple(read(a) for a in op.operands[nc + nb :])

                carry = run_while(op, consts, carry, nc, loc, origin)
                env.update((v.id, a) for v, a in zip(op.results, carry, strict=True))
                continue
            if op.access is not None:
                access = op.access
                resource = slot(access.ref)
                if resource not in memories:
                    raise UnsupportedJaxprError(loc, "tensor access is not an array Ref")
                if resource in unbuffered:
                    raise UnsupportedJaxprError(loc, "ANY storage requires an explicit DMA access")
                if access.transforms:
                    storage_shape = types[resource].shape
                    storage_indices = tuple(SliceIndex(0, n, 1) for n in storage_shape)
                    storage_index = (
                        storage_indices[0]
                        if len(storage_shape) == 1
                        else TensorIndex(storage_indices, storage_shape)
                    )
                    if access.index != storage_index:
                        raise UnsupportedJaxprError(
                            loc, "Ref transforms require an original-storage full base index"
                        )
                indices = (
                    access.index.indices
                    if isinstance(access.index, TensorIndex)
                    else (access.index,)
                )
                selected = None
                for index_axis, index in enumerate(indices):
                    operand = index.start if isinstance(index, DynamicSliceIndex) else index
                    if not isinstance(operand, ValueRef) or shape(operand):
                        continue
                    candidate = value(operand)
                    if isinstance(candidate, IndexExpr) and (
                        candidate.kind == "constant" or access.value is None and access.mask is None
                    ):
                        continue
                    interval = integer_value_bounds(candidate, ranges)
                    if interval is None:
                        raise UnsupportedJaxprError(
                            loc, "dynamic address needs finite integer bounds"
                        )
                    tail = (
                        (index.size - 1) * index.stride
                        if isinstance(index, DynamicSliceIndex)
                        else 0
                    )
                    if interval[0] < 0 or interval[1] + tail >= types[resource].shape[index_axis]:
                        raise UnsupportedJaxprError(
                            loc, "dynamic address bounds leave its original storage"
                        )
                    selected = operand, candidate, interval
                    break
                if selected is None and access.transforms:
                    selected = dynamic_view_case(
                        view_indices(types[resource].shape, access.transforms), loc
                    )
                if selected is not None:
                    operand, candidate, (low, high) = selected
                    return fork_address(
                        operand, candidate, low, high, loc, origin, operation_position
                    )
                boxes, to_ref, to_value, result_shape = access_region(
                    access.index, resource, loc, readonly_access=access.value is None
                )
                if access.transforms:
                    if boxes != (full_box(types[resource].shape),):
                        raise UnsupportedJaxprError(
                            loc, "Ref coordinate transforms require a full storage view"
                        )
                    result_shape, to_ref, to_value = view_coordinates(
                        types[resource].shape, fixed_views(access.transforms, loc), loc, budget
                    )
                    boxes = view_footprint(
                        types[resource].shape,
                        result_shape,
                        to_ref,
                        fixed_views(access.transforms, loc),
                        loc,
                        budget,
                    )
                if (
                    len(op.results) != 1
                    or not isinstance(op.results[0].type, ArrayType)
                    or op.results[0].type.shape != result_shape
                ):
                    raise UnsupportedJaxprError(
                        loc, "tensor access result shape differs from its index map"
                    )
                mask = value(access.mask) if access.mask is not None else True
                require_defined(
                    mask,
                    {**ranges, **coordinate_ranges("local", result_shape)},
                    budget,
                    loc + ".mask",
                )
                storage_mask = _substitute(mask, to_value)
                active = mask_regions(storage_mask, boxes, ranges, budget, loc)
                domain = domains[resource - npre] if npre <= resource < count else None
                valid: Expression = True
                if domain is not None and domain.padded:
                    valid = Predicate(
                        "and",
                        tuple(
                            condition
                            for address, size in zip(domain.address, domain.shape, strict=True)
                            for condition in (
                                Predicate(
                                    "ge", (substitute(address, iteration_coordinates), integer(0))
                                ),
                                Predicate(
                                    "lt",
                                    (substitute(address, iteration_coordinates), integer(size)),
                                ),
                            )
                        ),
                    )
                for region in active:
                    if resource in unbuffered:
                        require_true(valid, region.ranges, budget, loc)
                    dma_state.access(
                        resource,
                        region.box.linear_bounds(types[resource].shape),
                        write=access.value is not None,
                        path=loc,
                        region=region.box,
                    )
                pieces = tuple(
                    piece
                    for box in boxes
                    for piece in memories[resource].read(box, loc, initialized=False)
                )
                old = partial_value(pieces, types[resource])
                if access.value is None:
                    # An inactive lane does not read the underlying storage.
                    # Active scratch/output lanes must already be initialized.
                    for region in active:
                        checked = (
                            old
                            if region.guard is True
                            else ValueExpr(
                                "select_n",
                                scalar_type(types[resource]),
                                (
                                    region.guard,
                                    ValueExpr(
                                        "literal",
                                        scalar_type(types[resource]),
                                        data=np.zeros((), dtype=types[resource].dtype).tobytes(),
                                    ),
                                    old,
                                ),
                            )
                        )
                        require_defined(checked, region.ranges, budget, loc)
                old = _substitute(old, to_ref)
                if resource in readonly:
                    old = _substitute(old, iteration_coordinates)
                if domain is not None and domain.padded and resource in readonly:
                    valid_result = _substitute(valid, to_ref)
                    if defer_padding and access.mask is None and resource not in unbuffered:
                        old = ValueExpr(
                            "padded_value", scalar_type(types[resource]), (old, valid_result)
                        )
                    else:
                        old = ValueExpr(
                            "select_n",
                            scalar_type(types[resource]),
                            (
                                valid_result,
                                ValueExpr("undefined", scalar_type(types[resource])),
                                old,
                            ),
                        )
                if access.value is not None:
                    if shape(access.value) != result_shape:
                        raise UnsupportedJaxprError(
                            loc, "tensor write shape differs from its rectangle"
                        )
                    incoming = _substitute(value(access.value), to_value)
                    for region in active:
                        replacement = incoming
                        if region.guard is not True:
                            previous = partial_value(
                                memories[resource].read(region.box, loc, initialized=False),
                                types[resource],
                            )
                            replacement = ValueExpr(
                                "select_n",
                                scalar_type(types[resource]),
                                (region.guard, previous, incoming),
                            )
                        write(resource, region.box, replacement, origin)
                if access.mask is not None:
                    other = (
                        value(access.other)
                        if access.other is not None
                        else ValueExpr("undefined", scalar_type(types[resource]))
                    )
                    old = ValueExpr("select_n", scalar_type(types[resource]), (mask, other, old))
                elif access.other is not None:
                    value(access.other)
                env[op.results[0].id] = old
                continue
            if op.regions or op.call:
                raise UnsupportedJaxprError(loc, "tensor control region needs a resource proof")
            if len(op.results) != 1 or not isinstance(op.results[0].type, ArrayType):
                raise UnsupportedJaxprError(loc, "unexpected tensor operation result")
            local_ranges = {**ranges, **coordinate_ranges("local", op.results[0].type.shape)}
            env[op.results[0].id] = tensor_value(
                op,
                tuple(value(a) for a in op.operands),
                tuple(array_type(a) for a in op.operands),
                local_ranges,
                coordinates,
                grid,
                budget,
            )
        return tuple(read(a) for a in source.outputs)

    # Each index_map determines its own staging lifetime. Switching a tiled
    # output must not commit or invalidate an accumulator in another output.
    window_axes = tuple(
        tuple(i for i in serial_axes if i not in domain.serial_axes)
        for domain in domains[call.num_inputs :]
    )
    window_keys: list[tuple[int, ...] | None] = [None] * call.num_outputs
    windows: list[dict[tuple[IndexExpr, ...], tuple[tuple[int, ...], Expression]]] = [
        {} for _ in range(call.num_outputs)
    ]

    def window_identity(output: int, key: tuple[int, ...]) -> tuple[IndexExpr, ...]:
        if plan.windows:
            return plan.windows[output][key[0]].mapping
        fixed = {f"program:{i}": integer(k) for i, k in zip(window_axes[output], key, strict=True)}
        return tuple(substitute(v, fixed) for v in domains[call.num_inputs + output].mapping)

    def finish_window(output: int, key: tuple[int, ...]) -> None:
        resource = npre + call.num_inputs + output
        memory = memories[resource]
        dma_state.access(
            resource,
            memory.extent.linear_bounds(types[resource].shape),
            write=True,
            path=path + ".output_staging_boundary",
            region=memory.extent,
        )
        extent = (
            valid_output_box(resource, window_identity(output, key))
            if resource in restoring
            else memory.extent
        )
        value = partial_value(memory.read(extent, path, initialized=False), types[resource])
        windows[output][window_identity(output, key)] = (key, value)
        memory.invalidate(Origin(path + ".output_staging_boundary"))

    window_positions = [0] * call.num_outputs
    points = (
        (program_point,)
        if program_point is not None
        else product(*(range(grid[i]) for i in serial_axes))
    )
    for iteration, point in enumerate(points):
        step = dict(zip(serial_axes, point, strict=True))
        for j, axes in enumerate(window_axes):
            if plan.windows:
                if iteration == plan.windows[j][window_positions[j]].stop:
                    window_positions[j] += 1
                key = (window_positions[j],)
            else:
                key = tuple(step[i] for i in axes)
            previous = window_keys[j]
            if previous is not None and key != previous:
                finish_window(j, previous)
                block = call.blocks[call.num_inputs + j]
                saved = windows[j].get(window_identity(j, key))
                if (
                    saved is not None
                    and block.pipeline_mode is not None
                    and block.pipeline_mode.revisit == "any"
                ):
                    # Exact mapping equality retains each symbolic parallel
                    # owner. Only a previously completed full writeback can
                    # supply this block; first visits stay uninitialized.
                    resource = npre + call.num_inputs + j
                    origin = Origin(path + f".output_restore[{j}].grid{point}")
                    extent = valid_output_box(resource, window_identity(j, key))
                    write(resource, extent, saved[1], origin)
            window_keys[j] = key
        # The backend may reload staged inputs between grid programs. Their
        # old contents cannot remain DMA sources across this boundary. ANY
        # full-array references have no automatic staging overwrite.
        for i, block in enumerate(call.blocks[: call.num_inputs]):
            if not block.unbuffered:
                resource = npre + i
                box = full_box(types[resource].shape)
                dma_state.access(
                    resource,
                    box.linear_bounds(types[resource].shape),
                    write=True,
                    path=path + ".input_staging_boundary",
                    region=box,
                )
        iteration_coordinates = {
            f"program:{i}": integer(k) for i, k in zip(serial_axes, point, strict=True)
        }
        coordinates.update(iteration_coordinates)
        ranges.update({f"program:{i}": (k, k) for i, k in zip(serial_axes, point, strict=True)})
        evaluate(
            call.kernel,
            tuple(TensorRef(i) for i in range(len(call.kernel.inputs))),
            owner_suffix + (f".grid{point}" if serial_axes else ""),
        )
        if banked_interleaving and dma_state.pending:
            raise UnsupportedJaxprError(
                path, "interleaved banked programs require quiescent DMA at each program boundary"
            )
    if any(regular.values()):
        raise UnsupportedJaxprError(
            path, "tensor semaphore counts are not conserved at program exit"
        )
    communication = dma_state.finish(path)
    for j, key in enumerate(window_keys):
        assert key is not None
        finish_window(j, key)
    outputs = []
    for j, domain in enumerate(domains[call.num_inputs :]):
        result: Expression | None = None
        axes = window_axes[j]
        for key, value in reversed(tuple(windows[j].values())):
            if control_guard is not True:
                value = ValueExpr(
                    "select_n",
                    scalar_type(types[npre + call.num_inputs + j]),
                    (control_guard, value, value),
                    parameters=(ParameterView("__control_guard", True),),
                )
            bindings = plan.windows[j][key[0]].inverse if plan.windows else domain.inverse
            parts = []
            if plan.windows:
                for d, span in enumerate(plan.windows[j][key[0]].box.axes):
                    out = coordinate("output", d)
                    parts.extend(
                        (
                            Predicate("ge", (out, integer(span.start))),
                            Predicate("lt", (out, integer(span.stop))),
                        )
                    )
            elif axes:
                # Each formula is defined on this output rectangle. Use its
                # actual origin rather than modulo coordinates: the latter
                # obscure the correspondence to concatenated/sliced JAX output.
                bindings.update(
                    (f"program:{i}", integer(k)) for i, k in zip(axes, key, strict=True)
                )
                for d, (mapping, width) in enumerate(
                    zip(domain.mapping, domain.block_shape, strict=True)
                ):
                    start = scale(substitute(mapping, bindings), width)
                    out = coordinate("output", d)
                    bindings[f"local:{d}"] = add(out, scale(start, -1))
                    parts.extend(
                        (
                            Predicate("ge", (out, start)),
                            Predicate("lt", (out, add(start, integer(width)))),
                        )
                    )
            value = project_coordinates(
                value,
                bindings,
                {
                    substitute(a, bindings): coordinate("output", i)
                    for i, a in enumerate(domain.address)
                },
                {**(axis_ranges or {}), **coordinate_ranges("output", domain.shape)},
                budget,
            )
            if owner_suffix:
                guard = _substitute(
                    range_guard(
                        {
                            key: interval
                            for key, interval in (axis_ranges or {}).items()
                            if key.startswith("program:")
                        },
                        coordinate_ranges("program", grid),
                    ),
                    bindings,
                )
                value = ValueExpr(
                    "select_n",
                    scalar_type(types[npre + call.num_inputs + j]),
                    (
                        guard,
                        ValueExpr(
                            "literal",
                            scalar_type(types[npre + call.num_inputs + j]),
                            data=np.zeros(
                                (), dtype=types[npre + call.num_inputs + j].dtype
                            ).tobytes(),
                        ),
                        value,
                    ),
                )
            if result is None or expressions_equal((value,), (result,)):
                result = value
            else:
                result = ValueExpr(
                    "select_n",
                    scalar_type(types[npre + call.num_inputs + j]),
                    (Predicate("and", tuple(parts)), result, value),
                )
        assert result is not None
        outputs.append(
            resolve_defined(
                result,
                {**(axis_ranges or {}), **coordinate_ranges("output", domain.shape)},
                budget,
                path,
            )
        )
    return MemoryProjection(
        tuple(outputs),
        domains,
        tuple(versions),
        tuple(origins),
        (
            "grid/index_map/local coordinates have exact per-axis address correspondence",
            "every valid output coordinate belongs to one owning grid prefix",
            "trailing serial programs preserve scratch and counters in lexicographic order; "
            "each output index_map independently determines its staging window boundaries",
            "synchronous ANY output revisits restore the last completed version of the same "
            "window and parallel owner; first visits remain uninitialized",
            (
                "rectangular accesses preserve row boundaries and storage versions; "
                "input padding carries a separate complete-output initialization obligation"
                if defer_padding
                else "rectangular accesses preserve row boundaries and initialized versions"
            ),
            "serial execution exits with zero counters and quiescent DMA; "
            "pending transfers cannot overlap input reloads or output window changes",
            *(
                (
                    "each independent owner group initializes scratch before use and exits "
                    "with quiescent DMA; no memory state is assumed to survive between groups",
                )
                if plan.independent
                else ()
            ),
        ),
        communication,
        tuple(dma_state.waits),
    )
