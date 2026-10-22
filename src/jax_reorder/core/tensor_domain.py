"""Compose grid, Blocked index maps and local coordinates over exact integers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import cast

import numpy as np

from ..errors import UnsupportedJaxprError
from .index_domain import (
    AnalysisBudget,
    IndexExpr,
    add,
    axis,
    bounds,
    check_integer_range,
    divide,
    integer,
    integer_operation,
    literal_integer,
    lookup,
    modulo,
    scale,
    substitute,
)
from .model import (
    ArrayType,
    BlockMappingView,
    EffectView,
    LiteralValue,
    Operand,
    PallasCallView,
    RegionView,
    SemaphoreType,
    SliceIndex,
    TensorIndex,
    ValueRef,
    ValueType,
)

# Scalarization may erase array shapes only for these reviewed lane-wise rules.
# Cross-lane operations must retain coordinate maps and contraction domains even
# when every array involved is rank one.
POINTWISE_PRIMITIVES = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "rem",
        "neg",
        "sign",
        "abs",
        "max",
        "min",
        "clamp",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
        "xor",
        "not",
        "select_n",
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "tanh",
        "floor",
        "ceil",
        "is_finite",
        "stop_gradient",
        "integer_pow",
        "convert_element_type",
    }
)
TENSOR_VALUE_PRIMITIVES = frozenset(
    {
        "dot_general",
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_and",
        "reduce_or",
        "transpose",
        "reshape",
        "slice",
        "concatenate",
        "tile",
        "gather",
        "broadcast_to",
        "reshard",
    }
)


def requires_tensor_coordinates(
    primitive: str, operands: tuple[ValueType, ...], results: tuple[ValueType, ...]
) -> bool:
    """Keep input coordinates whenever scalarization would erase a shape map."""
    if primitive in TENSOR_VALUE_PRIMITIVES or any(
        isinstance(typ, ArrayType) and len(typ.shape) > 1 for typ in operands + results
    ):
        return True
    if primitive == "broadcast_in_dim":
        return any(isinstance(typ, ArrayType) and typ.shape for typ in operands)
    if primitive not in POINTWISE_PRIMITIVES or len(results) != 1:
        return False
    result = results[0]
    if not isinstance(result, ArrayType):
        return False
    # Scalars need no coordinate substitution. A non-scalar singleton operand
    # does: its lane must stay zero even when the result has many lanes.
    return any(
        isinstance(typ, ArrayType) and typ.shape and typ.shape != result.shape for typ in operands
    )


def restoring_outputs(call: PallasCallView) -> bool:
    """Whether a local executor needs the explicit output restoration path."""
    return any(
        block.pipeline_mode is not None and block.pipeline_mode.revisit == "any"
        for block in call.blocks[call.num_inputs :]
    )


def mutable_prefetch(call: PallasCallView) -> bool:
    """SMEM copies written by the kernel carry state across serial programs."""
    ports = {v.id for v in call.kernel.inputs[: len(call.prefetch)]}
    return any(
        isinstance(effect, EffectView) and effect.kind == "write" and effect.ref.id in ports
        for effect in call.kernel.effects
    )


def multidimensional(call: PallasCallView, *, masked: bool = True) -> bool:
    """Whether memory projection needs tensor coordinates, including rank-one values."""
    if (
        mutable_prefetch(call)
        or any(block.index_map.effects for block in call.blocks)
        or len(call.grid) != 1
        or any(len(b.block_shape) != 1 or b.unbuffered or b.squeezed_axes for b in call.blocks)
    ):
        return True
    if (call.scratch or masked) and any(b.array.shape[0] % b.block_shape[0] for b in call.blocks):
        # Padding with resources needs per-Ref valid-domain and lifetime checks.
        return True
    if any(isinstance(s.type.inner, SemaphoreType) and s.type.inner.shape for s in call.scratch):
        return True
    pending = [call.kernel]
    while pending:
        region = pending.pop()
        types = {v.id: v.type for v in region.inputs}
        types.update((v.id, v.type) for v, _ in region.constants)
        for op in region.operations:
            if (
                op.primitive == "run_scoped"
                or requires_tensor_coordinates(
                    op.primitive,
                    tuple(types[v.id] if isinstance(v, ValueRef) else v.type for v in op.operands),
                    tuple(v.type for v in op.results),
                )
                or op.access
                and (
                    op.access.transforms
                    or masked
                    and op.access.mask is not None
                    or isinstance(op.access.index, TensorIndex)
                    or isinstance(op.access.index, SliceIndex)
                    and op.access.index.stride != 1
                )
                or op.dma
                and (
                    op.dma.source_transforms
                    or op.dma.destination_transforms
                    or op.dma.semaphore_index is not None
                    or op.dma.source_tensor_slice is not None
                    or op.dma.destination_tensor_slice is not None
                )
            ):
                return True
            types.update((v.id, v.type) for v in op.results)
            pending.extend(r.region for r in op.regions)
    return False


def coordinate(prefix: str, dimension: int) -> IndexExpr:
    return axis(f"{prefix}:{dimension}")


def coordinate_ranges(prefix: str, shape: tuple[int, ...]) -> dict[str, tuple[int, int]]:
    return {f"{prefix}:{i}": (0, n - 1) for i, n in enumerate(shape)}


def index_map(
    region: RegionView, grid: tuple[int, ...], budget: AnalysisBudget
) -> tuple[IndexExpr, ...]:
    path = region.origin.path
    if region.effects or len(region.inputs) < len(grid):
        raise UnsupportedJaxprError(path, "index map requires pure static grid bindings")
    ranges = coordinate_ranges("program", grid)
    arguments = tuple(coordinate("program", i) if n > 1 else integer(0) for i, n in enumerate(grid))

    def literal(v: LiteralValue) -> IndexExpr | bool:
        if v.type.shape == () and np.dtype(v.type.dtype).kind == "b":
            return bool(np.frombuffer(v.data, dtype=v.type.dtype)[0])
        return literal_integer(v, path)

    class UndecidedControl(Exception):
        def __init__(self, path: str, dependencies: frozenset[int]) -> None:
            super().__init__(path)
            self.dependencies = dependencies

    def evaluate(
        source: RegionView,
        args: tuple[IndexExpr | bool | None, ...],
        input_dependencies: tuple[frozenset[int], ...] | None = None,
    ) -> tuple[IndexExpr | bool | None, ...]:
        # Unused prefetch inputs are deliberately unbound; reading one fails.
        env = {v.id: a for v, a in zip(source.inputs, args, strict=False)}
        env.update((v.id, literal(c)) for v, c in source.constants)
        dependencies = dict(
            zip(
                (v.id for v in source.inputs),
                input_dependencies
                if input_dependencies is not None
                else tuple(frozenset((i,)) if n > 1 else frozenset() for i, n in enumerate(grid)),
                strict=False,
            )
        )

        def dependency(v: Operand) -> frozenset[int]:
            return dependencies.get(v.id, frozenset()) if isinstance(v, ValueRef) else frozenset()

        def read(v: Operand) -> IndexExpr | bool | None:
            if isinstance(v, ValueRef):
                if v.id not in env:
                    raise UnsupportedJaxprError(path, "index map depends on prefetched data")
                return env[v.id]
            return literal(v)

        for op in source.operations:
            budget.spend(op.origin.path)
            depends = frozenset().union(*(dependency(a) for a in op.operands))
            dependencies.update((v.id, depends) for v in op.results)
            if op.primitive == "jit" and len(op.regions) == 1:
                child = op.regions[0]
                outputs = evaluate(
                    child.region,
                    tuple(read(a) for a in child.arguments),
                    tuple(dependency(a) for a in child.arguments),
                )
                env.update((v.id, a) for v, a in zip(op.results, outputs, strict=True))
                continue
            if op.primitive in ("scan", "while"):
                parameters = {p.name: p.value for p in op.parameters}
                if op.primitive == "scan":
                    const_slots, carry_slots, xs_slots = cast(
                        tuple[tuple[int, ...], ...], parameters["input_groups"]
                    )
                    carry_out, ys_out = cast(
                        tuple[tuple[int, ...], ...], parameters["output_groups"]
                    )
                    if xs_slots or ys_out:
                        raise UnsupportedJaxprError(
                            op.origin.path, "index-map scan requires scalar carry without a history"
                        )
                    consts = tuple(read(op.operands[i]) for i in const_slots)
                    carry = tuple(read(op.operands[i]) for i in carry_slots)
                    for _ in range(cast(int, parameters["length"])):
                        budget.spend(op.origin.path + ".iteration")
                        result = evaluate(
                            op.regions[0].region,
                            consts + carry,
                            (depends,) * (len(consts) + len(carry)),
                        )
                        carry = tuple(result[i] for i in carry_out)
                else:
                    nc, nb = (
                        cast(int, parameters["cond_nconsts"]),
                        cast(int, parameters["body_nconsts"]),
                    )
                    consts = tuple(read(v) for v in op.operands[: nc + nb])
                    carry = tuple(read(v) for v in op.operands[nc + nb :])
                    while True:
                        budget.spend(op.origin.path + ".iteration")
                        (condition,) = evaluate(
                            op.regions[0].region,
                            consts[:nc] + carry,
                            (depends,) * (nc + len(carry)),
                        )
                        if type(condition) is not bool:
                            raise UndecidedControl(op.origin.path, depends)
                        if not condition:
                            break
                        carry = evaluate(
                            op.regions[1].region,
                            consts[nc:] + carry,
                            (depends,) * (nb + len(carry)),
                        )
                env.update((v.id, a) for v, a in zip(op.results, carry, strict=True))
                continue
            if op.primitive == "cond":
                selector = read(op.operands[0])
                if not (
                    type(selector) is bool
                    or isinstance(selector, IndexExpr)
                    and selector.kind == "constant"
                ):
                    raise UndecidedControl(op.origin.path, dependency(op.operands[0]))
                selected = (
                    int(selector)
                    if type(selector) is bool
                    else int(cast(IndexExpr, selector).value)
                )
                selected = min(max(selected, 0), len(op.regions) - 1)
                child = op.regions[selected]
                outputs = evaluate(
                    child.region,
                    tuple(read(a) for a in child.arguments),
                    tuple(dependency(a) for a in child.arguments),
                )
                env.update((v.id, a) for v, a in zip(op.results, outputs, strict=True))
                continue
            if op.access or op.call or op.dma or op.semaphore or op.regions:
                raise UnsupportedJaxprError(
                    op.origin.path, "unsupported tensor index-map operation"
                )
            if (
                len(op.results) != 1
                or not isinstance(op.results[0].type, ArrayType)
                or op.results[0].type.shape
            ):
                raise UnsupportedJaxprError(op.origin.path, "index map must return integer scalars")
            values = tuple(read(a) for a in op.operands)
            name = op.primitive
            value: IndexExpr | bool | None
            if name == "convert_element_type" and type(values[0]) is bool:
                value = integer(int(values[0]))
            elif name in ("eq", "ne", "lt", "le", "gt", "ge"):
                if any(v is None for v in values):
                    env[op.results[0].id] = None
                    continue
                a, b = tuple(
                    integer(int(v)) if type(v) is bool else cast(IndexExpr, v) for v in values
                )
                low, high = bounds(add(a, scale(b, -1)), ranges)
                known = {
                    "eq": True if low == high == 0 else False if low > 0 or high < 0 else None,
                    "ne": False if low == high == 0 else True if low > 0 or high < 0 else None,
                    "lt": True if high < 0 else False if low >= 0 else None,
                    "le": True if high <= 0 else False if low > 0 else None,
                    "gt": True if low > 0 else False if high <= 0 else None,
                    "ge": True if low >= 0 else False if high < 0 else None,
                }[name]
                value = known
            elif name in ("and", "or", "xor", "not") and all(
                v is None or type(v) is bool for v in values
            ):
                if name == "and":
                    value = (
                        False
                        if False in values
                        else True
                        if all(v is True for v in values)
                        else None
                    )
                elif name == "or":
                    value = (
                        True
                        if True in values
                        else False
                        if all(v is False for v in values)
                        else None
                    )
                else:
                    value = (
                        None
                        if None in values
                        else not values[0]
                        if name == "not"
                        else values[0] != values[1]
                    )
            elif name == "select_n":
                selector = values[0]
                if not (
                    type(selector) is bool
                    or isinstance(selector, IndexExpr)
                    and selector.kind == "constant"
                ):
                    raise UndecidedControl(op.origin.path, dependency(op.operands[0]))
                selected = (
                    int(selector)
                    if type(selector) is bool
                    else int(cast(IndexExpr, selector).value)
                )
                if not 0 <= selected < len(values) - 1:
                    raise UnsupportedJaxprError(op.origin.path, "invalid index-map selector")
                value = values[selected + 1]
            elif any(v is None for v in values):
                value = None
            elif name == "sign" and isinstance(values[0], IndexExpr):
                lo, hi = bounds(values[0], ranges)
                value = (
                    integer(1)
                    if lo > 0
                    else integer(-1)
                    if hi < 0
                    else integer(0)
                    if lo == hi == 0
                    else None
                )
            elif all(isinstance(v, IndexExpr) for v in values):
                value = integer_operation(
                    name, cast(tuple[IndexExpr, ...], values), ranges, op.origin.path
                )
            else:
                raise UnsupportedJaxprError(
                    op.origin.path, "unsupported index-map boolean operation"
                )
            if isinstance(value, IndexExpr):
                check_integer_range(value, op.results[0].type.dtype, ranges, op.origin.path)
            env[op.results[0].id] = value
        result = tuple(read(v) for v in source.outputs)
        if any(v is None for v in result):
            raise UndecidedControl(
                path,
                frozenset().union(
                    *(
                        dependency(v)
                        for v, a in zip(source.outputs, result, strict=True)
                        if a is None
                    )
                ),
            )
        return result

    for v, value in zip(region.inputs, arguments, strict=False):
        if not isinstance(v.type, ArrayType) or v.type.shape:
            raise UnsupportedJaxprError(path, "invalid grid coordinate type")
        check_integer_range(value, v.type.dtype, ranges, path)
    try:
        outputs = evaluate(region, arguments)
    except UndecidedControl as undecided:
        # Endpoint refinement often proves sign helpers without enumerating
        # the grid. Only controls that still vary require a finite table.
        splits = tuple(
            ((0, 0),)
            if n == 1
            else ((0, 0), (1, 1))
            if n == 2
            else ((0, 0), (1, n - 2), (n - 1, n - 1))
            for n in grid
        )
        budget.spend(path, prod(len(s) for s in splits))
        agreed = None
        varying = undecided.dependencies
        try:
            for rectangle in product(*splits):
                ranges = {f"program:{i}": interval for i, interval in enumerate(rectangle)}
                candidate = evaluate(region, arguments)
                if agreed is not None and candidate != agreed:
                    raise UndecidedControl(path, varying)
                agreed = candidate
            assert agreed is not None
            outputs = agreed
        except UndecidedControl as error:
            varying |= error.dependencies
            while True:
                axes = tuple(sorted(varying))
                size = prod(grid[i] for i in axes)
                budget.spend(path + ".finite_index_map", size)
                if not axes:
                    raise UnsupportedJaxprError(
                        path, "index-map control has no static grid proof"
                    ) from error
                entries = []
                try:
                    for point in product(*(range(grid[i]) for i in axes)):
                        fixed = dict(zip(axes, point, strict=True))
                        ranges = {
                            f"program:{i}": (fixed[i], fixed[i]) if i in fixed else (0, n - 1)
                            for i, n in enumerate(grid)
                        }
                        args = tuple(
                            integer(fixed[i]) if i in fixed else v for i, v in enumerate(arguments)
                        )
                        entries.append(evaluate(region, args))
                except UndecidedControl as more:
                    additional = more.dependencies - varying
                    if not additional:
                        raise UnsupportedJaxprError(
                            path, "index-map control has no static grid proof"
                        ) from more
                    varying |= additional
                    continue
                selector = add(
                    *(
                        scale(arguments[i], prod(grid[j] for j in axes[k + 1 :]))
                        for k, i in enumerate(axes)
                    )
                )
                if any(not isinstance(v, IndexExpr) for entry in entries for v in entry):
                    raise UnsupportedJaxprError(
                        path, "index map must return integer coordinates"
                    ) from error
                outputs = tuple(
                    lookup(selector, tuple(cast(IndexExpr, entry[k]) for entry in entries))
                    for k in range(len(region.outputs))
                )
                break
    if not all(isinstance(v, IndexExpr) for v in outputs):
        raise UnsupportedJaxprError(path, "index map must return integer coordinates")
    return cast(tuple[IndexExpr, ...], outputs)


@dataclass(frozen=True)
class TensorDomain:
    shape: tuple[int, ...]
    block_shape: tuple[int, ...]
    grid: tuple[int, ...]
    mapping: tuple[IndexExpr, ...]
    # One grid axis per nontrivial output tile axis; -1 denotes a singleton.
    writer_axes: tuple[tuple[int, bool], ...] = ()
    linear_order: tuple[tuple[int, bool], ...] = ()
    serial_axes: tuple[int, ...] = ()
    finite_inverse: tuple[IndexExpr, ...] = ()
    squeezed_axes: tuple[int, ...] = ()

    @property
    def local_axes(self) -> tuple[int, ...]:
        return tuple(i for i in range(len(self.block_shape)) if i not in self.squeezed_axes)

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(self.block_shape[i] for i in self.local_axes)

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def block(self) -> int:
        return prod(self.block_shape)

    @property
    def programs(self) -> int:
        return prod(self.grid)

    @property
    def padded(self) -> bool:
        return any(n % b for n, b in zip(self.shape, self.block_shape, strict=True))

    @property
    def address(self) -> tuple[IndexExpr, ...]:
        local = {axis: coordinate("local", i) for i, axis in enumerate(self.local_axes)}
        return tuple(
            add(scale(p, b), local.get(i, integer(0)))
            for i, (p, b) in enumerate(zip(self.mapping, self.block_shape, strict=True))
        )

    @property
    def inverse(self) -> dict[str, IndexExpr]:
        result = {f"program:{i}": integer(0) for i in range(len(self.grid))}
        result.update(
            {
                f"local:{i}": modulo(coordinate("output", d), self.block_shape[d])
                for i, d in enumerate(self.local_axes)
            }
        )
        if self.finite_inverse:
            result.update((f"program:{i}", v) for i, v in enumerate(self.finite_inverse))
            return result
        if self.linear_order:
            tiles = tuple(
                (n + b - 1) // b for n, b in zip(self.shape, self.block_shape, strict=True)
            )
            linear = add(
                *(
                    scale(divide(coordinate("output", i), b), prod(tiles[i + 1 :]))
                    for i, b in enumerate(self.block_shape)
                )
            )
            for pos, (p, reverse) in enumerate(self.linear_order):
                stride = prod(self.grid[j] for j, _ in self.linear_order[pos + 1 :])
                value = modulo(divide(linear, stride), self.grid[p])
                result[f"program:{p}"] = (
                    add(integer(self.grid[p] - 1), scale(value, -1)) if reverse else value
                )
            return result
        for i, (b, (p, reverse)) in enumerate(zip(self.block_shape, self.writer_axes, strict=True)):
            item = coordinate("output", i)
            if p >= 0:
                group = divide(item, b)
                result[f"program:{p}"] = (
                    add(integer(self.grid[p] - 1), scale(group, -1)) if reverse else group
                )
        return result


def _prove_tensor_domain(
    block: BlockMappingView,
    grid: tuple[int, ...],
    budget: AnalysisBudget,
    *,
    writer: bool,
    mapping: tuple[IndexExpr, ...] | None = None,
) -> TensorDomain:
    path = block.origin.path
    budget.spend(path)
    shape, widths = block.array.shape, block.block_shape
    if len(shape) != len(widths) or not shape or any(n <= 0 for n in (*shape, *widths, *grid)):
        raise UnsupportedJaxprError(path, "invalid static tensor domain")
    squeezed = block.squeezed_axes
    if (
        type(squeezed) is not tuple
        or any(type(i) is not int or not 0 <= i < len(widths) for i in squeezed)
        or tuple(sorted(set(squeezed))) != squeezed
        or any(widths[i] != 1 for i in squeezed)
        or block.ref.inner.shape != tuple(n for i, n in enumerate(widths) if i not in squeezed)
    ):
        raise UnsupportedJaxprError(path, "invalid squeezed block coordinate binding")
    if mapping is None:
        mapping = index_map(block.index_map, grid, budget)
    if len(mapping) != len(shape):
        raise UnsupportedJaxprError(path, "index-map rank differs from array rank")
    tiles = tuple((n + b - 1) // b for n, b in zip(shape, widths, strict=True))
    ranges = coordinate_ranges("program", grid)
    axes: list[tuple[int, bool]] = []
    used: set[int] = set()
    separable = True
    for expression, extent in zip(mapping, tiles, strict=True):
        lo, hi = bounds(expression, ranges)
        if not 0 <= lo <= hi < extent:
            raise UnsupportedJaxprError(path, "index map reaches an out-of-bounds block")
        if not writer:
            continue
        if extent == 1 and lo == hi == 0:
            axes.append((-1, False))
            continue
        matched = False
        for i, size in enumerate(grid):
            forward = coordinate("program", i)
            reverse = add(integer(size - 1), scale(forward, -1))
            if i not in used and size == extent and expression in (forward, reverse):
                used.add(i)
                axes.append((i, expression == reverse))
                matched = True
                break
        if not matched:
            separable = False
    if writer and (not separable or used != {i for i, size in enumerate(grid) if size > 1}):
        linear = add(*(scale(v, prod(tiles[i + 1 :])) for i, v in enumerate(mapping)))
        for _ in mapping:
            budget.spend(path)
            reduced = add(linear)
            if reduced == linear:
                break
            linear = reduced
        coefficients: dict[int, int] = {}
        for term in linear.args if linear.kind == "sum" else (linear,):
            coefficient, item = (
                (int(term.value), term.args[0]) if term.kind == "scale" else (1, term)
            )
            if item.kind == "constant":
                continue
            if item.kind != "axis" or not str(item.value).startswith("program:"):
                raise UnsupportedJaxprError(path, "output block map lacks a proven axis bijection")
            coefficients[int(str(item.value).split(":")[1])] = coefficient
        order = tuple(
            (i, coefficients[i] < 0)
            for i in sorted(coefficients, key=lambda i: abs(coefficients[i]), reverse=True)
        )
        if set(coefficients) != {i for i, n in enumerate(grid) if n > 1} or prod(grid) != prod(
            tiles
        ):
            raise UnsupportedJaxprError(
                path, "grid has redundant writers or an uncovered output axis"
            )
        expected = []
        for position, (i, reverse) in enumerate(order):
            stride = prod(grid[j] for j, _ in order[position + 1 :])
            if abs(coefficients[i]) != stride:
                raise UnsupportedJaxprError(path, "output map is not a mixed-radix bijection")
            value = coordinate("program", i)
            expected.append(
                scale(add(integer(grid[i] - 1), scale(value, -1)) if reverse else value, stride)
            )
        if linear != add(*expected):
            raise UnsupportedJaxprError(path, "output linearized map is shifted outside its grid")
        return TensorDomain(shape, widths, grid, mapping, (), order, squeezed_axes=squeezed)
    result = TensorDomain(shape, widths, grid, mapping, tuple(axes), squeezed_axes=squeezed)
    if writer:
        for i, address in enumerate(result.address):
            if substitute(address, result.inverse) != coordinate("output", i):
                # A singleton block axis has no varying tile coordinate.
                if not (tiles[i] == 1 and shape[i] <= widths[i]):
                    raise UnsupportedJaxprError(path, "tensor address inverse was not proved")
    return result


def _separable_finite_inverse(
    domain: TensorDomain, budget: AnalysisBudget, path: str
) -> tuple[IndexExpr, ...] | None:
    """Finite permutations of individual axes retain other symbolic owners."""
    result = [integer(0) for _ in domain.grid]
    used = set()
    for d, (mapping, extent, width) in enumerate(
        zip(domain.mapping, domain.shape, domain.block_shape, strict=True)
    ):
        tiles = (extent + width - 1) // width
        pending, axes = [mapping], set()
        while pending:
            item = pending.pop()
            if item.kind == "axis" and str(item.value).startswith("program:"):
                axis_id = int(str(item.value).split(":")[1])
                if domain.grid[axis_id] > 1:
                    axes.add(axis_id)
            pending.extend(item.args)
        if not axes and tiles == 1:
            continue
        if len(axes) != 1:
            return None
        (axis_id,) = axes
        if axis_id in used or domain.grid[axis_id] != tiles:
            return None
        used.add(axis_id)
        selector = divide(coordinate("output", d), width)
        forward = coordinate("program", axis_id)
        reverse = add(integer(tiles - 1), scale(forward, -1))
        if mapping == forward:
            result[axis_id] = selector
        elif mapping == reverse:
            result[axis_id] = add(integer(tiles - 1), scale(selector, -1))
        else:
            budget.spend(path + ".finite_axis_bijection", tiles)
            values = tuple(
                substitute(mapping, {f"program:{axis_id}": integer(i)}) for i in range(tiles)
            )
            if any(v.kind != "constant" for v in values):
                return None
            inverse = {int(v.value): i for i, v in enumerate(values)}
            if len(inverse) != tiles:
                raise UnsupportedJaxprError(
                    path, "output map has redundant writers or an uncovered output axis"
                )
            result[axis_id] = lookup(selector, tuple(integer(inverse[i]) for i in range(tiles)))
    if used != {i for i, n in enumerate(domain.grid) if n > 1}:
        return None
    return tuple(result)


def prove_tensor_domain(
    block: BlockMappingView,
    grid: tuple[int, ...],
    budget: AnalysisBudget,
    *,
    writer: bool,
    mapping: tuple[IndexExpr, ...] | None = None,
) -> TensorDomain:
    if mapping is None:
        mapping = index_map(block.index_map, grid, budget)
    try:
        return _prove_tensor_domain(block, grid, budget, writer=writer, mapping=mapping)
    except UnsupportedJaxprError as original:
        if not writer:
            raise
        # First retain the ordinary range proof. Enumeration is a finite
        # writer-bijection fallback, never permission to accept invalid maps.
        domain = _prove_tensor_domain(block, grid, budget, writer=False, mapping=mapping)
        separated = _separable_finite_inverse(domain, budget, block.origin.path)
        if separated is not None:
            return TensorDomain(
                domain.shape,
                domain.block_shape,
                grid,
                mapping,
                finite_inverse=separated,
                squeezed_axes=domain.squeezed_axes,
            )
        tiles = tuple(
            (n + b - 1) // b for n, b in zip(domain.shape, domain.block_shape, strict=True)
        )
        if prod(grid) != prod(tiles):
            raise UnsupportedJaxprError(
                block.origin.path, "grid has redundant writers or an uncovered output axis"
            ) from original
        budget.spend(block.origin.path + ".finite_bijection", prod(grid))
        inverse: dict[int, tuple[int, ...]] = {}
        for point in product(*(range(n) for n in grid)):
            bindings = {f"program:{i}": integer(k) for i, k in enumerate(point)}
            target = tuple(substitute(v, bindings) for v in mapping)
            if any(v.kind != "constant" for v in target):
                raise UnsupportedJaxprError(
                    block.origin.path, "output map lacks a static finite bijection"
                ) from original
            position = sum(int(v.value) * prod(tiles[i + 1 :]) for i, v in enumerate(target))
            if position in inverse:
                raise UnsupportedJaxprError(
                    block.origin.path,
                    "output map has redundant writers or an uncovered output axis",
                ) from original
            inverse[position] = point
        selector = add(
            *(
                scale(divide(coordinate("output", i), b), prod(tiles[i + 1 :]))
                for i, b in enumerate(domain.block_shape)
            )
        )
        return TensorDomain(
            domain.shape,
            domain.block_shape,
            grid,
            mapping,
            finite_inverse=tuple(
                lookup(selector, tuple(integer(inverse[k][i]) for k in range(prod(tiles))))
                for i in range(len(grid))
            ),
            squeezed_axes=domain.squeezed_axes,
        )
