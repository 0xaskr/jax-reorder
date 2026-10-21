"""Coordinate-preserving scalarization of reviewed tensor value operations."""

from __future__ import annotations

from dataclasses import replace
from math import prod
from typing import cast

import numpy as np

from ..errors import UnsupportedJaxprError
from .index_domain import (
    AnalysisBudget,
    IndexExpr,
    add,
    bounds,
    check_integer_range,
    divide,
    integer,
    modulo,
    scale,
)
from .lane_gather import lane_gather
from .local_reshard import validate_local_broadcast, validate_reshard_operation
from .memory import Expression, Predicate, ValueExpr, _substitute, scalar_type, truth
from .model import ArrayType, OperationView, ParameterView
from .tensor_domain import coordinate
from .value_operations import pure_value


def tensor_value(
    op: OperationView,
    args: tuple[Expression, ...],
    input_types: tuple[ArrayType, ...],
    ranges: dict[str, tuple[int, int]],
    coordinates: dict[str, IndexExpr],
    grid: tuple[int, ...],
    budget: AnalysisBudget,
) -> Expression:
    path, name = op.origin.path, op.primitive
    if len(op.results) != 1 or not isinstance(op.results[0].type, ArrayType):
        raise UnsupportedJaxprError(path, "tensor value needs one array result")
    typ = op.results[0].type
    input_shapes = tuple(t.shape for t in input_types)
    if typ.sharding is None or any(typ.sharding.partitions):
        raise UnsupportedJaxprError(path, "tensor values must be shard-local")
    p = {v.name: v.value for v in op.parameters}
    if name == "reshard":
        validate_reshard_operation(op, input_types)
        return args[0]
    if name == "gather":
        return lane_gather(op, args, input_types, budget)
    if name in ("program_id", "num_programs"):
        dimension = cast(int, p["axis"])
        if not 0 <= dimension < len(grid):
            raise UnsupportedJaxprError(path, "program axis is outside the grid")
        return (
            integer(grid[dimension])
            if name == "num_programs"
            else coordinates.get(
                f"program:{dimension}",
                coordinate("program", dimension) if grid[dimension] > 1 else integer(0),
            )
        )
    if name in ("dot_general", "reduce_sum", "reduce_max"):
        # The operands are functions of bound input coordinates. Output-address
        # substitution must never capture a reduction/contraction coordinate.
        pending, seen, depth = list(args), set(), 0
        while pending:
            item = pending.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            budget.spend(path + ".tensor_bindings")
            if isinstance(item, ValueExpr):
                if item.operation.startswith("tensor."):
                    depth = max(
                        depth,
                        cast(int, dict((p.name, p.value) for p in item.parameters)["__depth"]),
                    )
                pending.extend(item.operands)
        depth += 1
        axes = tuple(
            tuple(f"tensor:{depth}:{i}:{shape}:{j}" for j in range(len(shape)))
            for i, shape in enumerate(input_shapes)
        )
        return ValueExpr(
            "tensor." + name,
            scalar_type(typ),
            tuple(
                _substitute(
                    a, {f"local:{j}": IndexExpr("axis", value=key) for j, key in enumerate(keys)}
                )
                for a, keys in zip(args, axes, strict=True)
            )
            + tuple(coordinate("local", i) for i in range(len(typ.shape))),
            op.parameters
            + (
                ParameterView("__depth", depth),
                ParameterView("__input_shapes", input_shapes),
                ParameterView("__input_dtypes", tuple(t.dtype for t in input_types)),
                ParameterView("__input_axes", axes),
                ParameterView("__output_shape", typ.shape),
            ),
        )
    if name == "iota":
        dimension = cast(int, p["dimension"])
        if p["shape"] != typ.shape or not 0 <= dimension < len(typ.shape):
            raise UnsupportedJaxprError(path, "invalid tensor iota dimension")
        if np.dtype(typ.dtype).kind not in "iu":
            raise UnsupportedJaxprError(path, "tensor iota requires an integer coordinate dtype")
        value = coordinate("local", dimension)
        check_integer_range(value, typ.dtype, ranges, path)
        return value
    if name == "broadcast_in_dim":
        if "__pallas_kernel_broadcast" in p or p.get("sharding") not in (None, ("none",)):
            validate_local_broadcast(op, input_types)
        dimensions = cast(tuple[int, ...], p["broadcast_dimensions"])
        shape = input_shapes[0]
        if len(dimensions) != len(shape):
            raise UnsupportedJaxprError(path, "invalid tensor broadcast interface")
        return _substitute(
            args[0],
            {
                f"local:{i}": integer(0) if n == 1 else coordinate("local", dimensions[i])
                for i, n in enumerate(shape)
            },
        )
    if name == "transpose":
        permutation = cast(tuple[int, ...], p["permutation"])
        return _substitute(
            args[0], {f"local:{d}": coordinate("local", i) for i, d in enumerate(permutation)}
        )
    if name == "tile":
        shape = input_shapes[0]
        offset = len(typ.shape) - len(shape)
        if offset < 0:
            raise UnsupportedJaxprError(path, "tile reduces input rank")
        return _substitute(
            args[0],
            {f"local:{i}": modulo(coordinate("local", i + offset), n) for i, n in enumerate(shape)},
        )
    if name == "reshape":
        input_shape = input_shapes[0]
        permutation = (
            tuple(range(len(input_shape)))
            if p["dimensions"] in (None, ("none",))
            else cast(tuple[int, ...], p["dimensions"])
        )
        if sorted(permutation) != list(range(len(input_shape))):
            raise UnsupportedJaxprError(path, "reshape dimensions are not an input permutation")
        shape = tuple(input_shape[i] for i in permutation)
        if prod(shape) != prod(typ.shape):
            raise UnsupportedJaxprError(path, "reshape changes the number of elements")
        linear = add(
            *(
                scale(coordinate("local", i), prod(typ.shape[i + 1 :]))
                for i in range(len(typ.shape))
            )
        )
        return _substitute(
            args[0],
            {
                f"local:{permutation[i]}": modulo(divide(linear, prod(shape[i + 1 :])), n)
                for i, n in enumerate(shape)
            },
        )
    broadcasted = []
    for value, shape in zip(args, input_shapes, strict=True):
        offset = len(typ.shape) - len(shape)
        if offset < 0 or any(n not in (1, typ.shape[offset + i]) for i, n in enumerate(shape)):
            raise UnsupportedJaxprError(path, "unreviewed tensor value broadcasting")
        broadcasted.append(
            _substitute(
                value,
                {
                    f"local:{i}": integer(0) if n == 1 else coordinate("local", offset + i)
                    for i, n in enumerate(shape)
                },
            )
        )
    args = tuple(broadcasted)
    if name == "broadcast_to":
        return args[0]
    scalar_op = replace(op, results=(replace(op.results[0], type=replace(typ, shape=())),))
    return pure_value(scalar_op, args, ranges, coordinates)


def project_coordinates(
    value: Expression,
    bindings: dict[str, IndexExpr],
    identities: dict[IndexExpr, IndexExpr],
    ranges: dict[str, tuple[int, int]],
    budget: AnalysisBudget,
) -> Expression:
    """Apply proved address identities and bounded integer rules, never CAS queries."""
    root = _substitute(value, bindings)
    cache: dict[int, Expression] = {}
    # Binder names include their shapes, so each name has one exact range even
    # when distinct tensor operations share a lexical depth.
    ranges = dict(ranges)
    pending, seen = [root], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        budget.spend("tensor.binders")
        if isinstance(item, ValueExpr | Predicate):
            pending.extend(item.operands)
            if isinstance(item, ValueExpr) and item.operation.startswith("tensor."):
                p = dict((a.name, a.value) for a in item.parameters)
                for names, shape in zip(
                    cast(tuple[tuple[str, ...], ...], p["__input_axes"]),
                    cast(tuple[tuple[int, ...], ...], p["__input_shapes"]),
                    strict=True,
                ):
                    ranges.update((key, (0, n - 1)) for key, n in zip(names, shape, strict=True))

    def visit(item: Expression) -> Expression:
        if id(item) in cache:
            return cache[id(item)]
        budget.spend("tensor.coordinates")
        result: Expression
        if isinstance(item, IndexExpr):
            if item in identities:
                result = identities[item]
            else:
                args = tuple(cast(IndexExpr, visit(a)) for a in item.args)
                if item.kind == "sum":
                    result = add(*args)
                elif item.kind == "scale":
                    result = scale(args[0], int(item.value))
                elif item.kind in ("div", "mod"):
                    divisor = int(item.value)
                    lo, hi = bounds(args[0], ranges)
                    if lo // divisor == hi // divisor:
                        result = (
                            integer(lo // divisor)
                            if item.kind == "div"
                            else add(args[0], integer(-(lo // divisor) * divisor))
                        )
                    else:
                        result = (divide if item.kind == "div" else modulo)(args[0], divisor)
                else:
                    result = replace(item, args=args)
            assert isinstance(result, IndexExpr)
            low, high = bounds(result, ranges)
            if low == high:
                result = integer(low)
        elif isinstance(item, Predicate):
            result = Predicate(
                item.operation,
                cast(
                    tuple[IndexExpr | Predicate | bool, ...], tuple(visit(a) for a in item.operands)
                ),
            )
            known = truth(result, ranges)
            if known is not None:
                result = known
        elif isinstance(item, ValueExpr):
            if item.operation == "select_n":
                selector = visit(item.operands[0])
                selected = (
                    int(selector)
                    if type(selector) is bool
                    else int(selector.value)
                    if isinstance(selector, IndexExpr) and selector.kind == "constant"
                    else None
                )
                if selected is not None and 0 <= selected < len(item.operands) - 1:
                    result = visit(item.operands[selected + 1])
                    cache[id(item)] = result
                    return result
            children = tuple(visit(a) for a in item.operands)
            result = replace(item, operands=children)
            if (
                item.operation == "convert_element_type"
                and len(children) == 1
                and isinstance(children[0], IndexExpr)
                and children[0].kind == "constant"
                and item.type.dtype in ("float32", "float64")
                and abs(int(children[0].value)) <= 2**53
            ):
                # The integer is exactly representable by the host conversion's
                # intermediate double; retain the actual target dtype rounding.
                result = ValueExpr(
                    "literal",
                    item.type,
                    data=np.asarray(int(children[0].value), dtype=item.type.dtype).tobytes(),
                )
        else:
            result = item
        cache[id(item)] = result
        return result

    return visit(root)
