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
    *,
    tensor_depths: dict[int, tuple[Expression, int]] | None = None,
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
    if name in ("dot_general", "reduce_sum", "reduce_max", "reduce_min", "reduce_and", "reduce_or"):
        # The operands are functions of bound input coordinates. Output-address
        # substitution must never capture a reduction/contraction coordinate.
        depths = {} if tensor_depths is None else tensor_depths
        pending = [(item, False) for item in args]
        while pending:
            item, ready = pending.pop()
            if id(item) in depths:
                continue
            if not ready:
                budget.spend(path + ".tensor_bindings")
                pending.append((item, True))
                if isinstance(item, ValueExpr):
                    pending.extend((child, False) for child in item.operands)
                continue
            depth = (
                cast(int, dict((p.name, p.value) for p in item.parameters)["__depth"])
                if isinstance(item, ValueExpr) and item.operation.startswith("tensor.")
                else 0
            )
            if isinstance(item, ValueExpr):
                depth = max(depth, *(depths[id(child)][1] for child in item.operands), 0)
            # Keep the immutable source alive with its metadata, preventing an
            # unrelated later object from reusing a memoized Python identity.
            depths[id(item)] = item, depth
        depth = max((depths[id(item)][1] for item in args), default=0) + 1
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
    if name == "squeeze":
        if (
            len(args) != 1
            or len(input_types) != 1
            or len(op.parameters) != 1
            or set(p) != {"dimensions"}
        ):
            raise UnsupportedJaxprError(path, "invalid squeeze interface")
        raw_dimensions = p["dimensions"]
        shape = input_shapes[0]
        if (
            type(raw_dimensions) is not tuple
            or any(type(d) is not int or not 0 <= d < len(shape) for d in raw_dimensions)
            or len(set(raw_dimensions)) != len(raw_dimensions)
        ):
            raise UnsupportedJaxprError(path, "squeeze dimensions must select distinct unit axes")
        dimensions = cast(tuple[int, ...], raw_dimensions)
        if any(shape[d] != 1 for d in dimensions):
            raise UnsupportedJaxprError(path, "squeeze dimensions must select distinct unit axes")
        kept = tuple(i for i in range(len(shape)) if i not in dimensions)
        if typ.shape != tuple(shape[i] for i in kept) or scalar_type(typ) != scalar_type(
            input_types[0]
        ):
            raise UnsupportedJaxprError(path, "squeeze output shape or scalar type differs")
        # Fix removed lanes at zero while retaining the source expression and
        # its conditions; squeezing is a coordinate map, not a new scalar input.
        bindings = {f"local:{d}": integer(0) for d in dimensions}
        bindings.update({f"local:{d}": coordinate("local", i) for i, d in enumerate(kept)})
        return _substitute(args[0], bindings)
    if name == "concatenate":
        dimension = p["dimension"]
        rank = len(typ.shape)
        if (
            not args
            or len(args) != len(input_types)
            or type(dimension) is not int
            or not 0 <= dimension < rank
            or any(
                value.dtype != typ.dtype
                or len(value.shape) != rank
                or any(n != typ.shape[i] for i, n in enumerate(value.shape) if i != dimension)
                for value in input_types
            )
            or sum(shape[dimension] for shape in input_shapes) != typ.shape[dimension]
        ):
            raise UnsupportedJaxprError(path, "invalid static concatenate interface")
        position = coordinate("local", dimension)
        offset = 0
        pieces = []
        for value, shape in zip(args, input_shapes, strict=True):
            pieces.append(
                (
                    offset + shape[dimension],
                    _substitute(value, {f"local:{dimension}": add(position, integer(-offset))}),
                )
            )
            offset += shape[dimension]
        result = pieces[-1][1]
        for stop, value in reversed(pieces[:-1]):
            result = ValueExpr(
                "select_n",
                scalar_type(typ),
                (Predicate("lt", (position, integer(stop))), result, value),
            )
        return result
    if name == "slice":
        original = input_shapes[0]
        starts = cast(tuple[int, ...], p["start_indices"])
        limits = cast(tuple[int, ...], p["limit_indices"])
        strides = (
            (1,) * len(original)
            if p["strides"] in (None, ("none",))
            else cast(tuple[int, ...], p["strides"])
        )
        if not len(starts) == len(limits) == len(strides) == len(original) == len(typ.shape):
            raise UnsupportedJaxprError(path, "slice coordinate rank mismatch")
        if any(
            not 0 <= start <= limit <= extent
            or stride <= 0
            or size != (limit - start + stride - 1) // stride
            for start, limit, stride, extent, size in zip(
                starts, limits, strides, original, typ.shape, strict=True
            )
        ):
            raise UnsupportedJaxprError(path, "invalid static value slice")
        return _substitute(
            args[0],
            {
                f"local:{i}": add(integer(start), scale(coordinate("local", i), stride))
                for i, (start, stride) in enumerate(zip(starts, strides, strict=True))
            },
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
        if prod(shape) == 0:
            # No output lane can observe a coordinate of an empty view. Keep
            # the operand and its conditions, but use unreachable placeholders
            # instead of dividing/modding by an empty axis. Consumers retain
            # the actual empty shape and therefore never read these lanes.
            return _substitute(args[0], {f"local:{i}": integer(0) for i in range(len(input_shape))})
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
    # A tensor's bound coordinates only occur inside its operands. Install
    # their ranges on entry, in the same traversal that projects coordinates;
    # rescanning the entire DAG beforehand repeats work for every output.
    # Names include shapes, so these ranges cannot shadow one another even
    # when distinct tensor operations share a lexical depth.
    ranges = dict(ranges)

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
            if item.operation.startswith("tensor."):
                p = dict((a.name, a.value) for a in item.parameters)
                for names, shape in zip(
                    cast(tuple[tuple[str, ...], ...], p["__input_axes"]),
                    cast(tuple[tuple[int, ...], ...], p["__input_shapes"]),
                    strict=True,
                ):
                    ranges.update((key, (0, n - 1)) for key, n in zip(names, shape, strict=True))
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
