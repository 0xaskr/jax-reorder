"""Scalar/vector value expressions shared by local state and call bindings."""

from __future__ import annotations

from typing import cast

import numpy as np

from ..errors import UnsupportedJaxprError
from .index_domain import IndexExpr, axis, check_integer_range, integer, integer_operation
from .integer_ranges import integer_value_bounds as integer_value_bounds
from .memory import Expression, Predicate, ValueExpr, scalar_type, truth
from .model import ArrayType, OperationView
from .tensor_domain import POINTWISE_PRIMITIVES


def pure_value(
    op: OperationView,
    args: tuple[Expression, ...],
    ranges: dict[str, tuple[int, int]],
    coordinates: dict[str, IndexExpr],
) -> Expression:
    path, name = op.origin.path, op.primitive
    if len(op.results) != 1 or not isinstance(op.results[0].type, ArrayType):
        raise UnsupportedJaxprError(path, "value operation requires one array result")
    typ = op.results[0].type
    if len(typ.shape) > 1 or typ.sharding is None or any(typ.sharding.partitions):
        raise UnsupportedJaxprError(path, "value projection requires local scalars or vectors")
    p = {v.name: v.value for v in op.parameters}
    if name == "stop_gradient":
        if len(args) != 1 or p:
            raise UnsupportedJaxprError(path, "stop_gradient requires one unmodified value")
        return args[0]
    if name == "axis_index":
        key = str(p["axis_name"])
        if key not in coordinates:
            raise UnsupportedJaxprError(path, "missing device axis binding")
        return coordinates[key]
    if name in ("program_id", "num_programs"):
        if p["axis"] != 0 or name not in coordinates:
            raise UnsupportedJaxprError(path, "missing program coordinate binding")
        return coordinates[name]
    if name == "iota":
        if p["dimension"] != 0 or len(typ.shape) != 1 or p["shape"] != typ.shape:
            raise UnsupportedJaxprError(path, "iota requires one vector axis")
        value = axis("global")
        # A local iota can enumerate a smaller window than the surrounding block.
        check_integer_range(value, typ.dtype, {"global": (0, typ.shape[0] - 1)}, path)
        return value
    if name == "broadcast_in_dim":
        if p["broadcast_dimensions"] != () or len(args) != 1:
            raise UnsupportedJaxprError(path, "broadcast requires a scalar source")
        return args[0]
    if name not in POINTWISE_PRIMITIVES:
        raise UnsupportedJaxprError(path, "value operation requires a tensor coordinate rule")
    value: Expression
    if name in ("eq", "ne", "lt", "le", "gt", "ge") and all(isinstance(v, IndexExpr) for v in args):
        value = Predicate(name, cast(tuple[IndexExpr, ...], args))
        known = truth(value, ranges)
        return known if known is not None else value
    if name in ("eq", "ne") and all(type(v) is bool for v in args):
        # JAX's integer remainder helper compares the signs of two integers.
        # During finite replay these signs are concrete booleans; retaining a
        # symbolic comparison would obscure an otherwise static buffer slot.
        return args[0] == args[1] if name == "eq" else args[0] != args[1]
    if name in ("and", "or", "not") and all(isinstance(v, Predicate | bool) for v in args):
        value = Predicate(name, cast(tuple[Predicate | bool, ...], args))
        known = truth(value, ranges)
        return known if known is not None else value
    if name == "xor" and all(type(v) is bool for v in args):
        return args[0] != args[1]
    if name == "convert_element_type":
        if typ.dtype == "bool" and isinstance(args[0], bool | Predicate):
            return args[0]
        if typ.dtype == "bool" and isinstance(args[0], IndexExpr):
            value = Predicate("ne", (args[0], integer(0)))
            known = truth(value, ranges)
            return known if known is not None else value
        if type(args[0]) is bool and np.dtype(typ.dtype).kind in "iu":
            return integer(int(args[0]))
    if name == "select_n" and isinstance(args[0], IndexExpr | bool):
        selected = (
            int(args[0])
            if type(args[0]) is bool
            else (
                int(args[0].value)
                if isinstance(args[0], IndexExpr) and args[0].kind == "constant"
                else None
            )
        )
        if selected is not None and 0 <= selected < len(args) - 1:
            return args[selected + 1]
    if all(isinstance(v, IndexExpr) for v in args) and np.dtype(typ.dtype).kind in "iu":
        if name in ("max", "min", "clamp", "sign", "xor", "and", "or", "not"):
            if all(isinstance(v, IndexExpr) and v.kind == "constant" for v in args):
                numbers = tuple(int(cast(IndexExpr, v).value) for v in args)
                result = (
                    max(numbers)
                    if name == "max"
                    else min(numbers)
                    if name == "min"
                    else min(max(numbers[1], numbers[0]), numbers[2])
                    if name == "clamp"
                    else (numbers[0] > 0) - (numbers[0] < 0)
                    if name == "sign"
                    else numbers[0] ^ numbers[1]
                    if name == "xor"
                    else numbers[0] & numbers[1]
                    if name == "and"
                    else numbers[0] | numbers[1]
                    if name == "or"
                    else ~numbers[0]
                )
                value = integer(result)
                check_integer_range(value, typ.dtype, ranges, path)
                return value
            # These remain typed values. A finite lane gather may evaluate them
            # exhaustively; general grid/address consumers still reject them.
            return ValueExpr(name, scalar_type(typ), args, op.parameters)
        value = integer_operation(name, cast(tuple[IndexExpr, ...], args), ranges, path)
        check_integer_range(value, typ.dtype, ranges, path)
        return value
    return ValueExpr(name, scalar_type(typ), args, op.parameters)


def static_integer(value: Expression, path: str) -> int:
    if type(value) is bool:
        return int(value)
    if isinstance(value, IndexExpr) and value.kind == "constant":
        return int(value.value)
    raise UnsupportedJaxprError(path, "a static integer is required for this resource operation")
