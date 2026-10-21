"""Exhaustive, budgeted proof of a static lane-only gather address function."""

from typing import cast

import numpy as np

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, IndexExpr, integer
from .memory import Expression, Predicate, ValueExpr, _substitute
from .model import ArrayType, OperationView
from .tensor_domain import coordinate


def lane_gather(
    op: OperationView,
    args: tuple[Expression, ...],
    input_types: tuple[ArrayType, ...],
    budget: AnalysisBudget,
) -> Expression:
    path = op.origin.path
    typ = cast(ArrayType, op.results[0].type)
    source, indices = input_types
    rank = len(source.shape)
    p = {v.name: v.value for v in op.parameters}
    batch = tuple(range(rank - 1))
    if (
        not rank
        or p["dimension_numbers"] != ((), (rank - 1,), (rank - 1,), batch, batch)
        or p["slice_sizes"] != (1,) * rank
        or p["mode"] != "PROMISE_IN_BOUNDS"
        or p["fill_value"] != ("none",)
        or p["unique_indices"] is not False
        or p["indices_are_sorted"] is not False
        or indices.shape != typ.shape + (1,)
        or len(typ.shape) != rank
        or source.shape[:-1] != typ.shape[:-1]
        or np.dtype(indices.dtype).kind not in "iu"
    ):
        raise UnsupportedJaxprError(path, "gather needs the reviewed static lane layout")
    root = _substitute(args[1], {f"local:{rank}": integer(0)})
    extent = typ.shape[-1]
    budget.spend(path + ".lanes", extent)

    def evaluate(value: Expression, lane: int, cache: dict[int, int | bool]) -> int | bool:
        if id(value) in cache:
            return cache[id(value)]
        budget.spend(path + ".lane_value")
        if type(value) is bool:
            return value
        if isinstance(value, IndexExpr):
            values = [evaluate(a, lane, cache) for a in value.args]
            if value.kind == "constant":
                result = int(value.value)
            elif value.kind == "axis" and value.value == f"local:{rank - 1}":
                result = lane
            elif value.kind == "sum":
                result = sum(values)
            elif value.kind == "scale":
                result = int(value.value) * values[0]
            elif value.kind == "div":
                result = values[0] // int(value.value)
            elif value.kind == "mod":
                result = values[0] % int(value.value)
            elif value.kind == "lookup":
                if not 0 <= values[0] < len(values) - 1:
                    raise UnsupportedJaxprError(path, "lane address outside its table")
                result = values[values[0] + 1]
            else:
                raise UnsupportedJaxprError(path, "gather index depends on a non-lane coordinate")
        elif isinstance(value, ValueExpr | Predicate):
            name = value.operation
            if isinstance(value, ValueExpr) and (
                np.dtype(value.type.dtype).kind not in "iub" or value.input is not None
            ):
                raise UnsupportedJaxprError(
                    path, "gather index must be a pure integer lane function"
                )
            values = [evaluate(a, lane, cache) for a in value.operands]
            if name == "literal" and isinstance(value, ValueExpr):
                result = int(np.frombuffer(value.data, dtype=value.type.dtype)[0])
            elif name in ("add", "sub", "mul", "div", "rem", "xor", "max", "min"):
                a, b = values
                if name in ("div", "rem") and b == 0:
                    raise UnsupportedJaxprError(path, "zero divisor in lane address")
                quotient = (
                    abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
                    if name in ("div", "rem")
                    else 0
                )
                result = {
                    "add": lambda: a + b,
                    "sub": lambda: a - b,
                    "mul": lambda: a * b,
                    "div": lambda: quotient,
                    "rem": lambda: a - quotient * b,
                    "xor": lambda: a ^ b,
                    "max": lambda: max(a, b),
                    "min": lambda: min(a, b),
                }[name]()
            elif name in ("eq", "ne", "lt", "le", "gt", "ge"):
                a, b = values
                result = {
                    "eq": a == b,
                    "ne": a != b,
                    "lt": a < b,
                    "le": a <= b,
                    "gt": a > b,
                    "ge": a >= b,
                }[name]
            elif name in ("and", "or", "not"):
                boolean = isinstance(value, Predicate) or np.dtype(value.type.dtype).kind == "b"
                if boolean:
                    result = (
                        all(values)
                        if name == "and"
                        else any(values)
                        if name == "or"
                        else not values[0]
                    )
                else:
                    result = (
                        values[0] & values[1]
                        if name == "and"
                        else values[0] | values[1]
                        if name == "or"
                        else ~values[0]
                    )
            elif name == "sign":
                result = (values[0] > 0) - (values[0] < 0)
            elif name == "neg":
                result = -values[0]
            elif name == "convert_element_type":
                assert isinstance(value, ValueExpr)
                result = (
                    bool(values[0]) if np.dtype(value.type.dtype).kind == "b" else int(values[0])
                )
            elif name == "select_n":
                if not 0 <= values[0] < len(values) - 1:
                    raise UnsupportedJaxprError(path, "invalid lane selector")
                result = values[values[0] + 1]
            else:
                raise UnsupportedJaxprError(path, f"unreviewed lane operation {name}")
            if isinstance(value, ValueExpr) and np.dtype(value.type.dtype).kind in "iu":
                limits = np.iinfo(value.type.dtype)
                if not limits.min <= result <= limits.max:
                    raise UnsupportedJaxprError(path, "integer lane arithmetic may overflow")
        else:
            raise UnsupportedJaxprError(path, "invalid lane expression")
        cache[id(value)] = result
        return result

    table = tuple(int(evaluate(root, i, {})) for i in range(extent))
    if any(not 0 <= i < source.shape[-1] for i in table):
        raise UnsupportedJaxprError(path, "lane gather violates its in-bounds promise")
    address = IndexExpr("lookup", (coordinate("local", rank - 1), *map(integer, table)))
    return _substitute(args[0], {f"local:{rank - 1}": address})
