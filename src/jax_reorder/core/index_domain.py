"""Exact bounded integer expressions and symbolic blocked-domain certificates.

No source objects, element enumeration, SymPy, or floating-point arithmetic are
used here. Unsupported relations fail instead of falling back to enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ComparisonBudgetError, UnsupportedJaxprError
from .model import ArrayType, BlockMappingView, LiteralValue, Operand, RegionView, ValueRef


@dataclass
class AnalysisBudget:
    limit: int = 10000
    used: int = 0

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit <= 0:
            raise ValueError("analysis budget must be a positive integer")

    def spend(self, path: str, count: int = 1) -> None:
        self.used += count
        if self.used > self.limit:
            raise ComparisonBudgetError(
                path, "semantic analysis budget exhausted", used=self.used, limit=self.limit
            )


@dataclass(frozen=True)
class IndexExpr:
    kind: str
    args: tuple[IndexExpr, ...] = ()
    value: int | str = 0


def integer(value: int) -> IndexExpr:
    return IndexExpr("constant", value=int(value))


def axis(name: str) -> IndexExpr:
    return IndexExpr("axis", value=name)


def scale(value: IndexExpr, factor: int) -> IndexExpr:
    if factor == 0:
        return integer(0)
    if factor == 1:
        return value
    if value.kind == "constant":
        return integer(int(value.value) * factor)
    if value.kind == "scale":
        return scale(value.args[0], int(value.value) * factor)
    if value.kind == "sum":
        return add(*(scale(v, factor) for v in value.args))
    return IndexExpr("scale", (value,), factor)


def add(*values: IndexExpr) -> IndexExpr:
    constant = 0
    terms: dict[IndexExpr, int] = {}
    pending = list(values)
    while pending:
        value = pending.pop()
        if value.kind == "sum":
            pending.extend(value.args)
        elif value.kind == "constant":
            constant += int(value.value)
        elif value.kind == "scale":
            terms[value.args[0]] = terms.get(value.args[0], 0) + int(value.value)
        else:
            terms[value] = terms.get(value, 0) + 1
    # q * B + r = i, where q=floor(i/B), r=i mod B. The divisor is
    # positive by construction. This identity is over mathematical integers.
    for term, coefficient in tuple(terms.items()):
        if term.kind != "mod" or not coefficient:
            continue
        quotient = divide(term.args[0], int(term.value))
        if terms.get(quotient) == coefficient * int(term.value):
            terms[term] = terms[quotient] = 0
            terms[term.args[0]] = terms.get(term.args[0], 0) + coefficient
    result = [scale(term, coef) for term, coef in terms.items() if coef]
    if constant:
        result.append(integer(constant))
    if not result:
        return integer(0)
    if len(result) == 1:
        return result[0]
    return IndexExpr("sum", tuple(sorted(result, key=repr)))


def divide(value: IndexExpr, divisor: int) -> IndexExpr:
    if divisor <= 0:
        raise ValueError("index divisor must be positive")
    if value.kind == "constant":
        return integer(int(value.value) // divisor)
    if divisor == 1:
        return value
    return IndexExpr("div", (value,), divisor)


def modulo(value: IndexExpr, divisor: int) -> IndexExpr:
    if divisor <= 0:
        raise ValueError("index divisor must be positive")
    if value.kind == "constant":
        return integer(int(value.value) % divisor)
    if divisor == 1:
        return integer(0)
    return IndexExpr("mod", (value,), divisor)


def substitute(expr: IndexExpr, bindings: dict[str, IndexExpr]) -> IndexExpr:
    if expr.kind == "axis":
        return bindings.get(str(expr.value), expr)
    args = tuple(substitute(a, bindings) for a in expr.args)
    if expr.kind == "sum":
        return add(*args)
    if expr.kind == "scale":
        return scale(args[0], int(expr.value))
    if expr.kind == "div":
        return divide(args[0], int(expr.value))
    if expr.kind == "mod":
        return modulo(args[0], int(expr.value))
    return expr


def bounds(expr: IndexExpr, ranges: dict[str, tuple[int, int]]) -> tuple[int, int]:
    if expr.kind == "constant":
        return int(expr.value), int(expr.value)
    if expr.kind == "axis":
        return ranges[str(expr.value)]
    limits = tuple(bounds(a, ranges) for a in expr.args)
    if expr.kind == "sum":
        return sum(x[0] for x in limits), sum(x[1] for x in limits)
    lo, hi = limits[0]
    k = int(expr.value)
    if expr.kind == "scale":
        return min(lo * k, hi * k), max(lo * k, hi * k)
    if expr.kind == "div":
        return lo // k, hi // k
    if expr.kind == "mod":
        return (lo % k, hi % k) if lo // k == hi // k else (0, k - 1)
    raise ValueError(f"invalid index expression: {expr.kind}")


def check_integer_range(
    expr: IndexExpr, dtype: str, ranges: dict[str, tuple[int, int]], path: str
) -> None:
    if np.dtype(dtype).kind not in "iu":
        raise UnsupportedJaxprError(path, "index expression must have integer dtype")
    limit = np.iinfo(dtype)
    lo, hi = bounds(expr, ranges)
    if lo < limit.min or hi > limit.max:
        raise UnsupportedJaxprError(
            path, "integer index arithmetic may overflow", lower=lo, upper=hi, dtype=dtype
        )


def literal_integer(value: LiteralValue, path: str) -> IndexExpr:
    if value.type.shape or np.dtype(value.type.dtype).kind not in "iu":
        raise UnsupportedJaxprError(path, "index literal must be an integer scalar")
    return integer(int(np.frombuffer(value.data, dtype=value.type.dtype)[0]))


def integer_operation(
    name: str, args: tuple[IndexExpr, ...], ranges: dict[str, tuple[int, int]], path: str
) -> IndexExpr:
    if name == "add":
        return add(*args)
    if name == "sub":
        return add(args[0], scale(args[1], -1))
    if name == "neg":
        return scale(args[0], -1)
    if name == "mul":
        if args[0].kind == "constant":
            return scale(args[1], int(args[0].value))
        if args[1].kind == "constant":
            return scale(args[0], int(args[1].value))
    if name in ("div", "rem") and args[1].kind == "constant" and int(args[1].value) > 0:
        if bounds(args[0], ranges)[0] >= 0:
            return (divide if name == "div" else modulo)(args[0], int(args[1].value))
    if name == "convert_element_type":
        return args[0]  # Caller proves representability in the destination dtype.
    raise UnsupportedJaxprError(path, f"unsupported integer index operation {name}")


def evaluate_index_map(region: RegionView, programs: int, budget: AnalysisBudget) -> IndexExpr:
    path = region.origin.path
    if len(region.inputs) != 1 or len(region.outputs) != 1 or region.effects:
        raise UnsupportedJaxprError(path, "expected a scalar pure index map")
    env = {region.inputs[0].id: axis("program")}
    ranges = {"program": (0, programs - 1)}
    typ = region.inputs[0].type
    if not isinstance(typ, ArrayType):
        raise UnsupportedJaxprError(path, "index map Ref input")
    check_integer_range(axis("program"), typ.dtype, ranges, path)
    for var, value in region.constants:
        env[var.id] = literal_integer(value, path)

    def read(operand: Operand) -> IndexExpr:
        return env[operand.id] if isinstance(operand, ValueRef) else literal_integer(operand, path)

    for op in region.operations:
        budget.spend(op.origin.path)
        if len(op.results) != 1 or op.access is not None or op.call is not None or op.regions:
            raise UnsupportedJaxprError(op.origin.path, "unsupported index map operation")
        result = op.results[0]
        if not isinstance(result.type, ArrayType) or result.type.shape:
            raise UnsupportedJaxprError(op.origin.path, "index map must remain scalar")
        expr = integer_operation(
            op.primitive, tuple(read(v) for v in op.operands), ranges, op.origin.path
        )
        check_integer_range(expr, result.type.dtype, ranges, op.origin.path)
        env[result.id] = expr
    return read(region.outputs[0])


@dataclass(frozen=True)
class BlockDomain:
    size: int
    block: int
    programs: int
    mapping: IndexExpr
    reverse: bool

    @property
    def address(self) -> IndexExpr:
        return add(scale(self.mapping, self.block), axis("lane"))

    @property
    def inverse(self) -> tuple[IndexExpr, IndexExpr]:
        group = divide(axis("global"), self.block)
        program = add(integer(self.programs - 1), scale(group, -1)) if self.reverse else group
        return program, modulo(axis("global"), self.block)


def prove_block_domain(
    block: BlockMappingView, programs: int, budget: AnalysisBudget
) -> BlockDomain:
    """Prove a blocked permutation and its inverse, in constant space in N."""
    budget.spend(block.origin.path)
    size, width = block.array.shape[0], block.block_shape[0]
    if programs != (size + width - 1) // width:
        raise UnsupportedJaxprError(
            block.origin.path, "grid leaves a coverage gap or redundant programs"
        )
    mapping = evaluate_index_map(block.index_map, programs, budget)
    forward = axis("program")
    reverse = add(integer(programs - 1), scale(forward, -1))
    if mapping not in (forward, reverse):
        raise UnsupportedJaxprError(
            block.origin.path, "blocked mapping is not a proven forward/reverse bijection"
        )
    result = BlockDomain(size, width, programs, mapping, mapping == reverse and mapping != forward)
    program, lane = result.inverse
    if substitute(result.address, {"program": program, "lane": lane}) != axis("global"):
        raise UnsupportedJaxprError(block.origin.path, "blocked inverse identity was not proved")
    return result
