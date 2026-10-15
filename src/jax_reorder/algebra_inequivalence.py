"""完整真实输出的受限实数多项式不恒等证据。

候选生成展开有限索引表达式，证据仅保存一个非零系数。独立复核器从原图
重建语义，再按指定单项式递归计算系数，不调用规范化器、CAS 或候选搜索。
输入元素是独立实变量；不把未解释函数、区域状态或特殊值当作独立变量。
"""

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from math import prod

import jax
import numpy as np
from jax._src import core

from .algebra_calls import CommonCalls
from .algebra_ir import Budget, Expr, Index, ratio
from .algebra_lower import Lowerer, encode
from .correspondence import GraphCorrespondence, verify_correspondence
from .errors import CorrespondenceError, UnsupportedJaxprError


@dataclass(frozen=True, order=True)
class InputElement:
    """按原始有序参数及合法坐标标识独立实变量。"""

    argument: int
    coordinate: tuple[int, ...]


@dataclass(frozen=True)
class InequivalenceSource:
    """只保存结构适配来源；不接受规范化器声称的输出项表。"""

    graph: core.Jaxpr = field(compare=False, repr=False)
    correspondence: GraphCorrespondence


@dataclass(frozen=True)
class PolynomialInequivalenceProof:
    """给定输出坐标的差多项式含有指定非零有理系数。"""

    left: InequivalenceSource
    right: InequivalenceSource
    output: int
    coordinate: tuple[int, ...]
    monomial: tuple[InputElement, ...]
    coefficient: tuple[int, int]
    budget: int
    version: int = 1


class _OutsidePolynomialDomain(Exception):
    """已知的片段边界；候选搜索保留 unknown。"""


def _require(condition, reason):
    if not condition:
        raise _OutsidePolynomialDomain(reason)


def _coordinate(coordinate, shape):
    return (
        type(coordinate) is tuple
        and len(coordinate) == len(shape)
        and all(type(i) is int and 0 <= i < n for i, n in zip(coordinate, shape, strict=True))
    )


def _input_shapes(graph):
    return tuple(tuple(v.aval.shape) for v in graph.invars)


def _coordinates(shape):
    """惰性遍历坐标；itertools.product 会先物化巨大 range，不适合预算停止。"""
    if not shape:
        yield ()
    else:
        for first in range(shape[0]):
            for rest in _coordinates(shape[1:]):
                yield (first, *rest)


def _argument(expression, graph):
    semantic, indices = expression.data
    _require(
        type(semantic) is tuple
        and len(semantic) == 3
        and semantic[0] == "input"
        and type(semantic[1]) is tuple
        and len(semantic[1]) == 2
        and semantic[1][0] == "argument",
        "polynomial variables must be original input elements",
    )
    position = semantic[1][1]
    _require(type(position) is int and 0 <= position < len(graph.invars), "invalid input slot")
    aval = graph.invars[position].aval
    _require(jax.dtypes.issubdtype(aval.dtype, np.floating), "input is outside real domain")
    _require(semantic[2] == encode(aval), "input metadata does not match source")
    _require(type(indices) is tuple and len(indices) == len(aval.shape), "input rank differs")
    return position, indices


def _validate(lowered, graph, budget):
    _require(not lowered.barriers, "regions cannot establish output inequivalence")
    pending = [output.body for output in lowered.outputs]
    seen = set()
    while pending:
        expression = pending.pop()
        if expression in seen:
            continue
        seen.add(expression)
        budget.spend(path="algebra.inequivalence.domain")
        _require(isinstance(expression, Expr) and expression.total, "partial or non-finite term")
        if expression.op == "constant":
            _require(not expression.args, "constant has children")
            _require(
                type(expression.data) is tuple
                and len(expression.data) == 2
                and all(type(n) is int for n in expression.data),
                "invalid rational constant",
            )
            Fraction(*expression.data)
        elif expression.op == "atom":
            _require(not expression.args, "input element has children")
            _argument(expression, graph)
        elif expression.op in ("add", "mul"):
            _require(not expression.data, "unexpected polynomial operator metadata")
        elif expression.op == "sum":
            _require(
                len(expression.args) == 1
                and len(expression.data) == 2
                and all(type(n) is int for n in expression.data)
                and expression.data[1] >= 0,
                "invalid finite reduction",
            )
        else:
            raise _OutsidePolynomialDomain("output contains an unsupported polynomial operation")
        pending.extend(expression.args)
    for output in lowered.outputs:
        _require(not output.nonfinite, "non-finite output")
        _require(jax.dtypes.issubdtype(output.dtype, np.floating), "output is outside real domain")


def _index(index, coordinate, bindings, budget):
    budget.spend(path="algebra.inequivalence.index")
    _require(isinstance(index, Index) and index.scope == 0, "unsupported index scope")
    if index.kind in ("zero", "const"):
        _require(not index.args and type(index.number) is int, "invalid constant index")
        return index.number
    if index.kind == "free":
        _require(not index.args and 0 <= index.number < len(coordinate), "invalid output axis")
        return coordinate[index.number]
    if index.kind == "sum":
        _require(not index.args and index.number in bindings, "unbound sum index")
        return bindings[index.number]
    values = tuple(_index(child, coordinate, bindings, budget) for child in index.args)
    if index.kind == "add":
        return sum(values)
    _require(len(values) == 1 and type(index.number) is int, "invalid index expression")
    if index.kind == "scale":
        return values[0] * index.number
    _require(index.kind in ("div", "mod") and index.number > 0, "unsupported index operation")
    return values[0] // index.number if index.kind == "div" else values[0] % index.number


def _element(expression, graph, coordinate, bindings, budget):
    position, indices = _argument(expression, graph)
    values = tuple(_index(i, coordinate, bindings, budget) for i in indices)
    _require(
        _coordinate(values, tuple(graph.invars[position].aval.shape)), "input read out of range"
    )
    return InputElement(position, values)


def _combine(left, right, budget):
    budget.spend(len(left) * len(right), "algebra.inequivalence.product")
    result = {}
    for first, a in left.items():
        for second, b in right.items():
            budget.spend(len(first) + len(second), "algebra.inequivalence.degree")
            budget.coefficient(a)
            budget.coefficient(b)
            monomial = tuple(sorted((*first, *second)))
            coefficient = result.get(monomial, Fraction(0)) + a * b
            budget.coefficient(coefficient)
            if coefficient:
                result[monomial] = coefficient
            else:
                result.pop(monomial, None)
    return result


def _polynomial(expression, graph, coordinate, budget):
    """搜索用稀疏展开；复核器不调用这个函数。"""
    cache = {}

    def visit(node, bindings):
        key = node, tuple(sorted(bindings.items()))
        if key in cache:
            return cache[key]
        budget.spend(path="algebra.inequivalence.expand")
        if node.op == "constant":
            value = Fraction(*node.data)
            result = {(): value} if value else {}
        elif node.op == "atom":
            result = {(_element(node, graph, coordinate, bindings, budget),): Fraction(1)}
        elif node.op == "mul":
            result = {(): Fraction(1)}
            for child in node.args:
                result = _combine(result, visit(child, bindings), budget)
        else:
            if node.op == "sum":
                uid, extent = node.data
                budget.spend(extent, "algebra.inequivalence.finite_sum")
                children = (visit(node.args[0], {**bindings, uid: i}) for i in range(extent))
            else:
                children = (visit(child, bindings) for child in node.args)
            result = {}
            for polynomial in children:
                budget.spend(len(polynomial), "algebra.inequivalence.collect")
                for monomial, value in polynomial.items():
                    coefficient = result.get(monomial, Fraction(0)) + value
                    budget.coefficient(coefficient)
                    if coefficient:
                        result[monomial] = coefficient
                    else:
                        result.pop(monomial, None)
        cache[key] = result
        return result

    return visit(expression, {})


def _coefficient(expression, target, graph, coordinate, budget):
    """按单项式系数定义复核，不展开完整多项式或寻找非零项。"""
    cache, products = {}, {}

    def splits(monomial):
        counts = sorted(Counter(monomial).items())
        budget.spend(prod(count + 1 for _, count in counts), "algebra.inequivalence.divisors")
        for powers in product(*(range(count + 1) for _, count in counts)):
            budget.spend(len(monomial), "algebra.inequivalence.divisor_size")
            left = tuple(
                item for (item, _), count in zip(counts, powers, strict=True) for _ in range(count)
            )
            right = tuple(
                item
                for (item, count), taken in zip(counts, powers, strict=True)
                for _ in range(count - taken)
            )
            yield left, right

    def multiply(children, monomial, bindings):
        key = children, monomial, tuple(sorted(bindings.items()))
        if key in products:
            return products[key]
        budget.spend(path="algebra.inequivalence.check_product")
        if not children:
            value = Fraction(not monomial)
        elif len(children) == 1:
            value = coefficient(children[0], monomial, bindings)
        else:
            value = Fraction(0)
            for first, second in splits(monomial):
                a = coefficient(children[0], first, bindings)
                if a:
                    b = multiply(children[1:], second, bindings)
                    budget.coefficient(a)
                    budget.coefficient(b)
                    value += a * b
                    budget.coefficient(value)
        products[key] = value
        return value

    def coefficient(node, monomial, bindings):
        key = node, monomial, tuple(sorted(bindings.items()))
        if key in cache:
            return cache[key]
        budget.spend(path="algebra.inequivalence.check_coefficient")
        if node.op == "constant":
            value = Fraction(0) if monomial else Fraction(*node.data)
        elif node.op == "atom":
            element = _element(node, graph, coordinate, bindings, budget)
            value = Fraction(monomial == (element,))
        elif node.op == "add":
            value = sum(
                (coefficient(child, monomial, bindings) for child in node.args), Fraction(0)
            )
        elif node.op == "mul":
            value = multiply(node.args, monomial, bindings)
        else:
            uid, extent = node.data
            budget.spend(extent, "algebra.inequivalence.check_sum")
            value = sum(
                (coefficient(node.args[0], monomial, {**bindings, uid: i}) for i in range(extent)),
                Fraction(0),
            )
        budget.coefficient(value)
        cache[key] = value
        return value

    return coefficient(expression, target, {})


def try_prove_inequivalent(left, right, *, sources, lowered, budget):
    """从已完整降低的同签名输出寻找证据；片段外返回 None，超预算抛错。"""
    try:
        _require(len(sources) == len(lowered) == 2, "expected two sources")
        _require(lowered[0].signature == lowered[1].signature, "output contracts differ")
        for source, value in zip(sources, lowered, strict=True):
            _validate(value, source.graph, budget)
        for output, (first, second) in enumerate(
            zip(lowered[0].outputs, lowered[1].outputs, strict=True)
        ):
            _require(first.shape == second.shape, "output shapes differ")
            if 0 in first.shape:
                continue
            for coordinate in _coordinates(first.shape):
                budget.spend(path="algebra.inequivalence.output_coordinate")
                a = _polynomial(first.body, sources[0].graph, coordinate, budget)
                b = _polynomial(second.body, sources[1].graph, coordinate, budget)
                budget.spend(len(a) + len(b), "algebra.inequivalence.candidate")
                for monomial in sorted(a.keys() | b.keys()):
                    coefficient = a.get(monomial, Fraction(0)) - b.get(monomial, Fraction(0))
                    if not coefficient:
                        continue
                    proof = PolynomialInequivalenceProof(
                        *(InequivalenceSource(s.graph, s.correspondence) for s in sources),
                        output,
                        coordinate,
                        monomial,
                        ratio(coefficient),
                        budget.limit,
                    )
                    verify_inequivalence_proof(left, right, proof)
                    return proof
    except _OutsidePolynomialDomain:
        return None
    return None


def verify_inequivalence_proof(left, right, proof):
    """从原图复核非零系数；无效证据统一抛 CorrespondenceError。"""
    try:
        _require(
            isinstance(proof, PolynomialInequivalenceProof)
            and type(proof.version) is int
            and proof.version == 1,
            "unsupported proof format",
        )
        budget = Budget(proof.budget)
        for original, source in ((left, proof.left), (right, proof.right)):
            _require(isinstance(source, InequivalenceSource), "invalid polynomial source")
            verify_correspondence(
                original, source.graph, source.correspondence, derivative_scope="primal"
            )
            core.check_jaxpr(source.graph)
        calls = CommonCalls(
            (
                (left, proof.left.graph, proof.left.correspondence),
                (right, proof.right.graph, proof.right.correspondence),
            ),
            budget,
        )
        lowered = tuple(Lowerer(budget, calls).graph(s.graph) for s in (proof.left, proof.right))
        _require(lowered[0].signature == lowered[1].signature, "input or output contracts differ")
        for source, value in zip((proof.left, proof.right), lowered, strict=True):
            _validate(value, source.graph, budget)
        _require(
            type(proof.output) is int and 0 <= proof.output < len(lowered[0].outputs),
            "invalid output slot",
        )
        shape = lowered[0].outputs[proof.output].shape
        _require(_coordinate(proof.coordinate, shape), "output coordinate is outside its domain")
        _require(type(proof.monomial) is tuple, "invalid monomial")
        budget.spend(len(proof.monomial), "algebra.inequivalence.proof_degree")
        _require(all(isinstance(v, InputElement) for v in proof.monomial), "invalid monomial")
        _require(tuple(sorted(proof.monomial)) == proof.monomial, "monomial must be ordered")
        shapes = _input_shapes(proof.left.graph)
        for variable in proof.monomial:
            _require(
                type(variable.argument) is int and 0 <= variable.argument < len(shapes),
                "invalid monomial input",
            )
            _require(
                _coordinate(variable.coordinate, shapes[variable.argument]),
                "monomial input coordinate is outside its domain",
            )
        _require(
            type(proof.coefficient) is tuple
            and len(proof.coefficient) == 2
            and all(type(n) is int for n in proof.coefficient),
            "invalid rational coefficient",
        )
        budget.spend(
            max(1, sum(n.bit_length() for n in proof.coefficient) // 64),
            "algebra.inequivalence.proof_coefficient",
        )
        claimed = Fraction(*proof.coefficient)
        _require(
            claimed and ratio(claimed) == proof.coefficient, "coefficient is zero or noncanonical"
        )
        coefficients = tuple(
            _coefficient(
                value.outputs[proof.output].body,
                proof.monomial,
                source.graph,
                proof.coordinate,
                budget,
            )
            for source, value in zip((proof.left, proof.right), lowered, strict=True)
        )
        _require(
            coefficients[0] - coefficients[1] == claimed, "incorrect output polynomial coefficient"
        )
    except CorrespondenceError:
        raise
    except (
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        KeyError,
        ZeroDivisionError,
        RecursionError,
        UnsupportedJaxprError,
        _OutsidePolynomialDomain,
    ) as exc:
        raise CorrespondenceError("algebra.inequivalence_proof", f"invalid proof: {exc}") from exc
