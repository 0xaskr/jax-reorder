"""独立复核索引代数证据，不导入或调用多项式规范化器。

验证器逐项检查局部规则、完整乘法组合及给定绑定双射。
索引编号是否最小不影响证明正确性，因此不重复规范化搜索。
"""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from itertools import product

from jax._src import core

from .algebra_domains import apply_domain_witness, prefix_terms, ranges
from .algebra_indices import coordinates, leaves, map_factors, scalar_factor, used_axes
from .algebra_ir import AlgebraStep, Budget, Factor, Index, Term, ratio
from .algebra_lower import Lowerer
from .algebra_padding import simplify_prefix_factors
from .correspondence import GraphCorrespondence, verify_correspondence
from .errors import CorrespondenceError
from .unrooted import order_key


@dataclass(frozen=True)
class AlgebraSource:
    """原图到仅作结构适配的图的独立对应证据。"""

    graph: core.Jaxpr = field(compare=False, repr=False)
    correspondence: GraphCorrespondence
    roots: tuple[int, ...]


@dataclass(frozen=True)
class AlgebraProof:
    """双图证明；每个有序输出及保留区域均有根，所有局部步骤可复核。"""

    left: AlgebraSource | None
    right: AlgebraSource | None
    steps: tuple[AlgebraStep, ...]
    budget: int
    version: int = 2
    structural: GraphCorrespondence | None = None


def _require(condition, reason):
    """证据不合法统一抛出带位置的错误，调用方不得据此报告等价。"""
    if not condition:
        raise CorrespondenceError("algebra.proof", reason)


def _raw_rule(step, children, budget, key=order_key):
    """按局部规则的数学定义独立列出必须覆盖的项，不进行规范编号。"""
    expr = step.expression
    if expr.op == "constant":
        _require(not children, "constant has premises")
        return (Term(ratio(Fraction(*expr.data))),)
    if expr.op == "call":
        return (Term((1, 1), factors=(scalar_factor(expr.data, children, expr.total),)),)
    if expr.op in ("atom", "bag"):
        semantic, indices = expr.data
        args = tuple(sorted(children, key=key)) if expr.op == "bag" else children
        return (Term((1, 1), factors=(Factor(semantic, indices, args, expr.total),)),)
    if expr.op == "add":
        return tuple(term for terms in children for term in terms)
    if expr.op == "sum":
        _require(len(children) == 1, "sum must have one premise")
        uid, extent = expr.data
        _require(type(extent) is int and extent >= 0, "invalid finite sum range")
        terms = []
        for term in children[0]:
            size = len(term.bounds)
            factors = map_factors(
                term.factors,
                lambda i, size=size: Index("bound", size) if i == Index("sum", uid) else i,
                key=key,
            )
            terms.append(Term(term.coefficient, term.bounds + (extent,), tuple(factors)))
        return tuple(terms)
    _require(expr.op == "mul", "unknown proof rule")
    count = 1
    for terms in children:
        count *= len(terms)
        _require(count <= budget.limit, "multiplication proof exceeds budget")
    _require(count or expr.total, "multiplication erases a partial expression")
    budget.spend(count, "algebra.proof.distribution")
    result = []
    # 枚举每个子多项式的项索引；缺少一个交叉项或重复一次都会改变预期项表。
    for choices in product(*(range(len(terms)) for terms in children)):
        coefficient = Fraction(1)
        ranges, factors = [], []
        for child, choice in enumerate(choices):
            term = children[child][choice]
            coefficient *= Fraction(term.coefficient[0], term.coefficient[1])
            size = len(ranges)
            factors.extend(
                map_factors(
                    term.factors,
                    lambda i, size=size: (
                        Index("bound", i.number + size) if i.kind == "bound" and i.scope == 0 else i
                    ),
                    key=key,
                )
            )
            ranges.extend(term.bounds)
        result.append(Term(ratio(coefficient), tuple(ranges), tuple(factors)))
    return tuple(result)


def _check_term(term, order, key=order_key):
    """校验给出的绑定置换、单位轴替换及无用轴重数，不寻找最优置换。"""
    _require(type(order) is tuple and all(type(i) is int for i in order), "invalid index witness")
    _require(all(type(n) is int and n >= 0 for n in term.bounds), "invalid bound extent")
    coefficient = Fraction(*term.coefficient)
    if not coefficient or any(n == 0 for n in term.bounds):
        _require(all(f.total for f in term.factors), "zero erases a partial expression")
        _require(not order, "zero term contains a spurious mapping")
        return Term((0, 1))
    simplified = map_factors(
        term.factors,
        lambda i: (
            Index("zero")
            if i.kind == "bound" and i.scope == 0 and term.bounds[i.number] == 1
            else i
        ),
        ranges(term),
        key=key,
    )
    simplified = simplify_prefix_factors(simplified, key=key)
    used = used_axes(simplified)
    for coordinate in coordinates(simplified):
        for index in leaves(coordinate):
            if index.kind == "bound" and index.scope == 0:
                _require(0 <= index.number < len(term.bounds), "bound index is outside its scope")
    _require(len(order) == len(used) and set(order) == used, "index witness is not a bijection")
    for axis, extent in enumerate(term.bounds):
        if axis not in used:
            coefficient *= extent
    renaming = {old: new for new, old in enumerate(order)}
    factors = map_factors(
        simplified,
        lambda i: Index("bound", renaming[i.number]) if i.kind == "bound" and i.scope == 0 else i,
        key=key,
    )
    return Term(
        ratio(coefficient),
        tuple(term.bounds[i] for i in order),
        tuple(sorted(factors, key=key)),
    )


def _check_step(step, children, budget, key=order_key):
    """独立汇总经见证转换的每一项，精确核对结果及消项定义域。"""
    originals = _raw_rule(step, children, budget, key)
    _require(len(step.domains) == len(originals), "missing domain witnesses")
    raw = []
    for original, witnesses in zip(originals, step.domains, strict=True):
        terms = [original]
        for instruction in witnesses:
            kind, position, *detail = instruction
            _require(type(position) is int and 0 <= position < len(terms), "invalid domain branch")
            term = terms[position]
            budget.spend(1 + len(term.factors) + len(term.bounds), "algebra.proof.domain")
            if kind == "rewrite":
                _require(len(detail) == 1, "invalid domain rewrite")
                terms[position] = apply_domain_witness(term, detail[0], key=key)
            else:
                _require(kind == "prefix" and not detail, "invalid domain instruction")
                terms[position : position + 1] = prefix_terms(term, budget)
        _require(
            all(type(n) is int for term in terms for n in term.bounds), "unresolved domain interval"
        )
        raw.extend(terms)
    _require(len(step.orders) == len(raw), "proof does not cover all expanded terms")
    budget.spend(len(raw) + 1, "algebra.proof.steps")
    coefficients = {}
    for original, order in zip(raw, step.orders, strict=True):
        term = _check_term(original, order, key)
        identity = term.bounds, term.factors
        coefficients[identity] = coefficients.get(identity, Fraction(0)) + term.rational
    expected = []
    for (bounds, factors), value in coefficients.items():
        _require(value or all(f.total for f in factors), "cancellation erases a partial expression")
        if value:
            expected.append(Term(ratio(value), bounds, factors))
    _require(tuple(sorted(expected, key=key)) == step.result, "incorrect local algebra result")


def verify_algebraic_proof(left, right, proof):
    """从原始图逐项复核证明；成功返回 None，失败抛出 CorrespondenceError。

    语义转换不执行多项式展开或索引搜索。验证器不信任证据中附带图的来源，
    必须先通过完整结构对应，再核对转换后的每个表达式根和全部局部推导。
    """
    try:
        _require(isinstance(proof, AlgebraProof) and proof.version == 2, "unsupported proof format")
        budget = Budget(proof.budget)
        key = partial(order_key, cache={})
        if proof.structural is not None:
            _require(
                proof.left is None and proof.right is None and not proof.steps,
                "mixed structural and algebraic certificate",
            )
            verify_correspondence(left, right, proof.structural, derivative_scope="primal")
            core.check_jaxpr(left)
            core.check_jaxpr(right)
            return None
        signatures, expressions = [], []
        for original, source in ((left, proof.left), (right, proof.right)):
            verify_correspondence(
                original, source.graph, source.correspondence, derivative_scope="primal"
            )
            core.check_jaxpr(source.graph)
        from .algebra_calls import CommonCalls

        common_calls = CommonCalls(
            (
                (left, proof.left.graph, proof.left.correspondence),
                (right, proof.right.graph, proof.right.correspondence),
            ),
            budget,
        )
        for source in (proof.left, proof.right):
            lowered = Lowerer(budget, common_calls).graph(source.graph)
            signatures.append(lowered.signature)
            expressions.append(lowered.roots)
        _require(signatures[0] == signatures[1], "input or output abstract contract differs")
        _require(len(proof.steps) <= proof.budget, "too many proof steps")
        for number, step in enumerate(proof.steps):
            _require(isinstance(step, AlgebraStep), "invalid proof step")
            _require(
                all(type(i) is int and 0 <= i < number for i in step.premises),
                "invalid proof dependency",
            )
            _require(
                tuple(proof.steps[i].expression for i in step.premises) == step.expression.args,
                "premises do not match the source expression",
            )
            _check_step(step, tuple(proof.steps[i].result for i in step.premises), budget, key)
        results = []
        for source, roots in zip((proof.left, proof.right), expressions, strict=True):
            _require(
                all(type(i) is int and 0 <= i < len(proof.steps) for i in source.roots),
                "invalid output proof root",
            )
            _require(
                tuple(proof.steps[i].expression for i in source.roots) == roots,
                "proof roots do not match the original graph",
            )
            results.append(tuple(proof.steps[i].result for i in source.roots))
        _require(results[0] == results[1], "ordered outputs or regions do not have equal proofs")
    except CorrespondenceError:
        raise
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RecursionError) as exc:
        raise CorrespondenceError("algebra.proof", f"invalid proof: {exc}") from exc
