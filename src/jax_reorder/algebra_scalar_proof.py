"""按有限实数恒等式重放标量证据；不导入 CAS 或规范化搜索器。"""

from dataclasses import replace
from fractions import Fraction

from jax import lax

from .algebra_ir import Factor, ScalarRewrite, Term, ratio
from .errors import CorrespondenceError
from .unrooted import order_key


def require(condition, reason):
    if not condition:
        raise CorrespondenceError("algebra.scalar_proof", reason)


def scalar(factor, primitive, arity):
    return (
        factor.scalar
        and not factor.indices
        and len(factor.semantic) == 4
        and factor.semantic[0] == "scalar"
        and factor.semantic[1] is primitive
        and len(factor.arguments) == arity
    )


def total(polynomial):
    return all(f.total for term in polynomial for f in term.factors)


def exponential(factor):
    return scalar(factor, lax.exp_p, 1) and factor.total and total(factor.arguments[0])


def constant(polynomial):
    if not polynomial:
        return Fraction(0)
    if len(polynomial) == 1 and not polynomial[0].bounds and not polynomial[0].factors:
        return polynomial[0].rational
    return None


def nonzero(polynomial, budget):
    """精确常数或同号正指数乘积的非空有限和；未知原子不获得符号假设。"""
    value = constant(polynomial)
    if value is not None:
        return value != 0
    signs = set()
    for term in polynomial:
        budget.spend(1 + len(term.bounds) + len(term.factors), "algebra.scalar_domain")
        if not term.rational or 0 in term.bounds:
            continue
        if any(type(size) is not int or size < 0 for size in term.bounds):
            return False
        if not all(exponential(factor) for factor in term.factors):
            return False
        signs.add(term.rational > 0)
    return len(signs) == 1


def combine(polynomials, weights, budget, key=order_key):
    """只合并相同完整索引单项式的有理系数，保持所有绑定与捕获作用域。"""
    coefficients = {}
    for polynomial, weight in zip(polynomials, weights, strict=True):
        for term in polynomial:
            budget.spend(1 + len(term.factors) + len(term.bounds), "algebra.scalar_argument")
            identity = term.bounds, term.factors
            value = coefficients.get(identity, Fraction(0)) + weight * term.rational
            budget.coefficient(value, "algebra.scalar_coefficient")
            coefficients[identity] = value
    return tuple(
        sorted(
            (Term(ratio(value), *identity) for identity, value in coefficients.items() if value),
            key=key,
        )
    )


def replace_factors(term, positions, argument):
    first = positions[0]
    selected = set(positions)
    replacement = (
        (Factor(term.factors[first].semantic, (), (argument,), True, True),) if argument else ()
    )
    factors = tuple(
        factor
        for index, item in enumerate(term.factors)
        for factor in (replacement if index == first else (() if index in selected else (item,)))
    )
    return (replace(term, factors=factors),)


def expected(term, witness, budget, key=order_key):
    """给定一个规则和位置后直接验算，不寻找改写、不调用候选生成器。"""
    kind = witness.kind
    if kind in {"exp_merge", "exp_zero", "exp_defined"}:
        positions = witness.data
        require(
            type(positions) is tuple
            and positions
            and len(positions) <= len(term.factors)
            and all(type(index) is int and 0 <= index < len(term.factors) for index in positions)
            and tuple(sorted(set(positions))) == positions,
            "invalid exponential factor positions",
        )
        factors = tuple(term.factors[index] for index in positions)
        if kind == "exp_defined":
            require(
                len(factors) == 1
                and scalar(factors[0], lax.exp_p, 1)
                and not factors[0].total
                and total(factors[0].arguments[0]),
                "exponential argument has no finite total proof",
            )
            budget.spend(1 + len(term.factors), "algebra.scalar_defined")
            return (
                replace(
                    term,
                    factors=tuple(
                        replace(factor, total=True) if index == positions[0] else factor
                        for index, factor in enumerate(term.factors)
                    ),
                ),
            )
        require(
            all(exponential(factor) for factor in factors), "exponential is not finite and total"
        )
        require(
            all(f.semantic == factors[0].semantic for f in factors), "exponential semantics differ"
        )
        if kind == "exp_zero":
            require(
                len(factors) == 1 and not factors[0].arguments[0],
                "exponential argument is not zero",
            )
            argument = ()
        else:
            require(len(factors) >= 2, "exponential merge needs two factors")
            argument = combine(
                tuple(f.arguments[0] for f in factors), (Fraction(1),) * len(factors), budget, key
            )
        return replace_factors(term, positions, argument)
    require(
        term.coefficient == (1, 1)
        and not term.bounds
        and len(term.factors) == 1
        and not witness.data,
        "division rule must target the original scalar call",
    )
    division = term.factors[0]
    require(scalar(division, lax.div_p, 2), "division rule targets a different primitive")
    numerator, denominator = division.arguments
    require(total(numerator) and total(denominator), "division has a partial argument")
    if kind == "div_constant":
        divisor = constant(denominator)
        require(divisor is not None and divisor != 0, "constant divisor must be nonzero")
        result = []
        for item in numerator:
            budget.spend(1 + len(item.factors) + len(item.bounds), "algebra.scalar_division")
            coefficient = item.rational / divisor
            budget.coefficient(coefficient, "algebra.scalar_coefficient")
            result.append(replace(item, coefficient=ratio(coefficient)))
        return tuple(sorted(result, key=key))
    if kind == "div_self":
        require(numerator == denominator and nonzero(denominator, budget), "missing nonzero proof")
        return (Term((1, 1)),)
    require(kind == "div_exp", "unknown scalar rewrite")
    factors = []
    for polynomial in (numerator, denominator):
        require(
            len(polynomial) == 1
            and polynomial[0].coefficient == (1, 1)
            and not polynomial[0].bounds
            and len(polynomial[0].factors) == 1,
            "division operands must be single exponentials",
        )
        factor = polynomial[0].factors[0]
        require(exponential(factor), "division exponential must be finite and total")
        factors.append(factor)
    require(factors[0].semantic == factors[1].semantic, "division exponential semantics differ")
    require(factors[0].semantic[-1] == division.semantic[-1], "division dtype differs")
    argument = combine(
        tuple(f.arguments[0] for f in factors), (Fraction(1), Fraction(-1)), budget, key
    )
    if not argument:
        return (Term((1, 1)),)
    return (Term((1, 1), factors=(replace(factors[0], arguments=(argument,)),)),)


def replay(raw, witnesses, budget, key=order_key):
    """逐项检查规则结果与索引位置；证据不能跳过前一步项表。"""
    require(type(witnesses) is tuple and len(witnesses) <= budget.limit, "invalid rewrite sequence")
    terms = list(raw)
    for witness in witnesses:
        require(isinstance(witness, ScalarRewrite), "invalid scalar rewrite certificate")
        require(
            type(witness.position) is int and 0 <= witness.position < len(terms),
            "invalid scalar rewrite position",
        )
        term = terms[witness.position]
        budget.spend(1 + len(term.factors) + len(term.bounds), "algebra.scalar_replay")
        result = expected(term, witness, budget, key)
        require(witness.result == result, "incorrect scalar rewrite result")
        terms[witness.position : witness.position + 1] = result
    return tuple(terms)
