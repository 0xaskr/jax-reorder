"""受控 SymPy 候选生成；索引/函数原子保持完整身份，结果须独立复核。"""

from dataclasses import replace
from fractions import Fraction

from jax import lax

from .algebra_ir import ScalarRewrite, Term, ratio
from .algebra_scalar_proof import constant, exponential, nonzero, replace_factors, scalar, total
from .errors import UnsupportedJaxprError


class LinearBridge:
    """CAS 仅看到规范单项式的独立符号；不展开求和或重新解释索引。"""

    def __init__(self, polynomials, budget, key):
        size = sum(1 + len(term.factors) + len(term.bounds) for p in polynomials for term in p)
        budget.spend(1 + size * size, "algebra.sympy.candidate")
        for polynomial in polynomials:
            for term in polynomial:
                budget.coefficient(term.rational, "algebra.sympy.coefficient")
        import sympy

        self.sp = sympy
        self.budget, self.key = budget, key
        identities = sorted(
            {(t.bounds, t.factors) for p in polynomials for t in p if t.bounds or t.factors},
            key=key,
        )
        self.symbols = {
            identity: sympy.Dummy(f"term{i}", real=True) for i, identity in enumerate(identities)
        }
        self.identities = {symbol: identity for identity, symbol in self.symbols.items()}

    def encode(self, polynomial):
        sp = self.sp
        return sp.Add(
            *(
                sp.Rational(*term.coefficient)
                * self.symbols.get((term.bounds, term.factors), sp.S.One)
                for term in polynomial
            )
        )

    def decode(self, expression):
        result = []
        for atom, coefficient in expression.as_coefficients_dict().items():
            if not coefficient.is_Rational or (atom != 1 and atom not in self.identities):
                raise UnsupportedJaxprError(
                    "algebra.sympy.result", "candidate left the linear bridge"
                )
            self.budget.spend(path="algebra.sympy.result")
            value = Fraction(int(coefficient.p), int(coefficient.q))
            self.budget.coefficient(value, "algebra.sympy.coefficient")
            if value:
                bounds, factors = self.identities.get(atom, ((), ()))
                result.append(Term(ratio(value), bounds, factors))
        return tuple(sorted(result, key=self.key))

    def exponent(self, expression):
        if expression == 1:
            return ()
        if expression == self.sp.E:
            return self.decode(self.sp.S.One)
        if expression.func is not self.sp.exp:
            raise UnsupportedJaxprError(
                "algebra.sympy.result", "expected one exponential candidate"
            )
        return self.decode(expression.args[0])


def _single_exp(polynomial):
    if (
        len(polynomial) == 1
        and polynomial[0].coefficient == (1, 1)
        and not polynomial[0].bounds
        and len(polynomial[0].factors) == 1
        and exponential(polynomial[0].factors[0])
    ):
        return polynomial[0].factors[0]
    return None


def _division(term, budget, key):
    if term.coefficient != (1, 1) or term.bounds or len(term.factors) != 1:
        return None
    factor = term.factors[0]
    if not scalar(factor, lax.div_p, 2):
        return None
    numerator, denominator = factor.arguments
    if not total(numerator) or not total(denominator):
        return None
    divisor = constant(denominator)
    if divisor is not None and divisor:
        bridge = LinearBridge((numerator,), budget, key)
        result = bridge.sp.cancel(
            bridge.encode(numerator) / bridge.sp.Rational(divisor.numerator, divisor.denominator)
        )
        return "div_constant", bridge.decode(result), ()
    if numerator == denominator and nonzero(denominator, budget):
        bridge = LinearBridge((), budget, key)
        symbol = bridge.sp.Dummy("nonzero", real=True, nonzero=True)
        candidate = bridge.sp.cancel(
            bridge.sp.Mul(symbol, bridge.sp.Pow(symbol, -1, evaluate=False), evaluate=False)
        )
        return "div_self", bridge.decode(candidate), ()
    left, right = _single_exp(numerator), _single_exp(denominator)
    if left is None or right is None or left.semantic != right.semantic:
        return None
    if left.semantic[-1] != factor.semantic[-1]:
        return None
    bridge = LinearBridge((left.arguments[0], right.arguments[0]), budget, key)
    candidate = bridge.sp.powsimp(
        bridge.sp.exp(bridge.encode(left.arguments[0]))
        / bridge.sp.exp(bridge.encode(right.arguments[0])),
        combine="exp",
    )
    argument = bridge.exponent(candidate)
    result = (
        (Term((1, 1), factors=(replace(left, arguments=(argument,)),)),)
        if argument
        else (Term((1, 1)),)
    )
    return "div_exp", result, ()


def _candidate(term, budget, key):
    division = _division(term, budget, key)
    if division is not None:
        return division
    groups = {}
    for index, factor in enumerate(term.factors):
        if scalar(factor, lax.exp_p, 1) and not factor.total and total(factor.arguments[0]):
            budget.spend(1 + len(term.factors), "algebra.sympy.defined")
            factors = tuple(
                replace(item, total=True) if position == index else item
                for position, item in enumerate(term.factors)
            )
            return "exp_defined", (replace(term, factors=factors),), (index,)
        if not exponential(factor):
            continue
        if not factor.arguments[0]:
            bridge = LinearBridge((), budget, key)
            argument = bridge.exponent(bridge.sp.exp(0))
            return "exp_zero", replace_factors(term, (index,), argument), (index,)
        groups.setdefault(factor.semantic, []).append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        positions = tuple(indices)
        arguments = tuple(term.factors[index].arguments[0] for index in positions)
        bridge = LinearBridge(arguments, budget, key)
        candidate = bridge.sp.powsimp(
            bridge.sp.Mul(*(bridge.sp.exp(bridge.encode(argument)) for argument in arguments)),
            combine="exp",
        )
        return "exp_merge", replace_factors(term, positions, bridge.exponent(candidate)), positions
    return None


def rewrite(raw, budget, key):
    """只在白名单形状上调用 CAS；每步候选保留可独立验算的规则位置。"""
    terms, witnesses, position = list(raw), [], 0
    while position < len(terms):
        candidate = _candidate(terms[position], budget, key)
        if candidate is None:
            position += 1
            continue
        kind, result, data = candidate
        budget.spend(1 + len(result), "algebra.sympy.proof")
        witnesses.append(ScalarRewrite(kind, position, result, data))
        terms[position : position + 1] = result
    return terms, tuple(witnesses)
