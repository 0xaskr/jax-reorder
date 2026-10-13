"""生成稀疏索引多项式及局部证明；搜索与证据复核由不同模块承担。"""

from dataclasses import replace
from fractions import Fraction
from itertools import permutations, product
from math import factorial, prod

from .algebra_ir import AlgebraStep, Budget, Expr, Factor, Index, Term, ratio
from .errors import UnsupportedJaxprError
from .unrooted import order_key


def _map_factors(factors, change):
    """只替换当前项读取的索引，调用参数内部的绑定保持独立。"""
    return tuple(replace(f, indices=tuple(change(i) for i in f.indices)) for f in factors)


class PolynomialNormalizer:
    """按子表达式后序展开；保留证明节点，复用共享子表达式。"""

    def __init__(self, budget: Budget):
        """同一图对及嵌套区域共用预算。"""
        self.budget = budget
        self.steps = []
        self.cache = {}

    def normalize(self, expression: Expr):
        """返回证明根编号；显式栈避免依赖链消耗 Python 递归栈。"""
        pending = [(False, expression)]
        while pending:
            ready, expr = pending.pop()
            if expr in self.cache:
                continue
            if not ready:
                self.budget.spend(path="algebra.expression")
                pending.append((True, expr))
                pending.extend((False, child) for child in reversed(expr.args))
                continue
            premises = tuple(self.cache[child] for child in expr.args)
            children = tuple(self.steps[i].result for i in premises)
            raw = self._expand(expr, children)
            orders, terms = [], []
            for term in raw:
                normal, order = self._term(term)
                terms.append(normal)
                orders.append(order)
            result = self._collect(terms)
            self.budget.spend(path="algebra.proof")
            self.cache[expr] = len(self.steps)
            self.steps.append(AlgebraStep(expr, premises, result, tuple(orders)))
        return self.cache[expression]

    def _expand(self, expr, children):
        """执行一个局部代数规则；先检查笛卡尔积规模再生成组合。"""
        if expr.op == "constant":
            self.budget.spend()
            return [Term(expr.data)]
        if expr.op in ("atom", "bag"):
            self.budget.spend()
            semantic, indices = expr.data
            if expr.op == "bag":
                children = tuple(sorted(children, key=order_key))
            return [Term((1, 1), factors=(Factor(semantic, indices, children, expr.total),))]
        if expr.op == "add":
            self.budget.spend(sum(map(len, children)), "algebra.add")
            return [term for child in children for term in child]
        if expr.op == "mul":
            if any(not child for child in children) and not expr.total:
                raise UnsupportedJaxprError("algebra.domain", "cannot erase a partial expression")
            self.budget.spend(prod(map(len, children)), "algebra.distribute")
            output = []
            for combination in product(*children):
                self.budget.spend(
                    sum(len(t.factors) + len(t.bounds) for t in combination), "algebra.product_size"
                )
                coefficient, bounds, factors = Fraction(1), (), ()
                for term in combination:
                    self.budget.coefficient(coefficient)
                    coefficient *= term.rational
                    offset = len(bounds)
                    factors += _map_factors(
                        term.factors,
                        lambda i, offset=offset: (
                            Index("bound", i.number + offset) if i.kind == "bound" else i
                        ),
                    )
                    bounds += term.bounds
                output.append(Term(ratio(coefficient), bounds, factors))
            return output
        if expr.op == "sum":
            self.budget.spend(
                sum(1 + len(t.factors) + len(t.bounds) for t in children[0]), "algebra.sum"
            )
            uid, extent = expr.data
            return [
                Term(
                    term.coefficient,
                    (*term.bounds, extent),
                    _map_factors(
                        term.factors,
                        lambda i, bounds=term.bounds: (
                            Index("bound", len(bounds)) if i == Index("sum", uid) else i
                        ),
                    ),
                )
                for term in children[0]
            ]
        raise UnsupportedJaxprError("algebra.expression", f"unsupported expression {expr.op}")

    def _term(self, term):
        """移除单位/无用轴后，精确搜索同色绑定轴的最小序列化。"""
        total = all(f.total for f in term.factors)
        if not term.rational or 0 in term.bounds:
            if not total:
                raise UnsupportedJaxprError("algebra.domain", "cannot erase a partial expression")
            return Term((0, 1)), ()
        factors = _map_factors(
            term.factors,
            lambda i: Index("zero") if i.kind == "bound" and term.bounds[i.number] == 1 else i,
        )
        used = {i.number for f in factors for i in f.indices if i.kind == "bound"}
        coefficient = term.rational
        for i, extent in enumerate(term.bounds):
            if i not in used:
                coefficient *= extent
                self.budget.coefficient(coefficient)
        groups = {}
        for i in sorted(used):
            # 每次读取的轴端口是固定锚点；其他绑定仅用范围着色，不能使用原编号。
            incidence = []
            for f in factors:
                for port, index in enumerate(f.indices):
                    if index == Index("bound", i):
                        colors = tuple(
                            ("bound", term.bounds[j.number])
                            if j.kind == "bound"
                            else (j.kind, j.number)
                            for j in f.indices
                        )
                        incidence.append(order_key((f.semantic, f.arguments, port, colors)))
            color = (term.bounds[i], tuple(sorted(incidence)))
            groups.setdefault(color, []).append(i)
        ordered_groups = [groups[key] for key in sorted(groups)]
        count = 1
        for group in ordered_groups:
            count *= factorial(len(group))
            if count > self.budget.limit - self.budget.used:
                self.budget.spend(count, "algebra.indices")
        self.budget.spend(count * (1 + len(factors) + len(used)), "algebra.indices")
        best, best_key, best_order = None, None, None
        for choices in product(*(permutations(group) for group in ordered_groups)):
            order = tuple(i for choice in choices for i in choice)
            renaming = {old: new for new, old in enumerate(order)}
            remapped = _map_factors(
                factors,
                lambda i, renaming=renaming: (
                    Index("bound", renaming[i.number]) if i.kind == "bound" else i
                ),
            )
            candidate = Term(
                ratio(coefficient),
                tuple(term.bounds[i] for i in order),
                tuple(sorted(remapped, key=order_key)),
            )
            key = order_key(candidate)
            if best_key is None or key < best_key:
                best, best_key, best_order = candidate, key, order
        return best, best_order

    def _collect(self, terms):
        """按完整项键合并系数；部分函数的消项必须拒绝。"""
        collected = {}
        for term in terms:
            key = term.bounds, term.factors
            value = collected.get(key, Fraction(0)) + term.rational
            self.budget.coefficient(value)
            collected[key] = value
        output = []
        for (bounds, factors), coefficient in collected.items():
            if coefficient:
                output.append(Term(ratio(coefficient), bounds, factors))
            elif any(not factor.total for factor in factors):
                raise UnsupportedJaxprError("algebra.domain", "cannot cancel a partial expression")
        return tuple(sorted(output, key=order_key))
