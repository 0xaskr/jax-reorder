"""生成稀疏索引多项式及局部证明；搜索与证据复核由不同模块承担。"""

from fractions import Fraction
from functools import partial
from itertools import permutations, product
from math import factorial, prod

from .algebra_domains import apply_domain_witness, prefix_terms, ranges, translation
from .algebra_indices import coordinates, map_factors, scalar_factor, used_axes
from .algebra_ir import AlgebraStep, Budget, Expr, Factor, Index, Term, ratio
from .algebra_padding import simplify_prefix_factors
from .errors import UnsupportedJaxprError
from .unrooted import order_key


class PolynomialNormalizer:
    """按子表达式后序展开；保留证明节点，复用共享子表达式。"""

    def __init__(self, budget: Budget):
        """同一图对及嵌套区域共用预算。"""
        self.budget = budget
        self.steps = []
        self.cache = {}
        self._key = partial(order_key, cache={})

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
            orders, terms, domains = [], [], []
            for original in raw:
                converted, witnesses = self._domains(original)
                domains.append(witnesses)
                for term in converted:
                    normal, order = self._term(term)
                    terms.append(normal)
                    orders.append(order)
            result = self._collect(terms)
            self.budget.spend(path="algebra.proof")
            self.cache[expr] = len(self.steps)
            self.steps.append(AlgebraStep(expr, premises, result, tuple(orders), tuple(domains)))
        return self.cache[expression]

    def _expand(self, expr, children):
        """执行一个局部代数规则；先检查笛卡尔积规模再生成组合。"""
        if expr.op == "constant":
            self.budget.spend()
            return [Term(expr.data)]
        if expr.op == "call":
            self.budget.spend(path="algebra.scalar_call")
            return [Term((1, 1), factors=(scalar_factor(expr.data, children, expr.total),))]
        if expr.op in ("atom", "bag"):
            self.budget.spend()
            semantic, indices = expr.data
            if expr.op == "bag":
                children = tuple(sorted(children, key=self._key))
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
                    factors += map_factors(
                        term.factors,
                        lambda i, offset=offset: (
                            Index("bound", i.number + offset)
                            if i.kind == "bound" and i.scope == 0
                            else i
                        ),
                        key=self._key,
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
                    map_factors(
                        term.factors,
                        lambda i, bounds=term.bounds: (
                            Index("bound", len(bounds)) if i == Index("sum", uid) else i
                        ),
                        key=self._key,
                    ),
                )
                for term in children[0]
            ]
        raise UnsupportedJaxprError("algebra.expression", f"unsupported expression {expr.op}")

    def _domains(self, term):
        """在前缀拆分后继续规范化各个域，保留每次分支与双射见证。"""
        terms, witnesses, position = [term], [], 0
        while position < len(terms):
            converted, path = self._domain_path(terms[position])
            witnesses.extend(("rewrite", position, witness) for witness in path)
            if any(witness[0] == "shift" for witness in path):
                witnesses.append(("prefix", position))
                terms[position : position + 1] = converted
            else:
                terms[position] = converted[0]
                position += 1
        return terms, tuple(witnesses)

    def _domain_path(self, term):
        """选择域双射并记录见证；独立验证器只检查见证，不执行此搜索。"""
        witnesses = []

        def apply(witness):
            nonlocal term
            self.budget.spend(1 + len(term.factors) + len(term.bounds), "algebra.domain_bijection")
            term = apply_domain_witness(term, witness, key=self._key)
            witnesses.append(witness)

        def nodes(index):
            yield index
            for child in index.args:
                yield from nodes(child)

        # 混合进制坐标的完整矩形先合成一根轴；不依赖 reshape 的存在。
        while True:
            found = None
            for coordinate in coordinates(term.factors):
                for node in nodes(coordinate):
                    numerator = node.args[0] if node.kind in ("div", "mod") else node
                    if numerator.kind != "add" or len(numerator.args) != 2:
                        continue
                    for major, minor in (numerator.args, numerator.args[::-1]):
                        if major.kind != "scale" or minor.kind != "bound" or minor.scope != 0:
                            continue
                        axis = major.args[0]
                        if (
                            axis.kind == "bound"
                            and axis.scope == 0
                            and axis.number != minor.number
                            and major.number == term.bounds[minor.number]
                            and major.number > 1
                            and term.bounds[axis.number] > 1
                        ):
                            found = ("fuse", axis.number, minor.number)
                            break
                    if found:
                        break
                if found:
                    break
            if found is None:
                break
            apply(found)
        # 商/余数读取按其进制拆域；每次拆分都严格缩小被拆轴。
        while True:
            found = None
            for coordinate in coordinates(term.factors):
                for node in nodes(coordinate):
                    if node.kind not in ("div", "mod"):
                        continue
                    axis = node.args[0]
                    if axis.kind != "bound" or axis.scope != 0:
                        continue
                    extent = term.bounds[axis.number]
                    if 1 < node.number < extent and extent % node.number == 0:
                        found = ("split", axis.number, node.number)
                        break
                if found:
                    break
            if found is None:
                break
            apply(found)
        for axis in range(len(term.bounds)):
            offset = translation(term, axis)
            if offset and all(f.total for f in term.factors):
                apply(("shift", axis, offset))
        return prefix_terms(term, self.budget), tuple(witnesses)

    def _term(self, term):
        """移除单位/无用轴后，精确搜索同色绑定轴的最小序列化。"""
        total = all(f.total for f in term.factors)
        if not term.rational or 0 in term.bounds:
            if not total:
                raise UnsupportedJaxprError("algebra.domain", "cannot erase a partial expression")
            return Term((0, 1)), ()
        factors = map_factors(
            term.factors,
            lambda i: (
                Index("zero")
                if i.kind == "bound" and i.scope == 0 and term.bounds[i.number] == 1
                else i
            ),
            key=self._key,
        )
        factors = map_factors(factors, lambda i: i, ranges(term), key=self._key)
        factors = simplify_prefix_factors(factors, key=self._key)
        used = used_axes(factors)
        coefficient = term.rational
        for i, extent in enumerate(term.bounds):
            if i not in used:
                coefficient *= extent
                self.budget.coefficient(coefficient)
        groups = {}
        for i in sorted(used):
            # 以完整标量调用树着色，外层绑定编号不泄漏进颜色。
            colored = map_factors(
                factors,
                lambda j, axis=i: (
                    Index("target" if j.number == axis else "color", term.bounds[j.number])
                    if j.kind == "bound" and j.scope == 0
                    else j
                ),
                key=self._key,
            )
            color = (term.bounds[i], self._key(tuple(sorted(colored, key=self._key))))
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
            remapped = map_factors(
                factors,
                lambda i, renaming=renaming: (
                    Index("bound", renaming[i.number]) if i.kind == "bound" and i.scope == 0 else i
                ),
                key=self._key,
            )
            candidate = Term(
                ratio(coefficient),
                tuple(term.bounds[i] for i in order),
                tuple(sorted(remapped, key=self._key)),
            )
            key = self._key(candidate)
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
        return tuple(sorted(output, key=self._key))
