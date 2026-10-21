"""受预算约束的 SymPy 化简和恒等查询。

这里放的是纯表达式操作，不依赖 SymbolicBuilder 或 worker：展开成本预估、
化简、ArrayExpr 兼容性判断、恒等比较、定义条件规范化，以及把坐标级反例
符号化。trusted 的 worker 协议和 symbolic 的构造器共用这些函数，避免两者
互相导入。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb, prod

import sympy as sp

from ..errors import MathematicalUndecidedError
from .budget import Budget
from .expression import rename_bounds
from .extrema import distinct_extrema, maximum, minimum


def finite_expansion(expression: sp.Basic, budget: Budget) -> bool:
    """在调用会隐式展开的 SymPy 之前，预估显式 Sum/Product 的展开成本。

    只接受上下限为精确非负整数的显式有限和；估算超过 256 项时返回 False，
    调用方保留惰性形式或判定未决，而不是让 SymPy 无预算地展开。数值 shape
    小不代表展开小，因此这里按项数而不是按张量大小计费。
    """
    costs = {}

    def cost(node: sp.Basic) -> int:
        """递归估算节点的展开项数；缓存避免共享子图重复计算。"""
        if node in costs:
            return costs[node]
        if isinstance(node, (sp.Sum, sp.Product)):
            size = 1
            for variable, start, stop in node.limits:
                extent = stop - start + 1
                if not extent.is_Integer or extent < 0:
                    return 257
                if isinstance(node, sp.Sum) and variable not in node.function.free_symbols:
                    # 与求和变量无关的行求和相当于一次标量乘法，即使声明轴很大。
                    continue
                size *= int(extent)
                if size > 256:
                    return 257
            count = size * max(1, cost(node.function))
        else:
            count = 0
            for child in node.args:
                count += cost(child)
                if count > 256:
                    break
        costs[node] = count
        return count

    count = cost(expression)
    # 257 是“已超过上限”的哨兵值；此时不扣预算，直接交回调用方。
    if count > 256:
        return False
    budget.spend(count, path="algebra.sympy.expand")
    return True


def compact_sums(expression: sp.Basic, budget: Budget) -> sp.Basic:
    """Combine overlapping finite intervals without adding any summand points.

    Only rational linear combinations of identical one-variable summands qualify.
    Cuts are exact integer endpoints; cancellation occurs before finite expansion.
    """

    def combine(node: sp.Basic) -> sp.Basic:
        groups: dict[sp.Basic, list[tuple[sp.Rational, int, int]]] = {}
        other = []
        budget.spend(len(node.args), path="algebra.sympy.sum_intervals")
        for term in node.args:
            coefficient, summation = term.as_coeff_Mul()
            if (
                not coefficient.is_Rational
                or not isinstance(summation, sp.Sum)
                or len(summation.limits) != 1
            ):
                other.append(term)
                continue
            variable, start, stop = summation.limits[0]
            if not start.is_Integer or not stop.is_Integer or stop < start:
                other.append(term)
                continue
            key = sp.Sum(summation.function, (variable, 0, 0))
            groups.setdefault(key, []).append((coefficient, int(start), int(stop) + 1))
        for key, intervals in groups.items():
            changes: dict[int, sp.Basic] = {}
            budget.spend(2 * len(intervals), path="algebra.sympy.sum_intervals")
            for coefficient, start, stop in intervals:
                changes[start] = changes.get(start, sp.S.Zero) + coefficient
                changes[stop] = changes.get(stop, sp.S.Zero) - coefficient
            cuts = sorted(changes)
            coefficient = sp.S.Zero
            variable = key.limits[0][0]
            for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
                coefficient += changes[start]
                if coefficient:
                    value = (
                        key.function.xreplace({variable: sp.Integer(start)})
                        if stop == start + 1
                        else sp.Sum(key.function, (variable, start, stop - 1))
                    )
                    other.append(coefficient * value)
        return sp.Add(*other)

    return rename_bounds(expression).replace(lambda node: node.is_Add, combine)


def simplify(
    expression: sp.Basic, budget: Budget, *, assumptions: sp.Basic = sp.true, expand: bool = True
) -> sp.Basic:
    """预算受控的 SymPy 化简入口，供转换和查询共用。

    顺序固定：先按组合上界预扣预算，再决定是否 expand；随后只对预算内的显式
    有限和调用 doit，最后做 Piecewise 规范化和假设下的 refine。默认
    simplify(doit=False) 防止 SymPy 隐式展开绕过上面的预算。
    """
    from sympy.core.function import AppliedUndef

    budget.spend(path="algebra.sympy.simplify")
    # 在 SymPy 真正展开幂/乘积之前，按组合上界扣减预算。
    terms = {}
    for node in reversed(list(sp.preorder_traversal(expression))):
        count = (
            sum(terms.get(arg, 1) for arg in node.args)
            if node.is_Add
            else prod(terms.get(arg, 1) for arg in node.args)
            if node.is_Mul
            else 1
        )
        if node.is_Pow and node.exp.is_Integer and node.exp > 0 and node.base.is_Add:
            # 多项式展开项数由二项式系数给出；幂次本身超过预算时直接拒绝。
            exponent = int(node.exp)
            if exponent > budget.limit:
                budget.spend(budget.limit + 1, path="algebra.sympy.expand")
            count = comb(len(node.base.args) + exponent - 1, exponent)
        if count > budget.limit - budget.used:
            budget.spend(count, path="algebra.sympy.expand")
        terms[node] = count
    expression = rename_bounds(
        expression.expand() if expand and isinstance(expression, sp.Expr) else expression
    )
    if expression.has(sp.Sum):
        expression = compact_sums(expression, budget)
    if expression.has(AppliedUndef) and expression.has(sp.Max, sp.Min):
        # 含不透明函数的离散索引条件缺少实数不等式模型，保持原样。
        return expression
    if expand and expression.has(sp.Sum, sp.Product) and finite_expansion(expression, budget):
        expression = expression.doit()
    if expression.has(sp.Piecewise, sp.Max, sp.Min, sp.Abs):
        # 统一分段形式，使两侧的谓词分支能按同一结构比较。
        expression = sp.piecewise_exclusive(
            sp.piecewise_fold(expression.rewrite(sp.Piecewise)), deep=True
        )
        if expression.has(sp.Sum):
            expression = compact_sums(expression, budget)
    if assumptions != sp.true:
        expression = sp.refine(expression, assumptions)
    # 上面的显式有限展开已经计费；simplify 默认的 doit 会绕过该限制，并让等价
    # 的求和以不一致的方式规范化。
    return sp.simplify(expression, doit=False)


def array_compatible(expression: sp.Basic, indices: Sequence[sp.Expr]) -> bool:
    """判断表达式能否安全走 SymPy ArrayExpr 的索引到数组转换。

    convert_indexed_to_array 把 x[i+1]、x[2*i] 的坐标整体当作轴标签，只有纯
    张量收缩（完整自由轴、完整范围求和、标签本身就是轴符号）才保持语义。
    切片、偏移、缺轴或坐标算术返回 False，继续保留 Indexed 表示。
    """
    bound = {}
    for summation in expression.atoms(sp.Sum):
        for variable, start, stop in summation.limits:
            if not isinstance(variable, sp.Symbol) or start != 0 or not stop.is_Integer:
                return False
            if variable in bound and bound[variable] != stop + 1:
                return False
            bound[variable] = stop + 1
    labels = set(indices) | bound.keys()
    reads = expression.atoms(sp.Indexed)
    # 缺少某个自由轴会把行广播和列广播混成同一个数组，因此标签集合必须精确相等。
    if {index for read in reads for index in read.indices} != labels:
        return False

    def supported(node: sp.Basic) -> bool:
        """只允许标签读取、完整求和和加减乘组合；其他结构一律保留 Indexed。"""
        if isinstance(node, sp.Indexed):
            return all(
                isinstance(index, sp.Symbol)
                and index in labels
                and (index not in bound or bound[index] == extent)
                for index, extent in zip(node.indices, node.base.shape, strict=True)
            )
        if isinstance(node, sp.Sum):
            return supported(node.function)
        if node.is_Add or node.is_Mul:
            return all(supported(arg) for arg in node.args)
        return node.is_Number or isinstance(node, sp.Symbol) and node not in labels

    return supported(expression)


def boolean_piecewise(
    expression: sp.Basic, budget: Budget, methods: list[str] | None = None
) -> sp.Basic:
    """把逻辑值分支的 Piecewise 转成 ITE，允许它参与布尔组合和比较。"""
    # SymPy 的 simplify 可把定义条件变成 Piecewise；它继承 Expr，
    # 但布尔分支不能相减，也不能直接作为 And 的操作数。
    if isinstance(expression, sp.Piecewise) and all(
        isinstance(branch, sp.logic.boolalg.Boolean) and not isinstance(branch, sp.Expr)
        for branch, _ in expression.args
    ):
        budget.spend(len(expression.args), path="algebra.sympy.logic")
        if methods is not None:
            methods.append("Piecewise.rewrite(ITE)")
        try:
            return expression.rewrite(sp.ITE)
        except (NotImplementedError, ValueError) as error:
            # SymPy cannot always prove coverage of periodic or multivariate
            # guards. Never invent a missing default branch or label this CAS
            # limitation as an internal implementation failure.
            raise MathematicalUndecidedError(
                "algebra.conditions", "Boolean Piecewise coverage is not established"
            ) from error
    return expression


def equal(
    left: sp.Basic,
    right: sp.Basic,
    budget: Budget,
    *,
    indices: Sequence[sp.Expr] | None = None,
    assumptions: sp.Basic = sp.true,
    methods: list[str] | None = None,
) -> bool | None:
    """判断两个表达式在给定假设下是否恒等，返回 True/False/None。

    依次尝试：字面相同、布尔 Piecewise→ITE、ArrayExpr 数组比较、布尔逻辑、
    Sum/Product 分侧化简，最后比较差值是否为零。None 表示当前预算和证据下
    无法判定，调用方必须与 False 区分；methods 记录实际使用的步骤供审计。
    """
    from sympy.core.function import AppliedUndef

    steps = [] if methods is None else methods
    # 含 NaN/zoo 的表达式没有可用的实数恒等语义，不能据此给出正/负结论。
    if left.has(sp.nan, sp.zoo) or right.has(sp.nan, sp.zoo):
        steps.append("nonfinite_expression")
        return None
    if left == right:
        steps.append("identity")
        return True

    extrema = left.atoms(sp.Max, sp.Min) | right.atoms(sp.Max, sp.Min)
    if any(len(node.args) > 32 for node in extrema):
        # Congruence only: identical exact operand sets may be represented by a
        # shared real atom. A negative answer for independent atoms says nothing
        # about the actual extrema, so it must remain undecided.
        budget.spend(sum(len(node.args) for node in extrema), path="algebra.sympy.extrema")
        contexts = (sp.Derivative, sp.Integral, sp.Limit, sp.Subs, sp.Lambda)
        if left.has(*contexts) or right.has(*contexts):
            # These contexts transform or bind a variable-dependent term; a
            # constant atom does not preserve their semantics. They are not
            # generated by current JAX lowering, so keep the helper conservative.
            steps.append("non-pointwise extrema context")
            return None
        bound_variables = {
            limit[0]
            for expression in (left, right)
            for contraction in expression.atoms(sp.Sum, sp.Product)
            for limit in contraction.limits
        }
        if any(node.free_symbols & bound_variables for node in extrema):
            # A Max depending on a contraction index is a function of that
            # index, never a constant atom shared by different intervals.
            steps.append("bound-dependent extrema")
            return None
        if assumptions == sp.true and distinct_extrema(left, right):
            steps.append("exact rational extremum witness")
            return False
        canonical_atoms = {}
        atoms = {}
        for node in extrema:
            canonical = (maximum if isinstance(node, sp.Max) else minimum)(*node.args)
            if canonical not in canonical_atoms:
                canonical_atoms[canonical] = sp.Dummy("extremum", real=True)
            atoms[node] = canonical_atoms[canonical]
        steps.append("finite extrema congruence")
        answer = equal(
            left.xreplace(atoms),
            right.xreplace(atoms),
            budget,
            indices=indices,
            assumptions=assumptions,
            methods=steps,
        )
        return True if answer is True else None

    left, right = (boolean_piecewise(x, budget, steps) for x in (left, right))
    if (
        indices is not None
        and (left.has(sp.Sum, sp.Indexed) or right.has(sp.Sum, sp.Indexed))
        and array_compatible(left, indices)
        and array_compatible(right, indices)
    ):
        from sympy.tensor.array.expressions import convert_indexed_to_array

        budget.spend(path="algebra.sympy.array")
        steps.append("ArrayExpr.doit")
        try:
            a = convert_indexed_to_array(left, first_indices=indices).doit()
            b = convert_indexed_to_array(right, first_indices=indices).doit()
            if a == b:
                return True
        except (TypeError, ValueError, NotImplementedError, IndexError, AttributeError, KeyError):
            # 转换失败只说明该表示不适用，不影响后续标量查询；不把异常当结论。
            pass
    if isinstance(left, sp.logic.boolalg.Boolean) and not isinstance(left, sp.Expr):
        # 布尔条件含不透明函数时缺少实数不等式模型，只做逻辑等价化简。
        steps.append(
            "simplify_logic(Equivalent)"
            if left.has(AppliedUndef) or right.has(AppliedUndef)
            else "simplify(Equivalent)"
        )
        residual = (
            sp.simplify_logic(sp.Equivalent(left, right), deep=False)
            if left.has(AppliedUndef) or right.has(AppliedUndef)
            else simplify(sp.Equivalent(left, right), budget)
        )
        return True if residual == sp.true else False if residual == sp.false else None
    if (left.has(sp.Sum, sp.Product) or right.has(sp.Sum, sp.Product)) and not (
        left.has(sp.Piecewise, sp.Max, sp.Min, sp.Abs)
        or right.has(sp.Piecewise, sp.Max, sp.Min, sp.Abs)
    ):
        # 先分别化简两侧，保留 exp(Sum(...)) 这类符号求和恒等式；直接展开会得到
        # 难以合并的大 Product。含 Piecewise/Max/Min/Abs 的求和则必须先消项或
        # 有界展开，否则分支会膨胀，因此不走这条快捷路径。
        steps.append("simplify(each side, doit=False)")
        left = simplify(left, budget, assumptions=assumptions, expand=False)
        right = simplify(right, budget, assumptions=assumptions, expand=False)
        if left == right:
            return True
    steps.append("simplify(left-right)")
    residual = simplify(left - right, budget, assumptions=assumptions)
    if residual.has(sp.nan, sp.zoo):
        return None
    if residual.has(AppliedUndef):
        # 不透明残差只能支持同余正结论；泛化问题的非零残差不是源程序反例。
        steps.append("opaque_residual")
        return None
    if residual.has(sp.Sum, sp.Product) and not finite_expansion(residual, budget):
        # Expr.equals 可能内部调用 doit，绕过显式展开预算；此时保留未决，
        # 由调用方在没有其他方法判定时抛出分类异常。
        steps.append("unevaluated_sum")
        return None
    steps.append("Expr.equals(0)")
    return residual.equals(0)


def domain_condition(condition: sp.Basic, budget: Budget) -> sp.Basic:
    """规范化定义条件；true/false 直接返回，其余尽量减少可比较的形式差异。

    含不透明函数时没有实数不等式模型，只做浅层逻辑化简；节点过多时仅重命名
    绑定，避免在大条件上触发高成本化简。
    """
    if condition in (sp.true, sp.false):
        return condition
    from sympy.core.function import AppliedUndef

    if condition.has(AppliedUndef):
        if sp.count_ops(condition) > 30:
            return rename_bounds(condition)
        return rename_bounds(sp.simplify_logic(condition, deep=False))
    return simplify(condition, budget)


def explicit_symbols(
    left: sp.Basic, right: sp.Basic, *, methods: list[str] | None = None
) -> tuple[sp.Basic, sp.Basic] | None:
    """把具体下标读取替换成两侧共享的独立入口符号，用于构造坐标级反例。

    抽象坐标下的 False 不构成合法输出反例；这里只接受常量下标且不越界的读取，
    并让左右两侧对应项共享同一个符号身份，从而把“存在某组输入元素使两侧不同”
    落实为可用 SymPy 判定的命题。纯整数读取用可表示的 0/1 见证；含不透明函数
    或其他无界整数符号时返回 None。
    """
    from sympy.core.function import AppliedUndef

    if left.has(AppliedUndef) or right.has(AppliedUndef):
        # 不透明函数只能提供同余正结论，不能给出一般反例。
        return None
    integer_reads = all(isinstance(v, sp.Indexed) and v.is_integer for v in (left, right))
    if (
        any(
            isinstance(symbol, sp.Symbol) and symbol.is_integer
            for symbol in left.free_symbols | right.free_symbols
        )
        and not integer_reads
    ):
        # 无界 SymPy 整数可以取到源整数 dtype 之外的值；同余仍有用，但 False
        # 不能当作源程序的反例。
        return None
    reads = left.atoms(sp.Indexed) | right.atoms(sp.Indexed)
    replacements = {}
    for position, read in enumerate(sorted(reads, key=sp.default_sort_key)):
        # 抽象下标或越界读取不在源输出的合法坐标上，无法作为反例。
        if any(not index.is_Integer for index in read.indices):
            return None
        if any(
            not 0 <= index < extent
            for index, extent in zip(read.indices, read.base.shape, strict=True)
        ):
            return None
        assumptions = {"real": True} if read.is_real else {}
        if read.is_integer:
            assumptions = {"integer": True}
        # 两侧都只是入口整数元素读取时，最多涉及两个元素。使用各整数 dtype
        # 都能表示的 0/1 构造具体见证；不把这条规则推广到整数算术或不透明值。
        replacements[read] = (
            sp.Integer(position) if integer_reads else sp.Symbol(f"entry_{position}", **assumptions)
        )
    if integer_reads and methods is not None:
        methods.append(f"bounded integer read witness: {replacements}")
    return left.xreplace(replacements), right.xreplace(replacements)
