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

from .budget import Budget
from .expression import rename_bounds


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

    def boolean_piecewise(expression: sp.Basic) -> sp.Basic:
        """把全部分支都是逻辑值的 Piecewise 转成 ITE，交给布尔比较。"""
        # SymPy 的 simplify 可把定义条件变成 Piecewise；它继承 Expr，
        # 但布尔分支不能相减。只把全部分支均为逻辑值的情况交给 ITE。
        if isinstance(expression, sp.Piecewise) and all(
            isinstance(branch, sp.logic.boolalg.Boolean) and not isinstance(branch, sp.Expr)
            for branch, _ in expression.args
        ):
            budget.spend(len(expression.args), path="algebra.sympy.logic")
            steps.append("Piecewise.rewrite(ITE)")
            return expression.rewrite(sp.ITE)
        return expression

    left, right = boolean_piecewise(left), boolean_piecewise(right)
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


def explicit_symbols(left: sp.Basic, right: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """把具体下标读取替换成两侧共享的独立入口符号，用于构造坐标级反例。

    抽象坐标下的 False 不构成合法输出反例；这里只接受常量下标且不越界的读取，
    并让左右两侧对应项共享同一个符号身份，从而把“存在某组输入元素使两侧不同”
    落实为可用 SymPy 判定的命题。含不透明函数或无界整数符号时返回 None。
    """
    from sympy.core.function import AppliedUndef

    if left.has(AppliedUndef) or right.has(AppliedUndef):
        # 不透明函数只能提供同余正结论，不能给出一般反例。
        return None
    if any(
        isinstance(symbol, sp.Symbol) and symbol.is_integer
        for symbol in left.free_symbols | right.free_symbols
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
        replacements[read] = sp.Symbol(f"entry_{position}", **assumptions)
    return left.xreplace(replacements), right.xreplace(replacements)
