"""符号值模型：坐标、精确常量、绑定重命名和标量公式。

本模块只依赖 SymPy 和 NumPy。symbolic、regions、trusted 共用这些基础类型，
放在单独文件里是为了避免它们互相导入成环。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from jax.typing import DTypeLike
from numpy.typing import ArrayLike
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair

from ..errors import MathematicalUndecidedError
from ._typing import ObjectArray, Shape

if TYPE_CHECKING:
    from .finiteness import Source


def coordinates(shape: Sequence[int]) -> tuple[sp.Expr, ...]:
    """按输出 shape 生成自由坐标符号；单位轴固定为 0。

    同名整数符号全图共享身份，广播和单位轴读取直接写 0，不额外引入绑定。
    """
    return tuple(
        sp.S.Zero if size == 1 else sp.Symbol(f"_i{axis}", integer=True, nonnegative=True)
        for axis, size in enumerate(shape)
    )


def _exact_array(value: ArrayLike) -> ObjectArray:
    """把常量数组逐元素转成精确 SymPy 值，保留 shape 与完整内容。

    浮点字面量按其 Python 二进制值转成有理数，不做十进制“取整”解释；
    NaN/±Inf 不在这里处理，由调用方显式拒绝或按特殊值保留。
    """

    def exact_scalar(x: object) -> sp.Basic:
        """按 Python 标量类型分派；布尔必须先于整数判断（bool 是 int 子类）。"""
        x = np.asarray(x).item()
        if isinstance(x, (bool, np.bool_)):
            return sp.true if x else sp.false
        if isinstance(x, (int, np.integer)):
            return sp.Integer(int(x))
        if isinstance(x, (float, np.floating)) and np.isfinite(x):
            return sp.Rational(float(x))
        # 其余标量（如 -inf 或扩展浮点）交给 sympify；是否有实数解释由调用方决定。
        return sp.sympify(x)

    return np.vectorize(exact_scalar, otypes=[object])(np.asarray(value))


def _finite_contraction[T: (sp.Sum, sp.Product)](
    kind: type[T], function: sp.Basic, *limits: Sequence[sp.Basic | int]
) -> T:
    """Build a definite scalar contraction without eager Piecewise distribution.

    SymPy's public contraction constructor folds every nested Piecewise on every
    coordinate substitution. The exact three-argument limits and scalar body
    are already supplied by lowering; preserving this same Sum node postpones
    that optional algebraic transformation until the budgeted query stage.
    """
    if not isinstance(function, sp.Expr) or any(
        len(limit) != 3 or not isinstance(limit[0], sp.Symbol) for limit in limits
    ):
        return kind(function, *limits)
    if isinstance(function, kind):
        limits = (*function.limits, *limits)
        function = function.function
    result = sp.Expr.__new__(kind, function, *(sp.Tuple(*limit) for limit in limits))
    result.is_commutative = function.is_commutative
    return result


def finite_sum(function: sp.Basic, *limits: Sequence[sp.Basic | int]) -> sp.Sum:
    return _finite_contraction(sp.Sum, function, *limits)


def finite_product(function: sp.Basic, *limits: Sequence[sp.Basic | int]) -> sp.Product:
    return _finite_contraction(sp.Product, function, *limits)


def dag_atoms(expression: sp.Basic, *types: type[sp.Basic]) -> set[sp.Basic]:
    """Return exactly the requested node types, visiting shared nodes once."""
    found, seen, pending = set(), set(), [expression]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if isinstance(node, types):
            found.add(node)
        pending.extend(node.args)
    return found


def indicator(condition: sp.Basic) -> sp.Basic:
    """Exact 0/1 indicator without distributing its nested Boolean domain."""
    if condition in (sp.true, sp.false):
        return sp.S.One if condition == sp.true else sp.S.Zero
    return sp.Expr.__new__(
        sp.Piecewise,
        sp.Basic.__new__(ExprCondPair, sp.S.One, condition),
        sp.Basic.__new__(ExprCondPair, sp.S.Zero, sp.true),
    )


def _rebuild(expression: sp.Basic, args: tuple[sp.Basic, ...]) -> sp.Basic:
    if args == expression.args:
        return expression
    if expression.func == sp.exp:
        # A coordinate substitution is exact and needs no transcendental
        # simplification. exp.eval otherwise recursively asks sign/integrality
        # of every contraction inside a large softmax at every renamed read.
        return sp.exp(*args, evaluate=False)
    if isinstance(expression, sp.Sum):
        return finite_sum(args[0], *args[1:])
    if isinstance(expression, sp.Product):
        return finite_product(args[0], *args[1:])
    if isinstance(expression, (sp.Mul, sp.Pow)) and any(
        arg.has(sp.Sum, sp.Product) for arg in args
    ):
        return expression.func(*args, evaluate=False)
    if isinstance(expression, ExprCondPair) and args[1].has(sp.Sum, sp.Product):
        return sp.Basic.__new__(ExprCondPair, *args)
    if isinstance(expression, sp.Piecewise) and any(
        pair.cond.has(sp.Sum, sp.Product) or pair.expr.has(sp.Piecewise) for pair in args
    ):
        # Coordinate substitution needs branch selection, not Boolean normal
        # form conversion of every nested page-selection branch.
        pairs = []
        for pair in args:
            if pair.cond == sp.false:
                continue
            if pair.cond == sp.true and not pairs:
                return pair.expr
            pairs.append(pair)
            if pair.cond == sp.true:
                break
        return sp.Expr.__new__(sp.Piecewise, *pairs) if pairs else sp.nan
    if isinstance(expression, Relational) and any(arg.has(sp.Sum, sp.Product) for arg in args):
        # Alpha-renaming a universal domain must not trigger assumptions that
        # recursively distribute every Piecewise inside its contractions.
        return expression.func(*args, evaluate=False)
    return expression.func(*args)


def substitute(expression: sp.Basic, replacements: Mapping[sp.Basic, sp.Basic]) -> sp.Basic:
    """Exact simultaneous replacement with shared DAG nodes and lazy exp rebuilds."""
    cache: dict[sp.Basic, sp.Basic] = {}

    def visit(node: sp.Basic) -> sp.Basic:
        if node in replacements:
            return replacements[node]
        if node not in cache:
            cache[node] = _rebuild(node, tuple(visit(arg) for arg in node.args))
        return cache[node]

    return visit(expression)


def rename_bounds(expression: sp.Basic, *, fresh: bool = False) -> sp.Basic:
    """按词法作用域重命名 Sum/Product 的绑定变量，防止嵌套求和捕获外层读取。

    SymPy 1.14 的 as_dummy 在嵌套求和里会复用 `_0` 这类名字，把外层引用误绑到
    内层。构造期用新 Dummy 保证唯一；查询键用带深度和位置的固定名字，使同一表达
    式得到可复现的规范文本，不依赖 Dummy 的全局计数。缓存键包含当前绑定环境：
    同一节点在不同环境下不能共用结果。
    """
    cache = {}

    def visit(node: sp.Basic, environment: Mapping[sp.Basic, sp.Basic], depth: int) -> sp.Basic:
        """在给定绑定环境下递归改写；环境只解释被绑定的符号，不改变其他节点。"""
        if node in environment:
            return environment[node]
        if not node.args:
            return node
        key = (node, tuple(environment.items()), depth)
        if key in cache:
            return cache[key]
        if isinstance(node, (sp.Sum, sp.Product)):
            bound = dict(environment)
            limits = []
            for position, (variable, start, stop) in enumerate(node.limits):
                # fresh 用于实例化，每次读取生成互不冲突的 Dummy；非 fresh 用于
                # 查询键，名字由深度和位置决定，便于稳定比较。
                symbol = (
                    sp.Dummy("bound", integer=True)
                    if fresh
                    else sp.Symbol(f"_bound_{depth}_{position}", integer=True)
                )
                bound[variable] = symbol
                # 上下限仍在外层环境求值，只有被积函数进入新的绑定层。
                limits.append(
                    (symbol, visit(start, environment, depth), visit(stop, environment, depth))
                )
            function = visit(node.function, bound, depth + 1)
            result = (
                finite_sum(function, *limits)
                if isinstance(node, sp.Sum)
                else finite_product(function, *limits)
            )
        else:
            result = _rebuild(node, tuple(visit(arg, environment, depth) for arg in node.args))
        cache[key] = result
        return result

    return visit(expression, {}, 0)


def same_bound_expression(left: sp.Basic, right: sp.Basic) -> bool:
    """Check exact alpha identity without reconstructing large scalar formulas.

    This is only a sufficient identity check. In particular, no algebraic
    rewrite, extrema comparison or condition simplification is performed.
    """
    identities: dict[object, int] = {}
    cache: dict[tuple[sp.Basic, tuple[tuple[sp.Basic, int], ...], int], int] = {}

    def intern(key: object) -> int:
        if key not in identities:
            identities[key] = len(identities)
        return identities[key]

    def visit(node: sp.Basic, environment: dict[sp.Basic, int], depth: int) -> int:
        if node in environment:
            return environment[node]
        if not node.args:
            return intern(("atom", node))
        key = node, tuple(environment.items()), depth
        if key not in cache:
            if isinstance(node, (sp.Sum, sp.Product)):
                bound = dict(environment)
                limits = []
                for position, (variable, start, stop) in enumerate(node.limits):
                    bound[variable] = intern(("bound", depth, position))
                    limits.append(
                        (visit(start, environment, depth), visit(stop, environment, depth))
                    )
                value = node.func, visit(node.function, bound, depth + 1), tuple(limits)
            else:
                children = tuple(visit(arg, environment, depth) for arg in node.args)
                if isinstance(node, (sp.Add, sp.Mul, sp.And, sp.Or, sp.Max, sp.Min)):
                    children = tuple(sorted(children))
                value = node.func, children
            cache[key] = intern(value)
        return cache[key]

    return left == right or visit(left, {}, 0) == visit(right, {}, 0)


def _boolean_piecewise(expression: sp.Piecewise) -> sp.Basic:
    """Preserve ordered Boolean branches without numeric Piecewise consumers.

    Piecewise construction already charges its branches to the builder budget.
    This pass adds a linear number of shared ITE nodes, never expands a truth
    table, and balances them to avoid making a flat constant array recursive.
    An explicit default is required, so no uncovered position becomes defined.
    The tensor's original shape, definedness and memory validity are unchanged.
    """
    converted: dict[sp.Basic, sp.Basic] = {}
    pending = [(expression, False)]
    while pending:
        node, ready = pending.pop()
        if node in converted:
            continue
        if not ready:
            if node.args[-1][1] != sp.true:
                raise MathematicalUndecidedError(
                    "algebra.conditions", "Boolean Piecewise coverage is not established"
                )
            pending.append((node, True))
            pending.extend(
                (child, False)
                for pair in node.args
                for child in pair.args
                if isinstance(child, sp.Piecewise) and child not in converted
            )
            continue
        choices = []
        for value, guard in node.args:
            value = converted.get(value, value)
            # Region assembly can use numeric zero as an unreachable Boolean
            # placeholder. The tensor dtype supplies the interpretation.
            value = sp.false if value == sp.S.Zero else sp.true if value == sp.S.One else value
            choices.append((converted.get(guard, guard), value))
        while len(choices) > 1:
            merged = []
            for index in range(0, len(choices), 2):
                left_guard, left_value = choices[index]
                if index + 1 == len(choices):
                    merged.append((left_guard, left_value))
                    continue
                right_guard, right_value = choices[index + 1]
                merged.append(
                    (
                        sp.ITE(left_guard, sp.true, right_guard, evaluate=False),
                        sp.ITE(left_guard, left_value, right_value, evaluate=False),
                    )
                )
            choices = merged
        guard, value = choices[0]
        converted[node] = sp.ITE(guard, value, sp.false, evaluate=False)
    return converted[expression]


@dataclass(frozen=True)
class SymbolicTensor:
    """一个输出端口的元素公式及附加事实。

    expression 在自由坐标上取值；defined 是原始定义条件，表示该元素何时有实数
    解释，不是化简后的域；finite 表示在保留条件下可作有限实数解释，与
    source.finite 的“无需附加条件的有限”事实分开使用。source 保存源图节点，
    供有限性分析和查询键使用。
    """

    shape: Shape
    dtype: DTypeLike
    expression: sp.Basic
    defined: sp.Basic = sp.true
    finite: bool = True
    source: Source | None = None
    # Pallas allocated-but-uninitialized lanes are a total-output safety
    # obligation, separate from mathematical partial-function definedness.
    memory_validity: sp.Basic = sp.true

    def __post_init__(self) -> None:
        if np.dtype(self.dtype).kind == "b" and isinstance(self.expression, sp.Piecewise):
            object.__setattr__(self, "expression", _boolean_piecewise(self.expression))

    def read(self, indices: Sequence[sp.Expr | int]) -> tuple[sp.Basic, sp.Basic]:
        """用具体或符号坐标实例化元素公式，返回 (expression, defined)。

        替换键或值与内部 Sum/Product 的绑定重名时先换新 Dummy，避免捕获。
        不相交时复用原绑定；单位轴坐标是常量 0，不出现在替换表中。
        """
        replacements = dict(zip(coordinates(self.shape), map(sp.sympify, indices), strict=True))
        replacements.pop(sp.S.Zero, None)
        symbols = set(replacements)
        for value in replacements.values():
            symbols.update(value.free_symbols)

        def instantiate(expression: sp.Basic) -> sp.Basic:
            if any(
                variable in symbols
                for node in dag_atoms(expression, sp.Sum, sp.Product)
                for variable, _, _ in node.limits
            ):
                expression = rename_bounds(expression, fresh=True)
            return substitute(expression, replacements)

        return instantiate(self.expression), instantiate(self.defined)
