# Traversal and scalar mappings adapted from johnviljoen/jax2sympy,
# revision 2cb6a6568716b26af0abd542a941f87ae0580e25.
# MIT License
#
# Copyright (c) 2026 John Viljoen
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""把已通过结构校验的 JAX 图构造成带 shape 的 SymPy 表达式。

沿用 jax2sympy 的 SSA 遍历和逐点映射：每个输出端口表示为在自由坐标上取值
的 Indexed 或 Sum 表达式，同时保存原始定义条件、有限性事实和来源路径。
本模块只负责忠实构造和显式拒绝，不做数学判断；化简和恒等查询交给
algebra.trusted 里的 SymPy。未适配的原语、区域或 dtype 必须报错，不能为了
得出结论而丢掉语义。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from functools import reduce
from itertools import product
from math import prod
from operator import add, mul, neg, sub, truediv
from typing import Any, ClassVar, cast

import numpy as np
import sympy as sp
from jax import dtypes, lax
from jax.typing import DTypeLike
from numpy.typing import ArrayLike

from ..errors import UnsupportedJaxprError
from ._typing import ArrayAtom, ArrayAval, ArrayLiteral, Equation, Shape, SymbolicGraph
from .budget import Budget
from .expression import SymbolicTensor, _exact_array, coordinates, rename_bounds
from .extrema import maximum, minimum
from .finiteness import Finiteness, Source
from .indices import Index
from .regions import region
from .sympy_ops import domain_condition, simplify

type Limit = tuple[sp.Symbol, int, int]


def dtype_kind(dtype: DTypeLike) -> str:
    """把 dtype 归并成 "f"/"b"/"i"/"u" 等类别；所有浮点格式共用 "f"。"""
    return "f" if dtypes.issubdtype(dtype, np.floating) else np.dtype(dtype).kind


def normalize_coordinates(expression: sp.Basic, shape: Shape) -> sp.Basic:
    """用已知 shape 与求和上下限，把精确整数坐标适配到统一的索引规范形式。

    该函数服务读取实例化：把自由轴符号、求和绑定变量以及常量上的 floor/mod
    运算翻译成 algebra.indices 的标量索引，再回写到 Indexed 读取上。这样
    slice/reshape/广播得到的坐标与展开后的求和坐标可以比较；无法解释的坐标
    保持原样，不为了匹配而猜测改写。
    """
    from .indices import Index, add, divide, integer, scale

    symbols = {
        symbol: Index("free", axis) for axis, symbol in enumerate(coordinates(shape)) if symbol != 0
    }
    # Index 的域用半开区间 (start, stop)；Sum/Product 的上下限是闭区间。
    domains = {index: (0, shape[index.number]) for index in symbols.values()}
    for node in expression.atoms(sp.Sum, sp.Product):
        for symbol, start, stop in node.limits:
            if start.is_Integer and stop.is_Integer and symbol not in symbols:
                index = Index("sum", len(symbols))
                symbols[symbol] = index
                domains[index] = (int(start), int(stop) + 1)
    reverse = {value: key for key, value in symbols.items()}

    def parse(value: sp.Expr) -> Index:
        """把 SymPy 表达式转成标量索引；无法表达时抛 ValueError。"""
        if value in symbols:
            return symbols[value]
        if value.is_Integer:
            return integer(int(value))
        if value.is_Add:
            return add(*(parse(child) for child in value.args))
        if value.is_Mul:
            coefficient, rest = value.as_coeff_Mul()
            if coefficient.is_Integer:
                return scale(parse(rest), int(coefficient))
        if value.func == sp.Mod and value.args[1].is_Integer:
            return divide(parse(value.args[0]), int(value.args[1]), domains, modulo=True)
        if value.func == sp.floor:
            numerator, denominator = value.args[0].as_numer_denom()
            if denominator.is_Integer:
                return divide(parse(numerator), int(denominator), domains)
        raise ValueError("not a supported coordinate expression")

    def emit(index: Index) -> sp.Expr:
        """把标量索引还原成 SymPy 表达式；优先复用原符号，其余按索引种类展开。"""
        if index in reverse:
            return reverse[index]
        if index.kind in ("zero", "const"):
            return sp.Integer(index.number)
        if index.kind == "add":
            return sp.Add(*(emit(child) for child in index.args))
        if index.kind == "scale":
            return index.number * emit(index.args[0])
        if index.kind == "div":
            return sp.floor(emit(index.args[0]) / index.number)
        if index.kind == "mod":
            return sp.Mod(emit(index.args[0]), index.number)
        raise ValueError("unknown coordinate kind")

    replacements = {}
    for read in expression.atoms(sp.Indexed):
        try:
            replacements[read] = read.base[tuple(emit(parse(i)) for i in read.indices)]
        except ValueError:
            # 只替换能完整解释的读取；其余保留原始坐标，避免丢失映射。
            continue
    return expression.xreplace(replacements)


# 已核对语义的逐点原语到 SymPy 函数的映射；这里只建立数学表达式。
# 部分函数（除法、log、sqrt、反三角函数等）的定义条件由 _operation 另行补充。
SCALAR: dict[str, Callable[..., sp.Basic]] = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": truediv,
    "neg": neg,
    "pow": pow,
    "integer_pow": pow,
    "sign": sp.sign,
    "square": lambda a: a**2,
    "abs": sp.Abs,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "sqrt": sp.sqrt,
    "eq": sp.Eq,
    "ne": sp.Ne,
    "lt": sp.StrictLessThan,
    "le": sp.LessThan,
    "gt": sp.StrictGreaterThan,
    "ge": sp.GreaterThan,
    "not": sp.Not,
    "and": sp.And,
    "or": sp.Or,
    "max": maximum,
    "min": minimum,
    "logaddexp": lambda a, b: sp.log(sp.exp(a) + sp.exp(b)),
    "atan2": sp.atan2,
    "add_any": lambda *args: reduce(add, args),
}


class SymbolicBuilder:
    """把一份已校验的 jaxpr 构造成 SymbolicTensor 序列。

    构造器只维护 SSA 环境、定义条件、来源、有限性和资源计数；所有数学判断
    （恒等、化简、分支合并）都交给可信 SymPy。遇到无法完整表达的语义就抛出
    UnsupportedJaxprError，不用近似或丢弃条件换取转换成功。
    """

    pointwise_primitives: ClassVar[frozenset[str]] = frozenset(SCALAR)

    def __init__(self, budget: Budget | int = 10000) -> None:
        """budget 可以是 Budget 或正整数上限；同一构造器内共享扣减。"""
        self.budget = budget if isinstance(budget, Budget) else Budget(budget)
        # rules 记录实际使用的适配/规则名，作为后端审计信息，不参与判定。
        self.rules: set[str] = set()
        # 预留的常量缓存槽位；当前构造流程直接返回 SymbolicTensor，不读取该列表。
        self.constants: list[SymbolicTensor] = []
        # 归约绑定变量按次递增命名，保证同一表达式内互不冲突。
        self.next_bound = 0
        self.finiteness = Finiteness()
        # 不透明运算的稳定身份表；只有完整内容相同才共享同一个函数编号。
        self.definitions: dict[tuple[Any, ...], int] = {}
        # 规范化表达式缓存；键含 shape，只影响构造成本，不改变运算身份。
        self.canonical_values: dict[tuple[sp.Basic, Shape], sp.Basic] = {}

    def input(self, aval: ArrayAval, position: int) -> SymbolicTensor:
        """为第 position 个函数输入建立符号值。

        浮点输入取实数假设，整数输入取整数假设，布尔输入用 Eq(x, 1) 表示真值；
        输入按位置命名，不跟随原 Python 变量名。source 标记该值无条件有限。
        """
        kind = dtype_kind(aval.dtype)
        if kind not in "fbiu":
            raise UnsupportedJaxprError("algebra.dtype", "requires real, integer or Boolean inputs")
        assumptions = {"real": True} if kind == "f" else {"integer": True} if kind in "iu" else {}
        shape = tuple(aval.shape)
        symbol = sp.Symbol(f"x{position}", **assumptions)
        expression = (
            sp.IndexedBase(symbol, shape=shape, **assumptions)[coordinates(shape)]
            if shape
            else symbol
        )
        if kind == "b":
            expression = sp.Eq(expression, 1)
        return SymbolicTensor(
            shape, aval.dtype, expression, source=Source(None, shape, aval.dtype, finite=True)
        )

    def constant(self, value: ArrayLike, aval: ArrayAval) -> SymbolicTensor:
        """把一个字面量或捕获常量编码成精确表达式，并记录有限性。

        +Inf 与 NaN 没有当前实数表示，直接拒绝；-Inf 保留为 SymPy 的 -oo。
        常量数组按具体坐标展开成 Piecewise，而不是引入新的自由符号，因此相同
        常量的两次出现拥有同一身份，不同常量不会被误认。
        """
        array = np.asarray(value, dtype=aval.dtype)
        if dtype_kind(aval.dtype) not in "fbiu":
            raise UnsupportedJaxprError("algebra.constant", "non-real constants are unsupported")
        if dtype_kind(aval.dtype) == "f" and (
            np.any(np.isnan(array)) or np.any(np.isposinf(array))
        ):
            raise UnsupportedJaxprError(
                "algebra.constant", "positive infinity and NaN literals are not represented"
            )
        # 先按元素个数扣预算：常量内容本身也可能很大。
        self.budget.spend(max(1, array.size), path="algebra.constants")
        exact = _exact_array(array)
        if array.size == 0:
            # 空张量没有元素可读；表达式只作为占位。
            expression = sp.S.Zero
        elif array.size == 1 or np.all(array == array.flat[0]):
            # 标量或所有元素相同：用一个统一表达式表示，避免无意义的 Piecewise。
            expression = sp.sympify(exact.flat[0])
        else:
            # 具体常量保留字面值，不引入新的自由符号身份。
            indices = coordinates(array.shape)
            expression = sp.Piecewise(
                *[
                    (
                        exact[index],
                        sp.And(*(sp.Eq(i, j) for i, j in zip(indices, index, strict=True))),
                    )
                    for index in np.ndindex(array.shape)
                ]
            )
        finite = bool(np.all(np.isfinite(array)))
        if not finite:
            self.rules.add("negative_infinity_constant")
        source = Source(
            "constant",
            tuple(aval.shape),
            aval.dtype,
            finite=finite,
            constant=array.item() if array.shape == () else None,
        )
        return SymbolicTensor(
            tuple(aval.shape), aval.dtype, expression, finite=finite, source=source
        )

    def graph(
        self, graph: SymbolicGraph, inputs: Sequence[SymbolicTensor] | None = None
    ) -> list[SymbolicTensor]:
        """按 SSA 顺序转换一张（子）图并返回有序输出。

        未提供实参的 invar 变成位置符号输入，constvar 绑定常量；inputs 用于区域
        适配器把父图已构造的值传入子图。Literal 操作数不进入环境，读取时单独转成
        常量。未被输出使用的计算不会删除，而是在 finish 中保留其定义条件。
        """
        environment = {} if inputs is None else dict(zip(graph.invars, inputs, strict=True))
        for position, var in enumerate(graph.invars):
            if var not in environment:
                environment[var] = self.input(var.aval, position)
        for var, value in zip(graph.constvars, graph.consts, strict=True):
            environment[var] = self.constant(value, var.aval)

        def read(var: ArrayAtom) -> SymbolicTensor:
            """把操作数解析成 SymbolicTensor；字面量直接编码，变量查 SSA 环境。"""
            if hasattr(var, "val"):
                return self.constant(cast(ArrayLiteral, var).val, var.aval)
            return environment[var]

        for position, equation in enumerate(graph.eqns):
            # 每条方程单独计费，便于定位预算耗尽的具体位置。
            self.budget.spend(path=f"algebra.convert.eqns[{position}]")
            values = [read(var) for var in equation.invars]
            outputs = self.operation(equation, values, position)
            for var, output in zip(equation.outvars, outputs, strict=True):
                if tuple(var.aval.shape) != output.shape:
                    # 构造器自身形状错误属于适配器缺陷，按内部错误向上传递。
                    raise ValueError("symbolic output shape differs from the jaxpr")
                environment[var] = output

        # 未被使用的部分计算仍可能产生除零等定义条件，不能当作 DCE 删除。
        used = {var for var in graph.outvars if not hasattr(var, "val")}
        unused = []
        for equation in reversed(graph.eqns):
            if any(var in used for var in equation.outvars):
                used.update(var for var in equation.invars if not hasattr(var, "val"))
            else:
                unused.extend(environment[var] for var in equation.outvars)
        unused.extend(environment[var] for var in graph.invars if var not in used)
        return self.finish([read(var) for var in graph.outvars], unused)

    def finish(
        self, outputs: Sequence[SymbolicTensor], unused: Iterable[SymbolicTensor]
    ) -> list[SymbolicTensor]:
        """把未使用计算的定义条件合取到所有输出上，返回最终值。

        删除未使用计算会放宽原程序的定义域，从而可能把只在部分输入上有定义的
        程序误判为等价；因此这些条件必须并入每个输出。若整张图没有输出却存在
        部分计算，没有任何输出能承载该条件，只能明确拒绝。
        """

        condition = sp.And(*(value.defined for value in unused))
        if condition != sp.true:
            self.budget.spend(path="algebra.dead.conditions")
            condition = domain_condition(condition, self.budget)
        if not outputs and condition != sp.true:
            raise UnsupportedJaxprError(
                "algebra.dead", "unused partial computation with no outputs"
            )
        return [replace(value, defined=sp.And(value.defined, condition)) for value in outputs]

    def definition(
        self,
        operation: tuple[Any, ...],
        values: Sequence[SymbolicTensor],
        outputs: Sequence[ArrayAtom],
    ) -> tuple[int, tuple[sp.Symbol, ...]]:
        """为一次不透明运算登记稳定身份，并返回需要捕获的输入符号。

        身份由原语名、编码后的参数、各输入的 shape/dtype/规范化表达式/条件以及
        输出抽象值共同决定；只有完整内容相同才共享同一个函数编号。返回的
        captured 是表达式中来自外层的整数/Dummy 符号，之后会改名成显式参数，
        避免把张量容器或循环绑定误当成普通实数参数。
        """
        from sympy.core.function import AppliedUndef

        captured = set()
        for value in values:
            local = set(coordinates(value.shape))
            for symbol in value.expression.free_symbols | value.defined.free_symbols:
                if (
                    isinstance(symbol, sp.Symbol)
                    and symbol not in local
                    and (symbol.is_Dummy or symbol.is_integer)
                ):
                    captured.add(symbol)
        captured = tuple(sorted(captured, key=sp.default_sort_key))
        # 改写为固定参数名，使同一运算在不同组合中的身份保持一致。
        replacements = {
            s: sp.Symbol(f"_capture{i}", **({"integer": True} if s.is_integer else {"real": True}))
            for i, s in enumerate(captured)
        }
        signature = []
        for value in values:
            expression = value.expression
            cache_key = (expression, value.shape)
            if cache_key not in self.canonical_values:
                bounds = sp.And(
                    *(i < n for i, n in zip(coordinates(value.shape), value.shape, strict=True))
                )
                opaque_guard = any(
                    condition.has(AppliedUndef)
                    for pw in expression.atoms(sp.Piecewise)
                    for _, condition in pw.args
                )
                # 整数/布尔表达式没有编码溢出语义，含不透明函数的条件缺少实数
                # 不等式模型；这两类只统一绑定名，不做实数化简。
                self.canonical_values[cache_key] = (
                    rename_bounds(expression)
                    if dtype_kind(value.dtype) in "biu" or opaque_guard
                    else simplify(expression, self.budget, assumptions=bounds)
                )
            canonical = self.canonical_values[cache_key].xreplace(replacements)
            if operation[0] == "convert_element_type":
                # cast 的元素值依赖具体坐标；在小张量上把至多 64 个点的规范表达式
                # 展开成元组，防止不同输入被错误合并成同一换算身份。
                selected, count = [], 1
                for axis, extent in enumerate(value.shape):
                    if 1 < extent and count * extent <= 64:
                        selected.append((coordinates(value.shape)[axis], extent))
                        count *= extent
                if selected:
                    self.budget.spend(count, path="algebra.cast.coordinates")
                    canonical = tuple(
                        canonical.xreplace(dict(zip((i for i, _ in selected), point, strict=True)))
                        for point in product(*(range(n) for _, n in selected))
                    )
            signature.append(
                (
                    value.shape,
                    str(value.dtype),
                    canonical,
                    rename_bounds(domain_condition(value.defined, self.budget)).xreplace(
                        replacements
                    ),
                )
            )
        key = (
            operation,
            tuple(signature),
            tuple((tuple(v.aval.shape), str(v.aval.dtype)) for v in outputs),
        )
        if key not in self.definitions:
            self.definitions[key] = len(self.definitions)
        return self.definitions[key], captured

    def bound(self, extent: int) -> Limit:
        """生成一次归约的绑定变量，返回 (symbol, 0, extent - 1)。

        使用 Dummy 保证同一表达式中多次归约不会共用绑定名；名字只用于打印，
        身份由对象本身决定。extent 为 0 时上下限退化，调用方需另行处理空归约。
        """
        symbol = sp.Dummy(f"k{self.next_bound}", integer=True, nonnegative=True)
        self.next_bound += 1
        return symbol, 0, extent - 1

    def broadcast(self, value: SymbolicTensor, shape: Shape) -> tuple[sp.Basic, sp.Basic]:
        """把值按 JAX 广播规则读取到 shape，返回 (expression, defined)。

        广播只允许在左侧补轴或扩展长度为 1 的轴；扩展轴读 0 号元素。
        """
        indices = coordinates(shape)[len(shape) - len(value.shape) :]
        return value.read(
            tuple(
                sp.S.Zero if size == 1 else i for size, i in zip(value.shape, indices, strict=True)
            )
        )

    def universal(self, condition: sp.Basic, limits: Sequence[Limit]) -> sp.Basic:
        """把逐点定义条件提升为“对整个归约域都成立”的条件。

        小域直接逐点合取；大域用 Product(Piecewise) == 1 的惰性等式表达同一
        语义，避免生成长布尔表达式。该提升只用于归约轴，不能拿单点条件代表
        整个域。
        """
        if condition == sp.true or not limits:
            return condition
        count = prod(int(stop - start + 1) for _, start, stop in limits)
        if count <= 32:
            self.budget.spend(max(1, count), path="algebra.conditions")
            return sp.And(
                *(
                    condition.xreplace(
                        dict(zip((limit[0] for limit in limits), point, strict=True))
                    )
                    for point in product(
                        *(range(int(start), int(stop) + 1) for _, start, stop in limits)
                    )
                )
            )
        return sp.Eq(sp.Product(sp.Piecewise((1, condition), (0, True)), *limits), 1)

    def summation(
        self, expression: sp.Basic, limits: Sequence[Limit], defined: sp.Basic = sp.true
    ) -> sp.Basic:
        """构造有限求和，并在可证明安全时改写为零基前缀差分形式。

        只改写保持同一覆盖的矩形域：单点轴先代入，所有读取都带同一非负偏移时
        平移绑定，读取完整轴时用“全长前缀减前缀”扩展起点。任何可能越界、部分
        定义或含非有限值的求和都保留原始 Sum，交给 SymPy 以惰性形式比较。
        """
        # 长度为 1 的轴没有可选坐标；先代入常量即可消去绑定。
        singletons = {
            variable: sp.sympify(start) for variable, start, stop in limits if start == stop
        }
        expression = expression.xreplace(singletons)
        limits = tuple(limit for limit in limits if limit[1] != limit[2])
        if not limits:
            return expression
        node = sp.Sum(expression, *limits)
        # 部分函数在未定义点不能按实数恒等式改写；非有限值同理。
        if defined != sp.true or expression.has(sp.oo, -sp.oo, sp.nan, sp.zoo):
            return node
        for variable, _, _ in tuple(node.limits):
            offsets = set()
            for read in node.function.atoms(sp.Indexed):
                for coordinate in read.indices:
                    if coordinate.has(variable):
                        offset = coordinate - variable
                        if not offset.is_Integer or offset < 0:
                            offsets.add(None)
                        else:
                            offsets.add(int(offset))
            if len(offsets) == 1 and None not in offsets and next(iter(offsets)) > 0:
                offset = next(iter(offsets))
                node = node.change_index(variable, variable + offset)
        # 只有读取范围不越界时，才能把起点扩展成 0..stop 的完整前缀。
        for variable, start, stop in node.limits:
            if start == 0:
                continue
            if start < 0:
                return node
            for read in node.function.atoms(sp.Indexed):
                for coordinate, extent in zip(read.indices, read.base.shape, strict=True):
                    if coordinate.has(variable) and (coordinate != variable or stop >= extent):
                        return node
        choices = [
            ((1, (v, 0, stop)),) if start == 0 else ((1, (v, 0, stop)), (-1, (v, 0, start - 1)))
            for v, start, stop in node.limits
        ]
        # 多轴同时扩展会产生前缀差分的笛卡尔积，逐项计费后再构造。
        self.budget.spend(prod(len(choice) for choice in choices), path="algebra.index_domains")
        return sp.Add(
            *(
                prod(sign for sign, _ in selection)
                * sp.Sum(node.function, *(limit for _, limit in selection))
                for selection in product(*choices)
            )
        )

    def operation(
        self, eq: Equation, values: Sequence[SymbolicTensor], position: int
    ) -> Sequence[SymbolicTensor]:
        """转换一条方程：先建立来源与有限性，再选择逐点映射或不透明同余。

        Source 沿用源图的有限性分析（如 guarded softmax），作为“在保留条件下可
        作有限实数解释”的依据。区域原语交给 algebra.regions 绑定；非有限
        输入、离散原语和窄化 cast 没有完整实数模型，改走 opaque() 只保留整个运算
        的身份与条件，禁止套用实数代数。返回值必须与方程输出端口一一对应。
        """
        from sympy.core.function import AppliedUndef

        name = eq.primitive.name
        if name == "pallas_global_values":
            from .pallas import projected_outputs

            return projected_outputs(self, eq, values)
        # Partial memory values are emitted only through the Pallas adapter,
        # which propagates their separate validity channel. Never let an outer
        # ordinary-JAX/opaque/region consumer silently reset that obligation.
        if any(v.memory_validity != sp.true for v in values):
            raise UnsupportedJaxprError(
                "algebra.pallas.initialization",
                "unresolved memory validity at an ordinary JAX boundary",
            )
        # 区域原语由 algebra.regions 统一绑定，这里不对内部逐点展开。
        region = name in (
            "jit",
            "pjit",
            "cond",
            "scan",
            "while",
            "custom_jvp_call",
            "custom_vjp_call",
        )
        sources = []
        for var in eq.outvars:
            # 部分值（如区域适配器新构造的输出）没有来源，用有限性标记构造占位来源。
            inputs = tuple(
                v.source or Source(None, v.shape, v.dtype, finite=v.finite) for v in values
            )
            source = Source(
                getattr(lax, name + "_p", None),
                tuple(var.aval.shape),
                var.aval.dtype,
                inputs,
                eq.params,
            )
            # 区域和多输出没有逐点有限性规则；保守标为非有限，避免误用实数消项。
            source.finite = (
                self.finiteness.is_finite(source) if not region and len(eq.outvars) == 1 else False
            )
            if dtype_kind(var.aval.dtype) in "biu":
                # 整数/布尔值没有浮点非有限状态。
                source.finite = True
            sources.append(source)
        self.rules.update(self.finiteness.rules)
        # 纯布局/视图原语：输出元素与输入元素一一对应，有限性只取决于输入。
        layouts = {
            "broadcast_in_dim",
            "reshape",
            "transpose",
            "squeeze",
            "slice",
            "rev",
            "tile",
            "split",
            "stop_gradient",
            "copy",
            "pad",
        }
        if name == "reduce_sum" and all(values[0].shape[axis] == 1 for axis in eq.params["axes"]):
            # 单位轴求和等于恒等，不产生新值。
            layouts.add(name)
        narrowing = (
            name == "convert_element_type"
            and values
            and values[0].finite
            and not sources[0].finite
            and dtype_kind(values[0].dtype) == dtype_kind(eq.outvars[0].aval.dtype) == "f"
            and bool(values[0].expression.free_symbols)
        )
        if narrowing:
            # 浮点窄化 cast 可能把有限输入变成 inf；不能用实数恒等式表示，
            # 改为逐元素不透明函数，并把“该点转换有定义”作为条件保留。
            value = values[0]
            bounds = sp.And(
                *(i < n for i, n in zip(coordinates(value.shape), value.shape, strict=True))
            )
            argument = simplify(value.expression, self.budget, assumptions=bounds)
            label = f"cast_{value.dtype}_to_{eq.outvars[0].aval.dtype}"
            domain = sp.And(value.defined, sp.Eq(sp.Function("defined_" + label)(argument), 1))
            self.rules.add("scalar:convert_element_type")
            return [
                SymbolicTensor(
                    value.shape,
                    eq.outvars[0].aval.dtype,
                    sp.Function(label, real=True)(argument),
                    domain,
                    True,
                    sources[0],
                )
            ]
        if (
            not region
            and name not in layouts
            and (
                any(not v.finite for v in values)
                or narrowing
                or name
                in (
                    "top_k",
                    "approx_top_k",
                    "reduce",
                    "gather",
                    "sort",
                    "scatter",
                    "scatter-add",
                    "dynamic_update_slice",
                    "rem",
                    "shift_left",
                    "shift_right_logical",
                    "shift_right_arithmetic",
                    "xor",
                )
                or (
                    name in SCALAR
                    and dtype_kind(eq.outvars[0].aval.dtype) in "iu"
                    and any(v.expression.free_symbols for v in values)
                )
                or (
                    name == "convert_element_type"
                    and values[0].dtype != eq.outvars[0].aval.dtype
                    and dtype_kind(values[0].dtype) in "iu"
                    and values[0].expression.free_symbols
                )
                or (
                    name == "select_n"
                    and (
                        dtype_kind(eq.outvars[0].aval.dtype) in "biu"
                        or values[0].expression.has(AppliedUndef)
                    )
                )
                or (
                    name == "convert_element_type"
                    and values[0].expression.has(AppliedUndef)
                    and dtype_kind(eq.outvars[0].aval.dtype) in "iu"
                )
            )
        ):
            # 以上情况没有已核对的实数模型：非有限输入、窄化、离散/比较类原语、
            # 含符号的整数运算，以及依赖不透明函数结果的值域转换。失败方向必须
            # 保守：保留整个运算同余，而不是猜测数学等价。
            return self.opaque(eq, values, sources)
        outputs = self._operation(eq, values, position)
        if region:
            return outputs
        # 纯布局运算的输出有限性由输入决定；其余运算的 finite 已由 result() 按
        # 表达式是否含非有限值给出。
        return [
            replace(
                value,
                source=source,
                finite=all(v.finite for v in values) if name in layouts else value.finite,
            )
            for value, source in zip(outputs, sources, strict=True)
        ]

    def opaque(
        self, eq: Equation, values: Sequence[SymbolicTensor], sources: Sequence[Source]
    ) -> list[SymbolicTensor]:
        """把整个运算表示成一个未解释函数调用，只保留完整同余。

        参数是捕获变量和输出坐标，另设 defined_ 函数保存“该输出端口有定义”的
        条件。适用于非有限值、离散语义和窄化：对这些值套用实数代数没有依据，
        而相同运算加相同输入条件仍可安全判等。该表示只能支持同余的正结论，
        泛化问题的 False 不能当作源程序反例。
        """
        from .adapter.semantics import encode

        name = eq.primitive.name
        self.rules.add(name)
        identity, captured = self.definition((name, encode(eq.params)), values, eq.outvars)
        result = []
        for port, (var, source) in enumerate(zip(eq.outvars, sources, strict=True)):
            shape = tuple(var.aval.shape)
            args = (*captured, *coordinates(shape))
            label = f"{name}_{identity}_{port}"
            # 非有限输出不加 real 假设：有限实数恒等式不能用于消项。
            expression = sp.Function("opaque_" + label, real=True if source.finite else None)(*args)
            if dtype_kind(var.aval.dtype) == "b":
                expression = sp.Eq(expression, 1)
            domain = sp.Eq(sp.Function("defined_" + label)(*args), 1)
            result.append(
                SymbolicTensor(shape, var.aval.dtype, expression, domain, source.finite, source)
            )
        return result

    def _operation(
        self, eq: Equation, values: Sequence[SymbolicTensor], position: int
    ) -> Sequence[SymbolicTensor]:
        """把一条已核对原语映射成逐点/索引表达式，返回按端口排列的 SymbolicTensor。

        每个分支都必须保留完整的操作数、坐标和参数语义；没有适配的分支在末尾
        显式抛出 UnsupportedJaxprError，不能返回近似表达式。
        """
        name, params = eq.primitive.name, eq.params
        path = f"algebra.convert.eqns[{position}].{name}"
        self.rules.add(name)
        if "call_jaxpr" in params and name in ("custom_jvp_call", "custom_vjp_call"):
            # 自定义求导调用保留调用边界，内部图交给同一个构造器。
            return self.graph(params["call_jaxpr"], values)
        if name in ("jit", "pjit"):
            if (
                "common_call" in params
                and all(value.finite for value in values)
                and all(dtype_kind(v.aval.dtype) == "f" for v in eq.outvars)
            ):
                # 两份图中已核对相同的共享调用不必再次展开；用不透明函数按端口
                # 保留身份和整体定义条件，避免依赖各自的私有实现细节。
                identity, inputs, outputs = params["common_call"]
                identity, captured = self.definition(
                    ("common_jit", identity),
                    [values[i] for i in inputs],
                    [eq.outvars[i] for i in outputs],
                )
                # 定义条件覆盖整个调用，所有输出端口共用同一份条件。
                domain = sp.Eq(sp.Function(f"defined_call_{identity}")(*captured), 1)
                call_outputs: list[SymbolicTensor | None] = [None] * len(outputs)
                for port, mapped in enumerate(outputs):
                    aval = eq.outvars[mapped].aval
                    if dtype_kind(aval.dtype) != "f":
                        raise UnsupportedJaxprError(path, "abstract call needs real-valued outputs")
                    expression = sp.Function(f"call_{identity}_{port}", real=True)(
                        *captured, *coordinates(aval.shape)
                    )
                    call_outputs[mapped] = SymbolicTensor(
                        tuple(aval.shape), aval.dtype, expression, domain
                    )
                self.rules.add("common_jit")
                return cast(list[SymbolicTensor], call_outputs)
            return self.graph(params["jaxpr"], values)
        if name in ("cond", "scan", "while"):
            # 区域内部用同一个构造器绑定实参、carry 和返回口。
            return region(self, eq, values, path)
        if len(eq.outvars) != 1 and name != "split":
            raise UnsupportedJaxprError(path, "multiple-output primitive has no symbolic adapter")
        aval = eq.outvars[0].aval
        shape, axes = tuple(aval.shape), coordinates(aval.shape)

        def result(expression: sp.Basic, defined: sp.Basic = sp.true) -> list[SymbolicTensor]:
            """包装单输出：统一坐标规范化，并按表达式是否含非有限值标记 finite。"""
            return [
                SymbolicTensor(
                    shape,
                    aval.dtype,
                    normalize_coordinates(sp.sympify(expression), shape),
                    normalize_coordinates(sp.sympify(defined), shape),
                    finite=not sp.sympify(expression).has(sp.oo, -sp.oo, sp.nan, sp.zoo),
                )
            ]

        if name == "empty2":
            # 锁定版本 JAX 的 _empty2_lower 返回标量零（含转置规则产生的残差占位），
            # 因此按全定义零处理；该结论不能外推到其他原语。
            return result(0)

        if name in SCALAR:
            if params.get("accuracy") is not None:
                # 非默认 accuracy 改变浮点语义，当前映射只覆盖默认值。
                raise UnsupportedJaxprError(path, "non-default scalar accuracy")
            args, conditions = zip(*(self.broadcast(v, shape) for v in values), strict=True)
            if name in ("not", "and", "or") and any(dtype_kind(v.dtype) != "b" for v in values):
                # 整数的 and/or/not 是位运算，不是布尔逻辑。
                raise UnsupportedJaxprError(path, "integer bitwise arithmetic is not Boolean logic")
            if (
                dtype_kind(aval.dtype) in "iu"
                and name not in ("max", "min")
                and any(arg.free_symbols for arg in args)
            ):
                # 固定宽度整数没有编码溢出语义，不对符号整数套用实数代数。
                raise UnsupportedJaxprError(
                    path, "fixed-width integer arithmetic has no real interpretation"
                )
            # 为部分函数补上原始定义条件；条件成立前不做实数化简。
            domain = sp.And(*conditions)
            if name == "div":
                domain &= sp.Ne(args[1], 0)
            elif name == "log":
                domain &= args[0] > 0
            elif name == "sqrt":
                domain &= args[0] >= 0
            elif name == "integer_pow" and params["y"] < 0:
                domain &= sp.Ne(args[0], 0)
            elif name == "pow":
                domain &= args[0] > 0
            elif name in ("asin", "acos"):
                domain &= sp.And(args[0] >= -1, args[0] <= 1)
            elif name == "tan":
                domain &= sp.Ne(sp.cos(args[0]), 0)
            elif name == "atan2":
                domain &= sp.Or(sp.Ne(args[0], 0), sp.Ne(args[1], 0))
            scalar_args = (*args, params["y"]) if name == "integer_pow" else args
            expression = sp.sympify(SCALAR[name](*scalar_args))
            if dtype_kind(aval.dtype) in "iu" and not expression.free_symbols:
                # 常量整数运算按 dtype 值域检查；溢出值没有可表示的数学结果。
                limits = np.iinfo(aval.dtype)
                if not expression.is_Integer or not limits.min <= expression <= limits.max:
                    raise UnsupportedJaxprError(
                        path, "integer constant arithmetic overflows its dtype"
                    )
            return result(expression, domain)
        if name in ("stop_gradient", "copy"):
            # 两者不改变元素值，只是视图/停止梯度包装。
            return result(*values[0].read(axes))
        if name == "convert_element_type":
            value = values[0]
            if np.dtype(value.dtype) == np.dtype(aval.dtype):
                # 同 dtype 转换（含 weak_type 变化）不改变元素值。
                return result(*value.read(axes))
            expression, domain = value.read(axes)
            if np.dtype(value.dtype).kind == "b":
                return result(sp.Piecewise((1, expression), (0, True)), domain)
            if not expression.free_symbols:
                # 常量转换交给 NumPy 按目标 dtype 舍入，再取精确值。
                converted = np.asarray(float(expression), dtype=aval.dtype)
                return result(_exact_array(converted).item(), domain)
            if dtype_kind(value.dtype) == dtype_kind(aval.dtype) == "f":
                # 不同浮点格式之间的舍入没有实数恒等式，只能逐元素保留为不透明函数。
                function = sp.Function(f"cast_{value.dtype}_to_{aval.dtype}", real=True)
                self.rules.add("scalar:convert_element_type")
                return result(function(expression), domain)
            raise UnsupportedJaxprError(
                path, "value-changing cast requires an explicit mathematical adapter"
            )
        if name == "broadcast_in_dim":
            indices = tuple(
                0 if size == 1 else axes[axis]
                for size, axis in zip(values[0].shape, params["broadcast_dimensions"], strict=True)
            )
            return result(*values[0].read(cast(Sequence[sp.Expr | int], indices)))
        if name == "transpose":
            indices = [None] * len(axes)
            for axis, source in enumerate(params["permutation"]):
                indices[source] = axes[axis]
            return result(*values[0].read(cast(Sequence[sp.Expr | int], indices)))
        if name == "squeeze":
            iterator = iter(axes)
            return result(
                *values[0].read(
                    tuple(
                        0 if i in params["dimensions"] else next(iterator)
                        for i in range(len(values[0].shape))
                    )
                )
            )
        if name == "reshape":
            original = values[0]
            if prod(shape) == 0:
                # 空张量没有元素可读，表达式只作为占位。
                return result(0)
            order = params.get("dimensions") or tuple(range(len(original.shape)))
            old_shape = tuple(original.shape[i] for i in order)
            # 把新坐标压成行主序线性下标，再按旧 shape 的步长还原每个轴。
            flat = sum(i * prod(shape[axis + 1 :]) for axis, i in enumerate(axes))
            reshape_indices: list[sp.Expr | int | None] = [None] * len(old_shape)
            for axis, source in enumerate(order):
                extent, stride = old_shape[axis], prod(old_shape[axis + 1 :])
                reshape_indices[source] = (
                    0 if extent <= 1 else sp.Mod(sp.floor(flat / stride), extent)
                )
            return result(*original.read(reshape_indices))
        if name == "slice":
            strides = params.get("strides") or (1,) * len(axes)
            return result(
                *values[0].read(
                    tuple(
                        start + i * stride
                        for start, i, stride in zip(
                            params["start_indices"], axes, strides, strict=True
                        )
                    )
                )
            )
        if name == "pad":
            original, padding = values
            indices, guards = [], []
            for coordinate, extent, (low, _, interior) in zip(
                axes, original.shape, params["padding_config"], strict=True
            ):
                # 输出坐标减去低侧 padding 后还原到原数组坐标；只有落在内部且
                # 对齐到内部采样点时才读取，否则使用填充值。
                shifted = coordinate - low
                indices.append(sp.floor(shifted / (interior + 1)))
                guards.extend(
                    (
                        shifted >= 0,
                        shifted < extent * (interior + 1),
                        sp.Eq(sp.Mod(shifted, interior + 1), 0),
                    )
                )
            guard = sp.And(*guards)
            expression, domain = original.read(indices)
            fill, fill_domain = padding.read(())
            return result(
                sp.Piecewise((expression, guard), (fill, True)),
                sp.Or(sp.And(guard, domain), sp.And(sp.Not(guard), fill_domain)),
            )
        if name == "dynamic_slice":
            # JAX 会把动态起点钳制到 [0, size - extent]；起点的定义条件也必须保留。
            indices = tuple(
                i + sp.Min(sp.Max(start.expression, 0), size - extent)
                for i, start, size, extent in zip(
                    axes, values[1:], values[0].shape, shape, strict=True
                )
            )
            expression, domain = values[0].read(indices)
            return result(expression, sp.And(domain, *(v.defined for v in values[1:])))
        if name == "rev":
            return result(
                *values[0].read(
                    tuple(
                        size - 1 - i if axis in params["dimensions"] else i
                        for axis, (size, i) in enumerate(zip(shape, axes, strict=True))
                    )
                )
            )
        if name == "tile":
            # 块内坐标取原数组坐标的模；左侧补 1 便于与输出轴对齐。
            padded = (1,) * (len(shape) - len(values[0].shape)) + values[0].shape
            indices = tuple(
                0 if size == 1 else sp.Mod(i, size) for size, i in zip(padded, axes, strict=True)
            )
            return result(*values[0].read(indices[len(padded) - len(values[0].shape) :]))
        if name == "iota":
            return result(axes[params["dimension"]])
        if name in (
            "reduce_sum",
            "reduce_prod",
            "reduce_max",
            "reduce_min",
            "reduce_or",
            "reduce_and",
        ):
            original = values[0]
            limits = [self.bound(original.shape[axis]) for axis in params["axes"]]
            reductions = dict(zip(params["axes"], limits, strict=True))
            free = iter(axes)
            indices = tuple(
                reductions[i][0] if i in reductions else next(free)
                for i in range(len(original.shape))
            )
            expression, domain = original.read(indices)
            empty = any(original.shape[axis] == 0 for axis in params["axes"])
            if empty:
                # 空归约使用各原语的单位元；-inf/+inf 在这里只作为该原语的值。
                return result(
                    {
                        "reduce_sum": 0,
                        "reduce_prod": 1,
                        "reduce_max": -sp.oo,
                        "reduce_min": sp.oo,
                        "reduce_or": sp.false,
                        "reduce_and": sp.true,
                    }[name]
                )
            if not limits:
                return result(expression, domain)
            if name in ("reduce_sum", "reduce_prod"):
                function = sp.Sum if name == "reduce_sum" else sp.Product
                return result(
                    self.summation(expression, limits, domain)
                    if name == "reduce_sum"
                    else function(expression, *limits),
                    self.universal(domain, limits),
                )
            # 有限域上的 max/min/布尔归约逐点展开；展开成本先扣预算。
            count = prod(stop + 1 for _, _, stop in limits)
            self.budget.spend(count, path=path)
            terms = [
                expression.xreplace(dict(zip((lim[0] for lim in limits), point, strict=True)))
                for point in product(*(range(stop + 1) for _, _, stop in limits))
            ]
            function = {
                "reduce_max": maximum,
                "reduce_min": minimum,
                "reduce_or": sp.Or,
                "reduce_and": sp.And,
            }[name]
            return result(function(*terms), self.universal(domain, limits))
        if name == "dot_general":
            if params.get("precision") not in (None, (None, None)):
                # 显式精度属于浮点执行细节，不在理想实数适配的契约内。
                raise UnsupportedJaxprError(path, "non-default dot precision")
            left, right = values
            (ca, cb), (ba, bb) = params["dimension_numbers"]
            ia, ib = [None] * len(left.shape), [None] * len(right.shape)
            free = iter(axes)
            # 批次维共享输出自由坐标；收缩维稍后变成同一个求和绑定；其余维按
            # 输出位置依次取自由坐标。
            for a, b in zip(ba, bb, strict=True):
                ia[a] = ib[b] = next(free)
            for indices, contracted, batched in ((ia, ca, ba), (ib, cb, bb)):
                for axis in range(len(indices)):
                    if axis not in contracted + batched:
                        indices[axis] = next(free)
            limits = []
            for a, b in zip(ca, cb, strict=True):
                limit = self.bound(left.shape[a])
                limits.append(limit)
                ia[a] = ib[b] = limit[0]
            a, da = left.read(ia)
            b, db = right.read(ib)
            if any(left.shape[axis] == 0 for axis in ca):
                # 空收缩维求和是零。
                return result(0)
            return result(
                self.summation(a * b, limits, sp.And(da, db)),
                self.universal(sp.And(da, db), limits),
            )
        if name == "select_n":
            pred, pred_domain = self.broadcast(values[0], shape)
            branches = [self.broadcast(value, shape) for value in values[1:]]
            boolean = np.dtype(values[0].dtype).kind == "b"
            # 布尔谓词只有两支（谓词真/假）；整数谓词按取值选择对应分支。
            guards = [
                (sp.Not(pred) if i == 0 else pred) if boolean else sp.Eq(pred, i)
                for i in range(len(branches))
            ]
            expression = sp.Piecewise(
                *[(value, guard) for (value, _), guard in zip(branches, guards, strict=True)],
                (sp.false if dtype_kind(aval.dtype) == "b" else sp.S.Zero, True),
            )
            domain = sp.And(
                pred_domain,
                sp.Or(
                    *[
                        sp.And(guard, domain)
                        for (_, domain), guard in zip(branches, guards, strict=True)
                    ]
                ),
            )
            return result(expression, domain)
        if name in ("concatenate", "stack"):
            axis = cast(int, params.get("dimension", params.get("axis")))
            branches, domains, offset = [], [], 0
            # 沿拼接轴给每个输入加半开区间守卫，再合成 Piecewise。
            for value in values:
                index = list(axes)
                if name == "stack":
                    del index[axis]
                    extent = 1
                else:
                    index[axis] -= offset
                    extent = value.shape[axis]
                expression, domain = value.read(index)
                guard = sp.And(axes[axis] >= offset, axes[axis] < offset + extent)
                branches.append((expression, guard))
                domains.append(sp.And(guard, domain))
                offset += extent
            return result(
                sp.Piecewise(
                    *branches, (sp.false if dtype_kind(aval.dtype) == "b" else sp.S.Zero, True)
                ),
                sp.Or(*domains),
            )
        if name == "split":
            outputs, offset = [], 0
            # 每个输出只在 split 轴的一段区间上读取输入；各段首尾相接覆盖该轴。
            for var, extent in zip(eq.outvars, params["sizes"], strict=True):
                index = list(coordinates(var.aval.shape))
                index[params["axis"]] += offset
                expression, domain = values[0].read(index)
                outputs.append(
                    SymbolicTensor(tuple(var.aval.shape), var.aval.dtype, expression, domain)
                )
                offset += extent
            return outputs
        raise UnsupportedJaxprError(path, "no complete symbolic adapter for primitive")
