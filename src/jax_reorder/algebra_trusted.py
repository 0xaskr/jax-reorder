"""用可信 SymPy 查询数学等价，并给出可终止的进程边界。

algebra_symbolic 负责把两份图构造成完整公式和原始定义条件；本模块负责在
声明的理想实数解释下判定输出恒等与条件等价，返回 equivalent/inequivalent，
其余情况（不支持、条件未判定、CAS 未判定、预算、超时、内部错误）一律抛分类
异常。可能无界的符号构造、化简和查询在独立 worker 中执行，超时由父进程 kill
并回收；查询记录只用于审计，不是独立证明证书。
"""

import os
import pickle
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from itertools import islice, product
from math import comb, prod

import sympy as sp

from .algebra_symbolic import SymbolicBuilder, coordinates, rename_bounds
from .errors import (
    ComparisonContractError,
    ComparisonError,
    ComparisonInternalError,
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    UnsupportedJaxprError,
)


@dataclass(frozen=True)
class SympyQuery:
    """单次数学查询的审计记录。

    记录输出来源、查询范围（整输出/定义条件/具体坐标）、True/False/None、实际
    方法链和左右表达式文本。它用于解释结论与复现问题，不是可独立复核的证书。
    """

    output: int
    scope: str
    answer: bool | None
    method: str
    left: str
    right: str


@dataclass(frozen=True)
class SympyResult:
    """成功的数学结论，只有 equivalent/inequivalent 两种状态。

    条件未判定、CAS 未判定、预算耗尽、超时和内部错误都由异常承载，不混入结果
    状态；queries/path/output/work/rules/definitions 记录判定路径和资源消耗，
    供审计和失败定位，不参与结论本身。
    """

    status: str
    queries: tuple[SympyQuery, ...] = ()
    path: str = "algebra.outputs"
    output: int | None = None
    backend: str = "sympy"
    version: str = sp.__version__
    work: int = 0
    elapsed_seconds: float = field(default=0, compare=False)
    rules: tuple[str, ...] = ()
    definitions: tuple[tuple[int, str], ...] = ()

    def __post_init__(self):
        """构造时校验状态，避免把第三种“未决”状态混进结果对象。"""
        if self.status not in ("equivalent", "inequivalent"):
            raise ValueError(f"invalid mathematical result status: {self.status}")


@dataclass(frozen=True)
class WorkerFailure:
    """失败消息与 SympyResult 分离，不携带第四类比较状态。

    worker 把异常分类、路径、诊断和 traceback 序列化回来，父进程据此重建同类
    异常，并把远端 traceback 保存在异常的 cause 上。
    """

    error_type: type[ComparisonError]
    path: str
    reason: str
    stage: str
    failure_kind: str
    diagnostics: dict
    traceback: str


class RemoteWorkerError(RuntimeError):
    """只作为父进程异常的 cause 保存远端原始异常链，不代表比较结论。"""


def _finite_expansion(expression, budget):
    """在调用会隐式展开的 SymPy 之前，预估显式 Sum/Product 的展开成本。

    只接受上下限为精确非负整数的显式有限和；估算超过 256 项时返回 False，
    调用方保留惰性形式或判定未决，而不是让 SymPy 无预算地展开。数值 shape
    小不代表展开小，因此这里按项数而不是按张量大小计费。
    """
    costs = {}

    def cost(node):
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


def simplify(expression, budget, *, assumptions=sp.true, expand=True):
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
    if expand and expression.has(sp.Sum, sp.Product) and _finite_expansion(expression, budget):
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


def _array_compatible(expression, indices):
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

    def supported(node):
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


def equal(left, right, budget, *, indices=None, assumptions=sp.true, methods=None):
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

    def boolean_piecewise(expression):
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
        and _array_compatible(left, indices)
        and _array_compatible(right, indices)
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
    if residual.has(sp.Sum, sp.Product) and not _finite_expansion(residual, budget):
        # Expr.equals 可能内部调用 doit，绕过显式展开预算；此时保留未决，
        # 由调用方在没有其他方法判定时抛出分类异常。
        steps.append("unevaluated_sum")
        return None
    steps.append("Expr.equals(0)")
    return residual.equals(0)


def _domain(condition, budget):
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


def _explicit_symbols(left, right):
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


def compare_symbolic(left, right, algebra_budget=10000):
    """比较两份已结构校验的 jaxpr 的完整数学输出。

    构造两侧输出后逐输出核对抽象契约、定义条件和元素表达式：所有输出及条件
    都通过才返回 equivalent；任一输出在合法坐标上被判为不恒等才返回
    inequivalent；条件未建立或 CAS 无法判定则抛 MathematicalUndecidedError。
    中间表达式不同、泛化变量上的 False 都不能直接推出源函数不等价。
    """
    started = time.monotonic()
    builder = SymbolicBuilder(algebra_budget)
    records = []

    def query(output, scope, x, y, **kwargs):
        """执行一次恒等查询并记录审计信息；None 原样返回给调用方处理。"""
        methods = []
        answer = equal(x, y, builder.budget, methods=methods, **kwargs)
        records.append(SympyQuery(output, scope, answer, " -> ".join(methods), str(x), str(y)))
        return answer

    def report(*args, **kwargs):
        """构造成功结果；不透明定义表用于解释表达式中的函数身份。"""
        return SympyResult(
            *args,
            **kwargs,
            definitions=tuple((index, repr(key)) for key, index in builder.definitions.items()),
        )

    try:
        a, b = builder.graph(left), builder.graph(right)
        # 两侧输出数量、shape、dtype 必须一致；构造后再检查一次，防止适配器改变形状。
        if len(a) != len(b) or any(
            x.shape != y.shape or x.dtype != y.dtype for x, y in zip(a, b, strict=True)
        ):
            raise ComparisonContractError("algebra.outputs", "output abstract contracts differ")
        undecided, domain_unknown = False, False
        for output, (x, y) in enumerate(zip(a, b, strict=True)):
            if prod(x.shape) == 0:
                # 空输出没有元素可比较；shape/dtype 已在上方核对。
                continue
            if x.defined == y.defined:
                da, db = x.defined, y.defined
            else:
                da, db = _domain(x.defined, builder.budget), _domain(y.defined, builder.budget)
            same_domain = query(output, "definedness", da, db)
            if same_domain is not True and prod(x.shape) <= 64:
                # 抽象条件可能因绑定名或分段形式不同而无法整体化简；小输出域
                # 逐坐标复核，避免把等价条件误判为未决。
                same_domain = all(
                    query(
                        output,
                        f"definedness_coordinate={point}",
                        _domain(x.read(point)[1], builder.budget),
                        _domain(y.read(point)[1], builder.budget),
                    )
                    is True
                    for point in product(*(range(n) for n in x.shape))
                )
            if same_domain is not True:
                domain_unknown = True
                continue
            if da == sp.false:
                # 输出处处无定义时没有可比较的值，也不能据此判等价。
                domain_unknown = True
                continue
            # 给符号查询补上合法输出坐标范围，供 ArrayExpr/refine 使用。
            bounds = sp.And(*(i < n for i, n in zip(coordinates(x.shape), x.shape, strict=True)))
            answer = query(
                output,
                "symbolic_output",
                x.expression,
                y.expression,
                indices=coordinates(x.shape),
                assumptions=bounds,
            )
            if answer is True:
                continue
            # 抽象坐标上的 False 不能证明存在合法输出坐标；只实例化索引，输入值
            # 仍保持符号，再在具体坐标上构造反例。
            concrete_answers = []
            fully_enumerated = prod(x.shape) <= 64
            # 完整枚举上限 64 个点；大输出域每轴只取前 2 个点、最多 16 个，
            # 只用于寻找反例，不能作为“全部输出通过”的依据。
            points = product(
                *(range(size if fully_enumerated else min(size, 2)) for size in x.shape)
            )
            for point in islice(points, 64 if fully_enumerated else 16):
                indices = tuple(map(sp.Integer, point))
                ex, dx = x.read(indices)
                ey, dy = y.read(indices)
                dx, dy = _domain(dx, builder.budget), _domain(dy, builder.budget)
                if dx == sp.false or equal(dx, dy, builder.budget) is not True:
                    # 该坐标下两侧定义条件不一致或未建立，不能比较值。
                    concrete_answers.append(None)
                    continue
                if not (
                    _finite_expansion(ex, builder.budget) and _finite_expansion(ey, builder.budget)
                ):
                    # 显式展开超出预算时保持惰性，不调用会隐式 doit 的查询。
                    concrete_answers.append(None)
                    continue
                ex, ey = ex.doit(), ey.doit()
                explicit = _explicit_symbols(ex, ey)
                methods = []
                if explicit is None:
                    # 不透明调用/cast 只有同余正结论可靠；泛化 False 不算反例。
                    decision = (
                        True if equal(ex, ey, builder.budget, methods=methods) is True else None
                    )
                else:
                    decision = equal(*explicit, builder.budget, methods=methods)
                if decision is False and (dx != sp.true or dy != sp.true):
                    # 条件未完全建立时，False 不能提升为源程序的不等价结论。
                    decision = None
                concrete_answers.append(decision)
                records.append(
                    SympyQuery(
                        output,
                        f"output_coordinate={point}",
                        decision,
                        " -> ".join(methods),
                        str(ex),
                        str(ey),
                    )
                )
                if decision is False:
                    return report(
                        "inequivalent",
                        tuple(records),
                        output=output,
                        work=builder.budget.used,
                        rules=tuple(sorted(builder.rules)),
                        elapsed_seconds=time.monotonic() - started,
                    )
            # 只有完整枚举且全部坐标都判定为真，才算该输出通过；其余留待失败分类。
            undecided |= not (
                fully_enumerated and concrete_answers and all(v is True for v in concrete_answers)
            )
        # 优先报告条件问题：条件未建立时，值比较的前提本身就不完整。
        failure = (
            "condition_undecided" if domain_unknown else "cas_undecided" if undecided else None
        )
        if failure:
            raise MathematicalUndecidedError(
                "algebra.outputs",
                "original conditions are not established"
                if domain_unknown
                else "SymPy did not establish output identity",
                failure_kind=failure,
            )
        return report(
            "equivalent",
            tuple(records),
            work=builder.budget.used,
            rules=tuple(sorted(builder.rules)),
            elapsed_seconds=time.monotonic() - started,
        )
    except ComparisonError as exc:
        # 保留已完成的查询和资源使用，便于定位；不改变原异常分类。
        exc.diagnostics.update(
            queries=tuple(records),
            work=builder.budget.used,
            elapsed_seconds=time.monotonic() - started,
            rules=tuple(sorted(builder.rules)),
            definitions=tuple((index, repr(key)) for key, index in builder.definitions.items()),
        )
        if isinstance(exc, UnsupportedJaxprError):
            # 转换缺失统一标记为 algebra 阶段的 unsupported_conversion。
            raise UnsupportedJaxprError(
                exc.path,
                exc.reason,
                stage="algebra",
                failure_kind="unsupported_conversion",
                **exc.diagnostics,
            ) from exc
        raise
    except Exception as exc:
        # 未预期异常按内部错误报告，绝不当作未决或数学结论。
        raise ComparisonInternalError(
            "algebra.sympy",
            f"{type(exc).__name__}: {exc}",
            stage="algebra",
            queries=tuple(records),
            work=builder.budget.used,
            elapsed_seconds=time.monotonic() - started,
        ) from exc


def isolated_compare(left, right, *, algebra_budget=10000, timeout=10.0):
    """在独立解释器进程中调用 compare_symbolic，并用墙钟上限兜底无界化简。

    输入是已冻结的图数据，不执行 JAX 数组；超时先 kill 并回收 worker，非零退出
    和非法响应转为分类异常，保证后续比较可以继续。输入/结果经 pickle 传输，
    worker 失败时以 WorkerFailure 保留分类、诊断和远端 traceback。
    """
    if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
        # bool 不是合法超时；只接受正有限数。
        raise ValueError("algebra_timeout must be a positive finite number")
    started = time.monotonic()
    try:
        payload = pickle.dumps((left, right, algebra_budget))
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.transport", f"symbolic transport failed: {exc}", stage="algebra"
        ) from exc
    environment = os.environ.copy()
    # 覆盖继承自父进程的 BLAS/OpenMP 线程配置，固定 worker 的数值库线程数。
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "jax_reorder.algebra_trusted"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        raise ComparisonInternalError(
            "algebra.sympy.worker", f"worker could not start: {exc}", stage="algebra"
        ) from exc
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # 超时后必须 kill + communicate 回收，否则子进程可能继续消耗资源。
        process.kill()
        process.communicate()
        raise ComparisonTimeoutError(
            "algebra.sympy.timeout",
            f"symbolic worker exceeded {timeout:g}s",
            timeout=timeout,
            elapsed_seconds=time.monotonic() - started,
        ) from exc
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.sympy.worker", f"worker communication failed: {exc}", stage="algebra"
        ) from exc
    finally:
        # 无论正常返回还是异常路径，都不留下仍在运行的 worker。
        if process.poll() is None:
            process.kill()
            process.wait()
    if process.returncode:
        # 非零退出表示 worker 没有返回可解析结果；stderr 尾部用于定位。
        raise ComparisonInternalError(
            "algebra.sympy.worker",
            f"worker exited {process.returncode}: {stderr.decode(errors='replace')[-1000:]}",
            stage="algebra",
            elapsed_seconds=time.monotonic() - started,
        )
    try:
        result = pickle.loads(stdout)
        if isinstance(result, WorkerFailure):
            # 只接受 ComparisonError 子类，防止把任意异常类型当结果重建。
            if not isinstance(result.error_type, type) or not issubclass(
                result.error_type, ComparisonError
            ):
                raise TypeError("invalid symbolic worker exception type")
            error = result.error_type(
                result.path,
                result.reason,
                stage=result.stage,
                failure_kind=result.failure_kind,
                **result.diagnostics,
            )
            error.diagnostics["elapsed_seconds"] = time.monotonic() - started
        elif not isinstance(result, SympyResult) or result.status not in (
            "equivalent",
            "inequivalent",
        ):
            raise TypeError("invalid symbolic worker result")
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.sympy.worker", f"invalid worker response: {exc}", stage="algebra"
        ) from exc
    if isinstance(result, WorkerFailure):
        # 用 RemoteWorkerError 作为 cause 保留远端 traceback，同时保持异常类型不变。
        raise error from RemoteWorkerError(result.traceback)
    # 父进程墙钟覆盖 worker 内部计时，包含进程启动与传输时间。
    return replace(result, elapsed_seconds=time.monotonic() - started)


def _worker_main():
    """worker 进程入口：读取 pickle 输入，写回结果或失败消息。"""
    # 使用规范模块名序列化类型，父进程无需导入 __main__ 中的类。
    from jax_reorder.algebra_trusted import WorkerFailure, compare_symbolic

    try:
        payload = pickle.load(sys.stdin.buffer)
        result = compare_symbolic(*payload)
    except Exception as exc:
        # 任何异常都转成 WorkerFailure：非 ComparisonError 统一按内部错误分类。
        error = (
            exc
            if isinstance(exc, ComparisonError)
            else ComparisonInternalError(
                "algebra.sympy.worker", f"{type(exc).__name__}: {exc}", stage="algebra"
            )
        )
        result = WorkerFailure(
            type(error),
            error.path,
            error.reason,
            error.stage,
            error.failure_kind,
            error.diagnostics,
            traceback.format_exc(),
        )
    # stdout 只写一个 pickle 结果；父进程按 returncode 和可解析性校验响应。
    pickle.dump(result, sys.stdout.buffer)


if __name__ == "__main__":
    _worker_main()
