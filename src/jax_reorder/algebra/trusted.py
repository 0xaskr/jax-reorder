"""用可信 SymPy 判定数学等价，并把可能跑不完的计算放进独立进程。

algebra.symbolic 负责把两张图构造成完整公式和原始定义条件；本模块在声明的
“理想实数”解释下判断输出恒等、条件等价，只返回 equivalent 或 inequivalent。
其余情况——不支持、条件未判定、CAS 未判定、超出预算、超时、内部错误——全部
抛分类异常。
可能无界的符号构造、化简和查询都会在独立 worker 里跑，超时由父进程 kill 并
回收。查询记录只用于审计，不是可独立复核的证明证书。
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import islice, product
from math import prod
from typing import cast

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from ..errors import (
    ComparisonContractError,
    ComparisonError,
    ComparisonFailureKind,
    ComparisonInternalError,
    ComparisonStage,
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    UnsupportedJaxprError,
)
from ._typing import SymbolicGraph
from .budget import validate_timeout
from .expression import SymbolicTensor, coordinates, dag_atoms, same_bound_expression
from .pallas_precision import pair_pallas_precision
from .symbolic import SymbolicBuilder
from .sympy_ops import (
    boolean_piecewise,
    domain_condition,
    equal,
    explicit_symbols,
    finite_expansion,
    simplify,
)
from .witnesses import DomainCache, exact_witness


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

    def __post_init__(self) -> None:
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
    stage: ComparisonStage
    failure_kind: ComparisonFailureKind
    diagnostics: dict[str, object]
    traceback: str


class RemoteWorkerError(RuntimeError):
    """只作为父进程异常的 cause 保存远端原始异常链，不代表比较结论。"""


def _witness_points(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Visit each axis before a Cartesian prefix can exhaust the search limit."""
    origin = (0,) * len(shape)
    seen: set[tuple[int, ...]] = {origin}
    yield origin
    for axis, size in enumerate(shape):
        for position in sorted({min(1, size - 1), size // 2, size - 1} - {0}):
            point = origin[:axis] + (position,) + origin[axis + 1 :]
            seen.add(point)
            yield point
    for point in product(*(sorted({0, min(1, n - 1), n // 2, n - 1}) for n in shape)):
        if point not in seen:
            yield point


def compare_symbolic(
    left: SymbolicGraph, right: SymbolicGraph, algebra_budget: int = 10000
) -> SympyResult:
    """比较两份已结构校验的 jaxpr 的完整数学输出。

    构造两侧输出后逐输出核对抽象契约、定义条件和元素表达式：所有输出及条件
    都通过才返回 equivalent；任一输出在合法坐标上被判为不恒等才返回
    inequivalent；条件未建立或 CAS 无法判定则抛 MathematicalUndecidedError。
    中间表达式不同、泛化变量上的 False 都不能直接推出源函数不等价。
    """
    started = time.monotonic()
    builder = SymbolicBuilder(algebra_budget)
    records: list[SympyQuery] = []
    witness_domains: DomainCache = {}

    def query(
        output: int,
        scope: str,
        x: sp.Basic,
        y: sp.Basic,
        *,
        indices: Sequence[sp.Expr] | None = None,
        assumptions: sp.Basic = sp.true,
    ) -> bool | None:
        """执行一次恒等查询并记录审计信息；None 原样返回给调用方处理。"""
        methods: list[str] = []
        answer = equal(
            x, y, builder.budget, methods=methods, indices=indices, assumptions=assumptions
        )
        records.append(SympyQuery(output, scope, answer, " -> ".join(methods), str(x), str(y)))
        return answer

    def report(
        status: str,
        queries: tuple[SympyQuery, ...],
        *,
        output: int | None = None,
        work: int,
        rules: tuple[str, ...],
        elapsed_seconds: float,
    ) -> SympyResult:
        """构造成功结果；不透明定义表用于解释表达式中的函数身份。"""
        return SympyResult(
            status,
            queries,
            output=output,
            work=work,
            rules=rules,
            elapsed_seconds=elapsed_seconds,
            definitions=tuple((index, repr(key)) for key, index in builder.definitions.items()),
        )

    def witness_at(
        output: int, point: tuple[int, ...], x: SymbolicTensor, y: SymbolicTensor
    ) -> SympyResult | None:
        """Find an exact counterexample before an expensive whole-formula CAS query."""
        ex, dx = x.read(point)
        ey, dy = y.read(point)
        witness = exact_witness(
            ex,
            ey,
            (*full_domains, dx, dy),
            [var.aval for var in left.invars],
            builder.budget,
            builder.element_casts,
            finite_values=x.finite and y.finite,
            domain_cache=witness_domains,
        )
        if witness is None:
            return None
        pattern, a_value, b_value = witness
        records.append(
            SympyQuery(
                output,
                f"output_coordinate={point}",
                False,
                f"exact representable input witness: {pattern}",
                str(a_value),
                str(b_value),
            )
        )
        return report(
            "inequivalent",
            tuple(records),
            output=output,
            work=builder.budget.used,
            rules=tuple(sorted(builder.rules)),
            elapsed_seconds=time.monotonic() - started,
        )

    try:
        builder.pallas_precision_paired = pair_pallas_precision(left, right, builder.budget)
        a, b = builder.graph(left), builder.graph(right)
        # 两侧输出数量、shape、dtype 必须一致；构造后再检查一次，防止适配器改变形状。
        if len(a) != len(b) or any(
            x.shape != y.shape or x.dtype != y.dtype for x, y in zip(a, b, strict=True)
        ):
            raise ComparisonContractError("algebra.outputs", "output abstract contracts differ")
        # Preflight every output before any negative result can return. An
        # earlier unequal output cannot hide an uninitialized later output.
        for output, (x, y) in enumerate(zip(a, b, strict=True)):
            if prod(x.shape) == 0:
                continue
            bounds = sp.And(*(i < n for i, n in zip(coordinates(x.shape), x.shape, strict=True)))
            # Memory initialization must hold everywhere in each output,
            # independently of matching partial real-function domains. Two
            # identical undefined lanes are never a mathematical proof.
            for side, value in (("left", x), ("right", y)):
                valid = domain_condition(value.memory_validity, builder.budget)
                if valid == sp.true:
                    continue
                answer = query(
                    output,
                    f"{side}_memory_initialization",
                    sp.And(bounds, boolean_piecewise(valid, builder.budget)),
                    bounds,
                )
                if answer is not True:
                    raise MathematicalUndecidedError(
                        "algebra.pallas.initialization",
                        "complete output memory initialization is not proved",
                    )
        # Search complete-output witnesses across all ports before allowing an
        # undecidable opaque port to consume the CAS budget for later ports.
        full_domains = tuple(
            builder.universal(
                value.defined,
                tuple(
                    (coordinate, 0, size - 1)
                    for coordinate, size in zip(coordinates(value.shape), value.shape, strict=True)
                    if size > 1
                ),
            )
            for value in (*a, *b)
            if prod(value.shape)
        )
        unconditionally_defined = all(domain == sp.true for domain in full_domains)
        totality_checked = unconditionally_defined

        def complete_domains_true() -> bool:
            nonlocal totality_checked, unconditionally_defined
            if not totality_checked:
                # Prove the original domain under each complete output's
                # coordinate bounds. Simplifying a quantified indicator first
                # can lose the bounds needed for branch-coverage tautologies.
                unconditionally_defined = True
                for output, pair in enumerate(zip(a, b, strict=True)):
                    for side, value in zip(("left", "right"), pair, strict=True):
                        if not prod(value.shape):
                            continue
                        defined = domain_condition(value.defined, builder.budget)
                        if defined == sp.true:
                            continue
                        bounds = sp.And(
                            *(
                                i < n
                                for i, n in zip(coordinates(value.shape), value.shape, strict=True)
                            )
                        )
                        answer = query(
                            output,
                            f"{side}_complete_domain",
                            sp.And(bounds, boolean_piecewise(defined, builder.budget)),
                            bounds,
                        )
                        if answer is not True:
                            unconditionally_defined = False
                            totality_checked = True
                            return False
                totality_checked = True
            return unconditionally_defined

        # Prefer a scalar formula around shared extrema, or a dynamic-read
        # formula whose page dependence can be exposed by an exact input.
        # Otherwise keep output order: a later scale may be invariant even when
        # an earlier quantized value changes. All original domains still apply.
        indirect_outputs = {
            output
            for output, pair in enumerate(zip(a, b, strict=True))
            if any(
                index.has(sp.Indexed)
                for value in pair
                for read in dag_atoms(value.expression, sp.Indexed)
                for index in read.indices
            )
        }

        def preferred_witness(item: tuple[int, tuple[SymbolicTensor, SymbolicTensor]]) -> bool:
            output, pair = item
            opaque = set().union(*(dag_atoms(v.expression, AppliedUndef) for v in pair))
            extrema = [dag_atoms(v.expression, sp.Max, sp.Min) for v in pair]
            if not opaque and extrema[0] == extrema[1]:
                return True
            # A page-dependent extremum can expose a different source before
            # nonlinear consumers. Known casts remain checked by exact_witness;
            # their presence is not a reason to search only the later consumer.
            return (
                output in indirect_outputs
                and any(extrema)
                and not any(v.expression.has(sp.exp) for v in pair)
                and all(node.func.__name__ in builder.element_casts for node in opaque)
            )

        candidates = sorted(
            enumerate(zip(a, b, strict=True)),
            key=lambda item: not preferred_witness(item),
        )
        for output, (x, y) in candidates:
            if prod(x.shape) == 0:
                continue
            large_formula = any(
                len(node.args) > 32
                for node in dag_atoms(x.expression, sp.Max, sp.Min)
                | dag_atoms(y.expression, sp.Max, sp.Min)
            )
            if large_formula and not same_bound_expression(x.expression, y.expression):
                # A shared, defined input with unequal output values is a
                # counterexample even if equality of the entire domains is
                # not yet established. The witness checks both original
                # domains exactly; it never treats unknown conditions as true.
                point = (
                    tuple(size // 2 for size in x.shape)
                    if output not in indirect_outputs
                    and (x.expression.has(AppliedUndef) or y.expression.has(AppliedUndef))
                    else (0,) * len(x.shape)
                )
                witnessed = witness_at(output, point, x, y)
                if witnessed is not None:
                    return witnessed
        undecided, domain_unknown = False, False
        for output, (x, y) in enumerate(zip(a, b, strict=True)):
            if prod(x.shape) == 0:
                continue
            bounds = sp.And(*(i < n for i, n in zip(coordinates(x.shape), x.shape, strict=True)))
            if same_bound_expression(x.defined, y.defined):
                da = db = x.defined
            else:
                da = domain_condition(x.defined, builder.budget)
                db = domain_condition(y.defined, builder.budget)
            if da != db:
                # 拼接等构造可显式携带坐标上界；域外的条件差异没有输出语义。
                # 对两侧同时限制坐标，仍保留各自对输入值的定义条件。
                da, db = (sp.And(bounds, boolean_piecewise(d, builder.budget)) for d in (da, db))
            same_domain = query(output, "definedness", da, db)
            if same_domain is not True and prod(x.shape) <= 64:
                # 抽象条件可能因绑定名或分段形式不同而无法整体化简；小输出域
                # 逐坐标复核，避免把等价条件误判为未决。
                same_domain = all(
                    query(
                        output,
                        f"definedness_coordinate={point}",
                        domain_condition(x.read(point)[1], builder.budget),
                        domain_condition(y.read(point)[1], builder.budget),
                    )
                    is True
                    for point in product(*(range(n) for n in x.shape))
                )
            if same_domain is not True:
                # Distinct partial domains do not themselves prove inequality.
                # A shared input may still distinguish a complete output;
                # witness_at checks both programs' full original domains.
                if not same_bound_expression(x.expression, y.expression):
                    small_output = prod(x.shape) <= 64
                    points = (
                        product(*(range(size) for size in x.shape))
                        if small_output
                        else _witness_points(x.shape)
                    )
                    for point in islice(points, 64 if small_output else 16):
                        witnessed = witness_at(output, point, x, y)
                        if witnessed is not None:
                            return witnessed
                domain_unknown = True
                continue
            if da == sp.false:
                # 输出处处无定义时没有可比较的值，也不能据此判等价。
                domain_unknown = True
                continue
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
            # 完整枚举上限 64 个点；大输出域取端点和中点、最多 16 个。
            # 分块计数等差异可能在前两个坐标尚未出现。这些点只用于找反例，
            # 不能作为“全部输出通过”的依据，输入值仍保持符号形式。
            points = (
                product(*(range(size) for size in x.shape))
                if fully_enumerated
                else _witness_points(x.shape)
            )
            for point in islice(points, 64 if fully_enumerated else 16):
                indices = tuple(map(sp.Integer, point))
                ex, dx = x.read(indices)
                ey, dy = y.read(indices)
                dx, dy = domain_condition(dx, builder.budget), domain_condition(dy, builder.budget)
                if dx == sp.false or equal(dx, dy, builder.budget) is not True:
                    # 该坐标下两侧定义条件不一致或未建立，不能比较值。
                    concrete_answers.append(None)
                    continue
                expandable = finite_expansion(ex, builder.budget) and finite_expansion(
                    ey, builder.budget
                )
                if not expandable or any(
                    len(node.args) > 32
                    for node in dag_atoms(ex, sp.Max, sp.Min) | dag_atoms(ey, sp.Max, sp.Min)
                ):
                    witnessed = witness_at(output, point, x, y)
                    if witnessed is not None:
                        return witnessed
                if not expandable:
                    # Compare the complete output difference before expanding
                    # two overlapping contractions. Prefix sums may cancel,
                    # leaving a small, exactly expandable counterexample.
                    if (
                        x.finite
                        and y.finite
                        and complete_domains_true()
                        and dx == dy == sp.true
                        and np.dtype(x.dtype).kind == "f"
                    ):
                        residual = simplify(ex - ey, builder.budget, expand=False)
                        if finite_expansion(residual, builder.budget):
                            explicit = explicit_symbols(residual.doit(), sp.S.Zero)
                            if explicit is not None:
                                methods = ["complete-output residual"]
                                decision = equal(*explicit, builder.budget, methods=methods)
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
                    # 显式展开超出预算时保持惰性，不调用会隐式 doit 的查询。
                    concrete_answers.append(None)
                    continue
                ex, ey = ex.doit(), ey.doit()
                methods = []
                explicit = explicit_symbols(ex, ey, methods=methods)
                if explicit is None:
                    # 不透明调用/cast 只有同余正结论可靠；泛化 False 不算反例。
                    decision = (
                        True if equal(ex, ey, builder.budget, methods=methods) is True else None
                    )
                    if decision is None:
                        # Dynamic array addresses cannot become independent
                        # scalar symbols. Specialize their source inputs and
                        # validate every read/domain before accepting a witness.
                        witnessed = witness_at(output, point, x, y)
                        if witnessed is not None:
                            return witnessed
                else:
                    decision = equal(*explicit, builder.budget, methods=methods)
                if decision is False and not complete_domains_true():
                    # A symbolic counterexample may violate another output's
                    # domain. Require one shared, exactly valid source input.
                    witnessed = witness_at(output, point, x, y)
                    if witnessed is not None:
                        return witnessed
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
            ComparisonFailureKind.CONDITION_UNDECIDED
            if domain_unknown
            else ComparisonFailureKind.CAS_UNDECIDED
            if undecided
            else None
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
                stage=ComparisonStage.ALGEBRA,
                failure_kind=ComparisonFailureKind.UNSUPPORTED_CONVERSION,
                **exc.diagnostics,
            ) from exc
        raise
    except Exception as exc:
        # 未预期异常按内部错误报告，绝不当作未决或数学结论。
        raise ComparisonInternalError(
            "algebra.sympy",
            f"{type(exc).__name__}: {exc}",
            stage=ComparisonStage.ALGEBRA,
            queries=tuple(records),
            work=builder.budget.used,
            elapsed_seconds=time.monotonic() - started,
        ) from exc


def isolated_compare(
    left: SymbolicGraph, right: SymbolicGraph, *, algebra_budget: int = 10000, timeout: float = 60.0
) -> SympyResult:
    """在独立解释器进程中调用 compare_symbolic，并用墙钟上限兜底无界化简。

    输入是已冻结的图数据，不执行 JAX 数组；超时先 kill 并回收 worker，非零退出
    和非法响应转为分类异常，保证后续比较可以继续。输入/结果经 pickle 传输，
    worker 失败时以 WorkerFailure 保留分类、诊断和远端 traceback。
    """
    validate_timeout(timeout)
    started = time.monotonic()
    try:
        payload = pickle.dumps((left, right, algebra_budget))
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.transport", f"symbolic transport failed: {exc}", stage=ComparisonStage.ALGEBRA
        ) from exc
    environment = os.environ.copy()
    # 覆盖继承自父进程的 BLAS/OpenMP 线程配置，固定 worker 的数值库线程数。
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "jax_reorder.algebra.trusted"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        raise ComparisonInternalError(
            "algebra.sympy.worker", f"worker could not start: {exc}", stage=ComparisonStage.ALGEBRA
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
            "algebra.sympy.worker",
            f"worker communication failed: {exc}",
            stage=ComparisonStage.ALGEBRA,
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
            stage=ComparisonStage.ALGEBRA,
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
            "algebra.sympy.worker", f"invalid worker response: {exc}", stage=ComparisonStage.ALGEBRA
        ) from exc
    if isinstance(result, WorkerFailure):
        # 用 RemoteWorkerError 作为 cause 保留远端 traceback，同时保持异常类型不变。
        raise cast(ComparisonError, error) from RemoteWorkerError(result.traceback)
    # 父进程墙钟覆盖 worker 内部计时，包含进程启动与传输时间。
    return replace(result, elapsed_seconds=time.monotonic() - started)


def _worker_main() -> None:
    """worker 进程入口：读取 pickle 输入，写回结果或失败消息。"""
    # 使用规范模块名序列化类型，父进程无需导入 __main__ 中的类。
    from jax_reorder.algebra.trusted import WorkerFailure, compare_symbolic

    try:
        payload = pickle.load(sys.stdin.buffer)
        result = compare_symbolic(*payload)
    except Exception as exc:
        # 任何异常都转成 WorkerFailure：非 ComparisonError 统一按内部错误分类。
        error = (
            exc
            if isinstance(exc, ComparisonError)
            else ComparisonInternalError(
                "algebra.sympy.worker",
                f"{type(exc).__name__}: {exc}",
                stage=ComparisonStage.ALGEBRA,
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
