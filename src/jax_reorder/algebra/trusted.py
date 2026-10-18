"""用可信 SymPy 查询数学等价，并给出可终止的进程边界。

algebra.symbolic 负责把两份图构造成完整公式和原始定义条件；本模块负责在
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
from math import prod

import sympy as sp

from ..errors import (
    ComparisonContractError,
    ComparisonError,
    ComparisonInternalError,
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    UnsupportedJaxprError,
)
from .budget import validate_timeout
from .expression import coordinates
from .symbolic import SymbolicBuilder
from .sympy_ops import domain_condition, equal, explicit_symbols, finite_expansion


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
                da = domain_condition(x.defined, builder.budget)
                db = domain_condition(y.defined, builder.budget)
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
                dx, dy = domain_condition(dx, builder.budget), domain_condition(dy, builder.budget)
                if dx == sp.false or equal(dx, dy, builder.budget) is not True:
                    # 该坐标下两侧定义条件不一致或未建立，不能比较值。
                    concrete_answers.append(None)
                    continue
                if not (
                    finite_expansion(ex, builder.budget) and finite_expansion(ey, builder.budget)
                ):
                    # 显式展开超出预算时保持惰性，不调用会隐式 doit 的查询。
                    concrete_answers.append(None)
                    continue
                ex, ey = ex.doit(), ey.doit()
                explicit = explicit_symbols(ex, ey)
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
    validate_timeout(timeout)
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
            [sys.executable, "-m", "jax_reorder.algebra.trusted"],
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
