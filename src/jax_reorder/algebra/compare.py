"""代数比较的编排层：结构能对上就结束，对不上再交给 SymPy。

流程是：先做结构比较，结构等价且没有显式执行条件就直接返回，不进入数学路径，
也不加浮点警告；显式执行绑定始终先检查对应协议并保留证据。
结构不同时再核对输入输出契约、找出两侧共同且完整相同的纯 JIT 调用，把图
冻结后交给独立 worker 判定，最后汇总成结果。数学结论只来自 worker，本模块
不执行 JAX 数组，也不自己推导恒等式。未完成和失败都用异常报告。
"""

from __future__ import annotations

from dataclasses import replace

from jax._src import core

from ..core.canonical import canonicalize, compare_jaxprs
from ..core.core_execution import CoreExecutionBinding, validate_binding_shape
from ..core.correspondence import GraphCorrespondence, build_correspondence
from ..errors import (
    ComparisonContractError,
    ComparisonError,
    ComparisonFailureKind,
    ComparisonInternalError,
    ComparisonStage,
    UnsupportedJaxprError,
)
from ..status import Status
from .adapter.calls import CommonCalls
from .budget import Budget, validate_timeout
from .result import ASSUMPTIONS, AlgebraComparison


def _prepare(graph: core.Jaxpr) -> tuple[core.Jaxpr, GraphCorrespondence]:
    """完成全图语义检查，并生成原图到区域适配图的结构对应证据。

    第一次规范化确认没有未知语义，并得到可执行图；第二次规范化适配后的
    jaxpr，为 build_correspondence 提供适配图一侧的来源记录（lineage）和
    区域视图。原图索引与适配图索引不能混用。
    """
    normalized = canonicalize(graph, include_derivative_rules=False)
    again = canonicalize(normalized.jaxpr, include_derivative_rules=False)
    correspondence = build_correspondence(graph, normalized.jaxpr, normalized, again)
    return normalized.jaxpr, correspondence


def compare_algebraic_jaxprs(
    left: core.Jaxpr,
    right: core.Jaxpr,
    *,
    algebra_budget: int = 10000,
    algebra_timeout: float = 60.0,
    scope: str = "function",
    execution: CoreExecutionBinding
    | tuple[CoreExecutionBinding, CoreExecutionBinding]
    | None = None,
) -> AlgebraComparison:
    """以可信 SymPy 比较两份完整前向 jaxpr 的数学输出。

    先做结构比较：没有显式执行绑定时，结构等价直接返回结构证据，不消耗代数预算，也不进入实数
    解释。结构不同时先检查输入输出抽象契约；普通 JAX 识别共同纯 JIT，Pallas
    则要求中间层完成全局值投影。冻结后在可终止的独立 worker 中查询；其 True/False 是
    唯一数学结论，失败以异常报告。algebra_timeout 只覆盖 worker 内的转换
    与查询，不包含 JAX tracing 和父进程结构比较。

    scope="device_local" 比较完整父图中提取的一个 Pallas kernel，保留实参来源、
    有界设备环境与协议证明；不把父函数输出组装或后处理作为局部等价的前提。
    此模式先完成资源证明，再尝试局部结构/语义快路径，最后才进入数学 worker。

    execution 为两侧共用的远端单 core 绑定，或分别属于左/右侧的绑定；调用方
    必须保证实际执行遵守该顺序。显式绑定时，两侧都先完成协议投影并保留证据，
    包括结构相同的情况；scope="device_local" 另接受已接入证明的本地多 core
    条件并保留 core_executions，普通 JAX 不接受该条件。
    """
    from .adapter.transport import freeze
    from .trusted import isolated_compare

    Budget(algebra_budget)
    validate_timeout(algebra_timeout)
    if scope not in ("function", "device_local"):
        raise ValueError("scope must be function or device_local")
    if execution is None:
        executions = (None, None)
    elif isinstance(execution, CoreExecutionBinding):
        executions = (execution, execution)
    elif (
        type(execution) is tuple
        and len(execution) == 2
        and all(isinstance(item, CoreExecutionBinding) for item in execution)
    ):
        executions = execution
    else:
        raise ValueError("execution requires one CoreExecutionBinding or a left/right binding pair")
    for binding in executions:
        if binding is not None:
            validate_binding_shape(binding)
    if scope == "device_local":
        from .local import compare_local_values

        return compare_local_values(
            left,
            right,
            algebra_budget=algebra_budget,
            algebra_timeout=algebra_timeout,
            executions=executions,
        )
    structural = compare_jaxprs(left, right, include_derivative_rules=False)
    if structural.status == Status.EQUIVALENT and execution is None:
        return AlgebraComparison(
            Status.EQUIVALENT,
            rules=("structural",),
            assumptions=("structural equivalence of the supplied forward jaxprs",),
            warnings=(),
            correspondence=structural.correspondence,
        )
    try:
        # 先完成两侧区域适配，再按适配后的端口构造符号输入。
        prepared = tuple(_prepare(graph) for graph in (left, right))
        a, b = (entry[0] for entry in prepared)
        from .adapter.semantics import encode

        # 数学阶段只能在同一输入/输出抽象契约上比较；否则差值没有同一含义。
        if tuple(encode(v.aval) for v in a.invars + a.outvars) != tuple(
            encode(v.aval) for v in b.invars + b.outvars
        ) or len(a.invars) != len(b.invars):
            raise ComparisonContractError(
                "algebra.contract", "input or output abstract contract differs"
            )
        from .adapter.pallas import contains_pallas, project

        budget = Budget(algebra_budget)
        has_pallas = contains_pallas(a), contains_pallas(b)
        normalizations = ()
        assumptions = (
            ("structural equivalence of the supplied forward jaxprs",)
            if structural.status == Status.EQUIVALENT
            else ASSUMPTIONS
        )
        if execution is not None:
            if not all(has_pallas):
                side = ("left", "right")[has_pallas.index(False)]
                raise UnsupportedJaxprError(
                    f"{side}.algebra.pallas",
                    "execution mapping requires a remote Pallas protocol on both sides",
                )
            assumptions += (
                "caller binds every device to the supplied core-0 program stream for each source; "
                "ordinary pallas_call does not enforce this execution mapping and order",
            )
        if any(has_pallas):
            converted, evidence = [], []
            identities = {}
            for side, graph, present, binding in zip(
                ("left", "right"), (a, b), has_pallas, executions, strict=True
            ):
                try:
                    value, proof = (
                        project(graph, budget, execution=binding)
                        if present
                        else (freeze(graph, common_ids=identities), None)
                    )
                except ComparisonError as error:
                    raise error.at(side) from error
                converted.append(value)
                evidence.append(proof)
            frozen = tuple(converted)
            normalizations = tuple(evidence)
            if any(proof and proof.projection.communication for proof in evidence):
                from ..core.communication import DMA_PROGRESS_ASSUMPTION

                assumptions += (DMA_PROGRESS_ASSUMPTION,)
            assumptions += tuple(
                dict.fromkeys(
                    value
                    for proof in evidence
                    if proof is not None
                    for value in proof.projection.assumptions
                )
            )
            assumptions += tuple(
                dict.fromkeys(
                    value
                    for proof in evidence
                    if proof and proof.remote is not None
                    for value in proof.remote.protocol.protocol.assumptions
                )
            )
            if structural.status == Status.EQUIVALENT:
                return AlgebraComparison(
                    Status.EQUIVALENT,
                    rules=("structural",),
                    assumptions=assumptions,
                    warnings=(),
                    correspondence=structural.correspondence,
                    work=budget.used,
                    normalizations=normalizations,
                )
        else:
            # 共同调用只涵盖完整相同的纯 JIT，不能将状态调用当作不透明纯函数。
            calls = CommonCalls(
                tuple(
                    (original, *adapted)
                    for original, adapted in zip((left, right), prepared, strict=True)
                ),
                budget,
            )
            identities = {}
            frozen = freeze(a, calls, identities), freeze(b, calls, identities)
        # worker 进程本身记 1 单位；下面用 -1/+1 把这 1 单位与父进程工作量一起结算。
        budget.spend(path="algebra.worker")
        result = isolated_compare(
            *frozen,
            algebra_budget=algebra_budget - budget.used + 1,
            timeout=algebra_timeout,
        )
        # 父进程已用的预算（含 worker 那 1 单位）加回 worker 报告的工作量。
        result = replace(result, work=result.work + budget.used - 1)
        return AlgebraComparison(
            Status(result.status),
            rules=result.rules,
            work=result.work,
            backend_result=result,
            normalizations=normalizations,
            assumptions=assumptions,
        )
    except UnsupportedJaxprError as exc:
        raise UnsupportedJaxprError(
            exc.path,
            exc.reason,
            stage=ComparisonStage.ALGEBRA,
            failure_kind=ComparisonFailureKind.UNSUPPORTED_CONVERSION,
            **exc.diagnostics,
        ) from exc
    except ComparisonError:
        raise
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.convert", f"{type(exc).__name__}: {exc}", stage=ComparisonStage.ALGEBRA
        ) from exc
