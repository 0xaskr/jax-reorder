"""代数比较的编排层：结构能对上就结束，对不上再交给 SymPy。

流程是：先做结构比较，结构等价就直接返回，不进入数学路径，也不加浮点警告；
结构不同时再核对输入输出契约、找出两侧共同且完整相同的纯 JIT 调用，把图
冻结后交给独立 worker 判定，最后汇总成结果。数学结论只来自 worker，本模块
不执行 JAX 数组，也不自己推导恒等式。未完成和失败都用异常报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from jax._src import core

from ..core.canonical import canonicalize, compare_jaxprs
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

if TYPE_CHECKING:
    from ..core.normalization import RewriteEvidence
    from .trusted import SympyResult

# 每条代数结论都随结果一起报告这些前提；它们限定结论成立的范围，
# 也说明结构快路径为何可以替换成不带浮点警告的结构等价。
ASSUMPTIONS = (
    "finite floating-point subexpressions are interpreted as ideal real numbers",
    "expressions expanded algebraically are finite and defined in the declared real interpretation",
    "negative infinity constants and possibly non-finite dependent operations "
    "retain their semantics",
    "only the supplied input signature, static arguments and trace environment are covered",
)
# 任何实际依赖代数改写的结论都必须携带这条浮点警告；结构等价不需要。
FLOAT_WARNING = (
    "Algebraic comparison uses ideal real arithmetic. Floating-point commutation, "
    "reassociation and distribution may change rounding, overflow, NaN/Inf and signed zero; "
    "neither small error nor bitwise equivalence is guaranteed."
)


@dataclass(frozen=True)
class AlgebraComparison:
    """结构对应或可信 SymPy 数学结果，按实际判定路径记录依据。

    correspondence 只在结构快路径填充，backend_result 只在数学路径填充，
    两者恰好有一个非 None。work 是实际消耗的预算单位数，不含结构比较。
    assumptions 与 warnings 描述该条结论的适用边界；rules 记录用到的规则名。
    normalizations 按左右侧保存 Pallas 的完整值投影及状态消除前提；普通 JAX
    侧为 None，没有 Pallas 转换时为空。它不替代 backend_result 的数学判定。
    失败（未完成、预算、超时、内部错误）不会进入本对象，而是抛出异常。
    """

    status: Status
    rules: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ASSUMPTIONS
    warnings: tuple[str, ...] = (FLOAT_WARNING,)
    correspondence: GraphCorrespondence | None = field(default=None, repr=False)
    work: int = 0
    backend_result: SympyResult | None = field(default=None, repr=False)
    normalizations: tuple[RewriteEvidence | None, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        """拒绝未完成状态：失败只能经异常通道报告，不能伪装成结果。"""
        # 未完成或失败必须走异常通道，不能退化成第四种结果状态被调用方忽略。
        if self.status not in (Status.EQUIVALENT, Status.INEQUIVALENT):
            raise ValueError(f"invalid complete comparison status: {self.status}")

    @property
    def equivalence_kind(self) -> Literal["structural", "algebraic"] | None:
        """区分结构证据与可信 SymPy 数学结论。"""
        if self.status != Status.EQUIVALENT:
            return None
        return "structural" if self.correspondence is not None else "algebraic"

    @property
    def output(self) -> int | None:
        """数学路径判定所用的完整输出坐标；结构路径没有单一输出，返回 None。"""
        if self.backend_result is not None:
            return self.backend_result.output
        return None


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
) -> AlgebraComparison:
    """以可信 SymPy 比较两份完整前向 jaxpr 的数学输出。

    先做结构比较：结构等价直接返回结构证据，不消耗代数预算，也不进入实数
    解释。结构不同时先检查输入输出抽象契约；普通 JAX 识别共同纯 JIT，Pallas
    则要求中间层完成全局值投影。冻结后在可终止的独立 worker 中查询；其 True/False 是
    唯一数学结论，失败以异常报告。algebra_timeout 只覆盖 worker 内的转换
    与查询，不包含 JAX tracing 和父进程结构比较。
    """
    from .adapter.transport import freeze
    from .trusted import isolated_compare

    Budget(algebra_budget)
    validate_timeout(algebra_timeout)
    structural = compare_jaxprs(left, right, include_derivative_rules=False)
    if structural.status == Status.EQUIVALENT:
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
        assumptions = ASSUMPTIONS
        if any(has_pallas):
            converted, evidence = [], []
            identities = {}
            for side, graph, present in zip(("left", "right"), (a, b), has_pallas, strict=True):
                try:
                    value, proof = (
                        project(graph, budget)
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
                    if proof and proof.remote is not None
                    for value in proof.remote.protocol.protocol.assumptions
                )
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
