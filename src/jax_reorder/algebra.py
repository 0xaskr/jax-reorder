"""理想实数解释下的索引代数比较；不改变原有结构比较入口。

本模块是代数后端的编排层：先复用结构等价结论，再对结构不同的图做完整语义
校验，把图和共同调用冻结后交给独立 SymPy worker 判定，最后汇总结果。
数学结论只来自 worker；本模块不执行 JAX 数组，也不自行推导恒等式。
结果对象只表示已完成的结构等价、数学等价或数学不等价；未完成与失败以异常报告。
"""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .algebra_budget import Budget
from .algebra_calls import CommonCalls
from .canonical import Issue, Status, canonicalize, compare_jaxprs
from .correspondence import GraphCorrespondence, build_correspondence
from .errors import (
    ComparisonContractError,
    ComparisonError,
    ComparisonInternalError,
    UnsupportedJaxprError,
)

if TYPE_CHECKING:
    from .algebra_trusted import SympyResult

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
    失败（未完成、预算、超时、内部错误）不会进入本对象，而是抛出异常。
    """

    status: Status
    issues: tuple[Issue, ...] = ()
    rules: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ASSUMPTIONS
    warnings: tuple[str, ...] = (FLOAT_WARNING,)
    correspondence: GraphCorrespondence | None = field(default=None, repr=False)
    work: int = 0
    backend_result: "SympyResult | None" = field(default=None, repr=False)

    def __post_init__(self):
        """拒绝未完成状态：失败只能经异常通道报告，不能伪装成结果。"""
        # 未完成或失败必须走异常通道，不能退化成第四种结果状态被调用方忽略。
        if self.status not in (Status.EQUIVALENT, Status.INEQUIVALENT):
            raise ValueError(f"invalid complete comparison status: {self.status}")

    @property
    def equivalence_kind(self):
        """区分结构证据与可信 SymPy 数学结论。"""
        if self.status != Status.EQUIVALENT:
            return None
        return "structural" if self.correspondence is not None else "algebraic"

    @property
    def output(self):
        """数学路径判定所用的完整输出坐标；结构路径没有单一输出，返回 None。"""
        if self.backend_result is not None:
            return self.backend_result.output
        return None


def _prepare(graph):
    """完成全图语义检查，并生成原图到区域适配图的结构对应证据。

    第一次规范化确认没有未知语义或未完成部分，并得到可执行图；
    第二次规范化适配后的 jaxpr，为 build_correspondence 提供适配图一侧的
    来源记录（lineage）和区域视图。原图索引与适配图索引不能混用。
    """
    normalized = canonicalize(graph, derivative_scope="primal")
    if not normalized.complete:
        issue = normalized.issues[0]
        raise UnsupportedJaxprError(issue.path, issue.reason)
    again = canonicalize(normalized.jaxpr, derivative_scope="primal")
    if not again.complete:
        issue = again.issues[0]
        raise UnsupportedJaxprError(issue.path, issue.reason)
    correspondence = build_correspondence(graph, normalized.jaxpr, normalized, again)
    return normalized.jaxpr, correspondence


def validate_timeout(value):
    """要求正整数或浮点数的有限正超时；布尔值被显式排除。

    用 type 精确比较而不是 isinstance，避免 True/False 被当成 int；
    无穷与 NaN 也不能表达 worker 的墙钟上限。
    """
    if type(value) not in (int, float) or not 0 < value < float("inf"):
        raise ValueError("algebra_timeout must be a positive finite number")


def compare_algebraic_jaxprs(left, right, *, algebra_budget=10000, algebra_timeout=10.0):
    """以可信 SymPy 比较两份完整前向 jaxpr 的数学输出。

    先做结构比较：结构等价直接返回结构证据，不消耗代数预算，也不进入实数
    解释。结构不同时依次检查输入输出抽象契约、共同纯 JIT 调用、进程边界
    输入冻结，然后在可终止的独立 worker 中查询；worker 的 True/False 是
    唯一数学结论，失败以异常报告。algebra_timeout 只覆盖 worker 内的转换
    与查询，不包含 JAX tracing 和父进程结构比较。
    """
    from .algebra_transport import freeze
    from .algebra_trusted import isolated_compare

    Budget(algebra_budget)
    validate_timeout(algebra_timeout)
    structural = compare_jaxprs(left, right, derivative_scope="primal")
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
        from .algebra_semantics import encode

        # 数学阶段只能在同一输入/输出抽象契约上比较；否则差值没有同一含义。
        if tuple(encode(v.aval) for v in a.invars + a.outvars) != tuple(
            encode(v.aval) for v in b.invars + b.outvars
        ) or len(a.invars) != len(b.invars):
            raise ComparisonContractError(
                "algebra.contract", "input or output abstract contract differs"
            )
        # 共同调用键在父进程、以结构证据为输入构造，只覆盖两侧完整相同的纯 JIT。
        calls = CommonCalls(
            tuple(
                (original, *adapted)
                for original, adapted in zip((left, right), prepared, strict=True)
            ),
            Budget(algebra_budget),
        )
        identities = {}
        # 两侧共享 id 表，使同一共同调用/不透明 reducer 在 worker 中得到同一符号。
        frozen = freeze(a, calls, identities), freeze(b, calls, identities)
        # worker 进程本身记 1 单位；下面用 -1/+1 把这 1 单位与父进程工作量一起结算。
        calls.budget.spend(path="algebra.worker")
        result = isolated_compare(
            *frozen,
            algebra_budget=algebra_budget - calls.budget.used + 1,
            timeout=algebra_timeout,
        )
        # 父进程已用的预算（含 worker 那 1 单位）加回 worker 报告的工作量。
        result = replace(result, work=result.work + calls.budget.used - 1)
        return AlgebraComparison(
            Status(result.status), rules=result.rules, work=result.work, backend_result=result
        )
    except UnsupportedJaxprError as exc:
        raise UnsupportedJaxprError(
            exc.path,
            exc.reason,
            stage="algebra",
            failure_kind="unsupported_conversion",
            **exc.diagnostics,
        ) from exc
    except ComparisonError:
        raise
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.convert", f"{type(exc).__name__}: {exc}", stage="algebra"
        ) from exc
