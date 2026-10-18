"""比较失败时抛出的分类异常。

失败一律通过抛异常报告，不会伪装成“等价”或“不同”的正常结果。
每个异常把这些信息存成属性，而不是只拼进报错文本：
- stage：在哪个阶段失败（structural / contract / algebra / trace / compile）；
- path：错误在图里的位置，例如 jaxpr.eqns[3].params.jaxpr；
- failure_kind：失败类别；diagnostics：附加上下文（预算用量、超时秒数等）。

底层只知道自己所在的子图，路径是相对的。比较入口在跨边界时用 at(prefix)
在前面补上 left / right 或模式名（例如 left.vjp），异常类型和诊断保持不变。
调用方因此可以按类型或属性处理错误，不必解析异常字符串。

类型分开是为了区分“输入或调用方式有问题”（非法图、契约不兼容、追踪失败）
和“实现或资源边界”（不支持语义、内部错误、预算耗尽、超时），好决定是跳过
这个用例、报错给用户，还是补实现。
"""

from enum import StrEnum
from typing import ClassVar, Self


class ComparisonStage(StrEnum):
    """比较处理阶段；与同值字符串兼容，新增阶段在此集中声明。"""

    COMPARISON = "comparison"
    STRUCTURAL = "structural"
    CONTRACT = "contract"
    ALGEBRA = "algebra"
    TRACE = "trace"
    COMPILE = "compile"


class ComparisonFailureKind(StrEnum):
    """失败分类；异常子类选择默认分类，诊断字段保留额外上下文。"""

    INTERNAL_ERROR = "internal_error"
    INVALID_JAXPR = "invalid_jaxpr"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    CONTRACT_MISMATCH = "contract_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    CAS_UNDECIDED = "cas_undecided"
    CONDITION_UNDECIDED = "condition_undecided"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RECURSION_LIMIT = "recursion_limit"
    TIMEOUT = "timeout"
    UNSUPPORTED_CONVERSION = "unsupported_conversion"
    TRACE_REJECTED = "trace_rejected"
    CPU_COMPILE_REJECTED = "cpu_compile_rejected"


# Base Class
class ComparisonError(ValueError):
    """统一异常边界，携带阶段、位置、原因和可选查询诊断。"""

    default_stage: ClassVar[ComparisonStage] = ComparisonStage.COMPARISON
    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.INTERNAL_ERROR

    path: str
    reason: str
    stage: ComparisonStage
    failure_kind: ComparisonFailureKind
    diagnostics: dict[str, object]

    def __init__(
        self,
        path: str,
        reason: str,
        *,
        stage: ComparisonStage | str | None = None,
        failure_kind: ComparisonFailureKind | str | None = None,
        **diagnostics: object,
    ) -> None:
        """保存位置、原因和分类，并把剩余关键字参数原样保留为诊断字段。

        stage/failure_kind 接受枚举或已声明的字符串值，省略时取类默认值；
        at() 重建异常依赖这些已保存字段，
        因此异常字符串只是展示形式，不作为后续解析依据。
        """
        self.path = path
        self.reason = reason
        self.stage = ComparisonStage(stage or self.default_stage)
        self.failure_kind = ComparisonFailureKind(failure_kind or self.default_kind)
        self.diagnostics = diagnostics
        super().__init__(f"{self.stage}:{path} [{self.failure_kind}]: {reason}")

    def at(self, prefix: str) -> Self:
        """边界补充侧别或模式，不改写底层异常和其诊断。"""
        return type(self)(
            f"{prefix}.{self.path}",
            self.reason,
            stage=self.stage,
            failure_kind=self.failure_kind,
            **self.diagnostics,
        )


class JaxprError(ComparisonError):
    """图处理错误的基类；默认阶段为 structural，便于集中捕获非法或不支持图。"""

    default_stage: ClassVar[ComparisonStage] = ComparisonStage.STRUCTURAL


class InvalidJaxprError(JaxprError):
    """非法 jaxpr，例如悬空引用、重复定义或 JAX 类型检查失败。"""

    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.INVALID_JAXPR


class UnsupportedJaxprError(JaxprError):
    """尚不能完整解释的 primitive、参数或区域语义。"""

    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.UNSUPPORTED_SEMANTICS


class CorrespondenceError(JaxprError):
    """原始图对应证据缺失或不保持绑定，按比较器内部错误处理。"""


class ComparisonContractError(ComparisonError):
    """签名、模式或环境不兼容，比较前提不成立。"""

    default_stage: ClassVar[ComparisonStage] = ComparisonStage.CONTRACT
    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.CONTRACT_MISMATCH


class MathematicalUndecidedError(ComparisonError):
    """数学查询或必要定义条件仍未决，不能据此判不等价。"""

    default_stage: ClassVar[ComparisonStage] = ComparisonStage.ALGEBRA
    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.CAS_UNDECIDED


class ComparisonBudgetError(ComparisonError):
    """确定性工作量或实现资源边界耗尽。"""

    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.BUDGET_EXHAUSTED


class ComparisonTimeoutError(ComparisonError):
    """数学 worker 超时，已终止并回收。"""

    default_stage: ClassVar[ComparisonStage] = ComparisonStage.ALGEBRA
    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.TIMEOUT


class ComparisonInternalError(ComparisonError):
    """转换、比较或 worker 运行故障，保留原始异常链。"""


class TraceError(ComparisonError):
    """指定函数签名或变换无法追踪。"""

    default_stage: ClassVar[ComparisonStage] = ComparisonStage.TRACE
    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.TRACE_REJECTED


class CompilationError(ComparisonError):
    """可选的 CPU 编译检查失败。"""

    default_stage: ClassVar[ComparisonStage] = ComparisonStage.COMPILE
    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.CPU_COMPILE_REJECTED
