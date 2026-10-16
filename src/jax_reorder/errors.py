"""比较失败的分类异常；失败不作为正常等价结论返回。"""


class ComparisonError(ValueError):
    """统一异常边界，携带阶段、位置、原因和可选查询诊断。"""

    default_stage = "comparison"
    default_kind = "internal_error"

    def __init__(self, path, reason, *, stage=None, failure_kind=None, **diagnostics):
        self.path = path
        self.reason = reason
        self.stage = stage or self.default_stage
        self.failure_kind = failure_kind or self.default_kind
        self.diagnostics = diagnostics
        super().__init__(f"{self.stage}:{path} [{self.failure_kind}]: {reason}")

    def at(self, prefix):
        """边界补充侧别或模式，不改写底层异常和其诊断。"""
        return type(self)(
            f"{prefix}.{self.path}",
            self.reason,
            stage=self.stage,
            failure_kind=self.failure_kind,
            **self.diagnostics,
        )


class JaxprError(ComparisonError):
    """图处理错误；保留原有基类以便调用方集中捕获图错误。"""

    default_stage = "structural"


class InvalidJaxprError(JaxprError):
    """非法 jaxpr，例如悬空引用、重复定义或 JAX 类型检查失败。"""

    default_kind = "invalid_jaxpr"


class UnsupportedJaxprError(JaxprError):
    """尚不能完整解释的 primitive、参数或区域语义。"""

    default_kind = "unsupported_semantics"


class CorrespondenceError(JaxprError):
    """结构对应证据校验失败，属于比较器内部错误。"""


class ComparisonContractError(ComparisonError):
    """签名、模式或环境不兼容，比较前提不成立。"""

    default_stage = "contract"
    default_kind = "contract_mismatch"


class MathematicalUndecidedError(ComparisonError):
    """数学查询或必要定义条件仍未决，不能据此判不等价。"""

    default_stage = "algebra"
    default_kind = "cas_undecided"


class ComparisonBudgetError(ComparisonError):
    """确定性工作量或实现资源边界耗尽。"""

    default_kind = "budget_exhausted"


class ComparisonTimeoutError(ComparisonError):
    """数学 worker 超时，已终止并回收。"""

    default_stage = "algebra"
    default_kind = "timeout"


class ComparisonInternalError(ComparisonError):
    """转换、比较或 worker 运行故障，保留原始异常链。"""


class TraceError(ComparisonError):
    """指定函数签名或变换无法追踪。"""

    default_stage = "trace"
    default_kind = "trace_rejected"


class CompilationError(ComparisonError):
    """可选的 CPU 编译检查失败。"""

    default_stage = "compile"
    default_kind = "cpu_compile_rejected"
