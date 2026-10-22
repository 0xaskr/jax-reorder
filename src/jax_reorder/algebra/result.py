"""Completed comparison results shared by full-function and local orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..core.correspondence import GraphCorrespondence
from ..status import Status

if TYPE_CHECKING:
    from ..core.core_execution import CoreExecutionEvidence
    from ..core.local import LocalKernelEvidence
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

    全函数结构快路径填 correspondence，局部结构/规范化填 local_evidence，
    数学路径填 backend_result。work 是实际消耗的预算单位数，不含源结构比较。
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
    scope: str = "function"
    local_evidence: LocalKernelEvidence | None = field(default=None, repr=False)
    core_executions: tuple[CoreExecutionEvidence | None, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        """拒绝未完成状态：失败只能经异常通道报告，不能伪装成结果。"""
        # 未完成或失败必须走异常通道，不能退化成第四种结果状态被调用方忽略。
        if self.status not in (Status.EQUIVALENT, Status.INEQUIVALENT):
            raise ValueError(f"invalid complete comparison status: {self.status}")

    @property
    def equivalence_kind(self) -> Literal["structural", "normalized", "algebraic"] | None:
        """区分结构证据与可信 SymPy 数学结论。"""
        if self.status != Status.EQUIVALENT:
            return None
        if self.local_evidence is not None:
            return "structural" if self.local_evidence.kind == "structural" else "normalized"
        return "structural" if self.correspondence is not None else "algebraic"

    @property
    def output(self) -> int | None:
        """数学路径判定所用的完整输出坐标；结构路径没有单一输出，返回 None。"""
        if self.backend_result is not None:
            return self.backend_result.output
        return None
