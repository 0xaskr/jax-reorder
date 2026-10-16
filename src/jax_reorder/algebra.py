"""理想实数解释下的索引代数比较；不改变原有结构比较入口。"""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .algebra_budget import Budget
from .algebra_calls import CommonCalls
from .canonical import Issue, Status, canonicalize, compare_jaxprs
from .correspondence import GraphCorrespondence, build_correspondence
from .errors import JaxprError, UnsupportedJaxprError

if TYPE_CHECKING:
    from .algebra_trusted import SympyResult

ASSUMPTIONS = (
    "finite floating-point subexpressions are interpreted as ideal real numbers",
    "expressions expanded algebraically are finite and defined in the declared real interpretation",
    "negative infinity constants and possibly non-finite dependent operations "
    "retain their semantics",
    "only the supplied input signature, static arguments and trace environment are covered",
)
FLOAT_WARNING = (
    "Algebraic comparison uses ideal real arithmetic. Floating-point commutation, "
    "reassociation and distribution may change rounding, overflow, NaN/Inf and signed zero; "
    "neither small error nor bitwise equivalence is guaranteed."
)


@dataclass(frozen=True)
class AlgebraComparison:
    """结构对应或可信 SymPy 数学结果，按实际判定路径记录依据。"""

    status: Status
    issues: tuple[Issue, ...] = ()
    rules: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ASSUMPTIONS
    warnings: tuple[str, ...] = (FLOAT_WARNING,)
    correspondence: GraphCorrespondence | None = field(default=None, repr=False)
    work: int = 0
    backend_result: "SympyResult | None" = field(default=None, repr=False)

    @property
    def equivalence_kind(self):
        """区分结构证据与可信 SymPy 数学结论。"""
        if self.status != Status.EQUIVALENT:
            return None
        return "structural" if self.correspondence is not None else "algebraic"

    @property
    def output(self):
        if self.backend_result is not None:
            return self.backend_result.output
        return None


def _prepare(graph):
    """完成全图语义检查，并生成原图到区域适配图的结构对应证据。"""
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
    if type(value) not in (int, float) or not 0 < value < float("inf"):
        raise ValueError("algebra_timeout must be a positive finite number")


def compare_algebraic_jaxprs(left, right, *, algebra_budget=10000, algebra_timeout=10.0):
    """Compare complete mathematical outputs with trusted SymPy, after structural validation."""
    from .algebra_transport import freeze
    from .algebra_trusted import SympyResult, isolated_compare

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
    if structural.status == Status.UNKNOWN:
        return AlgebraComparison(Status.UNKNOWN, structural.issues)
    try:
        # Adapt graph bindings before constructing the corresponding symbolic inputs.
        prepared = tuple(_prepare(graph) for graph in (left, right))
        a, b = (entry[0] for entry in prepared)
        from .algebra_semantics import encode

        if tuple(encode(v.aval) for v in a.invars + a.outvars) != tuple(
            encode(v.aval) for v in b.invars + b.outvars
        ) or len(a.invars) != len(b.invars):
            raise UnsupportedJaxprError(
                "algebra.contract", "input or output abstract contract differs"
            )
        calls = CommonCalls(
            tuple(
                (original, *adapted)
                for original, adapted in zip((left, right), prepared, strict=True)
            ),
            Budget(algebra_budget),
        )
        identities = {}
        frozen = freeze(a, calls, identities), freeze(b, calls, identities)
        calls.budget.spend(path="algebra.worker")
        result = isolated_compare(
            *frozen,
            algebra_budget=algebra_budget - calls.budget.used + 1,
            timeout=algebra_timeout,
        )
        result = replace(result, work=result.work + calls.budget.used - 1)
        issues = () if result.reason is None else (Issue(result.path, result.reason),)
        return AlgebraComparison(
            Status(result.status), issues, result.rules, work=result.work, backend_result=result
        )
    except JaxprError as exc:
        result = SympyResult(
            "unknown",
            reason=exc.reason,
            path=exc.path,
            failure_kind="budget_exhausted" if "budget" in exc.reason else "unsupported_conversion",
        )
        return AlgebraComparison(
            Status.UNKNOWN, (Issue(exc.path, exc.reason),), backend_result=result
        )
    except (ValueError, TypeError, RecursionError) as exc:
        result = SympyResult(
            "unknown",
            reason=f"conversion rejected: {exc}",
            path="algebra.convert",
            failure_kind="unsupported_conversion",
        )
        return AlgebraComparison(
            Status.UNKNOWN, (Issue(result.path, result.reason),), backend_result=result
        )
