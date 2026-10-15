"""理想实数解释下的索引代数比较；不改变原有结构比较入口。"""

from dataclasses import dataclass, field

from .algebra_calls import CommonCalls
from .algebra_inequivalence import PolynomialInequivalenceProof, try_prove_inequivalent
from .algebra_ir import Budget
from .algebra_lower import Lowerer
from .algebra_normalize import PolynomialNormalizer
from .algebra_proof import AlgebraProof, AlgebraSource, verify_algebraic_proof
from .canonical import Issue, Status, canonicalize, compare_jaxprs
from .correspondence import build_correspondence
from .errors import JaxprError, UnsupportedJaxprError

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
    """带独立正/负证据的数学比较；证明失败保留为 unknown。"""

    status: Status
    issues: tuple[Issue, ...] = ()
    rules: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ASSUMPTIONS
    warnings: tuple[str, ...] = (FLOAT_WARNING,)
    proof: AlgebraProof | None = field(default=None, repr=False)
    work: int = 0
    inequivalence_proof: PolynomialInequivalenceProof | None = field(default=None, repr=False)

    @property
    def equivalence_kind(self):
        """报告通过复核的等价结论所依赖的结构或代数证据。"""
        if self.status != Status.EQUIVALENT:
            return None
        return "structural" if self.proof.structural is not None else "algebraic"


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


def compare_algebraic_jaxprs(left, right, *, algebra_budget=10000):
    """比较给定图的有序输出，生成并复核索引代数证明。

    按前向语义解释已物化图；JVP/VJP 须由函数入口分别追踪后调用。
    不能自行恢复函数 pytree 或追踪环境，不执行数值抽样或优化原始程序。
    """
    budget = Budget(algebra_budget)
    rules = set()
    try:
        structural = compare_jaxprs(left, right, derivative_scope="primal")
        if structural.status == Status.EQUIVALENT:
            # 结构证据已经更强，不要求这类图还能被有限的代数规则重新展开。
            proof = AlgebraProof(
                None, None, (), algebra_budget, structural=structural.correspondence
            )
            verify_algebraic_proof(left, right, proof)
            return AlgebraComparison(
                Status.EQUIVALENT,
                rules=("structural",),
                assumptions=("structural equivalence of the supplied forward jaxprs",),
                warnings=(),
                proof=proof,
            )
        if structural.status == Status.UNKNOWN:
            return AlgebraComparison(Status.UNKNOWN, structural.issues)
        normalizer = PolynomialNormalizer(budget)
        sources, signatures, lowered_graphs = [], [], []
        prepared = tuple(_prepare(original) for original in (left, right))
        common_calls = CommonCalls(
            tuple(
                (original, *adapted)
                for original, adapted in zip((left, right), prepared, strict=True)
            ),
            budget,
        )
        for graph, correspondence in prepared:
            lowerer = Lowerer(budget, common_calls)
            lowered = lowerer.graph(graph)
            lowered_graphs.append(lowered)
            rules.update(lowerer.rules)
            roots = tuple(normalizer.normalize(expr) for expr in lowered.roots)
            sources.append(AlgebraSource(graph, correspondence, roots))
            signatures.append(lowered.signature)
        if signatures[0] != signatures[1]:
            raise UnsupportedJaxprError(
                "algebra.contract", "input or output abstract contract differs"
            )
        forms = [tuple(normalizer.steps[i].result for i in source.roots) for source in sources]
        if forms[0] != forms[1]:
            negative = try_prove_inequivalent(
                left, right, sources=tuple(sources), lowered=tuple(lowered_graphs), budget=budget
            )
            if negative is not None:
                return AlgebraComparison(
                    Status.INEQUIVALENT,
                    rules=tuple(sorted(rules | {"polynomial:nonzero_output_coefficient"})),
                    inequivalence_proof=negative,
                    work=budget.used,
                )
            raise UnsupportedJaxprError(
                "algebra.outputs", "normal forms differ; no algebraic proof found"
            )
        proof = AlgebraProof(*sources, tuple(normalizer.steps), algebra_budget)
        verify_algebraic_proof(left, right, proof)
        rules.update(step.expression.op for step in proof.steps)
        rules.update(
            f"scalar:{rewrite.kind}" for step in proof.steps for rewrite in step.scalar_rewrites
        )
        return AlgebraComparison(
            Status.EQUIVALENT, rules=tuple(sorted(rules)), proof=proof, work=budget.used
        )
    except JaxprError as exc:
        return AlgebraComparison(
            Status.UNKNOWN, (Issue(exc.path, exc.reason),), tuple(sorted(rules)), work=budget.used
        )
    except RecursionError:
        return AlgebraComparison(
            Status.UNKNOWN,
            (Issue("algebra.depth", "expression depth exceeds supported recursion limit"),),
            tuple(sorted(rules)),
            work=budget.used,
        )
    except (TypeError, ValueError, NotImplementedError) as exc:
        return AlgebraComparison(
            Status.UNKNOWN,
            (Issue("algebra.lower", f"conversion rejected: {exc}"),),
            tuple(sorted(rules)),
            work=budget.used,
        )
