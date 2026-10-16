"""Historical certificate generation used only by compatibility and tampering tests."""

from jax_reorder.algebra import AlgebraComparison, _prepare
from jax_reorder.algebra_calls import CommonCalls
from jax_reorder.algebra_inequivalence import try_prove_inequivalent
from jax_reorder.algebra_ir import Budget
from jax_reorder.algebra_lower import Lowerer
from jax_reorder.algebra_normalize import PolynomialNormalizer
from jax_reorder.algebra_proof import AlgebraProof, AlgebraSource, verify_algebraic_proof
from jax_reorder.canonical import Issue, Status, compare_jaxprs
from jax_reorder.errors import JaxprError, UnsupportedJaxprError


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
