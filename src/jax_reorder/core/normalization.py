"""Internal Pallas comparison: source structure, then proven retiling.

No JAX/Pallas source objects or SymPy imports are allowed in this consumer.
Failure to match normalized formulas is not mathematical inequivalence.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import (
    ComparisonBudgetError,
    ComparisonContractError,
    ComparisonFailureKind,
    CorrespondenceError,
    MathematicalUndecidedError,
    UnsupportedJaxprError,
)
from ..status import Status
from .communication import DMA_PROGRESS_ASSUMPTION
from .index_domain import AnalysisBudget, IndexExpr, bounds
from .memory import MemoryProjection, ValueExpr, expressions_equal, functionalize_call
from .model import (
    Attribute,
    PallasCallView,
    ParameterView,
    ProgramView,
    ShardMapView,
    ValueRef,
    ValueType,
)
from .view_graph import StructuralEvidence, match_views


@dataclass(frozen=True)
class CallBoundary:
    primitive: str
    inputs: tuple[ValueType, ...]
    outputs: tuple[ValueType, ...]
    parameters: tuple[ParameterView, ...]
    context: Attribute
    is_high: bool
    shard_map: ShardMapView | None = None
    compiler_backend: str | None = None
    compiler_options: tuple[ParameterView, ...] = ()
    kernel_argument_policy: str | None = None


@dataclass(frozen=True)
class RewriteEvidence:
    rule: str
    version: int
    source: str
    projection: MemoryProjection
    boundaries: tuple[CallBoundary, ...]
    shard_dependencies: tuple[tuple[tuple[int, ...], ...], ...] = ()


@dataclass(frozen=True)
class NormalizedEvidence:
    left: RewriteEvidence
    right: RewriteEvidence
    # Terminal relation is many source operations to each ordered scalar output.
    terminal_outputs: tuple[tuple[int, int], ...]
    kind: str = "normalized"


@dataclass(frozen=True)
class IndexDifferenceEvidence:
    output: int
    coordinate: int
    left: int
    right: int
    kind: str = "index_difference"


@dataclass(frozen=True)
class ViewComparison:
    status: Status
    evidence: StructuralEvidence | NormalizedEvidence | IndexDifferenceEvidence
    work: int
    assumptions: tuple[str, ...] = ("supplied forward graph and adapted execution context only",)
    warnings: tuple[str, ...] = ()

    @property
    def equivalence_kind(self) -> str | None:
        return self.evidence.kind if self.status == Status.EQUIVALENT else None


def _same_rewrite(left: RewriteEvidence, right: RewriteEvidence) -> bool:
    a, b = left.projection, right.projection
    return (
        (left.rule, left.version, left.source, left.boundaries, left.shard_dependencies)
        == (right.rule, right.version, right.source, right.boundaries, right.shard_dependencies)
        and (a.domains, a.origins, a.conditions, a.communication, a.waits)
        == (b.domains, b.origins, b.conditions, b.communication, b.waits)
        and expressions_equal(a.outputs, b.outputs)
        and tuple((v.storage, v.version, v.origin, v.span) for v in a.versions)
        == tuple((v.storage, v.version, v.origin, v.span) for v in b.versions)
        and expressions_equal(
            tuple(v.value for v in a.versions), tuple(v.value for v in b.versions)
        )
    )


def _call(view: ProgramView) -> tuple[PallasCallView, tuple[CallBoundary, ...]]:
    root = view.root
    boundaries = []
    while True:
        if root.constants or len(root.operations) != 1:
            raise UnsupportedJaxprError(
                root.origin.path,
                "normalization requires a single pallas_call under reviewed call wrappers",
            )
        op = root.operations[0]
        if op.operands != tuple(ValueRef(v.id) for v in root.inputs) or root.outputs != tuple(
            ValueRef(v.id) for v in op.results
        ):
            raise UnsupportedJaxprError(
                root.origin.path, "outer call bindings are not the ordered program interface"
            )
        boundaries.append(
            CallBoundary(
                op.primitive,
                tuple(v.type for v in root.inputs),
                tuple(v.type for v in op.results),
                op.parameters,
                op.context,
                root.is_high,
                op.shard_map,
                op.call.compiler.backend if op.call and op.call.compiler else None,
                tuple(p for p in op.call.compiler.parameters if p.name != "dimension_semantics")
                if op.call and op.call.compiler
                else (),
                op.call.kernel_argument_policy if op.call else None,
            )
        )
        if op.call is not None:
            return op.call, tuple(boundaries)
        if (
            op.primitive not in ("jit", "shard_map")
            or len(op.regions) != 1
            or op.regions[0].role != "call"
        ):
            raise UnsupportedJaxprError(op.origin.path, "unsupported wrapper around pallas_call")
        if op.primitive == "shard_map" and op.shard_map is None:
            raise UnsupportedJaxprError(op.origin.path, "missing shard mesh and partition contract")
        binding = op.regions[0]
        if binding.arguments != op.operands or binding.results != root.outputs:
            raise UnsupportedJaxprError(
                op.origin.path, "wrapper region bindings differ from call ports"
            )
        root = binding.region


def normalize_view(view: ProgramView, *, budget: AnalysisBudget) -> RewriteEvidence:
    call, boundaries = _call(view)
    try:
        projection = functionalize_call(call, budget=budget)
    except RecursionError as error:
        raise ComparisonBudgetError(
            call.kernel.origin.path,
            "index analysis nesting limit exceeded",
            failure_kind=ComparisonFailureKind.RECURSION_LIMIT,
        ) from error
    shard_proofs = []
    for boundary in boundaries:
        shard = boundary.shard_map
        if shard is None:
            continue
        output_dependencies = []
        for output, expr in enumerate(projection.outputs):
            pending, seen, used = [expr], set(), set()
            while pending:
                value = pending.pop()
                if id(value) in seen:
                    continue
                seen.add(id(value))
                budget.spend("shard_dependencies")
                if isinstance(value, ValueExpr):
                    if value.operation == "input" and value.input is not None:
                        used.add(value.input)
                    pending.extend(value.operands)
            varying = {
                axis
                for port in used
                for dimension in shard.input_partitions[port]
                for axis in dimension
            }
            assembled = {
                axis for dimension in shard.output_partitions[output] for axis in dimension
            }
            # A replicated result must actually be invariant on omitted axes.
            # Pallas out_avals/check_vma declarations alone do not prove this,
            # because the kernel intentionally clears manual-axis variation.
            if varying - assembled:
                raise UnsupportedJaxprError(
                    call.kernel.origin.path,
                    "output replication is not proved from shard-local input dependencies",
                )
            output_dependencies.append(tuple(sorted(used)))
        shard_proofs.append(tuple(output_dependencies))
    return RewriteEvidence(
        "blocked-memory-to-global-values",
        2,
        call.kernel.origin.path,
        projection,
        boundaries,
        tuple(shard_proofs),
    )


def verify_normalized_evidence(
    left: ProgramView, right: ProgramView, evidence: NormalizedEvidence, *, budget: AnalysisBudget
) -> None:
    """Replay rule preconditions against the source views and check the terminal match.

    Replay shares rule implementations; it is an audit, not an independent proof
    of adaptation or of the rewrite implementation itself.
    """
    a, b = normalize_view(left, budget=budget), normalize_view(right, budget=budget)
    if not _same_rewrite(evidence.left, a) or not _same_rewrite(evidence.right, b):
        raise CorrespondenceError("normalized", "rewrite evidence does not replay")
    expected = tuple((i, i) for i in range(len(a.projection.outputs)))
    if (
        evidence.terminal_outputs != expected
        or not expressions_equal(a.projection.outputs, b.projection.outputs)
        or a.boundaries != b.boundaries
    ):
        raise CorrespondenceError("normalized", "terminal projections do not match")


def compare_program_views(
    left: ProgramView, right: ProgramView, *, analysis_budget: int = 10000
) -> ViewComparison:
    """Compare admitted views; the public JAX API is connected in milestone M3."""
    budget = AnalysisBudget(analysis_budget)
    if left.context != right.context:
        raise ComparisonContractError("context", "adaptation environments differ")
    if tuple(v.type for v in left.root.inputs) != tuple(v.type for v in right.root.inputs):
        raise ComparisonContractError("inputs", "input abstract contracts differ")
    witness = match_views(left, right)
    if witness is not None:
        return ViewComparison(Status.EQUIVALENT, witness, budget.used)
    (a_call, a_boundary), (b_call, b_boundary) = _call(left), _call(right)
    if a_boundary != b_boundary:
        raise ComparisonContractError(
            "call_boundaries", "wrapper/interface execution contracts differ"
        )
    if tuple(b.array for b in a_call.blocks[a_call.num_inputs :]) != tuple(
        b.array for b in b_call.blocks[b_call.num_inputs :]
    ):
        raise ComparisonContractError("outputs", "output abstract contracts differ")
    a, b = normalize_view(left, budget=budget), normalize_view(right, budget=budget)
    assumptions = ("supplied forward graph and adapted execution context only",)
    if a.projection.communication or b.projection.communication:
        assumptions += (DMA_PROGRESS_ASSUMPTION,)
    if expressions_equal(a.projection.outputs, b.projection.outputs):
        evidence = NormalizedEvidence(a, b, tuple((i, i) for i in range(len(a.projection.outputs))))
        verify_normalized_evidence(left, right, evidence, budget=budget)
        return ViewComparison(Status.EQUIVALENT, evidence, budget.used, assumptions)
    # A coordinate-only integer output admits an exact symbolic counterexample,
    # independent of all tensor input values. Never infer inequality merely from
    # different formulas (including different opaque input applications).
    size = a.projection.domains[0].size
    candidates = {0, size - 1, *(d.block for d in (*a.projection.domains, *b.projection.domains))}
    for output, (x, y) in enumerate(zip(a.projection.outputs, b.projection.outputs, strict=True)):
        if not isinstance(x, IndexExpr) or not isinstance(y, IndexExpr):
            continue
        for coordinate in sorted(c for c in candidates if 0 <= c < size):
            budget.spend("index_difference")
            lo_x, hi_x = bounds(x, {"global": (coordinate, coordinate)})
            lo_y, hi_y = bounds(y, {"global": (coordinate, coordinate)})
            if lo_x == hi_x and lo_y == hi_y and lo_x != lo_y:
                return ViewComparison(
                    Status.INEQUIVALENT,
                    IndexDifferenceEvidence(output, coordinate, lo_x, lo_y),
                    budget.used,
                    assumptions,
                )
    raise MathematicalUndecidedError(
        "normalized", "normalized output expressions differ; a mathematical backend is required"
    )
