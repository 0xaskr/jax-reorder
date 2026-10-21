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
from .call_bindings import BindingExpr, binding_value, extract_call
from .communication import DMA_PROGRESS_ASSUMPTION
from .coordinate_partition import equal_on_output_domain
from .core_execution import CoreExecutionBinding
from .index_domain import AnalysisBudget, IndexExpr, axis, bounds
from .local_memory import functionalize_local_call, has_unmasked_padding
from .memory import MemoryProjection, ValueExpr, _substitute, expressions_equal, functionalize_call
from .model import (
    ArrayType,
    Attribute,
    PallasCallView,
    ParameterView,
    ProgramView,
    RefType,
    ShardMapView,
    SliceIndex,
    TensorIndex,
    ValueRef,
    ValueType,
)
from .remote_memory import (
    RemoteValueEvidence,
    functionalize_remote_call,
    has_protocol,
    same_remote_evidence,
)
from .tensor_domain import TensorDomain, multidimensional, restoring_outputs
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
    remote: RemoteValueEvidence | None = None
    bindings: tuple[BindingExpr, ...] = ()
    # This is a memory projection with an outstanding algebra-only obligation,
    # never a complete normalized equality certificate.
    deferred_padding: bool = False
    execution_cores: int | None = None
    execution: CoreExecutionBinding | None = None


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
        and left.bindings == right.bindings
        and left.deferred_padding == right.deferred_padding
        and (type(left.execution_cores), left.execution_cores)
        == (type(right.execution_cores), right.execution_cores)
        and left.execution == right.execution
        and same_remote_evidence(left.remote, right.remote)
        and (a.domains, a.origins, a.conditions, a.communication, a.waits)
        == (b.domains, b.origins, b.conditions, b.communication, b.waits)
        and expressions_equal(a.outputs, b.outputs)
        and tuple((v.storage, v.version, v.origin, v.span, v.rectangle) for v in a.versions)
        == tuple((v.storage, v.version, v.origin, v.span, v.rectangle) for v in b.versions)
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
            op.primitive not in ("jit", "shard_map", "custom_jvp_call", "custom_vjp_call")
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


def _simple_entry(view: ProgramView) -> bool:
    root = view.root
    while len(root.operations) == 1 and not root.constants:
        op = root.operations[0]
        if op.operands != tuple(ValueRef(v.id) for v in root.inputs) or root.outputs != tuple(
            ValueRef(v.id) for v in op.results
        ):
            return False
        if op.call is not None:
            return True
        if (
            op.primitive not in ("jit", "shard_map", "custom_jvp_call", "custom_vjp_call")
            or len(op.regions) != 1
        ):
            return False
        root = op.regions[0].region
    return False


def _extended_local(call: PallasCallView) -> bool:
    pending = [call.kernel]
    output_blocks = call.blocks[call.num_inputs :]
    different_output_windows = len({(b.array.shape, b.block_shape) for b in output_blocks}) > 1
    different_scratch_windows = bool(output_blocks) and any(
        scratch.bound_array is not None
        and scratch.bound_array.shape != output_blocks[0].block_shape
        for scratch in call.scratch
    )
    needed = bool(
        multidimensional(call, masked=False)
        or restoring_outputs(call)
        or call.prefetch
        or call.aliases
        or different_output_windows
        or different_scratch_windows
        or (call.grid[0].size > 1 and call.scratch)
    )
    while pending:
        region = pending.pop()
        types = {v.id: v.type for v in region.inputs}
        types.update((v.id, v.type) for v, _ in region.constants)
        types.update((v.id, v.type) for op in region.operations for v in op.results)
        for op in region.operations:
            if (
                op.dma
                and (op.dma.device_coordinates is not None or op.dma.source_semaphore is not None)
            ) or (op.semaphore and op.semaphore.device_coordinates is not None):
                return False
            scalar_index = False
            partial_window = False
            if op.access and not isinstance(op.access.index, SliceIndex | TensorIndex):
                index = op.access.index
                typ = types[index.id] if isinstance(index, ValueRef) else index.type
                scalar_index = isinstance(typ, ArrayType) and not typ.shape
            if op.access and isinstance(op.access.index, SliceIndex):
                ref_type = types[op.access.ref.id]
                if (
                    isinstance(ref_type, RefType)
                    and isinstance(ref_type.inner, ArrayType)
                    and len(ref_type.inner.shape) == 1
                ):
                    partial_window = op.access.index != SliceIndex(0, ref_type.inner.shape[0], 1)
            needed |= (
                op.primitive in ("cond", "scan", "while", "jit")
                or scalar_index
                or partial_window
                or bool(op.access and op.access.scalar)
            )
            pending.extend(b.region for b in op.regions)
    return needed


def _normalize_bound_call(
    view: ProgramView,
    budget: AnalysisBudget,
    *,
    defer_padding: bool = False,
    execution_cores: int | None = None,
) -> RewriteEvidence:
    site = extract_call(view, budget=budget)
    call = site.operation.call
    assert call is not None
    if any(w.shard_map is not None for w in site.wrappers):
        raise UnsupportedJaxprError(
            call.kernel.origin.path, "outer shard bindings need global assembly"
        )
    if tuple((o.operation, o.port) for o in site.outputs) != tuple(
        ("kernel_output", i) for i in range(call.num_outputs)
    ):
        raise UnsupportedJaxprError(
            call.kernel.origin.path, "outer postprocessing needs output composition"
        )
    if call.aliases and len(set(site.arguments)) != len(site.arguments):
        raise UnsupportedJaxprError(
            call.kernel.origin.path, "aliased call repeats an outer input binding"
        )
    projection = functionalize_local_call(
        call,
        budget=budget,
        inputs=tuple(binding_value(a, budget=budget) for a in site.arguments),
        defer_padding=defer_padding,
        execution_cores=execution_cores,
    )
    boundary = CallBoundary(
        "pallas_call",
        tuple(v.type for v in view.root.inputs),
        tuple(o.type for o in site.outputs),
        site.operation.parameters,
        site.operation.context,
        view.root.is_high,
        compiler_backend=call.compiler.backend if call.compiler else None,
        compiler_options=tuple(
            p for p in call.compiler.parameters if p.name != "dimension_semantics"
        )
        if call.compiler
        else (),
        kernel_argument_policy=call.kernel_argument_policy,
    )
    return RewriteEvidence(
        "bound-local-memory-to-values",
        1,
        call.kernel.origin.path,
        projection,
        (boundary,),
        bindings=site.arguments,
        deferred_padding=defer_padding,
        execution_cores=execution_cores,
    )


def normalize_view(
    view: ProgramView,
    *,
    budget: AnalysisBudget,
    defer_padding: bool = False,
    execution_cores: int | None = None,
    execution: CoreExecutionBinding | None = None,
) -> RewriteEvidence:
    """Project memory values, optionally leaving input-padding validity to algebra.

    The default entry proves complete values. ``defer_padding`` carries unresolved
    lane validity and cannot support a normalized-equality certificate by itself.
    ``execution_cores=1`` is a caller-established launch condition, not a compiler
    setting. Remote parallel grids additionally need ``execution`` to establish
    one exact core-0 program stream shared by all participating devices.
    """
    if execution_cores is not None and (type(execution_cores) is not int or execution_cores != 1):
        raise ValueError("execution_cores currently requires an explicit single-core binding")
    if execution is not None and (
        not isinstance(execution, CoreExecutionBinding) or execution_cores is not None
    ):
        raise ValueError(
            "execution requires a CoreExecutionBinding and no execution_cores argument"
        )
    if not _simple_entry(view):
        if execution is not None:
            raise UnsupportedJaxprError(
                view.root.origin.path, "execution mapping normalization requires a remote protocol"
            )
        try:
            return _normalize_bound_call(
                view, budget, defer_padding=defer_padding, execution_cores=execution_cores
            )
        except RecursionError as error:
            raise ComparisonBudgetError(
                view.root.origin.path,
                "outer binding nesting limit exceeded",
                failure_kind=ComparisonFailureKind.RECURSION_LIMIT,
            ) from error
    call, boundaries = _call(view)
    try:
        local = _extended_local(call) and not any(b.shard_map for b in boundaries)
        if (
            defer_padding
            and not has_protocol(call.kernel)
            and not any(b.shard_map for b in boundaries)
        ):
            local |= has_unmasked_padding(call)
        if has_protocol(call.kernel) and not local:
            shards = [b for b in boundaries if b.shard_map is not None]
            if len(shards) > 1:
                raise UnsupportedJaxprError(
                    call.kernel.origin.path, "remote DMA requires one global shard_map boundary"
                )
            boundary = shards[0] if shards else boundaries[0]
            projection, remote = functionalize_remote_call(
                call,
                boundary.shard_map,
                boundary.inputs,
                boundary.outputs,
                budget=budget,
                execution_cores=execution_cores,
                execution=execution,
            )
            return RewriteEvidence(
                "remote-memory-to-global-values" if shards else "communication-memory-to-values",
                2 if shards else 1,
                call.kernel.origin.path,
                projection,
                boundaries,
                remote=remote,
                execution_cores=execution_cores,
                execution=execution,
            )
        if execution is not None:
            raise UnsupportedJaxprError(
                call.kernel.origin.path,
                "execution mapping normalization requires a remote protocol",
            )
        if local:
            projection = functionalize_local_call(
                call,
                budget=budget,
                defer_padding=defer_padding,
                execution_cores=execution_cores,
            )
        else:
            try:
                projection = functionalize_call(call, budget=budget)
            except UnsupportedJaxprError:
                if not multidimensional(call) or has_protocol(call.kernel):
                    raise
                projection = functionalize_local_call(
                    call,
                    budget=budget,
                    defer_padding=defer_padding,
                    execution_cores=execution_cores,
                )
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
        "local-control-memory-to-values" if local else "blocked-memory-to-global-values",
        1 if local else 2,
        call.kernel.origin.path,
        projection,
        boundaries,
        tuple(shard_proofs),
        deferred_padding=defer_padding,
        execution_cores=execution_cores,
    )


def verify_normalized_evidence(
    left: ProgramView, right: ProgramView, evidence: NormalizedEvidence, *, budget: AnalysisBudget
) -> None:
    """Replay rule preconditions against the source views and check the terminal match.

    Replay shares rule implementations; it is an audit, not an independent proof
    of adaptation or of the rewrite implementation itself.
    """
    if evidence.left.deferred_padding or evidence.right.deferred_padding:
        raise CorrespondenceError(
            "normalized", "deferred padding requires an algebra initialization proof"
        )
    execution_cores = evidence.left.execution_cores
    if evidence.right.execution_cores != execution_cores or any(
        (
            proof.execution_cores is not None
            and (type(proof.execution_cores) is not int or proof.execution_cores != 1)
        )
        or (
            proof.execution is not None
            and (
                not isinstance(proof.execution, CoreExecutionBinding)
                or proof.execution_cores is not None
            )
        )
        or (
            proof.remote is not None
            and (
                (
                    type(proof.remote.protocol.execution_cores),
                    proof.remote.protocol.execution_cores,
                )
                != (type(proof.execution_cores), proof.execution_cores)
                or proof.remote.protocol.execution != proof.execution
            )
        )
        for proof in (evidence.left, evidence.right)
    ):
        raise CorrespondenceError("normalized", "execution conditions do not replay")
    try:
        a, b = (
            normalize_view(
                source,
                budget=budget,
                execution_cores=proof.execution_cores,
                execution=proof.execution,
            )
            for source, proof in ((left, evidence.left), (right, evidence.right))
        )
    except (UnsupportedJaxprError, ValueError) as error:
        raise CorrespondenceError("normalized", "rewrite evidence does not replay") from error
    if not _same_rewrite(evidence.left, a) or not _same_rewrite(evidence.right, b):
        raise CorrespondenceError("normalized", "rewrite evidence does not replay")
    expected = tuple((i, i) for i in range(len(a.projection.outputs)))
    if (
        evidence.terminal_outputs != expected
        or not _outputs_equal(a, b, budget)
        or a.boundaries != b.boundaries
    ):
        raise CorrespondenceError("normalized", "terminal projections do not match")


def _outputs_equal(left: RewriteEvidence, right: RewriteEvidence, budget: AnalysisBudget) -> bool:
    if left.deferred_padding or right.deferred_padding:
        return False
    a, b = left.projection.outputs, right.projection.outputs
    if expressions_equal(a, b):
        return True
    if (
        len(a) != len(b)
        or not any(isinstance(d, TensorDomain) for d in left.projection.domains)
        or not any(isinstance(d, TensorDomain) for d in right.projection.domains)
    ):
        return False
    for x, y, typ in zip(a, b, left.boundaries[0].outputs, strict=True):
        assert isinstance(typ, ArrayType)
        # Remote rank-one assembly retains its flattened global coordinate.
        # Bind it to this complete output domain before comparing partitions;
        # memory/protocol obligations have already been proved by both rewrites.
        if len(typ.shape) == 1:
            if left.remote is not None:
                x = _substitute(x, {"global": axis("output:0")})
            if right.remote is not None:
                y = _substitute(y, {"global": axis("output:0")})
        if not expressions_equal((x,), (y,)) and not equal_on_output_domain(
            x, y, typ.shape, budget
        ):
            return False
    return True


def compare_program_views(
    left: ProgramView,
    right: ProgramView,
    *,
    analysis_budget: int = 10000,
    execution_cores: int | None = None,
    execution: CoreExecutionBinding
    | tuple[CoreExecutionBinding, CoreExecutionBinding]
    | None = None,
) -> ViewComparison:
    """Compare views under optional caller-established remote program streams.

    A pair of execution bindings supplies separate grid coordinates for the two
    sources. A single binding applies to both; neither changes native scheduling.
    """
    if execution_cores is not None and (type(execution_cores) is not int or execution_cores != 1):
        raise ValueError("execution_cores currently requires an explicit single-core binding")
    if execution is None:
        executions = (None, None)
    elif execution_cores is not None:
        raise ValueError("execution requires no execution_cores argument")
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
    budget = AnalysisBudget(analysis_budget)
    assumptions = ("supplied forward graph and adapted execution context only",)
    if execution_cores == 1:
        assumptions += ("caller binds execution to one local core (core 0) on each device",)
    if execution is not None:
        assumptions += (
            "caller binds every device to the supplied core-0 program stream for each source; "
            "ordinary pallas_call does not enforce this execution mapping and order",
        )
    if left.context != right.context:
        raise ComparisonContractError("context", "adaptation environments differ")
    if tuple(v.type for v in left.root.inputs) != tuple(v.type for v in right.root.inputs):
        raise ComparisonContractError("inputs", "input abstract contracts differ")
    witness = match_views(left, right)
    if witness is not None and execution is None:
        return ViewComparison(Status.EQUIVALENT, witness, budget.used, assumptions)
    (a_call, a_boundary), (b_call, b_boundary) = _call(left), _call(right)
    if a_boundary != b_boundary:
        raise ComparisonContractError(
            "call_boundaries", "wrapper/interface execution contracts differ"
        )
    if tuple(b.array for b in a_call.blocks[a_call.num_inputs :]) != tuple(
        b.array for b in b_call.blocks[b_call.num_inputs :]
    ):
        raise ComparisonContractError("outputs", "output abstract contracts differ")
    a, b = (
        normalize_view(source, budget=budget, execution_cores=execution_cores, execution=binding)
        for source, binding in zip((left, right), executions, strict=True)
    )
    if a.projection.communication or b.projection.communication:
        assumptions += (DMA_PROGRESS_ASSUMPTION,)
    assumptions += tuple(
        dict.fromkeys(
            value
            for proof in (a, b)
            if proof.remote is not None
            for value in proof.remote.protocol.protocol.assumptions
        )
    )
    if _outputs_equal(a, b, budget):
        evidence = NormalizedEvidence(a, b, tuple((i, i) for i in range(len(a.projection.outputs))))
        verify_normalized_evidence(left, right, evidence, budget=budget)
        return ViewComparison(Status.EQUIVALENT, evidence, budget.used, assumptions)
    # A coordinate-only integer output admits an exact symbolic counterexample,
    # independent of all tensor input values. Never infer inequality merely from
    # different formulas (including different opaque input applications).
    for output, (x, y) in enumerate(zip(a.projection.outputs, b.projection.outputs, strict=True)):
        if not isinstance(x, IndexExpr) or not isinstance(y, IndexExpr):
            continue
        output_type = a.boundaries[0].outputs[output]
        assert isinstance(output_type, ArrayType)
        if any(isinstance(d, TensorDomain) for proof in (a, b) for d in proof.projection.domains):
            continue  # This rule binds "global", not tensor "output:i" coordinates.
        size = (
            output_type.shape[0]
            if a.remote is not None
            else a.projection.domains[a_call.num_inputs + output].size
        )
        candidates = {
            0,
            size - 1,
            *(d.block for d in (*a.projection.domains, *b.projection.domains)),
        }
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
