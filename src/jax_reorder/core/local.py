"""Device-local structure and optional resource checks, with explicit scope."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, cast

from ..errors import (
    ComparisonBudgetError,
    ComparisonContractError,
    ComparisonFailureKind,
    CorrespondenceError,
    UnsupportedJaxprError,
)
from ..status import Status
from .adapters.attributes import project_mesh_attribute
from .adapters.jax import adapt
from .call_bindings import BindingExpr, CallSite, binding_value, extract_call
from .canonical import canonicalize
from .communication import DMA_PROGRESS_ASSUMPTION
from .core_execution import (
    CoreExecutionBinding,
    CoreExecutionEvidence,
    functionalize_core_execution,
)
from .index_domain import AnalysisBudget, axis
from .local_memory import functionalize_local_call, has_unmasked_padding
from .memory import Expression, Predicate, ValueExpr
from .model import ArrayType, MeshView, OperationView, PallasCallView, ParameterView, ProgramView
from .normalization import (
    CallBoundary,
    RewriteEvidence,
    _outputs_equal,
    _retain_call_source,
    _same_rewrite,
)
from .remote_memory import functionalize_remote_call


@dataclass(frozen=True)
class LocalKernelEvidence:
    left: CallSite
    right: CallSite
    axes: tuple[tuple[str, int, str], ...]
    kind: str = "structural"


@dataclass(frozen=True)
class LocalKernelComparison:
    status: Status
    evidence: LocalKernelEvidence | None
    protocol_checked: bool = False
    left_protocol: RewriteEvidence | None = None
    right_protocol: RewriteEvidence | None = None
    scope: str = "device_local"
    assumptions: tuple[str, ...] = (
        "corresponding local input values, bound logical axes and reviewed TPU execution policy",
        "local structure does not establish communication validity or whole-program equivalence",
    )
    execution_cores: int | None = None
    execution: CoreExecutionBinding | None = None
    left_execution: CoreExecutionEvidence | None = None
    right_execution: CoreExecutionEvidence | None = None
    equivalence: str = "structural"

    @property
    def equivalence_kind(self) -> str | None:
        return self.evidence.kind if self.status is Status.EQUIVALENT and self.evidence else None


def _axes(site: CallSite) -> tuple[tuple[str, int, str], ...]:
    if sum(w.shard_map is not None for w in site.wrappers) > 1:
        raise UnsupportedJaxprError(
            "local.axes", "nested shard environments need explicit axis scope mapping"
        )
    available = {
        name: (name, size, kind)
        for w in site.wrappers
        if w.shard_map
        for name, size, kind in w.shard_map.mesh.axes
    }
    used: set[str] = set()
    remote = False
    pending: list[Any] = [site.operation.call, *site.arguments]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, OperationView | BindingExpr):
            primitive = value.primitive if isinstance(value, OperationView) else value.operation
            if primitive == "axis_index":
                used.update(str(p.value) for p in value.parameters if p.name == "axis_name")
        if isinstance(value, OperationView):
            remote |= bool(
                (value.dma and value.dma.device_coordinates is not None)
                or (value.semaphore and value.semaphore.device_coordinates is not None)
            )
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, f.name) for f in fields(value) if f.name != "origin")
        elif isinstance(value, tuple):
            pending.extend(value)
    if remote:
        used.update(available)
    if used - available.keys():
        raise UnsupportedJaxprError("local.axes", "kernel is missing a named-axis binding")
    return tuple(available[name] for name in available if name in used)


def _local_facts(site: CallSite) -> Callable[[Any], Any]:
    """Derive a local contract without changing any source facts or routing."""
    axes = _axes(site)
    names = {a[0] for a in axes}
    cache: dict[int, tuple[Any, Any]] = {}
    remote = site.operation.call is not None and _has_remote(site.operation.call)

    def local(value: Any) -> Any:
        if id(value) in cache:
            return cache[id(value)][1]
        original = value
        if isinstance(value, BindingExpr) and value.operation == "local_input" and not remote:
            (parent,) = value.operands
            if parent.operation == "input":
                value = replace(parent, type=value.type)
        if isinstance(value, ArrayType):
            # Manual variation and global partitioning are interface facts;
            # supplied local values are already bound at this entry boundary.
            value = replace(
                value, metadata=tuple((k, v) for k, v in value.metadata if k != "manual_axis_type")
            )
        if isinstance(value, MeshView):
            selected = tuple(a for a in value.axes if a[0] in names)
            result = replace(
                value,
                axes=selected,
                devices=(),
                abstract_device=value.abstract_device if selected or remote else ("none",),
                concrete=False,
            )
        elif is_dataclass(value) and not isinstance(value, type):
            result = replace(
                value, **{f.name: local(getattr(value, f.name)) for f in fields(value) if f.compare}
            )
        elif isinstance(value, tuple):
            projected = project_mesh_attribute(value, names) if not remote else value
            result = projected if projected is not value else tuple(local(v) for v in value)
        else:
            result = value
        # Replacements create temporary descriptors. Keep their original objects
        # alive too, so Python cannot recycle an identity inside this traversal.
        cache[id(original)] = (original, result)
        return result

    return local


def _key(site: CallSite) -> tuple[Any, ...]:
    local = _local_facts(site)
    return (
        local(site.operation.call),
        local(site.arguments),
        _axes(site),
        local(site.operation.context),
    )


def _has_remote(call: PallasCallView) -> bool:
    pending = [call.kernel]
    while pending:
        for op in pending.pop().operations:
            if (op.dma and op.dma.device_coordinates is not None) or (
                op.semaphore and op.semaphore.device_coordinates is not None
            ):
                return True
            pending.extend(b.region for b in op.regions)
    return False


def _boundary(
    view: ProgramView,
    site: CallSite,
    inputs: tuple[ArrayType, ...],
    outputs: tuple[ArrayType, ...],
    parameters: tuple[ParameterView, ...],
) -> CallBoundary:
    call = site.operation.call
    assert call is not None
    boundary = CallBoundary(
        "pallas_local_values",
        inputs,
        outputs,
        parameters,
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
    # Remote inputs still denote global source arrays and actual partitions.
    # Purely local values have already been bound at their local signatures.
    return boundary if _has_remote(call) else _local_facts(site)(boundary)


def _check_protocol(
    view: ProgramView,
    site: CallSite,
    budget: AnalysisBudget,
    execution_cores: int | None,
    execution: CoreExecutionBinding | None,
    *,
    defer_padding: bool = False,
) -> tuple[RewriteEvidence, CoreExecutionEvidence | None]:
    call = site.operation.call
    assert call is not None
    defer_padding = defer_padding and has_unmasked_padding(call)
    if _has_remote(call):
        return _remote_projection(view, site, budget, execution_cores, execution), None
    if call.aliases and len(set(site.arguments)) != len(site.arguments):
        raise UnsupportedJaxprError(
            call.kernel.origin.path, "aliased call repeats an outer input binding"
        )
    axes = _axes(site)
    coordinates = {name: axis("device:" + name) for name, _, _ in axes}
    ranges = {"device:" + name: (0, size - 1) for name, size, _ in axes}
    cache: dict[int, BindingExpr] = {}

    def local_input(v: BindingExpr) -> BindingExpr:
        if id(v) in cache:
            return cache[id(v)]
        budget.spend("local.bindings")
        if v.operation == "local_input":
            parent = v.operands[0]
            if parent.operation != "input":
                raise UnsupportedJaxprError(
                    "local.bindings", "computed global shard inputs need a shard value proof"
                )
            result = replace(parent, type=v.type)
        else:
            result = replace(v, operands=tuple(local_input(a) for a in v.operands))
        cache[id(v)] = result
        return result

    bound = tuple(local_input(a) for a in site.arguments)
    input_types: dict[int, ArrayType] = {}
    pending, seen = list(bound), set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if value.operation == "input":
            assert value.port is not None and isinstance(value.type, ArrayType)
            typ = replace(
                value.type,
                metadata=tuple((k, v) for k, v in value.type.metadata if k != "manual_axis_type"),
            )
            if value.port in input_types and input_types[value.port] != typ:
                raise UnsupportedJaxprError(
                    "local.inputs", "one source input has incompatible local views"
                )
            input_types[value.port] = typ
        pending.extend(value.operands)
    ports = tuple(sorted(input_types))
    renumbered: dict[int, BindingExpr] = {}

    def renumber(value: BindingExpr) -> BindingExpr:
        if id(value) not in renumbered:
            renumbered[id(value)] = replace(
                value,
                port=ports.index(value.port) if value.operation == "input" else value.port,
                operands=tuple(renumber(v) for v in value.operands),
            )
        return renumbered[id(value)]

    arguments = tuple(
        binding_value(renumber(a), budget=budget, coordinates=coordinates, ranges=ranges)
        for a in bound
    )
    core_evidence = None
    if execution is None:
        projection = functionalize_local_call(
            call,
            budget=budget,
            inputs=arguments,
            coordinates=coordinates,
            axis_ranges=ranges,
            execution_cores=execution_cores,
            defer_padding=defer_padding,
        )
    else:
        projection, core_evidence = functionalize_core_execution(
            call,
            execution,
            budget=budget,
            inputs=arguments,
            coordinates=coordinates,
            axis_ranges=ranges,
        )
    proof = RewriteEvidence(
        "device-local-bound-core-resource-check" if execution else "device-local-resource-check",
        1,
        call.kernel.origin.path,
        projection,
        (
            _boundary(
                view,
                site,
                tuple(input_types[p] for p in ports),
                tuple(b.array for b in call.blocks[call.num_inputs :]),
                (ParameterView("source_input_ports", ports), ParameterView("axes", axes)),
            ),
        ),
        bindings=site.arguments,
        deferred_padding=defer_padding,
        execution_cores=execution_cores,
        execution=execution,
    )
    return _retain_call_source(proof, call), core_evidence


def _remote_projection(
    view: ProgramView,
    site: CallSite,
    budget: AnalysisBudget,
    execution_cores: int | None,
    execution: CoreExecutionBinding | None,
) -> RewriteEvidence:
    call = site.operation.call
    assert call is not None
    axes = _axes(site)
    shards = [w.shard_map for w in site.wrappers if w.shard_map is not None]
    parents = tuple(a.operands[0] if a.operation == "local_input" else a for a in site.arguments)
    if any(a.operation != "input" or a.port is None for a in parents):
        raise UnsupportedJaxprError(
            "local.bindings", "remote computed inputs need a shard value proof"
        )
    ports = tuple(sorted({a.port for a in parents if a.port is not None}))
    types = {a.port: cast(ArrayType, a.type) for a in parents}
    shard = shards[0] if shards else None
    if shard is not None:
        if any(a.operation != "local_input" for a in site.arguments):
            raise UnsupportedJaxprError("local.bindings", "missing input shard placement")
        partitions = tuple(
            cast(
                tuple[tuple[str, ...], ...],
                next(p.value for p in a.parameters if p.name == "input_partition"),
            )
            for a in site.arguments
        )
        shard = replace(shard, input_partitions=partitions)
    projection, remote = functionalize_remote_call(
        call,
        shard,
        tuple(a.type for a in parents),
        tuple(b.array for b in call.blocks[call.num_inputs :]),
        budget=budget,
        execution_cores=execution_cores,
        execution=execution,
        scope="device_local",
    )
    mapping = tuple(ports.index(a.port) for a in parents)
    cache: dict[int, Expression] = {}

    def rebind(value: Expression) -> Expression:
        if id(value) not in cache:
            result = value
            if isinstance(value, ValueExpr | Predicate):
                result = replace(value, operands=tuple(rebind(a) for a in value.operands))
                if isinstance(result, ValueExpr) and result.input is not None:
                    result = replace(result, input=mapping[result.input])
            cache[id(value)] = result
        return cache[id(value)]

    projection = replace(projection, outputs=tuple(rebind(v) for v in projection.outputs))
    boundary = _boundary(
        view,
        site,
        tuple(types[p] for p in ports),
        tuple(call.blocks[call.num_inputs + p].array for p, _ in remote.local_outputs),
        (
            ParameterView("source_input_ports", ports),
            ParameterView("axes", axes),
            ParameterView("local_outputs", remote.local_outputs),
        ),
    )
    proof = RewriteEvidence(
        "device-local-remote-memory-to-values",
        1,
        call.kernel.origin.path,
        projection,
        (boundary,),
        remote=remote,
        bindings=site.arguments,
        execution_cores=execution_cores,
        execution=execution,
    )
    return _retain_call_source(proof, call)


def project_local_view(
    view: ProgramView,
    *,
    budget: AnalysisBudget,
    defer_padding: bool = False,
    execution: CoreExecutionBinding | None = None,
) -> tuple[CallSite, RewriteEvidence, CoreExecutionEvidence | None]:
    """Project one bound kernel without imposing the parent output assembly."""
    site = extract_call(view, budget=budget)
    _axes(site)
    try:
        evidence, core = _check_protocol(view, site, budget, None, execution)
    except UnsupportedJaxprError:
        if not defer_padding:
            raise
        # A fully proved projection can finish at the semantic stage. Defer
        # lane validity only when strict projection actually needs that proof.
        evidence, core = _check_protocol(view, site, budget, None, execution, defer_padding=True)
    return site, evidence, core


def compare_pallas_kernels(
    left: Any,
    right: Any,
    *,
    check_protocol: bool = False,
    equivalence: str = "structural",
    analysis_budget: int = 10000,
    execution_cores: int | None = None,
    execution: CoreExecutionBinding | None = None,
) -> LocalKernelComparison:
    """Compare one kernel per source graph, retaining lexical and axis bindings.

    A DIFFERENT result is a structural diagnostic. Resource checking is optional
    and separate; unsupported peer/control proofs raise instead of being assumed.
    equivalence="semantic" requires resource proofs and compares the complete
    local value projections; it never invokes mathematical rewriting.
    execution_cores=1 binds the launch to core 0 on each device. This is a caller
    supplied execution condition, recorded and replayed with the result.
    execution supplies exact multi-core program streams for a conditional proof;
    it does not configure native scheduling and requires check_protocol=True.
    Remote protocols accept only one core-0 stream, which the caller establishes
    identically on every device. execution_cores alone does not bind grid order.
    """
    if type(check_protocol) is not bool:
        raise ValueError("check_protocol must be a bool")
    if equivalence not in ("structural", "semantic"):
        raise ValueError("equivalence must be structural or semantic")
    check_protocol |= equivalence == "semantic"
    if execution_cores is not None and (type(execution_cores) is not int or execution_cores != 1):
        raise ValueError("execution_cores currently requires an explicit single-core binding")
    if execution is not None and (
        not isinstance(execution, CoreExecutionBinding)
        or not check_protocol
        or execution_cores is not None
    ):
        raise ValueError(
            "execution requires a CoreExecutionBinding, check_protocol=True, "
            "and no execution_cores argument"
        )
    try:
        return _compare_pallas_kernels(
            left,
            right,
            check_protocol,
            AnalysisBudget(analysis_budget),
            execution_cores,
            execution,
            equivalence,
        )
    except RecursionError as error:
        raise ComparisonBudgetError(
            "local",
            "local binding nesting limit exceeded",
            failure_kind=ComparisonFailureKind.RECURSION_LIMIT,
        ) from error


def _compare_pallas_kernels(
    left: Any,
    right: Any,
    check_protocol: bool,
    budget: AnalysisBudget,
    execution_cores: int | None,
    execution: CoreExecutionBinding | None,
    equivalence: str,
) -> LocalKernelComparison:
    a, b = (
        adapt(
            canonicalize(source, include_derivative_rules=False, search_budget=budget.limit).jaxpr
        ).view
        for source in (left, right)
    )
    if a.context != b.context:
        raise ComparisonContractError("local.context", "adaptation environments differ")
    x, y = extract_call(a, budget=budget), extract_call(b, budget=budget)
    (px, cx), (py, cy) = (
        (
            _check_protocol(a, x, budget, execution_cores, execution),
            _check_protocol(b, y, budget, execution_cores, execution),
        )
        if check_protocol
        else ((None, None), (None, None))
    )
    equal = _key(x) == _key(y)
    kind = "structural"
    if equivalence == "semantic":
        assert px is not None and py is not None
        kind = "structural" if equal else "normalized"
        equal = px.boundaries == py.boundaries and _outputs_equal(px, py, budget)
    evidence = LocalKernelEvidence(x, y, _axes(x), kind) if equal else None
    result = LocalKernelComparison(
        Status.EQUIVALENT if equal else Status.DIFFERENT,
        evidence,
        check_protocol,
        px,
        py,
        execution_cores=execution_cores,
        execution=execution,
        left_execution=cx,
        right_execution=cy,
        equivalence=equivalence,
    )
    if check_protocol:
        protocols = tuple(p for p in (px, py) if p is not None)
        assumptions = tuple(
            dict.fromkeys(value for proof in protocols for value in proof.projection.assumptions)
        )
        assumptions += tuple(
            dict.fromkeys(
                value
                for proof in protocols
                if proof.remote is not None
                for value in proof.remote.protocol.protocol.assumptions
            )
        )
        if any(p.projection.communication or p.remote for p in protocols):
            assumptions = (DMA_PROGRESS_ASSUMPTION,) + assumptions
        result = replace(result, assumptions=result.assumptions + assumptions)
    if execution_cores == 1:
        result = replace(
            result,
            assumptions=result.assumptions + ("caller binds execution to one local core (core 0)",),
        )
    if execution is not None:
        remote_execution = bool(px is not None and px.remote is not None)
        result = replace(
            result,
            assumptions=result.assumptions
            + (
                "conditional on every device using the caller-supplied core-0 program stream; "
                "ordinary pallas_call does not enforce this execution mapping and order"
                if remote_execution
                else "conditional on the caller-supplied exact local-core program mapping "
                "and order; "
                "ordinary pallas_call does not enforce this execution binding",
            ),
        )
    return result


def verify_pallas_kernel_comparison(
    left: Any, right: Any, result: LocalKernelComparison, *, analysis_budget: int = 10000
) -> None:
    """Re-extract source bindings and replay requested resource proofs."""
    if result.status is not Status.EQUIVALENT or result.evidence is None:
        raise CorrespondenceError("local", "an equivalent local result with evidence is required")
    if any(
        proof is not None
        and proof.remote is not None
        and (
            (type(proof.execution_cores), proof.execution_cores)
            != (type(result.execution_cores), result.execution_cores)
            or (
                type(proof.remote.protocol.execution_cores),
                proof.remote.protocol.execution_cores,
            )
            != (type(result.execution_cores), result.execution_cores)
            or proof.execution != result.execution
            or proof.remote.protocol.execution != result.execution
        )
        for proof in (result.left_protocol, result.right_protocol)
    ):
        raise CorrespondenceError("local", "remote execution conditions do not replay")
    replay = compare_pallas_kernels(
        left,
        right,
        check_protocol=result.protocol_checked,
        analysis_budget=analysis_budget,
        execution_cores=result.execution_cores,
        execution=result.execution,
        equivalence=result.equivalence,
    )
    if (result.status, result.evidence, result.scope, result.assumptions) != (
        replay.status,
        replay.evidence,
        replay.scope,
        replay.assumptions,
    ):
        raise CorrespondenceError("local", "local source bindings do not replay")
    for actual, expected in (
        (result.left_protocol, replay.left_protocol),
        (result.right_protocol, replay.right_protocol),
    ):
        if (actual is None) != (expected is None) or (
            actual is not None and expected is not None and not _same_rewrite(actual, expected)
        ):
            raise CorrespondenceError("local", "local protocol evidence does not replay")
    for actual, expected in (
        (result.left_execution, replay.left_execution),
        (result.right_execution, replay.right_execution),
    ):
        if (actual is None) != (expected is None):
            raise CorrespondenceError("local", "local core execution evidence is missing")
        if actual is None or expected is None:
            continue
        if (
            actual.binding != expected.binding
            or actual.protocol != expected.protocol
            or tuple((p.core, p.point) for p in actual.programs)
            != tuple((p.core, p.point) for p in expected.programs)
        ):
            raise CorrespondenceError("local", "local core execution evidence does not replay")
        for a, b in zip(actual.programs, expected.programs, strict=True):
            left = RewriteEvidence("bound-program", 1, "", a.projection, ())
            right = RewriteEvidence("bound-program", 1, "", b.projection, ())
            if not _same_rewrite(left, right):
                raise CorrespondenceError("local", "bound core program projection does not replay")
