"""Device-local structure and optional resource checks, with explicit scope."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from ..errors import (
    ComparisonBudgetError,
    ComparisonContractError,
    ComparisonFailureKind,
    CorrespondenceError,
    UnsupportedJaxprError,
)
from ..status import Status
from .adapters.jax import adapt
from .call_bindings import BindingExpr, CallSite, binding_value, extract_call
from .canonical import canonicalize
from .core_execution import (
    CoreExecutionBinding,
    CoreExecutionEvidence,
    functionalize_core_execution,
)
from .index_domain import AnalysisBudget, axis
from .local_memory import functionalize_local_call
from .model import ArrayType, MeshView, OperationView, PallasCallView, ProgramView
from .normalization import RewriteEvidence, _same_rewrite, normalize_view


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

    @property
    def equivalence_kind(self) -> str | None:
        return "structural" if self.status is Status.EQUIVALENT else None


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


def _key(site: CallSite) -> tuple[Any, ...]:
    axes = _axes(site)
    names = {a[0] for a in axes}
    cache: dict[int, tuple[Any, Any]] = {}

    def local(value: Any) -> Any:
        if id(value) in cache:
            return cache[id(value)][1]
        original = value
        if isinstance(value, BindingExpr) and value.operation == "local_input":
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
            result = replace(
                value,
                axes=tuple(a for a in value.axes if a[0] in names),
                devices=(),
                concrete=False,
            )
        elif is_dataclass(value) and not isinstance(value, type):
            result = replace(
                value, **{f.name: local(getattr(value, f.name)) for f in fields(value) if f.compare}
            )
        elif isinstance(value, tuple):
            result = tuple(local(v) for v in value)
        else:
            result = value
        # Replacements create temporary descriptors. Keep their original objects
        # alive too, so Python cannot recycle an identity inside this traversal.
        cache[id(original)] = (original, result)
        return result

    return local(site.operation.call), local(site.arguments), axes


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


def _check_protocol(
    view: ProgramView,
    site: CallSite,
    budget: AnalysisBudget,
    execution_cores: int | None,
    execution: CoreExecutionBinding | None,
) -> tuple[RewriteEvidence, CoreExecutionEvidence | None]:
    call = site.operation.call
    assert call is not None
    if _has_remote(call):
        return (
            normalize_view(
                view, budget=budget, execution_cores=execution_cores, execution=execution
            ),
            None,
        )
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

    arguments = tuple(
        binding_value(local_input(a), budget=budget, coordinates=coordinates, ranges=ranges)
        for a in site.arguments
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
    return (
        RewriteEvidence(
            "device-local-bound-core-resource-check"
            if execution
            else "device-local-resource-check",
            1,
            call.kernel.origin.path,
            projection,
            (),
            bindings=site.arguments,
        ),
        core_evidence,
    )


def compare_pallas_kernels(
    left: Any,
    right: Any,
    *,
    check_protocol: bool = False,
    analysis_budget: int = 10000,
    execution_cores: int | None = None,
    execution: CoreExecutionBinding | None = None,
) -> LocalKernelComparison:
    """Compare one kernel per source graph, retaining lexical and axis bindings.

    A DIFFERENT result is a structural diagnostic. Resource checking is optional
    and separate; unsupported peer/control proofs raise instead of being assumed.
    execution_cores=1 binds the launch to core 0 on each device. This is a caller
    supplied execution condition, recorded and replayed with the result.
    execution supplies exact multi-core program streams for a conditional proof;
    it does not configure native scheduling and requires check_protocol=True.
    Remote protocols accept only one core-0 stream, which the caller establishes
    identically on every device. execution_cores alone does not bind grid order.
    """
    if type(check_protocol) is not bool:
        raise ValueError("check_protocol must be a bool")
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
            left, right, check_protocol, AnalysisBudget(analysis_budget), execution_cores, execution
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
    evidence = LocalKernelEvidence(x, y, _axes(x)) if equal else None
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
    )
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
