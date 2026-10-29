"""Mathematical orchestration of explicitly scoped device-local Pallas values."""

from dataclasses import replace
from typing import Any

from ..core.adapters.jax import adapt
from ..core.canonical import canonicalize
from ..core.communication import DMA_PROGRESS_ASSUMPTION
from ..core.core_execution import CoreExecutionBinding
from ..core.index_domain import AnalysisBudget
from ..core.local import LocalKernelEvidence, _axes, _key, project_local_view
from ..core.normalization import _outputs_equal
from ..errors import ComparisonContractError, ComparisonError
from ..status import Status
from .adapter.pallas import freeze_local_projection
from .budget import Budget
from .result import ASSUMPTIONS, AlgebraComparison
from .trusted import isolated_compare


def compare_local_values(
    left: Any,
    right: Any,
    *,
    algebra_budget: int,
    algebra_timeout: float,
    executions: tuple[CoreExecutionBinding | None, CoreExecutionBinding | None],
    output_boundary: str = "kernel",
) -> AlgebraComparison:
    budget = Budget(algebra_budget)
    analysis = AnalysisBudget(algebra_budget)
    views = tuple(
        adapt(canonicalize(g, include_derivative_rules=False).jaxpr).view for g in (left, right)
    )
    if views[0].context != views[1].context:
        raise ComparisonContractError("local.context", "adaptation environments differ")
    sites, proofs, cores = [], [], []
    for side, view, execution in zip(("left", "right"), views, executions, strict=True):
        try:
            site, proof, core = project_local_view(
                view,
                budget=analysis,
                defer_padding=True,
                execution=execution,
                output_boundary=output_boundary,
            )
        except ComparisonError as error:
            raise error.at(side) from error
        sites.append(site)
        proofs.append(proof)
        cores.append(core)
    if proofs[0].boundaries != proofs[1].boundaries:
        raise ComparisonContractError(
            "local.contract", "local inputs, outputs or axis bindings differ"
        )
    assumptions = (
        (
            "device-local kernel values observed through complete ordered parent output views"
            if output_boundary == "parent"
            else "device-local kernel values only; parent output assembly "
            "and postprocessing are excluded"
        ),
        "corresponding source input ports, local signatures and bound logical axes",
    )
    if any(proof.projection.communication or proof.remote for proof in proofs):
        assumptions += (DMA_PROGRESS_ASSUMPTION,)
    assumptions += tuple(
        dict.fromkeys(value for proof in proofs for value in proof.projection.assumptions)
    )
    assumptions += tuple(
        dict.fromkeys(
            value
            for proof in proofs
            if proof.remote is not None
            for value in proof.remote.protocol.protocol.assumptions
        )
    )
    if any(binding is not None for binding in executions):
        assumptions += tuple(
            dict.fromkeys(
                value for core in cores if core is not None for value in core.protocol.assumptions
            )
        )
        assumptions += (
            "conditional on the caller-supplied exact core program mapping "
            "and order for each source",
        )
    kind = (
        "structural"
        if _key(sites[0], output_boundary) == _key(sites[1], output_boundary)
        else "normalized"
    )
    if not any(proof.deferred_padding for proof in proofs) and (
        kind == "structural" or _outputs_equal(proofs[0], proofs[1], analysis)
    ):
        return AlgebraComparison(
            Status.EQUIVALENT,
            rules=(kind,),
            assumptions=assumptions,
            warnings=(),
            work=analysis.used,
            normalizations=tuple(proofs),
            scope="device_local",
            local_evidence=LocalKernelEvidence(sites[0], sites[1], _axes(sites[0]), kind),
            core_executions=tuple(cores),
        )
    budget.spend(analysis.used, path="algebra.pallas.local")
    frozen = tuple(freeze_local_projection(proof, budget) for proof in proofs)
    budget.spend(path="algebra.worker")
    result = isolated_compare(
        *frozen,
        algebra_budget=algebra_budget - budget.used + 1,
        timeout=algebra_timeout,
    )
    result = replace(result, work=result.work + budget.used - 1)
    return AlgebraComparison(
        Status(result.status),
        rules=result.rules,
        assumptions=assumptions + ASSUMPTIONS,
        work=result.work,
        backend_result=result,
        normalizations=tuple(proofs),
        scope="device_local",
        core_executions=tuple(cores),
    )
