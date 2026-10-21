"""Bridge proven Pallas value projections to the isolated mathematical worker."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from jax._src import core

from ...core.adapters.jax import adapt
from ...core.adapters.pallas.structural import pallas_call_p
from ...core.core_execution import CoreExecutionBinding
from ...core.index_domain import AnalysisBudget
from ...core.normalization import RewriteEvidence, normalize_view
from ...core.params import parameter_view
from ...errors import ComparisonBudgetError, ComparisonStage, UnsupportedJaxprError
from .._typing import SymbolicGraph
from ..budget import Budget
from .transport import Variable


def contains_pallas(graph: core.Jaxpr) -> bool:
    pending, seen = [graph], set()
    while pending:
        region = pending.pop()
        if region in seen:
            continue
        seen.add(region)
        for eq in region.eqns:
            if eq.primitive is pallas_call_p:
                return True
            _, children = parameter_view(
                eq, lambda _: None, include_derivative_rules=False, parent=region
            )
            pending.extend(children.values())
    return False


def project(
    graph: core.Jaxpr,
    budget: Budget,
    *,
    execution: CoreExecutionBinding | None = None,
) -> tuple[SymbolicGraph, RewriteEvidence]:
    # Adaptation admits every operation, even if its value is unused. Effects
    # disappear only after the memory/protocol consumer proves its preconditions.
    budget.spend(path="algebra.pallas.adapt")
    view = adapt(graph).view
    if budget.used == budget.limit:
        budget.spend(path="algebra.pallas.projection")
    analysis = AnalysisBudget(budget.limit - budget.used)
    try:
        evidence = normalize_view(view, budget=analysis, defer_padding=True, execution=execution)
    except ComparisonBudgetError as error:
        raise ComparisonBudgetError(
            error.path,
            error.reason,
            stage=ComparisonStage.ALGEBRA,
            failure_kind=error.failure_kind,
            **error.diagnostics,
        ) from error
    budget.spend(analysis.used, path="algebra.pallas.projection")
    if evidence.remote is None and any(
        boundary.shard_map is not None for boundary in evidence.boundaries
    ):
        raise UnsupportedJaxprError(
            "algebra.pallas", "shard-local projections need global shard assembly before CAS"
        )

    def variable(i: int, v: core.Atom) -> Variable:
        aval = cast(core.ShapedArray, v.aval)
        return Variable(i, tuple(aval.shape), aval.dtype)

    inputs = tuple(variable(i, v) for i, v in enumerate(graph.invars))
    outputs = tuple(variable(len(inputs) + i, v) for i, v in enumerate(graph.outvars))
    return cast(
        SymbolicGraph,
        SimpleNamespace(
            invars=inputs,
            constvars=(),
            consts=(),
            outvars=outputs,
            eqns=(
                SimpleNamespace(
                    primitive=SimpleNamespace(name="pallas_global_values"),
                    params={
                        "projection": evidence.projection,
                        "rule": f"{evidence.rule}:v{evidence.version}",
                    },
                    invars=inputs,
                    outvars=outputs,
                ),
            ),
        ),
    ), evidence
