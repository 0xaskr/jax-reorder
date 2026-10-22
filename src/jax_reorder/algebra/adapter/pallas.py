"""Bridge proven Pallas value projections to the isolated mathematical worker."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import prod
from types import SimpleNamespace
from typing import cast

from jax._src import core

from ...core.adapters.jax import adapt
from ...core.adapters.pallas.structural import pallas_call_p
from ...core.core_execution import CoreExecutionBinding
from ...core.index_domain import AnalysisBudget, integer
from ...core.memory import _substitute
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
    return _projection_graph(evidence, inputs, outputs), evidence


def freeze_local_projection(evidence: RewriteEvidence, budget: Budget) -> SymbolicGraph:
    boundary = evidence.boundaries[0]
    from ...core.model import ArrayType

    types = boundary.inputs + boundary.outputs
    if not all(isinstance(v, ArrayType) for v in types):
        raise UnsupportedJaxprError("algebra.pallas.local", "local contract requires array values")
    outputs_types = tuple(cast(ArrayType, v) for v in boundary.outputs)
    axes = cast(
        tuple[tuple[str, int, str], ...],
        next(p.value for p in boundary.parameters if p.name == "axes"),
    )
    if axes and evidence.remote is None:
        # Instantiating the proven finite device environment prevents a CAS
        # counterexample from choosing an out-of-range logical device index.
        budget.spend(prod(a[1] for a in axes), path="algebra.pallas.device_instances")
        points = tuple(product(*(range(a[1]) for a in axes)))
        roots = tuple(
            _substitute(
                root, {"device:" + a[0]: integer(v) for a, v in zip(axes, point, strict=True)}
            )
            for root in evidence.projection.outputs
            for point in points
        )
        outputs_types = tuple(v for v in outputs_types for _ in points)
        evidence = replace(evidence, projection=replace(evidence.projection, outputs=roots))
    inputs = tuple(
        Variable(i, cast(ArrayType, v).shape, cast(ArrayType, v).dtype)
        for i, v in enumerate(boundary.inputs)
    )
    outputs = tuple(
        Variable(len(inputs) + i, v.shape, v.dtype) for i, v in enumerate(outputs_types)
    )
    return _projection_graph(evidence, inputs, outputs)


def _projection_graph(
    evidence: RewriteEvidence, inputs: tuple[Variable, ...], outputs: tuple[Variable, ...]
) -> SymbolicGraph:
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
    )
