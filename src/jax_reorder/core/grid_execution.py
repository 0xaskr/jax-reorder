"""Interleaved grid state through disjoint actual scratch banks, not private copies."""

from collections.abc import Callable
from itertools import product
from math import prod

from ..errors import UnsupportedJaxprError
from .grid_state import independent_domains
from .index_domain import AnalysisBudget, IndexExpr, integer
from .memory import Expression, MemoryProjection, Predicate, ValueExpr, scalar_type
from .model import PallasCallView, SemaphoreType
from .tensor_storage import Box


def functionalize_banked_grid(
    call: PallasCallView,
    *,
    budget: AnalysisBudget,
    coordinates: dict[str, IndexExpr] | None,
    axis_ranges: dict[str, tuple[int, int]] | None,
    execution_cores: int | None,
    project: Callable[[dict[str, IndexExpr], dict[str, tuple[int, int]]], MemoryProjection],
) -> MemoryProjection:
    path = call.kernel.origin.path + ".interleaved_banks"
    if execution_cores != 1:
        raise UnsupportedJaxprError(
            path,
            "stateful grid requires a parallel prefix and serial suffix or an explicit "
            "single-core binding for banked state; native owner-to-core stability is unbound",
        )
    # Output accumulators would revisit their staging buffer around intervening
    # owners. First admit only unique complete output blocks for every program.
    domains = independent_domains(call, budget).domains
    if any(
        isinstance(s.type.inner, SemaphoreType) and s.type.inner.kind == "regular"
        for s in call.scratch
    ):
        raise UnsupportedJaxprError(
            path, "interleaved bank proof does not bind regular semaphore core routes"
        )
    axes = tuple(i for i, g in enumerate(call.grid) if g.size > 1 and g.semantics == "parallel")
    budget.spend(path + ".owners", prod(call.grid[i].size for i in axes))
    projections: list[tuple[tuple[int, ...], MemoryProjection]] = []
    scratch_start = len(call.prefetch) + call.num_inputs + call.num_outputs
    allocated = set(range(scratch_start, len(call.kernel.inputs)))
    occupied: dict[int, list[tuple[int, Box]]] = {}
    for owner, point in enumerate(product(*(range(call.grid[i].size) for i in axes))):
        selected = {f"program:{i}": k for i, k in zip(axes, point, strict=True)}
        projection = project(
            {**(coordinates or {}), **{key: integer(k) for key, k in selected.items()}},
            {**(axis_ranges or {}), **{key: (k, k) for key, k in selected.items()}},
        )
        regions = {
            (v.storage, Box(v.rectangle)) for v in projection.versions if v.storage in allocated
        }
        for resource, box in regions:
            for earlier, previous in occupied.get(resource, ()):
                if earlier != owner:
                    budget.spend(path + ".overlap")
                    if box.intersection(previous) is not None:
                        raise UnsupportedJaxprError(
                            path,
                            "stateful grid requires a parallel prefix and serial suffix unless "
                            "actual scratch banks are disjoint; interleaved owners overlap",
                        )
            occupied.setdefault(resource, []).append((owner, box))
        projections.append((point, projection))
    outputs = []
    for output, domain in enumerate(domains[call.num_inputs :]):
        result: Expression | None = None
        for point, projection in reversed(projections):
            value = projection.outputs[output]
            if result is None:
                result = value
            else:
                budget.spend(path + ".output_piece")
                predicates = tuple(
                    Predicate("eq", (domain.inverse[f"program:{i}"], integer(k)))
                    for i, k in zip(axes, point, strict=True)
                )
                result = ValueExpr(
                    "select_n",
                    scalar_type(call.blocks[call.num_inputs + output].array),
                    (Predicate("and", predicates), result, value),
                )
        assert result is not None
        outputs.append(result)
    return MemoryProjection(
        tuple(outputs),
        domains,
        tuple(v for _, p in projections for v in p.versions),
        tuple(o for _, p in projections for o in p.origins),
        (
            "caller binds execution to one local core (core 0); "
            "no owner-to-core mapping is inferred",
            "native arbitrary dimensions retain lexicographic order for each parallel owner",
            "actual call-lifetime scratch writes of distinct owners occupy disjoint rectangles",
            "each owner's scratch reads are initialized by its own preceding writes",
            "each program completes DMA before any intervening parallel program",
            "complete output blocks have unique writers; no output staging revisit is assumed",
        ),
        tuple(t for _, p in projections for t in p.communication),
        tuple(w for _, p in projections for w in p.waits),
        assumptions=tuple(dict.fromkeys(a for _, p in projections for a in p.assumptions)),
    )
