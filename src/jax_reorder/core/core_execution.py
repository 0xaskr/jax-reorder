"""Conditional local-core routing proofs over explicitly supplied program streams."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, IndexExpr, bounds, integer
from .memory import Expression, MemoryProjection, Predicate, ValueExpr, scalar_type
from .model import Origin, PallasCallView
from .protocol import (
    Command,
    DeviceProgram,
    Instance,
    ProtocolEvidence,
    Semaphore,
    Signal,
    Wait,
    analyze_protocol,
)
from .tensor_domain import TensorDomain, coordinate
from .tensor_memory import functionalize_tensor_call


@dataclass(frozen=True)
class CoreExecutionBinding:
    """Caller-supplied launch condition, not a compiler setting.

    ``programs[c]`` is the ordered sequence of grid coordinates executed by
    local core ``c``. The caller must establish that the actual launch uses
    exactly this mapping and order. Ordinary pallas_call does not establish it.
    """

    programs: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class CoreProgramProjection:
    core: int
    point: tuple[int, ...]
    projection: MemoryProjection


@dataclass(frozen=True)
class CoreExecutionEvidence:
    binding: CoreExecutionBinding
    protocol: ProtocolEvidence
    programs: tuple[CoreProgramProjection, ...]


def validate_binding_shape(binding: CoreExecutionBinding) -> None:
    """Reject malformed caller conditions before graph-specific proof work."""
    if not isinstance(binding, CoreExecutionBinding) or type(binding.programs) is not tuple:
        raise ValueError("execution must be a CoreExecutionBinding with tuple program streams")
    if not binding.programs:
        raise ValueError("execution must name at least one local core")
    if any(type(stream) is not tuple for stream in binding.programs):
        raise ValueError("each core program stream must be a tuple")


def validate_binding(
    binding: CoreExecutionBinding, call: PallasCallView, budget: AnalysisBudget
) -> None:
    path = "local.core_execution"
    validate_binding_shape(binding)
    grid = tuple(g.size for g in call.grid)
    budget.spend(path + ".grid", prod(grid))
    budget.spend(path + ".cores", len(binding.programs))
    seen: set[tuple[int, ...]] = set()
    owners: dict[tuple[int, ...], int] = {}
    parallel = tuple(i for i, g in enumerate(call.grid) if g.semantics == "parallel")
    if any(g.semantics not in ("parallel", "arbitrary") for g in call.grid):
        raise UnsupportedJaxprError(path, "explicit binding requires reviewed grid semantics")
    for core, stream in enumerate(binding.programs):
        previous = None
        for point in stream:
            budget.spend(path + ".points", max(1, len(grid)))
            if (
                type(point) is not tuple
                or len(point) != len(grid)
                or any(
                    type(v) is not int or not 0 <= v < n for v, n in zip(point, grid, strict=True)
                )
            ):
                raise UnsupportedJaxprError(path, "core binding has an out-of-range grid point")
            if point in seen:
                raise UnsupportedJaxprError(path, "core binding repeats a grid point")
            if previous is not None and point <= previous:
                raise UnsupportedJaxprError(
                    path, "core binding must preserve the native lexicographic grid order"
                )
            previous = point
            seen.add(point)
            owner = tuple(point[i] for i in parallel)
            if owners.setdefault(owner, core) != core:
                raise UnsupportedJaxprError(
                    path, "arbitrary grid iterations for one parallel owner must use one core"
                )
    if len(seen) != prod(grid):
        raise UnsupportedJaxprError(path, "core binding does not cover the complete grid")


def functionalize_core_execution(
    call: PallasCallView,
    binding: CoreExecutionBinding,
    *,
    budget: AnalysisBudget,
    inputs: tuple[Expression, ...] | None = None,
    coordinates: dict[str, IndexExpr] | None = None,
    axis_ranges: dict[str, tuple[int, int]] | None = None,
) -> tuple[MemoryProjection, CoreExecutionEvidence]:
    """Prove independent local values and exact cross-core regular-counter topology.

    Every invocation proves its memory from unknown scratch, so retaining a real
    per-core allocation across programs cannot introduce an unmodeled read.
    Semaphore state, in contrast, is retained across the complete core streams.
    """
    validate_binding(binding, call, budget)
    commands: list[list[Command]] = [[] for _ in binding.programs]
    resources: set[int] = set()
    pieces: list[CoreProgramProjection] = []
    for core, stream in enumerate(binding.programs):
        for point in stream:

            def observe(
                phase: str,
                resource: int,
                target: int,
                amount: int,
                origin: Origin,
                core: int = core,
            ) -> None:
                resources.add(resource)
                if phase == "signal":
                    if not 0 <= target < len(binding.programs):
                        raise UnsupportedJaxprError(origin.path, "semaphore target core is unbound")
                    commands[core].append(Signal(Instance(target, resource), amount, origin))
                elif phase == "wait":
                    commands[core].append(Wait(Instance(core, resource), "regular", amount, origin))
                else:
                    raise UnsupportedJaxprError(origin.path, "unreviewed explicit core operation")

            projection = functionalize_tensor_call(
                call,
                budget=budget,
                inputs=inputs,
                coordinates=coordinates,
                axis_ranges=axis_ranges,
                program_point=point,
                semaphore_observer=observe,
            )
            pieces.append(CoreProgramProjection(core, point, projection))
    protocol = analyze_protocol(
        tuple(DeviceProgram(core, tuple(stream)) for core, stream in enumerate(commands)),
        tuple(
            Semaphore(Instance(core, resource), 0, "regular")
            for core in range(len(binding.programs))
            for resource in sorted(resources)
        ),
        budget=budget,
    )
    protocol = replace(
        protocol,
        assumptions=(
            "participating local cores do not fail",
            "enabled core streams are not permanently starved",
            "issued valid cross-core semaphore signals eventually become visible",
            "protocol participant IDs denote local cores under the explicit binding",
        ),
    )
    outputs: list[Expression] = []
    domains = pieces[0].projection.domains
    for output, raw_domain in enumerate(domains[call.num_inputs :]):
        assert isinstance(raw_domain, TensorDomain)
        domain = raw_domain
        result: Expression | None = None
        for item in reversed(pieces):
            value = item.projection.outputs[output]
            point_ranges = {f"program:{i}": (k, k) for i, k in enumerate(item.point)}
            predicates = []
            for i, (mapping, width) in enumerate(
                zip(domain.mapping, domain.block_shape, strict=True)
            ):
                block, last = bounds(mapping, point_ranges)
                assert block == last
                predicates.extend(
                    (
                        Predicate("ge", (coordinate("output", i), integer(block * width))),
                        Predicate("lt", (coordinate("output", i), integer((block + 1) * width))),
                    )
                )
            if result is None:
                result = value
            else:
                budget.spend("local.core_execution.output_piece")
                result = ValueExpr(
                    "select_n",
                    scalar_type(call.blocks[call.num_inputs + output].array),
                    (Predicate("and", tuple(predicates)), result, value),
                )
        assert result is not None
        outputs.append(result)
    combined = MemoryProjection(
        tuple(outputs),
        domains,
        tuple(v for item in pieces for v in item.projection.versions),
        tuple(o for item in pieces for o in item.projection.origins),
        (
            "conditional on the caller-supplied exact core mapping and ordered program streams",
            "ordinary pallas_call does not enforce this execution binding",
            "each program initializes local scratch and completes local DMA independently",
            "each complete output block has one writer; no inter-core shared value is inferred",
            "regular semaphore targets and waits satisfy the complete bound core topology",
        ),
        tuple(t for item in pieces for t in item.projection.communication),
        tuple(w for item in pieces for w in item.projection.waits),
    )
    return combined, CoreExecutionEvidence(binding, protocol, tuple(pieces))
