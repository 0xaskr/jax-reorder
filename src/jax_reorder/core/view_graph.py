"""Initial structural consumer and independent witness checks for source views.

The fast path currently keeps source equation order. It is deliberately narrower
than the existing JAX canonicalizer; a mismatch is only a structural diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import CorrespondenceError
from .model import ProgramView, RegionView


@dataclass(frozen=True)
class ViewGraph:
    # Views already use region-local SSA IDs, independent of JAX variable names.
    root: RegionView


@dataclass(frozen=True)
class RegionCorrespondence:
    left: str
    right: str
    equations: tuple[tuple[int, int], ...]
    values: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StructuralEvidence:
    regions: tuple[RegionCorrespondence, ...]
    kind: str = "structural"


def _regions(root: RegionView) -> tuple[RegionView, ...]:
    result = [root]
    for op in root.operations:
        for binding in op.regions:
            result.extend(_regions(binding.region))
        if op.call is not None:
            result.extend(_regions(op.call.kernel))
            for block in op.call.blocks:
                result.extend(_regions(block.index_map))
    return tuple(result)


def _values(region: RegionView) -> tuple[int, ...]:
    return (
        tuple(v.id for v, _ in region.constants)
        + tuple(v.id for v in region.inputs)
        + tuple(v.id for op in region.operations for v in op.results)
    )


def canonicalize_view(view: ProgramView) -> ViewGraph:
    return ViewGraph(view.root)


def verify_view_correspondence(
    left: ProgramView, right: ProgramView, witness: StructuralEvidence
) -> None:
    """Verify against original views, without calling the graph matcher."""
    a, b = _regions(left.root), _regions(right.root)
    if left.context != right.context or len(a) != len(b) or len(a) != len(witness.regions):
        raise CorrespondenceError("view", "incomplete region/context correspondence")
    for x, y, pair in zip(a, b, witness.regions, strict=True):
        if (pair.left, pair.right) != (x.origin.path, y.origin.path):
            raise CorrespondenceError("view", "incorrect region origins")
        expected_eqs = tuple((op.id, op.id) for op in x.operations)
        expected_values = tuple((i, i) for i in _values(x))
        if pair.equations != expected_eqs or pair.values != expected_values:
            raise CorrespondenceError(x.origin.path, "incomplete or non-bijective correspondence")
        # Direct source-view field comparison covers every port, effect, access,
        # constant and nested binding. It shares source adaptation, not matching.
        if x != y:
            raise CorrespondenceError(
                x.origin.path, "original views do not preserve complete semantics"
            )


def match_views(left: ProgramView, right: ProgramView) -> StructuralEvidence | None:
    if left.context != right.context or canonicalize_view(left) != canonicalize_view(right):
        return None
    regions = tuple(
        RegionCorrespondence(
            a.origin.path,
            b.origin.path,
            tuple((op.id, op.id) for op in a.operations),
            tuple((i, i) for i in _values(a)),
        )
        for a, b in zip(_regions(left.root), _regions(right.root), strict=True)
    )
    evidence = StructuralEvidence(regions)
    verify_view_correspondence(left, right, evidence)
    return evidence
