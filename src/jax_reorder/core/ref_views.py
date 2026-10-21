"""Exact static Ref coordinate maps and footprints over unchanged storage."""

from __future__ import annotations

from itertools import product
from math import prod

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, IndexExpr, add, divide, integer, modulo, scale, substitute
from .model import RefTransform, SliceIndex
from .storage import Span
from .tensor_domain import coordinate
from .tensor_storage import Box, full_box


def _unresolved(path: str) -> IndexExpr:
    raise UnsupportedJaxprError(path, "Ref address must be specialized before coordinate mapping")


def view_coordinates(
    storage_shape: tuple[int, ...],
    transforms: tuple[RefTransform, ...],
    path: str,
    budget: AnalysisBudget,
) -> tuple[tuple[int, ...], dict[str, IndexExpr], dict[str, IndexExpr]]:
    budget.spend(
        path + ".ref_views", sum(1 + len(t.dimensions) + len(t.indices) for t in transforms)
    )

    def reshape(
        coordinates: tuple[IndexExpr, ...], source: tuple[int, ...], target: tuple[int, ...]
    ) -> tuple[IndexExpr, ...]:
        common = 0
        for a, b in zip(reversed(source), reversed(target), strict=False):
            if a != b:
                break
            common += 1
        if common:
            # Equal row-major tail dimensions keep their coordinates exactly.
            # Re-linearizing them would introduce redundant div/mod guards,
            # obscuring both storage identity and padded-lane validity.
            return (
                reshape(coordinates[:-common], source[:-common], target[:-common])
                + coordinates[-common:]
            )
        linear = add(*(scale(c, prod(source[i + 1 :])) for i, c in enumerate(coordinates)))
        return tuple(modulo(divide(linear, prod(target[i + 1 :])), n) for i, n in enumerate(target))

    shapes = [storage_shape]
    forward = tuple(coordinate("local", i) for i in range(len(storage_shape)))
    for transform in transforms:
        shape = shapes[-1]
        dimensions = transform.dimensions
        if transform.kind != "index" and transform.indices:
            raise UnsupportedJaxprError(path, "non-index Ref transform carries index metadata")
        if transform.kind == "reshape":
            if any(n <= 0 for n in dimensions) or prod(dimensions) != prod(shape):
                raise UnsupportedJaxprError(path, "Ref reshape changes its element count")
            forward = reshape(forward, shape, dimensions)
            shape = dimensions
        elif transform.kind == "transpose":
            if sorted(dimensions) != list(range(len(shape))):
                raise UnsupportedJaxprError(path, "Ref transpose is not a permutation")
            shape = tuple(shape[i] for i in dimensions)
            forward = tuple(forward[i] for i in dimensions)
        elif transform.kind == "index":
            if dimensions or len(transform.indices) != len(shape):
                raise UnsupportedJaxprError(path, "Ref index transform rank differs from its view")
            selected, selected_shape = [], []
            for index, extent, coord in zip(transform.indices, shape, forward, strict=True):
                if isinstance(index, SliceIndex):
                    if (
                        index.size < 1
                        or index.stride < 1
                        or index.start < 0
                        or index.start + (index.size - 1) * index.stride >= extent
                    ):
                        raise UnsupportedJaxprError(
                            path, "Ref slice is outside its intermediate view"
                        )
                    selected_shape.append(index.size)
                    selected.append(divide(add(coord, integer(-index.start)), index.stride))
                elif type(index) is not int or not 0 <= index < extent:
                    raise UnsupportedJaxprError(
                        path, "Ref scalar index is outside its intermediate view"
                    )
            forward, shape = tuple(selected), tuple(selected_shape)
        else:
            raise UnsupportedJaxprError(path, "unknown Ref coordinate transform")
        shapes.append(shape)
    backward = tuple(coordinate("local", i) for i in range(len(shapes[-1])))
    for i in range(len(transforms) - 1, -1, -1):
        transform = transforms[i]
        if transform.kind == "reshape":
            backward = reshape(backward, shapes[i + 1], shapes[i])
        elif transform.kind == "transpose":
            backward = tuple(backward[transform.dimensions.index(j)] for j in range(len(backward)))
        else:
            expanded = []
            current = iter(backward)
            for index in transform.indices:
                expanded.append(
                    add(integer(index.start), scale(next(current), index.stride))
                    if isinstance(index, SliceIndex)
                    else integer(index)
                    if type(index) is int
                    else _unresolved(path)
                )
            backward = tuple(expanded)
    return (
        shapes[-1],
        {f"local:{i}": c for i, c in enumerate(backward)},
        {f"local:{i}": c for i, c in enumerate(forward)},
    )


def view_footprint(
    storage_shape: tuple[int, ...],
    result_shape: tuple[int, ...],
    backward: dict[str, IndexExpr],
    transforms: tuple[RefTransform, ...],
    path: str,
    budget: AnalysisBudget,
) -> tuple[Box, ...]:
    if not any(t.kind == "index" for t in transforms):
        return (full_box(storage_shape),)
    size = prod(result_shape)
    budget.spend(path + ".ref_view_points", size)
    points = []
    for point in product(*(range(n) for n in result_shape)):
        bindings = {f"local:{i}": integer(k) for i, k in enumerate(point)}
        address = tuple(
            substitute(backward[f"local:{i}"], bindings) for i in range(len(storage_shape))
        )
        if any(a.kind != "constant" for a in address):
            raise UnsupportedJaxprError(path, "Ref footprint did not resolve to static coordinates")
        coords = tuple(int(a.value) for a in address)
        if any(not 0 <= k < n for k, n in zip(coords, storage_shape, strict=True)):
            raise UnsupportedJaxprError(path, "Ref footprint leaves original storage")
        points.append(coords)
    if len(set(points)) != size:
        raise UnsupportedJaxprError(path, "Ref view is not injective")
    boxes = [Box(tuple(Span(k, k + 1) for k in point)) for point in sorted(points)]
    # Coalesce exact adjacent boxes, never replace a scattered footprint by
    # its bounding rectangle. A transpose changes order, not storage identity.
    for axis in reversed(range(len(storage_shape))):
        groups: dict[tuple[Span, ...], list[Box]] = {}
        for box in boxes:
            key = box.axes[:axis] + box.axes[axis + 1 :]
            groups.setdefault(key, []).append(box)
        boxes = []
        for group in groups.values():
            group.sort(key=lambda box: box.axes[axis])
            current = group[0]
            for following in group[1:]:
                if current.axes[axis].stop == following.axes[axis].start:
                    axes = list(current.axes)
                    axes[axis] = Span(current.axes[axis].start, following.axes[axis].stop)
                    current = Box(tuple(axes))
                else:
                    boxes.append(current)
                    current = following
            boxes.append(current)
    return tuple(sorted(boxes))
