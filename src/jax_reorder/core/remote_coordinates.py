"""Exact row-major addresses for finite remote Ref views, without tensor lanes."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from math import prod

from ..errors import UnsupportedJaxprError
from .index_domain import (
    AnalysisBudget,
    IndexExpr,
    add,
    axis,
    divide,
    integer,
    modulo,
    scale,
    substitute,
)
from .memory import Expression, _substitute
from .model import (
    ArrayType,
    DynamicSliceIndex,
    LiteralValue,
    Operand,
    RefTransform,
    SliceIndex,
    TensorIndex,
    ValueRef,
)
from .ref_views import view_coordinates
from .storage import Span
from .tensor_storage import Box, full_box


def linear(coordinates: tuple[IndexExpr, ...], shape: tuple[int, ...]) -> IndexExpr:
    return add(*(scale(c, prod(shape[i + 1 :])) for i, c in enumerate(coordinates)))


def coordinates(value: IndexExpr, shape: tuple[int, ...]) -> tuple[IndexExpr, ...]:
    # The leading coordinate needs no modulo: all callers prove the storage domain.
    return tuple(
        divide(value, prod(shape[i + 1 :]))
        if i == 0
        else modulo(divide(value, prod(shape[i + 1 :])), n)
        for i, n in enumerate(shape)
    )


def localize(value: Expression, shape: tuple[int, ...], budget: AnalysisBudget) -> Expression:
    """Apply the exact row-major coordinate inverse before composing a new view."""
    from .tensor_domain import coordinate_ranges
    from .tensor_values import project_coordinates

    local = tuple(axis(f"local:{i}") for i in range(len(shape)))
    identities = dict(zip(coordinates(axis("global"), shape), local, strict=True))
    return project_coordinates(
        value,
        {},
        identities,
        {**coordinate_ranges("local", shape), "global": (0, prod(shape) - 1)},
        budget,
    )


def merge_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    result: list[Span] = []
    for span in sorted(spans):
        if result and result[-1].stop == span.start:
            result[-1] = Span(result[-1].start, span.stop)
        elif result and result[-1].stop > span.start:
            raise ValueError("remote footprint overlaps itself")
        else:
            result.append(span)
    return tuple(result)


def box_spans(
    box: Box, shape: tuple[int, ...], budget: AnalysisBudget, path: str
) -> tuple[Span, ...]:
    if not shape:
        return (Span(0, 1),)
    # Collapse the contiguous suffix before enumeration. A full enormous tensor
    # is one interval; a column tile costs rows, never rows times tile width.
    split = len(shape) - 1
    while split > 0 and box.axes[split] == Span(0, shape[split]):
        split -= 1
    count = prod(a.size for a in box.axes[:split])
    budget.spend(path + ".linear_spans", count)
    stride = prod(shape[split + 1 :])
    result = []
    for prefix in product(*(range(a.start, a.stop) for a in box.axes[:split])):
        offset = sum(p * prod(shape[i + 1 :]) for i, p in enumerate(prefix))
        result.append(
            Span(offset + box.axes[split].start * stride, offset + box.axes[split].stop * stride)
        )
    return tuple(result)


def span_boxes(
    span: Span, shape: tuple[int, ...], budget: AnalysisBudget, path: str
) -> tuple[Box, ...]:
    """Split a row-major interval into maximal aligned rectangular suffixes."""
    if not shape:
        return (Box(()),)
    result = []
    start = span.start
    while start < span.stop:
        budget.spend(path + ".reshape_boxes")
        point = tuple((start // prod(shape[i + 1 :])) % n for i, n in enumerate(shape))
        dimension = len(shape) - 1
        while (
            dimension > 0 and point[dimension] == 0 and prod(shape[dimension:]) <= span.stop - start
        ):
            dimension -= 1
        stride = prod(shape[dimension + 1 :])
        size = min(shape[dimension] - point[dimension], (span.stop - start) // stride)
        result.append(
            Box(
                tuple(Span(p, p + 1) for p in point[:dimension])
                + (Span(point[dimension], point[dimension] + size),)
                + tuple(Span(0, n) for n in shape[dimension + 1 :])
            )
        )
        start += size * stride
    return tuple(result)


@dataclass(frozen=True)
class RemoteView:
    shape: tuple[int, ...]
    spans: tuple[Span, ...]
    to_storage: IndexExpr
    to_view: IndexExpr
    conditional: bool = False

    @property
    def bound(self) -> Span:
        return Span(self.spans[0].start, self.spans[-1].stop)


def remote_view(
    shape: tuple[int, ...],
    index: SliceIndex | TensorIndex | Operand | None,
    transforms: tuple[RefTransform, ...],
    read: Callable[[Operand], int],
    budget: AnalysisBudget,
    path: str,
    *,
    boxes: tuple[Box, ...] | None = None,
) -> RemoteView:
    def fixed(item: SliceIndex | DynamicSliceIndex | int | Operand) -> SliceIndex | int:
        if isinstance(item, DynamicSliceIndex):
            return SliceIndex(read(item.start), item.size, item.stride)
        if isinstance(item, ValueRef | LiteralValue):
            return read(item)
        return item

    chain = tuple(replace(t, indices=tuple(fixed(i) for i in t.indices)) for t in transforms)
    if transforms and index is not None:
        full = tuple(SliceIndex(0, n, 1) for n in shape)
        if not (
            (
                isinstance(index, TensorIndex)
                and index.shape == shape
                and not index.integer_shape
                and index.indices == full
            )
            or (len(shape) == 1 and isinstance(index, SliceIndex) and index == full[0])
        ):
            raise UnsupportedJaxprError(
                path, "remote transformed Ref requires a full original-storage base"
            )
        index = None
    current = shape
    for transform in chain:
        if transform.kind == "reshape":
            current = transform.dimensions
        elif transform.kind == "transpose":
            current = tuple(current[i] for i in transform.dimensions)
        else:
            current = tuple(i.size for i in transform.indices if isinstance(i, SliceIndex))
    if isinstance(index, TensorIndex):
        if index.shape != current or index.integer_shape:
            raise UnsupportedJaxprError(path, "remote Ref index has an incompatible tensor shape")
        selected = tuple(fixed(i) for i in index.indices)
    elif index is not None:
        if len(current) != 1:
            raise UnsupportedJaxprError(path, "remote scalar/slice index requires a vector Ref")
        selected = (fixed(index),)
    else:
        selected = tuple(SliceIndex(0, n, 1) for n in current)
    if selected != tuple(SliceIndex(0, n, 1) for n in current):
        chain += (RefTransform("index", (), selected),)
    output_shape, backward, forward = view_coordinates(shape, chain, path, budget)
    boxes = (full_box(output_shape),) if boxes is None else boxes
    shapes = [shape]
    for transform in chain:
        previous = shapes[-1]
        shapes.append(
            transform.dimensions
            if transform.kind == "reshape"
            else tuple(previous[i] for i in transform.dimensions)
            if transform.kind == "transpose"
            else tuple(i.size for i in transform.indices if isinstance(i, SliceIndex))
        )
    for pos in reversed(range(len(chain))):
        transform = chain[pos]
        if transform.kind == "reshape":
            boxes = tuple(
                b
                for box in boxes
                for span in box_spans(box, shapes[pos + 1], budget, path)
                for b in span_boxes(span, shapes[pos], budget, path)
            )
        elif transform.kind == "transpose":
            boxes = tuple(
                Box(tuple(box.axes[transform.dimensions.index(i)] for i in range(len(box.axes))))
                for box in boxes
            )
        else:
            expanded = []
            for box in boxes:
                kept = iter(box.axes)
                axes = []
                for item in transform.indices:
                    if isinstance(item, SliceIndex):
                        selected_axis = next(kept)
                        if item.stride == 1:
                            axes.append(
                                (
                                    Span(
                                        item.start + selected_axis.start,
                                        item.start + selected_axis.stop,
                                    ),
                                )
                            )
                        else:
                            budget.spend(path + ".strided_spans", selected_axis.size)
                            axes.append(
                                tuple(
                                    Span(
                                        item.start + p * item.stride,
                                        item.start + p * item.stride + 1,
                                    )
                                    for p in range(selected_axis.start, selected_axis.stop)
                                )
                            )
                    else:
                        assert type(item) is int  # Resolved and validated by view_coordinates.
                        axes.append((Span(item, item + 1),))
                budget.spend(path + ".indexed_boxes", prod(len(a) for a in axes))
                expanded.extend(Box(tuple(parts)) for parts in product(*axes))
            boxes = tuple(expanded)
    spans = merge_spans(tuple(s for box in boxes for s in box_spans(box, shape, budget, path)))
    to_storage = linear(tuple(backward[f"local:{i}"] for i in range(len(shape))), shape)
    to_view = linear(tuple(forward[f"local:{i}"] for i in range(len(output_shape))), output_shape)
    return RemoteView(
        output_shape,
        spans,
        substitute(
            to_storage,
            dict(
                zip(
                    (f"local:{i}" for i in range(len(output_shape))),
                    coordinates(axis("global"), output_shape),
                    strict=True,
                )
            ),
        ),
        substitute(
            to_view,
            dict(
                zip(
                    (f"local:{i}" for i in range(len(shape))),
                    coordinates(axis("global"), shape),
                    strict=True,
                )
            ),
        ),
    )


def remote_access_view(
    shape: tuple[int, ...],
    index: SliceIndex | TensorIndex | Operand | None,
    transforms: tuple[RefTransform, ...],
    read: Callable[[Operand], Expression],
    budget: AnalysisBudget,
    path: str,
    mask: Expression | None = None,
    operand_type: Callable[[Operand], ArrayType] | None = None,
) -> RemoteView:
    """Use one exact active footprint for protocol and value execution."""
    from .masked_access import mask_regions
    from .tensor_domain import coordinate_ranges
    from .value_operations import static_integer

    indices = index.indices if isinstance(index, TensorIndex) else (index,)

    def array_type(item: Operand) -> ArrayType:
        assert operand_type is not None
        return operand_type(item)

    vector = (
        bool(index.integer_shape)
        if isinstance(index, TensorIndex)
        else (isinstance(index, ValueRef | LiteralValue) and bool(array_type(index).shape))
    )
    if vector:
        if transforms or (isinstance(index, TensorIndex) and index.integer_shape != shape):
            raise UnsupportedJaxprError(path, "remote array indices require a full identity grid")
        for dimension, item in enumerate(indices):
            if not isinstance(item, ValueRef | LiteralValue):
                raise UnsupportedJaxprError(
                    path, "remote array indices require a full identity grid"
                )
            item_shape = array_type(item).shape
            offset = len(shape) - len(item_shape)
            if offset < 0 or any(n not in (1, shape[offset + i]) for i, n in enumerate(item_shape)):
                raise UnsupportedJaxprError(path, "remote array index broadcast shape differs")
            value = _substitute(
                localize(read(item), item_shape, budget),
                {
                    f"local:{i}": integer(0) if n == 1 else axis(f"local:{offset + i}")
                    for i, n in enumerate(item_shape)
                },
            )
            if value != axis(f"local:{dimension}"):
                raise UnsupportedJaxprError(
                    path, "remote vector index is not the identity coordinate"
                )
        index = None
    selected = remote_view(
        shape, index, transforms, lambda v: static_integer(read(v), path), budget, path
    )
    if mask is None:
        return selected
    localized = localize(mask, selected.shape, budget)
    regions = mask_regions(
        localized,
        (full_box(selected.shape),),
        coordinate_ranges("local", selected.shape),
        budget,
        path,
    )
    return replace(
        remote_view(
            shape,
            index,
            transforms,
            lambda v: static_integer(read(v), path),
            budget,
            path,
            boxes=tuple(region.box for region in regions),
        ),
        conditional=any(region.guard is not True for region in regions),
    )
