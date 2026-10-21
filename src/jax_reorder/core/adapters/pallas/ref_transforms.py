"""Decode scalar/slice Ref view chains without changing the owning storage."""

from __future__ import annotations

from dataclasses import fields
from math import prod
from typing import Any, cast

from jax._src.state import indexing
from jax._src.state import types as st

from ...index_domain import literal_integer
from ...model import DynamicSliceIndex, LiteralValue, Operand, RefTransform, SliceIndex, ValueRef
from ..attributes import require


def adapt_ref_transforms(
    transforms: tuple[Any, ...], storage_shape: tuple[int, ...], path: str
) -> tuple[RefTransform, ...]:
    current_shape = storage_shape
    views = []
    for transform in transforms:
        if type(transform) is indexing.NDIndexer:
            require(
                transform.shape == current_shape
                and transform.int_indexer_shape == ()
                and len(transform.indices) == len(current_shape),
                path,
                "reshape/transpose Ref chains require scalar/slice indices",
            )
            selected: list[SliceIndex | DynamicSliceIndex | int | Operand] = []
            selected_shape = []
            for index, extent in zip(transform.indices, current_shape, strict=True):
                if type(index) is indexing.Slice:
                    require(
                        type(index.size) is int
                        and type(index.stride) is int
                        and index.size > 0
                        and index.stride > 0,
                        path,
                        "Ref slice requires static positive size/stride",
                    )
                    size, stride = cast(int, index.size), index.stride
                    if type(index.start) is int:
                        require(
                            0 <= index.start and index.start + (size - 1) * stride < extent,
                            path,
                            "Ref slice is outside its intermediate view",
                        )
                        selected.append(SliceIndex(index.start, size, stride))
                    else:
                        require(
                            isinstance(index.start, ValueRef | LiteralValue),
                            path,
                            "unsupported dynamic Ref slice start",
                        )
                        selected.append(DynamicSliceIndex(cast(Operand, index.start), size, stride))
                    selected_shape.append(size)
                else:
                    require(
                        isinstance(index, ValueRef | LiteralValue), path, "unsupported Ref index"
                    )
                    if isinstance(index, LiteralValue):
                        number = int(literal_integer(index, path).value)
                        require(0 <= number < extent, path, "Ref scalar index is out of bounds")
                        selected.append(number)
                    else:
                        selected.append(cast(ValueRef, index))
            identity = tuple(SliceIndex(0, n, 1) for n in current_shape)
            if tuple(selected) != identity:
                views.append(RefTransform("index", (), tuple(selected)))
            current_shape = tuple(selected_shape)
        elif type(transform) is st.ReshapeTransform:
            shape = transform.shape
            require(
                {f.name for f in fields(transform)} == {"shape"}
                and type(shape) is tuple
                and all(type(n) is int and n > 0 for n in shape)
                and prod(shape) == prod(current_shape),
                path,
                "Ref reshape must preserve its static element count",
            )
            views.append(RefTransform("reshape", shape))
            current_shape = shape
        elif type(transform) is st.TransposeTransform:
            permutation = transform.permutation
            require(
                {f.name for f in fields(transform)} == {"permutation"}
                and type(permutation) is tuple
                and all(type(n) is int for n in permutation)
                and sorted(permutation) == list(range(len(current_shape))),
                path,
                "Ref transpose requires a static axis permutation",
            )
            views.append(RefTransform("transpose", permutation))
            current_shape = tuple(current_shape[i] for i in permutation)
        else:
            require(False, path, "unsupported Ref transform chain")
    return tuple(views)
