"""Locked local DMA source schemas; event pairing belongs to analysis."""

from __future__ import annotations

from typing import Any, cast

import jax
from jax._src.pallas import primitives as pp
from jax._src.pallas.mosaic import core as mc
from jax._src.pallas.mosaic import primitives as mp
from jax._src.state import indexing
from jax._src.state import types as st

from ...model import (
    DmaView,
    DynamicSliceIndex,
    Effect,
    EffectView,
    LiteralValue,
    Operand,
    RefTransform,
    SliceIndex,
    TensorIndex,
    ValueRef,
)
from ..attributes import require
from .ref_transforms import adapt_ref_transforms

DMA_SCHEMAS = {
    mp.dma_start_p: frozenset({"tree", "device_id_type", "priority", "add"}),
    mp.dma_wait_p: frozenset({"tree", "device_id_type", "insert_dummy_device", "is_wait_send"}),
}


def _tree_schema(tree: Any, path: str) -> None:
    require(type(tree) is jax.tree_util.PyTreeDef, path, "unknown DMA tree descriptor")
    node = tree.node_data()
    if node is not None:
        kind, metadata = node
        require(
            kind
            in (
                tuple,
                type(None),
                st.TransformedRef,
                indexing.NDIndexer,
                indexing.Slice,
                st.ReshapeTransform,
                st.TransposeTransform,
            ),
            path,
            "unknown DMA transform tree node",
        )
        if kind is st.TransformedRef:
            require(metadata == (), path, "unknown transformed Ref metadata")
        elif kind in (st.ReshapeTransform, st.TransposeTransform):
            require(
                type(metadata) is tuple
                and len(metadata) == 1
                and type(metadata[0]) is tuple
                and all(type(n) is int for n in metadata[0]),
                path,
                "invalid DMA view metadata",
            )
        elif kind is indexing.NDIndexer:
            require(
                type(metadata) is tuple and len(metadata) == 3, path, "unknown indexer metadata"
            )
            _tree_schema(metadata[0], path)
        elif kind is indexing.Slice:
            require(
                type(metadata) is tuple
                and len(metadata) == 3
                and all(n is None or type(n) is int for n in metadata),
                path,
                "dynamic DMA slice",
            )
    for child in tree.children():
        _tree_schema(child, path)


def _ref(
    value: Any, shapes: dict[int, tuple[int, ...]], path: str
) -> tuple[
    ValueRef,
    SliceIndex | TensorIndex | None,
    tuple[tuple[int, ...], ...],
    tuple[RefTransform, ...],
]:
    if isinstance(value, ValueRef):
        return value, None, (), ()
    require(type(value) is st.TransformedRef, path, "unknown DMA Ref transform")
    require(
        isinstance(value.ref, ValueRef) and bool(value.transforms),
        path,
        "DMA requires an indexed Ref",
    )
    if any(type(t) in (st.ReshapeTransform, st.TransposeTransform) for t in value.transforms):
        views = adapt_ref_transforms(value.transforms, shapes[value.ref.id], path)
        return value.ref, None, (), views
    ix = value.transforms[0]
    require(
        type(ix) is indexing.NDIndexer
        and bool(ix.indices)
        and len(ix.indices) == len(ix.shape)
        and ix.int_indexer_shape == ()
        and ix.shape == shapes[value.ref.id],
        path,
        "unsupported DMA indexer shape",
    )
    indices: list[SliceIndex | DynamicSliceIndex | Operand] = []
    for index in ix.indices:
        if type(index) is indexing.Slice:
            require(
                type(index.size) is int
                and type(index.stride) is int
                and (type(index.start) is int or isinstance(index.start, ValueRef | LiteralValue)),
                path,
                "DMA slice size/stride must be static",
            )
            indices.append(
                SliceIndex(index.start, cast(int, index.size), index.stride)
                if type(index.start) is int
                else DynamicSliceIndex(
                    cast(Operand, index.start), cast(int, index.size), index.stride
                )
            )
        else:
            require(isinstance(index, ValueRef | LiteralValue), path, "unsupported DMA index")
            indices.append(cast(Operand, index))
    result_shape = tuple(i.size for i in indices if isinstance(i, SliceIndex | DynamicSliceIndex))
    identity_views = []
    for transform in value.transforms[1:]:
        identity = (
            type(transform) is indexing.NDIndexer
            and transform.shape == result_shape
            and transform.int_indexer_shape == ()
            and len(transform.indices) == len(result_shape)
            and all(
                type(i) is indexing.Slice and (i.start, i.size, i.stride) == (0, n, 1)
                for i, n in zip(transform.indices, result_shape, strict=True)
            )
        )
        if not identity:
            return (
                value.ref,
                None,
                (),
                adapt_ref_transforms(value.transforms, shapes[value.ref.id], path),
            )
        identity_views.append(result_shape)
    if len(indices) == 1 and isinstance(indices[0], SliceIndex):
        return value.ref, indices[0], tuple(identity_views), ()
    return value.ref, TensorIndex(tuple(indices), ix.shape), tuple(identity_views), ()


def adapt_dma(
    eq: Any, operands: tuple[Operand, ...], effects: tuple[Effect, ...], path: str
) -> DmaView:
    p = eq.params
    require(set(p) == DMA_SCHEMAS[eq.primitive], path, "unaudited DMA parameter schema")
    _tree_schema(p["tree"], path)
    require(
        p["tree"].num_leaves == len(operands) and len(operands) >= 3,
        path,
        "invalid DMA tree operand count",
    )
    require(p["device_id_type"] is pp.DeviceIdType.MESH, path, "unsupported DMA device policy")
    require(not eq.outvars, path, "DMA unexpectedly returns values")
    unpacked = p["tree"].unflatten(operands)
    require(
        type(unpacked) is tuple and len(unpacked) == 5,
        path,
        "invalid DMA tree interface",
    )
    unpacked = cast(tuple[Any, ...], unpacked)
    remote = unpacked[3] is not None or unpacked[4] is not None
    source_sem, coordinates = None, None
    if remote:
        require(
            isinstance(unpacked[3], ValueRef | st.TransformedRef)
            and type(unpacked[4]) is tuple
            and bool(unpacked[4])
            and all(isinstance(c, (ValueRef, LiteralValue)) for c in unpacked[4]),
            path,
            "remote DMA requires a send semaphore and mesh coordinate tuple",
        )
        coordinates = cast(tuple[Operand, ...], unpacked[4])
        types = dict(zip(operands, (v.aval for v in eq.invars), strict=True))
        require(
            all(types[c].shape == () and str(types[c].dtype) == "int32" for c in coordinates),
            path,
            "remote mesh coordinates must be scalar int32",
        )
    ref_types = {
        cast(ValueRef, v).id: raw.aval
        for v, raw in zip(operands, eq.invars, strict=True)
        if isinstance(v, ValueRef) and type(raw.aval) is st.AbstractRef
    }
    shapes = {key: tuple(aval.shape) for key, aval in ref_types.items()}
    source, source_slice, source_views, source_transforms = _ref(unpacked[0], shapes, path)
    destination, destination_slice, destination_views, destination_transforms = _ref(
        unpacked[1], shapes, path
    )
    sem, sem_index, sem_views, sem_transforms = _ref(unpacked[2], shapes, path)
    source_sem_index = None
    if remote:
        source_sem, source_sem_index, source_sem_views, source_sem_transforms = _ref(
            unpacked[3], shapes, path
        )
        require(
            not source_sem_views and not source_sem_transforms,
            path,
            "additional send semaphore transforms are unsupported",
        )
        require(
            source_sem_index is None
            or isinstance(source_sem_index, TensorIndex)
            and all(isinstance(i, ValueRef | LiteralValue) for i in source_sem_index.indices),
            path,
            "send semaphore index must select a scalar element",
        )
        assert source_sem_index is None or isinstance(source_sem_index, TensorIndex)
    require(
        not sem_views and not sem_transforms,
        path,
        "additional semaphore transforms are not supported",
    )
    require(
        sem_index is None
        or isinstance(sem_index, TensorIndex)
        and all(isinstance(i, ValueRef | LiteralValue) for i in sem_index.indices),
        path,
        "DMA semaphore index must select a scalar element",
    )
    assert sem_index is None or isinstance(sem_index, TensorIndex)
    require(
        sem_index is not None or shapes[sem.id] == (), path, "DMA needs a scalar semaphore slot"
    )
    require(
        source_sem is None or source_sem_index is not None or shapes[source_sem.id] == (),
        path,
        "remote send semaphore must be scalar",
    )
    require(
        all(
            type(ref_types[s.id].dtype) is mc.DMASemaphore
            and ref_types[s.id].memory_space is mc.MemorySpace.SEMAPHORE
            for s in (sem, source_sem)
            if s is not None
        ),
        path,
        "DMA requires DMA semaphore storage",
    )
    wait_send = eq.primitive is mp.dma_wait_p and p["is_wait_send"] is True
    expected: set[Effect] = {
        EffectView("write" if wait_send else "read", source),
        EffectView("read" if wait_send else "write", destination),
        EffectView("write", sem),
    }
    if source_sem is not None:
        expected.add(EffectView("write", source_sem))
    require(set(effects) == expected, path, "inconsistent DMA Ref effects")
    if eq.primitive is mp.dma_start_p:
        require(
            type(p["priority"]) is int and p["priority"] >= 0 and type(p["add"]) is bool,
            path,
            "invalid DMA start parameters",
        )
        return DmaView(
            "start",
            source,
            destination,
            sem,
            "mesh",
            p["priority"],
            p["add"],
            source_slice=source_slice if isinstance(source_slice, SliceIndex) else None,
            destination_slice=destination_slice
            if isinstance(destination_slice, SliceIndex)
            else None,
            source_tensor_slice=source_slice if isinstance(source_slice, TensorIndex) else None,
            destination_tensor_slice=destination_slice
            if isinstance(destination_slice, TensorIndex)
            else None,
            source_semaphore=source_sem,
            device_coordinates=coordinates,
            semaphore_index=sem_index,
            source_semaphore_index=source_sem_index,
            source_identity_views=source_views,
            destination_identity_views=destination_views,
            source_transforms=source_transforms,
            destination_transforms=destination_transforms,
        )
    require(
        type(p["is_wait_send"]) is bool
        and type(p["insert_dummy_device"]) is bool
        and p["insert_dummy_device"] == wait_send
        and (remote or not wait_send),
        path,
        "inconsistent DMA send/receive wait policy",
    )
    return DmaView(
        "wait",
        source,
        destination,
        sem,
        "mesh",
        insert_dummy_device=p["insert_dummy_device"],
        is_wait_send=p["is_wait_send"],
        source_slice=source_slice if isinstance(source_slice, SliceIndex) else None,
        destination_slice=destination_slice if isinstance(destination_slice, SliceIndex) else None,
        source_tensor_slice=source_slice if isinstance(source_slice, TensorIndex) else None,
        destination_tensor_slice=destination_slice
        if isinstance(destination_slice, TensorIndex)
        else None,
        source_semaphore=source_sem,
        device_coordinates=coordinates,
        semaphore_index=sem_index,
        source_semaphore_index=source_sem_index,
        source_identity_views=source_views,
        destination_identity_views=destination_views,
        source_transforms=source_transforms,
        destination_transforms=destination_transforms,
    )
