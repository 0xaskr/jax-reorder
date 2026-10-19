"""Locked local DMA source schemas; event pairing belongs to analysis."""

from __future__ import annotations

from typing import Any, cast

import jax
from jax._src.pallas import primitives as pp
from jax._src.pallas.mosaic import primitives as mp
from jax._src.state import indexing
from jax._src.state import types as st

from ...model import DmaView, Effect, EffectView, LiteralValue, Operand, SliceIndex, ValueRef
from ..attributes import require

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
            kind in (tuple, type(None), st.TransformedRef, indexing.NDIndexer, indexing.Slice),
            path,
            "unknown DMA transform tree node",
        )
        if kind is st.TransformedRef:
            require(metadata == (), path, "unknown transformed Ref metadata")
        elif kind is indexing.NDIndexer:
            require(
                type(metadata) is tuple and len(metadata) == 3, path, "unknown indexer metadata"
            )
            _tree_schema(metadata[0], path)
        elif kind is indexing.Slice:
            require(
                type(metadata) is tuple
                and len(metadata) == 3
                and all(type(n) is int for n in metadata),
                path,
                "dynamic DMA slice",
            )
    for child in tree.children():
        _tree_schema(child, path)


def _ref(
    value: Any, shapes: dict[int, tuple[int, ...]], path: str
) -> tuple[ValueRef, SliceIndex | None]:
    if isinstance(value, ValueRef):
        return value, None
    require(type(value) is st.TransformedRef, path, "unknown DMA Ref transform")
    require(
        isinstance(value.ref, ValueRef) and len(value.transforms) == 1,
        path,
        "only a single static DMA slice is adapted",
    )
    ix = value.transforms[0]
    require(
        type(ix) is indexing.NDIndexer
        and len(ix.indices) == 1
        and ix.int_indexer_shape == ()
        and ix.shape == shapes[value.ref.id],
        path,
        "unsupported DMA indexer shape",
    )
    index = ix.indices[0]
    require(
        type(index) is indexing.Slice
        and all(type(n) is int for n in (index.start, index.size, index.stride)),
        path,
        "only static DMA slices are adapted",
    )
    index = cast(indexing.Slice, index)
    return value.ref, SliceIndex(cast(int, index.start), cast(int, index.size), index.stride)


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
            isinstance(unpacked[3], ValueRef)
            and type(unpacked[4]) is tuple
            and bool(unpacked[4])
            and all(isinstance(c, (ValueRef, LiteralValue)) for c in unpacked[4]),
            path,
            "remote DMA requires a scalar send semaphore and mesh coordinate tuple",
        )
        source_sem = cast(ValueRef, unpacked[3])
        coordinates = cast(tuple[Operand, ...], unpacked[4])
        types = dict(zip(operands, (v.aval for v in eq.invars), strict=True))
        require(
            all(types[c].shape == () and str(types[c].dtype) == "int32" for c in coordinates),
            path,
            "remote mesh coordinates must be scalar int32",
        )
    shapes = {
        cast(ValueRef, v).id: tuple(raw.aval.shape)
        for v, raw in zip(operands, eq.invars, strict=True)
        if isinstance(v, ValueRef) and type(raw.aval) is st.AbstractRef
    }
    source, source_slice = _ref(unpacked[0], shapes, path)
    destination, destination_slice = _ref(unpacked[1], shapes, path)
    require(isinstance(unpacked[2], ValueRef), path, "DMA semaphore must be an unsliced scalar Ref")
    sem = cast(ValueRef, unpacked[2])
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
            source_slice=source_slice,
            destination_slice=destination_slice,
            source_semaphore=source_sem,
            device_coordinates=coordinates,
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
        source_slice=source_slice,
        destination_slice=destination_slice,
        source_semaphore=source_sem,
        device_coordinates=coordinates,
    )
