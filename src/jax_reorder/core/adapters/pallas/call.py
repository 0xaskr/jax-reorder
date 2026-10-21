"""Decode the locked Pallas call, grid, block and memory-operation schemas."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

import jax
from jax._src import config, core
from jax._src.pallas import core as pc
from jax._src.pallas import primitives as pp
from jax._src.pallas.mosaic import core as mc
from jax._src.state import indexing
from jax._src.state import primitives as sp
from jax._src.state import types as st

from ....errors import UnsupportedJaxprError
from ...model import (
    ArrayType,
    BlockMappingView,
    DynamicSliceIndex,
    GridAxis,
    LiteralValue,
    MemoryAccess,
    Operand,
    Origin,
    PallasCallView,
    PipelineModeView,
    RefType,
    ScratchView,
    SemaphoreType,
    SliceIndex,
    TensorIndex,
    ValueRef,
)
from .compiler import adapt_compiler
from .ref_transforms import adapt_ref_transforms

if TYPE_CHECKING:
    from ..jax import JaxAdapter

CALL_FIELDS = frozenset(
    {
        "out_avals",
        "jaxpr",
        "debug",
        "interpret",
        "grid_mapping",
        "mesh",
        "input_output_aliases",
        "compiler_params",
        "cost_estimate",
        "metadata",
        "name",
    }
)
GRID_FIELDS = frozenset(
    {
        "grid",
        "grid_names",
        "block_mappings",
        "index_map_tree",
        "index_map_avals",
        "vmapped_dims",
        "scratch_avals",
        "num_index_operands",
        "num_inputs",
        "num_outputs",
        "get_grid_indices",
        "local_grid_env",
        "debug",
    }
)
BLOCK_FIELDS = frozenset(
    {
        "block_shape",
        "transformed_block_aval",
        "index_map_jaxpr",
        "index_map_out_tree",
        "array_aval",
        "origin",
        "transforms",
        "pipeline_mode",
        "debug",
    }
)


def _require(condition: bool, path: str, reason: str) -> None:
    if not condition:
        raise UnsupportedJaxprError(path, reason)


def _schema(value: Any, cls: type, fields: frozenset[str], path: str) -> None:
    _require(type(value) is cls, path, "unaudited descriptor type")
    _require(
        {f.name for f in dataclasses.fields(value)} == fields, path, "unaudited descriptor schema"
    )


def _pipeline_mode(value: Any, path: str) -> PipelineModeView | None:
    if value is None:
        return None
    _schema(
        value,
        pc.Buffered,
        frozenset({"buffer_count", "use_lookahead", "revisit", "prefetched_count"}),
        path,
    )
    _require(
        type(value.buffer_count) is int
        and value.buffer_count == 1
        and type(value.use_lookahead) is bool
        and not value.use_lookahead
        and type(value.prefetched_count) is int
        and value.prefetched_count == 0
        and (value.revisit is None or type(value.revisit) is pc.RevisitMode),
        path,
        "only synchronous single-buffer pipelines without lookahead/prefetch are adapted",
    )
    return PipelineModeView(
        value.buffer_count,
        value.use_lookahead,
        value.revisit.value if value.revisit is not None else None,
        value.prefetched_count,
    )


def adapt_call(eq: Any, adapter: JaxAdapter, path: str) -> PallasCallView:
    p = eq.params
    _require(set(p) == CALL_FIELDS, path, "unaudited pallas_call parameter schema")
    _require(
        p["interpret"] is False,
        path + ".interpret",
        "only native interpret=False source graphs are supported",
    )
    for name in ("mesh", "metadata"):
        _require(p[name] is None, path + "." + name, f"{name} is not yet adapted")
    cost = p["cost_estimate"]
    if cost is not None:
        _schema(
            cost,
            pc.CostEstimate,
            frozenset({"flops", "transcendentals", "bytes_accessed", "remote_bytes_transferred"}),
            path + ".cost_estimate",
        )
        cost = (
            cost.flops,
            cost.transcendentals,
            cost.bytes_accessed,
            cost.remote_bytes_transferred,
        )
        _require(all(type(n) is int and n >= 0 for n in cost), path, "invalid cost estimate")
    aliases = p["input_output_aliases"]
    _require(
        type(aliases) is tuple
        and all(
            type(pair) is tuple and len(pair) == 2 and all(type(i) is int for i in pair)
            for pair in aliases
        ),
        path,
        "invalid input/output alias schema",
    )
    aliases = cast(tuple[tuple[int, int], ...], aliases)
    _require(
        type(p["debug"]) is bool and (p["name"] is None or type(p["name"]) is str),
        path,
        "invalid diagnostic parameters",
    )
    gm = p["grid_mapping"]
    _schema(gm, pc.GridMapping, GRID_FIELDS, path + ".grid_mapping")
    _require(
        all(type(n) is int and n > 0 for n in gm.grid),
        path,
        "expected positive static grid axes",
    )
    for name in ("grid_names", "get_grid_indices", "local_grid_env"):
        _require(
            getattr(gm, name) is None, path + "." + name, "custom grid context is not yet adapted"
        )
    for name in ("vmapped_dims",):
        _require(getattr(gm, name) == (), path + "." + name, f"{name} is not yet supported")
    prefetch_count = gm.num_index_operands
    _require(type(prefetch_count) is int and prefetch_count >= 0, path, "invalid prefetch arity")
    _require(type(gm.debug) is bool, path, "invalid grid debug flag")
    _require(
        gm.num_inputs + prefetch_count == len(eq.invars) and gm.num_outputs == len(eq.outvars),
        path,
        "inconsistent call arity",
    )
    _require(
        all(prefetch_count <= i < len(eq.invars) and 0 <= o < gm.num_outputs for i, o in aliases)
        and len({i for i, _ in aliases}) == len(aliases)
        and len({o for _, o in aliases}) == len(aliases),
        path,
        "invalid or conflicting input/output aliases",
    )
    _require(
        adapter.context.pallas_backend == "tpu",
        path,
        "only the TPU Pallas source policy is reviewed",
    )
    compiler, grid_semantics = adapt_compiler(
        p["compiler_params"], len(gm.grid), gm.num_inputs, path + ".compiler_params"
    )
    _require(
        len(gm.block_mappings) == gm.num_inputs + gm.num_outputs,
        path,
        "inconsistent block mapping arity",
    )
    rank = len(gm.grid)
    expected_tree = jax.tree.structure(((0,) * (rank + prefetch_count), {}))
    _require(
        type(gm.index_map_tree) is jax.tree_util.PyTreeDef
        and gm.index_map_tree == expected_tree
        and len(gm.index_map_avals) == rank + prefetch_count,
        path,
        "unexpected index map bindings",
    )
    index_avals = tuple(adapter.array_type(a, path) for a in gm.index_map_avals[:rank])
    _require(
        all(a.shape == () and a.dtype == "int32" for a in index_avals),
        path,
        "unexpected program coordinate type",
    )
    prefetch = tuple(adapter.value_type(a, path + ".prefetch") for a in gm.index_map_avals[rank:])
    _require(
        all(isinstance(t, RefType) and isinstance(t.inner, ArrayType) for t in prefetch),
        path,
        "prefetch operands must be array Refs",
    )
    blocks = []
    for i, block in enumerate(gm.block_mappings):
        loc = f"{path}.blocks[{i}]"
        _schema(block, pc.BlockMapping, BLOCK_FIELDS, loc)
        _require(
            bool(block.block_shape) and all(type(b) is pc.Blocked for b in block.block_shape),
            loc,
            "only static Blocked mappings are adapted",
        )
        _require(block.transforms == (), loc, "block transforms are not yet adapted")
        pipeline_mode = _pipeline_mode(block.pipeline_mode, loc + ".pipeline_mode")
        _require(
            type(block.debug) is bool and type(block.origin) is str,
            loc,
            "invalid block diagnostics",
        )
        _require(
            type(block.index_map_out_tree) is jax.tree_util.PyTreeDef
            and block.index_map_out_tree == jax.tree.structure((0,) * len(block.block_shape)),
            loc,
            "unexpected index map output bindings",
        )
        shape = tuple(b.block_size for b in block.block_shape)
        _require(
            all(type(n) is int and n > 0 for n in shape),
            loc,
            "expected positive static block sizes",
        )
        ref = adapter.value_type(block.transformed_block_aval, loc)
        _require(
            isinstance(ref, RefType) and ref.inner.shape == shape, loc, "block Ref shape mismatch"
        )
        array = adapter.array_type(block.array_aval, loc)
        _require(
            len(array.shape) == len(shape) and all(n > 0 for n in array.shape),
            loc,
            "expected a nonempty array matching the block rank",
        )
        _require(type(block.transformed_block_aval) is st.AbstractRef, loc, "unknown Ref type")
        _require(
            block.transformed_block_aval.memory_space
            in (pc.MemorySpace.DEFAULT, pc.MemorySpace.ANY)
            and block.transformed_block_aval.kind is None,
            loc,
            "non-default Ref storage is not yet adapted",
        )
        region = adapter.region(block.index_map_jaxpr, loc + ".index_map")
        _require(
            len(region.inputs) == rank + prefetch_count and len(region.outputs) == len(shape),
            loc,
            "effectful or non-scalar index map",
        )
        _require(
            tuple(v.type for v in region.inputs) == (*index_avals, *prefetch),
            loc,
            "index map input binding mismatch",
        )
        unbuffered = block.transformed_block_aval.memory_space == pc.MemorySpace.ANY
        _require(not unbuffered or shape == array.shape, loc, "ANY storage requires a full Ref")
        _require(
            not unbuffered or pipeline_mode is None,
            loc,
            "unbuffered ANY storage cannot carry an automatic pipeline policy",
        )
        blocks.append(
            BlockMappingView(
                shape, array, cast(RefType, ref), region, Origin(loc), unbuffered, pipeline_mode
            )
        )
    scratch = []
    for i, aval in enumerate(gm.scratch_avals):
        typ = adapter.value_type(aval, f"{path}.scratch[{i}]")
        _require(
            isinstance(typ, RefType),
            path,
            "scratch must be a Ref",
        )
        typ = cast(RefType, typ)
        semaphore = isinstance(typ.inner, SemaphoreType)
        _require(
            semaphore
            or (
                isinstance(typ.inner, ArrayType)
                and aval.memory_space is mc.MemorySpace.VMEM
                and aval.kind is None
            ),
            path,
            "only VMEM arrays and scalar semaphore scratch are adapted",
        )
        # The locked TPU scratch semaphore allocation starts at zero. Its
        # lifetime spans the call, not an independently allocated grid instance.
        scratch.append(ScratchView(typ, "call", 0 if semaphore else None))
    kernel = adapter.kernel_region(p["jaxpr"], path + ".kernel")
    _require(
        not kernel.constants and not kernel.outputs,
        path,
        "captured kernel constants/return values are not yet supported",
    )
    clear_variation = config._check_vma.value
    kernel_ref_types = tuple(
        adapter.value_type(
            b.transformed_block_aval.update_manual_axis_type(core.empty_mat)
            if clear_variation
            else b.transformed_block_aval,
            path + ".kernel_binding",
        )
        for b in gm.block_mappings
    )
    if any(isinstance(s.type.inner, ArrayType) for s in scratch):
        local_arrays = tuple(cast(ArrayType, cast(RefType, t).inner) for t in kernel_ref_types)
        _require(bool(local_arrays), path, "scratch needs a kernel array binding")
        local_sharding = local_arrays[0].sharding
        _require(
            all(
                t.sharding is not None
                and local_sharding is not None
                and t.sharding
                == dataclasses.replace(local_sharding, partitions=t.sharding.partitions)
                for t in local_arrays
            ),
            path,
            "scratch binding requires a common local abstract mesh",
        )
        bound_scratch = []
        for s, source_aval in zip(scratch, gm.scratch_avals, strict=True):
            if isinstance(s.type.inner, ArrayType):
                # MemorySpace.VMEM constructs a ShapedArray with an empty
                # default abstract mesh, including inside shard_map. Allocation
                # is kernel-local; that empty descriptor does not mean globally
                # replicated storage. Keep it and record the binding separately.
                default = adapter.array_type(
                    core.ShapedArray(
                        source_aval.shape, source_aval.dtype, weak_type=source_aval.weak_type
                    ),
                    path + ".scratch_default",
                )
                _require(
                    s.type.inner.sharding in (default.sharding, local_sharding),
                    path,
                    "scratch descriptor names a different abstract mesh",
                )
                assert local_sharding is not None
                s = dataclasses.replace(
                    s,
                    bound_array=dataclasses.replace(
                        s.type.inner,
                        sharding=dataclasses.replace(
                            local_sharding, partitions=((),) * len(s.type.inner.shape)
                        ),
                    ),
                )
            bound_scratch.append(s)
        scratch = bound_scratch
    _require(
        tuple(v.type for v in kernel.inputs)
        == prefetch + kernel_ref_types + tuple(s.type for s in scratch),
        path,
        "kernel Ref bindings differ from blocks",
    )
    call_types = tuple(adapter.array_type(v.aval, path) for v in (*eq.invars, *eq.outvars))
    _require(
        call_types[prefetch_count:] == tuple(b.array for b in blocks),
        path,
        "call array bindings differ from blocks",
    )
    _require(
        all(
            (t.shape, t.dtype)
            == (
                cast(ArrayType, cast(RefType, r).inner).shape,
                cast(ArrayType, cast(RefType, r).inner).dtype,
            )
            for t, r in zip(call_types[:prefetch_count], prefetch, strict=True)
        ),
        path,
        "prefetch array binding differs from Ref",
    )
    _require(
        all(call_types[i] == call_types[len(eq.invars) + o] for i, o in aliases),
        path,
        "aliased arrays have different abstract contracts",
    )
    _require(
        tuple(adapter.array_type(a, path) for a in p["out_avals"]) == call_types[len(eq.invars) :],
        path,
        "out_avals mismatch",
    )
    return PallasCallView(
        tuple(GridAxis(n, s) for n, s in zip(gm.grid, grid_semantics, strict=True)),
        tuple(blocks),
        kernel,
        gm.num_inputs,
        gm.num_outputs,
        False,
        aliases=aliases,
        compiler=compiler,
        kernel_argument_policy="clear_variation" if clear_variation else "identity",
        scratch=tuple(scratch),
        prefetch=cast(tuple[RefType, ...], prefetch),
        cost_estimate=cost,
    )


def adapt_memory_access(eq: Any, operands: tuple[Operand, ...], path: str) -> MemoryAccess:
    p = eq.params
    tree = p["tree"] if eq.primitive in (sp.get_p, sp.swap_p) else p["args_tree"]
    _require(type(tree) is jax.tree_util.PyTreeDef, path, "unknown access tree descriptor")
    if eq.primitive in (sp.get_p, sp.swap_p):
        writing = eq.primitive is sp.swap_p
        ref, value = operands[0], operands[1] if writing else None
        transforms = p["tree"].unflatten(operands[2 if writing else 1 :])
        mask, other = None, None
    else:
        _require(
            p.get("cache_modifier") is None
            and p.get("eviction_policy") is None
            and not p.get("is_volatile", False),
            path,
            "non-default masked access attributes are not yet supported",
        )
        unpacked = p["args_tree"].unflatten(operands)
        if eq.primitive is pp.load_p:
            ref, transforms, mask, other = unpacked
            value = None
        else:
            ref, transforms, value, mask = unpacked
            other = None
    _require(isinstance(ref, ValueRef), path, "memory access must bind a Ref")
    if any(
        type(transform) in (st.ReshapeTransform, st.TransposeTransform) for transform in transforms
    ):
        storage_shape = eq.invars[0].aval.inner_aval.shape
        views = adapt_ref_transforms(transforms, storage_shape, path)
        indices = tuple(SliceIndex(0, n, 1) for n in storage_shape)
        index = indices[0] if len(storage_shape) == 1 else TensorIndex(indices, storage_shape)
        return MemoryAccess(
            cast(ValueRef, ref),
            index,
            value,
            mask,
            other,
            scalar=storage_shape == (),
            transforms=tuple(views),
        )
    if not transforms:
        shape = eq.invars[0].aval.inner_aval.shape
        if not shape:
            return MemoryAccess(
                cast(ValueRef, ref), SliceIndex(0, 1, 1), value, mask, other, scalar=True
            )
        indices = tuple(SliceIndex(0, n, 1) for n in shape)
        index = indices[0] if len(shape) == 1 else TensorIndex(indices, shape)
        return MemoryAccess(cast(ValueRef, ref), index, value, mask, other)
    _require(
        all(type(transform) is indexing.NDIndexer for transform in transforms),
        path,
        "unsupported Ref transform chain",
    )
    ix: Any = transforms[0]
    for following in transforms[1:]:
        _require(
            ix.int_indexer_shape == following.int_indexer_shape == ()
            and all(
                type(index) is indexing.Slice
                and all(type(v) is int for v in (index.start, index.size, index.stride))
                for index in (*ix.indices, *following.indices)
            ),
            path,
            "Ref index composition requires static slices",
        )
        _require(
            following.shape == tuple(index.size for index in ix.indices),
            path,
            "Ref transform intermediate shape differs from its preceding view",
        )
        _require(
            all(
                before.stride > 0
                and before.size > 0
                and 0 <= before.start
                and before.start + (before.size - 1) * before.stride < extent
                for before, extent in zip(ix.indices, ix.shape, strict=True)
            ),
            path,
            "Ref transform slice is outside its original storage",
        )
        _require(
            all(
                after.stride > 0
                and after.size > 0
                and 0 <= after.start
                and after.start + (after.size - 1) * after.stride < before.size
                for before, after in zip(ix.indices, following.indices, strict=True)
            ),
            path,
            "Ref transform slice is outside its intermediate view",
        )
        ix = dataclasses.replace(
            ix,
            indices=tuple(
                indexing.Slice(
                    before.start + after.start * before.stride,
                    after.size,
                    before.stride * after.stride,
                )
                for before, after in zip(ix.indices, following.indices, strict=True)
            ),
        )
    if ix.shape == () and ix.indices == () and ix.int_indexer_shape == ():
        _require(
            eq.invars[0].aval.inner_aval.shape == (), path, "scalar indexer/Ref shape mismatch"
        )
        return MemoryAccess(
            cast(ValueRef, ref), SliceIndex(0, 1, 1), value, mask, other, scalar=True
        )
    _require(len(ix.indices) == len(ix.shape), path, "indexer rank differs from Ref rank")
    _require(
        ix.shape == eq.invars[0].aval.inner_aval.shape, path, "indexer shape differs from Ref shape"
    )
    indices = []
    for index in ix.indices:
        if type(index) is indexing.Slice:
            _require(
                all(type(v) is int for v in (index.size, index.stride))
                and (type(index.start) is int or isinstance(index.start, ValueRef | LiteralValue)),
                path,
                "slice size and stride must be static",
            )
            indices.append(
                SliceIndex(index.start, cast(int, index.size), index.stride)
                if type(index.start) is int
                else DynamicSliceIndex(
                    cast(Operand, index.start), cast(int, index.size), index.stride
                )
            )
        else:
            _require(isinstance(index, ValueRef | LiteralValue), path, "unsupported index operand")
            indices.append(cast(Operand, index))
    if len(ix.shape) == 1 and not isinstance(indices[0], DynamicSliceIndex):
        _require(
            ix.int_indexer_shape in ((), ix.shape),
            path,
            "partial/broadcast integer index is not yet adapted",
        )
        return MemoryAccess(cast(ValueRef, ref), indices[0], value, mask, other)
    _require(
        all(type(n) is int for n in ix.int_indexer_shape), path, "dynamic integer broadcast shape"
    )
    return MemoryAccess(
        cast(ValueRef, ref),
        TensorIndex(tuple(indices), ix.shape, ix.int_indexer_shape),
        value,
        mask,
        other,
    )
