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
    GridAxis,
    MemoryAccess,
    Operand,
    Origin,
    PallasCallView,
    RefType,
    ScratchView,
    SemaphoreType,
    SliceIndex,
    ValueRef,
)
from .compiler import adapt_compiler

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


def adapt_call(eq: Any, adapter: JaxAdapter, path: str) -> PallasCallView:
    p = eq.params
    _require(set(p) == CALL_FIELDS, path, "unaudited pallas_call parameter schema")
    _require(
        p["interpret"] is False,
        path + ".interpret",
        "only native interpret=False source graphs are supported",
    )
    for name in ("mesh", "cost_estimate", "metadata"):
        _require(p[name] is None, path + "." + name, f"{name} is not yet adapted")
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
        len(gm.grid) == 1 and all(type(n) is int and n > 0 for n in gm.grid),
        path,
        "expected one positive static grid axis",
    )
    for name in ("grid_names", "get_grid_indices", "local_grid_env"):
        _require(
            getattr(gm, name) is None, path + "." + name, "custom grid context is not yet adapted"
        )
    for name in ("vmapped_dims",):
        _require(getattr(gm, name) == (), path + "." + name, f"{name} is not yet supported")
    _require(
        gm.num_index_operands == 0,
        path,
        "scalar prefetch/dynamic index operands are not yet supported",
    )
    _require(type(gm.debug) is bool, path, "invalid grid debug flag")
    _require(
        gm.num_inputs == len(eq.invars) and gm.num_outputs == len(eq.outvars),
        path,
        "inconsistent call arity",
    )
    _require(
        all(0 <= i < gm.num_inputs and 0 <= o < gm.num_outputs for i, o in aliases)
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
    expected_tree = jax.tree.structure(((0,), {}))
    _require(
        type(gm.index_map_tree) is jax.tree_util.PyTreeDef
        and gm.index_map_tree == expected_tree
        and len(gm.index_map_avals) == 1,
        path,
        "unexpected index map bindings",
    )
    index_aval = adapter.array_type(gm.index_map_avals[0], path)
    _require(
        index_aval.shape == () and index_aval.dtype == "int32",
        path,
        "unexpected program coordinate type",
    )
    blocks = []
    for i, block in enumerate(gm.block_mappings):
        loc = f"{path}.blocks[{i}]"
        _schema(block, pc.BlockMapping, BLOCK_FIELDS, loc)
        _require(
            len(block.block_shape) == 1 and type(block.block_shape[0]) is pc.Blocked,
            loc,
            "only one-dimensional Blocked mappings are adapted",
        )
        _require(
            block.transforms == () and block.pipeline_mode is None,
            loc,
            "block transforms/pipeline are not yet adapted",
        )
        _require(
            type(block.debug) is bool and type(block.origin) is str,
            loc,
            "invalid block diagnostics",
        )
        _require(
            type(block.index_map_out_tree) is jax.tree_util.PyTreeDef
            and block.index_map_out_tree == jax.tree.structure((0,)),
            loc,
            "unexpected index map output bindings",
        )
        size = block.block_shape[0].block_size
        _require(type(size) is int and size > 0, loc, "expected a positive static block size")
        ref = adapter.value_type(block.transformed_block_aval, loc)
        _require(
            isinstance(ref, RefType) and ref.inner.shape == (size,), loc, "block Ref shape mismatch"
        )
        array = adapter.array_type(block.array_aval, loc)
        _require(
            len(array.shape) == 1 and array.shape[0] > 0, loc, "expected a nonempty rank-one array"
        )
        _require(type(block.transformed_block_aval) is st.AbstractRef, loc, "unknown Ref type")
        _require(
            block.transformed_block_aval.memory_space == pc.MemorySpace.DEFAULT
            and block.transformed_block_aval.kind is None,
            loc,
            "non-default Ref storage is not yet adapted",
        )
        region = adapter.region(block.index_map_jaxpr, loc + ".index_map")
        _require(
            not region.effects and len(region.inputs) == 1 and len(region.outputs) == 1,
            loc,
            "effectful or non-scalar index map",
        )
        _require(region.inputs[0].type == index_aval, loc, "index map input binding mismatch")
        blocks.append(BlockMappingView((size,), array, cast(RefType, ref), region, Origin(loc)))
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
            "only VMEM arrays and DMA semaphore scratch are adapted",
        )
        # The locked TPU scratch semaphore allocation starts at zero. Its
        # lifetime spans the call, not an independently allocated grid instance.
        scratch.append(ScratchView(typ, "call", 0 if semaphore else None))
    kernel = adapter.region(p["jaxpr"], path + ".kernel")
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
            all(t.sharding == local_sharding for t in local_arrays),
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
                s = dataclasses.replace(
                    s, bound_array=dataclasses.replace(s.type.inner, sharding=local_sharding)
                )
            bound_scratch.append(s)
        scratch = bound_scratch
    _require(
        tuple(v.type for v in kernel.inputs) == kernel_ref_types + tuple(s.type for s in scratch),
        path,
        "kernel Ref bindings differ from blocks",
    )
    call_types = tuple(adapter.array_type(v.aval, path) for v in (*eq.invars, *eq.outvars))
    _require(
        call_types == tuple(b.array for b in blocks), path, "call array bindings differ from blocks"
    )
    _require(
        all(call_types[i] == call_types[gm.num_inputs + o] for i, o in aliases),
        path,
        "aliased arrays have different abstract contracts",
    )
    _require(
        tuple(adapter.array_type(a, path) for a in p["out_avals"]) == call_types[gm.num_inputs :],
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
    _require(
        len(transforms) == 1 and type(transforms[0]) is indexing.NDIndexer,
        path,
        "unsupported Ref transform chain",
    )
    ix = transforms[0]
    _require(len(ix.indices) == 1 and len(ix.shape) == 1, path, "expected rank-one indexing")
    _require(
        ix.shape == eq.invars[0].aval.inner_aval.shape, path, "indexer shape differs from Ref shape"
    )
    index = ix.indices[0]
    if type(index) is indexing.Slice:
        _require(ix.int_indexer_shape == (), path, "unexpected integer-index broadcast shape")
        _require(
            all(type(v) is int for v in (index.start, index.size, index.stride)),
            path,
            "dynamic slices are not yet adapted",
        )
        index = SliceIndex(cast(int, index.start), cast(int, index.size), index.stride)
    else:
        _require(isinstance(index, (ValueRef,)), path, "unsupported index operand")
        # The current fragment has one rank-one integer array index. A different
        # broadcast shape would change the element correspondence.
        _require(
            ix.int_indexer_shape == ix.shape,
            path,
            "partial/broadcast integer index is not yet adapted",
        )
    return MemoryAccess(cast(ValueRef, ref), index, value, mask, other)
