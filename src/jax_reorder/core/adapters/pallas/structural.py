"""Locked TPU source facts for the common structural graph consumer.

This adapter preserves operations and resources. It neither pairs DMA events nor
interprets semaphore counters, storage contents, or program schedules.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

import jax
from jax._src import api_util, checkify, core, debugging, effects, hijax
from jax._src import flattree as ft
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import _jax
from jax._src.pallas import core as pc
from jax._src.pallas import primitives as pp
from jax._src.pallas.einshape import Einshape
from jax._src.pallas.mosaic import core as mc
from jax._src.pallas.mosaic import lowering
from jax._src.pallas.mosaic import primitives as mp
from jax._src.pallas.mosaic.pipeline import emit_pipeline_p, pipeline_body_p
from jax._src.pallas.pallas_call import pallas_call_p
from jax._src.random import prng
from jax._src.state import primitives as sp
from jax._src.state import types as st
from jax._src.state.discharge import run_state_p

# Import the official frontend before taking registry snapshots: it registers
# pipeline/einshape primitives and Pallas PRNG implementations lazily.
from jax.experimental.pallas import tpu as _tpu  # noqa: F401

from ....errors import UnsupportedJaxprError

run_scoped_p = pp.run_scoped_p


class _Encoder(Protocol):
    def encode(self, value: Any, path: str) -> Any: ...

    def fields(self, obj: Any, names: tuple[str, ...], path: str) -> Any: ...


# Registry identities are captured once from the pinned backend, not accepted by
# primitive name. A caller registering a same-named primitive cannot enter it.
TPU_PRIMITIVES = frozenset(
    {pallas_call_p} | {p for rules in lowering.lowering_rules.values() for p in rules}
) - {debugging.debug_callback_p}

REF_EFFECTS = (st.ReadEffect, st.WriteEffect, st.AccumEffect)
STATE_EFFECTS = (
    pc.CommsEffect,
    pp.SemaphoreEffect,
    pp.DelayEffect,
    mp.PRNGEffect,
    mp.TraceEffect,
    mp.MXUEffect,
    core.InternalMutableArrayEffect,
    debugging.DebugEffect,
    debugging.OrderedDebugEffect,
)
ERROR_TYPES = (
    checkify.FailedCheckError,
    checkify.NaNError,
    checkify.DivisionByZeroError,
    checkify.OOBError,
)
PRNG_IMPLEMENTATIONS = frozenset(prng.prngs.values())

CALL_SCHEMA = frozenset(
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

# These descriptors contain nested index-map jaxprs. Diagnostic names do not
# identify values or resources; all execution fields remain in the graph.
GRID_FIELDS = (
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
)
BLOCK_FIELDS = (
    "block_shape",
    "transformed_block_aval",
    "index_map_jaxpr",
    "index_map_out_tree",
    "array_aval",
    "transforms",
    "pipeline_mode",
)

PARAMETER_SCHEMAS = {
    run_state_p: {"jaxpr", "which_linear", "is_initialized"},
    hijax.call_hi_primitive_p: {"_prim"},
    emit_pipeline_p: {
        "grid_mapping",
        "body_jaxpr",
        "tiling",
        "core_axis",
        "core_axis_name",
        "args_tree",
        "refs_tree",
        "dimension_semantics",
        "trace_scopes",
        "no_pipelining",
        "_explicit_indices",
        "num_cores",
    },
    pipeline_body_p: {"jaxpr", "in_tree", "num_inputs", "_explicit_indices"},
    pp.program_id_p: {"axis"},
    pp.num_programs_p: {"axis"},
    pp.multiple_of_p: {"values"},
    pp.load_p: {"args_tree", "cache_modifier", "eviction_policy", "is_volatile"},
    pp.swap_p: {"args_tree", "eviction_policy"},
    pp.reciprocal_p: {"approx", "full_range"},
    pp.run_scoped_p: {"jaxpr", "collective_axes", "ref_transforms"},
    pp.get_global_p: {"what"},
    pp.semaphore_read_p: {"args_tree"},
    pp.semaphore_signal_p: {"args_tree", "device_id_type"},
    pp.semaphore_wait_p: {"args_tree"},
    pp.delay_p: set(),
    mp.bitcast_p: {"ty"},
    mp.roll_p: {"axis", "stride", "stride_axis"},
    mp.dma_start_p: {"tree", "device_id_type", "priority", "add"},
    mp.dma_wait_p: {"tree", "device_id_type", "insert_dummy_device", "is_wait_send"},
    mp.get_barrier_semaphore_p: {"tag"},
    mp.prng_seed_p: set(),
    mp.prng_random_bits_p: {"shape"},
    mp.split_key_p: set(),
    mp.join_key_p: {"impl"},
    mp.stochastic_round_p: {"target_dtype"},
    mp.pack_elementwise_p: {"packed_dtype"},
    mp.unpack_elementwise_p: {"index", "packed_dtype", "unpacked_dtype"},
    mp.touch_p: set(),
    mp.trace_value_p: {"label"},
    mp.matmul_push_rhs_p: {"staging_register", "mxu_index", "transpose"},
    mp.matmul_acc_lhs_p: {"acc_transforms_tree", "load_staged_rhs"},
    mp.matmul_pop_p: {"acc_transforms_tree"},
    mp.matmul_lhs_fifo_p: {"mxu_index", "load_staged_rhs"},
    mp.matmul_pop_fifo_p: {"shape", "mxu_index", "dtype"},
    mp.conv_p: {
        "dimension_numbers",
        "window_strides",
        "padding",
        "lhs_dilation",
        "rhs_dilation",
        "window_reversal",
        "feature_group_count",
        "batch_group_count",
        "precision",
        "preferred_element_type",
    },
    sp.get_p: {"tree"},
    sp.swap_p: {"tree"},
    sp.addupdate_p: {"tree"},
    sp.broadcast_to_p: {"shape"},
    sp.create_linear_p: {"ty", "memory_space"},
    sp.pin_p: {"to"},
    sp.unpin_p: set(),
    sp.with_memory_space_constraint_p: {"memory_space"},
    core.ref_p: {"memory_space", "kind", "pin"},
    core.empty_ref_p: {"ty", "memory_space", "pin"},
    core.free_ref_p: set(),
    core.freeze_p: set(),
}
# Some source-level Ref operations are lowered to simpler primitives before
# the TPU registry is consulted. They still belong in source graph comparison.
TPU_PRIMITIVES |= frozenset(PARAMETER_SCHEMAS)


@dataclasses.dataclass(frozen=True)
class _PipelineTree:
    node_type: type | None
    metadata: Any
    children: tuple[_PipelineTree, ...]


def _pipeline_tree(tree: Any, grid: pc.GridMapping) -> _PipelineTree:
    """Materialize allocation index maps; Python callable identity is not semantics."""

    def metadata(value: Any) -> Any:
        if type(value) is pc.BlockSpec:
            if {f.name for f in dataclasses.fields(value)} != {
                "block_shape",
                "index_map",
                "memory_space",
                "pipeline_mode",
            }:
                raise UnsupportedJaxprError("emit_pipeline", "unaudited allocation BlockSpec")
            index_map, out_tree = None, None
            if value.index_map is not None:
                args, kwargs = grid.index_map_tree.unflatten([False] * len(grid.index_map_avals))
                debug = api_util.debug_info("allocation index_map", value.index_map, args, kwargs)
                with grid.trace_env():
                    index_map, outputs = pe.trace_to_jaxpr(
                        value.index_map,
                        ft.FTPyTree(grid.index_map_avals, grid.index_map_tree),
                        debug,
                    )
                if any(isinstance(c, core.Tracer) for c in index_map.consts):
                    raise UnsupportedJaxprError(
                        "emit_pipeline", "allocation index map has unbound tracer captures"
                    )
                out_tree = outputs.tree
            return (
                "allocation_block_spec",
                value.block_shape,
                index_map,
                out_tree,
                value.memory_space,
                value.pipeline_mode,
            )
        if isinstance(value, tuple):
            return tuple(metadata(v) for v in value)
        return value

    node = tree.node_data()
    return _PipelineTree(
        None if node is None else node[0],
        None if node is None else metadata(node[1]),
        tuple(_pipeline_tree(child, grid) for child in tree.children()),
    )


def capture_bindings(equation: core.JaxprEqn) -> dict[str, tuple[core.Atom, ...]]:
    """Explicit parent operands replacing child const payloads at execution."""
    p = equation.params
    if equation.primitive is run_scoped_p:
        return {"params.jaxpr": tuple(equation.invars)}
    if equation.primitive is pipeline_body_p:
        _, consts, _ = p["in_tree"].unflatten(equation.invars)
        return {"params.jaxpr": tuple(jax.tree.leaves(consts))}
    if equation.primitive is emit_pipeline_p:
        args = p["args_tree"].unflatten(equation.invars)
        result = {"params.body_jaxpr": tuple(args.body_consts)}
        start = 0
        for i, block in enumerate(p["grid_mapping"].block_mappings):
            end = start + len(block.index_map_jaxpr.constvars)
            result[f"params.grid_mapping.block_mappings[{i}].index_map_jaxpr"] = tuple(
                args.all_index_map_consts[start:end]
            )
            start = end
        return result
    return {}


def captures_child(equation: core.JaxprEqn, path: str) -> bool:
    return any(path == role or path.endswith("." + role) for role in capture_bindings(equation))


def effect_supported(effect: Any) -> bool:
    return type(effect) in (*REF_EFFECTS, *STATE_EFFECTS, effects.ErrorEffect)


def ordered(equation: core.JaxprEqn) -> bool:
    """Keep even resource reads ordered; no alias independence is assumed."""
    # free_ref has empty effects in JAX 0.11.2, but invalidates a resource and
    # has a dedicated non-DCE rule. Its position is part of resource lifetime.
    return any(effect_supported(e) for e in equation.effects) or equation.primitive in {
        pallas_call_p,
        core.free_ref_p,
        # JAX 0.11.2 reports no effects for this resource observation. Moving it
        # across signal/wait changes the observed count despite identical inputs.
        pp.semaphore_read_p,
    }


def parameters(equation: core.JaxprEqn) -> dict[str, Any]:
    """Drop only audited diagnostics; preserve the original executable params."""
    p = equation.params
    schema = PARAMETER_SCHEMAS.get(equation.primitive)
    if schema is not None and p.keys() != schema:
        raise UnsupportedJaxprError(
            equation.primitive.name, "unaudited Pallas/state primitive parameter schema"
        )
    if equation.primitive is emit_pipeline_p:
        return {**p, "args_tree": _pipeline_tree(p["args_tree"], p["grid_mapping"])}
    if equation.primitive is not pallas_call_p:
        return p
    if p.keys() != CALL_SCHEMA:
        raise UnsupportedJaxprError("pallas_call", "unaudited pallas_call parameter schema")
    if type(p["grid_mapping"]) is not pc.GridMapping:
        raise UnsupportedJaxprError("pallas_call", "unreviewed GridMapping type")
    if p["compiler_params"] is not None and type(p["compiler_params"]) is not mc.CompilerParams:
        raise UnsupportedJaxprError("pallas_call", "only the locked TPU compiler is admitted")
    return {k: v for k, v in p.items() if k not in {"name", "debug"}}


def encode(value: Any, path: str, encoder: _Encoder) -> tuple[bool, Any]:
    """Return an exact encoding when this is a Pallas/Ref source descriptor."""
    if type(value) is _PipelineTree:
        return True, (
            _PipelineTree,
            value.node_type,
            encoder.encode(value.metadata, f"{path}.metadata"),
            encoder.encode(value.children, f"{path}.children"),
        )
    if isinstance(value, hijax.HiPrim):
        if type(value) is not Einshape:
            raise UnsupportedJaxprError(path, "unreviewed high-level primitive")
        # Include both the declared parameters and all flattened type/tree
        # fields used by HiPrim's typecheck and lowering. No callable fallback.
        return True, (
            Einshape,
            encoder.encode(vars(value), path),
            encoder.encode(value.effects, path),
        )
    if type(value) is effects.ErrorEffect:
        return True, encoder.fields(value, ("error_type", "shape_dtypes"), path)
    if isinstance(value, type) and value in ERROR_TYPES:
        return True, ("check_error_type", value)
    if type(value) is _jax.Traceback:
        return True, (
            "traceback",
            tuple(
                (f.file_name, f.function_name, f.function_start_line, f.line_num)
                for f in value.frames
            ),
        )
    if type(value) is prng.PRNGImpl:
        if value not in PRNG_IMPLEMENTATIONS:
            raise UnsupportedJaxprError(path, "unreviewed random implementation")
        # Includes the implementation functions' identities, not only its name.
        return True, ("prng_implementation", prng.KeyTy(value))
    if type(value) is jax.ShapeDtypeStruct:
        return True, encoder.fields(value, tuple(jax.ShapeDtypeStruct.__slots__), path)
    for known in (st.AbstractRef, *REF_EFFECTS, *STATE_EFFECTS, pc.AbstractSemaphoreTy):
        if isinstance(value, known):
            if known is pc.AbstractSemaphoreTy:
                if type(value) not in (pc.Semaphore, pc.BarrierSemaphore, mc.DMASemaphore):
                    raise UnsupportedJaxprError(path, "unreviewed semaphore dtype")
                return True, ("semaphore_dtype", type(value))
            if type(value) is not known:
                raise UnsupportedJaxprError(path, f"unreviewed descriptor subclass {type(value)}")
            if known is st.AbstractRef:
                return True, encoder.fields(value, ("inner_aval", "memory_space", "kind"), path)
            if known in REF_EFFECTS:
                return True, encoder.fields(value, ("input",), path)
            return True, ("state_effect", known)
    if isinstance(value, (pc.GridMapping, pc.BlockMapping)):
        if type(value) not in (pc.GridMapping, pc.BlockMapping):
            raise UnsupportedJaxprError(path, "unreviewed grid/block descriptor subclass")
        names = GRID_FIELDS if type(value) is pc.GridMapping else BLOCK_FIELDS
        diagnostics = {"debug"} if type(value) is pc.GridMapping else {"debug", "origin"}
        if {f.name for f in dataclasses.fields(value)} != set(names) | diagnostics:
            raise UnsupportedJaxprError(path, "unaudited grid/block descriptor fields")
        return True, encoder.fields(value, names, path)
    if value is pc.dynamic_grid_dim:
        return True, ("dynamic_grid_dim",)
    if type(value) is FrozenDict:
        return True, (FrozenDict, encoder.encode(dict(value), path))
    return False, None


def replace_descriptor(value: Any, fields: dict[str, Any]) -> Any:
    if type(value) is pc.BlockMapping:
        # Its InitVar is not a stored field; .replace preserves captured consts.
        return value.replace(**fields)
    return dataclasses.replace(value, **fields)
