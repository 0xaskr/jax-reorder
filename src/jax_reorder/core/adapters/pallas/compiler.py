"""Versioned Mosaic TPU compiler policy, without erasing execution attributes."""

from __future__ import annotations

import dataclasses
from typing import Any

from jax._src.frozen_dict import FrozenDict
from jax._src.pallas.mosaic import core as tpu

from ...model import CompilerView, ParameterView
from ..attributes import attribute, require

COMPILER_FIELDS = frozenset(
    {
        "dimension_semantics",
        "allow_input_fusion",
        "vmem_limit_bytes",
        "collective_id",
        "has_side_effects",
        "flags",
        "internal_scratch_in_bytes",
        "serialization_format",
        "disable_bounds_checks",
        "disable_semaphore_checks",
        "skip_device_barrier",
        "allow_collective_id_without_custom_barrier",
        "shape_invariant_numerics",
        "use_tc_tiling_on_sc",
        "needs_layout_passes",
        "fuse_transposed_lhs_in_matmul",
        "opt_level",
    }
)


def adapt_compiler(
    source: Any, dimensions: int, inputs: int, path: str
) -> tuple[CompilerView, tuple[str, ...]]:
    defaults = tpu.CompilerParams()
    explicit = source is not None
    source = defaults if source is None else source
    require(
        type(source) is tpu.CompilerParams,
        path,
        "only reviewed Mosaic TPU compiler_params are admitted",
    )
    require(
        {f.name for f in dataclasses.fields(source)} == COMPILER_FIELDS,
        path,
        "unaudited TPU CompilerParams schema",
    )
    dimensions_source = source.dimension_semantics
    semantics = (
        ("arbitrary",) * dimensions
        if dimensions_source is None
        else tuple(
            item.value if type(item) is tpu.GridDimensionSemantics else item
            for item in dimensions_source
        )
    )
    require(
        len(semantics) == dimensions
        and all(
            type(s) is str and s in {"parallel", "arbitrary", "core_parallel", "subcore_parallel"}
            for s in semantics
        ),
        path,
        "invalid grid dimension semantics",
    )
    require(
        source.flags is None or not source.flags,
        path + ".flags",
        "compiler flags need per-key semantic review",
    )
    require(
        source.flags is None or type(source.flags) is FrozenDict,
        path,
        "invalid compiler flag mapping",
    )
    require(
        source.allow_input_fusion is None
        or (
            len(source.allow_input_fusion) == inputs
            and all(type(v) is bool for v in source.allow_input_fusion)
        ),
        path,
        "invalid input fusion contract",
    )
    for name in ("vmem_limit_bytes", "internal_scratch_in_bytes", "collective_id"):
        value = getattr(source, name)
        require(
            value is None or (type(value) is int and value >= 0),
            path + "." + name,
            "invalid compiler integer option",
        )
    require(
        type(source.serialization_format) is int and source.serialization_format > 0,
        path,
        "invalid serialization format",
    )
    for name in (
        "disable_bounds_checks",
        "disable_semaphore_checks",
        "skip_device_barrier",
        "allow_collective_id_without_custom_barrier",
        "shape_invariant_numerics",
        "needs_layout_passes",
        "fuse_transposed_lhs_in_matmul",
    ):
        require(
            type(getattr(source, name)) is bool,
            path + "." + name,
            "invalid compiler boolean option",
        )
    require(
        type(source.has_side_effects) in (bool, tpu.SideEffectType),
        path,
        "invalid compiler side-effect policy",
    )
    require(
        source.use_tc_tiling_on_sc is None or type(source.use_tc_tiling_on_sc) is bool,
        path,
        "invalid core tiling policy",
    )
    require(
        source.opt_level is None or type(source.opt_level) is tpu.OptLevel,
        path,
        "invalid optimization policy",
    )
    parameters = tuple(
        ParameterView(
            name,
            ("empty_flags",)
            if name == "flags" and source.flags is not None
            else attribute(getattr(source, name), path + "." + name),
        )
        for name in sorted(COMPILER_FIELDS)
    )

    def matches_defaults(excluded: set[str]) -> bool:
        return all(
            (
                not source.flags
                if name == "flags"
                else attribute(getattr(source, name), path)
                == attribute(getattr(defaults, name), path)
            )
            for name in COMPILER_FIELDS - excluded
        )

    standard = matches_defaults({"dimension_semantics"})
    reviewed_layout = (
        source.needs_layout_passes is False or source.fuse_transposed_lhs_in_matmul is True
    ) and matches_defaults(
        {"dimension_semantics", "needs_layout_passes", "fuse_transposed_lhs_in_matmul"}
    )
    reviewed_resources = (
        source.vmem_limit_bytes is not None or source.disable_bounds_checks
    ) and matches_defaults(
        {
            "dimension_semantics",
            "needs_layout_passes",
            "fuse_transposed_lhs_in_matmul",
            "vmem_limit_bytes",
            "disable_bounds_checks",
        }
    )
    return (
        CompilerView(
            explicit, "mosaic_tpu", parameters, standard, reviewed_layout, reviewed_resources
        ),
        semantics,
    )
