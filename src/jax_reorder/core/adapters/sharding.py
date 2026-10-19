"""Typed device/mesh/spec facts and positional shard_map bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
from jax._src import config, core, shard_map
from jax._src import mesh as mesh_lib
from jax.sharding import NamedSharding, PartitionSpec

from ..model import DeviceView, MeshView, RegionView, ShardingView, ShardMapView
from .attributes import attribute, require

if TYPE_CHECKING:
    from .jax import JaxAdapter

SHARD_SCHEMA = frozenset(
    {"jaxpr", "mesh", "in_specs", "out_specs", "check_vma", "newly_manual_axes"}
)


def mesh_view(source: Any, path: str) -> MeshView:
    require(
        type(source) in (mesh_lib.Mesh, mesh_lib.AbstractMesh), path, "unaudited mesh descriptor"
    )
    concrete = type(source) is mesh_lib.Mesh
    abstract = source.abstract_mesh if concrete else source
    require(
        all(type(n) is str for n in abstract.axis_names), path, "mesh axes must have string names"
    )
    axes = tuple(
        (name, int(size), kind.name)
        for name, size, kind in zip(
            abstract.axis_names, abstract.axis_sizes, abstract.axis_types, strict=True
        )
    )
    devices = []
    if concrete:
        for device in source.devices.flat:
            require(
                isinstance(device, jax.Device) and device.platform in ("cpu", "tpu"),
                path,
                "only CPU oracle/TPU device meshes are admitted",
            )
            coordinates = getattr(device, "coords", None)
            devices.append(
                DeviceView(
                    int(device.id),
                    int(device.process_index),
                    device.platform,
                    device.device_kind,
                    tuple(int(i) for i in coordinates) if coordinates is not None else None,
                    getattr(device, "core_on_chip", None),
                )
            )
    return MeshView(axes, tuple(devices), attribute(abstract.abstract_device, path), concrete)


def partition_view(spec: Any, path: str) -> tuple[tuple[str, ...], ...]:
    require(type(spec) is PartitionSpec, path, "unaudited partition specification")
    require(
        not spec.unreduced and not spec.reduced and spec.unreduced_kind is None,
        path,
        "reduction-state partitions are not yet admitted",
    )
    result = []
    for part in spec.partitions:
        names = () if part is None else part if type(part) is tuple else (part,)
        require(all(type(n) is str for n in names), path, "unconstrained/unknown partition axis")
        result.append(names)
    flat = tuple(n for names in result for n in names)
    require(len(flat) == len(set(flat)), path, "a mesh axis cannot partition two dimensions")
    return tuple(result)


def sharding_view(source: Any, path: str) -> ShardingView:
    require(type(source) is NamedSharding, path, "only NamedSharding abstract values are admitted")
    return ShardingView(
        mesh_view(source.mesh, path),
        partition_view(source.spec, path),
        source.memory_kind,
        source._logical_device_ids,
    )


def adapt_shard_map(eq: Any, adapter: JaxAdapter, path: str) -> tuple[ShardMapView, RegionView]:
    p = eq.params
    require(set(p) == SHARD_SCHEMA, path, "unaudited shard_map parameter schema")
    require(
        type(p["mesh"]) is mesh_lib.Mesh and not p["mesh"].empty,
        path,
        "shard_map requires a bound concrete mesh",
    )
    mesh = mesh_view(p["mesh"], path)
    require(
        type(p["check_vma"]) is bool and type(p["newly_manual_axes"]) is frozenset,
        path,
        "invalid shard_map execution options",
    )
    names = {axis[0] for axis in mesh.axes}
    require(p["newly_manual_axes"] <= names, path, "unknown manual axis")
    require(
        all(type(p[key]) is tuple for key in ("in_specs", "out_specs")),
        path,
        "invalid partition bindings",
    )
    ins = tuple(partition_view(spec, path) for spec in p["in_specs"])
    outs = tuple(partition_view(spec, path) for spec in p["out_specs"])
    require(
        len(ins) == len(eq.invars) and len(outs) == len(eq.outvars),
        path,
        "shard_map port/spec arity mismatch",
    )
    for spec in (*ins, *outs):
        require(
            all(n in names for part in spec for n in part), path, "partition axis missing from mesh"
        )
    source = p["jaxpr"]
    require(
        type(source) is core.Jaxpr
        and len(source.invars) == len(ins)
        and len(source.outvars) == len(outs),
        path,
        "invalid shard_map region interface",
    )
    inner_mesh = shard_map._as_manual_mesh(p["mesh"], p["newly_manual_axes"])
    with (
        shard_map._extend_axis_env(p["mesh"], p["newly_manual_axes"]),
        mesh_lib.use_abstract_mesh(inner_mesh),
        config._check_vma(p["check_vma"]),
    ):
        child = adapter.region(source, path + ".shards")
    # Verify the shape and abstract bindings using the same locked source rules
    # as JAX, once at the frontend. Consumers need not interpret sharding objects.
    for var, actual, spec in zip(eq.invars, source.invars, p["in_specs"], strict=True):
        expected = shard_map.shard_aval(
            p["mesh"], p["newly_manual_axes"], p["check_vma"], spec, var.aval
        )
        require(
            adapter.value_type(expected, path) == adapter.value_type(actual.aval, path),
            path,
            "shard input abstract binding mismatch",
        )
    for actual, var, spec in zip(source.outvars, eq.outvars, p["out_specs"], strict=True):
        expected = shard_map.unshard_aval(p["mesh"], p["check_vma"], spec, actual.aval)
        require(
            adapter.value_type(expected, path) == adapter.value_type(var.aval, path),
            path,
            "shard output abstract binding mismatch",
        )
    require(not child.effects, path, "effectful shard_map region is not yet admitted")
    return ShardMapView(
        mesh, ins, outs, p["check_vma"], tuple(sorted(p["newly_manual_axes"]))
    ), child
