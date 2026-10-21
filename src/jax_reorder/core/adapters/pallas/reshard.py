"""Admit only the Pallas lowering's local replicated metadata-only reshard."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from jax._src import mesh as mesh_lib

from ...local_reshard import sharding_contract, validate_local_reshard
from ...model import Attribute, ParameterView
from ..attributes import require
from ..sharding import mesh_view, sharding_view

if TYPE_CHECKING:
    from ..jax import JaxAdapter


def adapt_local_reshard(eq: Any, adapter: JaxAdapter, path: str) -> tuple[ParameterView, ...]:
    require(adapter.in_pallas_kernel, path, "reshard is admitted only inside a Pallas kernel")
    require(
        set(eq.params) == {"dst_sharding", "concrete_mesh"}
        and eq.params["concrete_mesh"] is None
        and not eq.effects
        and len(eq.invars) == len(eq.outvars) == 1,
        path,
        "unaudited Pallas reshard parameter/effect schema",
    )
    source = adapter.array_type(eq.invars[0].aval, path + ".source")
    target = adapter.array_type(eq.outvars[0].aval, path + ".target")
    validate_local_reshard(source, target, path)
    destination = sharding_view(eq.params["dst_sharding"], path + ".dst_sharding")
    require(
        destination == target.sharding
        and destination.mesh == mesh_view(mesh_lib.get_abstract_mesh(), path + ".kernel_mesh"),
        path,
        "Pallas reshard target differs from its bound manual mesh or output contract",
    )
    # Mosaic's registered lowering returns its operand. Keep the operation and
    # all descriptor facts so the core can replay this narrow identity rule.
    return (
        ParameterView("__pallas_kernel_reshard", True),
        ParameterView("concrete_mesh", None),
        ParameterView("dst_sharding", sharding_contract(destination)),
    )


def local_broadcast_sharding(eq: Any, adapter: JaxAdapter, path: str) -> Attribute:
    """Validate the replicated explicit broadcast emitted next to where's reshard."""
    require(adapter.in_pallas_kernel, path, "explicit broadcast sharding is Pallas-kernel-only")
    require(
        len(eq.invars) == len(eq.outvars) == 1 and not eq.effects,
        path,
        "invalid local broadcast interface",
    )
    source = adapter.array_type(eq.invars[0].aval, path + ".source")
    target = adapter.array_type(eq.outvars[0].aval, path + ".target")
    validate_local_reshard(replace(source, shape=target.shape), target, path)
    destination = sharding_view(eq.params["sharding"], path + ".sharding")
    require(
        destination == target.sharding
        and destination.mesh == mesh_view(mesh_lib.get_abstract_mesh(), path + ".kernel_mesh"),
        path,
        "Pallas broadcast target differs from its bound manual mesh or output contract",
    )
    return sharding_contract(destination)
