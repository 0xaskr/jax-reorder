"""The reviewed Pallas-only replicated sharding metadata transition."""

from dataclasses import replace

from ..errors import UnsupportedJaxprError
from .model import ArrayType, Attribute, OperationView, ShardingView


def sharding_contract(sharding: ShardingView) -> Attribute:
    """Portable complete facts for the admitted abstract, replicated descriptor."""
    return (
        sharding.mesh.axes,
        sharding.mesh.abstract_device,
        sharding.mesh.concrete,
        sharding.partitions,
        sharding.memory_kind,
        sharding.logical_device_ids,
    )


def validate_local_reshard(source: ArrayType, target: ArrayType, path: str) -> None:
    a, b = source.sharding, target.sharding
    if (
        replace(source, sharding=target.sharding) != target
        or a is None
        or b is None
        or any(a.partitions)
        or any(b.partitions)
        or a.mesh.concrete
        or b.mesh.concrete
        or a.mesh.devices
        or b.mesh.devices
        or not b.mesh.axes
        or any(kind != "Manual" for _, _, kind in b.mesh.axes)
        or a.mesh.axes
        and a.mesh != b.mesh
        or a.memory_kind != b.memory_kind
        or a.logical_device_ids != b.logical_device_ids
    ):
        raise UnsupportedJaxprError(path, "reshard must preserve a local replicated array contract")


def validate_reshard_operation(op: OperationView, inputs: tuple[ArrayType, ...]) -> None:
    if len(inputs) != 1 or len(op.results) != 1 or not isinstance(op.results[0].type, ArrayType):
        raise UnsupportedJaxprError(op.origin.path, "invalid Pallas local reshard arity")
    target = op.results[0].type
    validate_local_reshard(inputs[0], target, op.origin.path)
    assert target.sharding is not None
    parameters = {p.name: p.value for p in op.parameters}
    if (
        parameters.get("__pallas_kernel_reshard") is not True
        or parameters
        != {
            "__pallas_kernel_reshard": True,
            "concrete_mesh": None,
            "dst_sharding": sharding_contract(target.sharding),
        }
        or len(op.parameters) != 3
    ):
        raise UnsupportedJaxprError(op.origin.path, "Pallas local reshard metadata does not replay")


def validate_local_broadcast(op: OperationView, inputs: tuple[ArrayType, ...]) -> None:
    if len(inputs) != 1 or len(op.results) != 1 or not isinstance(op.results[0].type, ArrayType):
        raise UnsupportedJaxprError(op.origin.path, "invalid Pallas local broadcast arity")
    target = op.results[0].type
    validate_local_reshard(replace(inputs[0], shape=target.shape), target, op.origin.path)
    assert target.sharding is not None
    parameters = {p.name: p.value for p in op.parameters}
    if (
        parameters.get("__pallas_kernel_broadcast") is not True
        or parameters.get("sharding") != sharding_contract(target.sharding)
        or parameters.get("shape") != target.shape
        or set(parameters)
        != {"__pallas_kernel_broadcast", "sharding", "shape", "broadcast_dimensions"}
        or len(op.parameters) != 4
    ):
        raise UnsupportedJaxprError(
            op.origin.path, "Pallas local broadcast metadata does not replay"
        )
